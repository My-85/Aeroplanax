# /home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/aeroplanax_waypoint_smanuer.py
import os
import functools
from typing import Dict, Optional, Sequence, Any, Tuple, Callable
import jax
import jax.numpy as jnp
import numpy as np
import chex
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
import optax
import orbax.checkpoint as ocp
from gymnax.environments import spaces

# 框架依赖（与既有环境保持一致）
from .aeroplanax import AgentName, AgentID, EnvState, EnvParams, AeroPlanaxEnv
from .core.simulators import fighterplane

from .core.simulators.fighterplane.dynamics import atmos  # 新增

# ========== Baseline 控制器（RNN / LSTM，两种都支持） ==========
class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        zeros = self.initialize_carry(*rnn_state.shape)
        rnn_state = jnp.where(resets[:, np.newaxis], jax.lax.stop_gradient(zeros), rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ScannedLSTM(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        (h, c) = carry
        ins, resets = x
        zeros_h, zeros_c = self.initialize_carry(*h.shape)
        h = jnp.where(resets[:, np.newaxis], jax.lax.stop_gradient(zeros_h), h)
        c = jnp.where(resets[:, np.newaxis], jax.lax.stop_gradient(zeros_c), c)
        (h2, c2), y = nn.LSTMCell(features=ins.shape[1])((h, c), ins)
        return (h2, c2), y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.LSTMCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x  # (T,B,22), (T,B)

        # 前端 MLP
        emb = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        emb = activation(emb)

        # GRU（按时间 scan）
        hidden, emb = ScannedRNN()(hidden, (emb, dones))

        # trunk
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        # 预测头（t -> t+1）：vt_norm、pitch(rad)、az
        pred_h = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(nn_fc2)
        pred_h = activation(pred_h)
        pred = nn.Dense(3, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(pred_h)

        # stop‑grad 拼回：obs + pred → 小 MLP
        pred_sg = jax.lax.stop_gradient(pred)
        obs_aug = jnp.concatenate([obs, pred_sg], axis=-1)
        obs_aug = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs_aug)
        obs_aug = activation(obs_aug)

        # 汇总增强特征
        aug = jnp.concatenate([nn_fc2, obs_aug, pred_sg], axis=-1)
        aug = nn.LayerNorm()(aug)

        # actor（四个离散通道）
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2.0), bias_init=constant(0.0))(aug)
        actor_mean = activation(actor_mean)
        pi_throttle = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_aileron  = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_rudder   = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))

        # critic
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2.0), bias_init=constant(0.0))(aug)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1), pred

class ActorCriticLSTM(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        embedding = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedLSTM()(hidden, rnn_in)
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)
        pi_throttle = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_aileron  = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_rudder   = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)

# ========== 任务参数 / 状态 ==========
@struct.dataclass(frozen=True)
class WaypointTaskParams(EnvParams):
    max_steps: int = 3000
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    action_type: int = 1 # 0: continuous, 1: discrete
    max_altitude: float = 20000.0
    min_altitude: float = 2000.0
    max_vt: float = 360.0
    min_vt: float = 100.0

    #-----------------------------------------------------------------#
    # ①普通航点跟踪任务，逐渐变难，测试baseline极限性能
    #=================================================#
    # 航点范围（相对初始中心）
    wp_min_xy: float = 1500.0           # 允许更近的平面位移
    #=================================================#
    wp_max_xy: float = 30000.0
    wp_min_alt: float = 1000.0
    wp_max_alt: float = 15000.0
    #=================================================#

    #=================================================#
    # min_turn_deg_init: float = 20.0
    # min_turn_deg_step: float = 10.0  # 每达成一个航点提升最小转向角

    min_turn_deg_init: float = 10.0     # 初始最小转弯角小一点
    min_turn_deg_step: float = 5.0      # 难度上升慢一点
    #=================================================#

    #-----------------------------------------------------------------#
    # ②s机动
    # --- S 形机动（可开关） ---
    """
    想“每次转弯更大”（更圆、更缓）：增大 s_half_period_north（半周期更长 → 半径更大）；
    想“左右偏得更明显”：增大 s_amplitude。
    """
    use_s_curve: bool = struct.field(pytree_node=False, default=False)  # 开关
    # s_step_north: float = 1500.0    # 沿北向的步进（米），越小越密
    s_amplitude: float = 2000.0                 # 左右摆幅 A（米）

    s_half_period_north: float = 8000.0         # 北向半周期长度（米）← 半径更大就把它调大
    s_points_per_half: int = 12                 # 每个半周期上取多少个航点（越大越平滑）

    s_altitude_lock: bool = struct.field(pytree_node=False, default=True)  # 锁高：期望俯仰=0

    s_target_vt: float = struct.field(pytree_node=False, default=250.0)   # S 形恒定巡航速度(m/s)



    # 航点判定与难度提升
    # reach_radius_init: float = 800.0
    # reach_radius_decay: float = 0.9

    # # 先放宽达成判定
    # reach_radius_init: float = 1500.0   # 初始半径放大
    # reach_radius_decay: float = 0.95    # 衰减慢一点
    #=================================================#

    #-----------------------------------------------------------------#

    # ③垂直方向筋斗
    #=================================================#
    # 垂直回环（筋斗）
    use_vertical_loop: bool = struct.field(pytree_node=False, default=False)
    loop_radius: float = 2500.0
    loop_points_per_circle: int = 72
    loop_forward_north: float = 4000.0
    loop_target_vt: float = struct.field(pytree_node=False, default=250.0)
    # 起始相位与方向（新增）
    loop_phase0_deg: float = struct.field(pytree_node=False, default=120.0)  # 120°：前上方
    loop_direction: int = struct.field(pytree_node=False, default=1)         # +1 顺时针，-1 逆时针

    loop_pitch_limit_deg: float = struct.field(pytree_node=False, default=55.0)
    #=================================================#

    #-----------------------------------------------------------------#

    # 航点判定
    reach_radius_init: float = 600.0
    reach_radius_decay: float = 1.0
    max_waypoints: int = 100 # 达成目标航点个数就终止（设置一个很大的数）（到够数就 SUCCESS）

    #-----------------------------------------------------------------#

    # baseline 控制器
    baseline_type: str = struct.field(pytree_node=False, default="rnn")  # "rnn" or "lstm"
    baseline_seed: int = 42
    baseline_hidden: int = 128
    baseline_fc: int = 128
    baseline_loaddir: str = struct.field(pytree_node=False, default="")  # checkpoint 目录
    # 控制维度（离散）
    action_dims: Sequence[int] = struct.field(pytree_node=False, default=(31, 41, 41, 41))
    use_internal_baseline: bool = struct.field(pytree_node=False, default=True)  # 新增：是否用内置基线驱动

    # WaypointTaskParams 中新增（保持 pytree_node=False）
    use_high_level_action: bool = struct.field(pytree_node=False, default=False)
    hl_bins_heading: int = struct.field(pytree_node=False, default=17)
    hl_bins_pitch: int = struct.field(pytree_node=False, default=17)
    hl_bins_speed: int = struct.field(pytree_node=False, default=9)

    #=================================================#
    # 垂直方向最小分离（米）
    # min_alt_sep: float = 1000.0
    
    min_alt_sep: float = 500.0          # 垂直最小分离小一点，避免爬升/下降太狠
    #=================================================#

@struct.dataclass
class WaypointTaskState(EnvState):
    hstate: jnp.ndarray               # RNN/LSTM 隐藏态
    waypoint: jnp.ndarray             # (3,) [north, east, altitude]
    reached: jnp.ndarray              # 已到达的航点数
    reach_radius: jnp.ndarray         # 当前判定半径
    difficulty: jnp.ndarray           # 当前难度级别（影响最小转向角）
    time: jnp.ndarray                 # 仍沿用父类，计步

    s_origin_n: jnp.ndarray                  # 正弦起点（北向锚点）
    s_origin_e: jnp.ndarray                  # 正弦起点（东向锚点）

    # ===== 筋斗（垂直回环）状态 =====
    loop_center_n: jnp.ndarray
    loop_center_e: jnp.ndarray
    loop_center_alt: jnp.ndarray
    loop_idx: jnp.ndarray

    cmd_heading: jnp.ndarray
    cmd_pitch: jnp.ndarray
    cmd_vt: jnp.ndarray
    # 新增：给渲染提供参考指令
    target_heading: jnp.ndarray
    target_pitch: jnp.ndarray
    @classmethod
    def create(cls, env_state: EnvState, hstate, waypoint, reached, reach_radius, difficulty,
               s_origin_n=jnp.array(0.0), s_origin_e=jnp.array(0.0),
               loop_center_n=jnp.array(0.0), loop_center_e=jnp.array(0.0),
               loop_center_alt=jnp.array(0.0), loop_idx=jnp.array(0),
               target_heading=jnp.array(0.0), target_pitch=jnp.array(0.0)):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            pre_rewards=env_state.pre_rewards,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            hstate=hstate,
            waypoint=waypoint,
            reached=reached,
            reach_radius=reach_radius,
            difficulty=difficulty,

            s_origin_n=s_origin_n,
            s_origin_e=s_origin_e,

            loop_center_n=loop_center_n,
            loop_center_e=loop_center_e,
            loop_center_alt=loop_center_alt,
            loop_idx=loop_idx,

            cmd_heading=jnp.array(0.0),
            cmd_pitch=jnp.array(0.0),
            cmd_vt=jnp.array(0.0),

            target_heading=target_heading,
            target_pitch=target_pitch,
        )
        
# ========== 工具 ==========
def _wrap_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

# 计算筋斗圆的相位：n = c_n - R*cos(phi), alt = c_a + R*sin(phi)
def _loop_phase(center_n, center_alt, R, n, alt):
    x = jnp.clip((center_n - n) / jnp.maximum(R, 1e-6), -1e6, 1e6)
    y = jnp.clip((alt - center_alt) / jnp.maximum(R, 1e-6), -1e6, 1e6)
    return jnp.arctan2(y, x)  # [-pi, pi]

def _bearing(north, east):
    return jnp.arctan2(east, north)

def _desired_pitch(d_alt, h_dist):
    # return jnp.arctan2(-d_alt, jnp.maximum(h_dist, 1e-6)) # 当航点在上方（d_alt > 0）时，正确的俯仰目标应该是正的；但你的实现用 -d_alt，会让飞机向下俯冲，高度误差越飞越大 → 永远到不了航点，reached_cnt 一直是 0。
    return jnp.arctan2(d_alt, jnp.maximum(h_dist, 1e-6))
    # 改用正的 d_alt 即可。
    # 当航点在上方（d_alt > 0）时，飞机向上爬升，俯仰目标为正；
    # 当航点在下方（d_alt < 0）时，飞机向下俯冲，俯仰目标为负。
    # 这样改后，飞机会根据航点高度自动调整俯仰姿态，不再“越飞越远”。

def _controller_obs(state: fighterplane.FighterPlaneState, target_pitch, target_heading, target_vt):
    altitude = state.altitude
    roll, pitch, yaw = state.roll, state.pitch, state.yaw
    vt = state.vt
    alpha, beta = state.alpha, state.beta
    P, Q, R = state.P, state.Q, state.R

    # 误差/基础量（与训练一致的归一）
    norm_delta_heading = _wrap_pi(yaw - target_heading)
    norm_delta_pitch   = _wrap_pi(pitch - target_pitch)
    norm_delta_vt      = (vt - target_vt) / 340.0
    norm_altitude      = altitude / 5000.0
    norm_vt            = vt / 340.0

    roll_sin,  roll_cos  = jnp.sin(roll),  jnp.cos(roll)
    pitch_sin, pitch_cos = jnp.sin(pitch), jnp.cos(pitch)
    alpha_sin, alpha_cos = jnp.sin(alpha), jnp.cos(alpha)
    beta_sin,  beta_cos  = jnp.sin(beta),  jnp.cos(beta)

    # 气动/能量/轨迹角
    az = state.az  # g-load（垂直）
    alt_ft = altitude / 0.3048
    vt_ft  = jnp.clip(vt / 0.3048, 0.1, 1e6)
    mach, qbar, _ = atmos(alt_ft, vt_ft)
    # 参考动压：中位高度 + max vt
    alt_mid_ft = ((WaypointTaskParams.min_altitude + WaypointTaskParams.max_altitude) * 0.5) / 0.3048
    vt_ref_ft  = WaypointTaskParams.max_vt / 0.3048
    _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
    qbar_norm = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 2.0)

    spec_energy = 9.81 * altitude + 0.5 * jnp.clip(vt, 0.0, 1e4) ** 2
    e_ref = 9.81 * WaypointTaskParams.max_altitude + 0.5 * (WaypointTaskParams.max_vt ** 2)
    spec_energy_norm = jnp.clip(spec_energy / (e_ref + 1e-6), 0.0, 1.5)

    vx, vy, vz = state.vel_x, state.vel_y, state.vel_z
    vh = jnp.sqrt(jnp.maximum(vx * vx + vy * vy, 1e-6))
    gamma = jnp.arctan2(-vz, vh)
    gamma_sin, gamma_cos = jnp.sin(gamma), jnp.cos(gamma)

    obs = jnp.vstack((
        norm_delta_heading, norm_delta_pitch, norm_delta_vt,
        norm_altitude, norm_vt,
        roll_sin, roll_cos, pitch_sin, pitch_cos,
        alpha_sin, alpha_cos, beta_sin, beta_cos,
        P, Q, R,
        az, mach, qbar_norm, spec_energy_norm, gamma_sin, gamma_cos
    ))
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    low = jnp.array([
        -jnp.pi, -jnp.pi, -2.0,
        0.0, 0.0,
        -1., -1., -1., -1.,
        -1., -1., -1., -1.,
        -10., -10., -10.,
        -6.0, 0.0, 0.0, 0.0, -1.0, -1.0
    ]).reshape(-1, 1)
    high = jnp.array([
        jnp.pi, jnp.pi, 2.0,
        5.0, 2.0,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        10., 10., 10.,
        12.0, 3.0, 2.0, 2.0, 1.0, 1.0
    ]).reshape(-1, 1)
    obs = jnp.clip(obs, low, high)
    return obs  # (22, B=1)


def _sample_waypoint(key, center_n_e_alt, params: WaypointTaskParams, min_turn_rad: float, current_yaw: float):
    # 在矩形环带内采样，确保相对当前位置的方位变化 >= min_turn_rad
    key_xy, key_alt = jax.random.split(key)
    def sample_once(key):
        rxy = jax.random.uniform(key, shape=(2,), minval=-params.wp_max_xy, maxval=params.wp_max_xy)
        rxy = jnp.where(jnp.abs(rxy) < params.wp_min_xy, jnp.sign(rxy) * params.wp_min_xy, rxy)
        alt = jax.random.uniform(key_alt, minval=params.wp_min_alt, maxval=params.wp_max_alt)
        return rxy[0], rxy[1], alt

    nx, ex, alt = sample_once(key_xy)
    # 以当前位置为原点，计算方位角与当前朝向差
    bearing = _bearing(nx, ex)
    ok = jnp.abs(_wrap_pi(bearing - current_yaw)) >= min_turn_rad

    # 若不满足，则镜像加大角度
    nx = jnp.where(ok, nx, -nx)
    ex = jnp.where(ok, ex, -ex)
    return jnp.array([center_n_e_alt[0] + nx, center_n_e_alt[1] + ex, alt])

# ========== 环境 ==========
class AeroPlanaxWaypointEnv(AeroPlanaxEnv[WaypointTaskState, WaypointTaskParams]):
    def __init__(self, env_params: Optional[WaypointTaskParams] = None):
        super().__init__(env_params)
        # 记住外部传入的 params；若没传，用默认
        self._default_params = env_params or WaypointTaskParams()
        # baseline config
        self.cfg = {
            "SEED": env_params.baseline_seed,
            "LR": 3e-4,
            "NUM_ENVS": 1,
            "NUM_ACTORS": 1,
            "FC_DIM_SIZE": env_params.baseline_fc,
            "GRU_HIDDEN_DIM": env_params.baseline_hidden,
            "ACTIVATION": "relu",
        }
        self.action_dims = list(env_params.action_dims)
        self.use_internal_baseline = env_params.use_internal_baseline # 新增：是否用内置基线驱动
        # controller init & load
        self._init_controller(env_params)

        # 供上层 wrapper 查询
        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        # 奖励函数（下方实现）
        self.reward_functions = [
            functools.partial(self._reward_distance, scale=1.0),
            functools.partial(self._reward_alignment, scale=0.3),
            functools.partial(self._reward_speed_profile, scale=0.1),
            functools.partial(self._reward_reach_bonus, bonus=3.0),
            functools.partial(self._penalty_crash, pen=-5.0),

            functools.partial(self._reward_overload_penalty, scale=-0.1),
            functools.partial(self._reward_phase_error, scale=0.2),
            functools.partial(self._reward_speed_penalty, scale=-0.05),
        ]
        self.is_potential = [False, False, False, True, False, False, False, False]

        # 终止条件
        self.termination_conditions = [
            self._term_timeout,
            self._term_crashed,
            self._term_reached_enough
        ]

    # 放在 class AeroPlanaxWaypointEnv 内，__init__ 后、spaces 段之前均可
    def _get_obs_size(self) -> int:
        return 16
    # ---------- spaces ----------
    def _get_individual_obs_space(self, i) -> spaces.Space:
        # 16维控制观测（与 _controller_obs 一致）
        return spaces.Box(-jnp.inf, jnp.inf, (16,), dtype=jnp.float32)

    def _get_individual_action_space(self, i):
        if self.use_internal_baseline and self.default_params.use_high_level_action:
            return spaces.Dict({
                "d_heading": spaces.Discrete(self.default_params.hl_bins_heading),
                "d_pitch":   spaces.Discrete(self.default_params.hl_bins_pitch),
                "d_speed":   spaces.Discrete(self.default_params.hl_bins_speed),
            })
        else:
            return spaces.Dict({
                "throttle": spaces.Discrete(self.action_dims[0]),
                "elevator": spaces.Discrete(self.action_dims[1]),
                "aileron":  spaces.Discrete(self.action_dims[2]),
                "rudder":   spaces.Discrete(self.action_dims[3]),
            })
    # ---------- controller ----------
    def _init_controller(self, params: WaypointTaskParams):
        rng = jax.random.PRNGKey(self.cfg['SEED'])
        self.controller_type = params.baseline_type.lower()
        if self.controller_type == "lstm":
            self.controller = ActorCriticLSTM(self.action_dims, config=self.cfg)
            init_h = ScannedLSTM.initialize_carry(self.cfg["NUM_ACTORS"] * self.cfg["NUM_ENVS"], self.cfg["GRU_HIDDEN_DIM"])
        else:
            self.controller = ActorCriticRNN(self.action_dims, config=self.cfg)
            init_h = ScannedRNN.initialize_carry(self.cfg["NUM_ACTORS"] * self.cfg["NUM_ENVS"], self.cfg["GRU_HIDDEN_DIM"])

        init_x = (
            jnp.zeros((1, self.cfg["NUM_ENVS"] * self.cfg["NUM_ACTORS"], 22)),  # 由16改为22
            jnp.zeros((1, self.cfg["NUM_ENVS"] * self.cfg["NUM_ACTORS"]))
        )
        controller_params = self.controller.init(rng, init_h, init_x)

        tx = optax.adam(self.cfg["LR"])
        train_state = TrainState.create(apply_fn=self.controller.apply, params=controller_params, tx=tx)
        # 恢复 checkpoint（兼容多个 Orbax 版本）
        if params.baseline_loaddir and os.path.isdir(params.baseline_loaddir):
            ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
            state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
            try:
                restored = ckptr.restore(params.baseline_loaddir, item=state)
            except Exception:
                try:
                    restored = ckptr.restore(params.baseline_loaddir, args=ocp.args.StandardRestore(item=state))
                except Exception:
                    restored = ckptr.restore(params.baseline_loaddir)
            if isinstance(restored, dict) and "params" in restored:
                self.controller_params = restored["params"]
            elif isinstance(restored, dict) and "actor_params" in restored:
                self.controller_params = restored["actor_params"]
            else:
                self.controller_params = restored
        else:
            self.controller_params = train_state.params  # 没有 checkpoint 也可运行（仅用于占位）

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state: WaypointTaskState,
        state: WaypointTaskState,
        actions: Dict[AgentName, chex.Array],
    ):
        # 仅按第0个智能体计算（单机）
        pn, pe, pa = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
        dn, de, da = state.waypoint[0] - pn, state.waypoint[1] - pe, state.waypoint[2] - pa
        hdist = jnp.sqrt(dn * dn + de * de)
        dist3d = jnp.sqrt(hdist * hdist + da * da)

        desired_heading = _bearing(dn, de)
        desired_pitch = _desired_pitch(da, hdist)

        # 筋斗模式：只做几何限幅，不做“增量限速”
        if self.default_params.use_vertical_loop:
            pitch_max = jnp.deg2rad(self.default_params.loop_pitch_limit_deg)
            pitch_min = jnp.deg2rad(-70.0)
            desired_pitch = jnp.clip(desired_pitch, pitch_min, pitch_max)

        # S形锁高（如开启）
        if self.default_params.s_altitude_lock:
            desired_pitch = 0.0
            dist3d = hdist

        # 速度目标：模式优先，其次远快近慢
        if self.default_params.use_vertical_loop:
            target_vt = state.plane_state.vt[0]  # 自由速度
        elif self.default_params.use_s_curve:
            target_vt = self.default_params.s_target_vt
        else:
            vt_far = self.default_params.max_vt * 0.9
            vt_near = self.default_params.min_vt * 1.2
            blend = jnp.clip(dist3d / 15000.0, 0.0, 1.0)
            target_vt = vt_near * (1.0 - blend) + vt_far * blend

        # 高层微调（如启用）
        if self.default_params.use_high_level_action:
            ah = actions[self.agents[0]]["d_heading"]
            ap = actions[self.agents[0]]["d_pitch"]
            asv= actions[self.agents[0]]["d_speed"]
            def map_bin(idx, bins, lo, hi):
                step = (hi - lo) / (bins - 1)
                return lo + idx * step
            d_head = map_bin(ah, self.default_params.hl_bins_heading, -jnp.deg2rad(30.0), jnp.deg2rad(30.0))
            d_pitch= map_bin(ap, self.default_params.hl_bins_pitch,   -jnp.deg2rad(15.0), jnp.deg2rad(15.0))
            d_v    = map_bin(asv,self.default_params.hl_bins_speed,   -30.0, 30.0)
            desired_heading = _wrap_pi(desired_heading + d_head)
            desired_pitch   = jnp.clip(desired_pitch + d_pitch, -jnp.deg2rad(45), jnp.deg2rad(45))
            target_vt       = jnp.clip(target_vt + d_v, self.default_params.min_vt, self.default_params.max_vt)

        # 直接用几何目标喂基线（无平滑）
        last_obs = _controller_obs(state.plane_state, desired_pitch, desired_heading, target_vt).T  # (B,22)
        last_done = jnp.zeros((1,), dtype=bool)
        ac_in = (last_obs[None, :], last_done[None, :])  # (1,B,22), (1,B)
        hstate, pi, _, _ = self.controller.apply(self.controller_params, state.hstate, ac_in)
        pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

        # 采样四通道
        key1, key2, key3, key4 = jax.random.split(key, 4)
        a_th  = pi_throttle.sample(seed=key1)
        a_elv = pi_elevator.sample(seed=key2)
        a_ail = pi_aileron.sample(seed=key3)
        a_rud = pi_rudder.sample(seed=key4)
        a = jnp.concatenate([a_th[:, :, None], a_elv[:, :, None], a_ail[:, :, None], a_rud[:, :, None]], axis=-1).squeeze(0)
        a = jax.vmap(self._decode_discrete_actions)(a)  # (B,4)
        ctrl = jax.vmap(fighterplane.FighterPlaneControlState.create)(a)

        # 状态仅同步隐藏态（cmd_* 不再作为控制输入）
        new_state = state.replace(hstate=hstate)
        return new_state, ctrl
    # ---------- 必要接口 ----------
    @property
    def default_params(self) -> WaypointTaskParams:
        # return WaypointTaskParams()
        return self._default_params

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(self, key: jax.Array, params: WaypointTaskParams) -> WaypointTaskState:
        s = super()._init_state(key, params)
        # 初始朝向/速度/高度
        yaw = jnp.array([0.0])
        q0 = jnp.array([1.0]); q3 = jnp.array([0.0])
        key, key_vt, key_alt, key_sign, key_d = jax.random.split(key, 5)
        vt0 = jax.random.uniform(key_vt, shape=(1,), minval=params.min_vt, maxval=params.max_vt)
        alt0 = jax.random.uniform(key_alt, shape=(1,), minval=params.min_altitude, maxval=params.max_altitude)
        s = s.replace(plane_state=s.plane_state.replace(yaw=yaw, vt=vt0, q0=q0, q3=q3, altitude=alt0))

        if params.use_vertical_loop:
            # 回环圆心在机头正前方 R；首航点取圆周的第 1 个离散点
            n0 = s.plane_state.north[0]
            e0 = s.plane_state.east[0]
            a0 = s.plane_state.altitude[0]
            R = params.loop_radius
            c_n = n0 + R
            c_e = e0
            c_a = a0
            dphi = 2 * jnp.pi / params.loop_points_per_circle
            dir_ = jnp.sign(jnp.asarray(params.loop_direction, dtype=jnp.float32)) # 1 顺时针，-1 逆时针
            phi0 = jnp.deg2rad(params.loop_phase0_deg) # 初始相位
            idx0 = 0
            phi = phi0 + dir_ * idx0 * dphi
            n1 = c_n - R * jnp.cos(phi)
            e1 = c_e
            a1 = c_a + R * jnp.sin(phi)
            wp = jnp.array([n1, e1, a1])
            reach_radius = jnp.array(params.reach_radius_init)
            difficulty = jnp.array(0)

        elif params.use_s_curve:

            # s形机动
            n0 = s.plane_state.north[0]
            e0 = s.plane_state.east[0]
            a0 = s.plane_state.altitude[0]
            dn = params.s_half_period_north / params.s_points_per_half  # 单步北向间距
            idx0 = 1
            n1 = n0 + idx0 * dn
            # e(n) = e0 + A * sin(pi*(n-n0)/half_period)
            e1 = e0 + params.s_amplitude * jnp.sin(jnp.pi * (n1 - n0) / params.s_half_period_north)
            wp = jnp.array([n1, e1, a0])
            
            reach_radius = jnp.array(params.reach_radius_init)
            difficulty = jnp.array(0)  # 用作“当前步数/奇偶”的计数器

        else:
            # 原随机首航点逻辑...
            # 采样首个航点（在前方）且与当前高度至少 min_alt_sep 分离
            base_wp_n = s.plane_state.north[0] + 10000.0
            base_wp_e = s.plane_state.east[0]
            sign = jax.random.choice(key_sign, jnp.array([-1.0, 1.0]))
            d_alt = jax.random.uniform(key_d, shape=(1,), minval=params.min_alt_sep, maxval=3000.0) * sign
            wp_alt = jnp.clip(alt0 + d_alt, params.min_altitude, params.max_altitude)
            wp = jnp.array([base_wp_n, base_wp_e, wp_alt[0]])

            reach_radius = jnp.array(params.reach_radius_init)
            difficulty = jnp.array(0)

        # 隐藏态
        if self.controller_type == "lstm":
            hstate = ScannedLSTM.initialize_carry(1, self.cfg["GRU_HIDDEN_DIM"])
        else:
            hstate = ScannedRNN.initialize_carry(1, self.cfg["GRU_HIDDEN_DIM"])

        # 隐藏态初始化保持不变
        return WaypointTaskState.create(
            s,
            hstate=hstate,
            waypoint=wp,
            reached=jnp.array(0),
            reach_radius=reach_radius,
            difficulty=difficulty,
            s_origin_n=jnp.array(n0 if params.use_s_curve else 0.0),
            s_origin_e=jnp.array(e0 if params.use_s_curve else 0.0),
            loop_center_n=jnp.array(c_n if params.use_vertical_loop else 0.0),
            loop_center_e=jnp.array(c_e if params.use_vertical_loop else 0.0),
            loop_center_alt=jnp.array(c_a if params.use_vertical_loop else 0.0),
            loop_idx=jnp.array(idx0 if params.use_vertical_loop else 0.0),
        ).replace(
            cmd_heading=_bearing(wp[0]-s.plane_state.north[0], wp[1]-s.plane_state.east[0]),
            cmd_pitch=_desired_pitch(wp[2]-s.plane_state.altitude[0],
                                     jnp.sqrt((wp[0]-s.plane_state.north[0])**2+(wp[1]-s.plane_state.east[0])**2)),
            cmd_vt=s.plane_state.vt[0],
            # 新增：target_* 供渲染画参考
            target_heading=_bearing(wp[0]-s.plane_state.north[0], wp[1]-s.plane_state.east[0]),
            target_pitch=_desired_pitch(wp[2]-s.plane_state.altitude[0],
                                        jnp.sqrt((wp[0]-s.plane_state.north[0])**2+(wp[1]-s.plane_state.east[0])**2))
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(self, key: chex.PRNGKey, state: WaypointTaskState, params: WaypointTaskParams) -> WaypointTaskState:

        if params.use_vertical_loop:
            n0 = state.plane_state.north[0]
            e0 = state.plane_state.east[0]
            a0 = state.plane_state.altitude[0]
            R = params.loop_radius
            c_n = n0 + R
            c_e = e0
            c_a = a0
            dphi = 2 * jnp.pi / params.loop_points_per_circle
            dir_ = jnp.sign(jnp.asarray(params.loop_direction, dtype=jnp.float32))
            phi0 = jnp.deg2rad(params.loop_phase0_deg)
            idx0 = 0
            phi = phi0 + dir_ * idx0 * dphi
            n1 = c_n - R * jnp.cos(phi)
            e1 = c_e
            a1 = c_a + R * jnp.sin(phi)
            wp = jnp.array([n1, e1, a1])
            reach_radius = jnp.array(params.reach_radius_init)
            difficulty = jnp.array(0)
            return state.replace(waypoint=wp, reached=jnp.array(0),
                                 reach_radius=reach_radius, difficulty=difficulty,
                                 loop_center_n=jnp.array(c_n), loop_center_e=jnp.array(c_e),
                                 loop_center_alt=jnp.array(c_a), loop_idx=jnp.array(idx0)
                                 ).replace(
                                 cmd_heading=_bearing(wp[0]-state.plane_state.north[0], wp[1]-state.plane_state.east[0]),
                                 cmd_pitch=_desired_pitch(wp[2]-state.plane_state.altitude[0],
                                                          jnp.sqrt((wp[0]-state.plane_state.north[0])**2+(wp[1]-state.plane_state.east[0])**2)),
                                 cmd_vt=state.plane_state.vt[0],
                                 target_heading=_bearing(wp[0]-state.plane_state.north[0], wp[1]-state.plane_state.east[0]),
                                 target_pitch=_desired_pitch(wp[2]-state.plane_state.altitude[0],
                                                             jnp.sqrt((wp[0]-state.plane_state.north[0])**2+(wp[1]-state.plane_state.east[0])**2))
                                 )

        elif params.use_s_curve:
            n0 = state.plane_state.north[0]
            e0 = state.plane_state.east[0]
            a0 = state.plane_state.altitude[0]
            dn = params.s_half_period_north / params.s_points_per_half
            idx0 = 1
            n1 = n0 + idx0 * dn
            e1 = e0 + params.s_amplitude * jnp.sin(jnp.pi * (n1 - n0) / params.s_half_period_north)
            wp = jnp.array([n1, e1, a0])
            reach_radius = jnp.array(params.reach_radius_init)
            difficulty = jnp.array(idx0)
            return state.replace(waypoint=wp, reached=jnp.array(0),
                                 reach_radius=reach_radius, difficulty=difficulty,
                                 s_origin_n=jnp.array(n0), s_origin_e=jnp.array(e0)
                                 ).replace(
                                 cmd_heading=_bearing(wp[0]-state.plane_state.north[0], wp[1]-state.plane_state.east[0]),
                                 cmd_pitch=_desired_pitch(wp[2]-state.plane_state.altitude[0],
                                                          jnp.sqrt((wp[0]-state.plane_state.north[0])**2+(wp[1]-state.plane_state.east[0])**2)),
                                 cmd_vt=state.plane_state.vt[0]
                                 )
        else:

            key, ksign, kdelta = jax.random.split(key, 3)
            base_wp_n = state.plane_state.north[0] + 12000.0
            base_wp_e = state.plane_state.east[0]
            sign = jax.random.choice(ksign, jnp.array([-1.0, 1.0]))
            d_alt = jax.random.uniform(kdelta, shape=(1,), minval=params.min_alt_sep, maxval=3000.0) * sign
            wp_alt = jnp.clip(state.plane_state.altitude[0] + d_alt, params.min_altitude, params.max_altitude)
            wp = jnp.array([base_wp_n, base_wp_e, wp_alt[0]])

            reach_radius = jnp.array(params.reach_radius_init)
            difficulty = jnp.array(0)

            return state.replace(waypoint=wp, reached=jnp.array(0),
                                 reach_radius=reach_radius, difficulty=difficulty
                                 ).replace(
                                 cmd_heading=_bearing(wp[0]-state.plane_state.north[0], wp[1]-state.plane_state.east[0]),
                                 cmd_pitch=_desired_pitch(wp[2]-state.plane_state.altitude[0],
                                                          jnp.sqrt((wp[0]-state.plane_state.north[0])**2+(wp[1]-state.plane_state.east[0])**2)),
                                 cmd_vt=state.plane_state.vt[0]
                                 )
    
        # return state.replace(waypoint=wp, reached=jnp.array(0), reach_radius=jnp.array(params.reach_radius_init), difficulty=jnp.array(0))
        # return state.replace(waypoint=wp, reached=jnp.array(0), reach_radius=reach_radius, difficulty=difficulty)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: WaypointTaskParams,
    ) -> Tuple[WaypointTaskState, Dict[str, Any]]:
        pn, pe, pa = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
        dn, de, da = state.waypoint[0] - pn, state.waypoint[1] - pe, state.waypoint[2] - pa
        hdist = jnp.sqrt(dn**2 + de**2)
        dist3d = jnp.sqrt(hdist**2 + da**2)

        desired_heading = _bearing(dn, de)
        desired_pitch = _desired_pitch(da, hdist)

        if params.use_vertical_loop:
            pitch_max = jnp.deg2rad(params.loop_pitch_limit_deg)
            pitch_min = jnp.deg2rad(-70.0)
            desired_pitch = jnp.clip(desired_pitch, pitch_min, pitch_max)

        if params.s_altitude_lock:
            desired_pitch = 0.0
            dist3d = hdist

        if params.use_vertical_loop:
            target_vt = state.plane_state.vt[0]  # 自由速度
        elif params.use_s_curve:
            target_vt = params.s_target_vt
        else:
            vt_far = params.max_vt * 0.9
            vt_near = params.min_vt * 1.2
            blend = jnp.clip(dist3d / 15000.0, 0.0, 1.0)
            target_vt = vt_near * (1.0 - blend) + vt_far * blend

        # 直接把几何目标作为“指令”（仅用于日志/渲染）
        cmd_heading = desired_heading
        cmd_pitch   = desired_pitch
        cmd_vt      = target_vt

        # 写指标/诊断（保持原字段名）
        info['cmd_heading'] = cmd_heading
        info['cmd_pitch']   = cmd_pitch
        info['cmd_vt']      = cmd_vt
        info['dist_to_wp']  = dist3d
        info['hdist_to_wp'] = hdist

        # # 达成判定（筋斗带相位门）
        # if params.use_vertical_loop:
        #     R    = params.loop_radius
        #     dphi = 2 * jnp.pi / params.loop_points_per_circle
        #     dir_ = jnp.sign(jnp.asarray(params.loop_direction, dtype=jnp.float32))
        #     phi0 = jnp.deg2rad(params.loop_phase0_deg)
        #     phi_curr = _loop_phase(state.loop_center_n, state.loop_center_alt, R, pn, pa)
        #     phi_wp = phi0 + dir_ * (state.loop_idx.astype(jnp.float32)) * dphi
        #     def _phase_progress(phi_c, phi_w, d):
        #         raw = jnp.where(d >= 0.0, phi_c - phi_w, phi_w - phi_c)
        #         return jnp.mod(raw, 2.0 * jnp.pi)
        #     prog = _phase_progress(phi_curr, phi_wp, dir_)
        #     passed_gate = prog >= (0.15 * dphi)
        #     reached_now = jnp.logical_and(dist3d <= state.reach_radius, passed_gate)
        #     info['dbg_phi_curr'] = phi_curr
        #     info['dbg_phi_wp']   = phi_wp
        #     info['dbg_phi_rel']  = prog
        #     info['dbg_passed_gate'] = passed_gate.astype(jnp.int32)
        # else:
        #     reached_now = dist3d <= state.reach_radius

        # 仅半径达成，不用相位门
        reached_now = dist3d <= state.reach_radius

        info['reached_this_step'] = reached_now

        # 其它诊断与奖励（原样）
        info['dbg_desired_pitch'] = desired_pitch
        info['dbg_desired_heading'] = desired_heading
        info['dbg_target_vt'] = target_vt
        info['dbg_dist3d'] = dist3d
        info['dbg_hdist'] = hdist
        info['dbg_reach_radius'] = state.reach_radius
        info['dbg_reach_now'] = reached_now.astype(jnp.int32)
        info['dbg_reached_count'] = state.reached
        info['plane_status_before'] = state.plane_state.status[0]
        info['time_before'] = state.time

        info['reward_distance']  = self._reward_distance(state, params, agent_id=0, scale=1.0)
        info['reward_alignment'] = self._reward_alignment(state, params, agent_id=0, scale=0.3)
        info['reward_speed_profile'] = self._reward_speed_profile(state, params, agent_id=0, scale=0.1)
        info['reward_reach_bonus'] = self._reward_reach_bonus(state, params, agent_id=0, bonus=3.0)
        info['penalty_crash'] = self._penalty_crash(state, params, agent_id=0, pen=-5.0)
        info['reach_radius'] = state.reach_radius
        info['reached_count'] = state.reached

        # 达成则推进到圆上的下一个离散航点（逻辑不变）
        def on_reach(_):
            if params.use_vertical_loop:
                new_idx = state.loop_idx + 1
                full = new_idx >= params.loop_points_per_circle
                new_idx = jnp.where(full, 0, new_idx)
                c_n = jnp.where(full, state.loop_center_n + params.loop_forward_north, state.loop_center_n)
                c_e = state.loop_center_e
                c_a = state.loop_center_alt
                R = params.loop_radius
                dphi = 2 * jnp.pi / params.loop_points_per_circle
                dir_ = jnp.sign(jnp.asarray(params.loop_direction, dtype=jnp.float32))
                phi0 = jnp.deg2rad(params.loop_phase0_deg)
                phi = phi0 + dir_ * (new_idx.astype(jnp.float32)) * dphi
                n_next = c_n - R * jnp.cos(phi)
                e_next = c_e
                a_next = c_a + R * jnp.sin(phi)
                wp = jnp.array([n_next, e_next, a_next])
                reach_radius = jnp.maximum(200.0, state.reach_radius * params.reach_radius_decay)
                return state.replace(
                    waypoint=wp,
                    reach_radius=reach_radius,
                    reached=state.reached + 1,
                    loop_center_n=c_n,
                    loop_center_e=c_e,
                    loop_center_alt=c_a,
                    loop_idx=new_idx,
                )
            else:
                # 其它模式原样
                reach_radius = jnp.maximum(100.0, state.reach_radius * params.reach_radius_decay)
                difficulty = state.difficulty + 1
                min_turn_deg = params.min_turn_deg_init + params.min_turn_deg_step * difficulty
                min_turn_rad = jnp.deg2rad(jnp.clip(min_turn_deg, 0.0, 170.0))
                wp = _sample_waypoint(key, jnp.array([pn, pe, pa]), params, min_turn_rad=min_turn_rad, current_yaw=state.plane_state.yaw[0])
                return state.replace(waypoint=wp, reach_radius=reach_radius, reached=state.reached + 1, difficulty=difficulty)

        def on_keep(_):
            return state

        timeout = state.time >= params.max_steps * params.sim_freq / params.agent_interaction_steps
        crashed = (state.plane_state.status[0] == 2)
        enough  = (state.reached >= params.max_waypoints)
        info['dbg_timeout'] = timeout
        info['dbg_crashed'] = crashed
        info['dbg_enough']  = enough

        # 写回（cmd_* 仅作日志同步）
        state = state.replace(cmd_heading=cmd_heading, cmd_pitch=cmd_pitch, cmd_vt=cmd_vt, target_heading=desired_heading, target_pitch=desired_pitch)
        state = jax.lax.cond(reached_now, on_reach, on_keep, operand=None)
        return state, info

    # 建议放在 _step_task 之后或终止条件之前
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        # 目标（由航点决定）
        dn = state.waypoint[0] - state.plane_state.north
        de = state.waypoint[1] - state.plane_state.east
        da = state.waypoint[2] - state.plane_state.altitude
        hdist = jnp.sqrt(jnp.maximum(dn * dn + de * de, 1e-6))
        dist3d = jnp.sqrt(hdist * hdist + da * da)

        desired_heading = _bearing(dn, de)                  # [-pi, pi]
        desired_pitch   = _desired_pitch(da, hdist)         # [-pi/2, pi/2]

        if params.s_altitude_lock:
            desired_pitch = 0.0

        # 距离分段的目标空速：远快近慢 / S 形 / 筋斗恒速
        if params.use_s_curve:
            target_vt = params.s_target_vt
        elif params.use_vertical_loop:
            target_vt = state.plane_state.vt  # 自由速度
        else:
            vt_far = params.max_vt * 0.9
            vt_near = params.min_vt * 1.2
            blend = jnp.clip(dist3d / 15000.0, 0.0, 1.0)
            target_vt = vt_near * (1.0 - blend) + vt_far * blend

        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha, beta = state.plane_state.alpha, state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        # _get_obs() 同步成和训练一致的顺序与夹限
        obs = jnp.vstack((
            _wrap_pi(yaw - desired_heading),
            _wrap_pi(pitch - desired_pitch),
            (vt - target_vt) / 340.0,
            altitude / 5000.0,
            vt / 340.0,
            jnp.sin(roll),  jnp.cos(roll),
            jnp.sin(pitch), jnp.cos(pitch),
            jnp.sin(alpha), jnp.cos(alpha),
            jnp.sin(beta),  jnp.cos(beta),
            P, Q, R
        ))
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        low = jnp.array([
            -jnp.pi, -jnp.pi, -2.0,
            0.0,
            0.0,            # norm_vt
            -1., -1., -1., -1.,
            -1., -1., -1., -1.,
            -10., -10., -10.
        ]).reshape(-1, 1)
        high = jnp.array([
            jnp.pi, jnp.pi, 2.0,
            5.0,
            2.0,            # norm_vt
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            10., 10., 10.
        ]).reshape(-1, 1)
        obs = jnp.clip(obs, low, high)


        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    # ---------- 终止条件 ----------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _term_timeout(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID):
        # 改成浮点计算
        limit = (jnp.asarray(params.max_steps, jnp.float32)
                * jnp.asarray(params.sim_freq, jnp.float32)
                / jnp.asarray(params.agent_interaction_steps, jnp.float32))
        done = jnp.asarray(state.time, jnp.float32) >= limit
        return done, jnp.array(False)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _term_crashed(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID):
        crashed = state.plane_state.status[agent_id] == 2
        return crashed, jnp.array(False)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _term_reached_enough(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID):
        done = state.reached >= params.max_waypoints
        success = done  # 只有真正达成条件时才记为成功
        return done, success

    # ---------- 奖励 ----------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_distance(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID, scale: float = 1.0):
        dn = state.waypoint[0] - state.plane_state.north[0]
        de = state.waypoint[1] - state.plane_state.east[0]
        da = state.waypoint[2] - state.plane_state.altitude[0]
        dist = jnp.sqrt(dn*dn + de*de + da*da)
        # 距离越小奖励越高（取负距离并归一）
        return scale * (-dist / 10000.0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_alignment(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID, scale: float = 0.3):
        dn = state.waypoint[0] - state.plane_state.north[0]
        de = state.waypoint[1] - state.plane_state.east[0]
        da = state.waypoint[2] - state.plane_state.altitude[0]
        hdist = jnp.sqrt(jnp.maximum(dn*dn + de*de, 1e-6))
        desired_heading = _bearing(dn, de)
        desired_pitch = _desired_pitch(da, hdist)
        yaw, pitch = state.plane_state.yaw[0], state.plane_state.pitch[0]
        # 指向性奖励（高斯）
        align_h = jnp.exp(-((_wrap_pi(yaw - desired_heading))/(jnp.pi/8))**2)
        align_p = jnp.exp(-((_wrap_pi(pitch - desired_pitch))/(jnp.pi/12))**2)
        return scale * (0.5 * align_h + 0.5 * align_p)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_speed_profile(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID, scale: float = 0.1):
        # 简单的速度窗口：远快近慢
        dn = state.waypoint[0] - state.plane_state.north[0]
        de = state.waypoint[1] - state.plane_state.east[0]
        da = state.waypoint[2] - state.plane_state.altitude[0]
        dist3d = jnp.sqrt(dn*dn + de*de + da*da)
        vt_far = params.max_vt * 0.9
        vt_near = params.min_vt * 1.2
        blend = jnp.clip(dist3d / 15000.0, 0.0, 1.0)
        target_vt = vt_near * (1.0 - blend) + vt_far * blend
        vt = state.plane_state.vt[0]
        return scale * jnp.exp(-((vt - target_vt)/30.0)**2)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_reach_bonus(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID, bonus: float = 3.0):
        # 在 LogWrapper 里统计不到达事件，这里用潜在奖励：达到半径内给小额正奖
        dn = state.waypoint[0] - state.plane_state.north[0]
        de = state.waypoint[1] - state.plane_state.east[0]
        da = state.waypoint[2] - state.plane_state.altitude[0]
        dist = jnp.sqrt(dn*dn + de*de + da*da)
        return jnp.where(dist <= state.reach_radius, bonus, 0.0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _penalty_crash(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID, pen: float = -5.0):
        crashed = state.plane_state.status[agent_id] == 2
        return jnp.where(crashed, pen, 0.0)

    # 新增：Nz过载惩罚
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_overload_penalty(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID, scale: float = -0.1, nz_lim: float = 8.0):
        nz = jnp.abs(state.plane_state.az[0])
        penalty = jnp.clip(nz - nz_lim, 0.0)**2
        return scale * penalty

    # 新增：相位误差奖励（筋斗专用）
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_phase_error(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID, scale: float = 0.2):
        def _phase_reward():
            # 计算当前相位误差（简化：用当前/目标俯仰近似相位）
            desired_pitch = _desired_pitch(state.waypoint[2] - state.plane_state.altitude[0], 1.0)  # 归一 hdist
            actual_pitch = state.plane_state.pitch[0]
            err = _wrap_pi(actual_pitch - desired_pitch)
            return jnp.exp(-(err / (jnp.pi / 6))**2)  # 高斯奖励，sigma=30°

        return jnp.where(params.use_vertical_loop, scale * _phase_reward(), 0.0)

    # 新增：超速惩罚
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_speed_penalty(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID, scale: float = -0.05):
        vt = state.plane_state.vt[0]
        target_vt = params.loop_target_vt if params.use_vertical_loop else params.max_vt * 0.8  # 通用默认
        penalty = jnp.clip(vt - target_vt, 0.0)**2 / (params.max_vt**2)
        return scale * penalty

    # # ---------- 日志回调（wandb） ----------
    # def train_callback(self, metric: chex.Array, wandb_run: Any, train_mode: bool):
    #     if wandb_run is None:
    #         return
    #     # 训练步数
    #     env_steps = int(metric["update_steps"]) if "update_steps" in metric else None

    #     # 奖励分量均值（按训练管线聚合的字段名做健壮兼容）
    #     def log_if_exists(prefix: str, keys):
    #         payload = {}
    #         for k in keys:
    #             v = None
    #             if k in metric:
    #                 v = metric[k]
    #             elif "info_mean" in metric and isinstance(metric["info_mean"], dict) and k in metric["info_mean"]:
    #                 v = metric["info_mean"][k]
    #             elif "infos" in metric and isinstance(metric["infos"], dict) and k in metric["infos"]:
    #                 v = metric["infos"][k]
    #             if v is not None:
    #                 try:
    #                     payload[f"{prefix}/{k}"] = float(v)
    #                 except Exception:
    #                     pass
    #         if payload:
    #             wandb_run.log(payload, step=env_steps)

    #     # 奖励分量 & 任务相关
    #     log_if_exists("reward", ["r_dist", "r_align", "r_speed", "r_bonus", "r_crash"])
    #     log_if_exists("waypoint", ["dist_to_wp", "hdist_to_wp", "reach_radius", "reached_count"])

# if __name__ == "__main__":

