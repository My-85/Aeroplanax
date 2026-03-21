# /home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/aeroplanax_waypoint.py
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
        obs, dones = x
        embedding = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)
        pi_throttle = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_aileron  = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_rudder   = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)

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
    min_altitude: float = 10000.0
    max_vt: float = 340.0
    min_vt: float = 100.0

    # 航点范围（相对初始中心）
    #=================================================#
    # wp_min_xy: float = 1000.0

    wp_min_xy : float  = 1500.0           # 允许更近的平面位移
    #=================================================#
    wp_max_xy : float  = 30000.0
    wp_min_alt: float  = 1000.0
    wp_max_alt: float  = 15000.0

    #=================================================#
    # 航点判定与难度提升
    # reach_radius_init: float = 800.0
    # reach_radius_decay: float = 0.9

    # 先放宽达成判定
    reach_radius_init: float = 1500.0   # 初始半径放大
    reach_radius_decay: float = 0.95    # 衰减慢一点
    #=================================================#

    max_waypoints: int = 100 # 达成目标航点个数就终止（设置一个很大的数）

    #=================================================#
    # min_turn_deg_init: float = 20.0
    # min_turn_deg_step: float = 10.0  # 每达成一个航点提升最小转向角

    min_turn_deg_init: float = 10.0     # 初始最小转弯角小一点
    min_turn_deg_step: float = 5.0      # 难度上升慢一点
    #=================================================#

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

    @classmethod
    def create(cls, env_state: EnvState, hstate, waypoint, reached, reach_radius, difficulty):
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
        )
# ========== 工具 ==========
def _wrap_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

def _bearing(north, east):
    return jnp.arctan2(east, north)

def _desired_pitch(d_alt, h_dist):
    # return jnp.arctan2(-d_alt, jnp.maximum(h_dist, 1e-6)) # 当航点在上方（d_alt > 0）时，正确的俯仰目标应该是正的；但你的实现用 -d_alt，会让飞机向下俯冲，高度误差越飞越大 → 永远到不了航点，reached_cnt 一直是 0。
    return jnp.arctan2(d_alt, jnp.maximum(h_dist, 1e-6))
    # 改用正的 d_alt 即可。
    # 当航点在上方（d_alt > 0）时，飞机向上爬升，俯仰目标为正；
    # 当航点在下方（d_alt < 0）时，飞机向下俯冲，俯仰目标为负。
    # 这样改后，飞机会根据航点高度自动调整俯仰姿态，不再“越飞越远”。

# 替换 _controller_obs() 为训练同序
def _controller_obs(state: fighterplane.FighterPlaneState, target_pitch, target_heading, target_vt):
    altitude = state.altitude
    roll, pitch, yaw = state.roll, state.pitch, state.yaw
    vt = state.vt
    alpha, beta = state.alpha, state.beta
    P, Q, R = state.P, state.Q, state.R
    obs = jnp.vstack((
        _wrap_pi(yaw - target_heading),           # 0
        _wrap_pi(pitch - target_pitch),           # 1
        (vt - target_vt) / 340.0,                 # 2
        altitude / 5000.0,                        # 3
        vt / 340.0,                               # 4  <== 注意：norm_vt 放这里
        jnp.sin(roll),  jnp.cos(roll),            # 5,6
        jnp.sin(pitch), jnp.cos(pitch),           # 7,8
        jnp.sin(alpha), jnp.cos(alpha),           # 9,10
        jnp.sin(beta),  jnp.cos(beta),            # 11,12
        P, Q, R                                    # 13,14,15
    ))
    return obs  # (16, B)


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
        ]
        self.is_potential = [False, False, False, True, False]

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
            jnp.zeros((1, self.cfg["NUM_ENVS"] * self.cfg["NUM_ACTORS"], 16)),
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
        # 评测模式：用内置基线产生动作
        def use_internal(_):
            # 仅按第0个智能体计算（当前任务是单机）
            pn, pe, pa = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
            dn, de, da = state.waypoint[0] - pn, state.waypoint[1] - pe, state.waypoint[2] - pa
            hdist = jnp.sqrt(dn * dn + de * de)
            dist3d = jnp.sqrt(hdist * hdist + da * da)
            desired_heading = _bearing(dn, de)
            desired_pitch = _desired_pitch(da, hdist)
            vt_far = self.default_params.max_vt * 0.9
            vt_near = self.default_params.min_vt * 1.2
            blend = jnp.clip(dist3d / 15000.0, 0.0, 1.0)
            target_vt = vt_near * (1.0 - blend) + vt_far * blend

            if self.default_params.use_high_level_action:
                ah = actions[self.agents[0]]["d_heading"]
                ap = actions[self.agents[0]]["d_pitch"]
                asv= actions[self.agents[0]]["d_speed"]
                # 把离散索引映射到实际增量
                def map_bin(idx, bins, lo, hi): 
                    step = (hi - lo) / (bins - 1)
                    return lo + idx * step
                d_head = map_bin(ah, self.default_params.hl_bins_heading, -jnp.deg2rad(30.0), jnp.deg2rad(30.0))
                d_pitch= map_bin(ap, self.default_params.hl_bins_pitch,   -jnp.deg2rad(15.0), jnp.deg2rad(15.0))
                d_v    = map_bin(asv,self.default_params.hl_bins_speed,   -30.0, 30.0)
                desired_heading = _wrap_pi(desired_heading + d_head)
                desired_pitch   = jnp.clip(desired_pitch + d_pitch, -jnp.deg2rad(45), jnp.deg2rad(45))
                target_vt       = jnp.clip(target_vt + d_v, self.default_params.min_vt, self.default_params.max_vt)
            # 然后构造 controller_obs → 前向 baseline → 采样四通道 → 推进

            # 前向基线
            last_obs = _controller_obs(state.plane_state, desired_pitch, desired_heading, target_vt).T  # (B,16)
            last_done = jnp.zeros((1,), dtype=bool)
            ac_in = (last_obs[None, :], last_done[None, :])  # (1,B,16), (1,B)
            hstate, pi, _ = self.controller.apply(self.controller_params, state.hstate, ac_in)
            pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

            # 采样离散动作 -> 连续四通道
            key1, key2, key3, key4 = jax.random.split(key, 4)
            a_th  = pi_throttle.sample(seed=key1)
            a_elv = pi_elevator.sample(seed=key2)
            a_ail = pi_aileron.sample(seed=key3)
            a_rud = pi_rudder.sample(seed=key4)
            a = jnp.concatenate([a_th[:, :, None], a_elv[:, :, None], a_ail[:, :, None], a_rud[:, :, None]], axis=-1).squeeze(0)
            a = jax.vmap(self._decode_discrete_actions)(a)  # (B,4)
            ctrl = jax.vmap(fighterplane.FighterPlaneControlState.create)(a)
            new_state = state.replace(hstate=hstate)
            return new_state, ctrl

        # 训练模式：走父类的离散动作解码（外部 PPO 驱动）
        def use_external(_):
            return super(AeroPlanaxWaypointEnv, self)._decode_actions(key, init_state, state, actions)

        return jax.lax.cond(self.use_internal_baseline, use_internal, use_external, operand=None)
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

        # 采样首个航点（在前方）且与当前高度至少 min_alt_sep 分离
        base_wp_n = s.plane_state.north[0] + 10000.0
        base_wp_e = s.plane_state.east[0]
        sign = jax.random.choice(key_sign, jnp.array([-1.0, 1.0]))
        d_alt = jax.random.uniform(key_d, shape=(1,), minval=params.min_alt_sep, maxval=3000.0) * sign
        wp_alt = jnp.clip(alt0 + d_alt, params.min_altitude, params.max_altitude)
        wp = jnp.array([base_wp_n, base_wp_e, wp_alt[0]])
        # 隐藏态
        if self.controller_type == "lstm":
            hstate = ScannedLSTM.initialize_carry(1, self.cfg["GRU_HIDDEN_DIM"])
        else:
            hstate = ScannedRNN.initialize_carry(1, self.cfg["GRU_HIDDEN_DIM"])
        return WaypointTaskState.create(
            s,
            hstate=hstate,
            waypoint=wp,
            reached=jnp.array(0),
            reach_radius=jnp.array(params.reach_radius_init),
            difficulty=jnp.array(0),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(self, key: chex.PRNGKey, state: WaypointTaskState, params: WaypointTaskParams) -> WaypointTaskState:
        key, ksign, kdelta = jax.random.split(key, 3)
        base_wp_n = state.plane_state.north[0] + 12000.0
        base_wp_e = state.plane_state.east[0]
        sign = jax.random.choice(ksign, jnp.array([-1.0, 1.0]))
        d_alt = jax.random.uniform(kdelta, shape=(1,), minval=params.min_alt_sep, maxval=3000.0) * sign
        wp_alt = jnp.clip(state.plane_state.altitude[0] + d_alt, params.min_altitude, params.max_altitude)
        wp = jnp.array([base_wp_n, base_wp_e, wp_alt[0]])
        return state.replace(waypoint=wp, reached=jnp.array(0), reach_radius=jnp.array(params.reach_radius_init), difficulty=jnp.array(0))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],  # 外部动作无效，忽略
        params: WaypointTaskParams,
    ) -> Tuple[WaypointTaskState, Dict[str, Any]]:
        # 目标（由航点决定）
        pn, pe, pa = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
        dn, de, da = state.waypoint[0] - pn, state.waypoint[1] - pe, state.waypoint[2] - pa
        hdist = jnp.sqrt(dn**2 + de**2)
        dist3d = jnp.sqrt(hdist**2 + da**2)
        desired_heading = _bearing(dn, de)
        desired_pitch = _desired_pitch(da, hdist)

        # 距离分段的速度目标（远快近慢）
        vt_far = params.max_vt * 0.9
        vt_near = params.min_vt * 1.2
        blend = jnp.clip(dist3d / 15000.0, 0.0, 1.0)
        target_vt = vt_near * (1.0 - blend) + vt_far * blend

        # 写指标
        info['dist_to_wp'] = dist3d
        info['hdist_to_wp'] = hdist

        # 航点达成检测
        reached_now = dist3d <= state.reach_radius
        info['reached_this_step'] = reached_now

        #======================================================================#
        # ==== 调试字段（供渲染端完整记录）====
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
        #======================================================================#

        # 记录奖励分量（用于可视化）
        info['reward_distance']  = self._reward_distance(state, params, agent_id=0, scale=1.0)
        info['reward_alignment'] = self._reward_alignment(state, params, agent_id=0, scale=0.3)
        info['reward_speed_profile'] = self._reward_speed_profile(state, params, agent_id=0, scale=0.1)
        info['reward_reach_bonus'] = self._reward_reach_bonus(state, params, agent_id=0, bonus=3.0)
        info['penalty_crash'] = self._penalty_crash(state, params, agent_id=0, pen=-5.0)
        info['reach_radius'] = state.reach_radius
        info['reached_count'] = state.reached

        # 达成则生成新航点 & 提升难度
        def on_reach(_):
            # 提升难度：半径衰减、最小转向角提高
            reach_radius = jnp.maximum(100.0, state.reach_radius * params.reach_radius_decay)
            difficulty = state.difficulty + 1
            min_turn_deg = params.min_turn_deg_init + params.min_turn_deg_step * difficulty
            min_turn_rad = jnp.deg2rad(jnp.clip(min_turn_deg, 0.0, 170.0))
            # 相对当前位置采样新航点（要求大转向）
            wp = _sample_waypoint(key, jnp.array([pn, pe, pa]), params, min_turn_rad=min_turn_rad, current_yaw=state.plane_state.yaw[0])
            return state.replace(
                waypoint=wp,
                reach_radius=reach_radius,
                reached=state.reached + 1,
                difficulty=difficulty,
            )

        def on_keep(_):
            return state

        #======================================================================#
        timeout = state.time >= params.max_steps * params.sim_freq / params.agent_interaction_steps
        crashed = (state.plane_state.status[0] == 2)
        enough  = (state.reached >= params.max_waypoints)
        info['dbg_timeout'] = timeout
        info['dbg_crashed'] = crashed
        info['dbg_enough']  = enough
        #======================================================================#

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

        # 距离分段的目标空速：远快近慢
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
        done = state.time >= params.max_steps * params.sim_freq / params.agent_interaction_steps
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

