# /home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/aeroplanax_waypoint_vertical_loop.py
import os
import functools
from typing import Dict, Optional, Sequence, Any, Tuple
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
from .core.simulators.fighterplane.dynamics import atmos  # 可留作扩展（当前未强依赖）

# ========== Baseline 控制器（RNN / LSTM） ==========
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
        obs, dones = x  # (T,B,16), (T,B)

        # 前端 MLP
        emb = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        emb = activation(emb)

        # GRU（按时间 scan）
        hidden, emb = ScannedRNN()(hidden, (emb, dones))

        # trunk
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        # actor（四个离散通道）
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2.0), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)
        pi_throttle = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_aileron  = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_rudder   = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))

        # critic
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2.0), bias_init=constant(0.0))(nn_fc2)
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
    max_altitude: float = 40000.0
    min_altitude: float = 2000.0
    max_vt: float = 360.0
    min_vt: float = 100.0

    # 航点范围
    wp_min_xy: float = 1500.0
    wp_max_xy: float = 30000.0
    wp_min_alt: float = 1000.0
    wp_max_alt: float = 15000.0

    # 难度爬升
    min_turn_deg_init: float = 10.0
    min_turn_deg_step: float = 5.0

    # --- S 形机动（可开关） ---
    use_s_curve: bool = struct.field(pytree_node=False, default=False)
    s_amplitude: float = 2000.0
    s_half_period_north: float = 8000.0
    s_points_per_half: int = 12
    s_altitude_lock: bool = struct.field(pytree_node=False, default=True)
    s_target_vt: float = struct.field(pytree_node=False, default=250.0)

    # --- 垂直回环（筋斗） ---
    use_vertical_loop: bool = struct.field(pytree_node=False, default=False)
    loop_radius: float = 2500.0
    loop_points_per_circle: int = 72
    loop_forward_north: float = 4000.0
    loop_target_vt: float = struct.field(pytree_node=False, default=250.0)
    loop_phase0_deg: float = struct.field(pytree_node=False, default=120.0)
    loop_direction: int = struct.field(pytree_node=False, default=1)   # +1 顺时针，-1 逆时针
    loop_pitch_limit_deg: float = struct.field(pytree_node=False, default=55.0)
    loop_tilt_deg: float = struct.field(pytree_node=False, default=0.0)
    loop_enter_offset: float = struct.field(pytree_node=False, default=3000.0)
    loop_floor_margin: float = struct.field(pytree_node=False, default=800.0)

    # 起始状态（可选；为 None 则沿用当前逻辑）
    start_north: Optional[float] = struct.field(pytree_node=False, default=None)
    start_east:  Optional[float] = struct.field(pytree_node=False, default=None)
    start_alt:   Optional[float] = struct.field(pytree_node=False, default=None)
    start_yaw_deg: Optional[float] = struct.field(pytree_node=False, default=None)
    start_vt:    Optional[float] = struct.field(pytree_node=False, default=None)


    # 航点判定
    reach_radius_init: float = 600.0
    reach_radius_decay: float = 1.0
    max_waypoints: int = 100

    # baseline 控制器
    baseline_type: str = struct.field(pytree_node=False, default="rnn")
    baseline_seed: int = 42
    baseline_hidden: int = 128
    baseline_fc: int = 128
    baseline_loaddir: str = struct.field(pytree_node=False, default="")
    action_dims: Sequence[int] = struct.field(pytree_node=False, default=(31, 41, 41, 41))
    use_internal_baseline: bool = struct.field(pytree_node=False, default=True)

    # 高层动作（如后续需要）
    use_high_level_action: bool = struct.field(pytree_node=False, default=False)
    hl_bins_heading: int = struct.field(pytree_node=False, default=17)
    hl_bins_pitch: int = struct.field(pytree_node=False, default=17)
    hl_bins_speed: int = struct.field(pytree_node=False, default=9)

    # 垂直最小分离
    min_alt_sep: float = 500.0

    # 向量场回正项
    vf_k_radial: float = struct.field(pytree_node=False, default=1.50)  # 径向回正系数
    vf_k_plane:  float = struct.field(pytree_node=False, default=0.50)  # 离面回正系数    

@struct.dataclass
class WaypointTaskState(EnvState):
    hstate: jnp.ndarray
    waypoint: jnp.ndarray
    reached: jnp.ndarray
    reach_radius: jnp.ndarray
    difficulty: jnp.ndarray
    time: jnp.ndarray

    s_origin_n: jnp.ndarray
    s_origin_e: jnp.ndarray

    # 筋斗（垂直回环）
    loop_center_n: jnp.ndarray
    loop_center_e: jnp.ndarray
    loop_center_alt: jnp.ndarray
    loop_idx: jnp.ndarray

    cmd_heading: jnp.ndarray
    cmd_pitch: jnp.ndarray
    cmd_vt: jnp.ndarray

    target_heading: jnp.ndarray
    target_pitch: jnp.ndarray

    loop_ref_heading: jnp.ndarray
    loop_wps: jnp.ndarray

    @classmethod
    def create(cls, env_state: EnvState, hstate, waypoint, reached, reach_radius, difficulty,
               s_origin_n=jnp.array(0.0), s_origin_e=jnp.array(0.0),
               loop_center_n=jnp.array(0.0), loop_center_e=jnp.array(0.0),
               loop_center_alt=jnp.array(0.0), loop_idx=jnp.array(0),
               target_heading=jnp.array(0.0), target_pitch=jnp.array(0.0),
               loop_ref_heading=jnp.array(0.0), loop_wps=jnp.zeros((0, 3), dtype=jnp.float32)):
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
            loop_ref_heading=loop_ref_heading,
            loop_wps=loop_wps,
        )

# ========== 工具 ==========

# ===================== 新增：四元数工具函数 =====================
def _q_normalize(q):
    q = jnp.asarray(q, jnp.float32)
    n = jnp.linalg.norm(q) + 1e-9
    return q / n

def _q_conj(q):
    # q = [w, x, y, z]
    return jnp.array([q[0], -q[1], -q[2], -q[3]], dtype=jnp.float32)

def _q_mul(q1, q2):
    # Hamilton 乘法，q=[w,x,y,z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return jnp.array([w, x, y, z], dtype=jnp.float32)

def _q_from_axis_angle(axis, angle):
    # axis: 'x' | 'y' | 'z'
    h = 0.5 * angle
    s, c = jnp.sin(h), jnp.cos(h)
    if axis == 'x':
        return jnp.array([c, s, 0.0, 0.0], dtype=jnp.float32)
    if axis == 'y':
        return jnp.array([c, 0.0, s, 0.0], dtype=jnp.float32)
    # 'z'
    return jnp.array([c, 0.0, 0.0, s], dtype=jnp.float32)

def _q_from_euler_zyx(roll, pitch, yaw):
    # yaw-Z -> pitch-Y -> roll-X
    qz = _q_from_axis_angle('z', yaw)
    qy = _q_from_axis_angle('y', pitch)
    qx = _q_from_axis_angle('x', roll)
    return _q_normalize(_q_mul(_q_mul(qz, qy), qx))


def _wrap_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

# 计算筋斗圆的相位：n = c_n - R*cos(phi), alt = c_a + R*sin(phi)
def _loop_phase(center_n, center_alt, R, n, alt):
    x = jnp.clip((center_n - n) / jnp.maximum(R, 1e-6), -1e6, 1e6)
    y = jnp.clip((alt - center_alt) / jnp.maximum(R, 1e-6), -1e6, 1e6)
    return jnp.arctan2(y, x)

def _heading_axes(yaw):
    u_n = jnp.cos(yaw)
    u_e = jnp.sin(yaw)
    w_n = -u_e
    w_e =  u_n
    return u_n, u_e, w_n, w_e

def _tilted_b(yaw, tilt_rad):
    u_n, u_e, w_n, w_e = _heading_axes(yaw)
    b_n = jnp.sin(tilt_rad) * w_n
    b_e = jnp.sin(tilt_rad) * w_e
    b_a = jnp.cos(tilt_rad)
    return b_n, b_e, b_a

def _course_from_vel(vn, ve):
    # 航迹角 χ（由速度矢量确定的地速方位）
    return jnp.arctan2(ve, vn)

def _loop_basis(yaw_ref, tilt_rad):
    # 回环平面的三基：前向 u（水平）、倾斜“上” b（含垂直分量）、平面法向 n
    u = jnp.array([jnp.cos(yaw_ref), jnp.sin(yaw_ref), 0.0], dtype=jnp.float32)        # (u_n,u_e,0)
    b = jnp.array(_tilted_b(yaw_ref, tilt_rad), dtype=jnp.float32)                      # (b_n,b_e,b_a)
    n = jnp.cross(u, b)
    n = n / (jnp.linalg.norm(n) + 1e-9)
    return u, b, n

def _project_on_plane(v, n):
    # 向量 v 在法向 n 的平面投影
    return v - jnp.dot(v, n) * n

def _loop_vf_cmd(center, pos, yaw_ref, tilt_rad, R, dir_, k_r, k_w):
    """
    向量场引导：
    v_des ∝  切向 t_hat
           − k_r * ε_r * ρ_hat        （半径回正）
           − k_w * ε_w * n_plane_hat  （离面回正）
    最后把 v_des 归一化→得到 (χ_cmd, γ_cmd)
    """
    u, b, n_plane = _loop_basis(yaw_ref, tilt_rad)
    r  = pos - center

    # 环平面内分量 & 径向单位向量
    r_p      = _project_on_plane(r, n_plane)
    r_p_norm = jnp.linalg.norm(r_p) + 1e-9
    rho_hat  = r_p / r_p_norm

    # 在 (u,b) 基下的系数：r_p = r_u * u + r_b * b
    r_u = jnp.dot(r_p, u)
    r_b = jnp.dot(r_p, b)

    # 切向量（在平面内与半径垂直）：t ∝ (r_b * u - r_u * b)
    t = r_b * u - r_u * b
    t_hat = dir_ * t / (jnp.linalg.norm(t) + 1e-9)

    # # 误差
    # eps_r = (r_p_norm - R) / jnp.maximum(R, 1.0)   # 半径归一化偏差
    # eps_w = jnp.dot(r, n_plane)                    # 离面距离（米）

    # v_des = t_hat - k_r * eps_r * rho_hat - k_w * eps_w * n_plane

    # 误差
    eps_r_lin = (r_p_norm - R) / jnp.maximum(R, 1.0)         # 原线性
    eps_w_lin = jnp.dot(r, n_plane)                           # 离面距离（米）

    # 软饱和（捕获管道：半径的 30%R，离面的 15%R）
    eps_r = jnp.tanh(eps_r_lin / 0.30)
    eps_w = jnp.tanh((eps_w_lin / jnp.maximum(R, 1.0)) / 0.15)

    v_des = t_hat - k_r * eps_r * rho_hat - k_w * eps_w * n_plane

    v_hat = v_des / (jnp.linalg.norm(v_des) + 1e-9)

    chi_cmd   = jnp.arctan2(v_hat[1], v_hat[0])                             # 航迹角
    gamma_cmd = jnp.arctan2(v_hat[2], jnp.sqrt(v_hat[0]**2 + v_hat[1]**2))  # 飞行路径角
    return chi_cmd, gamma_cmd


def _build_loop_wps(center_n, center_e, center_a, yaw_ref, tilt_rad, R, N, dir_, phi0):
    N_int = int(N)
    dphi = 2.0 * jnp.pi / jnp.asarray(N_int, jnp.float32)
    k = jnp.arange(N_int, dtype=jnp.float32)
    phi = phi0 + dir_ * k * dphi

    u_n, u_e, _, _ = _heading_axes(yaw_ref)
    b_n, b_e, b_a = _tilted_b(yaw_ref, tilt_rad)

    n = center_n - R * jnp.cos(phi) * u_n + R * jnp.sin(phi) * b_n
    e = center_e - R * jnp.cos(phi) * u_e + R * jnp.sin(phi) * b_e
    a = center_a + R * jnp.sin(phi) * b_a
    return jnp.stack([n, e, a], axis=1)

def _loop_tangent(yaw_ref, tilt_rad, phi, dir_):
    u_n, u_e, _, _ = _heading_axes(yaw_ref)
    b_n, b_e, b_a  = _tilted_b(yaw_ref, tilt_rad)
    t_n = dir_ * (jnp.sin(phi) * u_n + jnp.cos(phi) * b_n)
    t_e = dir_ * (jnp.sin(phi) * u_e + jnp.cos(phi) * b_e)
    t_a = dir_ * (               jnp.cos(phi) * b_a)
    return t_n, t_e, t_a

def _bearing(north, east):
    return jnp.arctan2(east, north)

def _desired_pitch(d_alt, h_dist):
    # 注意：这里必须是正的 d_alt（航点在上方时俯仰应为正）
    return jnp.arctan2(d_alt, jnp.maximum(h_dist, 1e-6))

# ===================== 覆盖：基线观测构建（四元数误差版） =====================
def _controller_obs(state: fighterplane.FighterPlaneState, target_pitch, target_heading, target_vt):
    # 1) 当前四元数（优先使用状态里的 q0..q3；若没有可回退用欧拉生成）
    if hasattr(state, "q0"):
        q_cur = jnp.array([state.q0[0], state.q1[0], state.q2[0], state.q3[0]], dtype=jnp.float32)
        # 数值保护与归一化
        q_cur = _q_normalize(q_cur)
    else:
        q_cur = _q_from_euler_zyx(state.roll[0], state.pitch[0], state.yaw[0])

    # 2) 目标四元数（roll 目标为 0）
    q_tgt = _q_from_euler_zyx(0.0, target_pitch, target_heading) # Body2NED
    q_tgt_nb = _q_conj(q_tgt)                                     # NED2Body

    # 3) 四元数误差（w>=0，取向量部3维）
    q_err = _q_normalize(_q_mul(q_tgt_nb, _q_conj(q_cur))) # [w,x,y,z]
    q_err = jnp.where(q_err[0] < 0.0, -q_err, q_err)     # 消歧
    qv = jnp.clip(q_err[1:4], -1.0, 1.0)                 # (3,)

    # 4) 机体系下的目标方向 v_b（先在 NED 构出单位向量，再旋到 Body）
    c_th, s_th = jnp.cos(target_heading), jnp.sin(target_heading)
    c_ph, s_ph = jnp.cos(target_pitch),   jnp.sin(target_pitch)
    v_n = jnp.array([c_ph * c_th, c_ph * s_th, s_ph], dtype=jnp.float32)  # (3,)

    # 旋转 NED->Body：v_b = q_bn * (0,v_n) * q_bn^*
    # 注意 q_cur 在你代码里表示 q_BN（Body from NED），前面已规范化
    p   = jnp.array([0.0, v_n[0], v_n[1], v_n[2]], dtype=jnp.float32)
    qp  = _q_mul(q_cur, p)
    qpq = _q_mul(qp, _q_conj(q_cur))
    v_b = jnp.clip(qpq[1:4], -1.0, 1.0)  # (3,)

    # 5) 其它归一化量（与训练版保持一致）
    vt        = state.vt[0]
    altitude  = state.altitude[0]
    alpha     = state.alpha[0]
    beta      = state.beta[0]
    P, Q, R   = state.P[0], state.Q[0], state.R[0]

    norm_dvt  = (vt - target_vt) / 340.0
    norm_alt  = altitude / 5000.0
    norm_vt   = vt / 340.0

    alpha_sin, alpha_cos = jnp.sin(alpha), jnp.cos(alpha)
    beta_sin,  beta_cos  = jnp.sin(beta),  jnp.cos(beta)

    # 6) 按“训练 env 的 16D 顺序”拼接：
    # [ qv(3), dvt, alt/5000, vt/340, v_b(3), P,Q,R, sin(alpha),cos(alpha), sin(beta),cos(beta) ]
    obs = jnp.array([
        qv[0], qv[1], qv[2],             # 0-2
        norm_dvt,                        # 3
        norm_alt,                        # 4
        norm_vt,                         # 5
        v_b[0], v_b[1], v_b[2],          # 6-8
        P, Q, R,                          # 9-11
        alpha_sin, alpha_cos,            # 12-13
        beta_sin,  beta_cos              # 14-15
    ], dtype=jnp.float32).reshape(16, 1)

    # 7) 数值清理与裁剪（和训练 env 对齐）
    low  = jnp.array([-1., -1., -1., -2., 0., 0., -1., -1., -1., -10., -10., -10., -1., -1., -1., -1.], dtype=jnp.float32).reshape(16,1)
    high = jnp.array([ 1.,  1.,  1.,  2., 5., 2.,  1.,  1.,  1.,  10.,  10.,  10.,  1.,  1.,  1.,  1.], dtype=jnp.float32).reshape(16,1)
    obs  = jnp.clip(jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), low, high)
    return obs  # (16,1)


def _sample_waypoint(key, center_n_e_alt, params: WaypointTaskParams, min_turn_rad: float, current_yaw: float):
    key_xy, key_alt = jax.random.split(key)
    def sample_once(key):
        rxy = jax.random.uniform(key, shape=(2,), minval=-params.wp_max_xy, maxval=params.wp_max_xy)
        rxy = jnp.where(jnp.abs(rxy) < params.wp_min_xy, jnp.sign(rxy) * params.wp_min_xy, rxy)
        alt = jax.random.uniform(key_alt, minval=params.wp_min_alt, maxval=params.wp_max_alt)
        return rxy[0], rxy[1], alt

    nx, ex, alt = sample_once(key_xy)
    bearing = _bearing(nx, ex)
    ok = jnp.abs(_wrap_pi(bearing - current_yaw)) >= min_turn_rad
    nx = jnp.where(ok, nx, -nx)
    ex = jnp.where(ok, ex, -ex)
    return jnp.array([center_n_e_alt[0] + nx, center_n_e_alt[1] + ex, alt])

# ========== 环境 ==========
class AeroPlanaxWaypointEnv(AeroPlanaxEnv[WaypointTaskState, WaypointTaskParams]):
    def __init__(self, env_params: Optional[WaypointTaskParams] = None):
        super().__init__(env_params)
        self._default_params = env_params or WaypointTaskParams()

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
        self.use_internal_baseline = env_params.use_internal_baseline
        self._init_controller(env_params)

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        # 终止条件
        self.termination_conditions = [
            self._term_timeout,
            self._term_crashed,
            self._term_reached_enough
        ]

        self._los_Lmin_m = 300.0   # atan2(da, hdist) 的水平距离下限

    def _get_obs_size(self) -> int:
        return 16

    # ---------- spaces ----------
    def _get_individual_obs_space(self, i) -> spaces.Space:
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
            jnp.zeros((1, self.cfg["NUM_ENVS"] * self.cfg["NUM_ACTORS"])),
        )
        controller_params = self.controller.init(rng, init_h, init_x)

        tx = optax.adam(self.cfg["LR"])
        train_state = TrainState.create(apply_fn=self.controller.apply, params=controller_params, tx=tx)

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
            self.controller_params = train_state.params

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state: WaypointTaskState,
        state: WaypointTaskState,
        actions: Dict[AgentName, chex.Array],
    ):
        # 仅按第0个智能体计算
        pn, pe, pa = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
        wp = state.waypoint
        dn, de, da = wp[0] - pn, wp[1] - pe, wp[2] - pa
        hdist = jnp.sqrt(jnp.maximum(dn*dn + de*de, 1e-6))
        dist3d = jnp.sqrt(hdist*hdist + da*da)

        # # —— LOS & 指令链路（四元数友好）——
        # base_heading  = _bearing(dn, de)
        # desired_heading = jnp.where(self.default_params.use_vertical_loop, state.loop_ref_heading, base_heading)

        # 替换为（最近点 + 前视半步的切线航向）：
        base_heading = _bearing(dn, de)

        if self.default_params.use_vertical_loop:
            # =========================================================
            # 2. 强力姿态引导 (High-Gain Vector Field Guidance)
            # =========================================================
            R    = jnp.asarray(self.default_params.loop_radius, jnp.float32) # _step_task 中用 params
            dir_ = jnp.sign(jnp.asarray(self.default_params.loop_direction, jnp.float32)) # _step_task 中用 params
            tilt = jnp.deg2rad(self.default_params.loop_tilt_deg) # _step_task 中用 params
            
            # 构造向量 (标量->向量)
            center = jnp.array([state.loop_center_n, state.loop_center_e, state.loop_center_alt], jnp.float32)
            pn_s, pe_s, pa_s = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
            pos    = jnp.array([pn_s, pe_s, pa_s], jnp.float32)

            # --- 动态增益调优 ---
            # 关键点：爬升时 (Pitch > 10)，重力使得飞机严重掉半径。
            # 我们给一个巨大的径向纠偏增益 k_r = 4.0 (默认通常是 0.8)
            # 俯冲时恢复正常，以免修正过度
            pitch_rad = state.plane_state.pitch[0]
            pitch_deg = jnp.rad2deg(pitch_rad)
            is_climbing_steep = pitch_deg > 40.0
            k_r_val = jnp.where(is_climbing_steep, 0.2, 1.2)
            
            # 增大离面增益 k_w，防止左右偏离
            k_w_val = 1.0 

            chi_cmd, gamma_cmd = _loop_vf_cmd(
                center=center, pos=pos,
                yaw_ref=state.loop_ref_heading, tilt_rad=tilt,
                R=R, dir_=dir_,
                k_r=k_r_val,
                k_w=k_w_val,
            )

            # =========================================================
            # 3. 姿态补偿与保护 (Compensation & Protection)
            # =========================================================
            
            # A. Pitch 前馈 (Feedforward Bias)
            # 同样的逻辑：爬升时，VF 算出的只是切线，我们需要额外的迎角
            # 强行给一个 +15 度的 Target 偏置
            pitch_bias = jnp.where(is_climbing_steep, jnp.deg2rad(5.0), 0.0)
            desired_pitch = gamma_cmd + pitch_bias

            # B. 垂直段航向锁定 (Yaw Lock)
            # 当 Pitch 绝对值很大时，忽略 VF 的航向指令，锁定跑道方向
            # 扩大锁定范围到 +/- 50 度，避免进入高仰角后的偏航震荡
            is_vertical_zone = jnp.abs(pitch_deg) > 50.0
            
            desired_heading = jnp.where(
                is_vertical_zone, 
                state.loop_ref_heading, 
                chi_cmd
            )

            # C. 限制范围 (Clip)
            desired_pitch = jnp.clip(desired_pitch, jnp.deg2rad(-89.9), jnp.deg2rad(+89.9))
            
            #######################################################################################

        else:
            desired_heading = base_heading
            # desired_pitch = jnp.arctan2(da, hdist)
            hdist_sat = jnp.maximum(hdist, jnp.asarray(self._los_Lmin_m))
            desired_pitch = jnp.arctan2(da, hdist_sat)
            # desired_pitch = jnp.clip(desired_pitch, jnp.deg2rad(-89.0), jnp.deg2rad(+89.0))


        # 水平距离饱和（避免 90°尖峰）
        # hdist_sat = jnp.maximum(hdist, jnp.asarray(self._los_Lmin_m))
        # desired_pitch = jnp.arctan2(da, hdist_sat)
        # 限制范围
        desired_pitch = jnp.clip(desired_pitch, jnp.deg2rad(-89.0), jnp.deg2rad(+89.0))

        # 速度目标
        if self.default_params.use_vertical_loop:
            # target_vt = state.plane_state.vt[0]      # 自由速度
            # target_vt = jnp.asarray(self.default_params.loop_target_vt, jnp.float32)

            ###################################
            # =========================================================
            # 1. 极简动能策略 (Pure Pitch-based Kinetic Energy Management)
            # =========================================================
            # 获取标量 (注意：state.plane_state.* 是数组，需要 [0]；loop_center* 是标量)
            pitch_rad = state.plane_state.pitch[0]
            pitch_deg = jnp.rad2deg(pitch_rad)
            
            # 逻辑：只要机头没有大幅度朝下 (Pitch > -40)，就认为需要动力对抗重力或维持能量
            # 只有在明显的俯冲阶段 (Pitch < -40)，重力势能转化为动能时，才刹车
            is_power_phase = pitch_deg > -40.0
            
            vt_boost = 380.0  # 全力冲刺/爬升
            vt_brake = 180.0  # 俯冲刹车
            
            # 标量赋值，避免 Shape 错误
            target_vt_val = jnp.where(is_power_phase, vt_boost, vt_brake)
            target_vt = target_vt_val

            # === 修改结束 ===
            ###################################

        elif self.default_params.use_s_curve:
            target_vt = self.default_params.s_target_vt
        else:
            vt_far = self.default_params.max_vt * 0.9
            vt_near = self.default_params.min_vt * 1.2
            blend = jnp.clip(dist3d / 15000.0, 0.0, 1.0)
            target_vt = vt_near * (1.0 - blend) + vt_far * blend

        # S 形才锁高
        if self.default_params.use_s_curve and self.default_params.s_altitude_lock:
            desired_pitch = 0.0
            dist3d = hdist

        # baseline 观测（与训练一致）
        last_obs = _controller_obs(state.plane_state, desired_pitch, desired_heading, target_vt).T  # (B,16)
        last_done = jnp.zeros((1,), dtype=bool)
        ac_in = (last_obs[None, :], last_done[None, :])  # (1,B,16), (1,B)

        hstate, pi, _ = self.controller.apply(self.controller_params, state.hstate, ac_in)
        pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

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

    # ---------- 必要接口 ----------
    @property
    def default_params(self) -> WaypointTaskParams:
        return self._default_params

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(self, key: jax.Array, params: WaypointTaskParams) -> WaypointTaskState:
        # s = super()._init_state(key, params)
        # # 初始航向/速度/高度（四元数字段可选赋默认）
        # yaw = jnp.array([0.0])
        # q0 = jnp.array([1.0]); q3 = jnp.array([0.0])
        # key, key_vt, key_alt, key_sign, key_d = jax.random.split(key, 5)
        # vt0 = jax.random.uniform(key_vt, shape=(1,), minval=params.min_vt, maxval=params.max_vt)
        # alt0 = jax.random.uniform(key_alt, shape=(1,), minval=params.min_altitude, maxval=params.max_altitude)
        # s = s.replace(plane_state=s.plane_state.replace(yaw=yaw, vt=vt0, q0=q0, q3=q3, altitude=alt0))

        ###################################################################################
        s = super()._init_state(key, params)

        # ========= 起始姿态/速度 =========
        # yaw
        if params.start_yaw_deg is not None:
            yaw = jnp.array([jnp.deg2rad(params.start_yaw_deg)], dtype=jnp.float32)
        else:
            yaw = jnp.array([0.0], dtype=jnp.float32)

        # 四元数（与 yaw 一致，roll/pitch=0）
        q0 = jnp.array([1.0]); q1 = jnp.array([0.0]); q2 = jnp.array([0.0]); q3 = jnp.array([0.0])

        # 速度
        if params.start_vt is not None:
            vt0 = jnp.array([params.start_vt], dtype=jnp.float32)
        else:
            key, key_vt = jax.random.split(key)
            vt0 = jax.random.uniform(key_vt, shape=(1,),
                                    minval=params.min_vt, maxval=params.max_vt)

        # 高度
        if params.start_alt is not None:
            alt0 = jnp.array([params.start_alt], dtype=jnp.float32)
        else:
            key, key_alt = jax.random.split(key)
            alt0 = jax.random.uniform(key_alt, shape=(1,),
                                    minval=params.min_altitude, maxval=params.max_altitude)

        # 位置
        n0 = s.plane_state.north if params.start_north is None else jnp.array([params.start_north], dtype=jnp.float32)
        e0 = s.plane_state.east  if params.start_east  is None else jnp.array([params.start_east],  dtype=jnp.float32)

        s = s.replace(plane_state=s.plane_state.replace(
            north=n0, east=e0, altitude=alt0, vt=vt0, yaw=yaw, 
            q0=q0, q1=q1, q2=q2, q3=q3
        ))
        ###################################################################################

        # 隐藏态
        if self.controller_type == "lstm":
            hstate = ScannedLSTM.initialize_carry(1, self.cfg["GRU_HIDDEN_DIM"])
        else:
            hstate = ScannedRNN.initialize_carry(1, self.cfg["GRU_HIDDEN_DIM"])

        # —— 三种任务的首航点初始化 —— #
        if params.use_vertical_loop:
            n0, e0, a0 = s.plane_state.north[0], s.plane_state.east[0], s.plane_state.altitude[0]
            R    = float(self.default_params.loop_radius)
            N    = int(self.default_params.loop_points_per_circle)
            dir_ = jnp.sign(jnp.asarray(self.default_params.loop_direction, dtype=jnp.float32))
            phi0 = jnp.deg2rad(self.default_params.loop_phase0_deg)
            yaw0 = s.plane_state.yaw[0]   # 固定参考航向
            tilt = jnp.deg2rad(self.default_params.loop_tilt_deg)

            # 圆心：前移 (R + enter_offset)
            u_n, u_e, _, _ = _heading_axes(yaw0)
            center_shift = R + jnp.asarray(self.default_params.loop_enter_offset, jnp.float32)
            c_n = n0 + center_shift * u_n
            c_e = e0 + center_shift * u_e

            # 最低高度保护
            b_a = jnp.cos(tilt)
            floor_abs = self.default_params.min_altitude + self.default_params.loop_floor_margin
            c_a_min = floor_abs + R * jnp.abs(b_a)
            c_a = jnp.maximum(a0, c_a_min)

            # 整圈航点
            wps = _build_loop_wps(c_n, c_e, c_a, yaw0, tilt, R, N, dir_, phi0)

            # 以最近点为首点
            p0 = jnp.array([n0, e0, a0])
            d2 = jnp.sum((wps - p0[None, :]) ** 2, axis=1)
            k0 = jnp.argmin(d2)
            wp0 = wps[k0]

            dphi = 2.0 * jnp.pi / N
            reach_radius = jnp.array(jnp.minimum(self.default_params.reach_radius_init, R * dphi))
            difficulty   = jnp.array(0)

            return WaypointTaskState.create(
                s,
                hstate=hstate,
                waypoint=wp0,
                reached=jnp.array(0),
                reach_radius=reach_radius,
                difficulty=difficulty,
                s_origin_n=jnp.array(0.0),
                s_origin_e=jnp.array(0.0),
                loop_center_n=jnp.array(c_n),
                loop_center_e=jnp.array(c_e),
                loop_center_alt=jnp.array(c_a),
                loop_idx=jnp.asarray(k0),
            ).replace(
                cmd_heading=_bearing(wp0[0]-s.plane_state.north[0], wp0[1]-s.plane_state.east[0]),
                cmd_pitch=_desired_pitch(wp0[2]-s.plane_state.altitude[0],
                                         jnp.sqrt((wp0[0]-s.plane_state.north[0])**2+(wp0[1]-s.plane_state.east[0])**2)),
                cmd_vt=s.plane_state.vt[0],
                target_heading=_bearing(wp0[0]-s.plane_state.north[0], wp0[1]-s.plane_state.east[0]),
                target_pitch=_desired_pitch(wp0[2]-s.plane_state.altitude[0],
                                            jnp.sqrt((wp0[0]-s.plane_state.north[0])**2+(wp0[1]-s.plane_state.east[0])**2)),
            ).replace(loop_ref_heading=yaw0, loop_wps=wps)

        elif params.use_s_curve:
            n0 = s.plane_state.north[0]
            e0 = s.plane_state.east[0]
            a0 = s.plane_state.altitude[0]
            dn = params.s_half_period_north / params.s_points_per_half
            idx0 = 1
            n1 = n0 + idx0 * dn
            e1 = e0 + params.s_amplitude * jnp.sin(jnp.pi * (n1 - n0) / params.s_half_period_north)
            wp = jnp.array([n1, e1, a0])
            reach_radius = jnp.array(params.reach_radius_init)
            difficulty = jnp.array(idx0)

            return WaypointTaskState.create(
                s, hstate=hstate, waypoint=wp, reached=jnp.array(0),
                reach_radius=reach_radius, difficulty=difficulty,
                s_origin_n=jnp.array(n0), s_origin_e=jnp.array(e0),
            ).replace(
                cmd_heading=_bearing(wp[0]-s.plane_state.north[0], wp[1]-s.plane_state.east[0]),
                cmd_pitch=_desired_pitch(wp[2]-s.plane_state.altitude[0],
                                         jnp.sqrt((wp[0]-s.plane_state.north[0])**2+(wp[1]-s.plane_state.east[0])**2)),
                cmd_vt=s.plane_state.vt[0]
            )

        else:
            # 普通航点
            key, ksign, kdelta = jax.random.split(key, 3)
            base_wp_n = s.plane_state.north[0] + 12000.0
            base_wp_e = s.plane_state.east[0]
            sign = jax.random.choice(ksign, jnp.array([-1.0, 1.0]))
            d_alt = jax.random.uniform(kdelta, shape=(1,), minval=params.min_alt_sep, maxval=3000.0) * sign
            wp_alt = jnp.clip(s.plane_state.altitude[0] + d_alt, params.min_altitude, params.max_altitude)
            wp = jnp.array([base_wp_n, base_wp_e, wp_alt[0]])
            reach_radius = jnp.array(params.reach_radius_init)
            difficulty = jnp.array(0)

            return WaypointTaskState.create(
                s, hstate=hstate, waypoint=wp, reached=jnp.array(0),
                reach_radius=reach_radius, difficulty=difficulty
            ).replace(
                cmd_heading=_bearing(wp[0]-s.plane_state.north[0], wp[1]-s.plane_state.east[0]),
                cmd_pitch=_desired_pitch(wp[2]-s.plane_state.altitude[0],
                                         jnp.sqrt((wp[0]-s.plane_state.north[0])**2+(wp[1]-s.plane_state.east[0])**2)),
                cmd_vt=s.plane_state.vt[0]
            )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(self, key: chex.PRNGKey, state: WaypointTaskState, params: WaypointTaskParams) -> WaypointTaskState:
        if params.use_vertical_loop:
            n0, e0, a0 = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
            R    = float(self.default_params.loop_radius)
            N    = int(self.default_params.loop_points_per_circle)
            dir_ = jnp.sign(jnp.asarray(self.default_params.loop_direction, dtype=jnp.float32))
            phi0 = jnp.deg2rad(self.default_params.loop_phase0_deg)
            yaw0 = state.plane_state.yaw[0]
            tilt = jnp.deg2rad(self.default_params.loop_tilt_deg)

            u_n, u_e, _, _ = _heading_axes(yaw0)
            center_shift = R + jnp.asarray(self.default_params.loop_enter_offset, jnp.float32)
            c_n = n0 + center_shift * u_n
            c_e = e0 + center_shift * u_e

            b_a = jnp.cos(tilt)
            floor_abs = self.default_params.min_altitude + self.default_params.loop_floor_margin
            c_a_min = floor_abs + R * jnp.abs(b_a)
            c_a = jnp.maximum(a0, c_a_min)

            wps = _build_loop_wps(c_n, c_e, c_a, yaw0, tilt, R, N, dir_, phi0)
            p0 = jnp.array([n0, e0, a0])
            d2 = jnp.sum((wps - p0[None, :])**2, axis=1)
            k0 = jnp.argmin(d2)
            wp0 = wps[k0]

            dphi = 2.0 * jnp.pi / N
            reach_radius = jnp.array(jnp.minimum(self.default_params.reach_radius_init, R * dphi))
            difficulty   = jnp.array(0)

            return state.replace(
                waypoint=wp0, reached=jnp.array(0),
                reach_radius=reach_radius, difficulty=difficulty,
                loop_center_n=jnp.array(c_n), loop_center_e=jnp.array(c_e),
                loop_center_alt=jnp.array(c_a), loop_idx=jnp.asarray(k0)
            ).replace(
                cmd_heading=_bearing(wp0[0]-state.plane_state.north[0], wp0[1]-state.plane_state.east[0]),
                cmd_pitch=_desired_pitch(wp0[2]-state.plane_state.altitude[0],
                                         jnp.sqrt((wp0[0]-state.plane_state.north[0])**2+(wp0[1]-state.plane_state.east[0])**2)),
                cmd_vt=state.plane_state.vt[0],
                target_heading=_bearing(wp0[0]-state.plane_state.north[0], wp0[1]-state.plane_state.east[0]),
                target_pitch=_desired_pitch(wp0[2]-state.plane_state.altitude[0],
                                            jnp.sqrt((wp0[0]-state.plane_state.north[0])**2+(wp0[1]-state.plane_state.east[0])**2))
            ).replace(
                loop_ref_heading=yaw0, loop_wps=wps
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

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: WaypointTaskParams,
    ) -> Tuple[WaypointTaskState, Dict[str, Any]]:
        # 用预计算的 loop_wps 驱动当前 waypoint（非筋斗则 state.waypoint 不改）
        if params.use_vertical_loop:
            wp = state.loop_wps[state.loop_idx]
            state = state.replace(waypoint=wp)
        else:
            wp = state.waypoint

        pn, pe, pa = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
        dn, de, da = wp[0] - pn, wp[1] - pe, wp[2] - pa
        hdist = jnp.sqrt(jnp.maximum(dn*dn + de*de, 1e-6))
        dist3d = jnp.sqrt(hdist*hdist + da*da)

        # —— LOS & 指令链路 —— #
        # hdist_raw = jnp.sqrt(dn*dn + de*de)
        # hdist_sat = jnp.maximum(hdist_raw, jnp.asarray(self._los_Lmin_m))

        # base_heading = _bearing(dn, de)
        # desired_heading = jnp.where(params.use_vertical_loop, state.loop_ref_heading, base_heading)

        # 替换为（最近点 + 前视半步的切线航向）：
        base_heading = _bearing(dn, de)

        # if params.use_vertical_loop:
        #     R    = jnp.asarray(params.loop_radius, jnp.float32)
        #     N    = jnp.asarray(params.loop_points_per_circle, jnp.float32)
        #     dphi = 2.0 * jnp.pi / N
        #     dir_ = jnp.sign(jnp.asarray(params.loop_direction, jnp.float32))
        #     tilt = jnp.deg2rad(params.loop_tilt_deg)

        #     phi_curr = _loop_phase(state.loop_center_n, state.loop_center_alt, R, pn, pa)
        #     phi_look = phi_curr + dir_ * (0.5 * dphi)

        #     # t_n, t_e, _ = _loop_tangent(state.loop_ref_heading, tilt, phi_look, dir_)

        #     # 切向量 -> 飞行路径角 γ_cmd
        #     t_n, t_e, t_a = _loop_tangent(state.loop_ref_heading, tilt, phi_look, dir_)
        #     gamma_cmd = jnp.arctan2(t_a, jnp.sqrt(jnp.maximum(t_n*t_n + t_e*t_e, 1e-9)))

        #     # γ_cmd 直接作为俯仰指令的“几何参考”
        #     pitch_preclip = gamma_cmd
        #     desired_heading = jnp.arctan2(t_e, t_n)

        # else:
        #     desired_heading = base_heading
        #     pitch_preclip = jnp.arctan2(da, hdist_sat)

        # # pitch_preclip = jnp.arctan2(da, hdist_sat)
        # pitch_clip    = jnp.clip(pitch_preclip, jnp.deg2rad(-89.0), jnp.deg2rad(+89.0))

        # cmd_heading = desired_heading
        # cmd_pitch   = pitch_clip

        if params.use_vertical_loop:
            # =========================================================
            # 2. 强力姿态引导 (High-Gain Vector Field Guidance)
            # =========================================================
            R    = jnp.asarray(self.default_params.loop_radius, jnp.float32) # _step_task 中用 params
            dir_ = jnp.sign(jnp.asarray(self.default_params.loop_direction, jnp.float32)) # _step_task 中用 params
            tilt = jnp.deg2rad(self.default_params.loop_tilt_deg) # _step_task 中用 params
            
            # 构造向量 (标量->向量)
            center = jnp.array([state.loop_center_n, state.loop_center_e, state.loop_center_alt], jnp.float32)
            pn_s, pe_s, pa_s = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
            pos    = jnp.array([pn_s, pe_s, pa_s], jnp.float32)

            # --- 动态增益调优 ---
            # 关键点：爬升时 (Pitch > 10)，重力使得飞机严重掉半径。
            # 我们给一个巨大的径向纠偏增益 k_r = 4.0 (默认通常是 0.8)
            # 俯冲时恢复正常，以免修正过度
            pitch_rad = state.plane_state.pitch[0]
            pitch_deg = jnp.rad2deg(pitch_rad)
            is_climbing_steep = pitch_deg > 40.0
            k_r_val = jnp.where(is_climbing_steep, 0.2, 1.2)
            
            # 增大离面增益 k_w，防止左右偏离
            k_w_val = 1.0 

            chi_cmd, gamma_cmd = _loop_vf_cmd(
                center=center, pos=pos,
                yaw_ref=state.loop_ref_heading, tilt_rad=tilt,
                R=R, dir_=dir_,
                k_r=k_r_val,
                k_w=k_w_val,
            )

            # =========================================================
            # 3. 姿态补偿与保护 (Compensation & Protection)
            # =========================================================
            
            # A. Pitch 前馈 (Feedforward Bias)
            # 同样的逻辑：爬升时，VF 算出的只是切线，我们需要额外的迎角
            # 强行给一个 +15 度的 Target 偏置
            pitch_bias = jnp.where(is_climbing_steep, jnp.deg2rad(5.0), 0.0)
            desired_pitch = gamma_cmd + pitch_bias

            # B. 垂直段航向锁定 (Yaw Lock)
            # 当 Pitch 绝对值很大时，忽略 VF 的航向指令，锁定跑道方向
            # 扩大锁定范围到 +/- 50 度，避免进入高仰角后的偏航震荡
            is_vertical_zone = jnp.abs(pitch_deg) > 50.0
            
            desired_heading = jnp.where(
                is_vertical_zone, 
                state.loop_ref_heading, 
                chi_cmd
            )

            # C. 限制范围 (Clip)
            desired_pitch = jnp.clip(desired_pitch, jnp.deg2rad(-89.9), jnp.deg2rad(+89.9))
            
            #######################################################################################
        else:
            desired_heading = base_heading
            # pitch_preclip = jnp.arctan2(da, hdist)
            hdist_sat = jnp.maximum(hdist, jnp.asarray(self._los_Lmin_m))
            desired_pitch = jnp.arctan2(da, hdist_sat)
            desired_pitch = jnp.clip(desired_pitch, jnp.deg2rad(-89.0), jnp.deg2rad(+89.0))

        # pitch_clip = jnp.clip(pitch_preclip, jnp.deg2rad(-89.0), jnp.deg2rad(+89.0))
        cmd_heading = desired_heading
        cmd_pitch   = desired_pitch

        # 速度目标
        if params.use_vertical_loop:
            # target_vt = state.plane_state.vt[0]
            # target_vt = jnp.asarray(params.loop_target_vt, jnp.float32)

            ###################################################
            # =========================================================
            # 1. 极简动能策略 (Pure Pitch-based Kinetic Energy Management)
            # =========================================================
            # 获取标量 (注意：state.plane_state.* 是数组，需要 [0]；loop_center* 是标量)
            pitch_rad = state.plane_state.pitch[0]
            pitch_deg = jnp.rad2deg(pitch_rad)
            
            # 逻辑：只要机头没有大幅度朝下 (Pitch > -40)，就认为需要动力对抗重力或维持能量
            # 只有在明显的俯冲阶段 (Pitch < -40)，重力势能转化为动能时，才刹车
            is_power_phase = pitch_deg > -40.0
            
            vt_boost = 380.0  # 全力冲刺/爬升
            vt_brake = 180.0  # 俯冲刹车
            
            # 标量赋值，避免 Shape 错误
            target_vt_val = jnp.where(is_power_phase, vt_boost, vt_brake)
            target_vt = target_vt_val

            # === 修改结束 ===
            ###################################################

        elif params.use_s_curve:
            target_vt = params.s_target_vt
        else:
            vt_far = params.max_vt * 0.9
            vt_near = params.min_vt * 1.2
            blend = jnp.clip(dist3d / 15000.0, 0.0, 1.0)
            target_vt = vt_near * (1.0 - blend) + vt_far * blend

        if params.use_s_curve and params.s_altitude_lock:
            cmd_pitch = 0.0
            dist3d = hdist

        # 写 info（度量统一度/米）
        # rad2deg = (180.0 / jnp.pi)
        # info['dbg_hdist_raw_m'] = hdist_raw
        # info['dbg_hdist_sat_m'] = hdist_sat
        # info['dbg_da_m']        = da
        # info['dbg_gamma_los_raw_deg']    = gamma_los_raw * rad2deg
        # info['dbg_gamma_los_clip89_deg'] = gamma_los_clip89 * rad2deg
        # info['dbg_gamma_cmd_preclip_deg']= pitch_preclip * rad2deg
        # info['dbg_gamma_cmd_clip_deg']   = pitch_clip * rad2deg
        # info['dbg_gamma_cmd_rate_deg']   = pitch_clip * rad2deg
        # info['dbg_gamma_cmd_deg']        = pitch_clip * rad2deg

        info['cmd_heading'] = cmd_heading
        info['cmd_pitch']   = cmd_pitch
        info['cmd_vt']      = target_vt
        info['dist_to_wp']  = dist3d
        info['hdist_to_wp'] = hdist

        # yaw   = state.plane_state.yaw[0]
        # pitch = state.plane_state.pitch[0]
        # vt    = state.plane_state.vt[0]
        # info['obs_norm_dheading'] = _wrap_pi(yaw - cmd_heading)
        # info['obs_norm_dpitch']   = _wrap_pi(pitch - cmd_pitch)
        # info['obs_norm_dvt']      = (vt - target_vt) / 340.0

        # 现在：用速度向量算航迹角 chi = atan2(ve, vn)
        vn = state.plane_state.vel_x[0]
        ve = state.plane_state.vel_y[0]
        chi = _course_from_vel(vn, ve)
        pitch = state.plane_state.pitch[0]
        vt    = state.plane_state.vt[0]
        info['obs_norm_dheading'] = _wrap_pi(chi - cmd_heading)
        info['obs_norm_dpitch']   = _wrap_pi(pitch - cmd_pitch)
        info['obs_norm_dvt']      = (vt - target_vt) / 340.0

        # —— 航点达成：筋斗 = 半径 + 相位门；普通 = 半径 —— #
        if params.use_vertical_loop:
            R    = params.loop_radius
            dphi = 2 * jnp.pi / params.loop_points_per_circle
            dir_ = jnp.sign(jnp.asarray(params.loop_direction, dtype=jnp.float32))
            phi0 = jnp.deg2rad(params.loop_phase0_deg)

            phi_curr = _loop_phase(state.loop_center_n, state.loop_center_alt, R, pn, pa)
            phi_wp   = phi0 + dir_ * (state.loop_idx.astype(jnp.float32)) * dphi

            def _phase_progress(phi_c, phi_w, d):
                raw = jnp.where(d >= 0.0, phi_c - phi_w, phi_w - phi_c)
                return jnp.mod(raw, 2.0 * jnp.pi)

            prog = _phase_progress(phi_curr, phi_wp, dir_)
            passed_gate = prog >= (0.15 * dphi)

            # reached_now = jnp.logical_and(dist3d <= state.reach_radius, passed_gate)

            # 修改：只看距离，不看相位门
            reached_now = dist3d <= state.reach_radius  # 原: jnp.logical_and(dist3d <= state.reach_radius, passed_gate)

            info['dbg_phi_curr'] = phi_curr
            info['dbg_phi_wp']   = phi_wp
            info['dbg_passed_gate'] = passed_gate.astype(jnp.int32)
        else:
            reached_now = dist3d <= state.reach_radius

        info['reached_this_step'] = reached_now
        info['dbg_cmd_pitch'] = cmd_pitch
        info['dbg_cmd_heading'] = cmd_heading
        info['dbg_target_vt'] = target_vt
        info['dbg_dist3d'] = dist3d
        info['dbg_hdist'] = hdist
        info['dbg_reach_radius'] = state.reach_radius
        info['dbg_reach_now'] = reached_now.astype(jnp.int32)
        info['dbg_reached_count'] = state.reached
        info['plane_status_before'] = state.plane_state.status[0]
        info['time_before'] = state.time
        info['reached_wp_n'] = wp[0]
        info['reached_wp_e'] = wp[1]
        info['reached_wp_a'] = wp[2]

        info['reach_radius'] = state.reach_radius
        info['reached_count'] = state.reached

        # 达成推进
        def on_reach(_):
            if params.use_vertical_loop:
                N    = int(self.default_params.loop_points_per_circle)
                R    = self.default_params.loop_radius
                dir_ = jnp.sign(jnp.asarray(params.loop_direction, dtype=jnp.float32))
                phi0 = jnp.deg2rad(params.loop_phase0_deg)
                tilt = jnp.deg2rad(params.loop_tilt_deg)

                new_idx = state.loop_idx + 1
                full = new_idx >= N

                def _next_same_center(_):
                    wp_next = state.loop_wps[new_idx]
                    # reach_radius2 = jnp.maximum(200.0, state.reach_radius * params.reach_radius_decay)
                    reach_radius2 = jnp.asarray(params.reach_radius_init, dtype=jnp.float32) # 不衰减
                    return state.replace(
                        waypoint=wp_next, reach_radius=reach_radius2, reached=state.reached + 1,
                        loop_idx=new_idx
                    )

                def _rebuild_next_circle(_):
                    u_n, u_e, _, _ = _heading_axes(state.loop_ref_heading)
                    c_n = state.loop_center_n + params.loop_forward_north * u_n
                    c_e = state.loop_center_e + params.loop_forward_north * u_e

                    b_a = jnp.cos(tilt)
                    floor_abs = params.min_altitude + params.loop_floor_margin
                    c_a_min = floor_abs + R * jnp.abs(b_a)
                    c_a = jnp.maximum(state.loop_center_alt, c_a_min)

                    wps2 = _build_loop_wps(c_n, c_e, c_a, state.loop_ref_heading, tilt, R, N, dir_, phi0)

                    p = jnp.array([pn, pe, pa])
                    d2 = jnp.sum((wps2 - p[None, :]) ** 2, axis=1)
                    k_start = jnp.argmin(d2)
                    wp_next = wps2[k_start]

                    # reach_radius2 = jnp.maximum(200.0, state.reach_radius * params.reach_radius_decay)
                    reach_radius2 = jnp.asarray(params.reach_radius_init, dtype=jnp.float32)
                    return state.replace(
                        waypoint=wp_next, reach_radius=reach_radius2, reached=state.reached + 1,
                        loop_center_n=c_n, loop_center_e=c_e, loop_center_alt=c_a,
                        loop_idx=jnp.asarray(k_start), loop_wps=wps2
                    )

                return jax.lax.cond(full, _rebuild_next_circle, _next_same_center, operand=None)

            else:
                reach_radius = jnp.maximum(100.0, state.reach_radius * params.reach_radius_decay)
                difficulty = state.difficulty + 1
                min_turn_deg = params.min_turn_deg_init + params.min_turn_deg_step * difficulty
                min_turn_rad = jnp.deg2rad(jnp.clip(min_turn_deg, 0.0, 170.0))
                wp_next = _sample_waypoint(key, jnp.array([pn, pe, pa]), params, min_turn_rad=min_turn_rad, current_yaw=state.plane_state.yaw[0])
                return state.replace(waypoint=wp_next, reach_radius=reach_radius, reached=state.reached + 1, difficulty=difficulty)

        def on_keep(_):
            return state

        timeout = state.time >= params.max_steps * params.sim_freq / params.agent_interaction_steps
        crashed = (state.plane_state.status[0] == 2)
        enough  = (state.reached >= params.max_waypoints)
        info['dbg_timeout'] = timeout
        info['dbg_crashed'] = crashed
        info['dbg_enough']  = enough

        # 同步指令（供渲染/日志）
        state = state.replace(cmd_heading=cmd_heading, cmd_pitch=cmd_pitch, cmd_vt=target_vt,
                              target_heading=cmd_heading, target_pitch=cmd_pitch)

        state = jax.lax.cond(reached_now, on_reach, on_keep, operand=None)
        return state, info

    # ===================== 覆盖：环境对外观测（四元数误差版） =====================
    # ===================== 覆盖：环境对外观测（与 _controller_obs 完全一致的 16D） =====================
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        观测顺序严格对齐 _controller_obs：
        [ qv(3), dvt, alt/5000, vt/340, v_b(3), P,Q,R, sin(alpha),cos(alpha), sin(beta),cos(beta) ]
        取值裁剪范围同 _controller_obs。
        """
        # --- 计算当前 waypoint 的几何量（逐 agent） ---
        dn = state.waypoint[0] - state.plane_state.north
        de = state.waypoint[1] - state.plane_state.east
        da = state.waypoint[2] - state.plane_state.altitude
        hdist  = jnp.sqrt(jnp.maximum(dn * dn + de * de, 1e-6))
        dist3d = jnp.sqrt(hdist * hdist + da * da)

        # --- 目标航向/俯仰 ---
        base_heading = _bearing(dn, de)

        if params.use_vertical_loop:
            # # 最近相位 + 前视半步的环切线（单机 i=0，写成函数便于以后多机扩展）
            # R    = jnp.asarray(params.loop_radius, jnp.float32)
            # N    = jnp.asarray(params.loop_points_per_circle, jnp.float32)
            # dphi = 2.0 * jnp.pi / N
            # dir_ = jnp.sign(jnp.asarray(params.loop_direction, jnp.float32))
            # tilt = jnp.deg2rad(params.loop_tilt_deg)

            # pn_i = state.plane_state.north[0]
            # pa_i = state.plane_state.altitude[0]
            # phi_curr = _loop_phase(state.loop_center_n, state.loop_center_alt, R, pn_i, pa_i)
            # phi_look = phi_curr + dir_ * (0.5 * dphi)

            # t_n, t_e, t_a = _loop_tangent(state.loop_ref_heading, tilt, phi_look, dir_)
            # # 切向量 → 航向 & 飞行路径角（作为俯仰目标）
            # heading_i   = jnp.arctan2(t_e, t_n)
            # gamma_cmd_i = jnp.arctan2(t_a, jnp.sqrt(jnp.maximum(t_n * t_n + t_e * t_e, 1e-9)))

            # desired_heading = jnp.array([heading_i], dtype=jnp.float32)
            # desired_pitch   = jnp.array([gamma_cmd_i], dtype=jnp.float32)

            # 向量场纠偏 / 切向航向
            # R    = jnp.asarray(params.loop_radius, jnp.float32)
            # dir_ = jnp.sign(jnp.asarray(params.loop_direction, jnp.float32))
            # tilt = jnp.deg2rad(params.loop_tilt_deg)

            # center = jnp.array([state.loop_center_n, state.loop_center_e, state.loop_center_alt], jnp.float32)

            # pn_i = state.plane_state.north[0]
            # pe_i = state.plane_state.east[0]
            # pa_i = state.plane_state.altitude[0]
            # pos_i = jnp.array([pn_i, pe_i, pa_i], jnp.float32)

            # chi_cmd_i, gamma_cmd_i = _loop_vf_cmd(
            #     center=center,
            #     pos=pos_i,
            #     yaw_ref=state.loop_ref_heading,
            #     tilt_rad=tilt,
            #     R=R,
            #     dir_=dir_,
            #     k_r=params.vf_k_radial,
            #     k_w=params.vf_k_plane,
            # )

            # desired_heading = jnp.array([chi_cmd_i], dtype=jnp.float32)
            # desired_pitch   = jnp.array([gamma_cmd_i], dtype=jnp.float32)

            # 目标航向/俯仰（纯 LOS 版本）
            desired_heading = base_heading          # ← 用回 LOS 航向
            # desired_pitch   = jnp.arctan2(da, hdist)
            hdist_sat = jnp.maximum(hdist, jnp.asarray(self._los_Lmin_m))
            desired_pitch = jnp.arctan2(da, hdist_sat)
            desired_pitch = jnp.clip(desired_pitch, jnp.deg2rad(-89.0), jnp.deg2rad(+89.0))


        else:
            desired_heading = base_heading
            # desired_pitch   = _desired_pitch(da, hdist)
            hdist_sat = jnp.maximum(hdist, jnp.asarray(self._los_Lmin_m))
            desired_pitch = jnp.arctan2(da, hdist_sat)
            desired_pitch = jnp.clip(desired_pitch, jnp.deg2rad(-89.0), jnp.deg2rad(+89.0))


        # S 形锁高：俯仰固定为 0
        if params.use_s_curve and params.s_altitude_lock:
            desired_pitch = jnp.array([0.0], dtype=jnp.float32)

        # --- 目标空速（与 _decode_actions 对齐） ---
        if params.use_vertical_loop:
            target_vt = jnp.full_like(state.plane_state.vt, params.loop_target_vt, dtype=jnp.float32)
        elif params.use_s_curve:
            target_vt = jnp.full_like(state.plane_state.vt, params.s_target_vt, dtype=jnp.float32)
        else:
            vt_far  = params.max_vt * 0.9
            vt_near = params.min_vt * 1.2
            blend   = jnp.clip(dist3d / 15000.0, 0.0, 1.0)
            target_vt = vt_near * (1.0 - blend) + vt_far * blend

        # --- 取出飞机当前状态 ---
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha, beta = state.plane_state.alpha, state.plane_state.beta
        P, Q, Rrate = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        # --- 单机（i=0）构造与 _controller_obs 完全一致的 16D ---
        def _build_obs_single(i):
            # 当前姿态四元数（优先取状态中的 q0..q3；若无则由 ZYX 欧拉生成）
            if hasattr(state.plane_state, "q0"):
                q_cur = jnp.array([
                    state.plane_state.q0[i],
                    state.plane_state.q1[i],
                    state.plane_state.q2[i],
                    state.plane_state.q3[i],
                ], dtype=jnp.float32)
                q_cur = _q_normalize(q_cur)
            else:
                q_cur = _q_from_euler_zyx(roll[i], pitch[i], yaw[i])

            # 目标姿态四元数（roll 目标 = 0）
            q_tgt = _q_from_euler_zyx(0.0, desired_pitch[i], desired_heading[i])  # Body2NED
            q_tgt_nb = _q_conj(q_tgt)                                            # NED2Body

            # 四元数误差（取向量部3维；保证 w>=0 消歧）
            q_err = _q_normalize(_q_mul(q_tgt_nb, _q_conj(q_cur)))  # [w,x,y,z]
            q_err = jnp.where(q_err[0] < 0.0, -q_err, q_err)
            qv = jnp.clip(q_err[1:4], -1.0, 1.0)  # (3,)

            # 机体系目标方向 v_b：先在 NED 构单位向量，再旋到 Body
            c_th, s_th = jnp.cos(desired_heading[i]), jnp.sin(desired_heading[i])
            c_ph, s_ph = jnp.cos(desired_pitch[i]),   jnp.sin(desired_pitch[i])
            v_n = jnp.array([c_ph * c_th, c_ph * s_th, s_ph], dtype=jnp.float32)  # NED

            p   = jnp.array([0.0, v_n[0], v_n[1], v_n[2]], dtype=jnp.float32)
            qp  = _q_mul(q_cur, p)
            qpq = _q_mul(qp, _q_conj(q_cur))
            v_b = jnp.clip(qpq[1:4], -1.0, 1.0)  # Body, (3,)

            # 其它归一化量
            norm_dvt = (vt[i] - target_vt[i]) / 340.0
            norm_alt = altitude[i] / 5000.0
            norm_vt  = vt[i] / 340.0

            alpha_sin_i, alpha_cos_i = jnp.sin(alpha[i]), jnp.cos(alpha[i])
            beta_sin_i,  beta_cos_i  = jnp.sin(beta[i]),  jnp.cos(beta[i])

            # 拼接 16D（严格同 _controller_obs 顺序）
            vec = jnp.array([
                qv[0], qv[1], qv[2],         # 0-2  四元数误差向量部
                norm_dvt,                    # 3    (vt - target_vt) / 340
                norm_alt,                    # 4    alt / 5000
                norm_vt,                     # 5    vt / 340
                v_b[0], v_b[1], v_b[2],      # 6-8  机体系目标方向
                P[i], Q[i], Rrate[i],        # 9-11 角速率
                alpha_sin_i, alpha_cos_i,    # 12-13
                beta_sin_i,  beta_cos_i      # 14-15
            ], dtype=jnp.float32)

            # 裁剪范围与 _controller_obs 保持一致
            low = jnp.array([
                -1., -1., -1.,   # qv
                -2.,             # dvt/340
                0.,             # alt/5000
                0.,             # vt/340
                -1., -1., -1.,   # v_b
                -10., -10., -10.,# P,Q,R
                -1., -1.,        # sin(alpha),cos(alpha)
                -1., -1.         # sin(beta),cos(beta)
            ], dtype=jnp.float32)

            high = jnp.array([
                1.,  1.,  1.,   # qv
                2.,             # dvt/340
                5.,             # alt/5000
                2.,             # vt/340
                1.,  1.,  1.,   # v_b
                10., 10., 10.,   # P,Q,R
                1.,  1.,        # sin(alpha),cos(alpha)
                1.,  1.         # sin(beta),cos(beta)
            ], dtype=jnp.float32)

            vec = jnp.clip(jnp.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0), low, high)
            return vec

        # 单机（当前仅 1 个 agent）
        obs_vec = _build_obs_single(0)  # (16,)

        # 返回 {agent_name: (16,)}
        return {agent: obs_vec for _, agent in enumerate(self.agents)}



    # ---------- 终止条件 ----------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _term_timeout(self, state: WaypointTaskState, params: WaypointTaskParams, agent_id: AgentID):
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
        success = done
        return done, success
