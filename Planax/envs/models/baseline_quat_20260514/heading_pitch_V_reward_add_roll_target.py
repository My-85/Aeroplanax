# 配合新版训练代码（做了reward clip） —— 四元数版
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID

# --- 工具函数：与动力学一致的姿态约定 ---
def _euler_to_quat_nb(roll: float, pitch: float, yaw: float):
    """
    按 Z(yaw)-Y(pitch)-X(roll) 组合生成 q_{NB}（Body->NED 的四元数）。
    这与 dynamics 里 quaternion_to_rpy 的约定是互逆的：
      rpy = quaternion_to_rpy( conj(q_{BN}) )  // 你在 dynamics.update里就是这么做的
    """
    cr, sr = jnp.cos(0.5*roll),  jnp.sin(0.5*roll)
    cp, sp = jnp.cos(0.5*pitch), jnp.sin(0.5*pitch)
    cy, sy = jnp.cos(0.5*yaw),   jnp.sin(0.5*yaw)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return jnp.stack([qw, qx, qy, qz], axis=0)  # q_{NB}

def _quat_conj(q):
    # 共轭：q*，对应旋转的逆
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def _quat_normalize(q):
    return q / (jnp.linalg.norm(q) + 1e-6)

def _quat_geodesic_angle(q_a, q_b): # q_a: NED -> Body, q_b: NED -> Body
    """
    计算两个单位四元数之间的测地角：theta = 2*arccos(|dot(q_a, q_b)|)
    注意：用绝对值处理双覆盖（q 与 -q 等价）
    """
    q_a = _quat_normalize(q_a)
    q_b = _quat_normalize(q_b)
    cos_half = jnp.abs(jnp.dot(q_a, q_b))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half)

def heading_pitch_V_reward_fn_add_roll_target(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """
    四元数姿态跟踪 + 速度跟踪 的加权几何平均 reward。
    - 姿态误差：当前姿态四元数 q_{BN}^{curr} 与 目标姿态四元数 q_{BN}^{tgt} 的测地角
    - 速度误差：与原实现一致
    """
    # === 读取状态 ===
    vt = state.plane_state.vt[agent_id]
    q_curr = jnp.array([
        state.plane_state.q0[agent_id],
        state.plane_state.q1[agent_id],
        state.plane_state.q2[agent_id],
        state.plane_state.q3[agent_id],
    ])
    q_curr = jnp.nan_to_num(q_curr, nan=0.0)
    q_curr = _quat_normalize(q_curr)  # 状态里虽有归一，但这里再保护一次

    # === 构造目标姿态四元数 ===
    # 任务侧给了目标航向/俯仰；roll 目标设为 0（翼水平）。如需“放开滚转”，可把 roll_t 换成 state.plane_state.roll[agent_id] 的缓慢滤波版本。
    yaw_t   = state.target_heading[agent_id]
    pitch_t = state.target_pitch[agent_id]
    roll_t  = state.target_roll[agent_id]   # <-- 改这里


    # 先得到 q_{NB}^{tgt}（Body->NED），再取共轭变为 q_{BN}^{tgt}，与状态里的存储一致（见 dynamics.update 的用法）
    q_tgt_nb = _euler_to_quat_nb(roll_t, pitch_t, yaw_t)   # Body -> NED
    q_tgt_bn = _quat_conj(q_tgt_nb)     # NED -> Body
    
    # === 姿态 reward：四元数测地角 ===
    theta = _quat_geodesic_angle(q_curr, q_tgt_bn)  # ∈ [0, π]
    theta_scale = jnp.deg2rad(5.0)  # 5° ≈ 你原先 heading/pitch 的误差尺度
    att_r = jnp.exp(- (theta / theta_scale) ** 2)

    # === 速度 reward：保留原高斯形 ===
    delta_vt = vt - state.target_vt[agent_id]
    delta_vt = jnp.clip(jnp.nan_to_num(delta_vt, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
    speed_error_scale = 12.0  # m/s
    speed_r = jnp.exp(- (delta_vt / speed_error_scale) ** 2)

    # === 加权几何平均（保持原总权重思想：姿态0.8，速度0.2） ===
    w_att   = 0.7
    w_speed = 0.3
    reward = (att_r ** w_att) * (speed_r ** w_speed)

    # === clip + 掩码 ===
    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
