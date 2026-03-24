# quat_baseline_reward.py — Quaternion attitude tracking reward (iterable copy)
# Refactored from heading_pitch_V_reward_add_roll_target.py with extractable REWARD_CONFIG.
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID

# --- quaternion helpers (same as original) ---
def _euler_to_quat_nb(roll, pitch, yaw):
    cr, sr = jnp.cos(0.5*roll),  jnp.sin(0.5*roll)
    cp, sp = jnp.cos(0.5*pitch), jnp.sin(0.5*pitch)
    cy, sy = jnp.cos(0.5*yaw),   jnp.sin(0.5*yaw)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return jnp.stack([qw, qx, qy, qz], axis=0)

def _quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def _quat_normalize(q):
    return q / (jnp.linalg.norm(q) + 1e-6)

def _quat_geodesic_angle(q_a, q_b):
    q_a = _quat_normalize(q_a)
    q_b = _quat_normalize(q_b)
    cos_half = jnp.abs(jnp.dot(q_a, q_b))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half)


# ---- REWARD_CONFIG: all tunable parameters extracted here ----
REWARD_CONFIG = {
    "theta_scale_deg": 30.0,
    "speed_error_scale": 40.0,
    "w_att": 0.7,
    "w_speed": 0.3,
    "att_exponent": 4.0,
    "dot_product_weight": 0.25,
}


def quat_baseline_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """Hybrid: champion Gaussian (75%) + direct quaternion dot product reward (25%).
    
    The dot product reward (1+|q_curr·q_tgt|)/2 has a guaranteed non-vanishing gradient
    at ALL angles including 180°, and its gradient is proportional to sin(theta/2) which
    peaks at theta=90° — exactly where curriculum L2-3 agents need learning signal.
    Unlike cosine similarity of Euler angles, this operates in quaternion space directly.
    """
    _cfg = REWARD_CONFIG

    vt = state.plane_state.vt[agent_id]
    q_curr = jnp.array([
        state.plane_state.q0[agent_id],
        state.plane_state.q1[agent_id],
        state.plane_state.q2[agent_id],
        state.plane_state.q3[agent_id],
    ])
    q_curr = jnp.nan_to_num(q_curr, nan=0.0)
    q_curr = _quat_normalize(q_curr)

    yaw_t   = state.target_heading[agent_id]
    pitch_t = state.target_pitch[agent_id]
    roll_t  = state.target_roll[agent_id]

    q_tgt_nb = _euler_to_quat_nb(roll_t, pitch_t, yaw_t)
    q_tgt_nb = _quat_conj(q_tgt_nb)

    theta = _quat_geodesic_angle(q_curr, q_tgt_nb)

    # Component 1: Champion quartic Gaussian (precision at small angles)
    theta_scale = jnp.deg2rad(_cfg["theta_scale_deg"])
    gaussian_r = jnp.exp(-((theta / theta_scale) ** _cfg["att_exponent"]))

    # Component 2: Quaternion dot product reward
    # |q_a · q_b| = cos(theta/2), so (1 + |q_a · q_b|) / 2 maps:
    #   theta=0   -> 1.0 (perfect)
    #   theta=90  -> 0.854 (still high, provides gradient)
    #   theta=180 -> 0.5 (minimum, non-zero gradient everywhere)
    cos_half = jnp.abs(jnp.dot(q_curr, q_tgt_nb))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    dot_r = (1.0 + cos_half) / 2.0

    # Weighted combination: mostly Gaussian for precision, dot for curriculum gradient
    w_dot = _cfg["dot_product_weight"]
    att_r = (1.0 - w_dot) * gaussian_r + w_dot * dot_r

    # Speed reward (unchanged from champion)
    delta_vt = vt - state.target_vt[agent_id]
    delta_vt = jnp.clip(jnp.nan_to_num(delta_vt, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
    speed_r = jnp.exp(-(delta_vt / _cfg["speed_error_scale"]) ** 2)

    # Weighted geometric mean
    reward = (att_r ** _cfg["w_att"]) * (speed_r ** _cfg["w_speed"])

    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
