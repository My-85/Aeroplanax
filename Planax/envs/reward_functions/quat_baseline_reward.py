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
    "theta_scale_low_deg": 30.0,
    "theta_scale_mid_deg": 60.0,
    "theta_scale_high_deg": 90.0,
    "theta_exponent": 2.0,
    "speed_error_scale": 40.0,
    "w_att": 0.75,
    "w_speed": 0.25,
    "settled_bonus_weight": 0.15,
    "settled_threshold_deg": 5.0,
}


def quat_baseline_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0) -> float:
    """Level-adaptive single-scale Gaussian reward for curriculum learning.

    Simplified design to avoid crashes:
    - Single Gaussian with level-adaptive sigma
    - Level 0-1: sigma=30deg (Phase 1 compatible, precise tracking)
    - Level 2-3: sigma=60deg (medium, covers ±90deg targets)
    - Level 4-5: sigma=90deg (wide, covers full ±180deg domain)
    - Multiplicative settled bonus (1 + weight) for theta < threshold
    - No clip(0,1) on final reward to preserve settled bonus signal
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
    theta = jnp.nan_to_num(theta, nan=0.0)

    # --- Level-adaptive theta scale ---
    curriculum_level = state.curriculum_level[agent_id]

    scale_low  = jnp.deg2rad(_cfg["theta_scale_low_deg"])
    scale_mid  = jnp.deg2rad(_cfg["theta_scale_mid_deg"])
    scale_high = jnp.deg2rad(_cfg["theta_scale_high_deg"])

    theta_scale = jnp.where(
        curriculum_level <= 1,
        scale_low,
        jnp.where(curriculum_level <= 3, scale_mid, scale_high)
    )

    # --- Single Gaussian attitude reward ---
    att_r = jnp.exp(-((theta / theta_scale) ** _cfg["theta_exponent"]))
    att_r = jnp.clip(att_r, 0.0, 1.0)

    # --- Speed reward ---
    delta_vt = vt - state.target_vt[agent_id]
    delta_vt = jnp.clip(
        jnp.nan_to_num(delta_vt, nan=0.0, posinf=1e6, neginf=-1e6),
        -1e3, 1e3
    )
    speed_r = jnp.exp(-(delta_vt / _cfg["speed_error_scale"]) ** 2)

    # --- Base reward (product form) ---
    base_reward = (att_r ** _cfg["w_att"]) * (speed_r ** _cfg["w_speed"])

    # --- Settled bonus: multiplicative when theta < threshold ---
    # Use (1 + bonus) multiplier; allow reward > 1.0 to preserve signal
    # Final clip is set to 1.15 to cap the bonus without wasting it
    settled_threshold = jnp.deg2rad(_cfg["settled_threshold_deg"])
    settled_multiplier = jnp.where(
        theta < settled_threshold,
        1.0 + _cfg["settled_bonus_weight"],
        1.0
    )
    reward = base_reward * settled_multiplier

    reward = jnp.clip(
        jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0 + _cfg["settled_bonus_weight"]  # allow bonus to exceed 1.0
    )
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
