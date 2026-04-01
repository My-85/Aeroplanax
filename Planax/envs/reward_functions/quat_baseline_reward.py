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
    "theta_scale_fine_deg": 25.0,
    "theta_scale_coarse_deg": 100.0,
    "theta_exponent_fine": 4.0,
    "theta_exponent_coarse": 1.5,
    "blend_weight_fine_l01": 0.85,
    "blend_weight_fine_l23": 0.6,
    "blend_weight_fine_l45": 0.4,
    "speed_error_scale": 40.0,
    "w_att": 0.75,
    "w_speed": 0.25,
    "settled_bonus_weight": 0.25,
    "settled_threshold_deg": 6.0,
    "overload_penalty_weight": 0.15,
    "overload_onset_g": 6.0,
    "overload_max_g": 10.0,
}


def quat_baseline_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0) -> float:
    """Dual-scale blended Gaussian with curriculum-adaptive weighting and overload penalty.

    Design rationale:
    - Fine scale (25°, exp=4): Precise tracking for small angles
    - Coarse scale (100°, exp=1.5): Non-vanishing gradient at 120-180°
    - Curriculum-adaptive blend: More fine at L0-1, more coarse at L4-5
    - Enhanced settled bonus (1.25x) to improve stability
    - Overload penalty: Soft onset at 6G, saturate at 10G to prevent crash
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

    # --- Dual-scale Gaussian ---
    scale_fine = jnp.deg2rad(_cfg["theta_scale_fine_deg"])
    scale_coarse = jnp.deg2rad(_cfg["theta_scale_coarse_deg"])

    att_r_fine = jnp.exp(-((theta / scale_fine) ** _cfg["theta_exponent_fine"]))
    att_r_coarse = jnp.exp(-((theta / scale_coarse) ** _cfg["theta_exponent_coarse"]))

    # --- Curriculum-adaptive blending ---
    curriculum_level = state.curriculum_level[agent_id]
    w_fine = jnp.where(
        curriculum_level <= 1,
        _cfg["blend_weight_fine_l01"],
        jnp.where(curriculum_level <= 3, _cfg["blend_weight_fine_l23"], _cfg["blend_weight_fine_l45"])
    )

    att_r = w_fine * att_r_fine + (1.0 - w_fine) * att_r_coarse
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

    # --- Enhanced settled bonus for stability ---
    settled_threshold = jnp.deg2rad(_cfg["settled_threshold_deg"])
    settled_multiplier = jnp.where(
        theta < settled_threshold,
        1.0 + _cfg["settled_bonus_weight"],
        1.0
    )
    reward = base_reward * settled_multiplier

    # --- Overload penalty: soft onset at 6G, saturate at 10G ---
    az = state.plane_state.az[agent_id]
    overload_nz = jnp.abs(jnp.nan_to_num(az, nan=0.0))
    overload_penalty = (
        _cfg["overload_penalty_weight"]
        * jnp.clip(
            (overload_nz - _cfg["overload_onset_g"]) / (_cfg["overload_max_g"] - _cfg["overload_onset_g"]),
            0.0, 1.0
        ) ** 2
    )
    reward = reward - overload_penalty

    reward = jnp.clip(
        jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0 + _cfg["settled_bonus_weight"]
    )
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
