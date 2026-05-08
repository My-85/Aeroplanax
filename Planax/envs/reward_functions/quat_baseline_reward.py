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
    "overload_penalty_weight": 0.2,
    "overload_onset_g": 6.0,
    "overload_max_g": 10.0,
    "jerk_penalty_weight": 0.05,
    "jerk_gate_deg": 25.0,
    "progress_weight": 0.05,
    "stability_penalty_weight": 0.1,
    "osc_gate_deg": 12.0,
    "osc_omega_scale": 0.5,
}


def quat_baseline_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0) -> float:
    """Dual-scale blended Gaussian with curriculum-adaptive weighting and full penalty suite.

    Design rationale:
    - Fine scale (25°, exp=4): Precise tracking for small angles
    - Coarse scale (100°, exp=1.5): Non-vanishing gradient at 120-180°
    - Curriculum-adaptive blend: More fine at L0-1, more coarse at L4-5
    - Settled bonus (1.25x multiplicative): rewards stability without clip waste
    - Overload penalty: (nz-6)^2/16 onset at 6G, saturate at 10G (weight=0.2)
    - Jerk penalty: action smoothness, gated off when theta>25° for large maneuvers
    - Progress reward: potential-based shaping from prev_theta->theta
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

    # --- Settled bonus (multiplicative, avoids clip waste) ---
    settled_threshold = jnp.deg2rad(_cfg["settled_threshold_deg"])
    settled_multiplier = jnp.where(
        theta < settled_threshold,
        1.0 + _cfg["settled_bonus_weight"],
        1.0
    )
    r_total = base_reward * settled_multiplier

    # --- Overload penalty: MULTIPLICATIVE form, kills reward when nz > threshold ---
    # Previous additive form (weight=0.2) was insufficient: net_reward = 0.54+0.017-0.2 > 0
    # Multiplicative: reward → 0 as nz → 10G, so "pull 10G to turn fast" is never profitable
    az = state.plane_state.az[agent_id]
    overload_nz = jnp.abs(jnp.nan_to_num(az, nan=0.0))
    overload_ratio = jnp.clip(
        (overload_nz - _cfg["overload_onset_g"]) / (_cfg["overload_max_g"] - _cfg["overload_onset_g"]),
        0.0, 1.0
    )
    # Multiplier goes from 1.0 at 6G down to 0.0 at 10G
    overload_multiplier = 1.0 - overload_ratio
    r_total = r_total * overload_multiplier

    # --- Jerk penalty: penalize rapid control surface changes near target ---
    # Gated off when theta > jerk_gate_deg to allow free large maneuvers
    el  = jnp.nan_to_num(state.plane_state.el[agent_id],  nan=0.0)
    ail = jnp.nan_to_num(state.plane_state.ail[agent_id], nan=0.0)
    rud = jnp.nan_to_num(state.plane_state.rud[agent_id], nan=0.0)
    p_el  = jnp.nan_to_num(state.prev_el[agent_id],  nan=0.0)
    p_ail = jnp.nan_to_num(state.prev_ail[agent_id], nan=0.0)
    p_rud = jnp.nan_to_num(state.prev_rud[agent_id], nan=0.0)
    delta_u = jnp.abs(el - p_el) + jnp.abs(ail - p_ail) + jnp.abs(rud - p_rud)
    jerk_gate_rad = jnp.deg2rad(_cfg["jerk_gate_deg"])
    jerk_active = theta < jerk_gate_rad
    P_jerk = jnp.minimum(delta_u / 5.0, 1.0) * _cfg["jerk_penalty_weight"] * jerk_active

    # --- Progress reward: potential-based shaping (prev_theta -> theta) ---
    prev_theta = jnp.nan_to_num(state.prev_theta[agent_id], nan=0.0)
    R_progress = (prev_theta - theta) * _cfg["progress_weight"]

    # --- Final reward ---
    reward = r_total + R_progress - P_jerk

    reward = jnp.clip(
        jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0 + _cfg["settled_bonus_weight"]
    )
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
