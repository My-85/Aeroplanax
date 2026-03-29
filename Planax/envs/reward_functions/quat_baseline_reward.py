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
    # Theta-adaptive dual-path reward
    "theta_scale_precision_deg": 25.0,   # precision path: narrow Gaussian
    "theta_scale_guidance_deg": 90.0,    # guidance path: wide Gaussian for large angles
    "theta_blend_scale_deg": 30.0,       # blend transition point
    "precision_exponent": 4.0,           # quartic for sharp precision peak
    "guidance_exponent": 2.0,            # quadratic for smooth gradient
    # Speed reward
    "speed_error_scale": 40.0,
    # Product weights
    "w_att": 0.7,
    "w_speed": 0.3,
    # Settled bonus
    "settled_bonus_weight": 0.15,
    "settled_threshold_deg": 5.0,
}


def quat_baseline_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0) -> float:
    """Theta-adaptive dual-path reward.

    Two parallel paths automatically blend based on current theta:
    - Precision path (scale=25°, quartic): sharp reward near target
    - Guidance path (scale=90°, quadratic): gradient signal at large angles

    Blend weight = exp(-(theta/30°)^2):
    - theta small → precision dominates (accurate tracking incentive)
    - theta large → guidance dominates (non-zero gradient for learning)

    This eliminates the need for curriculum_level-based switching:
    the reward automatically adapts to the current error magnitude.

    At theta=90°: att_r ≈ 0.368 (vs ~0.13 in Iter11) — 3x more gradient signal.
    At theta=5°: att_r ≈ 0.985 (high precision incentive maintained).
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

    # --- Precision path: narrow quartic Gaussian ---
    theta_scale_prec = jnp.deg2rad(_cfg["theta_scale_precision_deg"])
    gaussian_precision = jnp.exp(-((theta / theta_scale_prec) ** _cfg["precision_exponent"]))

    # --- Guidance path: wide quadratic Gaussian ---
    theta_scale_guid = jnp.deg2rad(_cfg["theta_scale_guidance_deg"])
    gaussian_guidance = jnp.exp(-((theta / theta_scale_guid) ** _cfg["guidance_exponent"]))

    # --- Theta-adaptive blend weight ---
    # blend → 1 when theta small (precision dominates)
    # blend → 0 when theta large (guidance dominates)
    theta_blend_scale = jnp.deg2rad(_cfg["theta_blend_scale_deg"])
    blend = jnp.exp(-((theta / theta_blend_scale) ** 2))

    att_r = blend * gaussian_precision + (1.0 - blend) * gaussian_guidance
    att_r = jnp.clip(att_r, 0.0, 1.0)

    # --- Settled bonus: extra reward when theta < 5° ---
    settled_threshold = jnp.deg2rad(_cfg["settled_threshold_deg"])
    settled_bonus = _cfg["settled_bonus_weight"] * jnp.where(theta < settled_threshold, 1.0, 0.0)

    # --- Speed reward ---
    delta_vt = vt - state.target_vt[agent_id]
    delta_vt = jnp.clip(jnp.nan_to_num(delta_vt, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
    speed_r = jnp.exp(-(delta_vt / _cfg["speed_error_scale"]) ** 2)

    # --- Product form + settled bonus ---
    base_reward = (att_r ** _cfg["w_att"]) * (speed_r ** _cfg["w_speed"])
    reward = base_reward + settled_bonus

    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
