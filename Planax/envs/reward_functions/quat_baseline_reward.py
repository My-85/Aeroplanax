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
    "theta_scale_deg": 30.0,        # fine Gaussian scale (champion-proven)
    "theta_scale_coarse_deg": 90.0, # coarse Gaussian scale for large angles
    "speed_error_scale": 40.0,
    "w_att": 0.7,
    "w_speed": 0.3,
    "att_exponent": 4.0,
    "coarse_exponent": 2.0,         # quadratic for coarse (smoother gradient)
    # curriculum-adaptive coarse weight: lower at L0 for precision, higher at L5 for gradient
    "coarse_w_l01": 0.05,   # L0-1: 5% coarse, 95% fine — champion-like precision
    "coarse_w_l23": 0.20,   # L2-3: 20% coarse
    "coarse_w_l45": 0.45,   # L4-5: 45% coarse — strong gradient signal for large angles
    "dot_product_weight": 0.0,      # disabled (was 0.2 in current, trying pure Gaussian)
    "use_arithmetic_mean": 1.0,
}


def quat_baseline_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """Curriculum-adaptive dual-scale Gaussian attitude reward.
    
    Core idea: Use curriculum_level to blend between:
    - Fine scale (30°, quartic): high precision signal for L0-1
    - Coarse scale (90°, quadratic): gradient signal for L4-5 large angles
    
    At L0-1: 95% fine + 5% coarse → champion-level L0 precision
    At L2-3: 80% fine + 20% coarse → balanced
    At L4-5: 55% fine + 45% coarse → sufficient gradient for theta>90°
    
    Different from #49 (fixed 75/25 split) and #22 (fixed 70/30 split):
    adaptive blending preserves L0 precision while fixing L4-5 gradient.
    Different from #38 (similar concept but used scale adaptation not weight adaptation).
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

    # --- Fine-scale Gaussian (champion-proven, 30°, quartic) ---
    theta_scale_fine = jnp.deg2rad(_cfg["theta_scale_deg"])
    gaussian_fine = jnp.exp(-((theta / theta_scale_fine) ** _cfg["att_exponent"]))

    # --- Coarse-scale Gaussian (90°, quadratic) for large-angle gradient ---
    theta_scale_coarse = jnp.deg2rad(_cfg["theta_scale_coarse_deg"])
    gaussian_coarse = jnp.exp(-((theta / theta_scale_coarse) ** _cfg["coarse_exponent"]))

    # --- Curriculum-adaptive blending weight for coarse component ---
    curriculum_level = state.curriculum_level[agent_id]
    
    coarse_w = jnp.where(
        curriculum_level <= 1,
        _cfg["coarse_w_l01"],   # L0-1: minimal coarse (preserve precision)
        jnp.where(
            curriculum_level <= 3,
            _cfg["coarse_w_l23"],  # L2-3: moderate coarse
            _cfg["coarse_w_l45"]   # L4-5: large coarse (gradient for big angles)
        )
    )
    fine_w = 1.0 - coarse_w

    att_r = fine_w * gaussian_fine + coarse_w * gaussian_coarse

    # Speed reward (unchanged from champion)
    delta_vt = vt - state.target_vt[agent_id]
    delta_vt = jnp.clip(jnp.nan_to_num(delta_vt, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
    speed_r = jnp.exp(-(delta_vt / _cfg["speed_error_scale"]) ** 2)

    # Arithmetic weighted sum (same as current champion)
    reward = _cfg["w_att"] * att_r + _cfg["w_speed"] * speed_r

    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
