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
    "theta_scale_coarse_deg": 90.0,
    "speed_error_scale": 40.0,
    "w_att": 0.7,
    "w_speed": 0.3,
    "att_exponent": 4.0,
    "coarse_exponent": 2.0,
    "coarse_w_l01": 0.05,
    "coarse_w_l23": 0.25,
    "coarse_w_l45": 0.45,
    "use_product_form": 1.0,
}


def quat_baseline_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """Curriculum-adaptive dual-scale Gaussian with champion product form.
    
    Key insight: Previous Phase 2b experiments used arithmetic sum (w_att*att_r + w_speed*speed_r)
    but the champion uses geometric product ((att_r^w_att) * (speed_r^w_speed)).
    This experiment restores the product form while using dual-scale att_r.
    
    att_r = fine_w * gaussian_fine(30°, quartic) + coarse_w * gaussian_coarse(90°, quadratic)
    reward = (att_r^0.7) * (speed_r^0.3)  ← champion product form restored
    
    Curriculum-adaptive coarse weight:
    - L0-1: 5% coarse (champion-like precision)
    - L2-3: 25% coarse (balanced gradient)
    - L4-5: 45% coarse (large angle gradient)
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
        _cfg["coarse_w_l01"],
        jnp.where(
            curriculum_level <= 3,
            _cfg["coarse_w_l23"],
            _cfg["coarse_w_l45"]
        )
    )
    fine_w = 1.0 - coarse_w

    att_r = fine_w * gaussian_fine + coarse_w * gaussian_coarse
    att_r = jnp.clip(att_r, 0.0, 1.0)

    # Speed reward (unchanged from champion)
    delta_vt = vt - state.target_vt[agent_id]
    delta_vt = jnp.clip(jnp.nan_to_num(delta_vt, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
    speed_r = jnp.exp(-(delta_vt / _cfg["speed_error_scale"]) ** 2)

    # CHAMPION PRODUCT FORM: (att_r^w_att) * (speed_r^w_speed)
    # This is the key difference from recent Phase 2b experiments that used arithmetic sum
    reward = (att_r ** _cfg["w_att"]) * (speed_r ** _cfg["w_speed"])

    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
