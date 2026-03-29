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
    "theta_scale_precision_deg": 20.0,
    "theta_scale_guidance_deg": 120.0,
    "theta_blend_scale_deg": 40.0,
    "precision_exponent": 4.0,
    "guidance_exponent": 2.0,
    "speed_error_scale": 40.0,
    "w_att": 0.75,
    "w_speed": 0.25,
    "settled_bonus_weight": 0.2,
    "settled_threshold_deg": 5.0,
}


def quat_baseline_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0) -> float:
    """Theta-adaptive dual-path reward v2 — wider guidance for curriculum levels 2-5.

    Key changes from v1:
    - guidance scale: 90° → 120° (theta=120° now has exp(-1)≈0.368 gradient)
    - blend scale: 30° → 40° (precision mode lasts longer)
    - precision scale: 25° → 20° (sharper near-target incentive)
    - settled bonus: additive → multiplicative (avoids clip waste)
    - w_att: 0.7 → 0.75 (attitude tracking more important for curriculum)

    At theta=120°: att_r ≈ 0.368 (vs ~0.10 in v1) — 3.7x more gradient signal.
    At theta=90°:  att_r ≈ 0.467 (vs ~0.368 in v1) — 27% more gradient signal.
    At theta=5°:   att_r ≈ 0.999 (precision maintained).
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

    # --- Precision path: narrow quartic Gaussian (sharp near-target incentive) ---
    theta_scale_prec = jnp.deg2rad(_cfg["theta_scale_precision_deg"])
    gaussian_precision = jnp.exp(-((theta / theta_scale_prec) ** _cfg["precision_exponent"]))

    # --- Guidance path: wide quadratic Gaussian (gradient signal at large angles) ---
    # scale=120° ensures theta=120° gives exp(-1)≈0.368 — enough gradient for curriculum L2-5
    theta_scale_guid = jnp.deg2rad(_cfg["theta_scale_guidance_deg"])
    gaussian_guidance = jnp.exp(-((theta / theta_scale_guid) ** _cfg["guidance_exponent"]))

    # --- Theta-adaptive blend weight ---
    # blend → 1 when theta small (precision dominates)
    # blend → 0 when theta large (guidance dominates)
    # blend_scale=40° gives a smoother transition than 30°
    theta_blend_scale = jnp.deg2rad(_cfg["theta_blend_scale_deg"])
    blend = jnp.exp(-((theta / theta_blend_scale) ** 2))

    att_r = blend * gaussian_precision + (1.0 - blend) * gaussian_guidance
    att_r = jnp.clip(att_r, 0.0, 1.0)

    # --- Speed reward ---
    delta_vt = vt - state.target_vt[agent_id]
    delta_vt = jnp.clip(jnp.nan_to_num(delta_vt, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
    speed_r = jnp.exp(-(delta_vt / _cfg["speed_error_scale"]) ** 2)

    # --- Base reward (product form) ---
    base_reward = (att_r ** _cfg["w_att"]) * (speed_r ** _cfg["w_speed"])

    # --- Settled bonus: multiplicative boost when theta < 5° ---
    # Multiplicative form: avoids being wasted by clip(0,1) when base_reward is already high
    settled_threshold = jnp.deg2rad(_cfg["settled_threshold_deg"])
    settled_multiplier = jnp.where(
        theta < settled_threshold,
        1.0 + _cfg["settled_bonus_weight"],
        1.0
    )
    reward = base_reward * settled_multiplier

    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
