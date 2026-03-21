# envs/reward_functions/full_domain_reward.py
# -*- coding: utf-8 -*-
"""
Full-domain maneuver reward: quaternion attitude tracking (triple-scale),
speed tracking, altitude safety, smoothness, alive bonus.

Key changes (v10 - iteration 5):
  ISSUE: Agent found reward exploit at 1.6B steps:
    - theta stable at 13° (never enters 10° on-target zone)
    - delta_vt degraded from 17.7 to 29.2 (completely abandoned speed tracking)
    - episodic_return = 64 (0.64/step × 100 steps timeout cycling)
    - curriculum regressed from 0.19 to 0.002 (agent learned to avoid success)

  Root causes:
    1. full_bonus=4.0 required theta<10° AND delta_vt<25 simultaneously — agent couldn't
       achieve either → bonus completely inaccessible → no gradient
    2. track_weight_spd=0.25 too weak — no incentive to track speed
    3. Timeout free (20s, no penalty) → agent cycles timeouts for guaranteed reward
    4. Curriculum advancement = harder targets = lower reward → agent avoids success

  Fixes (v10):
    1. Split on-target: att-only(1.5) + spd-only(1.0) + combined(2.0) + close(0.1)
       - Removed near_bonus entirely
    2. Rebalanced tracking: att 0.75→0.60, spd 0.25→0.40
    3. (Env-side) Timeout 20s→12s (shrink timeout-cycling profit from 64 to ~33)
    4. Curriculum level scale: 1.0 + 0.1*level (level 7 → 1.7× reward)

  The prev_specific_energy field in state is repurposed to store prev_theta
  (initialized to pi, updated in _step_task).
"""
import jax
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID


# ---- quaternion helpers ----
def _quat_normalize(q):
    return q / (jnp.linalg.norm(q) + 1e-9)

def _quat_conj(q):
    return jnp.stack([q[0], -q[1], -q[2], -q[3]], axis=0)

def _quat_from_euler_bn(roll, pitch, yaw):
    """ZYX Euler angles to q_BN (Body-to-NED rotation quaternion)."""
    cr, sr = jnp.cos(0.5 * roll),  jnp.sin(0.5 * roll)
    cp, sp = jnp.cos(0.5 * pitch), jnp.sin(0.5 * pitch)
    cy, sy = jnp.cos(0.5 * yaw),   jnp.sin(0.5 * yaw)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return jnp.stack([qw, qx, qy, qz], axis=0)

def _quat_geodesic_angle(q_a, q_b):
    """Geodesic angle between two quaternions. Convention-independent."""
    q_a = _quat_normalize(q_a)
    q_b = _quat_normalize(q_b)
    cos_half = jnp.abs(jnp.dot(q_a, q_b))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half)


REWARD_CONFIG = {
    # v17: fix "theta=85° no learning" — 3 root causes:
    #   1. crash_penalty=-5.0 drowns tracking signal (crash every ~45 steps)
    #   2. Gaussian-only attitude reward has near-zero gradient at theta>70°
    #   3. grad_norm=12 clipped to 2, value loss dominates actor gradient
    # Fix: add cosine attitude term (strong gradient everywhere), reduce crash penalty,
    # increase progress coefficient for dense directional signal.
    "gaussian_scale_coarse_deg": 80.0,
    "gaussian_scale_medium_deg": 20.0,
    "gaussian_scale_fine_deg": 5.0,
    "gaussian_weight_coarse": 0.4,
    "gaussian_weight_medium": 0.35,
    "gaussian_weight_fine": 0.25,
    "cosine_weight": 0.6,                 # NEW v17: cosine term weight in attitude mix
    "gaussian_mix_weight": 0.4,           # NEW v17: gaussian term weight in attitude mix
    "speed_sigma": 30.0,
    "track_weight_att": 0.75,
    "track_weight_spd": 0.25,
    "tracking_scale": 2.0,                # v17: 3.0→2.0, cosine provides enough signal
    "progress_coeff": 5.0,                # v17: 3.0→5.0, stronger dense improvement signal
    "progress_clip_lo": -0.5,
    "progress_clip_hi": 1.5,              # v17: 1.0→1.5, allow bigger positive progress
    # on-target bonuses (v10 split, v11 speed gate — proven design, keep)
    "on_target_att_bonus": 1.5,
    "on_target_att_thresh_deg": 10.0,
    "on_target_spd_bonus": 1.0,
    "on_target_spd_thresh": 15.0,
    "on_target_spd_att_gate_deg": 30.0,
    "on_target_full_bonus": 2.0,
    "on_target_full_thresh_deg": 10.0,
    "on_target_full_vt_thresh": 25.0,
    "on_target_close_bonus": 0.3,         # v17: 0.1→0.3, stronger close-zone incentive
    "on_target_close_thresh_deg": 30.0,   # v17: 20→30°, wider close zone
    # error penalty — light touch, just to discourage extreme theta
    "error_penalty_coeff": -0.01,
    "error_penalty_thresh_deg": 60.0,
    "error_extreme_coeff": -0.02,
    "error_extreme_thresh_deg": 90.0,
    # smoothness/overload — keep light penalties
    "smoothness_omega_thresh": 10.0,
    "smoothness_coeff": -0.001,
    "smoothness_gate_deg": 180.0,
    "overload_omega_thresh": 15.0,
    "overload_coeff": -0.003,
    # v17: crash_penalty small so it doesn't drown tracking signal
    # At theta=85°, tracking≈1.1/step. crash_penalty=-0.3 means crash is only
    # 1.4 worse than normal step — detectable but not dominant.
    "alive_bonus": 0.05,                  # v17: 0.005→0.05, small positive bias
    "crash_penalty": -0.3,                # v17: -5.0→-0.3, stop drowning tracking signal
    "reward_clip_lo": -1.0,               # v17: match crash_penalty
    "reward_clip_hi": 8.0,
    "curriculum_level_scale": 0.1,
}


def full_domain_reward_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    """
    v10: Fix reward exploit (theta=13° plateau + speed abandonment + curriculum avoidance).

    Root causes (1.6B step analysis):
      1. full_bonus=4.0 required theta<10° AND delta_vt<25 simultaneously — agent couldn't
         achieve either → 4.0 was completely inaccessible → no gradient toward on-target.
      2. No speed incentive — track_weight_spd=0.25 too low, delta_vt degraded 17→29.
      3. Agent exploited timeout cycling (0.64/step × 100 steps = 64 episodic return).
      4. Curriculum advancement → harder targets → lower reward → agent avoided success.

    Fixes:
      - Split on-target: att-only (1.5) + spd-only (1.0) + combined (2.0) + close (0.1)
      - Rebalance tracking weights: att 0.75→0.60, spd 0.25→0.40
      - Curriculum level scale: level_scale = 1.0 + 0.1*level (level 7 → 1.7×)
      - (Env-side) Timeout reduced 20s→12s to shrink timeout-cycling profit

    Reward at key operating points:
      - theta=13°, delta_vt=29: ~0.55/step (no bonus, punishes not tracking)
      - theta=8°,  delta_vt=29: ~2.1/step  (att bonus fires, 3.9× incentive)
      - theta=8°,  delta_vt=10: ~5.2/step  (all bonuses fire, 9.5× incentive)

    Clipped to [-5, 5.0].
    """
    # ---- read state ----
    vt = jnp.nan_to_num(state.plane_state.vt[agent_id], nan=0.0)
    alt = jnp.nan_to_num(state.plane_state.altitude[agent_id], nan=0.0)
    vel_z = jnp.nan_to_num(state.plane_state.vel_z[agent_id], nan=0.0)
    P = jnp.nan_to_num(state.plane_state.P[agent_id], nan=0.0)
    Q = jnp.nan_to_num(state.plane_state.Q[agent_id], nan=0.0)
    R = jnp.nan_to_num(state.plane_state.R[agent_id], nan=0.0)

    q_curr = jnp.array([
        jnp.nan_to_num(state.plane_state.q0[agent_id], nan=1.0),
        jnp.nan_to_num(state.plane_state.q1[agent_id], nan=0.0),
        jnp.nan_to_num(state.plane_state.q2[agent_id], nan=0.0),
        jnp.nan_to_num(state.plane_state.q3[agent_id], nan=0.0),
    ])
    q_curr = _quat_normalize(q_curr)

    yaw_t   = state.target_heading[agent_id]
    pitch_t = state.target_pitch[agent_id]
    roll_t  = state.target_roll[agent_id]
    vt_tgt  = state.target_vt[agent_id]

    # target quaternion q_NB = conj(q_BN), matching dynamics state convention
    q_tgt = _quat_conj(_quat_from_euler_bn(roll_t, pitch_t, yaw_t))

    # ---- alive mask ----
    is_alive = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]

    _cfg = REWARD_CONFIG

    # ---- CRASH PENALTY: strong negative reward for dead planes ----
    crash_penalty = jnp.where(is_alive, 0.0, _cfg["crash_penalty"])

    # ---- attitude tracking: sharper Gaussian scales for stronger gradient ----
    theta = _quat_geodesic_angle(q_curr, q_tgt)  # [0, pi]
    theta_deg = theta * 180.0 / jnp.pi

    # Coarse: strong gradient in 20-60° range
    r_coarse = jnp.exp(-(theta / jnp.deg2rad(_cfg["gaussian_scale_coarse_deg"])) ** 2)
    # Medium: strong gradient in 8-20° range for mid-accuracy convergence
    r_medium = jnp.exp(-(theta / jnp.deg2rad(_cfg["gaussian_scale_medium_deg"])) ** 2)
    # Fine: precision tracking for final convergence
    r_fine   = jnp.exp(-(theta / jnp.deg2rad(_cfg["gaussian_scale_fine_deg"])) ** 2)

    # Weights: balanced across scales
    r_att_gaussian = (_cfg["gaussian_weight_coarse"] * r_coarse
                      + _cfg["gaussian_weight_medium"] * r_medium
                      + _cfg["gaussian_weight_fine"] * r_fine)

    # v17: cosine attitude term — provides strong gradient at ALL theta values
    # At theta=85°, Gaussian gradient ≈ 0.008/deg (undetectable in noise).
    # cos(theta) gradient = sin(theta)/2 ≈ 0.5/rad = 0.009/deg at theta=85°.
    # Combined, the cosine term ensures the agent always sees which direction improves.
    r_att_cosine = (1.0 + jnp.cos(theta)) / 2.0  # theta=0→1.0, 90°→0.5, 180°→0.0

    r_att = _cfg["cosine_weight"] * r_att_cosine + _cfg["gaussian_mix_weight"] * r_att_gaussian

    # ---- PROGRESS REWARD: explicit dense signal for theta reduction ----
    prev_theta = jnp.nan_to_num(
        jnp.clip(state.prev_specific_energy[agent_id], 0.0, jnp.pi),
        nan=jnp.pi
    )
    theta_delta = prev_theta - theta  # positive = improvement
    r_progress = jnp.clip(
        _cfg["progress_coeff"] * theta_delta / jnp.pi,
        _cfg["progress_clip_lo"],
        _cfg["progress_clip_hi"],
    )

    # ---- speed tracking ----
    delta_vt = jnp.clip(jnp.nan_to_num(vt - vt_tgt, nan=0.0), -1e3, 1e3)
    r_spd = jnp.exp(-(delta_vt / _cfg["speed_sigma"]) ** 2)

    # ---- combined tracking (weighted sum, scaled up for stronger gradient signal) ----
    r_tracking = _cfg["tracking_scale"] * (_cfg["track_weight_att"] * r_att + _cfg["track_weight_spd"] * r_spd)

    # ---- on-target bonus: split into attitude-only, speed-only, combined ----
    # v10: decoupled bonuses so agent gets stepwise incentive (att → spd → both)
    # Previously full_bonus=4.0 required theta<10° AND delta_vt<25 simultaneously;
    # agent couldn't achieve either condition → 4.0 was completely inaccessible.

    # Attitude-only: theta<10° gives 1.5 regardless of speed
    on_target_att = jnp.where(
        theta_deg <= _cfg["on_target_att_thresh_deg"],
        _cfg["on_target_att_bonus"],
        0.0,
    )

    # Speed bonus: delta_vt<15 AND theta<30° (gate prevents speed-only exploit)
    # v11 fix: without gate, agent at theta=87° got 1.0 spd_bonus → 1.39/step exploit
    on_target_spd = jnp.where(
        (jnp.abs(delta_vt) <= _cfg["on_target_spd_thresh"]) & (theta_deg <= _cfg["on_target_spd_att_gate_deg"]),
        _cfg["on_target_spd_bonus"],
        0.0,
    )

    # Combined: both conditions met gives extra 2.0
    on_target_full = jnp.where(
        (theta_deg <= _cfg["on_target_full_thresh_deg"]) & (jnp.abs(delta_vt) <= _cfg["on_target_full_vt_thresh"]),
        _cfg["on_target_full_bonus"],
        0.0,
    )

    # Close zone: theta 10-20° gives 0.1 (minimal, no plateau)
    on_target_close = jnp.where(
        (theta_deg <= _cfg["on_target_close_thresh_deg"]) & (theta_deg > _cfg["on_target_att_thresh_deg"]),
        _cfg["on_target_close_bonus"],
        0.0,
    )

    on_target_bonus = on_target_att + on_target_spd + on_target_full + on_target_close

    # ---- large error penalty: stronger push to reduce error ----
    r_error_penalty = jnp.where(
        theta_deg > _cfg["error_penalty_thresh_deg"],
        _cfg["error_penalty_coeff"] * (theta_deg - _cfg["error_penalty_thresh_deg"]) / _cfg["error_penalty_thresh_deg"],
        0.0,
    )
    r_error_penalty_extreme = jnp.where(
        theta_deg > _cfg["error_extreme_thresh_deg"],
        _cfg["error_extreme_coeff"] * (theta_deg - _cfg["error_extreme_thresh_deg"]) / _cfg["error_extreme_thresh_deg"],
        0.0,
    )

    # ---- altitude safety (soft penalty, only below safe_alt) ----
    safe_alt   = getattr(params, "safe_altitude", 2.5)     # km
    danger_alt = getattr(params, "danger_altitude", 1.5)    # km
    alt_km = alt / 1000.0

    margin_denom = jnp.maximum(safe_alt - danger_alt, 0.01)
    margin = jnp.clip((safe_alt - alt_km) / margin_denom, 0.0, 1.0)
    vel_z_term = jnp.clip(-vel_z / 340.0, 0.0, 1.0)
    r_alt_soft = -0.5 * margin ** 2 * (1.0 + vel_z_term)
    r_alt_hard = jnp.where(alt_km < danger_alt, -2.0, 0.0)
    r_alt_active = r_alt_soft + r_alt_hard
    r_alt = jnp.where(alt_km <= safe_alt, r_alt_active, 0.0)

    # ---- smoothness (always active to prevent overload crashes) ----
    omega_mag = jnp.sqrt(P ** 2 + Q ** 2 + R ** 2)
    omega_excess = jnp.clip(omega_mag - _cfg["smoothness_omega_thresh"], 0.0)
    r_smooth_raw = _cfg["smoothness_coeff"] * omega_excess ** 2
    # v12: gate opened to 180° — smoothness penalty always active
    # Previously gated at 30°, so agent had NO smoothness penalty when theta>30°
    # → extreme angular rates → overload crashes (58 crashes/episode)
    gate = 1.0

    # ---- overload penalty (high angular rate → likely structural overload) ----
    omega_overload_excess = jnp.clip(omega_mag - _cfg["overload_omega_thresh"], 0.0)
    r_overload = _cfg["overload_coeff"] * omega_overload_excess ** 2

    # ---- alive bonus ----
    r_alive = _cfg["alive_bonus"]

    # ---- alive reward: tracking + progress + bonuses + safety ----
    r_alive_total = (
        r_tracking
        + r_progress
        + on_target_bonus
        + r_error_penalty
        + r_error_penalty_extreme
        + r_alt
        + r_smooth_raw * gate
        + r_overload
        + r_alive
    )

    # ---- curriculum level reward scale: higher levels → more reward per step ----
    # v10: compensates for harder targets at higher curriculum levels.
    # Without this, advancing the curriculum = harder targets = lower reward, so the agent
    # rationally avoids success. With scale, level 1 → 1.1×, level 7 → 1.7× reward.
    curriculum_level = getattr(state, 'curriculum_level', jnp.int32(0))
    level_scale = 1.0 + _cfg["curriculum_level_scale"] * jnp.float32(curriculum_level)
    r_alive_total = r_alive_total * level_scale

    # ---- CRITICAL: do NOT multiply by mask ----
    reward = jnp.where(is_alive, r_alive_total, crash_penalty)

    reward = jnp.clip(
        jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0),
        _cfg["reward_clip_lo"],
        _cfg["reward_clip_hi"],
    )

    return reward * reward_scale
