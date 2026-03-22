# envs/reward_functions/full_domain_reward.py
# -*- coding: utf-8 -*-
"""
Full-domain maneuver reward: quaternion attitude tracking (triple-scale),
speed tracking, altitude safety, smoothness, alive bonus.

Key changes (v7 - iteration 3):
  ISSUE: Agent plateaued at theta~51 deg, curriculum stuck at level 0.
  Root causes:
    1. Gaussian scales too wide (70/25/5 deg) — shallow gradient at 30-60 deg range
    2. On-target bonuses too small (2.0/0.5/0.15) — weak attractor
    3. Alive bonus (0.01/step) competed with tracking incentive
    4. No progress reward — no dense signal for theta reduction

  Fixes:
    1. Sharper Gaussian scales: 40/12/4 deg — strong gradient where agent currently is (30-60 deg)
    2. Progress reward: explicit reward for reducing theta (potential-based shaping)
       r_progress = 0.3 * (prev_theta - curr_theta) / pi  (positive when improving)
    3. Larger on_target bonuses: full=3.0, close=1.0, near=0.4
    4. Stronger large-error penalty: -0.15 * (theta-60)/60 (was -0.08)
    5. Reduced alive bonus: 0.005/step (was 0.01) — survival still valued but less dominant
    6. Clip range expanded: [-5, 5.0]

  The prev_specific_energy field in state is repurposed to store prev_theta
  (this was already done in v6 — initialized to pi, updated in _step_task).
"""
import jax
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID


# ---- quaternion helpers ----
def _quat_normalize(q):
    return q / (jnp.linalg.norm(q) + 1e-9)

def _quat_conj(q):
    return jnp.stack([q[0], -q[1], -q[2], -q[3]], axis=0)

def _quat_from_euler_nb(roll, pitch, yaw):
    """ZYX Euler angles to quaternion (same formula as env helpers)."""
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


def full_domain_reward_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    """
    v7: Sharper reward gradients to break the theta~50 deg plateau.

    Key improvements:
      1. Triple-scale Gaussian with scales 40/12/4 deg (was 70/25/5)
         — much stronger gradient in the 20-60 deg range where agent is stuck
      2. Progress reward: r_progress = 0.4*(prev_theta - curr_theta)/pi
         — dense per-step signal for any theta reduction
         — prev_theta stored in state.prev_specific_energy (reused field)
      3. Larger on_target bonuses: full=3.0, close=1.0, near=0.4
      4. Stronger large-error penalty
      5. Reduced alive bonus (0.005) so tracking dominates survival

    Structure:
      - If plane is alive/locked: tracking_reward + progress + bonuses + safety + alive_bonus
      - If plane is crashed/dead: crash_penalty = -3.0

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

    # target quaternion — conjugated to match dynamics state convention
    q_tgt = _quat_conj(_quat_from_euler_nb(roll_t, pitch_t, yaw_t))

    # ---- alive mask ----
    is_alive = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]

    # ---- CRASH PENALTY: strong negative reward for dead planes ----
    crash_penalty = jnp.where(is_alive, 0.0, -3.0)

    # ---- attitude tracking: sharper Gaussian scales for stronger gradient ----
    theta = _quat_geodesic_angle(q_curr, q_tgt)  # [0, pi]
    theta_deg = theta * 180.0 / jnp.pi

    # Coarse (40°): strong gradient in 20-60° range (where agent is stuck at ~50°)
    r_coarse = jnp.exp(-(theta / jnp.deg2rad(40.0)) ** 2)
    # Medium (12°): strong gradient in 8-20° range for mid-accuracy convergence
    r_medium = jnp.exp(-(theta / jnp.deg2rad(12.0)) ** 2)
    # Fine (4°): precision tracking for final convergence <8°
    r_fine   = jnp.exp(-(theta / jnp.deg2rad(4.0)) ** 2)

    # Weights: balanced across scales, coarse slightly dominant to help convergence from far
    r_att = 0.40 * r_coarse + 0.40 * r_medium + 0.20 * r_fine

    # ---- PROGRESS REWARD: explicit dense signal for theta reduction ----
    # prev_specific_energy stores prev_theta (in radians), initialized to pi
    # This gives +reward for any improvement, -reward for getting worse
    prev_theta = jnp.nan_to_num(
        jnp.clip(state.prev_specific_energy[agent_id], 0.0, jnp.pi),
        nan=jnp.pi
    )
    theta_delta = prev_theta - theta  # positive = improvement
    # Scale: max improvement=pi, max reward=0.4; capped to avoid exploit
    r_progress = jnp.clip(0.4 * theta_delta / jnp.pi, -0.3, 0.3)

    # ---- speed tracking ----
    delta_vt = jnp.clip(jnp.nan_to_num(vt - vt_tgt, nan=0.0), -1e3, 1e3)
    r_spd = jnp.exp(-(delta_vt / 30.0) ** 2)

    # ---- combined tracking (weighted sum instead of product for better gradient) ----
    r_tracking = 0.75 * r_att + 0.25 * r_spd

    # ---- on-target bonus: LARGER rewards to create strong attractor ----
    # Tier 1: full on-target (theta<10° AND speed<15 m/s) → +3.0
    # Tier 2: close attitude (theta<20°) → +1.0
    # Tier 3: approaching (theta<35°) → +0.4
    on_target_full = jnp.where(
        (theta_deg <= 10.0) & (jnp.abs(delta_vt) <= 15.0),
        3.0,
        0.0,
    )
    on_target_close = jnp.where(
        (theta_deg <= 20.0) & (on_target_full == 0.0),
        1.0,
        0.0,
    )
    on_target_near = jnp.where(
        (theta_deg <= 35.0) & (on_target_close == 0.0) & (on_target_full == 0.0),
        0.4,
        0.0,
    )
    on_target_bonus = on_target_full + on_target_close + on_target_near

    # ---- large error penalty: stronger push to reduce error ----
    r_error_penalty = jnp.where(
        theta_deg > 60.0,
        -0.15 * (theta_deg - 60.0) / 60.0,
        0.0,
    )
    # Also penalize being >90° — much too far
    r_error_penalty_extreme = jnp.where(
        theta_deg > 90.0,
        -0.15 * (theta_deg - 90.0) / 90.0,
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

    # ---- smoothness (only when near target to avoid penalizing aggressive maneuvers) ----
    omega_mag = jnp.sqrt(P ** 2 + Q ** 2 + R ** 2)
    omega_excess = jnp.clip(omega_mag - 5.0, 0.0)
    r_smooth_raw = -0.008 * omega_excess ** 2
    # Gate by attitude accuracy: only penalize smoothness when relatively close
    gate = jnp.where(theta_deg < 30.0, r_att, 0.0)

    # ---- alive bonus: REDUCED to 0.005/step ----
    # 0.005 * 500 steps = 2.5 total — still positive but not dominant over tracking
    # On-target bonuses (up to 3.0/step when tracking well) dominate
    r_alive = 0.005

    # ---- alive reward: tracking + progress + bonuses + safety ----
    r_alive_total = (
        r_tracking
        + r_progress
        + on_target_bonus
        + r_error_penalty
        + r_error_penalty_extreme
        + r_alt
        + r_smooth_raw * gate
        + r_alive
    )

    # ---- CRITICAL: do NOT multiply by mask ----
    # Instead: alive → full reward, dead → crash_penalty
    reward = jnp.where(is_alive, r_alive_total, crash_penalty)

    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)

    return reward * reward_scale
