# envs/reward_functions/full_domain_reward.py
# -*- coding: utf-8 -*-
"""
Full-domain maneuver reward: quaternion attitude tracking (triple-scale),
speed tracking, altitude safety, smoothness, alive bonus.

Key changes (v5 - iteration 2 fixes):
  - REMOVED broken progress reward (prev_specific_energy was never updated = always wrong)
  - Increased on_target_bonus 1.5 → 2.0 (must dominate survival strategy)
  - Added smooth partial bonus: 0.5 for theta<20deg, 0.15 for theta<35deg
  - Rebalanced triple-scale weights: more coarse signal for large-error gradient
  - Reduced alive_bonus 0.005 → 0.002 (500 steps = 1.0 total, much less than one on_target_bonus)
  - Kept all safety penalties
  - Clip range: [-5, 2.5]
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
    Total reward = r_tracking + on_target_bonus + r_alt + r_smooth * gate + r_alive
    clipped to [-5, 2.5].

    Design philosophy:
      - on_target_bonus (2.0) >> total alive_bonus (0.002*500=1.0)
        so agent MUST track to maximize reward
      - Smooth partial bonuses at 35° and 20° create gradient toward on_target
      - Triple-scale tracking gives gradient at ALL error ranges
      - NO broken progress reward (prev_specific_energy not tracked reliably)
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

    # ---- attitude tracking (triple-scale for smooth gradient across all ranges) ----
    theta = _quat_geodesic_angle(q_curr, q_tgt)  # [0, pi]
    theta_deg = theta * 180.0 / jnp.pi

    # Coarse (70°): gives gradient signal at large errors (35-120°)
    r_coarse = jnp.exp(-(theta / jnp.deg2rad(70.0)) ** 2)
    # Medium (25°): bridges the gap, strong gradient at 10-40° range
    r_medium = jnp.exp(-(theta / jnp.deg2rad(25.0)) ** 2)
    # Fine (5°): precision tracking for final convergence
    r_fine   = jnp.exp(-(theta / jnp.deg2rad(5.0)) ** 2)

    # Weights: more coarse to give gradient at large errors (agent starts far from target)
    r_att = 0.25 * r_coarse + 0.50 * r_medium + 0.25 * r_fine

    # ---- speed tracking ----
    delta_vt = jnp.clip(jnp.nan_to_num(vt - vt_tgt, nan=0.0), -1e3, 1e3)
    r_spd = jnp.exp(-(delta_vt / 30.0) ** 2)

    # ---- combined tracking (weighted sum instead of product for better gradient) ----
    r_tracking = 0.75 * r_att + 0.25 * r_spd

    # ---- on-target bonus: LARGE reward for being close ----
    # Tiered: creates clear gradient toward precision
    # Tier 1: full on-target (theta<10° AND speed<15 m/s) → +2.0
    # Tier 2: close attitude (theta<20°) → +0.5
    # Tier 3: approaching (theta<35°) → +0.15
    # These bonuses STACK on top of r_tracking for maximum gradient signal
    on_target_full = jnp.where(
        (theta_deg <= 10.0) & (jnp.abs(delta_vt) <= 15.0),
        2.0,
        0.0,
    )
    on_target_close = jnp.where(
        (theta_deg <= 20.0) & (on_target_full == 0.0),
        0.5,
        0.0,
    )
    on_target_near = jnp.where(
        (theta_deg <= 35.0) & (on_target_close == 0.0) & (on_target_full == 0.0),
        0.15,
        0.0,
    )
    on_target_bonus = on_target_full + on_target_close + on_target_near

    # ---- large error penalty: penalize staying far from target ----
    # Gentle penalty to create urgency without dominating
    r_error_penalty = jnp.where(
        theta_deg > 60.0,
        -0.08 * (theta_deg - 60.0) / 60.0,  # scales from 0 at 60° to -0.08 at 120°
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
    omega_excess = jnp.clip(omega_mag - 5.0, 0.0)  # allow higher rates for aggressive maneuvers
    r_smooth_raw = -0.008 * omega_excess ** 2
    # Gate by attitude accuracy: only penalize smoothness when relatively close
    gate = jnp.where(theta_deg < 30.0, r_att, 0.0)

    # ---- alive bonus (MINIMAL: 0.002/step * 500steps = 1.0 total) ----
    # Must be less than one on_target_bonus (2.0) to force tracking behavior
    r_alive = 0.002

    # ---- total ----
    reward = (r_tracking + on_target_bonus + r_error_penalty
              + r_alt + r_smooth_raw * gate + r_alive)
    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 2.5)

    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
