# envs/termination_conditions/unreach_full_domain.py
# -*- coding: utf-8 -*-
"""
Quaternion-based success detection for full-domain maneuver.
Uses geodesic angle between current and target quaternions.
Target quaternion = conj(_quat_from_euler_bn(...)) = q_NB, matching dynamics state convention.

Key change: real success requires sustained on-target tracking
(checked via on_target_steps in _step_task). Here we use:
  - sustained success: on_target_steps >= sustained_on_target_steps param
  - timeout: elapsed >= max_interval triggers target switch but is NOT a real success
Both trigger success=True (to cause target switch in _step_task),
but curriculum advancement only counts sustained successes.

v7 changes (iteration 3):
  - min_elapsed_sec: 5 → 3 (faster feedback, agent can succeed sooner)
  - max_interval_sec: 30 → 20 (reduce timeout-gaming; agent forced to track faster)
  - These changes increase the curriculum advancement pressure appropriately

v10 changes (iteration 5):
  - max_interval_sec: 20 → 12 (further reduce timeout-cycling exploit;
    agent was collecting 0.64/step × 100 steps = 64 episodic return by cycling timeouts)
"""
from typing import Tuple
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState


# ---- quaternion helpers ----
# Convention: dynamics stores q_NB (NED-to-Body).
#   _quat_from_euler_bn: ZYX Euler → q_BN (Body-to-NED)
#   target = conj(_quat_from_euler_bn(...)) = q_NB, matching dynamics state.
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
    q_a = _quat_normalize(q_a)
    q_b = _quat_normalize(q_b)
    cos_half = jnp.abs(jnp.dot(q_a, q_b))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half)


def unreach_full_domain_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    min_elapsed_sec: float = 3.0,
    theta_tol_deg: float = 10.0,
    vt_tol: float = 15.0,
    max_interval_sec: float = 12.0,
) -> Tuple[bool, bool]:
    """
    Success condition (does not terminate episode, triggers target switch):
      success = sustained_on_target (on_target_steps >= threshold, checked in _step_task)
              | timeout (elapsed >= max_interval, gives target switch but NOT curriculum credit)

    v7 changes:
      - min_elapsed_sec: 5 → 3 (faster feedback)
      - max_interval_sec: 30 → 20 → 12 (reduce timeout-gaming)
      - Real success requires sustained tracking (on_target_steps >= params.sustained_on_target_steps)
    """
    plane_state: FighterPlaneState = state.plane_state

    # time bookkeeping
    sim_per_decision = params.sim_freq / params.agent_interaction_steps
    check_time = state.time - state.last_check_time
    elapsed_sec = check_time / sim_per_decision

    # current quaternion from dynamics state
    q_curr = jnp.array([
        jnp.nan_to_num(plane_state.q0[agent_id], nan=1.0),
        jnp.nan_to_num(plane_state.q1[agent_id], nan=0.0),
        jnp.nan_to_num(plane_state.q2[agent_id], nan=0.0),
        jnp.nan_to_num(plane_state.q3[agent_id], nan=0.0),
    ])
    q_curr = _quat_normalize(q_curr)

    # target quaternion — conjugated to match dynamics state convention
    yaw_t   = state.target_heading[agent_id]
    pitch_t = state.target_pitch[agent_id]
    roll_t  = state.target_roll[agent_id]
    q_tgt = _quat_conj(_quat_from_euler_bn(roll_t, pitch_t, yaw_t))

    theta = _quat_geodesic_angle(q_curr, q_tgt)

    # speed error
    vt = jnp.nan_to_num(plane_state.vt[agent_id], nan=0.0)
    vt_tgt = jnp.nan_to_num(state.target_vt[agent_id], nan=0.0)
    delta_vt = jnp.abs(vt - vt_tgt)

    # Sustained on-target check: uses curriculum-dependent threshold
    base_steps = getattr(params, 'sustained_on_target_steps', 5)
    per_level = getattr(params, 'sustained_on_target_per_level', 2)
    curr_level = getattr(state, 'curriculum_level', jnp.int32(0))
    sustained_threshold = base_steps + curr_level * per_level
    on_target_steps = getattr(state, 'on_target_steps', jnp.int32(0))
    sustained_success = on_target_steps >= sustained_threshold

    # Time gate: at least min_elapsed_sec must have passed
    time_ok = elapsed_sec >= min_elapsed_sec

    # Timeout: too long without success, switch target anyway
    timeout = elapsed_sec >= max_interval_sec

    # success triggers target switch: either sustained success OR timeout
    success = (time_ok & sustained_success) | timeout

    done = jnp.array(False, dtype=jnp.bool_)
    return done, success
