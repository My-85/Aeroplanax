"""
Vertical-energy quaternion task switching.

The environment stores a per-agent task_duration_steps field.  This
termination condition only emits success when that duration elapses; the
environment then decides whether the previous target was earned and samples
the next curriculum target.  It never terminates the episode by itself.
"""
from typing import Tuple

import jax.numpy as jnp

from ..aeroplanax import TEnvState, TEnvParams, AgentID


def unreach_heading_pitch_V_quat_vertical_energy_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
) -> Tuple[bool, bool]:
    check_time = state.time - state.last_check_time
    task_duration_steps = getattr(state, "task_duration_steps", None)
    if task_duration_steps is None:
        level = jnp.clip(jnp.asarray(state.level_selected[agent_id], dtype=jnp.int32), 0, 3)
        intervals_sec = jnp.array([11.0, 24.0, 42.0, 50.0])
        steps_per_sec = params.sim_freq / params.agent_interaction_steps
        min_steps = intervals_sec[level] * steps_per_sec
    else:
        min_steps = jnp.maximum(jnp.asarray(task_duration_steps[agent_id], dtype=jnp.float32), 1.0)
    success = check_time >= min_steps
    done = jnp.array(False, dtype=jnp.bool_)
    return done, success
