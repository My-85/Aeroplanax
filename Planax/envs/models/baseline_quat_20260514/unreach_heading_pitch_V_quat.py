"""
Quaternion-env-only: adaptive check intervals based on difficulty level.
Reads state.level_selected [0-3] and uses progressively longer intervals.
"""
from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax.numpy as jnp

# Check intervals per level: [L0, L1, L2, L3] in seconds
_LEVEL_INTERVAL = jnp.array([11.0, 24.0, 42.0, 50.0])


def unreach_heading_pitch_V_quat_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
) -> Tuple[bool, bool]:
    """
    Adaptive target switching.  Interval depends on state.level_selected.
    The env's _step_task further checks tracking quality to decide
    whether to advance the curriculum counter.
    """
    check_time = state.time - state.last_check_time  # RL steps
    steps_per_sec = params.sim_freq / params.agent_interaction_steps
    level = jnp.clip(jnp.asarray(state.level_selected[agent_id], dtype=jnp.int32), 0, 3)
    min_steps = _LEVEL_INTERVAL[level] * steps_per_sec
    success = check_time >= min_steps
    done = jnp.array(False, dtype=jnp.bool_)
    return done, success
