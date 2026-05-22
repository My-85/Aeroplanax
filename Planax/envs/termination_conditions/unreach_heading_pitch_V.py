from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax.numpy as jnp


def unreach_heading_pitch_V_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    check_interval: float = 90.0,  # seconds between target switches (agent needs time to complete ~117° turns)
) -> Tuple[bool, bool]:
    """
    Time-driven target switching.  After check_interval seconds, trigger
    a target switch.  Simple, predictable, and decoupled from tracking quality
    (tracking is handled entirely by the reward function).
    """
    check_time = state.time - state.last_check_time
    min_steps = check_interval * params.sim_freq / params.agent_interaction_steps
    success = check_time >= min_steps
    done = False
    return done, success
