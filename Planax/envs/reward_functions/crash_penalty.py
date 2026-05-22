"""Crash penalty: mild negative reward when aircraft exceeds overload / crashes."""
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID


def crash_penalty_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    penalty: float = -3.0,
) -> float:
    """Return `penalty` when the aircraft is in crashed state (status==2)."""
    return jnp.where(state.plane_state.is_crashed[agent_id], penalty, 0.0)
