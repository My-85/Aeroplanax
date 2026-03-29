# from typing import Tuple
# import jax.numpy as jnp
# import jax
# from ..aeroplanax import TEnvState, TEnvParams, AgentID
# from ..core.simulators.fighterplane.dynamics import FighterPlaneState


# def crashed_fn(
#     state: TEnvState,
#     params: TEnvParams,
#     agent_id: AgentID,
# ) -> Tuple[bool, bool]:
#     """
#     End up the simulation if the aircraft is on an extreme state.
#     """
#     plane_state: FighterPlaneState = state.plane_state
#     done = plane_state.is_crashed[agent_id]
#     success = False
#     return done, success


from typing import Tuple
import jax.numpy as jnp
import jax
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState, atmos

def crashed_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
) -> Tuple[bool, bool]:
    """
    End up the simulation if the aircraft is on an extreme state,
    or any load factor component exceeds 10G.
    """
    plane_state: FighterPlaneState = state.plane_state
    done_engine = plane_state.is_crashed[agent_id]

    # Load factor limit: crash if any component |nx|, |ny|, |nz| > 10G
    # Note: ax, ay, az are already normalized load factors (in G units)
    nx = plane_state.ax[agent_id]
    ny = plane_state.ay[agent_id]
    nz = plane_state.az[agent_id]
    done_overload = jnp.logical_or(
        jnp.logical_or(jnp.abs(nx) > 10.0, jnp.abs(ny) > 10.0),
        jnp.abs(nz) > 10.0
    )

    done = jnp.logical_or(done_engine, done_overload)
    success = False
    return done, success