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
    or dynamic pressure is too low (stall-like).
    """
    plane_state: FighterPlaneState = state.plane_state
    done_engine = plane_state.is_crashed[agent_id]

    # qbar-based stall crash
    alt_ft = plane_state.altitude[agent_id] / 0.3048
    vt_ft  = jnp.maximum(plane_state.vt[agent_id] / 0.3048, 0.1)
    _, qbar, _ = atmos(alt_ft, vt_ft)

    # alt_mid_ft = ((getattr(params, "min_altitude", 2000.0) + getattr(params, "max_altitude", 20000.0)) * 0.5) / 0.3048
    # vt_ref_ft  = getattr(params, "max_vt", 360.0) / 0.3048
    # _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)

    alt_mid_ft = ((getattr(params, "min_altitude", 2000.0) + getattr(params, "max_altitude", 20000.0)) * 0.5) / 0.3048
    vt_ref_ft  = getattr(params, "qbar_ref_vt", getattr(params, "max_vt", 360.0)) / 0.3048
    _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)

    qn = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 10.0)
    thresh = getattr(params, "qbar_crash_frac", 0.30)  # 默认0.30，可在 TaskParams 配置
    done_qbar = qn < thresh

    done = jnp.logical_or(done_engine, done_qbar)
    success = False
    return done, success