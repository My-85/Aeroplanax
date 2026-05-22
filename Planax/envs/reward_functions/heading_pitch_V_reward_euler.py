"""Euler-angle version: Fixed Laplacian reward.

Widened from 15° to 30° (heading) to provide gradient across the full
error range needed by absolute-target tracking.  No annealing, no
curriculum — simple, predictable, effective.
"""
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI


def heading_pitch_V_reward_euler_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    roll  = jnp.nan_to_num(state.plane_state.roll[agent_id],  nan=0.0)
    pitch = jnp.nan_to_num(state.plane_state.pitch[agent_id], nan=0.0)
    yaw   = jnp.nan_to_num(state.plane_state.yaw[agent_id],   nan=0.0)
    vt    = jnp.nan_to_num(state.plane_state.vt[agent_id],    nan=0.0)

    delta_heading = wrap_PI(yaw - state.target_heading[agent_id])
    delta_pitch   = wrap_PI(pitch - state.target_pitch[agent_id])
    delta_roll    = wrap_PI(roll - state.target_roll[agent_id])
    delta_vt      = vt - state.target_vt[agent_id]

    # ---- Fixed Laplacian scales (widened for full-space targets) ----
    # |e|=15°: exp(-15/30)=0.61  gradient=0.020/°
    # |e|=30°: exp(-30/30)=0.37  gradient=0.012/°
    # |e|=60°: exp(-60/30)=0.14  gradient=0.005/°
    # |e|=90°: exp(-90/30)=0.05  gradient=0.0017/°
    # |e|=180°:exp(-180/30)=0.002 gradient~0
    heading_scale = jnp.deg2rad(30.0)
    pitch_scale   = jnp.deg2rad(25.0)
    roll_scale    = jnp.deg2rad(25.0)
    speed_scale   = 25.0

    heading_r = jnp.exp(-jnp.abs(jnp.clip(delta_heading, -jnp.pi, jnp.pi)) / heading_scale)
    pitch_r   = jnp.exp(-jnp.abs(jnp.clip(delta_pitch,   -jnp.pi, jnp.pi)) / pitch_scale)
    roll_r    = jnp.exp(-jnp.abs(jnp.clip(delta_roll,    -jnp.pi, jnp.pi)) / roll_scale)
    speed_r   = jnp.exp(-jnp.abs(jnp.clip(delta_vt, -1e3, 1e3)) / speed_scale)

    # ---- weights (sum = 1.0) ----
    w_heading = 0.35
    w_pitch   = 0.20
    w_roll    = 0.20
    w_speed   = 0.25

    reward = (heading_r ** w_heading) * (pitch_r ** w_pitch) * \
             (roll_r ** w_roll) * (speed_r ** w_speed)

    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
