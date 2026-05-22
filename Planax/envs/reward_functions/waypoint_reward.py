"""Waypoint tracking reward: attitude tracking + waypoint proximity bonus.

Same fixed-Laplacian attitude/speed tracking as the universal baseline,
plus a bonus for closing distance to the waypoint.
"""
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI


def waypoint_reward_fn(
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

    # Fixed Laplacian scales (same as universal baseline)
    heading_scale = jnp.deg2rad(30.0)
    pitch_scale   = jnp.deg2rad(25.0)
    roll_scale    = jnp.deg2rad(25.0)
    speed_scale   = 25.0

    heading_r = jnp.exp(-jnp.abs(jnp.clip(delta_heading, -jnp.pi, jnp.pi)) / heading_scale)
    pitch_r   = jnp.exp(-jnp.abs(jnp.clip(delta_pitch,   -jnp.pi, jnp.pi)) / pitch_scale)
    roll_r    = jnp.exp(-jnp.abs(jnp.clip(delta_roll,    -jnp.pi, jnp.pi)) / roll_scale)
    speed_r   = jnp.exp(-jnp.abs(jnp.clip(delta_vt, -1e3, 1e3)) / speed_scale)

    w_heading = 0.35
    w_pitch   = 0.20
    w_roll    = 0.20
    w_speed   = 0.25

    attitude_r = (heading_r ** w_heading) * (pitch_r ** w_pitch) * \
                 (roll_r ** w_roll) * (speed_r ** w_speed)
    attitude_r = jnp.clip(jnp.nan_to_num(attitude_r, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    # Waypoint proximity bonus: reward closing distance
    wp_n = jnp.nan_to_num(state.waypoint_n[agent_id], nan=0.0)
    wp_e = jnp.nan_to_num(state.waypoint_e[agent_id], nan=0.0)
    plane_n = jnp.nan_to_num(state.plane_state.north[agent_id], nan=0.0)
    plane_e = jnp.nan_to_num(state.plane_state.east[agent_id], nan=0.0)
    dist_to_wp = jnp.sqrt((wp_n - plane_n)**2 + (wp_e - plane_e)**2)

    # Bonus: exp(-dist / 2000m) — half at 1386m, 0.05 at 6000m
    wp_bonus = jnp.exp(-dist_to_wp / 2000.0) * 0.2

    reward = attitude_r + wp_bonus
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
