"""Waypoint tracking environment for trajectory-following baseline.

Generates random waypoints.  Converts them to heading/pitch/roll/vt targets.
Agent learns to track these targets, which naturally guide it toward waypoints.

Key differences from heading_pitch_V:
  - Targets derived from waypoint geometry (not random)
  - Heading locked at waypoint switch (prevents chasing)
  - Waypoint proximity bonus in reward
  - Same observation space -> same network architecture
"""
from typing import Dict, Optional, Tuple, Any
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import functools
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
from .reward_functions import (
    waypoint_reward_fn,
    altitude_reward_fn,
    event_driven_reward_fn,
)
from .termination_conditions import (
    crashed_fn,
    timeout_fn,
)
from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class WaypointTaskState(EnvState):
    waypoint_n: ArrayLike
    waypoint_e: ArrayLike
    waypoint_alt: ArrayLike
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_roll: ArrayLike
    target_vt: ArrayLike
    last_waypoint_time: ArrayLike
    waypoints_reached: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            pre_rewards=env_state.pre_rewards,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            waypoint_n=extra_state[0],
            waypoint_e=extra_state[1],
            waypoint_alt=extra_state[2],
            target_heading=extra_state[3],
            target_pitch=extra_state[4],
            target_roll=extra_state[5],
            target_vt=extra_state[6],
            last_waypoint_time=jnp.zeros_like(env_state.plane_state.north, dtype=jnp.int32),
            waypoints_reached=jnp.zeros_like(env_state.plane_state.north, dtype=jnp.int32),
        )


@struct.dataclass(frozen=True)
class WaypointTaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    formation_type: int = 0
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 20000.0
    min_altitude: float = 2000.0
    max_vt: float = 360.0
    min_vt: float = 120.0
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000
    safe_distance: float = 3000
    # Waypoint params
    wp_switch_interval: float = 90.0       # seconds between waypoint switches
    wp_min_dist: float = 3000.0            # min waypoint distance from plane (m)
    wp_max_dist: float = 15000.0           # max waypoint distance
    wp_reach_radius: float = 1000.0        # waypoint reach radius (m)


class AeroPlanaxWaypointEnv(AeroPlanaxEnv[WaypointTaskState, WaypointTaskParams]):
    def __init__(self, env_params: Optional[WaypointTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(waypoint_reward_fn, reward_scale=1.0),
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=0),
        ]
        self.is_potential = [False] * len(self.reward_functions)

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
        ]

    def _get_obs_size(self) -> int:
        return 22  # same as heading_pitch_V baseline

    @property
    def default_params(self) -> WaypointTaskParams:
        return WaypointTaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: chex.PRNGKey,
        params: WaypointTaskParams,
    ) -> WaypointTaskState:
        state = super()._init_state(key, params)

        key, key_heading = jax.random.split(key)
        initial_heading = jax.random.uniform(key_heading, shape=(self.num_agents,),
                                             minval=0.0, maxval=2.0 * jnp.pi)

        vt = jnp.full((self.num_agents,), 250.0)

        half_heading = initial_heading / 2.0
        q0 = -jnp.cos(half_heading)
        q1 = jnp.zeros((self.num_agents,))
        q2 = jnp.zeros((self.num_agents,))
        q3 = jnp.sin(half_heading)

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=initial_heading, vt=vt, vel_y=vt,
                q0=q0, q1=q1, q2=q2, q3=q3,
            )
        )

        extra = jnp.zeros((7, self.num_agents))
        state = WaypointTaskState.create(state, extra_state=extra)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> WaypointTaskState:
        state = self._generate_formation(key, state, params)

        # Trimmed cruise altitude
        state = state.replace(
            plane_state=state.plane_state.replace(
                altitude=jnp.full((self.num_agents,), 5000.0),
            )
        )

        # Random initial heading
        key, key_heading = jax.random.split(key)
        initial_heading = jax.random.uniform(key_heading, shape=(self.num_agents,),
                                             minval=0.0, maxval=2.0 * jnp.pi)

        vt = jnp.full((self.num_agents,), 250.0)
        vel_y = vt

        half_heading = initial_heading / 2.0
        q0 = -jnp.cos(half_heading)
        q1 = jnp.zeros((self.num_agents,))
        q2 = jnp.zeros((self.num_agents,))
        q3 = jnp.sin(half_heading)

        # Initial waypoint: in front of the plane
        key, key_wp = jax.random.split(key)
        wp_dist = jax.random.uniform(key_wp, shape=(self.num_agents,),
                                     minval=params.wp_min_dist, maxval=params.wp_max_dist)
        wp_n = wp_dist * jnp.cos(initial_heading)
        wp_e = wp_dist * jnp.sin(initial_heading)
        wp_alt = jnp.full((self.num_agents,), 5000.0)

        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_y=vel_y, vt=vt, yaw=initial_heading,
                q0=q0, q1=q1, q2=q2, q3=q3,
            ),
            waypoint_n=wp_n,
            waypoint_e=wp_e,
            waypoint_alt=wp_alt,
            target_heading=initial_heading,
            target_pitch=jnp.zeros((self.num_agents,)),
            target_roll=jnp.zeros((self.num_agents,)),
            target_vt=vt,
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: WaypointTaskParams,
    ) -> Tuple[WaypointTaskState, Dict[str, Any]]:
        """Generate new waypoint on schedule or when reached."""

        # Check if time for a new waypoint
        steps_per_sec = params.sim_freq / params.agent_interaction_steps
        wp_interval_steps = params.wp_switch_interval * steps_per_sec
        time_for_new_wp = (state.time - state.last_waypoint_time) >= wp_interval_steps

        # Check if current waypoint reached
        north = jnp.nan_to_num(state.plane_state.north, nan=0.0)
        east  = jnp.nan_to_num(state.plane_state.east,  nan=0.0)
        alt   = jnp.nan_to_num(state.plane_state.altitude, nan=0.0)
        wp_n = jnp.nan_to_num(state.waypoint_n, nan=0.0)
        wp_e = jnp.nan_to_num(state.waypoint_e, nan=0.0)
        wp_a = jnp.nan_to_num(state.waypoint_alt, nan=0.0)

        dist_to_wp = jnp.sqrt((wp_n - north)**2 + (wp_e - east)**2 + (wp_a - alt)**2)
        wp_reached = dist_to_wp < params.wp_reach_radius

        need_new_wp = time_for_new_wp | wp_reached

        # Generate new random waypoint (in world frame, relative to plane)
        key_wp, key_h, key_v = jax.random.split(key, 3)
        new_wp_dist = jax.random.uniform(key_wp, shape=(self.num_agents,),
                                         minval=params.wp_min_dist, maxval=params.wp_max_dist)
        new_wp_heading = jax.random.uniform(key_h, shape=(self.num_agents,),
                                            minval=-jnp.pi, maxval=jnp.pi)
        new_wp_alt = jax.random.uniform(key_v, shape=(self.num_agents,),
                                        minval=params.min_altitude + 1000,
                                        maxval=params.max_altitude - 1000)

        new_wp_n = north + new_wp_dist * jnp.cos(new_wp_heading)
        new_wp_e = east  + new_wp_dist * jnp.sin(new_wp_heading)

        # Compute locked bearing to waypoint (heading is fixed until next switch)
        d_n = wp_n - north
        d_e = wp_e - east
        d_alt = wp_a - alt
        h_dist = jnp.sqrt(d_n**2 + d_e**2) + 1e-6
        locked_heading = jnp.arctan2(d_e, d_n)
        locked_pitch   = jnp.arctan2(d_alt, h_dist)

        target_vt = jnp.full((self.num_agents,), 250.0)

        state = state.replace(
            waypoint_n=jnp.where(need_new_wp, new_wp_n, wp_n),
            waypoint_e=jnp.where(need_new_wp, new_wp_e, wp_e),
            waypoint_alt=jnp.where(need_new_wp, new_wp_alt, wp_a),
            target_heading=jnp.where(need_new_wp, locked_heading, state.target_heading),
            target_pitch=jnp.where(need_new_wp, locked_pitch, state.target_pitch),
            target_roll=jnp.zeros((self.num_agents,)),
            target_vt=jnp.where(need_new_wp, target_vt, state.target_vt),
            last_waypoint_time=jnp.where(need_new_wp, state.time, state.last_waypoint_time),
            waypoints_reached=(state.waypoints_reached + need_new_wp.astype(jnp.int32)),
            success=False,
        )

        # Logging
        pre = state.pre_rewards
        info["r_attitude_v"] = pre[0]
        info["r_altitude"]   = pre[1]
        info["r_crash"]      = pre[2]

        roll  = jnp.nan_to_num(state.plane_state.roll,  nan=0.0)
        pitch = jnp.nan_to_num(state.plane_state.pitch, nan=0.0)
        yaw   = jnp.nan_to_num(state.plane_state.yaw,   nan=0.0)
        vt    = jnp.nan_to_num(state.plane_state.vt,    nan=0.0)
        tgt_h = jnp.nan_to_num(state.target_heading, nan=0.0)
        tgt_p = jnp.nan_to_num(state.target_pitch,   nan=0.0)
        tgt_r = jnp.nan_to_num(state.target_roll,    nan=0.0)
        tgt_v = jnp.nan_to_num(state.target_vt,      nan=0.0)

        info["dbg_heading_err_deg"] = jnp.abs(wrap_PI(yaw - tgt_h)) * 180.0 / jnp.pi
        info["dbg_pitch_err_deg"]   = jnp.abs(wrap_PI(pitch - tgt_p)) * 180.0 / jnp.pi
        info["dbg_roll_err_deg"]    = jnp.abs(wrap_PI(roll - tgt_r)) * 180.0 / jnp.pi
        info["dbg_speed_err_ms"]    = jnp.abs(vt - tgt_v)
        info["dbg_alt_km"]          = alt / 1000.0
        info["dbg_vt_ms"]           = vt
        info["dbg_wp_dist_km"]      = dist_to_wp / 1000.0
        info["waypoints_reached"]          = state.waypoints_reached

        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """Observation: 22 dims — same structure as heading_pitch_V baseline."""
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_delta_heading = wrap_PI((yaw - state.target_heading))
        norm_delta_pitch   = wrap_PI((pitch - state.target_pitch))
        norm_delta_roll    = wrap_PI((roll - state.target_roll))
        norm_delta_vt = (vt - state.target_vt) / 340
        norm_altitude = altitude / 5000
        norm_vt = vt / 340
        roll_sin = jnp.sin(roll);   roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch); pitch_cos = jnp.cos(pitch)
        alpha_sin = jnp.sin(alpha); alpha_cos = jnp.cos(alpha)
        beta_sin = jnp.sin(beta);   beta_cos = jnp.cos(beta)

        cs = state.control_state
        prev_thr = jnp.nan_to_num(cs.throttle,     nan=0.0)
        prev_el  = jnp.nan_to_num(cs.elevator,     nan=0.0)
        prev_ail = jnp.nan_to_num(cs.aileron,      nan=0.0)
        prev_rud = jnp.nan_to_num(cs.rudder,       nan=0.0)
        prev_sb  = jnp.nan_to_num(cs.speed_brake,  nan=0.0)

        obs = jnp.vstack((norm_delta_heading, norm_delta_pitch, norm_delta_roll, norm_delta_vt,
                            norm_altitude, norm_vt,
                            roll_sin, roll_cos, pitch_sin, pitch_cos,
                            alpha_sin, alpha_cos, beta_sin, beta_cos,
                            P, Q, R,
                            prev_thr, prev_el, prev_ail, prev_rud, prev_sb))
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}

    @functools.partial(jax.jit, static_argnums=(0,))
    def _generate_formation(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> WaypointTaskState:
        if self.formation_type == 0:
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 1:
            team_positions = line_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 2:
            team_positions = diamond_formation(self.num_allies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")

        team_center = jnp.zeros(3)
        key, key_altitude = jax.random.split(key)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        team_center = team_center.at[2].set(altitude)
        formation_positions = enforce_safe_distance(team_positions, team_center, params.safe_distance)
        initial_heading = jnp.full((self.num_agents,), jnp.pi / 2)
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2],
            yaw=initial_heading,
        ))
        return state
