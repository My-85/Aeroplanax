# envs/aeroplanax_s_maneuver.py
# -*- coding: utf-8 -*-
"""
S-maneuver task environment.

Reuses heading_pitch_V baseline's 16D observation, reward, and termination.
Replaces _step_task with S-curve waypoint tracking logic:
  waypoint_north = origin_n + idx * dn
  waypoint_east  = origin_e + amplitude * sin(pi * idx / points_per_half)
When aircraft reaches a waypoint (horizontal distance < reach_radius),
advance to next waypoint and update target_heading accordingly.
target_pitch = 0 (altitude lock), target_vt = constant cruise speed.
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
    heading_pitch_V_reward_fn,
    altitude_reward_fn,
)
from .termination_conditions import (
    crashed_fn,
    timeout_fn,
)
from .utils.utils import wrap_PI


@struct.dataclass
class S_Maneuver_TaskState(EnvState):
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_vt: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike
    # S-maneuver specific
    s_origin_n: ArrayLike
    s_origin_e: ArrayLike
    current_wp_idx: ArrayLike
    waypoints_reached: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array,
               s_origin_n=None, s_origin_e=None,
               current_wp_idx=None, waypoints_reached=None):
        # extra_state[0] shape = (num_agents,), use it to derive consistent shapes
        num_agents_shape = extra_state[0].shape
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            pre_rewards=env_state.pre_rewards,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            target_heading=extra_state[0],
            target_pitch=extra_state[1],
            target_vt=extra_state[2],
            last_check_time=jnp.full(num_agents_shape, env_state.time, dtype=jnp.float32),
            heading_turn_counts=jnp.zeros(num_agents_shape, dtype=jnp.int32),
            s_origin_n=s_origin_n,
            s_origin_e=s_origin_e,
            current_wp_idx=current_wp_idx,
            waypoints_reached=waypoints_reached,
        )


@struct.dataclass(frozen=True)
class S_Maneuver_TaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1  # discrete
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
    # S-maneuver parameters
    s_amplitude: float = 3000.0
    s_half_period_north: float = 12000.0
    s_points_per_half: int = 30
    s_target_vt: float = 250.0
    reach_radius: float = 400.0
    max_waypoints: int = 200
    s_altitude_lock: bool = struct.field(pytree_node=False, default=True)


class AeroPlanaxSManeuverEnv(AeroPlanaxEnv[S_Maneuver_TaskState, S_Maneuver_TaskParams]):
    def __init__(self, env_params: Optional[S_Maneuver_TaskParams] = None):
        super().__init__(env_params)

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(heading_pitch_V_reward_fn, reward_scale=1.0),
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
        ]
        self.is_potential = [False] * len(self.reward_functions)

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            self._term_s_maneuver_done,
        ]

    def _get_obs_size(self) -> int:
        return 16

    @property
    def default_params(self) -> S_Maneuver_TaskParams:
        return S_Maneuver_TaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _term_s_maneuver_done(self, state: S_Maneuver_TaskState,
                               params: S_Maneuver_TaskParams, agent_id: AgentID):
        done = state.waypoints_reached[agent_id] >= params.max_waypoints
        return done, done  # done=True, success=True

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: chex.PRNGKey,
        params: S_Maneuver_TaskParams,
    ) -> S_Maneuver_TaskState:
        state = super()._init_state(key, params)

        # Fixed heading = 0 (north)
        initial_heading = jnp.zeros((self.num_agents,))

        # Random speed and altitude
        key, key_vt, key_alt = jax.random.split(key, 3)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=200.0, maxval=280.0)
        alt = jax.random.uniform(key_alt, shape=(self.num_agents,), minval=5000.0, maxval=10000.0)

        # Quaternion for heading=0: q_NB
        # half_heading = 0 => q0 = -cos(0) = -1, q3 = sin(0) = 0
        q0 = jnp.full((self.num_agents,), -1.0)
        q1 = jnp.zeros((self.num_agents,))
        q2 = jnp.zeros((self.num_agents,))
        q3 = jnp.zeros((self.num_agents,))

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=initial_heading,
                vt=vt,
                vel_y=vt,
                altitude=alt,
                q0=q0, q1=q1, q2=q2, q3=q3,
            )
        )

        # S-curve origin = current position
        s_origin_n = state.plane_state.north
        s_origin_e = state.plane_state.east

        # Compute first waypoint (idx=1) heading
        dn = params.s_half_period_north / params.s_points_per_half
        wp_idx = jnp.ones((self.num_agents,), dtype=jnp.int32)
        wp_n = s_origin_n + 1.0 * dn
        wp_e = s_origin_e + params.s_amplitude * jnp.sin(
            jnp.pi * 1.0 / params.s_points_per_half
        )
        delta_n = wp_n - state.plane_state.north
        delta_e = wp_e - state.plane_state.east
        init_target_heading = jnp.arctan2(delta_e, delta_n)

        target_vt = jnp.full((self.num_agents,), params.s_target_vt)
        target_pitch = jnp.zeros((self.num_agents,))

        extra_state = jnp.stack([init_target_heading, target_pitch, target_vt], axis=0)

        state = S_Maneuver_TaskState.create(
            state, extra_state=extra_state,
            s_origin_n=s_origin_n,
            s_origin_e=s_origin_e,
            current_wp_idx=wp_idx,
            waypoints_reached=jnp.zeros((self.num_agents,), dtype=jnp.int32),
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: S_Maneuver_TaskState,
        params: S_Maneuver_TaskParams,
    ) -> S_Maneuver_TaskState:
        # On reset, restart S-curve from current position
        s_origin_n = state.plane_state.north
        s_origin_e = state.plane_state.east

        dn = params.s_half_period_north / params.s_points_per_half
        wp_n = s_origin_n + 1.0 * dn
        wp_e = s_origin_e + params.s_amplitude * jnp.sin(
            jnp.pi * 1.0 / params.s_points_per_half
        )
        delta_n = wp_n - state.plane_state.north
        delta_e = wp_e - state.plane_state.east
        target_heading = jnp.arctan2(delta_e, delta_n)

        state = state.replace(
            target_heading=target_heading,
            target_pitch=jnp.zeros_like(state.target_pitch),
            target_vt=jnp.full_like(state.target_vt, params.s_target_vt),
            s_origin_n=s_origin_n,
            s_origin_e=s_origin_e,
            current_wp_idx=jnp.ones_like(state.current_wp_idx),
            waypoints_reached=jnp.zeros_like(state.waypoints_reached),
            last_check_time=jnp.full_like(state.last_check_time, state.time),
            heading_turn_counts=jnp.zeros_like(state.heading_turn_counts),
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: S_Maneuver_TaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: S_Maneuver_TaskParams,
    ) -> Tuple[S_Maneuver_TaskState, Dict[str, Any]]:
        """Waypoint tracking along S-curve. Advance waypoint when reached."""
        dn_step = params.s_half_period_north / params.s_points_per_half

        # Current waypoint position (per agent)
        idx_f = state.current_wp_idx.astype(jnp.float32)
        wp_n = state.s_origin_n + idx_f * dn_step
        wp_e = state.s_origin_e + params.s_amplitude * jnp.sin(
            jnp.pi * idx_f / params.s_points_per_half
        )

        # Horizontal distance to waypoint
        delta_n = wp_n - state.plane_state.north
        delta_e = wp_e - state.plane_state.east
        hdist = jnp.sqrt(delta_n ** 2 + delta_e ** 2)

        # Reached?
        reached = hdist <= params.reach_radius

        # Advance index if reached
        new_idx = jnp.where(reached, state.current_wp_idx + 1, state.current_wp_idx)
        new_reached_count = jnp.where(reached,
                                       state.waypoints_reached + 1,
                                       state.waypoints_reached)

        # Compute heading to NEW waypoint (after potential advance)
        new_idx_f = new_idx.astype(jnp.float32)
        new_wp_n = state.s_origin_n + new_idx_f * dn_step
        new_wp_e = state.s_origin_e + params.s_amplitude * jnp.sin(
            jnp.pi * new_idx_f / params.s_points_per_half
        )
        new_delta_n = new_wp_n - state.plane_state.north
        new_delta_e = new_wp_e - state.plane_state.east
        new_target_heading = jnp.arctan2(new_delta_e, new_delta_n)

        # Update heading_turn_counts on reach (for compatibility with reward/termination)
        new_turn_counts = jnp.where(reached,
                                     state.heading_turn_counts + 1,
                                     state.heading_turn_counts)

        state = state.replace(
            target_heading=new_target_heading,
            target_pitch=jnp.zeros_like(state.target_pitch),
            target_vt=jnp.full_like(state.target_vt, params.s_target_vt),
            current_wp_idx=new_idx,
            waypoints_reached=new_reached_count,
            heading_turn_counts=new_turn_counts,
            last_check_time=jnp.where(reached,
                                       jnp.full_like(state.last_check_time, state.time),
                                       state.last_check_time),
        )

        info["waypoints_reached"] = new_reached_count
        info["hdist_to_wp"] = hdist
        info["heading_turn_counts"] = new_turn_counts

        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: S_Maneuver_TaskState,
        params: S_Maneuver_TaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        16D observation identical to heading_pitch_V:
            0. delta_heading (rad)
            1. delta_pitch (rad)
            2. delta_vt / 340
            3. altitude / 5000
            4. vt / 340
            5-6. sin/cos(roll)
            7-8. sin/cos(pitch)
            9-10. sin/cos(alpha)
            11-12. sin/cos(beta)
            13-15. P, Q, R
        """
        altitude = state.plane_state.altitude
        roll = state.plane_state.roll
        pitch = state.plane_state.pitch
        yaw = state.plane_state.yaw
        vt = state.plane_state.vt
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_delta_heading = wrap_PI(yaw - state.target_heading)
        norm_delta_pitch = wrap_PI(pitch - state.target_pitch)
        norm_delta_vt = (vt - state.target_vt) / 340
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vt = vt / 340
        alpha_sin = jnp.sin(alpha)
        alpha_cos = jnp.cos(alpha)
        beta_sin = jnp.sin(beta)
        beta_cos = jnp.cos(beta)

        obs = jnp.vstack((norm_delta_heading, norm_delta_pitch, norm_delta_vt,
                           norm_altitude, norm_vt,
                           roll_sin, roll_cos, pitch_sin, pitch_cos,
                           alpha_sin, alpha_cos, beta_sin, beta_cos,
                           P, Q, R))
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
