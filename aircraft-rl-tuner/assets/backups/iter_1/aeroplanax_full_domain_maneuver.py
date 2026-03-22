# envs/aeroplanax_full_domain_maneuver.py
# -*- coding: utf-8 -*-
"""
Full-domain attitude control environment.
Supports arbitrary 3D maneuvers: loops, barrel rolls, Split-S, Immelmann, etc.

Key differences from the heading_pitch_V_quaternion_version_add_full_roll env:
  - No pitch/roll clamping (full attitude domain)
  - 22D observation space (energy, load factor, flight path angle, dynamic pressure)
  - 8-level progressive curriculum
  - Relaxed crash limits (500m floor, 12G, etc.)
  - Dual-mode target generation (60% delta, 40% random)

v5 changes:
  - Fixed _step_task to actually update prev_specific_energy with current theta
  - Increased sustained_on_target_steps base: 3 → 10 (require real tracking)
  - Slowed curriculum: advance_threshold 3→5, advance_per_level 2→3
  - Reduced max_interval in termination from 30→20s (less timeout gaming)
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
    full_domain_reward_fn,
    reward_nz_soft_penalty,
    reward_low_qbar_penalty,
)
from .termination_conditions import (
    full_domain_crashed_fn,
    timeout_fn,
    unreach_full_domain_fn,
)
from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance
from .core.utils import check_crashed_full_domain
from .core.simulators import fighterplane
from jax import lax


# ======================== Task State ========================

@struct.dataclass
class FullDomain_TaskState(EnvState):
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_roll: ArrayLike
    target_vt: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike
    # energy tracking (also used to store prev_theta for progress reward)
    prev_specific_energy: ArrayLike
    # curriculum level & real success tracking
    curriculum_level: ArrayLike
    curriculum_success_counts: ArrayLike   # only counts real successes, not timeouts
    # sustained tracking: how many consecutive steps the agent has been on-target
    on_target_steps: ArrayLike
    # timeout tracking for penalty
    timeout_count: ArrayLike

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
            target_heading=extra_state[0],
            target_pitch=extra_state[1],
            target_roll=extra_state[2],
            target_vt=extra_state[3],
            last_check_time=env_state.time,
            heading_turn_counts=0,
            prev_specific_energy=jnp.full_like(extra_state[0], jnp.pi),  # init to pi (max error)
            curriculum_level=jnp.zeros((), dtype=jnp.int32),
            curriculum_success_counts=jnp.zeros((), dtype=jnp.int32),
            on_target_steps=jnp.zeros((), dtype=jnp.int32),
            timeout_count=jnp.zeros((), dtype=jnp.int32),
        )


# ======================== Task Params ========================

@struct.dataclass(frozen=True)
class FullDomain_TaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    formation_type: int = 0
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 20000.0
    min_altitude: float = 2000.0        # spawn altitude lower bound (formation generation)
    # NOTE: crash floor is crash_altitude_limit=500m, not min_altitude

    max_vt: float = 400.0               # from 360
    min_vt: float = 80.0                # from 120
    max_velocities_u_increment: float = 50.0

    max_heading_increment: float = jnp.pi      # from pi/2
    max_pitch_increment: float = jnp.pi / 2    # from pi/6
    max_roll_increment: float = jnp.pi          # from pi/2
    max_altitude_increment: float = 2100.0

    safe_altitude: float = 2.5          # km, from 4.0
    danger_altitude: float = 1.5        # km, from 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000
    safe_distance: float = 3000

    # crash limits (used by full_domain_crashed_fn)
    crash_altitude_limit: float = 500.0  # from 2500
    nz_hard_limit: float = 12.0          # from 10
    qbar_crash_frac: float = 0.10        # from 0.30

    # curriculum
    curriculum_advance_threshold: int = 5   # base success count to advance (increased from 3)
    curriculum_advance_per_level: int = 3   # extra per level (increased from 2)

    # sustained tracking requirement (base value; effective = base + per_level * level)
    sustained_on_target_steps: int = 10     # base: 10 consecutive steps at level 0 (increased from 3)
    sustained_on_target_per_level: int = 2  # +2 per curriculum level (reduced from 3, max 10+7*2=24)


# ======================== Quaternion Helpers ========================
# Convention (consistent with reference env heading_pitch_V_quaternion):
#   _quat_from_euler_nb(roll, pitch, yaw) → raw ZYX Euler-to-quaternion
#   _target_q_bn_from_euler(...)           → conj(raw), matching dynamics state
#   All comparisons (geodesic, q_err, rotation) use the conjugated (BN) convention.

def _quat_normalize(q):
    return q / (jnp.linalg.norm(q) + 1e-9)

def _quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_from_euler_nb(roll, pitch, yaw):
    """ZYX Euler angles to quaternion (same formula as reference env)."""
    cr, sr = jnp.cos(0.5*roll),  jnp.sin(0.5*roll)
    cp, sp = jnp.cos(0.5*pitch), jnp.sin(0.5*pitch)
    cy, sy = jnp.cos(0.5*yaw),   jnp.sin(0.5*yaw)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return jnp.array([qw, qx, qy, qz])

def _target_q_bn_from_euler(yaw_t, pitch_t, roll_t=0.0):
    """Target quaternion = conj(_quat_from_euler_nb(...)).

    Conjugated to match dynamics state convention.
    (Verified against reference env _target_q_bn_from_heading_pitch.)
    """
    return _quat_conj(_quat_from_euler_nb(roll_t, pitch_t, yaw_t))

def _quat_err_bn(q_curr_bn, yaw_t, pitch_t, roll_t):
    """Quaternion error: q_err = q_tgt_bn * conj(q_curr_bn).

    Both q_curr_bn (from dynamics state) and q_tgt_bn (from
    _target_q_bn_from_euler) are in the conjugated (BN) convention.
    Returns shortest-arc error; when on target, q_err ≈ [1,0,0,0].
    """
    q_curr_bn = _quat_normalize(q_curr_bn)
    q_tgt_bn  = _target_q_bn_from_euler(yaw_t, pitch_t, roll_t)
    q_err = _quat_mul(q_tgt_bn, _quat_conj(q_curr_bn))
    # disambiguate: ensure w >= 0 (shortest rotation)
    q_err = jnp.where(q_err[0] < 0.0, -q_err, q_err)
    return q_err

def _rotate_ned_to_body(q_bn, v_n):
    """Rotate vector from NED to Body frame: v_b = q_bn * (0,v_n) * conj(q_bn)."""
    q_bn = _quat_normalize(q_bn)
    p = jnp.array([0.0, v_n[0], v_n[1], v_n[2]])
    qp = _quat_mul(q_bn, p)
    qpq = _quat_mul(qp, _quat_conj(q_bn))
    return qpq[1:]

def _quat_geodesic_angle(q_a, q_b):
    """Geodesic angle between two quaternions."""
    q_a = _quat_normalize(q_a)
    q_b = _quat_normalize(q_b)
    cos_half = jnp.abs(jnp.dot(q_a, q_b))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half)


# ======================== Environment ========================

class AeroPlanaxFullDomainEnv(AeroPlanaxEnv[FullDomain_TaskState, FullDomain_TaskParams]):
    def __init__(self, env_params: Optional[FullDomain_TaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(full_domain_reward_fn, reward_scale=1.0),
            functools.partial(self._nz_penalty_wrapper, scale=0.5),
            functools.partial(self._qbar_penalty_wrapper, scale=0.3),
        ]

        self.is_potential = [False] * len(self.reward_functions)

        self.termination_conditions = [
            full_domain_crashed_fn,
            timeout_fn,
            unreach_full_domain_fn,
        ]

    def _get_obs_size(self) -> int:
        return 22

    # ======================== get_reward (store individual components) ========================

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_reward(self, state, params=None):
        """Override to store per-function rewards in pre_rewards for logging."""
        if params is None:
            params = self.default_params
        agent_ids = jnp.arange(self.num_agents)
        rewards = jnp.zeros(self.num_agents)
        individual = jnp.zeros((len(self.reward_functions), self.num_agents))
        for i in range(len(self.reward_functions)):
            r_i = jax.vmap(self.reward_functions[i], in_axes=(None, None, 0))(
                state, params, agent_ids)
            individual = individual.at[i].set(r_i)
            rewards += r_i
        state = state.replace(pre_rewards=individual)
        rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        return state, rewards

    # ======================== Step (override to use relaxed crash check) ========================

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: FullDomain_TaskState,
        actions: Dict[AgentName, chex.Array],
        params: Optional[FullDomain_TaskParams] = None,
    ):
        """Override base step to use check_crashed_full_domain instead of check_crashed."""
        if params is None:
            params = self.default_params

        def update_status(plane_states):
            """Relaxed crash check for full-domain training."""
            crashed = jax.vmap(
                check_crashed_full_domain, in_axes=(None, 0)
            )(plane_states, jnp.arange(self.num_agents))
            false_arr = jnp.zeros_like(crashed, dtype=bool)
            plane_alive = plane_states.is_alive | plane_states.is_locked
            # reset alive planes to status 0, then mark crashed
            plane_states = plane_states.replace(
                status=jnp.where(plane_alive, 0, plane_states.status)
            )
            plane_states = plane_states.replace(
                status=jnp.where(jnp.logical_and(crashed, plane_alive), 2, plane_states.status)
            )
            return plane_states

        def step_sim_fn(state_st, _):
            state_st, action = self._decode_actions(key, state, state_st, actions)
            next_plane_states = jax.vmap(
                fighterplane.update, in_axes=(0, 0, None)
            )(state_st.plane_state, action, 1 / params.sim_freq)
            next_plane_states = update_status(next_plane_states)
            state_st = state_st.replace(
                plane_state=next_plane_states,
                control_state=action,
            )
            return state_st, True

        state_st, _ = jax.lax.scan(
            step_sim_fn, init=state, xs=None, length=self.agent_interaction_steps,
        )
        state_st = state_st.replace(time=state.time + 1)

        obs_st = self._get_obs(state_st, params)
        state_st, dones = self.get_termination(state_st, params)
        dones["__all__"] = state_st.done
        state_st, rewards = self.get_reward(state_st, params)
        info = {"success": state_st.success}

        # --- Read individual reward components from pre_rewards (stored by get_reward) ---
        info["r_main"] = state_st.pre_rewards[0]
        info["r_nz"] = state_st.pre_rewards[1]
        info["r_qbar"] = state_st.pre_rewards[2]

        key, key_step = jax.random.split(key)
        state_st, info = self._step_task(key_step, state_st, info, actions, params)

        # Auto-reset
        key, key_reset = jax.random.split(key)
        obs_re, state_re = self.reset(key_reset, params)

        state_out = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), state_re, state_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )

        return lax.stop_gradient(obs), state_out, rewards, dones, info

    @staticmethod
    def _nz_penalty_wrapper(state, params, agent_id, scale=0.5):
        """Wrapper to strip jit static_argnums from reward_nz_soft_penalty for vmap compatibility."""
        return reward_nz_soft_penalty.__wrapped__(state, params, agent_id, scale=scale)

    @staticmethod
    def _qbar_penalty_wrapper(state, params, agent_id, scale=0.3):
        """Wrapper to strip jit static_argnums from reward_low_qbar_penalty for vmap compatibility."""
        return reward_low_qbar_penalty.__wrapped__(state, params, agent_id, scale=scale)

    @property
    def default_params(self) -> FullDomain_TaskParams:
        return FullDomain_TaskParams()

    # ======================== Init ========================

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: chex.PRNGKey,
        params: FullDomain_TaskParams,
    ) -> FullDomain_TaskState:
        state = super()._init_state(key, params)

        # random heading [0, 2pi)
        key, key_heading = jax.random.split(key)
        initial_heading = jax.random.uniform(
            key_heading, shape=(self.num_agents,),
            minval=0.0, maxval=2.0 * jnp.pi
        )

        # random speed (safe range for stable initial flight)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(
            key_vt, shape=(self.num_agents,),
            minval=150.0, maxval=300.0
        )

        # random pitch (mild: ±15°, difficulty comes from curriculum targets)
        key, key_pitch = jax.random.split(key)
        rand_pitch = jax.random.uniform(
            key_pitch, shape=(self.num_agents,),
            minval=jnp.radians(-15.0), maxval=jnp.radians(15.0)
        )

        # random roll (mild: ±30°, difficulty comes from curriculum targets)
        key, key_roll = jax.random.split(key)
        rand_roll = jax.random.uniform(
            key_roll, shape=(self.num_agents,),
            minval=jnp.radians(-30.0), maxval=jnp.radians(30.0)
        )

        # quaternion from euler
        q_init = jax.vmap(_quat_from_euler_nb)(rand_roll, rand_pitch, initial_heading)

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=initial_heading,
                roll=rand_roll,
                pitch=rand_pitch,
                vt=vt,
                vel_y=vt,
                q0=q_init[:, 0],
                q1=q_init[:, 1],
                q2=q_init[:, 2],
                q3=q_init[:, 3],
            )
        )

        extra = jnp.stack([initial_heading, rand_pitch, rand_roll, vt], axis=0)
        state = FullDomain_TaskState.create(state, extra)
        return state

    # ======================== Reset Task ========================

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: FullDomain_TaskState,
        params: FullDomain_TaskParams,
    ) -> FullDomain_TaskState:
        """Reset: generate new targets scaled by curriculum level."""
        return state

    # ======================== Step Task ========================

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: FullDomain_TaskState,
        info: Dict,
        actions: Dict[AgentName, chex.Array],
        params: FullDomain_TaskParams,
    ) -> Tuple[FullDomain_TaskState, Dict]:
        """
        Core task logic:
          1. Check if agent is on-target this step (update on_target_steps counter)
          2. Update prev_specific_energy with current theta (for progress tracking)
          3. On success signal: generate new target, update curriculum
          4. Update timeout counter
        """
        agent_id = 0  # single agent

        # ---- compute current quaternion error ----
        q_curr = jnp.array([
            jnp.nan_to_num(state.plane_state.q0[agent_id], nan=1.0),
            jnp.nan_to_num(state.plane_state.q1[agent_id], nan=0.0),
            jnp.nan_to_num(state.plane_state.q2[agent_id], nan=0.0),
            jnp.nan_to_num(state.plane_state.q3[agent_id], nan=0.0),
        ])
        q_curr = _quat_normalize(q_curr)
        q_tgt = _quat_conj(_quat_from_euler_nb(
            state.target_roll[agent_id],
            state.target_pitch[agent_id],
            state.target_heading[agent_id],
        ))
        theta = _quat_geodesic_angle(q_curr, q_tgt)
        theta_deg = theta * 180.0 / jnp.pi

        vt = jnp.nan_to_num(state.plane_state.vt[agent_id], nan=0.0)
        vt_tgt = state.target_vt[agent_id]
        delta_vt = jnp.abs(vt - vt_tgt)

        # ---- on-target check ----
        on_target_now = (theta_deg <= 10.0) & (delta_vt <= 15.0)
        new_on_target_steps = jnp.where(
            on_target_now,
            state.on_target_steps + 1,
            jnp.int32(0),
        )

        # ---- update prev_specific_energy with current theta (for future progress reward) ----
        new_prev_energy = state.prev_specific_energy.at[agent_id].set(theta)

        # ---- sustained success check ----
        curr_level = state.curriculum_level
        base_steps = params.sustained_on_target_steps
        per_level = params.sustained_on_target_per_level
        sustained_threshold = base_steps + curr_level * per_level
        sustained_success = new_on_target_steps >= sustained_threshold

        # ---- is this a success event? (from termination condition) ----
        success_signal = info.get("success", jnp.array(False))
        # Handle multi-agent format: success_signal may be dict or array
        success_flag = jnp.any(jnp.array(list(success_signal.values()) if isinstance(success_signal, dict) else [success_signal]))

        # ---- timeout check ----
        sim_per_decision = params.sim_freq / params.agent_interaction_steps
        elapsed_sec = (state.time - state.last_check_time) / sim_per_decision
        is_timeout = elapsed_sec >= 20.0  # reduced from 30s to discourage timeout gaming

        # A "real" success: sustained tracking (not just timeout)
        real_success = success_flag & sustained_success & (~is_timeout)
        timeout_event = success_flag & is_timeout & (~sustained_success)

        # ---- curriculum update ----
        advance_threshold = params.curriculum_advance_threshold + curr_level * params.curriculum_advance_per_level
        new_success_counts = jnp.where(
            real_success,
            state.curriculum_success_counts + 1,
            state.curriculum_success_counts,
        )
        should_advance = new_success_counts >= advance_threshold
        new_curriculum_level = jnp.where(
            should_advance,
            jnp.minimum(curr_level + 1, jnp.int32(7)),
            curr_level,
        )
        new_success_counts = jnp.where(
            should_advance,
            jnp.int32(0),
            new_success_counts,
        )

        # ---- timeout counter ----
        new_timeout_count = jnp.where(
            timeout_event,
            state.timeout_count + 1,
            state.timeout_count,
        )

        # ---- generate new target on success ----
        key, key_target = jax.random.split(key)
        new_heading, new_pitch, new_roll, new_vt = self._generate_target(
            key_target, state, new_curriculum_level, params
        )

        # Update targets only on success event
        new_target_heading = jnp.where(
            success_flag,
            state.target_heading.at[agent_id].set(new_heading),
            state.target_heading,
        )
        new_target_pitch = jnp.where(
            success_flag,
            state.target_pitch.at[agent_id].set(new_pitch),
            state.target_pitch,
        )
        new_target_roll = jnp.where(
            success_flag,
            state.target_roll.at[agent_id].set(new_roll),
            state.target_roll,
        )
        new_target_vt = jnp.where(
            success_flag,
            state.target_vt.at[agent_id].set(new_vt),
            state.target_vt,
        )
        new_last_check_time = jnp.where(
            success_flag,
            state.time,
            state.last_check_time,
        )
        # Reset on_target_steps when new target assigned
        new_on_target_steps_final = jnp.where(
            success_flag,
            jnp.int32(0),
            new_on_target_steps,
        )

        state = state.replace(
            target_heading=new_target_heading,
            target_pitch=new_target_pitch,
            target_roll=new_target_roll,
            target_vt=new_target_vt,
            last_check_time=new_last_check_time,
            curriculum_level=new_curriculum_level,
            curriculum_success_counts=new_success_counts,
            on_target_steps=new_on_target_steps_final,
            prev_specific_energy=new_prev_energy,
            timeout_count=new_timeout_count,
        )

        # ---- update info for logging ----
        info["theta_deg"] = theta_deg
        info["delta_vt"] = delta_vt
        info["on_target_steps"] = jnp.float32(new_on_target_steps_final)
        info["curriculum_level"] = jnp.float32(new_curriculum_level)
        info["curriculum_success_counts"] = jnp.float32(new_success_counts)
        info["success_times"] = jnp.float32(state.timeout_count + jnp.where(real_success, 1, 0))
        info["timeout_count"] = jnp.float32(new_timeout_count)

        return state, info

    def _generate_target(
        self,
        key: chex.PRNGKey,
        state: FullDomain_TaskState,
        curriculum_level: jnp.ndarray,
        params: FullDomain_TaskParams,
    ) -> Tuple:
        """
        Generate new target based on curriculum level.

        Level 0: small heading changes, zero pitch/roll, small speed changes
        Level 1: moderate heading, small pitch (±15°), small roll (±20°)
        Level 2: full heading, moderate pitch (±30°), moderate roll (±45°)
        Level 3: full heading, moderate pitch (±45°), moderate roll (±90°)
        Level 4: full heading, full pitch (±60°), full roll (±135°)
        Level 5: full heading, full pitch (±70°), full roll (±180°)
        Level 6+: same as 5 but with speed extremes

        Dual-mode: 60% delta from current, 40% random absolute
        """
        agent_id = 0
        key, key_mode = jax.random.split(key)
        key, key_heading = jax.random.split(key)
        key, key_pitch = jax.random.split(key)
        key, key_roll = jax.random.split(key)
        key, key_vt = jax.random.split(key)

        # Current state
        curr_heading = state.plane_state.yaw[agent_id]
        curr_pitch = state.plane_state.pitch[agent_id]
        curr_roll = state.plane_state.roll[agent_id]
        curr_vt = state.plane_state.vt[agent_id]

        # Curriculum-scaled limits
        # Pitch: 0→15°, 1→20°, 2→30°, 3→45°, 4→60°, 5→70°, 6→70°, 7→70°
        pitch_limits = jnp.array([
            jnp.radians(15.0),
            jnp.radians(20.0),
            jnp.radians(30.0),
            jnp.radians(45.0),
            jnp.radians(60.0),
            jnp.radians(70.0),
            jnp.radians(70.0),
            jnp.radians(70.0),
        ])
        # Roll: 0→20°, 1→45°, 2→90°, 3→135°, 4→180°, 5→180°, 6→180°, 7→180°
        roll_limits = jnp.array([
            jnp.radians(20.0),
            jnp.radians(45.0),
            jnp.radians(90.0),
            jnp.radians(135.0),
            jnp.radians(180.0),
            jnp.radians(180.0),
            jnp.radians(180.0),
            jnp.radians(180.0),
        ])
        # Speed range: 0→[150,250], 1→[130,280], 2→[120,300], 3→[100,320], 4→[90,360], 5→[80,400]
        vt_min_limits = jnp.array([150.0, 130.0, 120.0, 100.0, 90.0, 80.0, 80.0, 80.0])
        vt_max_limits = jnp.array([250.0, 280.0, 300.0, 320.0, 360.0, 400.0, 400.0, 400.0])

        pitch_lim = pitch_limits[curriculum_level]
        roll_lim = roll_limits[curriculum_level]
        vt_min = vt_min_limits[curriculum_level]
        vt_max = vt_max_limits[curriculum_level]

        # ---- delta mode ----
        # Heading: always full range delta
        delta_heading = jax.random.uniform(key_heading, minval=-jnp.pi, maxval=jnp.pi)
        delta_pitch = jax.random.uniform(key_pitch, minval=-pitch_lim, maxval=pitch_lim)
        delta_roll = jax.random.uniform(key_roll, minval=-roll_lim, maxval=roll_lim)
        delta_vt = jax.random.uniform(key_vt, minval=-50.0, maxval=50.0)

        tgt_heading_delta = wrap_PI(curr_heading + delta_heading)
        tgt_pitch_delta = jnp.clip(curr_pitch + delta_pitch, -pitch_lim, pitch_lim)
        tgt_roll_delta = jnp.clip(curr_roll + delta_roll, -roll_lim, roll_lim)
        tgt_vt_delta = jnp.clip(curr_vt + delta_vt, vt_min, vt_max)

        # ---- random absolute mode ----
        tgt_heading_rand = jax.random.uniform(key_heading, minval=0.0, maxval=2.0 * jnp.pi)
        tgt_pitch_rand = jax.random.uniform(key_pitch, minval=-pitch_lim, maxval=pitch_lim)
        tgt_roll_rand = jax.random.uniform(key_roll, minval=-roll_lim, maxval=roll_lim)
        tgt_vt_rand = jax.random.uniform(key_vt, minval=vt_min, maxval=vt_max)

        # ---- mode selection: 60% delta, 40% random ----
        use_delta = jax.random.uniform(key_mode) < 0.6

        tgt_heading = jnp.where(use_delta, tgt_heading_delta, tgt_heading_rand)
        tgt_pitch   = jnp.where(use_delta, tgt_pitch_delta,   tgt_pitch_rand)
        tgt_roll    = jnp.where(use_delta, tgt_roll_delta,    tgt_roll_rand)
        tgt_vt      = jnp.where(use_delta, tgt_vt_delta,      tgt_vt_rand)

        return tgt_heading, tgt_pitch, tgt_roll, tgt_vt

    # ======================== Observations ========================

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: FullDomain_TaskState,
        params: FullDomain_TaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        22D observation space:
        [0-3]   quaternion error (q_err components, w,x,y,z)
        [4]     speed error (vt - vt_tgt) / 100
        [5]     speed ratio (vt / vt_tgt)
        [6-8]   angular rates (P, Q, R) / 10
        [9-11]  velocity in body frame (vx, vy, vz) / 100 (approx)
        [12]    altitude / 10000
        [13]    altitude rate (vel_z) / 100
        [14]    specific energy proxy (vt^2/2 + g*alt) / 1e5
        [15]    load factor Nz / 9
        [16]    dynamic pressure qbar / 1e4
        [17]    flight path angle (FPA) / pi
        [18]    roll angle / pi
        [19]    pitch angle / pi
        [20]    curriculum level / 7
        [21]    on-target time ratio (on_target_steps / sustained_threshold)
        """
        def get_agent_obs(agent_id):
            ps = state.plane_state

            # Quaternion error components
            q_curr = jnp.array([
                jnp.nan_to_num(ps.q0[agent_id], nan=1.0),
                jnp.nan_to_num(ps.q1[agent_id], nan=0.0),
                jnp.nan_to_num(ps.q2[agent_id], nan=0.0),
                jnp.nan_to_num(ps.q3[agent_id], nan=0.0),
            ])
            q_err = _quat_err_bn(
                q_curr,
                state.target_heading[agent_id],
                state.target_pitch[agent_id],
                state.target_roll[agent_id],
            )

            # Speed quantities
            vt = jnp.nan_to_num(ps.vt[agent_id], nan=150.0)
            vt_tgt = state.target_vt[agent_id]
            speed_err = (vt - vt_tgt) / 100.0
            speed_ratio = jnp.clip(vt / (vt_tgt + 1e-6), 0.5, 2.0) - 1.0

            # Angular rates
            P = jnp.nan_to_num(ps.P[agent_id], nan=0.0) / 10.0
            Q = jnp.nan_to_num(ps.Q[agent_id], nan=0.0) / 10.0
            R = jnp.nan_to_num(ps.R[agent_id], nan=0.0) / 10.0

            # Velocity body frame (approximate from NED velocity)
            vel_x = jnp.nan_to_num(ps.vel_x[agent_id] if hasattr(ps, 'vel_x') else 0.0, nan=0.0)
            vel_y_val = jnp.nan_to_num(ps.vel_y[agent_id], nan=vt)
            vel_z = jnp.nan_to_num(ps.vel_z[agent_id], nan=0.0)

            # Altitude quantities
            alt = jnp.nan_to_num(ps.altitude[agent_id], nan=5000.0)
            alt_norm = alt / 10000.0
            alt_rate = vel_z / 100.0

            # Specific energy
            g = 9.81
            spec_energy = (0.5 * vt ** 2 + g * alt) / 1e5

            # Load factor (Nz)
            nz = jnp.nan_to_num(ps.nz[agent_id] if hasattr(ps, 'nz') else 1.0, nan=1.0)
            nz_norm = jnp.clip(nz / 9.0, -2.0, 2.0)

            # Dynamic pressure
            rho = 1.225 * jnp.exp(-alt / 8500.0)  # rough ISA
            qbar = 0.5 * rho * vt ** 2
            qbar_norm = qbar / 1e4

            # Flight path angle
            fpa = jnp.arctan2(-vel_z, jnp.maximum(vel_y_val, 1.0))
            fpa_norm = fpa / jnp.pi

            # Euler angles
            roll = jnp.nan_to_num(ps.roll[agent_id], nan=0.0)
            pitch = jnp.nan_to_num(ps.pitch[agent_id], nan=0.0)
            roll_norm = roll / jnp.pi
            pitch_norm = pitch / (jnp.pi / 2.0)

            # Curriculum info
            curr_level_norm = jnp.float32(state.curriculum_level) / 7.0
            base_steps = params.sustained_on_target_steps
            per_level = params.sustained_on_target_per_level
            sustained_threshold = jnp.float32(base_steps + state.curriculum_level * per_level)
            on_target_ratio = jnp.clip(
                jnp.float32(state.on_target_steps) / jnp.maximum(sustained_threshold, 1.0),
                0.0, 1.0,
            )

            obs = jnp.array([
                q_err[0], q_err[1], q_err[2], q_err[3],  # [0-3]
                speed_err, speed_ratio,                    # [4-5]
                P, Q, R,                                   # [6-8]
                vel_x / 100.0, vel_y_val / 100.0, vel_z / 100.0,  # [9-11]
                alt_norm, alt_rate,                        # [12-13]
                spec_energy,                               # [14]
                nz_norm,                                   # [15]
                qbar_norm,                                 # [16]
                fpa_norm,                                  # [17]
                roll_norm, pitch_norm,                     # [18-19]
                curr_level_norm,                           # [20]
                on_target_ratio,                           # [21]
            ], dtype=jnp.float32)

            return jnp.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        obs = {
            agent: get_agent_obs(i)
            for i, agent in enumerate(self.agents)
        }
        return obs
