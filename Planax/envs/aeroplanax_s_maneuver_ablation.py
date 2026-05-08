"""
S-Maneuver Fidelity Ablation Environment
==========================================
Periodic ±45° heading-switching task ("S-maneuver") with two aerodynamic fidelity modes:
  - "high": Full F-16 hifi aerodynamics (NASA table lookups via hifi_F16)
  - "low":  Simplified linear aerodynamics (no stall, no dynamic derivatives, no coupling)

The fidelity_mode is stored as a Python instance attribute (self.fidelity_mode) so that
JAX JIT can branch at trace time via Python if/else (no jax.lax.cond overhead).

S-maneuver schedule:
  - target_heading alternates between (s_base_heading ± s_heading_amplitude)
  - switch every s_switch_steps agent-interaction steps (default 50 → 10 s)
  - target_pitch = 0 (level flight), target_vt = fixed cruise speed
"""

from typing import Dict, Optional, Tuple, Any
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import functools
import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from gymnax.environments import spaces

from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
from .reward_functions import heading_pitch_V_reward_fn, altitude_reward_fn
from .termination_conditions import crashed_fn, timeout_fn
from .utils.utils import wrap_PI
from .core.simulators import fighterplane, missile
from .core.simulators.fighterplane.dynamics import (
    FighterPlaneState, FighterPlaneControlState,
    atmos, accels, quaternion_to_rpy,
)
from .core.utils import (
    update_blood, check_crashed, check_locked,
    check_shotdown, check_shotdown_by_missile, check_hit, check_miss,
)


# =============================================================================
# Low-fidelity aerodynamic plant
# =============================================================================

def nlplant_lofi(xu):
    """
    Low-fidelity linear aerodynamics. Same interface as nlplant() but replaces
    ~20 NASA table lookups with closed-form expressions:

    Gap vs hifi:
      - CL is purely linear in alpha (no deep stall above ~20°)
      - CD is a parabolic polar (no alpha-beta cross terms)
      - Cm has no Cmq pitch-rate damping → pitch oscillations do not decay
      - Cn has no Cn_beta (directional stability removed) → sideslip grows freely
      - No dynamic derivatives: Clp, Cnr, Cyp, Cyr = 0
      - No aileron-yaw coupling (adverse yaw absent)
      - No leading-edge flap corrections
    """
    xdot = jnp.zeros_like(xu)
    g = 32.17
    m = 636.94
    B = 30.0
    S = 300.0
    cbar = 11.32
    Heng = 0.0
    Jy = 55814.0
    Jxz = 982.0
    Jz = 63100.0
    Jx = 9496.0
    r2d = 180.0 / jnp.pi

    alt = xu[2]
    vt = xu[6]
    alpha_deg = xu[7] * r2d   # degrees
    beta_deg  = xu[8] * r2d   # degrees
    P = xu[9]
    Q = xu[10]
    R = xu[11]

    sa = jnp.sin(xu[7])
    ca = jnp.cos(xu[7])
    sb = jnp.sin(xu[8])
    cb = jnp.cos(xu[8])

    vt = (vt <= 0.01) * 0.01 + (vt > 0.01) * vt

    q0 = xu[12]; q1 = xu[13]; q2 = xu[14]; q3 = xu[15]
    q0sq = q0**2; q1sq = q1**2; q2sq = q2**2; q3sq = q3**2
    q0q1 = q0*q1; q0q2 = q0*q2; q0q3 = q0*q3
    q1q2 = q1*q2; q1q3 = q1*q3; q2q3 = q2*q3

    T      = xu[16]
    el_deg = xu[17]   # degrees
    ail_deg= xu[18]   # degrees
    rud_deg= xu[19]   # degrees
    dail   = ail_deg / 21.5
    drud   = rud_deg / 30.0

    temp = atmos(alt, vt)
    qbar = temp[1]

    U = vt * ca * cb
    V = vt * sb
    W = vt * sa * cb

    # --- Navigation equations (identical to hifi) ---
    xdot = xdot.at[0].set((q0sq+q1sq-q2sq-q3sq)*U + 2*(q1q2+q0q3)*V + 2*(q1q3-q0q2)*W)
    xdot = xdot.at[1].set(2*(q1q2-q0q3)*U + (q0sq-q1sq+q2sq-q3sq)*V + 2*(q2q3+q0q1)*W)
    xdot = xdot.at[2].set(-(2*(q1q3+q0q2)*U + 2*(q2q3-q0q1)*V + (q0sq-q1sq-q2sq+q3sq)*W))
    xdot = xdot.at[3].set(0.0)
    xdot = xdot.at[4].set(0.0)
    xdot = xdot.at[5].set(0.0)

    # --- Lofi aerodynamic coefficients ---
    # Lift: linear slope 0.095/deg, NO stall above ~20°
    CL = 0.095 * alpha_deg
    # Drag: parabolic polar (no beta coupling)
    CD = 0.020 + 0.05 * CL * CL
    # Body-axis: X forward (+thrust direction), Z down in body
    Cx_tot =  CL * sa - CD * ca          # axial
    Cz_tot = -CL * ca - CD * sa          # normal (downward in body)
    # Side force: linear sideslip only, NO aileron/rudder contribution
    Cy_tot = -0.020 * beta_deg
    # Pitch moment: elevator only — NO Cmq rate damping
    Cm_tot = -0.050 * el_deg / 25.0
    # Yaw moment: rudder only — NO Cn_beta (weathercock stability removed)
    Cn_tot =  0.025 * drud
    # Roll moment: aileron only — NO Clp roll damping
    Cl_tot =  0.060 * dail

    # --- Force equations (identical structure to hifi) ---
    Udot = R*V - Q*W + g*2*(q1q3+q0q2) + qbar*S*Cx_tot/m + T/m
    Vdot = P*W - R*U + g*2*(q2q3-q0q1) + qbar*S*Cy_tot/m
    Wdot = Q*U - P*V + g*(q0sq-q1sq-q2sq+q3sq) + qbar*S*Cz_tot/m

    xdot = xdot.at[6].set((U*Udot + V*Vdot + W*Wdot) / (vt + 1e-6))
    xdot = xdot.at[7].set((U*Wdot - W*Udot) / (U*U + W*W + 1e-6))
    xdot = xdot.at[8].set((Vdot*vt - V*xdot[6]) / (vt*vt*cb + 1e-6))

    L_tot = Cl_tot * qbar * S * B
    M_tot = Cm_tot * qbar * S * cbar
    N_tot = Cn_tot * qbar * S * B
    denom = Jx*Jz - Jxz*Jxz + 1e-6

    xdot = xdot.at[9].set( (Jz*L_tot + Jxz*N_tot - (Jz*(Jz-Jy)+Jxz*Jxz)*Q*R + Jxz*(Jx-Jy+Jz)*P*Q + Jxz*Q*Heng) / denom)
    xdot = xdot.at[10].set((M_tot + (Jz-Jx)*P*R - Jxz*(P*P-R*R) - R*Heng) / Jy)
    xdot = xdot.at[11].set((Jx*N_tot + Jxz*L_tot + (Jx*(Jx-Jy)+Jxz*Jxz)*P*Q - Jxz*(Jx-Jy+Jz)*Q*R + Jx*Q*Heng) / denom)

    # --- Quaternion kinematics (identical to hifi) ---
    xdot = xdot.at[12].set(0.5*(       P*q1 + Q*q2 + R*q3))
    xdot = xdot.at[13].set(0.5*(-P*q0       + R*q2 - Q*q3))
    xdot = xdot.at[14].set(0.5*(-Q*q0 - R*q1       + P*q3))
    xdot = xdot.at[15].set(0.5*(-R*q0 + Q*q1 - P*q2      ))

    return xdot


def update_lofi(
    state: FighterPlaneState,
    action: FighterPlaneControlState,
    dt: float,
) -> FighterPlaneState:
    """Low-fidelity aircraft integrator. Same interface as fighterplane.update()."""
    x = jnp.hstack((
        state.north    / 0.3048,
        state.east     / 0.3048,
        state.altitude / 0.3048,
        state.roll,
        state.pitch,
        state.yaw,
        state.vt       / 0.3048,
        state.alpha,
        state.beta,
        state.P, state.Q, state.R,
        state.q0, state.q1, state.q2, state.q3,
    ))

    T   = 0.9 * state.T   + 0.1 * action.throttle * 0.225 * 76300 / 0.3048
    el  = 0.9 * state.el  + 0.1 * action.elevator * 45
    ail = 0.9 * state.ail + 0.1 * action.aileron  * 45
    rud = 0.9 * state.rud + 0.1 * action.rudder   * 45
    u  = jnp.hstack((T, el, ail, rud, action.leading_edge_flap))
    xu = jnp.hstack((x, u))

    xdot = nlplant_lofi(xu)

    nx_cg, ny_cg, nz_cg = accels(
        xu[3], xu[4], xu[7], xu[8], xu[6],
        xdot[7], xdot[8], xdot[6],
        xu[9], xu[10], xu[11],
    )

    new_x = x + xdot[:16] * dt

    new_q0, new_q1, new_q2, new_q3 = new_x[12], new_x[13], new_x[14], new_x[15]
    norm_q = jnp.sqrt(new_q0**2 + new_q1**2 + new_q2**2 + new_q3**2) + 1e-6
    new_q0 /= norm_q; new_q1 /= norm_q; new_q2 /= norm_q; new_q3 /= norm_q

    dot  = state.q0*new_q0 + state.q1*new_q1 + state.q2*new_q2 + state.q3*new_q3
    sign = jnp.where(dot < 0.0, -1.0, 1.0)
    new_q0 *= sign; new_q1 *= sign; new_q2 *= sign; new_q3 *= sign

    roll, pitch, yaw = quaternion_to_rpy(new_q0, -new_q1, -new_q2, -new_q3)

    new_state = state.replace(
        north    = new_x[0]  * 0.3048,
        east     = new_x[1]  * 0.3048,
        altitude = new_x[2]  * 0.3048,
        roll=roll, pitch=pitch, yaw=yaw,
        vel_x = xdot[0] * 0.3048,
        vel_y = xdot[1] * 0.3048,
        vel_z = xdot[2] * 0.3048,
        vt    = new_x[6] * 0.3048,
        alpha = new_x[7],  beta  = new_x[8],
        P     = new_x[9],  Q     = new_x[10], R = new_x[11],
        q0=new_q0, q1=new_q1, q2=new_q2, q3=new_q3,
        T=T, el=el, ail=ail, rud=rud,
        ax=nx_cg, ay=ny_cg, az=nz_cg,
    )
    mask = state.is_alive | state.is_locked
    return jax.lax.cond(mask, lambda: new_state, lambda: state)


# =============================================================================
# Quaternion helpers (same as heading_pitch_V_quaternion_version.py)
# =============================================================================

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
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _quat_from_euler_bn(roll, pitch, yaw):
    cr, sr = jnp.cos(0.5*roll),  jnp.sin(0.5*roll)
    cp, sp = jnp.cos(0.5*pitch), jnp.sin(0.5*pitch)
    cy, sy = jnp.cos(0.5*yaw),   jnp.sin(0.5*yaw)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return jnp.array([qw, qx, qy, qz])

def _target_q_nb_from_heading_pitch(yaw_t, pitch_t, roll_t=0.0):
    return _quat_conj(_quat_from_euler_bn(roll_t, pitch_t, yaw_t))

def _quat_err_nb(q_curr_nb, yaw_t, pitch_t, roll_t=0.0):
    q_curr_nb = _quat_normalize(q_curr_nb)
    q_tgt_nb  = _target_q_nb_from_heading_pitch(yaw_t, pitch_t, roll_t)
    q_err = _quat_mul(q_tgt_nb, _quat_conj(q_curr_nb))
    q_err = jnp.where(q_err[0] < 0.0, -q_err, q_err)
    return q_err

def _rotate_ned_to_body(q_nb, v_n):
    q_nb = _quat_normalize(q_nb)
    p = jnp.array([0.0, v_n[0], v_n[1], v_n[2]])
    qp  = _quat_mul(q_nb, p)
    qpq = _quat_mul(qp, _quat_conj(q_nb))
    return qpq[1:]


# =============================================================================
# Task dataclasses
# =============================================================================

@struct.dataclass
class SManeuverTaskState(EnvState):
    target_heading:      ArrayLike   # (num_agents,)  current heading target
    target_pitch:        ArrayLike   # (num_agents,)  fixed at 0 (level flight)
    target_vt:           ArrayLike   # (num_agents,)  fixed cruise speed
    s_base_heading:      ArrayLike   # (num_agents,)  reference heading at episode start
    s_phase:             ArrayLike   # scalar int 0/1 (0 → +amplitude, 1 → −amplitude)
    last_switch_time:    ArrayLike   # scalar int, env-step time of last heading switch
    heading_turn_counts: ArrayLike   # scalar int, total number of heading switches

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        """extra_state: array of shape (7, num_agents) or compatible."""
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
            s_base_heading=extra_state[3],
            s_phase=jnp.array(0, dtype=jnp.int32),
            last_switch_time=jnp.array(0, dtype=jnp.int32),
            heading_turn_counts=jnp.array(0, dtype=jnp.int32),
        )


@struct.dataclass(frozen=True)
class SManeuverTaskParams(EnvParams):
    num_allies:              int   = 1
    num_enemies:             int   = 0
    num_missiles:            int   = 0
    agent_type:              int   = 0
    action_type:             int   = 1      # discrete
    sim_freq:                int   = 50
    agent_interaction_steps: int   = 10
    max_altitude:          float   = 20000.0
    min_altitude:          float   = 2000.0
    max_vt:                float   = 360.0
    min_vt:                float   = 100.0
    safe_altitude:         float   = 4.0    # km — used by altitude_reward_fn
    danger_altitude:       float   = 3.5    # km

    # S-maneuver timing
    # s_switch_steps: heading switches every this many agent-interaction steps.
    # Default = 50 steps × (10 sim-substeps / 50 Hz) = 10 seconds real time.
    # To change to T seconds: s_switch_steps = T * sim_freq / agent_interaction_steps
    #   e.g. 5 s → 25, 20 s → 100
    s_switch_steps:        int   = 50
    # s_heading_tol: if > 0, also require heading error < this (rad) before switching.
    # Set to 0.0 to use pure time-based switching (original behaviour).
    # e.g. jnp.deg2rad(10.0) → wait until within 10° AND time elapsed.
    s_heading_tol:         float = 0.0            # rad; 0 = time-only
    s_heading_amplitude:   float = jnp.pi / 4    # ±45° heading swing
    s_target_vt:           float = 200.0          # m/s cruise speed
    s_target_pitch:        float = 0.0            # level flight

    # Fidelity switch (Python-level, NOT a JAX array — resolved at JIT trace time)
    fidelity_mode: str = struct.field(pytree_node=False, default="high")

    # Episode length: max_steps is in SECONDS (timeout_fn multiplies by sim_freq/steps)
    max_steps: int = 400   # 400 s → 2000 agent-interaction steps


# =============================================================================
# Environment
# =============================================================================

class AeroPlanaxSManeuverAblationEnv(
    AeroPlanaxEnv[SManeuverTaskState, SManeuverTaskParams]
):
    """
    S-Maneuver ablation environment.

    fidelity_mode="high" → standard F-16 hifi physics (fighterplane.update)
    fidelity_mode="low"  → simplified linear aerodynamics (update_lofi)

    The mode is fixed at __init__ time and selected via Python if/else inside
    the overridden step(), so JIT compiles two distinct graphs.
    """

    def __init__(self, env_params: Optional[SManeuverTaskParams] = None):
        super().__init__(env_params)
        # Store fidelity mode as Python attribute for JIT branching
        self.fidelity_mode: str = env_params.fidelity_mode if env_params else "high"

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i)
            for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i)
            for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(heading_pitch_V_reward_fn, reward_scale=1.0),
            functools.partial(altitude_reward_fn,        reward_scale=1.0, Kv=0.2),
        ]
        self.is_potential = [False, False]

        self.termination_conditions = [
            crashed_fn,
            functools.partial(timeout_fn, max_steps=self.default_params.max_steps),
        ]

    def _get_obs_size(self) -> int:
        return 16

    @property
    def default_params(self) -> SManeuverTaskParams:
        return SManeuverTaskParams()

    # ------------------------------------------------------------------
    # Physics step override — selects hifi or lofi at trace time
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SManeuverTaskState,
        actions: Dict[AgentName, chex.Array],
        params: Optional[SManeuverTaskParams] = None,
    ):
        if params is None:
            params = self.default_params

        # Select physics update function based on fidelity mode
        _physics_fn = update_lofi if self.fidelity_mode == "low" else fighterplane.update

        def update_status(plane_states, missile_states):
            def update_plane_status(ps, crashed, shotdown, locked):
                alive = ps.is_alive | ps.is_locked
                ps = ps.replace(status=jnp.where(alive, jnp.where(locked, 1, 0), ps.status))
                ps = ps.replace(status=jnp.where(jnp.logical_and(crashed, alive), 2, ps.status))
                ps = ps.replace(status=jnp.where(jnp.logical_and(shotdown, alive), 3, ps.status))
                return ps

            crashed      = jax.vmap(check_crashed, in_axes=(None, 0))(plane_states, jnp.arange(self.num_agents))
            false_locked = jnp.zeros_like(crashed, dtype=bool)
            # num_enemies == 0, num_missiles == 0 → always the else branch
            plane_states = update_plane_status(plane_states, crashed, false_locked, false_locked)
            return plane_states, missile_states

        def step_sim_fn(state_st, _):
            plane_states, missile_states = state_st.plane_state, state_st.missile_state
            state_st, action = self._decode_actions(key, state, state_st, actions)
            next_plane_states = jax.vmap(
                _physics_fn, in_axes=(0, 0, None)
            )(plane_states, action, 1 / params.sim_freq)
            next_plane_states, next_missile_states = update_status(next_plane_states, missile_states)
            state_st = state_st.replace(
                plane_state=next_plane_states,
                missile_state=next_missile_states,
                control_state=action,
            )
            return state_st, True

        state_st, _ = jax.lax.scan(
            step_sim_fn, init=state, xs=None, length=self.agent_interaction_steps
        )
        state_st = state_st.replace(time=state.time + 1)

        obs_st = self._get_obs(state_st, params)
        state_st, dones = self.get_termination(state_st, params)
        dones["__all__"] = state_st.done
        state_st, rewards = self.get_reward(state_st, params)
        info = {"success": state_st.success}

        key, key_step = jax.random.split(key)
        state_st, info = self._step_task(key_step, state_st, info, actions, params)

        key, key_reset = jax.random.split(key)
        obs_re, state_re = self.reset(key_reset, params)

        state = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), state_re, state_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return lax.stop_gradient(obs), state, rewards, dones, info

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: chex.PRNGKey,
        params: SManeuverTaskParams,
    ) -> SManeuverTaskState:
        state = super()._init_state(key, params)

        key, k_heading, k_vt, k_alt, k_pitch = jax.random.split(key, 5)
        initial_heading = jax.random.uniform(
            k_heading, shape=(self.num_agents,), minval=0.0, maxval=2.0 * jnp.pi
        )
        vt = jax.random.uniform(
            k_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt
        )
        rand_pitch = jax.random.uniform(
            k_pitch, shape=(self.num_agents,),
            minval=jnp.radians(-30.0), maxval=jnp.radians(30.0),
        )
        q_init = jax.vmap(_quat_from_euler_bn)(
            jnp.zeros_like(rand_pitch), rand_pitch, initial_heading
        )

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=initial_heading,
                vt=vt,
                vel_y=vt,
                q0=q_init[:, 0], q1=q_init[:, 1],
                q2=q_init[:, 2], q3=q_init[:, 3],
            )
        )

        target_heading = wrap_PI(initial_heading + params.s_heading_amplitude)
        target_pitch   = jnp.full((self.num_agents,), params.s_target_pitch)
        target_vt      = jnp.full((self.num_agents,), params.s_target_vt)

        extra = jnp.stack([target_heading, target_pitch, target_vt, initial_heading], axis=0)
        return SManeuverTaskState.create(state, extra)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: SManeuverTaskState,
        params: SManeuverTaskParams,
    ) -> SManeuverTaskState:
        # Random altitude within safe range
        key, k_alt, k_heading, k_vt, k_pitch = jax.random.split(key, 5)
        altitude = jax.random.uniform(
            k_alt, minval=params.min_altitude, maxval=params.max_altitude
        )
        initial_heading = jax.random.uniform(
            k_heading, shape=(self.num_agents,), minval=0.0, maxval=2.0 * jnp.pi
        )
        vt = jax.random.uniform(
            k_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt
        )
        rand_pitch = jax.random.uniform(
            k_pitch, shape=(self.num_agents,),
            minval=jnp.radians(-10.0), maxval=jnp.radians(10.0),
        )
        q_init = jax.vmap(_quat_from_euler_bn)(
            jnp.zeros_like(rand_pitch), rand_pitch, initial_heading
        )

        state = state.replace(
            plane_state=state.plane_state.replace(
                altitude=jnp.full((self.num_agents,), altitude),
                north=jnp.zeros((self.num_agents,)),
                east=jnp.zeros((self.num_agents,)),
                yaw=initial_heading,
                vt=vt,
                vel_y=vt,
                q0=q_init[:, 0], q1=q_init[:, 1],
                q2=q_init[:, 2], q3=q_init[:, 3],
            ),
            target_heading   = wrap_PI(initial_heading + params.s_heading_amplitude),
            target_pitch     = jnp.full((self.num_agents,), params.s_target_pitch),
            target_vt        = jnp.full((self.num_agents,), params.s_target_vt),
            s_base_heading   = initial_heading,
            s_phase          = jnp.array(0, dtype=jnp.int32),
            last_switch_time = jnp.array(0, dtype=jnp.int32),
            heading_turn_counts = jnp.array(0, dtype=jnp.int32),
        )
        return state

    # ------------------------------------------------------------------
    # Task step: periodic heading switch
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: SManeuverTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: SManeuverTaskParams,
    ) -> Tuple[SManeuverTaskState, Dict[str, Any]]:
        elapsed = state.time - state.last_switch_time
        time_ready = elapsed >= params.s_switch_steps

        # Optional heading-tolerance gate: also require heading error < s_heading_tol
        # (only active when s_heading_tol > 0)
        yaw = state.plane_state.yaw[0]
        hdg_err = jnp.abs(wrap_PI(yaw - state.target_heading[0]))
        heading_reached = jnp.where(
            params.s_heading_tol > 0.0,
            hdg_err < params.s_heading_tol,
            jnp.array(True),   # tol=0 → always satisfied
        )
        should_switch = time_ready & heading_reached

        new_phase = jnp.where(should_switch, 1 - state.s_phase, state.s_phase)
        # Phase 0 → base + amplitude, phase 1 → base − amplitude
        sign = jnp.where(new_phase == 0, 1.0, -1.0)
        new_target_heading = wrap_PI(state.s_base_heading + sign * params.s_heading_amplitude)

        new_last_switch_time = jnp.where(
            should_switch, state.time, state.last_switch_time
        )
        new_turn_counts = state.heading_turn_counts + should_switch.astype(jnp.int32)

        state = state.replace(
            s_phase          = new_phase,
            target_heading   = new_target_heading,
            last_switch_time = new_last_switch_time,
            heading_turn_counts = new_turn_counts,
        )

        info["heading_turn_counts"] = state.heading_turn_counts
        info["s_phase"]             = state.s_phase

        return state, info

    # ------------------------------------------------------------------
    # Observation (16-dim quaternion, identical to heading_pitch_V_quat)
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: SManeuverTaskState,
        params: SManeuverTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        16-dim quaternion observation (same as heading_pitch_V_quaternion_version):
          [0:3]  quaternion error vector part
          [3]    (vt - target_vt) / 340
          [4]    altitude / 5000
          [5]    vt / 340
          [6:9]  target direction in body frame
          [9:12] P, Q, R  (rad/s)
          [12:15] sin/cos(alpha), sin/cos(beta)
        """
        q_curr = jnp.stack([
            jnp.nan_to_num(state.plane_state.q0, nan=0.0),
            jnp.nan_to_num(state.plane_state.q1, nan=0.0),
            jnp.nan_to_num(state.plane_state.q2, nan=0.0),
            jnp.nan_to_num(state.plane_state.q3, nan=0.0),
        ], axis=1)  # (B, 4)

        yaw_t   = state.target_heading
        pitch_t = state.target_pitch
        vt_tgt  = state.target_vt

        def _err_row(q_row, yh, ph):
            return _quat_err_nb(q_row, yh, ph, 0.0)

        q_err_batch = jax.vmap(_err_row, in_axes=(0, 0, 0))(q_curr, yaw_t, pitch_t)
        qv = jnp.clip(q_err_batch[:, 1:4], -1.0, 1.0)

        c_th = jnp.cos(yaw_t);   s_th = jnp.sin(yaw_t)
        c_ph = jnp.cos(pitch_t); s_ph = jnp.sin(pitch_t)
        v_n = jnp.stack([c_ph * c_th, c_ph * s_th, s_ph], axis=1)
        v_b = jax.vmap(_rotate_ned_to_body, in_axes=(0, 0))(q_curr, v_n)
        v_b = jnp.clip(v_b, -1.0, 1.0)

        altitude = state.plane_state.altitude
        vt       = state.plane_state.vt
        alpha    = state.plane_state.alpha
        beta     = state.plane_state.beta
        P, Q, R  = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_dvt = (vt - vt_tgt) / 340.0
        norm_alt = altitude / 5000.0
        norm_vt  = vt / 340.0

        obs_mat = jnp.stack([
            qv[:, 0], qv[:, 1], qv[:, 2],
            norm_dvt,
            norm_alt,
            norm_vt,
            v_b[:, 0], v_b[:, 1], v_b[:, 2],
            P, Q, R,
            jnp.sin(alpha), jnp.cos(alpha),
            jnp.sin(beta),  jnp.cos(beta),
        ], axis=0)  # (16, B)

        low  = jnp.array([-1.,-1.,-1.,-2., 0., 0.,-1.,-1.,-1.,-10.,-10.,-10.,-1.,-1.,-1.,-1.]).reshape(-1, 1)
        high = jnp.array([ 1., 1., 1., 2., 5., 2., 1., 1., 1., 10., 10., 10., 1., 1., 1., 1.]).reshape(-1, 1)
        obs_mat = jnp.clip(
            jnp.nan_to_num(obs_mat, nan=0.0, posinf=0.0, neginf=0.0),
            low, high,
        )
        return {agent: obs_mat[:, i] for i, agent in enumerate(self.agents)}
