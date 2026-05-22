"""
planax_hold_env.py — Altitude / Velocity / Attitude Hold Task

Goal: Stabilize an F-16 at:
  - altitude 5000 m
  - speed    170 m/s  (≈ 0.5 Mach)
  - roll     0 rad
  - beta     0 rad  (coordinated flight)
  - pitch    at the natural trim angle

Reset: Forces the initial state near the goal then adds small random
disturbances to roll (±15°), pitch (±5°), and body rates P/Q/R (±0.1 rad/s).

Observation (15-dim):
  [0]  Δh / 1000            altitude error normalised
  [1]  ΔVt / 170            speed error normalised
  [2]  roll                 rad
  [3]  sin(pitch)
  [4]  cos(pitch)
  [5]  sin(yaw)
  [6]  cos(yaw)
  [7]  sin(alpha)
  [8]  cos(alpha)
  [9]  sin(beta)
  [10] cos(beta)
  [11] P                    roll rate  rad/s
  [12] Q                    pitch rate rad/s
  [13] R                    yaw rate   rad/s
  [14] prev_throttle_norm   0→1
  [15] prev_elevator_norm  -1→1
  [16] prev_aileron_norm   -1→1
  [17] prev_rudder_norm    -1→1

Action space: discrete (same convention as existing Planax envs)
  throttle: 31 bins  [0/30, 1/30, …, 1.0]
  elevator: 41 bins  [index*2/40 - 1]
  aileron:  41 bins
  rudder:   41 bins
"""

import functools
from typing import Dict, Optional, Tuple, Any

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from .aeroplanax import (
    AgentID,
    AgentName,
    AeroPlanaxEnv,
    EnvParams,
    EnvState,
)
from .core.simulators.fighterplane.dynamics import quaternion_to_rpy

# ──────────────────────────────────────────────────────────────────────────
# Target constants (SI units)
# ──────────────────────────────────────────────────────────────────────────
TARGET_ALT_M: float = 5000.0   # metres
TARGET_VT_MS: float = 170.0    # m/s
TRIM_ALPHA_RAD: float = 0.0714  # ≈ 4.09°  JSBSim trim value

# ──────────────────────────────────────────────────────────────────────────
# Helper: euler → quaternion  (NED→Body convention, scalar-first)
# ──────────────────────────────────────────────────────────────────────────
def _euler_to_quat_nb(roll, pitch, yaw):
    """Returns q_Body_to_NED stored in (q0,q1,q2,q3) form expected by
    dynamics.py (conjugate of the NED-to-Body quaternion)."""
    cr, sr = jnp.cos(0.5 * roll),  jnp.sin(0.5 * roll)
    cp, sp = jnp.cos(0.5 * pitch), jnp.sin(0.5 * pitch)
    cy, sy = jnp.cos(0.5 * yaw),   jnp.sin(0.5 * yaw)
    # q_NED_to_Body
    qw =  cr*cp*cy + sr*sp*sy
    qx =  sr*cp*cy - cr*sp*sy
    qy =  cr*sp*cy + sr*cp*sy
    qz =  cr*cp*sy - sr*sp*cy
    # dynamics.py stores q_Body_to_NED = conjugate
    return qw, -qx, -qy, -qz


# ──────────────────────────────────────────────────────────────────────────
# Task state
# ──────────────────────────────────────────────────────────────────────────
@struct.dataclass
class HoldTaskState(EnvState):
    # previous normalised control outputs (for obs & smoothness penalty)
    prev_thr: chex.Array    # shape (num_agents,)  0→1
    prev_el:  chex.Array    # shape (num_agents,) -1→1
    prev_ail: chex.Array    # shape (num_agents,)
    prev_rud: chex.Array    # shape (num_agents,)


# ──────────────────────────────────────────────────────────────────────────
# Task params
# ──────────────────────────────────────────────────────────────────────────
@struct.dataclass(frozen=True)
class HoldTaskParams(EnvParams):
    num_allies:            int   = 1
    num_enemies:           int   = 0
    num_missiles:          int   = 0
    agent_type:            int   = 0       # 0 = fighterplane
    action_type:           int   = 1       # 1 = discrete
    sim_freq:              int   = 50      # Hz
    agent_interaction_steps: int = 10      # physics steps per RL step → dt_RL = 0.2 s
    max_steps:             int   = 500     # 500 RL steps × 0.2 s = 100 s episode

    # Episode safety limits
    min_altitude: float = 4000.0
    max_altitude: float = 6000.0

    # targets
    target_alt:  float = TARGET_ALT_M
    target_vt:   float = TARGET_VT_MS

    # initial perturbation magnitudes
    init_roll_noise:  float = jnp.radians(15.0)
    init_pitch_noise: float = jnp.radians(5.0)
    init_pqr_noise:   float = 0.1          # rad/s

    # reward weights
    w_alt:      float = 0.20
    w_vt:       float = 0.15
    w_roll:     float = 0.20
    w_beta:     float = 0.15
    w_pqr:      float = 0.10
    w_alive:    float = 0.05
    # smoothness penalty weights
    w_thr_rate: float = 0.10
    w_srf_rate: float = 0.05


# ──────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────
class AeroPlanaxHoldEnv(AeroPlanaxEnv[HoldTaskState, HoldTaskParams]):
    """Altitude / Speed / Attitude Hold environment for fixed-wing RL research."""

    def __init__(self, env_params: Optional[HoldTaskParams] = None):
        if env_params is None:
            env_params = HoldTaskParams()
        super().__init__(env_params)

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i)
            for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i)
            for i, agent in enumerate(self.agents)
        }

        # no external reward_functions / termination_conditions lists needed;
        # we implement everything inline for clarity
        self.reward_functions = []
        self.termination_conditions = []
        self.is_potential = []

    # ------------------------------------------------------------------
    @property
    def default_params(self) -> HoldTaskParams:
        return HoldTaskParams()

    def _get_obs_size(self) -> int:
        return 18   # see module docstring

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(self, key: chex.PRNGKey, params: HoldTaskParams) -> HoldTaskState:
        """Build the base EnvState (from parent) then wrap with HoldTaskState."""
        base = super()._init_state(key, params)
        n = self.num_agents
        return HoldTaskState(
            plane_state=base.plane_state,
            missile_state=base.missile_state,
            control_state=base.control_state,
            pre_rewards=base.pre_rewards,
            done=base.done,
            success=base.success,
            time=base.time,
            prev_thr=jnp.zeros((n,)),
            prev_el=jnp.zeros((n,)),
            prev_ail=jnp.zeros((n,)),
            prev_rud=jnp.zeros((n,)),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(self, key: chex.PRNGKey,
                    state: HoldTaskState,
                    params: HoldTaskParams) -> HoldTaskState:
        """Perturb the trimmed flight condition and set control history to zero."""
        n = self.num_agents

        key, k_roll, k_pitch, k_pqr, k_yaw = jax.random.split(key, 5)

        # Small random initial perturbations
        d_roll  = jax.random.uniform(k_roll,  (n,), minval=-params.init_roll_noise,  maxval=params.init_roll_noise)
        d_pitch = jax.random.uniform(k_pitch, (n,), minval=-params.init_pitch_noise, maxval=params.init_pitch_noise)
        d_pqr   = jax.random.uniform(k_pqr,   (n, 3), minval=-params.init_pqr_noise, maxval=params.init_pqr_noise)

        # Initial heading: random so the policy sees diverse yaw angles
        init_yaw = jax.random.uniform(k_yaw, (n,), minval=0.0, maxval=2.0 * jnp.pi)

        # trim pitch = TRIM_ALPHA_RAD (aircraft flies level at this pitch)
        init_pitch = jnp.full((n,), TRIM_ALPHA_RAD) + d_pitch
        init_roll  = d_roll   # near-zero roll

        # Quaternion consistent with the perturbed attitude
        q0, q1, q2, q3 = jax.vmap(_euler_to_quat_nb)(init_roll, init_pitch, init_yaw)

        # rl trim throttle is ~0.065 (from JSBSim trim study), corresponding
        # to action index 2/30  ≈ 0.067, normalised value ≈ 0.065
        trim_throttle_norm = jnp.full((n,), 0.065)
        # elevator at trim: el_norm = el_deg / 45 where el_deg ≈ -1.27°
        trim_el_norm = jnp.full((n,), -1.27 / 45.0)

        plane = state.plane_state.replace(
            altitude = jnp.full((n,), TARGET_ALT_M),
            vt       = jnp.full((n,), TARGET_VT_MS),
            roll     = init_roll,
            pitch    = init_pitch,
            yaw      = init_yaw,
            vel_x    = TARGET_VT_MS * jnp.cos(init_pitch),  # rough decomposition
            vel_y    = jnp.zeros((n,)),
            vel_z    = jnp.zeros((n,)),
            alpha    = jnp.full((n,), TRIM_ALPHA_RAD),
            beta     = jnp.zeros((n,)),
            P        = d_pqr[:, 0],
            Q        = d_pqr[:, 1],
            R        = d_pqr[:, 2],
            q0       = q0,
            q1       = q1,
            q2       = q2,
            q3       = q3,
            north    = jnp.zeros((n,)),
            east     = jnp.zeros((n,)),
        )

        return state.replace(
            plane_state = plane,
            time        = 0,
            done        = False,
            success     = False,
            prev_thr    = trim_throttle_norm,
            prev_el     = trim_el_norm,
            prev_ail    = jnp.zeros((n,)),
            prev_rud    = jnp.zeros((n,)),
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: HoldTaskState,
                 params: HoldTaskParams) -> Dict[AgentName, chex.Array]:
        ps = state.plane_state
        alt   = jnp.nan_to_num(ps.altitude)
        vt    = jnp.nan_to_num(ps.vt)
        roll  = jnp.nan_to_num(ps.roll)
        pitch = jnp.nan_to_num(ps.pitch)
        yaw   = jnp.nan_to_num(ps.yaw)
        alpha = jnp.nan_to_num(ps.alpha)
        beta  = jnp.nan_to_num(ps.beta)
        P     = jnp.nan_to_num(ps.P)
        Q     = jnp.nan_to_num(ps.Q)
        R     = jnp.nan_to_num(ps.R)

        d_alt = (alt - params.target_alt) / 1000.0      # normalise to ~[-1,1]
        d_vt  = (vt  - params.target_vt)  / params.target_vt

        obs = jnp.stack([
            d_alt,
            d_vt,
            roll,
            jnp.sin(pitch),
            jnp.cos(pitch),
            jnp.sin(yaw),
            jnp.cos(yaw),
            jnp.sin(alpha),
            jnp.cos(alpha),
            jnp.sin(beta),
            jnp.cos(beta),
            P,
            Q,
            R,
            state.prev_thr,
            state.prev_el,
            state.prev_ail,
            state.prev_rud,
        ], axis=0)   # shape (18, n_agents)

        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}

    # ------------------------------------------------------------------
    # Reward   (inline, called from within step)
    # ------------------------------------------------------------------
    def _compute_reward(self, state: HoldTaskState,
                        params: HoldTaskParams,
                        cur_thr_norm, cur_el_norm,
                        cur_ail_norm, cur_rud_norm) -> jnp.ndarray:
        """Returns reward array of shape (num_agents,)."""
        ps = state.plane_state
        alt  = jnp.nan_to_num(ps.altitude)
        vt   = jnp.nan_to_num(ps.vt)
        roll = jnp.nan_to_num(ps.roll)
        beta = jnp.nan_to_num(ps.beta)
        P    = jnp.nan_to_num(ps.P)
        Q    = jnp.nan_to_num(ps.Q)
        R    = jnp.nan_to_num(ps.R)
        alive = ps.is_alive.astype(jnp.float32)

        # ── state-keeping rewards (Gaussian-shaped, range 0→1) ──
        r_alt  = jnp.exp(-((alt - params.target_alt) / 200.0)   ** 2)
        r_vt   = jnp.exp(-((vt  - params.target_vt)  / 20.0)    ** 2)
        r_roll = jnp.exp(-(roll                        / 0.2617) ** 2)  # ±15° = ±0.26 rad
        r_beta = jnp.exp(-(beta                        / 0.0873) ** 2)  # ±5°

        # angular rate penalty (want P,Q,R ≈ 0)
        pqr_mag = jnp.sqrt(P**2 + Q**2 + R**2 + 1e-8)
        r_pqr = jnp.exp(-(pqr_mag / 0.3) ** 2)

        # ── control-smoothness penalties ──
        d_thr = jnp.abs(cur_thr_norm - state.prev_thr)
        d_srf = (jnp.abs(cur_el_norm  - state.prev_el)
               + jnp.abs(cur_ail_norm - state.prev_ail)
               + jnp.abs(cur_rud_norm - state.prev_rud))

        r_thr_smooth = -d_thr        # range 0→-1
        r_srf_smooth = -d_srf / 3.0  # normalised

        # ── survival bonus ──
        r_alive = alive  # 1 when alive, 0 when crashed

        # ── weighted sum ──
        reward = (
              params.w_alt      * r_alt
            + params.w_vt       * r_vt
            + params.w_roll     * r_roll
            + params.w_beta     * r_beta
            + params.w_pqr      * r_pqr
            + params.w_alive    * r_alive
            + params.w_thr_rate * r_thr_smooth
            + params.w_srf_rate * r_srf_smooth
        )
        return jnp.nan_to_num(reward, nan=0.0)

    # ------------------------------------------------------------------
    # Termination  (inline)
    # ------------------------------------------------------------------
    def _compute_done(self, state: HoldTaskState,
                      params: HoldTaskParams) -> jnp.ndarray:
        """Returns done bool array (num_agents,)."""
        ps = state.plane_state
        alt   = jnp.nan_to_num(ps.altitude)
        roll  = jnp.nan_to_num(ps.roll)
        pitch = jnp.nan_to_num(ps.pitch)
        alpha = jnp.nan_to_num(ps.alpha)

        out_of_alt  = (alt < params.min_altitude) | (alt > params.max_altitude)
        roll_limit  = jnp.abs(roll)  > jnp.radians(85.0)
        pitch_limit = jnp.abs(pitch) > jnp.radians(60.0)
        stall       = jnp.abs(alpha) > jnp.radians(40.0)
        timeout     = state.time >= params.max_steps

        return out_of_alt | roll_limit | pitch_limit | stall | timeout | (~ps.is_alive)

    # ------------------------------------------------------------------
    # Step task (state updates that require the raw normalised action)
    # ------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(self, key: chex.PRNGKey,
                   state: HoldTaskState,
                   info: Dict[str, Any],
                   action: Dict[AgentName, chex.Array],
                   params: HoldTaskParams) -> Tuple[HoldTaskState, Dict[str, Any]]:
        """Decode action to normalised values and store in state for next obs."""
        # reconstruct the normalised actions from the discrete indices
        acts = jnp.stack([action[a] for a in self.agents])  # (n,4)
        cur_thr_norm = acts[:, 0] / 30.0
        cur_el_norm  = acts[:, 1] * 2.0 / 40.0 - 1.0
        cur_ail_norm = acts[:, 2] * 2.0 / 40.0 - 1.0
        cur_rud_norm = acts[:, 3] * 2.0 / 40.0 - 1.0

        # compute reward using PRE-step dynamics but POST-step action
        reward_arr = self._compute_reward(state, params,
                                          cur_thr_norm, cur_el_norm,
                                          cur_ail_norm, cur_rud_norm)
        info["hold_reward"] = reward_arr

        state = state.replace(
            prev_thr = cur_thr_norm,
            prev_el  = cur_el_norm,
            prev_ail = cur_ail_norm,
            prev_rud = cur_rud_norm,
        )
        return state, info

    # ------------------------------------------------------------------
    # Override get_reward to use our inline function
    # ------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_reward(self, state: HoldTaskState,
                   params: HoldTaskParams):
        # reward is computed in _step_task; pull it from info-stored field
        # For agent ordering, we build a trivial dict
        # (Called by the parent's step() after _step_task)
        # We use prev_thr etc. which are already updated
        reward_arr = self._compute_reward(
            state, params,
            state.prev_thr, state.prev_el,
            state.prev_ail, state.prev_rud,
        )
        rewards = {agent: reward_arr[i] for i, agent in enumerate(self.agents)}
        return state, rewards

    # ------------------------------------------------------------------
    # Override get_termination
    # ------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_termination(self, state: HoldTaskState,
                        params: HoldTaskParams):
        done_arr = self._compute_done(state, params)
        # mark crashed planes
        new_status = jnp.where(done_arr, 2, state.plane_state.status)  # 2 = CRASHED
        state = state.replace(
            plane_state=state.plane_state.replace(status=new_status),
            done=jnp.any(done_arr),          # scalar bool
            success=jnp.array(False),         # scalar bool, matches EnvState default
        )
        dones = {agent: done_arr[i] for i, agent in enumerate(self.agents)}
        return state, dones
