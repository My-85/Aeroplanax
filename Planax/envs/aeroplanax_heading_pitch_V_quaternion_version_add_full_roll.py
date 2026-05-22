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
    heading_reward_fn,
    # heading_pitch_V_reward_fn,
    heading_pitch_V_reward_fn_add_roll_target,
    altitude_reward_fn,
    event_driven_reward_fn,
    heading_pitch_v_event_driven_reward_fn,
    reward_nz_soft_penalty,
)

from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_pitch_V_quat_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class Heading_Pitch_V_TaskState(EnvState):
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_vt: ArrayLike
    target_roll: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike
    level_selected: ArrayLike = 0  # current difficulty level (0-3) for adaptive check interval

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
            heading_turn_counts=jnp.zeros_like(extra_state[0], dtype=jnp.int32),
            level_selected=jnp.zeros_like(extra_state[0], dtype=jnp.int32),
        )


@struct.dataclass(frozen=True)
class Heading_Pitch_V_TaskParams(EnvParams):
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
    max_velocities_u_increment: float = 50.0

    max_heading_increment: float = jnp.pi/2
    max_pitch_increment: float = jnp.pi/6
    max_roll_increment: float = jnp.pi / 2
    max_altitude_increment: float = 2100.0

    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000
    safe_distance: float = 3000

    # G-load soft penalty params
    nz_limit: float = 9.0
    r_nz_coef: float = 0.05  # increased from 0.02 to deter G-spiking
    r_nz_clip: float = 5.0   # increased to match stronger coefficient
    nz_hard_cap: float = 15.0

# ---------------- Quaternion helpers (convention: q = Body->NED) ----------------
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
    """Build quaternion from Tait-Bryan (Z-Y-X) Euler angles.
    Returns q representing Body->NED."""
    cr, sr = jnp.cos(0.5*roll),  jnp.sin(0.5*roll)
    cp, sp = jnp.cos(0.5*pitch), jnp.sin(0.5*pitch)
    cy, sy = jnp.cos(0.5*yaw),   jnp.sin(0.5*yaw)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return jnp.array([qw, qx, qy, qz])

def _target_q_bn_from_heading_pitch(yaw_t, pitch_t, roll_t=0.0):
    q_nb = _quat_from_euler_nb(roll_t, pitch_t, yaw_t)
    return _quat_conj(q_nb)

def _quat_err_bn(q_curr_bn, yaw_t, pitch_t, roll_t):
    q_curr_bn = _quat_normalize(jnp.nan_to_num(q_curr_bn, nan=0.0, posinf=0.0, neginf=0.0))
    q_tgt_bn  = _quat_normalize(_target_q_bn_from_heading_pitch(yaw_t, pitch_t, roll_t))
    q_err = _quat_mul(q_tgt_bn, _quat_conj(q_curr_bn))
    q_err = jnp.where(q_err[0] < 0.0, -q_err, q_err)
    return q_err

def _rotate_ned_to_body(q_bn, v_n):
    q_bn = _quat_normalize(q_bn)
    p = jnp.array([0.0, v_n[0], v_n[1], v_n[2]])
    qpq = _quat_mul(_quat_mul(q_bn, p), _quat_conj(q_bn))
    return qpq[1:]


# ── Adaptive check interval: maps difficulty level to RL steps ──
# Level 0: small deltas (~55 steps), Level 3: full-maneuver (~250 steps)
_LEVEL_CHECK_INTERVAL = jnp.array([55, 120, 210, 250], dtype=jnp.float32)

class AeroPlanaxHeading_Pitch_V_Env(AeroPlanaxEnv[Heading_Pitch_V_TaskState, Heading_Pitch_V_TaskParams]):
    def __init__(self, env_params: Optional[Heading_Pitch_V_TaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(heading_pitch_V_reward_fn_add_roll_target, reward_scale=2.0),
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=0),
            functools.partial(reward_nz_soft_penalty, scale=1.0),
        ]

        self.is_potential = [False] * len(self.reward_functions)

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_pitch_V_quat_fn,
        ]

    def _get_obs_size(self) -> int:
        return 21  # 16 base + 5 past action

    @property
    def default_params(self) -> Heading_Pitch_V_TaskParams:
        return Heading_Pitch_V_TaskParams()


    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: chex.PRNGKey,
        params: Heading_Pitch_V_TaskParams,
    ) -> Heading_Pitch_V_TaskState:
        state = super()._init_state(key, params)

        # Trim initialization: level flight at cruise speed
        # Only yaw is randomized; roll/pitch/vt/alt are fixed for stability
        key, key_heading = jax.random.split(key)
        initial_heading = jax.random.uniform(
            key_heading,
            shape=(self.num_agents,),
            minval=0.0,
            maxval=2.0 * jnp.pi,
        )

        trim_roll = jnp.zeros((self.num_agents,))
        trim_pitch = jnp.zeros((self.num_agents,))
        trim_vt = jnp.full((self.num_agents,), 250.0)
        trim_alt = jnp.full((self.num_agents,), 5000.0)

        q_init_nb = jax.vmap(_quat_from_euler_nb)(trim_roll, trim_pitch, initial_heading)
        q_init_bn = jax.vmap(_quat_conj)(q_init_nb)

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=initial_heading,
                roll=trim_roll,
                pitch=trim_pitch,
                vt=trim_vt,
                vel_y=trim_vt,
                altitude=trim_alt,
                alpha=jnp.radians(2.0) * jnp.ones((self.num_agents,)),
                q0=q_init_bn[:, 0],
                q1=q_init_bn[:, 1],
                q2=q_init_bn[:, 2],
                q3=q_init_bn[:, 3],
            )
        )

        extra = jnp.stack([initial_heading, trim_pitch, trim_roll, trim_vt], axis=0)
        state = Heading_Pitch_V_TaskState.create(state, extra_state=extra)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: Heading_Pitch_V_TaskState,
        params: Heading_Pitch_V_TaskParams,
    ) -> Heading_Pitch_V_TaskState:
        # Targets = current state (zero-error start). _step_task will set real targets.
        state = state.replace(
            target_heading=state.plane_state.yaw,
            target_pitch=state.plane_state.pitch,
            target_roll=state.plane_state.roll,
            target_vt=state.plane_state.vt,
            heading_turn_counts=jnp.zeros_like(state.plane_state.yaw, dtype=jnp.int32),
            level_selected=jnp.zeros_like(state.plane_state.yaw, dtype=jnp.int32),
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: Heading_Pitch_V_TaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: Heading_Pitch_V_TaskParams,
    ) -> Tuple[Heading_Pitch_V_TaskState, Dict[str, Any]]:
        """Task step with mixed-sampling curriculum for diverse maneuver training."""

        B = self.num_agents

        # ── Curriculum progress: 0 (early) -> 1 (late) ──
        progress = jnp.clip(state.heading_turn_counts.astype(jnp.float32) / 300.0, 0.0, 1.0)

        # ── Level probability schedules ──
        # L0: 80% -> 5%,  L1: 15% -> 10%,  L2: 5% -> 25%,  L3: 0% -> 60%
        p0 = 0.80 * (1.0 - progress) + 0.05 * progress
        p1 = 0.15 * (1.0 - progress) + 0.10 * progress
        p2 = 0.05 * (1.0 - progress) + 0.25 * progress
        cum0 = p0
        cum1 = p0 + p1
        cum2 = p0 + p1 + p2

        # 4 independent dice per axis
        key_h, key_p, key_r, key_v, key_info = jax.random.split(key, 5)
        dice_h = jax.random.uniform(key_h, shape=(B,))
        dice_p = jax.random.uniform(key_p, shape=(B,))
        dice_r = jax.random.uniform(key_r, shape=(B,))
        dice_v = jax.random.uniform(key_v, shape=(B,))

        def sample_level(dice):
            return jnp.where(dice < cum0, 0,
                   jnp.where(dice < cum1, 1,
                   jnp.where(dice < cum2, 2, 3)))

        lv_h = sample_level(dice_h)
        lv_p = sample_level(dice_p)
        lv_r = sample_level(dice_r)
        lv_v = sample_level(dice_v)
        max_level = jnp.max(jnp.stack([lv_h, lv_p, lv_r, lv_v], axis=0), axis=0)

        # ── Target generation ──
        # Level 0: hdg +-30deg, pitch +-10deg, roll +-30deg, speed +-20 m/s
        # Level 1: hdg +-90deg, pitch +-30deg, roll +-90deg, speed +-50 m/s
        # Level 2: hdg +-180deg, pitch +-60deg, roll +-180deg, speed +-100 m/s
        # Level 3: hdg absolute, pitch +-89deg, roll absolute, speed absolute [120,360]

        def target_heading_fn(lv, k):
            dh0 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/6, maxval=jnp.pi/6)
            dh1 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/2, maxval=jnp.pi/2)
            dh2 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi,  maxval=jnp.pi)
            dh3 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi,  maxval=jnp.pi)
            dh = jnp.where(lv == 0, dh0, jnp.where(lv == 1, dh1, jnp.where(lv == 2, dh2, dh3)))
            return jnp.where(lv < 3, wrap_PI(state.plane_state.yaw + dh), wrap_PI(dh))

        def target_pitch_fn(lv, k):
            dp0 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/18, maxval=jnp.pi/18)
            dp1 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/6,  maxval=jnp.pi/6)
            dp2 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/3,  maxval=jnp.pi/3)
            dp3 = jax.random.uniform(k, shape=(B,), minval=-89*jnp.pi/180, maxval=89*jnp.pi/180)
            dp = jnp.where(lv == 0, dp0, jnp.where(lv == 1, dp1, jnp.where(lv == 2, dp2, dp3)))
            raw = state.plane_state.pitch + dp
            return jnp.clip(raw, jnp.radians(-89), jnp.radians(89))

        def target_roll_fn(lv, k):
            dr0 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/6, maxval=jnp.pi/6)
            dr1 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/2, maxval=jnp.pi/2)
            dr2 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi,   maxval=jnp.pi)
            dr3 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi,   maxval=jnp.pi)
            dr = jnp.where(lv == 0, dr0, jnp.where(lv == 1, dr1, jnp.where(lv == 2, dr2, dr3)))
            return jnp.where(lv < 3, wrap_PI(state.plane_state.roll + dr), wrap_PI(dr))

        def target_vt_fn(lv, k):
            dv0 = jax.random.uniform(k, shape=(B,), minval=-20.0,  maxval=20.0)
            dv1 = jax.random.uniform(k, shape=(B,), minval=-50.0,  maxval=50.0)
            dv2 = jax.random.uniform(k, shape=(B,), minval=-100.0, maxval=100.0)
            dv3 = jax.random.uniform(k, shape=(B,), minval=-130.0, maxval=110.0)
            dv = jnp.where(lv == 0, dv0, jnp.where(lv == 1, dv1, jnp.where(lv == 2, dv2, dv3)))
            return jnp.where(lv < 3,
                   jnp.clip(state.plane_state.vt + dv, 120.0, 360.0),
                   jnp.clip(250.0 + dv3, 120.0, 360.0))

        key_h2, key_p2, key_r2, key_v2 = jax.random.split(key_info, 4)

        target_heading = target_heading_fn(lv_h, key_h2)
        target_pitch   = target_pitch_fn(lv_p, key_p2)
        target_roll    = target_roll_fn(lv_r, key_r2)
        target_vt      = target_vt_fn(lv_v, key_v2)

        # Check if this success was EARNED (tracking good) vs safety timeout
        # Only advance curriculum on earned successes
        def _track_ok_row(q_row, yh, ph, rh, vt, vt_tgt):
            q_err = _quat_err_bn(q_row, yh, ph, rh)
            w = jnp.clip(jnp.nan_to_num(jnp.abs(q_err[0]), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0 - 1e-7)
            theta = 2.0 * jnp.arccos(w)
            att_ok = theta < jnp.deg2rad(5.0)
            spd_ok = jnp.abs(vt - vt_tgt) < 15.0
            return att_ok & spd_ok

        q_curr_pre = jnp.stack([
            jnp.nan_to_num(state.plane_state.q0, nan=0.0),
            jnp.nan_to_num(state.plane_state.q1, nan=0.0),
            jnp.nan_to_num(state.plane_state.q2, nan=0.0),
            jnp.nan_to_num(state.plane_state.q3, nan=0.0),
        ], axis=1)
        tracking_ok = jax.vmap(_track_ok_row, in_axes=(0,0,0,0,0,0))(
            q_curr_pre, state.target_heading, state.target_pitch, state.target_roll,
            state.plane_state.vt, state.target_vt
        )
        earned_and_ready = tracking_ok & state.success

        new_state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,
            target_heading=target_heading,
            target_pitch=target_pitch,
            target_roll=target_roll,
            target_vt=target_vt,
            last_check_time=state.time,
            heading_turn_counts=jnp.where(earned_and_ready,
                                          state.heading_turn_counts + 1,
                                          state.heading_turn_counts),
            level_selected=max_level,
        )
        state = jax.lax.cond(state.success, lambda: new_state, lambda: state)
        info["heading_turn_counts"] = state.heading_turn_counts
        info["level_selected"] = state.level_selected

        # ── Per-reward-component logging ──
        ego_z_km = jnp.nan_to_num(state.plane_state.altitude / 1000.0, nan=0.0, posinf=1e6, neginf=-1e6)
        ego_vz_mh = jnp.nan_to_num(state.plane_state.vel_z / 340.0,    nan=0.0, posinf=1e6, neginf=-1e6)
        safe_alt = self.default_params.safe_altitude
        danger_alt = self.default_params.danger_altitude
        Pv = -jnp.clip(ego_vz_mh / 0.2 * (safe_alt - ego_z_km) / safe_alt, 0., 1.)
        Pv = jnp.where(ego_z_km <= safe_alt, Pv, jnp.zeros_like(Pv))
        PH = jnp.clip(ego_z_km / danger_alt, 0., 1.) - 2.0
        PH = jnp.where(ego_z_km <= danger_alt, PH, jnp.zeros_like(PH))
        altitude_reward_raw = Pv + PH
        altitude_reward_will_clip = jnp.abs(altitude_reward_raw) > 10.0

        q_curr = jnp.stack([
            jnp.nan_to_num(state.plane_state.q0, nan=0.0),
            jnp.nan_to_num(state.plane_state.q1, nan=0.0),
            jnp.nan_to_num(state.plane_state.q2, nan=0.0),
            jnp.nan_to_num(state.plane_state.q3, nan=0.0),
        ], axis=1)

        def _theta_row(q_row, yh, ph, rh, vt, vt_tgt):
            q_err = _quat_err_bn(q_row, yh, ph, rh)
            w = jnp.clip(jnp.nan_to_num(jnp.abs(q_err[0]), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0 - 1e-7)
            theta = 2.0 * jnp.arccos(w)
            theta_scale = jnp.pi / 36.0
            ori_r   = jnp.exp(- (theta/theta_scale)**2)
            speed_r = jnp.exp(- ((jnp.clip(jnp.nan_to_num(vt - vt_tgt, nan=0.0), -1e3, 1e3)/24.0)**2))
            return (ori_r**0.8) * (speed_r**0.2)

        hpv_reward_raw = jax.vmap(_theta_row, in_axes=(0,0,0,0,0,0))(
            q_curr, state.target_heading, state.target_pitch, state.target_roll,
            state.plane_state.vt, state.target_vt
        )
        heading_pitch_V_reward_will_clip = hpv_reward_raw > 1.0

        info["clipped_altitude_reward_count"] = altitude_reward_will_clip.astype(jnp.float32)
        info["clipped_heading_pitch_V_reward_count"] = heading_pitch_V_reward_will_clip.astype(jnp.float32)
        info["clipped_any_reward_count"] = (altitude_reward_will_clip | heading_pitch_V_reward_will_clip).astype(jnp.float32)

        info["r_att_speed"] = jnp.nan_to_num(hpv_reward_raw, nan=0.0, posinf=0.0, neginf=0.0)
        info["r_altitude"] = jnp.nan_to_num(altitude_reward_raw, nan=0.0, posinf=0.0, neginf=0.0)
        die = state.plane_state.is_crashed | state.plane_state.is_shotdown
        info["r_crash"] = jnp.nan_to_num((-200.0 * die).astype(jnp.float32), nan=0.0, posinf=0.0, neginf=0.0)

        nx_g = jnp.nan_to_num(jnp.abs(state.plane_state.ax), nan=0.0, posinf=0.0, neginf=0.0)
        ny_g = jnp.nan_to_num(jnp.abs(state.plane_state.ay), nan=0.0, posinf=0.0, neginf=0.0)
        nz_g = jnp.nan_to_num(jnp.abs(state.plane_state.az), nan=0.0, posinf=0.0, neginf=0.0)
        load_max = jnp.max(jnp.array([nx_g, ny_g, nz_g]), axis=0)
        over_g = jnp.clip(load_max - self.default_params.nz_limit, 0.0)
        raw_pen = -self.default_params.r_nz_coef * (over_g * over_g)
        info["r_nz_penalty"] = jnp.clip(raw_pen, -self.default_params.r_nz_clip, 0.0).astype(jnp.float32)
        info["g_load_max"] = jnp.nan_to_num(load_max, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)

        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: Heading_Pitch_V_TaskState,
        params: Heading_Pitch_V_TaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        21-dim observation:
        [0:3]   qv (quaternion error vector part)
        [3]     (vt - target_vt) / 340
        [4]     altitude / 5000
        [5]     vt / 340
        [6:9]   v_b (target direction in body frame)
        [9:12]  P, Q, R
        [12:14] sin/cos alpha
        [14:16] sin/cos beta
        [16:21] prev_action (throttle, elevator, aileron, rudder, speed_brake)  — normalized
        """
        B = self.num_agents

        q_curr = jnp.stack([
            jnp.nan_to_num(state.plane_state.q0, nan=0.0),
            jnp.nan_to_num(state.plane_state.q1, nan=0.0),
            jnp.nan_to_num(state.plane_state.q2, nan=0.0),
            jnp.nan_to_num(state.plane_state.q3, nan=0.0),
        ], axis=1)

        yaw_t   = state.target_heading
        pitch_t = state.target_pitch
        roll_t  = state.target_roll
        vt_tgt  = state.target_vt

        def _err_row(q_row, yh, ph, rh):
            return _quat_err_bn(q_row, yh, ph, rh)
        q_err_batch = jax.vmap(_err_row, in_axes=(0,0,0,0))(q_curr, yaw_t, pitch_t, roll_t)
        qv = jnp.clip(q_err_batch[:, 1:4], -1.0, 1.0)

        c_th, s_th = jnp.cos(yaw_t),   jnp.sin(yaw_t)
        c_ph, s_ph = jnp.cos(pitch_t), jnp.sin(pitch_t)
        v_n = jnp.stack([c_ph * c_th, c_ph * s_th, -s_ph], axis=1)
        v_b = jax.vmap(_rotate_ned_to_body, in_axes=(0,0))(q_curr, v_n)
        v_b = jnp.clip(v_b, -1.0, 1.0)

        altitude = state.plane_state.altitude
        vt       = state.plane_state.vt
        alpha    = state.plane_state.alpha
        beta     = state.plane_state.beta
        P, Q, R  = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_dvt = (vt - vt_tgt) / 340.0
        norm_alt = altitude / 5000.0
        norm_vt  = vt / 340.0

        alpha_sin, alpha_cos = jnp.sin(alpha), jnp.cos(alpha)
        beta_sin,  beta_cos  = jnp.sin(beta),  jnp.cos(beta)

        # Past action from control_state (normalized, zero if first step)
        cs = state.control_state
        prev_thr = jnp.nan_to_num(cs.throttle, nan=0.0)
        prev_el  = jnp.nan_to_num(cs.elevator, nan=0.0)
        prev_ail = jnp.nan_to_num(cs.aileron, nan=0.0)
        prev_rud = jnp.nan_to_num(cs.rudder, nan=0.0)
        prev_sb  = jnp.nan_to_num(jnp.where(cs.speed_brake > 0, cs.speed_brake, jnp.zeros_like(cs.speed_brake)),
                                  nan=0.0)

        obs_mat = jnp.stack([
            qv[:,0], qv[:,1], qv[:,2],       # 0-2
            norm_dvt,                        # 3
            norm_alt,                        # 4
            norm_vt,                         # 5
            v_b[:,0], v_b[:,1], v_b[:,2],    # 6-8
            P, Q, R,                          # 9-11
            alpha_sin, alpha_cos,            # 12-13
            beta_sin,  beta_cos,              # 14-15
            prev_thr, prev_el, prev_ail,      # 16-18
            prev_rud, prev_sb,                # 19-20
        ], axis=0)

        low_base  = jnp.array([-1., -1., -1., -2., 0., 0., -1., -1., -1., -10., -10., -10., -1., -1., -1., -1.])
        high_base = jnp.array([ 1.,  1.,  1.,  2., 5., 2.,  1.,  1.,  1.,  10.,  10.,  10.,  1.,  1.,  1.,  1.])
        low_act   = jnp.array([0.,  -1., -1., -1., 0.])
        high_act  = jnp.array([1.,   1.,  1.,  1., 1.])
        low  = jnp.concatenate([low_base, low_act]).reshape(-1, 1)
        high = jnp.concatenate([high_base, high_act]).reshape(-1, 1)
        obs_mat = jnp.clip(jnp.nan_to_num(obs_mat, nan=0.0, posinf=0.0, neginf=0.0), low, high)

        return {agent: obs_mat[:, i] for i, agent in enumerate(self.agents)}


    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: Heading_Pitch_V_TaskState,
            params: Heading_Pitch_V_TaskParams,
        ) -> Heading_Pitch_V_TaskState:

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
        initial_heading = jnp.full((self.num_agents,), jnp.pi/2)
        # Update quaternion to stay consistent with yaw/pitch/roll
        q_nb = jax.vmap(_quat_from_euler_nb)(
            state.plane_state.roll, state.plane_state.pitch, initial_heading)
        q_bn = jax.vmap(_quat_conj)(q_nb)
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2],
            yaw=initial_heading,
            q0=q_bn[:, 0], q1=q_bn[:, 1], q2=q_bn[:, 2], q3=q_bn[:, 3],
        ))
        return state
