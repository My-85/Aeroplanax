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
    heading_pitch_V_reward_fn_vertical_energy,
    altitude_reward_fn,
    event_driven_reward_fn,
    reward_nz_soft_penalty,
)

from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_pitch_V_quat_vertical_energy_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance
from experiments.hierarchical_trajectory_tracking.loop_attitude_target import loop_plane_hpr_jax


@struct.dataclass
class Heading_Pitch_V_TaskState(EnvState):
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_vt: ArrayLike
    target_roll: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike
    level_selected: ArrayLike
    task_mode: ArrayLike
    vertical_stage: ArrayLike
    task_duration_steps: ArrayLike
    task_start_heading: ArrayLike
    task_start_pitch: ArrayLike
    task_start_roll: ArrayLike
    task_start_vt: ArrayLike
    task_start_altitude: ArrayLike
    task_start_energy: ArrayLike
    task_target_pitch_final: ArrayLike
    task_arc_angle: ArrayLike
    task_arc_start_angle: ArrayLike
    task_arc_radius: ArrayLike
    prev_energy: ArrayLike
    prev_throttle: ArrayLike
    prev_elevator: ArrayLike
    prev_aileron: ArrayLike
    prev_rudder: ArrayLike
    prev_speed_brake: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        zeros_f = jnp.zeros_like(extra_state[0])
        zeros_i = jnp.zeros_like(extra_state[0], dtype=jnp.int32)
        energy = 0.5 * extra_state[3] * extra_state[3] + 9.80665 * env_state.plane_state.altitude
        cruise_steps = jnp.full_like(extra_state[0], 55.0, dtype=jnp.float32)
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
            heading_turn_counts=zeros_i,
            level_selected=zeros_i,
            task_mode=zeros_i,
            vertical_stage=zeros_i,
            task_duration_steps=cruise_steps,
            task_start_heading=extra_state[0],
            task_start_pitch=extra_state[1],
            task_start_roll=extra_state[2],
            task_start_vt=extra_state[3],
            task_start_altitude=env_state.plane_state.altitude,
            task_start_energy=energy,
            task_target_pitch_final=extra_state[1],
            task_arc_angle=zeros_f,
            task_arc_start_angle=zeros_f,
            task_arc_radius=jnp.full_like(extra_state[0], 10000.0, dtype=jnp.float32),
            prev_energy=energy,
            prev_throttle=zeros_f,
            prev_elevator=zeros_f,
            prev_aileron=zeros_f,
            prev_rudder=zeros_f,
            prev_speed_brake=zeros_f,
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
    r_nz_coef: float = 0.035
    r_nz_clip: float = 4.0
    nz_hard_cap: float = 15.0

    # Vertical-energy fine-tuning curriculum.
    original_task_prob: float = 0.25
    horizontal_proxy_task_prob: float = 0.15
    level_altitude_task_prob: float = 0.10
    vertical_stage_successes: int = 8
    vertical_stage_offset: int = 0
    vertical_cruise_vt: float = 250.0
    pitch_ramp_duration_sec: float = 8.0
    climb_duration_sec: float = 18.0
    proxy_task_duration_sec: float = 48.0
    min_vertical_duration_sec: float = 8.0
    max_vertical_duration_sec: float = 35.0
    circle_proxy_radius_m: float = 5000.0
    circle_proxy_radius_tight_m: float = 3000.0
    circle_proxy_tight_prob: float = 0.0
    circle_proxy_left_prob: float = 0.50
    s_curve_proxy_amplitude_m: float = 3000.0
    s_curve_heading_amplitude_deg: float = 32.0
    s_curve_period_sec: float = 85.0
    figure_eight_proxy_radius_m: float = 5000.0
    figure_eight_heading_amplitude_deg: float = 42.0
    figure_eight_period_sec: float = 120.0
    circle_proxy_prob: float = 0.34
    s_curve_proxy_prob: float = 0.33
    vertical_arc_90_prob: float = 0.30
    vertical_arc_60_radius_prob: float = 0.50
    use_loop_plane_targets_for_vertical_arc: float = 0.0
    half_loop_curriculum_prob: float = 0.0
    half_loop_pullup_retention_prob: float = 0.16
    half_loop_climb_retention_prob: float = 0.10
    half_loop_vertical_retention_prob: float = 0.34
    half_loop_transition_prob: float = 0.28
    half_loop_exit_recovery_prob: float = 0.0
    half_loop_bridge_transition_prob: float = 0.0
    half_loop_partial_prob: float = 0.12
    half_loop_partial_exit_prob: float = 0.0
    half_loop_partial_bridge_prob: float = 0.0
    half_loop_max_phase_deg: float = 180.0

    # Energy-aware reward parameters.
    ve_theta_scale_deg: float = 7.5
    ve_speed_under_scale: float = 18.0
    ve_speed_over_scale: float = 35.0
    ve_low_speed_threshold: float = 180.0
    ve_strong_low_speed_threshold: float = 170.0
    ve_alpha_soft_deg: float = 15.0
    ve_alpha_hard_deg: float = 18.0
    ve_beta_soft_deg: float = 10.0
    ve_g_soft: float = 9.0
    ve_g_hard: float = 10.0
    ve_altitude_retention_weight: float = 0.14
    ve_altitude_retention_deadband_m: float = 80.0
    ve_altitude_retention_scale_m: float = 220.0
    ve_altitude_retention_vz_weight: float = 0.03
    ve_altitude_drift_weight: float = 0.04
    ve_altitude_drift_scale_m: float = 500.0
    ve_loop_geom_weight: float = 0.0
    ve_loop_roll_weight: float = 0.0
    ve_loop_nose_tangent_weight: float = 0.0
    ve_loop_wing_plane_weight: float = 0.0
    ve_loop_velocity_tangent_weight: float = 0.0
    ve_loop_nose_velocity_weight: float = 0.0
    ve_high_speed_alpha_weight: float = 0.0
    ve_action_saturation_weight: float = 0.0

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

def _rotate_body_to_ned(q_bn, v_b):
    q_bn = _quat_normalize(q_bn)
    p = jnp.array([0.0, v_b[0], v_b[1], v_b[2]])
    q_nb = _quat_conj(q_bn)
    qpq = _quat_mul(_quat_mul(q_nb, p), q_bn)
    return qpq[1:]

def _unit_vec(v):
    return v / (jnp.linalg.norm(v) + 1e-6)

def _axis_angle_error(a, b):
    a = _unit_vec(a)
    b = _unit_vec(b)
    c = jnp.clip(jnp.dot(a, b), -1.0, 1.0)
    return jnp.arccos(c)


# ── Adaptive check interval: maps difficulty level to RL steps ──
# Level 0: small deltas (~55 steps), Level 3: full-maneuver (~250 steps)
_LEVEL_CHECK_INTERVAL = jnp.array([55, 120, 210, 250], dtype=jnp.float32)

class AeroPlanaxHeading_Pitch_V_Env(AeroPlanaxEnv[Heading_Pitch_V_TaskState, Heading_Pitch_V_TaskParams]):
    def __init__(self, env_params: Optional[Heading_Pitch_V_TaskParams] = None):
        if env_params is None:
            env_params = Heading_Pitch_V_TaskParams()
        self._configured_params = env_params
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(heading_pitch_V_reward_fn_vertical_energy, reward_scale=2.0),
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=0),
            functools.partial(reward_nz_soft_penalty, scale=1.0),
        ]

        self.is_potential = [False] * len(self.reward_functions)

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_pitch_V_quat_vertical_energy_fn,
        ]

    def _get_obs_size(self) -> int:
        return 21  # 16 base + 5 past action

    @property
    def default_params(self) -> Heading_Pitch_V_TaskParams:
        return getattr(self, "_configured_params", Heading_Pitch_V_TaskParams())


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
        energy = 0.5 * state.plane_state.vt * state.plane_state.vt + 9.80665 * state.plane_state.altitude
        zeros_f = jnp.zeros_like(state.plane_state.yaw)
        zeros_i = jnp.zeros_like(state.plane_state.yaw, dtype=jnp.int32)
        state = state.replace(
            target_heading=state.plane_state.yaw,
            target_pitch=state.plane_state.pitch,
            target_roll=state.plane_state.roll,
            target_vt=state.plane_state.vt,
            last_check_time=state.time,
            heading_turn_counts=zeros_i,
            level_selected=zeros_i,
            task_mode=zeros_i,
            vertical_stage=zeros_i,
            task_duration_steps=jnp.full_like(state.plane_state.yaw, 55.0, dtype=jnp.float32),
            task_start_heading=state.plane_state.yaw,
            task_start_pitch=state.plane_state.pitch,
            task_start_roll=state.plane_state.roll,
            task_start_vt=state.plane_state.vt,
            task_start_altitude=state.plane_state.altitude,
            task_start_energy=energy,
            task_target_pitch_final=state.plane_state.pitch,
            task_arc_angle=zeros_f,
            task_arc_start_angle=zeros_f,
            task_arc_radius=jnp.full_like(state.plane_state.yaw, 10000.0, dtype=jnp.float32),
            prev_energy=energy,
            prev_throttle=zeros_f,
            prev_elevator=zeros_f,
            prev_aileron=zeros_f,
            prev_rudder=zeros_f,
            prev_speed_brake=zeros_f,
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
        """Vertical-energy curriculum with original-task replay."""
        B = self.num_agents

        def _seconds_to_steps(sec):
            return jnp.asarray(sec, dtype=jnp.float32) * params.sim_freq / params.agent_interaction_steps

        def _smooth01(x):
            x = jnp.clip(x, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)

        progress = jnp.clip(state.heading_turn_counts.astype(jnp.float32) / 300.0, 0.0, 1.0)
        p0 = 0.80 * (1.0 - progress) + 0.05 * progress
        p1 = 0.15 * (1.0 - progress) + 0.10 * progress
        p2 = 0.05 * (1.0 - progress) + 0.25 * progress
        cum0 = p0
        cum1 = p0 + p1
        cum2 = p0 + p1 + p2

        key_h, key_p, key_r, key_v, key_info, key_mode = jax.random.split(key, 6)
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

        def target_heading_fn(lv, k):
            dh0 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/6, maxval=jnp.pi/6)
            dh1 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/2, maxval=jnp.pi/2)
            dh2 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi, maxval=jnp.pi)
            dh3 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi, maxval=jnp.pi)
            dh = jnp.where(lv == 0, dh0, jnp.where(lv == 1, dh1, jnp.where(lv == 2, dh2, dh3)))
            return jnp.where(lv < 3, wrap_PI(state.plane_state.yaw + dh), wrap_PI(dh))

        def target_pitch_fn(lv, k):
            dp0 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/18, maxval=jnp.pi/18)
            dp1 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/6, maxval=jnp.pi/6)
            dp2 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/3, maxval=jnp.pi/3)
            dp3 = jax.random.uniform(k, shape=(B,), minval=-89 * jnp.pi / 180, maxval=89 * jnp.pi / 180)
            dp = jnp.where(lv == 0, dp0, jnp.where(lv == 1, dp1, jnp.where(lv == 2, dp2, dp3)))
            return jnp.clip(state.plane_state.pitch + dp, jnp.radians(-89), jnp.radians(89))

        def target_roll_fn(lv, k):
            dr0 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/6, maxval=jnp.pi/6)
            dr1 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi/2, maxval=jnp.pi/2)
            dr2 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi, maxval=jnp.pi)
            dr3 = jax.random.uniform(k, shape=(B,), minval=-jnp.pi, maxval=jnp.pi)
            dr = jnp.where(lv == 0, dr0, jnp.where(lv == 1, dr1, jnp.where(lv == 2, dr2, dr3)))
            return jnp.where(lv < 3, wrap_PI(state.plane_state.roll + dr), wrap_PI(dr))

        def target_vt_fn(lv, k):
            dv0 = jax.random.uniform(k, shape=(B,), minval=-20.0, maxval=20.0)
            dv1 = jax.random.uniform(k, shape=(B,), minval=-50.0, maxval=50.0)
            dv2 = jax.random.uniform(k, shape=(B,), minval=-100.0, maxval=100.0)
            dv3 = jax.random.uniform(k, shape=(B,), minval=-130.0, maxval=110.0)
            dv = jnp.where(lv == 0, dv0, jnp.where(lv == 1, dv1, jnp.where(lv == 2, dv2, dv3)))
            return jnp.where(lv < 3,
                             jnp.clip(state.plane_state.vt + dv, 120.0, 360.0),
                             jnp.clip(250.0 + dv3, 120.0, 360.0))

        key_h2, key_p2, key_r2, key_v2 = jax.random.split(key_info, 4)
        original_heading = target_heading_fn(lv_h, key_h2)
        original_pitch = target_pitch_fn(lv_p, key_p2)
        original_roll = target_roll_fn(lv_r, key_r2)
        original_vt = target_vt_fn(lv_v, key_v2)
        original_duration = _LEVEL_CHECK_INTERVAL[jnp.clip(max_level, 0, 3)]

        count_for_stage = state.heading_turn_counts + state.success.astype(jnp.int32)
        stage_from_success = count_for_stage // jnp.maximum(params.vertical_stage_successes, 1)
        vertical_stage = jnp.clip(stage_from_success + params.vertical_stage_offset, 0, 10)
        (
            key_mode2,
            key_proxy,
            key_pitch_mag,
            key_pitch_sign,
            key_climb,
            key_radius,
            key_arc,
            key_half_dice,
            key_half_retention,
            key_half_transition,
            key_half_partial,
            key_half_radius,
        ) = jax.random.split(key_mode, 12)
        mode_dice = jax.random.uniform(key_mode2, shape=(B,))
        original_prob = jnp.clip(params.original_task_prob, 0.0, 0.95)
        proxy_prob = jnp.clip(params.horizontal_proxy_task_prob, 0.0, 0.95 - original_prob)
        level_prob = jnp.clip(params.level_altitude_task_prob, 0.0, 0.95 - original_prob - proxy_prob)
        choose_original = mode_dice < original_prob
        choose_proxy = (mode_dice >= original_prob) & (mode_dice < original_prob + proxy_prob)
        choose_level_hold = (
            (mode_dice >= original_prob + proxy_prob)
            & (mode_dice < original_prob + proxy_prob + level_prob)
        )
        vertical_mode = jnp.where(vertical_stage <= 0, 1,
                        jnp.where(vertical_stage <= 2, 2,
                        jnp.where(vertical_stage <= 3, 3,
                        jnp.where(vertical_stage <= 7, 4, 5))))
        proxy_dice = jax.random.uniform(key_proxy, shape=(B,))
        circle_proxy_prob = jnp.clip(params.circle_proxy_prob, 0.0, 1.0)
        s_curve_proxy_prob = jnp.clip(params.s_curve_proxy_prob, 0.0, 1.0 - circle_proxy_prob)
        proxy_mode = jnp.where(
            proxy_dice < circle_proxy_prob,
            jnp.full((B,), 6, dtype=jnp.int32),
            jnp.where(
                proxy_dice < circle_proxy_prob + s_curve_proxy_prob,
                jnp.full((B,), 7, dtype=jnp.int32),
                jnp.full((B,), 8, dtype=jnp.int32),
            ),
        )
        task_mode_base = jnp.where(
            choose_original,
            jnp.zeros((B,), dtype=jnp.int32),
            jnp.where(
                choose_proxy,
                proxy_mode,
                jnp.where(choose_level_hold, jnp.ones((B,), dtype=jnp.int32), vertical_mode),
            ),
        ).astype(jnp.int32)
        choose_vertical_branch = ~(choose_original | choose_proxy | choose_level_hold)
        use_half_loop_curriculum = (
            choose_vertical_branch
            & (jax.random.uniform(key_half_dice, shape=(B,)) < jnp.clip(params.half_loop_curriculum_prob, 0.0, 1.0))
        )

        half_mix_dice = jax.random.uniform(key_half_dice, shape=(B,))
        half_pull_prob = jnp.clip(params.half_loop_pullup_retention_prob, 0.0, 1.0)
        half_climb_prob = jnp.clip(params.half_loop_climb_retention_prob, 0.0, 1.0 - half_pull_prob)
        half_ret_prob = jnp.clip(params.half_loop_vertical_retention_prob, 0.0, 1.0 - half_pull_prob - half_climb_prob)
        half_trans_prob = jnp.clip(params.half_loop_transition_prob, 0.0, 1.0 - half_pull_prob - half_climb_prob - half_ret_prob)
        half_bridge_prob = jnp.clip(
            params.half_loop_bridge_transition_prob,
            0.0,
            1.0 - half_pull_prob - half_climb_prob - half_ret_prob - half_trans_prob,
        )
        half_exit_prob = jnp.clip(
            params.half_loop_exit_recovery_prob,
            0.0,
            1.0 - half_pull_prob - half_climb_prob - half_ret_prob - half_trans_prob - half_bridge_prob,
        )
        half_pull_cut = half_pull_prob
        half_climb_cut = half_pull_prob + half_climb_prob
        half_ret_cut = half_climb_cut + half_ret_prob
        half_trans_cut = half_ret_cut + half_trans_prob
        half_bridge_cut = half_trans_cut + half_bridge_prob
        half_exit_cut = half_bridge_cut + half_exit_prob
        half_mode = jnp.where(
            half_mix_dice < half_pull_cut,
            jnp.full((B,), 4, dtype=jnp.int32),
            jnp.where(
                half_mix_dice < half_climb_cut,
                jnp.full((B,), 3, dtype=jnp.int32),
                jnp.where(
                    half_mix_dice < half_ret_cut,
                    jnp.full((B,), 5, dtype=jnp.int32),
                    jnp.where(
                        half_mix_dice < half_trans_cut,
                        jnp.full((B,), 9, dtype=jnp.int32),
                        jnp.full((B,), 9, dtype=jnp.int32),
                    ),
                ),
            ),
        )
        task_mode_new = jnp.where(use_half_loop_curriculum, half_mode, task_mode_base).astype(jnp.int32)

        mag_idx = jax.random.randint(key_pitch_mag, shape=(B,), minval=0, maxval=4)
        ramp_mags_stage1 = jnp.array([5.0, 5.0, 10.0, 10.0])
        ramp_mags_stage2 = jnp.array([10.0, 15.0, 15.0, 20.0])
        ramp_mags_late = jnp.array([15.0, 20.0, 20.0, 20.0])
        ramp_mag = jnp.where(vertical_stage <= 1,
                             ramp_mags_stage1[mag_idx],
                             jnp.where(vertical_stage <= 2,
                                       ramp_mags_stage2[mag_idx],
                                       ramp_mags_late[mag_idx]))
        ramp_sign = jnp.where(jax.random.bernoulli(key_pitch_sign, p=0.65, shape=(B,)), 1.0, -1.0)
        ramp_final = jnp.deg2rad(ramp_sign * ramp_mag)

        climb_idx = jax.random.randint(key_climb, shape=(B,), minval=0, maxval=5)
        climb_final = jnp.deg2rad(jnp.array([5.0, 10.0, 15.0, -5.0, -10.0]))[climb_idx]

        radius_idx = jax.random.randint(key_radius, shape=(B,), minval=0, maxval=4)
        radii_easy = jnp.array([10000.0, 10000.0, 8000.0, 8000.0])
        radii_medium = jnp.array([8000.0, 5000.0, 5000.0, 3000.0])
        radii_hard = jnp.array([5000.0, 5000.0, 3000.0, 2000.0])
        pullup_radius = jnp.where(vertical_stage <= 4,
                                  radii_easy[radius_idx],
                                  jnp.where(vertical_stage <= 6,
                                            radii_medium[radius_idx],
                                            radii_hard[radius_idx]))
        pullup_angle = jnp.deg2rad(jnp.where(vertical_stage <= 5, 15.0, 30.0))

        key_arc_kind, key_arc_radius, key_proxy_sign = jax.random.split(key_arc, 3)
        use_90_arc = jax.random.uniform(key_arc_kind, shape=(B,)) < jnp.clip(params.vertical_arc_90_prob, 0.0, 1.0)
        use_r8000_60 = jax.random.uniform(key_arc_radius, shape=(B,)) < jnp.clip(params.vertical_arc_60_radius_prob, 0.0, 1.0)
        vertical_arc_angle = jnp.deg2rad(jnp.where(use_90_arc, 90.0, 60.0))
        vertical_arc_radius = jnp.where(use_90_arc, 10000.0, jnp.where(use_r8000_60, 8000.0, 10000.0))
        half_ret_idx = jax.random.randint(key_half_retention, shape=(B,), minval=0, maxval=4)
        half_ret_angle = jnp.deg2rad(jnp.array([60.0, 90.0, 120.0, 150.0], dtype=jnp.float32))[half_ret_idx]
        half_transition_idx = jax.random.randint(key_half_transition, shape=(B,), minval=0, maxval=5)
        half_exit_idx = jax.random.randint(key_half_transition, shape=(B,), minval=0, maxval=4)
        half_transition_start_base = jnp.deg2rad(
            jnp.array([90.0, 120.0, 135.0, 150.0, 160.0], dtype=jnp.float32)
        )[half_transition_idx]
        half_transition_end_base = jnp.deg2rad(
            jnp.array([120.0, 150.0, 165.0, 175.0, 180.0], dtype=jnp.float32)
        )[half_transition_idx]
        half_bridge_idx = jax.random.randint(key_half_transition, shape=(B,), minval=0, maxval=3)
        half_bridge_start = jnp.deg2rad(
            jnp.array([150.0, 155.0, 160.0], dtype=jnp.float32)
        )[half_bridge_idx]
        half_bridge_end = jnp.deg2rad(
            jnp.array([165.0, 170.0, 175.0], dtype=jnp.float32)
        )[half_bridge_idx]
        half_exit_start = jnp.deg2rad(
            jnp.array([160.0, 170.0, 175.0, 180.0], dtype=jnp.float32)
        )[half_exit_idx]
        half_exit_end = jnp.deg2rad(
            jnp.array([180.0, 190.0, 200.0, 210.0], dtype=jnp.float32)
        )[half_exit_idx]
        half_is_bridge_transition = (
            use_half_loop_curriculum
            & (task_mode_new == 9)
            & (half_mix_dice >= half_trans_cut)
            & (half_mix_dice < half_bridge_cut)
        )
        half_is_exit_transition = (
            use_half_loop_curriculum
            & (task_mode_new == 9)
            & (half_mix_dice >= half_bridge_cut)
            & (half_mix_dice < half_exit_cut)
        )
        half_transition_start = jnp.where(
            half_is_exit_transition,
            half_exit_start,
            jnp.where(half_is_bridge_transition, half_bridge_start, half_transition_start_base),
        )
        half_transition_end = jnp.where(
            half_is_exit_transition,
            half_exit_end,
            jnp.where(half_is_bridge_transition, half_bridge_end, half_transition_end_base),
        )
        half_partial_idx = jax.random.randint(key_half_partial, shape=(B,), minval=0, maxval=5)
        half_partial_angle_base = jnp.deg2rad(
            jnp.array([160.0, 165.0, 170.0, 175.0, 180.0], dtype=jnp.float32)
        )[half_partial_idx]
        half_partial_bridge_idx = jax.random.randint(key_half_partial, shape=(B,), minval=0, maxval=3)
        half_partial_angle_bridge = jnp.deg2rad(
            jnp.array([165.0, 170.0, 175.0], dtype=jnp.float32)
        )[half_partial_bridge_idx]
        half_partial_exit_idx = jax.random.randint(key_half_partial, shape=(B,), minval=0, maxval=4)
        half_partial_angle_exit = jnp.deg2rad(
            jnp.array([180.0, 190.0, 200.0, 210.0], dtype=jnp.float32)
        )[half_partial_exit_idx]
        use_bridge_partial = (
            jax.random.uniform(key_half_partial, shape=(B,))
            < jnp.clip(params.half_loop_partial_bridge_prob, 0.0, 1.0)
        )
        use_exit_partial = (
            jax.random.uniform(key_half_partial, shape=(B,))
            < jnp.clip(params.half_loop_partial_exit_prob, 0.0, 1.0)
        )
        half_partial_angle = jnp.where(
            use_exit_partial,
            half_partial_angle_exit,
            jnp.where(use_bridge_partial, half_partial_angle_bridge, half_partial_angle_base),
        )
        half_radius_idx = jax.random.randint(key_half_radius, shape=(B,), minval=0, maxval=3)
        half_radius = jnp.array([15000.0, 12000.0, 10000.0], dtype=jnp.float32)[half_radius_idx]
        half_is_partial = use_half_loop_curriculum & (task_mode_new == 5) & (half_mix_dice >= half_trans_cut)
        half_loop_angle = jnp.where(half_is_partial, half_partial_angle, half_ret_angle)
        half_loop_radius = half_radius
        proxy_left = jax.random.bernoulli(
            key_proxy_sign,
            p=jnp.clip(params.circle_proxy_left_prob, 0.0, 1.0),
            shape=(B,),
        )
        proxy_direction = jnp.where(proxy_left, -1.0, 1.0)
        circle_tight = jax.random.uniform(key_arc_radius, shape=(B,)) < jnp.clip(params.circle_proxy_tight_prob, 0.0, 1.0)
        circle_radius = jnp.where(circle_tight, params.circle_proxy_radius_tight_m, params.circle_proxy_radius_m)
        proxy_radius = jnp.where(task_mode_new == 6, params.circle_proxy_radius_m,
                         jnp.where(task_mode_new == 7, params.s_curve_proxy_amplitude_m,
                                   params.figure_eight_proxy_radius_m))
        proxy_radius = jnp.where(task_mode_new == 6, circle_radius, proxy_radius)

        loop_arc_angle = jnp.where(use_half_loop_curriculum & (task_mode_new == 5), half_loop_angle, vertical_arc_angle)
        loop_arc_radius = jnp.where(
            use_half_loop_curriculum & ((task_mode_new == 5) | (task_mode_new == 9)),
            half_loop_radius,
            vertical_arc_radius,
        )
        loop_arc_start_angle = jnp.where(task_mode_new == 9, half_transition_start, jnp.zeros((B,)))
        loop_arc_delta = jnp.where(task_mode_new == 9, half_transition_end - half_transition_start, loop_arc_angle)

        target_pitch_final_new = jnp.where(task_mode_new == 0, original_pitch,
                                  jnp.where(task_mode_new == 1, jnp.zeros((B,)),
                                  jnp.where(task_mode_new == 2, ramp_final,
                                  jnp.where(task_mode_new == 3, climb_final,
                                  jnp.where(task_mode_new == 4, pullup_angle,
                                  jnp.where((task_mode_new == 6) | (task_mode_new == 7) | (task_mode_new == 8), jnp.zeros((B,)),
                                           loop_arc_start_angle + loop_arc_delta)))))
                                  )
        arc_angle_new = jnp.where(task_mode_new == 4, pullup_angle,
                          jnp.where((task_mode_new == 5) | (task_mode_new == 9), loop_arc_delta,
                          jnp.where((task_mode_new == 6) | (task_mode_new == 7) | (task_mode_new == 8), proxy_direction, jnp.zeros((B,)))))
        arc_radius_new = jnp.where(task_mode_new == 4, pullup_radius,
                           jnp.where((task_mode_new == 5) | (task_mode_new == 9), loop_arc_radius,
                           jnp.where((task_mode_new == 6) | (task_mode_new == 7) | (task_mode_new == 8), proxy_radius,
                                     jnp.full((B,), 10000.0)))
                           )
        arc_start_angle_new = jnp.where((task_mode_new == 5) | (task_mode_new == 9), loop_arc_start_angle, jnp.zeros((B,)))
        arc_duration_sec = jnp.abs(arc_angle_new) * arc_radius_new / jnp.maximum(params.vertical_cruise_vt, 1.0)
        vertical_duration = jnp.where(task_mode_new == 1, _seconds_to_steps(11.0),
                              jnp.where(task_mode_new == 2, _seconds_to_steps(params.pitch_ramp_duration_sec),
                              jnp.where(task_mode_new == 3, _seconds_to_steps(params.climb_duration_sec),
                              jnp.where((task_mode_new == 6) | (task_mode_new == 7) | (task_mode_new == 8),
                                       _seconds_to_steps(params.proxy_task_duration_sec),
                                       _seconds_to_steps(jnp.clip(arc_duration_sec,
                                                                  params.min_vertical_duration_sec,
                                                                  params.max_vertical_duration_sec))))))
        duration_new = jnp.where(task_mode_new == 0, original_duration, vertical_duration)
        energy_now = 0.5 * state.plane_state.vt * state.plane_state.vt + 9.80665 * state.plane_state.altitude

        vertical_heading = state.plane_state.yaw
        vertical_roll = jnp.zeros((B,))
        vertical_vt = jnp.full((B,), params.vertical_cruise_vt)
        target_heading_new = jnp.where(task_mode_new == 0, original_heading, vertical_heading)
        target_pitch_new = jnp.where(task_mode_new == 0, original_pitch,
                             jnp.where(task_mode_new == 1, jnp.zeros((B,)), state.plane_state.pitch))
        target_roll_new = jnp.where(task_mode_new == 0, original_roll, vertical_roll)
        target_vt_new = jnp.where(task_mode_new == 0, original_vt, vertical_vt)

        def _track_ok_row(q_row, yh, ph, rh, vt, vt_tgt):
            q_err = _quat_err_bn(q_row, yh, ph, rh)
            w = jnp.clip(jnp.nan_to_num(jnp.abs(q_err[0]), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0 - 1e-7)
            theta = 2.0 * jnp.arccos(w)
            att_ok = theta < jnp.deg2rad(7.0)
            spd_ok = vt > params.ve_low_speed_threshold
            return att_ok & spd_ok

        q_curr_pre = jnp.stack([
            jnp.nan_to_num(state.plane_state.q0, nan=0.0),
            jnp.nan_to_num(state.plane_state.q1, nan=0.0),
            jnp.nan_to_num(state.plane_state.q2, nan=0.0),
            jnp.nan_to_num(state.plane_state.q3, nan=0.0),
        ], axis=1)
        tracking_ok = jax.vmap(_track_ok_row, in_axes=(0, 0, 0, 0, 0, 0))(
            q_curr_pre, state.target_heading, state.target_pitch, state.target_roll,
            state.plane_state.vt, state.target_vt
        )
        earned_and_ready = tracking_ok & state.success

        new_state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,
            target_heading=target_heading_new,
            target_pitch=target_pitch_new,
            target_roll=target_roll_new,
            target_vt=target_vt_new,
            last_check_time=state.time,
            heading_turn_counts=jnp.where(earned_and_ready, state.heading_turn_counts + 1, state.heading_turn_counts),
            level_selected=max_level,
            task_mode=task_mode_new,
            vertical_stage=vertical_stage,
            task_duration_steps=duration_new,
            task_start_heading=state.plane_state.yaw,
            task_start_pitch=state.plane_state.pitch,
            task_start_roll=state.plane_state.roll,
            task_start_vt=state.plane_state.vt,
            task_start_altitude=state.plane_state.altitude,
            task_start_energy=energy_now,
            task_target_pitch_final=target_pitch_final_new,
            task_arc_angle=arc_angle_new,
            task_arc_start_angle=arc_start_angle_new,
            task_arc_radius=arc_radius_new,
            prev_energy=energy_now,
        )
        state = jax.lax.cond(state.success, lambda: new_state, lambda: state)

        elapsed = jnp.asarray(state.time - state.last_check_time, dtype=jnp.float32)
        elapsed_sec = elapsed * params.agent_interaction_steps / jnp.maximum(params.sim_freq, 1)
        frac = _smooth01(elapsed / jnp.maximum(state.task_duration_steps, 1.0))
        ramp_pitch = state.task_start_pitch + (state.task_target_pitch_final - state.task_start_pitch) * frac
        arc_pitch = state.task_start_pitch + state.task_arc_angle * frac
        proxy_direction_active = jnp.where(state.task_arc_angle >= 0.0, 1.0, -1.0)
        circle_heading = wrap_PI(
            state.task_start_heading
            + proxy_direction_active
            * params.vertical_cruise_vt
            / jnp.maximum(state.task_arc_radius, 1.0)
            * elapsed_sec
        )
        s_curve_heading = wrap_PI(
            state.task_start_heading
            + proxy_direction_active
            * jnp.deg2rad(params.s_curve_heading_amplitude_deg)
            * jnp.sin(2.0 * jnp.pi * elapsed_sec / jnp.maximum(params.s_curve_period_sec, 1.0))
        )
        figure_eight_heading = wrap_PI(
            state.task_start_heading
            + proxy_direction_active
            * jnp.deg2rad(params.figure_eight_heading_amplitude_deg)
            * jnp.sin(4.0 * jnp.pi * elapsed_sec / jnp.maximum(params.figure_eight_period_sec, 1.0))
        )
        loop_theta_max = jnp.deg2rad(jnp.maximum(params.half_loop_max_phase_deg, 180.0))
        loop_theta = jnp.clip(state.task_arc_start_angle + state.task_arc_angle * frac, 0.0, loop_theta_max)
        loop_heading, loop_pitch, loop_roll = loop_plane_hpr_jax(loop_theta, state.task_start_heading, 1.0)
        loop_target_active = (
            (state.task_mode == 9)
            | (
                (state.task_mode == 5)
                & (
                    (params.use_loop_plane_targets_for_vertical_arc > 0.5)
                    | ((state.task_arc_start_angle + state.task_arc_angle) > (0.5 * jnp.pi))
                )
            )
        )
        non_loop_pitch = jnp.where(state.task_mode == 1, state.task_target_pitch_final,
                           jnp.where(state.task_mode == 2, ramp_pitch,
                           jnp.where(state.task_mode == 3, state.task_target_pitch_final,
                           jnp.where((state.task_mode == 4) | (state.task_mode == 5),
                                     arc_pitch,
                                     state.target_pitch))))
        non_loop_pitch = jnp.clip(non_loop_pitch, jnp.deg2rad(-89.0), jnp.deg2rad(89.0))
        active_pitch = jnp.where(loop_target_active, loop_pitch, non_loop_pitch)
        active_heading = jnp.where(loop_target_active, loop_heading,
                           jnp.where(state.task_mode == 6, circle_heading,
                           jnp.where(state.task_mode == 7, s_curve_heading,
                           jnp.where(state.task_mode == 8, figure_eight_heading,
                           jnp.where(state.task_mode > 0, state.task_start_heading, state.target_heading)))
                           ))
        active_roll = jnp.where(
            loop_target_active,
            loop_roll,
            jnp.where(state.task_mode > 0, jnp.zeros_like(state.target_roll), state.target_roll),
        )
        active_vt = jnp.where(state.task_mode > 0, jnp.full((B,), params.vertical_cruise_vt), state.target_vt)
        state = state.replace(
            target_heading=active_heading,
            target_pitch=active_pitch,
            target_roll=active_roll,
            target_vt=active_vt,
            prev_energy=energy_now,
            prev_throttle=jnp.nan_to_num(state.control_state.throttle, nan=0.0),
            prev_elevator=jnp.nan_to_num(state.control_state.elevator, nan=0.0),
            prev_aileron=jnp.nan_to_num(state.control_state.aileron, nan=0.0),
            prev_rudder=jnp.nan_to_num(state.control_state.rudder, nan=0.0),
            prev_speed_brake=jnp.nan_to_num(state.control_state.speed_brake, nan=0.0),
        )

        info["heading_turn_counts"] = state.heading_turn_counts
        info["level_selected"] = state.level_selected
        info["task_mode"] = state.task_mode
        info["vertical_stage"] = state.vertical_stage
        info["task_duration_steps"] = state.task_duration_steps
        info["target_pitch_deg"] = jnp.rad2deg(state.target_pitch)
        info["target_roll_deg"] = jnp.rad2deg(state.target_roll)
        info["target_heading_deg"] = jnp.rad2deg(state.target_heading)
        info["target_vt"] = state.target_vt

        ego_z_km = jnp.nan_to_num(state.plane_state.altitude / 1000.0, nan=0.0, posinf=1e6, neginf=-1e6)
        ego_vz_mh = jnp.nan_to_num(state.plane_state.vel_z / 340.0, nan=0.0, posinf=1e6, neginf=-1e6)
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
            ori_r = jnp.exp(-((theta / theta_scale) ** 2))
            speed_r = jnp.exp(-((jnp.clip(jnp.nan_to_num(vt - vt_tgt, nan=0.0), -1e3, 1e3) / 24.0) ** 2))
            return (ori_r ** 0.8) * (speed_r ** 0.2)

        hpv_reward_raw = jax.vmap(_theta_row, in_axes=(0, 0, 0, 0, 0, 0))(
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

        energy = 0.5 * state.plane_state.vt * state.plane_state.vt + 9.80665 * state.plane_state.altitude
        info["energy_proxy"] = jnp.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)
        info["energy_loss"] = jnp.nan_to_num(state.task_start_energy - energy, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)
        altitude_delta = state.plane_state.altitude - state.task_start_altitude
        altitude_hold_err = jnp.clip(
            jnp.abs(altitude_delta) - params.ve_altitude_retention_deadband_m,
            0.0,
            1e6,
        )
        level_hold_gate = (state.task_mode == 1) | (state.task_mode == 6) | (state.task_mode == 7) | (state.task_mode == 8)
        info["altitude_gain"] = jnp.nan_to_num(altitude_delta, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)
        info["altitude_hold_error"] = jnp.nan_to_num(altitude_hold_err, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)
        info["altitude_hold_task"] = level_hold_gate.astype(jnp.float32)
        info["alpha_deg"] = jnp.rad2deg(jnp.nan_to_num(state.plane_state.alpha, nan=0.0)).astype(jnp.float32)
        info["beta_deg"] = jnp.rad2deg(jnp.nan_to_num(state.plane_state.beta, nan=0.0)).astype(jnp.float32)
        info["vt"] = jnp.nan_to_num(state.plane_state.vt, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)
        info["altitude"] = jnp.nan_to_num(state.plane_state.altitude, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)
        info["pitch_deg"] = jnp.rad2deg(jnp.nan_to_num(state.plane_state.pitch, nan=0.0)).astype(jnp.float32)
        info["roll_deg"] = jnp.rad2deg(jnp.nan_to_num(state.plane_state.roll, nan=0.0)).astype(jnp.float32)
        info["yaw_deg"] = jnp.rad2deg(jnp.nan_to_num(state.plane_state.yaw, nan=0.0)).astype(jnp.float32)

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
