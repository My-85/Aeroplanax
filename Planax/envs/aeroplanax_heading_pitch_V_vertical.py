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
from .core.simulators.fighterplane.dynamics import atmos
from .reward_functions import (
    heading_reward_fn,
    heading_pitch_V_reward_fn,
    altitude_reward_fn,
    event_driven_reward_fn,
    heading_pitch_v_event_driven_reward_fn,
    reward_nz_soft_penalty,
    reward_low_qbar_penalty,
    reward_energy_track,
)

from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_pitch_V_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class Heading_Pitch_V_TaskState(EnvState):
    # —— “实际目标”（用于观测/奖励/终止判据）——
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_vt: ArrayLike

    # —— “指令目标”（只在切换时改变，随后用一阶滤波逐步追上）——
    cmd_target_heading: ArrayLike
    cmd_target_pitch: ArrayLike
    cmd_target_vt: ArrayLike

    last_check_time: ArrayLike
    last_switch_time: ArrayLike # 为防死锁，需要一个很轻的“强制切换”保底：阶段驻留太久也切一次
    heading_turn_counts: ArrayLike

    # 竖直统计（总）
    vertical_success_counts: ArrayLike   # int32
    is_vertical_target: ArrayLike        # bool

    # === NEW: 上/下方向细分统计 ===
    vertical_up_success_counts: ArrayLike
    vertical_down_success_counts: ArrayLike
    vertical_cmd_up_counts: ArrayLike
    vertical_cmd_down_counts: ArrayLike
    is_vertical_up_target: ArrayLike     # bool

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        # extra_state = stack([init_heading, init_pitch, init_vt])
        zeros_like = jnp.zeros_like(extra_state[0])
        time_vec   = jnp.full_like(extra_state[0], env_state.time)  # 关键：把 time 扩成向量
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

            cmd_target_heading=extra_state[0],
            cmd_target_pitch=extra_state[1],
            cmd_target_vt=extra_state[2],

            last_check_time=time_vec,        # <- 由标量改为向量
            last_switch_time=time_vec,       # <- 由标量改为向量
            heading_turn_counts=zeros_like.astype(jnp.int32),  # <- 由标量0改为向量0

            vertical_success_counts=zeros_like.astype(jnp.int32),
            is_vertical_target=zeros_like.astype(jnp.bool_),

            # NEW
            vertical_up_success_counts=zeros_like.astype(jnp.int32),
            vertical_down_success_counts=zeros_like.astype(jnp.int32),
            vertical_cmd_up_counts=zeros_like.astype(jnp.int32),
            vertical_cmd_down_counts=zeros_like.astype(jnp.int32),
            is_vertical_up_target=zeros_like.astype(jnp.bool_),
        )


@struct.dataclass(frozen=True)
class Heading_Pitch_V_TaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    formation_type: int = 0  # 0: wedge, 1: line, 2: diamond
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 20000.0
    min_altitude: float = 2000.0
    max_vt: float = 360.0
    min_vt: float = 120.0
    max_heading_increment: float = jnp.pi/2   # 90°
    max_pitch_increment: float = jnp.pi/6     # 30°
    max_altitude_increment: float = 2100.0
    max_velocities_u_increment: float = 50.0
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000
    safe_distance: float = 3000  # 编队最小安全间距

    # —— 竖直段课程 —— 
    loop_mode_prob: float = 0.5
    loop_pitch_max_deg: float = 90.0
    loop_phase_steps: int = 200
    loop_speed_low: float = 210.0
    loop_cmd_pitch_cap_deg: float = 85.0 # 竖直段目标俯仰角上限

    # —— ramp 速度（非竖直/退出竖直）——
    ramp_steps_normal: int = 40

    # —— 奖励塑形参数 —— 
    r_nz_coef: float = 0.005
    r_qbar_coef: float = 0.02
    r_energy_coef: float = 0.05
    nz_limit: float = 9.0
    qbar_low_frac: float = 0.35
    energy_ref_frac: float = 0.90
    nz_hard_cap: float = 15.0
    r_nz_clip: float = 3.0

    # === NEW: 向下竖直控制参数 ===
    loop_down_prob: float = 0.5              # 进入竖直后选择“向下”的概率
    down_alt_buffer: float = 2500.0          # 低于 min_altitude + buffer 时禁用向下竖直
    loop_speed_down: float = 300.0           # 向下竖直时趋向的目标高速

    # # === 竖直能量/门控新增参数 ===
    # vert_up_min_vt: float = 260.0
    # vert_up_min_qbar_norm: float = 0.35
    # vert_up_extra_reward: float = 0.35
    # vert_up_fail_penalty: float = -0.5
    # vert_down_max_pitch_deg: float = -40.0
    # vert_down_min_alt_buffer: float = 1800.0
    # vert_down_max_qbar_norm: float = 1.25
    # vert_down_extra_reward: float = 0.25
    # vert_down_fail_penalty: float = -0.4


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
            functools.partial(heading_pitch_V_reward_fn, reward_scale=1.0),
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            functools.partial(reward_nz_soft_penalty, scale=1.0),
            functools.partial(reward_low_qbar_penalty, scale=1.0),
            functools.partial(reward_energy_track,   scale=1.0),
        ]
        self.is_potential = [False] * len(self.reward_functions)

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_pitch_V_fn,
        ]

        # 课程学习的步进强度（保留）
        self.increment_size = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10)

    def _get_obs_size(self) -> int:
        return 24 # 22 + 2个新维度

    @property
    def default_params(self) -> Heading_Pitch_V_TaskParams:
        return Heading_Pitch_V_TaskParams()

    # ---------------- init/reset ----------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(self, key: chex.PRNGKey, params: Heading_Pitch_V_TaskParams) -> Heading_Pitch_V_TaskState:
        state = super()._init_state(key, params)

        # 初始航向 ~ U[0, 2π)
        key, key_heading = jax.random.split(key)
        initial_heading = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=0.0, maxval=2.0*jnp.pi)

        # 初始速度 ~ U[min_vt, max_vt]
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        # 初始四元数（按你的约定）
        half = initial_heading / 2.0
        q0 = -jnp.cos(half); q1 = jnp.zeros((self.num_agents,)); q2 = q1; q3 = jnp.sin(half)

        state = state.replace(
            plane_state=state.plane_state.replace(yaw=initial_heading, vt=vt, vel_y=vt, q0=q0, q1=q1, q2=q2, q3=q3)
        )

        init_target_heading = initial_heading
        init_target_pitch   = state.plane_state.pitch
        init_target_vt      = vt
        extra = jnp.vstack((init_target_heading, init_target_pitch, init_target_vt))
        state = Heading_Pitch_V_TaskState.create(state, extra_state=extra)

        # 清零统计
        zeros_like = jnp.zeros_like(init_target_heading, dtype=jnp.int32)
        state = state.replace(
            vertical_success_counts=zeros_like,
            is_vertical_target=jnp.zeros_like(init_target_heading, dtype=jnp.bool_),

            # NEW
            vertical_up_success_counts=zeros_like,
            vertical_down_success_counts=zeros_like,
            vertical_cmd_up_counts=zeros_like,
            vertical_cmd_down_counts=zeros_like,
            is_vertical_up_target=jnp.zeros_like(init_target_heading, dtype=jnp.bool_),
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(self, key: chex.PRNGKey, state: Heading_Pitch_V_TaskState, params: Heading_Pitch_V_TaskParams
                    ) -> Heading_Pitch_V_TaskState:
        state = self._generate_formation(key, state, params)

        # 初始航向/速度
        key, key_heading = jax.random.split(key)
        initial_heading = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=0.0, maxval=2.0*jnp.pi)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        vel_y = vt

        half = initial_heading / 2.0
        q0 = -jnp.cos(half); q1 = jnp.zeros((self.num_agents,)); q2 = q1; q3 = jnp.sin(half)

        state = state.replace(
            plane_state=state.plane_state.replace(vel_y=vel_y, vt=vt, yaw=initial_heading, q0=q0, q1=q1, q2=q2, q3=q3),
            target_heading=initial_heading,
            target_pitch=state.plane_state.pitch,
            target_vt=vt,
            cmd_target_heading=initial_heading,
            cmd_target_pitch=state.plane_state.pitch,
            cmd_target_vt=vt,

            # last_check_time=state.time,
            # last_switch_time=state.time,

            # heading_turn_counts=0,

            last_check_time=jnp.full_like(initial_heading, state.time),   # <- 标量改向量
            last_switch_time=jnp.full_like(initial_heading, state.time),  # <- 标量改向量

            heading_turn_counts=jnp.zeros_like(initial_heading, dtype=jnp.int32),  # <- 标量改向量

            vertical_success_counts=jnp.zeros_like(initial_heading, dtype=jnp.int32),
            is_vertical_target=jnp.zeros_like(initial_heading, dtype=jnp.bool_),

            # NEW
            vertical_up_success_counts=jnp.zeros_like(initial_heading, dtype=jnp.int32),
            vertical_down_success_counts=jnp.zeros_like(initial_heading, dtype=jnp.int32),
            vertical_cmd_up_counts=jnp.zeros_like(initial_heading, dtype=jnp.int32),
            vertical_cmd_down_counts=jnp.zeros_like(initial_heading, dtype=jnp.int32),
            is_vertical_up_target=jnp.zeros_like(initial_heading, dtype=jnp.bool_),
        )
        return state

    # ---------------- task step（关键改动） ----------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: Heading_Pitch_V_TaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: Heading_Pitch_V_TaskParams,
    ) -> Tuple[Heading_Pitch_V_TaskState, Dict[str, Any]]:

        # 多分一个 key_down
        key_h, key_p, key_v, key_mode, key_vert, key_down = jax.random.split(key, 6)  # CHANGED
        delta = jax.random.uniform(key_h, shape=(self.num_agents,), minval=0.5, maxval=1.0)

        # —— 常规桶：随机增量（俯仰限幅到 ±45°）——
        delta_heading = jax.random.uniform(key_h, shape=(self.num_agents,),
                                           minval=-params.max_heading_increment, maxval=params.max_heading_increment)

        current_altitude = state.plane_state.altitude
        max_pitch = jnp.where(current_altitude > params.max_altitude - 1000.0, -params.max_pitch_increment * 0.5,
                              params.max_pitch_increment)
        min_pitch = jnp.where(current_altitude < params.min_altitude + 1000.0, params.max_pitch_increment * 0.5,
                              -params.max_pitch_increment)
        delta_pitch = jax.random.uniform(key_p, shape=(self.num_agents,), minval=min_pitch, maxval=max_pitch)
        new_pitch_rand = jnp.clip(state.plane_state.pitch + delta_pitch, jnp.radians(-45.0), jnp.radians(45.0))
        delta_pitch = new_pitch_rand - state.plane_state.pitch

        delta_vt = jax.random.uniform(key_v, shape=(self.num_agents,),
                                      minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        rand_heading = wrap_PI(state.plane_state.yaw   + delta_heading * delta)
        rand_pitch   = wrap_PI(state.plane_state.pitch + delta_pitch   * delta)
        rand_vt      = state.plane_state.vt + delta_vt * delta

        # # —— 竖直桶：上/下都可以 —— 
        # choose_vertical = jax.random.bernoulli(key_mode, p=params.loop_mode_prob, shape=(self.num_agents,))

        # —— 竖直桶：上/下皆可，但上拉需“动压/速度门控” —— 
        choose_vertical = jax.random.bernoulli(key_mode, p=params.loop_mode_prob, shape=(self.num_agents,))

        # 动压估计（与 obs 中一致的归一化方式）
        alt_ft = state.plane_state.altitude / 0.3048
        vt_ft  = jnp.maximum(state.plane_state.vt / 0.3048, 0.1)
        _, qbar, _ = atmos(alt_ft, vt_ft)
        alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
        vt_ref_ft  = params.max_vt / 0.3048
        _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
        qbar_norm = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 2.0)

        # 原始下探意愿 & 门控
        choose_down_raw = jax.random.bernoulli(key_down, p=params.loop_down_prob, shape=(self.num_agents,))  # NEW
        
        # 低空保护：低于 min_altitude + buffer 禁止向下竖直
        allow_down = current_altitude > (params.min_altitude + params.down_alt_buffer)                       # NEW
        
        choose_down = choose_down_raw & allow_down                                                           # NEW
        
        # 上拉门控：速度/动压不足则不进入上拉竖直（先“攒能量”）
        allow_up = (state.plane_state.vt > params.loop_speed_low * 0.85) & (qbar_norm > params.qbar_low_frac * 0.90)

        # 组合得到“有效竖直选择”
        choose_vert_up   = choose_vertical & allow_up
        choose_vert_down = choose_vertical & choose_down_raw & allow_down
        choose_vertical_eff = choose_vert_up | choose_vert_down

        # vertical_sign = jnp.where(choose_vert_down, -1.0, 1.0)  # 仅当进入竖直时才有意义    
        vertical_sign_raw = jnp.where(choose_vert_down, -1.0, 1.0)
        vertical_sign = jnp.where(choose_vertical_eff, vertical_sign_raw, 0.0)

        thr_deg = 45.0  # CHANGED: 降低竖直最小阈值，减小瞬时跳变

        pitch_max_rad = jnp.deg2rad(params.loop_pitch_max_deg)
        pitch_thr_rad = jnp.deg2rad(thr_deg)
        pitch_vert_abs = jax.random.uniform(key_vert, shape=(self.num_agents,), minval=pitch_thr_rad, maxval=pitch_max_rad)

        cmd_cap = jnp.deg2rad(params.loop_cmd_pitch_cap_deg)
        pitch_vertical_cmd = jnp.clip(vertical_sign * pitch_vert_abs, -cmd_cap, cmd_cap)                     # CHANGED

        heading_vertical = state.cmd_target_heading  # 竖直时保持航向

        sinw = jnp.abs(jnp.sin(pitch_vertical_cmd))
        base_vt = jnp.clip(state.cmd_target_vt, params.min_vt, params.max_vt)
        #=============================================================================#
        # # 向上竖直：使用低速以保持控制性
        # # 向下竖直：使用中等速度，避免过度加速
        # up_speed = params.loop_speed_low      # 向上：低速（210）
        # down_speed = jnp.minimum(base_vt * 1.1, params.max_vt * 0.8)  # 向下：当前速度的1.1倍，但不超过最大速度的80%

        # vert_speed_target = jnp.where(vertical_sign > 0.0, up_speed, down_speed)
        # vt_vertical_cmd = jnp.clip((1.0 - sinw) * base_vt + sinw * vert_speed_target,
        #                         params.min_vt, params.max_vt)

        # 自适应竖直期望空速：上拉不再强行降到 210，避免 qbar 过低；qbar 过低时给一点速度补偿
        up_speed   = jnp.maximum(params.loop_speed_low, base_vt * 0.85)
        down_speed = jnp.minimum(base_vt * 1.05, params.max_vt * 0.8)

        vert_speed_target = jnp.where(vertical_sign > 0.0, up_speed, down_speed)
        vt_vertical_cmd = (1.0 - jnp.clip(sinw, 0.0, 1.0)) * base_vt + jnp.clip(sinw, 0.0, 1.0) * vert_speed_target

        # qbar 低时给一个温和的“救场增益”，缓解低动压长时间驻留
        shortfall = jnp.clip(params.qbar_low_frac - qbar_norm, 0.0, 1.0)
        vt_vertical_cmd = vt_vertical_cmd + 40.0 * shortfall
        vt_vertical_cmd = jnp.clip(vt_vertical_cmd, params.min_vt, params.max_vt)


        #==============================================================================#

        # # —— 本次“指令目标”（仅在 success=True 的切换瞬间刷新）——
        # sample_cmd_heading = jnp.where(choose_vertical, heading_vertical, rand_heading)
        # sample_cmd_pitch   = jnp.where(choose_vertical, pitch_vertical_cmd, rand_pitch)
        # sample_cmd_vt      = jnp.where(choose_vertical, vt_vertical_cmd,   rand_vt)

        # —— 本次“指令目标”（仅在切换瞬间刷新）——
        sample_cmd_heading = jnp.where(choose_vertical_eff, heading_vertical, rand_heading)
        sample_cmd_pitch   = jnp.where(choose_vertical_eff, pitch_vertical_cmd, rand_pitch)
        sample_cmd_vt      = jnp.where(choose_vertical_eff, vt_vertical_cmd,   rand_vt)

        #==============================================================================================#
        # _step_task 里引入强制切换（竖直更长，普通更短）
        # success_now = state.success

        # new_cmd_heading = jnp.where(success_now, sample_cmd_heading, state.cmd_target_heading)
        # new_cmd_pitch   = jnp.where(success_now, sample_cmd_pitch,   state.cmd_target_pitch)
        # new_cmd_vt      = jnp.where(success_now, sample_cmd_vt,      state.cmd_target_vt)

        success_now = state.success
        # === 强制切换：驻留时间过长也切一次 ===
        steps_per_sec = params.sim_freq // params.agent_interaction_steps
        dwell = (state.time - state.last_switch_time).astype(jnp.int32)
        force_vert = 20 * steps_per_sec   # 竖直驻留上限（秒）
        force_norm = 10  * steps_per_sec   # 普通驻留上限（秒）
        force_switch = jnp.where(state.is_vertical_target, dwell >= force_vert, dwell >= force_norm)
        do_switch = success_now | force_switch

        new_cmd_heading = jnp.where(do_switch, sample_cmd_heading, state.cmd_target_heading)
        new_cmd_pitch   = jnp.where(do_switch, sample_cmd_pitch,   state.cmd_target_pitch)
        new_cmd_vt      = jnp.where(do_switch, sample_cmd_vt,      state.cmd_target_vt)

        #==============================================================================================#

        # —— 切换后：标记是否“竖直目标” & 方向 & 计数 —— 
        completed_vertical = state.is_vertical_target & success_now
        # 方向以“切换前的目标方向”为准
        completed_up   = (state.is_vertical_target & state.is_vertical_up_target) & success_now               # NEW
        completed_down = (state.is_vertical_target & (~state.is_vertical_up_target)) & success_now           # NEW

        new_vertical_success_counts = state.vertical_success_counts + completed_vertical.astype(jnp.int32)
        new_vertical_up_success_counts = state.vertical_up_success_counts + completed_up.astype(jnp.int32)    # NEW
        new_vertical_down_success_counts = state.vertical_down_success_counts + completed_down.astype(jnp.int32) # NEW

        # 本次切换（若发生）采纳的新模式
        # new_is_vertical_target = jnp.where(success_now, choose_vertical, state.is_vertical_target).astype(jnp.bool_)
        new_is_vertical_target = jnp.where(do_switch, choose_vertical_eff, state.is_vertical_target).astype(jnp.bool_) # 目标成功达成后切换或者超时切换

        # new_is_vertical_up_target = jnp.where(  # NEW
        #     success_now,
        new_is_vertical_up_target = jnp.where(
            do_switch,
            jnp.where(choose_vertical_eff, (vertical_sign > 0.0), state.is_vertical_up_target),
            state.is_vertical_up_target
        ).astype(jnp.bool_)

        # 统计“发起次数”（仅在切换瞬间 + 选择了竖直）
        # cmd_up_inc   = (success_now & choose_vertical & (vertical_sign > 0.0)).astype(jnp.int32)              # NEW
        # cmd_down_inc = (success_now & choose_vertical & (vertical_sign < 0.0)).astype(jnp.int32)              # NEW

        cmd_up_inc   = (do_switch & choose_vertical_eff & (vertical_sign > 0.0)).astype(jnp.int32)
        cmd_down_inc = (do_switch & choose_vertical_eff & (vertical_sign < 0.0)).astype(jnp.int32)

        new_cmd_up_counts   = state.vertical_cmd_up_counts   + cmd_up_inc                                     # NEW
        new_cmd_down_counts = state.vertical_cmd_down_counts + cmd_down_inc                                   # NEW

        # new_last_check_time   = jnp.where(success_now, state.time, state.last_check_time)

        new_last_check_time   = jnp.where(do_switch, state.time, state.last_check_time)
        new_last_switch_time  = jnp.where(do_switch, state.time, state.last_switch_time)

        new_turn_counts       = state.heading_turn_counts + success_now.astype(state.heading_turn_counts.dtype)

        # —— 一阶平滑：实际目标追随指令目标（竖直慢、普通快）——
        alpha_vert = 1.0 / jnp.maximum(params.loop_phase_steps, 1)
        alpha_norm = 1.0 / jnp.maximum(params.ramp_steps_normal, 1)
        alpha = jnp.where(new_is_vertical_target, alpha_vert, alpha_norm)

        tgt_heading = wrap_PI(state.target_heading + alpha * wrap_PI(new_cmd_heading - state.target_heading))
        tgt_pitch   = wrap_PI(state.target_pitch   + alpha * wrap_PI(new_cmd_pitch   - state.target_pitch))
        tgt_vt      = jnp.clip(state.target_vt     + alpha * (new_cmd_vt - state.target_vt),
                               params.min_vt, params.max_vt)

        # —— 回写状态 —— 
        state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,  # 用过 success_now 后清零
            # 实际目标
            target_heading=tgt_heading,
            target_pitch=tgt_pitch,
            target_vt=tgt_vt,
            # 指令目标（仅切换时被刷新）
            cmd_target_heading=new_cmd_heading,
            cmd_target_pitch=new_cmd_pitch,
            cmd_target_vt=new_cmd_vt,
            # 统计
            last_check_time=new_last_check_time,
            last_switch_time=new_last_switch_time,

            heading_turn_counts=new_turn_counts,

            vertical_success_counts=new_vertical_success_counts,
            is_vertical_target=new_is_vertical_target,

            # NEW: 方向细分统计
            vertical_up_success_counts=new_vertical_up_success_counts,
            vertical_down_success_counts=new_vertical_down_success_counts,
            vertical_cmd_up_counts=new_cmd_up_counts,
            vertical_cmd_down_counts=new_cmd_down_counts,
            is_vertical_up_target=new_is_vertical_up_target,
        )

        # —— info（给训练/渲染打印）——
        info["heading_turn_counts"]    = state.heading_turn_counts
        info["vertical_success_counts"]= state.vertical_success_counts
        info["is_vertical_target"]     = state.is_vertical_target.astype(jnp.float32)
        info["is_vertical_up_target"]  = state.is_vertical_up_target.astype(jnp.float32)   # NEW
        # info["is_vertical_cmd"]        = (choose_vertical & success_now).astype(jnp.float32)
        # info["is_vertical_cmd_up"]     = (choose_vertical & (vertical_sign > 0.0) & success_now).astype(jnp.float32)   # NEW
        # info["is_vertical_cmd_down"]   = (choose_vertical & (vertical_sign < 0.0) & success_now).astype(jnp.float32)   # NEW

        info["switch_event"]           = do_switch.astype(jnp.float32)               # NEW：切换事件（成功或强制）
        info["is_vertical_cmd"]        = (choose_vertical_eff & do_switch).astype(jnp.float32)
        info["is_vertical_cmd_up"]     = (choose_vertical_eff & (vertical_sign > 0.0) & do_switch).astype(jnp.float32)
        info["is_vertical_cmd_down"]   = (choose_vertical_eff & (vertical_sign < 0.0) & do_switch).astype(jnp.float32)

        # 累计计数导出（episode 末步取最后一个就是“本局总计”）
        info["vertical_up_success_counts"]   = state.vertical_up_success_counts      # NEW
        info["vertical_down_success_counts"] = state.vertical_down_success_counts    # NEW
        info["vertical_cmd_up_counts"]       = state.vertical_cmd_up_counts          # NEW
        info["vertical_cmd_down_counts"]     = state.vertical_cmd_down_counts        # NEW

        # Debug: 指令 vs 实际
        info["target_heading_cmd_deg"] = jnp.rad2deg(state.cmd_target_heading)
        info["target_pitch_cmd_deg"]   = jnp.rad2deg(state.cmd_target_pitch)
        info["target_vt_cmd"]          = state.cmd_target_vt
        info["target_pitch_deg"]       = jnp.rad2deg(state.target_pitch)
        info["target_heading_deg"]     = jnp.rad2deg(state.target_heading)
        info["target_vt"]              = state.target_vt

        # ====== 下面保持你原来的监控/裁剪标记 ======
        ego_z_km = jnp.nan_to_num(state.plane_state.altitude / 1000.0, nan=0.0, posinf=1e6, neginf=-1e6)
        ego_vz_mh = jnp.nan_to_num(state.plane_state.vel_z / 340.0,    nan=0.0, posinf=1e6, neginf=-1e6)
        Kv = 0.2
        safe_alt = self.default_params.safe_altitude
        danger_alt = self.default_params.danger_altitude
        Pv = -jnp.clip(ego_vz_mh / Kv * (safe_alt - ego_z_km) / safe_alt, 0., 1.)
        Pv = jnp.where(ego_z_km <= safe_alt, Pv, jnp.zeros_like(Pv))
        PH = jnp.clip(ego_z_km / danger_alt, 0., 1.) - 1. - 1.
        PH = jnp.where(ego_z_km <= danger_alt, PH, jnp.zeros_like(PH))
        altitude_reward_raw = Pv + PH
        altitude_reward_will_clip = jnp.abs(altitude_reward_raw) > 10.0

        roll  = jnp.nan_to_num(state.plane_state.roll,  nan=0.0)
        pitch = jnp.nan_to_num(state.plane_state.pitch, nan=0.0)
        yaw   = jnp.nan_to_num(state.plane_state.yaw,   nan=0.0)
        vt    = jnp.nan_to_num(state.plane_state.vt,    nan=0.0)

        delta_heading = wrap_PI(yaw   - state.target_heading)
        delta_pitch   = wrap_PI(pitch - state.target_pitch)
        delta_vt      = jnp.nan_to_num(vt - state.target_vt, nan=0.0, posinf=1e6, neginf=-1e6)

        heading_error_scale = jnp.pi / 72
        pitch_error_scale   = jnp.pi / 72
        roll_error_scale    = 0.35
        speed_error_scale   = 24.0

        w_heading = 0.4; w_pitch = 0.3; w_roll = 0.1; w_speed = 0.2
        heading_r = jnp.exp(-((jnp.clip(delta_heading, -jnp.pi, jnp.pi) / heading_error_scale) ** 2))
        pitch_r   = jnp.exp(-((jnp.clip(delta_pitch,   -jnp.pi, jnp.pi) / pitch_error_scale) ** 2))
        roll_r    = jnp.exp(-((jnp.clip(roll, -10.0, 10.0) / roll_error_scale) ** 2))
        speed_r   = jnp.exp(-((jnp.clip(delta_vt, -1e3, 1e3)  / speed_error_scale) ** 2))
        hpv_reward_raw = (heading_r**w_heading) * (pitch_r**w_pitch) * (roll_r**w_roll) * (speed_r**w_speed)
        heading_pitch_V_reward_will_clip = hpv_reward_raw > 1.0

        info["clipped_altitude_reward_count"]      = altitude_reward_will_clip.astype(jnp.float32)
        info["clipped_heading_pitch_V_reward_count"] = heading_pitch_V_reward_will_clip.astype(jnp.float32)
        info["clipped_any_reward_count"]           = (altitude_reward_will_clip | heading_pitch_V_reward_will_clip).astype(jnp.float32)

        fn_hpv = functools.partial(heading_pitch_V_reward_fn, reward_scale=1.0)
        fn_alt = functools.partial(altitude_reward_fn,       reward_scale=1.0, Kv=0.2)
        r_hpv = jax.vmap(fn_hpv, in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))
        r_alt = jax.vmap(fn_alt, in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))

        from .reward_functions import reward_nz_soft_penalty, reward_low_qbar_penalty, reward_energy_track
        r_nz   = jax.vmap(reward_nz_soft_penalty, in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))
        r_qbar = jax.vmap(reward_low_qbar_penalty, in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))
        r_eng  = jax.vmap(reward_energy_track,     in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))
        info["dbg_r_hpv_mean"]  = jnp.mean(r_hpv)
        info["dbg_r_alt_mean"]  = jnp.mean(r_alt)
        info["dbg_r_nz_mean"]   = jnp.mean(r_nz)
        info["dbg_r_qbar_mean"] = jnp.mean(r_qbar)
        info["dbg_r_eng_mean"]  = jnp.mean(r_eng)

        info["has_nan_r_nz"]   = jnp.any(~jnp.isfinite(r_nz)).astype(jnp.float32)
        info["has_nan_r_qbar"] = jnp.any(~jnp.isfinite(r_qbar)).astype(jnp.float32)
        info["has_nan_r_eng"]  = jnp.any(~jnp.isfinite(r_eng)).astype(jnp.float32)

        return state, info

    # ---------------- observation（保持不变） ----------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: Heading_Pitch_V_TaskState,
        params: Heading_Pitch_V_TaskParams,
    ) -> Dict[AgentName, chex.Array]:
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_delta_heading = wrap_PI((yaw - state.target_heading))
        norm_delta_pitch = wrap_PI((pitch - state.target_pitch))
        norm_delta_vt = (vt - state.target_vt) / 340
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll); roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch); pitch_cos = jnp.cos(pitch)
        norm_vt = vt / 340
        alpha_sin = jnp.sin(alpha); alpha_cos = jnp.cos(alpha)
        beta_sin = jnp.sin(beta);   beta_cos = jnp.cos(beta)

        az = state.plane_state.az

        alt_ft = altitude / 0.3048
        vt_ft  = jnp.clip(vt / 0.3048, 0.1, 1e6)
        mach, qbar, _ = atmos(alt_ft, vt_ft)
        alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
        vt_ref_ft  = params.max_vt / 0.3048
        _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
        qbar_norm = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 2.0)

        spec_energy = 9.81 * altitude + 0.5 * jnp.clip(vt, 0.0, 1e4) ** 2
        e_ref = 9.81 * params.max_altitude + 0.5 * (params.max_vt ** 2)
        spec_energy_norm = jnp.clip(spec_energy / (e_ref + 1e-6), 0.0, 1.5)

        vx, vy, vz = state.plane_state.vel_x, state.plane_state.vel_y, state.plane_state.vel_z
        vh = jnp.sqrt(jnp.maximum(vx * vx + vy * vy, 1e-6))
        gamma = jnp.arctan2(-vz, vh)
        gamma_sin, gamma_cos = jnp.sin(gamma), jnp.cos(gamma)

        #===================================================================#
        # 在原有obs后面添加竖直状态
        # 添加竖直状态指示
        is_vertical = state.is_vertical_target.astype(jnp.float32)
        is_vertical_up = state.is_vertical_up_target.astype(jnp.float32) 
        vertical_direction = jnp.where(state.is_vertical_target, 
                                    jnp.where(state.is_vertical_up_target, 1.0, -1.0), 
                                    0.0)
        #===================================================================#

        obs = jnp.vstack((
            norm_delta_heading, norm_delta_pitch, norm_delta_vt,
            norm_altitude, norm_vt,
            roll_sin, roll_cos, pitch_sin, pitch_cos,
            alpha_sin, alpha_cos, beta_sin, beta_cos,
            P, Q, R,
            az, mach, qbar_norm, spec_energy_norm, gamma_sin, gamma_cos,
            is_vertical, vertical_direction  # 新增两个观测维度
        ))
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        low  = jnp.array([
            -jnp.pi, -jnp.pi, -2.0,
            0.0,     0.0,
            -1.0,    -1.0,   -1.0,   -1.0,
            -1.0,    -1.0,   -1.0,   -1.0,
            -10.0,   -10.0,  -10.0,
            -6.0,     0.0,    0.0,    0.0,   -1.0,  -1.0,
            0.0,     -1.0    # 新增：is_vertical [0,1], vertical_direction [-1,1]
        ]).reshape(-1, 1)

        high = jnp.array([
            jnp.pi,  jnp.pi,  2.0,
            5.0,     2.0,
            1.0,     1.0,     1.0,   1.0,
            1.0,     1.0,     1.0,   1.0,
            10.0,    10.0,    10.0,
            12.0,    3.0,     2.0,   2.0,    1.0,   1.0,
            1.0,     1.0      # 新增：is_vertical [0,1], vertical_direction [-1,1]
        ]).reshape(-1, 1)

        obs = jnp.clip(obs, low, high)
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}

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
        team_center =  team_center.at[2].set(altitude)
        formation_positions = enforce_safe_distance(team_positions, team_center, params.safe_distance)
        initial_heading = jnp.full((self.num_agents,), jnp.pi/2)
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2],
            yaw=initial_heading,
        ))
        return state



#========================================================================================#
#老版本：竖直桶切换俯仰比较剧烈，且可能出现从正俯仰瞬间跳到负俯仰

# from typing import Dict, Optional, Tuple, Any
# from jax import Array
# from jax.typing import ArrayLike
# import chex
# from .aeroplanax import AgentName, AgentID

# import functools
# import jax
# import jax.numpy as jnp
# from flax import struct
# from gymnax.environments import spaces
# from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
# from .core.simulators.fighterplane.dynamics import atmos
# from .reward_functions import (
#     heading_reward_fn,
#     heading_pitch_V_reward_fn,
#     altitude_reward_fn,
#     event_driven_reward_fn,
#     heading_pitch_v_event_driven_reward_fn,
#     reward_nz_soft_penalty,
#     reward_low_qbar_penalty,
#     reward_energy_track,
# )

# from .termination_conditions import (
#     crashed_fn,
#     timeout_fn,
#     unreach_heading_pitch_V_fn,
# )

# from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


# @struct.dataclass
# class Heading_Pitch_V_TaskState(EnvState):
#     target_heading: ArrayLike 
#     target_pitch: ArrayLike  # 新增目标俯仰角
#     target_vt: ArrayLike
#     last_check_time: ArrayLike
#     heading_turn_counts: ArrayLike

#     # 新增：竖直目标统计
#     vertical_success_counts: ArrayLike   # int32，累计完成竖直到达次数
#     is_vertical_target: ArrayLike        # bool，当前目标是否为竖直目标

#     @classmethod
#     def create(cls, env_state: EnvState, extra_state: Array):
#         return cls(
#             plane_state=env_state.plane_state,
#             missile_state=env_state.missile_state,
#             control_state=env_state.control_state,
#             pre_rewards=env_state.pre_rewards,  # 必须包含这一行
#             done=env_state.done,
#             success=env_state.success,
#             time=env_state.time,
#             target_heading=extra_state[0],
#             target_pitch=extra_state[1],  # 新增
#             target_vt=extra_state[2],
#             last_check_time=env_state.time,
#             heading_turn_counts=0,

#             vertical_success_counts=jnp.zeros_like(extra_state[0], dtype=jnp.int32),
#             is_vertical_target=jnp.zeros_like(extra_state[0], dtype=jnp.bool_),
#         )


# @struct.dataclass(frozen=True)
# class Heading_Pitch_V_TaskParams(EnvParams):
#     num_allies: int = 1
#     num_enemies: int = 0
#     num_missiles: int = 0
#     agent_type: int = 0
#     action_type: int = 1
#     formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
#     sim_freq: int = 50
#     agent_interaction_steps: int = 10
#     max_altitude: float = 20000.0
#     min_altitude: float = 2000.0
#     max_vt: float = 360.0
#     min_vt: float = 120.0
#     max_heading_increment: float = jnp.pi/2  # 最大航向变化量(90°)
#     max_pitch_increment: float = jnp.pi/6  # 最大俯仰角变化量(30°)
#     max_altitude_increment: float = 2100.0
#     max_velocities_u_increment: float = 50.0
#     safe_altitude: float = 4.0
#     danger_altitude: float = 3.5
#     noise_scale: float = 0.0
#     team_spacing: float = 15000       
#     safe_distance: float = 3000 # 编队最小安全间距

#     # 竖直段课程（新增）
#     loop_mode_prob: float = 0.5            # 每次 reset 有 50% 概率进入“竖直课程”
#     loop_pitch_max_deg: float = 90.0       # 目标俯仰极限（训练信号允许到 90°）
#     loop_phase_steps: int = 200            # 从水平→竖直的相位步数（越小拉升越快）
#     loop_speed_low: float = 210.0          # 竖直段低速目标（动压控制）
#     loop_cmd_pitch_cap_deg: float = 85.0   # 指令俯仰限幅（工程保护，建议 80~88）

#     # —— 新增：奖励塑形参数 —— 
#     r_nz_coef: float = 0.005         # 从 0.05 降到 0.005
#     r_qbar_coef: float = 0.02        # 低动压惩罚权重
#     r_energy_coef: float = 0.05      # 比能惩罚权重
#     nz_limit: float = 9.0            # g 限（正向）
#     qbar_low_frac: float = 0.35      # 动压下限（占参考动压的比例）
#     energy_ref_frac: float = 0.90    # 竖直段所需比能阈值（占参考比能的比例）
#     nz_hard_cap: float = 15.0    # 硬封顶（数值保险）
#     r_nz_clip: float = 3.0       # 单步最大惩罚（|r_nz| ≤ 3）

# class AeroPlanaxHeading_Pitch_V_Env(AeroPlanaxEnv[Heading_Pitch_V_TaskState, Heading_Pitch_V_TaskParams]):
#     def __init__(self, env_params: Optional[Heading_Pitch_V_TaskParams] = None):
#         super().__init__(env_params)
#         self.formation_type = env_params.formation_type

#         self.observation_spaces: Dict[AgentName, spaces.Space] = {
#             agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
#         }
#         self.action_spaces: Dict[AgentName, spaces.Space] = {
#             agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
#         }

#         self.reward_functions = [
#             functools.partial(heading_pitch_V_reward_fn, reward_scale=1.0),
#             functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
#             # functools.partial(event_driven_reward_fn, fail_reward=-20, success_reward=20),
#             # functools.partial(heading_pitch_v_event_driven_reward_fn, fail_reward=-50, success_reward=50),

#             functools.partial(reward_nz_soft_penalty, scale=1.0),
#             functools.partial(reward_low_qbar_penalty, scale=1.0),
#             functools.partial(reward_energy_track,   scale=1.0),
#         ]

#         # 与 reward_functions 一一对应，表示这些奖励是否做势能差分
#         # 这里全部设为 False 即可
#         self.is_potential = [False] * len(self.reward_functions)

#         self.termination_conditions = [
#             crashed_fn,
#             timeout_fn,
#             unreach_heading_pitch_V_fn,
#         ]

#         # 课程学习：
#         self.increment_size = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10)
#         # 前5个元素是 [0.2, 0.4, 0.6, 0.8, 1.0]
#         # 后10个元素是 [1.0] 重复10次
#         # 该数组用于控制航向/俯仰/速度变化量的增量系数
#         # 每次 heading_turn_counts 增加时，会按索引取对应的系数值进行缩放
#         # 前5次任务切换时增量系数逐步增大（0.2→1.0），后续保持1.0不变

#     def _get_obs_size(self) -> int:
#         return 22  # 16(原) + 6(气动/能量/轨迹角)

#     @property
#     def default_params(self) -> Heading_Pitch_V_TaskParams:
#         return Heading_Pitch_V_TaskParams()


#     @functools.partial(jax.jit, static_argnums=(0,))
#     def _init_state(
#         self,
#         key: chex.PRNGKey,
#         params: Heading_Pitch_V_TaskParams,
#     ) -> Heading_Pitch_V_TaskState:
#         state = super()._init_state(key, params)

#         # 随机初始航向 [0, 2π)
#         key, key_heading = jax.random.split(key)
#         initial_heading = jax.random.uniform(
#             key_heading,
#             shape=(self.num_agents,),
#             minval=0.0,
#             maxval=2.0 * jnp.pi,
#         )

#         # 随机初始速度 [min_vt, max_vt]
#         key, key_vt = jax.random.split(key)
#         vt = jax.random.uniform(
#             key_vt,
#             shape=(self.num_agents,),
#             minval=params.min_vt,
#             maxval=params.max_vt,
#         )

#         # 四元数（保持你的约定：q0=-cos(ψ/2), q3=sin(ψ/2)）
#         half_heading = initial_heading / 2.0
#         q0 = -jnp.cos(half_heading)
#         q1 = jnp.zeros((self.num_agents,))
#         q2 = jnp.zeros((self.num_agents,))
#         q3 = jnp.sin(half_heading)

#         # 写回飞机状态
#         state = state.replace(
#             plane_state=state.plane_state.replace(
#                 yaw=initial_heading,
#                 vt=vt,
#                 vel_y=vt,
#                 q0=q0,
#                 q1=q1,
#                 q2=q2,
#                 q3=q3,
#             )
#         )

#         # 初始目标：当前姿态/速度（不做任何竖直模式偏置）
#         init_target_heading = initial_heading
#         init_target_pitch   = state.plane_state.pitch
#         init_target_vt      = vt

#         extra = jnp.vstack((init_target_heading, init_target_pitch, init_target_vt))
#         state = Heading_Pitch_V_TaskState.create(state, extra_state=extra)

#         # 新增：竖直统计字段初始化
#         state = state.replace(
#             vertical_success_counts=jnp.zeros_like(init_target_heading, dtype=jnp.int32),
#             is_vertical_target=jnp.zeros_like(init_target_heading, dtype=jnp.bool_),
#         )

#         return state

#     @functools.partial(jax.jit, static_argnums=(0,))
#     def _reset_task(
#         self,
#         key: chex.PRNGKey,
#         state: Heading_Pitch_V_TaskState,
#         params: Heading_Pitch_V_TaskParams,
#     ) -> Heading_Pitch_V_TaskState:
#         # 生成编队/位置（沿用既有逻辑）
#         state = self._generate_formation(key, state, params)

#         # 随机初始航向 [0, 2π)
#         key, key_heading = jax.random.split(key)
#         initial_heading = jax.random.uniform(
#             key_heading,
#             shape=(self.num_agents,),
#             minval=0.0,
#             maxval=2.0 * jnp.pi,
#         )

#         # 随机初始速度 [min_vt, max_vt]
#         key, key_vt = jax.random.split(key)
#         vt = jax.random.uniform(
#             key_vt,
#             shape=(self.num_agents,),
#             minval=params.min_vt,
#             maxval=params.max_vt,
#         )
#         vel_y = vt

#         # 四元数（保持你的约定）
#         half_heading = initial_heading / 2.0
#         q0 = -jnp.cos(half_heading)
#         q1 = jnp.zeros((self.num_agents,))
#         q2 = jnp.zeros((self.num_agents,))
#         q3 = jnp.sin(half_heading)

#         # 写回飞机状态 + 初始目标（=当前）
#         state = state.replace(
#             plane_state=state.plane_state.replace(
#                 vel_y=vel_y,
#                 vt=vt,
#                 yaw=initial_heading,
#                 q0=q0,
#                 q1=q1,
#                 q2=q2,
#                 q3=q3,
#             ),
#             target_heading=initial_heading,
#             target_pitch=state.plane_state.pitch,
#             target_vt=vt,
#         )

#         # 新增：竖直统计字段初始化
#         state = state.replace(
#             vertical_success_counts=jnp.zeros_like(initial_heading, dtype=jnp.int32),
#             is_vertical_target=jnp.zeros_like(initial_heading, dtype=jnp.bool_),
#         )

#         return state

#     @functools.partial(jax.jit, static_argnums=(0,))
#     def _step_task(
#         self,
#         key: chex.PRNGKey,
#         state: Heading_Pitch_V_TaskState,
#         info: Dict[str, Any],
#         action: Dict[AgentName, chex.Array],
#         params: Heading_Pitch_V_TaskParams,
#     ) -> Tuple[Heading_Pitch_V_TaskState, Dict[str, Any]]:
#         """Task-specific step transition（不区分模式；到达/时间到才切换）."""
#         key_h, key_p, key_v, key_mode, key_sign, key_vert = jax.random.split(key, 6)
#         # 若需渐进难度，可用：delta = self.increment_size[state.heading_turn_counts]
#         delta = jax.random.uniform(key_h, shape=(self.num_agents,), minval=0.5, maxval=1.0)

#         # 常规桶：随机航向/俯仰/速度增量（俯仰考虑高度安全带，并限幅到 ±45°）
#         delta_heading = jax.random.uniform(
#             key_h, shape=(self.num_agents,),
#             minval=-params.max_heading_increment, maxval=params.max_heading_increment
#         )

#         current_altitude = state.plane_state.altitude
#         max_pitch = jnp.where(
#             current_altitude > params.max_altitude - 1000.0,
#             -params.max_pitch_increment * 0.5,
#             params.max_pitch_increment
#         )
#         min_pitch = jnp.where(
#             current_altitude < params.min_altitude + 1000.0,
#             params.max_pitch_increment * 0.5,
#             -params.max_pitch_increment
#         )
#         delta_pitch = jax.random.uniform(key_p, shape=(self.num_agents,), minval=min_pitch, maxval=max_pitch)
#         new_pitch_rand = jnp.clip(state.plane_state.pitch + delta_pitch, jnp.radians(-45.0), jnp.radians(45.0))
#         delta_pitch = new_pitch_rand - state.plane_state.pitch

#         delta_vt = jax.random.uniform(
#             key_v, shape=(self.num_agents,),
#             minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment
#         )

#         rand_target_heading = wrap_PI(state.plane_state.yaw + delta_heading * delta)
#         rand_target_pitch   = wrap_PI(state.plane_state.pitch + delta_pitch * delta)
#         rand_target_vt      = state.plane_state.vt + delta_vt * delta

#         # 竖直桶：以一定概率采样极端俯仰（工程限幅 + 速度降到低速）
#         choose_vertical = jax.random.bernoulli(key_mode, p=params.loop_mode_prob, shape=(self.num_agents,)) # 每次 新的目标值 有 50% 概率进入“竖直课程”
#         sign = jax.random.choice(key_sign, jnp.array([-1.0, 1.0]), shape=(self.num_agents,))
#         thr_deg = 70.0  # 竖直阈值
#         pitch_max_rad = jnp.deg2rad(params.loop_pitch_max_deg)
#         pitch_thr_rad = jnp.deg2rad(thr_deg)
#         pitch_vert_abs = jax.random.uniform(key_vert, shape=(self.num_agents,), minval=pitch_thr_rad, maxval=pitch_max_rad)
#         pitch_vertical = sign * pitch_vert_abs
#         cmd_cap = jnp.deg2rad(params.loop_cmd_pitch_cap_deg)
#         pitch_vertical_cmd = jnp.clip(pitch_vertical, -cmd_cap, cmd_cap)
#         heading_vertical = state.target_heading  # 竖直桶保持航向不变（也可设小扰动）
#         sinw = jnp.abs(jnp.sin(pitch_vertical_cmd))
#         base_vt = jnp.clip(state.target_vt, params.min_vt, params.max_vt)
#         vt_vertical = jnp.clip((1.0 - sinw) * base_vt + sinw * params.loop_speed_low, params.min_vt, params.max_vt)

#         # 在“切换”时选择新目标（choose_vertical 决定从哪个桶采样）
#         sample_heading = jnp.where(choose_vertical, heading_vertical, rand_target_heading)
#         sample_pitch   = jnp.where(choose_vertical, pitch_vertical_cmd, rand_target_pitch)
#         sample_vt      = jnp.where(choose_vertical, vt_vertical,        rand_target_vt)

#         # 仅在 success=True（到达或时间到）时更新目标与计时点
#         success_now = state.success  # 新增：先缓存

#         new_target_heading = jnp.where(state.success, sample_heading, state.target_heading)
#         new_target_pitch   = jnp.where(state.success, sample_pitch,   state.target_pitch)
#         new_target_vt      = jnp.where(state.success, sample_vt,      state.target_vt)

#         new_last_check_time = jnp.where(state.success, state.time, state.last_check_time)
#         new_heading_turn_counts = state.heading_turn_counts + state.success.astype(state.heading_turn_counts.dtype)

#         # === 新增：竖直完成计数（完成=切换前的目标是竖直 且 本步 success=True） ===
#         completed_vertical = state.is_vertical_target & state.success
#         new_vertical_success_counts = state.vertical_success_counts + completed_vertical.astype(jnp.int32)

#         # === 新增：更新“当前目标是否竖直”的标志（仅在切换时，用本次采样的 choose_vertical） ===
#         new_is_vertical_target = jnp.where(state.success, choose_vertical, state.is_vertical_target).astype(jnp.bool_)

#         # 写回
#         new_state = state.replace(
#             plane_state=state.plane_state.replace(
#                 status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
#             ),
#             success=False,  # 注意：在用 success_now 之后再清零
#             target_heading=new_target_heading,
#             target_pitch=new_target_pitch,
#             target_vt=new_target_vt,
#             last_check_time=new_last_check_time,
#             heading_turn_counts=new_heading_turn_counts,
#             vertical_success_counts=new_vertical_success_counts,
#             is_vertical_target=new_is_vertical_target,
#         )
#         state = new_state

#         # info：用 success_now，而不是 state.success（此时已被置 False）
#         info["heading_turn_counts"] = state.heading_turn_counts
#         info["is_vertical_cmd"] = (choose_vertical & success_now).astype(jnp.float32)
#         info["vertical_success_counts"] = state.vertical_success_counts
#         info["is_vertical_target"] = state.is_vertical_target.astype(jnp.float32)
#         info["target_pitch_cmd_deg"] = jnp.rad2deg(state.target_pitch)
#         info["target_vt_cmd"] = state.target_vt
#         info["target_heading_cmd_deg"] = jnp.rad2deg(state.target_heading)
#         info["switch_event"] = success_now.astype(jnp.float32)

#         #================================================================#
#         # ============================
#         # 在 info 中记录“奖励被裁剪”的标志
#         # ============================
#         # 1) altitude 奖励：你在 altitude_reward_fn 里裁剪到 [-10, 10]，并且有“超过 10 表示将被裁剪”的监控变量（见 altitude_reward.py）
#         #    参考：reward 输出裁剪阈值 [-10, 10] 与监控：cite__turn20file1
#         ego_z_km = jnp.nan_to_num(state.plane_state.altitude / 1000.0, nan=0.0, posinf=1e6, neginf=-1e6)
#         ego_vz_mh = jnp.nan_to_num(state.plane_state.vel_z / 340.0,    nan=0.0, posinf=1e6, neginf=-1e6)
#         Kv = 0.2
#         safe_alt = self.default_params.safe_altitude
#         danger_alt = self.default_params.danger_altitude
#         Pv = -jnp.clip(ego_vz_mh / Kv * (safe_alt - ego_z_km) / safe_alt, 0., 1.)
#         # Pv = jax.lax.select(ego_z_km <= safe_alt, Pv, 0.0)
#         Pv = jnp.where(ego_z_km <= safe_alt, Pv, jnp.zeros_like(Pv))
#         PH = jnp.clip(ego_z_km / danger_alt, 0., 1.) - 1. - 1.
#         # PH = jax.lax.select(ego_z_km <= danger_alt, PH, 0.0)
#         PH = jnp.where(ego_z_km <= danger_alt, PH, jnp.zeros_like(PH))
#         altitude_reward_raw = Pv + PH
#         # “会被裁剪”的判定（与 altitude_reward_fn 里保持一致：绝对值>10 会被 clip 到 [-10,10]）
#         altitude_reward_will_clip = jnp.abs(altitude_reward_raw) > 10.0

#         # 2) heading_pitch_V 奖励：你在 heading_pitch_V_reward_fn 里裁剪到 [0,1]
#         #    参考：输出 clip 到 [0,1] 与监控变量：cite__turn20file2
#         roll  = jnp.nan_to_num(state.plane_state.roll,  nan=0.0)
#         pitch = jnp.nan_to_num(state.plane_state.pitch, nan=0.0)
#         yaw   = jnp.nan_to_num(state.plane_state.yaw,   nan=0.0)
#         vt    = jnp.nan_to_num(state.plane_state.vt,    nan=0.0)

#         delta_heading = wrap_PI(yaw   - state.target_heading)
#         delta_pitch   = wrap_PI(pitch - state.target_pitch)
#         delta_vt      = jnp.nan_to_num(vt - state.target_vt, nan=0.0, posinf=1e6, neginf=-1e6)

#         # 与奖励函数保持同一尺度
#         heading_error_scale = jnp.pi / 72
#         pitch_error_scale   = jnp.pi / 72
#         roll_error_scale    = 0.35
#         speed_error_scale   = 24.0

#         # 按你的权重
#         w_heading = 0.4
#         w_pitch   = 0.3
#         w_roll    = 0.1
#         w_speed   = 0.2

#         # 这里计算“裁剪前”的 raw 值
#         heading_r = jnp.exp(-((jnp.clip(delta_heading, -jnp.pi, jnp.pi) / heading_error_scale) ** 2))
#         pitch_r   = jnp.exp(-((jnp.clip(delta_pitch,   -jnp.pi, jnp.pi) / pitch_error_scale) ** 2))
#         roll_r    = jnp.exp(-((jnp.clip(roll, -10.0, 10.0) / roll_error_scale) ** 2))
#         speed_r   = jnp.exp(-((jnp.clip(delta_vt, -1e3, 1e3)  / speed_error_scale) ** 2))

#         hpv_reward_raw = (heading_r**w_heading) * (pitch_r**w_pitch) * (roll_r**w_roll) * (speed_r**w_speed)
#         # “会被裁剪”的判定（>1 的部分会被 clip 到 1.0；一般不会>1，主要是数值噪声保护）
#         heading_pitch_V_reward_will_clip = hpv_reward_raw > 1.0

#         # 合并到 info（逐智能体布尔向量）
#         info["clipped_altitude_reward_count"] = altitude_reward_will_clip.astype(jnp.float32)
#         info["clipped_heading_pitch_V_reward_count"] = heading_pitch_V_reward_will_clip.astype(jnp.float32)
#         info["clipped_any_reward_count"] = (altitude_reward_will_clip | heading_pitch_V_reward_will_clip).astype(jnp.float32)

#         #================================================================#
#         #=== 奖励分量监控（均值，定位异常用） ===
#         fn_hpv = functools.partial(heading_pitch_V_reward_fn, reward_scale=1.0)
#         fn_alt = functools.partial(altitude_reward_fn,       reward_scale=1.0, Kv=0.2)
#         r_hpv = jax.vmap(fn_hpv, in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))
#         r_alt = jax.vmap(fn_alt, in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))
#         # 你新增的三项
#         from .reward_functions import reward_nz_soft_penalty, reward_low_qbar_penalty, reward_energy_track
#         r_nz   = jax.vmap(reward_nz_soft_penalty, in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))
#         r_qbar = jax.vmap(reward_low_qbar_penalty, in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))
#         r_eng  = jax.vmap(reward_energy_track,     in_axes=(None, None, 0))(state, params, jnp.arange(self.num_agents))
#         info["dbg_r_hpv_mean"]  = jnp.mean(r_hpv)
#         info["dbg_r_alt_mean"]  = jnp.mean(r_alt)
#         info["dbg_r_nz_mean"]   = jnp.mean(r_nz)
#         info["dbg_r_qbar_mean"] = jnp.mean(r_qbar)
#         info["dbg_r_eng_mean"]  = jnp.mean(r_eng)

#         # 监控：NaN 检测
#         info["has_nan_r_nz"]   = jnp.any(~jnp.isfinite(r_nz)).astype(jnp.float32)
#         info["has_nan_r_qbar"] = jnp.any(~jnp.isfinite(r_qbar)).astype(jnp.float32)
#         info["has_nan_r_eng"]  = jnp.any(~jnp.isfinite(r_eng)).astype(jnp.float32)
#         #================================================================#

#         return state, info

#     @functools.partial(jax.jit, static_argnums=(0,))
#     def _get_obs(
#         self,
#         state: Heading_Pitch_V_TaskState,
#         params: Heading_Pitch_V_TaskParams,
#     ) -> Dict[AgentName, chex.Array]:
#         """
#         Task-specific observation function to state.

#         observation(dim 16 + 6 = 22):
#             0. ego_delta_heading       (unit rad)
#             1. ego_delta_pitch         (unit rad)  # 新增
#             2. ego_delta_vt            (unit: mh)
#             3. ego_altitude            (unit: 5km)
#             4. ego_roll_sin
#             5. ego_roll_cos
#             6. ego_pitch_sin
#             7. ego_pitch_cos
#             8. ego_vt                  (unit: mh)
#             9. ego_alpha_sin
#             10. ego_alpha_cos
#             11. ego_beta_sin
#             12. ego_beta_cos
#             13. ego_P                  (unit: rad/s)
#             14. ego_Q                  (unit: rad/s)
#             15. ego_R                  (unit: rad/s)

#             16 nz(=az)
#             17 mach
#             18 qbar_norm
#             19 spec_energy_norm
#             20 gamma_sin
#             21 gamma_cos
#         """
#         altitude = state.plane_state.altitude
#         roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
#         vt = state.plane_state.vt
#         alpha = state.plane_state.alpha
#         beta = state.plane_state.beta
#         P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

#         norm_delta_heading = wrap_PI((yaw - state.target_heading))
#         norm_delta_pitch = wrap_PI((pitch - state.target_pitch))  # 新增
#         norm_delta_vt = (vt - state.target_vt) / 340
#         norm_altitude = altitude / 5000
#         roll_sin = jnp.sin(roll)
#         roll_cos = jnp.cos(roll)
#         pitch_sin = jnp.sin(pitch)
#         pitch_cos = jnp.cos(pitch)
#         norm_vt = vt / 340
#         alpha_sin = jnp.sin(alpha)
#         alpha_cos = jnp.cos(alpha)
#         beta_sin = jnp.sin(beta)
#         beta_cos = jnp.cos(beta)

#         # 气动/能量/轨迹角度，这里参数自适应归一化：用任务参数自动计算参考值，既覆盖全域又有合适动态范围。
#         az = state.plane_state.az  # 过载(垂直向)

#         # qbar 自适应归一化（按“中位高度 + 最大速度”计算 qbar_ref）（动压归一化）q̄ = 0.5·ρ·V²，反映“可用升力/气动载荷”的量级；q̄高→更容易拉出大过载，q̄低→更容易失速。
#         # 我们用参考值 q̄ref 归一化：qbar_norm = q̄ / q̄_ref，其中 q̄_ref 取“中位高度 + params.max_vt”计算得到，保证量纲一致且数值分布不塌缩。
#         alt_ft = altitude / 0.3048
#         vt_ft = jnp.clip(vt / 0.3048, 0.1, 1e6)
#         mach, qbar, _ = atmos(alt_ft, vt_ft)
#         alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
#         vt_ref_ft  = params.max_vt / 0.3048
#         _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
#         qbar_norm = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 2.0)

#         # 比能自适应归一化
#         # E = g·h + 0.5·V²（J/kg），反映飞机“总能量状态”；回环中它在势能和动能之间转换。
#         # 用参考值 E_ref = g·max_altitude + 0.5·max_vt² 做归一化：spec_energy_norm = E / E_ref，便于策略判断“还能不能继续抬头/是否需要先攒能量”。
#         spec_energy = 9.81 * altitude + 0.5 * jnp.clip(vt, 0.0, 1e4) ** 2
#         e_ref = 9.81 * params.max_altitude + 0.5 * (params.max_vt ** 2)
#         spec_energy_norm = jnp.clip(spec_energy / (e_ref + 1e-6), 0.0, 1.5) # spec_energy_norm（单位质量机械能归一化）

#         # γ（flight-path angle，航迹角）:
#         # 定义为相对地平线的航迹俯仰：γ = atan2(-v_z, sqrt(v_x²+v_y²))，上升为正、下降为负。
#         # 为避免角度在 ±π 处不连续，输入给策略的是 sinγ、cosγ 两个分量。
#         vx, vy, vz = state.plane_state.vel_x, state.plane_state.vel_y, state.plane_state.vel_z
#         vh = jnp.sqrt(jnp.maximum(vx * vx + vy * vy, 1e-6))
#         gamma = jnp.arctan2(-vz, vh)  # 向上为正
#         gamma_sin, gamma_cos = jnp.sin(gamma), jnp.cos(gamma)

#         obs = jnp.vstack((
#             norm_delta_heading, norm_delta_pitch, norm_delta_vt,
#             norm_altitude, norm_vt,
#             roll_sin, roll_cos, pitch_sin, pitch_cos,
#             alpha_sin, alpha_cos, beta_sin, beta_cos,
#             P, Q, R,
#             az, mach, qbar_norm, spec_energy_norm, gamma_sin, gamma_cos
#         ))
#         # 数值稳定化：去 NaN/Inf，并做合理限幅
#         obs = jnp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

#         """
#         逐维夹限（示例取值，结合你参数上限可再收紧）：
#             norm_delta_heading/pitch: [-π, π]
#             norm_delta_vt: [-2, 2]（(vt-target_vt)/340）
#             norm_altitude: [0, 5]（2~20km → 0.4~4）
#             norm_vt: [0, 2]（120~360 → 0.35~1.06）
#             roll_sin/cos, pitch_sin/cos, alpha_sin/cos, beta_sin/cos: [-1, 1]
#             P, Q, R（角速度，rad/s）: [-5, 5]（按你仿真上限调整）
#             如需更稳，可把 P/Q/R 改为 [-10, 10] 起步，再观察裁剪率收紧
#         """
#         low  = jnp.array([
#             -jnp.pi, -jnp.pi, -2.0,   # delta_heading, delta_pitch, delta_vt
#             0.0,     0.0,            # norm_altitude, norm_vt
#             -1.0,    -1.0,   -1.0,   -1.0,   # sin/cos
#             -1.0,    -1.0,   -1.0,   -1.0,   # sin/cos
#             -10.0,   -10.0,  -10.0,    # P, Q, R
#             -6.0,     0.0,    0.0,     0.0,   -1.0,  -1.0  # az, mach, qbar_norm, spec_energy_norm, gamma_sin, gamma_cos
#         ]).reshape(-1, 1)

#         high = jnp.array([
#             jnp.pi,  jnp.pi,  2.0,
#             5.0,     2.0,
#             1.0,     1.0,     1.0,   1.0,
#             1.0,     1.0,     1.0,   1.0,
#             10.0,    10.0,    10.0,
#             12.0,    3.0,     2.0,    2.0,    1.0,   1.0 # az, mach, qbar_norm, spec_energy_norm, gamma_sin, gamma_cos
#         ]).reshape(-1, 1)

#         obs = jnp.clip(obs, low, high)
#         return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    
#     @functools.partial(jax.jit, static_argnums=(0, ))
#     def _generate_formation(
#             self,
#             key: chex.PRNGKey,
#             state: Heading_Pitch_V_TaskState,
#             params: Heading_Pitch_V_TaskParams,
#         ) -> Heading_Pitch_V_TaskState:

#         # 根据队形类型选择生成函数
#         if self.formation_type == 0:
#             team_positions = wedge_formation(self.num_allies, params.team_spacing)
#         elif self.formation_type == 1:
#             team_positions = line_formation(self.num_allies, params.team_spacing)
#         elif self.formation_type == 2:
#             team_positions = diamond_formation(self.num_allies, params.team_spacing)
#         else:
#             raise ValueError("Provided formation type is not valid")
        
#         # 转换为全局坐标并确保安全距离        
#         team_center = jnp.zeros(3)
#         key, key_altitude = jax.random.split(key)
#         altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
#         team_center =  team_center.at[2].set(altitude)
#         formation_positions = enforce_safe_distance(team_positions, team_center, params.safe_distance)
#         initial_heading = jnp.full((self.num_agents,), jnp.pi/2)
#         state = state.replace(plane_state=state.plane_state.replace(
#             north=formation_positions[:, 0],
#             east=formation_positions[:, 1],
#             altitude=formation_positions[:, 2],
#             yaw=initial_heading,
#         ))
#         return state
