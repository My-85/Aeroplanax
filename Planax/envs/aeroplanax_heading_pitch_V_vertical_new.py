# Planax/envs/aeroplanax_heading_pitch_V.py
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
from .utils.utils import wrap_PI
from .reward_functions import (
    heading_pitch_V_reward_fn,
    altitude_reward_fn,
    reward_nz_soft_penalty,
    reward_low_qbar_penalty,
    reward_energy_track,
)
from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_pitch_V_fn,
)
@struct.dataclass
class Heading_Pitch_V_TaskState(EnvState):
    # “实际目标”（观测/奖励）
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_vt: ArrayLike
    # “指令目标”（只在切换时改变，随后一阶跟随）
    cmd_target_heading: ArrayLike
    cmd_target_pitch: ArrayLike
    cmd_target_vt: ArrayLike
    # 统计/调度
    last_switch_time: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike
    # 竖直统计与方向（简洁版）
    is_vertical_target: ArrayLike         # bool
    is_vertical_up_target: ArrayLike      # bool
    vertical_up_success_counts: ArrayLike # int
    vertical_down_success_counts: ArrayLike # int
    vertical_cmd_up_counts: ArrayLike
    vertical_cmd_down_counts: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        zeros_like = jnp.zeros_like(extra_state[0])
        time_vec   = jnp.full_like(extra_state[0], env_state.time)
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
            last_switch_time=time_vec,
            last_check_time=time_vec,
            heading_turn_counts=zeros_like.astype(jnp.int32),
            is_vertical_target=zeros_like.astype(jnp.bool_),
            is_vertical_up_target=zeros_like.astype(jnp.bool_),
            vertical_up_success_counts=zeros_like.astype(jnp.int32),
            vertical_down_success_counts=zeros_like.astype(jnp.int32),   # 修复：由 zeros_like(jnp.int32) → .astype(...)
            vertical_cmd_up_counts=zeros_like.astype(jnp.int32),
            vertical_cmd_down_counts=zeros_like.astype(jnp.int32),       # 修复：由 zeros_like(jnp.int32) → .astype(...)
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
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    max_vt: float = 360.0
    min_vt: float = 120.0

    # 随机桶最大增量
    max_heading_increment: float = jnp.pi/2     # ±90°
    max_pitch_increment: float   = jnp.pi/6     # ±30°
    max_velocities_u_increment: float = 50.0

    # 课程与竖直参与（简洁）
    loop_mode_prob: float = 0.35               # 每次切换进入“竖直指令”的概率
    pitch_curriculum_steps: int = 10           # 多少次切换后把竖直上限从 30° 提到 ~85°
    loop_pitch_max_deg: float = 88.0           # 课程上限
    loop_cmd_pitch_cap_deg: float = 88.0       # 指令硬限幅（工程保护）
    loop_speed_low: float = 210.0              # 向上竖直偏低速
    loop_speed_down: float = 300.0             # 向下竖直时趋向的目标高速
    loop_down_prob: float = 0.3                # 进入竖直后选“向下”的概率
    down_alt_buffer: float = 1800.0            # 低于 min_altitude+buffer 禁止向下竖直
    qbar_low_frac: float = 0.32                # 动压下限判据

    # 贴近竖直的采样偏置
    near_vertical_prob: float = 0.80           # 概率落在“上限窗口”内采样
    near_vertical_window_deg: float = 8.0      # 上限窗口宽度（度）

    # 平滑（避免一步突变）
    loop_phase_steps: int = 120                # 竖直阶段的目标跟随步数（越大越慢）
    ramp_steps_normal: int = 40                # 普通阶段的目标跟随步数

    # 奖励塑形（新增“回落稳定”）
    r_nz_coef: float = 0.005
    r_qbar_coef: float = 0.02
    r_energy_coef: float = 0.05
    nz_limit: float = 9.0
    nz_hard_cap: float = 15.0
    r_nz_clip: float = 3.0

class AeroPlanaxHeading_Pitch_V_Env(AeroPlanaxEnv[Heading_Pitch_V_TaskState, Heading_Pitch_V_TaskParams]):
    def __init__(self, env_params: Optional[Heading_Pitch_V_TaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type
        self.observation_spaces = {agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)}
        self.action_spaces = {agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)}
        # 奖励：保持简洁，外加“回落稳定奖励”
        self.reward_functions = [
            functools.partial(heading_pitch_V_reward_fn, reward_scale=1.0),
            functools.partial(altitude_reward_fn,       reward_scale=1.0, Kv=0.2),
            functools.partial(reward_nz_soft_penalty,   scale=1.0),
            functools.partial(reward_low_qbar_penalty,  scale=1.0),
            functools.partial(reward_energy_track,      scale=1.0),
            functools.partial(self._reward_vertical_down_stability, scale=1.0),
        ]
        self.is_potential = [False, False, True, True, True, False]

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_pitch_V_fn,
        ]
    def _get_obs_size(self) -> int:
        # 16(原) + 6(az,mach,qbar_norm,energy_norm,gamma_sin, gamma_cos)
        return 22

    @property
    def default_params(self) -> Heading_Pitch_V_TaskParams:
        return Heading_Pitch_V_TaskParams()

    # ---------------- init/reset ----------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(self, key: chex.PRNGKey, params: Heading_Pitch_V_TaskParams) -> Heading_Pitch_V_TaskState:
        state = super()._init_state(key, params)
        # 初始航向 [0, 2π)
        key, kh, kv, ka = jax.random.split(key, 4)
        hdg = jax.random.uniform(kh, shape=(self.num_agents,), minval=0.0, maxval=2*jnp.pi)
        # 初速
        vt = jax.random.uniform(kv, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        # 初始高度（避免一上来就触发高度终止）
        alt = jax.random.uniform(ka, shape=(self.num_agents,), minval=params.min_altitude + 3000.0, maxval=params.max_altitude - 2000.0)
        # 四元数（沿用你的约定：q0=-cos(ψ/2), q3=sin(ψ/2)）
        half = hdg / 2.0
        q0 = -jnp.cos(half); q3 = jnp.sin(half)
        state = state.replace(plane_state=state.plane_state.replace(
            yaw=hdg, vt=vt, vel_y=vt, vel_z=jnp.zeros_like(vt), altitude=alt, q0=q0, q1=jnp.zeros_like(hdg), q2=jnp.zeros_like(hdg), q3=q3
        ))
        extra = jnp.vstack((hdg, state.plane_state.pitch, vt))
        return Heading_Pitch_V_TaskState.create(state, extra_state=extra)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(self, key: chex.PRNGKey, state: Heading_Pitch_V_TaskState, params: Heading_Pitch_V_TaskParams) -> Heading_Pitch_V_TaskState:
        # 仅重新采样初始朝向/速度，目标=当前（平稳开局）
        key, kh, kv, ka = jax.random.split(key, 4)
        hdg = jax.random.uniform(kh, shape=(self.num_agents,), minval=0.0, maxval=2*jnp.pi)
        vt  = jax.random.uniform(kv, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        alt = jax.random.uniform(ka, shape=(self.num_agents,), minval=params.min_altitude + 3000.0, maxval=params.max_altitude - 2000.0)
        half = hdg / 2.0
        q0 = -jnp.cos(half); q3 = jnp.sin(half)
        state = state.replace(
            plane_state=state.plane_state.replace(vel_y=vt, vel_z=jnp.zeros_like(vt), altitude=alt, vt=vt, yaw=hdg, q0=q0, q1=jnp.zeros_like(hdg), q2=jnp.zeros_like(hdg), q3=q3),
            target_heading=hdg, target_pitch=state.plane_state.pitch, target_vt=vt,
            cmd_target_heading=hdg, cmd_target_pitch=state.plane_state.pitch, cmd_target_vt=vt,
            last_switch_time=jnp.full_like(hdg, state.time),
            last_check_time=jnp.full_like(hdg, state.time),
            heading_turn_counts=jnp.zeros_like(hdg, dtype=jnp.int32),
            is_vertical_target=jnp.zeros_like(hdg, dtype=jnp.bool_),
            is_vertical_up_target=jnp.zeros_like(hdg, dtype=jnp.bool_),
            vertical_up_success_counts=jnp.zeros_like(hdg, dtype=jnp.int32),
            vertical_down_success_counts=jnp.zeros_like(hdg, dtype=jnp.int32),
            vertical_cmd_up_counts=jnp.zeros_like(hdg, dtype=jnp.int32),
            vertical_cmd_down_counts=jnp.zeros_like(hdg, dtype=jnp.int32),
        )
        return state

    # ---------------- step（简洁竖直课程 + 平滑） ----------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(self, key: chex.PRNGKey, state: Heading_Pitch_V_TaskState, info: Dict[str, Any], action: Dict[AgentName, chex.Array], params: Heading_Pitch_V_TaskParams) -> Tuple[Heading_Pitch_V_TaskState, Dict[str, Any]]:
        key_h, key_p, key_v, key_mode, key_down = jax.random.split(key, 5)
        B = self.num_agents

        # 常规随机增量（小步）
        delta = jax.random.uniform(key_h, shape=(B,), minval=0.5, maxval=1.0)
        d_head = jax.random.uniform(key_h, shape=(B,), minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        cur_alt = state.plane_state.altitude
        max_pitch = jnp.where(cur_alt > params.max_altitude-1000.0, -params.max_pitch_increment*0.5, params.max_pitch_increment)
        min_pitch = jnp.where(cur_alt < params.min_altitude+1000.0, params.max_pitch_increment*0.5, -params.max_pitch_increment)
        d_pitch_small = jax.random.uniform(key_p, shape=(B,), minval=min_pitch, maxval=max_pitch)
        new_pitch_small = jnp.clip(state.plane_state.pitch + d_pitch_small, jnp.radians(-45.0), jnp.radians(45.0))
        d_pitch_small = new_pitch_small - state.plane_state.pitch
        d_vt = jax.random.uniform(key_v, shape=(B,), minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        # 是否进入竖直课程
        choose_vertical = jax.random.bernoulli(key_mode, p=params.loop_mode_prob, shape=(B,))
        # 课程：俯仰“可采上限”从 30° 线性拉到 loop_pitch_max_deg
        step_idx = jnp.clip(state.heading_turn_counts, 0, params.pitch_curriculum_steps)
        frac = step_idx / jnp.maximum(params.pitch_curriculum_steps, 1)
        pitch_hi_deg = 30.0 + (params.loop_pitch_max_deg - 30.0) * frac
        pitch_thr_deg = 45.0  # 进入竖直的最小阈值（降低瞬时跳变）
        pitch_hi = jnp.deg2rad(pitch_hi_deg)
        pitch_thr = jnp.deg2rad(pitch_thr_deg)

        # 上/下选择 + 门控
        choose_down_raw = jax.random.bernoulli(key_down, p=params.loop_down_prob, shape=(B,))
        allow_down = cur_alt > (params.min_altitude + params.down_alt_buffer)
        choose_down = choose_down_raw & allow_down
        # 动压估计
        alt_ft = state.plane_state.altitude / 0.3048
        vt_ft  = jnp.maximum(state.plane_state.vt / 0.3048, 0.1)
        _, qbar, _ = atmos(alt_ft, vt_ft)
        alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
        vt_ref_ft  = params.max_vt / 0.3048
        _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
        qbar_norm = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 2.0)
        allow_up = (state.plane_state.vt > params.loop_speed_low * 0.85) & (qbar_norm > params.qbar_low_frac * 0.90)

        choose_vert_up   = choose_vertical & allow_up
        choose_vert_down = choose_vertical & choose_down
        choose_vertical_eff = choose_vert_up | choose_vert_down
        vert_sign = jnp.where(choose_vert_down, -1.0, 1.0)

        #======================================================================================================#
        # # 竖直桶：在 [pitch_thr, pitch_hi] 内采样绝对值，再限幅到 cmd_cap
        # key_p2 = jax.random.split(key_p, 1)[0]
        # pitch_abs = jax.random.uniform(key_p2, shape=(B,), minval=pitch_thr, maxval=pitch_hi)
        # cmd_cap = jnp.deg2rad(params.loop_cmd_pitch_cap_deg)
        # pitch_vert_cmd = jnp.clip(vert_sign * pitch_abs, -cmd_cap, cmd_cap)

        # 竖直桶：优先在 [pitch_hi - Δ, pitch_hi] 贴近上限采样；否则在 [pitch_thr, pitch_hi] 采样
        key_p2, key_bias = jax.random.split(key_p, 2)
        pitch_hi = jnp.deg2rad(pitch_hi_deg)
        pitch_thr = jnp.deg2rad(pitch_thr_deg)
        nv = jax.random.bernoulli(key_bias, p=params.near_vertical_prob, shape=(B,))
        win = jnp.deg2rad(params.near_vertical_window_deg)
        low_win = jnp.maximum(pitch_hi - win, pitch_thr)
        low_bound = jnp.where(nv, low_win, pitch_thr)
        pitch_abs = jax.random.uniform(key_p2, shape=(B,), minval=low_bound, maxval=pitch_hi)
        cmd_cap = jnp.deg2rad(params.loop_cmd_pitch_cap_deg)
        pitch_vert_cmd = jnp.clip(vert_sign * pitch_abs, -cmd_cap, cmd_cap)

        #======================================================================================================#

        # 速度目标（竖直向上用低速，向下略提；并在动压偏低时给补偿）
        sinw = jnp.abs(jnp.sin(pitch_vert_cmd))
        base_vt = jnp.clip(state.cmd_target_vt, params.min_vt, params.max_vt)
        # up_speed   = jnp.maximum(params.loop_speed_low, base_vt * 0.85)
        # down_speed = jnp.minimum(base_vt * 1.05, params.max_vt * 0.8)
        # vt_vert_cmd= (1.0 - jnp.clip(sinw, 0.0, 1.0)) * base_vt + jnp.clip(sinw, 0.0, 1.0) * jnp.where(vert_sign>0, up_speed, down_speed)
        
        # 有了 loop_speed_down，你可以直接在配置里调向下竖直期望速度（例如 230~260 m/s），更易控制回落稳态与低动压风险。
        up_speed   = jnp.maximum(params.loop_speed_low, base_vt * 0.85)
        down_speed = jnp.clip(params.loop_speed_down, params.min_vt, params.max_vt * 0.9)
        vt_vert_cmd= (1.0 - jnp.clip(sinw, 0.0, 1.0)) * base_vt + jnp.clip(sinw, 0.0, 1.0) * jnp.where(vert_sign>0, up_speed, down_speed)    
        
        shortfall = jnp.clip(params.qbar_low_frac - qbar_norm, 0.0, 1.0)
        vt_vert_cmd = jnp.clip(vt_vert_cmd + 30.0 * shortfall, params.min_vt, params.max_vt)

        # 采样本次“指令目标”
        sample_cmd_heading = jnp.where(choose_vertical_eff, state.cmd_target_heading, wrap_PI(state.plane_state.yaw + d_head * delta))
        sample_cmd_pitch   = jnp.where(choose_vertical_eff, pitch_vert_cmd, wrap_PI(state.plane_state.pitch + (d_pitch_small * delta)))
        sample_cmd_vt      = jnp.where(choose_vertical_eff, vt_vert_cmd, state.plane_state.vt + d_vt * delta)

        # 切换判据：到达/强制驻留上限
        steps_per_sec = params.sim_freq // params.agent_interaction_steps
        dwell = (state.time - state.last_switch_time).astype(jnp.int32)
        force_vert = 20*steps_per_sec
        force_norm = 10*steps_per_sec
        force_switch = jnp.where(state.is_vertical_target, dwell>=force_vert, dwell>=force_norm)
        do_switch = state.success | force_switch

        new_cmd_heading = jnp.where(do_switch, sample_cmd_heading, state.cmd_target_heading)
        new_cmd_pitch   = jnp.where(do_switch, sample_cmd_pitch,   state.cmd_target_pitch)
        new_cmd_vt      = jnp.where(do_switch, sample_cmd_vt,      state.cmd_target_vt)

        # 切换后竖直标记与方向
        new_is_vertical_target = jnp.where(do_switch, choose_vertical_eff, state.is_vertical_target).astype(jnp.bool_)
        new_is_vertical_up     = jnp.where(do_switch, jnp.where(choose_vertical_eff, (vert_sign>0), state.is_vertical_up_target), state.is_vertical_up_target)
        # 统计“发起次数”
        cmd_up_inc   = (do_switch & choose_vertical_eff & (vert_sign>0)).astype(jnp.int32)
        cmd_down_inc = (do_switch & choose_vertical_eff & (vert_sign<0)).astype(jnp.int32)

        # 一阶跟随（避免一步突变）
        alpha_vert = 1.0 / jnp.maximum(params.loop_phase_steps, 1)
        alpha_norm = 1.0 / jnp.maximum(params.ramp_steps_normal, 1)
        alpha = jnp.where(new_is_vertical_target, alpha_vert, alpha_norm)
        tgt_heading = wrap_PI(state.target_heading + alpha * wrap_PI(new_cmd_heading - state.target_heading))
        tgt_pitch   = wrap_PI(state.target_pitch   + alpha * wrap_PI(new_cmd_pitch   - state.target_pitch))
        tgt_vt      = jnp.clip(state.target_vt     + alpha * (new_cmd_vt - state.target_vt), params.min_vt, params.max_vt)

        # 完成统计（上一段的目标若为竖直 & 本步成功）
        completed_up   = (state.is_vertical_target & state.is_vertical_up_target) & state.success
        completed_down = (state.is_vertical_target & (~state.is_vertical_up_target)) & state.success

        # 回写状态
        state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,
            target_heading=tgt_heading, target_pitch=tgt_pitch, target_vt=tgt_vt,
            cmd_target_heading=new_cmd_heading, cmd_target_pitch=new_cmd_pitch, cmd_target_vt=new_cmd_vt,
            last_switch_time=jnp.where(do_switch, state.time, state.last_switch_time),
            last_check_time=jnp.where(do_switch, state.time, state.last_check_time),
            heading_turn_counts=state.heading_turn_counts + do_switch.astype(jnp.int32),
            is_vertical_target=new_is_vertical_target,
            is_vertical_up_target=new_is_vertical_up.astype(jnp.bool_),
            vertical_cmd_up_counts=state.vertical_cmd_up_counts + cmd_up_inc,
            vertical_cmd_down_counts=state.vertical_cmd_down_counts + cmd_down_inc,
            vertical_up_success_counts=state.vertical_up_success_counts + completed_up.astype(jnp.int32),
            vertical_down_success_counts=state.vertical_down_success_counts + completed_down.astype(jnp.int32),
        )

        # ---------------- info：训练/渲染用 ----------------
        info["switch_event"] = do_switch.astype(jnp.float32)
        info["is_vertical_target"] = state.is_vertical_target.astype(jnp.float32)
        info["is_vertical_up_target"] = state.is_vertical_up_target.astype(jnp.float32)
        info["is_vertical_cmd"] = (do_switch & choose_vertical_eff).astype(jnp.float32)
        info["is_vertical_cmd_up"] = (do_switch & choose_vertical_eff & (vert_sign>0)).astype(jnp.float32)
        info["is_vertical_cmd_down"] = (do_switch & choose_vertical_eff & (vert_sign<0)).astype(jnp.float32)
        info["vertical_up_success_counts"] = state.vertical_up_success_counts
        info["vertical_down_success_counts"] = state.vertical_down_success_counts
        info["vertical_cmd_up_counts"] = state.vertical_cmd_up_counts
        info["vertical_cmd_down_counts"] = state.vertical_cmd_down_counts
        info["target_heading_cmd_deg"] = jnp.rad2deg(state.cmd_target_heading)
        info["target_pitch_cmd_deg"] = jnp.rad2deg(state.cmd_target_pitch)
        info["target_vt_cmd"] = state.cmd_target_vt
        info["target_heading_deg"] = jnp.rad2deg(state.target_heading)
        info["target_pitch_deg"]   = jnp.rad2deg(state.target_pitch)
        info["target_vt"]          = state.target_vt

        # 监控：低动压/能量/航迹角
        altitude = state.plane_state.altitude
        vt       = state.plane_state.vt
        alt_ft   = altitude / 0.3048
        vt_ft    = jnp.clip(vt / 0.3048, 0.1, 1e6)
        mach, qbar, _ = atmos(alt_ft, vt_ft)
        _, qbar_ref, _ = atmos(((params.min_altitude + params.max_altitude)*0.5)/0.3048, params.max_vt/0.3048)
        qbar_norm = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 2.0)
        vx, vy, vz = state.plane_state.vel_x, state.plane_state.vel_y, state.plane_state.vel_z
        vh = jnp.sqrt(jnp.maximum(vx*vx + vy*vy, 1e-6))
        gamma = jnp.arctan2(-vz, vh)
        spec_energy = 9.81 * altitude + 0.5 * jnp.clip(vt, 0.0, 1e4)**2
        e_ref = 9.81 * params.max_altitude + 0.5 * (params.max_vt**2)
        spec_energy_norm = jnp.clip(spec_energy / (e_ref + 1e-6), 0.0, 1.5)

        info["qbar_norm"] = qbar_norm
        info["low_qbar"]  = (qbar_norm < params.qbar_low_frac).astype(jnp.float32)
        info["gamma_deg"] = jnp.rad2deg(gamma)
        info["spec_energy_norm"] = spec_energy_norm
        return state, info

    # ---------------- observation（22维） ----------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: Heading_Pitch_V_TaskState, params: Heading_Pitch_V_TaskParams) -> Dict[AgentName, chex.Array]:
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha = state.plane_state.alpha; beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_delta_heading = wrap_PI(yaw - state.target_heading)
        norm_delta_pitch   = wrap_PI(pitch - state.target_pitch)
        norm_delta_vt      = (vt - state.target_vt) / 340.0
        norm_altitude      = altitude / 5000.0
        norm_vt            = vt / 340.0
        roll_sin, roll_cos   = jnp.sin(roll), jnp.cos(roll)
        pitch_sin,pitch_cos  = jnp.sin(pitch), jnp.cos(pitch)
        alpha_sin,alpha_cos  = jnp.sin(alpha), jnp.cos(alpha)
        beta_sin, beta_cos   = jnp.sin(beta),  jnp.cos(beta)

        az = state.plane_state.az
        alt_ft = altitude / 0.3048
        vt_ft  = jnp.clip(vt / 0.3048, 0.1, 1e6)
        mach, qbar, _ = atmos(alt_ft, vt_ft)
        _, qbar_ref, _ = atmos(((params.min_altitude + params.max_altitude)*0.5)/0.3048, params.max_vt/0.3048)
        qbar_norm = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 2.0)
        spec_energy = 9.81 * altitude + 0.5 * jnp.clip(vt, 0.0, 1e4)**2
        e_ref = 9.81 * params.max_altitude + 0.5 * (params.max_vt**2)
        spec_energy_norm = jnp.clip(spec_energy / (e_ref + 1e-6), 0.0, 1.5)
        vx, vy, vz = state.plane_state.vel_x, state.plane_state.vel_y, state.plane_state.vel_z
        vh = jnp.sqrt(jnp.maximum(vx*vx + vy*vy, 1e-6))
        gamma = jnp.arctan2(-vz, vh)
        gamma_sin, gamma_cos = jnp.sin(gamma), jnp.cos(gamma)

        obs = jnp.vstack((
            norm_delta_heading, norm_delta_pitch, norm_delta_vt,
            norm_altitude, norm_vt,
            roll_sin, roll_cos, pitch_sin, pitch_cos,
            alpha_sin, alpha_cos, beta_sin, beta_cos,
            P, Q, R,
            az, mach, qbar_norm, spec_energy_norm, gamma_sin, gamma_cos
        ))
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        low  = jnp.array([-jnp.pi, -jnp.pi, -2.0, 0.0, 0.0, -1,-1,-1,-1, -1,-1,-1,-1, -10,-10,-10, -6.0,0.0,0.0,0.0,-1.0,-1.0]).reshape(-1,1)
        high = jnp.array([ jnp.pi,  jnp.pi,  2.0, 5.0, 2.0,  1, 1, 1, 1,  1, 1, 1, 1,  10, 10, 10, 12.0,3.0,2.0,2.0, 1.0, 1.0]).reshape(-1,1)
        obs = jnp.clip(obs, low, high)
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}

    # ---------------- rewards（简洁） ----------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_hpv_like(self, state: Heading_Pitch_V_TaskState, params: Heading_Pitch_V_TaskParams, agent_id: AgentID, scale: float=1.0):
        # 与你原 hpv 奖励同型（简化）
        yaw, pitch, roll = state.plane_state.yaw[agent_id], state.plane_state.pitch[agent_id], state.plane_state.roll[agent_id]
        dh = wrap_PI(yaw - state.target_heading[agent_id])
        dp = wrap_PI(pitch - state.target_pitch[agent_id])
        dv = jnp.clip(state.plane_state.vt[agent_id] - state.target_vt[agent_id], -1e3, 1e3)
        h_scale = jnp.pi/72; p_scale = jnp.pi/72; r_scale = 0.35; v_scale = 24.0
        w_h, w_p, w_r, w_v = 0.4, 0.3, 0.1, 0.2
        Rh = jnp.exp(-((jnp.clip(dh,-jnp.pi,jnp.pi)/h_scale)**2))
        Rp = jnp.exp(-((jnp.clip(dp,-jnp.pi,jnp.pi)/p_scale)**2))
        Rr = jnp.exp(-((jnp.clip(roll,-10.0,10.0)/r_scale)**2))
        Rv = jnp.exp(-((dv / v_scale)**2))
        return scale * ((Rh**w_h)*(Rp**w_p)*(Rr**w_r)*(Rv**w_v))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_altitude_guard(self, state: Heading_Pitch_V_TaskState, params: Heading_Pitch_V_TaskParams, agent_id: AgentID, scale: float=1.0):
        z_km = jnp.nan_to_num(state.plane_state.altitude[agent_id]/1000.0, nan=0.0)
        vz_mh= jnp.nan_to_num(state.plane_state.vel_z[agent_id]/340.0,    nan=0.0)
        Kv=0.2; safe=4.0; danger=3.5
        Pv = -jnp.clip(vz_mh/Kv*(safe-z_km)/safe, 0., 1.); Pv = jnp.where(z_km<=safe, Pv, 0.0)
        PH = jnp.clip(z_km/danger,0.,1.)-1.-1.;    PH = jnp.where(z_km<=danger,PH,0.0)
        raw = Pv+PH
        return scale * jnp.clip(raw, -10.0, 10.0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reward_vertical_down_stability(self, state: Heading_Pitch_V_TaskState, params: Heading_Pitch_V_TaskParams, agent_id: AgentID, scale: float=1.0):
        # 仅在“目标为竖直且向下”时生效：用掩码避免 Python 分支
        is_vert_down = (state.is_vertical_target[agent_id] & (~state.is_vertical_up_target[agent_id]))
        mask = is_vert_down.astype(jnp.float32)

        roll = jnp.abs(state.plane_state.roll[agent_id])
        q    = jnp.abs(state.plane_state.Q[agent_id])         # 俯仰角速度
        pitch_err = jnp.abs(wrap_PI(state.plane_state.pitch[agent_id] - state.target_pitch[agent_id]))
        vz   = -state.plane_state.vel_z[agent_id]             # 向下为正

        r_roll = jnp.exp(- (jnp.clip(roll, 0.0, jnp.deg2rad(30.0)) / jnp.deg2rad(8.0))**2)
        r_q    = jnp.exp(- (jnp.clip(q,    0.0, 0.8) / 0.25)**2)
        r_pe   = jnp.exp(- (jnp.clip(pitch_err, 0.0, jnp.deg2rad(25.0)) / jnp.deg2rad(8.0))**2)
        p_vz   = -jnp.clip(vz - 90.0, 0.0) / 90.0            # 下冲过快惩罚（>90 m/s 开始罚）

        reward = scale * (0.4*r_roll + 0.3*r_q + 0.3*r_pe + 0.5*p_vz)
        return reward * mask