# Planax/envs/aeroplanax_full_pitch.py
# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, Any
from jax.typing import ArrayLike
import chex
import functools
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv, AgentName, AgentID
from .utils.utils import wrap_PI
from .core.simulators.fighterplane.dynamics import atmos
from .reward_functions import (
    heading_pitch_reward_fn,      # 奖励里可按 params.use_vt_in_reward 开关速度项
    altitude_reward_fn,
    reward_low_qbar_penalty,
    reward_nz_soft_penalty,
    reward_energy_track,
)
from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_pitch_fn,     # 仅俯仰/竖直/常规三模式兼容（见下文）
)


@struct.dataclass
class Pitch_Curriculum_TaskState(EnvState):
    # “实际目标”（被观测/被奖励判定）
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_vt: ArrayLike

    # 统计
    last_check_time: ArrayLike                # 任务切换/判定窗口的起始时刻（单位：决策步）
    heading_turn_counts: ArrayLike            # 切换计数（可视化）

    # 课程：俯仰单桶推进（共9个桶：0..8 → [0,10]…[80,90]）
    curriculum_max_bin: ArrayLike             # 已解锁最高桶（每智能体独立）
    current_bin_idx: ArrayLike                # 当前训练所在桶
    bin_trials: ArrayLike                     # (B, 9) 每桶尝试数
    bin_successes: ArrayLike                  # (B, 9) 每桶成功数

    @classmethod
    def create(cls, env_state: EnvState, extra_state: jnp.ndarray):
        """
        extra_state 约定：shape = (3, B)
        [0] -> 初始目标 heading
        [1] -> 初始目标 pitch
        [2] -> 初始目标 vt
        """
        B = extra_state.shape[1]
        zeros_bins = jnp.zeros((B, 9), dtype=jnp.int32)
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
            last_check_time=env_state.time,
            heading_turn_counts=jnp.zeros((B,), dtype=jnp.int32),
            curriculum_max_bin=jnp.zeros((B,), dtype=jnp.int32),
            current_bin_idx=jnp.zeros((B,), dtype=jnp.int32),
            bin_trials=zeros_bins,
            bin_successes=zeros_bins,
        )


@struct.dataclass(frozen=True)
class Pitch_Curriculum_TaskParams(EnvParams):
    # 常规 env 参数
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
    max_vt: float = 1000.0
    min_vt: float = 120.0

    qbar_ref_vt:    float = 360.0   # 用于 qbar_norm 归一化的参考速度
    energy_ref_vt:  float = 360.0   # 用于 energy_norm 的参考速度
    energy_ref_alt: float = 20000.0 # 能量参考高度（m），保留原 20km

    max_heading_increment: float = jnp.pi / 2   # 小幅航向扰动上限（避免完全静态）
    safe_altitude: float = 4.0                  # km，altitude_reward 用
    danger_altitude: float = 3.5                # km，altitude_reward 用

    # 仅俯仰课程参数
    pitch_only_mode: bool = True                # 仅俯仰判定&训练
    pitch_bin_deg: int = 10                     # 单桶宽度
    curriculum_target_rate: float = 0.75         # 当前桶达标率阈值    # 0.80 -> 0.75
    curriculum_min_trials: int = 60             # 当前桶最小尝试数
    target_max_jump_deg: float = 90.0           # 每次目标最大跃迁角，防大跳

    # 奖励开关
    use_vt_in_reward: bool = False

    # ---- 安全/惩罚/控制参数（新增）----
    qbar_low_frac: float = 0.35          # 低动压阈值（reward/控制门）
    r_qbar_coef:  float = 0.06           # 低动压惩罚强度（原0.02→0.06）
    qbar_crash_frac: float = 0.20        # 动压坠毁阈值（见 crashed_fn）
    promote_min_qbar: float = 0.45       # 升桶条件①动压安全门：需满足的最小 qbar_norm
    promote_min_energy: float = 0.75     # 升桶条件②能量安全门：比能归一化阈值

    # 新增：用于渲染固定模式
    render_no_switch: bool = struct.field(pytree_node=False, default=False)          # True 时完全不切换目标（也不升桶）
    render_fixed_bin_idx: int = struct.field(pytree_node=False, default=-1)          # >=0 固定到指定桶（0..8），否则用当前桶
    render_fixed_pitch_deg: float = struct.field(pytree_node=False, default=float("nan"))  # 非 NaN 则强制把目标俯仰固定到该角度（度）

    only_negative: bool = True

class AeroPlanaxPitchCurriculumEnv(AeroPlanaxEnv[Pitch_Curriculum_TaskState, Pitch_Curriculum_TaskParams]):
    """
    仅俯仰课程（单桶逐级推进、不控速度）的训练环境：
    - 目标俯仰只在“当前桶” [k*10°, (k+1)*10°] 内采样。
    - 方向自动择“更接近上一次目标”的一侧，并限幅目标跃迁，避免剧烈改变。
    - 目标速度恒等于当前速度，策略专注俯仰跟踪；奖励里速度项可关闭。
    - 观测22维：在原16维基础上加入 az、mach、qbar_norm、energy_norm、gamma_sin、gamma_cos。
    """

    def __init__(self, env_params: Optional[Pitch_Curriculum_TaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        # 奖励：heading-pitch（可选速度）+ 海拔守恒惩罚
        self.reward_functions = [
            functools.partial(heading_pitch_reward_fn, reward_scale=1.0),
            functools.partial(altitude_reward_fn,       reward_scale=1.0, Kv=0.2),
            # 新增：低qbar惩罚（scale=-1.0表示负奖励，强度可调）
            functools.partial(reward_low_qbar_penalty,  scale=1.0),   # FIX: 保持为负奖励
            # 新增：过载软惩罚（scale=-0.5，强度适中）
            functools.partial(reward_nz_soft_penalty, scale=0.5),   # FIX: 保持为负奖励
            # 新增：能量储备惩罚（scale=-0.5，只在竖直目标时激活）
            functools.partial(reward_energy_track, scale=4.0),   # FIX: 保持为负奖励  scale=0.5 可能太弱
        ]
        self.is_potential = [False] * len(self.reward_functions)

        # 终止条件：不结束 episode，仅返回 success 用于任务切换
        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_pitch_fn,
        ]
    def get_reward_functions(self):
        return self.reward_functions
    def _get_obs_size(self) -> int:
        return 22

    @property
    def default_params(self) -> Pitch_Curriculum_TaskParams:
        return Pitch_Curriculum_TaskParams()

    # ---------- 初始化 / 重置 ----------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(self, key: chex.PRNGKey, params: Pitch_Curriculum_TaskParams) -> Pitch_Curriculum_TaskState:
        state = super()._init_state(key, params)

        # 初始 yaw ∈ [0, 2π)
        key, kh, kv, kz = jax.random.split(key, 4)
        hdg = jax.random.uniform(kh, shape=(self.num_agents,), minval=0.0, maxval=2.0 * jnp.pi)

        # 初速 ∈ [min_vt, max_vt]
        vt = jax.random.uniform(kv, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        # 安全高度 ∈ [min_altitude, max_altitude]
        # alt = jax.random.uniform(kz, shape=(self.num_agents,), minval=params.min_altitude, maxval=params.max_altitude)
        # 安全高度 ∈ [10000, max_altitude]  # 修改：从高海拔起步
        alt = jax.random.uniform(kz, shape=(self.num_agents,), minval=10000.0, maxval=params.max_altitude)


        # 四元数（约定）：q0=-cos(ψ/2), q3=sin(ψ/2)
        half = hdg / 2.0
        q0, q3 = -jnp.cos(half), jnp.sin(half)

        # 回写飞机状态（把 vel_y=vt，便于平直开始）
        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=hdg, vt=vt, vel_y=vt,
                altitude=alt,
                q0=q0, q1=jnp.zeros_like(hdg), q2=jnp.zeros_like(hdg), q3=q3
            )
        )

        # 初始目标 = 当前状态
        extra = jnp.vstack((hdg, state.plane_state.pitch, vt))
        return Pitch_Curriculum_TaskState.create(state, extra)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(self, key: chex.PRNGKey, state: Pitch_Curriculum_TaskState, params: Pitch_Curriculum_TaskParams) -> Pitch_Curriculum_TaskState:
        # 与 _init_state 一致，且课程统计清零
        key, kh, kv, kz = jax.random.split(key, 4)
        hdg = jax.random.uniform(kh, shape=(self.num_agents,), minval=0.0, maxval=2.0 * jnp.pi)
        vt  = jax.random.uniform(kv, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        # alt = jax.random.uniform(kz, shape=(self.num_agents,), minval=params.min_altitude, maxval=params.max_altitude)
        alt = jax.random.uniform(kz, shape=(self.num_agents,), minval=10000.0, maxval=params.max_altitude)  # 修改：从高海拔起步

        half = hdg / 2.0
        q0, q3 = -jnp.cos(half), jnp.sin(half)

        B = self.num_agents
        zeros_bins = jnp.zeros((B, 9), dtype=jnp.int32)

        # 在 _reset_task 返回前，替换课程相关字段为“继承”或“衰减继承”
        # decay = 0.95  # 可调，或直接不衰减
        # bin_trials     = (state.bin_trials.astype(jnp.float32)     * decay).astype(jnp.int32)
        # bin_successes  = (state.bin_successes.astype(jnp.float32)  * decay).astype(jnp.int32)

        # - 每次 reset 都会向下取整。比如 61 → 57（*0.95=57.95 → int32=57），多来几次就低于 60 了；均值统计（跨并行环境）也会被新启动的小 t_cur 稀释，于是你看到 `t_cur` “中期下降”。  
        # - 目的本来是“不要清零”，那就直接保留为 float，不要回写成 int；阈值比较时用 float 一样工作。改成：
        decay = 0.995  # 或 1.0 先验证
        bin_trials    = state.bin_trials.astype(jnp.float32)    * decay  # 不再 cast 回 int32
        bin_successes = state.bin_successes.astype(jnp.float32) * decay


        return state.replace(
            plane_state=state.plane_state.replace(
                yaw=hdg, vt=vt, vel_y=vt, altitude=alt,
                q0=q0, q1=jnp.zeros_like(hdg), q2=jnp.zeros_like(hdg), q3=q3
            ),
            target_heading=hdg,
            target_pitch=state.plane_state.pitch,
            target_vt=vt,
            last_check_time=jnp.broadcast_to(state.time, (B,)).astype(jnp.int32),

            # 关键：不要清零课程进度
            curriculum_max_bin=state.curriculum_max_bin,
            current_bin_idx=jnp.maximum(state.current_bin_idx, state.curriculum_max_bin),  # 从最高桶继续

            # 关键：不要清空统计；如需防老化，用衰减继承
            bin_trials=bin_trials,
            bin_successes=bin_successes,

            heading_turn_counts=jnp.zeros((B,), dtype=jnp.int32),
        )

    # ---------- 单步任务逻辑：仅俯仰课程 ----------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: Pitch_Curriculum_TaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: Pitch_Curriculum_TaskParams
    ) -> Tuple[Pitch_Curriculum_TaskState, Dict[str, Any]]:
        B = self.num_agents
        key, k_head, k_bin = jax.random.split(key, 3)

        # (1) 记录上一段桶/成功
        prev_bin = state.current_bin_idx
        was_success = state.success.astype(jnp.int32)

        # (2) 轻微航向扰动（避免静态；也可改成直接保持 yaw 不变）
        delta = jax.random.uniform(k_head, shape=(B,), minval=0.3, maxval=0.8)
        d_head = jax.random.uniform(
            k_head, shape=(B,),
            minval=-params.max_heading_increment * 0.05,
            maxval= params.max_heading_increment * 0.05
        )
        sampled_heading = wrap_PI(state.plane_state.yaw + d_head * delta)

        # (3) 只在“当前桶”采样目标俯仰绝对值
        num_bins = 9
        cur_bin = state.current_bin_idx
        lo_deg = cur_bin * params.pitch_bin_deg
        hi_deg = (cur_bin + 1) * params.pitch_bin_deg
        # pitch_abs_deg = jax.random.uniform(k_bin, shape=(B,), minval=lo_deg, maxval=hi_deg)
        # pitch_abs_rad = jnp.deg2rad(pitch_abs_deg)

        # （低端偏置，降低难度）
        u = jax.random.uniform(k_bin, shape=(B,))               # U(0,1)
        bias = u * u                                            # 低端偏置
        pitch_abs_deg = lo_deg + (hi_deg - lo_deg) * bias
        pitch_abs_rad = jnp.deg2rad(pitch_abs_deg)

        # 方向选择：取“更接近上一目标”的一侧，若 only_negative=True 则强制负号，并做目标跃迁限幅，避免 +60° → -60° 大跳
        cand_pos =  pitch_abs_rad
        cand_neg = -pitch_abs_rad
        prev_tgt = state.target_pitch
        d_pos = jnp.abs(wrap_PI(cand_pos - prev_tgt))
        d_neg = jnp.abs(wrap_PI(cand_neg - prev_tgt))
        # picked = jnp.where(d_pos <= d_neg, cand_pos, cand_neg)

        picked_default = jnp.where(d_pos <= d_neg, cand_pos, cand_neg)

        only_neg = jnp.asarray(params.only_negative, jnp.bool_)
        picked = jnp.where(only_neg, -pitch_abs_rad, picked_default)
 
        # 限制每次目标跃迁角，防止 +60°→-60° 大跳
        max_jump = jnp.deg2rad(params.target_max_jump_deg)
        delta_pitch = jnp.clip(wrap_PI(picked - prev_tgt), -max_jump, max_jump)
        sampled_pitch = wrap_PI(prev_tgt + delta_pitch)

        # (4) 不控速度：目标速度=当前速度（课程阶段只训练俯仰）
        sampled_vt = state.plane_state.vt

        # ===================== [NEW] 渲染固定模式：不切换/不升桶（仅用于可视化） =====================
        # 说明：
        # - render_no_switch=True 时，本步跳过“切换与升桶”，固定目标俯仰（两个优先级：render_fixed_pitch_deg > render_fixed_bin_idx）
        # - 仍然输出 info 中的课程/奖励/qbar 相关监控，便于 Tacview 与日志对齐
        if params.render_no_switch:
            # 以当前桶中心角度为后备固定角
            lo_deg_s = cur_bin * params.pitch_bin_deg
            hi_deg_s = (cur_bin + 1) * params.pitch_bin_deg
            center_deg = 0.5 * (lo_deg_s + hi_deg_s)
            center_rad = jnp.deg2rad(center_deg)

            # 计算固定目标俯仰
            # 1) 若指定了固定角（度），优先使用
            fixed_pitch_rad = jnp.where(
                jnp.isnan(jnp.asarray(params.render_fixed_pitch_deg, jnp.float32)),
                center_rad,
                jnp.deg2rad(jnp.asarray(params.render_fixed_pitch_deg, jnp.float32)),
            )
            # 2) 若指定了固定桶，则用固定桶中心角；否则沿用当前桶中心角
            has_fixed_bin = params.render_fixed_bin_idx >= 0
            fixed_bin_center_deg = 0.5 * (
                params.render_fixed_bin_idx * params.pitch_bin_deg
                + (params.render_fixed_bin_idx + 1) * params.pitch_bin_deg
            )
            fixed_bin_center_rad = jnp.deg2rad(jnp.asarray(fixed_bin_center_deg, jnp.float32))
            fixed_pitch_rad = jnp.where(has_fixed_bin, fixed_bin_center_rad, fixed_pitch_rad)

            # 与上一目标同号，避免正负号频繁翻转（0 视作正）
            sign_prev = jnp.sign(state.target_pitch + 1e-6)
            sign_prev = jnp.where(sign_prev == 0.0, 1.0, sign_prev)
            new_tgt_pitch = fixed_pitch_rad * sign_prev

            # 目标航向保持不变（也可选择轻扰动）
            new_tgt_heading = state.target_heading

            # 组临时 state 用于计算奖励分量
            state2 = state.replace(
                target_heading=new_tgt_heading,
                target_pitch=new_tgt_pitch,
                target_vt=state.target_vt,
                success=False,
            )

            # 运动学/气动监控（与原路径一致）
            ps_cur = state2.plane_state
            alt_ft_cur = ps_cur.altitude / 0.3048
            vt_ft_cur  = jnp.clip(ps_cur.vt / 0.3048, 0.1, 1e6)
            _, qbar_cur, _ = atmos(alt_ft_cur, vt_ft_cur)

            # alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
            # vt_ref_ft  = params.max_vt / 0.3048
            # _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
            # qbar_norm_cur = jnp.clip(qbar_cur / (qbar_ref + 1e-6), 0.0, 2.0)

            alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
            vt_ref_ft  = getattr(params, "qbar_ref_vt", 360.0) / 0.3048
            _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
            qbar_norm_cur = jnp.clip(qbar_cur / (qbar_ref + 1e-6), 0.0, 2.0)

            # 奖励分量（与原路径一致，只是用 state2）
            comps_hp  = jax.vmap(lambda aid: self.reward_functions[0](state2, params, aid))(jnp.arange(B))
            comps_alt = jax.vmap(lambda aid: self.reward_functions[1](state2, params, aid))(jnp.arange(B))
            comps_qb  = jax.vmap(lambda aid: self.reward_functions[2](state2, params, aid))(jnp.arange(B))
            comps_nz  = jax.vmap(lambda aid: self.reward_functions[3](state2, params, aid))(jnp.arange(B))
            comps_en  = jax.vmap(lambda aid: self.reward_functions[4](state2, params, aid))(jnp.arange(B))

            # info：课程窗口/进度（沿用当前桶与累计计数，不做新增累计）
            info["pitch_bin_lo_deg"]   = lo_deg.astype(jnp.float32)
            info["pitch_bin_hi_deg"]   = hi_deg.astype(jnp.float32)

            # 计算当前桶成功率（累计）
            t_cur  = state.bin_trials[jnp.arange(B, dtype=jnp.int32), cur_bin] 
            s_cur  = state.bin_successes[jnp.arange(B, dtype=jnp.int32), cur_bin]
            rate_cur = s_cur.astype(jnp.float32) / jnp.maximum(t_cur, 1)

            info["pitch_bin_rate_cur"] = rate_cur.astype(jnp.float32)
            info["pitch_current_bin"]  = state.current_bin_idx.astype(jnp.float32)
            info["curriculum_max_bin"] = state.curriculum_max_bin.astype(jnp.float32)

            # 切换统计（全 0）
            info["heading_turn_counts"] = state.heading_turn_counts
            info["did_switch"]      = jnp.zeros((B,), dtype=jnp.float32)
            info["switch_success"]  = jnp.zeros((B,), dtype=jnp.float32)
            info["switch_timeout"]  = jnp.zeros((B,), dtype=jnp.float32)

            # 奖励分量
            info["reward_heading_pitch"]    = jnp.asarray(comps_hp,  jnp.float32)
            info["reward_altitude"]         = jnp.asarray(comps_alt, jnp.float32)
            info["reward_low_qbar_penalty"] = jnp.asarray(comps_qb,  jnp.float32)
            info["reward_nz_soft_penalty"]  = jnp.asarray(comps_nz,  jnp.float32)
            info["reward_energy_track"]     = jnp.asarray(comps_en,  jnp.float32)

            # 安全门监控
            info["qbar_norm_cur"]     = jnp.asarray(qbar_norm_cur, jnp.float32)
            info["promote_qbar_gate"] = (qbar_norm_cur >= jnp.asarray(params.promote_min_qbar, jnp.float32)).astype(jnp.float32)

            info["promote_at_switch"] = (promote.astype(jnp.float32) * do_switch.astype(jnp.float32))
            info["bin_trials_cur"]    = t_cur.astype(jnp.float32)
            info["bin_success_cur"]   = s_cur.astype(jnp.float32)

            # 当前桶奖励均值
            mask_cur = (state.current_bin_idx == cur_bin).astype(jnp.float32)  # 逐环境匹配
            hp = info["reward_heading_pitch"]  # (B,)
            info["reward_heading_pitch_cur_bin"] = (hp * mask_cur).sum() / (mask_cur.sum() + 1e-8)

            return state2, info
        # ===================== [NEW] 渲染固定模式 END =====================

        # === 气动/能量：用于安全门、计次与 info ===
        ps_cur = state.plane_state
        alt_ft_cur = ps_cur.altitude / 0.3048
        vt_ft_cur  = jnp.clip(ps_cur.vt / 0.3048, 0.1, 1e6)
        _, qbar_cur, _ = atmos(alt_ft_cur, vt_ft_cur)
        alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
        # vt_ref_ft  = params.max_vt / 0.3048
        vt_ref_ft  = getattr(params, "qbar_ref_vt", 360.0) / 0.3048
        _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
        qbar_norm_cur = jnp.clip(qbar_cur / (qbar_ref + 1e-6), 0.0, 2.0)

        # g = 9.81
        # spec_energy = g * ps_cur.altitude + 0.5 * jnp.clip(ps_cur.vt, 0.0, 1e4) ** 2
        # e_ref = g * params.max_altitude + 0.5 * (params.max_vt ** 2)
        # energy_norm_cur = jnp.clip(spec_energy / (e_ref + 1e-6), 0.0, 1.5)

        g = 9.81
        spec_energy = g * ps_cur.altitude + 0.5 * jnp.clip(ps_cur.vt, 0.0, 1e4) ** 2
        e_ref = g * getattr(params, "energy_ref_alt", params.max_altitude) + 0.5 * (getattr(params, "energy_ref_vt", 360.0) ** 2)
        energy_norm_cur = jnp.clip(spec_energy / (e_ref + 1e-6), 0.0, 1.5)

        # === 气动/能量：用于安全门、计次与 info ===

        # (5) 本段是否切换：success 或 驻留超时（把失败也记入试次）
        steps_per_sec = jnp.maximum(params.sim_freq // params.agent_interaction_steps, 1)
        dwell_steps = (state.time - state.last_check_time).astype(jnp.int32)
        # force_steps = 8 * steps_per_sec     # 最长驻留 8s
        force_steps   = 4 * steps_per_sec   # 原来是 8 * steps_per_sec  把强制切换窗口从 8s 改 4s（升桶机会翻倍）
        force_switch = dwell_steps >= force_steps
        # do_switch = state.success | force_switch
        crashed_now = (qbar_norm_cur < jnp.asarray(params.qbar_crash_frac, jnp.float32))
        success_switch = state.success
        # do_switch = state.success | force_switch | crashed_now
        do_switch = success_switch | force_switch | crashed_now

        # (6) 统计累加到“上一段所属桶”（crash 记试次不记成功）
        bin_trials = state.bin_trials
        bin_successes = state.bin_successes
        rows = jnp.arange(B, dtype=jnp.int32)
        incr_trials = jnp.zeros_like(bin_trials).at[rows, prev_bin].add(do_switch.astype(jnp.int32))
        incr_succs  = jnp.zeros_like(bin_successes).at[rows, prev_bin].add(was_success)
        bin_trials = bin_trials + incr_trials
        bin_successes = bin_successes + incr_succs

        # 运动学/气动附加维度（用于 promote 安全门）
        ps_cur = state.plane_state
        alt_ft_cur = ps_cur.altitude / 0.3048
        vt_ft_cur  = jnp.clip(ps_cur.vt / 0.3048, 0.1, 1e6)
        _, qbar_cur, _ = atmos(alt_ft_cur, vt_ft_cur)
        alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
        # vt_ref_ft  = params.max_vt / 0.3048
        vt_ref_ft  = getattr(params, "qbar_ref_vt", 360.0) / 0.3048
        _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
        qbar_norm_cur = jnp.clip(qbar_cur / (qbar_ref + 1e-6), 0.0, 2.0)

        # (7) 桶推进：只看“当前桶”的成功率与试次 + 安全门（动压门 + 能量门）
        t_cur  = bin_trials[rows, cur_bin]
        s_cur  = bin_successes[rows, cur_bin]
        rate_cur = s_cur.astype(jnp.float32) / jnp.maximum(t_cur, 1)
        promote = (rate_cur >= jnp.asarray(params.curriculum_target_rate, jnp.float32)) & \
                  (t_cur    >= jnp.asarray(params.curriculum_min_trials,  jnp.int32))
        # 安全门（当前步动压需达标）
        promote = promote & (qbar_norm_cur   >= jnp.asarray(params.promote_min_qbar, jnp.float32))
        promote = promote & (energy_norm_cur >= jnp.asarray(params.promote_min_energy, jnp.float32))

        # next_bin = jnp.minimum(cur_bin + promote.astype(jnp.int32), num_bins - 1)

        # # 替换为（只在切换步且满足 promote 时升桶）：
        # next_bin = jnp.where(
        #     do_switch & promote,
        #     jnp.minimum(cur_bin + 1, num_bins - 1),
        #     cur_bin
        # )

        # 仅“成功步”触发升桶；超时/坠毁只累计试次不升桶
        next_bin = jnp.where(
            success_switch & promote,
            jnp.minimum(cur_bin + 1, num_bins - 1),
            cur_bin
        )

        new_cur_bin = next_bin
        new_max_bin = jnp.maximum(state.curriculum_max_bin, new_cur_bin)

        # (8) 切换时才更新目标 / last_check_time / 统计计数
        new_tgt_heading = jnp.where(do_switch, sampled_heading, state.target_heading)
        new_tgt_pitch   = jnp.where(do_switch, sampled_pitch,  state.target_pitch)
        new_tgt_vt      = jnp.where(do_switch, sampled_vt,     state.target_vt)

        state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,
            target_heading=new_tgt_heading,
            target_pitch=new_tgt_pitch,
            target_vt=new_tgt_vt,
            last_check_time=jnp.where(do_switch, jnp.broadcast_to(state.time, (B,)), state.last_check_time),
            heading_turn_counts=state.heading_turn_counts + do_switch.astype(jnp.int32),
            current_bin_idx=new_cur_bin,
            curriculum_max_bin=new_max_bin,
            bin_trials=bin_trials,
            bin_successes=bin_successes,
        )

        # (9) info 输出（训练/可视化）
        info["pitch_bin_lo_deg"]   = lo_deg.astype(jnp.float32)
        info["pitch_bin_hi_deg"]   = hi_deg.astype(jnp.float32)
        info["pitch_bin_rate_cur"] = rate_cur.astype(jnp.float32)
        info["pitch_current_bin"]  = state.current_bin_idx.astype(jnp.float32)
        info["curriculum_max_bin"] = state.curriculum_max_bin.astype(jnp.float32)
        info["curr_t_cur"]         = t_cur.astype(jnp.float32)  # 新增：当前桶尝试次数
        
        info["heading_turn_counts"] = state.heading_turn_counts

        # 新增：本步是否切换、因成功/因超时（基于上一段统计）
        info["did_switch"]      = do_switch.astype(jnp.float32)
        info["switch_success"]  = (do_switch & (was_success.astype(jnp.bool_))).astype(jnp.float32)
        info["switch_timeout"]  = (do_switch & (~was_success.astype(jnp.bool_))).astype(jnp.float32)

        # 奖励分量（沿用原实现）
        comps_hp  = jax.vmap(lambda aid: self.reward_functions[0](state, params, aid))(jnp.arange(B))
        comps_alt = jax.vmap(lambda aid: self.reward_functions[1](state, params, aid))(jnp.arange(B))
        comps_qb  = jax.vmap(lambda aid: self.reward_functions[2](state, params, aid))(jnp.arange(B))
        comps_nz  = jax.vmap(lambda aid: self.reward_functions[3](state, params, aid))(jnp.arange(B))
        comps_en  = jax.vmap(lambda aid: self.reward_functions[4](state, params, aid))(jnp.arange(B))
        info["reward_heading_pitch"]    = jnp.asarray(comps_hp,  jnp.float32)
        info["reward_altitude"]         = jnp.asarray(comps_alt, jnp.float32)
        info["reward_low_qbar_penalty"] = jnp.asarray(comps_qb,  jnp.float32)
        info["reward_nz_soft_penalty"]  = jnp.asarray(comps_nz,  jnp.float32)
        info["reward_energy_track"]     = jnp.asarray(comps_en,  jnp.float32)

        # 安全门相关观测
        info["qbar_norm_cur"]     = jnp.asarray(qbar_norm_cur, jnp.float32)
        info["energy_norm_cur"]   = jnp.asarray(energy_norm_cur, jnp.float32)
        info["promote_qbar_gate"] = (qbar_norm_cur >= jnp.asarray(params.promote_min_qbar, jnp.float32)).astype(jnp.float32)

        # 新增：与渲染分支保持一致，防止 KeyError
        info["promote_at_switch"] = (promote.astype(jnp.float32) * do_switch.astype(jnp.float32))
        info["bin_trials_cur"]    = t_cur.astype(jnp.float32)
        info["bin_success_cur"]   = s_cur.astype(jnp.float32)

        # 新增：当前桶奖励均值（用本路径计算的 comps_hp）
        mask_cur = (state.current_bin_idx == cur_bin).astype(jnp.float32)  # 逐环境匹配
        hp = comps_hp  # jax.vmap 后的 (B,) 或 (B,) 形状
        info["reward_heading_pitch_cur_bin"] = (hp * mask_cur).sum() / (mask_cur.sum() + 1e-8)

        # 新增：仅“成功步”为条件的 promote 命中率与门通过率
        qbar_gate = (qbar_norm_cur   >= jnp.asarray(params.promote_min_qbar,   jnp.float32)).astype(jnp.float32)
        ener_gate = (energy_norm_cur >= jnp.asarray(params.promote_min_energy, jnp.float32)).astype(jnp.float32)
        succ_mask = success_switch.astype(jnp.float32)
        info["promote_on_success"]       = (promote.astype(jnp.float32) * succ_mask)
        info["qbar_gate_on_success"]     = (qbar_gate * succ_mask)
        info["energy_gate_on_success"]   = (ener_gate * succ_mask)

        # 新增：就绪度（达成 rate_cur/t_cur 的样本占比）
        ready = ((rate_cur >= jnp.asarray(params.curriculum_target_rate, jnp.float32)) &
                (t_cur    >= jnp.asarray(params.curriculum_min_trials,  jnp.int32))).astype(jnp.float32)
        info["ready_cur_bin"] = ready

        return state, info

    # ---------- 观测：22维 ----------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: Pitch_Curriculum_TaskState, params: Pitch_Curriculum_TaskParams) -> Dict[AgentName, chex.Array]:
        """
        22维观测（列向量拼接，返回给每个智能体一列）：
          0: norm_delta_heading = wrap_PI(yaw - target_heading)
          1: norm_delta_pitch   = wrap_PI(pitch - target_pitch)
          2: norm_delta_vt      = (vt - target_vt)/340
          3: norm_altitude      = altitude/5000
          4: norm_vt            = vt/340
          5-8:  roll_sin, roll_cos, pitch_sin, pitch_cos
          9-12: alpha_sin, alpha_cos, beta_sin, beta_cos
          13-15: P, Q, R
          16: az（过载，g）
          17: mach
          18: qbar_norm（动压相对参考值，见下）
          19: energy_norm（比能相对参考值，见下）
          20-21: gamma_sin, gamma_cos（航迹角）
        """
        ps = state.plane_state
        altitude = ps.altitude
        roll, pitch, yaw = ps.roll, ps.pitch, ps.yaw
        vt = ps.vt
        alpha, beta = ps.alpha, ps.beta
        P, Q, R = ps.P, ps.Q, ps.R

        # 一组归一化差
        norm_delta_heading = wrap_PI(yaw - state.target_heading)
        norm_delta_pitch   = wrap_PI(pitch - state.target_pitch)
        norm_delta_vt      = (vt - state.target_vt) / 340.0
        norm_altitude      = altitude / 5000.0
        norm_vt            = vt / 340.0

        # 三角/角速度
        roll_sin, roll_cos = jnp.sin(roll), jnp.cos(roll)
        pitch_sin, pitch_cos = jnp.sin(pitch), jnp.cos(pitch)
        alpha_sin, alpha_cos = jnp.sin(alpha), jnp.cos(alpha)
        beta_sin,  beta_cos  = jnp.sin(beta),  jnp.cos(beta)

        # 运动学/气动附加维度
        az = ps.az  # 过载（单位 g）
        alt_ft = altitude / 0.3048
        vt_ft  = jnp.clip(vt / 0.3048, 0.1, 1e6)
        mach, qbar, _ = atmos(alt_ft, vt_ft)

        # # 动压归一：相对于“中位高度 + 最大速度”的参考动压
        # alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
        # vt_ref_ft  = params.max_vt / 0.3048
        # _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
        # qbar_norm = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 2.0)

        # 动压归一：相对于“中位高度 + 参考速度”的参考动压
        alt_mid_ft = ((params.min_altitude + params.max_altitude) * 0.5) / 0.3048
        vt_ref_ft  = getattr(params, "qbar_ref_vt", 360.0) / 0.3048
        _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)
        qbar_norm = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 2.0)

        # 比能归一
        # g = 9.81
        # spec_energy = g * altitude + 0.5 * jnp.clip(vt, 0.0, 1e4) ** 2
        # e_ref = g * params.max_altitude + 0.5 * (params.max_vt ** 2)
        # energy_norm = jnp.clip(spec_energy / (e_ref + 1e-6), 0.0, 1.5)

        g = 9.81
        spec_energy = g * altitude + 0.5 * jnp.clip(vt, 0.0, 1e4) ** 2
        e_ref = g * getattr(params, "energy_ref_alt", params.max_altitude) \
            + 0.5 * (getattr(params, "energy_ref_vt", 360.0) ** 2)
        energy_norm = jnp.clip(spec_energy / (e_ref + 1e-6), 0.0, 1.5)

        # 航迹角 gamma（水平速度向量与速度矢量夹角）
        vx, vy, vz = ps.vel_x, ps.vel_y, ps.vel_z
        vh = jnp.sqrt(jnp.maximum(vx * vx + vy * vy, 1e-6))
        gamma = jnp.arctan2(-vz, vh)  # 向上为正
        gamma_sin, gamma_cos = jnp.sin(gamma), jnp.cos(gamma)

        # 拼观测
        obs = jnp.vstack((
            norm_delta_heading, norm_delta_pitch, norm_delta_vt,
            norm_altitude, norm_vt,
            roll_sin, roll_cos, pitch_sin, pitch_cos,
            alpha_sin, alpha_cos, beta_sin, beta_cos,
            P, Q, R,
            az, mach, qbar_norm, energy_norm, gamma_sin, gamma_cos
        ))
        # NaN/Inf 保护 + 合理夹限
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        low  = jnp.array([-jnp.pi, -jnp.pi, -2.0, 0.0, 0.0,
                          -1,-1,-1,-1, -1,-1,-1,-1,
                          -10,-10,-10,   0.0, 0.0, 0.0, 0.0, -1.0, -1.0]).reshape(-1,1)
        high = jnp.array([ jnp.pi,  jnp.pi,  2.0, 5.0, 2.0,
                           1, 1, 1, 1,  1, 1, 1, 1,
                           10, 10, 10,  12.0, 3.0, 2.0, 2.0,  1.0,  1.0]).reshape(-1,1)
        obs = jnp.clip(obs, low, high)
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}