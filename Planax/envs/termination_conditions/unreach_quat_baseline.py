# # envs/termination_conditions/unreach_heading_pitch_V.py
# # -*- coding: utf-8 -*-
# from typing import Tuple
# import jax
# import jax.numpy as jnp

# from ..aeroplanax import TEnvState, TEnvParams, AgentID
# from ..core.simulators.fighterplane.dynamics import FighterPlaneState
# from ..utils.utils import wrap_PI


# @jax.jit
# def unreach_heading_pitch_V_fn(
#     state: TEnvState,
#     params: TEnvParams,
#     agent_id: AgentID,
#     max_check_interval: float = 5.0,
#     # min_check_interval: float = 0.2, # 这里实际指的是秒（实际物理仿真里面的时间）
#     min_check_interval: float = 0.5,  # CHANGED: 1.0 → 0.5 秒.把最小检查窗口从 1.0s 降到 0.5s（响应更快）
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     是否到达当前 heading/pitch/V 目标（不终止 episode，只返回 success 触发任务切换）
#     - 竖直目标: 放宽到 pitch & vt 达标即可，检查窗口取 12 秒
#     - 非竖直目标: 要求 heading & pitch & vt 全到位，检查窗口取 max_check_interval
#     注：全程使用 JAX array 逻辑，避免 Python 标量化。
#     """
#     plane_state: FighterPlaneState = state.plane_state

#     # ------- 取每个智能体的 last_check_time（兼容标量/向量存储） -------
#     lct_raw = jnp.asarray(state.last_check_time)
#     last_check_time = lct_raw if lct_raw.shape == () else lct_raw[agent_id]

#     # ------- 当前是否竖直目标（JAX 布尔） -------
#     is_vert_vec = getattr(state, "is_vertical_target", None)
#     if is_vert_vec is None:
#         is_vertical_target = jnp.array(False, dtype=jnp.bool_)
#     else:
#         is_vertical_target = jnp.asarray(is_vert_vec[agent_id], dtype=jnp.bool_)

#     # ------- 把“秒”换成“步”，并做下限约束（JAX 流程） -------
#     # 每次 agent 决策跨度（仿真步/一次决策）
#     sim_per_decision = jnp.asarray(params.sim_freq / params.agent_interaction_steps, dtype=jnp.float32) # e.g. 50/10=5 5是hz

#     # window_sec_vert = jnp.asarray(12.0, dtype=jnp.float32)
#     # window_sec_norm = jnp.asarray(max_check_interval, dtype=jnp.float32)
#     # window_sec      = jnp.where(is_vertical_target, window_sec_vert, window_sec_norm)

#     # min_sec = jnp.asarray(min_check_interval, dtype=jnp.float32)
#     # eff_sec = jnp.maximum(min_sec, window_sec)

#     # check_steps_f   = eff_sec * sim_per_decision
#     # max_check_steps = jnp.maximum(1, jnp.round(check_steps_f).astype(jnp.int32))

#     # 只用“最小检查窗口0.2s”，避免必须等满 12s 才算成功
#     min_sec = jnp.asarray(min_check_interval, dtype=jnp.float32)
#     min_steps = jnp.maximum(1, jnp.round(min_sec * sim_per_decision)).astype(jnp.int32)

#     # ------- 是否到了评估窗口尾端（按“步”计） -------
#     t_cur  = jnp.asarray(state.time, dtype=jnp.int32)
#     t_last = jnp.asarray(last_check_time, dtype=jnp.int32)
#     elapsed_steps = t_cur - t_last
#     # mask_time = elapsed_steps >= max_check_steps  # bool
#     ready = elapsed_steps >= min_steps           # bool

#     # ------- 角度/速度误差 -------
#     yaw   = jnp.nan_to_num(plane_state.yaw[agent_id],   nan=0.0)
#     pitch = jnp.nan_to_num(plane_state.pitch[agent_id], nan=0.0)
#     vt    = jnp.nan_to_num(plane_state.vt[agent_id],    nan=0.0)

#     tgt_h = jnp.nan_to_num(state.target_heading[agent_id], nan=0.0)
#     tgt_p = jnp.nan_to_num(state.target_pitch[agent_id],   nan=0.0)
#     tgt_v = jnp.nan_to_num(state.target_vt[agent_id],      nan=0.0)

#     d_h = jnp.abs(wrap_PI(yaw   - tgt_h))
#     d_p = jnp.abs(wrap_PI(pitch - tgt_p))
#     d_v = jnp.abs(vt - tgt_v)

#     # ------- 阈值（竖直/非竖直） -------
#     head_tol_norm  = jnp.deg2rad(5.0)
#     pitch_tol_norm = jnp.deg2rad(5.0)
#     vt_tol_norm    = jnp.asarray(10.0, dtype=jnp.float32)

#     # 竖直放宽：只考 pitch & vt
#     # 说明：你原注释“heading 仍可观测，但不强制”
#     pitch_tol_vert = jnp.deg2rad(8.0)
#     vt_tol_vert    = jnp.asarray(12.0, dtype=jnp.float32)

#     # ------- 两套 success 判定，然后按 is_vertical_target 选择 -------
#     # 非竖直
#     mask_h_norm = d_h <= head_tol_norm
#     mask_p_norm = d_p <= pitch_tol_norm
#     mask_v_norm = d_v <= vt_tol_norm
#     # success_norm = mask_time & (mask_h_norm & mask_p_norm & mask_v_norm)
#     success_norm = ready & (mask_h_norm & mask_p_norm & mask_v_norm)

#     # 竖直
#     mask_p_vert = d_p <= pitch_tol_vert
#     mask_v_vert = d_v <= vt_tol_vert
#     # success_vert = mask_time & (mask_p_vert & mask_v_vert)
#     success_vert = ready & (mask_p_vert & mask_v_vert)

#     success = jnp.where(is_vertical_target, success_vert, success_norm)

#     # 本终止函数不结束 episode，仅返回 success 用于切换目标
#     done = jnp.array(False, dtype=jnp.bool_)
#     return done, success



#=====================================================================================#
# 老版本：水平桶和竖直桶的reward不做区分

from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState
import jax.numpy as jnp
from ..utils.utils import wrap_PI


def unreach_quat_baseline_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_check_interval: int = 5,
    min_check_interval: int = 0.2
) -> Tuple[bool, bool]:
    """
    检查飞机是否在限定时间内达到目标航向角、俯仰角和速度
    """
    plane_state: FighterPlaneState = state.plane_state
    yaw = plane_state.yaw[agent_id]
    altitude = plane_state.altitude[agent_id]
    vt = plane_state.vt[agent_id]
    check_time = state.time - state.last_check_time
    # 判断时间
    max_check_interval = max_check_interval * params.sim_freq / params.agent_interaction_steps # 50*50/10=250
    # min_check_interval = min_check_interval * params.sim_freq / params.agent_interaction_steps # 0.2*50/10=1
    # mask1 = check_time >= max_check_interval
    mask1 = check_time >= max_check_interval
    

    # 检查是否达到目标航向角（误差在5度以内）
    delta_heading = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading[agent_id])
    mask_heading = jnp.abs(delta_heading) <= jnp.pi / 72  # 5度

    # 检查是否达到目标俯仰角（误差在5度以内）
    delta_pitch = wrap_PI(state.plane_state.pitch[agent_id] - state.target_pitch[agent_id])
    mask_pitch = jnp.abs(delta_pitch) <= jnp.pi / 72  # 5度

    # 检查是否达到目标速度（误差在10m/s以内）
    delta_velocity = jnp.abs(state.plane_state.vt[agent_id] - state.target_vt[agent_id])
    mask_velocity = delta_velocity <= 10.0  # 10m/s

    # 在满足时间间隔的基础上，只要完成任意一个目标就算成功
    # success = mask1 & (mask_heading | mask_pitch | mask_velocity)

    # # 第一版success（较严格）
    # success = mask1 & mask_heading & mask_pitch & mask_velocity

    # 第二版success（较宽松）
    success = mask1 # 只考虑时间间隔,时间到了就换方向
    
    # 如果达到检查时间但未达到任何目标，则失败
    fail = mask1 & (~mask_heading & ~mask_pitch & ~mask_velocity)

    # 如果飞机已经死亡，则任务结束
    # done = ~state.plane_state.is_alive[agent_id]
    done = False

    return done, success 