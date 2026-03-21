# Planax/envs/termination_conditions/unreach_heading_pitch.py
# -*- coding: utf-8 -*-
from typing import Tuple
import jax
import jax.numpy as jnp

from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI

@jax.jit
def unreach_heading_pitch_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_check_interval: float = 5.0,   # 兼容接口，不使用
    min_check_interval: float = 0.5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    仅俯仰/竖直/常规三模式兼容的 success 判定：
      - pitch_only_mode=True  → 只判 pitch
      - 否则若 state.is_vertical_target 存在且为真 → 竖直阈值(宽松，仅 pitch+vt)
      - 否则 → 常规阈值(heading+pitch+vt)
    采用最小检查窗口 min_check_interval（秒）换算到“决策步”后再判定，抑制抖动。
    """
    # # last_check_time（向量/标量兼容；用 lax.cond 避免 tracer 布尔）
    # lct_raw = jnp.asarray(state.last_check_time)
    # last_check_time = jax.lax.cond(
    #     (lct_raw.shape == ()), lambda _: jnp.asarray(lct_raw, jnp.int32),
    #     lambda _: jnp.asarray(lct_raw[agent_id], jnp.int32),
    #     operand=None,
    # )
    last_check_time = jnp.asarray(state.last_check_time, jnp.int32)[agent_id]

    # 到评估窗口
    sim_per_decision = jnp.asarray(params.sim_freq / params.agent_interaction_steps, dtype=jnp.float32)
    min_steps = jnp.maximum(1, jnp.round(jnp.asarray(min_check_interval, jnp.float32) * sim_per_decision)).astype(jnp.int32)
    t_cur  = jnp.asarray(state.time, dtype=jnp.int32)
    elapsed_steps = t_cur - last_check_time
    ready = elapsed_steps >= min_steps

    # 误差（NaN保护）
    yaw   = jnp.nan_to_num(state.plane_state.yaw[agent_id],   nan=0.0)
    pitch = jnp.nan_to_num(state.plane_state.pitch[agent_id], nan=0.0)
    vt    = jnp.nan_to_num(state.plane_state.vt[agent_id],    nan=0.0)
    tgt_h = jnp.nan_to_num(state.target_heading[agent_id], nan=0.0)
    tgt_p = jnp.nan_to_num(state.target_pitch[agent_id],   nan=0.0)
    tgt_v = jnp.nan_to_num(state.target_vt[agent_id],      nan=0.0)

    d_h = jnp.abs(wrap_PI(yaw   - tgt_h))
    d_p = jnp.abs(wrap_PI(pitch - tgt_p))
    d_v = jnp.abs(vt - tgt_v)

    # 阈值
    head_tol_norm  = jnp.deg2rad(5.0)
    pitch_tol_norm = jnp.deg2rad(5.0)
    vt_tol_norm    = jnp.asarray(10.0, dtype=jnp.float32)
    pitch_tol_vert = jnp.deg2rad(8.0)
    vt_tol_vert    = jnp.asarray(12.0, dtype=jnp.float32)

    # 模式
    pitch_only_mode = jnp.asarray(getattr(params, "pitch_only_mode", False), dtype=jnp.bool_)

    # 三套 success（去掉竖直判定）
    success_norm        = ready & (d_h <= head_tol_norm) & (d_p <= pitch_tol_norm) & (d_v <= vt_tol_norm)
    success_pitch_only  = ready & (d_p <= pitch_tol_norm)

    # 选择：仅俯仰则只判俯仰；否则按常规阈值
    success = jnp.where(pitch_only_mode, success_pitch_only, success_norm)

    done = jnp.array(False, dtype=jnp.bool_)
    return done, success