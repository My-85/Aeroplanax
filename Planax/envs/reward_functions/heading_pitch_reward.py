# Planax/envs/reward_functions/heading_pitch_reward.py
# -*- coding: utf-8 -*-
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI

def heading_pitch_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """
    加权几何平均奖励：
      - heading/pitch 误差 → exp(- (e/scale)^2)
      - roll 约束（稳定性）
      - 速度项可按 params.use_vt_in_reward 关闭（课程阶段建议 False）
    输出做 [0,1] 裁剪与 NaN 保护。
    """
    roll = state.plane_state.roll[agent_id]
    pitch = state.plane_state.pitch[agent_id]
    yaw = state.plane_state.yaw[agent_id]
    vt = state.plane_state.vt[agent_id]

    # 误差
    delta_heading = wrap_PI(yaw - state.target_heading[agent_id])
    delta_pitch   = wrap_PI(pitch - state.target_pitch[agent_id])
    delta_vt      = (vt - state.target_vt[agent_id])

    # 数值保护与夹限
    delta_heading = jnp.clip(jnp.nan_to_num(delta_heading, nan=0.0), -jnp.pi, jnp.pi)
    delta_pitch   = jnp.clip(jnp.nan_to_num(delta_pitch,   nan=0.0), -jnp.pi, jnp.pi)
    delta_vt      = jnp.clip(jnp.nan_to_num(delta_vt,      nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
    roll          = jnp.clip(jnp.nan_to_num(roll, nan=0.0), -10.0, 10.0)

    # 尺度
    heading_error_scale = jnp.pi / 72  # 5°
    pitch_error_scale   = jnp.pi / 72  # 5°
    roll_error_scale    = 0.35         # ≈20°
    speed_error_scale   = 24.0         # m/s

    # 分量
    heading_r = jnp.exp(-((delta_heading / heading_error_scale) ** 2))
    pitch_r   = jnp.exp(-((delta_pitch   / pitch_error_scale)   ** 2))
    roll_r    = jnp.exp(-((roll / roll_error_scale) ** 2))
    speed_r   = jnp.exp(-((delta_vt / speed_error_scale) ** 2))

    # 课程阶段可关闭速度项
    use_vt = getattr(params, "use_vt_in_reward", True)
    speed_r = jnp.where(use_vt, speed_r, jnp.ones_like(speed_r)) # 在仅pitch课程中，关闭速度项

    # 加权几何平均
    w_heading, w_pitch, w_roll, w_speed = 0.4, 0.3, 0.1, 0.2
    reward = (heading_r**w_heading) * (pitch_r**w_pitch) * (roll_r**w_roll) * (speed_r**w_speed)

    # 归一化与掩码
    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0), 0.0, 1.0)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return jnp.nan_to_num(reward * reward_scale * mask, nan=0.0, posinf=0.0, neginf=0.0)