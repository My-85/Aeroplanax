# import jax.numpy as jnp
# from ..aeroplanax import TEnvState, TEnvParams, AgentID
# import jax


# def altitude_reward_fn(
#         state: TEnvState,
#         params: TEnvParams,
#         agent_id: AgentID,
#         reward_scale: float = 1.0,
#         Kv: float = 0.2,
#     ) -> float:
#     """
#     Reward is the sum of all the punishments.
#     """
#     safe_altitude = params.safe_altitude
#     danger_altitude = params.danger_altitude
#     ego_z = state.plane_state.altitude[agent_id] / 1000    # unit: km
#     ego_vz = state.plane_state.vel_z[agent_id] / 340    # unit: mh
#     Pv = -jnp.clip(ego_vz / Kv * (safe_altitude - ego_z) / safe_altitude, 0., 1.)
#     Pv = jax.lax.select(ego_z <= safe_altitude, Pv, 0.0)
#     PH = jnp.clip(ego_z / danger_altitude, 0., 1.) - 1. - 1.
#     PH = jax.lax.select(ego_z <= danger_altitude, PH, 0.0)
#     reward = Pv + PH
#     mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
#     return reward * mask * reward_scale

################################################################################
# 配合新版训练代码（做了reward clip）
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax


def altitude_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
        Kv: float = 0.2,
    ) -> float:
    """
    Reward is the sum of all the punishments.
    """
    safe_altitude = params.safe_altitude
    danger_altitude = params.danger_altitude
    ego_z = state.plane_state.altitude[agent_id] / 1000    # unit: km
    ego_vz = state.plane_state.vel_z[agent_id] / 340    # unit: mh

    # 输入裁剪，避免极端值放大数值。作用：把异常数值先“转成有限数”再处理，防止 NaN/Inf 继续传播。
    ego_z  = jnp.clip(jnp.nan_to_num(ego_z,  nan=0.0, posinf=1e6, neginf=-1e6),  -1e3,  1e3)
    ego_vz = jnp.clip(jnp.nan_to_num(ego_vz, nan=0.0, posinf=1e6, neginf=-1e6),  -1e3,  1e3)

    Pv = -jnp.clip(ego_vz / Kv * (safe_altitude - ego_z) / safe_altitude, 0., 1.)
    # MODIFIED: 用 jnp.where 替换 jax.lax.select 以支持形状广播和避免 AssertionError
    Pv = jnp.where(ego_z <= safe_altitude, Pv, 0.0)
    PH = jnp.clip(ego_z / danger_altitude, 0., 1.) - 1. - 1.
    # MODIFIED: 同上
    PH = jnp.where(ego_z <= danger_altitude, PH, 0.0)

    reward = Pv + PH

    reward_raw = reward
    altitude_reward_clipped = (jnp.abs(reward_raw) > 10.0).mean() # 裁剪率监控，避免过度饱和

    # 输出裁剪，避免极端值放大数值。作用：把异常数值先“转成有限数”再处理，防止 NaN/Inf 继续传播。阈值含义：[-10, 10] 是一个“经验量级”。若你的奖励天然更小，可改为 [-5, 5] 或 [-1, 1]；若经常被裁剪，说明阈值太紧或奖励设计需重标定。
    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=1e3, neginf=-1e3), -10.0, 10.0)

    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return jnp.nan_to_num(reward * mask * reward_scale, nan=0.0, posinf=0.0, neginf=0.0)