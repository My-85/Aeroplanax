import functools
import jax
import jax.numpy as jnp
from ..core.simulators.fighterplane.dynamics import atmos

@functools.partial(jax.jit, static_argnums=(2,))
def reward_low_qbar_penalty(state, params, agent_id: int, scale: float = 1.0):
    """
    低动压惩罚：在需要较大俯仰指令时(qbar_norm 低于阈值)给惩罚，防失速。
    需要在 params 中提供：
      - qbar_low_frac: float（阈值，默认 0.35，按参考动压的比例）
      - r_qbar_coef: float（权重，默认 0.02）
    参考动压 qbar_ref 取“(min_altitude+max_altitude)/2 + max_vt”计算。
    """
    # alt_m = jnp.nan_to_num(state.plane_state.altitude[agent_id], nan=0.0)
    # vt_m  = jnp.nan_to_num(state.plane_state.vt[agent_id],       nan=0.0)

    # 输入数值保险：同时处理 NaN 与 ±Inf
    alt_m = jnp.nan_to_num(state.plane_state.altitude[agent_id], nan=0.0, posinf=1e6, neginf=-1e6)
    vt_m  = jnp.nan_to_num(state.plane_state.vt[agent_id],       nan=0.0, posinf=1e6, neginf=0.0)

    alt_ft = alt_m / 0.3048
    vt_ft  = jnp.maximum(vt_m / 0.3048, 0.1)
    _, qbar, _ = atmos(alt_ft, vt_ft)
    qbar = jnp.nan_to_num(qbar, nan=0.0, posinf=1e6, neginf=0.0)

    # alt_mid_ft = ((getattr(params, "min_altitude", 2000.0) + getattr(params, "max_altitude", 20000.0)) * 0.5) / 0.3048
    # vt_ref_ft  = getattr(params, "max_vt", 360.0) / 0.3048
    # _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)

    alt_mid_ft = ((getattr(params, "min_altitude", 2000.0) + getattr(params, "max_altitude", 20000.0)) * 0.5) / 0.3048
    vt_ref_ft  = getattr(params, "qbar_ref_vt", getattr(params, "max_vt", 360.0)) / 0.3048
    _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)

    qbar_ref = jnp.nan_to_num(qbar_ref, nan=1.0, posinf=1e6, neginf=1.0)

    # 分母保护 + 合理范围夹限
    denom = jnp.maximum(qbar_ref, 1e-6)
    qn = jnp.clip(qbar / denom, 0.0, 10.0)

    # 仅在目标俯仰较大时启用，避免巡航期惩罚
    tgt_pitch = getattr(state, "target_pitch", jnp.array([0.0]))[agent_id]
    gate = (jnp.abs(tgt_pitch) > jnp.deg2rad(45.0)).astype(jnp.float32)

    thresh = getattr(params, "qbar_low_frac", 0.35)
    coef   = getattr(params, "r_qbar_coef", 0.02)

    lack = jnp.clip(thresh - qn, 0.0)
    reward = -coef * lack * lack * gate

    # 输出兜底：防止任何残余 NaN/Inf 泄漏
    return jnp.nan_to_num(scale * reward, nan=0.0, posinf=0.0, neginf=0.0)