import functools
import jax
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnums=(2,))
def reward_energy_track(state, params, agent_id: int, scale: float = 1.0):
    """
    能量储备惩罚：竖直目标期间若比能低于阈值惩罚，鼓励提前攒能量。
    需要在 params 中提供：
      - energy_ref_frac: float（默认 0.90，按参考比能的比例）
      - r_energy_coef: float（默认 0.05）
    参考比能 E_ref = g*max_altitude + 0.5*max_vt^2
    """
    g = 9.81
    alt = jnp.nan_to_num(state.plane_state.altitude[agent_id], nan=0.0)
    vt  = jnp.nan_to_num(state.plane_state.vt[agent_id],       nan=0.0)
    E = g * alt + 0.5 * vt * vt

    E_ref = g * getattr(params, "max_altitude", 20000.0) + 0.5 * (getattr(params, "max_vt", 360.0) ** 2)
    En = E / (E_ref + 1e-6)

    # 仅在“当前目标为竖直目标”时启用；若字段不存在则默认开启
    is_vert_all = getattr(state, "is_vertical_target", None)
    gate = (is_vert_all[agent_id].astype(jnp.float32) if is_vert_all is not None else jnp.array(1.0, jnp.float32))

    thresh = getattr(params, "energy_ref_frac", 0.90)
    coef   = getattr(params, "r_energy_coef", 0.1)

    lack = jnp.clip(thresh - En, 0.0)
    reward = -coef * lack * lack * gate

    return jnp.nan_to_num(scale * reward, nan=0.0, posinf=0.0, neginf=0.0)