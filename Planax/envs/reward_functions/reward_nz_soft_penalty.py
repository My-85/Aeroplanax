import functools
import jax
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnums=(2,))
def reward_nz_soft_penalty(state, params, agent_id: int, scale: float = 1.0):
    """
    过载软惩罚（饱和+限幅）:
      - 超过 nz_limit 后按二次惩罚；
      - nz 在 nz_hard_cap 处硬封顶（数值保险）；
      - 单步惩罚绝对值不超过 r_nz_clip。
    """
    nz = jnp.abs(jnp.nan_to_num(state.plane_state.az[agent_id], nan=0.0))
    nz_limit    = getattr(params, "nz_limit", 9.0)
    nz_hard_cap = getattr(params, "nz_hard_cap", 15.0)
    # coef        = getattr(params, "r_nz_coef", 0.005)
    coef        = getattr(params, "r_nz_coef", 0.02)  # 增加惩罚强度
    r_clip      = getattr(params, "r_nz_clip", 3.0)

    # 硬封顶，避免异常尖峰把 loss 带爆
    nz = jnp.clip(nz, 0.0, nz_hard_cap)

    over = jnp.clip(nz - nz_limit, 0.0)        # 仅对超阈部分惩罚
    raw_pen = -coef * (over * over)            # 二次软惩罚
    reward = jnp.clip(raw_pen, -r_clip, 0.0)   # 单步最大惩罚限幅

    return jnp.nan_to_num(scale * reward, nan=0.0, posinf=0.0, neginf=0.0)