import jax.numpy as jnp

def reward_nz_soft_penalty(state, params, agent_id: int, scale: float = 1.0):
    """
    G-load soft penalty covering all three body axes (nx, ny, nz).

    The crashed_fn hard-terminates at |nx|>10G, |ny|>10G, or |nz|>10G.
    This penalty provides a progressive warning signal before reaching that
    hard limit, so the policy learns to ease off BEFORE crashing.

    Uses the same nz_limit / r_nz_coef / r_nz_clip params, applied to
    the MAX absolute load factor across all three axes.
    """
    nx = jnp.abs(jnp.nan_to_num(state.plane_state.ax[agent_id], nan=0.0))
    ny = jnp.abs(jnp.nan_to_num(state.plane_state.ay[agent_id], nan=0.0))
    nz = jnp.abs(jnp.nan_to_num(state.plane_state.az[agent_id], nan=0.0))

    # max load factor across all three body axes
    load_max = jnp.max(jnp.array([nx, ny, nz]))

    nz_limit    = getattr(params, "nz_limit", 7.0)
    nz_hard_cap = getattr(params, "nz_hard_cap", 15.0)
    coef        = getattr(params, "r_nz_coef", 0.02)
    r_clip      = getattr(params, "r_nz_clip", 3.0)

    load_max = jnp.clip(load_max, 0.0, nz_hard_cap)

    over = jnp.clip(load_max - nz_limit, 0.0)        # only penalize beyond threshold
    raw_pen = -coef * (over * over)                    # quadratic soft penalty
    reward = jnp.clip(raw_pen, -r_clip, 0.0)           # per-step cap

    return jnp.nan_to_num(scale * reward, nan=0.0, posinf=0.0, neginf=0.0)
