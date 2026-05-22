import jax.numpy as jnp


def reward_action_smoothness(state, params, agent_id: int, scale: float = 1.0):
    """
    Penalise large single-step changes in control surface commands.

    Encourages smooth, coordinated manoeuvres instead of jerky, G-spiking
    control inputs. Reads current and previous control_state from the env.

    Penalty = -scale * sum_c (|delta_c| / max_range_c)^2
    clipped to [-1.0, 0.0] per step.
    """
    cur = state.control_state
    # prev is not stored separately — we use the current plane_state's
    # stored control surface values (T, el, ail, rud, sb) which reflect
    # the command applied in the PREVIOUS step.
    prev_thr = jnp.nan_to_num(state.plane_state.T[agent_id], nan=0.0) / 19000.0
    prev_el  = jnp.nan_to_num(state.plane_state.el[agent_id], nan=0.0)
    prev_ail = jnp.nan_to_num(state.plane_state.ail[agent_id], nan=0.0)
    prev_rud = jnp.nan_to_num(state.plane_state.rud[agent_id], nan=0.0)

    cur_thr = jnp.nan_to_num(cur.throttle[agent_id], nan=0.0)
    cur_el  = jnp.nan_to_num(cur.elevator[agent_id], nan=0.0)
    cur_ail = jnp.nan_to_num(cur.aileron[agent_id], nan=0.0)
    cur_rud = jnp.nan_to_num(cur.rudder[agent_id], nan=0.0)

    d_thr = (cur_thr - prev_thr) / 1.0
    d_el  = (cur_el  - prev_el)  / 2.0
    d_ail = (cur_ail - prev_ail) / 2.0
    d_rud = (cur_rud - prev_rud) / 2.0

    raw = (d_thr ** 2 + d_el ** 2 + d_ail ** 2 + d_rud ** 2) / 4.0
    penalty = jnp.clip(-raw, -1.0, 0.0)

    return scale * jnp.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)
