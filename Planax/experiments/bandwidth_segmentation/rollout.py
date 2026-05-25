"""
Closed-loop rollout: runs a trained RL policy on waypoint-tracking task.
"""

import numpy as np
from typing import Dict, Tuple

# JAX imports done inside run() to avoid triggering GPU allocation at import time.


def run_rollout(
    waypoints: np.ndarray,
    env,
    env_params,
    network,
    net_params,
    hstate_init,
    rng_seed: int,
    max_steps: int = 3000,
    reach_radius: float = 500.0,
    cruise_vt: float = 250.0,
) -> Dict:
    """
    Run a single closed-loop waypoint-tracking episode.

    Returns dict with:
      - actual_traj: [N, 3] north, east, altitude
      - actions: [N, 4] raw discrete action indices
      - state: dict of recorded state variables
      - t: [N] time array
      - waypoints_reached: int
      - total_waypoints: int
      - steps: int
      - termination_reason: str
    """
    import jax
    import jax.numpy as jnp

    rng = jax.random.PRNGKey(rng_seed)
    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, env_params)
    hstate = hstate_init
    done_flag = jnp.zeros((1,))

    rec_north, rec_east, rec_alt = [], [], []
    rec_roll, rec_pitch, rec_yaw = [], [], []
    rec_vt, rec_alpha, rec_beta = [], [], []
    rec_thr, rec_el, rec_ail, rec_rud = [], [], [], []
    rec_t = []

    current_wp = 0
    total_reached = 0

    def _f(x):
        return float(np.asarray(x).reshape(-1)[0])

    termination_reason = "max_steps"

    for step in range(max_steps):
        ps = state.plane_state
        north = _f(ps.north); east = _f(ps.east); alt = _f(ps.altitude)
        vt = _f(ps.vt)
        yaw = _f(ps.yaw); pitch = _f(ps.pitch); roll = _f(ps.roll)
        alpha = _f(ps.alpha); beta = _f(ps.beta)

        # Waypoint target
        wp_idx = min(current_wp, len(waypoints) - 1)
        wp_n, wp_e, wp_a = waypoints[wp_idx]
        d_n, d_e, d_alt = wp_n - north, wp_e - east, wp_a - alt
        h_dist = max(float(np.sqrt(d_n**2 + d_e**2)), 1e-6)
        dist_3d = float(np.sqrt(h_dist**2 + d_alt**2))
        target_heading = float(np.arctan2(d_e, d_n))
        target_pitch = float(np.arctan2(d_alt, h_dist))

        if dist_3d < reach_radius and current_wp < len(waypoints) - 1:
            current_wp += 1
            total_reached += 1

        # Build observation with overridden targets
        state_w = state.replace(
            target_heading=jnp.array([target_heading]),
            target_pitch=jnp.array([target_pitch]),
            target_vt=jnp.array([cruise_vt]),
        )
        obs_dict_w = env._get_obs(state_w, env_params)
        obs_vec = obs_dict_w[env.agents[0]]
        obs_in = obs_vec[None, None, :]
        done_in = done_flag[None, :]

        # Policy forward (greedy)
        hstate, pi, _ = network.apply(net_params, hstate, (obs_in, done_in))
        acts = [int(p.mode()[0, 0]) for p in pi]
        action_dict = {env.agents[0]: jnp.array(acts)}

        rng, step_key = jax.random.split(rng)
        obs_dict2, state, rew_dict, done_dict, info = env.step(
            step_key, state, action_dict, env_params)

        # Record
        rec_t.append(step * 0.2)
        rec_north.append(north); rec_east.append(east); rec_alt.append(alt)
        rec_roll.append(np.degrees(roll)); rec_pitch.append(np.degrees(pitch))
        rec_yaw.append(np.degrees(yaw))
        rec_vt.append(vt); rec_alpha.append(np.degrees(alpha))
        rec_beta.append(np.degrees(beta))
        rec_thr.append(acts[0]); rec_el.append(acts[1])
        rec_ail.append(acts[2]); rec_rud.append(acts[3])

        if bool(done_dict["__all__"]):
            status = int(_f(ps.status))
            termination_reason = "crashed" if status == 2 else "timeout"
            break

    actions_arr = np.column_stack([rec_thr, rec_el, rec_ail, rec_rud])
    state_dict = {
        "altitude": rec_alt, "airspeed": rec_vt,
        "alpha": rec_alpha, "beta": rec_beta,
        "roll": rec_roll, "pitch": rec_pitch, "yaw": rec_yaw,
    }
    actual_traj = np.column_stack([rec_north, rec_east, rec_alt])

    return {
        "actual_traj": actual_traj,
        "actions": actions_arr,
        "state": state_dict,
        "t": np.array(rec_t),
        "waypoints_reached": total_reached,
        "total_waypoints": len(waypoints),
        "steps": len(rec_t),
        "termination_reason": termination_reason,
    }
