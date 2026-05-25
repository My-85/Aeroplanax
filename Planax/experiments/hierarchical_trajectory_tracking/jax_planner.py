"""
Pure JAX clone of PurePursuitPlanner + PathManager + TargetBlender.
Drop-in JAX-compatible target generator for scan/vmap/jit.
"""
import jax.numpy as jnp
from jax import lax
from functools import partial


def precompute_path(waypoints):
    """Precompute arc-length and diffs for JAX path operations."""
    wps = jnp.array(waypoints)
    diffs = jnp.diff(wps, axis=0)
    seg_lens = jnp.sqrt(jnp.sum(diffs ** 2, axis=1))
    arc = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(seg_lens)])
    total_arc = arc[-1]
    return wps, arc, total_arc


def _nearest_segment_jax(point, wps, current_idx, search_window=15):
    """JAX clone of nearest_segment: find closest point on nearby segments."""
    n = len(wps)
    start = jnp.maximum(0, current_idx)
    end = jnp.minimum(n - 1, current_idx + search_window)

    best_dist = jnp.inf
    best_idx = start
    best_t = 0.0
    best_proj = wps[start]

    # For loop over search window (small, can't vmap easily due to variable range)
    for i in range(search_window):
        seg_idx = start + i
        # Only process if within bounds
        in_range = (seg_idx < end)
        a = wps[jnp.where(in_range, seg_idx, 0)]
        b = wps[jnp.where(in_range, jnp.minimum(seg_idx + 1, n - 1), 0)]
        seg = b - a
        seg_len_sq = jnp.sum(seg * seg) + 1e-9
        t = jnp.clip(jnp.sum((point - a) * seg) / seg_len_sq, 0.0, 1.0)
        proj = a + t * seg
        dist = jnp.sqrt(jnp.sum((point - proj) ** 2))
        update = in_range & (dist < best_dist)
        best_dist = jnp.where(update, dist, best_dist)
        best_idx = jnp.where(update, seg_idx, best_idx)
        best_t = jnp.where(update, t, best_t)
        best_proj = jnp.where(update, proj, best_proj)

    return best_idx, best_t, best_proj


def _interpolate_along_arc_jax(wps, arc, s):
    """JAX clone of interpolate_along_arc."""
    s = jnp.clip(s, 0.0, arc[-1])
    idx = jnp.searchsorted(arc, s, side='right') - 1
    idx = jnp.clip(idx, 0, len(wps) - 2)
    seg_len = arc[idx + 1] - arc[idx] + 1e-9
    t = jnp.clip((s - arc[idx]) / seg_len, 0.0, 1.0)
    return wps[idx] + t * (wps[idx + 1] - wps[idx])


def jax_path_update(pos, wps, arc, current_idx, path_progress,
                    lookahead_dist, reach_radius, wp_reached_count):
    """JAX clone of PathManager._update_lookahead.

    Returns: (new_idx, new_progress, lookahead_point, tangent, is_done)
    """
    # Find nearest segment (look back 1, search ahead 15 — matches PurePursuitPlanner)
    seg_idx, t, proj = _nearest_segment_jax(pos, wps,
                                             jnp.maximum(0, current_idx - 1))

    # Update path_progress
    n_wp = len(wps)
    t_clipped = jnp.clip(t, 0.0, 1.0)
    new_progress = arc[seg_idx] + t_clipped * (
        arc[jnp.minimum(seg_idx + 1, n_wp - 1)] - arc[seg_idx])

    # Lookahead point
    lookahead_s = new_progress + lookahead_dist
    lookahead_s = jnp.clip(lookahead_s, 0.0, arc[-1])
    lookahead = _interpolate_along_arc_jax(wps, arc, lookahead_s)

    # Tangent at current segment
    a = wps[seg_idx]
    b = wps[jnp.minimum(seg_idx + 1, n_wp - 1)]
    seg = b - a
    seg_len = jnp.sqrt(jnp.sum(seg * seg)) + 1e-9
    tangent = seg / seg_len

    # Check if done
    dist_to_end = jnp.sqrt(jnp.sum((pos - wps[-1]) ** 2))
    just_reached = dist_to_end < reach_radius
    new_wp_count = wp_reached_count + jnp.where(just_reached, 1, 0)
    is_done = (new_progress >= arc[-1] - reach_radius) & (new_wp_count > 0)

    return (seg_idx, new_progress, new_wp_count, lookahead, tangent, is_done, just_reached)


def jax_compute_target(lookahead_point, pos, target_vt):
    """JAX clone of pure_pursuit_subgoal: heading/pitch from lookahead error."""
    error = lookahead_point - pos
    d_n, d_e, d_a = error[0], error[1], error[2]
    h_dist = jnp.sqrt(d_n ** 2 + d_e ** 2) + 1e-9
    hdg = jnp.arctan2(d_e, d_n)
    pitch = jnp.arctan2(d_a, h_dist)
    return hdg, pitch, 0.0, target_vt


def jax_blend_target(raw_h, raw_p, raw_r, raw_v,
                     prev_h, prev_p, prev_r, prev_v,
                     cur_yaw, cur_pitch, cur_roll, cur_vt,
                     step_counter, blend_steps=250, dt=0.2):
    """JAX clone of TargetBlender.blend (simplified: blend only, no rate limit)."""
    blend = jnp.minimum(1.0, step_counter / blend_steps)

    # Heading blend with wrap
    hdg_err = jnp.arctan2(jnp.sin(raw_h - cur_yaw), jnp.cos(raw_h - cur_yaw))
    t_h = jnp.arctan2(jnp.sin(cur_yaw + blend * hdg_err), jnp.cos(cur_yaw + blend * hdg_err))

    # Pitch (clamped)
    raw_p_clipped = jnp.clip(raw_p, -jnp.radians(89.0), jnp.radians(89.0))
    t_p = cur_pitch + blend * (raw_p_clipped - cur_pitch)

    # Roll
    roll_err = jnp.arctan2(jnp.sin(raw_r - cur_roll), jnp.cos(raw_r - cur_roll))
    t_r = jnp.arctan2(jnp.sin(cur_roll + blend * roll_err), jnp.cos(cur_roll + blend * roll_err))

    # Speed
    t_v = cur_vt + blend * (raw_v - cur_vt)

    return t_h, t_p, t_r, t_v


# ═══ Full planner step (all three combined, single JAX call) ═══
def jax_planner_step(pos, cur_yaw, cur_pitch, cur_roll, cur_vt,
                     wps, arc, path_idx, path_progress, wp_count,
                     lookahead_dist, target_vt, reach_radius,
                     step_counter, blend_steps=250):
    """Complete PurePursuitPlanner step in JAX.

    Returns:
      (target_h, target_p, target_r, target_vt,
       new_path_idx, new_path_progress, new_wp_count, is_done)
    """
    # 1. Path update
    new_idx, new_progress, new_wp_count, lookahead, tangent, is_done, just_reached = \
        jax_path_update(pos, wps, arc, path_idx, path_progress,
                        lookahead_dist, reach_radius, wp_count)

    # 2. Raw target from lookahead
    raw_h, raw_p, raw_r, raw_v = jax_compute_target(lookahead, pos, target_vt)

    # 3. Blend — step_counter+1 matches PP's self.step_counter += 1 at top of blend()
    # Reset to 0 (+1 = 1) on waypoint reached (matches PurePursuitPlanner)
    effective_step = jnp.where(just_reached, 0.0, step_counter) + 1.0
    t_h, t_p, t_r, t_v = jax_blend_target(
        raw_h, raw_p, raw_r, raw_v,
        0.0, 0.0, 0.0, 0.0,
        cur_yaw, cur_pitch, cur_roll, cur_vt,
        effective_step, blend_steps)

    return (t_h, t_p, t_r, t_v,
            new_idx, new_progress, new_wp_count, is_done)
