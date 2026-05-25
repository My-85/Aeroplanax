"""
Path utilities for 3D trajectory following.

Key design choice: use local-segment projection + path index
rather than nearest-point-on-whole-trajectory, to avoid phase
loss on closed trajectories (circles, loops).
"""

import numpy as np
from typing import Tuple, Optional


def arc_length(waypoints: np.ndarray) -> np.ndarray:
    """Cumulative arc length along waypoint sequence (N, 3)."""
    diffs = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def nearest_segment(point: np.ndarray, waypoints: np.ndarray,
                    current_idx: int = 0, search_window: int = 5) -> Tuple[int, float, np.ndarray]:
    """
    Find nearest point on local segments near current_idx.

    Returns: (segment_start_idx, t_in_[0,1], projected_point)
    """
    n = len(waypoints)
    start = max(0, current_idx)
    end = min(n - 1, current_idx + search_window)
    best_dist = float('inf')
    best_idx = start
    best_t = 0.0
    best_proj = waypoints[start].copy()

    for i in range(start, end):
        a, b = waypoints[i], waypoints[i + 1]
        seg = b - a
        seg_len_sq = np.dot(seg, seg)
        if seg_len_sq < 1e-9:
            proj = a.copy()
            t = 0.0
        else:
            t = np.clip(np.dot(point - a, seg) / seg_len_sq, 0.0, 1.0)
            proj = a + t * seg
        dist = np.linalg.norm(point - proj)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            best_t = t
            best_proj = proj

    return best_idx, best_t, best_proj


def path_progress(north: float, east: float, alt: float,
                  waypoints: np.ndarray, arc: np.ndarray,
                  current_idx: int, lookahead_idx: int = 2) -> dict:
    """
    Compute tracking state at current position.

    Returns dict with:
      - cross_track_error: perpendicular distance to local segment
      - along_track_pos:   cumulative arc length at projection
      - tangent:           local path tangent unit vector
      - normal:            local path normal (in horizontal plane)
      - curvature_proxy:   angle between consecutive segments
      - lookahead_point:   waypoint lookahead_idx steps ahead
      - wp_idx:            current segment index
      - dist_to_lookahead: 3D distance to lookahead point
    """
    point = np.array([north, east, alt])
    idx, t, proj = nearest_segment(point, waypoints, current_idx)

    # Cross-track error
    cross_track = np.linalg.norm(point - proj)

    # Along-track position
    along_track = arc[idx] + t * (arc[idx + 1] - arc[idx]) if idx + 1 < len(arc) else arc[-1]

    # Tangent at projection point
    if idx + 1 < len(waypoints):
        tangent = waypoints[idx + 1] - waypoints[idx]
        tangent_len = np.linalg.norm(tangent)
        tangent = tangent / max(tangent_len, 1e-9)
    else:
        tangent = np.array([1.0, 0.0, 0.0])

    # Normal (horizontal plane)
    normal_h = np.array([-tangent[1], tangent[0], 0.0])
    normal_h_len = np.linalg.norm(normal_h)
    normal_h = normal_h / max(normal_h_len, 1e-9)

    # Curvature proxy: angle between this and next segment
    curvature_proxy = 0.0
    if idx + 2 < len(waypoints):
        t1 = waypoints[idx + 1] - waypoints[idx]
        t2 = waypoints[idx + 2] - waypoints[idx + 1]
        t1 = t1 / max(np.linalg.norm(t1), 1e-9)
        t2 = t2 / max(np.linalg.norm(t2), 1e-9)
        curvature_proxy = np.arccos(np.clip(np.dot(t1, t2), -1.0, 1.0))

    # Lookahead point
    la_idx = min(idx + lookahead_idx, len(waypoints) - 1)
    lookahead = waypoints[la_idx]
    dist_to_lookahead = np.linalg.norm(point - lookahead)

    return {
        "cross_track_error": cross_track,
        "along_track_pos": along_track,
        "tangent": tangent,
        "normal_h": normal_h,
        "curvature_proxy": curvature_proxy,
        "lookahead_point": lookahead,
        "wp_idx": idx,
        "dist_to_lookahead": dist_to_lookahead,
        "projected_point": proj,
    }


def cross_track_error_2d(north: float, east: float,
                         wp_a: np.ndarray, wp_b: np.ndarray) -> Tuple[float, float]:
    """Cross-track error from line segment AB (horizontal only)."""
    seg = wp_b[:2] - wp_a[:2]
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-9:
        return np.linalg.norm(np.array([north, east]) - wp_a[:2]), 0.0
    t = np.clip(np.dot(np.array([north, east]) - wp_a[:2], seg) / seg_len_sq, 0.0, 1.0)
    proj = wp_a[:2] + t * seg
    return np.linalg.norm(np.array([north, east]) - proj), t


def tangent_alignment(velocity_ned: np.ndarray, tangent: np.ndarray) -> float:
    """Cosine similarity between velocity direction and path tangent."""
    v_hat = velocity_ned / max(np.linalg.norm(velocity_ned), 1e-9)
    return np.dot(v_hat, tangent)


def shift_trajectory_to_aircraft(waypoints: np.ndarray,
                                  aircraft_n: float, aircraft_e: float,
                                  aircraft_alt: float) -> np.ndarray:
    """Translate waypoints so first WP starts at aircraft position."""
    offset = np.array([aircraft_n, aircraft_e, aircraft_alt]) - waypoints[0]
    return waypoints + offset


def check_waypoint_reached(point: np.ndarray, waypoint: np.ndarray,
                            radius: float) -> bool:
    return np.linalg.norm(point - waypoint) < radius


def body_frame_error(north_err: float, east_err: float, alt_err: float,
                     yaw: float) -> np.ndarray:
    """Convert ENU position error to body-frame error."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    forward_err =  north_err * cy + east_err * sy
    right_err   = -north_err * sy + east_err * cy
    return np.array([forward_err, right_err, alt_err])


def compute_true_cte(point: np.ndarray, waypoints: np.ndarray,
                     current_idx: int = 0, search_window: int = 10) -> float:
    """
    True cross-track error: perpendicular distance from aircraft
    to the closest projection on the reference trajectory.

    Returns: cross_track_error (meters, always >= 0)
    """
    n = len(waypoints)
    start = max(0, current_idx - 1)
    end = min(n - 1, current_idx + search_window)
    best_dist = float('inf')

    for i in range(start, end):
        a, b = waypoints[i], waypoints[i + 1]
        seg = b - a
        seg_len_sq = np.dot(seg, seg)
        if seg_len_sq < 1e-9:
            dist = np.linalg.norm(point - a)
        else:
            t = np.clip(np.dot(point - a, seg) / seg_len_sq, 0.0, 1.0)
            proj = a + t * seg
            dist = np.linalg.norm(point - proj)
        if dist < best_dist:
            best_dist = dist

    return float(best_dist)


def compute_cte_statistics(positions: np.ndarray, waypoints: np.ndarray,
                           wp_indices: np.ndarray = None) -> dict:
    """
    Compute CTE statistics for a recorded trajectory.

    Args:
        positions: (N, 3) array of [north, east, alt]
        waypoints: (M, 3) reference trajectory
        wp_indices: (N,) approximate segment indices (optional, for speed)

    Returns dict with mean, p50, p90, max
    """
    ctes = []
    for i in range(len(positions)):
        idx = wp_indices[i] if wp_indices is not None else 0
        ctes.append(compute_true_cte(positions[i], waypoints, int(idx), 10))
    cte_a = np.array(ctes)
    return {
        "cte_mean": float(cte_a.mean()),
        "cte_p50": float(np.percentile(cte_a, 50)),
        "cte_p90": float(np.percentile(cte_a, 90)),
        "cte_max": float(cte_a.max()),
    }
