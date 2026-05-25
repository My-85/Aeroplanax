"""
PathManager v2: tracks waypoint index and path progress.
Outputs clearly-named absolute positions and relative error vectors.
"""
import numpy as np
from .path_utils import nearest_segment, arc_length


def interpolate_along_arc(waypoints: np.ndarray, arc: np.ndarray, s: float) -> np.ndarray:
    """Interpolate position at arc-length s along the waypoint path."""
    s = np.clip(s, 0.0, arc[-1])
    idx = np.searchsorted(arc, s, side='right') - 1
    idx = np.clip(idx, 0, len(waypoints) - 2)
    seg_len = arc[idx + 1] - arc[idx]
    t = (s - arc[idx]) / max(seg_len, 1e-9)
    t = np.clip(t, 0.0, 1.0)
    return waypoints[idx] + t * (waypoints[idx + 1] - waypoints[idx])


class PathManager:
    def __init__(self, waypoints: np.ndarray, mode: str = "waypoint",
                 lookahead_dist: float = 1000.0, reach_radius: float = 500.0):
        self.waypoints = waypoints
        self.n_wp = len(waypoints)
        self.mode = mode
        self.lookahead_dist = lookahead_dist
        self.reach_radius = reach_radius
        self.arc = arc_length(waypoints)
        self.current_idx = 0
        self.path_progress = 0.0
        self.wp_reached_count = 0
        self.just_reached = False
        self._started = False

    def update(self, north: float, east: float, alt: float) -> dict:
        aircraft = np.array([north, east, alt])
        self.just_reached = False
        self._started = True

        if self.mode == "waypoint":
            ctx = self._update_waypoint(aircraft)
        else:
            ctx = self._update_lookahead(aircraft)

        # Add absolute/relative fields with clear naming
        ctx["aircraft_pos"] = aircraft
        ctx["current_wp_abs"] = self.waypoints[min(self.current_idx, self.n_wp - 1)].copy()
        ctx["lookahead_wp_abs"] = ctx["lookahead_point"]
        ctx["lookahead_error_world"] = ctx["lookahead_point"] - aircraft
        ctx["current_wp_error_world"] = ctx["current_wp_abs"] - aircraft
        return ctx

    def _update_waypoint(self, aircraft: np.ndarray) -> dict:
        wp = self.waypoints[min(self.current_idx, self.n_wp - 1)]
        dist = np.linalg.norm(aircraft - wp)

        if dist < self.reach_radius and self.current_idx < self.n_wp - 1:
            self.current_idx += 1
            self.wp_reached_count += 1
            self.just_reached = True
            wp = self.waypoints[self.current_idx]

        # Lookahead: the WP `la_steps` ahead
        la_steps = max(1, min(3, self.n_wp - self.current_idx - 1))
        la_idx = min(self.current_idx + la_steps, self.n_wp - 1)
        lookahead = self.waypoints[la_idx]

        # Tangent at current segment
        if self.current_idx + 1 < self.n_wp:
            tangent = self.waypoints[self.current_idx + 1] - self.waypoints[self.current_idx]
            tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
        else:
            tangent = np.array([1.0, 0.0, 0.0])

        self.path_progress = self.arc[self.current_idx]

        return {
            "lookahead_point": lookahead,
            "tangent_world": tangent,
            "wp_idx": self.current_idx,
            "dist_to_wp": float(dist),
            "path_progress": self.path_progress,
            "just_reached": self.just_reached,
        }

    def _update_lookahead(self, aircraft: np.ndarray) -> dict:
        # Find nearest segment
        seg_idx, t, proj = nearest_segment(aircraft, self.waypoints,
                                            max(0, self.current_idx - 1), 15)
        self.current_idx = seg_idx
        self.path_progress = (self.arc[seg_idx] +
                              t * (self.arc[min(seg_idx+1, self.n_wp-1)] - self.arc[seg_idx]))

        # Lookahead point: interpolated at path_progress + lookahead_dist
        lookahead_s = self.path_progress + self.lookahead_dist
        lookahead = interpolate_along_arc(self.waypoints, self.arc, lookahead_s)

        # Tangent at projection
        if seg_idx + 1 < self.n_wp:
            tangent = self.waypoints[seg_idx + 1] - self.waypoints[seg_idx]
            tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
        else:
            tangent = np.array([1.0, 0.0, 0.0])

        # Check if last WP reached
        end_wp = self.waypoints[-1]
        dist_to_end = np.linalg.norm(aircraft - end_wp)
        if dist_to_end < self.reach_radius:
            self.wp_reached_count += 1
            self.just_reached = True

        return {
            "lookahead_point": lookahead,
            "tangent_world": tangent,
            "wp_idx": seg_idx,
            "dist_to_wp": float(np.linalg.norm(aircraft - lookahead)),
            "path_progress": self.path_progress,
            "just_reached": self.just_reached,
        }

    def is_done(self) -> bool:
        if not self._started:
            return False
        if self.mode == "waypoint":
            return self.current_idx >= self.n_wp - 1 and self.wp_reached_count > 0
        else:
            return self.path_progress >= self.arc[-1] - self.reach_radius and self.wp_reached_count > 0
