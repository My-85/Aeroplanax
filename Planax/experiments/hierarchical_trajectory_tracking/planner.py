"""
Unified trajectory planner interface.

All planners take a reference trajectory and aircraft state,
return (target_heading, target_pitch, target_roll, target_vt).
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from .path_manager import PathManager
from .subgoal_generator import (
    pure_pursuit_subgoal, tangent_following_subgoal,
    pursuit_tangent_blend, tangent_plus_roll_subgoal,
)
from .target_blender import TargetBlender
from .path_utils import arc_length


@dataclass
class PlannerConfig:
    lookahead_dist: float = 1000.0
    reach_radius: float = 500.0
    blend_steps: int = 200
    target_vt: float = 250.0
    path_mode: str = "lookahead"  # "lookahead" | "waypoint"
    roll_mode: str = "zero"       # "zero" | "heading_error" | "turn_rate"
    max_roll_deg: float = 0.0
    w_pursuit: float = 0.6        # for pursuit_tangent_blend
    curvature_lookahead: bool = False  # ScheduledLookahead
    min_lookahead: float = 200.0
    max_lookahead: float = 2000.0
    energy_aware: bool = False     # EnergyAware mode
    max_pitch_rate_deg_s: float = 20.0


class BaseTrajectoryPlanner:
    """Abstract planner: trajectory + state → targets."""
    def __init__(self, config: PlannerConfig = None):
        self.cfg = config or PlannerConfig()

    def reset(self, waypoints: np.ndarray, init_heading: float, init_pitch: float,
              init_roll: float, init_vt: float):
        self.waypoints = waypoints
        self.total_arc = arc_length(waypoints)[-1]
        self.path = PathManager(waypoints, mode=self.cfg.path_mode,
                                lookahead_dist=self.cfg.lookahead_dist,
                                reach_radius=self.cfg.reach_radius)
        self.blender = TargetBlender(blend_steps=self.cfg.blend_steps)
        self.blender.reset(init_heading, init_pitch, init_roll, init_vt)

    def step(self, north: float, east: float, alt: float,
             yaw: float, pitch: float, roll: float, vt: float) -> Dict:
        path_ctx = self.path.update(north, east, alt)
        if self.path.just_reached:
            self.blender.reset(yaw, pitch, roll, vt)
        raw_h, raw_p, raw_r, raw_v = self._compute_raw_target(path_ctx, yaw, pitch, roll, vt)
        t_h, t_p, t_r, t_v = self.blender.blend(raw_h, raw_p, raw_r, raw_v, yaw, pitch, roll, vt)
        return {
            "target_heading": t_h, "target_pitch": t_p,
            "target_roll": t_r, "target_vt": t_v,
            "lookahead_dist": self.cfg.lookahead_dist,
            "blend_steps": self.cfg.blend_steps,
            "path_ctx": path_ctx,
            "just_reached": self.path.just_reached,
        }

    def _compute_raw_target(self, path_ctx, yaw, pitch, roll, vt):
        raise NotImplementedError

    def is_done(self) -> bool:
        return self.path.is_done()

    @property
    def path_progress(self):
        return self.path.path_progress


class PurePursuitPlanner(BaseTrajectoryPlanner):
    """Pure pursuit: target = direction to lookahead point. Roll = 0."""
    def _compute_raw_target(self, path_ctx, yaw, pitch, roll, vt):
        h, p, r, v, _ = pure_pursuit_subgoal(path_ctx, yaw, pitch, vt, self.cfg.target_vt)
        return h, p, 0.0, v

    def step(self, *args, **kwargs):
        # Adjust lookahead if curvature-aware
        if self.cfg.curvature_lookahead:
            ctx = kwargs.get('path_ctx', None)
            if ctx is None:
                # Need to update path first
                pass
        return super().step(*args, **kwargs)


class ScheduledLookaheadPlanner(BaseTrajectoryPlanner):
    """Lookahead scales with curvature: shorter on tight turns."""
    def _compute_raw_target(self, path_ctx, yaw, pitch, roll, vt):
        curvature = path_ctx.get("curvature_proxy", 0.0)
        if hasattr(self.path, '_current_curvature'):
            curvature = self.path._current_curvature
        # Shorter lookahead for high curvature
        if curvature > 0:
            la = np.clip(500.0 / (curvature + 0.0001),
                         self.cfg.min_lookahead, self.cfg.max_lookahead)
            self.path.lookahead_dist = la
        h, p, r, v, _ = pure_pursuit_subgoal(path_ctx, yaw, pitch, vt, self.cfg.target_vt)
        return h, p, 0.0, v


class EnergyAwarePlanner(BaseTrajectoryPlanner):
    """Conservative pitch on climbs to prevent energy loss."""
    def _compute_raw_target(self, path_ctx, yaw, pitch, roll, vt):
        tangent = path_ctx["tangent_world"]
        tangent_pitch = np.arctan2(tangent[2], np.sqrt(tangent[0]**2 + tangent[1]**2) + 1e-9)
        # Limit climb rate based on energy state
        energy = 0.5 * vt**2
        max_climb_pitch = np.radians(np.clip((energy - 15000) / 500, 5, 20))
        if tangent_pitch > 0:
            tangent_pitch = min(tangent_pitch, max_climb_pitch)
        h, p, _, v, _ = pure_pursuit_subgoal(path_ctx, yaw, pitch, vt, self.cfg.target_vt)
        # Conservative: limit pitch to not exceed energy-based cap
        p = np.clip(p, -np.radians(20), tangent_pitch)
        return h, p, 0.0, v


# Registry
PLANNER_REGISTRY = {
    "pure_pursuit": PurePursuitPlanner,
    "scheduled_lookahead": ScheduledLookaheadPlanner,
    "energy_aware": EnergyAwarePlanner,
}
