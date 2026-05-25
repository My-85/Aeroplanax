"""
SubgoalGenerator v2: uses lookahead_error_world (relative) for all computations.
All target heading/pitch derived from position ERRORS, not absolute coords.
"""

import numpy as np
from typing import Tuple, Dict

GRAVITY = 9.81


def _heading_pitch_from_error(error_vec: np.ndarray) -> Tuple[float, float]:
    """Convert world-frame position error to heading and pitch."""
    d_n, d_e, d_a = error_vec[0], error_vec[1], error_vec[2]
    h_dist = np.sqrt(d_n**2 + d_e**2) + 1e-9
    hdg = np.arctan2(d_e, d_n)
    pitch = np.arctan2(d_a, h_dist)
    return hdg, pitch


def pure_pursuit_subgoal(path_ctx: dict, yaw: float, pitch: float,
                         vt: float, cruise_vt: float = 250.0) -> Tuple[float, float, float, float, Dict]:
    """Target = direction to lookahead point (using relative error)."""
    err = path_ctx["lookahead_error_world"]
    t_hdg, t_pitch = _heading_pitch_from_error(err)
    return t_hdg, t_pitch, 0.0, cruise_vt, {"mode": "pure_pursuit_subgoal"}


def tangent_following_subgoal(path_ctx: dict, yaw: float, pitch: float,
                               vt: float, cruise_vt: float = 250.0) -> Tuple[float, float, float, float, Dict]:
    """Target heading/pitch = local tangent direction (world frame)."""
    t = path_ctx["tangent_world"]
    t_hdg, t_pitch = _heading_pitch_from_error(t)
    return t_hdg, t_pitch, 0.0, cruise_vt, {"mode": "tangent_following_subgoal"}


def pursuit_tangent_blend(path_ctx: dict, yaw: float, pitch: float,
                           vt: float, w_pursuit: float = 0.6,
                           cruise_vt: float = 250.0) -> Tuple[float, float, float, float, Dict]:
    """Blend pursuit direction (relative error) and tangent (world)."""
    err = path_ctx["lookahead_error_world"]
    t = path_ctx["tangent_world"]

    pursuit_dir = err / max(np.linalg.norm(err), 1e-9)
    tangent_dir = t / max(np.linalg.norm(t), 1e-9)

    blended = w_pursuit * pursuit_dir + (1.0 - w_pursuit) * tangent_dir
    blended = blended / max(np.linalg.norm(blended), 1e-9)

    t_hdg, t_pitch = _heading_pitch_from_error(blended)
    return t_hdg, t_pitch, 0.0, cruise_vt, {"mode": "pursuit_tangent_blend", "w_pursuit": w_pursuit}


def tangent_plus_roll_subgoal(path_ctx: dict, yaw: float, pitch: float, roll: float,
                                vt: float, cruise_vt: float = 250.0,
                                max_roll_deg: float = 75.0) -> Tuple[float, float, float, float, Dict]:
    """Tangent following with heading-error-based roll."""
    t = path_ctx["tangent_world"]
    t_hdg, t_pitch = _heading_pitch_from_error(t)

    hdg_err = np.arctan2(np.sin(t_hdg - yaw), np.cos(t_hdg - yaw))
    t_roll = np.clip(1.5 * hdg_err, -np.radians(max_roll_deg), np.radians(max_roll_deg))

    return t_hdg, t_pitch, t_roll, cruise_vt, {
        "mode": "tangent_plus_roll_subgoal",
        "heading_error_deg": np.degrees(hdg_err),
        "target_roll_deg": np.degrees(t_roll),
    }


SUBGGOAL_REGISTRY = {
    "pure_pursuit": pure_pursuit_subgoal,
    "tangent_following": tangent_following_subgoal,
    "pursuit_tangent_blend": pursuit_tangent_blend,
    "tangent_plus_roll": tangent_plus_roll_subgoal,
}
