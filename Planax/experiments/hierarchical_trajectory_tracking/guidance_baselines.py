"""
Hand-crafted guidance baselines for no-training validation.

All guidance functions take aircraft state and path info,
return (target_heading, target_pitch, target_roll, target_vt).
"""

import numpy as np
from typing import Tuple, Optional, Dict
from .path_utils import path_progress, tangent_alignment
from .target_interface import UpperActionConfig


# ────────────────────────────────────────────────────────────────────
# 9.1 Pure Pursuit Guidance
# ────────────────────────────────────────────────────────────────────
def pure_pursuit(north: float, east: float, alt: float,
                 vt: float, yaw: float, pitch: float, roll: float,
                 waypoints: np.ndarray, current_wp: int,
                 cruise_vt: float = 250.0,
                 heuristic_roll: bool = False,
                 heading_error_gain: float = 1.5,
                 max_roll_deg: float = 75.0) -> Tuple[float, float, float, float, Dict]:
    """
    Pure pursuit: point directly at the current waypoint.

    Returns: (target_heading, target_pitch, target_roll, target_vt, info)
    """
    wp = waypoints[min(current_wp, len(waypoints) - 1)]
    d_n = wp[0] - north
    d_e = wp[1] - east
    d_a = wp[2] - alt

    h_dist = np.sqrt(d_n**2 + d_e**2)
    target_heading = np.arctan2(d_e, d_n)
    target_pitch   = np.arctan2(d_a, max(h_dist, 1e-6))
    target_vt      = cruise_vt

    if heuristic_roll:
        hdg_err = np.arctan2(np.sin(target_heading - yaw), np.cos(target_heading - yaw))
        target_roll = np.clip(heading_error_gain * hdg_err,
                              -np.radians(max_roll_deg), np.radians(max_roll_deg))
    else:
        target_roll = 0.0

    info = {
        "guidance_type": "pure_pursuit", "dist_to_wp": np.sqrt(h_dist**2 + d_a**2),
        "heading_error": target_heading - yaw, "target_roll": target_roll,
    }
    return target_heading, target_pitch, target_roll, target_vt, info


# ────────────────────────────────────────────────────────────────────
# 9.2 Tangent-Following Guidance
# ────────────────────────────────────────────────────────────────────
def tangent_following(north: float, east: float, alt: float,
                      vt: float, yaw: float, pitch: float, roll: float,
                      waypoints: np.ndarray, arc: np.ndarray,
                      current_wp: int, lookahead: int = 2,
                      cruise_vt: float = 250.0,
                      heuristic_roll: bool = False,
                      heading_error_gain: float = 1.5,
                      max_roll_deg: float = 75.0) -> Tuple[float, float, float, float, Dict]:
    """
    Follow the local path tangent, with lookahead for curvature anticipation.
    """
    path = path_progress(north, east, alt, waypoints, arc, current_wp, lookahead)
    tangent = path["tangent"]
    lookahead_pt = path["lookahead_point"]

    # Target heading/pitch from tangent
    target_heading = np.arctan2(tangent[1], tangent[0])
    target_pitch   = np.arctan2(tangent[2], np.sqrt(tangent[0]**2 + tangent[1]**2))
    target_vt      = cruise_vt

    if heuristic_roll:
        hdg_err = np.arctan2(np.sin(target_heading - yaw), np.cos(target_heading - yaw))
        target_roll = np.clip(heading_error_gain * hdg_err,
                              -np.radians(max_roll_deg), np.radians(max_roll_deg))
    else:
        target_roll = 0.0

    info = {
        "guidance_type": "tangent_following",
        "cross_track": path["cross_track_error"],
        "curvature": path["curvature_proxy"],
        "tangent_heading": np.degrees(target_heading),
        "tangent_pitch": np.degrees(target_pitch),
        "target_roll": target_roll,
    }
    return target_heading, target_pitch, target_roll, target_vt, info


# ────────────────────────────────────────────────────────────────────
# 9.3 Tangent + Roll Guidance
# ────────────────────────────────────────────────────────────────────
def tangent_plus_roll(north: float, east: float, alt: float,
                      vt: float, yaw: float, pitch: float, roll: float,
                      waypoints: np.ndarray, arc: np.ndarray,
                      current_wp: int, lookahead: int = 2,
                      cruise_vt: float = 250.0,
                      turn_rate_gain: float = 1.0) -> Tuple[float, float, float, float, Dict]:
    """
    Tangent following with turn-rate-based roll command.

    target_roll = arctan(V * curvature / g)
    This produces a coordinated turn for the local path curvature.
    """
    G = 9.81
    path = path_progress(north, east, alt, waypoints, arc, current_wp, lookahead)
    tangent = path["tangent"]
    curvature = path["curvature_proxy"]

    # Target heading/pitch from tangent
    target_heading = np.arctan2(tangent[1], tangent[0])
    target_pitch   = np.arctan2(tangent[2], np.sqrt(tangent[0]**2 + tangent[1]**2))
    target_vt      = cruise_vt

    # Turn-rate-based roll
    if curvature > 1e-6:
        turn_rate = vt * curvature * turn_rate_gain
        target_roll = np.arctan(np.clip(vt * turn_rate / G, -3.0, 3.0))
        target_roll = np.clip(target_roll, -np.radians(75), np.radians(75))
    else:
        target_roll = 0.0

    info = {
        "guidance_type": "tangent_plus_roll",
        "cross_track": path["cross_track_error"],
        "curvature": curvature,
        "turn_rate_est": float(vt * curvature) if curvature > 1e-6 else 0.0,
        "target_roll_deg": np.degrees(target_roll),
    }
    return target_heading, target_pitch, target_roll, target_vt, info


# ────────────────────────────────────────────────────────────────────
# Guidance factory
# ────────────────────────────────────────────────────────────────────
GUIDANCE_REGISTRY = {
    "pure_pursuit": pure_pursuit,
    "tangent_following": tangent_following,
    "tangent_plus_roll": tangent_plus_roll,
}
