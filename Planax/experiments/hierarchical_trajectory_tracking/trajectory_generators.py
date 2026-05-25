"""
Trajectory generators for curriculum-based 3D trajectory tracking.

Each generator returns (waypoints, metadata).
Waypoints shape: (N, 3) = [north, east, altitude].
"""

import numpy as np
from typing import Tuple, Dict, Any


def single_waypoint(origin_n: float, origin_e: float, origin_alt: float,
                    init_yaw: float, distance: float = 5000.0) -> Tuple[np.ndarray, Dict]:
    """One waypoint straight ahead."""
    wp_n = origin_n + distance * np.cos(init_yaw)
    wp_e = origin_e + distance * np.sin(init_yaw)
    waypoints = np.array([[wp_n, wp_e, origin_alt]])
    meta = {
        "name": "single_waypoint", "n_points": 1,
        "total_length_m": distance, "radius": None,
        "altitude_range": (origin_alt, origin_alt),
        "max_pitch_proxy_deg": 0.0, "max_curvature": 0.0,
        "max_heading_rate_proxy": 0.0, "max_vertical_angle": 0.0,
        "singularity_risk": "none",
    }
    return waypoints, meta


def straight_line(origin_n: float, origin_e: float, origin_alt: float,
                  init_yaw: float, length: float = 20000.0,
                  n_points: int = 10) -> Tuple[np.ndarray, Dict]:
    """Straight line along initial heading."""
    t = np.linspace(0, length, n_points)
    wp_n = origin_n + t * np.cos(init_yaw)
    wp_e = origin_e + t * np.sin(init_yaw)
    wp_a = np.full(n_points, origin_alt)
    waypoints = np.column_stack([wp_n, wp_e, wp_a])
    meta = {
        "name": "straight_line", "n_points": n_points,
        "total_length_m": length, "radius": None,
        "altitude_range": (origin_alt, origin_alt),
        "max_pitch_proxy_deg": 0.0, "max_curvature": 0.0,
        "max_heading_rate_proxy": 0.0, "max_vertical_angle": 0.0,
        "singularity_risk": "none",
    }
    return waypoints, meta


def level_circle(origin_n: float, origin_e: float, origin_alt: float,
                 init_yaw: float, radius: float = 3000.0,
                 n_points: int = 30, direction: int = 1) -> Tuple[np.ndarray, Dict]:
    """
    Horizontal circle. Centre is `radius` to the RIGHT of the aircraft.
    direction=1: right turn, -1: left turn.
    """
    centre_n = origin_n - radius * np.sin(init_yaw) * direction
    centre_e = origin_e + radius * np.cos(init_yaw) * direction
    theta0 = init_yaw - np.pi/2 * direction
    theta = np.linspace(theta0, theta0 + 2*np.pi*direction, n_points, endpoint=False)
    wp_n = centre_n + radius * np.cos(theta)
    wp_e = centre_e + radius * np.sin(theta)
    wp_a = np.full(n_points, origin_alt)
    waypoints = np.column_stack([wp_n, wp_e, wp_a])
    meta = {
        "name": f"level_circle_R{int(radius)}", "n_points": n_points,
        "total_length_m": 2 * np.pi * radius, "radius": radius,
        "altitude_range": (origin_alt, origin_alt),
        "max_pitch_proxy_deg": 0.0,
        "max_curvature": 1.0 / radius,
        "max_heading_rate_proxy": 30.0, "max_vertical_angle": 0.0,
        "singularity_risk": "none",
    }
    return waypoints, meta


def s_curve(origin_n: float, origin_e: float, origin_alt: float,
            init_yaw: float, amplitude: float = 3000.0,
            half_period: float = 10000.0, n_points: int = 60) -> Tuple[np.ndarray, Dict]:
    """Horizontal S-curve along initial heading."""
    forward = np.linspace(0, half_period * 2, n_points)
    lateral = amplitude * np.sin(np.pi * forward / half_period)
    cy, sy = np.cos(init_yaw), np.sin(init_yaw)
    wp_n = origin_n + forward * cy - lateral * sy
    wp_e = origin_e + forward * sy + lateral * cy
    wp_a = np.full(n_points, origin_alt)
    waypoints = np.column_stack([wp_n, wp_e, wp_a])
    meta = {
        "name": f"s_curve_A{int(amplitude)}", "n_points": n_points,
        "total_length_m": half_period * 2, "radius": None,
        "altitude_range": (origin_alt, origin_alt),
        "max_pitch_proxy_deg": 0.0,
        "max_curvature": amplitude * (np.pi / half_period)**2,
        "max_heading_rate_proxy": 15.0, "max_vertical_angle": 0.0,
        "singularity_risk": "none",
    }
    return waypoints, meta


def figure_eight(origin_n: float, origin_e: float, origin_alt: float,
                 init_yaw: float, radius: float = 3000.0,
                 n_points: int = 60) -> Tuple[np.ndarray, Dict]:
    """Horizontal figure-8."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = radius * np.sin(t)
    y = radius * np.sin(2*t) / 2
    cy, sy = np.cos(init_yaw), np.sin(init_yaw)
    wp_n = origin_n + x * cy - y * sy
    wp_e = origin_e + x * sy + y * cy
    wp_a = np.full(n_points, origin_alt)
    waypoints = np.column_stack([wp_n, wp_e, wp_a])
    meta = {
        "name": f"figure_eight_R{int(radius)}", "n_points": n_points,
        "total_length_m": 6 * radius, "radius": radius,
        "altitude_range": (origin_alt, origin_alt),
        "max_pitch_proxy_deg": 0.0, "max_curvature": 2.0 / radius,
        "max_heading_rate_proxy": 30.0, "max_vertical_angle": 0.0,
        "singularity_risk": "none",
    }
    return waypoints, meta


def mild_climb(origin_n: float, origin_e: float, origin_alt: float,
               init_yaw: float, length: float = 15000.0, delta_alt: float = 2000.0,
               n_points: int = 20) -> Tuple[np.ndarray, Dict]:
    """Mild climb/descent along straight line. Positive delta_alt = climb."""
    t = np.linspace(0, length, n_points)
    wp_n = origin_n + t * np.cos(init_yaw)
    wp_e = origin_e + t * np.sin(init_yaw)
    wp_a = origin_alt + t * delta_alt / length
    waypoints = np.column_stack([wp_n, wp_e, wp_a])
    gamma = np.degrees(np.arctan2(delta_alt, length))
    meta = {
        "name": f"mild_climb_{int(delta_alt)}m", "n_points": n_points,
        "total_length_m": length, "radius": None,
        "altitude_range": (min(origin_alt, origin_alt+delta_alt),
                           max(origin_alt, origin_alt+delta_alt)),
        "max_pitch_proxy_deg": gamma, "max_curvature": 0.0,
        "max_heading_rate_proxy": 0.0, "max_vertical_angle": abs(gamma),
        "singularity_risk": "none",
    }
    return waypoints, meta


def vertical_arc(origin_n: float, origin_e: float, origin_alt: float,
                 init_yaw: float, radius: float = 2000.0,
                 arc_angle_deg: float = 30.0, n_points: int = 15,
                 start_angle_deg: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """
    Vertical arc in the North-Altitude plane.
    arc_angle_deg: how many degrees of the circle to trace.
    30° = pull-up arc, 90° = quarter loop, 180° = half loop, 360° = full loop.
    start_angle_deg: 0 = bottom of loop.
    """
    centre_n = origin_n + radius
    centre_alt = origin_alt + radius
    theta = np.linspace(np.radians(start_angle_deg),
                        np.radians(start_angle_deg + arc_angle_deg),
                        n_points, endpoint=True)
    wp_n = centre_n - radius * np.cos(theta)
    wp_a = centre_alt - radius * np.sin(theta)
    wp_e = np.full(n_points, origin_e)
    waypoints = np.column_stack([wp_n, wp_e, wp_a])
    meta = {
        "name": f"vertical_arc_{int(arc_angle_deg)}deg_R{int(radius)}",
        "n_points": n_points,
        "total_length_m": radius * np.radians(arc_angle_deg),
        "radius": radius,
        "altitude_range": (wp_a.min(), wp_a.max()),
        "max_pitch_proxy_deg": arc_angle_deg,
        "max_curvature": 1.0 / radius,
        "max_heading_rate_proxy": 0.0,
        "max_vertical_angle": arc_angle_deg,
        "singularity_risk": "high" if arc_angle_deg >= 180 else
                            "medium" if arc_angle_deg >= 90 else "low",
    }
    return waypoints, meta


def vertical_loop(origin_n: float, origin_e: float, origin_alt: float,
                  init_yaw: float, radius: float = 2000.0,
                  n_points: int = 30) -> Tuple[np.ndarray, Dict]:
    """Full 360° vertical loop (convenience wrapper around vertical_pullup_arc)."""
    return vertical_pullup_arc(origin_n, origin_e, origin_alt, init_yaw,
                               radius=radius, arc_angle_deg=360.0, n_points=n_points)


# ── New vertical arc generators (correct geometry) ──

def vertical_pullup_arc(
    origin_n: float, origin_e: float, origin_alt: float,
    init_yaw: float,
    radius: float = 2000.0,
    arc_angle_deg: float = 30.0,
    n_points: int = 20,
) -> Tuple[np.ndarray, Dict]:
    """
    Pull-up vertical arc starting FROM current aircraft position.

    theta = 0:  position = origin, tangent pitch = 0 deg
    theta = arc_angle: tangent pitch ≈ arc_angle_deg
    """
    theta = np.linspace(0.0, np.radians(arc_angle_deg), n_points)
    forward = radius * np.sin(theta)
    altitude_gain = radius * (1.0 - np.cos(theta))

    cy, sy = np.cos(init_yaw), np.sin(init_yaw)
    wp_n = origin_n + forward * cy
    wp_e = origin_e + forward * sy
    wp_a = origin_alt + altitude_gain
    waypoints = np.column_stack([wp_n, wp_e, wp_a])

    arc_len = radius * np.radians(abs(arc_angle_deg))
    avg_pitch = np.degrees(np.arctan2(altitude_gain[-1], max(forward[-1], 1e-9)))

    meta = {
        "name": f"vertical_pullup_{int(arc_angle_deg)}deg_R{int(radius)}",
        "n_points": n_points,
        "total_length_m": float(arc_len),
        "radius": radius,
        "arc_angle_deg": float(arc_angle_deg),
        "start_alt": float(origin_alt),
        "end_alt": float(wp_a[-1]),
        "altitude_gain": float(wp_a[-1] - origin_alt),
        "forward_distance": float(forward[-1]),
        "altitude_range": (float(wp_a.min()), float(wp_a.max())),
        "max_tangent_pitch_deg": float(abs(arc_angle_deg)),
        "average_climb_angle_deg": float(avg_pitch),
        "max_curvature": float(1.0 / radius),
        "singularity_risk": (
            "high" if abs(arc_angle_deg) >= 90 else
            "medium" if abs(arc_angle_deg) >= 60 else
            "low"
        ),
    }
    return waypoints, meta


def vertical_pushdown_arc(
    origin_n: float, origin_e: float, origin_alt: float,
    init_yaw: float,
    radius: float = 2000.0,
    arc_angle_deg: float = 30.0,
    n_points: int = 20,
) -> Tuple[np.ndarray, Dict]:
    """
    Push-down vertical arc starting FROM current aircraft position.

    theta = 0:  position = origin, tangent pitch = 0 deg
    theta = arc_angle: tangent pitch ≈ -arc_angle_deg
    """
    theta = np.linspace(0.0, np.radians(arc_angle_deg), n_points)
    forward = radius * np.sin(theta)
    altitude_loss = radius * (1.0 - np.cos(theta))

    cy, sy = np.cos(init_yaw), np.sin(init_yaw)
    wp_n = origin_n + forward * cy
    wp_e = origin_e + forward * sy
    wp_a = origin_alt - altitude_loss
    waypoints = np.column_stack([wp_n, wp_e, wp_a])

    arc_len = radius * np.radians(abs(arc_angle_deg))
    avg_pitch = -np.degrees(np.arctan2(altitude_loss[-1], max(forward[-1], 1e-9)))

    meta = {
        "name": f"vertical_pushdown_{int(arc_angle_deg)}deg_R{int(radius)}",
        "n_points": n_points,
        "total_length_m": float(arc_len),
        "radius": radius,
        "arc_angle_deg": float(-abs(arc_angle_deg)),
        "start_alt": float(origin_alt),
        "end_alt": float(wp_a[-1]),
        "altitude_gain": float(wp_a[-1] - origin_alt),
        "forward_distance": float(forward[-1]),
        "altitude_range": (float(wp_a.min()), float(wp_a.max())),
        "max_tangent_pitch_deg": float(abs(arc_angle_deg)),
        "average_climb_angle_deg": float(avg_pitch),
        "max_curvature": float(1.0 / radius),
        "singularity_risk": (
            "high" if abs(arc_angle_deg) >= 90 else
            "medium" if abs(arc_angle_deg) >= 60 else
            "low"
        ),
    }
    return waypoints, meta


# ── New 3D complex trajectory generators ──

def helix_trajectory(
    origin_n: float, origin_e: float, origin_alt: float,
    init_yaw: float,
    radius: float = 5000.0,
    turns: float = 1.0,
    delta_alt: float = 1000.0,
    n_points: int = 120,
    direction: int = 1,
) -> Tuple[np.ndarray, Dict]:
    """
    Helix: horizontal circle + linear altitude change.

    direction=1: right turn, -1: left turn.
    delta_alt>0: climbing helix.
    """
    centre_n = origin_n + radius * np.sin(init_yaw) * direction
    centre_e = origin_e - radius * np.cos(init_yaw) * direction
    theta0 = init_yaw - np.pi/2 * direction
    theta = np.linspace(theta0, theta0 + 2*np.pi*turns*direction, n_points, endpoint=False)

    wp_n = centre_n + radius * np.cos(theta)
    wp_e = centre_e + radius * np.sin(theta)
    frac = np.linspace(0, 1, n_points)
    wp_a = origin_alt + delta_alt * frac

    waypoints = np.column_stack([wp_n, wp_e, wp_a])
    total_len = n_points * (2*np.pi*radius) * abs(turns) / n_points  # approx
    total_len = abs(turns) * 2 * np.pi * radius * np.sqrt(1 + (delta_alt/total_len)**2)
    avg_gamma = np.degrees(np.arctan2(delta_alt, abs(turns)*2*np.pi*radius))

    meta = {
        "name": f"helix_R{int(radius)}_t{turns}_dAlt{int(delta_alt)}",
        "n_points": n_points, "total_length_m": float(total_len),
        "radius": radius, "turns": turns, "delta_alt": delta_alt,
        "altitude_range": (float(wp_a.min()), float(wp_a.max())),
        "max_pitch_proxy_deg": avg_gamma, "max_curvature": float(1.0/radius),
        "average_climb_angle_deg": float(avg_gamma),
        "singularity_risk": "low",
    }
    return waypoints, meta


def climbing_figure_eight(
    origin_n: float, origin_e: float, origin_alt: float,
    init_yaw: float,
    radius: float = 5000.0,
    delta_alt: float = 1000.0,
    n_points: int = 120,
) -> Tuple[np.ndarray, Dict]:
    """Figure-eight with linear altitude change."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = radius * np.sin(t)
    y = radius * np.sin(2*t) / 2
    cy, sy = np.cos(init_yaw), np.sin(init_yaw)
    wp_n = origin_n + x * cy - y * sy
    wp_e = origin_e + x * sy + y * cy
    frac = np.linspace(0, 1, n_points)
    wp_a = origin_alt + delta_alt * frac
    waypoints = np.column_stack([wp_n, wp_e, wp_a])

    total_len = 6 * radius
    avg_gamma = np.degrees(np.arctan2(delta_alt, total_len))
    meta = {
        "name": f"climbing_fig8_R{int(radius)}_dAlt{int(delta_alt)}",
        "n_points": n_points, "total_length_m": float(total_len),
        "radius": radius, "delta_alt": delta_alt,
        "altitude_range": (float(wp_a.min()), float(wp_a.max())),
        "max_pitch_proxy_deg": avg_gamma, "max_curvature": float(2.0/radius),
        "singularity_risk": "low",
    }
    return waypoints, meta


def climbing_s_curve(
    origin_n: float, origin_e: float, origin_alt: float,
    init_yaw: float,
    amplitude: float = 3000.0,
    half_period: float = 10000.0,
    delta_alt: float = 1000.0,
    n_points: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """S-curve with linear altitude change."""
    forward = np.linspace(0, half_period * 2, n_points)
    lateral = amplitude * np.sin(np.pi * forward / half_period)
    cy, sy = np.cos(init_yaw), np.sin(init_yaw)
    wp_n = origin_n + forward * cy - lateral * sy
    wp_e = origin_e + forward * sy + lateral * cy
    frac = np.linspace(0, 1, n_points)
    wp_a = origin_alt + delta_alt * frac
    waypoints = np.column_stack([wp_n, wp_e, wp_a])

    total_len = half_period * 2
    avg_gamma = np.degrees(np.arctan2(delta_alt, total_len))
    meta = {
        "name": f"climbing_s_A{int(amplitude)}_dAlt{int(delta_alt)}",
        "n_points": n_points, "total_length_m": float(total_len),
        "amplitude": amplitude, "delta_alt": delta_alt,
        "altitude_range": (float(wp_a.min()), float(wp_a.max())),
        "max_pitch_proxy_deg": avg_gamma, "max_curvature": float(amplitude*(np.pi/half_period)**2),
        "singularity_risk": "low",
    }
    return waypoints, meta


# ── Curriculum registry ──
TRAJECTORY_REGISTRY = {
    "single_waypoint": single_waypoint,
    "straight_line": straight_line,
    "level_circle": level_circle,
    "s_curve": s_curve,
    "figure_eight": figure_eight,
    "mild_climb": mild_climb,
    "vertical_arc": vertical_arc,
    "vertical_loop": vertical_loop,
    "vertical_pullup_arc": vertical_pullup_arc,
    "vertical_pushdown_arc": vertical_pushdown_arc,
    "helix": helix_trajectory,
    "climbing_figure_eight": climbing_figure_eight,
    "climbing_s_curve": climbing_s_curve,
}
