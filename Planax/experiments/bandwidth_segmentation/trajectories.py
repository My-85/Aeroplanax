"""
Reference trajectory generators with metadata.

Each generator returns (traj, metadata) where metadata includes:
  - total_length_m: total arc length
  - max_curvature_rad: maximum local direction change
  - max_heading_rate_proxy: max |Δψ|/Δt along trajectory
  - max_pitch_deg: max flight-path angle (≈ pitch)
  - has_altitude_change: whether altitude varies
  - singularity_risk: whether pitch exceeds Euler singularity threshold (>80°)
"""

import numpy as np
from typing import Tuple, Dict, Optional


def _arc_length(traj: np.ndarray) -> float:
    diffs = np.diff(traj, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _max_curvature(traj: np.ndarray) -> float:
    """Max local direction change between adjacent tangents."""
    diffs = np.diff(traj, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    norms = np.maximum(norms, 1e-9)
    t = diffs / norms[:, None]
    max_angle = 0.0
    for k in range(len(t) - 1):
        dot = np.clip(np.dot(t[k], t[k + 1]), -1.0, 1.0)
        angle = np.arccos(dot)
        max_angle = max(max_angle, angle)
    return float(max_angle)


def _max_pitch_angle(traj: np.ndarray) -> float:
    """Max flight-path elevation angle (≈ pitch) along trajectory."""
    diffs = np.diff(traj, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    norms = np.maximum(norms, 1e-9)
    max_pitch = 0.0
    for k in range(len(diffs)):
        pitch = np.arcsin(np.clip(diffs[k, 2] / norms[k], -1.0, 1.0))
        max_pitch = max(max_pitch, abs(float(pitch)))
    return float(np.degrees(max_pitch))


def _max_heading_rate_proxy(traj: np.ndarray, v_ref: float = 250.0) -> float:
    """Max |Δψ|/Δt proxy along trajectory."""
    diffs = np.diff(traj, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    norms = np.maximum(norms, 1e-9)
    t = diffs / norms[:, None]
    max_rate = 0.0
    for k in range(len(t) - 1):
        psi_k = np.arctan2(t[k, 1], t[k, 0])
        psi_k1 = np.arctan2(t[k + 1, 1], t[k + 1, 0])
        d_psi = abs(float(np.arctan2(np.sin(psi_k1 - psi_k), np.cos(psi_k1 - psi_k))))
        dt = norms[k] / v_ref
        rate = d_psi / max(dt, 0.01)
        max_rate = max(max_rate, rate)
    return float(np.degrees(max_rate))


def _make_metadata(traj: np.ndarray, name: str, v_ref: float = 250.0) -> Dict:
    length = _arc_length(traj)
    max_pitch = _max_pitch_angle(traj)
    return {
        "name": name,
        "n_points": len(traj),
        "total_length_m": length,
        "max_curvature_rad": _max_curvature(traj),
        "max_heading_rate_proxy_deg_s": _max_heading_rate_proxy(traj, v_ref),
        "max_pitch_deg": max_pitch,
        "has_altitude_change": max_pitch > 1.0,
        "singularity_risk": max_pitch > 80.0,
    }


# ────────────────────────────────────────────────────────────────────────────
# Trajectory generators
# ────────────────────────────────────────────────────────────────────────────

def generate_s_curve(
    origin: np.ndarray = np.array([0.0, 0.0, 5000.0]),
    amplitude: float = 5000.0,
    half_period_north: float = 20000.0,
    points_per_half: int = 50,
    n_half_periods: int = 4,
    dt: float = 0.2,
    v_ref: float = 250.0,
) -> Tuple[np.ndarray, Dict]:
    """S-curve trajectory (level flight, east-west oscillation)."""
    dn = half_period_north / points_per_half
    total_points = n_half_periods * points_per_half + 1
    waypoints = []
    for i in range(total_points):
        wp_n = origin[0] + i * dn
        wp_e = origin[1] + amplitude * np.sin(np.pi * i / points_per_half)
        wp_a = origin[2]
        waypoints.append([wp_n, wp_e, wp_a])
    traj = np.array(waypoints)
    meta = _make_metadata(traj, f"s_curve_A{amplitude:.0f}_P{points_per_half}_N{n_half_periods}", v_ref)
    return traj, meta


def generate_circle(
    origin: np.ndarray = np.array([0.0, 0.0, 5000.0]),
    radius: float = 5000.0,
    n_points: int = 500,
    altitude: float = 5000.0,
) -> Tuple[np.ndarray, Dict]:
    """Level circle trajectory."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    east = origin[1] + radius * np.cos(theta)
    north = origin[0] + radius * np.sin(theta)
    alt = np.full_like(east, altitude)
    traj = np.column_stack([north, east, alt])
    meta = _make_metadata(traj, f"circle_R{radius:.0f}")
    return traj, meta


def generate_figure_eight(
    origin: np.ndarray = np.array([0.0, 0.0, 5000.0]),
    radius: float = 4000.0,
    n_points: int = 500,
    altitude: float = 5000.0,
) -> Tuple[np.ndarray, Dict]:
    """Figure-eight trajectory (Lemniscate of Gerono), level flight."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    east = origin[1] + radius * np.cos(theta)
    north = origin[0] + radius * np.sin(theta) * np.cos(theta)
    alt = np.full_like(east, altitude)
    traj = np.column_stack([north, east, alt])
    meta = _make_metadata(traj, f"figure_eight_R{radius:.0f}")
    return traj, meta


def generate_climbing_spiral(
    origin: np.ndarray = np.array([0.0, 0.0, 5000.0]),
    radius: float = 5000.0,
    n_points: int = 500,
    alt_start: float = 5000.0,
    alt_end: float = 7000.0,
) -> Tuple[np.ndarray, Dict]:
    """Spiraling climb with gradual altitude increase."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    east = origin[1] + radius * np.cos(theta)
    north = origin[0] + radius * np.sin(theta)
    alt = np.linspace(alt_start, alt_end, n_points)
    traj = np.column_stack([north, east, alt])
    meta = _make_metadata(traj, f"climbing_spiral_R{radius:.0f}_A{alt_start:.0f}-{alt_end:.0f}")
    return traj, meta


def generate_slalom(
    origin: np.ndarray = np.array([0.0, 0.0, 5000.0]),
    amplitude: float = 3000.0,
    length: float = 30000.0,
    n_points: int = 500,
    altitude: float = 5000.0,
) -> Tuple[np.ndarray, Dict]:
    """Long-distance slalom (level flight, high-frequency direction changes)."""
    north = np.linspace(origin[0], origin[0] + length, n_points)
    east = origin[1] + amplitude * np.sin(2 * np.pi * north / (length / 5))
    alt = np.full_like(north, altitude)
    traj = np.column_stack([north, east, alt])
    meta = _make_metadata(traj, f"slalom_A{amplitude:.0f}_L{length:.0f}")
    return traj, meta


def generate_ascending_s_curve(
    origin: np.ndarray = np.array([0.0, 0.0, 5000.0]),
    amplitude: float = 5000.0,
    half_period_north: float = 20000.0,
    points_per_half: int = 50,
    n_half_periods: int = 4,
    alt_start: float = 5000.0,
    alt_end: float = 7000.0,
) -> Tuple[np.ndarray, Dict]:
    """S-curve with ascending altitude (moderate climb)."""
    dn = half_period_north / points_per_half
    total_points = n_half_periods * points_per_half + 1
    waypoints = []
    for i in range(total_points):
        frac = i / max(total_points - 1, 1)
        wp_n = origin[0] + i * dn
        wp_e = origin[1] + amplitude * np.sin(np.pi * i / points_per_half)
        wp_a = alt_start + frac * (alt_end - alt_start)
        waypoints.append([wp_n, wp_e, wp_a])
    traj = np.array(waypoints)
    meta = _make_metadata(traj, f"ascending_s_A{amplitude:.0f}_P{points_per_half}_N{n_half_periods}")
    return traj, meta


# ────────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────────

ALL_TRAJECTORIES = {
    "s_curve": lambda: generate_s_curve(n_half_periods=4),
    "s_curve_long": lambda: generate_s_curve(amplitude=5000, half_period_north=20000, n_half_periods=8),
    "circle": lambda: generate_circle(radius=5000, n_points=500),
    "figure_eight": lambda: generate_figure_eight(radius=4000, n_points=500),
    "climbing_spiral": lambda: generate_climbing_spiral(radius=5000, alt_start=5000, alt_end=7000),
    "slalom": lambda: generate_slalom(amplitude=3000, length=30000, n_points=500),
    "ascending_s_curve": lambda: generate_ascending_s_curve(),
}
