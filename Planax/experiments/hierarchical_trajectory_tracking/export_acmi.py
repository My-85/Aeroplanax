"""
Reusable ACMI (Tacview) exporter for hierarchical trajectory tracking.

Guarantees both critical fixes:
1. ENU meters → geodetic lon/lat degrees conversion
2. Reference waypoint / trajectory markers in output

Usage:
    from experiments.hierarchical_trajectory_tracking.export_acmi import write_acmi
    write_acmi("output.acmi", waypoints, trajectory_frames)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

METERS_PER_DEG = 111320.0


def enu_to_geodetic(east_m: float, north_m: float, alt_m: float) -> Tuple[float, float, float]:
    """Convert ENU meters (relative to origin) to geodetic lon/lat/alt.

    Critical fix: raw east/north in METERS must be converted to DEGREES.
    Without this, aircraft at north=12019m renders as latitude=12019°.
    """
    lat = north_m / METERS_PER_DEG
    lon = east_m / (METERS_PER_DEG * np.cos(np.radians(lat)))
    return lon, lat, alt_m


def write_acmi(
    filepath: str,
    waypoints: np.ndarray,
    trajectory: Dict[str, List[float]],
    aircraft_name: str = "F16",
    color: str = "Cyan",
    reference_time: str = "2023-04-01T00:00:00Z",
    waypoint_color: str = "Yellow",
) -> None:
    """Write a Tacview ACMI file with corrected coordinates and waypoint markers.

    Waypoints use incremental timestamps starting from 5000 (verified working
    with Tacview). Aircraft track uses timestamp 100.

    Args:
        filepath: Output .acmi file path
        waypoints: (N, 3) array in (north_m, east_m, alt_m) [NEU]
        trajectory: dict with keys t/n/e/a/roll/pitch/yaw
        aircraft_name: Label in Tacview
        color: Tacview color for aircraft track
        reference_time: ISO 8601 reference time string
        waypoint_color: Tacview color for waypoint markers
    """
    def _wp_to_lonlat(wp):
        """waypoints stored as (north, east, alt)"""
        return enu_to_geodetic(wp[1], wp[0], wp[2])

    with open(filepath, 'w') as f:
        # Header
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        f.write(f"0,ReferenceTime={reference_time}\n")

        # Waypoint markers with incremental timestamps starting from 5000
        for k, wp in enumerate(waypoints):
            wlon, wlat, walt = _wp_to_lonlat(wp)
            f.write(
                f"{5000 + k},Type=Navaid+Static+Waypoint,"
                f"Name=WP_{k},Color={waypoint_color},"
                f"T={wlon}|{wlat}|{walt}|0|0|0\n"
            )

        # Aircraft trajectory
        n_frames = len(trajectory['t'])
        for i in range(n_frames):
            lon, lat, alt = enu_to_geodetic(
                trajectory['e'][i], trajectory['n'][i], trajectory['a'][i]
            )
            f.write(f"#{trajectory['t'][i]:.2f}\n")
            f.write(
                f"100,T={lon:.8f}|{lat:.8f}|{alt:.2f}|"
                f"{trajectory['roll'][i]:.2f}|{trajectory['pitch'][i]:.2f}|{trajectory['yaw'][i]:.2f},"
                f"Type=Air+FixedWing,Name={aircraft_name},Color={color}\n"
            )
