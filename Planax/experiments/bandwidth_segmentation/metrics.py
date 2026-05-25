"""
Metrics computation for closed-loop trajectory tracking evaluation.

All metrics are computed from rollout data and stored with clearly named fields.
"""

import numpy as np
from typing import Dict, Optional, List


def cross_track_error(
    actual_traj: np.ndarray,
    ref_traj: np.ndarray,
) -> np.ndarray:
    """
    Cross-track error: minimum distance from each point on the actual trajectory
    to ANY point on the continuous reference trajectory.

    This is the error to the continuous reference, not to discrete waypoints.
    """
    cte = np.zeros(len(actual_traj))
    for i in range(len(actual_traj)):
        dists = np.linalg.norm(ref_traj - actual_traj[i], axis=1)
        cte[i] = np.min(dists)
    return cte


def waypoint_error(
    actual_traj: np.ndarray,
    waypoints: np.ndarray,
) -> np.ndarray:
    """Minimum distance from each actual point to any waypoint (per-waypoint)."""
    errors = np.zeros(len(waypoints))
    for k in range(len(waypoints)):
        dists = np.linalg.norm(actual_traj - waypoints[k], axis=1)
        errors[k] = np.min(dists)
    return errors


def compute_all_metrics(
    ref_traj: np.ndarray,
    actual_traj: np.ndarray,
    waypoints: np.ndarray,
    actions: Optional[np.ndarray] = None,
    state_dict: Optional[Dict] = None,
    termination_reason: str = "unknown",
    steps_completed: int = 0,
) -> Dict:
    """
    Compute comprehensive tracking, control, and stability metrics.

    Args:
        ref_traj: [M, 3] continuous reference trajectory
        actual_traj: [N, 3] actual flown trajectory
        waypoints: [K, 3] waypoint coordinates used for guidance
        actions: [N, 4] raw discrete actions [thr, el, ail, rud]
        state_dict: dict with keys altitude, airspeed, alpha, beta, roll, pitch, yaw
        termination_reason: why the episode ended
        steps_completed: number of RL steps completed

    Returns:
        Dict with all metrics, using clearly named fields.
    """
    metrics = {"termination_reason": termination_reason,
               "steps_completed": steps_completed,
               "trajectory_completion": steps_completed / max(len(actual_traj), 1)}

    # ── Tracking: to continuous reference trajectory ──
    cte = cross_track_error(actual_traj, ref_traj)
    metrics["cross_track_error_continuous_max_m"] = float(np.max(cte))
    metrics["cross_track_error_continuous_mean_m"] = float(np.mean(cte))
    metrics["cross_track_error_continuous_rms_m"] = float(np.sqrt(np.mean(cte ** 2)))
    metrics["final_position_error_m"] = float(np.linalg.norm(actual_traj[-1] - ref_traj[-1]))

    # ── Tracking: to discrete waypoints ──
    wp_err = waypoint_error(actual_traj, waypoints)
    metrics["waypoint_error_mean_m"] = float(np.mean(wp_err))
    metrics["waypoint_error_max_m"] = float(np.max(wp_err))

    # ── Control metrics ──
    if actions is not None and len(actions) > 1:
        # Decode discrete actions to normalized [-1, 1] range
        # action format: [throttle(0-30), elevator(0-40), aileron(0-40), rudder(0-40)]
        thr_norm = actions[:, 0].astype(float) / 30.0         # [0, 1]
        ele_norm = actions[:, 1].astype(float) * 2.0 / 40.0 - 1.0  # [-1, 1], ±25° physical
        ail_norm = actions[:, 2].astype(float) * 2.0 / 40.0 - 1.0  # [-1, 1], ±21.5° physical
        rud_norm = actions[:, 3].astype(float) * 2.0 / 40.0 - 1.0  # [-1, 1], ±30° physical

        # Normalized action saturation: |action| > 0.95 of range
        for name, vals in [("elevator", ele_norm), ("aileron", ail_norm),
                           ("rudder", rud_norm), ("throttle", thr_norm)]:
            sat = np.mean(np.abs(vals) > 0.95)
            metrics[f"actuator_{name}_saturation_rate_normalized"] = float(sat)
            metrics[f"actuator_{name}_rms_normalized"] = float(np.sqrt(np.mean(vals ** 2)))
            metrics[f"actuator_{name}_max_abs_normalized"] = float(np.max(np.abs(vals)))

        # True physical surface deflection saturation (deg)
        # Elevator: ±25°, Aileron: ±21.5°, Rudder: ±30°
        ele_deg = ele_norm * 25.0
        ail_deg = ail_norm * 21.5
        rud_deg = rud_norm * 30.0
        for name, vals, limit in [("elevator", ele_deg, 25.0), ("aileron", ail_deg, 21.5),
                                   ("rudder", rud_deg, 30.0)]:
            sat = np.mean(np.abs(vals) > 0.95 * limit)
            metrics[f"actuator_{name}_saturation_rate_physical"] = float(sat)
            metrics[f"actuator_{name}_max_deflection_deg"] = float(np.max(np.abs(vals)))

        # Combined saturation: any actuator > 95% of range
        total_sat = np.mean(
            (np.abs(ele_norm) > 0.95) | (np.abs(ail_norm) > 0.95) |
            (np.abs(rud_norm) > 0.95) | (np.abs(thr_norm) > 0.95)
        )
        metrics["actuator_total_saturation_rate"] = float(total_sat)

        # Command smoothness
        act_diff = np.diff(actions.astype(float), axis=0)
        metrics["actuator_command_smoothness_sum_sq"] = float(np.sum(act_diff ** 2))
        metrics["actuator_command_smoothness_rms"] = float(np.sqrt(np.mean(act_diff ** 2)))

    # ── Stability / safety metrics ──
    if state_dict is not None:
        for key in ["altitude", "airspeed", "alpha", "beta", "roll", "pitch", "yaw"]:
            if key in state_dict and len(state_dict[key]) > 0:
                vals = np.array(state_dict[key])
                metrics[f"state_{key}_min"] = float(np.min(vals))
                metrics[f"state_{key}_max"] = float(np.max(vals))
                metrics[f"state_{key}_mean"] = float(np.mean(vals))
                if key in ["alpha", "beta"]:
                    metrics[f"state_{key}_max_abs_deg"] = float(np.max(np.abs(vals)))

    return metrics
