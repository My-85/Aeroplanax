"""
Bandwidth-Aware Adaptive Subgoal Segmentation DP Solver.

Given a continuous 3D reference trajectory, computes an optimal sequence of
waypoint indices that balances geometric accuracy against closed-loop
executability constraints.

Core contribution: the segment cost jointly penalizes cross-track error,
local curvature, attitude rate demands, and switching frequency, with hard
constraints on minimum segment time/length, max rate demands, and max curvature.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field


# ────────────────────────────────────────────────────────────────────────────
@dataclass
class SegmentationConfig:
    """Configuration for the bandwidth-aware DP segmentation solver."""

    # ── Objective weights ──
    w_geo:    float = 1.0
    w_curv:   float = 0.5
    w_rate:   float = 2.0
    w_switch: float = 0.15

    # ── Reference ──
    v_ref:    float = 250.0
    dt_ref:   float = 0.2

    # ── Hard constraints ──
    tau_cmd:          float = 2.0
    psi_dot_max:      float = np.radians(90.0)
    theta_dot_max:    float = np.radians(45.0)
    phi_dot_max:      float = np.radians(120.0)
    max_turn_angle:   float = np.radians(150.0)
    hard_reject_curvature: bool = True

    # ── Geometry ──
    geo_mode: str = "max"

    # ── DP ──
    min_waypoints:    int = 3
    max_waypoints:    int = 200
    force_endpoints:  bool = True


@dataclass
class SegmentationResult:
    waypoint_indices: np.ndarray
    waypoints: np.ndarray
    segment_costs: List[float]
    geo_errors: List[float]
    curv_costs: List[float]
    rate_costs: List[float]
    total_cost: float
    num_segments: int
    debug: Dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _arc_length(traj):
    diffs = np.diff(traj, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])


def _tangents(traj):
    diffs = np.diff(traj, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    t = diffs / norms
    return np.concatenate([t, t[-1:]], axis=0)


def _cross_track_errors(traj, i, j, mode="max"):
    if j <= i + 1:
        return 0.0, 0.0
    p_i, p_j = traj[i], traj[j]
    chord = p_j - p_i
    chord_len_sq = np.dot(chord, chord)
    if chord_len_sq < 1e-9:
        return 0.0, 0.0
    errors = []
    for k in range(i + 1, j):
        t = np.dot(traj[k] - p_i, chord) / chord_len_sq
        t = np.clip(t, 0.0, 1.0)
        proj = p_i + t * chord
        errors.append(np.linalg.norm(traj[k] - proj))
    errors = np.array(errors)
    rms = np.sqrt(np.mean(errors ** 2))
    return float(np.max(errors)), float(rms)


def _segment_curvature_cost(traj, i, j):
    if j <= i + 1:
        return 0.0, 0.0
    t = _tangents(traj)
    acc, max_angle = 0.0, 0.0
    for k in range(i + 1, min(j, len(t))):
        dot = np.clip(np.dot(t[k - 1], t[k]), -1.0, 1.0)
        angle = np.arccos(dot)
        acc += angle
        max_angle = max(max_angle, angle)
    return float(acc), float(max_angle)


def _segment_rate_cost(traj, i, j, arc, v_ref, psi_dot_max, theta_dot_max, phi_dot_max, attitude_ref=None):
    L = arc[j] - arc[i]
    if L < 1e-3:
        return 0.0, {}
    dt = L / v_ref
    t = _tangents(traj)
    psi_i = np.arctan2(t[i, 1], t[i, 0])
    psi_j = np.arctan2(t[j, 1], t[j, 0])
    d_psi = abs(_wrap_pi(psi_j - psi_i))
    psi_dot = d_psi / max(dt, 0.01)
    theta_i = np.arcsin(np.clip(t[i, 2], -1.0, 1.0))
    theta_j = np.arcsin(np.clip(t[j, 2], -1.0, 1.0))
    d_theta = abs(_wrap_pi(theta_j - theta_i))
    theta_dot = d_theta / max(dt, 0.01)
    ratios = [psi_dot / max(psi_dot_max, 1e-6), theta_dot / max(theta_dot_max, 1e-6)]
    phi_dot = 0.0
    if attitude_ref is not None and attitude_ref.shape[0] > max(i, j):
        phi_i = attitude_ref[i, 0]
        phi_j = attitude_ref[j, 0]
        d_phi = abs(_wrap_pi(phi_j - phi_i))
        phi_dot = d_phi / max(dt, 0.01)
        ratios.append(phi_dot / max(phi_dot_max, 1e-6))
    return float(max(ratios)), {
        "dt": float(dt), "L": float(L),
        "d_psi_deg": float(np.degrees(d_psi)),
        "d_theta_deg": float(np.degrees(d_theta)),
        "psi_dot_req": float(np.degrees(psi_dot)),
        "theta_dot_req": float(np.degrees(theta_dot)),
        "phi_dot_req": float(np.degrees(phi_dot)),
    }


def _check_hard_constraints(cfg, traj, arc, i, j, curv_max, rate_debug, attitude_ref=None):
    L = arc[j] - arc[i]
    dt = L / cfg.v_ref
    if dt < cfg.tau_cmd:
        return False, f"dt={dt:.3f}s < tau_cmd={cfg.tau_cmd}s"
    L_min = cfg.v_ref * cfg.tau_cmd
    if L < L_min:
        return False, f"L={L:.1f}m < L_min={L_min:.1f}m"
    if cfg.hard_reject_curvature and curv_max > cfg.max_turn_angle:
        return False, f"curv_max={np.degrees(curv_max):.1f}° > {np.degrees(cfg.max_turn_angle):.1f}°"
    if rate_debug:
        if np.degrees(rate_debug.get("psi_dot_req", 0)) > np.degrees(cfg.psi_dot_max):
            return False, f"psi_dot={np.degrees(rate_debug['psi_dot_req']):.1f} > {np.degrees(cfg.psi_dot_max):.1f}"
        if np.degrees(rate_debug.get("theta_dot_req", 0)) > np.degrees(cfg.theta_dot_max):
            return False, f"theta_dot={np.degrees(rate_debug['theta_dot_req']):.1f} > {np.degrees(cfg.theta_dot_max):.1f}"
        if np.degrees(rate_debug.get("phi_dot_req", 0)) > np.degrees(cfg.phi_dot_max):
            return False, f"phi_dot={np.degrees(rate_debug['phi_dot_req']):.1f} > {np.degrees(cfg.phi_dot_max):.1f}"
    return True, ""


# ────────────────────────────────────────────────────────────────────────────

def segment_cost(cfg, traj, arc, i, j, attitude_ref=None):
    """Compute J(i,j) for candidate segment. Returns (cost, debug_dict)."""
    geo_max, geo_rms = _cross_track_errors(traj, i, j, cfg.geo_mode)
    E_geo = geo_max if cfg.geo_mode == "max" else geo_rms
    curv_acc, curv_max = _segment_curvature_cost(traj, i, j)
    E_curv = curv_acc
    rate_val, rate_debug = _segment_rate_cost(
        traj, i, j, arc, cfg.v_ref, cfg.psi_dot_max, cfg.theta_dot_max, cfg.phi_dot_max, attitude_ref)
    E_rate = rate_val
    feasible, reason = _check_hard_constraints(cfg, traj, arc, i, j, curv_max, rate_debug, attitude_ref)
    if not feasible:
        return np.inf, {"feasible": False, "reason": reason}
    cost = (cfg.w_geo * E_geo + cfg.w_curv * E_curv + cfg.w_rate * E_rate + cfg.w_switch * 1.0)
    debug = {"feasible": True, "i": i, "j": j, "E_geo": float(E_geo), "E_curv": float(E_curv),
             "E_rate": float(E_rate), "total_cost": float(cost),
             "geo_max": float(geo_max), "geo_rms": float(geo_rms),
             "curv_acc_deg": float(np.degrees(curv_acc)), "curv_max_deg": float(np.degrees(curv_max)),
             "L": float(L := arc[j] - arc[i]), "dt": float(L / cfg.v_ref), **rate_debug}
    return cost, debug


def solve(cfg, traj, attitude_ref=None, verbose=False):
    """DP solver. Returns SegmentationResult."""
    M = traj.shape[0]
    arc = _arc_length(traj)
    dp = np.full(M, np.inf)
    back = np.full(M, -1, dtype=int)
    dp[0] = 0.0
    all_debug = {}
    if verbose:
        print(f"DP: {M} trajectory points, computing O(M²) segment costs...")
    for j in range(1, M):
        best_cost, best_i, best_debug = np.inf, -1, {}
        for i in range(j - 1, -1, -1):
            cost, dbg = segment_cost(cfg, traj, arc, i, j, attitude_ref)
            all_debug[(i, j)] = dbg
            if cost >= np.inf:
                continue
            total = dp[i] + cost
            if total < best_cost:
                best_cost, best_i, best_debug = total, i, dbg
        dp[j], back[j] = best_cost, best_i
        if verbose and j % max(1, M // 20) == 0:
            print(f"  [{100*j//M}%] j={j}/{M}, dp={best_cost:.3f}")
    waypoint_indices = [M - 1]
    current = M - 1
    while current > 0:
        prev = back[current]
        if prev < 0:
            raise RuntimeError(f"DP backtracking failed at index {current}")
        waypoint_indices.append(prev)
        current = prev
    waypoint_indices = np.array(waypoint_indices[::-1])
    if cfg.force_endpoints and waypoint_indices[0] != 0:
        waypoint_indices = np.concatenate([[0], waypoint_indices])
    waypoints = traj[waypoint_indices]
    seg_costs, geo_errs, curv_costs, rate_costs = [], [], [], []
    for k in range(len(waypoint_indices) - 1):
        ii, jj = int(waypoint_indices[k]), int(waypoint_indices[k + 1])
        cost, dbg = segment_cost(cfg, traj, arc, ii, jj, attitude_ref)
        seg_costs.append(float(cost) if cost < np.inf else np.inf)
        geo_errs.append(dbg.get("E_geo", 0.0))
        curv_costs.append(dbg.get("E_curv", 0.0))
        rate_costs.append(dbg.get("E_rate", 0.0))
    if verbose:
        print(f"  Done: {len(waypoint_indices)} waypoints, total_cost={dp[-1]:.3f}")
    return SegmentationResult(
        waypoint_indices=waypoint_indices, waypoints=waypoints,
        segment_costs=seg_costs, geo_errors=geo_errs, curv_costs=curv_costs,
        rate_costs=rate_costs, total_cost=float(dp[-1]) if dp[-1] < np.inf else np.inf,
        num_segments=len(waypoint_indices) - 1,
        debug={"dp_values": dp, "back_pointers": back, "segment_debug": all_debug})


def run_all_methods(traj, cfg, uniform_Ns=[5, 10, 20, 40, 80], rdp_epsilons=[50.0, 100.0, 200.0, 500.0],
                    attitude_ref=None, verbose=True):
    """Run all baseline methods + DP variants on a trajectory. Returns dict."""
    from . import baselines as bl
    results = {}
    for N in uniform_Ns:
        for prefix, fn in [("uniform", bl.uniform_arc_length), ("curvature", bl.curvature_based)]:
            indices = fn(traj, N)
            results[f"{prefix}_N{N}"] = {"method": prefix, "N": N, "indices": indices, "waypoints": traj[indices]}
    for eps in rdp_epsilons:
        indices = bl.rdp_simplify(traj, eps)
        results[f"rdp_eps{eps:.0f}"] = {"method": "rdp", "epsilon": eps, "indices": indices,
                                         "waypoints": traj[indices], "N": len(indices)}
    cfg_no_bw = SegmentationConfig(**{**cfg.__dict__, "w_rate": 0.0, "hard_reject_curvature": False})
    cfg_no_bw.psi_dot_max = 1e9; cfg_no_bw.theta_dot_max = 1e9; cfg_no_bw.phi_dot_max = 1e9
    cfg_no_bw.max_turn_angle = 1e9; cfg_no_bw.tau_cmd = 0.0
    r = solve(cfg_no_bw, traj, attitude_ref, verbose=verbose)
    results["dp_no_bandwidth"] = {"method": "dp_no_bandwidth", "indices": r.waypoint_indices,
                                   "waypoints": r.waypoints, "N": r.num_segments + 1, "result": r}
    r = solve(cfg, traj, attitude_ref, verbose=verbose)
    results["dp_with_bandwidth"] = {"method": "dp_with_bandwidth", "indices": r.waypoint_indices,
                                     "waypoints": r.waypoints, "N": r.num_segments + 1, "result": r}
    return results
