"""
Baseline waypoint selection methods: uniform, curvature, RDP.
"""

import numpy as np
from typing import List


def _arc_length(traj: np.ndarray) -> np.ndarray:
    diffs = np.diff(traj, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _tangents(traj: np.ndarray) -> np.ndarray:
    diffs = np.diff(traj, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    t = diffs / norms
    return np.concatenate([t, t[-1:]], axis=0)


def uniform_arc_length(traj: np.ndarray, N: int) -> np.ndarray:
    """Select N waypoints evenly spaced by arc length. Always includes endpoints."""
    arc = _arc_length(traj)
    total = arc[-1]
    if total < 1e-6:
        return np.array([0, len(traj) - 1])
    target_s = np.linspace(0, total, max(N, 2))
    indices = [0]
    for s in target_s[1:-1]:
        idx = int(np.searchsorted(arc, s))
        idx = np.clip(idx, 1, len(traj) - 2)
        indices.append(idx)
    indices.append(len(traj) - 1)
    return np.array(sorted(set(indices)))


def curvature_based(traj: np.ndarray, N: int) -> np.ndarray:
    """Select N waypoints biased toward high-curvature regions."""
    arc = _arc_length(traj)
    t = _tangents(traj)
    curvature = np.zeros(len(traj))
    for k in range(1, len(traj) - 1):
        dot = np.clip(np.dot(t[k - 1], t[k + 1]), -1.0, 1.0)
        ds = max(arc[k + 1] - max(arc[k - 1], 0), 0.001)
        curvature[k] = np.arccos(dot) / ds
    cumsum = np.cumsum(np.maximum(curvature, 1e-6))
    cumsum /= max(cumsum[-1], 1e-6)
    target_c = np.linspace(0, 1, max(N, 2))
    indices = [0]
    for c in target_c[1:-1]:
        idx = int(np.searchsorted(cumsum, c))
        idx = np.clip(idx, 1, len(traj) - 2)
        indices.append(idx)
    indices.append(len(traj) - 1)
    return np.array(sorted(set(indices)))


def rdp_simplify(traj: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification. Returns waypoint indices."""
    indices = [0, len(traj) - 1]
    stack = [(0, len(traj) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p_i, p_j = traj[i], traj[j]
        chord = p_j - p_i
        chord_len_sq = np.dot(chord, chord)
        if chord_len_sq < 1e-9:
            continue
        max_err, max_k = 0.0, i
        for k in range(i + 1, j):
            t = np.dot(traj[k] - p_i, chord) / chord_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj = p_i + t * chord
            err = np.linalg.norm(traj[k] - proj)
            if err > max_err:
                max_err = err
                max_k = k
        if max_err > epsilon:
            indices.append(max_k)
            stack.append((i, max_k))
            stack.append((max_k, j))
    return np.array(sorted(set(indices)))
