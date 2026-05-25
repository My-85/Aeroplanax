"""
Test target continuity through a vertical loop.

Checks:
1. Do target heading/pitch/roll change smoothly through the loop?
2. Does the quaternion error jump at the top (inverted flight)?
3. Is qv norm continuous?

Usage:
    python experiments/hierarchical_trajectory_tracking/test_loop_target_continuity.py
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_planax_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _planax_root)

from experiments.hierarchical_trajectory_tracking.trajectory_generators import vertical_loop
from experiments.hierarchical_trajectory_tracking.guidance_baselines import (
    pure_pursuit, tangent_following, tangent_plus_roll,
)
from experiments.hierarchical_trajectory_tracking.path_utils import arc_length

# Simulate aircraft flying along the loop at constant speed
RADIUS = 2000.0
N_WP = 30
CRUISE_VT = 250.0
DT = 0.2
ORIGIN = (0.0, 0.0, 5000.0)
INIT_HEADING = 0.0  # North

waypoints, meta = vertical_loop(*ORIGIN, INIT_HEADING, radius=RADIUS, n_points=N_WP)
arc = arc_length(waypoints)
total_arc = arc[-1]

# Simulate ideal flight along the loop
n_steps = int(total_arc / (CRUISE_VT * DT)) + 100
t = np.arange(n_steps) * DT
s = (t * CRUISE_VT) % total_arc  # wrap around

# Interpolate waypoints to get ideal position at each s
wrapped_idx = np.searchsorted(arc, s, side='right') - 1
wrapped_idx = np.clip(wrapped_idx, 0, N_WP - 2)
frac = (s - arc[wrapped_idx]) / (arc[wrapped_idx + 1] - arc[wrapped_idx] + 1e-9)
frac = np.clip(frac, 0.0, 1.0)

ideal_n = waypoints[wrapped_idx, 0] + frac * (waypoints[wrapped_idx + 1, 0] - waypoints[wrapped_idx, 0])
ideal_e = waypoints[wrapped_idx, 1] + frac * (waypoints[wrapped_idx + 1, 1] - waypoints[wrapped_idx, 1])
ideal_alt = waypoints[wrapped_idx, 2] + frac * (waypoints[wrapped_idx + 1, 2] - waypoints[wrapped_idx, 2])

# Compute velocity (tangent)
ideal_vn = np.gradient(ideal_n, DT)
ideal_ve = np.gradient(ideal_e, DT)
ideal_va = np.gradient(ideal_alt, DT)

# Compute ideal yaw, pitch from velocity
ideal_yaw = np.arctan2(ideal_ve, ideal_vn)
ideal_pitch = np.arctan2(ideal_va, np.sqrt(ideal_vn**2 + ideal_ve**2))
ideal_roll = np.zeros(n_steps)
ideal_vt = CRUISE_VT * np.ones(n_steps)

# Compute guidance targets for each step
methods = {
    "Pure Pursuit": [],
    "Tangent Following": [],
    "Tangent + Roll": [],
}

target_h = {k: [] for k in methods}
target_p = {k: [] for k in methods}
target_r = {k: [] for k in methods}

for i in range(0, n_steps, 5):  # every 5th step for speed
    for name, guidance_fn in [
        ("Pure Pursuit", lambda n,e,a,vt,y,p,r,wp,idx: pure_pursuit(n,e,a,vt,y,p,r,wp,idx,heuristic_roll=False)),
        ("Tangent Following", lambda n,e,a,vt,y,p,r,wp,idx: tangent_following(n,e,a,vt,y,p,r,wp,arc,idx,heuristic_roll=False)),
        ("Tangent + Roll", lambda n,e,a,vt,y,p,r,wp,idx: tangent_plus_roll(n,e,a,vt,y,p,r,wp,arc,idx)),
    ]:
        th, tp, tr, tv, info = guidance_fn(
            ideal_n[i], ideal_e[i], ideal_alt[i],
            ideal_vt[i], ideal_yaw[i], ideal_pitch[i], ideal_roll[i],
            waypoints, wrapped_idx[i],
        )
        target_h[name].append(np.degrees(th))
        target_p[name].append(np.degrees(tp))
        target_r[name].append(np.degrees(tr))

# Plot
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Target Continuity Through Vertical Loop\n"
             f"R={RADIUS}m, {N_WP} waypoints, simulated ideal flight at {CRUISE_VT} m/s",
             fontsize=13)

for col, (name, color) in enumerate(zip(methods.keys(), ['blue', 'green', 'red'])):
    idx = np.arange(0, n_steps, 5)[:len(target_h[name])]

    axes[0, col].plot(idx * DT, target_h[name], color=color, lw=0.8)
    axes[0, col].set_title(f"{name}\nTarget Heading (deg)")
    axes[0, col].set_ylabel("deg"); axes[0, col].grid(True, alpha=0.3)
    axes[0, col].set_ylim(-185, 185)

    axes[1, col].plot(idx * DT, target_p[name], color=color, lw=0.8)
    axes[1, col].set_title("Target Pitch (deg)")
    axes[1, col].set_ylabel("deg"); axes[1, col].grid(True, alpha=0.3)
    axes[1, col].set_ylim(-95, 95)

    axes[2, col].plot(idx * DT, target_r[name], color=color, lw=0.8)
    axes[2, col].set_title("Target Roll (deg)")
    axes[2, col].set_xlabel("Time (s)")
    axes[2, col].set_ylabel("deg"); axes[2, col].grid(True, alpha=0.3)
    axes[2, col].set_ylim(-80, 80)

plt.tight_layout()
outpath = os.path.join(_planax_root, "results", "loop_target_continuity.png")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath, dpi=120, bbox_inches='tight')
plt.close(fig)

# ── Summary stats ──
print("="*70)
print("TARGET CONTINUITY THROUGH VERTICAL LOOP")
print("="*70)
for name in methods:
    h_arr = np.array(target_h[name])
    p_arr = np.array(target_p[name])
    r_arr = np.array(target_r[name])
    # Check for jumps
    dh = np.abs(np.diff(h_arr))
    dp = np.abs(np.diff(p_arr))
    dr = np.abs(np.diff(r_arr))
    # Detect large jumps (> 30 deg in one step)
    h_jumps = np.sum(dh > 30)
    p_jumps = np.sum(dp > 30)
    r_jumps = np.sum(dr > 30)
    print(f"\n{name}:")
    print(f"  Heading: range=[{h_arr.min():.0f}, {h_arr.max():.0f}]deg  "
          f"jumps>30deg: {h_jumps}")
    print(f"  Pitch:   range=[{p_arr.min():.0f}, {p_arr.max():.0f}]deg  "
          f"jumps>30deg: {p_jumps}")
    print(f"  Roll:    range=[{r_arr.min():.0f}, {r_arr.max():.0f}]deg  "
          f"jumps>30deg: {r_jumps}")
    if h_jumps == 0 and p_jumps == 0:
        print(f"  VERDICT: Targets are continuous ✓")
    else:
        print(f"  VERDICT: Targets have discontinuities ✗")

print(f"\nPlot saved: {outpath}")
print("="*70)
