"""
Analyse vertical loop tracking quality from the render CSV output.
Computes cross-track error relative to the ideal loop circle,
phase-by-phase error breakdown, and identifies where tracking degrades.

Usage:
    python envs/models/baseline_quat_20260514/analyze_loop_tracking.py
"""
import sys, os, glob
import numpy as np

_planax_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOOP_RADIUS  = 2000.0
CENTER_NORTH = 1000.0
CENTER_ALT   = 7000.0

csv_dir = os.path.join(_planax_root, "results", "vertical_loop_test")
files = sorted(glob.glob(os.path.join(csv_dir, "vertical_loop_*.csv")))
if not files:
    print(f"No CSV files found in {csv_dir}")
    print("Run render_vertical_loop_test.py first.")
    sys.exit(1)

csv_path = files[-1]
print(f"Loading: {csv_path}")

data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
t     = data[:, 0]
north = data[:, 1]
east  = data[:, 2]
alt   = data[:, 3]
vt    = data[:, 4]
roll  = data[:, 5]
pitch = data[:, 6]
yaw   = data[:, 7]
wp_dist = data[:, 8]

# Cross-track error: distance from ideal circle in North-Altitude plane
radial_dist  = np.sqrt((north - CENTER_NORTH)**2 + (alt - CENTER_ALT)**2)
circle_error = np.abs(radial_dist - LOOP_RADIUS)
east_error   = np.abs(east)

# Angle on circle (0=bottom going up clockwise in North-Alt view)
# In North-Alt plane: angle from centre to aircraft
theta = np.arctan2(north - CENTER_NORTH, -(alt - CENTER_ALT))
# Convert to [0, 2pi] range with 0 at bottom going CCW
theta_deg = np.degrees(theta)

print(f"\n{'='*75}")
print(f"LOOP TRACKING QUALITY ANALYSIS")
print(f"{'='*75}")
print(f"  Ideal: circle R={LOOP_RADIUS:.0f}m, centre (N={CENTER_NORTH:.0f}, Alt={CENTER_ALT:.0f})")
print(f"  Steps: {len(t)},  Duration: {t[-1]:.1f}s")
print(f"  Altitude range: {alt.min():.0f} - {alt.max():.0f}m (ideal: {CENTER_ALT-LOOP_RADIUS:.0f} - {CENTER_ALT+LOOP_RADIUS:.0f})")
print(f"  North range:    {north.min():.0f} - {north.max():.0f}m   (ideal: {CENTER_NORTH-LOOP_RADIUS:.0f} - {CENTER_NORTH+LOOP_RADIUS:.0f})")
print(f"  Speed range:    {vt.min():.0f} - {vt.max():.0f}m/s")

print(f"\n  ── Cross-track error (deviation from ideal circle) ──")
p50 = np.percentile(circle_error, 50)
p75 = np.percentile(circle_error, 75)
p90 = np.percentile(circle_error, 90)
p95 = np.percentile(circle_error, 95)
print(f"  mean={circle_error.mean():.0f}m  median={np.median(circle_error):.0f}m  "
      f"p75={p75:.0f}m  p90={p90:.0f}m  p95={p95:.0f}m  max={circle_error.max():.0f}m")
print(f"  East deviation: mean={east_error.mean():.1f}m  p95={np.percentile(east_error,95):.1f}m  max={east_error.max():.0f}m")

# Rating
if p50 < 50 and p90 < 150:
    rating = "GOOD"
elif p50 < 100 and p90 < 300:
    rating = "FAIR"
elif p50 < 200:
    rating = "POOR"
else:
    rating = "BAD"
print(f"\n  Overall rating: {rating}")
print(f"  (GOOD: p50<50m & p90<150m, FAIR: p50<100m & p90<300m, POOR/BAD otherwise)")

# Phase breakdown
# Quadrant 1: climb right side (theta 0 to pi/2)
# Quadrant 2: near top (theta pi/2 to pi)
# Quadrant 3: dive left side (theta -pi to -pi/2)
# Quadrant 4: bottom (theta -pi/2 to 0)
q1 = (theta > -np.pi/8) & (theta <= np.pi/2 + np.pi/8)
q2 = (theta > np.pi/2 - np.pi/8) & (theta <= np.pi + np.pi/8)
q3 = (theta < -np.pi/2 + np.pi/8) | (theta > np.pi - np.pi/8)
q4 = (theta > -np.pi/2 - np.pi/8) & (theta <= np.pi/8)

phases = {
    "Climb (bottom→top, right side)": q1,
    "Top (near apex, inverted)":      q2,
    "Dive (top→bottom, left side)":   q3,
    "Bottom (transition)":            q4,
}

print(f"\n  ── Phase-by-phase cross-track error ──")
for label, mask in phases.items():
    if mask.sum() > 0:
        ce = circle_error[mask]
        spd = vt[mask]
        print(f"  [{label}]")
        print(f"    steps={mask.sum():4d}  "
              f"err: mean={ce.mean():.0f}m  p50={np.percentile(ce,50):.0f}m  "
              f"p90={np.percentile(ce,90):.0f}m  max={ce.max():.0f}m  "
              f"spd: {spd.mean():.0f}m/s")

# Worst sustained deviation
window = 30
rolling = np.convolve(circle_error, np.ones(window)/window, mode='valid')
worst_idx = np.argmax(rolling)
print(f"\n  Worst {window}-step rolling avg error: {rolling[worst_idx]:.0f}m")
print(f"    at t={t[worst_idx]:.1f}s  alt={alt[worst_idx]:.0f}m  N={north[worst_idx]:.0f}m  "
      f"radial_dist={radial_dist[worst_idx]:.0f}m (ideal={LOOP_RADIUS:.0f})")

# Error vs ideal velocity for loop
# At R=2000m, V=250m/s: centripetal accel = V^2/R = 31.25 m/s^2 = 3.2G
# Required G for loop at 250m/s: sqrt(1 + (V^2/(g*R))^2) = sqrt(1 + 3.18^2) = 3.3G
ideal_g = np.sqrt(1 + (vt**2 / (9.81 * LOOP_RADIUS))**2)
print(f"\n  Required G for loop: mean={ideal_g.mean():.1f}G  max={ideal_g.max():.1f}G  "
      f"(well within F-16 capability)")

print(f"{'='*75}")
