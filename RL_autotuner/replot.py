#!/usr/bin/env python3
"""Re-generate plots from saved metrics JSON files."""
import json, sys, os, numpy as np
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_MEM_FRACTION"] = "0.1"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "Planax"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = str(SCRIPT_DIR / "comparison_results")

with open(f"{OUT_DIR}/metrics_A.json") as f:
    metrics_a = json.load(f)
with open(f"{OUT_DIR}/metrics_B.json") as f:
    metrics_b = json.load(f)

LABEL_A = "baseline_θ24.88"
LABEL_B = "autotuned_θ20.65"
COLOR_A = "#2196F3"
COLOR_B = "#FF9800"
SETTLE_THRESH_DEG = 8
STEPS_PER_WP = 250

WAYPOINTS = [
    ("WP00_H+15",        15,   0,   0, 200),
    ("WP01_H-15",       -15,   0,   0, 200),
    ("WP02_P+8",          0,   8,   0, 200),
    ("WP03_P-8",          0,  -8,   0, 200),
    ("WP04_H+30_P+10",   30,  10,   0, 210),
    ("WP05_H-30_P-10",  -30, -10,   0, 190),
    ("WP06_R+30",         0,   0,  30, 200),
    ("WP07_R-45",         0,   0, -45, 200),
    ("WP08_H+60_P+20",   60,  20,   0, 220),
    ("WP09_H-60_P-20",  -60, -20,   0, 180),
    ("WP10_R+90",         0,   0,  90, 200),
    ("WP11_combo_L2",    45,  15,  30, 215),
    ("WP12_H+90_P+30",   90,  30,   0, 230),
    ("WP13_R+135",        0,   0, 135, 200),
    ("WP14_combo_L3",    60,  25,  60, 220),
    ("WP15_H-90_P-30",  -90, -30,   0, 175),
    ("WP16_H+120_P+45", 120,  45,   0, 240),
    ("WP17_R+180",        0,   0, 180, 200),
    ("WP18_combo_L4",    90,  40,  90, 240),
    ("WP19_full",       -120, -45, -90, 170),
]

def ts(m): return [x["sim_time"] for x in m]

wp_starts, wp_labels = [], []
for m in metrics_a:
    if m["step_in_wp"] == 0:
        wp_starts.append(m["sim_time"])
        wp_labels.append(m["wp_name"].split("_", 1)[1])

def add_wp_lines(ax, top=180):
    for xv, xl in zip(wp_starts, wp_labels):
        ax.axvline(x=xv, color="gray", linestyle=":", alpha=0.35, linewidth=0.7)
        ax.text(xv + 0.3, top * 0.97, xl, fontsize=5, rotation=90, va="top", color="gray")

n_wp = len(WAYPOINTS)
settle_a, settle_b, ss_a, ss_b = [], [], [], []
for wi in range(n_wp):
    for metrics, slist, sslist in [(metrics_a, settle_a, ss_a), (metrics_b, settle_b, ss_b)]:
        data = [m for m in metrics if m["wp_idx"] == wi]
        st = next((m["step_in_wp"] for m in data if m["theta_deg"] < SETTLE_THRESH_DEG), STEPS_PER_WP)
        slist.append(st * 0.2)
        tail = data[int(len(data) * 0.75):]
        sslist.append(np.mean([m["theta_deg"] for m in tail]) if tail else 0.0)

x_pos = np.arange(n_wp)
w = 0.38

# Fig 1: theta
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(ts(metrics_a), [m["theta_deg"] for m in metrics_a], color=COLOR_A, label=LABEL_A, alpha=0.85, linewidth=0.9)
ax.plot(ts(metrics_b), [m["theta_deg"] for m in metrics_b], color=COLOR_B, label=LABEL_B, alpha=0.85, linewidth=0.9)
ax.axhline(y=SETTLE_THRESH_DEG, color="green", linestyle="--", alpha=0.6, label=f"on-target ({SETTLE_THRESH_DEG}°)")
add_wp_lines(ax, top=185)
ax.set_ylim(0, 185); ax.set_ylabel("Theta (deg)"); ax.set_xlabel("Sim time (s)")
ax.set_title("Geodesic Attitude Error (theta_deg)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig1_theta.png", dpi=150); plt.close()

# Fig 2: delta_vt
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(ts(metrics_a), [m["delta_vt"] for m in metrics_a], color=COLOR_A, label=LABEL_A, alpha=0.85, linewidth=0.9)
ax.plot(ts(metrics_b), [m["delta_vt"] for m in metrics_b], color=COLOR_B, label=LABEL_B, alpha=0.85, linewidth=0.9)
ax.axhline(y=25, color="green", linestyle="--", alpha=0.6, label="on-target (25 m/s)")
add_wp_lines(ax, top=max(max(m["delta_vt"] for m in metrics_a), max(m["delta_vt"] for m in metrics_b)) * 0.95)
ax.set_ylim(bottom=0); ax.set_ylabel("Delta Vt (m/s)"); ax.set_xlabel("Sim time (s)")
ax.set_title("Speed Tracking Error (delta_vt)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig2_delta_vt.png", dpi=150); plt.close()

# Fig 3: settling time
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x_pos - w/2, settle_a, w, color=COLOR_A, alpha=0.75, label=LABEL_A)
ax.bar(x_pos + w/2, settle_b, w, color=COLOR_B, alpha=0.75, label=LABEL_B)
ax.axhline(y=STEPS_PER_WP * 0.2, color="red", linestyle="--", alpha=0.5, label=f"max ({STEPS_PER_WP*0.2:.0f}s)")
ax.set_xticks(x_pos); ax.set_xticklabels(wp_labels, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Settling Time (s)"); ax.set_title(f"Settling Time to theta < {SETTLE_THRESH_DEG}°")
ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig3_settling.png", dpi=150); plt.close()

# Fig 4: steady-state error
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x_pos - w/2, ss_a, w, color=COLOR_A, alpha=0.75, label=LABEL_A)
ax.bar(x_pos + w/2, ss_b, w, color=COLOR_B, alpha=0.75, label=LABEL_B)
ax.axhline(y=SETTLE_THRESH_DEG, color="green", linestyle="--", alpha=0.5, label=f"target ({SETTLE_THRESH_DEG}°)")
ax.set_xticks(x_pos); ax.set_xticklabels(wp_labels, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Steady-State Theta Error (deg)"); ax.set_title("Steady-State Attitude Error (last 25% of WP)")
ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig4_ss_error.png", dpi=150); plt.close()

# Fig 5: yaw
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(ts(metrics_a), [m["tgt_yaw"] for m in metrics_a], color="black", linestyle="--", linewidth=1.2, label="Target Yaw")
ax.plot(ts(metrics_a), [m["yaw"] for m in metrics_a], color=COLOR_A, alpha=0.85, linewidth=0.9, label=f"Yaw {LABEL_A}")
ax.plot(ts(metrics_b), [m["yaw"] for m in metrics_b], color=COLOR_B, alpha=0.85, linewidth=0.9, label=f"Yaw {LABEL_B}")
add_wp_lines(ax, top=200)
ax.set_ylabel("Yaw / Heading (deg)"); ax.set_xlabel("Sim time (s)")
ax.set_title("Yaw (Heading) Tracking"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig5_yaw.png", dpi=150); plt.close()

# Fig 6: pitch
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(ts(metrics_a), [m["tgt_pitch"] for m in metrics_a], color="black", linestyle="--", linewidth=1.2, label="Target Pitch")
ax.plot(ts(metrics_a), [m["pitch"] for m in metrics_a], color=COLOR_A, alpha=0.85, linewidth=0.9, label=f"Pitch {LABEL_A}")
ax.plot(ts(metrics_b), [m["pitch"] for m in metrics_b], color=COLOR_B, alpha=0.85, linewidth=0.9, label=f"Pitch {LABEL_B}")
add_wp_lines(ax, top=100)
ax.set_ylabel("Pitch (deg)"); ax.set_xlabel("Sim time (s)")
ax.set_title("Pitch Tracking"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig6_pitch.png", dpi=150); plt.close()

# Fig 7: roll
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(ts(metrics_a), [m["tgt_roll"] for m in metrics_a], color="black", linestyle="--", linewidth=1.2, label="Target Roll")
ax.plot(ts(metrics_a), [m["roll"] for m in metrics_a], color=COLOR_A, alpha=0.85, linewidth=0.9, label=f"Roll {LABEL_A}")
ax.plot(ts(metrics_b), [m["roll"] for m in metrics_b], color=COLOR_B, alpha=0.85, linewidth=0.9, label=f"Roll {LABEL_B}")
add_wp_lines(ax, top=200)
ax.set_ylabel("Roll (deg)"); ax.set_xlabel("Sim time (s)")
ax.set_title("Roll Tracking"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig7_roll.png", dpi=150); plt.close()

# Fig 8: speed
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(ts(metrics_a), [m["tgt_vt"] for m in metrics_a], color="black", linestyle="--", linewidth=1.2, label="Target Vt")
ax.plot(ts(metrics_a), [m["vt"] for m in metrics_a], color=COLOR_A, alpha=0.85, linewidth=0.9, label=f"Vt {LABEL_A}")
ax.plot(ts(metrics_b), [m["vt"] for m in metrics_b], color=COLOR_B, alpha=0.85, linewidth=0.9, label=f"Vt {LABEL_B}")
add_wp_lines(ax, top=max(max(m["vt"] for m in metrics_a), max(m["vt"] for m in metrics_b)) * 0.95)
ax.set_ylabel("Speed (m/s)"); ax.set_xlabel("Sim time (s)")
ax.set_title("Speed Tracking"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig8_speed.png", dpi=150); plt.close()

print(f"Saved 8 plots to {OUT_DIR}/")

# Console summary
theta_a = [m["theta_deg"] for m in metrics_a]
theta_b = [m["theta_deg"] for m in metrics_b]
dvt_a   = [m["delta_vt"]  for m in metrics_a]
dvt_b   = [m["delta_vt"]  for m in metrics_b]
print("\n" + "=" * 88)
print(f"{'WP Name':<24} {'Settle_A':>9} {'Settle_B':>9} {'SS_A':>8} {'SS_B':>8} {'Winner'}")
print("-" * 88)
for i, (name, *_) in enumerate(WAYPOINTS):
    short = name.split("_", 1)[1]
    winner = "B" if ss_b[i] < ss_a[i] else ("A" if ss_a[i] < ss_b[i] else "=")
    print(f"{short:<24} {settle_a[i]:>8.1f}s {settle_b[i]:>8.1f}s "
          f"{ss_a[i]:>7.1f}° {ss_b[i]:>7.1f}° {winner}")
print("=" * 88)
print(f"\nOverall mean theta:    {LABEL_A}={np.mean(theta_a):.2f}°   {LABEL_B}={np.mean(theta_b):.2f}°")
print(f"Overall mean delta_vt: {LABEL_A}={np.mean(dvt_a):.1f}   {LABEL_B}={np.mean(dvt_b):.1f}")
winner_overall = LABEL_B if np.mean(theta_b) < np.mean(theta_a) else LABEL_A
print(f"\n★  Overall winner: {winner_overall}")
