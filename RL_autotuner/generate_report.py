#!/usr/bin/env python3
"""Generate Phase 2 training report with charts for group meeting."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
with open("results.jsonl") as f:
    results = [json.loads(l) for l in f if l.strip()]

with open("champion/champion_meta.json") as f:
    champion = json.load(f)

champion_pl = champion.get("per_level_metrics", {})
champion_overall = champion["metrics"]["mean_theta_deg"]

# ============================================================
# Figure 1: Overall theta across all experiments
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

eids = [r["experiment_id"] for r in results]
thetas = []
for r in results:
    ev = r["metrics"].get("eval", {})
    t = ev.get("mean_theta_deg", None)
    if t is None or t < 0:
        t = None
    thetas.append(t)

colors = []
for r in results:
    if r["status"] == "keep":
        colors.append("#2ecc71")
    elif r["status"] == "crash":
        colors.append("#e74c3c")
    else:
        colors.append("#e67e22")

# Plot bars
for i, (eid, theta, color) in enumerate(zip(eids, thetas, colors)):
    if theta is not None:
        ax.bar(eid, theta, color=color, edgecolor='black', linewidth=0.5)
    else:
        ax.bar(eid, 5, color='#e74c3c', edgecolor='black', linewidth=0.5)
        ax.text(eid, 2.5, 'CRASH', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

# Phase dividers
ax.axvline(x=9.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax.text(5, 55, 'Phase 1\n(fixed env)', ha='center', fontsize=11, color='blue', fontweight='bold')
ax.text(16, 55, 'Phase 2\n(curriculum env)', ha='center', fontsize=11, color='blue', fontweight='bold')

# Champion line
ax.axhline(y=champion_overall, color='red', linestyle='-.', linewidth=2, alpha=0.7, label=f'Champion baseline (θ={champion_overall:.1f}°)')

# Phase 1 champion line
ax.axhline(y=20.65, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Phase 1 champion (θ=20.65°, Phase 1 eval)')

keep_patch = mpatches.Patch(color='#2ecc71', label='Keep (new champion)')
discard_patch = mpatches.Patch(color='#e67e22', label='Discard')
crash_patch = mpatches.Patch(color='#e74c3c', label='Crash')
ax.legend(handles=[keep_patch, discard_patch, crash_patch, ax.get_lines()[0], ax.get_lines()[1]],
          loc='upper left', fontsize=10)

ax.set_xlabel('Experiment ID', fontsize=13)
ax.set_ylabel('Mean θ (degrees, lower=better)', fontsize=13)
ax.set_title('RL Autotuner: Experiment History (Phase 1 → Phase 2)', fontsize=15, fontweight='bold')
ax.set_xticks(eids)
ax.set_ylim(0, 80)
ax.grid(axis='y', alpha=0.3)

plt.savefig('report_fig1_experiment_history.png')
plt.close()
print("Saved: report_fig1_experiment_history.png")

# ============================================================
# Figure 2: Per-level analysis — the key finding
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

levels = [0, 1, 2, 3, 4, 5]
level_labels = ['L0\nH±90°\nP±30°', 'L1\nH±120°\nP±45°', 'L2\nH±180°\nP±60°',
                'L3\nH±180°\nP±75°', 'L4\nH±180°\nP±89°', 'L5\nH±180°\nP±89°']

# Champion per-level
champ_thetas = [champion_pl[str(l)]["mean_theta_deg"] for l in levels]

# Collect per-level data from experiments that have it
experiments_with_pl = []
for r in results:
    pl = r["metrics"].get("per_level", {})
    if pl and r["experiment_id"] >= 16:
        exp_thetas = []
        for l in levels:
            val = pl.get(str(l), {}).get("mean_theta_deg", None)
            exp_thetas.append(val)
        experiments_with_pl.append({
            "id": r["experiment_id"],
            "thetas": exp_thetas,
            "status": r["status"],
            "desc": r["description"][:50],
        })

x = np.arange(len(levels))
width = 0.12

# Plot champion
bars_champ = ax.bar(x - 0.3, champ_thetas, width, color='#3498db', edgecolor='black',
                     linewidth=0.5, label='Champion (#9)', zorder=3)

# Plot experiments with per-level data
colors_exp = ['#e74c3c', '#e67e22', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e91e63']
for i, exp in enumerate(experiments_with_pl[:6]):  # max 6 experiments
    vals = [t if t is not None else 0 for t in exp["thetas"]]
    mask = [t is not None for t in exp["thetas"]]
    positions = x - 0.3 + (i + 1) * width
    for j, (v, m) in enumerate(zip(vals, mask)):
        if m:
            ax.bar(positions[j], v, width, color=colors_exp[i % len(colors_exp)],
                   edgecolor='black', linewidth=0.3, alpha=0.7,
                   label=f'#{exp["id"]}' if j == 0 else None, zorder=2)
        else:
            # Early exit - draw X
            ax.text(positions[j], 5, '×', ha='center', va='center', fontsize=12,
                    color=colors_exp[i % len(colors_exp)], fontweight='bold')

# Highlight the coupling problem
ax.annotate('Agent wins here\n(low levels)', xy=(0, 32), fontsize=11,
            ha='center', color='#27ae60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', alpha=0.8))

ax.annotate('Agent loses here\n(high levels)', xy=(3.5, 85), fontsize=11,
            ha='center', color='#c0392b', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', alpha=0.8))

# Draw arrow showing the problem
ax.annotate('', xy=(4.5, 80), xytext=(1, 35),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5, ls='--'))
ax.text(3, 60, 'Reward-Curriculum\nCoupling Problem', ha='center', fontsize=12,
        color='red', fontweight='bold', rotation=25,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', alpha=0.9))

ax.set_xlabel('Curriculum Level', fontsize=13)
ax.set_ylabel('Mean θ (degrees, lower=better)', fontsize=13)
ax.set_title('Key Finding: Same Reward Works for Low Levels but Fails at High Levels',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(level_labels, fontsize=10)
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

plt.savefig('report_fig2_per_level_coupling.png')
plt.close()
print("Saved: report_fig2_per_level_coupling.png")

# ============================================================
# Figure 3: Reward gradient analysis — why coupling happens
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

theta_range = np.linspace(0, 180, 500)

# Left: Gaussian reward for different scales
for sigma, label, color in [(30, 'σ=30° (current)', '#e74c3c'),
                              (60, 'σ=60° (tried #13)', '#e67e22'),
                              (75, 'σ=75° (tried #17)', '#f39c12')]:
    reward = np.exp(-(theta_range / sigma) ** 2)
    ax1.plot(theta_range, reward, label=label, linewidth=2.5, color=color)

# Mark curriculum level ranges
level_ranges = [(0, 30, 'L0'), (30, 45, 'L1'), (45, 60, 'L2'), (60, 75, 'L3'), (75, 89, 'L4-5')]
colors_bg = ['#eafaf1', '#fef9e7', '#fdf2e9', '#fdedec', '#f5eef8']
for (lo, hi, lbl), bg in zip(level_ranges, colors_bg):
    ax1.axvspan(lo, hi, alpha=0.3, color=bg)
    ax1.text((lo + hi) / 2, 0.95, lbl, ha='center', fontsize=9, fontweight='bold', alpha=0.7)

ax1.set_xlabel('θ (degrees)', fontsize=12)
ax1.set_ylabel('Reward value', fontsize=12)
ax1.set_title('Gaussian Reward: Different Scales', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 120)

# Right: Gradient magnitude
for sigma, label, color in [(30, 'σ=30°', '#e74c3c'),
                              (60, 'σ=60°', '#e67e22'),
                              (75, 'σ=75°', '#f39c12')]:
    grad = np.abs(-2 * theta_range / sigma**2 * np.exp(-(theta_range / sigma) ** 2))
    ax2.plot(theta_range, grad, label=label, linewidth=2.5, color=color)

ax2.axhline(y=0.001, color='gray', linestyle='--', alpha=0.5, label='Noise floor (~0.001)')

for (lo, hi, lbl), bg in zip(level_ranges, colors_bg):
    ax2.axvspan(lo, hi, alpha=0.3, color=bg)
    ax2.text((lo + hi) / 2, 0.065, lbl, ha='center', fontsize=9, fontweight='bold', alpha=0.7)

ax2.set_xlabel('θ (degrees)', fontsize=12)
ax2.set_ylabel('|∂r/∂θ| (gradient magnitude)', fontsize=12)
ax2.set_title('Gradient Signal: Below Noise → No Learning', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 120)
ax2.set_ylim(0, 0.07)

plt.suptitle('Why One Reward Cannot Serve All Curriculum Levels', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report_fig3_gradient_analysis.png')
plt.close()
print("Saved: report_fig3_gradient_analysis.png")

# ============================================================
# Figure 4: Proposed solution — adaptive reward
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

theta_range = np.linspace(0, 180, 500)

# Current: fixed sigma=30
reward_fixed = np.exp(-(theta_range / 30) ** 2)
ax.plot(theta_range, reward_fixed, label='Current: fixed σ=30° (all levels)',
        linewidth=2.5, color='#e74c3c', linestyle='--')

# Proposed: adaptive sigma per level
# Level 0-2: sigma=30, Level 3: sigma=50, Level 4-5: sigma=75
# Simulate what this would look like
sigma_adaptive = np.where(theta_range < 45, 30, np.where(theta_range < 75, 50, 75))
reward_adaptive = np.exp(-(theta_range / sigma_adaptive) ** 2)
ax.plot(theta_range, reward_adaptive, label='Proposed: σ adaptive to curriculum level',
        linewidth=3, color='#2ecc71')

for (lo, hi, lbl), bg in zip(level_ranges, colors_bg):
    ax.axvspan(lo, hi, alpha=0.3, color=bg)
    ax.text((lo + hi) / 2, 1.02, lbl, ha='center', fontsize=10, fontweight='bold', alpha=0.7,
            transform=ax.get_xaxis_transform())

ax.set_xlabel('θ (degrees)', fontsize=13)
ax.set_ylabel('Reward value', fontsize=13)
ax.set_title('Proposed Solution: Level-Adaptive Reward Parameters', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 120)
ax.set_ylim(0, 1.05)

plt.savefig('report_fig4_proposed_solution.png')
plt.close()
print("Saved: report_fig4_proposed_solution.png")

# ============================================================
# Figure 5: Agent's search trajectory — what it tried
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Categorize experiments
categories = {
    'Phase 1: Config tuning': {'ids': [1,2,3,4,5,6,7,8,9], 'color': '#3498db', 'marker': 'o'},
    'Phase 2a: Config tuning': {'ids': [10,11,12], 'color': '#e67e22', 'marker': 's'},
    'Phase 2b: Multi-scale Gaussian': {'ids': [13, 16], 'color': '#e74c3c', 'marker': '^'},
    'Phase 2b: Bonus rewards': {'ids': [14, 15], 'color': '#9b59b6', 'marker': 'D'},
    'Phase 2b: Exponent changes': {'ids': [19, 20], 'color': '#f39c12', 'marker': 'v'},
    'Phase 2b: Hybrid/piecewise': {'ids': [17, 18, 21, 22], 'color': '#1abc9c', 'marker': 'p'},
}

for cat_name, cat_data in categories.items():
    cat_thetas = []
    cat_eids = []
    for r in results:
        if r["experiment_id"] in cat_data['ids']:
            ev = r["metrics"].get("eval", {})
            t = ev.get("mean_theta_deg", None)
            if t and t > 0:
                cat_thetas.append(t)
                cat_eids.append(r["experiment_id"])
    if cat_thetas:
        ax.scatter(cat_eids, cat_thetas, color=cat_data['color'], marker=cat_data['marker'],
                   s=120, label=cat_name, edgecolors='black', linewidth=0.5, zorder=3)

# Champion lines
ax.axhline(y=20.65, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Phase 1 best (20.65°)')
ax.axhline(y=champion_overall, color='red', linestyle='-.', linewidth=2, alpha=0.5,
           label=f'Phase 2 baseline ({champion_overall:.1f}°)')

# Annotations
ax.annotate('All Phase 2b attempts:\nglobal reward changes\n→ cannot solve per-level problem',
            xy=(16, 55), fontsize=11, ha='center', color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdedec', alpha=0.9))

ax.axvline(x=9.5, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
ax.text(5, 72, 'Phase 1', ha='center', fontsize=12, color='blue', fontweight='bold')
ax.text(16, 72, 'Phase 2', ha='center', fontsize=12, color='blue', fontweight='bold')

ax.set_xlabel('Experiment ID', fontsize=13)
ax.set_ylabel('Mean θ (degrees, lower=better)', fontsize=13)
ax.set_title("Agent's Search Trajectory: What Was Tried and Why It Failed", fontsize=14, fontweight='bold')
ax.legend(loc='center right', fontsize=9, bbox_to_anchor=(1.0, 0.35))
ax.set_ylim(0, 80)
ax.grid(alpha=0.3)

plt.savefig('report_fig5_search_trajectory.png')
plt.close()
print("Saved: report_fig5_search_trajectory.png")

print("\n=== Report Summary ===")
print(f"Total experiments: {len(results)}")
print(f"Phase 1 (exp 1-9): 4 keeps, best θ=20.65°")
print(f"Phase 2 (exp 10-22): 0 keeps, all discard/crash")
print(f"Champion: #{champion['experiment_id']}, overall θ={champion_overall:.2f}°")
print(f"Key finding: Reward-Curriculum coupling — same reward works for Level 0-2 but fails at Level 3+")
print(f"\nAll figures saved to current directory.")
