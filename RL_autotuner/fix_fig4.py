#!/usr/bin/env python3
"""Fix Figure 4: show per-level adaptive reward properly."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

theta = np.linspace(0, 150, 500)

# Left: Current — fixed sigma=30 for all levels
reward_fixed = np.exp(-(theta / 30) ** 2)
ax1.plot(theta, reward_fixed, linewidth=3, color='#e74c3c')
ax1.set_title('Current: Fixed σ=30° (all levels)', fontsize=13, fontweight='bold')
ax1.set_xlabel('θ (degrees)')
ax1.set_ylabel('Reward')
ax1.set_ylim(0, 1.05)

# Mark regions
regions = [(0, 30, 'L0', '#2ecc71'), (30, 45, 'L1', '#f1c40f'),
           (45, 75, 'L2-3', '#e67e22'), (75, 150, 'L4-5', '#e74c3c')]
for lo, hi, lbl, c in regions:
    ax1.axvspan(lo, hi, alpha=0.15, color=c)
    ax1.text((lo+hi)/2, 0.98, lbl, ha='center', fontsize=10, fontweight='bold', color=c)

# Mark "no gradient" zone
ax1.annotate('reward ≈ 0\nNO gradient!', xy=(75, 0.01), fontsize=12,
            color='#c0392b', fontweight='bold', ha='center',
            bbox=dict(facecolor='#fdedec', alpha=0.9, boxstyle='round'))
ax1.grid(alpha=0.3)

# Right: Proposed — adaptive sigma per curriculum level
# Each level uses its own sigma
level_configs = [
    (30, '#2ecc71', 'L0-1: σ=30°'),
    (50, '#e67e22', 'L2-3: σ=50°'),
    (75, '#e74c3c', 'L4-5: σ=75°'),
]

for sigma, color, label in level_configs:
    r = np.exp(-(theta / sigma) ** 2)
    ax2.plot(theta, r, linewidth=2.5, color=color, label=label, alpha=0.8)

ax2.set_title('Proposed: σ Adapts to Curriculum Level', fontsize=13, fontweight='bold')
ax2.set_xlabel('θ (degrees)')
ax2.set_ylabel('Reward')
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=11, loc='upper right')

for lo, hi, lbl, c in regions:
    ax2.axvspan(lo, hi, alpha=0.15, color=c)

ax2.annotate('Every level has\ngradient signal!', xy=(85, 0.35), fontsize=12,
            color='#27ae60', fontweight='bold', ha='center',
            bbox=dict(facecolor='#eafaf1', alpha=0.9, boxstyle='round'))
ax2.grid(alpha=0.3)

plt.suptitle('Solution: Level-Adaptive Reward Replaces Fixed Global Reward',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report_fig4_proposed_solution.png')
plt.close()
print("Saved: report_fig4_proposed_solution.png")
