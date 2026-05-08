"""
Planax Throughput Benchmark - Visualization
Generates charts and formatted tables from benchmark_results.json
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── Load data ────────────────────────────────────────────────────────────────
with open("benchmark_results.json") as f:
    raw = json.load(f)

# Organize by env name
envs = list(dict.fromkeys(r["env"] for r in raw))
data = {e: [r for r in raw if r["env"] == e] for e in envs}

colors = {
    "HeadingPitchV (1-agent)":      "#2196F3",   # blue
    "SManeuver (1-agent)":          "#4CAF50",   # green
    "FullDomainManeuver (1-agent)": "#FF9800",   # orange
}
markers = {"HeadingPitchV (1-agent)": "o",
           "SManeuver (1-agent)": "s",
           "FullDomainManeuver (1-agent)": "^"}
labels  = {
    "HeadingPitchV (1-agent)":      "HeadingPitchV",
    "SManeuver (1-agent)":          "S-Maneuver",
    "FullDomainManeuver (1-agent)": "FullDomain",
}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f0f1a")
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])   # env_sps  (linear-y, log-x)
ax2 = fig.add_subplot(gs[0, 1])   # sim_sps  (log-log)
ax3 = fig.add_subplot(gs[1, 0])   # scaling efficiency
ax4 = fig.add_subplot(gs[1, 1])   # wall-clock ms per call

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=10)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color="#2a2a4a", linewidth=0.6, alpha=0.8)

# ── Plot 1: env_steps/s vs num_envs (linear-y, log-x) ────────────────────────
style_ax(ax1, "Env Steps/s vs Parallel Environments", "num_envs (log scale)", "env steps / s")
for env in envs:
    rows = data[env]
    xs = [r["num_envs"]    for r in rows]
    ys = [r["env_sps_mean"] for r in rows]
    yerr = [r["env_sps_std"] for r in rows]
    ax1.semilogx(xs, ys, marker=markers[env], color=colors[env],
                 label=labels[env], linewidth=2, markersize=7, zorder=3)
    ax1.fill_between(xs,
                     [y - e for y, e in zip(ys, yerr)],
                     [y + e for y, e in zip(ys, yerr)],
                     alpha=0.15, color=colors[env])

ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
ax1.legend(facecolor="#111130", labelcolor="white", fontsize=9, framealpha=0.8)
ax1.set_xscale("log")

# ── Plot 2: sim_steps/s (= env_sps × interaction_steps), log-log ─────────────
style_ax(ax2, "Physics Sim Steps/s  (interaction × env_sps)", "num_envs (log scale)", "sim steps / s  (log)")
for env in envs:
    rows = data[env]
    xs = [r["num_envs"]    for r in rows]
    ys = [r["sim_sps_mean"] for r in rows]
    ax2.loglog(xs, ys, marker=markers[env], color=colors[env],
               label=labels[env], linewidth=2, markersize=7, zorder=3)

# Ideal-linear reference line
ref_x = np.array([1, 10000])
ref_y = np.array([data[envs[0]][0]["sim_sps_mean"],
                  data[envs[0]][0]["sim_sps_mean"] * 10000])
ax2.loglog(ref_x, ref_y, "--", color="#666688", linewidth=1.2,
           label="ideal linear", zorder=1)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
))
ax2.legend(facecolor="#111130", labelcolor="white", fontsize=9, framealpha=0.8)

# ── Plot 3: Scaling efficiency (sps / (n * sps_1)) ────────────────────────────
style_ax(ax3, "Vectorization Efficiency  (vs ideal linear)", "num_envs (log scale)", "efficiency  (%)")
for env in envs:
    rows = data[env]
    sps_1 = rows[0]["env_sps_mean"]
    xs = [r["num_envs"] for r in rows]
    eff = [r["env_sps_mean"] / (r["num_envs"] * sps_1) * 100 for r in rows]
    ax3.semilogx(xs, eff, marker=markers[env], color=colors[env],
                 label=labels[env], linewidth=2, markersize=7, zorder=3)

ax3.axhline(100, color="#666688", linestyle="--", linewidth=1.2, label="ideal (100%)")
ax3.set_ylim(0, 130)
ax3.set_xscale("log")
ax3.legend(facecolor="#111130", labelcolor="white", fontsize=9, framealpha=0.8)

# ── Plot 4: Wall-clock ms per lax.scan call (200 steps) ───────────────────────
style_ax(ax4, "Wall-clock Time per Scan Call  (200-step rollout)", "num_envs (log scale)", "ms per call")
for env in envs:
    rows = data[env]
    xs = [r["num_envs"]       for r in rows]
    ys = [r["ms_per_call"]    for r in rows]
    ax4.semilogx(xs, ys, marker=markers[env], color=colors[env],
                 label=labels[env], linewidth=2, markersize=7, zorder=3)
ax4.set_xscale("log")
ax4.legend(facecolor="#111130", labelcolor="white", fontsize=9, framealpha=0.8)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Planax Platform  —  Throughput Benchmark\n"
    "2× NVIDIA A100 80GB  |  JAX 0.6.2  |  vmap + lax.scan  (single GPU)",
    color="white", fontsize=14, fontweight="bold", y=0.98
)

plt.savefig("benchmark_throughput.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved benchmark_throughput.png")
plt.close()


# ── ASCII / markdown summary table ────────────────────────────────────────────
print("\n" + "="*88)
print(f"{'Environment':<26} {'num_envs':>8}  {'env_sps':>10}  {'sim_sps':>13}  {'efficiency':>12}  {'ms/call':>9}")
print("="*88)

for env in envs:
    rows = data[env]
    sps_1 = rows[0]["env_sps_mean"]
    print(f"  {labels[env]}")
    for r in rows:
        eff = r["env_sps_mean"] / (r["num_envs"] * sps_1) * 100
        peak_marker = " ◀ peak" if r == max(rows, key=lambda x: x["env_sps_mean"]) else ""
        print(f"  {'':<24} {r['num_envs']:>8}  "
              f"{r['env_sps_mean']:>10,.0f}  "
              f"{r['sim_sps_mean']:>13,.0f}  "
              f"{eff:>10.1f}%  "
              f"{r['ms_per_call']:>7.0f}ms{peak_marker}")
    print()

# Peak summary
print("-"*88)
print("PEAK THROUGHPUT SUMMARY")
print("-"*88)
for env in envs:
    rows = data[env]
    best = max(rows, key=lambda x: x["env_sps_mean"])
    print(f"  {labels[env]:<28} "
          f"env_steps/s: {best['env_sps_mean']:>10,.0f}   "
          f"sim_steps/s: {best['sim_sps_mean']:>12,.0f}   "
          f"@ {best['num_envs']} envs")
print("="*88)
