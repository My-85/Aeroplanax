"""
Planax Maximum Throughput - Final Visualization
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

with open("benchmark_limits.json") as f:
    raw = json.load(f)

single = [r for r in raw if r["mode"] == "single_gpu"]
dual   = [r for r in raw if r["mode"] == "dual_gpu"]

s_envs    = [r["num_envs"]       for r in single]
s_sps     = [r["env_sps_mean"]   for r in single]
s_sim     = [r["sim_sps_mean"]   for r in single]
s_gpu_mb  = [r["gpu0_mb"]        for r in single]
s_ms      = [r["ms_per_call"]    for r in single]

d_envs    = [r["num_envs"]       for r in dual]
d_sps     = [r["env_sps_mean"]   for r in dual]
d_sim     = [r["sim_sps_mean"]   for r in dual]
d_ms      = [r["ms_per_call"]    for r in dual]

BLUE   = "#2196F3"
ORANGE = "#FF9800"
GREEN  = "#4CAF50"
GREY   = "#666688"

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor("#0f0f1a")
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])   # Main: throughput vs envs (wide)
ax2 = fig.add_subplot(gs[0, 2])    # GPU memory usage
ax3 = fig.add_subplot(gs[1, 0])    # Scaling efficiency
ax4 = fig.add_subplot(gs[1, 1])    # Wall-clock per call
ax5 = fig.add_subplot(gs[1, 2])    # sim_steps/s (physics throughput)

def style(ax, title, xl, yl, xlog=True, ylog=False):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xl, color="#aaaaaa", fontsize=9)
    ax.set_ylabel(yl, color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444466")
    ax.grid(True, color="#2a2a4a", linewidth=0.6, alpha=0.8)
    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")

fmt_M = mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
fmt_K = mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x < 1e6 else f"{x/1e6:.1f}M")

# ── Plot 1: env_steps/s vs num_envs (MAIN FIGURE) ────────────────────────────
style(ax1, "Planax Throughput  vs  Number of Parallel Environments",
      "Number of Parallel Environments  (log scale)", "env steps / second")

ax1.semilogx(s_envs, s_sps, "o-", color=BLUE, lw=2.5, ms=8,
             label="Single A100 (80GB)", zorder=4)
ax1.semilogx(d_envs, d_sps, "s--", color=ORANGE, lw=2.5, ms=8,
             label="Dual A100  (pmap)", zorder=4)

# Previous paper claim baseline
ax1.axhline(150_000, color="#FF4444", lw=1.5, ls=":", zorder=2,
            label="Paper claim: 150K steps/s")

# Annotations for peaks
peak_s = max(single, key=lambda r: r["env_sps_mean"])
ax1.annotate(f"Single GPU peak\n{peak_s['env_sps_mean']/1e6:.1f}M env_sps\n({peak_s['num_envs']//1e6:.0f}M envs)",
             xy=(peak_s["num_envs"], peak_s["env_sps_mean"]),
             xytext=(peak_s["num_envs"]*0.25, peak_s["env_sps_mean"]*1.05),
             color=BLUE, fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

peak_d = max(dual, key=lambda r: r["env_sps_mean"])
ax1.annotate(f"Dual GPU peak\n{peak_d['env_sps_mean']/1e6:.1f}M env_sps\n({peak_d['num_envs']//1e6:.0f}M envs)\n(not yet saturated)",
             xy=(peak_d["num_envs"], peak_d["env_sps_mean"]),
             xytext=(peak_d["num_envs"]*0.18, peak_d["env_sps_mean"]*0.85),
             color=ORANGE, fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))

ax1.yaxis.set_major_formatter(fmt_M)
ax1.legend(facecolor="#111130", labelcolor="white", fontsize=10, framealpha=0.9, loc="upper left")

# shade saturation region
ax1.axvspan(1_500_000, 5_500_000, alpha=0.07, color=BLUE, label="_")
ax1.text(2_200_000, max(s_sps)*0.55, "Compute\nsaturation\n(single GPU)",
         color=BLUE, fontsize=8, alpha=0.85, ha="center")

# ── Plot 2: GPU memory usage ──────────────────────────────────────────────────
style(ax2, "GPU VRAM Usage  (single GPU)", "num_envs", "Memory used (GB)")
ax2.semilogx(s_envs, [m/1024 for m in s_gpu_mb], "o-", color=GREEN, lw=2, ms=7)
ax2.axhline(80, color="#FF4444", lw=1.5, ls="--", label="A100 limit (80 GB)")
ax2.set_ylim(0, 85)
state_line = [n * 240 / 1024**3 for n in s_envs]
ax2.semilogx(s_envs, state_line, ":", color=GREY, lw=1.5, label="Raw state only")
ax2.legend(facecolor="#111130", labelcolor="white", fontsize=8)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f} GB"))
# annotate actual usage at 5M
ax2.annotate(f"{s_gpu_mb[-1]/1024:.1f} GB\n@ 5M envs",
             xy=(s_envs[-1], s_gpu_mb[-1]/1024),
             xytext=(s_envs[-3], s_gpu_mb[-1]/1024 + 12),
             color=GREEN, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=GREEN))

# ── Plot 3: Scaling efficiency (sps / n / sps_per_env_at_1) ──────────────────
style(ax3, "Vectorization Efficiency", "num_envs", "efficiency (%)")
sps_per_env_s = s_sps[0] / s_envs[0]
sps_per_env_d = d_sps[0] / d_envs[0]
eff_s = [y / (n * sps_per_env_s) * 100 for y, n in zip(s_sps, s_envs)]
eff_d = [y / (n * sps_per_env_d) * 100 for y, n in zip(d_sps, d_envs)]
ax3.semilogx(s_envs, eff_s, "o-", color=BLUE, lw=2, ms=7, label="Single GPU")
ax3.semilogx(d_envs, eff_d, "s--", color=ORANGE, lw=2, ms=7, label="Dual GPU")
ax3.axhline(100, color=GREY, ls="--", lw=1.2, label="Ideal")
ax3.set_ylim(0, 130)
ax3.legend(facecolor="#111130", labelcolor="white", fontsize=8)

# ── Plot 4: Wall-clock per call ───────────────────────────────────────────────
style(ax4, "Wall-clock per Scan Call  (100 steps)", "num_envs", "ms / call", ylog=True)
ax4.loglog(s_envs, s_ms, "o-", color=BLUE, lw=2, ms=7, label="Single GPU")
ax4.loglog(d_envs, d_ms, "s--", color=ORANGE, lw=2, ms=7, label="Dual GPU")
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}ms"))
ax4.legend(facecolor="#111130", labelcolor="white", fontsize=8)

# ── Plot 5: Physics sim steps/s ───────────────────────────────────────────────
style(ax5, "Physics Sim Steps/s  (×10 interaction steps)", "num_envs", "sim steps / s")
ax5.semilogx(s_envs, s_sim, "o-", color=BLUE, lw=2, ms=7, label="Single GPU")
ax5.semilogx(d_envs, d_sim, "s--", color=ORANGE, lw=2, ms=7, label="Dual GPU")
ax5.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:.2f}B"))
ax5.legend(facecolor="#111130", labelcolor="white", fontsize=8)
ax5.annotate(f"{max(d_sim)/1e6:.0f}M sim_sps\n(dual GPU)",
             xy=(d_envs[-1], d_sim[-1]),
             xytext=(d_envs[-2]*0.3, d_sim[-1]*0.82),
             color=ORANGE, fontsize=8, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ORANGE))

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Planax Platform  —  Maximum Throughput Exploration\n"
    "HeadingPitchV Env  |  2× NVIDIA A100 80GB PCIe  |  JAX 0.6.2  |  vmap + lax.scan",
    color="white", fontsize=13, fontweight="bold", y=0.99
)

plt.savefig("benchmark_limits.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved benchmark_limits.png")
plt.close()


# ── ASCII table ───────────────────────────────────────────────────────────────
print()
print("━"*80)
print(f"  SINGLE GPU (A100 #0)  —  env_steps/s")
print("━"*80)
print(f"  {'num_envs':>10}  {'env_sps':>12}  {'sim_sps':>14}  {'GPU VRAM':>10}  {'eff%':>7}")
sps0 = single[0]["env_sps_mean"] / single[0]["num_envs"]
for r in single:
    eff = r["env_sps_mean"] / (r["num_envs"] * sps0) * 100
    peak = " ◀ PEAK" if r == max(single, key=lambda x: x["env_sps_mean"]) else ""
    print(f"  {r['num_envs']:>10,}  {r['env_sps_mean']:>12,.0f}  "
          f"{r['sim_sps_mean']:>14,.0f}  {r['gpu0_mb']:>8} MB  "
          f"{eff:>6.1f}%{peak}")

print()
print("━"*80)
print(f"  DUAL GPU  (pmap, 2× A100)  —  env_steps/s")
print("━"*80)
print(f"  {'num_envs':>10}  {'env_sps':>12}  {'sim_sps':>14}  {'vs 1-GPU':>10}")
for r in dual:
    # compare to single-GPU at same num_envs (or closest)
    ref = next((s for s in single if s["num_envs"] == r["num_envs"]), None)
    ratio = f"{r['env_sps_mean']/ref['env_sps_mean']:.2f}×" if ref else "—"
    peak = " ◀ max tested" if r == max(dual, key=lambda x: x["env_sps_mean"]) else ""
    print(f"  {r['num_envs']:>10,}  {r['env_sps_mean']:>12,.0f}  "
          f"{r['sim_sps_mean']:>14,.0f}  {ratio:>10}{peak}")

print()
peak_s = max(single, key=lambda r: r["env_sps_mean"])
peak_d = max(dual,   key=lambda r: r["env_sps_mean"])
print("━"*80)
print("  SUMMARY")
print("━"*80)
print(f"  Single A100 peak  :  {peak_s['env_sps_mean']/1e6:.2f}M env_steps/s  "
      f"({peak_s['sim_sps_mean']/1e6:.0f}M sim_steps/s)  @ {peak_s['num_envs']/1e6:.0f}M envs")
print(f"  Dual A100 peak    :  {peak_d['env_sps_mean']/1e6:.2f}M env_steps/s  "
      f"({peak_d['sim_sps_mean']/1e6:.0f}M sim_steps/s)  @ {peak_d['num_envs']/1e6:.1f}M envs  (not yet saturated)")
print(f"  Compute saturation:  @ ~1-2M envs single GPU,  ~2M+ dual GPU")
print(f"  Memory limit      :  5M envs uses only {single[-1]['gpu0_mb']/1024:.1f} GB / 80 GB  → NOT memory bound")
print(f"  Paper claim       :  150,000 env_steps/s  →  actual is {peak_s['env_sps_mean']/150000:.0f}× higher")
print("━"*80)
