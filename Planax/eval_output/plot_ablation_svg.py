"""
Ablation plots — LoFi vs HiFi policy in HiFi environment.
Data: eval_output/ablation_results_20260420_145651.json
  LoFi: crashed=200/200, timeout=0/200,   mean_ep=2.1s,   hdg_RMSE=83.85°
  HiFi: crashed=30/200,  timeout=170/200, mean_ep=340.2s, hdg_RMSE=49.35°

Output: .pdf (vector, Overleaf-compatible) + .png (raster backup)
"""
import os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

CR = "#E74C3C"
CB = "#2980B9"
CG = "#27AE60"

def savefig(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)

# ── data ──────────────────────────────────────────────────────────────────────
time_s           = [0,  40,   80,  120,  160,  200,  240,  280,  320,  360,  400]
not_crashed_lofi = [100, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0]
not_crashed_hifi = [100, 85,  85,   85,   85,   85,   85,   85,   85,   85,   85]

# =============================================================================
# Figure 1 – Survival curve
# =============================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.2))

ax.fill_between(time_s, not_crashed_lofi, not_crashed_hifi,
                color=CB, alpha=0.10)

ax.plot(time_s, not_crashed_lofi, color=CR, lw=2.5, marker="o",
        ms=5, label="LoFi-trained policy (zero-shot → HiFi)")
ax.plot(time_s, not_crashed_hifi, color=CB, lw=2.5, marker="o",
        ms=5, label="HiFi-trained policy (in-domain)")

ax.axhline(85, color=CG, lw=1.5, ls="--")
ax.text(5, 87, "85% never crashed (170/200 survived full 400 s)",
        color=CG, fontsize=8, va="bottom")

ax.annotate("LoFi: all crash in 2.1 s",
            xy=(2.1, 0), xytext=(70, 22),
            color=CR, fontsize=8, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=CR, lw=1.2))

ax.axvline(400, color="#aaa", lw=1, ls="--")
ax.text(398, 55, "episode end", color="#888", fontsize=7, ha="right")

ax.set_xlabel("Simulation time (s)", fontweight="bold")
ax.set_ylabel("Not crashed (%)", fontweight="bold")
ax.set_title("Survival Curve: LoFi vs HiFi Policy in High-Fidelity Environment",
             fontsize=11, fontweight="bold")
ax.set_xlim(0, 420)
ax.set_ylim(-5, 110)
ax.set_xticks([0, 100, 200, 300, 400])
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.grid(axis="y", color="#e0e0e0", lw=0.8)
ax.legend(loc="center right", fontsize=9, framealpha=0.9)
fig.tight_layout()
savefig(fig, "fig1_survival_curve")

# =============================================================================
# Figure 2 – Stacked outcome bar  (图例移到左上，避免与柱子重叠)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.0, 3.8))

labels        = ["LoFi policy", "HiFi policy"]
crashed_pct   = [100.0,  15.0]
timeout_pct   = [  0.0,  85.0]
x             = np.array([0, 1])
width         = 0.55

bars_c = ax.bar(x, crashed_pct, width, color=CR, alpha=0.88, label="Crashed")
bars_t = ax.bar(x, timeout_pct, width, bottom=crashed_pct,
                color=CG, alpha=0.88, label="Survived to timeout")

for xi, cp, tp in zip(x, crashed_pct, timeout_pct):
    if cp > 3:
        ax.text(xi, cp / 2, f"{cp:.0f}%\n({int(cp*2)}/200)",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    if tp > 0:
        ax.text(xi, cp + tp / 2, f"{tp:.0f}%\n({int(tp*2)}/200)",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight="bold", fontsize=11)
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.set_ylim(0, 115)
ax.grid(axis="y", color="#e8e8e8", lw=0.8)
ax.set_title("Episode Outcome Breakdown (n=200 each)",
             fontsize=11, fontweight="bold")
# 图例放左上，远离柱子
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
fig.tight_layout()
savefig(fig, "fig2_outcome_bars")

# =============================================================================
# Figure 3 – Key metrics (4 panels)
# =============================================================================
panels = [
    ("Episode Length (s)\n[log scale]", 2.084,  340.201, True),
    ("Stall Rate\n(%/step)",            23.48,   23.13,  False),
    ("G-Overload Rate\n(%/step)",        1.40,    0.23,  False),
    ("Heading RMSE\n(deg)",             83.85,   49.35,  False),
]

fig, axes = plt.subplots(1, 4, figsize=(8.8, 4.0))
fig.suptitle("Key Metrics: LoFi vs HiFi Policy in High-Fidelity Environment",
             fontsize=11, fontweight="bold", y=1.01)

for ax, (title, vl, vh, is_log) in zip(axes, panels):
    vals   = [vl, vh]
    colors = [CR, CB]
    xlbls  = ["LoFi", "HiFi"]

    if is_log:
        bar_vals = [math.log10(v + 1) for v in vals]
        ax.set_ylabel("log₁₀(value+1)", fontsize=7)
    else:
        bar_vals = vals

    bars = ax.bar(xlbls, bar_vals, width=0.5, color=colors, alpha=0.88)

    for bar, raw in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bar_vals) * 0.02,
                f"{raw:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color=bar.get_facecolor())

    parts = title.split("\n")
    ax.set_title(parts[0], fontsize=9, fontweight="bold")
    if len(parts) > 1:
        ax.set_xlabel(parts[1], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#e8e8e8", lw=0.8)
    ax.set_ylim(0, max(bar_vals) * 1.3)

lofi_patch = mpatches.Patch(color=CR, alpha=0.88, label="LoFi-trained")
hifi_patch = mpatches.Patch(color=CB, alpha=0.88, label="HiFi-trained")
fig.legend(handles=[lofi_patch, hifi_patch], loc="lower center",
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)
fig.tight_layout()
savefig(fig, "fig3_metric_bars")

print(f"\nAll 3 figures saved to: {OUT}")
