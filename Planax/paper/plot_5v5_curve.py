"""
Plot 5v5 pursuit-evasion learning curves: HRL vs E2E, 5 seeds each.
Output: 5v5_learning_curve.pdf
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Paths ────────────────────────────────────────────────────────────────────
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Planax/paper/
E2E_DIR = os.path.join(PAPER_DIR, "experiment/5v5_selfplay_combat_E2E")
HRL_DIR = os.path.join(PAPER_DIR, "experiment/5v5_selfplay_combat_HRL")
SEEDS = [0, 10, 20, 30, 42]
TAG = "eval/episodic_return"
OUT_PDF = os.path.join(PAPER_DIR, "5v5_learning_curve.pdf")

# EMA smoothing factor (0 = no smooth, higher = smoother; 0.97 is heavy)
EMA_ALPHA = 0.97


# ── Helpers ──────────────────────────────────────────────────────────────────
def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average, same length as input."""
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * x[i]
    return out


def load_seed(log_dir: str) -> tuple[np.ndarray, np.ndarray]:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    scalars = ea.Scalars(TAG)
    steps = np.array([s.step for s in scalars], dtype=float)
    values = np.array([s.value for s in scalars], dtype=float)
    return steps, values


def load_experiment(base_dir: str, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common_steps (M steps), smoothed mean, SE across seeds."""
    all_steps, all_values = [], []
    for seed in SEEDS:
        log_dir = os.path.join(base_dir, f"{prefix}{seed}", "logs")
        steps, values = load_seed(log_dir)
        all_steps.append(steps)
        all_values.append(ema(values, EMA_ALPHA))

    max_common = min(s[-1] for s in all_steps)
    n_points = min(len(s) for s in all_steps)
    common_steps = np.linspace(0, max_common, n_points)

    interp_values = np.array([
        np.interp(common_steps, steps, values)
        for steps, values in zip(all_steps, all_values)
    ])

    mean = interp_values.mean(axis=0)
    se = interp_values.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    return common_steps / 1e6, mean, se


print("Loading E2E logs …")
e2e_steps, e2e_mean, e2e_se = load_experiment(
    E2E_DIR, "combat_end_to_end_agent10_selfplay_seed"
)
print("Loading HRL logs …")
hrl_steps, hrl_mean, hrl_se = load_experiment(
    HRL_DIR, "combat_agent10_selfplay_seed"
)
print(f"E2E  return range: [{e2e_mean.min():.1f}, {e2e_mean.max():.1f}]")
print(f"HRL  return range: [{hrl_mean.min():.1f}, {hrl_mean.max():.1f}]")


# ── Matplotlib style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HRL_COLOR = "#1f77b4"
E2E_COLOR = "#d62728"

# ── Broken-axis layout ────────────────────────────────────────────────────────
# Panel y-spans are proportional to the data range so tick spacing looks uniform.
TOP_YLIM = (-30, 75)    # HRL range with padding  (span = 105)
BOT_YLIM = (-800, -710) # E2E range with padding  (span =  90)
TOP_SPAN = TOP_YLIM[1] - TOP_YLIM[0]
BOT_SPAN = BOT_YLIM[1] - BOT_YLIM[0]

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    figsize=(5, 4.5),
    sharex=True,
    gridspec_kw={"height_ratios": [TOP_SPAN, BOT_SPAN], "hspace": 0.06},
)

for ax in (ax_top, ax_bot):
    ax.plot(hrl_steps, hrl_mean, color=HRL_COLOR, linewidth=1.8,
            label="Hierarchical (HRL)")
    ax.fill_between(hrl_steps, hrl_mean - hrl_se, hrl_mean + hrl_se,
                    color=HRL_COLOR, alpha=0.3)
    ax.plot(e2e_steps, e2e_mean, color=E2E_COLOR, linewidth=1.8,
            label="End-to-End (E2E)")
    ax.fill_between(e2e_steps, e2e_mean - e2e_se, e2e_mean + e2e_se,
                    color=E2E_COLOR, alpha=0.3)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.spines["right"].set_visible(False)

ax_top.set_ylim(*TOP_YLIM)
ax_bot.set_ylim(*BOT_YLIM)

# Hide the touching spines to create a visual gap
ax_top.spines["top"].set_visible(False)
ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)

# No x-ticks on the top panel
ax_top.tick_params(bottom=False)

# ── Diagonal break marks ──────────────────────────────────────────────────────
# Only on the LEFT spine (right spine is hidden, so marks there would float).
# Drawn in transAxes coordinates; clip_on=False crosses the spine boundary.
D = 0.030   # half-height of slash in axes-fraction units
W = 0.010   # half-width  of slash in axes-fraction units

ax_top.plot((-W, +W), (-D, +D),
            transform=ax_top.transAxes, color="black", lw=0.9, clip_on=False)
ax_bot.plot((-W, +W), (1 - D, 1 + D),
            transform=ax_bot.transAxes, color="black", lw=0.9, clip_on=False)

# ── Labels & legend ───────────────────────────────────────────────────────────
fig.text(
    0.02, 0.5, "Average Reward",
    va="center", ha="center", rotation="vertical",
    fontsize=matplotlib.rcParams["axes.labelsize"],
    fontfamily=matplotlib.rcParams["font.family"],
)
ax_bot.set_xlabel("Million Environment Steps")
ax_top.legend(loc="lower right", frameon=True, edgecolor="black",
              framealpha=0.9, fancybox=False)

plt.savefig(OUT_PDF, bbox_inches="tight", format="pdf")
print(f"Saved → {OUT_PDF}")
