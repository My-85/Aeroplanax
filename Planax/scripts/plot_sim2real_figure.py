#!/usr/bin/env python3
"""Generate publication-quality sim2real sensitivity figure.

Reads summary_metrics.csv from the sensitivity experiment and produces
figure_sim2real_sensitivity.pdf + .svg with grouped perturbation settings.

Usage:
  python scripts/plot_sim2real_figure.py <results_dir>
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Perturbation group definitions ─────────────────────────────────────
GROUP_ORDER = [
    "Nominal",
    "Aero ±10%",
    "Mass/Inertia ±5%",
    "Wind 5--10 m/s",
    "Obs. noise",
    "Delay 1--3 steps",
]

SETTING_TO_GROUP = {
    "nominal":      "Nominal",
    "aero_090":     "Aero ±10%",
    "aero_095":     "Aero ±10%",
    "aero_105":     "Aero ±10%",
    "aero_110":     "Aero ±10%",
    "mass_095":     "Mass/Inertia ±5%",
    "mass_105":     "Mass/Inertia ±5%",
    "wind_5ms":     "Wind 5--10 m/s",
    "wind_10ms":    "Wind 5--10 m/s",
    "obs_noise_001":"Obs. noise",
    "delay_1":      "Delay 1--3 steps",
    "delay_2":      "Delay 1--3 steps",
    "delay_3":      "Delay 1--3 steps",
}

GROUP_COLOR = {
    "Nominal":           "#333333",
    "Aero ±10%":         "#4477AA",
    "Mass/Inertia ±5%":  "#66AA55",
    "Wind 5--10 m/s":     "#CC8833",
    "Obs. noise":         "#AA5599",
    "Delay 1--3 steps":   "#CC4444",
}


def main():
    if len(sys.argv) < 2:
        results_dir = sorted(Path("results/sim2real_sensitivity").glob("2*"))
        if not results_dir:
            print("Usage: python scripts/plot_sim2real_figure.py <results_dir>")
            sys.exit(1)
        csv_path = results_dir[-1] / "summary_metrics.csv"
    else:
        csv_path = Path(sys.argv[1]) / "summary_metrics.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    out_dir = csv_path.parent

    # ── Load data ──
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    n_seeds = rows[0]["num_seeds"] if rows else "?"
    print(f"Loaded {len(rows)} rows, n_seeds={n_seeds}")

    # ── Group and aggregate ──
    # Support both old ("cte_p90_mean") and new ("tracking_error_p90_mean") column names
    track_col = ("tracking_error_p90_mean" if "tracking_error_p90_mean" in rows[0]
                 else "cte_p90_mean")
    print(f"Using tracking error column: {track_col}")

    group_data = {g: {"survivals": [], "gmax_vals": [], "vt_mins": [],
                       "aoa_maxs": [], "track_errs": []}
                  for g in GROUP_ORDER}

    for row in rows:
        setting = row["setting"]
        group = SETTING_TO_GROUP.get(setting)
        if group is None:
            print(f"  Warning: unknown setting '{setting}', skipping")
            continue
        group_data[group]["survivals"].append(float(row["survival_rate"]))
        group_data[group]["gmax_vals"].append(float(row["gmax_mean"]))
        group_data[group]["vt_mins"].append(float(row["vt_min_mean"]))
        group_data[group]["aoa_maxs"].append(float(row["alpha_max_mean"]))
        group_data[group]["track_errs"].append(float(row[track_col]))

    # Aggregate stats per group
    stats = {}
    for group in GROUP_ORDER:
        d = group_data[group]
        if not d["survivals"]:
            stats[group] = None
            continue
        surv_arr = np.array(d["survivals"])
        gmax_arr = np.array(d["gmax_vals"])
        stats[group] = {
            "surv_mean": float(np.mean(surv_arr)),
            "surv_std":  float(np.std(surv_arr, ddof=1)) if len(surv_arr) > 1 else 0.0,
            "surv_all":  surv_arr,
            "gmax_worst": float(np.max(gmax_arr)),
            "gmax_mean":  float(np.mean(gmax_arr)),
            "gmax_std":   float(np.std(gmax_arr, ddof=1)) if len(gmax_arr) > 1 else 0.0,
            "gmax_all":   gmax_arr,
            "n_settings": len(d["survivals"]),
        }

    # ── Create figure ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    groups = GROUP_ORDER
    x = np.arange(len(groups))
    bar_width = 0.55

    # ── Panel (a): Avg survival ──
    surv_means = []
    surv_stds = []
    for g in groups:
        s = stats[g]
        surv_means.append(s["surv_mean"] if s else 0)
        surv_stds.append(s["surv_std"] if s else 0)

    bars1 = ax1.bar(x, surv_means, bar_width,
                    color=[GROUP_COLOR[g] for g in groups],
                    edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.errorbar(x, surv_means, yerr=surv_stds, fmt="none",
                 ecolor="#333333", capsize=3, linewidth=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=20, ha="right", fontsize=7.5)
    ax1.set_ylabel("Avg Survival Rate")
    ax1.set_ylim(0, 1.12)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax1.set_title("(a) Survival Rate", fontweight="bold", loc="left")

    # ── Panel (b): Worst-case Max G ──
    gmax_worsts = []
    gmax_means = []
    for g in groups:
        s = stats[g]
        gmax_worsts.append(s["gmax_worst"] if s else 0)
        gmax_means.append(s["gmax_mean"] if s else 0)

    bars2 = ax2.bar(x, gmax_worsts, bar_width,
                    color=[GROUP_COLOR[g] for g in groups],
                    edgecolor="white", linewidth=0.5, alpha=0.85)

    # Highlight delay group with a text label (no special color)
    delay_idx = groups.index("Delay 1--3 steps")
    if stats["Delay 1--3 steps"]:
        delay_gmax = stats["Delay 1--3 steps"]["gmax_worst"]
        ax2.annotate("overload\ntermination",
                     xy=(delay_idx, delay_gmax),
                     xytext=(delay_idx + 0.7, delay_gmax + 0.6),
                     fontsize=7.5, color="#CC4444",
                     arrowprops=dict(arrowstyle="->", color="#CC4444",
                                     lw=1.0, connectionstyle="arc3,rad=0.15"),
                     ha="left", va="center")

    # G=10 reference line (overload threshold)
    ax2.axhline(y=10.0, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax2.text(len(groups) - 0.5, 10.05, "G=10 limit", fontsize=6.5,
             color="red", alpha=0.6, ha="right", va="bottom")

    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, rotation=20, ha="right", fontsize=7.5)
    ax2.set_ylabel("Worst-Case Max G")
    ax2.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax2.set_title("(b) Worst-Case Max G-load", fontweight="bold", loc="left")

    # ── Global annotation ──
    fig.text(0.5, -0.02,
             f"Zero-shot evaluation (no retraining) · "
             f"Averaged over S-curve & 90° pull-up · "
             f"{n_seeds} seeds per setting",
             ha="center", fontsize=7, style="italic", color="#555555")

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    # ── Save ──
    pdf_path = out_dir / "figure_sim2real_sensitivity.pdf"
    svg_path = out_dir / "figure_sim2real_sensitivity.svg"
    fig.savefig(pdf_path, dpi=200)
    fig.savefig(svg_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")

    # ── Print stats for caption ──
    print("\nGroup stats:")
    for g in groups:
        s = stats[g]
        if s:
            print(f"  {g}: surv={s['surv_mean']:.3f}±{s['surv_std']:.3f}, "
                  f"gmax_worst={s['gmax_worst']:.1f}, "
                  f"n_settings={s['n_settings']}")


if __name__ == "__main__":
    main()
