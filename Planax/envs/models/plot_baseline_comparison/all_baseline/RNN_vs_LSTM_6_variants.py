#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN vs LSTM (6 variants) training curves comparison.
- Metrics: eval/episodic_return, eval/episodic_length, eval/success_times
- Input: manually specified TensorBoard log directories (to `.../logs`)
- Smoothing + unified grid interpolation for fair alignment
- Output: per-metric PNG/PDF + CSV summaries (final mean±std, per-seed summaries)
"""

import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# =========================
# Fill your log directories here
# Each group has 3 seeds; each seed can have a list of logs dirs (if multiple runs)
# =========================
EXPERIMENTS: Dict[str, Dict[str, List[str]]] = {
    # RNN family
    "RNN-Orig": {
        "Seed 10": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/RNN新策略/PPO+RNN(原始版本，全无)/seed10/heading_pitch_V_discrete_rnn_2025-08-31-18-28/logs"],
        "Seed 20": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/RNN新策略/PPO+RNN(原始版本，全无)/seed20/heading_pitch_V_discrete_rnn_2025-08-31-18-29/logs"],
        "Seed 42": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/RNN新策略/PPO+RNN(原始版本，全无)/seed42/heading_pitch_V_discrete_rnn_2025-08-31-22-24/logs"],
    },
    "RNN-Actor": {
        "Seed 10": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/RNN新策略/PPO+RNN(仅actor加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed10/heading_pitch_V_discrete_rnn_2025-09-01-01-07/logs"],
        "Seed 20": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/RNN新策略/PPO+RNN(仅actor加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed42/heading_pitch_V_discrete_rnn_2025-09-01-00-57/logs"],
        "Seed 42": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/RNN新策略/PPO+RNN(仅actor加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed42/heading_pitch_V_discrete_rnn_2025-09-01-00-57/logs"],
    },
    "RNN-ActorCritic": {
        "Seed 10": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/RNN新策略/PPO+RNN(actor与critic均加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed10/heading_pitch_V_discrete_rnn_2025-08-29-12-46/logs"],
        "Seed 20": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/RNN新策略/PPO+RNN(actor与critic均加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed20/heading_pitch_V_discrete_rnn_2025-08-29-12-47/logs"],
        "Seed 42": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/RNN新策略/PPO+RNN(actor与critic均加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed42/heading_pitch_V_discrete_rnn_2025-08-29-15-55/logs"],
    },
    # LSTM family
    "LSTM-Orig": {
        "Seed 10": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/LSTM新策略/PPO+LSTM(原始版本，全无)/seed10/heading_pitch_V_discrete_lstm_2025-08-31-11-37/logs"],
        "Seed 20": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/LSTM新策略/PPO+LSTM(原始版本，全无)/seed20/heading_pitch_V_discrete_lstm_2025-08-31-11-38/logs"],
        "Seed 42": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/LSTM新策略/PPO+LSTM(原始版本，全无)/seed42/heading_pitch_V_discrete_lstm_2025-08-31-18-31/logs"],
    },
    "LSTM-Actor": {
        "Seed 10": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/LSTM新策略/PPO+LSTM(仅actor加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)（不带中间奖励并且unreach_heading_pitch_V.py里面mask1只由时间决定（宽松版）训练出来的）/seed10/heading_pitch_V_discrete_lstm_2025-08-29-12-34/logs"],
        "Seed 20": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/LSTM新策略/PPO+LSTM(仅actor加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)（不带中间奖励并且unreach_heading_pitch_V.py里面mask1只由时间决定（宽松版）训练出来的）/seed20/heading_pitch_V_discrete_lstm_2025-08-29-14-56/logs"],
        "Seed 42": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/LSTM新策略/PPO+LSTM(仅actor加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)（不带中间奖励并且unreach_heading_pitch_V.py里面mask1只由时间决定（宽松版）训练出来的）/seed42/heading_pitch_V_discrete_lstm_2025-08-29-15-01/logs"],
    },
    "LSTM-ActorCritic": {
        "Seed 10": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/LSTM新策略/PPO+LSTM(actor和critic均加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed10/heading_pitch_V_discrete_lstm_2025-08-31-22-34/logs"],
        "Seed 20": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/LSTM新策略/PPO+LSTM(actor和critic均加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed20/heading_pitch_V_discrete_lstm_2025-08-31-22-31/logs"],
        "Seed 42": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/LSTM新策略/PPO+LSTM(actor和critic均加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed42/heading_pitch_V_discrete_lstm_2025-08-31-18-33/logs"],
    },
}

# Metrics to plot
METRIC_TAGS = [
    "eval/episodic_return",
    "eval/episodic_length",
    "eval/success_times",
]

# Smoothing config
SMOOTH_MODE   = "window"  # "window" (moving average) or "moving" (EMA)
SMOOTH_WINDOW = 5
EMA_ALPHA     = 0.9

# Unified grid size
GRID_POINTS = 1000

# Whether to plot each run (per seed) as faint lines
PLOT_EACH_RUN = True

# Output directory
OUTPUT_DIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/all_baseline/plots_rnn_lstm_six_variants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette for 6 groups (distinct)
COLOR_MAP = {
    "RNN-Orig":         "#D62728",  # crimson-like
    "RNN-Actor":        "#FF7F0E",  # orange
    "RNN-ActorCritic":  "#2CA02C",  # green
    "LSTM-Orig":        "#1F77B4",  # blue
    "LSTM-Actor":       "#9467BD",  # purple
    "LSTM-ActorCritic": "#8C564B",  # brown
}
GROUP_ORDER = list(COLOR_MAP.keys())  # plot order and legend order

# -------------------------
# Utils
# -------------------------
def ensure_outdir(d: str):
    os.makedirs(d, exist_ok=True)

def smoother(x: np.ndarray, a=0.9, w=5, mode="window") -> np.ndarray:
    """1D smoothing: moving average or EMA."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    if mode == "window":
        out = np.empty_like(x)
        for i in range(len(x)):
            lo = max(0, i - w + 1)
            out[i] = np.mean(x[lo:i+1])
        return out
    elif mode == "moving":
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = (1 - a) * x[i] + a * out[i - 1]
        return out
    else:
        raise ValueError("Unknown smooth mode.")

def read_tb_scalar(log_dir: str, tag: str):
    """Read (step, value) for a scalar tag from a single TB log dir."""
    try:
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        if tag not in ea.scalars.Keys():
            print(f"[Warn] {log_dir} missing tag: {tag}")
            return None, None
        evs = ea.Scalars(tag)
        steps  = np.array([e.step for e in evs], dtype=float)
        values = np.array([e.value for e in evs], dtype=float)
        return steps, values
    except Exception as e:
        print(f"[Err ] read failure {log_dir}: {e}")
        return None, None

def collect_curves_group(group_dirs: Dict[str, List[str]], tag: str):
    """
    For one group: {seed_name: [log_dir, ...]} -> {seed_name: [(steps, values, log_dir), ...]}
    Cleaning: finite filter, sort by step, unique by step (keep first occurrence).
    """
    bag: Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]] = {}
    for seed_name, dirs in group_dirs.items():
        arr = []
        for d in dirs:
            s, v = read_tb_scalar(d, tag)
            if s is None or len(s) == 0:
                continue
            mask = np.isfinite(s) & np.isfinite(v)
            s, v = s[mask], v[mask]
            if len(s) < 2:
                continue
            idx = np.argsort(s)
            s, v = s[idx], v[idx]
            uniq_s, uniq_idx = np.unique(s, return_index=True)
            s, v = uniq_s, v[uniq_idx]
            arr.append((s, v, d))
        if len(arr) > 0:
            bag[seed_name] = arr
    return bag

def process_to_grid_multi(
    all_groups_bags: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]]],
    smooth_mode: str,
    smooth_w: int,
    ema_alpha: float,
    grid_points: int,
):
    """
    Align all groups to a common right-end step (min over all runs' last step).
    Returns:
      x_grid_million,
      per_group_pack: Dict[group_name] -> (mean, std, per_seed_curves, seed_lastvals)
    """
    def min_last_step(bag):
        arr = []
        for _, items in bag.items():
            for (s, _, _) in items:
                if len(s) > 0:
                    arr.append(s[-1])
        return np.min(arr) if len(arr) > 0 else None

    right_ends = []
    for g, bag in all_groups_bags.items():
        m = min_last_step(bag)
        if m is not None:
            right_ends.append(m)
    if len(right_ends) == 0:
        return None, {}

    right_end = float(np.min(right_ends))
    if right_end <= 0:
        return None, {}

    x_grid = np.linspace(0.0, right_end, grid_points)
    x_grid_million = x_grid / 1e6

    def smooth_and_interp(bag):
        per_seed_curves = {}   # {seed: [(xMillion, y_smooth), ...]}
        seed_final_vals = {}   # {seed: [last_y, ...]}
        interp_stack = []      # [n_runs, grid_points]

        for seed, items in bag.items():
            per_seed_curves.setdefault(seed, [])
            seed_final_vals.setdefault(seed, [])
            for (s, v, d) in items:
                y_s = smoother(v, w=smooth_w, mode="window") if smooth_mode == "window" \
                    else smoother(v, a=ema_alpha, mode="moving")
                per_seed_curves[seed].append((s / 1e6, y_s))
                seed_final_vals[seed].append(float(y_s[-1]))
                y_interp = np.interp(x_grid, s, y_s)
                interp_stack.append(y_interp)

        if len(interp_stack) == 0:
            return None, None, per_seed_curves, seed_final_vals
        interp_stack = np.stack(interp_stack, axis=0)
        return np.mean(interp_stack, axis=0), np.std(interp_stack, axis=0), per_seed_curves, seed_final_vals

    per_group_pack = {}
    for g, bag in all_groups_bags.items():
        mean, std, curves, lastvals = smooth_and_interp(bag)
        per_group_pack[g] = (mean, std, curves, lastvals)

    return x_grid_million, per_group_pack

def plot_and_report_multi(
    xg: np.ndarray,
    per_group_pack: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, List[Tuple[np.ndarray, np.ndarray]]], Dict[str, List[float]]]],
    label_order: List[str],
    colors: Dict[str, str],
    metric_tag: str,
    plot_each: bool,
    output_dir: str,
):
    """Draw 6 variants together; save PNG/PDF and CSV summaries."""
    sns.set_theme(style="darkgrid", font_scale=1.2, rc={"figure.figsize": (12, 8)})
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    def draw_one(group_name: str, zorder: int):
        mean, std, seed_curves, _ = per_group_pack[group_name]
        color = colors[group_name]
        if mean is None:
            return
        if plot_each:
            for _, curves in seed_curves.items():
                for (xs, ys) in curves:
                    ax.plot(xs, ys, color=color, alpha=0.12, linewidth=1, zorder=zorder-2)
        ax.plot(xg, mean, color=color, linewidth=2.2, label=group_name, zorder=zorder)
        ax.fill_between(xg, mean - std, mean + std, color=color, alpha=0.18, zorder=zorder-1)

    # Draw in requested order
    for i, g in enumerate(label_order):
        if g in per_group_pack and per_group_pack[g][0] is not None:
            draw_one(g, zorder=5-i)  # earlier (RNN) above later

    # Labels in English
    y_label_map = {
        "eval/episodic_return": "episodic_return",
        "eval/episodic_length": "episodic_length",
        "eval/success_times":   "success_times",
    }
    ylab = y_label_map.get(metric_tag, metric_tag.split("/")[-1])

    ax.set_title(f"RNN vs LSTM (6 variants) performance: {metric_tag}", fontsize=16)
    ax.set_xlabel("Environment steps (million)", fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.legend(loc="best", fontsize=10, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    tag_stub = metric_tag.replace("/", "_")
    png_path = os.path.join(output_dir, f"six_variants_{tag_stub}.png")
    pdf_path = os.path.join(output_dir, f"six_variants_{tag_stub}.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"\n[{metric_tag}] saved:\n- {png_path}\n- {pdf_path}")

    # Print & CSV summaries
    rows_final = []
    for g in label_order:
        if g not in per_group_pack: 
            continue
        mean, std, seed_curves, seed_lastvals = per_group_pack[g]
        if mean is None:
            print(f"[{metric_tag}][{g}] no valid data")
            continue
        print(f"[{metric_tag}][{g}] final (mean±std @ last grid): {mean[-1]:.3f} ± {std[-1]:.3f}")
        rows_final.append([g, float(mean[-1]), float(std[-1])])

        # Per-seed summary
        seed_rows = []
        for seed, vals in seed_lastvals.items():
            if len(vals) == 0:
                continue
            mu, sig = float(np.mean(vals)), float(np.std(vals))
            seed_rows.append([seed, mu, sig, len(vals)])
            print(f"  - {seed}: {mu:.3f} ± {sig:.3f}  (runs={len(vals)})")
        if len(seed_rows) > 0:
            df_seed = pd.DataFrame(seed_rows, columns=["seed", "final_mean", "final_std", "runs"])
            csv_path = os.path.join(output_dir, f"{g.replace('-', '_')}_{tag_stub}_final_per_seed.csv")
            df_seed.to_csv(csv_path, index=False)
            print(f"[{metric_tag}][{g}] per-seed CSV: {csv_path}")

    if len(rows_final) > 0:
        df_final = pd.DataFrame(rows_final, columns=["group", "final_mean", "final_std"])
        csv_path = os.path.join(output_dir, f"final_mean_std_six_variants_{tag_stub}.csv")
        df_final.to_csv(csv_path, index=False)
        print(f"[{metric_tag}][ALL] final mean±std CSV: {csv_path}")

    plt.close(fig)

# -------------------------
# Main
# -------------------------
def main():
    ensure_outdir(OUTPUT_DIR)
    for metric_tag in METRIC_TAGS:
        print(f"\n================ Processing metric: {metric_tag} ================")
        # Read all groups
        all_groups_bags = {}
        for g, seed_dirs in EXPERIMENTS.items():
            print(f"==> Reading group: {g}")
            bag = collect_curves_group(seed_dirs, metric_tag)
            all_groups_bags[g] = bag

        pack = process_to_grid_multi(
            all_groups_bags,
            smooth_mode=SMOOTH_MODE,
            smooth_w=SMOOTH_WINDOW,
            ema_alpha=EMA_ALPHA,
            grid_points=GRID_POINTS,
        )
        if pack[0] is None:
            print(f"[{metric_tag}] no valid logs or step ranges; skip.")
            continue

        xg, per_group_pack = pack
        plot_and_report_multi(
            xg,
            per_group_pack,
            label_order=GROUP_ORDER,
            colors=COLOR_MAP,
            metric_tag=metric_tag,
            plot_each=PLOT_EACH_RUN,
            output_dir=OUTPUT_DIR,
        )

if __name__ == "__main__":
    main()