# -*- coding: utf-8 -*-
"""
RNN vs LSTM 三指标训练曲线对比（手动路径版）
- 指标：eval/episodic_return、eval/episodic_length、eval/success_times
- 手动指定每个 seed 的 TensorBoard logs 路径
- 平滑 + 统一网格插值（保证两条基线在同一 x 范围内比较）
- 输出：每个指标各一张 PNG/PDF 图 + 两类 CSV（每基线各 seed 最终值；两基线最终点均值±std）
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# =========================
# 中文显示（可选）
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei']   # 中文标签
plt.rcParams['axes.unicode_minus'] = False     # 负号正常显示

# =========================
# 你只需要修改这里的路径映射
# =========================
# RNN/LSTM 的 3 个 seed 日志目录（到 logs 这一层）
# 如果某个 seed 有多次 run，可以在列表里放多个 logs 目录
RNN_EXPERIMENTS = {
    "Seed 10": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline/new_baseline/seed10/heading_pitch_V_discrete_rnn_2025-08-29-12-46/logs"],
    "Seed 20": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline/new_baseline/seed20/heading_pitch_V_discrete_rnn_2025-08-29-12-47/logs"],
    "Seed 42": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline/new_baseline/seed42/heading_pitch_V_discrete_rnn_2025-08-29-15-55/logs"],
}

LSTM_EXPERIMENTS = {
    "Seed 10": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/LSTM_baseline/new_baseline/seed10/heading_pitch_V_discrete_lstm_2025-08-29-12-34/logs"],
    "Seed 20": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/LSTM_baseline/new_baseline/seed20/heading_pitch_V_discrete_lstm_2025-08-29-14-56/logs"],
    "Seed 42": ["/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/LSTM_baseline/new_baseline/seed42/heading_pitch_V_discrete_lstm_2025-08-29-15-01/logs"],
}

# 要绘制的 3 个指标
METRIC_TAGS = [
    "eval/episodic_return",
    "eval/episodic_length",
    "eval/success_times",
]

# 平滑配置
SMOOTH_MODE   = "window"  # "window"（滑动均值）或 "moving"（指数滑动）
SMOOTH_WINDOW = 5         # window 模式的窗口
EMA_ALPHA     = 0.9       # moving 模式的指数权重

# 统一网格密度（对齐插值后的点数）
GRID_POINTS = 1000

# 是否把每条 seed 的原始（平滑后）曲线画成淡色细线
PLOT_EACH_RUN = True

# 输出目录
OUTPUT_DIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/plot_baseline_comparison/plots_rnn_vs_lstm_manual_3metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 工具函数
# =========================
def smoother(x, a=0.9, w=5, mode="window"):
    """1D 序列平滑：滑动均值 or 指数滑动"""
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

def read_tb_scalar(log_dir, tag):
    """读取单个日志目录的 (step, value)，失败返回 (None, None)"""
    try:
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        if tag not in ea.scalars.Keys():
            print(f"[Warn] {log_dir} 不含 tag: {tag}")
            return None, None
        evs = ea.Scalars(tag)
        steps  = np.array([e.step for e in evs], dtype=float)
        values = np.array([e.value for e in evs], dtype=float)
        return steps, values
    except Exception as e:
        print(f"[Err ] 读取失败 {log_dir}: {e}")
        return None, None

def collect_curves(experiments_dict, tag):
    """把 {seed_name: [log_dir, ...]} 读成 {seed_name: [(steps, values, log_dir), ...]}"""
    bag = {}
    for seed_name, dirs in experiments_dict.items():
        arr = []
        for d in dirs:
            s, v = read_tb_scalar(d, tag)
            if s is None or len(s) == 0:
                continue
            # 清理 NaN/Inf & 排序去重
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

def process_to_grid(bag_a, bag_b, smooth_mode, smooth_w, ema_alpha, grid_points):
    """两条基线（A/B）的曲线一起对齐：
       - 求所有曲线的共同最短尾步 (min_last_step)
       - 在 [0, min_last_step] 上建立统一网格
       - 对每条曲线：平滑 -> 线性插值到统一网格
       返回：
        x_grid_million,
        (A_mean, A_std, A_seed_curves, A_seed_last_values),
        (B_mean, B_std, B_seed_curves, B_seed_last_values)
    """
    def min_last_step(bag):
        arr = []
        for _, items in bag.items():
            for (s, _, _) in items:
                if len(s) > 0:
                    arr.append(s[-1])
        return np.min(arr) if len(arr) > 0 else None

    a_last = min_last_step(bag_a)
    b_last = min_last_step(bag_b)
    if a_last is None and b_last is None:
        return None, None, None

    # 统一右端（若有一边为空，就用另一边的右端）
    right_end = b_last if a_last is None else (a_last if b_last is None else min(a_last, b_last))
    if right_end is None or right_end <= 0:
        return None, None, None

    x_grid = np.linspace(0.0, right_end, grid_points)
    x_grid_million = x_grid / 1e6

    def smooth_and_interp(bag):
        per_seed_curves = {}   # {seed: [(xMillion, y_smooth), ...]}
        seed_final_vals = {}   # {seed: [last_y, ...]}
        interp_stack = []      # 统一网格上的 y 曲线集合

        for seed, items in bag.items():
            per_seed_curves.setdefault(seed, [])
            seed_final_vals.setdefault(seed, [])
            for (s, v, d) in items:
                if smooth_mode == "window":
                    y_s = smoother(v, w=smooth_w, mode="window")
                else:
                    y_s = smoother(v, a=ema_alpha, mode="moving")
                # 保存一份“原始平滑后”的曲线（按自身 step）
                per_seed_curves[seed].append((s / 1e6, y_s))
                seed_final_vals[seed].append(float(y_s[-1]))
                # 插值到统一网格
                y_interp = np.interp(x_grid, s, y_s)
                interp_stack.append(y_interp)

        if len(interp_stack) == 0:
            return None, None, per_seed_curves, seed_final_vals
        interp_stack = np.stack(interp_stack, axis=0)  # [num_runs, grid_points]
        return np.mean(interp_stack, axis=0), np.std(interp_stack, axis=0), per_seed_curves, seed_final_vals

    a_mean, a_std, a_curves, a_lastvals = smooth_and_interp(bag_a)
    b_mean, b_std, b_curves, b_lastvals = smooth_and_interp(bag_b)
    return x_grid_million, (a_mean, a_std, a_curves, a_lastvals), (b_mean, b_std, b_curves, b_lastvals)

def plot_and_report(
    xg, a_pack, b_pack,
    label_a="RNN", label_b="LSTM",
    color_a="crimson", color_b="dodgerblue",
    metric_tag="eval/episodic_return",
    plot_each=True,
    output_dir="."
):
    """画图 + 打印汇总 + 存 CSV"""
    sns.set_theme(style="darkgrid", font_scale=1.2, rc={"figure.figsize": (12, 8)})
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    def draw_pack(pack, color, label, zorder=2):
        mean, std, seed_curves, _ = pack
        if mean is not None:
            if plot_each:
                for _, curves in seed_curves.items():
                    for (xs, ys) in curves:
                        ax.plot(xs, ys, color=color, alpha=0.15, linewidth=1, zorder=1)
            ax.plot(xg, mean, color=color, linewidth=2.5, label=f"{label} (mean)", zorder=zorder)
            ax.fill_between(xg, mean - std, mean + std, color=color, alpha=0.20, zorder=zorder-1)

    if a_pack[0] is None and b_pack[0] is None:
        print(f"[{metric_tag}] 没有有效数据可画。")
        return

    if a_pack[0] is not None:
        draw_pack(a_pack, color_a, label_a, zorder=3)
    if b_pack[0] is not None:
        draw_pack(b_pack, color_b, label_b, zorder=2)

    # y 轴标题根据 tag 后缀更友好
    y_label_map = {
        "eval/episodic_return": "episodic_return",
        "eval/episodic_length": "episodic_length",
        "eval/success_times":   "success_times",
    }
    ylab = y_label_map.get(metric_tag, metric_tag.split("/")[-1])

    ax.set_title(f"RNN vs LSTM training performance({metric_tag})", fontsize=16)
    ax.set_xlabel("Environment steps(million)", fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    tag_stub = metric_tag.replace("/", "_")
    png_path = os.path.join(output_dir, f"rnn_vs_lstm_{tag_stub}.png")
    pdf_path = os.path.join(output_dir, f"rnn_vs_lstm_{tag_stub}.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"\n[{metric_tag}] 图已保存：\n- {png_path}\n- {pdf_path}")

    # ===== 打印与导出 CSV 汇总 =====
    def summarize(name, pack):
        mean, std, seed_curves, seed_lastvals = pack
        if mean is None:
            print(f"[{metric_tag}][{name}] 无有效数据")
            return
        print(f"[{metric_tag}][{name}] 最终性能（均值±标准差，基于统一网格最后一点）: {mean[-1]:.3f} ± {std[-1]:.3f}")
        rows = []
        for seed, vals in seed_lastvals.items():
            if len(vals) == 0:
                continue
            mu, sig = float(np.mean(vals)), float(np.std(vals))
            rows.append([seed, mu, sig, len(vals)])
            print(f"  - {seed}: {mu:.3f} ± {sig:.3f}  (runs={len(vals)})")
        if len(rows) > 0:
            df = pd.DataFrame(rows, columns=["seed", "final_mean", "final_std", "runs"])
            csv_path = os.path.join(output_dir, f"{name.lower()}_{tag_stub}_final_per_seed.csv")
            df.to_csv(csv_path, index=False)
            print(f"[{metric_tag}][{name}] 各 seed 最终值 CSV：{csv_path}")

    summarize("RNN", a_pack)
    summarize("LSTM", b_pack)

    # 两基线的最终点对比也导出一份（该指标）
    final_rows = []
    if a_pack[0] is not None:
        final_rows.append([ "RNN", float(a_pack[0][-1]), float(a_pack[1][-1]) ])
    if b_pack[0] is not None:
        final_rows.append([ "LSTM", float(b_pack[0][-1]), float(b_pack[1][-1]) ])
    if len(final_rows) > 0:
        df_final = pd.DataFrame(final_rows, columns=["baseline", "final_mean", "final_std"])
        csv_path = os.path.join(output_dir, f"final_mean_std_rnn_vs_lstm_{tag_stub}.csv")
        df_final.to_csv(csv_path, index=False)
        print(f"[{metric_tag}][总表] 最终点均值±std 已保存：{csv_path}")

    plt.close(fig)  # 关闭当前图，避免多指标时内存积累


# =========================
# 主流程
# =========================
def main():
    # 逐指标处理
    for metric_tag in METRIC_TAGS:
        print(f"\n================ 处理指标：{metric_tag} ================")
        print("==> 读取 RNN 日志...")
        rnn_bag  = collect_curves(RNN_EXPERIMENTS, metric_tag)
        print("==> 读取 LSTM 日志...")
        lstm_bag = collect_curves(LSTM_EXPERIMENTS, metric_tag)

        pack = process_to_grid(
            rnn_bag, lstm_bag,
            smooth_mode=SMOOTH_MODE,
            smooth_w=SMOOTH_WINDOW,
            ema_alpha=EMA_ALPHA,
            grid_points=GRID_POINTS
        )
        if pack[0] is None:
            print(f"[{metric_tag}] 未找到有效的日志或步数范围，跳过。")
            continue

        xg, rnn_pack, lstm_pack = pack
        # 画图 + 导出 CSV
        plot_and_report(
            xg, rnn_pack, lstm_pack,
            label_a="RNN", label_b="LSTM",
            color_a="crimson", color_b="dodgerblue",
            metric_tag=metric_tag,
            plot_each=PLOT_EACH_RUN,
            output_dir=OUTPUT_DIR
        )


if __name__ == "__main__":
    main()
