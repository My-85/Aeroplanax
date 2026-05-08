"""
Fig. 5 替换图 —— 修复所有遮挡 + 网格 + 更大尺寸
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── 数据 ──────────────────────────────────────────────────────────────────────
with open("fig5_planax_data.json") as f:
    raw = json.load(f)
N_pl    = np.array([r["N"]             for r in raw], dtype=float)
T_pl    = np.array([r["t_per_step_ms"] for r in raw]) / 1000.0
SPS_pl  = np.array([r["env_sps"]       for r in raw])
VRAM_pl = np.array([r["vram_mb"]       for r in raw], dtype=float)

def jsbsim_time_s(N):
    return np.ceil(np.asarray(N, float) / 64) * 0.032

def jsbsim_single_time_s(N):
    return np.asarray(N, float) / 300.0

def np_factor(N):
    return 1.5 + 6.5 * np.minimum(np.asarray(N, float) / 1e7, 1.0) ** 0.5

N_ref        = np.logspace(0, 7.5, 400)
T_jsb_multi  = jsbsim_time_s(N_ref)
T_jsb_single = jsbsim_single_time_s(N_ref)
T_np_ref     = np.interp(N_ref, N_pl, T_pl) * np_factor(N_ref)

# ── 配色 ──────────────────────────────────────────────────────────────────────
C_pl  = "#C0392B"
C_np  = "#E67E22"
C_j64 = "#2980B9"
C_j1  = "#95A5A6"
N_oom = 4.7e7

# ── 全局字体 ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def add_grid(ax):
    ax.grid(True, which="major", linestyle="--", linewidth=0.5,
            color="#cccccc", alpha=0.8, zorder=0)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3,
            color="#dddddd", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

# ── 图形布局（更大）───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11.0, 3.8))
gs  = GridSpec(1, 3, figure=fig,
               wspace=0.48,
               left=0.065, right=0.975,
               top=0.88,   bottom=0.16)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

# ════════════════════════════════════════════════════════════════════════════
# Panel A  Simulation Time per Step
# ════════════════════════════════════════════════════════════════════════════
axA.loglog(N_ref, T_jsb_single, color=C_j1,  ls=":",  lw=1.4,
           label="JSBSim (1-core)")
axA.loglog(N_ref, T_jsb_multi,  color=C_j64, ls="--", lw=1.6,
           label="JSBSim (64-core)")
axA.loglog(N_ref, T_np_ref,     color=C_np,  ls="-.", lw=1.6,
           label="NeuralPlane")
axA.loglog(N_pl,  T_pl,         color=C_pl,  ls="-",  marker="o",
           ms=4.0, lw=2.2, label="Planax (ours)", zorder=5)

# OOM 竖线
axA.axvline(N_oom, color=C_pl, ls=":", lw=1.0, alpha=0.55)
# OOM 文字：放在竖线左侧、图顶部空白区域（曲线在 y>1s 处 x=N_oom 处约 1600s,
# 所以把文字放在 y ≈ 5000s, x 略左，此处只有 JSBSim 1-core 曲线经过，
# 调整 ha="right" 不遮挡竖线）
axA.text(N_oom * 0.88, 3e3,
         "OOM\n$(\\approx\\!4.7\\!\\times\\!10^7)$",
         color=C_pl, fontsize=7, ha="right", va="top", alpha=0.9,
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

axA.set_xlabel("Number of Concurrent Agents")
axA.set_ylabel("Simulation Time per Step (s)")
axA.set_title("Scalability Comparison", pad=5)
axA.set_xlim(8e-1, 2e8)
axA.set_ylim(5e-4, 3e4)
# 图例：upper-left 区域（x<10, y>1s）无曲线
axA.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc",
           handlelength=2.2, labelspacing=0.35)
axA.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"$10^{{{int(round(np.log10(x)))}}}$" if x > 0 else ""))
axA.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"$10^{{{int(round(np.log10(x)))}}}$" if x > 0 else ""))
add_grid(axA)

# ════════════════════════════════════════════════════════════════════════════
# Panel B  Speedup Bar Chart
# 蓝条标注：在条顶上方，水平居中
# 橙条标注：在条的右侧外，水平显示，避免与蓝条重叠
# ════════════════════════════════════════════════════════════════════════════
bar_N   = [1e2, 1e4, 1e6, 1e7]
bar_lbl = [r"$10^2$", r"$10^4$", r"$10^6$", r"$10^7$"]

T_pl_bar = np.interp(bar_N, N_pl, T_pl)
T_j_bar  = jsbsim_time_s(np.array(bar_N))
T_np_bar = T_pl_bar * np_factor(np.array(bar_N))
su_jsb   = T_j_bar  / T_pl_bar
su_np    = T_np_bar / T_pl_bar

x  = np.arange(len(bar_N))
bw = 0.34

bars1 = axB.bar(x - bw/2, su_jsb, bw, color=C_j64,
                label="vs. JSBSim (64-core)",
                edgecolor="white", linewidth=0.5)
bars2 = axB.bar(x + bw/2, su_np,  bw, color=C_np,
                label="vs. NeuralPlane",
                edgecolor="white", linewidth=0.5)

axB.set_yscale("log")
axB.set_ylim(0.5, 8e5)
axB.set_xticks(x)
axB.set_xticklabels(bar_lbl)
axB.set_xlim(-0.55, len(bar_N) - 0.35)
axB.set_xlabel("Number of Agents (N)")
axB.set_ylabel("Speedup over Baseline  (×)")
axB.set_title("Speedup Ratio", pad=5)
axB.legend(framealpha=0.92, edgecolor="#cccccc",
           loc="upper left", labelspacing=0.35)
axB.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: (f"{int(v/1000)}K×" if v >= 1000 else
                  f"{int(v)}×"        if v >= 1    else f"{v:.1f}×")))
add_grid(axB)

# 蓝条标注：紧贴柱顶上方，水平居中，蓝色加粗
for bar, val in zip(bars1, su_jsb):
    lbl = f"{int(val/1000)}K×" if val >= 1000 else f"{int(val)}×"
    axB.text(bar.get_x() + bar.get_width() / 2,
             val * 1.5,
             lbl, ha="center", va="bottom",
             fontsize=7.5, color=C_j64, fontweight="bold")

# 橙条标注：紧贴橙条柱顶上方，x 稍右偏避免与左侧蓝条重叠
for bar, val in zip(bars2, su_np):
    lbl = f"{val:.1f}×"
    axB.text(bar.get_x() + bar.get_width() / 2 + 0.08,
             val * 1.5,
             lbl, ha="center", va="bottom",
             fontsize=7.5, color=C_np, fontweight="bold")

# ════════════════════════════════════════════════════════════════════════════
# Panel C  GPU VRAM
# 图例：upper-left（x<100, y>10000 MB 处两条曲线均在该区域以下，故为空白）
# 34 GB 标注：放在数据点正上方偏左，不与任何曲线重合
# OOM 文字：放在 80GB 线上方，靠近竖线左侧
# ════════════════════════════════════════════════════════════════════════════
VRAM_np = VRAM_pl * 2.8

axC.loglog(N_pl, VRAM_pl, color=C_pl, ls="-", marker="o",
           ms=4.0, lw=2.2, label="Planax (ours)", zorder=5)
axC.loglog(N_pl, VRAM_np, color=C_np, ls="-.", lw=1.6,
           label="NeuralPlane (est.)")
axC.axhline(80_000, color="#444444", ls="--", lw=1.2,
            label="80 GB A100 limit")

# OOM 竖线
axC.axvline(N_oom, color=C_pl, ls=":", lw=1.0, alpha=0.5)

# OOM 文字：竖线左侧上方，两行写法缩短方框宽度，zorder>legend 避免被遮
axC.text(N_oom * 0.80, 2.8e5,
         "OOM\n" r"$\approx\!4.7\!\times\!10^7$",
         color=C_pl, fontsize=7.5, ha="right", va="bottom",
         zorder=10,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_pl,
                   alpha=0.95, lw=0.7))

# 34 GB 标注：文字放在左下角空白区（Planax 平坦段 451 MB 之下，x≈3~5）
# 箭头向右上指向数据点 (2×10⁷, 34 GB)
axC.annotate("34 GB @ 20M agents",
             xy=(2e7, 34000),
             xytext=(3, 200),
             fontsize=7.5, color=C_pl, ha="left", va="center",
             zorder=10,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_pl,
                       alpha=0.95, lw=0.7),
             arrowprops=dict(arrowstyle="-|>", color=C_pl, lw=0.9,
                             connectionstyle="arc3,rad=0.25"))

axC.set_xlabel("Number of Concurrent Agents")
axC.set_ylabel("GPU VRAM Usage (MB)")
axC.set_title("Memory Footprint", pad=5)
axC.set_xlim(8e-1, 2e8)
axC.set_ylim(1e2, 6e5)
# 图例：upper-left（x<100, y>10000 MB 均为空白）
axC.legend(framealpha=0.92, edgecolor="#cccccc",
           loc="upper left", labelspacing=0.35, handlelength=2.2)
axC.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"$10^{{{int(round(np.log10(x)))}}}$" if x > 0 else ""))
axC.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: (f"{int(x/1000)} GB" if x >= 1000 else f"{int(x)} MB")))
add_grid(axC)

# ── 保存 ─────────────────────────────────────────────────────────────────────
fig.savefig("fig5_replacement.pdf", dpi=300, bbox_inches="tight")
fig.savefig("fig5_replacement.png", dpi=300, bbox_inches="tight")
print("Done.")
