"""
Architecture comparison figures for IEEE RA-L.
  bottleneck_traditional.pdf  – Traditional CPU-Centric MARL Pipeline
  bottleneck_planax.pdf       – Planax GPU-Resident Architecture
Transparent background, single-column width, no (a)/(b) labels.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT   = os.path.dirname(os.path.abspath(__file__))
C_RED   = "#FFCCCC"
C_BLUE  = "#CCE5FF"
C_GREEN = "#CCFFCC"
C_PCIE  = "#CC2222"
C_DARK  = "#222222"


def rbox(ax, cx, cy, w, h, text, fc, ec="#444444", lw=1.3,
         fs=8.5, bold=False, ls="-", zorder=3, tc="black"):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.025",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=ls, zorder=zorder,
    ))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=tc, zorder=zorder + 1, multialignment="center",
            linespacing=1.4)


def region_bg(ax, cx, cy, w, h, fc, ec, ls="--", lw=1.0, zorder=1):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=ls, zorder=zorder,
    ))


def solid_arr(ax, x1, y1, x2, y2, color=C_DARK, lw=1.5, zorder=5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=13), zorder=zorder)


def bidir_dashed(ax, x1, y1, x2, y2, color=C_PCIE, lw=2.6, zorder=5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=lw,
                                linestyle=(0, (5, 2)), mutation_scale=16),
                zorder=zorder)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Traditional CPU-Centric MARL Pipeline
# ─────────────────────────────────────────────────────────────────────────────
#
# Vertical layout (y increases upward, data range [0, 1]):
#
#   0.972  ← Title (va="top"; text bottom ≈ 0.945)
#            ← gap ≈ 0.017
#   0.928  ← CPU region top
#   0.916    "Host (CPU)" label (top of text; zorder=6 above all patches)
#   0.882    CPU main box top  (cy=0.832, h=0.100)
#   0.782    CPU main box bottom
#            ← connector arrow  0.782 → 0.631
#   0.730  ← PCIe region top
#   0.708    PCIe header line 1
#   0.678    PCIe header line 2
#   0.628    bidir arrow top
#   0.564    bidir arrow mid  (H2D@0.619 left / D2H@0.509 left / Latency@0.564 right)
#   0.500    bidir arrow bottom
#            ← connector arrow  0.435 → 0.362
#   0.435  ← PCIe region bottom
#   0.405  ← GPU region top
#   0.393    "Device (GPU)" label top (zorder=6)
#   0.360    GPU main box top   (cy=0.310, h=0.100)
#   0.260    GPU main box bottom
#   0.220  ← GPU region bottom
#            feedback arc x=0.888: (0.888,0.782)↔(0.888,0.360), rad=-0.22
#            "Repeat every step" rotated 90° at x=0.970
#
#   0.152    bottom note
#   0.108    formula
# ─────────────────────────────────────────────────────────────────────────────
def make_traditional():
    fig, ax = plt.subplots(figsize=(3.3, 5.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)

    cx  = 0.50
    BW  = 0.680   # main box width
    BH  = 0.100   # main box height
    RW  = 0.820   # region background width

    # ── CPU region ───────────────────────────────────────────────────
    # CPU_HI=0.905: actual drawn border top = 0.905+pad(0.015)=0.920
    # Title bottom ≈ 0.945  →  gap ≈ 0.025 (clearly separated at print res)
    # cpu_cy=0.816: box top=0.866; label bottom≈0.872 → small gap above box ✓
    CPU_HI, CPU_LO = 0.905, 0.748
    cpu_cy = 0.816                      # box center
    region_bg(ax, cx, (CPU_HI+CPU_LO)/2, RW, CPU_HI-CPU_LO,
              "#FFF0F0", "#CC3333")
    # Section label: zorder=6 so it always appears above all patches
    ax.text(cx - RW/2 + 0.018, CPU_HI - 0.014,
            "Host (CPU)",
            fontsize=7, color="#CC3333", fontweight="bold",
            ha="left", va="top", zorder=6)
    rbox(ax, cx, cpu_cy, BW, BH,
         "CPU: Environment Simulator\n(e.g. JSBSim)",
         fc=C_RED, ec="#CC3333", lw=1.5)

    # ── PCIe region ───────────────────────────────────────────────────
    PCIE_HI, PCIE_LO = 0.730, 0.435

    # connector: CPU box bottom → PCIe zone top border (stops before header text)
    solid_arr(ax, cx, cpu_cy - BH/2, cx, PCIE_HI - 0.002)

    region_bg(ax, cx, (PCIE_HI+PCIE_LO)/2, RW, PCIE_HI-PCIE_LO,
              "#FFF5F5", C_PCIE, ls=":", lw=1.3)

    # Two-line header, inside the PCIe zone top (well above arrow top at 0.628)
    ax.text(cx, PCIE_HI - 0.022,
            "Host-to-Device / Device-to-Host",
            ha="center", va="top", fontsize=7.3,
            color=C_PCIE, fontweight="bold", zorder=3)
    ax.text(cx, PCIE_HI - 0.053,
            "(PCIe Bottleneck)",
            ha="center", va="top", fontsize=7.0,
            color=C_PCIE, fontweight="bold", zorder=3)

    # Bidirectional dashed arrow
    ARR_T = 0.628
    ARR_B = PCIE_LO + 0.065   # 0.500
    ARR_M = (ARR_T + ARR_B) / 2  # 0.564
    bidir_dashed(ax, cx, ARR_T, cx, ARR_B)

    # H2D (upper-left), D2H (lower-left): ha="right" so text stays left of arrow
    ax.text(cx - 0.090, ARR_M + 0.055,
            "H2D  obs/reward",
            ha="right", va="center",
            fontsize=6.8, color=C_PCIE, style="italic", zorder=4)
    ax.text(cx - 0.090, ARR_M - 0.055,
            "D2H  actions",
            ha="right", va="center",
            fontsize=6.8, color=C_PCIE, style="italic", zorder=4)
    # Latency (right of arrow)
    ax.text(cx + 0.090, ARR_M,
            "Latency\n(bottleneck)",
            ha="left", va="center",
            fontsize=8.0, color=C_PCIE, fontweight="bold", zorder=4)

    # connector: PCIe bottom → GPU box top
    solid_arr(ax, cx, PCIE_LO, cx, 0.362)

    # ── GPU region ────────────────────────────────────────────────────
    GPU_HI, GPU_LO = 0.405, 0.220
    gpu_cy = 0.310                      # box center
    region_bg(ax, cx, (GPU_HI+GPU_LO)/2, RW, GPU_HI-GPU_LO,
              "#EEF5FF", "#3366CC")
    # Section label: zorder=6, positioned inside region top, above main box (box top = 0.360)
    ax.text(cx - RW/2 + 0.018, GPU_HI - 0.012,
            "Device (GPU)",
            fontsize=7, color="#3366CC", fontweight="bold",
            ha="left", va="top", zorder=6)
    rbox(ax, cx, gpu_cy, BW, BH,
         "GPU: Neural Network Policy\n(PPO / MAPPO)",
         fc=C_BLUE, ec="#3366CC", lw=1.5)

    # ── feedback arc (right side) ──────────────────────────────────────
    # Arc from GPU box top (0.360) up to CPU box bottom (0.782), bows right
    X_ARC = cx + 0.388   # = 0.888
    ax.annotate("",
                xy     = (X_ARC, cpu_cy - BH/2),   # destination: CPU box bottom
                xytext = (X_ARC, gpu_cy + BH/2),   # source:      GPU box top
                arrowprops=dict(arrowstyle="<|-|>",
                                color="#888888", lw=1.2,
                                connectionstyle="arc3,rad=-0.22",
                                mutation_scale=10),
                zorder=4)
    # Vertical label sits to the right of the arc (x=0.970 << 1.0 ✓)
    ax.text(0.970, (cpu_cy + gpu_cy) / 2,
            "Repeat every step",
            ha="center", va="center", rotation=90,
            fontsize=6.2, color="#888888", style="italic", zorder=5)

    # ── title ──────────────────────────────────────────────────────────
    # y=0.972, text bottom ≈ 0.945; CPU_HI=0.928 → gap ≈ 0.017  ✓
    ax.text(cx, 0.972,
            "Traditional CPU-Centric MARL Pipeline",
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=C_DARK, zorder=6)

    # ── bottom notes ────────────────────────────────────────────────────
    ax.text(cx, 0.152,
            "PCIe transfers dominate step latency",
            ha="center", va="center",
            fontsize=7.5, color="#555555", style="italic", zorder=5)
    ax.text(cx, 0.108,
            r"$t_{\mathrm{step}} = t_{\mathrm{sim}} + "
            r"t_{\mathrm{H2D}} + t_{\mathrm{inf}} + t_{\mathrm{D2H}}$",
            ha="center", va="center",
            fontsize=8.0, color="#555555", zorder=5)

    out = os.path.join(OUT, "bottleneck_traditional.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight",
                transparent=True, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Planax GPU-Resident Architecture
# ─────────────────────────────────────────────────────────────────────────────
#
# Vertical layout (y increases upward, data range [0, 1]):
#
#   0.972  ← Title (bottom ≈ 0.945)
#            ← gap ≈ 0.035
#   0.910  ← Outer GPU frame top  (O_HI)
#   0.882    "Fully GPU-Resident (Zero-Transfer)" label top
#            ← outer label bottom ≈ 0.859; S_HI=0.845 → gap ≈ 0.014  ✓
#   0.845  ← XLA sub-frame top  (S_HI)
#   0.820    "XLA Fused Kernel" label top  (bottom ≈ 0.797)
#            ← CTOP = S_HI - 0.065 = 0.780  (below XLA label)
#   0.663  ← 6-DOF Dynamics box center (y1); box [0.616, 0.710]
#   0.579    bidir arrow mid (label "VRAM Shared…" at cx+0.04)
#   0.495  ← Tensorized Aero LUT box center (y2)
#   0.411    bidir arrow mid
#   0.327  ← Policy Optimization box center (y3); box [0.280, 0.374]
#            ← CBOT = S_LO + 0.035 = 0.215  → bottom margin ≈ 0.065  ✓
#   0.180  ← XLA sub-frame bottom  (S_LO)
#   0.150    "GPU VRAM" note  (between outer=0.120 and XLA=0.180)
#   0.120  ← Outer frame bottom  (O_LO)
#
#   0.076    bottom note  (below outer frame)
#   0.035    formula
#
#   Left side feedback arc x=0.220 (curves left, min_x≈0.175)
#   Label "action→state" at x=0.125 ha="right" (right_edge=0.125 < arc_min=0.175) ✓
# ─────────────────────────────────────────────────────────────────────────────
def make_planax():
    fig, ax = plt.subplots(figsize=(3.3, 5.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)

    cx    = 0.50
    BW    = 0.570   # inner box width (leaves room on both sides)
    BH    = 0.095   # inner box height
    GREEN = "#228B22"

    # ── title ──────────────────────────────────────────────────────────
    ax.text(cx, 0.972,
            "Planax GPU-Resident Architecture",
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=C_DARK, zorder=6)

    # ── outer GPU-resident frame ───────────────────────────────────────
    O_LO, O_HI, O_W = 0.120, 0.910, 0.88
    ax.add_patch(FancyBboxPatch(
        (cx - O_W/2, O_LO), O_W, O_HI - O_LO,
        boxstyle="round,pad=0.02",
        facecolor="#F0FFF0", edgecolor="#1A7A1A",
        linewidth=2.2, zorder=1,
    ))
    # Outer frame label: top of text at O_HI-0.028=0.882
    # text height≈0.023; bottom≈0.859; S_HI=0.845 → gap≈0.014 ✓
    ax.text(cx, O_HI - 0.028,
            "Fully GPU-Resident  (Zero-Transfer)",
            ha="center", va="top", fontsize=8.2,
            color="#1A7A1A", fontweight="bold", zorder=6)

    # ── XLA fused kernel sub-frame ────────────────────────────────────
    # S_HI lowered to 0.818 so XLA top border (0.818+0.018=0.836) stays
    # clearly below the outer-frame label bottom (≈ 0.859).
    S_LO, S_HI, S_W = 0.180, 0.818, 0.75
    ax.add_patch(FancyBboxPatch(
        (cx - S_W/2, S_LO), S_W, S_HI - S_LO,
        boxstyle="round,pad=0.018",
        facecolor="#E8FFE8", edgecolor="#2CA02C",
        linewidth=1.5, linestyle="--", zorder=2,
    ))
    # XLA label: inside sub-frame top
    ax.text(cx, S_HI - 0.022,
            "XLA Fused Kernel",
            ha="center", va="top", fontsize=8.0,
            color="#2CA02C", fontweight="bold", zorder=6)

    # ── compute evenly-spaced box centers ─────────────────────────────
    CTOP = S_HI - 0.068   # below XLA label
    CBOT = S_LO + 0.035   # 0.215
    AVAIL = CTOP - CBOT
    GAP   = (AVAIL - 3 * BH) / 4

    y1 = CTOP - GAP - BH / 2
    y2 = y1   - BH / 2 - GAP - BH / 2
    y3 = y2   - BH / 2 - GAP - BH / 2

    rbox(ax, cx, y1, BW, BH,
         "6-DOF Dynamics\n(RK4 Integration)",
         fc=C_GREEN, ec="#2CA02C", lw=1.4, zorder=4)
    rbox(ax, cx, y2, BW, BH,
         "Tensorized Aero LUT\n(Batched Lookup)",
         fc=C_GREEN, ec="#2CA02C", lw=1.4, zorder=4)
    rbox(ax, cx, y3, BW, BH,
         "Policy Optimization\n(PPO / MAPPO)",
         fc=C_BLUE, ec="#3366CC", lw=1.4, zorder=4)

    # ── arrows between boxes ───────────────────────────────────────────
    G = BH / 2 + 0.006   # from box edge to arrow tip

    a1_t = y1 - G;  a1_b = y2 + G;  a1_m = (a1_t + a1_b) / 2  # ≈ 0.579
    a2_t = y2 - G;  a2_b = y3 + G;  a2_m = (a2_t + a2_b) / 2  # ≈ 0.411

    solid_arr(ax, cx, a1_t, cx, a1_b, color=GREEN, lw=2.0)
    solid_arr(ax, cx, a2_t, cx, a2_b, color=GREEN, lw=2.0)

    # Arrow labels: to the RIGHT of arrow (ha="left" at cx+0.04=0.54)
    # "VRAM Shared /" = 13 chars × ~0.010 units → right edge ≈ 0.67 < 1.0 ✓
    for mid_y in (a1_m, a2_m):
        ax.text(cx + 0.040, mid_y,
                "VRAM Shared /\nNo PCIe Transfer",
                ha="left", va="center",
                fontsize=6.3, color=GREEN, style="italic",
                zorder=5, linespacing=1.3)

    # ── feedback arc (RIGHT side: Policy → Dynamics) ──────────────────
    # Placed between box right edge (cx+BW/2=0.785) and XLA frame right
    # inner edge (cx+S_W/2=0.875).  Arc bows right (rad=-0.20).
    # max x of arc ≈ 0.840 + 0.20*(y1-G - (y3+G))/2 ≈ 0.840+0.022 = 0.862
    # XLA frame right inner = 0.875  ✓  (arc stays inside the frame)
    X_ARC = cx + 0.340   # 0.840
    ax.annotate("",
                xy     = (X_ARC, y1 - G),   # destination: bottom of 6-DOF box
                xytext = (X_ARC, y3 + G),   # source:      top of Policy box
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5,
                                connectionstyle="arc3,rad=-0.22",
                                mutation_scale=12),
                zorder=5)
    # Short vertical label between XLA inner border and outer frame
    ax.text(cx + S_W/2 + 0.012, (y1 + y3) / 2,
            "step\nloop",
            ha="left", va="center",
            fontsize=6.0, color=GREEN, style="italic",
            zorder=6, linespacing=1.3)

    # ── GPU VRAM note (between outer frame bottom and XLA sub-frame bottom) ──
    # y = (O_LO + S_LO) / 2 = (0.120 + 0.180) / 2 = 0.150
    ax.text(cx, (O_LO + S_LO) / 2,
            "GPU VRAM  (Shared Memory Pool)",
            ha="center", va="center",
            fontsize=7.0, color="#1A7A1A", style="italic", zorder=6)

    # ── bottom notes (outside outer frame) ────────────────────────────
    ax.text(cx, 0.076,
            "All computation fused in a single GPU kernel launch",
            ha="center", va="center",
            fontsize=7.5, color="#228B22", style="italic", zorder=5)
    ax.text(cx, 0.035,
            r"$t_{\mathrm{step}} \approx t_{\mathrm{kernel}}$"
            r"$\quad (t_{\mathrm{H2D}} = t_{\mathrm{D2H}} \approx 0)$",
            ha="center", va="center",
            fontsize=8.0, color="#228B22", zorder=5)

    out = os.path.join(OUT, "bottleneck_planax.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight",
                transparent=True, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_traditional()
    make_planax()
    print("Done.")
