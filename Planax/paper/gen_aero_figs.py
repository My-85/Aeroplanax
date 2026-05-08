#!/usr/bin/env python3
"""
IEEE-style aerodynamic comparison figures for the Planax paper.

Outputs (paper/ directory):
  aero_speed_comparison.pdf
  aero_fidelity_comparison.pdf
  aero_coeff_curves.pdf
"""

import sys, os, time, warnings
import numpy as np

TORCH_SITE = (
    "/home/dqy/aeroplanax/new/20251215最新代码库/"
    "autoresearch/autoresearch/.venv/lib/python3.10/site-packages"
)
sys.path.insert(0, TORCH_SITE)

import torch
import jax
import jax.numpy as jnp
from jax import jit
from scipy.interpolate import RegularGridInterpolator
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── IEEE serif style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "STIXGeneral",
    "mathtext.fontset":  "stix",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.linewidth":    1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "figure.dpi":        150,
})

ROOT       = "/home/dqy/aeroplanax/new/20251215最新代码库"
PLANAX_DIR = os.path.join(ROOT, "Planax")
NEURAL_DIR = os.path.join(ROOT, "neuralplane/NeuralPlane")
DATA_DIR   = os.path.join(PLANAX_DIR, "envs/core/simulators/fighterplane/data")
OUT_DIR    = os.path.join(PLANAX_DIR, "paper")

sys.path.insert(0, PLANAX_DIR)
sys.path.insert(0, NEURAL_DIR)

# ── Palette ───────────────────────────────────────────────────────────
C_GT    = "#000000"          # black  – ground truth
C_LUT   = "#1A6B1A"          # dark green – Planax JAX-LUT
C_MLP   = "#CC2222"          # dark red   – NeuralPlane MLP
C_BAR_L = "#1F3F7A"          # navy blue  – Planax bars
C_BAR_M = "#8B1A1A"          # dark red   – NeuralPlane bars


# ══════════════════════════════════════════════════════════════════════
# 1.  Load raw F-16 tables
# ══════════════════════════════════════════════════════════════════════
print("Loading F-16 aerodynamic tables …")

def _dat(name):
    return np.loadtxt(os.path.join(DATA_DIR, name))

ALPHA1 = _dat("ALPHA1.dat")
BETA1  = _dat("BETA1.dat")
DH1    = _dat("DH1.dat")
DH2    = _dat("DH2.dat")

Cx_raw = _dat("CX0120_ALPHA1_BETA1_DH1_201.dat").reshape(len(DH1), len(BETA1), len(ALPHA1))
Cz_raw = _dat("CZ0120_ALPHA1_BETA1_DH1_301.dat").reshape(len(DH1), len(BETA1), len(ALPHA1))
Cm_raw = _dat("CM0120_ALPHA1_BETA1_DH1_101.dat").reshape(len(DH1), len(BETA1), len(ALPHA1))
Cy_raw = _dat("CY0320_ALPHA1_BETA1_401.dat").reshape(len(BETA1), len(ALPHA1))
Cn_raw = _dat("CN0120_ALPHA1_BETA1_DH2_501.dat").reshape(len(DH2), len(BETA1), len(ALPHA1))
Cl_raw = _dat("CL0120_ALPHA1_BETA1_DH2_601.dat").reshape(len(DH2), len(BETA1), len(ALPHA1))

gt_Cx = RegularGridInterpolator((DH1, BETA1, ALPHA1), Cx_raw, method="linear",
                                 bounds_error=False, fill_value=None)
gt_Cz = RegularGridInterpolator((DH1, BETA1, ALPHA1), Cz_raw, method="linear",
                                 bounds_error=False, fill_value=None)
gt_Cm = RegularGridInterpolator((DH1, BETA1, ALPHA1), Cm_raw, method="linear",
                                 bounds_error=False, fill_value=None)
gt_Cy = RegularGridInterpolator((BETA1, ALPHA1), Cy_raw, method="linear",
                                 bounds_error=False, fill_value=None)
gt_Cn = RegularGridInterpolator((DH2, BETA1, ALPHA1), Cn_raw, method="linear",
                                 bounds_error=False, fill_value=None)
gt_Cl = RegularGridInterpolator((DH2, BETA1, ALPHA1), Cl_raw, method="linear",
                                 bounds_error=False, fill_value=None)


# ══════════════════════════════════════════════════════════════════════
# 2.  Planax JAX-LUT (batched, JIT-compiled)
# ══════════════════════════════════════════════════════════════════════
print("Building Planax JAX-LUT …")

_DH1j = jnp.array(DH1);  _DH2j = jnp.array(DH2)
_A1j  = jnp.array(ALPHA1); _B1j = jnp.array(BETA1)
_Cxj  = jnp.array(Cx_raw); _Czj = jnp.array(Cz_raw)
_Cmj  = jnp.array(Cm_raw); _Cyj = jnp.array(Cy_raw)
_Cnj  = jnp.array(Cn_raw); _Clj = jnp.array(Cl_raw)


def _tri(gx, gy, gz, v, xs, ys, zs):
    ix = jnp.clip(jnp.searchsorted(gx, xs) - 1, 0, len(gx) - 2)
    iy = jnp.clip(jnp.searchsorted(gy, ys) - 1, 0, len(gy) - 2)
    iz = jnp.clip(jnp.searchsorted(gz, zs) - 1, 0, len(gz) - 2)
    xd = (xs - gx[ix]) / (gx[ix+1] - gx[ix])
    yd = (ys - gy[iy]) / (gy[iy+1] - gy[iy])
    zd = (zs - gz[iz]) / (gz[iz+1] - gz[iz])
    c000 = v[ix,   iy,   iz  ]; c100 = v[ix+1, iy,   iz  ]
    c010 = v[ix,   iy+1, iz  ]; c110 = v[ix+1, iy+1, iz  ]
    c001 = v[ix,   iy,   iz+1]; c101 = v[ix+1, iy,   iz+1]
    c011 = v[ix,   iy+1, iz+1]; c111 = v[ix+1, iy+1, iz+1]
    c0 = (c000*(1-xd)+c100*xd)*(1-yd) + (c010*(1-xd)+c110*xd)*yd
    c1 = (c001*(1-xd)+c101*xd)*(1-yd) + (c011*(1-xd)+c111*xd)*yd
    return c0*(1-zd) + c1*zd


def _bi(gx, gy, v, xs, ys):
    ix = jnp.clip(jnp.searchsorted(gx, xs) - 1, 0, len(gx) - 2)
    iy = jnp.clip(jnp.searchsorted(gy, ys) - 1, 0, len(gy) - 2)
    xd = (xs - gx[ix]) / (gx[ix+1] - gx[ix])
    yd = (ys - gy[iy]) / (gy[iy+1] - gy[iy])
    return ((v[ix, iy]*(1-xd)+v[ix+1, iy]*xd)*(1-yd) +
            (v[ix, iy+1]*(1-xd)+v[ix+1, iy+1]*xd)*yd)


@jit
def _lut_C(a, b, e):
    return (_tri(_DH1j, _B1j, _A1j, _Cxj, e, b, a),
            _tri(_DH1j, _B1j, _A1j, _Czj, e, b, a),
            _tri(_DH1j, _B1j, _A1j, _Cmj, e, b, a),
            _bi (_B1j,  _A1j,       _Cyj, b, a),
            _tri(_DH2j, _B1j, _A1j, _Cnj, e, b, a),
            _tri(_DH2j, _B1j, _A1j, _Clj, e, b, a))


def lut_eval(a_np, b_np, e_np):
    r = _lut_C(jnp.array(a_np), jnp.array(b_np), jnp.array(e_np))
    jax.block_until_ready(r)
    return tuple(np.array(x) for x in r)


# warm-up
_d = np.zeros(16, dtype=np.float32)
lut_eval(_d, _d, _d)
print("  JAX JIT compiled.")


# ══════════════════════════════════════════════════════════════════════
# 3.  NeuralPlane PyTorch-MLP
# ══════════════════════════════════════════════════════════════════════
print("Loading NeuralPlane MLP …")
TORCH_DEV = "cuda" if torch.cuda.is_available() else "cpu"

spec = importlib.util.spec_from_file_location(
    "hifi_F16_AeroData",
    os.path.join(NEURAL_DIR, "envs/models/F16/hifi_F16_AeroData.py"))
hifi_mod = importlib.util.module_from_spec(spec)
hifi_mod.device = TORCH_DEV
spec.loader.exec_module(hifi_mod)

mlp_model = hifi_mod.hifi_F16(device=TORCH_DEV)
for attr in dir(mlp_model):
    obj = getattr(mlp_model, attr)
    if isinstance(obj, torch.nn.Module):
        obj.eval()


def mlp_eval(a_np, b_np, e_np):
    a = torch.tensor(a_np, dtype=torch.float32, device=TORCH_DEV)
    b = torch.tensor(b_np, dtype=torch.float32, device=TORCH_DEV)
    e = torch.tensor(e_np, dtype=torch.float32, device=TORCH_DEV)
    with torch.no_grad():
        r = mlp_model.hifi_C(a, b, e)
    if TORCH_DEV.startswith("cuda"):
        torch.cuda.synchronize()
    return tuple(x.cpu().numpy() for x in r)


mlp_eval(_d, _d, _d)
print("  MLP ready.")


# ══════════════════════════════════════════════════════════════════════
# 4.  Test points & ground-truth evaluation
# ══════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(2025)
N   = 10000
a_t = rng.uniform(ALPHA1.min(), ALPHA1.max(), N).astype(np.float32)
b_t = rng.uniform(BETA1.min(),  BETA1.max(),  N).astype(np.float32)
e_t = rng.uniform(DH1.min(),    DH1.max(),    N).astype(np.float32)

pts3_dh1 = np.stack([e_t, b_t, a_t], axis=-1)
pts3_dh2 = np.stack([e_t, b_t, a_t], axis=-1)
pts2     = np.stack([b_t, a_t], axis=-1)

NAMES = ["Cx", "Cz", "Cm", "Cy", "Cn", "Cl"]
gt_vals = {
    "Cx": gt_Cx(pts3_dh1), "Cz": gt_Cz(pts3_dh1), "Cm": gt_Cm(pts3_dh1),
    "Cy": gt_Cy(pts2),
    "Cn": gt_Cn(pts3_dh2), "Cl": gt_Cl(pts3_dh2),
}

lut_r = dict(zip(NAMES, lut_eval(a_t, b_t, e_t)))
mlp_r = dict(zip(NAMES, mlp_eval(a_t, b_t, e_t)))

lut_rmse = {c: float(np.sqrt(np.mean((lut_r[c] - gt_vals[c])**2))) for c in NAMES}
mlp_rmse = {c: float(np.sqrt(np.mean((mlp_r[c] - gt_vals[c])**2))) for c in NAMES}

print("\nFidelity (RMSE vs ground truth):")
for c in NAMES:
    ratio = mlp_rmse[c] / lut_rmse[c] if lut_rmse[c] > 0 else float("inf")
    print(f"  {c}: LUT={lut_rmse[c]:.2e}  MLP={mlp_rmse[c]:.2e}  ratio={ratio:.1e}")


# ══════════════════════════════════════════════════════════════════════
# 5.  Speed benchmark
# ══════════════════════════════════════════════════════════════════════
BATCHES   = [1, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
N_REPS    = 40
lut_ms, mlp_ms = [], []

print("\nSpeed benchmark …")
print(f"{'Batch':>6}  {'LUT ms':>9}  {'MLP ms':>9}  {'Speedup':>8}")
for B in BATCHES:
    a, b, e = a_t[:B], b_t[:B], e_t[:B]
    for _ in range(6): lut_eval(a, b, e)
    t0 = time.perf_counter()
    for _ in range(N_REPS): lut_eval(a, b, e)
    tl = (time.perf_counter() - t0) / N_REPS * 1e3

    for _ in range(6): mlp_eval(a, b, e)
    if TORCH_DEV.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        mlp_eval(a, b, e)
        if TORCH_DEV.startswith("cuda"): torch.cuda.synchronize()
    tm = (time.perf_counter() - t0) / N_REPS * 1e3

    lut_ms.append(tl); mlp_ms.append(tm)
    print(f"{B:>6}  {tl:>9.3f}  {tm:>9.3f}  {tm/tl:>7.2f}x")

lut_sps = [B / (t * 1e-3) for B, t in zip(BATCHES, lut_ms)]
mlp_sps = [B / (t * 1e-3) for B, t in zip(BATCHES, mlp_ms)]


# ══════════════════════════════════════════════════════════════════════
# 6.  Alpha-sweep curves (β=0, δe=0)
# ══════════════════════════════════════════════════════════════════════
N_SW = 400
a_sw = np.linspace(ALPHA1.min(), ALPHA1.max(), N_SW, dtype=np.float32)
b_sw = np.zeros(N_SW, dtype=np.float32)
e_sw = np.zeros(N_SW, dtype=np.float32)

pts3_sw = np.stack([e_sw, b_sw, a_sw], axis=-1)
pts2_sw = np.stack([b_sw, a_sw], axis=-1)
gt_sw = {
    "Cx": gt_Cx(pts3_sw), "Cz": gt_Cz(pts3_sw), "Cm": gt_Cm(pts3_sw),
    "Cy": gt_Cy(pts2_sw),
    "Cn": gt_Cn(pts3_sw), "Cl": gt_Cl(pts3_sw),
}
lut_sw = dict(zip(NAMES, lut_eval(a_sw, b_sw, e_sw)))
mlp_sw = dict(zip(NAMES, mlp_eval(a_sw, b_sw, e_sw)))


# ══════════════════════════════════════════════════════════════════════
# 7.  Figure A – Speed comparison
# ══════════════════════════════════════════════════════════════════════
print("\nGenerating aero_speed_comparison.pdf …")

fig_s, (ax_l, ax_t) = plt.subplots(1, 2, figsize=(8.5, 3.8))
fig_s.subplots_adjust(top=0.82, wspace=0.32)

kw_lut = dict(color=C_LUT, lw=2.0, marker="o", ms=5, label="Planax (JAX-LUT)")
kw_mlp = dict(color=C_MLP, lw=2.0, marker="s", ms=5, ls="--",
              label="NeuralPlane (PyTorch-MLP)")

ax_l.plot(BATCHES, lut_ms, **kw_lut)
ax_l.plot(BATCHES, mlp_ms, **kw_mlp)
ax_l.set_xscale("log"); ax_l.set_yscale("log")
ax_l.set_xlabel("Batch Size")
ax_l.set_ylabel("Latency (ms)")
ax_l.set_title("(a) Inference Latency")
ax_l.tick_params(which="both", direction="in")

ax_t.plot(BATCHES, lut_sps, **kw_lut)
ax_t.plot(BATCHES, mlp_sps, **kw_mlp)
ax_t.set_xscale("log"); ax_t.set_yscale("log")
ax_t.set_xlabel("Batch Size")
ax_t.set_ylabel("Throughput (samples/s)")
ax_t.set_title("(b) Throughput")
ax_t.tick_params(which="both", direction="in")

# single shared legend at top of figure
handles, labels = ax_l.get_legend_handles_labels()
fig_s.legend(handles, labels, loc="upper center", ncol=2,
             fontsize=11, frameon=True, framealpha=0.9,
             bbox_to_anchor=(0.5, 0.97))

fig_s.suptitle("Aerodynamic Model Computational Performance",
               fontsize=14, fontweight="bold", y=1.04)

out_s = os.path.join(OUT_DIR, "aero_speed_comparison.pdf")
fig_s.savefig(out_s, format="pdf", bbox_inches="tight", transparent=True, dpi=300)
plt.close(fig_s)
print(f"  Saved: {out_s}")


# ══════════════════════════════════════════════════════════════════════
# 8.  Figure B – Fidelity comparison
# ══════════════════════════════════════════════════════════════════════
print("Generating aero_fidelity_comparison.pdf …")

LABELS_TEX = [r"$C_x$", r"$C_z$", r"$C_m$", r"$C_y$", r"$C_n$", r"$C_l$"]
x = np.arange(len(NAMES))
w = 0.36

fig_f, ax_f = plt.subplots(figsize=(7.0, 4.2))

bars_l = ax_f.bar(x - w/2, [lut_rmse[c] for c in NAMES], w,
                  color=C_BAR_L, label="Planax (JAX-LUT)", alpha=0.92, zorder=3)
bars_m = ax_f.bar(x + w/2, [mlp_rmse[c] for c in NAMES], w,
                  color=C_BAR_M, label="NeuralPlane (PyTorch-MLP)", alpha=0.92, zorder=3)

ax_f.set_yscale("log")
ax_f.set_xticks(x)
ax_f.set_xticklabels(LABELS_TEX, fontsize=13)
ax_f.set_ylabel("RMSE (vs. Ground-Truth LUT Table)")
ax_f.set_title("Aerodynamic Model Fidelity Comparison", fontweight="bold")
ax_f.legend(fontsize=10, loc="upper right", frameon=True, framealpha=0.9)
ax_f.tick_params(which="both", direction="in")
ax_f.yaxis.grid(True, which="major", alpha=0.25, lw=0.6, zorder=0)

# Annotate each MLP bar with the relative error multiplier
for i, c in enumerate(NAMES):
    ratio = mlp_rmse[c] / lut_rmse[c]
    bar_top = mlp_rmse[c]
    # format as e.g. "6.0×10⁵×"
    exp   = int(np.floor(np.log10(ratio)))
    coeff = ratio / 10**exp
    label = rf"${coeff:.1f}\!\times\!10^{{{exp}}}$×"
    ax_f.text(x[i] + w/2, bar_top * 1.6, label,
              ha="center", va="bottom", fontsize=8.5,
              color=C_BAR_M, fontweight="bold")

# Add a horizontal reference line at LUT mean RMSE
mean_lut = np.mean(list(lut_rmse.values()))
ax_f.axhline(mean_lut, color=C_BAR_L, lw=1.2, ls=":", alpha=0.7,
             label=f"LUT mean RMSE = {mean_lut:.1e}")
ax_f.legend(fontsize=10, loc="upper right", frameon=True, framealpha=0.9)

fig_f.tight_layout()
out_f = os.path.join(OUT_DIR, "aero_fidelity_comparison.pdf")
fig_f.savefig(out_f, format="pdf", bbox_inches="tight", transparent=True, dpi=300)
plt.close(fig_f)
print(f"  Saved: {out_f}")


# ══════════════════════════════════════════════════════════════════════
# 9.  Figure C – Coefficient curves with zoom insets
# ══════════════════════════════════════════════════════════════════════
print("Generating aero_coeff_curves.pdf …")

TITLES = {
    "Cx": r"$C_x$ (Axial Force)",
    "Cz": r"$C_z$ (Normal Force)",
    "Cm": r"$C_m$ (Pitching Moment)",
    "Cy": r"$C_y$ (Side Force)",
    "Cn": r"$C_n$ (Yawing Moment)",
    "Cl": r"$C_l$ (Rolling Moment)",
}
YLABELS = {c: TITLES[c].split("(")[0].strip() for c in NAMES}

fig_c, axes = plt.subplots(2, 3, figsize=(11.0, 7.4))
fig_c.subplots_adjust(hspace=0.52, wspace=0.35, top=0.85, bottom=0.09,
                      left=0.07, right=0.97)
axes = axes.flatten()

# Zoom α∈[30°,60°] — MLP visibly diverges for both Cz and Cm
# Cz: curves fill left & bottom → inset upper-right
# Cm: curves fill bottom → inset upper-left
ZOOM_ALPHA = (30, 60)
INSET_POS  = {"Cz": [0.56, 0.54, 0.42, 0.42],
              "Cm": [0.02, 0.54, 0.42, 0.42]}

for i, c in enumerate(NAMES):
    ax    = axes[i]
    gt_y  = gt_sw[c]; lut_y = lut_sw[c]; mlp_y = mlp_sw[c]

    ax.plot(a_sw, gt_y,  color=C_GT,  lw=2.5, ls="-",  zorder=6,
            label="Ground Truth (LUT table)")
    ax.plot(a_sw, lut_y, color=C_LUT, lw=1.5, ls="--", zorder=5,
            label="Planax (JAX-LUT)")
    ax.plot(a_sw, mlp_y, color=C_MLP, lw=1.5, ls=":",  zorder=4,
            label="NeuralPlane (MLP)")
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(YLABELS[c])
    ax.set_title(TITLES[c])
    ax.tick_params(which="both", direction="in")

    if c in ("Cz", "Cm"):
        zm    = (a_sw >= ZOOM_ALPHA[0]) & (a_sw <= ZOOM_ALPHA[1])
        a_z   = a_sw[zm]; gt_z = gt_y[zm]; lut_z = lut_y[zm]; mlp_z = mlp_y[zm]
        axins = ax.inset_axes(INSET_POS[c])
        axins.plot(a_z, gt_z,  color=C_GT,  lw=1.8, ls="-",  zorder=6)
        axins.plot(a_z, lut_z, color=C_LUT, lw=1.1, ls="--", zorder=5)
        axins.plot(a_z, mlp_z, color=C_MLP, lw=1.1, ls=":",  zorder=4)
        axins.set_xlim(a_z.min(), a_z.max())
        y_lo = min(gt_z.min(), lut_z.min(), mlp_z.min())
        y_hi = max(gt_z.max(), lut_z.max(), mlp_z.max())
        pad  = (y_hi - y_lo) * 0.15
        axins.set_ylim(y_lo - pad, y_hi + pad)
        axins.tick_params(labelsize=7, direction="in", pad=1)
        axins.set_xlabel(r"$\alpha$ (deg)", fontsize=7, labelpad=1)
        lut_err = float(np.max(np.abs(lut_z - gt_z)))
        axins.set_title(f"|LUT$-$GT|$\\leq${lut_err:.0e}", fontsize=7, pad=2)
        ax.indicate_inset_zoom(axins, edgecolor="#888888", lw=0.8, alpha=0.7)

handles, labels = axes[0].get_legend_handles_labels()
fig_c.legend(handles, labels, loc="upper center", ncol=3,
             fontsize=10, frameon=True, framealpha=0.9,
             bbox_to_anchor=(0.5, 0.975))
fig_c.suptitle(
    r"F-16 Aerodynamic Coefficients: JAX-LUT vs.\ PyTorch-MLP"
    "\n" r"($\beta=0°$,  $\delta_e=0°$,  varying $\alpha$)",
    fontsize=13, fontweight="bold", y=1.03,
)

out_c = os.path.join(OUT_DIR, "aero_coeff_curves.pdf")
fig_c.savefig(out_c, format="pdf", bbox_inches="tight", transparent=True)
plt.close(fig_c)
print(f"  Saved: {out_c}")

print("\nAll figures saved.")
