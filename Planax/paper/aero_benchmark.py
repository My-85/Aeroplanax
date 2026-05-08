#!/usr/bin/env python3
"""
Aerodynamic model benchmark: Planax JAX-LUT vs NeuralPlane PyTorch-MLP.

Measures:
  1. Computational speed across batch sizes
  2. Fidelity vs scipy ground-truth interpolation (same raw F-16 .dat tables)
  3. Coefficient curves vs alpha

Outputs three publication-quality PDFs in the paper/ directory:
  aero_speed_comparison.pdf
  aero_fidelity_comparison.pdf
  aero_coeff_curves.pdf
"""

import sys, os, time, warnings
import numpy as np

# ── PyTorch lives in the autoresearch venv ────────────────────────────
TORCH_SITE = (
    "/home/dqy/aeroplanax/new/20251215最新代码库/"
    "autoresearch/autoresearch/.venv/lib/python3.10/site-packages"
)
sys.path.insert(0, TORCH_SITE)

import torch
import jax
import jax.numpy as jnp
from jax import jit, vmap
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.8,
})

# ── Paths ─────────────────────────────────────────────────────────────
ROOT       = "/home/dqy/aeroplanax/new/20251215最新代码库"
PLANAX_DIR = os.path.join(ROOT, "Planax")
NEURAL_DIR = os.path.join(ROOT, "neuralplane/NeuralPlane")
DATA_DIR   = os.path.join(PLANAX_DIR, "envs/core/simulators/fighterplane/data")
OUT_DIR    = os.path.join(PLANAX_DIR, "paper")

sys.path.insert(0, PLANAX_DIR)
sys.path.insert(0, NEURAL_DIR)


# ══════════════════════════════════════════════════════════════════════
# 1.  Load raw F-16 aerodynamic tables (shared ground truth)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading F-16 aerodynamic tables …")

def load_dat(name):
    return np.loadtxt(os.path.join(DATA_DIR, name))

ALPHA1 = load_dat("ALPHA1.dat")   # (20,) deg
BETA1  = load_dat("BETA1.dat")    # (19,) deg
DH1    = load_dat("DH1.dat")      # (5,)  deg  – elevator (Cx/Cz/Cm)
DH2    = load_dat("DH2.dat")      # (5,)  deg  – elevator (Cn/Cl)

Cx_raw = load_dat("CX0120_ALPHA1_BETA1_DH1_201.dat").reshape(len(DH1), len(BETA1), len(ALPHA1))
Cz_raw = load_dat("CZ0120_ALPHA1_BETA1_DH1_301.dat").reshape(len(DH1), len(BETA1), len(ALPHA1))
Cm_raw = load_dat("CM0120_ALPHA1_BETA1_DH1_101.dat").reshape(len(DH1), len(BETA1), len(ALPHA1))
Cy_raw = load_dat("CY0320_ALPHA1_BETA1_401.dat").reshape(len(BETA1), len(ALPHA1))
Cn_raw = load_dat("CN0120_ALPHA1_BETA1_DH2_501.dat").reshape(len(DH2), len(BETA1), len(ALPHA1))
Cl_raw = load_dat("CL0120_ALPHA1_BETA1_DH2_601.dat").reshape(len(DH2), len(BETA1), len(ALPHA1))

# Scipy reference interpolators (these ARE the ground truth)
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

print(f"  ALPHA range : [{ALPHA1.min():.0f}, {ALPHA1.max():.0f}] deg  (N={len(ALPHA1)})")
print(f"  BETA  range : [{BETA1.min():.0f},  {BETA1.max():.0f}] deg  (N={len(BETA1)})")
print(f"  DH    range : [{DH1.min():.0f},  {DH1.max():.0f}] deg  (N={len(DH1)})")


# ══════════════════════════════════════════════════════════════════════
# 2.  Planax JAX-LUT model
# ══════════════════════════════════════════════════════════════════════
print("\nLoading Planax JAX-LUT model …")

# Build fully-batched JAX trilinear interpolation (vectorised, JIT-compiled)
_DH1_j  = jnp.array(DH1)
_DH2_j  = jnp.array(DH2)
_A1_j   = jnp.array(ALPHA1)
_B1_j   = jnp.array(BETA1)
_Cx_j   = jnp.array(Cx_raw)
_Cz_j   = jnp.array(Cz_raw)
_Cm_j   = jnp.array(Cm_raw)
_Cy_j   = jnp.array(Cy_raw)
_Cn_j   = jnp.array(Cn_raw)
_Cl_j   = jnp.array(Cl_raw)


def _trilinear_batch(gx, gy, gz, vals, xs, ys, zs):
    """Vectorised trilinear interp: xs,ys,zs are 1-D arrays of length N."""
    ix = jnp.clip(jnp.searchsorted(gx, xs) - 1, 0, len(gx) - 2)
    iy = jnp.clip(jnp.searchsorted(gy, ys) - 1, 0, len(gy) - 2)
    iz = jnp.clip(jnp.searchsorted(gz, zs) - 1, 0, len(gz) - 2)

    x0, x1 = gx[ix], gx[ix + 1]
    y0, y1 = gy[iy], gy[iy + 1]
    z0, z1 = gz[iz], gz[iz + 1]

    xd = (xs - x0) / (x1 - x0)
    yd = (ys - y0) / (y1 - y0)
    zd = (zs - z0) / (z1 - z0)

    c000 = vals[ix,   iy,   iz  ]
    c100 = vals[ix+1, iy,   iz  ]
    c010 = vals[ix,   iy+1, iz  ]
    c110 = vals[ix+1, iy+1, iz  ]
    c001 = vals[ix,   iy,   iz+1]
    c101 = vals[ix+1, iy,   iz+1]
    c011 = vals[ix,   iy+1, iz+1]
    c111 = vals[ix+1, iy+1, iz+1]

    c00 = c000*(1-xd) + c100*xd
    c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd
    c11 = c011*(1-xd) + c111*xd
    c0  = c00*(1-yd)  + c10*yd
    c1  = c01*(1-yd)  + c11*yd
    return c0*(1-zd) + c1*zd


def _bilinear_batch(gx, gy, vals, xs, ys):
    ix = jnp.clip(jnp.searchsorted(gx, xs) - 1, 0, len(gx) - 2)
    iy = jnp.clip(jnp.searchsorted(gy, ys) - 1, 0, len(gy) - 2)
    x0, x1 = gx[ix], gx[ix+1]
    y0, y1 = gy[iy], gy[iy+1]
    xd = (xs - x0) / (x1 - x0)
    yd = (ys - y0) / (y1 - y0)
    c00 = vals[ix,   iy  ]
    c10 = vals[ix+1, iy  ]
    c01 = vals[ix,   iy+1]
    c11 = vals[ix+1, iy+1]
    c0  = c00*(1-xd) + c10*xd
    c1  = c01*(1-xd) + c11*xd
    return c0*(1-yd) + c1*yd


@jit
def _lut_hifi_C(alphas, betas, els):
    """alphas, betas, els: 1-D jnp arrays (deg). Returns (Cx, Cz, Cm, Cy, Cn, Cl)."""
    cx = _trilinear_batch(_DH1_j, _B1_j, _A1_j, _Cx_j, els,    betas, alphas)
    cz = _trilinear_batch(_DH1_j, _B1_j, _A1_j, _Cz_j, els,    betas, alphas)
    cm = _trilinear_batch(_DH1_j, _B1_j, _A1_j, _Cm_j, els,    betas, alphas)
    cy = _bilinear_batch(_B1_j,   _A1_j,         _Cy_j, betas,  alphas)
    cn = _trilinear_batch(_DH2_j, _B1_j, _A1_j, _Cn_j, els,    betas, alphas)
    cl = _trilinear_batch(_DH2_j, _B1_j, _A1_j, _Cl_j, els,    betas, alphas)
    return cx, cz, cm, cy, cn, cl


def planax_lut_eval(alphas_np, betas_np, els_np):
    """Evaluate Planax LUT (JAX), returns tuple of 6 numpy arrays."""
    a = jnp.array(alphas_np)
    b = jnp.array(betas_np)
    e = jnp.array(els_np)
    result = _lut_hifi_C(a, b, e)
    jax.block_until_ready(result)
    return tuple(np.array(r) for r in result)


# Warm-up JIT compilation
print("  Warming up JAX JIT …")
_dummy_a = np.linspace(-10, 45, 16, dtype=np.float32)
_dummy_b = np.zeros(16, dtype=np.float32)
_dummy_e = np.zeros(16, dtype=np.float32)
planax_lut_eval(_dummy_a, _dummy_b, _dummy_e)
print("  JAX JIT compiled.")


# ══════════════════════════════════════════════════════════════════════
# 3.  NeuralPlane PyTorch-MLP model
# ══════════════════════════════════════════════════════════════════════
print("\nLoading NeuralPlane PyTorch-MLP model …")

TORCH_DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  PyTorch device: {TORCH_DEV}")

# We need to patch the module-level device before the class default is captured
import importlib, types
# Temporarily override so MLP models load on our chosen device
_neural_f16_path = os.path.join(NEURAL_DIR, "envs/models/F16/hifi_F16_AeroData.py")

spec = importlib.util.spec_from_file_location("hifi_F16_AeroData", _neural_f16_path)
hifi_mod = importlib.util.module_from_spec(spec)
hifi_mod.device = TORCH_DEV           # patch before exec_module
spec.loader.exec_module(hifi_mod)
hifi_F16 = hifi_mod.hifi_F16

mlp_model = hifi_F16(device=TORCH_DEV)

# Set all component models to eval mode
for attr in dir(mlp_model):
    obj = getattr(mlp_model, attr)
    if isinstance(obj, torch.nn.Module):
        obj.eval()

print("  MLP models loaded and set to eval mode.")


def mlp_eval(alphas_np, betas_np, els_np):
    """Evaluate NeuralPlane MLP, returns tuple of 6 numpy arrays."""
    a = torch.tensor(alphas_np, dtype=torch.float32, device=TORCH_DEV)
    b = torch.tensor(betas_np,  dtype=torch.float32, device=TORCH_DEV)
    e = torch.tensor(els_np,    dtype=torch.float32, device=TORCH_DEV)
    with torch.no_grad():
        cx, cz, cm, cy, cn, cl = mlp_model.hifi_C(a, b, e)
    if TORCH_DEV.startswith("cuda"):
        torch.cuda.synchronize()
    return (cx.cpu().numpy(), cz.cpu().numpy(), cm.cpu().numpy(),
            cy.cpu().numpy(), cn.cpu().numpy(), cl.cpu().numpy())


# Warm-up
print("  Warming up PyTorch MLP …")
mlp_eval(_dummy_a, _dummy_b, _dummy_e)
print("  PyTorch MLP ready.")


# ══════════════════════════════════════════════════════════════════════
# 4.  Random test points within F-16 flight envelope
# ══════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(2025)
N_TEST  = 10000
alphas_t = rng.uniform(ALPHA1.min(), ALPHA1.max(), N_TEST).astype(np.float32)
betas_t  = rng.uniform(BETA1.min(),  BETA1.max(),  N_TEST).astype(np.float32)
els_t    = rng.uniform(DH1.min(),    DH1.max(),    N_TEST).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 5.  Speed benchmark
# ══════════════════════════════════════════════════════════════════════
BATCH_SIZES = [1, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
N_REPEATS   = 40

print("\n" + "=" * 60)
print("Speed benchmark …")
print(f"{'Batch':>6}  {'LUT (ms)':>10}  {'MLP (ms)':>10}  {'Speedup':>8}")
print("-" * 42)

lut_times_ms = []
mlp_times_ms = []

for B in BATCH_SIZES:
    a = alphas_t[:B]
    b = betas_t[:B]
    e = els_t[:B]

    # ── Planax LUT ───────────────────────────────────────────────────
    for _ in range(5):          # extra warm-up at this batch size
        r = planax_lut_eval(a, b, e)
    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        r = planax_lut_eval(a, b, e)
    t_lut = (time.perf_counter() - t0) / N_REPEATS * 1e3
    lut_times_ms.append(t_lut)

    # ── NeuralPlane MLP ──────────────────────────────────────────────
    for _ in range(5):
        mlp_eval(a, b, e)
    if TORCH_DEV.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        mlp_eval(a, b, e)
        if TORCH_DEV.startswith("cuda"):
            torch.cuda.synchronize()
    t_mlp = (time.perf_counter() - t0) / N_REPEATS * 1e3
    mlp_times_ms.append(t_mlp)

    ratio = t_mlp / t_lut if t_lut > 0 else float("nan")
    print(f"{B:>6}  {t_lut:>10.3f}  {t_mlp:>10.3f}  {ratio:>7.1f}×")


# ══════════════════════════════════════════════════════════════════════
# 6.  Fidelity benchmark – RMSE vs scipy ground truth
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Fidelity benchmark …")

pts3_dh1 = np.stack([els_t, betas_t, alphas_t], axis=-1)
pts3_dh2 = np.stack([els_t, betas_t, alphas_t], axis=-1)
pts2     = np.stack([betas_t, alphas_t], axis=-1)

gt = {
    "Cx": gt_Cx(pts3_dh1), "Cz": gt_Cz(pts3_dh1), "Cm": gt_Cm(pts3_dh1),
    "Cy": gt_Cy(pts2),
    "Cn": gt_Cn(pts3_dh2), "Cl": gt_Cl(pts3_dh2),
}

lut_r = planax_lut_eval(alphas_t, betas_t, els_t)
mlp_r = mlp_eval(alphas_t, betas_t, els_t)
COEFF_NAMES = ["Cx", "Cz", "Cm", "Cy", "Cn", "Cl"]
lut_d = dict(zip(COEFF_NAMES, lut_r))
mlp_d = dict(zip(COEFF_NAMES, mlp_r))

lut_rmse, mlp_rmse = {}, {}
lut_maxe, mlp_maxe = {}, {}

print(f"{'Coeff':>6}  {'LUT RMSE':>12}  {'MLP RMSE':>12}  {'LUT MaxE':>12}  {'MLP MaxE':>12}")
print("-" * 62)
for c in COEFF_NAMES:
    le = lut_d[c] - gt[c]
    me = mlp_d[c] - gt[c]
    lut_rmse[c] = float(np.sqrt(np.mean(le**2)))
    mlp_rmse[c] = float(np.sqrt(np.mean(me**2)))
    lut_maxe[c] = float(np.max(np.abs(le)))
    mlp_maxe[c] = float(np.max(np.abs(me)))
    print(f"{c:>6}  {lut_rmse[c]:>12.2e}  {mlp_rmse[c]:>12.2e}  "
          f"{lut_maxe[c]:>12.2e}  {mlp_maxe[c]:>12.2e}")

mean_lut = np.mean(list(lut_rmse.values()))
mean_mlp = np.mean(list(mlp_rmse.values()))
print(f"\n  Mean RMSE — LUT: {mean_lut:.2e}   MLP: {mean_mlp:.2e}   "
      f"Ratio: {mean_mlp/mean_lut:.1f}×")


# ══════════════════════════════════════════════════════════════════════
# 7.  Coefficient curves sweep (β=0°, δe=0°, varying α)
# ══════════════════════════════════════════════════════════════════════
N_SW = 300
alpha_sw = np.linspace(ALPHA1.min(), ALPHA1.max(), N_SW, dtype=np.float32)
beta_sw  = np.zeros(N_SW, dtype=np.float32)
el_sw    = np.zeros(N_SW, dtype=np.float32)

pts3_sw = np.stack([el_sw, beta_sw, alpha_sw], axis=-1)
pts2_sw = np.stack([beta_sw, alpha_sw], axis=-1)
gt_sw = {
    "Cx": gt_Cx(pts3_sw), "Cz": gt_Cz(pts3_sw), "Cm": gt_Cm(pts3_sw),
    "Cy": gt_Cy(pts2_sw),
    "Cn": gt_Cn(pts3_sw), "Cl": gt_Cl(pts3_sw),
}
lut_sw_r = planax_lut_eval(alpha_sw, beta_sw, el_sw)
mlp_sw_r = mlp_eval(alpha_sw, beta_sw, el_sw)
lut_sw = dict(zip(COEFF_NAMES, lut_sw_r))
mlp_sw = dict(zip(COEFF_NAMES, mlp_sw_r))


# ══════════════════════════════════════════════════════════════════════
# 8.  Figures
# ══════════════════════════════════════════════════════════════════════
C_LUT  = "#2CA02C"   # green – Planax
C_MLP  = "#CC2222"   # red   – NeuralPlane
C_GT   = "#111111"   # black – ground truth

# ── Figure 1: Speed Comparison (dual panel: latency + throughput) ─────
lut_tput = [B / (t * 1e-3) for B, t in zip(BATCH_SIZES, lut_times_ms)]   # queries/s
mlp_tput = [B / (t * 1e-3) for B, t in zip(BATCH_SIZES, mlp_times_ms)]

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(7.0, 3.0))

ax1a.plot(BATCH_SIZES, lut_times_ms, "o-", color=C_LUT, lw=2.0, ms=5,
          label="Planax  (JAX LUT)")
ax1a.plot(BATCH_SIZES, mlp_times_ms, "s--", color=C_MLP, lw=2.0, ms=5,
          label="NeuralPlane  (PyTorch MLP)")
ax1a.set_xscale("log"); ax1a.set_yscale("log")
ax1a.set_xlabel("Batch size")
ax1a.set_ylabel("Inference latency  (ms)")
ax1a.set_title("(a)  Latency", fontweight="bold", fontsize=9)
ax1a.legend(fontsize=7.5, loc="upper left")
ax1a.grid(True, which="both", alpha=0.25, lw=0.5)

ax1b.plot(BATCH_SIZES, lut_tput, "o-", color=C_LUT, lw=2.0, ms=5,
          label="Planax  (JAX LUT)")
ax1b.plot(BATCH_SIZES, mlp_tput, "s--", color=C_MLP, lw=2.0, ms=5,
          label="NeuralPlane  (PyTorch MLP)")
ax1b.set_xscale("log"); ax1b.set_yscale("log")
ax1b.set_xlabel("Batch size")
ax1b.set_ylabel("Throughput  (queries / s)")
ax1b.set_title("(b)  Throughput", fontweight="bold", fontsize=9)
ax1b.legend(fontsize=7.5, loc="upper left")
ax1b.grid(True, which="both", alpha=0.25, lw=0.5)

fig1.suptitle("Aerodynamic Model Computational Performance",
              fontweight="bold", fontsize=10, y=1.01)
fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "aero_speed_comparison.pdf")
fig1.savefig(out1, format="pdf", bbox_inches="tight", transparent=True, dpi=300)
plt.close(fig1)
print(f"\nSaved: {out1}")

# ── Figure 2: Fidelity – RMSE bar chart ─────────────────────────────
LABELS = [r"$C_x$", r"$C_z$", r"$C_m$", r"$C_y$", r"$C_n$", r"$C_l$"]
x  = np.arange(len(COEFF_NAMES))
w  = 0.38

fig2, ax2 = plt.subplots(figsize=(4.0, 3.2))
bars_l = ax2.bar(x - w/2, [lut_rmse[c] for c in COEFF_NAMES], w,
                 color=C_LUT, label="Planax  (JAX LUT)", alpha=0.90)
bars_m = ax2.bar(x + w/2, [mlp_rmse[c] for c in COEFF_NAMES], w,
                 color=C_MLP, label="NeuralPlane  (PyTorch MLP)", alpha=0.90)
ax2.set_xticks(x)
ax2.set_xticklabels(LABELS, fontsize=10)
ax2.set_ylabel("RMSE  (vs. ground-truth LUT table)")
ax2.set_title("Aerodynamic Model Fidelity", fontweight="bold", fontsize=9.5)
ax2.set_yscale("log")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, axis="y", alpha=0.25, lw=0.5)
# Annotate mean RMSE summary
ax2.text(0.98, 0.02,
         f"Mean RMSE:  LUT={mean_lut:.1e}  (≈0)\n"
         f"               MLP={mean_mlp:.1e}  ({mean_mlp/mean_lut:.0e}× worse)",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=7.2, color="#444444",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.7))
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "aero_fidelity_comparison.pdf")
fig2.savefig(out2, format="pdf", bbox_inches="tight", transparent=True, dpi=300)
plt.close(fig2)
print(f"Saved: {out2}")

# ── Figure 3: Coefficient curves vs α (6-panel) ──────────────────────
PRETTY = {
    "Cx": (r"$C_x$", r"$C_x$  vs.  $\alpha$  ($\beta$=0°, $\delta_e$=0°)"),
    "Cz": (r"$C_z$", r"$C_z$  vs.  $\alpha$"),
    "Cm": (r"$C_m$", r"$C_m$  vs.  $\alpha$"),
    "Cy": (r"$C_y$", r"$C_y$  vs.  $\alpha$"),
    "Cn": (r"$C_n$", r"$C_n$  vs.  $\alpha$"),
    "Cl": (r"$C_l$", r"$C_l$  vs.  $\alpha$"),
}
fig3, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
axes = axes.flatten()

for i, c in enumerate(COEFF_NAMES):
    ax = axes[i]
    ax.plot(alpha_sw, gt_sw[c], "-",  color=C_GT,  lw=2.2, zorder=6,
            label="Ground Truth (LUT table)")
    ax.plot(alpha_sw, lut_sw[c], "--", color=C_LUT, lw=1.8, zorder=5,
            label="Planax  JAX-LUT")
    ax.plot(alpha_sw, mlp_sw[c], ":",  color=C_MLP, lw=1.8, zorder=4,
            label="NeuralPlane  MLP")
    ylabel, title = PRETTY[c]
    ax.set_xlabel(r"$\alpha$ (deg)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=7.8)
    ax.grid(alpha=0.25, lw=0.5)
    if i == 0:
        ax.legend(fontsize=7, loc="best")

fig3.suptitle(
    r"F-16 Aerodynamic Coefficients: JAX-LUT vs. PyTorch-MLP  ($\beta$=0°, $\delta_e$=0°)",
    fontsize=9, fontweight="bold", y=1.02,
)
fig3.tight_layout()
out3 = os.path.join(OUT_DIR, "aero_coeff_curves.pdf")
fig3.savefig(out3, format="pdf", bbox_inches="tight", transparent=True, dpi=300)
plt.close(fig3)
print(f"Saved: {out3}")


# ══════════════════════════════════════════════════════════════════════
# 9.  Summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
idx1k  = BATCH_SIZES.index(1000)
idx10k = BATCH_SIZES.index(10000)
print(f"Speed @ batch=1000:")
print(f"  Planax  JAX-LUT : {lut_times_ms[idx1k]:.3f} ms  "
      f"({BATCH_SIZES[idx1k]/lut_times_ms[idx1k]*1e3:.0f} queries/s)")
print(f"  NeuralPlane MLP : {mlp_times_ms[idx1k]:.3f} ms  "
      f"({BATCH_SIZES[idx1k]/mlp_times_ms[idx1k]*1e3:.0f} queries/s)")
print(f"Speed @ batch=10000:")
print(f"  Planax  JAX-LUT : {lut_times_ms[idx10k]:.3f} ms  "
      f"({BATCH_SIZES[idx10k]/lut_times_ms[idx10k]*1e3:.0f} queries/s)")
print(f"  NeuralPlane MLP : {mlp_times_ms[idx10k]:.3f} ms  "
      f"({BATCH_SIZES[idx10k]/mlp_times_ms[idx10k]*1e3:.0f} queries/s)")
print(f"\nFidelity (mean RMSE, 6 coefficients, {N_TEST} test points):")
print(f"  Planax  JAX-LUT : {mean_lut:.3e}  (≈ machine precision)")
print(f"  NeuralPlane MLP : {mean_mlp:.3e}")
print(f"  LUT is {mean_mlp/mean_lut:.0f}× more accurate than MLP")
print("=" * 60)
