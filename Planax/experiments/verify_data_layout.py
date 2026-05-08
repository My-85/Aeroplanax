#!/usr/bin/env python3
"""
Pinpoint which interpretation of Planax data is correct, by comparing
against the NASA TP-1538 reference Cx table from f16_deq.f.

NASA TP-1538 Cx table at (alpha rows -2..9 in 5° steps, EL columns -2..2 in 12° steps):

  EL→     -25     -10      0      10     25
  α=-10  -.099  -.048  -.022  -.040  -.083
  α=-5   -.081  -.038  -.020  -.038  -.073
  α=0    -.081  -.040  -.021  -.039  -.076
  α=+5   -.063  -.021  -.004  -.025  -.072
  α=+10  -.025   .016   .032   .006  -.046
  α=+15   .044   .083   .094   .062   .012

These reference values were directly copied from f16_deq.f line 294-303.
The original NASA grid is α∈[-10,45]°(5° steps), EL∈[-25,25]°(12° steps but
interpreted as 5 elevator settings).

Key insight: NASA TP-1538 uses a 12×5 table with NO beta dependence on Cx.
Planax hifi data is a richer 20×19×5 table that adds beta dependence.

But the PHYSICAL TRENDS should match:
  - Cx should decrease (more negative drag) at large |alpha|
  - Cx at fixed alpha should be a smooth function of EL
"""
import numpy as np
import sys
sys.path.insert(0, '.')

# Load Planax data files
DH1 = np.loadtxt('envs/core/simulators/fighterplane/data/DH1.dat')
ALPHA1 = np.loadtxt('envs/core/simulators/fighterplane/data/ALPHA1.dat')
BETA1 = np.loadtxt('envs/core/simulators/fighterplane/data/BETA1.dat')
Cx_raw = np.loadtxt('envs/core/simulators/fighterplane/data/CX0120_ALPHA1_BETA1_DH1_201.dat')

print(f"Grid sizes: ALPHA1={len(ALPHA1)}, BETA1={len(BETA1)}, DH1={len(DH1)}")
print(f"Total Cx points: {Cx_raw.size}")
print()
print("ALPHA1:", ALPHA1)
print("BETA1: ", BETA1)
print("DH1:   ", DH1)
print()

# The two reshape interpretations
# Filename literal: ALPHA1, BETA1, DH1 → outer-to-inner = (ALPHA, BETA, DH)
# Planax current:  (DH, BETA, ALPHA)

reshapes = {
    'A: (ALPHA, BETA, DH) [filename order]': Cx_raw.reshape(len(ALPHA1), len(BETA1), len(DH1)),
    'B: (DH, BETA, ALPHA) [Planax current]': Cx_raw.reshape(len(DH1), len(BETA1), len(ALPHA1)),
}

# NASA reference values: at beta=0, alpha sweep, el sweep
NASA_REF = {  # [alpha_deg][el_deg]
    -10: {-25: -.099, -10: -.048, 0: -.022, 10: -.040, 25: -.083},
    -5:  {-25: -.081, -10: -.038, 0: -.020, 10: -.038, 25: -.073},
    0:   {-25: -.081, -10: -.040, 0: -.021, 10: -.039, 25: -.076},
    5:   {-25: -.063, -10: -.021, 0: -.004, 10: -.025, 25: -.072},
    10:  {-25: -.025, -10:  .016, 0:  .032, 10:  .006, 25: -.046},
    15:  {-25:  .044, -10:  .083, 0:  .094, 10:  .062, 25:  .012},
}

bi_zero = np.argmin(np.abs(BETA1))  # beta=0 index

print("="*80)
print(f"Comparing Cx values at beta=0:")
print(f"NASA TP-1538 reference vs Planax data with two reshape interpretations")
print("="*80)

for label, Cx_3d in reshapes.items():
    print(f"\n--- Interpretation: {label} ---")
    label_axe = "alpha/el"
    print(f"{label_axe:>10}", end='')
    for el_target in [-25, -10, 0, 10, 25]:
        print(f"{el_target:>10}", end='')
    print()

    for alpha_target in [-10, -5, 0, 5, 10, 15]:
        ai = np.argmin(np.abs(ALPHA1 - alpha_target))
        print(f"α={alpha_target:>3}°/REF: ", end='')
        # Print NASA ref first
        for el_target in [-25, -10, 0, 10, 25]:
            print(f"{NASA_REF[alpha_target][el_target]:>10.4f}", end='')
        print()

        print(f"α={alpha_target:>3}°/PLN: ", end='')
        for el_target in [-25, -10, 0, 10, 25]:
            di = np.argmin(np.abs(DH1 - el_target))
            if 'ALPHA, BETA, DH' in label:
                val = Cx_3d[ai, bi_zero, di]
            else:
                val = Cx_3d[di, bi_zero, ai]
            print(f"{val:>10.4f}", end='')
        print()
        # error
        errors = []
        for el_target in [-25, -10, 0, 10, 25]:
            di = np.argmin(np.abs(DH1 - el_target))
            if 'ALPHA, BETA, DH' in label:
                val = Cx_3d[ai, bi_zero, di]
            else:
                val = Cx_3d[di, bi_zero, ai]
            errors.append(val - NASA_REF[alpha_target][el_target])
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        print(f"            RMSE={rmse:.4f}")
        print()

print()
print("="*80)
print("CONCLUSION:")
print("  The interpretation closer to NASA TP-1538 Cx values is the CORRECT reshape.")
print("="*80)
