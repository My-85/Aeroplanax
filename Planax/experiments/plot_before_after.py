#!/usr/bin/env python3
"""Generate side-by-side before/after comparison plots."""
import os, sys, json
from pathlib import Path
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
BUGGY_DIR = ROOT / "results" / "fidelity_validation"
FIXED_DIR = ROOT / "results" / "fidelity_validation_fixed"


def load_csv(path):
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) if v not in ('', 'nan', 'NaN', 'inf', '-inf') else float('nan')
                         for k, v in row.items()})
    return rows


def get_arr(traj, key):
    if traj is None:
        return None
    return np.array([s.get(key, float('nan')) for s in traj])


def plot_scenario(scenario):
    """3 columns × 4 rows: BUGGY (left) | FIXED (middle) | overlay (right) for key vars."""
    bp = load_csv(BUGGY_DIR / f"{scenario}_planax_v2.csv")
    bj = load_csv(BUGGY_DIR / f"{scenario}_jsbsim_v2.csv")
    fp = load_csv(FIXED_DIR / f"{scenario}_planax_v3.csv")
    fj = load_csv(FIXED_DIR / f"{scenario}_jsbsim_v3.csv")

    if fp is None:
        print(f"  [skip] {scenario}: no fixed output")
        return

    variables = [
        ('vt', 'Airspeed Vt (m/s)'),
        ('alpha', 'Alpha (rad)'),
        ('P', 'Roll Rate p (rad/s)'),
        ('Q', 'Pitch Rate q (rad/s)'),
        ('roll', 'Roll (rad)'),
        ('pitch', 'Pitch (rad)'),
        ('altitude', 'Altitude (m)'),
        ('R', 'Yaw Rate r (rad/s)'),
    ]

    fig, axes = plt.subplots(len(variables), 2, figsize=(13, 2.4 * len(variables)),
                              sharex='col')
    fig.suptitle(f'Planax vs JSBSim — Scenario: {scenario}\nLeft: BUGGY dynamics | Right: FIXED dynamics',
                 fontsize=14, y=1.0)

    for i, (var, label) in enumerate(variables):
        # LEFT: buggy
        ax = axes[i, 0]
        if bp is not None and bj is not None:
            t_bp = get_arr(bp, 'time')
            t_bj = get_arr(bj, 'time')
            v_bp = get_arr(bp, var)
            v_bj = get_arr(bj, var)
            ax.plot(t_bp, v_bp, 'r-', lw=1.5, label='Planax (buggy)')
            ax.plot(t_bj, v_bj, 'b--', lw=1.2, label='JSBSim')
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        if i == 0:
            ax.set_title('BUGGY dynamics', fontsize=11)

        # RIGHT: fixed
        ax = axes[i, 1]
        t_fp = get_arr(fp, 'time')
        t_fj = get_arr(fj, 'time')
        v_fp = get_arr(fp, var)
        v_fj = get_arr(fj, var) if fj else None
        ax.plot(t_fp, v_fp, 'g-', lw=1.5, label='Planax (FIXED)')
        if v_fj is not None:
            ax.plot(t_fj, v_fj, 'b--', lw=1.2, label='JSBSim')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        if i == 0:
            ax.set_title('FIXED dynamics', fontsize=11)

        if i == len(variables) - 1:
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 1].set_xlabel('Time (s)')

    plt.tight_layout()
    out = FIXED_DIR / f'{scenario}_BEFORE_vs_AFTER.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out.name}")


def main():
    print("Generating before/after comparison plots...")
    for sc in ['trim', 'elevator_doublet', 'coordinated_turn', 'sinusoidal']:
        plot_scenario(sc)


if __name__ == '__main__':
    main()
