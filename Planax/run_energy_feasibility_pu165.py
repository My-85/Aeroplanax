import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-2c45b7fd-69c8-1697-23a0-fe7ce7a2a620")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import run_half_loop_bridge_micro_search as bridge
import run_half_loop_termination_trace as trace


PLANAX_ROOT = Path(__file__).resolve().parent
HARD_TASK = "pu165_R15000"
G0 = 9.80665
SAFE_VT = 190.0
RADII = [15000.0, 18000.0, 20000.0, 25000.0, 30000.0]
ANGLES = [150.0, 165.0]
ENTRY_VTS = [250.0, 300.0, 350.0, 400.0]
ALTITUDE_LIMITS = [8000.0, 12000.0, 16000.0, 20000.0]


def parse_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def csv_float(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def policy_configs():
    return {
        "base_only": None,
        "update2_scale1.0": bridge.make_residual_cfg(scale=1.0),
        "update2_scale0.2": bridge.make_residual_cfg(scale=0.20),
    }


def feasibility_rows():
    rows = []
    for radius in RADII:
        for angle in ANGLES:
            theta = math.radians(angle)
            alt_gain = radius * (1.0 - math.cos(theta))
            pe_per_kg = G0 * alt_gain
            for vt in ENTRY_VTS:
                ke_per_kg = 0.5 * vt * vt
                energy_height = ke_per_kg / G0
                exit_v_sq = vt * vt - 2.0 * G0 * alt_gain
                exit_v_unpowered = math.sqrt(max(0.0, exit_v_sq))
                energy_deficit_to_safe = alt_gain + SAFE_VT * SAFE_VT / (2.0 * G0) - energy_height
                bottom_load = 1.0 + vt * vt / (G0 * radius)
                top_load_abs = abs(vt * vt / (G0 * radius) - 1.0)
                load_est = max(bottom_load, top_load_abs)
                speed_margin = exit_v_unpowered - SAFE_VT
                if energy_deficit_to_safe <= 0.0 and load_est < 8.5:
                    cls = "feasible"
                elif energy_deficit_to_safe <= 5000.0 and load_est < 9.0:
                    cls = "marginal"
                else:
                    cls = "infeasible"
                rows.append(
                    {
                        "radius_m": radius,
                        "angle_deg": angle,
                        "altitude_gain_m": alt_gain,
                        "potential_energy_j_per_kg": pe_per_kg,
                        "entry_vt": vt,
                        "entry_ke_j_per_kg": ke_per_kg,
                        "entry_energy_height_m": energy_height,
                        "unpowered_exit_vt_est": exit_v_unpowered,
                        "speed_margin_to_safe_vt": speed_margin,
                        "energy_deficit_to_safe_vt_m": energy_deficit_to_safe,
                        "required_load_factor_est": load_est,
                        "bottom_load_factor_est": bottom_load,
                        "top_load_abs_est": top_load_abs,
                        "classification": cls,
                    }
                )
    return rows


FEASIBILITY_FIELDS = [
    "radius_m",
    "angle_deg",
    "altitude_gain_m",
    "potential_energy_j_per_kg",
    "entry_vt",
    "entry_ke_j_per_kg",
    "entry_energy_height_m",
    "unpowered_exit_vt_est",
    "speed_margin_to_safe_vt",
    "energy_deficit_to_safe_vt_m",
    "required_load_factor_est",
    "bottom_load_factor_est",
    "top_load_abs_est",
    "classification",
]


def candidate_variants():
    variants = []
    for vt in ENTRY_VTS:
        variants.append(
            {
                "variant": f"fixed_R15000_entry_vt_{int(vt)}",
                "family": "entry_vt_sweep_fixed_R15000",
                "entry_vt": vt,
                "target_vt": min(vt, 360.0),
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
            }
        )
    for limit in ALTITUDE_LIMITS:
        variants.append(
            {
                "variant": f"altcap_{int(limit)}_entry_vt_400",
                "family": "altitude_gain_limit_sweep",
                "entry_vt": 400.0,
                "target_vt": 360.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": limit,
                "target_altitude_gain_limit_m": limit,
                "pitch_rate_limit_deg_s": 10.0,
                "roll_rate_limit_deg_s": 35.0,
            }
        )
    for limit in ALTITUDE_LIMITS:
        variants.append(
            {
                "variant": f"altcap_{int(limit)}_entry_vt_300",
                "family": "altitude_gain_limit_sweep_v300",
                "entry_vt": 300.0,
                "target_vt": 300.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": limit,
                "target_altitude_gain_limit_m": limit,
                "pitch_rate_limit_deg_s": 8.0,
                "roll_rate_limit_deg_s": 30.0,
            }
        )
    variants.extend(
        [
            {
                "variant": "mpc_altcap8000_v300_smooth",
                "family": "constrained_mpc_rhtso_v300",
                "entry_vt": 300.0,
                "target_vt": 300.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 8000.0,
                "target_altitude_gain_limit_m": 8000.0,
                "bridge_target_vt": 300.0,
                "bridge_lookahead_dist": 900.0,
                "pitch_rate_limit_deg_s": 6.0,
                "roll_rate_limit_deg_s": 24.0,
                "pitch_blend_with_current": 0.12,
            },
            {
                "variant": "mpc_altcap12000_v300_smooth",
                "family": "constrained_mpc_rhtso_v300",
                "entry_vt": 300.0,
                "target_vt": 300.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 12000.0,
                "target_altitude_gain_limit_m": 12000.0,
                "bridge_target_vt": 300.0,
                "bridge_lookahead_dist": 1000.0,
                "pitch_rate_limit_deg_s": 6.0,
                "roll_rate_limit_deg_s": 24.0,
                "pitch_blend_with_current": 0.15,
            },
            {
                "variant": "load_factor_limited_altcap12000_v300",
                "family": "load_factor_limited_v300",
                "entry_vt": 300.0,
                "target_vt": 300.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 12000.0,
                "target_altitude_gain_limit_m": 12000.0,
                "bridge_target_vt": 290.0,
                "bridge_lookahead_dist": 1200.0,
                "pitch_rate_limit_deg_s": 5.0,
                "roll_rate_limit_deg_s": 20.0,
                "pitch_blend_with_current": 0.25,
            },
            {
                "variant": "profile_R20000_altcap12000_v300",
                "family": "variable_radius_profile_v300",
                "entry_vt": 300.0,
                "target_vt": 300.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 12000.0,
                "target_altitude_gain_limit_m": 12000.0,
                "target_radius_profile": [
                    {"start_deg": 60.0, "end_deg": 165.0, "radius_m": 20000.0, "transition_deg": 18.0}
                ],
                "bridge_target_vt": 300.0,
                "bridge_lookahead_dist": 1000.0,
                "pitch_rate_limit_deg_s": 6.0,
                "roll_rate_limit_deg_s": 24.0,
                "pitch_blend_with_current": 0.10,
            },
            {
                "variant": "mpc_altcap8000_v400_smooth",
                "family": "constrained_mpc_rhtso",
                "entry_vt": 400.0,
                "target_vt": 360.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 8000.0,
                "target_altitude_gain_limit_m": 8000.0,
                "bridge_target_vt": 340.0,
                "bridge_lookahead_dist": 900.0,
                "pitch_rate_limit_deg_s": 8.0,
                "roll_rate_limit_deg_s": 30.0,
                "pitch_blend_with_current": 0.10,
            },
            {
                "variant": "mpc_altcap12000_v400_smooth",
                "family": "constrained_mpc_rhtso",
                "entry_vt": 400.0,
                "target_vt": 360.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 12000.0,
                "target_altitude_gain_limit_m": 12000.0,
                "bridge_target_vt": 340.0,
                "bridge_lookahead_dist": 1000.0,
                "pitch_rate_limit_deg_s": 8.0,
                "roll_rate_limit_deg_s": 30.0,
                "pitch_blend_with_current": 0.12,
            },
            {
                "variant": "mpc_altcap16000_v400_smooth",
                "family": "constrained_mpc_rhtso",
                "entry_vt": 400.0,
                "target_vt": 360.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 16000.0,
                "target_altitude_gain_limit_m": 16000.0,
                "bridge_target_vt": 350.0,
                "bridge_lookahead_dist": 1100.0,
                "pitch_rate_limit_deg_s": 9.0,
                "roll_rate_limit_deg_s": 35.0,
                "pitch_blend_with_current": 0.10,
            },
            {
                "variant": "profile_R20000_altcap12000_v400",
                "family": "variable_radius_profile",
                "entry_vt": 400.0,
                "target_vt": 360.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 12000.0,
                "target_altitude_gain_limit_m": 12000.0,
                "target_radius_profile": [
                    {"start_deg": 60.0, "end_deg": 165.0, "radius_m": 20000.0, "transition_deg": 18.0}
                ],
                "bridge_target_vt": 340.0,
                "bridge_lookahead_dist": 1000.0,
                "pitch_rate_limit_deg_s": 8.0,
                "roll_rate_limit_deg_s": 30.0,
            },
            {
                "variant": "profile_R25000_altcap12000_v400",
                "family": "variable_radius_profile",
                "entry_vt": 400.0,
                "target_vt": 360.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 12000.0,
                "target_altitude_gain_limit_m": 12000.0,
                "target_radius_profile": [
                    {"start_deg": 60.0, "end_deg": 165.0, "radius_m": 25000.0, "transition_deg": 20.0}
                ],
                "bridge_target_vt": 340.0,
                "bridge_lookahead_dist": 1100.0,
                "pitch_rate_limit_deg_s": 8.0,
                "roll_rate_limit_deg_s": 30.0,
                "pitch_blend_with_current": 0.08,
            },
            {
                "variant": "phase_slow_altcap12000_v400",
                "family": "phase_rate_schedule",
                "entry_vt": 400.0,
                "target_vt": 360.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 12000.0,
                "target_altitude_gain_limit_m": 12000.0,
                "bridge_target_vt": 330.0,
                "bridge_lookahead_dist": 1300.0,
                "pitch_rate_limit_deg_s": 5.0,
                "roll_rate_limit_deg_s": 20.0,
                "pitch_blend_with_current": 0.18,
            },
            {
                "variant": "load_factor_limited_altcap12000_v400",
                "family": "load_factor_limited",
                "entry_vt": 400.0,
                "target_vt": 360.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 12000.0,
                "target_altitude_gain_limit_m": 12000.0,
                "bridge_target_vt": 330.0,
                "bridge_lookahead_dist": 1200.0,
                "pitch_rate_limit_deg_s": 6.0,
                "roll_rate_limit_deg_s": 25.0,
                "pitch_blend_with_current": 0.25,
            },
            {
                "variant": "energy_preserve_altcap8000_v350",
                "family": "target_vt_schedule",
                "entry_vt": 350.0,
                "target_vt": 340.0,
                "eval_radius_m": 15000.0,
                "target_radius_m": 15000.0,
                "eval_altitude_gain_limit_m": 8000.0,
                "target_altitude_gain_limit_m": 8000.0,
                "bridge_target_vt": 320.0,
                "bridge_lookahead_dist": 1000.0,
                "pitch_rate_limit_deg_s": 8.0,
                "roll_rate_limit_deg_s": 30.0,
                "pitch_blend_with_current": 0.10,
            },
        ]
    )
    return variants


def phase_stats(phase_rows):
    if not phase_rows:
        return {
            "CTE_p90": "",
            "CTE_max": "",
            "phase_max": "",
            "target_pitch_delta_mean": "",
            "target_pitch_delta_max": "",
            "target_roll_delta_mean": "",
            "target_roll_delta_max": "",
        }
    def arr(key):
        return np.asarray([csv_float(r, key) for r in phase_rows], dtype=np.float64)
    cte = arr("CTE")
    phase = arr("phase")
    dp = np.abs(np.diff(arr("target_pitch")))
    dr = np.abs(np.diff(arr("target_roll")))
    return {
        "CTE_p90": float(np.percentile(cte, 90)),
        "CTE_max": float(np.max(cte)),
        "phase_max": float(np.max(phase)),
        "target_pitch_delta_mean": float(np.mean(dp)) if len(dp) else 0.0,
        "target_pitch_delta_max": float(np.max(dp)) if len(dp) else 0.0,
        "target_roll_delta_mean": float(np.mean(dr)) if len(dr) else 0.0,
        "target_roll_delta_max": float(np.max(dr)) if len(dr) else 0.0,
    }


def success_gate(row):
    return (
        str(row.get("completed")) == "True"
        and row.get("terminal_reason_classified") == "success"
        and csv_float(row, "effective_Gmax", 99.0) < 9.0
        and csv_float(row, "effective_vt_min", 0.0) > SAFE_VT
        and csv_float(row, "env_alpha_max", 999.0) < 45.0
        and abs(csv_float(row, "env_beta_max", 999.0)) < 45.0
    )


def target_stream_cost(row):
    phase = csv_float(row, "terminal_phase_deg", 0.0)
    gmax = csv_float(row, "effective_Gmax", 99.0)
    vt_min = csv_float(row, "effective_vt_min", 0.0)
    cte = csv_float(row, "CTE_mean", 100000.0)
    wing = csv_float(row, "wing_plane_error_mean", 180.0)
    vtan = csv_float(row, "velocity_tangent_error_mean", 180.0)
    nose = csv_float(row, "nose_tangent_error_mean", 180.0)
    alpha = csv_float(row, "env_alpha_max", 999.0)
    beta = abs(csv_float(row, "env_beta_max", 999.0))
    pitch_smooth = csv_float(row, "target_pitch_delta_mean", 0.0)
    roll_smooth = csv_float(row, "target_roll_delta_mean", 0.0)
    completion = 0.0 if success_gate(row) else 5.0 + max(0.0, 165.0 - phase) / 20.0
    safety = 18.0 * max(0.0, gmax - 8.5) ** 2 + 6.0 * max(0.0, SAFE_VT - vt_min) / 40.0
    safety += 4.0 * max(0.0, alpha - 40.0) / 10.0 + 4.0 * max(0.0, beta - 40.0) / 10.0
    geometry = cte / 2500.0 + wing / 40.0 + (vtan + nose) / 100.0
    smooth = pitch_smooth / 4.0 + roll_smooth / 8.0
    return float(completion + safety + geometry + smooth)


def row_from_rollout(summary, terminal_row, phase_rows, policy_name, variant):
    row = dict(summary)
    row.update(phase_stats(phase_rows))
    terminal_g = csv_float(terminal_row, "terminal_G", csv_float(row, "Gmax", 0.0))
    terminal_vt = csv_float(terminal_row, "terminal_vt", csv_float(row, "vt_min", 0.0))
    row["base_policy"] = policy_name
    row["variant"] = variant["variant"]
    row["family"] = variant.get("family", "")
    row["entry_vt"] = variant.get("entry_vt", 250.0)
    row["target_radius_m"] = variant.get("target_radius_m", 15000.0)
    row["eval_radius_m"] = variant.get("eval_radius_m", 15000.0)
    row["altitude_gain_limit_m"] = variant.get("target_altitude_gain_limit_m", "")
    row["bridge_target_vt"] = variant.get("bridge_target_vt", "")
    row["bridge_lookahead_dist"] = variant.get("bridge_lookahead_dist", "")
    row["pitch_rate_limit_deg_s"] = variant.get("pitch_rate_limit_deg_s", "")
    row["roll_rate_limit_deg_s"] = variant.get("roll_rate_limit_deg_s", "")
    row["pitch_blend_with_current"] = variant.get("pitch_blend_with_current", "")
    row["target_radius_profile"] = json.dumps(variant.get("target_radius_profile", []), sort_keys=True)
    row["terminal_G"] = terminal_g
    row["terminal_vt"] = terminal_vt
    row["effective_Gmax"] = max(csv_float(row, "Gmax", 0.0), terminal_g)
    row["effective_vt_min"] = min(csv_float(row, "vt_min", 0.0), terminal_vt)
    row["success_gate"] = success_gate(row)
    row["target_stream_cost"] = target_stream_cost(row)
    return row


SWEEP_FIELDS = [
    "base_policy",
    "variant",
    "family",
    "entry_vt",
    "altitude_gain_limit_m",
    "eval_radius_m",
    "target_radius_m",
    "completed",
    "success_gate",
    "terminal_reason_classified",
    "terminal_phase_deg",
    "phase_max",
    "steps",
    "CTE_mean",
    "CTE_p90",
    "CTE_max",
    "velocity_tangent_error_mean",
    "nose_tangent_error_mean",
    "nose_velocity_error_mean",
    "wing_plane_error_mean",
    "q_error_mean_rad",
    "Gmax",
    "terminal_G",
    "effective_Gmax",
    "env_alpha_max",
    "env_beta_max",
    "vt_min",
    "terminal_vt",
    "effective_vt_min",
    "vt_max",
    "phase145_170_CTE_mean",
    "phase145_170_Gmax",
    "phase145_170_alpha_max",
    "phase145_170_vt_min",
    "target_pitch_delta_mean",
    "target_pitch_delta_max",
    "target_roll_delta_mean",
    "target_roll_delta_max",
    "action_diff_norm_mean",
    "jitter_action_diff_mean",
    "target_stream_cost",
    "bridge_target_vt",
    "bridge_lookahead_dist",
    "pitch_rate_limit_deg_s",
    "roll_rate_limit_deg_s",
    "pitch_blend_with_current",
    "target_radius_profile",
]


def write_report(root, feasibility, rows):
    successes = [r for r in rows if success_gate(r)]
    best_cost = min(rows, key=lambda r: csv_float(r, "target_stream_cost", 1e9)) if rows else None
    best_progress = max(rows, key=lambda r: csv_float(r, "terminal_phase_deg", -1.0)) if rows else None
    best_safe_progress = max(
        [r for r in rows if csv_float(r, "effective_Gmax", 99.0) < 9.0 and csv_float(r, "effective_vt_min", 0.0) > SAFE_VT],
        key=lambda r: csv_float(r, "terminal_phase_deg", -1.0),
        default=None,
    )
    infeasible_fixed = [
        r for r in feasibility
        if float(r["angle_deg"]) == 165.0 and r["classification"] == "infeasible"
    ]
    text = [
        "# pu165 Energy Feasibility And Maneuver Reparameterization Study",
        "",
        "## Decision",
        "",
        f"- successful_target_streams: `{len(successes)}`",
        f"- best_cost_variant: `{best_cost['variant'] if best_cost else 'none'}`",
        f"- best_progress_variant: `{best_progress['variant'] if best_progress else 'none'}`",
        "- residual_training_resumed: `False`",
        "- pu170_175_180_tested: `False`",
        "",
        "## Analytical Feasibility",
        "",
        f"- Fixed-radius 165 deg infeasible rows: `{len(infeasible_fixed)}` / `{len([r for r in feasibility if float(r['angle_deg']) == 165.0])}`.",
        "- Classification uses unpowered energy height, required load estimate, and a 190 m/s safe-speed floor.",
        "",
        "## Target-Stream Sweep Summary",
        "",
        "| variant | family | completed | reason | phase | CTE | Gmax | vt_min | cost |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: csv_float(r, "target_stream_cost", 1e9)):
        text.append(
            f"| {row['variant']} | {row['family']} | {row['completed']} | "
            f"{row['terminal_reason_classified']} | {float(row['terminal_phase_deg']):.1f} | "
            f"{float(row['CTE_mean']):.1f} | {float(row['effective_Gmax']):.2f} | "
            f"{float(row['effective_vt_min']):.1f} | {float(row['target_stream_cost']):.2f} |"
        )
    text.extend(["", "## Diagnosis", ""])
    if successes:
        best_success = min(successes, key=lambda r: csv_float(r, "target_stream_cost", 1e9))
        text.append(
            f"- Found an energy-feasible pu165 target stream: `{best_success['variant']}`. "
            "Residual training may resume only after visual/ACMI validation of this target stream."
        )
    else:
        text.append("- No tested pu165 target stream completed with effective_Gmax < 9 and vt_min above the safe threshold.")
        text.append("- Do not resume residual PPO.")
    if best_safe_progress:
        text.append(
            f"- Best safe progress: `{best_safe_progress['variant']}` reached phase "
            f"`{float(best_safe_progress['terminal_phase_deg']):.1f}` with Gmax "
            f"`{float(best_safe_progress['effective_Gmax']):.2f}` and vt_min "
            f"`{float(best_safe_progress['effective_vt_min']):.1f}`."
        )
    if best_progress:
        text.append(
            f"- Furthest progress overall: `{best_progress['variant']}` reached phase "
            f"`{float(best_progress['terminal_phase_deg']):.1f}` with reason "
            f"`{best_progress['terminal_reason_classified']}`."
        )
    text.extend(
        [
            "- The fixed-radius pu165 benchmark demands much more altitude gain than the entry kinetic energy at 250-400 m/s can support without engine/controller help.",
            "- If capped-altitude variants still fail, the remaining limitation is controller/target-stream compatibility, not residual-only learning.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if successes:
        text.append("- Keep pu165 only and validate the successful target stream visually before any small residual correction.")
    else:
        text.append("- Treat fixed-radius pu165 at current entry conditions as energy-infeasible for this controller stack.")
        text.append("- Revise the benchmark before more PPO: use capped altitude gain, lower terminal pitch demand, larger-radius curriculum with explicit speed floor, or MPC/RH-TSO target-stream optimization.")
        text.append("- Do not reintroduce pu170/175/180.")
    text.extend(
        [
            "",
            "## Files",
            "",
            f"- feasibility: `{(root / 'feasibility_diagnostics.csv').resolve()}`",
            f"- radius_sweep: `{(root / 'radius_sweep.csv').resolve()}`",
            f"- rhtso_sweep: `{(root / 'rhtso_sweep.csv').resolve()}`",
            f"- phasewise: `{(root / 'phasewise').resolve()}`",
            f"- raw_terminal_info: `{(root / 'raw_terminal_info').resolve()}`",
        ]
    )
    (root / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--policies", default="update2_scale0.2")
    parser.add_argument("--limit-candidates", type=int, default=0)
    parser.add_argument("--candidate-contains", default="")
    parser.add_argument("--feasibility-only", action="store_true")
    args = parser.parse_args()

    root = args.out_dir or PLANAX_ROOT / "results/energy_feasibility_pu165" / datetime.now().strftime("%Y%m%d_%H%M")
    root.mkdir(parents=True, exist_ok=True)
    (root / "phasewise").mkdir(exist_ok=True)
    (root / "raw_terminal_info").mkdir(exist_ok=True)

    feasibility = feasibility_rows()
    write_csv(root / "feasibility_diagnostics.csv", feasibility, FEASIBILITY_FIELDS)
    write_csv(root / "radius_sweep.csv", feasibility, FEASIBILITY_FIELDS)

    variants = candidate_variants()
    filters = parse_list(args.candidate_contains)
    if filters:
        variants = [v for v in variants if any(f in v["variant"] for f in filters)]
    if args.limit_candidates > 0:
        variants = variants[: args.limit_candidates]

    write_json(
        root / "config.json",
        {
            "base": str(bridge.BASE_CKPT),
            "residual": str(bridge.BEST_RESIDUAL),
            "hard_task": HARD_TASK,
            "policies": parse_list(args.policies),
            "safe_vt": SAFE_VT,
            "candidate_count": len(variants),
            "candidate_variants": variants,
        },
    )

    rows = []
    terminal_rows = []
    if not args.feasibility_only:
        env, net, net_params, residual_net, residual_params, _ = bridge.load_models()
        policies = policy_configs()
        for policy_name in parse_list(args.policies):
            cfg = policies[policy_name]
            for variant in variants:
                summary, terminal_row, phase_rows, raw_info = trace.run_trace_test(
                    env,
                    net,
                    net_params,
                    residual_net,
                    residual_params,
                    f"{policy_name}__{variant['variant']}",
                    HARD_TASK,
                    cfg,
                    variant=variant,
                )
                row = row_from_rollout(summary, terminal_row, phase_rows, policy_name, variant)
                rows.append(row)
                terminal_row["base_policy"] = policy_name
                terminal_row["variant"] = variant["variant"]
                terminal_rows.append(terminal_row)
                write_csv(root / "rhtso_sweep.csv", rows, SWEEP_FIELDS)
                write_csv(root / "terminal_states.csv", terminal_rows, ["base_policy", "variant"] + trace.TERMINAL_FIELDS)
                write_csv(root / "phasewise" / f"{policy_name}__{variant['variant']}.csv", phase_rows, bridge.PHASE_FIELDS)
                write_json(root / "raw_terminal_info" / f"{policy_name}__{variant['variant']}.json", raw_info)
                print(
                    f"{policy_name} {variant['variant']} completed={row['completed']} "
                    f"term={row['terminal_reason_classified']} phase={float(row['terminal_phase_deg']):.1f} "
                    f"CTE={float(row['CTE_mean']):.1f} G={float(row['effective_Gmax']):.2f} "
                    f"vtmin={float(row['effective_vt_min']):.1f} cost={float(row['target_stream_cost']):.2f}",
                    flush=True,
                )
    if not rows:
        write_csv(root / "rhtso_sweep.csv", [], SWEEP_FIELDS)
        write_csv(root / "terminal_states.csv", [], ["base_policy", "variant"] + trace.TERMINAL_FIELDS)
    write_report(root, feasibility, rows)
    print(f"energy_feasibility_dir={root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
