import argparse
import csv
import json
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
from experiments.hierarchical_trajectory_tracking.export_acmi import write_acmi


PLANAX_ROOT = Path(__file__).resolve().parent
TASK = "pu165_R15000"
SAFE_VT = 190.0


PHASE_FIELDS_EXT = [
    "policy",
    "task",
    "step",
    "time_sec",
    "north",
    "east",
    "phase",
    "CTE",
    "velocity_tangent_error",
    "nose_tangent_error",
    "nose_velocity_error",
    "wing_plane_error",
    "q_error_norm",
    "alpha",
    "beta",
    "G",
    "vt",
    "altitude",
    "pitch",
    "roll",
    "yaw",
    "target_pitch",
    "target_roll",
    "elevator_action",
    "aileron_action",
    "rudder_action",
    "throttle_action",
    "speedbrake_action",
    "base_logits_norm",
    "residual_logits_norm",
    "final_base_logits_norm",
    "residual_gate_value",
    "action_difference_from_base",
]


SUMMARY_FIELDS = [
    "candidate",
    "target_stream_mode",
    "completed",
    "useful_gate",
    "terminal_reason_classified",
    "terminal_phase_deg",
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
    "vt_min",
    "vt_max",
    "env_alpha_max",
    "env_beta_max",
    "phase80_100_CTE_mean",
    "phase80_100_velocity_tangent_error_mean",
    "phase80_100_nose_tangent_error_mean",
    "phase80_100_wing_plane_error_mean",
    "phase100_130_CTE_mean",
    "phase100_130_velocity_tangent_error_mean",
    "phase100_130_nose_tangent_error_mean",
    "phase100_130_wing_plane_error_mean",
    "phase130_165_CTE_mean",
    "phase130_165_velocity_tangent_error_mean",
    "phase130_165_nose_tangent_error_mean",
    "phase130_165_wing_plane_error_mean",
    "after100_velocity_tangent_error_mean",
    "after100_nose_tangent_error_mean",
    "after100_CTE_mean",
    "target_pitch_delta_mean",
    "target_pitch_delta_max",
    "target_roll_delta_mean",
    "target_roll_delta_max",
    "acmi_path",
]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(row, key, digits=1, default=""):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return default


def candidate_variants():
    common = {
        "entry_vt": 300.0,
        "target_vt": 300.0,
        "eval_radius_m": 15000.0,
        "target_radius_m": 15000.0,
        "eval_altitude_gain_limit_m": 20000.0,
        "target_altitude_gain_limit_m": 20000.0,
        "pitch_rate_limit_deg_s": 8.0,
        "roll_rate_limit_deg_s": 30.0,
    }
    return [
        {
            **common,
            "candidate": "pure_pursuit_moving_lookahead",
            "target_stream_mode": "pure_pursuit",
        },
        {
            **common,
            "candidate": "tangent_following",
            "target_stream_mode": "tangent_following",
        },
        {
            **common,
            "candidate": "pursuit_tangent_blend_w050",
            "target_stream_mode": "pursuit_tangent_blend",
            "w_pursuit": 0.50,
        },
        {
            **common,
            "candidate": "pursuit_tangent_blend_w025",
            "target_stream_mode": "pursuit_tangent_blend",
            "w_pursuit": 0.25,
        },
        {
            **common,
            "candidate": "phase_scheduled_blend",
            "target_stream_mode": "phase_scheduled_blend",
        },
        {
            **common,
            "candidate": "curvature_aware_smooth",
            "target_stream_mode": "curvature_aware",
            "bridge_lookahead_dist": 1100.0,
            "pitch_rate_limit_deg_s": 6.0,
            "roll_rate_limit_deg_s": 24.0,
        },
    ]


def eval_waypoints():
    wps, meta = bridge.ev.vertical_pullup_arc(
        0,
        0,
        5000,
        0.0,
        radius=15000.0,
        arc_angle_deg=165.0,
        n_points=110,
    )
    return trace.limit_altitude_gain(wps, meta, 20000.0)[0]


def arr(rows, key):
    return np.asarray([f(r, key) for r in rows], dtype=np.float64)


def phase_window_metrics(rows, start, end, prefix):
    if not rows:
        return {
            f"{prefix}_CTE_mean": "",
            f"{prefix}_velocity_tangent_error_mean": "",
            f"{prefix}_nose_tangent_error_mean": "",
            f"{prefix}_wing_plane_error_mean": "",
        }
    phase = arr(rows, "phase")
    mask = (phase >= start) & (phase <= end)
    if not np.any(mask):
        return {
            f"{prefix}_CTE_mean": "",
            f"{prefix}_velocity_tangent_error_mean": "",
            f"{prefix}_nose_tangent_error_mean": "",
            f"{prefix}_wing_plane_error_mean": "",
        }
    return {
        f"{prefix}_CTE_mean": float(arr(rows, "CTE")[mask].mean()),
        f"{prefix}_velocity_tangent_error_mean": float(arr(rows, "velocity_tangent_error")[mask].mean()),
        f"{prefix}_nose_tangent_error_mean": float(arr(rows, "nose_tangent_error")[mask].mean()),
        f"{prefix}_wing_plane_error_mean": float(arr(rows, "wing_plane_error")[mask].mean()),
    }


def summarize(candidate, summary, phase_rows, acmi_path):
    cte = arr(phase_rows, "CTE")
    target_pitch = arr(phase_rows, "target_pitch")
    target_roll = arr(phase_rows, "target_roll")
    phase = arr(phase_rows, "phase")
    after100 = phase >= 100.0
    row = {
        "candidate": candidate["candidate"],
        "target_stream_mode": candidate["target_stream_mode"],
        "completed": summary["completed"],
        "terminal_reason_classified": summary["terminal_reason_classified"],
        "terminal_phase_deg": summary["terminal_phase_deg"],
        "steps": summary["steps"],
        "CTE_mean": summary["CTE_mean"],
        "CTE_p90": float(np.percentile(cte, 90)) if len(cte) else "",
        "CTE_max": float(np.max(cte)) if len(cte) else "",
        "velocity_tangent_error_mean": summary["velocity_tangent_error_mean"],
        "nose_tangent_error_mean": summary["nose_tangent_error_mean"],
        "nose_velocity_error_mean": summary["nose_velocity_error_mean"],
        "wing_plane_error_mean": summary["wing_plane_error_mean"],
        "q_error_mean_rad": summary["q_error_mean_rad"],
        "Gmax": summary["Gmax"],
        "vt_min": summary["vt_min"],
        "vt_max": summary["vt_max"],
        "env_alpha_max": summary["env_alpha_max"],
        "env_beta_max": summary["env_beta_max"],
        "after100_velocity_tangent_error_mean": float(arr(phase_rows, "velocity_tangent_error")[after100].mean()) if np.any(after100) else "",
        "after100_nose_tangent_error_mean": float(arr(phase_rows, "nose_tangent_error")[after100].mean()) if np.any(after100) else "",
        "after100_CTE_mean": float(cte[after100].mean()) if np.any(after100) else "",
        "target_pitch_delta_mean": float(np.abs(np.diff(target_pitch)).mean()) if len(target_pitch) > 1 else 0.0,
        "target_pitch_delta_max": float(np.abs(np.diff(target_pitch)).max()) if len(target_pitch) > 1 else 0.0,
        "target_roll_delta_mean": float(np.abs(np.diff(target_roll)).mean()) if len(target_roll) > 1 else 0.0,
        "target_roll_delta_max": float(np.abs(np.diff(target_roll)).max()) if len(target_roll) > 1 else 0.0,
        "acmi_path": str(acmi_path),
    }
    for start, end, prefix in [
        (80.0, 100.0, "phase80_100"),
        (100.0, 130.0, "phase100_130"),
        (130.0, 165.0, "phase130_165"),
    ]:
        row.update(phase_window_metrics(phase_rows, start, end, prefix))
    row["useful_gate"] = (
        str(row["completed"]) == "True"
        and row["terminal_reason_classified"] == "success"
        and f(row, "Gmax", 99.0) < 9.0
        and f(row, "vt_min", 0.0) > SAFE_VT
        and f(row, "env_alpha_max", 999.0) < 45.0
        and abs(f(row, "env_beta_max", 999.0)) < 45.0
        and f(row, "after100_velocity_tangent_error_mean", 999.0) < 20.0
        and f(row, "after100_nose_tangent_error_mean", 999.0) < 25.0
    )
    return row


def write_candidate_acmi(path, waypoints, phase_rows, name):
    traj = {
        "t": [f(r, "time_sec") for r in phase_rows],
        "n": [f(r, "north") for r in phase_rows],
        "e": [f(r, "east") for r in phase_rows],
        "a": [f(r, "altitude") for r in phase_rows],
        "roll": [f(r, "roll") for r in phase_rows],
        "pitch": [f(r, "pitch") for r in phase_rows],
        "yaw": [f(r, "yaw") for r in phase_rows],
    }
    write_acmi(str(path), waypoints, traj, aircraft_name=name, color="Cyan")


def write_report(root, rows):
    baseline = next((r for r in rows if r["candidate"] == "pure_pursuit_moving_lookahead"), None)
    useful = [r for r in rows if str(r["useful_gate"]) == "True"]
    best_after100 = min(
        rows,
        key=lambda r: (
            f(r, "after100_velocity_tangent_error_mean", 999.0)
            + f(r, "after100_nose_tangent_error_mean", 999.0)
            + f(r, "after100_CTE_mean", 999999.0) / 1000.0
        ),
    ) if rows else None
    text = [
        "# pu165 Tangent / Curvature Target-Stream Report",
        "",
        "## Decision",
        "",
        f"- useful_candidates: `{len(useful)}`",
        f"- best_after100_candidate: `{best_after100['candidate'] if best_after100 else 'none'}`",
        "- residual_training_resumed: `False`",
        "- pu170_175_180_tested: `False`",
        "",
        "## Candidate Summary",
        "",
        "| candidate | completed | reason | phase | CTE | after100 CTE | after100 vtan | after100 nose | Gmax | vt_min | useful |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {row['candidate']} | {row['completed']} | {row['terminal_reason_classified']} | "
            f"{fmt(row, 'terminal_phase_deg')} | {fmt(row, 'CTE_mean')} | "
            f"{fmt(row, 'after100_CTE_mean')} | {fmt(row, 'after100_velocity_tangent_error_mean')} | "
            f"{fmt(row, 'after100_nose_tangent_error_mean')} | {fmt(row, 'Gmax', 2)} | "
            f"{fmt(row, 'vt_min')} | {row['useful_gate']} |"
        )
    text.extend(["", "## Interpretation", ""])
    if baseline and best_after100:
        text.append(
            f"- Baseline after-100 CTE/tangent/nose: "
            f"`{float(baseline['after100_CTE_mean']):.1f}`, "
            f"`{float(baseline['after100_velocity_tangent_error_mean']):.1f}`, "
            f"`{float(baseline['after100_nose_tangent_error_mean']):.1f}`."
        )
        text.append(
            f"- Best after-100 candidate `{best_after100['candidate']}`: "
            f"`{float(best_after100['after100_CTE_mean']):.1f}`, "
            f"`{float(best_after100['after100_velocity_tangent_error_mean']):.1f}`, "
            f"`{float(best_after100['after100_nose_tangent_error_mean']):.1f}`."
        )
    curvature = next((r for r in rows if r["candidate"] == "curvature_aware_smooth"), None)
    if curvature and baseline:
        text.append(
            f"- `curvature_aware_smooth` completes safely and lowers global CTE "
            f"from `{float(baseline['CTE_mean']):.1f}` to `{float(curvature['CTE_mean']):.1f}`, "
            f"but after-100 CTE worsens from `{float(baseline['after100_CTE_mean']):.1f}` "
            f"to `{float(curvature['after100_CTE_mean']):.1f}` and after-100 velocity-tangent "
            f"error worsens from `{float(baseline['after100_velocity_tangent_error_mean']):.1f}` "
            f"to `{float(curvature['after100_velocity_tangent_error_mean']):.1f}`."
        )
    text.append(
        "- Main answer: the tested tangent-aware streams do not demonstrate a fix for the post-100 deg inside-cut. "
        "Pure tangent and constant blends overload immediately; phase-scheduled tangent survives but drifts far off the arc; curvature-aware smoothing is safe but does not improve the after-100 tracking metrics over the pure-pursuit baseline."
    )
    text.append("- ACMI files were exported for visual inspection before any training decision.")
    text.extend(
        [
            "",
            "## ACMI Files",
            "",
        ]
    )
    for row in rows:
        text.append(f"- `{row['acmi_path']}`")
    text.extend(
        [
            "",
            "## Files",
            "",
            f"- summary: `{(root / 'summary.csv').resolve()}`",
            f"- phasewise: `{(root / 'phasewise').resolve()}`",
            f"- acmi: `{(root / 'acmi').resolve()}`",
        ]
    )
    (root / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--candidates", default="")
    args = parser.parse_args()

    root = args.out_dir or PLANAX_ROOT / "results/pu165_tangent_curvature_target_stream" / datetime.now().strftime("%Y%m%d_%H%M")
    for sub in ["phasewise", "raw_terminal_info", "acmi"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    candidates = candidate_variants()
    if args.candidates:
        keep = {x.strip() for x in args.candidates.split(",") if x.strip()}
        candidates = [c for c in candidates if c["candidate"] in keep]

    write_json(
        root / "config.json",
        {
            "base": str(bridge.BASE_CKPT),
            "residual": str(bridge.BEST_RESIDUAL),
            "policy": "base_epoch619_plus_residual_update_2_scale_0.2",
            "task": TASK,
            "candidates": candidates,
        },
    )

    env, net, net_params, residual_net, residual_params, _ = bridge.load_models()
    residual_cfg = bridge.make_residual_cfg(scale=0.20)
    waypoints = eval_waypoints()
    rows = []
    terminal_rows = []
    for candidate in candidates:
        summary, terminal_row, phase_rows, raw_info = trace.run_trace_test(
            env,
            net,
            net_params,
            residual_net,
            residual_params,
            f"update2_scale0.2__{candidate['candidate']}",
            TASK,
            residual_cfg,
            variant=candidate,
        )
        acmi_path = root / "acmi" / f"{candidate['candidate']}.acmi"
        write_candidate_acmi(acmi_path, waypoints, phase_rows, candidate["candidate"])
        row = summarize(candidate, summary, phase_rows, acmi_path)
        rows.append(row)
        terminal_row["candidate"] = candidate["candidate"]
        terminal_rows.append(terminal_row)
        write_csv(root / "summary.csv", rows, SUMMARY_FIELDS)
        write_csv(root / "terminal_states.csv", terminal_rows, ["candidate"] + trace.TERMINAL_FIELDS)
        write_csv(root / "phasewise" / f"{candidate['candidate']}.csv", phase_rows, PHASE_FIELDS_EXT)
        write_json(root / "raw_terminal_info" / f"{candidate['candidate']}.json", raw_info)
        print(
            f"{candidate['candidate']} completed={row['completed']} reason={row['terminal_reason_classified']} "
            f"phase={fmt(row, 'terminal_phase_deg')} CTE={fmt(row, 'CTE_mean')} "
            f"after100_vtan={fmt(row, 'after100_velocity_tangent_error_mean')} "
            f"after100_nose={fmt(row, 'after100_nose_tangent_error_mean')} "
            f"G={fmt(row, 'Gmax', 2)} vtmin={fmt(row, 'vt_min')}",
            flush=True,
        )
    write_report(root, rows)
    print(f"target_stream_dir={root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
