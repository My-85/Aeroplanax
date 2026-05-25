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
    "family",
    "band",
    "local_pitch_bias_deg",
    "local_lookahead_scale",
    "local_target_vt_delta",
    "completed",
    "termination",
    "terminal_phase",
    "CTE_mean",
    "CTE_p90",
    "CTE_max",
    "velocity_tangent_error_mean",
    "nose_tangent_error_mean",
    "nose_velocity_error_mean",
    "wing_plane_error_mean",
    "q_error_mean_rad",
    "after100_CTE_mean",
    "after100_velocity_tangent_error_mean",
    "after100_nose_tangent_error_mean",
    "after100_wing_plane_error_mean",
    "Gmax",
    "vt_min",
    "vt_max",
    "alpha_min",
    "alpha_max",
    "beta_min",
    "beta_max",
    "target_pitch_delta_mean",
    "target_pitch_delta_max",
    "target_roll_delta_mean",
    "target_roll_delta_max",
    "numeric_improves_after100",
    "safety_gate",
    "numeric_useful_gate",
    "visual_acmi_verdict",
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


def common_variant():
    return {
        "entry_vt": 300.0,
        "target_vt": 300.0,
        "eval_radius_m": 15000.0,
        "target_radius_m": 15000.0,
        "eval_altitude_gain_limit_m": 20000.0,
        "target_altitude_gain_limit_m": 20000.0,
        "target_stream_mode": "pure_pursuit",
        "pitch_rate_limit_deg_s": 8.0,
        "roll_rate_limit_deg_s": 30.0,
        "local_correction_margin_deg": 5.0,
    }


def band_name(band):
    return f"{int(band[0])}_{int(band[1])}"


def make_candidate(name, family, band=None, pitch_bias=None, lookahead_scale=None, vt_delta=None):
    v = common_variant()
    v["candidate"] = name
    v["family"] = family
    if band is not None:
        v["local_correction_band"] = list(band)
        v["band"] = band_name(band)
    else:
        v["band"] = ""
    if pitch_bias is not None:
        v["local_pitch_bias_deg"] = float(pitch_bias)
    if lookahead_scale is not None:
        v["local_lookahead_scale"] = float(lookahead_scale)
    if vt_delta is not None:
        v["local_target_vt_delta"] = float(vt_delta)
    return v


def candidate_variants():
    bands = [(90.0, 130.0), (100.0, 145.0)]
    variants = [make_candidate("baseline_pure_pursuit", "baseline")]
    for band in bands:
        for bias in [2.0, 4.0, 6.0]:
            variants.append(
                make_candidate(
                    f"pitch_p{int(bias)}_b{band_name(band)}",
                    "phase_gated_pitch_bias",
                    band=band,
                    pitch_bias=bias,
                )
            )
    for band in bands:
        for scale in [0.8, 0.6]:
            variants.append(
                make_candidate(
                    f"lookahead_x{str(scale).replace('.', '')}_b{band_name(band)}",
                    "phase_gated_smaller_lookahead",
                    band=band,
                    lookahead_scale=scale,
                )
            )
    for band in bands:
        for vt_delta in [-20.0, -10.0, 10.0]:
            sign = "m" if vt_delta < 0 else "p"
            variants.append(
                make_candidate(
                    f"vt_{sign}{int(abs(vt_delta))}_b{band_name(band)}",
                    "phase_gated_target_vt_adjustment",
                    band=band,
                    vt_delta=vt_delta,
                )
            )
    for band in bands:
        variants.extend(
            [
                make_candidate(
                    f"look08_pitch_p2_b{band_name(band)}",
                    "joint_local_correction",
                    band=band,
                    lookahead_scale=0.8,
                    pitch_bias=2.0,
                ),
                make_candidate(
                    f"look08_pitch_p4_b{band_name(band)}",
                    "joint_local_correction",
                    band=band,
                    lookahead_scale=0.8,
                    pitch_bias=4.0,
                ),
                make_candidate(
                    f"look08_vt_m10_b{band_name(band)}",
                    "joint_local_correction",
                    band=band,
                    lookahead_scale=0.8,
                    vt_delta=-10.0,
                ),
                make_candidate(
                    f"look08_vt_p10_b{band_name(band)}",
                    "joint_local_correction",
                    band=band,
                    lookahead_scale=0.8,
                    vt_delta=10.0,
                ),
                make_candidate(
                    f"pitch_p2_vt_m10_b{band_name(band)}",
                    "joint_local_correction",
                    band=band,
                    pitch_bias=2.0,
                    vt_delta=-10.0,
                ),
                make_candidate(
                    f"pitch_p2_vt_p10_b{band_name(band)}",
                    "joint_local_correction",
                    band=band,
                    pitch_bias=2.0,
                    vt_delta=10.0,
                ),
            ]
        )
    return variants


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


def phase_mean(rows, key, start=100.0, end=165.0):
    if not rows:
        return ""
    phase = arr(rows, "phase")
    mask = (phase >= start) & (phase <= end)
    if not np.any(mask):
        return ""
    return float(arr(rows, key)[mask].mean())


def summarize(candidate, summary, phase_rows, acmi_path, baseline=None):
    cte = arr(phase_rows, "CTE")
    alpha = arr(phase_rows, "alpha")
    beta = arr(phase_rows, "beta")
    target_pitch = arr(phase_rows, "target_pitch")
    target_roll = arr(phase_rows, "target_roll")
    row = {
        "candidate": candidate["candidate"],
        "family": candidate["family"],
        "band": candidate.get("band", ""),
        "local_pitch_bias_deg": candidate.get("local_pitch_bias_deg", ""),
        "local_lookahead_scale": candidate.get("local_lookahead_scale", ""),
        "local_target_vt_delta": candidate.get("local_target_vt_delta", ""),
        "completed": summary["completed"],
        "termination": summary["terminal_reason_classified"],
        "terminal_phase": summary["terminal_phase_deg"],
        "CTE_mean": summary["CTE_mean"],
        "CTE_p90": float(np.percentile(cte, 90)) if len(cte) else "",
        "CTE_max": float(np.max(cte)) if len(cte) else "",
        "velocity_tangent_error_mean": summary["velocity_tangent_error_mean"],
        "nose_tangent_error_mean": summary["nose_tangent_error_mean"],
        "nose_velocity_error_mean": summary["nose_velocity_error_mean"],
        "wing_plane_error_mean": summary["wing_plane_error_mean"],
        "q_error_mean_rad": summary["q_error_mean_rad"],
        "after100_CTE_mean": phase_mean(phase_rows, "CTE"),
        "after100_velocity_tangent_error_mean": phase_mean(phase_rows, "velocity_tangent_error"),
        "after100_nose_tangent_error_mean": phase_mean(phase_rows, "nose_tangent_error"),
        "after100_wing_plane_error_mean": phase_mean(phase_rows, "wing_plane_error"),
        "Gmax": summary["Gmax"],
        "vt_min": summary["vt_min"],
        "vt_max": summary["vt_max"],
        "alpha_min": float(np.min(alpha)) if len(alpha) else "",
        "alpha_max": float(np.max(alpha)) if len(alpha) else "",
        "beta_min": float(np.min(beta)) if len(beta) else "",
        "beta_max": float(np.max(beta)) if len(beta) else "",
        "target_pitch_delta_mean": float(np.abs(np.diff(target_pitch)).mean()) if len(target_pitch) > 1 else 0.0,
        "target_pitch_delta_max": float(np.abs(np.diff(target_pitch)).max()) if len(target_pitch) > 1 else 0.0,
        "target_roll_delta_mean": float(np.abs(np.diff(target_roll)).mean()) if len(target_roll) > 1 else 0.0,
        "target_roll_delta_max": float(np.abs(np.diff(target_roll)).max()) if len(target_roll) > 1 else 0.0,
        "acmi_path": str(acmi_path),
    }
    safety = (
        str(row["completed"]) == "True"
        and row["termination"] == "success"
        and f(row, "Gmax", 99.0) < 9.0
        and f(row, "vt_min", 0.0) > SAFE_VT
        and f(row, "alpha_max", 999.0) < 45.0
        and abs(f(row, "beta_min", 999.0)) < 45.0
        and abs(f(row, "beta_max", 999.0)) < 45.0
    )
    improves = False
    if baseline is not None:
        improves = (
            f(row, "after100_CTE_mean", 1e9) < f(baseline, "after100_CTE_mean", 1e9)
            and f(row, "after100_velocity_tangent_error_mean", 1e9)
            < f(baseline, "after100_velocity_tangent_error_mean", 1e9)
        )
    row["numeric_improves_after100"] = improves
    row["safety_gate"] = safety
    row["numeric_useful_gate"] = safety and improves
    if candidate["family"] == "baseline":
        row["visual_acmi_verdict"] = "baseline_reference"
    elif row["numeric_useful_gate"]:
        row["visual_acmi_verdict"] = "needs_tacview_confirmation_less_inward_cut_proxy"
    else:
        row["visual_acmi_verdict"] = "likely_still_inside_cut_or_not_safe_by_metrics"
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
    baseline = rows[0] if rows else None
    numeric_useful = [r for r in rows if str(r.get("numeric_useful_gate")) == "True"]
    best = min(
        rows,
        key=lambda r: (
            f(r, "after100_CTE_mean", 1e9) + 50.0 * f(r, "after100_velocity_tangent_error_mean", 1e9)
        ),
    ) if rows else None
    text = [
        "# pu165 Local Correction Search",
        "",
        "## Decision",
        "",
        f"- numeric_useful_candidates: `{len(numeric_useful)}`",
        f"- best_numeric_candidate: `{best['candidate'] if best else 'none'}`",
        "- residual_training_resumed: `False`",
        "- pu170_175_180_tested: `False`",
        "- promotion: `none`",
        "",
        "## Baseline",
        "",
    ]
    if baseline:
        text.extend(
            [
                f"- after100 CTE: `{float(baseline['after100_CTE_mean']):.1f}`",
                f"- after100 velocity_tangent_error: `{float(baseline['after100_velocity_tangent_error_mean']):.1f}`",
                f"- after100 nose_tangent_error: `{float(baseline['after100_nose_tangent_error_mean']):.1f}`",
                f"- Gmax: `{float(baseline['Gmax']):.2f}`",
                f"- vt_min: `{float(baseline['vt_min']):.1f}`",
            ]
        )
    text.extend(
        [
            "",
            "## Summary",
            "",
            "| candidate | family | completed | term | after100 CTE | after100 vtan | after100 nose | Gmax | vt_min | numeric useful | ACMI verdict |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        text.append(
            f"| {row['candidate']} | {row['family']} | {row['completed']} | {row['termination']} | "
            f"{fmt(row, 'after100_CTE_mean')} | {fmt(row, 'after100_velocity_tangent_error_mean')} | "
            f"{fmt(row, 'after100_nose_tangent_error_mean')} | {fmt(row, 'Gmax', 2)} | "
            f"{fmt(row, 'vt_min')} | {row['numeric_useful_gate']} | {row['visual_acmi_verdict']} |"
        )
    text.extend(["", "## Interpretation", ""])
    if numeric_useful:
        text.append(
            "- Some candidates improve the numeric after-100 CTE and velocity-tangent metrics while preserving safety. They are not promoted until Tacview confirms visibly less inward cutting."
        )
    else:
        text.append(
            "- No local hand-designed correction satisfied the numeric useful gate. This means the tested pitch/lookahead/vt micro-corrections are insufficient by the requested after-100 criteria."
        )
    text.append(
        "- `visual_acmi_verdict` is conservative: ACMI files are exported, but final visual confirmation must be made in Tacview. No candidate is promoted from metrics alone."
    )
    text.extend(
        [
            "",
            "## ACMI",
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
            f"- raw_terminal_info: `{(root / 'raw_terminal_info').resolve()}`",
            f"- acmi: `{(root / 'acmi').resolve()}`",
        ]
    )
    (root / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit-candidates", type=int, default=0)
    parser.add_argument("--candidate-contains", default="")
    args = parser.parse_args()

    root = args.out_dir or PLANAX_ROOT / "results/pu165_local_correction_search" / datetime.now().strftime("%Y%m%d_%H%M")
    for sub in ["phasewise", "raw_terminal_info", "acmi"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    candidates = candidate_variants()
    filters = [x.strip() for x in args.candidate_contains.split(",") if x.strip()]
    if filters:
        candidates = [c for c in candidates if any(flt in c["candidate"] for flt in filters)]
    if args.limit_candidates > 0:
        candidates = candidates[: args.limit_candidates]

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
    baseline = None
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
        row = summarize(candidate, summary, phase_rows, acmi_path, baseline=baseline)
        if candidate["family"] == "baseline":
            baseline = row
            row = summarize(candidate, summary, phase_rows, acmi_path, baseline=baseline)
        rows.append(row)
        terminal_row["candidate"] = candidate["candidate"]
        terminal_rows.append(terminal_row)
        write_csv(root / "summary.csv", rows, SUMMARY_FIELDS)
        write_csv(root / "terminal_states.csv", terminal_rows, ["candidate"] + trace.TERMINAL_FIELDS)
        write_csv(root / "phasewise" / f"{candidate['candidate']}.csv", phase_rows, PHASE_FIELDS_EXT)
        write_json(root / "raw_terminal_info" / f"{candidate['candidate']}.json", raw_info)
        print(
            f"{candidate['candidate']} completed={row['completed']} term={row['termination']} "
            f"phase={fmt(row, 'terminal_phase')} after100_CTE={fmt(row, 'after100_CTE_mean')} "
            f"after100_vtan={fmt(row, 'after100_velocity_tangent_error_mean')} "
            f"after100_nose={fmt(row, 'after100_nose_tangent_error_mean')} "
            f"G={fmt(row, 'Gmax', 2)} vtmin={fmt(row, 'vt_min')} useful={row['numeric_useful_gate']}",
            flush=True,
        )
    write_report(root, rows)
    print(f"local_correction_dir={root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
