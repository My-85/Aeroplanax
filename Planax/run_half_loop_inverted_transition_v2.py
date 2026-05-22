import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


PLANAX_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PLANAX_ROOT / "paper/second_paper/half_loop_inverted_transition_v2_config.json"
GRADE_VALUE = {"A": 4, "B": 3, "C": 2, "F": 1, "Fail": 0, "": 0}

HORIZONTAL_TASKS = [
    "level_circle_R3000_right",
    "level_circle_R3000_left",
    "level_circle_R5000_right",
    "level_circle_R5000_left",
    "s_curve_A3000",
    "figure_eight_R5000",
    "mild_climb_p1000m",
    "mild_descent_m1000m",
]

STABLE_LOOP_TASKS = ["pu060_R12000", "pu090_R12000", "pu150_R12000"]
TARGET_LOOP_TASKS = ["pu175_R15000", "pu180_R15000"]
GEOMETRY_KEYS = [
    "wing_plane_error_mean",
    "nose_tangent_error_mean",
    "velocity_tangent_error_mean",
]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def by_task(rows, policy=None):
    out = {}
    for row in rows:
        if policy is not None and row.get("policy") != policy:
            continue
        out[row.get("task") or row.get("name")] = row
    return out


def grade_not_worse(candidate, baseline, key="grade_loop_quality"):
    return GRADE_VALUE.get(candidate.get(key, ""), 0) >= GRADE_VALUE.get(baseline.get(key, ""), 0)


def metric_improved(candidate, baseline, key, min_delta):
    return f(candidate, key, 1e9) <= f(baseline, key, 1e9) - min_delta


def read_saved_checkpoint(train_log: Path) -> str:
    rows = read_csv(train_log)
    if not rows or not rows[-1].get("saved_checkpoint"):
        raise RuntimeError(f"Could not read saved_checkpoint from {train_log}")
    return rows[-1]["saved_checkpoint"]


def run_command(cmd, env, dry_run=False):
    print(" ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PLANAX_ROOT), env=env, check=True)


def horizontal_regressions(candidate_rows, baseline_rows):
    cand = by_task(candidate_rows, "candidate")
    base = by_task(baseline_rows, "candidate") or by_task(baseline_rows, "baseline_epoch600")
    regressions = []
    overload = []
    altitude = []
    for task in HORIZONTAL_TASKS:
        c = cand.get(task)
        b = base.get(task)
        if not c or not b:
            regressions.append(f"{task}:missing")
            continue
        if f(c, "success_rate") < f(b, "success_rate") - 0.10:
            regressions.append(f"{task}:success {f(b,'success_rate'):.2f}->{f(c,'success_rate'):.2f}")
        if f(c, "crash_rate") > f(b, "crash_rate") + 0.001:
            regressions.append(f"{task}:crash {f(b,'crash_rate'):.2f}->{f(c,'crash_rate'):.2f}")
        if f(c, "Gmax_mean") > f(b, "Gmax_mean") + 0.35:
            item = f"{task}:Gmax {f(b,'Gmax_mean'):.2f}->{f(c,'Gmax_mean'):.2f}"
            regressions.append(item)
            overload.append(task)
        if abs(f(c, "altitude_drift_mean")) > abs(f(b, "altitude_drift_mean")) + 100.0:
            item = f"{task}:alt_drift {f(b,'altitude_drift_mean'):.1f}->{f(c,'altitude_drift_mean'):.1f}"
            regressions.append(item)
            altitude.append(task)
    return regressions, overload, altitude


def loop_regressions(candidate_rows, baseline_rows):
    cand = by_task(candidate_rows)
    base = by_task(baseline_rows)
    regressions = []
    overload = []
    for task in STABLE_LOOP_TASKS:
        c = cand.get(task)
        b = base.get(task)
        if not c or not b:
            regressions.append(f"{task}:missing")
            continue
        if not grade_not_worse(c, b):
            regressions.append(f"{task}:grade {b.get('grade_loop_quality')}->{c.get('grade_loop_quality')}")
        if f(c, "Gmax") > f(b, "Gmax") + 0.35:
            regressions.append(f"{task}:Gmax {f(b,'Gmax'):.2f}->{f(c,'Gmax'):.2f}")
            overload.append(task)
        if f(c, "vt_min") < f(b, "vt_min") - 5.0:
            regressions.append(f"{task}:vt_min {f(b,'vt_min'):.1f}->{f(c,'vt_min'):.1f}")
    return regressions, overload


def loop_improvement(candidate_rows, baseline_rows):
    cand = by_task(candidate_rows)
    base = by_task(baseline_rows)
    per_task = {}
    any_target_all_geometry = False
    any_useful = False
    for task in TARGET_LOOP_TASKS:
        c = cand.get(task)
        b = base.get(task)
        if not c or not b:
            per_task[task] = {"missing": True}
            continue
        metrics = {
            key: metric_improved(c, b, key, 2.0)
            for key in GEOMETRY_KEYS
        }
        metrics["q_error_mean_rad"] = metric_improved(c, b, "q_error_mean_rad", 0.05)
        metrics["env_alpha_max"] = f(c, "env_alpha_max", 1e9) <= f(b, "env_alpha_max", 1e9) - 1.0
        metrics["grade_not_worse"] = grade_not_worse(c, b)
        metrics["completed_improved"] = int(str(c.get("completed")) == "True") > int(str(b.get("completed")) == "True")
        per_task[task] = metrics
        all_geom = all(metrics[key] for key in GEOMETRY_KEYS)
        any_target_all_geometry = any_target_all_geometry or all_geom
        any_useful = any_useful or all_geom or metrics["q_error_mean_rad"] or metrics["completed_improved"]
    return per_task, any_target_all_geometry, any_useful


def evaluate_gate(horizontal_rows, baseline_horizontal_rows, loop_rows, baseline_loop_rows):
    h_reg, h_overload, h_alt = horizontal_regressions(horizontal_rows, baseline_horizontal_rows)
    l_reg, l_overload = loop_regressions(loop_rows, baseline_loop_rows)
    improvements, target_geom_improved, any_useful = loop_improvement(loop_rows, baseline_loop_rows)
    overload = h_overload + l_overload
    gate_pass = (
        not h_reg
        and not l_reg
        and target_geom_improved
        and not overload
        and not h_alt
    )
    return {
        "gate_pass": gate_pass,
        "horizontal_regressions": h_reg,
        "loop_retention_regressions": l_reg,
        "overload_increase": overload,
        "altitude_drift_regressions": h_alt,
        "target_loop_improvements": improvements,
        "target_geometry_all_improved": target_geom_improved,
        "any_useful_improvement": any_useful and not h_reg and not l_reg,
    }


def adjust_config(base_cfg, gate, current_round):
    cfg = deepcopy(base_cfg)
    env_params = cfg.setdefault("ENV_PARAMS", {})
    if gate["horizontal_regressions"]:
        cfg["LOADDIR"] = base_cfg["LOADDIR"]
        cfg["LR"] = max(float(cfg.get("LR", 1e-5)) * 0.75, 5e-6)
        env_params["horizontal_proxy_task_prob"] = min(float(env_params.get("horizontal_proxy_task_prob", 0.40)) + 0.10, 0.50)
        env_params["original_task_prob"] = max(float(env_params.get("original_task_prob", 0.30)), 0.30)
        env_params["half_loop_partial_prob"] = 0.01
        env_params["half_loop_transition_prob"] = min(float(env_params.get("half_loop_transition_prob", 0.27)), 0.18)
        env_params["half_loop_vertical_retention_prob"] = 0.81
    elif not gate["target_geometry_all_improved"]:
        env_params["half_loop_transition_prob"] = min(float(env_params.get("half_loop_transition_prob", 0.27)) + 0.04, 0.35)
        env_params["half_loop_vertical_retention_prob"] = max(float(env_params.get("half_loop_vertical_retention_prob", 0.67)) - 0.04, 0.58)
        env_params["half_loop_partial_prob"] = min(float(env_params.get("half_loop_partial_prob", 0.06)) + 0.01, 0.08)
    cfg["HALF_LOOP_V2_ADJUSTMENT_FROM_ROUND"] = current_round
    return cfg


def build_round_report(path, round_idx, checkpoint, gate, horizontal_summary, loop_summary):
    lines = [
        f"# Half-Loop Inverted Transition V2 Round {round_idx:02d}",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- gate_pass: `{gate['gate_pass']}`",
        f"- horizontal_regressions: `{'; '.join(gate['horizontal_regressions']) if gate['horizontal_regressions'] else 'none'}`",
        f"- loop_retention_regressions: `{'; '.join(gate['loop_retention_regressions']) if gate['loop_retention_regressions'] else 'none'}`",
        f"- overload_increase: `{'; '.join(gate['overload_increase']) if gate['overload_increase'] else 'none'}`",
        f"- altitude_drift_regressions: `{'; '.join(gate['altitude_drift_regressions']) if gate['altitude_drift_regressions'] else 'none'}`",
        "",
        "## Target Geometry Improvement",
        "",
    ]
    for task, metrics in gate["target_loop_improvements"].items():
        lines.append(f"- {task}: `{metrics}`")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- horizontal eval: `{horizontal_summary}`",
            f"- loop-quality eval: `{loop_summary}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_report(path, selected, rounds, baseline_checkpoint):
    if selected is None:
        text = "# Half-Loop Inverted Transition V2 Final Report\n\nNo rounds were executed.\n"
        path.write_text(text, encoding="utf-8")
        return
    gate = selected["gate"]
    checkpoint = selected["checkpoint"] if gate["gate_pass"] else "none"
    lines = [
        "# Half-Loop Inverted Transition V2 Final Report",
        "",
        f"- baseline checkpoint: `{baseline_checkpoint}`",
        f"- promoted checkpoint: `{checkpoint}`",
        f"- latest diagnostic checkpoint: `{selected['checkpoint']}`",
        f"- selected round: `{selected['round']}`",
        f"- gate_pass: `{gate['gate_pass']}`",
        "",
        "## Required Answers",
        "",
        f"1. Did you find a checkpoint better than epoch619? `{gate['gate_pass']}`",
        f"2. Which checkpoint path? `{checkpoint}`",
        f"3. Did horizontal tasks stay stable? `{not gate['horizontal_regressions']}`",
        f"4. Did 60/90/150 stay stable? `{not gate['loop_retention_regressions']}`",
        f"5. Did 175/180 improve? `{gate['target_geometry_all_improved']}`",
        f"6. Which geometry metrics improved? `{gate['target_loop_improvements']}`",
        f"7. Should Claude run full ACMI regression? `{gate['gate_pass']}`",
        f"8. If no better checkpoint, what blocked progress? `{'none' if gate['gate_pass'] else '; '.join(gate['horizontal_regressions'] + gate['loop_retention_regressions']) or 'target loop geometry did not improve enough'}`",
        "",
        "## Round Summary",
        "",
    ]
    for record in rounds:
        lines.append(
            f"- round {record['round']}: gate={record['gate']['gate_pass']}, "
            f"useful={record['gate']['any_useful_improvement']}, checkpoint=`{record['checkpoint']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    max_rounds = args.rounds or int(base_cfg.get("MAX_ROUNDS", 5))
    root = PLANAX_ROOT / base_cfg.get("OUTPUT_ROOT", "results/half_loop_inverted_transition_v2")
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M")
    configs_dir = run_dir / "configs"
    reports_dir = run_dir / "round_reports"
    checkpoints_dir = run_dir / "checkpoint"
    eval_dir = run_dir / "eval"
    for d in [configs_dir, reports_dir, checkpoints_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "JAX_PLATFORMS": "cuda",
            "MPLCONFIGDIR": "/tmp",
            "WANDB_MODE": "offline",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )

    baseline_checkpoint = base_cfg["LOADDIR"]
    current_checkpoint = baseline_checkpoint

    baseline_horizontal_dir = eval_dir / "baseline_horizontal"
    run_command(
        [
            sys.executable,
            "eval_vertical_energy_checkpoints.py",
            "--baseline",
            baseline_checkpoint,
            "--new",
            baseline_checkpoint,
            "--out-dir",
            str(baseline_horizontal_dir),
            "--seeds",
            str(base_cfg.get("EVAL_SEEDS", 5)),
            "--suite",
            "horizontal_v2",
        ],
        env,
        dry_run=args.dry_run,
    )
    baseline_loop_dir = eval_dir / "baseline_loop_quality"
    run_command(
        [
            sys.executable,
            "eval_loop_quality_claude_aligned.py",
            "--checkpoint",
            baseline_checkpoint,
            "--out-dir",
            str(baseline_loop_dir),
            "--suite",
            "v2",
            "--no-compare",
        ],
        env,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"search_dir={run_dir}", flush=True)
        return

    baseline_horizontal_rows = read_csv(baseline_horizontal_dir / "eval_summary.csv")
    baseline_loop_rows = read_csv(baseline_loop_dir / "loop_quality_summary.csv")
    shutil.copyfile(baseline_horizontal_dir / "eval_summary.csv", run_dir / "baseline_eval_horizontal.csv")
    shutil.copyfile(baseline_loop_dir / "loop_quality_summary.csv", run_dir / "baseline_eval_loop_quality.csv")

    search_rows = []
    round_records = []
    best_record = None
    current_cfg = deepcopy(base_cfg)
    no_useful_rounds = 0
    stop_after = int(base_cfg.get("STOP_AFTER_NO_USEFUL_IMPROVEMENT", 2))

    for round_idx in range(1, max_rounds + 1):
        round_dir = checkpoints_dir / f"round_{round_idx:02d}"
        train_cfg = deepcopy(current_cfg)
        train_cfg["LOADDIR"] = current_checkpoint
        train_cfg["TOTAL_TIMESTEPS"] = int(base_cfg.get("TIMESTEPS_PER_ROUND", base_cfg["TOTAL_TIMESTEPS"]))
        train_cfg["FOR_LOOP_EPOCHS"] = 1
        train_cfg["OUTPUTDIR"] = str(round_dir)
        train_cfg["LOGDIR"] = str(round_dir / "logs")
        train_cfg["SAVEDIR"] = str(round_dir / "checkpoint")
        train_cfg["HALF_LOOP_V2_ROUND"] = round_idx
        cfg_path = configs_dir / f"round_{round_idx:02d}_config.json"
        write_json(cfg_path, train_cfg)

        env["CONFIG_JSON"] = str(cfg_path)
        run_command(
            [sys.executable, "train_heading_pitch_V_discrete_rnn_quaternion_vertical_energy_finetune.py"],
            env,
            dry_run=args.dry_run,
        )
        current_checkpoint = read_saved_checkpoint(round_dir / "train_log.csv")
        shutil.copyfile(round_dir / "train_log.csv", run_dir / "train_log.csv")

        horizontal_eval_dir = eval_dir / f"round_{round_idx:02d}_horizontal"
        run_command(
            [
                sys.executable,
                "eval_vertical_energy_checkpoints.py",
                "--baseline",
                baseline_checkpoint,
                "--new",
                current_checkpoint,
                "--out-dir",
                str(horizontal_eval_dir),
                "--seeds",
                str(base_cfg.get("EVAL_SEEDS", 5)),
                "--suite",
                "horizontal_v2",
            ],
            env,
            dry_run=args.dry_run,
        )

        loop_eval_dir = eval_dir / f"round_{round_idx:02d}_loop_quality"
        run_command(
            [
                sys.executable,
                "eval_loop_quality_claude_aligned.py",
                "--checkpoint",
                current_checkpoint,
                "--out-dir",
                str(loop_eval_dir),
                "--suite",
                "v2",
                "--no-compare",
            ],
            env,
            dry_run=args.dry_run,
        )

        horizontal_rows = read_csv(horizontal_eval_dir / "eval_summary.csv")
        loop_rows = read_csv(loop_eval_dir / "loop_quality_summary.csv")
        gate = evaluate_gate(horizontal_rows, baseline_horizontal_rows, loop_rows, baseline_loop_rows)
        score_report = {
            "round": round_idx,
            "checkpoint": current_checkpoint,
            "gate": gate,
            "baseline_checkpoint": baseline_checkpoint,
            "horizontal_eval": str(horizontal_eval_dir / "eval_summary.csv"),
            "loop_quality_eval": str(loop_eval_dir / "loop_quality_summary.csv"),
        }
        write_json(run_dir / "score_report.json", score_report)
        write_json(reports_dir / f"round_{round_idx:02d}_score_report.json", score_report)
        shutil.copyfile(horizontal_eval_dir / "eval_summary.csv", run_dir / "eval_horizontal.csv")
        shutil.copyfile(loop_eval_dir / "loop_quality_summary.csv", run_dir / "eval_loop_quality.csv")
        round_report = reports_dir / f"round_{round_idx:02d}_report.md"
        build_round_report(
            round_report,
            round_idx,
            current_checkpoint,
            gate,
            horizontal_eval_dir / "eval_summary.csv",
            loop_eval_dir / "loop_quality_summary.csv",
        )
        shutil.copyfile(round_report, run_dir / "report.md")

        record = {
            "round": round_idx,
            "checkpoint": current_checkpoint,
            "config": str(cfg_path),
            "train_log": str(round_dir / "train_log.csv"),
            "horizontal_eval": str(horizontal_eval_dir / "eval_summary.csv"),
            "loop_quality_eval": str(loop_eval_dir / "loop_quality_summary.csv"),
            "report": str(round_report),
            "gate": gate,
        }
        round_records.append(record)
        search_rows.append(
            {
                "round": round_idx,
                "checkpoint": current_checkpoint,
                "gate_pass": gate["gate_pass"],
                "target_geometry_all_improved": gate["target_geometry_all_improved"],
                "any_useful_improvement": gate["any_useful_improvement"],
                "horizontal_regressions": "; ".join(gate["horizontal_regressions"]),
                "loop_retention_regressions": "; ".join(gate["loop_retention_regressions"]),
                "overload_increase": "; ".join(gate["overload_increase"]),
                "altitude_drift_regressions": "; ".join(gate["altitude_drift_regressions"]),
                "horizontal_eval": str(horizontal_eval_dir / "eval_summary.csv"),
                "loop_quality_eval": str(loop_eval_dir / "loop_quality_summary.csv"),
                "report": str(round_report),
            }
        )
        write_csv(run_dir / "search_summary.csv", search_rows)

        if gate["gate_pass"]:
            best_record = record
            break
        if gate["any_useful_improvement"]:
            no_useful_rounds = 0
        else:
            no_useful_rounds += 1
        if no_useful_rounds >= stop_after:
            break
        current_cfg = adjust_config(base_cfg, gate, round_idx)
        if gate["horizontal_regressions"]:
            current_checkpoint = baseline_checkpoint

    selected = best_record or (round_records[-1] if round_records else None)
    manifest = {
        "best_checkpoint": best_record["checkpoint"] if best_record else None,
        "diagnostic_checkpoint": selected["checkpoint"] if selected else None,
        "main_baseline_if_no_gate": baseline_checkpoint,
        "gate_pass": selected["gate"]["gate_pass"] if selected else False,
        "training_config": str(args.config.resolve()),
        "rounds": round_records,
        "eval_alignment": "results/codex_eval_alignment_epoch619/20260517_172109",
        "notes": "Promotion requires Claude-aligned loop-quality geometry and horizontal_v2 stability.",
    }
    write_json(run_dir / "best_checkpoint_manifest.json", manifest)
    build_final_report(run_dir / "final_report.md", selected, round_records, baseline_checkpoint)
    print(f"search_dir={run_dir}", flush=True)
    print(f"latest_checkpoint={current_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
