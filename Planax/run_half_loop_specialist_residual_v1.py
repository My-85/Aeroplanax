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
DEFAULT_CONFIG = PLANAX_ROOT / "paper/second_paper/half_loop_specialist_residual_v1_config.json"
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
STABLE_LOOP_TASKS = ["pu060_R12000", "pu090_R12000", "pu120_R12000", "pu150_R12000"]
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
        metrics = {key: metric_improved(c, b, key, 2.0) for key in GEOMETRY_KEYS}
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


def adjust_config(config, gate):
    cfg = deepcopy(config)
    env_params = cfg.setdefault("ENV_PARAMS", {})
    if gate["horizontal_regressions"]:
        cfg["LR"] = max(float(cfg.get("LR", 1e-5)) * 0.5, 2.5e-6)
        cfg["RESIDUAL_LOGIT_CLIP"] = max(float(cfg.get("RESIDUAL_LOGIT_CLIP", 1.25)) * 0.75, 0.5)
        cfg["NON_LOOP_RESIDUAL_L2_COEF"] = max(float(cfg.get("NON_LOOP_RESIDUAL_L2_COEF", 0.20)), 0.50)
    elif not gate["target_geometry_all_improved"]:
        env_params["half_loop_transition_prob"] = min(
            float(env_params.get("half_loop_transition_prob", 0.48)) + 0.05, 0.58
        )
        env_params["half_loop_partial_prob"] = min(
            float(env_params.get("half_loop_partial_prob", 0.22)) + 0.03, 0.28
        )
        env_params["half_loop_vertical_retention_prob"] = max(
            float(env_params.get("half_loop_vertical_retention_prob", 0.30)) - 0.08, 0.18
        )
    return cfg


def build_round_report(path, round_idx, checkpoint, gate, horizontal_summary, loop_summary):
    lines = [
        f"# Half-Loop Specialist Residual V1 Round {round_idx:02d}",
        "",
        f"- residual checkpoint: `{checkpoint}`",
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
        path.write_text("# Half-Loop Specialist Residual V1 Final Report\n\nNo rounds were executed.\n", encoding="utf-8")
        return
    gate = selected["gate"]
    promoted = selected["checkpoint"] if gate["gate_pass"] else "none"
    lines = [
        "# Half-Loop Specialist Residual V1 Final Report",
        "",
        f"- architecture: `frozen epoch619 + phase-gated residual logits policy`",
        f"- base checkpoint: `{baseline_checkpoint}`",
        f"- promoted residual checkpoint: `{promoted}`",
        f"- latest diagnostic residual checkpoint: `{selected['checkpoint']}`",
        f"- selected round: `{selected['round']}`",
        f"- gate_pass: `{gate['gate_pass']}`",
        "",
        "## Required Answers",
        "",
        "1. Which architecture was chosen and why? `Option A, implemented as residual logits because the action space is discrete multi-head categorical.`",
        f"2. What files were created? `run_half_loop_specialist_residual_v1.py; train_half_loop_specialist_residual_v1.py; half_loop_residual_policy.py; paper/second_paper/half_loop_specialist_residual_v1_config.json`",
        f"3. Did training run? `True`",
        f"4. Whether horizontal behavior is preserved? `{not gate['horizontal_regressions']}`",
        f"5. Whether 175/180 improved? `{gate['target_geometry_all_improved']}`",
        f"6. Should this checkpoint go to Claude for full ACMI regression? `{gate['gate_pass']}`",
        f"7. If not successful, what blocked progress? `{'none' if gate['gate_pass'] else '; '.join(gate['horizontal_regressions'] + gate['loop_retention_regressions']) or 'target loop geometry did not improve enough'}`",
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


def write_paper_story(path):
    text = """# Paper 2 Story Plan

Working title: Phase-Conditioned Residual Control for Preserving Flight Skills During Aerobatic Specialization

Core story:

A monolithic PPO policy suffers from multi-task interference when learning full-loop aerobatics. Earlier mixed fine-tuning improved selected vertical/inverted metrics but repeatedly damaged horizontal behaviors. We therefore freeze a balanced base flight skill and learn a phase-conditioned residual/specialist skill for the inverted top-transition region. The method aims to improve half-loop geometry while preserving horizontal behaviors.

Contributions:

1. Demonstrate multi-task interference in monolithic flight-skill fine-tuning.
2. Introduce phase-conditioned residual/specialist control for aerobatic skill composition.
3. Preserve existing flight skills while improving 80-180 degree inverted transition.
4. Evaluate with geometry-aware loop-quality metrics.

Current claim boundary:

Do not claim success until the residual checkpoint passes horizontal retention and Claude-style loop-quality promotion gates.
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    max_rounds = args.rounds or int(base_cfg.get("MAX_ROUNDS", 5))
    root = PLANAX_ROOT / base_cfg.get("OUTPUT_ROOT", "results/half_loop_specialist_residual_v1")
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M")
    configs_dir = run_dir / "configs"
    reports_dir = run_dir / "round_reports"
    checkpoints_dir = run_dir / "checkpoint"
    eval_dir = run_dir / "eval"
    plots_dir = run_dir / "plots"
    for d in [configs_dir, reports_dir, checkpoints_dir, eval_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", base_cfg)
    write_paper_story(run_dir / "paper2_story_plan.md")

    env = os.environ.copy()
    env.update(
        {
            "JAX_PLATFORMS": "cuda",
            "MPLCONFIGDIR": "/tmp",
            "WANDB_MODE": "offline",
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE", "true"
            ),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90"
            ),
        }
    )

    baseline_checkpoint = base_cfg["BASE_CHECKPOINT"]
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
    baseline_exit_loop_dir = eval_dir / "baseline_loop_quality_exit_v2"
    run_command(
        [
            sys.executable,
            "eval_loop_quality_claude_aligned.py",
            "--checkpoint",
            baseline_checkpoint,
            "--out-dir",
            str(baseline_exit_loop_dir),
            "--suite",
            "exit_v2",
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
    shutil.copyfile(
        baseline_exit_loop_dir / "loop_quality_summary.csv",
        run_dir / "baseline_eval_loop_quality_exit_v2.csv",
    )

    current_cfg = deepcopy(base_cfg)
    current_residual = str(base_cfg.get("RESIDUAL_LOADDIR", ""))
    round_records = []
    search_rows = []
    best_record = None
    no_useful_rounds = 0
    stop_after = int(base_cfg.get("STOP_AFTER_NO_USEFUL_IMPROVEMENT", 2))

    for round_idx in range(1, max_rounds + 1):
        round_dir = checkpoints_dir / f"round_{round_idx:02d}"
        train_cfg = deepcopy(current_cfg)
        train_cfg["BASE_CHECKPOINT"] = baseline_checkpoint
        train_cfg["RESIDUAL_LOADDIR"] = current_residual
        train_cfg["TOTAL_TIMESTEPS"] = int(base_cfg.get("TIMESTEPS_PER_ROUND", base_cfg["TOTAL_TIMESTEPS"]))
        train_cfg["OUTPUTDIR"] = str(round_dir)
        train_cfg["LOGDIR"] = str(round_dir / "logs")
        train_cfg["SAVEDIR"] = str(round_dir / "checkpoint")
        train_cfg["ROUND"] = round_idx
        cfg_path = configs_dir / f"round_{round_idx:02d}_config.json"
        write_json(cfg_path, train_cfg)

        env["CONFIG_JSON"] = str(cfg_path)
        run_command(
            [sys.executable, "train_half_loop_specialist_residual_v1.py"],
            env,
            dry_run=args.dry_run,
        )
        current_residual = read_saved_checkpoint(round_dir / "train_log.csv")
        shutil.copyfile(round_dir / "train_log.csv", run_dir / "train_log.csv")

        horizontal_eval_dir = eval_dir / f"round_{round_idx:02d}_horizontal"
        run_command(
            [
                sys.executable,
                "eval_vertical_energy_checkpoints.py",
                "--baseline",
                baseline_checkpoint,
                "--new",
                baseline_checkpoint,
                "--residual-checkpoint",
                current_residual,
                "--residual-config",
                str(cfg_path),
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
                baseline_checkpoint,
                "--residual-checkpoint",
                current_residual,
                "--residual-config",
                str(cfg_path),
                "--out-dir",
                str(loop_eval_dir),
                "--suite",
                "v2",
                "--no-compare",
            ],
            env,
            dry_run=args.dry_run,
        )

        exit_loop_eval_dir = eval_dir / f"round_{round_idx:02d}_loop_quality_exit_v2"
        run_command(
            [
                sys.executable,
                "eval_loop_quality_claude_aligned.py",
                "--checkpoint",
                baseline_checkpoint,
                "--residual-checkpoint",
                current_residual,
                "--residual-config",
                str(cfg_path),
                "--out-dir",
                str(exit_loop_eval_dir),
                "--suite",
                "exit_v2",
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
            "architecture": "frozen_epoch619_plus_phase_gated_residual_logits",
            "base_checkpoint": baseline_checkpoint,
            "residual_checkpoint": current_residual,
            "gate": gate,
            "horizontal_eval": str(horizontal_eval_dir / "eval_summary.csv"),
            "loop_quality_eval": str(loop_eval_dir / "loop_quality_summary.csv"),
            "loop_quality_exit_v2_eval": str(exit_loop_eval_dir / "loop_quality_summary.csv"),
        }
        write_json(run_dir / "score_report.json", score_report)
        write_json(reports_dir / f"round_{round_idx:02d}_score_report.json", score_report)
        shutil.copyfile(horizontal_eval_dir / "eval_summary.csv", run_dir / "eval_horizontal.csv")
        shutil.copyfile(loop_eval_dir / "loop_quality_summary.csv", run_dir / "eval_loop_quality.csv")
        shutil.copyfile(exit_loop_eval_dir / "loop_quality_summary.csv", run_dir / "eval_loop_quality_exit_v2.csv")
        round_report = reports_dir / f"round_{round_idx:02d}_report.md"
        build_round_report(
            round_report,
            round_idx,
            current_residual,
            gate,
            horizontal_eval_dir / "eval_summary.csv",
            loop_eval_dir / "loop_quality_summary.csv",
        )

        record = {
            "round": round_idx,
            "checkpoint": current_residual,
            "gate": gate,
            "horizontal_eval": str(horizontal_eval_dir / "eval_summary.csv"),
            "loop_quality_eval": str(loop_eval_dir / "loop_quality_summary.csv"),
            "loop_quality_exit_v2_eval": str(exit_loop_eval_dir / "loop_quality_summary.csv"),
            "report": str(round_report),
        }
        round_records.append(record)
        search_rows.append(
            {
                "round": round_idx,
                "residual_checkpoint": current_residual,
                "gate_pass": gate["gate_pass"],
                "target_geometry_all_improved": gate["target_geometry_all_improved"],
                "any_useful_improvement": gate["any_useful_improvement"],
                "horizontal_regressions": "; ".join(gate["horizontal_regressions"]),
                "loop_retention_regressions": "; ".join(gate["loop_retention_regressions"]),
                "overload_increase": "; ".join(gate["overload_increase"]),
                "altitude_drift_regressions": "; ".join(gate["altitude_drift_regressions"]),
                "horizontal_eval": str(horizontal_eval_dir / "eval_summary.csv"),
                "loop_quality_eval": str(loop_eval_dir / "loop_quality_summary.csv"),
                "loop_quality_exit_v2_eval": str(exit_loop_eval_dir / "loop_quality_summary.csv"),
                "report": str(round_report),
            }
        )
        write_csv(run_dir / "search_summary.csv", search_rows)

        if gate["gate_pass"]:
            best_record = record
            break
        if gate["horizontal_regressions"]:
            best_record = record
            break
        if gate["any_useful_improvement"]:
            no_useful_rounds = 0
        else:
            no_useful_rounds += 1
        if no_useful_rounds >= stop_after:
            best_record = record
            break
        current_cfg = adjust_config(current_cfg, gate)

    selected = best_record or (round_records[-1] if round_records else None)
    build_final_report(run_dir / "final_report.md", selected, round_records, baseline_checkpoint)
    manifest = {
        "architecture": "frozen_epoch619_plus_phase_gated_residual_logits",
        "base_checkpoint": baseline_checkpoint,
        "best_residual_checkpoint": None
        if selected is None or not selected["gate"]["gate_pass"]
        else selected["checkpoint"],
        "latest_diagnostic_residual_checkpoint": None if selected is None else selected["checkpoint"],
        "gate_pass": False if selected is None else selected["gate"]["gate_pass"],
        "rounds": round_records,
    }
    write_json(run_dir / "best_checkpoint_manifest.json", manifest)
    print(f"search_dir={run_dir}", flush=True)
    print(f"gate_pass={manifest['gate_pass']}", flush=True)
    print(f"best_residual_checkpoint={manifest['best_residual_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
