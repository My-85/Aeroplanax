import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PLANAX_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PLANAX_ROOT / "paper/second_paper/half_loop_inverted_transition_finetune_config.json"


GRADE_VALUE = {"A": 4, "B": 3, "C": 2, "F": 1, "": 0}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def row_by_task(rows, policy):
    return {r["task"]: r for r in rows if r.get("policy") == policy}


def grade_not_worse(candidate, baseline):
    return GRADE_VALUE.get(candidate.get("grade", ""), 0) >= GRADE_VALUE.get(baseline.get("grade", ""), 0)


def metric_improved(candidate, baseline, key, min_delta=2.0):
    return f(candidate, key, 1e9) <= f(baseline, key, 1e9) - min_delta


def horizontal_regressions(cand, base):
    regressions = []
    for task in [
        "level_circle_R3000_left",
        "level_circle_R3000_right",
        "level_circle_R5000_left",
        "level_circle_R5000_right",
        "s_curve_A3000",
        "figure_eight_R5000",
    ]:
        c = cand.get(task)
        b = base.get(task)
        if not c or not b:
            regressions.append(f"{task}:missing")
            continue
        if f(c, "success_rate") < f(b, "success_rate") - 0.10:
            regressions.append(f"{task}:success {f(b,'success_rate'):.2f}->{f(c,'success_rate'):.2f}")
        if f(c, "crash_rate") > f(b, "crash_rate") + 0.001:
            regressions.append(f"{task}:crash {f(b,'crash_rate'):.2f}->{f(c,'crash_rate'):.2f}")
        if abs(f(c, "altitude_drift_mean")) > abs(f(b, "altitude_drift_mean")) + 100.0:
            regressions.append(
                f"{task}:alt_drift {f(b,'altitude_drift_mean'):.1f}->{f(c,'altitude_drift_mean'):.1f}"
            )
        if f(c, "Gmax_mean") > f(b, "Gmax_mean") + 0.35:
            regressions.append(f"{task}:Gmax {f(b,'Gmax_mean'):.2f}->{f(c,'Gmax_mean'):.2f}")
    return regressions


def evaluate_gate(summary_rows):
    base = row_by_task(summary_rows, "baseline_epoch600")
    cand = row_by_task(summary_rows, "candidate")

    h_reg = horizontal_regressions(cand, base)
    vertical_reg = []
    for task in ["60_loop_plane_arc_R10000", "90_loop_plane_arc_R10000", "150_loop_plane_arc_R15000"]:
        c = cand.get(task)
        b = base.get(task)
        if not c or not b:
            vertical_reg.append(f"{task}:missing")
            continue
        if not grade_not_worse(c, b):
            vertical_reg.append(f"{task}:grade {b.get('grade')}->{c.get('grade')}")
        if f(c, "success_rate") < f(b, "success_rate") - 0.10:
            vertical_reg.append(f"{task}:success {f(b,'success_rate'):.2f}->{f(c,'success_rate'):.2f}")

    task180 = cand.get("180_half_loop_R15000", {})
    base180 = base.get("180_half_loop_R15000", {})
    task175 = cand.get("175_half_loop_arc_R15000", {})
    base175 = base.get("175_half_loop_arc_R15000", {})
    half_loop_180_b = task180.get("grade") in ("A", "B")
    half_loop_175_better = (
        metric_improved(task175, base175, "wing_plane_error_mean")
        and metric_improved(task175, base175, "nose_tangent_error_mean")
        and metric_improved(task175, base175, "velocity_tangent_error_mean")
    )
    geom_improved = {
        "wing_plane_error": metric_improved(task180, base180, "wing_plane_error_mean")
        or metric_improved(task175, base175, "wing_plane_error_mean"),
        "nose_tangent_error": metric_improved(task180, base180, "nose_tangent_error_mean")
        or metric_improved(task175, base175, "nose_tangent_error_mean"),
        "velocity_tangent_error": metric_improved(task180, base180, "velocity_tangent_error_mean")
        or metric_improved(task175, base175, "velocity_tangent_error_mean"),
        "alpha": f(task180, "alpha_max_mean", 1e9) <= f(base180, "alpha_max_mean", 1e9) - 1.0
        or f(task175, "alpha_max_mean", 1e9) <= f(base175, "alpha_max_mean", 1e9) - 1.0,
    }
    overload_increase = []
    for task, c in cand.items():
        b = base.get(task)
        if b and f(c, "Gmax_mean") > f(b, "Gmax_mean") + 0.35:
            overload_increase.append(task)

    gate_pass = (
        (half_loop_180_b or half_loop_175_better)
        and not h_reg
        and not vertical_reg
        and all(geom_improved.values())
        and not overload_increase
    )
    return {
        "gate_pass": gate_pass,
        "horizontal_regressions": h_reg,
        "vertical_regressions": vertical_reg,
        "half_loop_180_b": half_loop_180_b,
        "half_loop_175_better": half_loop_175_better,
        "geom_improved": geom_improved,
        "overload_increase": overload_increase,
    }


def build_round_report(path: Path, round_idx: int, checkpoint: str, gate: dict, summary_rows):
    cand = row_by_task(summary_rows, "candidate")
    base = row_by_task(summary_rows, "baseline_epoch600")

    def line(task, metric):
        c = cand.get(task, {})
        b = base.get(task, {})
        return f"- {task}: {metric} {f(b, metric):.2f} -> {f(c, metric):.2f}, grade {b.get('grade','')} -> {c.get('grade','')}"

    report = [
        f"# Half-Loop Inverted Transition Round {round_idx:02d}",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- gate_pass: `{gate['gate_pass']}`",
        f"- horizontal_regressions: `{'; '.join(gate['horizontal_regressions']) if gate['horizontal_regressions'] else 'none'}`",
        f"- vertical_regressions: `{'; '.join(gate['vertical_regressions']) if gate['vertical_regressions'] else 'none'}`",
        f"- overload_increase: `{'; '.join(gate['overload_increase']) if gate['overload_increase'] else 'none'}`",
        "",
        "## Key Geometry",
        "",
        line("150_loop_plane_arc_R15000", "wing_plane_error_mean"),
        line("175_half_loop_arc_R15000", "wing_plane_error_mean"),
        line("180_half_loop_R15000", "wing_plane_error_mean"),
        line("180_half_loop_R15000", "nose_tangent_error_mean"),
        line("180_half_loop_R15000", "velocity_tangent_error_mean"),
        "",
        "## Required Decision",
        "",
        f"- 180 B/A: `{gate['half_loop_180_b']}`",
        f"- 175 materially better than ep619: `{gate['half_loop_175_better']}`",
        f"- geometry improved: `{gate['geom_improved']}`",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def build_final_report(path: Path, best: dict, rounds):
    if not rounds:
        text = "# Half-Loop Inverted Transition Search\n\nNo rounds were executed.\n"
        path.write_text(text, encoding="utf-8")
        return
    last = rounds[-1]
    selected = best or last
    gate = selected["gate"]
    promoted_checkpoint = selected["checkpoint"] if gate["gate_pass"] else "none"
    diagnostic_checkpoint = selected["checkpoint"]
    report = [
        "# Half-Loop Inverted Transition Search Final Report",
        "",
        f"- promoted checkpoint: `{promoted_checkpoint}`",
        f"- latest diagnostic checkpoint: `{diagnostic_checkpoint}`",
        f"- selected round: `{selected['round']}`",
        f"- gate_pass: `{gate['gate_pass']}`",
        "",
        "## Required Answers",
        "",
        f"1. 是否找到 half-loop-capable checkpoint？`{gate['gate_pass']}`",
        f"2. 180° 是否达到 B/A？`{gate['half_loop_180_b']}`",
        f"3. 175° 是否明显优于 ep619？`{gate['half_loop_175_better']}`",
        f"4. wing_plane_error 是否下降？`{gate['geom_improved']['wing_plane_error']}`",
        f"5. nose_tangent_error 是否下降？`{gate['geom_improved']['nose_tangent_error']}`",
        f"6. velocity_tangent_error 是否下降？`{gate['geom_improved']['velocity_tangent_error']}`",
        f"7. alpha 是否下降？`{gate['geom_improved']['alpha']}`",
        f"8. 60°/90°/150° 是否保持？`{not gate['vertical_regressions']}`",
        f"9. 水平轨迹是否保持？`{not gate['horizontal_regressions']}`",
        f"10. 是否建议交给 Claude 做完整 ACMI 回归？`{gate['gate_pass']}`",
        "",
        "## Notes",
        "",
        "- Promotion requires geometry improvement and horizontal retention together.",
        "- If gate_pass is false, keep checkpoint_epoch_619 as the main baseline and treat the latest checkpoint as diagnostic only.",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    max_rounds = args.rounds or int(base_cfg.get("MAX_ROUNDS", 5))
    root = PLANAX_ROOT / base_cfg.get("OUTPUT_ROOT", "results/half_loop_inverted_transition_search")
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M")
    configs_dir = run_dir / "configs"
    reports_dir = run_dir / "round_reports"
    checkpoints_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    for d in [configs_dir, reports_dir, checkpoints_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    current_checkpoint = base_cfg["LOADDIR"]
    baseline_checkpoint = base_cfg.get("PLANNER_PROXY_EVAL_BASELINE", current_checkpoint)
    env = os.environ.copy()
    env.update(
        {
            "JAX_PLATFORMS": "cuda",
            "MPLCONFIGDIR": "/tmp",
            "WANDB_MODE": "offline",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )

    search_rows = []
    round_records = []
    best_record = None
    for round_idx in range(1, max_rounds + 1):
        round_train_dir = checkpoints_dir / f"round_{round_idx:02d}"
        cycle_cfg = dict(base_cfg)
        cycle_cfg["LOADDIR"] = current_checkpoint
        cycle_cfg["TOTAL_TIMESTEPS"] = int(base_cfg.get("TIMESTEPS_PER_ROUND", base_cfg["TOTAL_TIMESTEPS"]))
        cycle_cfg["FOR_LOOP_EPOCHS"] = 1
        cycle_cfg["OUTPUTDIR"] = str(round_train_dir)
        cycle_cfg["LOGDIR"] = str(round_train_dir / "logs")
        cycle_cfg["SAVEDIR"] = str(round_train_dir / "checkpoint")
        cycle_cfg["HALF_LOOP_SEARCH_ROUND"] = round_idx
        cfg_path = configs_dir / f"round_{round_idx:02d}_config.json"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cycle_cfg, f, indent=2, ensure_ascii=False)

        env["CONFIG_JSON"] = str(cfg_path)
        run_command(
            [sys.executable, "train_heading_pitch_V_discrete_rnn_quaternion_vertical_energy_finetune.py"],
            env,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            current_checkpoint = str(round_train_dir / "checkpoint/checkpoint_epoch_DRY_RUN")
        else:
            current_checkpoint = read_saved_checkpoint(round_train_dir / "train_log.csv")

        eval_dir = reports_dir / f"round_{round_idx:02d}_eval"
        run_command(
            [
                sys.executable,
                "eval_vertical_energy_checkpoints.py",
                "--baseline",
                baseline_checkpoint,
                "--new",
                current_checkpoint,
                "--out-dir",
                str(eval_dir),
                "--seeds",
                str(base_cfg.get("EVAL_SEEDS", 5)),
                "--suite",
                "half_loop_search",
            ],
            env,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            break

        summary_rows = read_csv(eval_dir / "eval_summary.csv")
        gate = evaluate_gate(summary_rows)
        report_path = reports_dir / f"round_{round_idx:02d}_report.md"
        build_round_report(report_path, round_idx, current_checkpoint, gate, summary_rows)

        round_record = {
            "round": round_idx,
            "checkpoint": current_checkpoint,
            "config": str(cfg_path),
            "train_dir": str(round_train_dir),
            "eval_summary": str(eval_dir / "eval_summary.csv"),
            "eval_rollouts": str(eval_dir / "eval_rollouts.csv"),
            "report": str(report_path),
            "gate": gate,
        }
        round_records.append(round_record)
        search_rows.append(
            {
                "round": round_idx,
                "checkpoint": current_checkpoint,
                "gate_pass": gate["gate_pass"],
                "half_loop_180_b": gate["half_loop_180_b"],
                "half_loop_175_better": gate["half_loop_175_better"],
                "wing_plane_improved": gate["geom_improved"]["wing_plane_error"],
                "nose_tangent_improved": gate["geom_improved"]["nose_tangent_error"],
                "velocity_tangent_improved": gate["geom_improved"]["velocity_tangent_error"],
                "alpha_improved": gate["geom_improved"]["alpha"],
                "horizontal_regressions": "; ".join(gate["horizontal_regressions"]),
                "vertical_regressions": "; ".join(gate["vertical_regressions"]),
                "overload_increase": "; ".join(gate["overload_increase"]),
                "eval_summary": str(eval_dir / "eval_summary.csv"),
                "report": str(report_path),
            }
        )
        write_csv(run_dir / "search_summary.csv", search_rows)

        if gate["gate_pass"]:
            best_record = round_record
            break
        if gate["horizontal_regressions"]:
            break

    selected = best_record or (round_records[-1] if round_records else None)
    manifest = {
        "best_checkpoint": best_record["checkpoint"] if best_record else None,
        "diagnostic_checkpoint": selected["checkpoint"] if selected else None,
        "main_baseline_if_no_gate": base_cfg["LOADDIR"],
        "gate_pass": selected["gate"]["gate_pass"] if selected else False,
        "training_config": str(args.config.resolve()),
        "rounds": round_records,
        "eval_suite": "half_loop_search",
        "loop_plane_target": "experiments/hierarchical_trajectory_tracking/loop_attitude_target.py",
        "notes": "Do not promote a checkpoint if half-loop improves while horizontal proxy regresses.",
    }
    with (run_dir / "best_checkpoint_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    build_final_report(run_dir / "final_report.md", best_record, round_records)
    print(f"search_dir={run_dir}", flush=True)
    print(f"latest_checkpoint={current_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
