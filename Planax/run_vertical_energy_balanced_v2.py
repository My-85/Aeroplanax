import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PLANAX_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PLANAX_ROOT / "paper/second_paper/vertical_energy_balanced_finetune_v2_config.json"


def read_saved_checkpoint(train_log: Path) -> str:
    with train_log.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not rows[-1].get("saved_checkpoint"):
        raise RuntimeError(f"Could not read saved_checkpoint from {train_log}")
    return rows[-1]["saved_checkpoint"]


def run_command(cmd, env, dry_run: bool):
    print(" ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PLANAX_ROOT), env=env, check=True)


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


def float_field(row, name: str, default: float = 0.0) -> float:
    try:
        return float(row.get(name, default))
    except (TypeError, ValueError):
        return default


def candidate_rows(rows):
    return [r for r in rows if r.get("policy") == "candidate"]


def is_b_or_better(row) -> bool:
    return row.get("grade") in ("A", "B")


def split_eval_outputs(cycle_dir: Path, eval_dirs, vertical_filename: str = "eval_vertical.csv"):
    rows = []
    for eval_dir in eval_dirs:
        summary = eval_dir / "eval_summary.csv"
        if summary.exists():
            rows.extend(read_csv(summary))

    original = [r for r in rows if r.get("category") == "retention"]
    planner = [r for r in rows if r.get("category") == "old_skill_proxy"]
    vertical = [r for r in rows if r.get("category") in ("vertical", "vertical_arc")]
    write_csv(cycle_dir / "eval_original.csv", original)
    write_csv(cycle_dir / "eval_planner_proxy.csv", planner)
    write_csv(cycle_dir / vertical_filename, vertical)
    return rows


def build_report(cycle_dir: Path, cycle_cfg: dict, checkpoint: str, rows, quarter_loop_attempted: bool):
    cand = candidate_rows(rows)
    base = [r for r in rows if r.get("policy") != "candidate"]
    base_by_task = {r["task"]: r for r in base}
    cand_by_task = {r["task"]: r for r in cand}

    original = [r for r in cand if r.get("category") == "retention"]
    planner = [r for r in cand if r.get("category") == "old_skill_proxy"]
    vertical = [r for r in cand if r.get("category") in ("vertical", "vertical_arc")]

    planner_crashes = [r for r in planner if float_field(r, "crash_rate") > 0.0]
    original_crashes = [r for r in original if float_field(r, "crash_rate") > 0.0]
    drift_flags = []
    for r in planner:
        task = r["task"]
        drift = abs(float_field(r, "altitude_drift_mean", float_field(r, "altitude_gain_mean")))
        if task.startswith("level_circle") and drift > 200.0:
            drift_flags.append((task, drift))
        if (task.startswith("s_curve") or task.startswith("figure_eight")) and drift > 150.0:
            drift_flags.append((task, drift))

    pullup_regressions = []
    for task, r in cand_by_task.items():
        if not (task.startswith("15_pullup") or task.startswith("30_pullup")):
            continue
        b = base_by_task.get(task)
        if not b:
            continue
        vt_ok = float_field(r, "vt_min_mean") >= float_field(b, "vt_min_mean") - 1.0
        alpha_ok = float_field(r, "alpha_max_mean") <= float_field(b, "alpha_max_mean") + 2.0
        g_ok = float_field(r, "Gmax_mean") <= float_field(b, "Gmax_mean") + 0.3
        if not (vt_ok and alpha_ok and g_ok):
            pullup_regressions.append(task)

    sixty_rows = [r for r in cand if r["task"].startswith("60_vertical_arc")]
    sixty_b = [r for r in sixty_rows if is_b_or_better(r)]
    quarter_rows = [r for r in cand if r["task"].startswith("90_quarter_loop")]

    gate_pass = (
        not planner_crashes
        and not original_crashes
        and not drift_flags
        and not pullup_regressions
        and len(sixty_b) >= 1
    )
    if drift_flags or planner_crashes:
        decision = "switch altitude_retention_repair"
    elif pullup_regressions or original_crashes:
        decision = "stop and inspect/regress"
    elif gate_pass:
        decision = "continue balanced v2"
    else:
        decision = "hold vertical progression; inspect 60deg arc"

    env_params = cycle_cfg.get("ENV_PARAMS", {})
    report = [
        "# Balanced Vertical-Energy Fine-Tune V2 Cycle Report",
        "",
        "## 1. Training",
        "",
        f"- start checkpoint: `{cycle_cfg['LOADDIR']}`",
        f"- output checkpoint: `{checkpoint}`",
        f"- timesteps: `{cycle_cfg['TOTAL_TIMESTEPS']}`",
        f"- LR: `{cycle_cfg['LR']}`",
        f"- replay/curriculum: original={env_params.get('original_task_prob')}, proxy={env_params.get('horizontal_proxy_task_prob')}, level_altitude={env_params.get('level_altitude_task_prob')}, vertical_remaining≈{1.0 - env_params.get('original_task_prob', 0.0) - env_params.get('horizontal_proxy_task_prob', 0.0) - env_params.get('level_altitude_task_prob', 0.0):.2f}",
        "",
        "## 2. Original Skill Retention",
        "",
        f"- crash rows: `{', '.join(r['task'] for r in original_crashes) if original_crashes else 'none'}`",
        "- roll small target: unavailable in this target-level env",
        "",
        "## 3. Planner Proxy",
        "",
        f"- crash rows: `{', '.join(r['task'] for r in planner_crashes) if planner_crashes else 'none'}`",
        f"- altitude drift flags: `{', '.join(f'{t}:{d:.1f}m' for t, d in drift_flags) if drift_flags else 'none'}`",
        "",
        "## 4. Vertical Energy",
        "",
        f"- pull-up regressions vs start checkpoint: `{', '.join(pullup_regressions) if pullup_regressions else 'none'}`",
        f"- 60deg B/A rows: `{', '.join(r['task'] + ':' + r.get('grade', '') for r in sixty_b) if sixty_b else 'none'}`",
        f"- 90deg attempted: `{quarter_loop_attempted}`",
        f"- 90deg rows: `{', '.join(r['task'] + ':' + r.get('grade', '') for r in quarter_rows) if quarter_rows else 'none'}`",
        "",
        "## 5. Gate",
        "",
        f"- gate_pass: `{gate_pass}`",
        f"- decision: `{decision}`",
        f"- recommend Claude ACMI/demo regression: `{gate_pass}`",
    ]
    (cycle_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return gate_pass, decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    cycles = args.cycles or int(base_cfg.get("TRAIN_EVAL_CYCLES", 3))
    branch_root = PLANAX_ROOT / base_cfg.get("OUTPUT_ROOT", "results/vertical_energy_balanced_finetune_v2")
    branch_root.mkdir(parents=True, exist_ok=True)

    current_checkpoint = base_cfg["LOADDIR"]
    baseline_checkpoint = base_cfg.get("PLANNER_PROXY_EVAL_BASELINE", current_checkpoint)
    seeds = int(base_cfg.get("PLANNER_PROXY_EVAL_SEEDS", 10))
    run_proxy_eval = bool(base_cfg.get("RUN_PLANNER_PROXY_EVAL", True))
    eval_interval = max(1, int(base_cfg.get("PLANNER_PROXY_EVAL_INTERVAL_CYCLES", 1)))
    timesteps_per_cycle = int(base_cfg.get("TIMESTEPS_PER_CYCLE", base_cfg["TOTAL_TIMESTEPS"]))

    for cycle in range(1, cycles + 1):
        stamp_format = "%Y%m%d_%H%M" if base_cfg.get("FLAT_OUTPUT_FOR_SINGLE_CYCLE") and cycles == 1 else "%Y%m%d_%H%M%S"
        stamp = datetime.now().strftime(stamp_format)
        cycle_dir = branch_root / stamp if base_cfg.get("FLAT_OUTPUT_FOR_SINGLE_CYCLE") and cycles == 1 else branch_root / f"{stamp}_cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        cycle_cfg = dict(base_cfg)
        cycle_cfg["LOADDIR"] = current_checkpoint
        cycle_cfg["TOTAL_TIMESTEPS"] = timesteps_per_cycle
        cycle_cfg["FOR_LOOP_EPOCHS"] = 1
        cycle_cfg["OUTPUTDIR"] = str(cycle_dir)
        cycle_cfg["LOGDIR"] = str(cycle_dir / "logs")
        cycle_cfg["SAVEDIR"] = str(cycle_dir / "checkpoint")
        cycle_cfg["BALANCED_V2_CYCLE"] = cycle
        cycle_cfg_path = cycle_dir / "cycle_config.json"
        with cycle_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cycle_cfg, f, indent=2, ensure_ascii=False)

        env = os.environ.copy()
        env.update({
            "JAX_PLATFORMS": "cuda",
            "MPLCONFIGDIR": "/tmp",
            "WANDB_MODE": "offline",
            "CONFIG_JSON": str(cycle_cfg_path),
        })

        run_command(
            [sys.executable, "train_heading_pitch_V_discrete_rnn_quaternion_vertical_energy_finetune.py"],
            env,
            args.dry_run,
        )
        if args.dry_run:
            current_checkpoint = str(cycle_dir / "checkpoint/checkpoint_epoch_DRY_RUN")
        else:
            current_checkpoint = read_saved_checkpoint(cycle_dir / "train_log.csv")

        if run_proxy_eval and cycle % eval_interval == 0:
            eval_dir = cycle_dir / "planner_proxy_eval"
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
                    str(seeds),
                    "--suite",
                    "planner_proxy",
                ],
                env,
                args.dry_run,
            )
            eval_dirs = [eval_dir]
            quarter_loop_attempted = False
            vertical_filename = base_cfg.get("VERTICAL_EVAL_FILENAME", "eval_vertical.csv")
            rows = split_eval_outputs(cycle_dir, eval_dirs, vertical_filename=vertical_filename) if not args.dry_run else []
            sixty_candidate = [
                r for r in candidate_rows(rows)
                if r.get("task", "").startswith("60_vertical_arc")
            ]
            sixty_stable = len(sixty_candidate) >= 2 and all(is_b_or_better(r) for r in sixty_candidate)
            if sixty_stable and base_cfg.get("ALLOW_QUARTER_LOOP_EVAL", True):
                quarter_loop_attempted = True
                quarter_dir = cycle_dir / "quarter_loop_eval"
                run_command(
                    [
                        sys.executable,
                        "eval_vertical_energy_checkpoints.py",
                        "--baseline",
                        baseline_checkpoint,
                        "--new",
                        current_checkpoint,
                        "--out-dir",
                        str(quarter_dir),
                        "--seeds",
                        str(seeds),
                        "--suite",
                        "quarter_loop",
                    ],
                    env,
                    args.dry_run,
                )
                eval_dirs.append(quarter_dir)
                rows = split_eval_outputs(cycle_dir, eval_dirs, vertical_filename=vertical_filename) if not args.dry_run else rows
            if not args.dry_run:
                build_report(cycle_dir, cycle_cfg, current_checkpoint, rows, quarter_loop_attempted)

    print(f"latest_checkpoint={current_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
