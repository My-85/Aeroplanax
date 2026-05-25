import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PLANAX_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = (
    PLANAX_ROOT
    / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
)
DEFAULT_RESIDUAL = (
    PLANAX_ROOT
    / "results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2"
)
DEFAULT_RESIDUAL_CONFIG = (
    PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/configs/round_01_config.json"
)
WINDOWS = [
    (80, 180),
    (80, 190),
    (80, 200),
    (90, 200),
    (100, 200),
    (150, 210),
]
TASKS_OF_INTEREST = [
    "pu150_R12000",
    "pu175_R15000",
    "pu180_R15000",
    "pu185_R15000",
    "pu190_R15000",
    "pu200_R15000",
    "pu210_R15000",
]
SUMMARY_FIELDS = [
    "variant",
    "gate_start",
    "gate_end",
    "task",
    "termination",
    "completed",
    "steps",
    "CTE_mean",
    "velocity_tangent_error_mean",
    "nose_tangent_error_mean",
    "wing_plane_error_mean",
    "q_error_mean_rad",
    "env_alpha_max",
    "phase150_180_velocity_tangent_error_mean",
    "phase150_180_nose_tangent_error_mean",
    "phase150_180_wing_plane_error_mean",
    "phase170_200_velocity_tangent_error_mean",
    "phase170_200_nose_tangent_error_mean",
    "phase170_200_wing_plane_error_mean",
    "phase180_200_velocity_tangent_error_mean",
    "phase180_200_nose_tangent_error_mean",
    "phase180_200_wing_plane_error_mean",
    "eval_dir",
]


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_eval(cmd, env, done_file: Path, dry_run=False):
    if done_file.exists():
        print(f"skip existing {done_file.parent}", flush=True)
        return
    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(" ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PLANAX_ROOT), env=env, check=True)


def append_rows(summary_rows, variant, gate_start, gate_end, eval_dir):
    rows = read_csv(eval_dir / "loop_quality_summary.csv")
    by_name = {r["name"]: r for r in rows}
    for task in TASKS_OF_INTEREST:
        r = by_name.get(task)
        if not r:
            continue
        summary_rows.append(
            {
                "variant": variant,
                "gate_start": gate_start,
                "gate_end": gate_end,
                "task": task,
                "termination": r.get("termination", ""),
                "completed": r.get("completed", ""),
                "steps": r.get("steps", ""),
                "CTE_mean": r.get("CTE_mean", ""),
                "velocity_tangent_error_mean": r.get("velocity_tangent_error_mean", ""),
                "nose_tangent_error_mean": r.get("nose_tangent_error_mean", ""),
                "wing_plane_error_mean": r.get("wing_plane_error_mean", ""),
                "q_error_mean_rad": r.get("q_error_mean_rad", ""),
                "env_alpha_max": r.get("env_alpha_max", ""),
                "phase150_180_velocity_tangent_error_mean": r.get(
                    "phase150_180_velocity_tangent_error_mean", ""
                ),
                "phase150_180_nose_tangent_error_mean": r.get(
                    "phase150_180_nose_tangent_error_mean", ""
                ),
                "phase150_180_wing_plane_error_mean": r.get(
                    "phase150_180_wing_plane_error_mean", ""
                ),
                "phase170_200_velocity_tangent_error_mean": r.get(
                    "phase170_200_velocity_tangent_error_mean", ""
                ),
                "phase170_200_nose_tangent_error_mean": r.get(
                    "phase170_200_nose_tangent_error_mean", ""
                ),
                "phase170_200_wing_plane_error_mean": r.get(
                    "phase170_200_wing_plane_error_mean", ""
                ),
                "phase180_200_velocity_tangent_error_mean": r.get(
                    "phase180_200_velocity_tangent_error_mean", ""
                ),
                "phase180_200_nose_tangent_error_mean": r.get(
                    "phase180_200_nose_tangent_error_mean", ""
                ),
                "phase180_200_wing_plane_error_mean": r.get(
                    "phase180_200_wing_plane_error_mean", ""
                ),
                "eval_dir": str(eval_dir.resolve()),
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--residual-config", type=Path, default=DEFAULT_RESIDUAL_CONFIG)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PLANAX_ROOT
        / "results/half_loop_specialist_residual_v1_gate_window_ablations"
        / datetime.now().strftime("%Y%m%d_%H%M"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base": str(args.base.resolve()),
                "residual": str(args.residual.resolve()),
                "residual_config": str(args.residual_config.resolve()),
                "suite": "exit_v2",
                "windows": WINDOWS,
                "cuda_visible_devices": args.cuda_visible_devices,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

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
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    baseline_dir = out_dir / "baseline_ep619_exit_v2"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    run_eval(
        [
            sys.executable,
            "eval_loop_quality_claude_aligned.py",
            "--checkpoint",
            str(args.base),
            "--out-dir",
            str(baseline_dir),
            "--suite",
            "exit_v2",
            "--no-compare",
        ],
        env,
        baseline_dir / "loop_quality_summary.csv",
        dry_run=args.dry_run,
    )

    summary_rows = []
    if not args.dry_run:
        append_rows(summary_rows, "baseline_ep619", "", "", baseline_dir)

    for gate_start, gate_end in WINDOWS:
        variant = f"gate_{gate_start}_{gate_end}"
        eval_dir = out_dir / variant
        eval_dir.mkdir(parents=True, exist_ok=True)
        run_eval(
            [
                sys.executable,
                "eval_loop_quality_claude_aligned.py",
                "--checkpoint",
                str(args.base),
                "--residual-checkpoint",
                str(args.residual),
                "--residual-config",
                str(args.residual_config),
                "--gate-start",
                str(gate_start),
                "--gate-end",
                str(gate_end),
                "--out-dir",
                str(eval_dir),
                "--suite",
                "exit_v2",
                "--no-compare",
            ],
            env,
            eval_dir / "loop_quality_summary.csv",
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            append_rows(summary_rows, variant, gate_start, gate_end, eval_dir)
            write_csv(out_dir / "gate_window_ablation_summary.csv", summary_rows)

    report = [
        "# Residual Gate-Window Ablations",
        "",
        f"- base: `{args.base.resolve()}`",
        f"- residual: `{args.residual.resolve()}`",
        f"- residual_config: `{args.residual_config.resolve()}`",
        "- suite: `exit_v2`",
        "- residual_scale: `1.0`",
        "",
        "This is inference-only. No training is performed.",
    ]
    if summary_rows:
        report.extend(
            [
                "",
                "Summary CSV:",
                "",
                f"`{(out_dir / 'gate_window_ablation_summary.csv').resolve()}`",
            ]
        )
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"out_dir={out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
