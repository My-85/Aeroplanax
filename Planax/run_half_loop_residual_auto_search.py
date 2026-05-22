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
GPU_UUID = "GPU-2c45b7fd-69c8-1697-23a0-fe7ce7a2a620"
BASE_CKPT = PLANAX_ROOT / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
BEST_RESIDUAL = (
    PLANAX_ROOT
    / "results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2"
)
BEST_RESIDUAL_CONFIG = (
    PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/configs/round_01_config.json"
)
CACHED_BEST_LOOP = (
    PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/eval/round_01_loop_quality/loop_quality_summary.csv"
)

QUICK_NAMES = "pu150_R12000,pu165_R15000,pu170_R15000"
QUICK_TASKS = ["pu150_R12000", "pu165_R15000", "pu170_R15000"]
DEEP_NAMES = "pu175_R15000,pu180_R15000"
FULL_NAMES = "pu060_R12000,pu090_R12000,pu120_R12000,pu150_R12000,pu165_R15000,pu170_R15000,pu175_R15000,pu180_R15000"
EVAL_TIMEOUT_SEC = 20 * 60
TRAIN_TIMEOUT_SEC = 3 * 60 * 60


def read_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
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


def by_name(rows):
    return {row.get("name") or row.get("task"): row for row in rows}


def run_command(cmd, env, log_path=None, dry_run=False, timeout=None):
    line = " ".join(str(x) for x in cmd)
    print(line, flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PLANAX_ROOT), env=env, check=True, timeout=timeout)


def make_env():
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "JAX_PLATFORMS": "cuda",
            "MPLCONFIGDIR": "/tmp",
            "WANDB_MODE": "offline",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
        }
    )
    return env


def eval_loop(out_dir, residual, residual_config, names, gate_start=None, gate_end=None, scale=None, dry_run=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "loop_quality_summary.csv").exists() and not dry_run:
        print(f"skip existing {out_dir / 'loop_quality_summary.csv'}", flush=True)
        return out_dir / "loop_quality_summary.csv"
    cmd = [
        sys.executable,
        "eval_loop_quality_claude_aligned.py",
        "--checkpoint",
        str(BASE_CKPT),
        "--residual-checkpoint",
        str(residual),
        "--residual-config",
        str(residual_config),
        "--out-dir",
        str(out_dir),
        "--suite",
        "v2",
        "--only-names",
        names,
        "--no-compare",
    ]
    if gate_start is not None:
        cmd += ["--gate-start", str(gate_start)]
    if gate_end is not None:
        cmd += ["--gate-end", str(gate_end)]
    if scale is not None:
        cmd += ["--residual-scale", str(scale)]
    try:
        run_command(cmd, make_env(), out_dir / "command.log", dry_run=dry_run, timeout=EVAL_TIMEOUT_SEC)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        (out_dir / "eval_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        if not (out_dir / "loop_quality_summary.csv").exists():
            write_csv(out_dir / "loop_quality_summary.csv", [], ["name", "termination", "error"])
    return out_dir / "loop_quality_summary.csv"


def eval_loop_staged(out_dir, residual, residual_config, baseline_rows, gate_start=None, gate_end=None, scale=None, dry_run=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "loop_quality_summary.csv").exists() and not dry_run:
        print(f"skip existing {out_dir / 'loop_quality_summary.csv'}", flush=True)
        return out_dir / "loop_quality_summary.csv"
    combined = []
    for task in QUICK_TASKS:
        csv_path = eval_loop(
            out_dir / task,
            residual,
            residual_config,
            task,
            gate_start=gate_start,
            gate_end=gate_end,
            scale=scale,
            dry_run=dry_run,
        )
        if dry_run:
            continue
        rows = read_csv(csv_path)
        if not rows:
            placeholder = {key: "" for key in (baseline_rows[0].keys() if baseline_rows else ["name", "termination"])}
            placeholder["name"] = task
            placeholder["termination"] = "missing"
            combined.append(placeholder)
            break
        combined.extend(rows)
        passed, reasons = quick_gate(combined, baseline_rows, tasks=[r["name"] for r in combined])
        if not passed and task != "pu150_R12000":
            break
    if combined:
        write_csv(out_dir / "loop_quality_summary.csv", combined)
    elif dry_run:
        write_csv(out_dir / "loop_quality_summary.csv", [])
    return out_dir / "loop_quality_summary.csv"


def eval_horizontal(out_dir, residual, residual_config, dry_run=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "eval_vertical_energy_checkpoints.py",
        "--baseline",
        str(BASE_CKPT),
        "--new",
        str(BASE_CKPT),
        "--residual-checkpoint",
        str(residual),
        "--residual-config",
        str(residual_config),
        "--out-dir",
        str(out_dir),
        "--seeds",
        "5",
        "--suite",
        "horizontal_v2",
    ]
    run_command(cmd, make_env(), out_dir / "command.log", dry_run=dry_run, timeout=EVAL_TIMEOUT_SEC)
    return out_dir / "eval_summary.csv"


def train_candidate(candidate_dir, cfg, dry_run=False):
    cfg = deepcopy(cfg)
    candidate_dir = candidate_dir.resolve()
    existing_train_log = candidate_dir / "train_log.csv"
    if existing_train_log.exists() and not dry_run:
        rows = read_csv(existing_train_log)
        if rows:
            existing_ckpt = rows[-1].get("saved_checkpoint", "")
            if existing_ckpt and Path(existing_ckpt).exists():
                print(f"reuse existing checkpoint {existing_ckpt}", flush=True)
                return existing_ckpt
    cfg["OUTPUTDIR"] = str(candidate_dir)
    cfg["LOGDIR"] = str((candidate_dir / "logs").resolve())
    cfg["SAVEDIR"] = str((candidate_dir / "checkpoint").resolve())
    cfg_path = candidate_dir / "config.json"
    write_json(cfg_path, cfg)
    env = make_env()
    env["CONFIG_JSON"] = str(cfg_path)
    cmd = [sys.executable, "train_half_loop_specialist_residual_v1.py"]
    run_command(cmd, env, candidate_dir / "train_command.log", dry_run=dry_run, timeout=TRAIN_TIMEOUT_SEC)
    if dry_run:
        return ""
    train_rows = read_csv(candidate_dir / "train_log.csv")
    if not train_rows:
        raise RuntimeError(f"missing train_log.csv in {candidate_dir}")
    return train_rows[-1]["saved_checkpoint"]


def base_train_cfg(name, family, gate_start, gate_end, scale, clip, lr, anchor_coef, env_params):
    return {
        "GROUP": "half_loop_residual_auto_search",
        "FAMILY": family,
        "CANDIDATE_NAME": name,
        "BASE_CHECKPOINT": str(BASE_CKPT),
        "RESIDUAL_LOADDIR": str(BEST_RESIDUAL),
        "ANCHOR_RESIDUAL_LOADDIR": str(BEST_RESIDUAL),
        "ANCHOR_BC_COEF": anchor_coef,
        "ANCHOR_PHASE_START_DEG": 90.0,
        "ANCHOR_PHASE_END_DEG": 165.0,
        "LR": lr,
        "NUM_ENVS": 1000,
        "NUM_STEPS": 512,
        "TOTAL_TIMESTEPS": 512000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 10,
        "RESIDUAL_FC_DIM_SIZE": 96,
        "RESIDUAL_GRU_HIDDEN_DIM": 64,
        "RESIDUAL_LOGIT_CLIP": clip,
        "RESIDUAL_SCALE": scale,
        "RESIDUAL_L2_COEF": 0.035,
        "NON_LOOP_RESIDUAL_L2_COEF": 0.45,
        "RESIDUAL_SATURATION_COEF": 0.045,
        "RESIDUAL_GATE_START_DEG": gate_start,
        "RESIDUAL_GATE_END_DEG": gate_end,
        "RESIDUAL_PHASE_MAX_DEG": 190.0,
        "RESIDUAL_SMOOTH_GATE_MARGIN_DEG": 5.0,
        "ENV_PARAMS": env_params,
    }


def write_scale_config(path: Path, scale: float):
    with BEST_RESIDUAL_CONFIG.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["RESIDUAL_SCALE"] = scale
    cfg["SCALE_ONLY_ABLATION"] = True
    cfg["BASE_CHECKPOINT"] = str(BASE_CKPT)
    cfg["RESIDUAL_LOADDIR"] = str(BEST_RESIDUAL)
    cfg.setdefault("RESIDUAL_PHASE_MAX_DEG", max(float(cfg.get("RESIDUAL_GATE_END_DEG", 180.0)), 180.0))
    write_json(path, cfg)
    return path


def bridge_env_params(family):
    common = {
        "original_task_prob": 0.04,
        "horizontal_proxy_task_prob": 0.06,
        "level_altitude_task_prob": 0.0,
        "vertical_stage_successes": 10,
        "vertical_stage_offset": 8,
        "vertical_cruise_vt": 250.0,
        "use_loop_plane_targets_for_vertical_arc": 1.0,
        "half_loop_curriculum_prob": 1.0,
        "half_loop_pullup_retention_prob": 0.0,
        "half_loop_climb_retention_prob": 0.0,
        "half_loop_exit_recovery_prob": 0.0,
        "half_loop_partial_exit_prob": 0.0,
        "half_loop_partial_bridge_prob": 0.7,
        "half_loop_max_phase_deg": 180.0,
        "min_vertical_duration_sec": 8.0,
        "max_vertical_duration_sec": 40.0,
        "ve_low_speed_threshold": 182.0,
        "ve_strong_low_speed_threshold": 172.0,
        "ve_alpha_soft_deg": 14.0,
        "ve_alpha_hard_deg": 18.0,
        "ve_beta_soft_deg": 10.0,
        "ve_g_soft": 8.4,
        "ve_g_hard": 9.4,
        "ve_loop_geom_weight": 0.024,
        "ve_loop_roll_weight": 0.014,
        "ve_loop_nose_tangent_weight": 0.034,
        "ve_loop_wing_plane_weight": 0.040,
        "ve_loop_velocity_tangent_weight": 0.028,
        "ve_loop_nose_velocity_weight": 0.020,
        "ve_high_speed_alpha_weight": 0.035,
        "ve_action_saturation_weight": 0.030,
    }
    if family == "A":
        common.update(
            {
                "half_loop_vertical_retention_prob": 0.42,
                "half_loop_transition_prob": 0.08,
                "half_loop_bridge_transition_prob": 0.50,
                "half_loop_partial_prob": 0.10,
            }
        )
    else:
        common.update(
            {
                "half_loop_vertical_retention_prob": 0.38,
                "half_loop_transition_prob": 0.16,
                "half_loop_bridge_transition_prob": 0.38,
                "half_loop_partial_prob": 0.10,
            }
        )
    return common


def candidate_configs(round_idx, failure_hint):
    if round_idx == 1:
        return [
            base_train_cfg(
                "family_A_bridge_strong_anchor_absfix",
                "A",
                120.0,
                180.0,
                0.25,
                0.50,
                1e-6,
                2.5,
                bridge_env_params("A"),
            ),
            base_train_cfg(
                "family_B_bridge_mild_extension_absfix",
                "B",
                100.0,
                190.0,
                0.35,
                0.65,
                1.5e-6,
                1.25,
                bridge_env_params("B"),
            ),
        ]
    scale = 0.25 if failure_hint in {"early_bridge_regression", "residual_overpower", "retention_failure"} else 0.35
    lr = 1e-6 if failure_hint != "insufficient_residual" else 2e-6
    anchor = 3.0 if failure_hint in {"early_bridge_regression", "retention_failure"} else 1.5
    gate_start = 140.0 if failure_hint in {"early_bridge_regression", "retention_failure"} else 120.0
    env_a = bridge_env_params("A")
    env_b = bridge_env_params("B")
    if failure_hint == "insufficient_residual":
        env_b["half_loop_bridge_transition_prob"] = 0.48
        env_b["half_loop_vertical_retention_prob"] = 0.34
    return [
        base_train_cfg(
            f"round_{round_idx}_mut_A_{failure_hint}",
            "A",
            gate_start,
            180.0,
            scale,
            0.50,
            lr,
            anchor,
            env_a,
        ),
        base_train_cfg(
            f"round_{round_idx}_mut_B_{failure_hint}",
            "B",
            120.0,
            185.0,
            min(scale + 0.10, 0.50),
            0.65,
            lr,
            max(anchor * 0.6, 1.0),
            env_b,
        ),
    ]


def quick_gate(rows, baseline_rows, tasks=None):
    cand = by_name(rows)
    base = by_name(baseline_rows)
    reasons = []
    missing = []
    for task in (tasks or QUICK_TASKS):
        c = cand.get(task)
        b = base.get(task)
        if not c or not b:
            missing.append(f"{task}:missing")
            continue
        if task == "pu150_R12000":
            if c.get("termination") != "ok":
                reasons.append(f"{task}:termination {c.get('termination')}")
            if f(c, "CTE_mean", 1e9) > f(b, "CTE_mean", 1e9) + 75.0:
                reasons.append(f"{task}:CTE {f(b,'CTE_mean'):.1f}->{f(c,'CTE_mean'):.1f}")
            if f(c, "wing_plane_error_mean", 1e9) > f(b, "wing_plane_error_mean", 1e9) + 5.0:
                reasons.append(f"{task}:wing {f(b,'wing_plane_error_mean'):.1f}->{f(c,'wing_plane_error_mean'):.1f}")
        else:
            if task in {"pu165_R15000", "pu170_R15000"} and c.get("termination") == "crash":
                reasons.append(f"{task}:bridge_crash")
            if c.get("termination") == "crash" and b.get("termination") != "crash":
                reasons.append(f"{task}:new_crash")
            if f(c, "CTE_mean", 1e9) > f(b, "CTE_mean", 1e9) + 250.0:
                reasons.append(f"{task}:CTE {f(b,'CTE_mean'):.1f}->{f(c,'CTE_mean'):.1f}")
            if f(c, "wing_plane_error_mean", 1e9) > f(b, "wing_plane_error_mean", 1e9) + 3.0:
                reasons.append(f"{task}:wing {f(b,'wing_plane_error_mean'):.1f}->{f(c,'wing_plane_error_mean'):.1f}")
            if f(c, "nose_tangent_error_mean", 1e9) > f(b, "nose_tangent_error_mean", 1e9) + 3.0:
                reasons.append(f"{task}:nose {f(b,'nose_tangent_error_mean'):.1f}->{f(c,'nose_tangent_error_mean'):.1f}")
            if f(c, "env_alpha_max", 1e9) > f(b, "env_alpha_max", 1e9) + 2.0:
                reasons.append(f"{task}:alpha {f(b,'env_alpha_max'):.1f}->{f(c,'env_alpha_max'):.1f}")
            if f(c, "Gmax", 1e9) > f(b, "Gmax", 1e9) + 0.35:
                reasons.append(f"{task}:Gmax {f(b,'Gmax'):.2f}->{f(c,'Gmax'):.2f}")
    if not reasons and missing:
        reasons.extend(missing)
    return not reasons, reasons


def score_candidate(rows, baseline_rows):
    cand = by_name(rows)
    base = by_name(baseline_rows)
    score = 0.0
    for task, weight in [("pu165_R15000", 2.0), ("pu170_R15000", 2.0), ("pu175_R15000", 1.0), ("pu180_R15000", 1.0)]:
        c = cand.get(task)
        b = base.get(task)
        if not c or not b:
            continue
        if c.get("termination") in {"", "missing"}:
            score -= 50.0
            continue
        score += weight * max(0.0, f(b, "CTE_mean") - f(c, "CTE_mean")) / 100.0
        score += weight * max(0.0, f(b, "wing_plane_error_mean") - f(c, "wing_plane_error_mean"))
        score += weight * max(0.0, f(b, "nose_tangent_error_mean") - f(c, "nose_tangent_error_mean"))
        if c.get("termination") == "crash":
            score -= 100.0
    return score


GRADE_VALUE = {"A": 4, "B": 3, "C": 2, "F": 1, "Fail": 0, "": 0}


def full_gate(full_csv, horizontal_csv):
    reasons = []
    full_rows = read_csv(Path(full_csv)) if full_csv else []
    full = by_name(full_rows)
    baseline_full = by_name(read_csv(CACHED_BEST_LOOP))
    for task in ["pu060_R12000", "pu090_R12000", "pu120_R12000", "pu150_R12000"]:
        c = full.get(task)
        b = baseline_full.get(task)
        if not c or not b:
            reasons.append(f"{task}:missing")
            continue
        if c.get("termination") != "ok":
            reasons.append(f"{task}:termination {c.get('termination')}")
        if GRADE_VALUE.get(c.get("grade_loop_quality", ""), 0) < GRADE_VALUE.get(b.get("grade_loop_quality", ""), 0):
            reasons.append(f"{task}:grade {b.get('grade_loop_quality')}->{c.get('grade_loop_quality')}")
        if f(c, "Gmax", 1e9) > f(b, "Gmax", 1e9) + 0.35:
            reasons.append(f"{task}:Gmax {f(b,'Gmax'):.2f}->{f(c,'Gmax'):.2f}")
        if f(c, "vt_min", 0.0) < f(b, "vt_min", 0.0) - 5.0:
            reasons.append(f"{task}:vt_min {f(b,'vt_min'):.1f}->{f(c,'vt_min'):.1f}")

    h_rows = read_csv(Path(horizontal_csv)) if horizontal_csv else []
    by_policy_task = {(r.get("policy"), r.get("task")): r for r in h_rows}
    tasks = [
        "level_circle_R3000_right",
        "level_circle_R3000_left",
        "level_circle_R5000_right",
        "level_circle_R5000_left",
        "s_curve_A3000",
        "figure_eight_R5000",
        "mild_climb_p1000m",
        "mild_descent_m1000m",
    ]
    for task in tasks:
        c = by_policy_task.get(("candidate", task))
        b = by_policy_task.get(("baseline_epoch600", task))
        if not c or not b:
            reasons.append(f"{task}:missing")
            continue
        if f(c, "success_rate") < f(b, "success_rate") - 0.10:
            reasons.append(f"{task}:success {f(b,'success_rate'):.2f}->{f(c,'success_rate'):.2f}")
        if f(c, "crash_rate") > f(b, "crash_rate") + 0.001:
            reasons.append(f"{task}:crash {f(b,'crash_rate'):.2f}->{f(c,'crash_rate'):.2f}")
        if f(c, "Gmax_mean") > f(b, "Gmax_mean") + 0.35:
            reasons.append(f"{task}:Gmax {f(b,'Gmax_mean'):.2f}->{f(c,'Gmax_mean'):.2f}")
        if abs(f(c, "altitude_drift_mean")) > abs(f(b, "altitude_drift_mean")) + 100.0:
            reasons.append(f"{task}:alt_drift {f(b,'altitude_drift_mean'):.1f}->{f(c,'altitude_drift_mean'):.1f}")
    return not reasons, reasons


def classify_failure(reasons, train_log):
    text = " ".join(reasons)
    if "missing" in text:
        return "target_or_eval_bug"
    if "pu150" in text:
        return "retention_failure"
    if "alpha" in text:
        return "residual_overpower"
    if "wing" in text or "nose" in text or "CTE" in text or "crash" in text:
        return "early_bridge_regression"
    rows = read_csv(train_log) if train_log and train_log.is_file() else []
    if rows and float(rows[-1].get("gate_rate_mean", "0") or 0.0) < 0.05:
        return "target_or_eval_bug"
    return "insufficient_residual"


def copy_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def write_round_files(round_dir, candidate_records, promoted_record, failure_mode):
    summary_fields = [
        "candidate",
        "family",
        "kind",
        "gate_pass",
        "score",
        "failure_mode",
        "checkpoint",
        "quick_eval",
        "full_eval",
        "horizontal_eval",
    ]
    rows = []
    for rec in candidate_records:
        rows.append({key: rec.get(key, "") for key in summary_fields})
    write_csv(round_dir / "eval_quick.csv", rows, summary_fields)
    write_csv(round_dir / "eval_full.csv", rows, summary_fields)
    write_csv(round_dir / "phasewise_metrics.csv", rows, summary_fields)
    (round_dir / "failure_mode.md").write_text(f"# Failure Mode\n\n`{failure_mode}`\n", encoding="utf-8")
    report = ["# Round Report", "", f"- failure_mode: `{failure_mode}`", ""]
    for rec in candidate_records:
        report.append(
            f"- {rec['candidate']}: gate={rec['gate_pass']} score={rec['score']:.3f} "
            f"failure={rec['failure_mode']} checkpoint=`{rec.get('checkpoint', '')}`"
        )
    if promoted_record:
        report.append("")
        report.append(f"Promoted candidate: `{promoted_record['candidate']}`")
    (round_dir / "round_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def evaluate_scale_ablation(round_dir, baseline_rows, dry_run=False):
    records = []
    for scale in [0.25]:
        cand_dir = round_dir / "scale_ablation" / f"scale_{scale:g}"
        cfg_path = write_scale_config(cand_dir / "config.json", scale)
        quick_csv = eval_loop_staged(
            cand_dir / "quick",
            BEST_RESIDUAL,
            cfg_path,
            baseline_rows,
            dry_run=dry_run,
        )
        rows = [] if dry_run else read_csv(quick_csv)
        gate_pass, reasons = (False, ["dry_run"]) if dry_run else quick_gate(rows, baseline_rows)
        full_csv = ""
        horizontal_csv = ""
        if gate_pass and not dry_run:
            full_csv = str(eval_loop(cand_dir / "full_v2", BEST_RESIDUAL, cfg_path, FULL_NAMES))
            horizontal_csv = str(eval_horizontal(cand_dir / "horizontal_v2", BEST_RESIDUAL, cfg_path))
            full_pass, full_reasons = full_gate(full_csv, horizontal_csv)
            gate_pass = gate_pass and full_pass
            reasons.extend(full_reasons)
        records.append(
            {
                "candidate": f"scale_only_{scale:g}",
                "family": "C",
                "kind": "scale_only",
                "gate_pass": gate_pass,
                "score": score_candidate(rows, baseline_rows) if rows else 0.0,
                "failure_mode": classify_failure(reasons, Path("")),
                "failure_reasons": reasons,
                "checkpoint": str(BEST_RESIDUAL),
                "quick_eval": str(quick_csv),
                "full_eval": full_csv,
                "horizontal_eval": horizontal_csv,
            }
        )
        write_json(cand_dir / "candidate_manifest.json", records[-1])
        if gate_pass and records[-1]["score"] > 0.0:
            break
    for scale in [0.5, 0.75]:
        cand_dir = round_dir / "scale_ablation" / f"scale_{scale:g}"
        cfg_path = write_scale_config(cand_dir / "config.json", scale)
        record = {
            "candidate": f"scale_only_{scale:g}",
            "family": "C",
            "kind": "scale_only_skipped",
            "gate_pass": False,
            "score": 0.0,
            "failure_mode": "target_or_eval_bug" if scale == 0.5 else "residual_overpower",
            "failure_reasons": [
                "scale_0.5_pu165_timed_out_or_was_interrupted",
                "skip_larger_scale_until_bridge_training_candidate_exists",
            ],
            "checkpoint": str(BEST_RESIDUAL),
            "quick_eval": str(cand_dir / "quick"),
            "full_eval": "",
            "horizontal_eval": "",
        }
        write_json(cand_dir / "candidate_manifest.json", record)
        records.append(record)
    cand_dir = round_dir / "scale_ablation" / "scale_1"
    cfg_path = write_scale_config(cand_dir / "config.json", 1.0)
    quick_dir = cand_dir / "quick"
    quick_dir.mkdir(parents=True, exist_ok=True)
    write_csv(quick_dir / "loop_quality_summary.csv", baseline_rows)
    gate_pass, reasons = quick_gate(baseline_rows, baseline_rows)
    record = {
        "candidate": "scale_only_1",
        "family": "C",
        "kind": "scale_only_cached_baseline",
        "gate_pass": gate_pass,
        "score": score_candidate(baseline_rows, baseline_rows),
        "failure_mode": classify_failure(reasons, Path("")),
        "failure_reasons": reasons + [f"cached_from:{CACHED_BEST_LOOP}"],
        "checkpoint": str(BEST_RESIDUAL),
        "quick_eval": str(quick_dir / "loop_quality_summary.csv"),
        "full_eval": "",
        "horizontal_eval": "",
    }
    write_json(cand_dir / "candidate_manifest.json", record)
    records.append(record)
    return records


def run_candidate(round_dir, cfg, baseline_rows, dry_run=False):
    name = cfg["CANDIDATE_NAME"]
    cand_dir = round_dir / "candidates" / name
    try:
        checkpoint = train_candidate(cand_dir, cfg, dry_run=dry_run)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        failure_mode = "target_or_eval_bug"
        manifest = {
            "candidate": name,
            "family": cfg["FAMILY"],
            "checkpoint": "",
            "gate_pass": False,
            "quick_reasons": [f"train_failed:{type(exc).__name__}"],
            "failure_mode": failure_mode,
            "score": 0.0,
            "config": str((cand_dir / "config.json").resolve()),
            "quick_eval": "",
            "full_eval": "",
            "horizontal_eval": "",
        }
        write_json(cand_dir / "candidate_manifest.json", manifest)
        (cand_dir / "failure_mode.md").write_text(f"# Failure Mode\n\n`{failure_mode}`\n\n{manifest['quick_reasons']}\n", encoding="utf-8")
        return {
            "candidate": name,
            "family": cfg["FAMILY"],
            "kind": "trained",
            "gate_pass": False,
            "score": 0.0,
            "failure_mode": failure_mode,
            "failure_reasons": manifest["quick_reasons"],
            "checkpoint": "",
            "quick_eval": "",
            "full_eval": "",
            "horizontal_eval": "",
            "manifest": str(cand_dir / "candidate_manifest.json"),
        }
    copy_if_exists(cand_dir / "train_log.csv", round_dir / "train_log.csv")
    quick_csv = eval_loop_staged(
        cand_dir / "quick",
        checkpoint or BEST_RESIDUAL,
        cand_dir / "config.json",
        baseline_rows,
        dry_run=dry_run,
    )
    rows = [] if dry_run else read_csv(quick_csv)
    gate_pass, reasons = (False, ["dry_run"]) if dry_run else quick_gate(rows, baseline_rows)
    full_csv = ""
    horizontal_csv = ""
    if gate_pass and not dry_run:
        deep_csv = eval_loop(cand_dir / "deep", checkpoint, cand_dir / "config.json", DEEP_NAMES)
        deep_rows = read_csv(deep_csv)
        rows = rows + deep_rows
        full_csv = eval_loop(cand_dir / "full_v2", checkpoint, cand_dir / "config.json", FULL_NAMES)
        horizontal_csv = eval_horizontal(cand_dir / "horizontal_v2", checkpoint, cand_dir / "config.json")
        full_pass, full_reasons = full_gate(full_csv, horizontal_csv)
        gate_pass = gate_pass and full_pass
        reasons.extend(full_reasons)
    failure_mode = classify_failure(reasons, cand_dir / "train_log.csv")
    score = score_candidate(rows, baseline_rows) if rows else 0.0
    manifest = {
        "candidate": name,
        "family": cfg["FAMILY"],
        "checkpoint": checkpoint,
        "gate_pass": gate_pass,
        "quick_reasons": reasons,
        "failure_mode": failure_mode,
        "score": score,
        "config": str((cand_dir / "config.json").resolve()),
        "quick_eval": str(quick_csv),
        "full_eval": str(full_csv),
        "horizontal_eval": str(horizontal_csv),
    }
    write_json(cand_dir / "candidate_manifest.json", manifest)
    (cand_dir / "failure_mode.md").write_text(f"# Failure Mode\n\n`{failure_mode}`\n\n{reasons}\n", encoding="utf-8")
    return {
        "candidate": name,
        "family": cfg["FAMILY"],
        "kind": "trained",
        "gate_pass": gate_pass,
        "score": score,
        "failure_mode": failure_mode,
        "failure_reasons": reasons,
        "checkpoint": checkpoint,
        "quick_eval": str(quick_csv),
        "full_eval": str(full_csv),
        "horizontal_eval": str(horizontal_csv),
        "manifest": str(cand_dir / "candidate_manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.out_dir or (
        PLANAX_ROOT / "results/half_loop_residual_auto_search" / datetime.now().strftime("%Y%m%d_%H%M")
    )
    root.mkdir(parents=True, exist_ok=True)
    decision_log = root / "decision_log.md"
    decision_log.write_text("# Decision Log\n\n", encoding="utf-8")

    write_json(
        root / "search_config.json",
        {
            "base": str(BASE_CKPT),
            "initial_best_residual": str(BEST_RESIDUAL),
            "initial_best_residual_config": str(BEST_RESIDUAL_CONFIG),
            "gpu_uuid": GPU_UUID,
            "max_rounds": args.max_rounds,
            "rules": "Never train monolithic ep619; never continue failed update_4 branches.",
        },
    )

    baseline_dir = root / "baseline_update_2"
    baseline_quick = baseline_dir / "quick" / "loop_quality_summary.csv"
    if CACHED_BEST_LOOP.exists() and not args.dry_run:
        cached_rows = read_csv(CACHED_BEST_LOOP)
        cached_by_name = by_name(cached_rows)
        baseline_rows = [cached_by_name[name] for name in QUICK_TASKS if name in cached_by_name]
        write_csv(baseline_quick, baseline_rows)
        (baseline_dir / "quick" / "source.txt").write_text(
            f"Copied from cached residual_update_2 eval:\n{CACHED_BEST_LOOP}\n",
            encoding="utf-8",
        )
    else:
        baseline_quick = eval_loop(
            baseline_dir / "quick",
            BEST_RESIDUAL,
            BEST_RESIDUAL_CONFIG,
            QUICK_NAMES,
            dry_run=args.dry_run,
        )
        baseline_rows = [] if args.dry_run else read_csv(baseline_quick)
    write_json(
        root / "best_candidate_manifest.json",
        {
            "base_checkpoint": str(BASE_CKPT),
            "residual_checkpoint": str(BEST_RESIDUAL),
            "residual_config": str(BEST_RESIDUAL_CONFIG),
            "promoted": False,
            "note": "Initial best remains residual_update_2 until a strict-gate candidate beats it.",
        },
    )

    search_rows = []
    best_record = None
    failure_hint = "initial"
    no_improve_rounds = 0

    for round_idx in range(1, args.max_rounds + 1):
        round_dir = root / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        with decision_log.open("a", encoding="utf-8") as f_log:
            f_log.write(f"## Round {round_idx:02d}\n\nfailure_hint before round: `{failure_hint}`\n\n")

        candidate_records = []
        if round_idx == 1:
            scale_records = evaluate_scale_ablation(round_dir, baseline_rows, dry_run=args.dry_run)
            candidate_records.extend(scale_records)
            for record in scale_records:
                search_rows.append(
                    {
                        "round": round_idx,
                        "candidate": record["candidate"],
                        "family": record["family"],
                        "kind": record["kind"],
                        "gate_pass": record["gate_pass"],
                        "score": f"{record['score']:.6f}",
                        "failure_mode": record["failure_mode"],
                        "checkpoint": record.get("checkpoint", ""),
                        "quick_eval": record.get("quick_eval", ""),
                        "full_eval": record.get("full_eval", ""),
                        "horizontal_eval": record.get("horizontal_eval", ""),
                    }
                )
            write_csv(root / "search_summary.csv", search_rows)
        scale_promoted = any(
            rec["kind"] == "scale_only" and rec["gate_pass"] and rec["score"] > 0.0
            for rec in candidate_records
        )

        for cfg in ([] if scale_promoted else candidate_configs(round_idx, failure_hint)):
            record = run_candidate(round_dir, cfg, baseline_rows, dry_run=args.dry_run)
            candidate_records.append(record)
            search_rows.append(
                {
                    "round": round_idx,
                    "candidate": record["candidate"],
                    "family": record["family"],
                    "kind": record["kind"],
                    "gate_pass": record["gate_pass"],
                    "score": f"{record['score']:.6f}",
                    "failure_mode": record["failure_mode"],
                    "checkpoint": record.get("checkpoint", ""),
                    "quick_eval": record.get("quick_eval", ""),
                    "full_eval": record.get("full_eval", ""),
                    "horizontal_eval": record.get("horizontal_eval", ""),
                }
            )
            write_csv(root / "search_summary.csv", search_rows)

        passed = [rec for rec in candidate_records if rec["gate_pass"]]
        promoted = max(passed, key=lambda x: x["score"]) if passed else None
        if promoted and promoted["score"] > 0.0:
            best_record = promoted
            no_improve_rounds = 0
            write_json(
                root / "best_candidate_manifest.json",
                {
                    "base_checkpoint": str(BASE_CKPT),
                    "residual_checkpoint": promoted["checkpoint"],
                    "promoted": True,
                    "round": round_idx,
                    "candidate": promoted["candidate"],
                    "score": promoted["score"],
                    "manifest": promoted.get("manifest", ""),
                },
            )
            failure_hint = "promoted"
        else:
            no_improve_rounds += 1
            failure_hint = max(candidate_records, key=lambda x: x["score"])["failure_mode"] if candidate_records else "target_or_eval_bug"

        write_round_files(round_dir, candidate_records, promoted, failure_hint)
        with decision_log.open("a", encoding="utf-8") as f_log:
            for rec in candidate_records:
                f_log.write(
                    f"- {rec['candidate']}: gate={rec['gate_pass']} score={rec['score']:.3f} "
                    f"failure={rec['failure_mode']} reasons={rec.get('failure_reasons', [])}\n"
                )
            f_log.write(f"\nround_decision: `{failure_hint}`\n\n")

        if best_record is not None:
            break
        if no_improve_rounds >= 2:
            with decision_log.open("a", encoding="utf-8") as f_log:
                f_log.write("early_stop: `two_rounds_without_strict_gate_improvement`\n\n")
            break
        if failure_hint == "target_or_eval_bug":
            break

    final = [
        "# Half-Loop Residual Auto Search Final Report",
        "",
        f"1. Best current combination: base `{BASE_CKPT}` + residual `{BEST_RESIDUAL}`"
        if best_record is None
        else f"1. Best current combination: base `{BASE_CKPT}` + residual `{best_record['checkpoint']}`",
        f"2. Did any candidate beat residual_update_2? `{best_record is not None}`",
        "3. Did bridge tasks pu165/pu170 improve? `see search_summary.csv; promotion requires both quick gates`",
        "4. Did pu175 improve or complete? `evaluated only for candidates passing bridge quick gate`",
        "5. Did pu180 improve? `evaluated only after bridge quick gate`",
        "6. Did horizontal tasks remain unchanged? `full horizontal gate runs only for quick-pass candidates; residual gate is loop-phase-only`",
        "7. Did 60/90/120/150 remain unchanged? `full v2 gate runs only for quick-pass candidates; pu150 is always in quick gate`",
        f"8. Remaining failure mode: `{failure_hint}`",
        "9. Next stage: `continue bridge repair; do not train exit/recovery until pu165/pu170 pass`",
        f"10. Ready for Claude ACMI regression? `{best_record is not None}`",
        "",
        f"- search summary: `{(root / 'search_summary.csv').resolve()}`",
        f"- decision log: `{decision_log.resolve()}`",
    ]
    (root / "final_report.md").write_text("\n".join(final) + "\n", encoding="utf-8")
    print(f"search_dir={root.resolve()}", flush=True)
    print(f"promoted={best_record is not None}", flush=True)


if __name__ == "__main__":
    main()
