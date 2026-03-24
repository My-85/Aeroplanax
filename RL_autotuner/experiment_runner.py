#!/usr/bin/env python3
"""
experiment_runner.py — Main orchestrator for the RL autotuner.

Follows the autoresearch pattern:
  1. Single modifiable file (reward_config.json) — Phase 1
  2. Fixed evaluation protocol (evaluator.py)
  3. Champion/challenger/discard mechanism
  4. Results tracking (results.jsonl)

Modes:
  manual:  Human edits reward_config.json, runner does train→eval→keep/discard
  auto:    Claude suggests config changes via API
  dry-run: Validate config without training

Usage:
    python experiment_runner.py --mode manual [--budget 100000000]
    python experiment_runner.py --mode auto --api-key KEY [--budget 100000000]
    python experiment_runner.py --mode dry-run
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import sys
import json
import subprocess
import time
import re
import shutil
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional

# Paths
AUTOTUNER_DIR = Path(__file__).resolve().parent
PLANAX_DIR = AUTOTUNER_DIR.parent / "Planax"
REWARD_CONFIG_PATH = AUTOTUNER_DIR / "reward_config.json"
CHAMPION_META_PATH = AUTOTUNER_DIR / "champion" / "champion_meta.json"
RESULTS_PATH = AUTOTUNER_DIR / "results.jsonl"
TRAINING_SCRIPT = PLANAX_DIR / "train_quat_baseline_iter.py"

# Import local modules
sys.path.insert(0, str(AUTOTUNER_DIR))
from config_patcher import load_config, patch_reward_file, backup_reward_file, validate_config, patch_reward_file_code
from evaluator import extract_training_metrics


# ======================== Training Budget ========================

DEFAULT_BUDGET = 500_000_000   # 500M steps (match champion training scale)
EARLY_STOP_MIN_STEPS = 200_000_000  # 200M steps before checking early stop
EARLY_STOP_WINDOW = 50         # check improvement over last N return records
EARLY_STOP_RETURN_TOL = 0.5   # minimum return improvement over window


# ======================== Champion Management ========================

def load_champion() -> dict:
    """Load current champion metadata."""
    if CHAMPION_META_PATH.exists():
        with open(CHAMPION_META_PATH) as f:
            return json.load(f)
    return {"experiment_id": 0, "metrics": {}, "status": "placeholder"}


def save_champion(meta: dict):
    """Save new champion metadata."""
    CHAMPION_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAMPION_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"CHAMPION updated: experiment #{meta['experiment_id']}")


def is_better_than_champion(new_metrics: dict, champion: dict) -> bool:
    """Compare new experiment metrics against champion.

    Primary metric: mean_theta_deg or final_theta_deg (lower = better)
    Tiebreaker: mean_delta_vt or final_delta_vt (lower = better)

    Returns True if new experiment should become champion.
    """
    champ_metrics = champion.get("metrics", {})
    # Support both formal eval (mean_theta_deg) and training log fallback (final_theta_deg)
    champ_theta = champ_metrics.get("mean_theta_deg") or champ_metrics.get("final_theta_deg")
    new_theta = new_metrics.get("mean_theta_deg") or new_metrics.get("final_theta_deg")

    # If champion has no metrics (placeholder), any real result wins
    if champ_theta is None and new_theta is not None:
        return True
    if new_theta is None:
        return False

    # Primary: lower theta is better (0.3° margin — we use 3-seed eval to reduce noise)
    if new_theta < champ_theta - 0.3:
        return True
    if new_theta > champ_theta + 0.3:
        return False

    # Tiebreaker: lower delta_vt
    champ_dvt = champ_metrics.get("mean_delta_vt") or champ_metrics.get("final_delta_vt", float("inf"))
    new_dvt = new_metrics.get("mean_delta_vt") or new_metrics.get("final_delta_vt", float("inf"))
    if champ_dvt is None:
        champ_dvt = float("inf")
    if new_dvt is None:
        new_dvt = float("inf")
    return new_dvt < champ_dvt


# ======================== Results Logging ========================

def log_result(experiment_id: int, config: dict, metrics: dict, status: str, description: str):
    """Append result to results.jsonl."""
    record = {
        "experiment_id": experiment_id,
        "config_snapshot": config,
        "metrics": metrics,
        "status": status,
        "description": description,
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Logged experiment #{experiment_id}: status={status}")


def get_next_experiment_id() -> int:
    """Get the next experiment ID from results.jsonl."""
    if not RESULTS_PATH.exists():
        return 1
    count = 0
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                count += 1
    return count + 1


def load_results_history() -> list:
    """Load all past experiment results for context."""
    results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


# ======================== Training ========================

def run_training(budget: int, experiment_id: int) -> dict:
    """Launch training subprocess and monitor progress.

    Returns dict with:
      - log_lines: stdout lines
      - metrics: extracted training metrics
      - checkpoint_path: path to best checkpoint (or None)
      - status: 'completed', 'early_stopped', 'crashed'
      - wall_time: seconds elapsed
    """
    # Split budget into multiple iterations for intermediate checkpoints.
    # This allows early stop to kill training while still having a checkpoint to evaluate.
    num_iterations = max(1, int(budget) // 25_000_000)  # checkpoint every ~25M steps
    per_iter_steps = int(budget) // num_iterations

    env = os.environ.copy()
    env["DRYRUN_TIMESTEPS"] = str(per_iter_steps)
    env["FOR_LOOP_EPOCHS"] = str(num_iterations)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    # Load champion checkpoint for fine-tuning (don't train from scratch)
    champion = load_champion()
    champ_ckpt = champion.get("checkpoint_path")
    if champ_ckpt and os.path.exists(str(champ_ckpt)):
        env["LOADDIR"] = str(champ_ckpt)
        print(f"  Fine-tuning from champion checkpoint: {champ_ckpt}")
    else:
        print(f"  WARNING: No champion checkpoint found, training from scratch!")

    cmd = [sys.executable, str(TRAINING_SCRIPT)]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT #{experiment_id}: Training with budget={budget:.0e} steps")
    print(f"{'='*60}")

    t0 = time.time()
    log_lines = []
    return_history = []
    cooldown_records = 0  # after CHECKPOINT line, skip N return records before checking early stop
    COOLDOWN_AFTER_CHECKPOINT = 5  # skip first 5 records after iteration boundary

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PLANAX_DIR),
        text=True,
        bufsize=1,
    )

    status = "completed"
    try:
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            log_lines.append(line)

            # Print important lines
            if any(kw in line for kw in ["theta_deg", "episodic_return", "curriculum", "CHECKPOINT", "SUMMARY", "env_step"]):
                print(f"  [{time.time()-t0:.0f}s] {line}")

            # Detect iteration boundary — reset cooldown so early stop ignores cold-start
            if "CHECKPOINT" in line:
                cooldown_records = COOLDOWN_AFTER_CHECKPOINT

            # Track episodic_return
            m = re.search(r"episodic_return[=:\s]+([-\d.]+)", line)
            if m:
                return_history.append(float(m.group(1)))
                if cooldown_records > 0:
                    cooldown_records -= 1

            # Early stopping (only when not in post-checkpoint cooldown)
            m_step = re.search(r"env_step[=:\s]+(\d+)", line)
            if m_step and cooldown_records == 0:
                env_step = int(m_step.group(1))
                if (len(return_history) >= EARLY_STOP_MIN_STEPS // 1_000_000
                        and len(return_history) >= EARLY_STOP_WINDOW):
                    old_return = return_history[-EARLY_STOP_WINDOW]
                    new_return = return_history[-1]
                    improvement = new_return - old_return
                    if improvement < EARLY_STOP_RETURN_TOL:
                        print(f"  EARLY STOP: return improved only {improvement:.1f} over {EARLY_STOP_WINDOW} records")
                        print(f"    (old={old_return:.1f}, new={new_return:.1f})")
                        status = "early_stopped"
                        proc.send_signal(signal.SIGTERM)
                        time.sleep(2)
                        if proc.poll() is None:
                            proc.kill()
                        break

    except Exception as e:
        print(f"  ERROR during training: {e}")
        status = "crashed"
        proc.kill()

    proc.wait()
    wall_time = time.time() - t0

    if proc.returncode != 0 and status == "completed":
        status = "crashed"

    # Extract metrics from log
    metrics = extract_training_metrics(log_lines)

    # Find checkpoint path from log
    checkpoint_path = None
    for line in reversed(log_lines):
        # Match absolute path first (from updated training script)
        m = re.search(r"CHECKPOINT\s+(/\S+/checkpoint_epoch_\d+)", line)
        if m:
            checkpoint_path = m.group(1)
            break
        # Fallback: relative path
        m = re.search(r"(results/.*?/checkpoints/checkpoint_epoch_\d+)", line)
        if m:
            checkpoint_path = str(PLANAX_DIR / m.group(1))
            break

    # Fallback: scan results directory for checkpoint created during this run
    if checkpoint_path is None:
        results_dir = PLANAX_DIR / "results"
        if results_dir.exists():
            import glob as glob_mod
            ckpts = sorted(glob_mod.glob(str(results_dir / "*/checkpoints/checkpoint_epoch_*")))
            if ckpts:
                # Only consider checkpoints created after training started
                ckpts = [c for c in ckpts if os.path.getmtime(c) >= t0]
                if ckpts:
                    ckpts.sort(key=os.path.getmtime)
                    checkpoint_path = ckpts[-1]
                    print(f"  Found checkpoint via directory scan: {checkpoint_path}")

    print(f"\n  Training {status} in {wall_time:.0f}s")
    print(f"  Training log metrics (pre-eval): return={metrics.get('final_episodic_return', '?')}")

    return {
        "log_lines": log_lines,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
        "status": status,
        "wall_time": wall_time,
    }


# ======================== Experiment Loop ========================

def run_experiment(config: dict, budget: int, description: str = "") -> dict:
    """Run a single experiment: patch config → train → evaluate → compare → keep/discard."""
    experiment_id = get_next_experiment_id()

    # 1. Validate config
    warnings = validate_config(config)
    if any("MISSING" in w for w in warnings):
        print(f"FATAL: Config validation failed: {warnings}")
        log_result(experiment_id, config, {}, "crash", f"Config validation failed: {warnings}")
        return {"status": "crash", "reason": "config_validation"}

    if warnings:
        print(f"Config warnings: {warnings}")

    # 2. Backup and patch reward file
    backup_reward_file()
    success = patch_reward_file(config)
    if not success:
        log_result(experiment_id, config, {}, "crash", "Failed to patch reward file")
        return {"status": "crash", "reason": "patch_failed"}

    # 3. Run training
    train_result = run_training(budget, experiment_id)
    training_metrics = train_result["metrics"]

    # 4. Formal evaluation (if checkpoint exists and training didn't crash)
    eval_metrics = None
    per_level_results = None
    early_exit = False
    if train_result["status"] != "crashed" and train_result["checkpoint_path"]:
        try:
            from evaluator import evaluate_checkpoint_per_level, EVAL_CONFIG
            print(f"\n  Running per-level formal evaluation on checkpoint...")
            # Load champion per-level data for early-exit comparison
            champion = load_champion()
            champion_per_level = champion.get("per_level_metrics") if champion else None
            eval_result = evaluate_checkpoint_per_level(
                train_result["checkpoint_path"], dict(EVAL_CONFIG),
                champion_per_level=champion_per_level
            )
            eval_metrics = eval_result["overall"]
            per_level_results = eval_result["per_level"]
            early_exit = eval_result["overall"].get("early_exit", False)
            print(f"  Eval overall: theta={eval_metrics.get('mean_theta_deg', '?')}°, "
                  f"delta_vt={eval_metrics.get('mean_delta_vt', '?')}, "
                  f"crash_rate={eval_metrics.get('mean_crash_rate', '?')}")
        except Exception as e:
            print(f"  Formal evaluation failed: {e}, falling back to training metrics")
            import traceback
            traceback.print_exc()
            eval_metrics = None

    # Use eval_metrics if available, otherwise fallback to training log metrics
    metrics = eval_metrics if eval_metrics is not None else training_metrics

    # 5. Compare to champion
    champion = load_champion()
    if train_result["status"] == "crashed":
        status = "crash"
        decision = "discard"
    elif early_exit:
        status = "discard"
        decision = "discard"
    elif is_better_than_champion(metrics, champion):
        status = "keep"
        decision = "keep"
        # Update champion
        champion_data = {
            "experiment_id": experiment_id,
            "description": description,
            "config_snapshot": config,
            "checkpoint_path": train_result["checkpoint_path"],
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "status": "champion",
        }
        if per_level_results:
            champion_data["per_level_metrics"] = per_level_results
        save_champion(champion_data)
    else:
        status = "discard"
        decision = "discard"

    # 6. Log result (include both training and eval metrics)
    combined_metrics = dict(training_metrics)
    if eval_metrics:
        combined_metrics["eval"] = eval_metrics
    if per_level_results:
        combined_metrics["per_level"] = per_level_results
    log_result(experiment_id, config, combined_metrics, status, description)

    # 6. If discard, restore reward file from backup
    if decision == "discard":
        from config_patcher import restore_reward_file
        restore_reward_file()
        print(f"  DISCARD: Restored reward file to champion config")
    else:
        print(f"  KEEP: New champion! theta={metrics.get('mean_theta_deg') or metrics.get('final_theta_deg', '?')}°")

    return {
        "experiment_id": experiment_id,
        "status": status,
        "metrics": metrics,
        "checkpoint_path": train_result["checkpoint_path"],
        "wall_time": train_result["wall_time"],
    }


# ======================== Manual Mode ========================

def run_manual_mode(budget: int):
    """Run in manual mode: human edits reward_config.json, then press enter to train."""
    print("\n" + "=" * 60)
    print("RL AUTOTUNER — Manual Mode")
    print("=" * 60)
    print(f"Config file: {REWARD_CONFIG_PATH}")
    print(f"Training budget: {budget:.0e} steps")
    print(f"Champion: {CHAMPION_META_PATH}")
    print()

    while True:
        print("\n--- Ready for next experiment ---")
        print("Edit reward_config.json, then press ENTER to train.")
        print("Type 'q' to quit, 'status' to see champion, 'history' to see results.")
        user_input = input("> ").strip()

        if user_input.lower() == "q":
            break
        elif user_input.lower() == "status":
            champion = load_champion()
            print(json.dumps(champion, indent=2))
            continue
        elif user_input.lower() == "history":
            history = load_results_history()
            for r in history[-10:]:
                print(f"  #{r['experiment_id']}: {r['status']} | "
                      f"theta={r['metrics'].get('final_theta_deg', '?')} | "
                      f"{r['description']}")
            continue

        # Load config and run
        config = load_config()
        desc = input("Brief description of changes: ").strip() or "manual experiment"
        result = run_experiment(config, budget, desc)
        print(f"\nResult: {result['status']} (experiment #{result['experiment_id']})")


# ======================== Phase 2b Detection & Helpers ========================

PHASE2B_CONSECUTIVE_DISCARDS = 3  # trigger Phase 2b after N consecutive discards

REWARD_FILE = PLANAX_DIR / "envs" / "reward_functions" / "quat_baseline_reward.py"
PROJECT_ROOT = AUTOTUNER_DIR.parent


def _check_phase2b_trigger() -> bool:
    """Check if last N non-crash results are all 'discard' → enter Phase 2b."""
    history = load_results_history()
    if not history:
        return False
    # Filter out crash entries (they don't count as valid attempts)
    valid = [r for r in history if r.get("status") in ("keep", "discard")]
    if len(valid) < PHASE2B_CONSECUTIVE_DISCARDS:
        return False
    last_n = valid[-PHASE2B_CONSECUTIVE_DISCARDS:]
    return all(r["status"] == "discard" for r in last_n)


def _git_commit(message: str) -> bool:
    """Git add all and commit. Returns True if commit succeeded."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(PROJECT_ROOT), check=True,
                       capture_output=True, text=True)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  GIT: committed — {message}")
            return True
        # Nothing to commit is OK
        if "nothing to commit" in result.stdout:
            print(f"  GIT: nothing to commit")
            return True
        print(f"  GIT: commit failed — {result.stderr}")
        return False
    except Exception as e:
        print(f"  GIT: error — {e}")
        return False


def _git_reset_hard() -> bool:
    """Git reset --hard HEAD~1 to undo last commit."""
    try:
        result = subprocess.run(
            ["git", "reset", "--hard", "HEAD~1"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  GIT: reset to HEAD~1")
            return True
        print(f"  GIT: reset failed — {result.stderr}")
        return False
    except Exception as e:
        print(f"  GIT: error — {e}")
        return False


def _read_reward_code() -> str:
    """Read current reward file content."""
    return REWARD_FILE.read_text(encoding="utf-8")


def _add_reward_reflection(parts: list, history: list):
    """Add Reward Reflection diagnostic section to prompt.

    Extracts att_r, speed_r diagnostics from recent experiment results
    and formats them as actionable diagnostic info for Claude.
    """
    # Find recent experiments with per-level diagnostic data
    recent_with_diag = []
    for r in reversed(history[-10:]):
        pl = r.get("metrics", {}).get("per_level", {})
        if pl:
            has_diag = any(
                pl.get(str(lv), {}).get("diag_att_r_mean") is not None
                for lv in [0, 1, 2, 3, 5]
            )
            if has_diag:
                recent_with_diag.append(r)
        if len(recent_with_diag) >= 3:
            break

    if not recent_with_diag:
        return

    parts.append("### Reward Reflection (诊断信息)")
    parts.append("")
    parts.append("以下是近期实验中 reward 各分量的统计。用这些信息诊断 reward 函数的问题：")
    parts.append("**注意：这些信息仅供你理解 reward 函数的行为。keep/discard 判据仍然是 theta_deg < champion。**")
    parts.append("")

    for r in recent_with_diag:
        exp_id = r.get("experiment_id", "?")
        status = r.get("status", "?")
        parts.append(f"**实验 #{exp_id} ({status})**: {r.get('description', '')}")
        pl = r.get("metrics", {}).get("per_level", {})
        for lv in sorted(pl.keys(), key=lambda x: int(x)):
            lv_data = pl[lv]
            att_r = lv_data.get("diag_att_r_mean")
            att_std = lv_data.get("diag_att_r_std")
            spd_r = lv_data.get("diag_speed_r_mean")
            spd_std = lv_data.get("diag_speed_r_std")
            theta = lv_data.get("mean_theta_deg")
            if att_r is not None:
                line = f"  L{lv}: att_r={att_r:.3f}"
                if att_std is not None:
                    line += f"(±{att_std:.3f})"
                if spd_r is not None:
                    line += f", speed_r={spd_r:.3f}"
                    if spd_std is not None:
                        line += f"(±{spd_std:.3f})"
                if theta is not None:
                    line += f", theta={theta:.1f}°"
                # Add interpretation
                if att_r is not None and att_r < 0.05:
                    line += " ← att_r≈0: Gaussian 完全饱和，需更大 sigma 或不同函数"
                elif att_r is not None and att_r < 0.2:
                    line += " ← att_r 较低：姿态跟踪信号较弱"
                parts.append(line)
        parts.append("")


def _build_claude_prompt_phase2b(config: dict, champion: dict, history: list,
                                  reward_code: str, iteration: int) -> str:
    """Build prompt for Phase 2b: Claude modifies reward function code."""
    parts = [
        f"## Phase 2b — Iteration {iteration}",
        "",
        "Phase 2a 参数调优已饱和（连续 3 次 discard）。现在你可以修改 reward 函数逻辑。",
        "",
        f"**训练预算：{DEFAULT_BUDGET:.0e} 步（fine-tune from champion checkpoint）。**",
        f"Champion 训练了 ~1.35e9 步。你的 reward 函数必须让 agent 在 {DEFAULT_BUDGET:.0e} 步内收敛到各 level 均优于 champion。",
        "",
        "### 当前 reward 文件完整代码",
        "```python",
        reward_code,
        "```",
        "",
        "### 当前 Champion",
        f"theta_deg = {champion.get('metrics', {}).get('mean_theta_deg', '?')}°",
        f"delta_vt = {champion.get('metrics', {}).get('mean_delta_vt', '?')}",
        f"config = {json.dumps(champion.get('config_snapshot', config), indent=2)}",
        "",
    ]

    # Champion per-level data
    champ_pl = champion.get("per_level_metrics", {})
    if champ_pl:
        parts.append("### Champion 分 level 表现")
        for lv in sorted(champ_pl.keys(), key=lambda x: int(x)):
            t = champ_pl[lv].get("mean_theta_deg", "?")
            parts.append(f"  Level {lv}: theta={t}°")
        parts.append("")

    if history:
        parts.append("### 完整实验历史（含 per-level 结果）")
        parts.append("**重要：不要重复之前已经尝试过的方案！** 仔细看每个 discard/crash 实验的描述，避免生成相同或相似的 reward 代码。")
        parts.append("如果之前的方案是 'adaptive theta_scale 25/50/75°' 且 discard，你必须尝试不同的方法（比如改 exponent、加 progress reward、改 weight 策略等），而不是只微调数值。")
        parts.append("")
        for r in history:
            theta = r.get("metrics", {}).get("mean_theta_deg") or r.get("metrics", {}).get("final_theta_deg", "?")
            dvt = r.get("metrics", {}).get("mean_delta_vt") or r.get("metrics", {}).get("final_delta_vt", "?")
            line = f"  #{r['experiment_id']}: {r['status']} | overall_theta={theta}° | delta_vt={dvt} | {r.get('description', '')}"
            # Append per-level data if available
            pl = r.get("metrics", {}).get("per_level", {})
            if pl:
                pl_parts = []
                for lv in sorted(pl.keys(), key=lambda x: int(x)):
                    lt = pl[lv].get("mean_theta_deg")
                    if lt is not None:
                        pl_parts.append(f"L{lv}={lt:.1f}°")
                if pl_parts:
                    line += f" | per_level: [{', '.join(pl_parts)}]"
            parts.append(line)
        parts.append("")

    # Reward Reflection: diagnostic info from recent experiments
    _add_reward_reflection(parts, history)

    parts.extend([
        "### 关键发现：Reward-Curriculum 耦合问题",
        "",
        "从实验历史中发现一个核心问题：**同一个 reward 函数在低 level 表现好，在高 level 表现差。**",
        "例如实验 #18：Level 0 赢了 champion（36.6° < 38.2°），但 Level 3 大幅落后（89.2° >> 68.9°）。",
        "原因：固定 sigma=30° 的 Gaussian 在 theta>60° 时 reward≈0，梯度消失，agent 无法学习大角度跟踪。",
        "",
        "**你必须解决这个问题。** 方法：在 reward 函数中根据 curriculum_level 使用不同的参数。",
        "",
        "### 如何读取 curriculum_level",
        "在 quat_baseline_reward_fn(state, params, agent_id, reward_scale) 中：",
        "```",
        "curriculum_level = state.curriculum_level[agent_id]  # int, 0-5",
        "```",
        "然后用 jnp.where 做条件分支（不能用 Python if/else）：",
        "```",
        "theta_scale = jnp.where(curriculum_level <= 1, 30.0,",
        "              jnp.where(curriculum_level <= 3, 50.0, 75.0))",
        "```",
        "这样低 level 用小 sigma（精确跟踪），高 level 用大 sigma（保证梯度信号）。",
        "",
        "### Phase 2b 修改指导",
        "最高优先级方向：",
        "1. **Level-adaptive reward**：根据 curriculum_level 切换 theta_scale_deg（最重要！）",
        "2. 多尺度 Gaussian：添加 coarse + fine 两个尺度的 Gaussian 加权和",
        "3. Progress reward：奖励 θ 减小的方向",
        "4. Settled bonus：θ < 5° 时额外奖励",
        "",
        "### 约束",
        "- 函数签名不变：(state, params, agent_id, reward_scale) → float",
        "- reward 返回值范围 [0, 1]（或合理范围，不要爆到几百）",
        "- 不要删除 jnp.nan_to_num 保护",
        "- 不要删除 is_alive/is_locked mask",
        "- 不要用 Python if/else 判断 JAX traced values，用 jnp.where",
        "- 保持所有 quaternion helper 函数不变",
        "- REWARD_CONFIG 字典中可以添加新参数",
        "",
        "### 返回格式",
        "返回你的分析推理，然后：",
        "",
        "1. 一个 ```python 代码块，包含完整的新 REWARD_CONFIG 字典 + quat_baseline_reward_fn 函数",
        "   （从 `# ---- REWARD_CONFIG` 注释开始，到文件末尾）",
        "",
        "2. 一个 ```json 代码块，包含对应的 reward_config.json（至少包含 speed_error_scale, w_att, w_speed，可加新 key）",
        "",
        "3. DESCRIPTION: <一行描述你改了什么和为什么>",
    ])

    return "\n".join(parts)


def _parse_claude_response_phase2b(reply: str) -> tuple:
    """Parse Phase 2b response: extract Python code, JSON config, and description.

    Returns (reward_code, config, description) or (None, None, "") on failure.
    """
    # Extract Python code block
    py_match = re.search(r"```python\s*\n(.*?)\n```", reply, re.DOTALL)
    if not py_match:
        print("  Phase 2b parse: no python code block found")
        return None, None, ""
    reward_code = py_match.group(1)

    # Validate reward code has required elements
    if "REWARD_CONFIG" not in reward_code:
        print("  Phase 2b parse: REWARD_CONFIG not found in code")
        return None, None, ""
    if "quat_baseline_reward_fn" not in reward_code:
        print("  Phase 2b parse: quat_baseline_reward_fn not found in code")
        return None, None, ""

    # Extract JSON config block
    json_match = re.search(r"```json\s*\n(.*?)\n```", reply, re.DOTALL)
    if not json_match:
        print("  Phase 2b parse: no json code block found")
        return None, None, ""

    try:
        config = json.loads(json_match.group(1))
    except json.JSONDecodeError as e:
        print(f"  Phase 2b parse: JSON decode error — {e}")
        return None, None, ""

    config = {k: v for k, v in config.items() if not k.startswith("_")}

    # Extract description
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", reply)
    description = desc_match.group(1).strip() if desc_match else "Phase 2b: reward logic modification"

    return reward_code, config, description


def _parse_claude_response_phase2b_k4(reply: str) -> list:
    """Parse K=4 Phase 2b candidates from Claude response.

    Returns list of (reward_code, config, description) tuples.
    May return fewer than 4 if some candidates fail to parse.
    """
    # Split by CANDIDATE_N markers
    candidate_sections = re.split(r"CANDIDATE_\d+[:\s]*", reply)
    # First section is preamble (analysis), skip it
    if len(candidate_sections) > 1:
        candidate_sections = candidate_sections[1:]
    else:
        # No CANDIDATE markers — try to parse as single candidate (backward compat)
        result = _parse_claude_response_phase2b(reply)
        if result[0] is not None:
            return [result]
        return []

    candidates = []
    for i, section in enumerate(candidate_sections):
        code, config, desc = _parse_claude_response_phase2b(section)
        if code is not None:
            candidates.append((code, config, f"[K{i+1}] {desc}"))
        else:
            print(f"  Candidate {i+1}: parse failed, skipping")

    return candidates


def _parse_claude_response_k4(reply: str) -> list:
    """Parse K=4 Phase 2a config candidates from Claude response.

    Returns list of (config, description) tuples.
    """
    candidate_sections = re.split(r"CANDIDATE_\d+[:\s]*", reply)
    if len(candidate_sections) > 1:
        candidate_sections = candidate_sections[1:]
    else:
        # No CANDIDATE markers — try single candidate (backward compat)
        result = _parse_claude_response(reply)
        if result[0] is not None:
            return [result]
        return []

    candidates = []
    for i, section in enumerate(candidate_sections):
        config, desc = _parse_claude_response(section)
        if config is not None:
            candidates.append((config, f"[K{i+1}] {desc}"))
        else:
            print(f"  Candidate {i+1}: parse failed, skipping")

    return candidates


def _validate_reward_code_jax(reward_code: str) -> bool:
    """Validate reward code by JAX-tracing in a subprocess.

    Temporarily writes the code to the reward file, runs validate_reward.py,
    then restores the original code. Returns True if JAX trace succeeds.
    """
    original_code = REWARD_FILE.read_text(encoding="utf-8")

    try:
        # Apply the candidate code
        success = patch_reward_file_code(reward_code)
        if not success:
            print("  JAX validation: failed to patch reward file")
            return False

        # Run validation in subprocess (avoids import cache issues)
        validate_script = AUTOTUNER_DIR / "validate_reward.py"
        result = subprocess.run(
            [sys.executable, str(validate_script)],
            capture_output=True, text=True, timeout=60,
            cwd=str(PLANAX_DIR),
        )

        if result.returncode == 0:
            print(f"  JAX validation: PASS — {result.stdout.strip()}")
            return True
        else:
            print(f"  JAX validation: FAIL — {result.stderr.strip()[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print("  JAX validation: TIMEOUT (60s)")
        return False
    except Exception as e:
        print(f"  JAX validation: ERROR — {e}")
        return False
    finally:
        # Always restore original code
        REWARD_FILE.write_text(original_code, encoding="utf-8")


# ======================== Auto Mode (Claude API) ========================

def run_auto_mode(budget: int, api_key: str, api_base: str, max_iterations: int = 20):
    """Run in auto mode: Claude suggests config changes (Phase 2a) or code changes (Phase 2b)."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: 'anthropic' package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = Anthropic(api_key=api_key, base_url=api_base)

    # Load program.md for system prompt
    program_path = AUTOTUNER_DIR / "program.md"
    if program_path.exists():
        system_prompt = program_path.read_text(encoding="utf-8")
    else:
        system_prompt = "You are an RL reward tuning expert. Suggest changes to reward_config.json."

    print("\n" + "=" * 60)
    print("RL AUTOTUNER — Auto Mode (Claude)")
    print(f"Budget: {budget:.0e} steps/experiment, max {max_iterations} iterations")
    print("=" * 60)

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*40} Iteration {iteration}/{max_iterations} {'='*40}")

        # Build context
        current_config = load_config()
        champion = load_champion()
        history = load_results_history()

        # Check Phase 2b trigger
        phase2b = _check_phase2b_trigger()
        if phase2b:
            print(">>> PHASE 2b ACTIVATED (3 consecutive discards) — modifying reward code <<<")

        # Build prompt based on phase
        if phase2b:
            reward_code = _read_reward_code()
            user_message = _build_claude_prompt_phase2b(
                current_config, champion, history, reward_code, iteration
            )
            max_tokens = 16384
        else:
            user_message = _build_claude_prompt(current_config, champion, history, iteration)
            max_tokens = 8192

        # Call Claude API with retry logic
        phase_label = "Phase 2b" if phase2b else "Phase 2a"
        print(f"Calling Claude for {phase_label} suggestion...")
        reply = None
        for attempt in range(3):
            try:
                reply_parts = []
                with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=max_tokens,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                ) as stream:
                    for text in stream.text_stream:
                        reply_parts.append(text)
                reply = "".join(reply_parts)
                print(f"  Claude responded ({len(reply)} chars)")
                # Retry if response is too short (likely API error)
                if len(reply) < 100:
                    print(f"  Response too short (<100 chars), retry {attempt+1}/3...")
                    print(f"  Actual response: {repr(reply[:200])}")  # Log first 200 chars for debugging
                    time.sleep(15 * (attempt + 1))
                    reply = None
                    continue
                break  # success
            except Exception as e:
                print(f"  Claude API error (attempt {attempt+1}/3): {e}")
                time.sleep(15 * (attempt + 1))
                reply = None
        if reply is None:
            print(f"Claude API failed after 3 attempts, skipping iteration")
            log_result(get_next_experiment_id(), current_config, {},
                       "crash", "Claude API failed after 3 retries")
            time.sleep(10)
            continue

        # Parse response based on phase
        if phase2b:
            # Phase 2b: single candidate + JAX validation
            new_reward_code, new_config, description = _parse_claude_response_phase2b(reply)
            if new_reward_code is None:
                print(f"Failed to parse Phase 2b response, skipping iteration")
                log_result(get_next_experiment_id(), current_config, {},
                           "crash", f"Failed to parse Phase 2b response")
                continue

            print(f"Claude suggests (Phase 2b): {description}")

            # JAX validation before committing to training
            print("  Validating JAX compatibility...")
            if not _validate_reward_code_jax(new_reward_code):
                print("  JAX validation FAILED, skipping iteration")
                log_result(get_next_experiment_id(), new_config, {},
                           "crash", f"Phase 2b JAX validation failed: {description}")
                continue

            # Git commit current state before modifying code
            _git_commit(f"pre-phase2b: before experiment #{get_next_experiment_id()}")

            # Apply code changes
            backup_reward_file()
            success = patch_reward_file_code(new_reward_code)
            if not success:
                print("Failed to patch reward code, rolling back")
                _git_reset_hard()
                log_result(get_next_experiment_id(), new_config, {},
                           "crash", f"Phase 2b code patch failed")
                continue

            # Write updated config
            with open(REWARD_CONFIG_PATH, "w") as f:
                json.dump(new_config, f, indent=2)

            # Git commit the changes
            _git_commit(f"phase2b experiment: {description}")

            # Run experiment
            result = run_experiment(new_config, budget, f"[Phase 2b] {description}")
            print(f"Result: {result['status']} (experiment #{result['experiment_id']})")

            # If discard, rollback the code change via git
            if result["status"] in ("discard", "crash"):
                print("  Phase 2b discard — rolling back code changes via git reset")
                _git_reset_hard()

        else:
            # Phase 2a: single config candidate
            new_config, description = _parse_claude_response(reply)
            if new_config is None:
                print(f"Failed to parse Claude response, skipping iteration")
                log_result(get_next_experiment_id(), current_config, {},
                           "crash", f"Failed to parse Claude response")
                continue

            print(f"Claude suggests (Phase 2a): {description}")

            # Write new config
            with open(REWARD_CONFIG_PATH, "w") as f:
                json.dump(new_config, f, indent=2)

            # Run experiment
            result = run_experiment(new_config, budget, description)
            print(f"Result: {result['status']} (experiment #{result['experiment_id']})")

    print(f"\nAuto mode completed after {max_iterations} iterations.")
    champion = load_champion()
    print(f"Final champion: experiment #{champion.get('experiment_id', '?')}")
    print(f"  theta={champion.get('metrics', {}).get('mean_theta_deg', '?')}°")


def _build_claude_prompt(config: dict, champion: dict, history: list, iteration: int) -> str:
    """Build the prompt for Claude to suggest next config (Phase 2a)."""
    parts = [
        f"## Phase 2a — Iteration {iteration}",
        "",
        "### Current Champion Config",
        "```json",
        json.dumps(champion.get("config_snapshot", config), indent=2),
        "```",
        "",
        "### Champion Eval Metrics",
        json.dumps(champion.get("metrics", {}), indent=2),
        "",
    ]

    if history:
        parts.append("### Complete Experiment History")
        for r in history:
            metrics = r.get("metrics", {})
            # Prefer eval metrics over training log metrics
            eval_m = metrics.get("eval", {})
            theta = eval_m.get("mean_theta_deg") or metrics.get("final_theta_deg", "?")
            dvt = eval_m.get("mean_delta_vt") or metrics.get("final_delta_vt", "?")
            crash = eval_m.get("mean_crash_rate", "?")
            on_target = eval_m.get("mean_on_target_rate", "?")
            cfg = r.get("config_snapshot", {})
            parts.append(
                f"  #{r['experiment_id']}: {r['status']} | "
                f"theta={theta}° | delta_vt={dvt} | crash={crash} | on_target={on_target} | "
                f"config={json.dumps(cfg)} | "
                f"{r.get('description', '')}"
            )
        parts.append("")

    # Reward Reflection: diagnostic info from recent experiments
    _add_reward_reflection(parts, history)

    parts.extend([
        "### 分析要求",
        "在提出修改前，你必须：",
        "1. 整理出参数变化趋势表：每个参数往哪个方向调、结果 keep 还是 discard",
        "2. 识别哪些方向有效（keep 了），哪些无效（discard 了），是否有参数振荡",
        "3. 基于以上分析，形成明确的假设，再提出修改",
        "4. 禁止重复已经被 discard 的相同修改方向",
        "5. 每次最多改 2 个参数",
        "",
        "### 可调参数",
        "- theta_scale_deg: 姿态 Gaussian 宽度（当前 champion 有课程学习，高 level 会遇到 theta>90°）",
        "- speed_error_scale: 速度 Gaussian 宽度",
        "- w_att: 姿态权重（w_att + w_speed 必须 ≈ 1.0）",
        "- w_speed: 速度权重",
        "",
        "### Task",
        "Suggest the next reward_config.json. Goal: minimize mean_theta_deg.",
        "",
        "Return:",
        "1. 你的分析推理过程（趋势、假设）",
        "2. 一个 JSON code block with the complete new config:",
        "```json",
        "{ ... }",
        "```",
        "3. DESCRIPTION: <一行描述你改了什么和为什么>",
    ])

    return "\n".join(parts)


def _parse_claude_response(reply: str) -> tuple:
    """Extract JSON config and description from Claude's response."""
    # Find JSON block
    json_match = re.search(r"```json\s*\n(.*?)\n```", reply, re.DOTALL)
    if not json_match:
        return None, ""

    try:
        config = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None, ""

    # Remove comment fields
    config = {k: v for k, v in config.items() if not k.startswith("_")}

    # Find description
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", reply)
    description = desc_match.group(1).strip() if desc_match else "auto-suggested config"

    return config, description


# ======================== Main ========================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RL Autotuner Experiment Runner")
    parser.add_argument("--mode", choices=["manual", "auto", "manual-auto", "dry-run", "init-baseline"], default="manual")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="Training budget (timesteps)")
    parser.add_argument("--description", type=str, default="", help="Experiment description (for manual-auto mode)")
    parser.add_argument("--api-key", type=str, default="sk-bf907cb732da4390f4755aac9a25e00ca58abcf5e65cb830a0a64d162a788f68",
                        help="Claude API key (auto mode)")
    parser.add_argument("--api-base", type=str, default="https://ai.tokencloud.ai",
                        help="Claude API base URL")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max auto iterations")
    args = parser.parse_args()

    if args.mode == "dry-run":
        config = load_config()
        warnings = validate_config(config)
        print("Config loaded:")
        print(json.dumps(config, indent=2))
        if warnings:
            print(f"\nWarnings: {warnings}")
        else:
            print("\nConfig valid, ready to train.")
        # Also test patching
        print("\nDry-run patch output:")
        from config_patcher import config_to_python
        print(config_to_python(config))

    elif args.mode == "manual":
        run_manual_mode(int(args.budget))

    elif args.mode == "auto":
        if not args.api_key:
            # Try environment variable
            api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("ERROR: --api-key required for auto mode (or set CLAUDE_API_KEY env var)")
                sys.exit(1)
        else:
            api_key = args.api_key
        run_auto_mode(int(args.budget), api_key, args.api_base, args.max_iterations)

    elif args.mode == "manual-auto":
        config = load_config()
        desc = args.description or "manual-auto experiment"
        result = run_experiment(config, int(args.budget), desc)
        print(f"\nResult: {result['status']} (experiment #{result['experiment_id']})")

    elif args.mode == "init-baseline":
        from evaluator import evaluate_random_policy, EVAL_CONFIG
        print("Evaluating random policy as initial baseline...")
        result = evaluate_random_policy(dict(EVAL_CONFIG))
        metrics = result["aggregate"]
        save_champion({
            "experiment_id": 0,
            "description": "Random policy baseline (untrained network). Reference: quaternion baseline trained in heading_pitch_V env achieved ~20deg theta. This full_domain task is harder.",
            "config_snapshot": load_config(),
            "checkpoint_path": None,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "status": "baseline",
        })
        print(f"\nInitial baseline established:")
        print(f"  theta_deg = {metrics.get('mean_theta_deg', '?')}deg")
        print(f"  delta_vt  = {metrics.get('mean_delta_vt', '?')}")
        print(f"  crash_rate = {metrics.get('mean_crash_rate', '?')}")
        print(f"  on_target_rate = {metrics.get('mean_on_target_rate', '?')}")
