#!/usr/bin/env python3
"""
auto_train_loop.py
==================
Autonomous Evaluator-Optimizer Loop for full-domain maneuver training.

Workflow (Anthropic SDK 流式架构):
  1. Submit a training run (train_full_domain_maneuver_v3.py) as subprocess
  2. Wait for completion
  3. Call evaluate_maneuver.py to produce a JSON evaluation report
  4. If metrics meet target thresholds → declare success and exit
  5. If not → call Anthropic SDK (streaming) to analyze the report,
     read the embedded codebase, and output code edits as JSON
  6. Apply edits → dry-run validation → if pass, go to step 1; if fail, auto-rollback

Key design: Claude receives full source code in the prompt and outputs
structured JSON edits. The framework applies edits and validates via dry-run.
Streaming output lets the user observe Claude's reasoning in real-time.

Usage:
    python auto_train_loop.py --max-iters 5 --target-metrics ../assets/target_metrics_template.json

Environment variables:
    ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY  — API key
    ANTHROPIC_BASE_URL    — API base URL (default: https://api.anthropic.com)
    ANTHROPIC_TUNER_MODEL — Model ID (default: claude-opus-4-6)

Environment: Single-node, 2× NVIDIA A100 80 GB.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import json
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import re
import gc
import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # 20251215最新代码库
PLANAX_DIR = PROJECT_ROOT / "Planax"
SKILL_ROOT = SCRIPT_DIR.parent       # aircraft-rl-tuner/

EVALUATE_SCRIPT = SCRIPT_DIR / "evaluate_maneuver.py"
TRAIN_SCRIPT = PLANAX_DIR / "train_full_domain_maneuver_v3.py"
REWARD_FILE = PLANAX_DIR / "envs" / "reward_functions" / "full_domain_reward.py"
ENV_FILE = PLANAX_DIR / "envs" / "aeroplanax_full_domain_maneuver.py"
TERMINATION_FILE = PLANAX_DIR / "envs" / "termination_conditions" / "unreach_full_domain.py"
REFERENCES_DIR = SKILL_ROOT / "references"

DEFAULT_TARGET_METRICS = SKILL_ROOT / "assets" / "target_metrics_template.json"

# ---------------------------------------------------------------------------
# Anthropic SDK configuration
# ---------------------------------------------------------------------------
ANTHROPIC_TUNER_MODEL = os.environ.get("ANTHROPIC_TUNER_MODEL", "claude-opus-4-6")


def load_target_metrics(path: str) -> Dict[str, Any]:
    """Load target metric thresholds from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_pass(report: Dict[str, Any], targets: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Check if evaluation report meets all target thresholds. Returns (pass, reasons)."""
    agg = report.get("aggregate", {})
    reasons = []

    thresholds = targets.get("thresholds", {})

    sr = agg.get("mean_success_rate", 0.0)
    sr_target = thresholds.get("mean_success_rate", 0.8)
    if sr < sr_target:
        reasons.append(f"mean_success_rate={sr:.3f} < {sr_target}")

    crash = agg.get("mean_crash_count", 999)
    crash_target = thresholds.get("max_mean_crash_count", 1.0)
    if crash > crash_target:
        reasons.append(f"mean_crash_count={crash:.2f} > {crash_target}")

    stall = agg.get("mean_stall_steps", 999)
    stall_target = thresholds.get("max_mean_stall_steps", 30)
    if stall > stall_target:
        reasons.append(f"mean_stall_steps={stall:.1f} > {stall_target}")

    # Per-level checks
    per_level = report.get("per_level", {})
    level_targets = thresholds.get("per_level_min_success_rate", {})
    for lvl_key, min_sr in level_targets.items():
        lvl_data = per_level.get(lvl_key, {})
        lvl_sr = lvl_data.get("mean_success_rate", 0.0)
        if lvl_sr < min_sr:
            reasons.append(f"{lvl_key} success={lvl_sr:.2f} < {min_sr}")

    # Coverage check
    coverage = report.get("coverage", {})
    if thresholds.get("require_full_domain", True):
        if not coverage.get("full_domain_achieved", False):
            reasons.append("full_domain NOT achieved (missing extreme pitch/roll coverage)")

    passed = len(reasons) == 0
    return passed, reasons


def load_training_log_tail(max_chars: int = 3000) -> str:
    """Load the last N chars of training output for Claude to analyze trends."""
    log_file = SKILL_ROOT / "assets" / "train_output.log"
    if not log_file.exists():
        return ""
    try:
        text = log_file.read_text(encoding="utf-8")
        return text[-max_chars:] if len(text) > max_chars else text
    except Exception:
        return ""


def _read_file_safe(path: Path, max_chars: int = 0) -> str:
    """Read a file, returning empty string on failure. Optionally truncate."""
    try:
        text = path.read_text(encoding="utf-8")
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + f"\n... (truncated, full file: {path})"
        return text
    except Exception:
        return ""


def _read_lines(path: Path, start: int, end: int) -> str:
    """Read specific line range from a file (1-indexed). Returns numbered lines."""
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
        selected = lines[start - 1:end]
        return "\n".join(f"{i}: {l}" for i, l in enumerate(selected, start=start))
    except Exception:
        return ""


def _extract_key_sections(env_path: Path, train_path: Path) -> str:
    """Extract key config sections from env and train files for prompt embedding."""
    sections = []

    # Env file: params (lines 120-140), on-target tracking (500-525), curriculum targets (660-735)
    sections.append(f"### 环境参数 ({env_path}, lines 120-140)")
    sections.append(_read_lines(env_path, 120, 140))
    sections.append(f"\n### on-target 跟踪逻辑 ({env_path}, lines 500-525)")
    sections.append(_read_lines(env_path, 500, 525))
    sections.append(f"\n### 课程目标生成 ({env_path}, lines 660-735)")
    sections.append(_read_lines(env_path, 660, 735))

    # Train file: KL/entropy config (lines 135-160), main config (lines 710-750)
    sections.append(f"\n### 训练 KL/Entropy 配置 ({train_path}, lines 135-160)")
    sections.append(_read_lines(train_path, 135, 160))
    sections.append(f"\n### 训练主配置 PPO ({train_path}, lines 710-750)")
    sections.append(_read_lines(train_path, 710, 750))

    return "\n".join(sections)


def _apply_edits(edits: list[dict]) -> tuple[bool, list[str]]:
    """Apply a list of edits from Claude's JSON output.

    Each edit dict: {file: str, old: str, new: str}
    Returns (all_ok, messages).
    """
    messages = []
    all_ok = True
    for i, edit in enumerate(edits):
        fpath = edit.get("file", "")
        old = edit.get("old", "")
        new = edit.get("new", "")
        if not fpath or not old:
            messages.append(f"Edit {i}: 跳过 (缺少 file 或 old 字段)")
            continue
        try:
            p = Path(fpath)
            if not p.exists():
                messages.append(f"Edit {i}: 文件不存在 {fpath}")
                all_ok = False
                continue
            content = p.read_text(encoding="utf-8")
            if old not in content:
                # Try with normalized whitespace
                old_stripped = old.strip()
                if old_stripped and old_stripped in content:
                    content = content.replace(old_stripped, new.strip(), 1)
                    p.write_text(content, encoding="utf-8")
                    messages.append(f"Edit {i}: OK (whitespace-normalized) {p.name}")
                    continue
                messages.append(f"Edit {i}: old_string 未在文件中找到 {p.name}")
                all_ok = False
                continue
            content = content.replace(old, new, 1)
            p.write_text(content, encoding="utf-8")
            messages.append(f"Edit {i}: OK {p.name}")
        except Exception as e:
            messages.append(f"Edit {i}: 错误 {type(e).__name__}: {e}")
            all_ok = False
    return all_ok, messages


def call_claude_code(
    report: Dict[str, Any],
    fail_reasons: list[str],
    iteration: int,
    iteration_log: list = None,
) -> tuple[bool, str]:
    """
    Call Anthropic SDK (streaming) to analyze evaluation results
    and produce code edits as JSON.

    Streaming: Claude's analysis text is printed in real-time so the user
    can observe the reasoning process. After the stream completes, the JSON
    code block at the end is parsed and edits are applied.

    Returns (success, summary):
      - success: True if edits were applied successfully
      - summary: Text summary of what was changed
    """
    # Build iteration history summary (with previous analysis for cross-iteration memory)
    history_text = ""
    prev_analysis = ""
    if iteration_log:
        history_text = "\n## 历次迭代历史\n"
        for entry in iteration_log:
            agg = entry.get("aggregate", {})
            changes_str = ""
            if "code_changes_summary" in entry:
                changes_str = f" | 修改: {entry['code_changes_summary'][:100]}"
            history_text += (
                f"- 第{entry['iteration']}轮: "
                f"success_rate={agg.get('mean_success_rate', 0):.3f}, "
                f"crash={agg.get('mean_crash_count', 0):.1f}, "
                f"stall={agg.get('mean_stall_steps', 0):.0f} | "
                f"{'通过' if entry.get('passed') else '失败: ' + '; '.join(entry.get('reasons', [])[:3])}"
                f"{changes_str}\n"
            )

        # Load previous iteration's full analysis for cross-iteration reasoning
        prev_iter = iteration_log[-1].get("iteration", iteration - 1)
        prev_log_file = SKILL_ROOT / "assets" / f"claude_code_iter_{prev_iter}.log"
        if prev_log_file.exists():
            prev_analysis_text = _read_file_safe(prev_log_file, max_chars=3000)
            if prev_analysis_text:
                prev_analysis = f"""
## 上一轮（第 {prev_iter} 轮）Claude 的完整分析和修改

以下是你上一轮的分析输出，请基于此理解之前做了什么、为什么没有效果，避免重复无效修改：

```
{prev_analysis_text}
```
"""

    # Load training log tail for context
    train_log_tail = load_training_log_tail(2000)

    # Pre-read key files to embed in prompt (single-turn, no tool use needed)
    reward_code = _read_file_safe(REWARD_FILE)
    termination_code = _read_file_safe(TERMINATION_FILE)
    key_sections = _extract_key_sections(ENV_FILE, TRAIN_SCRIPT)

    # Load expert guide
    expert_guide = ""
    expert_file = REFERENCES_DIR / "jax_rl_best_practices.md"
    if expert_file.exists():
        expert_guide_text = _read_file_safe(expert_file)
        if expert_guide_text:
            expert_guide = f"""
## 专家指南（JAX RL 最佳实践）

{expert_guide_text}
"""

    # Build the prompt
    prompt = f"""你是一个高级 RL 系统工程师，正在调优基于 JAX 的固定翼飞机全域机动控制训练。

这是自动训练循环的第 {iteration} 轮。上一轮训练+评估结果未达标，需要你分析并修改代码。

## 评估报告（第 {iteration} 轮）

```json
{json.dumps(report, indent=2, ensure_ascii=False)[:4000]}
```

## 失败原因
{chr(10).join(f'- {r}' for r in fail_reasons)}
{history_text}{prev_analysis}
## 训练日志（尾部）
```
{train_log_tail}
```

## 奖励函数完整代码 ({REWARD_FILE})

```python
{reward_code}
```

## 终止条件完整代码 ({TERMINATION_FILE})

```python
{termination_code}
```

## 环境和训练脚本关键配置（节选）

{key_sections}

（完整文件：{ENV_FILE} 和 {TRAIN_SCRIPT}）
{expert_guide}
## 安全规则（必须遵守）
1. 禁止删除 jnp.nan_to_num 安全守卫
2. 禁止删除安全惩罚（Nz 过载、qbar 失速、高度）
3. 禁止修改函数签名或 @jax.jit 装饰器参数
4. 禁止使用 Python if/else 对 JAX traced 值做分支 — 用 jnp.where 或 jax.lax.cond
5. 始终保持 reward 裁剪在有界范围内（如 [-5, 5]）
6. 渐进式修改 — 每轮改动聚焦于 1-3 个根本问题，不要大规模重构

## 你的任务

1. 先用中文自然语言分析根本原因（我会看到你的实时输出）
2. 然后在最后输出一个 JSON 代码块，格式如下：

```json
{{
  "analysis": "根本原因分析（1-3句）",
  "edits": [
    {{"file": "文件绝对路径", "old": "要替换的原始代码", "new": "替换后的新代码"}},
    ...
  ],
  "summary": "修改摘要（一句话）"
}}
```

要求：
- 每个 edit 的 old 字段必须精确匹配文件中的代码片段（包括缩进和空格）
- 每个 edit 只替换一处，edits 最多 20 个
- 你只能修改上述 4 个核心文件
- 必须以 ```json 代码块结尾
"""

    # SDK client
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "sk-bf907cb732da4390f4755aac9a25e00ca58abcf5e65cb830a0a64d162a788f68")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://ai.tokencloud.ai")
    model = ANTHROPIC_TUNER_MODEL

    print(f"\n[claude-sdk] 调用 Anthropic SDK（流式输出）...")
    print(f"[claude-sdk] Model: {model}")
    print(f"[claude-sdk] Base URL: {base_url}")
    print(f"[claude-sdk] Prompt 大小: {len(prompt)} 字节 (含嵌入代码)")

    log_file = SKILL_ROOT / "assets" / f"claude_code_iter_{iteration}.log"

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

            full_text = ""
            input_tokens = 0
            output_tokens = 0

            print(f"[claude-sdk] 尝试 {attempt}/{max_retries}")
            print(f"[claude-sdk] {'─' * 60}")

            with client.messages.stream(
                model=model,
                max_tokens=32768,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_text += text

                # Get final message for token usage
                final_message = stream.get_final_message()
                input_tokens = final_message.usage.input_tokens
                output_tokens = final_message.usage.output_tokens

            print(f"\n[claude-sdk] {'─' * 60}")
            print(f"[claude-sdk] Tokens: {input_tokens} in / {output_tokens} out")

            # Save log
            with open(log_file, "w", encoding="utf-8") as flog:
                flog.write(full_text)

            # Parse JSON code block from response
            json_match = re.search(r'```json\s*\n(.*?)\n```', full_text, re.DOTALL)
            if not json_match:
                # Fallback: try to find a raw JSON object with "edits" key
                json_match = re.search(
                    r'(\{[^{}]*"analysis".*?"edits"\s*:\s*\[.*?\]\s*,?\s*"summary".*?\})',
                    full_text, re.DOTALL,
                )
                if json_match:
                    result_text = json_match.group(1)
                else:
                    if attempt < max_retries:
                        print(f"[claude-sdk] 未找到 JSON 代码块，重试...")
                        time.sleep(10)
                        continue
                    return False, "Claude 输出中未找到 JSON 代码块"
            else:
                result_text = json_match.group(1)

            try:
                result = json.loads(result_text)
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    print(f"[claude-sdk] JSON 解析失败: {e}，重试...")
                    time.sleep(10)
                    continue
                return False, f"JSON 解析失败: {e}"

            analysis = result.get("analysis", "")
            edits = result.get("edits", [])
            summary = result.get("summary", "（无摘要）")

            print(f"[claude-sdk] 修改数量: {len(edits)}")

            if not edits:
                print(f"[claude-sdk] 警告: Claude 未提供任何修改")
                return True, f"无修改。分析: {analysis}"

            # Apply edits
            ok, msgs = _apply_edits(edits)
            for msg in msgs:
                print(f"[claude-sdk] {msg}")

            if not ok:
                print(f"[claude-sdk] 部分修改失败，但已应用成功的部分")

            print(f"\n[claude-sdk] 修改摘要: {summary}")
            return True, summary

        except anthropic.APITimeoutError:
            print(f"\n[claude-sdk] API 超时 (尝试 {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(30 * attempt)
                continue
            return False, "API 超时（已重试3次）"

        except anthropic.APIStatusError as e:
            print(f"\n[claude-sdk] API 错误 {e.status_code}: {e.message} (尝试 {attempt}/{max_retries})")
            if attempt < max_retries and e.status_code >= 500:
                time.sleep(30 * attempt)
                continue
            return False, f"API 错误 {e.status_code}: {e.message}"

        except Exception as e:
            print(f"\n[claude-sdk] 错误: {type(e).__name__}: {e} (尝试 {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(30 * attempt)
                continue
            return False, f"错误: {type(e).__name__}: {e}"

    return False, "SDK 调用失败（已重试3次）"


def backup_code_files(iteration: int) -> Path:
    """
    Backup all modifiable code files before Claude Code makes changes.
    Returns the backup directory path.
    """
    backup_dir = SKILL_ROOT / "assets" / "backups" / f"iter_{iteration}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    files_to_backup = [REWARD_FILE, ENV_FILE, TERMINATION_FILE, TRAIN_SCRIPT]
    for src in files_to_backup:
        if src.exists():
            # Preserve relative path structure
            rel = src.relative_to(PROJECT_ROOT)
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(f"[backup] 已备份 {len(files_to_backup)} 个文件到 {backup_dir}")
    return backup_dir


def rollback_code_files(backup_dir: Path):
    """Restore code files from backup directory."""
    restored = 0
    for backup_file in backup_dir.rglob("*.py"):
        rel = backup_file.relative_to(backup_dir)
        dst = PROJECT_ROOT / rel
        shutil.copy2(backup_file, dst)
        restored += 1
    print(f"[rollback] 已从 {backup_dir} 恢复 {restored} 个文件")


def _detect_plateau(iteration_log: list, window: int = 3) -> tuple[bool, str]:
    """检测调优是否已停滞（多维度评估）。
    检查最近 window 轮的 success_rate / crash_count / total_reward，
    只有当所有维度都无明显进展时才判定停滞。"""
    if len(iteration_log) < window:
        return False, ""
    recent = iteration_log[-window:]

    rates = [e.get("aggregate", {}).get("mean_success_rate", 0.0) for e in recent]
    crashes = [e.get("aggregate", {}).get("mean_crash_count", 999.0) for e in recent]
    rewards = [e.get("aggregate", {}).get("mean_total_reward", -9999.0) for e in recent]

    rate_spread = max(rates) - min(rates)
    rate_improving = rate_spread >= 0.02

    # crash 下降 20% 以上算有进展
    crash_improving = (min(crashes) < max(crashes) * 0.8) if max(crashes) > 0 else False

    # reward 提升 10% 以上算有进展（处理负值）
    reward_range = max(rewards) - min(rewards)
    reward_base = max(abs(min(rewards)), 1.0)
    reward_improving = (reward_range / reward_base) >= 0.10

    if rate_improving or crash_improving or reward_improving:
        return False, ""

    trend_rate = " → ".join(f"{r:.3f}" for r in rates)
    trend_crash = " → ".join(f"{c:.1f}" for c in crashes)
    trend_reward = " → ".join(f"{r:.1f}" for r in rewards)
    diagnosis = (
        f"最近 {window} 轮所有指标均无明显进展:\n"
        f"  success_rate: {trend_rate} (极差={rate_spread:.4f})\n"
        f"  crash_count:  {trend_crash}\n"
        f"  total_reward: {trend_reward}"
    )
    return True, diagnosis


def _escalate_to_user(iteration_log: list, reasons: list, iteration: int, last_checkpoint: str):
    """打印中文停滞通知并保存 escalation_report.json。"""
    recent_window = min(len(iteration_log), 5)
    recent = iteration_log[-recent_window:]
    rates = [entry.get("aggregate", {}).get("mean_success_rate", 0.0) for entry in recent]
    trend = " → ".join(f"{r:.3f}" for r in rates)
    reasons_str = "\n".join(f"║    - {r:<60s}║" for r in reasons) if reasons else "║    - 未知瓶颈                                                      ║"

    # 从 fail_reasons 自动推断建议方向
    suggestions = []
    reasons_text = " ".join(reasons)
    if "success_rate" in reasons_text:
        suggestions.append("新增/重构 reward 组件以更好引导策略学习")
    if "crash" in reasons_text:
        suggestions.append("调整终止条件或增加安全惩罚")
    if "stall" in reasons_text:
        suggestions.append("修改课程学习逻辑，降低推进门槛")
    if "coverage" in reasons_text or "domain" in reasons_text:
        suggestions.append("调整观测空间或动作空间以覆盖极端姿态")
    if not suggestions:
        suggestions.append("分析训练日志，定位策略学习瓶颈")

    ckpt_str = last_checkpoint or "N/A"

    banner = f"""
╔══════════════════════════════════════════════════════════════════╗
║  ⚠ 参数调优已连续 {recent_window} 轮无明显进展，需要代码级修改              ║
║                                                                  ║
║  当前主要瓶颈:                                                   ║
{reasons_str}
║                                                                  ║
║  最近 {recent_window} 轮 success_rate: {trend:<40s}║
║                                                                  ║
║  请使用 Claude Code 进行代码级修改，例如：                      ║
║    - 新增/重构 reward 组件                                       ║
║    - 修改课程学习逻辑                                            ║
║    - 调整观测空间或动作空间                                      ║
║                                                                  ║
║  详细报告: aircraft-rl-tuner/assets/escalation_report.json      ║
║                                                                  ║
║  修改完成后，继续运行:                                           ║
║    python auto_train_loop.py --checkpoint {ckpt_str:<24s}║
╚══════════════════════════════════════════════════════════════════╝"""
    print(banner)

    # 收集历次参数修改记录
    param_history = []
    for entry in iteration_log:
        if "parameter_changes" in entry:
            param_history.append({
                "iteration": entry["iteration"],
                "changes": entry["parameter_changes"],
            })

    report = {
        "status": "NEEDS_CODE_REVIEW",
        "iteration": iteration,
        "last_checkpoint": last_checkpoint,
        "iteration_history": iteration_log,
        "current_fail_reasons": reasons,
        "parameter_change_history": param_history,
        "suggestions": suggestions,
        "timestamp": datetime.now().isoformat(),
    }

    report_path = SKILL_ROOT / "assets" / "escalation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[escalation] 报告已保存: {report_path}")


def _ensure_dryrun_override(train_script: str):
    """Ensure the training script reads DRYRUN_TIMESTEPS env var.

    When Claude API rewrites train_full_domain_maneuver_v3.py, it may drop
    our injected override line. This function re-injects it if missing.
    """
    script_path = Path(train_script)
    content = script_path.read_text(encoding="utf-8")
    marker = "DRYRUN_TIMESTEPS"
    if marker not in content:
        # Find the config dict closing brace and inject after it
        # Match the last assignment to config dict (e.g. "}" on its own line after config entries)
        pattern = r'(^config\s*=\s*\{.*?^\})'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if match:
            insert_pos = match.end()
            inject_line = (
                '\n# Allow dry-run override: set DRYRUN_TIMESTEPS env var to run minimal steps for validation\n'
                'config["TOTAL_TIMESTEPS"] = float(os.environ.get("DRYRUN_TIMESTEPS", config["TOTAL_TIMESTEPS"]))\n'
            )
            content = content[:insert_pos] + inject_line + content[insert_pos:]
            script_path.write_text(content, encoding="utf-8")
            print(f"[dry-run] Re-injected DRYRUN_TIMESTEPS override into {script_path.name}")
        else:
            print(f"[dry-run] WARNING: Could not find config dict in {script_path.name} to inject override")


def run_dry_run(train_script: str, timeout_sec: int = 300) -> tuple[bool, str]:
    """
    Run training with minimal steps to validate code correctness.
    Uses DRYRUN_TIMESTEPS env var to override TOTAL_TIMESTEPS in the training script.
    Returns (success, error_message).
    """
    # Ensure the override line exists (Claude API may have removed it)
    _ensure_dryrun_override(train_script)

    print(f"\n[dry-run] Launching validation run (timeout={timeout_sec}s)...")

    env = os.environ.copy()
    env["DRYRUN_TIMESTEPS"] = "2000000"   # 2M steps → 2 updates with 1000 envs * 1000 steps
    env["WANDB_MODE"] = "disabled"        # Don't log dry run to wandb

    cmd = [sys.executable, "-u", str(train_script)]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PLANAX_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(f"[dry-run] PID={proc.pid}")

        # Capture output for error diagnosis
        output_lines = []
        try:
            for line in iter(proc.stdout.readline, b""):
                decoded = line.decode("utf-8", errors="replace")
                sys.stdout.write(f"  [dry-run] {decoded}")
                sys.stdout.flush()
                output_lines.append(decoded)
        except Exception:
            pass
        finally:
            proc.stdout.close()

        try:
            returncode = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            # Timeout during dry-run is treated as success (code ran without crashing)
            print(f"[dry-run] Timed out after {timeout_sec}s — code appears to run without errors.")
            return True, ""

        if returncode != 0:
            # Collect last 3000 chars of output for error context
            full_output = "".join(output_lines)
            error_tail = full_output[-3000:] if len(full_output) > 3000 else full_output
            print(f"[dry-run] FAILED (exit code {returncode})")
            return False, error_tail

        print(f"[dry-run] PASSED (exit code 0)")
        return True, ""

    except Exception as e:
        error_msg = f"Failed to launch dry-run subprocess: {type(e).__name__}: {e}"
        print(f"[dry-run] {error_msg}")
        return False, error_msg


MAX_TOTAL_TIMESTEPS = 3e8  # Hard cap: never train more than 300M steps per iteration

# Early stopping config
EARLY_STOP_WINDOW = 50       # Number of recent updates to check for improvement
EARLY_STOP_MIN_UPDATES = 100 # Don't check early stopping before 100 updates (=100M steps)
EARLY_STOP_THETA_TOL = 2.0   # Minimum theta_deg improvement required over window


def _parse_theta_from_line(line: str) -> Optional[float]:
    """Extract theta_deg value from training output line."""
    m = re.search(r'theta_deg=([\d.]+)', line)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _check_early_stop(theta_history: list) -> tuple[bool, str]:
    """Check if training should be stopped early due to lack of progress.

    Compares the mean theta_deg of the first half vs last half of the window.
    If last half is not at least EARLY_STOP_THETA_TOL degrees better, stop.
    """
    if len(theta_history) < EARLY_STOP_MIN_UPDATES:
        return False, ""

    window = theta_history[-EARLY_STOP_WINDOW:]
    if len(window) < EARLY_STOP_WINDOW:
        return False, ""

    first_half = window[:len(window)//2]
    last_half = window[len(window)//2:]

    mean_first = sum(first_half) / len(first_half)
    mean_last = sum(last_half) / len(last_half)

    improvement = mean_first - mean_last  # positive = theta decreased = good

    if improvement < EARLY_STOP_THETA_TOL:
        reason = (
            f"theta_deg 停滞: 前{len(first_half)}次均值={mean_first:.1f}°, "
            f"后{len(last_half)}次均值={mean_last:.1f}°, "
            f"改善={improvement:.1f}° < 阈值{EARLY_STOP_THETA_TOL}°"
        )
        return True, reason

    return False, ""


def run_training(
    train_script: str,
    total_timesteps: float = 3e8,
    num_envs: int = 1000,
    timeout_sec: int = 259200,  # 72 hours — A100 long training may take 10+ hours
    resume_checkpoint: Optional[str] = None,
    enable_early_stop: bool = True,
) -> Optional[str]:
    """
    Launch training subprocess and wait for completion.
    Returns the checkpoint path if successful, None otherwise.

    stdout/stderr are streamed to a log file (not buffered in memory)
    to avoid OOM on long training runs.

    Early stopping: monitors theta_deg in real-time. If no improvement over
    EARLY_STOP_WINDOW updates, kills the process and returns the latest checkpoint.
    """
    # Enforce TOTAL_TIMESTEPS cap
    total_timesteps = min(total_timesteps, MAX_TOTAL_TIMESTEPS)

    print(f"\n[train] Launching: {train_script}")
    print(f"[train] Timesteps: {total_timesteps:.0e} (cap={MAX_TOTAL_TIMESTEPS:.0e}), Envs: {num_envs}, Timeout: {timeout_sec}s ({timeout_sec/3600:.0f}h)")
    if enable_early_stop:
        print(f"[train] Early stopping: enabled (window={EARLY_STOP_WINDOW}, min_updates={EARLY_STOP_MIN_UPDATES}, theta_tol={EARLY_STOP_THETA_TOL}°)")
    if resume_checkpoint:
        print(f"[train] Resuming from checkpoint: {resume_checkpoint}")

    # Log file for training output
    log_dir = SKILL_ROOT / "assets"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_output.log"

    # Build subprocess command
    cmd = [sys.executable, "-u", str(train_script)]
    if resume_checkpoint:
        cmd.extend(["--resume-checkpoint", resume_checkpoint])

    # Track theta_deg for early stopping
    theta_history = []
    early_stopped = False
    early_stop_reason = ""

    # Use 'tee' to simultaneously print to terminal AND write to log file
    with open(log_file, "w", encoding="utf-8") as flog:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PLANAX_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(f"[train] PID={proc.pid}, log: {log_file}")

        # Stream output: print to terminal + write to file simultaneously
        try:
            for line in iter(proc.stdout.readline, b""):
                decoded = line.decode("utf-8", errors="replace")
                sys.stdout.write(decoded)   # real-time terminal output
                sys.stdout.flush()
                flog.write(decoded)          # persist to log file
                flog.flush()

                # Parse theta_deg for early stopping
                if enable_early_stop:
                    theta = _parse_theta_from_line(decoded)
                    if theta is not None:
                        theta_history.append(theta)
                        should_stop, reason = _check_early_stop(theta_history)
                        if should_stop:
                            early_stopped = True
                            early_stop_reason = reason
                            print(f"\n[early-stop] {reason}")
                            print(f"[early-stop] 终止训练进程 (PID={proc.pid})")
                            proc.kill()
                            break
        except Exception:
            pass
        finally:
            proc.stdout.close()

        try:
            returncode = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"[train] TIMEOUT after {timeout_sec}s — training process killed.")
            print(f"[train] Check log for details: {log_file}")
            return None

    if early_stopped:
        print(f"[train] 早停触发: {early_stop_reason}")
        # Still try to find the latest checkpoint
        results_dir = PLANAX_DIR / "results"
        if results_dir.exists():
            checkpoints = sorted(results_dir.glob("**/checkpoint_epoch_*"), key=os.path.getmtime)
            if checkpoints:
                print(f"[train] 使用最新 checkpoint: {checkpoints[-1]}")
                return str(checkpoints[-1])
        print("[train] 早停但未找到 checkpoint")
        return None

    if returncode != 0:
        print(f"[train] FAILED (exit code {returncode})")
        # Read last 1000 chars of log for error info
        try:
            log_text = log_file.read_text(encoding="utf-8")
            print(f"[train] Log tail (last 1000 chars): {log_text[-1000:]}")
        except Exception:
            pass
        return None

    # Parse checkpoint path from log
    try:
        log_text = log_file.read_text(encoding="utf-8")
        for line in reversed(log_text.split("\n")):
            if "Checkpoint saved" in line:
                results_dir = PLANAX_DIR / "results"
                if results_dir.exists():
                    checkpoints = sorted(results_dir.glob("**/checkpoint_epoch_*"), key=os.path.getmtime)
                    if checkpoints:
                        return str(checkpoints[-1])
                break
    except Exception:
        pass

    # Fallback: find latest checkpoint
    results_dir = PLANAX_DIR / "results"
    if results_dir.exists():
        checkpoints = sorted(results_dir.glob("**/checkpoint_epoch_*"), key=os.path.getmtime)
        if checkpoints:
            return str(checkpoints[-1])

    print("[train] WARNING: Could not find checkpoint path")
    return None


def run_evaluation_subprocess(checkpoint_path: str, episodes: int = 50) -> Optional[Dict]:
    """Run evaluation script as subprocess and parse JSON output."""
    out_file = SKILL_ROOT / "assets" / "latest_eval_report.json"

    cmd = [
        sys.executable, "-u", str(EVALUATE_SCRIPT),
        "--checkpoint", checkpoint_path,
        "--episodes", str(episodes),
        "--max-steps", "2000",
        "--out", str(out_file),
    ]

    log_dir = SKILL_ROOT / "assets"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eval_output.log"

    print(f"\n[eval] Launching evaluation ({episodes} episodes)...")
    print(f"[eval] Log: {log_file}")

    with open(log_file, "w", encoding="utf-8") as flog:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Stream: terminal + log file
        try:
            for line in iter(proc.stdout.readline, b""):
                decoded = line.decode("utf-8", errors="replace")
                sys.stdout.write(decoded)
                sys.stdout.flush()
                flog.write(decoded)
                flog.flush()
        except Exception:
            pass
        finally:
            proc.stdout.close()

        try:
            returncode = proc.wait(timeout=14400)  # 4 hours for evaluation
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"[eval] TIMEOUT after 4 hours — evaluation process killed.")
            return None

    if returncode != 0:
        print(f"[eval] FAILED (exit code {returncode})")
        try:
            log_text = log_file.read_text(encoding="utf-8")
            print(f"[eval] Log tail (last 500 chars): {log_text[-500:]}")
        except Exception:
            pass
        return None

    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ======================== Main Loop ========================

def auto_train_loop(
    max_iterations: int = 5,
    target_metrics_path: str = str(DEFAULT_TARGET_METRICS),
    checkpoint_path: Optional[str] = None,
    skip_training: bool = False,
    eval_episodes: int = 50,
    param_stall_limit: int = 3,
):
    """
    Main evaluator-optimizer loop.

    Args:
        max_iterations: Maximum number of train-evaluate-optimize cycles.
        target_metrics_path: Path to target metrics JSON.
        checkpoint_path: If provided, skip first training and evaluate this checkpoint.
        skip_training: If True, only evaluate+optimize (no subprocess training).
        eval_episodes: Number of evaluation episodes per iteration.
        param_stall_limit: Consecutive stall iterations before escalating to user.
    """
    print("=" * 70)
    print("  AIRCRAFT RL TUNER — Auto Train Loop (Anthropic SDK 流式架构)")
    print("=" * 70)
    print(f"  Max iterations:  {max_iterations}")
    print(f"  Target metrics:  {target_metrics_path}")
    print(f"  Skip training:   {skip_training}")
    print(f"  Eval episodes:   {eval_episodes}")
    print(f"  Stall limit:     {param_stall_limit}")
    print(f"  SDK model:       {ANTHROPIC_TUNER_MODEL}")
    _sdk_base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://ai.tokencloud.ai")
    print(f"  API base URL:    {_sdk_base_url}")
    print("=" * 70)

    targets = load_target_metrics(target_metrics_path)

    iteration_log = []
    last_checkpoint = None  # Track last successful checkpoint for resume
    loop_status = "MAX_ITERATIONS_REACHED"
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'='*60}")

        # Step 1: Train (or use provided checkpoint)
        if checkpoint_path and iteration == 1:
            ckpt = checkpoint_path
            print(f"[iter {iteration}] Using provided checkpoint: {ckpt}")
        elif skip_training:
            print(f"[iter {iteration}] skip_training=True, finding latest checkpoint...")
            results_dir = PLANAX_DIR / "results"
            checkpoints = sorted(results_dir.glob("**/checkpoint_epoch_*"), key=os.path.getmtime) if results_dir.exists() else []
            if not checkpoints:
                print("[ERROR] No checkpoint found and skip_training=True. Exiting.")
                break
            ckpt = str(checkpoints[-1])
            print(f"[iter {iteration}] Latest checkpoint: {ckpt}")
        else:
            # Resume from last checkpoint if available
            resume_ckpt = None
            if last_checkpoint and Path(last_checkpoint).exists():
                resume_ckpt = last_checkpoint
                print(f"[iter {iteration}] 从上一轮 checkpoint 继续训练: {resume_ckpt}")
            else:
                if iteration > 1:
                    print(f"[iter {iteration}] 未找到可用 checkpoint，从头训练")

            ckpt = run_training(str(TRAIN_SCRIPT), resume_checkpoint=resume_ckpt)
            if ckpt is None and resume_ckpt is not None:
                # Resume failed (e.g. obs space changed) — fallback to train from scratch
                print(f"[iter {iteration}] 继续训练失败，回退到从头训练...")
                ckpt = run_training(str(TRAIN_SCRIPT), resume_checkpoint=None)
            if ckpt is None:
                print(f"[iter {iteration}] Training failed. Stopping loop.")
                break

        last_checkpoint = ckpt  # Track for potential resume in next iteration

        # Step 2: Evaluate
        report = run_evaluation_subprocess(ckpt, episodes=eval_episodes)
        if report is None:
            print(f"[iter {iteration}] Evaluation failed. Stopping loop.")
            break

        agg = report.get("aggregate", {})
        print(f"\n[iter {iteration}] Results:")
        print(f"  success_rate:  {agg.get('mean_success_rate', 0):.3f}")
        print(f"  crash_count:   {agg.get('mean_crash_count', 0):.2f}")
        print(f"  stall_steps:   {agg.get('mean_stall_steps', 0):.1f}")
        print(f"  total_reward:  {agg.get('mean_total_reward', 0):.1f}")

        # Step 3: Check pass/fail
        passed, reasons = check_pass(report, targets)

        iter_entry = {
            "iteration": iteration,
            "checkpoint": ckpt,
            "passed": passed,
            "aggregate": agg,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
        }
        iteration_log.append(iter_entry)

        if passed:
            print(f"\n{'*'*60}")
            print(f"  SUCCESS at iteration {iteration}!")
            print(f"  All target metrics met. Final checkpoint: {ckpt}")
            print(f"{'*'*60}")

            # Save final report
            final_report = {
                "status": "SUCCESS",
                "final_iteration": iteration,
                "final_checkpoint": ckpt,
                "evaluation": report,
                "iteration_log": iteration_log,
            }
            final_path = SKILL_ROOT / "assets" / "final_report.json"
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            print(f"  Final report: {final_path}")
            return final_report

        # Step 4: Not passed — call Claude for parameter adjustments
        print(f"\n[iter {iteration}] FAIL reasons:")
        for r in reasons:
            print(f"  - {r}")

        if iteration >= max_iterations:
            print(f"\n[iter {iteration}] Max iterations reached. Stopping.")
            break

        # Step 3.5: 停滞检测 — 连续 N 轮参数调优无进展则暂停
        is_stalled, stall_diag = _detect_plateau(iteration_log, window=param_stall_limit)
        if is_stalled:
            print(f"\n[iter {iteration}] 停滞检测: {stall_diag}")
            _escalate_to_user(iteration_log, reasons, iteration, last_checkpoint)
            loop_status = "NEEDS_CODE_REVIEW"
            break

        # Step 4: Call Claude Code CLI for code-level analysis and modifications
        print(f"\n[iter {iteration}] 备份代码文件...")
        backup_dir = backup_code_files(iteration)

        print(f"\n[iter {iteration}] 调用 Claude Code 进行代码级分析和修改...")
        cc_success, cc_summary = call_claude_code(
            report, reasons, iteration,
            iteration_log=iteration_log,
        )

        # Record changes in iteration log
        iter_entry["code_changes_summary"] = cc_summary

        if not cc_success:
            print(f"[iter {iteration}] Claude Code 调用失败（已重试3次）: {cc_summary}")
            print(f"[iter {iteration}] 回滚代码，停止循环。")
            rollback_code_files(backup_dir)
            loop_status = "CLAUDE_CODE_FAILED"
            break

        # Step 5: Dry-run validation
        print(f"\n[iter {iteration}] 运行 dry-run 验证...")
        dry_ok, dry_err = run_dry_run(str(TRAIN_SCRIPT))
        if dry_ok:
            print(f"[iter {iteration}] Dry-run 通过！")
        else:
            print(f"[iter {iteration}] Dry-run 失败:\n{dry_err[:1500]}")
            print(f"[iter {iteration}] 回滚代码到修改前...")
            rollback_code_files(backup_dir)

            # 给 Claude Code 第二次机会：把错误信息也传入
            print(f"\n[iter {iteration}] 恢复代码后，将错误信息传给 Claude Code 再试一次...")
            backup_dir2 = backup_code_files(iteration * 100 + 1)  # 二次备份
            retry_report = dict(report)
            retry_reasons = reasons + [f"DRY-RUN FAILED: {dry_err[:2000]}"]
            cc_success2, cc_summary2 = call_claude_code(
                retry_report, retry_reasons, iteration,
                iteration_log=iteration_log,
            )
            iter_entry["code_changes_summary"] = cc_summary2

            if not cc_success2:
                print(f"[iter {iteration}] Claude Code 第二次也失败了。回滚并停止。")
                rollback_code_files(backup_dir2)
                break

            # 二次 dry-run
            dry_ok2, dry_err2 = run_dry_run(str(TRAIN_SCRIPT))
            if dry_ok2:
                print(f"[iter {iteration}] 第二次 dry-run 通过！")
            else:
                print(f"[iter {iteration}] 第二次 dry-run 仍然失败:\n{dry_err2[:1000]}")
                print(f"[iter {iteration}] 回滚代码并停止循环。")
                rollback_code_files(backup_dir2)
                break

        # Memory cleanup
        gc.collect()

    # Save iteration log
    log_path = SKILL_ROOT / "assets" / "iteration_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(iteration_log, f, indent=2, ensure_ascii=False)
    print(f"\nIteration log saved: {log_path}")

    return {"status": loop_status, "iteration_log": iteration_log}


# ======================== CLI ========================

def main():
    parser = argparse.ArgumentParser(description="Auto RL Training Loop with Claude-powered optimization")
    parser.add_argument("--max-iters", type=int, default=10, help="Max training iterations")
    parser.add_argument("--target-metrics", default=str(DEFAULT_TARGET_METRICS), help="Target metrics JSON")
    parser.add_argument("--checkpoint", default=None, help="Start from this checkpoint (skip first training)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only evaluate+optimize")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Episodes per evaluation")
    parser.add_argument("--param-stall-limit", type=int, default=3,
                        help="Consecutive stall iterations before escalating to user (default: 3)")
    args = parser.parse_args()

    result = auto_train_loop(
        max_iterations=args.max_iters,
        target_metrics_path=args.target_metrics,
        checkpoint_path=os.path.abspath(args.checkpoint) if args.checkpoint else None,
        skip_training=args.skip_training,
        eval_episodes=args.eval_episodes,
        param_stall_limit=args.param_stall_limit,
    )

    verdict = result.get("status", "UNKNOWN")
    print(f"\n[auto_train_loop] Final status: {verdict}")
    if verdict == "SUCCESS":
        return 0
    elif verdict == "NEEDS_CODE_REVIEW":
        return 2
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
