import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-2c45b7fd-69c8-1697-23a0-fe7ce7a2a620")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import run_half_loop_bridge_micro_search as bridge
import run_half_loop_termination_trace as trace


PLANAX_ROOT = Path(__file__).resolve().parent

VARIANTS = {
    "baseline_target_stream": {},
    "larger_radius_bridge": {
        "target_radius_m": 18000.0,
    },
    "curvature_limited_bridge": {
        "target_radius_m": 20000.0,
        "pitch_rate_limit_deg_s": 10.0,
        "roll_rate_limit_deg_s": 30.0,
    },
    "load_factor_limited_bridge": {
        "pitch_blend_with_current": 0.35,
        "pitch_rate_limit_deg_s": 8.0,
        "roll_rate_limit_deg_s": 25.0,
    },
    "vt_scheduled_bridge": {
        "bridge_target_vt": 220.0,
    },
    "lookahead_scheduled_bridge": {
        "bridge_lookahead_dist": 650.0,
    },
    "pitch_rate_limited_bridge": {
        "pitch_rate_limit_deg_s": 8.0,
    },
    "combined_load_factor_aware_bridge": {
        "target_radius_m": 18000.0,
        "bridge_lookahead_dist": 800.0,
        "bridge_target_vt": 230.0,
        "pitch_rate_limit_deg_s": 10.0,
        "roll_rate_limit_deg_s": 30.0,
        "pitch_blend_with_current": 0.20,
    },
}


def parse_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def policy_configs():
    return {
        "base_only": None,
        "update2_scale1.0": bridge.make_residual_cfg(scale=1.0),
        "update2_scale0.2": bridge.make_residual_cfg(scale=0.20),
    }


def variant_success(row):
    return (
        str(row.get("task")) == "pu165_R15000"
        and str(row.get("completed")) == "True"
        and row.get("terminal_reason_classified") == "success"
        and float(row.get("Gmax", 999.0)) < 9.0
        and float(row.get("env_alpha_max", 999.0)) < 45.0
    )


def write_report(root, rows, terminal_rows):
    hard_rows = [r for r in rows if r["task"] == "pu165_R15000"]
    successes = [r for r in hard_rows if variant_success(r)]
    if successes:
        best = min(successes, key=lambda r: float(r["CTE_mean"]))
    else:
        best = min(hard_rows, key=lambda r: (r["terminal_reason_classified"] != "success", float(r["CTE_mean"]))) if hard_rows else None
    text = [
        "# Load-Factor-Aware Bridge Target Stream Report",
        "",
        "## Decision",
        "",
        f"- pu165_success_count: `{len(successes)}`",
        f"- best_candidate: `{best['policy'] if best else 'none'}`",
        f"- best_variant: `{best['target_stream_variant'] if best else 'none'}`",
        "- No residual training was run.",
        "",
        "## Hard Task Results",
        "",
        "| policy | variant | completed | reason | phase | CTE | Gmax | alpha_max | wing |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in hard_rows:
        text.append(
            f"| {row['policy']} | {row['target_stream_variant']} | {row['completed']} | "
            f"{row['terminal_reason_classified']} | {float(row['terminal_phase_deg']):.1f} | "
            f"{float(row['CTE_mean']):.1f} | {float(row['Gmax']):.2f} | "
            f"{float(row['env_alpha_max']):.1f} | {float(row['wing_plane_error_mean']):.1f} |"
        )
    text.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if successes:
        text.append("- At least one target-stream candidate made `pu165_R15000` complete with Gmax < 9.")
        text.append("- Residual training may resume only around the successful target stream and still only for pu165 snippets.")
    else:
        text.append("- No tested target-stream candidate completed `pu165_R15000` with Gmax < 9.")
        text.append("- Treat the best row as diagnostic only; do not resume residual PPO training yet.")
    text.extend(
        [
            "",
            "## Files",
            "",
            f"- summary: `{(root / 'target_stream_summary.csv').resolve()}`",
            f"- terminal states: `{(root / 'terminal_states.csv').resolve()}`",
            f"- phasewise: `{(root / 'phasewise').resolve()}`",
        ]
    )
    (root / "target_stream_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--variants",
        default="baseline_target_stream,larger_radius_bridge,curvature_limited_bridge,load_factor_limited_bridge,vt_scheduled_bridge,lookahead_scheduled_bridge,pitch_rate_limited_bridge,combined_load_factor_aware_bridge",
    )
    parser.add_argument("--policies", default="base_only,update2_scale1.0,update2_scale0.2")
    parser.add_argument("--hard-task", default="pu165_R15000")
    parser.add_argument("--retention-tasks", default="")
    args = parser.parse_args()
    root = args.out_dir or PLANAX_ROOT / "results/load_factor_aware_bridge_target_stream" / datetime.now().strftime("%Y%m%d_%H%M")
    root.mkdir(parents=True, exist_ok=True)
    (root / "phasewise").mkdir(exist_ok=True)
    (root / "raw_terminal_info").mkdir(exist_ok=True)
    selected_variants = parse_list(args.variants)
    selected_policies = parse_list(args.policies)
    retention_tasks = parse_list(args.retention_tasks)
    bridge.write_json(
        root / "config.json",
        {
            "base": str(bridge.BASE_CKPT),
            "residual": str(bridge.BEST_RESIDUAL),
            "variants": {name: VARIANTS[name] for name in selected_variants},
            "policies": selected_policies,
            "hard_task": args.hard_task,
            "retention_tasks": retention_tasks,
        },
    )
    env, net, net_params, residual_net, residual_params, _ = bridge.load_models()
    policies = policy_configs()
    rows = []
    terminal_rows = []
    run_plan = []
    for variant_name in selected_variants:
        for policy_name in selected_policies:
            run_plan.append((variant_name, policy_name, args.hard_task))
    for variant_name in selected_variants:
        if variant_name not in {"baseline_target_stream", "combined_load_factor_aware_bridge"}:
            continue
        for policy_name in selected_policies:
            for task in retention_tasks:
                run_plan.append((variant_name, policy_name, task))
    for variant_name, policy_name, task in run_plan:
        variant = VARIANTS[variant_name]
        cfg = policies[policy_name]
        summary, terminal_row, phase_rows, raw_info = trace.run_trace_test(
            env,
            net,
            net_params,
            residual_net,
            residual_params,
            f"{policy_name}__{variant_name}",
            task,
            cfg,
            variant=variant,
        )
        summary["base_policy"] = policy_name
        summary["target_stream_variant"] = variant_name
        terminal_row["base_policy"] = policy_name
        terminal_row["target_stream_variant"] = variant_name
        rows.append(summary)
        terminal_rows.append(terminal_row)
        fieldnames = ["base_policy", "target_stream_variant"] + trace.SUMMARY_FIELDS
        terminal_fields = ["base_policy", "target_stream_variant"] + trace.TERMINAL_FIELDS
        bridge.write_csv(root / "target_stream_summary.csv", rows, fieldnames)
        bridge.write_csv(root / "terminal_states.csv", terminal_rows, terminal_fields)
        bridge.write_csv(root / "phasewise" / f"{policy_name}__{variant_name}_{task}.csv", phase_rows, bridge.PHASE_FIELDS)
        bridge.write_json(root / "raw_terminal_info" / f"{policy_name}__{variant_name}_{task}.json", raw_info)
        print(
            f"{policy_name} {variant_name} {task} term={summary['terminal_reason_classified']} "
            f"completed={summary['completed']} phase={float(summary['terminal_phase_deg']):.1f} "
            f"CTE={float(summary['CTE_mean']):.1f} Gmax={float(summary['Gmax']):.2f}",
            flush=True,
        )
    write_report(root, rows, terminal_rows)
    print(f"target_stream_dir={root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
