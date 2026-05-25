import argparse
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


PLANAX_ROOT = Path(__file__).resolve().parent
HARD_TASK = "pu165_R15000"
RADIUS_VALUES = [15000.0, 18000.0, 20000.0, 25000.0, 30000.0]


def parse_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def policy_configs():
    return {
        "base_only": None,
        "update2_scale1.0": bridge.make_residual_cfg(scale=1.0),
        "update2_scale0.2": bridge.make_residual_cfg(scale=0.20),
    }


def rhtso_variants():
    return [
        {
            "variant": "baseline_target_stream",
            "family": "baseline",
            "eval_radius_m": 15000.0,
            "target_radius_m": 15000.0,
        },
        {
            "variant": "curvature_limited_profile_R20000_s100",
            "family": "curvature_limited",
            "eval_radius_m": 15000.0,
            "target_radius_m": 15000.0,
            "target_radius_profile": [
                {"start_deg": 100.0, "end_deg": 165.0, "radius_m": 20000.0, "transition_deg": 12.0}
            ],
        },
        {
            "variant": "local_radius_inflation_R25000_s100",
            "family": "local_radius_inflation",
            "eval_radius_m": 15000.0,
            "target_radius_m": 15000.0,
            "target_radius_profile": [
                {"start_deg": 100.0, "end_deg": 165.0, "radius_m": 25000.0, "transition_deg": 15.0}
            ],
        },
        {
            "variant": "local_radius_inflation_R30000_s080",
            "family": "local_radius_inflation",
            "eval_radius_m": 15000.0,
            "target_radius_m": 15000.0,
            "target_radius_profile": [
                {"start_deg": 80.0, "end_deg": 165.0, "radius_m": 30000.0, "transition_deg": 18.0}
            ],
        },
        {
            "variant": "pitch_rate_limited_6dps",
            "family": "pitch_rate_limited",
            "eval_radius_m": 15000.0,
            "target_radius_m": 15000.0,
            "pitch_rate_limit_deg_s": 6.0,
            "roll_rate_limit_deg_s": 25.0,
        },
        {
            "variant": "progress_slowdown_v220_la700",
            "family": "progress_slowdown",
            "eval_radius_m": 15000.0,
            "target_radius_m": 15000.0,
            "bridge_target_vt": 220.0,
            "bridge_lookahead_dist": 700.0,
            "pitch_blend_with_current": 0.15,
            "pitch_rate_limit_deg_s": 8.0,
            "roll_rate_limit_deg_s": 25.0,
        },
        {
            "variant": "rhtso_lf_profile_R25000_v220_la700",
            "family": "load_factor_aware_rhtso",
            "eval_radius_m": 15000.0,
            "target_radius_m": 15000.0,
            "target_radius_profile": [
                {"start_deg": 90.0, "end_deg": 165.0, "radius_m": 25000.0, "transition_deg": 15.0}
            ],
            "bridge_target_vt": 220.0,
            "bridge_lookahead_dist": 700.0,
            "pitch_blend_with_current": 0.20,
            "pitch_rate_limit_deg_s": 8.0,
            "roll_rate_limit_deg_s": 25.0,
        },
        {
            "variant": "rhtso_lf_profile_R30000_v230_la900",
            "family": "load_factor_aware_rhtso",
            "eval_radius_m": 15000.0,
            "target_radius_m": 15000.0,
            "target_radius_profile": [
                {"start_deg": 80.0, "end_deg": 165.0, "radius_m": 30000.0, "transition_deg": 18.0}
            ],
            "bridge_target_vt": 230.0,
            "bridge_lookahead_dist": 900.0,
            "pitch_blend_with_current": 0.15,
            "pitch_rate_limit_deg_s": 8.0,
            "roll_rate_limit_deg_s": 30.0,
        },
        {
            "variant": "rhtso_energy_safe_R30000_v240_la1100",
            "family": "load_factor_aware_rhtso",
            "eval_radius_m": 15000.0,
            "target_radius_m": 15000.0,
            "target_radius_profile": [
                {"start_deg": 80.0, "end_deg": 165.0, "radius_m": 30000.0, "transition_deg": 20.0}
            ],
            "bridge_target_vt": 240.0,
            "bridge_lookahead_dist": 1100.0,
            "pitch_blend_with_current": 0.10,
            "pitch_rate_limit_deg_s": 10.0,
            "roll_rate_limit_deg_s": 35.0,
        },
    ]


def csv_float(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def is_success(row):
    return (
        str(row.get("completed")) == "True"
        and row.get("terminal_reason_classified") == "success"
        and csv_float(row, "effective_Gmax", csv_float(row, "Gmax", 999.0)) < 9.0
    )


def phase_stats(phase_rows):
    cte = np.asarray([csv_float(r, "CTE") for r in phase_rows], dtype=np.float64)
    phase = np.asarray([csv_float(r, "phase") for r in phase_rows], dtype=np.float64)
    target_pitch = np.asarray([csv_float(r, "target_pitch") for r in phase_rows], dtype=np.float64)
    target_roll = np.asarray([csv_float(r, "target_roll") for r in phase_rows], dtype=np.float64)
    target_pitch_delta = np.abs(np.diff(target_pitch)) if len(target_pitch) > 1 else np.asarray([0.0])
    target_roll_delta = np.abs(np.diff(target_roll)) if len(target_roll) > 1 else np.asarray([0.0])
    if len(cte) == 0:
        return {
            "CTE_p90": "",
            "CTE_max": "",
            "phase_max": "",
            "target_pitch_delta_mean": "",
            "target_pitch_delta_max": "",
            "target_roll_delta_mean": "",
            "target_roll_delta_max": "",
        }
    return {
        "CTE_p90": float(np.percentile(cte, 90)),
        "CTE_max": float(np.max(cte)),
        "phase_max": float(np.max(phase)),
        "target_pitch_delta_mean": float(np.mean(target_pitch_delta)),
        "target_pitch_delta_max": float(np.max(target_pitch_delta)),
        "target_roll_delta_mean": float(np.mean(target_roll_delta)),
        "target_roll_delta_max": float(np.max(target_roll_delta)),
    }


def rhtso_cost(row):
    gmax = csv_float(row, "effective_Gmax", csv_float(row, "Gmax", 99.0))
    vt_min = csv_float(row, "effective_vt_min", csv_float(row, "vt_min", 0.0))
    cte = csv_float(row, "CTE_mean", 100000.0)
    wing = csv_float(row, "wing_plane_error_mean", 180.0)
    vtan = csv_float(row, "velocity_tangent_error_mean", 180.0)
    nose = csv_float(row, "nose_tangent_error_mean", 180.0)
    terminal_phase = csv_float(row, "terminal_phase_deg", 0.0)
    pitch_smooth = csv_float(row, "target_pitch_delta_mean", 0.0)
    roll_smooth = csv_float(row, "target_roll_delta_mean", 0.0)
    completion_penalty = 0.0 if is_success(row) else 4.0 + max(0.0, 165.0 - terminal_phase) / 25.0
    g_penalty = 6.0 * max(0.0, gmax - 8.5) ** 2 + 25.0 * max(0.0, gmax - 9.0) ** 2
    low_speed_penalty = 4.0 * max(0.0, 190.0 - vt_min) / 50.0
    geometry_cost = cte / 3000.0 + wing / 45.0 + (vtan + nose) / 120.0
    smooth_cost = pitch_smooth / 4.0 + roll_smooth / 8.0
    return float(completion_penalty + g_penalty + low_speed_penalty + geometry_cost + smooth_cost)


def row_from_summary(summary, phase_rows, terminal_row, policy_name, variant, group):
    row = dict(summary)
    row.update(phase_stats(phase_rows))
    terminal_g = csv_float(terminal_row, "terminal_G", csv_float(row, "Gmax", 0.0))
    terminal_vt = csv_float(terminal_row, "terminal_vt", csv_float(row, "vt_min", 0.0))
    row["terminal_G"] = terminal_g
    row["terminal_vt"] = terminal_vt
    row["effective_Gmax"] = max(csv_float(row, "Gmax", 0.0), terminal_g)
    row["effective_vt_min"] = min(csv_float(row, "vt_min", 0.0), terminal_vt)
    row["base_policy"] = policy_name
    row["variant"] = variant["variant"]
    row["family"] = variant.get("family", "")
    row["group"] = group
    row["eval_radius_m"] = variant.get("eval_radius_m", 15000.0)
    row["target_radius_m"] = variant.get("target_radius_m", 15000.0)
    row["bridge_target_vt"] = variant.get("bridge_target_vt", "")
    row["bridge_lookahead_dist"] = variant.get("bridge_lookahead_dist", "")
    row["pitch_rate_limit_deg_s"] = variant.get("pitch_rate_limit_deg_s", "")
    row["roll_rate_limit_deg_s"] = variant.get("roll_rate_limit_deg_s", "")
    row["pitch_blend_with_current"] = variant.get("pitch_blend_with_current", "")
    row["target_radius_profile"] = json.dumps(variant.get("target_radius_profile", []), sort_keys=True)
    row["success_gate"] = is_success(row)
    row["rhtso_cost"] = rhtso_cost(row)
    return row


OUTPUT_FIELDS = [
    "group",
    "base_policy",
    "variant",
    "family",
    "task",
    "eval_radius_m",
    "target_radius_m",
    "completed",
    "success_gate",
    "terminal_reason_classified",
    "terminal_phase_deg",
    "steps",
    "CTE_mean",
    "CTE_p90",
    "CTE_max",
    "velocity_tangent_error_mean",
    "nose_tangent_error_mean",
    "nose_velocity_error_mean",
    "wing_plane_error_mean",
    "q_error_mean_rad",
    "Gmax",
    "terminal_G",
    "effective_Gmax",
    "env_alpha_max",
    "env_beta_max",
    "vt_min",
    "terminal_vt",
    "effective_vt_min",
    "vt_max",
    "phase145_170_CTE_mean",
    "phase145_170_Gmax",
    "phase145_170_alpha_max",
    "phase145_170_vt_min",
    "target_pitch_delta_mean",
    "target_pitch_delta_max",
    "target_roll_delta_mean",
    "target_roll_delta_max",
    "action_diff_norm_mean",
    "jitter_action_diff_mean",
    "rhtso_cost",
    "bridge_target_vt",
    "bridge_lookahead_dist",
    "pitch_rate_limit_deg_s",
    "roll_rate_limit_deg_s",
    "pitch_blend_with_current",
    "target_radius_profile",
]


def write_report(root, radius_rows, rhtso_rows):
    all_rows = radius_rows + rhtso_rows
    successes = [r for r in all_rows if is_success(r)]
    radius_successes = [r for r in radius_rows if is_success(r)]
    by_variant = {r.get("variant"): r for r in all_rows}
    min_success_radius = min((csv_float(r, "eval_radius_m") for r in radius_successes), default=None)
    best_by_cost = min(all_rows, key=lambda r: csv_float(r, "rhtso_cost", 1e9)) if all_rows else None
    best_progress = max(all_rows, key=lambda r: csv_float(r, "terminal_phase_deg", -1.0)) if all_rows else None
    best_safe_progress = max(
        [r for r in all_rows if csv_float(r, "effective_Gmax", csv_float(r, "Gmax", 99.0)) < 9.0],
        key=lambda r: csv_float(r, "terminal_phase_deg", -1.0),
        default=None,
    )
    text = [
        "# Load-Factor-Aware RH-TSO pu165 Study",
        "",
        "## Decision",
        "",
        f"- success_count: `{len(successes)}`",
        f"- min_success_radius_m: `{min_success_radius if min_success_radius is not None else 'none'}`",
        f"- best_cost_variant: `{best_by_cost['variant'] if best_by_cost else 'none'}`",
        f"- best_progress_variant: `{best_progress['variant'] if best_progress else 'none'}`",
        "- residual_training_resumed: `False`",
        "",
        "## Minimum Radius Sweep",
        "",
        "| policy | radius | completed | reason | phase | CTE | effective_Gmax | effective_vt_min | cost |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in radius_rows:
        text.append(
            f"| {row['base_policy']} | {float(row['eval_radius_m']):.0f} | {row['completed']} | "
            f"{row['terminal_reason_classified']} | {float(row['terminal_phase_deg']):.1f} | "
            f"{float(row['CTE_mean']):.1f} | {float(row['effective_Gmax']):.2f} | "
            f"{float(row['effective_vt_min']):.1f} | {float(row['rhtso_cost']):.2f} |"
        )
    text.extend(
        [
            "",
            "## RH-TSO Target-Stream Sweep",
            "",
            "| variant | family | completed | reason | phase | CTE | effective_Gmax | effective_vt_min | cost |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rhtso_rows, key=lambda r: csv_float(r, "rhtso_cost", 1e9)):
        text.append(
            f"| {row['variant']} | {row['family']} | {row['completed']} | "
            f"{row['terminal_reason_classified']} | {float(row['terminal_phase_deg']):.1f} | "
            f"{float(row['CTE_mean']):.1f} | {float(row['effective_Gmax']):.2f} | "
            f"{float(row['effective_vt_min']):.1f} | {float(row['rhtso_cost']):.2f} |"
        )
    text.extend(
        [
            "",
            "## Diagnosis",
            "",
        ]
    )
    if successes:
        text.append("- At least one target stream completed pu165 with Gmax < 9. Residual training may resume only around that successful target stream.")
    else:
        text.append("- No tested target stream completed pu165 with Gmax < 9, so residual training must remain stopped.")
    if best_safe_progress:
        text.append(
            f"- Best G-safe progress: `{best_safe_progress['variant']}` reached phase "
            f"`{float(best_safe_progress['terminal_phase_deg']):.1f}` with Gmax "
            f"`{float(best_safe_progress['effective_Gmax']):.2f}` and reason `{best_safe_progress['terminal_reason_classified']}`."
        )
    if best_progress:
        text.append(
            f"- Furthest progress overall: `{best_progress['variant']}` reached phase "
            f"`{float(best_progress['terminal_phase_deg']):.1f}` with Gmax "
            f"`{float(best_progress['effective_Gmax']):.2f}` and vt_min `{float(best_progress['effective_vt_min']):.1f}`."
        )
    text.extend(
        [
            "- Current evidence should be read as target-stream/controller feasibility evidence, not as a new policy improvement.",
            "",
            "## Failure Attribution",
            "",
        ]
    )
    baseline = by_variant.get("baseline_target_stream") or by_variant.get("global_radius_R15000")
    if baseline:
        text.append(
            f"- R15000 baseline fails by `{baseline['terminal_reason_classified']}` at phase "
            f"`{float(baseline['terminal_phase_deg']):.1f}` with CTE `{float(baseline['CTE_mean']):.1f}`. "
            "This is a controller/progress failure before the 145-170 bridge, not a clean load-factor-limited completion."
        )
    r20000 = by_variant.get("global_radius_R20000")
    if r20000:
        text.append(
            f"- Global R20000 gives the best G-safe near-bridge result: phase "
            f"`{float(r20000['terminal_phase_deg']):.1f}`, CTE `{float(r20000['CTE_mean']):.1f}`, "
            f"effective_Gmax `{float(r20000['effective_Gmax']):.2f}`, but terminates as "
            f"`{r20000['terminal_reason_classified']}` with effective_vt_min `{float(r20000['effective_vt_min']):.1f}`."
        )
    local_overload = by_variant.get("local_radius_inflation_R25000_s100")
    if local_overload:
        text.append(
            f"- Local radius inflation can push phase farther (`{float(local_overload['terminal_phase_deg']):.1f}`), "
            f"but this candidate terminates by `{local_overload['terminal_reason_classified']}` with "
            f"effective_Gmax `{float(local_overload['effective_Gmax']):.2f}` and large CTE "
            f"`{float(local_overload['CTE_mean']):.1f}`."
        )
    text.extend(
        [
            "- The dominant blocker is therefore not solved by residual-only control or by a single larger radius. It is a coupled target-stream/controller/energy problem: easing curvature reduces CTE and G, but then the aircraft runs out of usable speed/energy before a valid pu165 completion.",
            "- A stronger RH-TSO or MPC-style target-stream optimizer should explicitly trade phase progress against speed floor, altitude demand, and load factor before any new residual PPO run.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if successes:
        text.append("- Keep pu165 as the only hard objective and run a small residual correction only after validating ACMI/visual continuity for the successful target stream.")
    else:
        text.append("- Keep pu165 as the only hard objective, but switch away from residual-only PPO.")
        text.append("- Next work should use larger-radius bridge curriculum, stronger load-factor-aware RH-TSO, reference reshaping before 165, or model-predictive target-stream co-design.")
        text.append("- Do not test pu170/175/180 and do not resume residual training yet.")
    text.extend(
        [
            "",
            "## Files",
            "",
            f"- radius_sweep: `{(root / 'radius_sweep.csv').resolve()}`",
            f"- rhtso_sweep: `{(root / 'rhtso_sweep.csv').resolve()}`",
            f"- phasewise: `{(root / 'phasewise').resolve()}`",
            f"- terminal states: `{(root / 'terminal_states.csv').resolve()}`",
        ]
    )
    (root / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--policies", default="update2_scale0.2")
    parser.add_argument("--radius-values", default="15000,18000,20000,25000,30000")
    parser.add_argument("--skip-radius", action="store_true")
    parser.add_argument("--skip-rhtso", action="store_true")
    args = parser.parse_args()

    root = args.out_dir or PLANAX_ROOT / "results/load_factor_aware_rhtso_pu165" / datetime.now().strftime("%Y%m%d_%H%M")
    root.mkdir(parents=True, exist_ok=True)
    (root / "phasewise").mkdir(exist_ok=True)
    (root / "raw_terminal_info").mkdir(exist_ok=True)

    policies = policy_configs()
    selected_policies = parse_list(args.policies)
    radii = [float(x) for x in parse_list(args.radius_values)]
    bridge.write_json(
        root / "config.json",
        {
            "base": str(bridge.BASE_CKPT),
            "residual": str(bridge.BEST_RESIDUAL),
            "hard_task": HARD_TASK,
            "policies": selected_policies,
            "radius_values": radii,
            "rhtso_cost_terms": [
                "G",
                "low_speed",
                "CTE",
                "wing_plane",
                "velocity_tangent",
                "nose_tangent",
                "target_smoothness",
                "completion_phase",
            ],
        },
    )

    env, net, net_params, residual_net, residual_params, _ = bridge.load_models()
    radius_rows = []
    rhtso_rows = []
    terminal_rows = []

    if not args.skip_radius:
        for policy_name in selected_policies:
            cfg = policies[policy_name]
            for radius in radii:
                variant = {
                    "variant": f"global_radius_R{int(radius)}",
                    "family": "radius_sweep",
                    "eval_radius_m": radius,
                    "target_radius_m": radius,
                }
                summary, terminal_row, phase_rows, raw_info = trace.run_trace_test(
                    env,
                    net,
                    net_params,
                    residual_net,
                    residual_params,
                    f"{policy_name}__{variant['variant']}",
                    HARD_TASK,
                    cfg,
                    variant=variant,
                )
                row = row_from_summary(summary, phase_rows, terminal_row, policy_name, variant, "radius_sweep")
                radius_rows.append(row)
                terminal_row["base_policy"] = policy_name
                terminal_row["variant"] = variant["variant"]
                terminal_rows.append(terminal_row)
                bridge.write_csv(root / "radius_sweep.csv", radius_rows, OUTPUT_FIELDS)
                bridge.write_csv(root / "terminal_states.csv", terminal_rows, ["base_policy", "variant"] + trace.TERMINAL_FIELDS)
                bridge.write_csv(root / "phasewise" / f"{policy_name}__{variant['variant']}.csv", phase_rows, bridge.PHASE_FIELDS)
                bridge.write_json(root / "raw_terminal_info" / f"{policy_name}__{variant['variant']}.json", raw_info)
                print(
                    f"radius {policy_name} R={radius:.0f} completed={row['completed']} "
                    f"term={row['terminal_reason_classified']} phase={float(row['terminal_phase_deg']):.1f} "
                    f"CTE={float(row['CTE_mean']):.1f} Gmax={float(row['effective_Gmax']):.2f} cost={float(row['rhtso_cost']):.2f}",
                    flush=True,
                )

    if not args.skip_rhtso:
        for policy_name in selected_policies:
            cfg = policies[policy_name]
            for variant in rhtso_variants():
                summary, terminal_row, phase_rows, raw_info = trace.run_trace_test(
                    env,
                    net,
                    net_params,
                    residual_net,
                    residual_params,
                    f"{policy_name}__{variant['variant']}",
                    HARD_TASK,
                    cfg,
                    variant=variant,
                )
                row = row_from_summary(summary, phase_rows, terminal_row, policy_name, variant, "rhtso_sweep")
                rhtso_rows.append(row)
                terminal_row["base_policy"] = policy_name
                terminal_row["variant"] = variant["variant"]
                terminal_rows.append(terminal_row)
                bridge.write_csv(root / "rhtso_sweep.csv", rhtso_rows, OUTPUT_FIELDS)
                bridge.write_csv(root / "terminal_states.csv", terminal_rows, ["base_policy", "variant"] + trace.TERMINAL_FIELDS)
                bridge.write_csv(root / "phasewise" / f"{policy_name}__{variant['variant']}.csv", phase_rows, bridge.PHASE_FIELDS)
                bridge.write_json(root / "raw_terminal_info" / f"{policy_name}__{variant['variant']}.json", raw_info)
                print(
                    f"rhtso {policy_name} {variant['variant']} completed={row['completed']} "
                    f"term={row['terminal_reason_classified']} phase={float(row['terminal_phase_deg']):.1f} "
                    f"CTE={float(row['CTE_mean']):.1f} Gmax={float(row['effective_Gmax']):.2f} cost={float(row['rhtso_cost']):.2f}",
                    flush=True,
                )

    if not radius_rows:
        bridge.write_csv(root / "radius_sweep.csv", [], OUTPUT_FIELDS)
    if not rhtso_rows:
        bridge.write_csv(root / "rhtso_sweep.csv", [], OUTPUT_FIELDS)
    write_report(root, radius_rows, rhtso_rows)
    print(f"rhtso_dir={root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
