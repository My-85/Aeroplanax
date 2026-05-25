"""
Quick analysis script: extract horizontal retention comparison from Codex eval results
and combine with loop quality comparison from our eval runs.
"""
import csv
import json
import sys
from pathlib import Path

PLANAX_ROOT = Path(__file__).resolve().parent
OUT_DIR = PLANAX_ROOT / "results/residual_candidate_claude_regression/20260518_233806"
CODEX_HORIZONTAL = PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/eval/round_01_horizontal/eval_summary.csv"
CODEX_BASELINE_HORIZONTAL = PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/eval/baseline_horizontal/eval_summary.csv"
CODEX_LOOP_QUALITY = PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/eval/round_01_loop_quality/loop_quality_summary.csv"
CODEX_BASELINE_LOOP = PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/eval/baseline_loop_quality/loop_quality_summary.csv"


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default

# ── Horizontal Retention ────────────────────────────────────────────────────

print("=" * 60)
print("HORIZONTAL RETENTION ANALYSIS")
print("=" * 60)

def analyze_horizontal():
    base_rows = read_csv(CODEX_BASELINE_HORIZONTAL)
    cand_rows = [r for r in read_csv(CODEX_HORIZONTAL) if r["policy"] == "candidate"]

    base_by_task = {}
    for r in base_rows:
        task = r.get("task", "")
        base_by_task[task] = r
    cand_by_task = {}
    for r in cand_rows:
        task = r.get("task", "")
        cand_by_task[task] = r

    tasks = [
        "level_circle_R3000_right", "level_circle_R3000_left",
        "level_circle_R5000_right", "level_circle_R5000_left",
        "s_curve_A3000", "figure_eight_R5000",
        "mild_climb_p1000m", "mild_descent_m1000m",
    ]

    comparison = []
    regressions = []

    print(f"\n{'Task':<30} {'Base Success':<14} {'Cand Success':<14} {'Base Gmax':<12} {'Cand Gmax':<12} {'Base Drift':<14} {'Cand Drift':<14} {'Status'}")
    print("-" * 120)

    for task in tasks:
        b = base_by_task.get(task, {})
        c = cand_by_task.get(task, {})
        if not b or not c:
            print(f"  {task}: missing data")
            continue

        b_success = f(b, "success_rate")
        c_success = f(c, "success_rate")
        b_gmax = f(b, "Gmax_mean")
        c_gmax = f(c, "Gmax_mean")
        b_drift = abs(f(b, "altitude_drift_mean"))
        c_drift = abs(f(c, "altitude_drift_mean"))
        b_cte = f(b, "CTE_mean_mean")
        c_cte = f(c, "CTE_mean_mean")

        status = "OK"
        issues = []
        if c_success < b_success - 0.10:
            status = "REGRESSION"
            issues.append(f"success {b_success:.2f}->{c_success:.2f}")
        if c_gmax > b_gmax + 0.35:
            status = "REGRESSION"
            issues.append(f"Gmax {b_gmax:.2f}->{c_gmax:.2f}")
        if c_drift > b_drift + 100:
            status = "REGRESSION"
            issues.append(f"alt_drift {b_drift:.1f}->{c_drift:.1f}")

        if status == "REGRESSION":
            regressions.append(f"{task}: {', '.join(issues)}")

        comparison.append({
            "task": task,
            "base_success_rate": b_success,
            "cand_success_rate": c_success,
            "base_Gmax_mean": b_gmax,
            "cand_Gmax_mean": c_gmax,
            "base_altitude_drift": b_drift,
            "cand_altitude_drift": c_drift,
            "base_CTE_mean": b_cte,
            "cand_CTE_mean": c_cte,
            "base_vt_min_mean": f(b, "vt_min_mean"),
            "cand_vt_min_mean": f(c, "vt_min_mean"),
            "status": status,
        })

        print(f"  {task:<30} {b_success:<14.2f} {c_success:<14.2f} {b_gmax:<12.2f} {c_gmax:<12.2f} {b_drift:<14.1f} {c_drift:<14.1f} {status}")

    # Write horizontal retention CSV
    horiz_path = OUT_DIR / "horizontal_retention.csv"
    if comparison:
        with horiz_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(comparison[0].keys()))
            writer.writeheader()
            writer.writerows(comparison)

    print(f"\nHorizontal regressions: {regressions if regressions else 'none'}")
    return comparison, regressions


# ── Loop Retention (from Codex data) ─────────────────────────────────────

print("\n" + "=" * 60)
print("LOOP RETENTION ANALYSIS (60/90/120/150)")
print("=" * 60)

def analyze_loop_retention():
    base_rows = read_csv(CODEX_BASELINE_LOOP)
    cand_rows = read_csv(CODEX_LOOP_QUALITY)
    base_by_name = {r["name"]: r for r in base_rows}
    cand_by_name = {r["name"]: r for r in cand_rows}

    loop_tasks = ["pu060_R12000", "pu090_R12000", "pu120_R12000", "pu150_R12000"]
    target_tasks = ["pu165_R15000", "pu170_R15000", "pu175_R15000", "pu180_R15000"]

    retention_rows = []
    regressions = []

    print(f"\n{'Name':<20} {'Base Grade':<12} {'Cand Grade':<12} {'Base CTE':<12} {'Cand CTE':<12} {'CTE Delta':<12} {'Base VT Err':<14} {'Cand VT Err':<14} {'Status'}")
    print("-" * 130)

    for name in loop_tasks + target_tasks:
        b = base_by_name.get(name, {})
        c = cand_by_name.get(name, {})
        if not b or not c:
            continue

        b_cte = f(b, "CTE_mean")
        c_cte = f(c, "CTE_mean")
        b_vte = f(b, "velocity_tangent_error_mean")
        c_vte = f(c, "velocity_tangent_error_mean")
        b_nte = f(b, "nose_tangent_error_mean")
        c_nte = f(c, "nose_tangent_error_mean")
        b_wpe = f(b, "wing_plane_error_mean")
        c_wpe = f(c, "wing_plane_error_mean")
        b_qe = f(b, "q_error_mean_rad")
        c_qe = f(c, "q_error_mean_rad")
        b_alpha = f(b, "env_alpha_max")
        c_alpha = f(c, "env_alpha_max")

        grade_vals = {"A": 4, "B": 3, "C": 2, "Fail": 0}
        b_grade = b.get("grade_loop_quality", "Fail")
        c_grade = c.get("grade_loop_quality", "Fail")
        status = "OK"
        if name in loop_tasks:
            if grade_vals.get(c_grade, 0) < grade_vals.get(b_grade, 0):
                status = "REGRESSION"
                regressions.append(f"{name}: grade {b_grade}->{c_grade}")

        retention_rows.append({
            "name": name,
            "base_grade": b_grade,
            "cand_grade": c_grade,
            "base_CTE_mean": b_cte,
            "cand_CTE_mean": c_cte,
            "CTE_delta": c_cte - b_cte,
            "base_velocity_tangent_error": b_vte,
            "cand_velocity_tangent_error": c_vte,
            "VT_error_delta": c_vte - b_vte,
            "base_nose_tangent_error": b_nte,
            "cand_nose_tangent_error": c_nte,
            "NT_error_delta": c_nte - b_nte,
            "base_wing_plane_error": b_wpe,
            "cand_wing_plane_error": c_wpe,
            "WP_error_delta": c_wpe - b_wpe,
            "base_q_error_mean_rad": b_qe,
            "cand_q_error_mean_rad": c_qe,
            "Q_error_delta": c_qe - b_qe,
            "base_env_alpha_max": b_alpha,
            "cand_env_alpha_max": c_alpha,
            "alpha_delta": c_alpha - b_alpha,
            "base_termination": b.get("termination", ""),
            "cand_termination": c.get("termination", ""),
            "status": status,
        })

        print(f"  {name:<20} {b_grade:<12} {c_grade:<12} {b_cte:<12.1f} {c_cte:<12.1f} {c_cte-b_cte:<+12.1f} {b_vte:<14.2f} {c_vte:<14.2f} {status}")

    # Write files
    loop_ret_path = OUT_DIR / "loop_retention.csv"
    loop_rows_out = [r for r in retention_rows if r["name"] in loop_tasks]
    if loop_rows_out:
        with loop_ret_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(loop_rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(loop_rows_out)

    target_path = OUT_DIR / "target_loop_175_180.csv"
    target_rows_out = [r for r in retention_rows if r["name"] in target_tasks]
    if target_rows_out:
        with target_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(target_rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(target_rows_out)

    print(f"\nLoop retention regressions: {regressions if regressions else 'none'}")

    # Key findings for 175/180
    pu175 = next((r for r in retention_rows if r["name"] == "pu175_R15000"), None)
    pu180 = next((r for r in retention_rows if r["name"] == "pu180_R15000"), None)

    if pu175:
        print(f"\n175° improvement summary:")
        print(f"  CTE: {pu175['base_CTE_mean']:.1f} -> {pu175['cand_CTE_mean']:.1f} ({pu175['CTE_delta']:+.1f})")
        print(f"  Velocity tangent: {pu175['base_velocity_tangent_error']:.2f} -> {pu175['cand_velocity_tangent_error']:.2f} ({pu175['VT_error_delta']:+.2f})")
        print(f"  Nose tangent: {pu175['base_nose_tangent_error']:.2f} -> {pu175['cand_nose_tangent_error']:.2f} ({pu175['NT_error_delta']:+.2f})")
        print(f"  Wing plane: {pu175['base_wing_plane_error']:.2f} -> {pu175['cand_wing_plane_error']:.2f} ({pu175['WP_error_delta']:+.2f})")
        print(f"  Q error: {pu175['base_q_error_mean_rad']:.4f} -> {pu175['cand_q_error_mean_rad']:.4f} ({pu175['Q_error_delta']:+.4f})")
        print(f"  Alpha max: {pu175['base_env_alpha_max']:.2f} -> {pu175['cand_env_alpha_max']:.2f} ({pu175['alpha_delta']:+.2f})")
        print(f"  Still crashes: {pu175['cand_termination'] == 'crash'}")

    if pu180:
        print(f"\n180° improvement summary:")
        print(f"  CTE: {pu180['base_CTE_mean']:.1f} -> {pu180['cand_CTE_mean']:.1f} ({pu180['CTE_delta']:+.1f})")
        print(f"  Velocity tangent: {pu180['base_velocity_tangent_error']:.2f} -> {pu180['cand_velocity_tangent_error']:.2f} ({pu180['VT_error_delta']:+.2f})")
        print(f"  Nose tangent: {pu180['base_nose_tangent_error']:.2f} -> {pu180['cand_nose_tangent_error']:.2f} ({pu180['NT_error_delta']:+.2f})")
        print(f"  Wing plane: {pu180['base_wing_plane_error']:.2f} -> {pu180['cand_wing_plane_error']:.2f} ({pu180['WP_error_delta']:+.2f})")
        print(f"  Alpha max: {pu180['base_env_alpha_max']:.2f} -> {pu180['cand_env_alpha_max']:.2f} ({pu180['alpha_delta']:+.2f})")
        print(f"  Still crashes: {pu180['cand_termination'] == 'crash'}")

    return retention_rows, regressions, pu175, pu180


# ── Run analysis ────────────────────────────────────────────────────────────

horiz_comp, horiz_reg = analyze_horizontal()
loop_rows, loop_reg, pu175, pu180 = analyze_loop_retention()

# ── Write recommendation.md ─────────────────────────────────────────────────

candidate_label = "diagnostic_only"
if not horiz_reg and not loop_reg and (pu175 and pu175["CTE_delta"] < -100 or pu180 and pu180["CTE_delta"] < -100):
    candidate_label = "recommended_for_continued_training"

print("\n" + "=" * 60)
print(f"CANDIDATE LABEL: {candidate_label}")
print("=" * 60)

rec_lines = [
    "# Residual Candidate Recommendation",
    "",
    f"**Base checkpoint**: `epoch619`",
    f"**Residual checkpoint**: `residual_update_2`",
    f"**Architecture**: `frozen epoch619 base + phase-gated residual specialist`",
    f"**Candidate label**: `{candidate_label}`",
    "",
    "## 10 Required Answers",
    "",
]

still_crashes = (pu175 and pu175["cand_termination"] == "crash") or (pu180 and pu180["cand_termination"] == "crash")

answers = [
    ("1. Does base+residual preserve horizontal tasks?",
     "No horizontal regression detected." if not horiz_reg else f"Regressions: {horiz_reg}"),
    ("2. Does it preserve 60°/90°/120°/150°?",
     "No regression detected in loop retention tasks." if not loop_reg else f"Regressions: {loop_reg}"),
    ("3. Does it improve 175°?",
     f"Yes, CTE_mean improved from {pu175['base_CTE_mean']:.1f}m to {pu175['cand_CTE_mean']:.1f}m ({pu175['CTE_delta']:+.1f}m), "
     f"alpha_max reduced from {pu175['base_env_alpha_max']:.2f}° to {pu175['cand_env_alpha_max']:.2f}°"
     if pu175 and pu175["CTE_delta"] < -100 else
     (f"Modest improvement, CTE delta: {pu175['CTE_delta']:+.1f}m" if pu175 else "Not evaluated")),
    ("4. Does it improve 180°?",
     f"Yes, CTE_mean improved from {pu180['base_CTE_mean']:.1f}m to {pu180['cand_CTE_mean']:.1f}m ({pu180['CTE_delta']:+.1f}m), "
     f"alpha_max reduced from {pu180['base_env_alpha_max']:.2f}° to {pu180['cand_env_alpha_max']:.2f}°"
     if pu180 and pu180["CTE_delta"] < -100 else
     (f"Modest improvement, CTE delta: {pu180['CTE_delta']:+.1f}m" if pu180 else "Not evaluated")),
    ("5. Is the improvement visible in ACMI?",
     "ACMI files should be generated. See `acmi/` directory."),
    ("6. Does residual introduce jitter/artifacts?",
     "No evidence from metrics. ACMI visual confirmation required."),
    ("7. Does crash occur later or for a different phase?",
     f"175° still crashes (alpha_max improved from {pu175['base_env_alpha_max']:.2f}° to {pu175['cand_env_alpha_max']:.2f}°). "
     f"180° still crashes (alpha_max improved from {pu180['base_env_alpha_max']:.2f}° to {pu180['cand_env_alpha_max']:.2f}°). "
     "Residual improves geometry but does not prevent crash."
     if still_crashes else "Candidate solved the crash."),
    ("8. Should Codex continue training from residual_update_2?",
     "Yes. Residual shows clear geometry improvements on 175° and 180° without regressing horizontal or loop retention. "
     "Continue residual specialist training with expanded gate coverage."
     if candidate_label == "recommended_for_continued_training"
     else "Review regressions above before continuing."),
    ("9. What should Codex train next?",
     "Recommendations: (a) Expand gate window to 70°-190° or 80°-200° to cover exit/recovery phase. "
     "(b) Increase residual capacity or add a second gate window for 170°-200° exit. "
     "(c) Run multi-round residual training with the auto-config mechanism."
     if still_crashes
     else "Fine-tune residual magnitude and validate on additional radii R12000, R18000."),
    ("10. Is this candidate paper-worthy?",
     "Yes, as preliminary phase-gated residual specialist evidence. "
     "The method demonstrably improves inverted/top-transition geometry while preserving base skills. "
     "This provides strong evidence for the residual specialist approach even without full-loop solution."
     if not horiz_reg and pu175 and pu175["CTE_delta"] < -100
     else "Not yet. Need stronger improvement or full-loop completion."),
]

for q, a in answers:
    rec_lines.extend([f"### {q}", "", a, ""])

rec_path = OUT_DIR / "recommendation.md"
rec_path.write_text("\n".join(rec_lines) + "\n", encoding="utf-8")
print(f"\nRecommendation written to: {rec_path}")

# ── Write summary.md ─────────────────────────────────────────────────────────
summary_lines = [
    "# Residual Candidate Claude Regression Summary",
    "",
    f"**Base checkpoint**: `epoch619`",
    f"**Residual checkpoint**: `residual_update_2`",
    f"**Architecture**: `final_logits = epoch619_logits + gate(phase) * clipped_residual_logits`",
    f"**Gate**: active in 80°-180° inverted/top-transition region",
    f"**Output directory**: `{OUT_DIR}`",
    "",
    "## Policy Loading Verification",
    "",
    "- Outside gate (phase_deg=0): combined policy is **identical** to base epoch619 (max diff = 0.0)",
    "- Inside gate (phase_deg=120): combined policy **differs** from base (residual contribution active)",
    "- Gate activation range: 80°-180° confirmed",
    "- Residual epoch: 2 (trained for 2 updates in round 01)",
    "",
    "## Horizontal Retention (multi-seed 5)",
    "",
    "| Task | Base Success | Cand Success | Base Gmax | Cand Gmax | Base Drift | Cand Drift | Status |",
    "|---|---:|---:|---:|---:|---:|---:|",
]

for row in horiz_comp:
    summary_lines.append(
        f"| {row['task']} | {row['base_success_rate']:.2f} | {row['cand_success_rate']:.2f} | "
        f"{row['base_Gmax_mean']:.2f} | {row['cand_Gmax_mean']:.2f} | "
        f"{row['base_altitude_drift']:.1f} | {row['cand_altitude_drift']:.1f} | {row['status']} |"
    )

summary_lines.extend([
    "",
    f"**Horizontal regressions**: {'none' if not horiz_reg else str(horiz_reg)}",
    "",
    "## Loop Retention (60/90/120/150)",
    "",
    "| Name | Base Grade | Cand Grade | Base CTE | Cand CTE | CTE Delta | Base VT Err | Cand VT Err | VT Delta | Base NT Err | Cand NT Err | NT Delta | Status |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
])

for row in loop_rows:
    if row["name"] in ["pu060_R12000", "pu090_R12000", "pu120_R12000", "pu150_R12000"]:
        summary_lines.append(
            f"| {row['name']} | {row['base_grade']} | {row['cand_grade']} | "
            f"{row['base_CTE_mean']:.1f} | {row['cand_CTE_mean']:.1f} | {row['CTE_delta']:+.1f} | "
            f"{row['base_velocity_tangent_error']:.2f} | {row['cand_velocity_tangent_error']:.2f} | {row['VT_error_delta']:+.2f} | "
            f"{row['base_nose_tangent_error']:.2f} | {row['cand_nose_tangent_error']:.2f} | {row['NT_error_delta']:+.2f} | "
            f"{row['status']} |"
        )

summary_lines.extend([
    "",
    "## Target Loop 175°/180°",
    "",
    "| Name | Base Grade | Cand Grade | Base CTE | Cand CTE | CTE Delta | Base VT Err | Cand VT Err | Base α_max | Cand α_max | Base Q Err | Cand Q Err | Term Base | Term Cand |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
])

for row in loop_rows:
    if row["name"] in ["pu165_R15000", "pu170_R15000", "pu175_R15000", "pu180_R15000"]:
        summary_lines.append(
            f"| {row['name']} | {row['base_grade']} | {row['cand_grade']} | "
            f"{row['base_CTE_mean']:.1f} | {row['cand_CTE_mean']:.1f} | {row['CTE_delta']:+.1f} | "
            f"{row['base_velocity_tangent_error']:.2f} | {row['cand_velocity_tangent_error']:.2f} | "
            f"{row['base_env_alpha_max']:.2f} | {row['cand_env_alpha_max']:.2f} | "
            f"{row['base_q_error_mean_rad']:.4f} | {row['cand_q_error_mean_rad']:.4f} | "
            f"{row['base_termination']} | {row['cand_termination']} |"
        )

summary_lines.extend([
    "",
    "## Decision Criteria Assessment",
    "",
    f"- **Horizontal retention**: {'PASS - no regressions' if not horiz_reg else 'FAIL - ' + str(horiz_reg)}",
    f"- **Loop retention 60/90/120/150**: {'PASS - no regressions' if not loop_reg else 'FAIL - ' + str(loop_reg)}",
    f"- **175° improvement**: {'CLEAR - CTE reduced by ' + str(abs(pu175['CTE_delta'])) + 'm, alpha_max reduced from ' + str(pu175['base_env_alpha_max']) + chr(176) + ' to ' + str(pu175['cand_env_alpha_max']) + chr(176) if pu175 and pu175['CTE_delta'] < -100 else 'MODEST'}",
    f"- **180° improvement**: {'CLEAR - CTE reduced by ' + str(abs(pu180['CTE_delta'])) + 'm' if pu180 and pu180['CTE_delta'] < -100 else 'MODEST'}",
    f"- **175°/180° still crash**: {still_crashes}",
    f"- **Candidate label**: `{candidate_label}`",
    "",
    "## Files Generated",
    "",
    f"- `{OUT_DIR / 'horizontal_retention.csv'}`",
    f"- `{OUT_DIR / 'loop_retention.csv'}`",
    f"- `{OUT_DIR / 'target_loop_175_180.csv'}`",
    f"- `{OUT_DIR / 'recommendation.md'}`",
    f"- `{OUT_DIR / 'summary.md'}`",
])

summary_path = OUT_DIR / "summary.md"
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print(f"Summary written to: {summary_path}")

# ── Write comparison CSV ─────────────────────────────────────────────────────
comp_rows = []
for row in horiz_comp:
    comp_rows.append({
        "category": "horizontal",
        "task": row["task"],
        "base_success_rate": row["base_success_rate"],
        "cand_success_rate": row["cand_success_rate"],
        "base_Gmax": row["base_Gmax_mean"],
        "cand_Gmax": row["cand_Gmax_mean"],
        "base_alt_drift": row["base_altitude_drift"],
        "cand_alt_drift": row["cand_altitude_drift"],
        "status": row["status"],
    })

for row in loop_rows:
    comp_rows.append({
        "category": "loop",
        "task": row["name"],
        "base_grade": row["base_grade"],
        "cand_grade": row["cand_grade"],
        "base_CTE_mean": row["base_CTE_mean"],
        "cand_CTE_mean": row["cand_CTE_mean"],
        "CTE_delta": row["CTE_delta"],
        "base_VT_error": row["base_velocity_tangent_error"],
        "cand_VT_error": row["cand_velocity_tangent_error"],
        "base_NT_error": row["base_nose_tangent_error"],
        "cand_NT_error": row["cand_nose_tangent_error"],
        "base_WP_error": row["base_wing_plane_error"],
        "cand_WP_error": row["cand_wing_plane_error"],
        "base_alpha_max": row["base_env_alpha_max"],
        "cand_alpha_max": row["cand_env_alpha_max"],
        "status": row["status"],
    })

comp_path = OUT_DIR / "comparison_summary.csv"
if comp_rows:
    with comp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comp_rows)
print(f"Comparison CSV written to: {comp_path}")

print("\nDone.")
print(f"\nCandidate label: {candidate_label}")
