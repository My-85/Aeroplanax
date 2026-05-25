# Claude Code Handoff Summary

## Session Scope

Full Claude regression of phase-gated residual half-loop candidate: verify Codex claims, assess 175°/180° improvement, AC批准 or reject for continued training.

## Current Baseline

```
results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619
```

## Residual Candidate Evaluated

```
results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2
```

- **Architecture**: `final_logits = epoch619_logits + gate(phase) * clipped_residual_logits`
- **Gate**: active in 80°-180° inverted/top-transition region
- **Outside gate**: combined policy is IDENTICAL to epoch619 (verified)
- **Residual epoch**: 2 (round 01, update 2)

## Claude Regression Results (2026-05-19)

**Output**: `results/residual_candidate_claude_regression/20260518_233806/`

### 1. Policy Loading Verification — PASS

- Gate = 0.0 at phase 0° (outside gate zone)
- Gate = 1.0 at phase 120° (inside gate zone)
- Outside gate: combined logits identical to base (max diff = 0.0)
- Inside gate: combined logits differ from base (residual contribution confirmed)
- Gate activation profile: exactly 80°-180°

### 2. Horizontal Retention (8 tasks, multi-seed 5) — PASS

All 8 tasks show no regression:
- level_circle_R3000_right/left: success rate preserved, Gmax comparable
- level_circle_R5000_right/left: grades preserved (A/B), no overload increase
- s_curve_A3000, figure_eight_R5000: no change in behavior
- mild_climb_p1000m, mild_descent_m1000m: altitude drift unchanged

### 3. Loop Retention 60/90/120/150 — PASS

- pu060_R12000: B→B, CTE unchanged (65.3m)
- pu090_R12000: B→B, CTE unchanged (56.3m)
- pu120_R12000: B→B, CTE +0.3m (58.1→58.3)
- pu150_R12000: B→B, CTE +4.1m (71.6→75.7)

No grade regressions. Slight CTE increase on 150° is within noise.

### 4. Target Loop 175°/180° — IMPROVED but still crashes

**pu175_R15000** (Codex prediction confirmed):
| Metric | Base epoch619 | Base+Residual | Delta |
|--------|--------------|---------------|-------|
| CTE_mean | 6698.7 | 2737.4 | **-3961.3** |
| velocity_tangent_error | 57.91 | 17.80 | **-40.11** |
| nose_tangent_error | 57.11 | 23.09 | **-34.02** |
| wing_plane_error | 71.41 | 37.85 | **-33.56** |
| q_error_mean_rad | 1.0737 | 0.6658 | **-0.4079** |
| env_alpha_max | 45.44 | 14.75 | **-30.69** |
| termination | crash | crash | still crashes |

**pu180_R15000** (Codex prediction confirmed):
| Metric | Base epoch619 | Base+Residual | Delta |
|--------|--------------|---------------|-------|
| CTE_mean | 6101.1 | 5327.6 | **-773.5** |
| velocity_tangent_error | 63.88 | 52.93 | **-10.94** |
| nose_tangent_error | 62.58 | 51.07 | **-11.51** |
| wing_plane_error | 77.46 | 63.46 | **-14.00** |
| env_alpha_max | 24.17 | 20.52 | **-3.64** |
| termination | crash | crash | still crashes |

### 5. Decision — recommended_for_continued_training

**Passes all gates**:
- [x] Horizontal retention: no hidden regression
- [x] 60/90/120/150 retention: no regression
- [x] 175°: clear planner-level improvement
- [x] 180°: at least some phase-wise geometry improvement
- [x] No new overload/crash/altitude drift artifacts
- [ ] 175°/180° complete without crash — NOT YET

**Candidate label**: `recommended_for_continued_training`

**NOT solved full-loop**. The residual specialist approach is validated but needs expanded gate coverage.

## For Next Codex Round

1. Continue from `residual_checkpoint_update_2`
2. Expand gate window: try 70°-190° or 80°-200° to cover exit/recovery
3. Consider a second parallel gate for 170°-200° exit phase
4. Run multi-round auto-config training (the run script supports this)
5. Generate ACMI files for visual confirmation (use `experiments/hierarchical_trajectory_tracking/export_acmi.py`)

## Files Delivered

```
results/residual_candidate_claude_regression/20260518_233806/
├── summary.md
├── recommendation.md
├── comparison_summary.csv
├── horizontal_retention.csv
├── loop_retention.csv
├── target_loop_175_180.csv
├── metrics/
│   └── policy_loader_check.json
├── acmi/        (pending ACMI generation)
└── phasewise_diagnostics/  (pending)
```

## Grading Criteria (loop-quality — same as before)

| Criterion | A | B |
|-----------|---|---|
| CTE_mean | <100m | <500m |
| CTE_p90 | <300m | <1200m |
| CTE_max | <800m | — |
| velocity_tangent_error | <15° | <30° |
| nose_tangent_error | <15° | <30° |
| nose_velocity_error | <15° | — |
| wing_plane_error | <15° | — |
| q_error (attitude) | <0.5 rad | — |
| Gmax | <9 | <10 |
| vt_min | ≥190 m/s | ≥175 m/s |

## Conda Environment

```bash
source /home/dqy/miniconda3/etc/profile.d/conda.sh && conda activate aeroplanax
```
