# Codex Handoff Summary

## 0. Latest Update - 2026-05-18 Corrected Specialist Residual V1

Current best baseline remains:

`results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`

Do not replace epoch619 with a monolithic checkpoint. The new candidate is a residual specialist that must be evaluated as:

`epoch619 base checkpoint + residual_checkpoint_update_2`

Corrected residual run:

`results/half_loop_specialist_residual_v1/20260518_1803/`

Best residual candidate:

`results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2`

Base checkpoint:

`results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`

Architecture:

`frozen epoch619 + phase-gated residual logits policy`

Reason: the existing policy uses five discrete categorical action heads, so the practical Option A is:

`final_logits = epoch619_logits + gate(phase) * clipped_residual_logits`

The residual gate is active only in the 80-180 degree inverted/top-transition region. Outside that region, including horizontal tasks, the final action distribution is epoch619 by construction.

Final decision for corrected run:

- `gate_pass`: true.
- Better than epoch619 as a residual candidate: yes.
- Horizontal regressions: none.
- 60/90/120/150 loop retention regressions: none.
- Overload increase: none.
- Altitude drift regressions: none.
- 175/180 geometry improved: yes.
- Claude full ACMI regression recommended: yes.

Important: 175/180 still terminate with crash/fail, so this is not a final solved full-loop policy. It is the first residual candidate that passes Codex's promotion gate and should be sent to Claude for full planner/ACMI regression.

Corrected run target-loop deltas:

- `pu175_R15000`
  - `CTE_mean`: 6698.7 -> 2737.4
  - `velocity_tangent_error`: 57.91 -> 17.80
  - `nose_tangent_error`: 57.11 -> 23.09
  - `wing_plane_error`: 71.41 -> 37.85
  - `q_error_mean_rad`: 1.0737 -> 0.6658
  - `env_alpha_max`: 45.44 -> 14.75
  - termination: still crash

- `pu180_R15000`
  - `CTE_mean`: 6101.1 -> 5327.6
  - `velocity_tangent_error`: 63.88 -> 52.93
  - `nose_tangent_error`: 62.58 -> 51.07
  - `wing_plane_error`: 77.46 -> 63.46
  - `q_error_mean_rad`: essentially unchanged by scorer threshold
  - `env_alpha_max`: 24.17 -> 20.52
  - termination: still crash

Training gate diagnostics for the corrected formal round:

`results/half_loop_specialist_residual_v1/20260518_1803/train_log.csv`

- `gate_rate_mean`: 0.51842773
- `loop_mode_rate_mean`: 0.87564063
- `mode5_rate_mean`: 0.54606843
- `mode9_rate_mean`: 0.32957229
- `mode5_gate_rate_mean`: 0.34462374
- `mode9_gate_rate_mean`: 1.00000000
- `loop_phase_mean`: 92.1773
- `loop_phase_max`: 179.9972

Root cause fixed:

The earlier residual run `results/half_loop_specialist_residual_v1/20260518_1104/` was effectively invalid as a training result because `AeroPlanaxHeading_Pitch_V_Env(env_params)` accepted configured params, but `LogWrapper.reset/step` called the underlying env without passing params, and the env's `default_params` property returned fresh default params. Therefore `half_loop_curriculum_prob=1.0` and the residual curriculum did not actually take effect.

Fix:

- `envs/aeroplanax_heading_pitch_V_quaternion_version_vertical_energy.py` now stores constructor params in `_configured_params`.
- `default_params` now returns `_configured_params`, preserving residual curriculum params through `LogWrapper.reset/step`.

Validation after fix:

- Gate activation smoke:
  `results/half_loop_specialist_residual_v1_gate_debug/20260518_175204/`
  - transition-only forced-success smoke reached `max_gate_rate=1.0`
  - first gate step: 1

- GPU training smoke:
  `results/half_loop_specialist_residual_v1_smoke/20260518_1800/`
  - `gate_rate_mean=0.51269531`
  - `loop_mode_rate_mean=0.69824219`
  - `mode9_gate_rate_mean=1.00000000`

New diagnostic file:

- `check_half_loop_residual_gate_activation.py`

New files:

- `half_loop_residual_policy.py`
- `train_half_loop_specialist_residual_v1.py`
- `run_half_loop_specialist_residual_v1.py`
- `paper/second_paper/half_loop_specialist_residual_v1_config.json`

Evaluators extended for residual candidates:

- `eval_loop_quality_claude_aligned.py`
- `eval_vertical_energy_checkpoints.py`

Do not promote old residual no-op run:

`results/half_loop_specialist_residual_v1/20260518_1104/`

Reason:

- It was run before the env-param fix.
- Its `train_log.csv` had `gate_rate_mean = 0.00000000`.
- Its residual checkpoint remained a no-op and must not be sent to Claude.

Next recommended action:

- Ask Claude to run full ACMI / planner-level regression on the pair:
  - base: `results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`
  - residual: `results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2`
- If Claude confirms no hidden horizontal/ACMI regression, continue from this residual candidate.
- If Claude finds residual-specific artifacts, keep epoch619 as baseline and use this candidate only as a diagnostic direction.

## Previous Update - 2026-05-17

Current best baseline remains:

`results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`

Do not promote or continue from:

- `results/half_loop_inverted_transition_v2/20260517_1800/checkpoint/round_01/checkpoint/checkpoint_epoch_621`
- `results/half_loop_inverted_transition_v2/20260517_1800/checkpoint/round_02/checkpoint/checkpoint_epoch_621`
- `results/half_loop_inverted_transition_search/20260517_0134/checkpoints/round_01/checkpoint/checkpoint_epoch_622`
- `checkpoint_epoch_658`
- `checkpoint_epoch_628`
- `checkpoint_epoch_632`

The latest v2 search did not find a checkpoint better than epoch619. The round_01 and round_02 `checkpoint_epoch_621` files are diagnostics only.

## 1. Evaluator Alignment

Codex now has a Claude-aligned planner-level loop-quality evaluator:

`eval_loop_quality_claude_aligned.py`

It reports:

- `CTE_mean`, `CTE_p90`, `CTE_max`
- `velocity_tangent_error`
- `nose_tangent_error`
- `nose_velocity_error`
- `wing_plane_error`
- `q_error_norm`
- `env_alpha` / `env_beta` range
- `target_roll` vs actual roll range
- `vt_min`, `vt_mean`, `vt_max`
- `Gmax`, `Gmean`
- `termination_reason`

Step 0 alignment was run on epoch619:

`results/codex_eval_alignment_epoch619/20260517_172109/`

Important result:

- `report.md` says `alignment_status: pass`.
- `comparison_to_claude.csv` matches Claude's official epoch619 loop-quality report within numerical tolerance.
- Epoch619 is B on 60/90/105/120/135/150 but Fail at 180 under Claude planner-level loop-quality metrics.
- Therefore target-level B grades for 175/180 must not be used as promotion evidence by themselves.

Claude official/codex-aligned epoch619 180-degree metrics:

- `CTE_mean`: 6101.1
- `velocity_tangent_error`: 63.88
- `nose_tangent_error`: 62.58
- `nose_velocity_error`: 18.89
- `wing_plane_error`: 77.46
- `q_error_norm`: 0.893
- `termination_reason`: crash

## 2. Latest V2 Training Attempt

New conservative branch:

`half_loop_inverted_transition_v2`

Config:

`paper/second_paper/half_loop_inverted_transition_v2_config.json`

Runner:

`run_half_loop_inverted_transition_v2.py`

The runner starts only from epoch619, evaluates horizontal proxy and Claude-aligned loop quality, applies promotion gates, and stops early after two rounds with no useful improvement.

Latest run:

`results/half_loop_inverted_transition_v2/20260517_1800/`

Outputs:

- `train_log.csv`
- `baseline_eval_horizontal.csv`
- `baseline_eval_loop_quality.csv`
- `eval_horizontal.csv`
- `eval_loop_quality.csv`
- `score_report.json`
- `search_summary.csv`
- `final_report.md`
- `checkpoint/round_01/checkpoint/checkpoint_epoch_621`
- `checkpoint/round_02/checkpoint/checkpoint_epoch_621`

Final decision:

- Better than epoch619: no.
- Promoted checkpoint: none.
- Claude full ACMI regression recommended: no.
- Stop reason: two short rounds showed no useful improvement.

## 3. V2 Round Results

Round 1 checkpoint:

`results/half_loop_inverted_transition_v2/20260517_1800/checkpoint/round_01/checkpoint/checkpoint_epoch_621`

Status: diagnostic only, not promoted.

Reason:

- Some 175/180 geometry metrics improved under planner-level loop-quality eval.
- Horizontal gate failed:
  - `level_circle_R3000_right:Gmax 8.43->9.33`
  - `level_circle_R3000_left:success 0.60->0.20`
  - `mild_climb_p1000m:Gmax 7.41->8.79`

Round 2 checkpoint:

`results/half_loop_inverted_transition_v2/20260517_1800/checkpoint/round_02/checkpoint/checkpoint_epoch_621`

Status: diagnostic only, not promoted.

Reason:

- 60/90/150 loop retention stayed stable.
- 175/180 geometry had partial improvements but still did not complete cleanly and still hit crash/fail behavior.
- Horizontal gate failed:
  - `level_circle_R3000_left:crash 0.00->0.20`
  - `mild_climb_p1000m:Gmax 7.41->8.55`

Round 2 target-loop improvement flags:

- `pu175_R15000`: nose tangent improved, velocity tangent improved, q error improved; wing-plane and alpha did not improve.
- `pu180_R15000`: wing-plane improved, nose tangent improved, velocity tangent improved; q error and alpha did not improve.

This is not promotable because the official gate requires horizontal stability, no overload increase, and no crash regression.

## 4. Current Best Baseline

Use only:

`results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`

Status:

- Current best main baseline.
- Best balanced checkpoint from prior Claude planner-level regression.
- Must remain the reference for future conservative experiments.

Do not replace it with ep621, ep622, ep658, ep628, or ep632.

## 5. Historical Diagnostic Checkpoints

- `checkpoint_epoch_658`
  - Status: do not use as baseline.
  - Reason: target-level pull-up looked better, but Claude planner-level regression showed horizontal degradation.

- `checkpoint_epoch_628`
  - Path: `results/vertical_energy_balanced_finetune_v2/20260516_012725_cycle_01/checkpoint/checkpoint_epoch_628`
  - Status: diagnostic only.
  - Reason: vertical energy improved, but planner proxy failed; R3000 circle regressed and overload appeared.

- `checkpoint_epoch_632`
  - Path: `results/altitude_retention_repair/20260516_0146/checkpoint/checkpoint_epoch_632`
  - Status: failed repair diagnostic only.
  - Reason: partially repaired ep628 but still failed R3000 circle / altitude retention gate.

- `checkpoint_epoch_622`
  - Path: `results/half_loop_inverted_transition_search/20260517_0134/checkpoints/round_01/checkpoint/checkpoint_epoch_622`
  - Status: diagnostic only, not promoted.
  - Reason: earlier corrected target-level eval showed B-grade 175/180, but Claude planner-level loop-quality metrics are the official gate and horizontal retention regressed.

## 6. Evaluation And Training Files Added

New or extended files:

- `eval_loop_quality_claude_aligned.py`
- `eval_vertical_energy_checkpoints.py`
- `paper/second_paper/half_loop_inverted_transition_v2_config.json`
- `run_half_loop_inverted_transition_v2.py`

`eval_vertical_energy_checkpoints.py` now includes a `horizontal_v2` suite with:

- `level_circle_R3000_right`
- `level_circle_R3000_left`
- `level_circle_R5000_right`
- `level_circle_R5000_left`
- `s_curve_A3000`
- `figure_eight_R5000`
- `mild_climb_p1000m`
- `mild_descent_m1000m`

## 7. What Worked

- The Claude-aligned evaluator now matches the official epoch619 loop-quality report.
- GPU training/eval works with:

  `CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda MPLCONFIGDIR=/tmp WANDB_MODE=offline`

- The v2 runner can perform baseline eval, short-round training, horizontal proxy eval, planner-level loop-quality eval, gate scoring, reports, and early stopping.
- 60/90/150 loop-plane retention remained stable in the latest v2 run.
- Some 175/180 geometry terms improved in short diagnostics, especially nose/velocity tangent errors.

## 8. What Failed

- No v2 checkpoint passed the promotion gate.
- Horizontal stability remains the blocker.
- R3000 left is still fragile under even conservative inverted-transition training.
- Mild climb overload worsened in both v2 rounds.
- 175/180 improvements were not complete: crash/fail behavior remained, and alpha/q/wing metrics were mixed.

## 9. Next Recommended Direction

Do not increase full 175/180 pressure yet.

Next attempt should start from epoch619 again and be more protective:

- Increase horizontal proxy to at least 50%.
- Make R3000 left/right and climb/descent explicit high-weight retention tasks.
- Reduce or zero 175/180 direct sampling for the first round.
- Prefer only short transition snippets such as 90->120, 120->150, and 135->165.
- Consider lowering LR below `1e-5`.
- Strengthen overload and climb/descent G penalties before adding more inverted pressure.
- Keep Claude-aligned loop-quality eval as the only official vertical geometry gate.

Promotion remains allowed only if all are true:

- horizontal proxy does not regress
- R3000 left/right do not regress
- figure-eight does not regress
- 60/90/150 do not regress
- 175 or 180 improves in Claude loop-quality geometry
- wing-plane, nose-tangent, and velocity-tangent errors improve
- no overload increase
- no altitude drift regression

## 10. What Not To Do

- Do not use ep621, ep622, ep658, ep628, or ep632 as the main baseline.
- Do not continue from any v2 round checkpoint unless explicitly doing a diagnostic repair.
- Do not hand ep621 or ep622 to Claude as a promoted ACMI checkpoint.
- Do not rely on target-level vertical success alone.
- Do not train full loop, barrel roll, Immelmann, or Split-S.
- Do not run training or eval on CPU.
- Do not use old roll=0 targets for >90-degree vertical arc or half-loop tasks.

## 11. 2026-05-19 Residual Exit/Recovery Continuation

User instruction for this continuation:

- Do not train monolithic ep619.
- Keep `checkpoint_epoch_619` frozen as base.
- Continue only from residual `results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2`.
- Focus on post-top recovery tasks: `160->180`, `170->190`, `175->200`, `180->210`.
- Keep horizontal and 60/90/120/150 retention gates strict.

Important GPU correction:

- Numeric `CUDA_VISIBLE_DEVICES=1` was not reliable in this environment because imported helper modules could override the variable or CUDA ordinals were confusing relative to `nvidia-smi`.
- Verified physical GPU 1 UUID:

  `GPU-2c45b7fd-69c8-1697-23a0-fe7ce7a2a620`

- Use this for future GPU 1 runs:

  `CUDA_VISIBLE_DEVICES=GPU-2c45b7fd-69c8-1697-23a0-fe7ce7a2a620`

- Fixed `experiments/hierarchical_trajectory_tracking/render_ablation_tests.py` so it uses `os.environ.setdefault(...)` instead of unconditionally forcing `CUDA_VISIBLE_DEVICES=0`.

Evaluator/runner fixes made:

- `eval_loop_quality_claude_aligned.py`
  - Added phase-wise output fields to `FIELDNAMES`.
  - Added `--only-names` for quick screens such as `pu150/pu165/pu175/pu180`.
- `run_residual_gate_window_ablations.py`
  - Added explicit child-process `CUDA_VISIBLE_DEVICES` reporting/override.
  - Honors high-memory JAX preallocation env vars.
- `run_half_loop_specialist_residual_v1.py`
  - Honors configured `RESIDUAL_LOADDIR` instead of always starting residual from scratch.
  - Honors high-memory JAX preallocation env vars.
  - Added exit_v2 eval hook, though the final continuation used direct train/eval for speed.

Inference-only gate-window ablation status:

- Completed earlier at `results/half_loop_specialist_residual_v1_gate_window_ablations/20260519_0115/`.
- Main readout:
  - `100..200` is bad and should be excluded.
  - `150..210` gives strong 175/180 geometry in inference but is bad beyond 185.
  - `90..200` preserves 150 well and improves 175, but does not solve 200/210.
  - `80..180` remains the safest default but does not solve crashes.

Training attempts from `residual_update_2`:

1. Exit-heavy run:
   - Config: `paper/second_paper/half_loop_specialist_residual_exit_recovery_v1_config.json`
   - Run dir: `results/half_loop_specialist_residual_exit_recovery_v1/20260519_1633/`
   - Checkpoint: `results/half_loop_specialist_residual_exit_recovery_v1/20260519_1633/checkpoint/residual_checkpoint_update_4`
   - Gate: `150..210`, scale `1.0`, LR `7.5e-6`
   - horizontal_v2: passed; candidate rows matched baseline rows exactly.
   - v2 partial:
     - 60/90/120/150 remained ok.
     - `pu165_R15000` failed: crash, CTE 7330.7, velocity tangent 63.05, nose tangent 63.73, wing-plane 78.25.
   - Decision: do not promote, do not send to Claude, do not continue from this checkpoint.

2. Conservative restart:
   - Config: `paper/second_paper/half_loop_specialist_residual_exit_recovery_v1_conservative_config.json`
   - Run dir: `results/half_loop_specialist_residual_exit_recovery_v1_conservative/20260519_1741/`
   - Checkpoint: `results/half_loop_specialist_residual_exit_recovery_v1_conservative/20260519_1741/checkpoint/residual_checkpoint_update_4`
   - Gate: `90..200`, scale `0.5`, clip `0.75`, LR `3e-6`
   - Quick screen:
     - `pu150_R12000`: ok/A, CTE 68.2, velocity tangent 3.73, nose tangent 4.92, wing-plane 14.75.
     - `pu165_R15000`: crash/Fail, CTE 6819.6, velocity tangent 64.06, nose tangent 62.42, wing-plane 86.42.
   - Decision: do not promote, do not send to Claude, do not continue from this checkpoint.

Current best remains:

`base epoch619 + residual_update_2`

Do not promote or continue from either new `residual_checkpoint_update_4`.

Next recommended direction:

- Stop pushing exit/recovery directly until 165 is stabilized.
- Add an explicit 150->165 / 155->170 bridge stage before any 175/180/200 pressure.
- Use very small residual scale (`0.25` or `0.5`) and lower LR (`1e-6` to `2e-6`).
- Add behavior-cloning/KL-to-residual_update_2 inside 90..150 and 150..165 so the entry geometry is not overwritten.
- Consider training only on short 150->165 and 160->175 snippets first, with exit tasks disabled or below 10%.
- Only after `pu165_R15000` is no worse than residual_update_2 should 170/175/180/200 recovery be reintroduced.

## 12. 2026-05-20 Residual Auto-Search Bridge Repair

User instruction for this continuation:

- Do not train monolithic ep619.
- Keep base `checkpoint_epoch_619` frozen.
- Start only from residual `residual_checkpoint_update_2`.
- Use physical GPU 1 UUID:

  `CUDA_VISIBLE_DEVICES=GPU-2c45b7fd-69c8-1697-23a0-fe7ce7a2a620`

- Focus on the 150->165 / 155->170 bridge before any exit/recovery or full-loop pressure.

Code/evaluator changes made:

- `half_loop_residual_policy.py`
  - Added smooth residual gate support via `RESIDUAL_SMOOTH_GATE_MARGIN_DEG`.
- `eval_loop_quality_claude_aligned.py`
  - Added smooth-gate-aware residual gate evaluation.
  - Kept `--only-names` quick-screen support.
- `envs/aeroplanax_heading_pitch_V_quaternion_version_vertical_energy.py`
  - Added bridge transition sampling:
    - `half_loop_bridge_transition_prob`
    - `half_loop_partial_bridge_prob`
    - bridge starts around 150/155/160 and ends around 165/170/175.
- `train_half_loop_specialist_residual_v1.py`
  - Added residual-update-2 behavior cloning anchor:
    - `ANCHOR_RESIDUAL_LOADDIR`
    - `ANCHOR_BC_COEF`
    - `ANCHOR_PHASE_START_DEG`
    - `ANCHOR_PHASE_END_DEG`
  - Added `final_anchor_bc` and `anchor_residual_checkpoint` to `train_log.csv`.
- `run_half_loop_residual_auto_search.py`
  - New auto-search runner for residual-only bridge repair.
  - Uses GPU UUID above.
  - Keeps ep619 frozen and starts from update_2.
  - Enforces bridge no-crash gate for `pu165_R15000` and `pu170_R15000`.
  - Rejects candidates with Gmax increase.
  - Adds eval timeout handling and early stop after two rounds without strict-gate improvement.

Search run:

- Root: `results/half_loop_residual_auto_search/20260520_1146/`
- Final report: `results/half_loop_residual_auto_search/20260520_1146/final_report.md`
- Best manifest: `results/half_loop_residual_auto_search/20260520_1146/best_candidate_manifest.json`

Important correction:

- An earlier pre-fix runner briefly printed `promoted=True` for scale-only 0.25.
- This was invalid because scale-only 0.25 still crashed on `pu165_R15000` and `pu170_R15000`, and increased Gmax.
- The root report and manifest were corrected. Final decision is `no_promotion`.

Baseline quick metrics for current best update_2:

- `pu150_R12000`: ok/B, CTE 75.7, wing 15.1, Gmax 5.84
- `pu165_R15000`: crash/Fail, CTE 7884.8, velocity tangent 62.8, nose tangent 61.5, wing 79.5, Gmax 7.55
- `pu170_R15000`: crash/Fail, CTE 6968.9, velocity tangent 61.6, nose tangent 59.4, wing 77.1, Gmax 7.53

Scale-only diagnostic:

- `scale=0.25`:
  - `pu150_R12000`: ok/A, CTE 69.8, wing 14.8
  - `pu165_R15000`: crash/Fail, CTE 3463.2, wing 38.8, Gmax 9.04
  - `pu170_R15000`: crash/Fail, CTE 2530.0, wing 33.4, Gmax 10.02
- Interpretation:
  - The residual is likely over-powered near 150-175.
  - Lower scale improves geometry strongly but still crashes and increases Gmax.
  - Not promotable.

Trained residual-only candidates:

1. Round 1 Family A:
   - Checkpoint: `results/half_loop_residual_auto_search/20260520_1146/round_01/candidates/family_A_bridge_strong_anchor_absfix/checkpoint/residual_checkpoint_update_3`
   - pu150: ok/B, CTE 71.6
   - pu165: timeout/missing
   - Decision: reject.

2. Round 1 Family B:
   - Checkpoint: `results/half_loop_residual_auto_search/20260520_1146/round_01/candidates/family_B_bridge_mild_extension_absfix/checkpoint/residual_checkpoint_update_3`
   - pu150: ok/A, CTE 70.9
   - pu165: timeout/missing
   - Decision: reject.

3. Round 2 Family A:
   - Checkpoint: `results/half_loop_residual_auto_search/20260520_1146/round_02/candidates/round_2_mut_A_residual_overpower/checkpoint/residual_checkpoint_update_3`
   - pu150: ok/B, CTE 71.6
   - pu165: timeout/missing
   - Decision: reject.

4. Round 2 Family B:
   - Checkpoint: `results/half_loop_residual_auto_search/20260520_1146/round_02/candidates/round_2_mut_B_residual_overpower/checkpoint/residual_checkpoint_update_3`
   - pu150: ok/A, CTE 69.7
   - pu165: crash/Fail, CTE 7127.7, wing 83.7, Gmax 8.74
   - Decision: reject.

Final decision:

- No candidate beat residual_update_2 under strict gates.
- Do not send any new checkpoint from this run to Claude for ACMI regression.
- Current best remains:

  `base checkpoint_epoch_619 + residual_checkpoint_update_2`

Next recommended direction:

- Continue bridge repair, not exit/recovery.
- Make `pu165_R15000` the only hard next objective before adding `pu170/pu175`.
- Use residual scale `<=0.25`.
- Add stronger anti-over-G and alpha shaping in the 145-170 phase band.
- Train only `150->165` and `155->165/170` snippets first.
- Keep update_2 behavior-cloning anchor but allow localized correction around 150-165.
- Require `pu165_R15000` no-crash before reintroducing 170/175/180/200 recovery tasks.

## 13. 2026-05-21 Bridge Micro-Search Diagnostics

User instruction for this step:

- Do not train monolithic ep619.
- Do not train any residual candidate until diagnostics explain why scale `0.25` improves geometry but still fails `pu165`.
- Keep current best:
  - base `results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`
  - residual `results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2`
- Use GPU UUID `GPU-2c45b7fd-69c8-1697-23a0-fe7ce7a2a620`.
- Stop residual-only bridge training if `pu165_R15000` cannot become no-crash with Gmax < 9.

New runner:

- `run_half_loop_bridge_micro_search.py`
  - Adds detailed phasewise diagnostics for base, update_2 scale 1.0, update_2 scale 0.25, update_2 scale 0.125.
  - Adds cached scale sweep, focused gate sweep, and target-stream sweep.
  - Writes incremental CSVs so long diagnostics are not lost.
  - Keeps ep619 frozen and performs inference-only diagnostics unless training is explicitly added later.

Run root:

- `results/half_loop_bridge_micro_search/20260520_phase_diag/`

Outputs:

- `diagnosis.md`
- `final_report.md`
- `phasewise_diagnostics.csv`
- `scale_sweep.csv`
- `gate_sweep.csv`
- `target_stream_sweep.csv`
- `target_stream_sananifest.json`
- `best_candidate_manifest.json`
- `phasewise/`
- `rounds/README.md`

Scale sweep result on `pu165_R15000`:

- `scale=0.05`: `done_unknown`, CTE 6655.5, wing 84.7, Gmax 8.16
- `scale=0.10`: `done_unknown`, CTE 2580.1, wing 35.1, Gmax 6.32
- `scale=0.125`: `done_unknown`, CTE 5852.1, wing 70.7, Gmax 8.17, alpha_max 22.86
- `scale=0.20`: `done_unknown`, CTE 2122.6, wing 28.3, Gmax 7.04
- `scale=0.25`: `done_unknown`, CTE 3463.2, wing 38.8, Gmax 9.04
- `scale=0.35`: `done_unknown`, CTE 7450.8, wing 83.3, Gmax 8.65

Interpretation:

- `scale=0.20` is the best safe inference geometry point for `pu165`.
- It keeps Gmax < 9 and preserves `pu150`, but still does not complete `pu165`.
- `scale=0.25` remains not promotable because Gmax crosses 9 and completion still fails.
- Scale-only inference cannot promote a checkpoint.

Gate sweep:

- Focused gate sweep tested `scale=0.20`, `pu165_R15000`, windows `80-180`, `140-175`, `145-170`, `150-170`, margins `10/20`.
- Rerun rows ended early at `steps=174`, `completed=False`, `termination=done_unknown`, CTE 249.1, Gmax 5.84, action diff 0.0.
- This is not a success. Residual was inactive before termination, so these rows are a termination/evaluator diagnostic.

Target-stream sweep:

- Tested target_vt `220/240/250/260` and lookahead `conservative/default/relaxed` with `scale=0.20`.
- No setting completed `pu165_R15000`.
- Best remained `250/default`: CTE 2122.6, wing 28.3, Gmax 7.04, but still `done_unknown`.
- Lower speed or conservative/relaxed lookahead generally reduced geometry quality or pushed G close to 10.

Final decision:

- No training was run after diagnostics.
- No checkpoint was promoted.
- No candidate should be sent to Claude ACMI regression.
- Current best remains `checkpoint_epoch_619 + residual_checkpoint_update_2`.

Immediate blocker:

- The evaluator records many failures as `done_unknown` because env auto-reset hides the pre-reset termination state.
- Before more training, fix evaluator/runner to capture true final-step termination reason and final pre-reset phase/action/aircraft state.

Next recommended direction:

- Do not reintroduce `pu170/pu175/pu180` yet.
- Keep `pu165_R15000` as the only hard objective.
- First fix termination tracing.
- Then test load-factor-aware bridge target shaping: larger radius or curvature/load-factor limited 145-170 target stream, combined with residual scale around `0.20`.
- Do not continue residual-only PPO blind training until `pu165` can be evaluated without `done_unknown` ambiguity.

## 14. 2026-05-22 Termination Trace Fix And Target-Stream Tests

User instruction for this step:

- Do not train residual PPO until termination tracing is fixed.
- Keep current best as `checkpoint_epoch_619 + residual_checkpoint_update_2`.
- Classify previous `done_unknown` cases using true pre-reset terminal state.
- Then test load-factor-aware bridge target streams before considering any residual training.

Code changes:

- `envs/aeroplanax.py`
  - `step()` now stores pre-reset terminal state and done flags in `info`:
    - `terminal_state_before_reset`
    - `terminal_dones_before_reset`
    - `terminal_env_done_before_reset`
    - `terminal_success_before_reset`
- `termination_trace_utils.py`
  - New helper for terminal-state extraction and classification.
  - Classifies terminal reason as `success`, `crash`, `overload`, `low_speed`, `timeout`, `altitude_limit`, `nan_or_invalid`, `unknown_done`, etc.
- `eval_loop_quality_claude_aligned.py`
  - Uses pre-reset terminal state for terminal reason classification.
- `run_half_loop_bridge_micro_search.py`
  - Uses the same fixed terminal classification.
- `run_half_loop_termination_trace.py`
  - New diagnostic runner for fixed terminal trace output.
- `run_load_factor_aware_bridge_target_stream.py`
  - New diagnostic runner for target-stream variants.

Termination trace run:

- Root: `results/half_loop_bridge_termination_trace/20260522_termination_fixed/`
- Report: `results/half_loop_bridge_termination_trace/20260522_termination_fixed/termination_trace_report.md`
- CSVs:
  - `scale_sweep_fixed.csv`
  - `terminal_states.csv`
  - `phasewise/`
  - `raw_terminal_info/`

Key fixed-trace results:

- `base_only / pu165_R15000`:
  - true reason `timeout`
  - terminal phase `103.2°`
  - CTE `7752.8`
  - Gmax `6.66`
- `update2_scale1.0 / pu165_R15000`:
  - true reason `timeout`
  - terminal phase `99.4°`
  - CTE `7593.8`
  - Gmax `8.67`
- `update2_scale0.2 / pu165_R15000`:
  - true reason `timeout`
  - terminal phase `107.5°`
  - CTE `7948.5`
  - Gmax `7.76`
- `update2_scale0.25 / pu165_R15000`:
  - true reason `timeout`
  - terminal phase `105.7°`
  - CTE `7958.7`
  - Gmax `7.76`
- `update2_scale0.2 / pu170_R15000`:
  - true reason `overload`
  - terminal phase `127.5°`
  - Gmax `10.15`

Important corrected interpretation:

- Previous `done_unknown` was mainly env timeout or overload hidden by auto-reset.
- `update2_scale0.2 / pu165` is not close to a 165° bridge success.
- It times out around phase `107.5°`, before the 145-170 bridge band.
- Residual-only PPO around 150-165 is not currently meaningful because the rollout does not reliably reach that phase.

Target-stream diagnostic runs:

- Main scale0.20 hard-task run:
  - `results/load_factor_aware_bridge_target_stream/20260522_scale02_hard/`
- Supplemental base/scale1 run:
  - `results/load_factor_aware_bridge_target_stream/20260522_policy_supplement/`

Scale0.20 target-stream variants on `pu165_R15000`:

- `baseline_target_stream`:
  - timeout, phase `107.5°`, CTE `7948.5`, Gmax `7.76`
- `larger_radius_bridge`:
  - timeout, phase `139.8°`, CTE `10509.4`, Gmax `9.26`
- `curvature_limited_bridge`:
  - low_speed, phase `132.5°`, CTE `4717.9`, Gmax `8.63`
- `load_factor_limited_bridge`:
  - low_speed, phase `160.5°`, CTE `1469.7`, Gmax `10.39`
- `vt_scheduled_bridge`:
  - timeout, phase `107.1°`, CTE `7764.4`, Gmax `8.58`
- `lookahead_scheduled_bridge`:
  - timeout, phase `121.7°`, CTE `10336.9`, Gmax `7.38`
- `pitch_rate_limited_bridge`:
  - timeout, phase `91.4°`, CTE `6863.7`, Gmax `8.68`
- `combined_load_factor_aware_bridge`:
  - timeout, phase `104.5°`, CTE `4951.9`, Gmax `9.89`

Supplemental policy checks:

- `base_only / load_factor_limited_bridge`:
  - timeout, phase `165.0°`, CTE `22295.0`, Gmax `7.94`
- `update2_scale1.0 / load_factor_limited_bridge`:
  - overload, phase `156.7°`, CTE `6351.6`, Gmax `12.28`
- `base_only / combined_load_factor_aware_bridge`:
  - timeout, phase `83.4°`, CTE `6148.7`, Gmax `9.89`
- `update2_scale1.0 / combined_load_factor_aware_bridge`:
  - overload, phase `165.0°`, CTE `5764.3`, Gmax `10.01`

Final next-stage report:

- `results/half_loop_bridge_next_stage/20260522_termination_trace_target_stream/final_report.md`

Final decision:

- No checkpoint promoted.
- No residual PPO training run.
- No target-stream candidate completed `pu165_R15000` with true reason `success` and Gmax < 9.
- Nothing is ready for Claude ACMI regression.
- Current best remains:

  `base checkpoint_epoch_619 + residual_checkpoint_update_2`

Next recommended direction:

- Stop blind residual-only PPO searches.
- Keep `pu165_R15000` as the only hard objective.
- Move to larger-radius bridge curriculum, load-factor-aware RH-TSO, reference reshaping before 165, or model-predictive target-stream co-design.
- Do not reintroduce `pu170/pu175/pu180` or exit/recovery until `pu165` completes with Gmax < 9.

## 15. 2026-05-22 Load-Factor-Aware RH-TSO pu165 Co-Design Study

User requested: do not train residual, do not test `pu170/175/180`, keep
`pu165_R15000` as the only hard target, and run a load-factor-aware target-stream
co-design study.

Code changes:

- `run_half_loop_termination_trace.py`
  - Added `profile_pullup_arc()` for phase-local radius inflation.
  - Added support for separate `eval_radius_m`, `target_radius_m`, and
    `target_radius_profile`.
  - Fixed summary `Gmax` / `vt_min` to include the terminal pre-reset state,
    so terminal overload / low-speed events are not hidden by step-before-action
    logging.
- `run_load_factor_aware_rhtso_pu165.py`
  - New inference-only runner.
  - Performs global radius sweep and target-stream / RH-TSO candidate sweep.
  - Uses costs on completion phase, effective G, low speed, CTE, wing-plane,
    tangent alignment, and target smoothness.
  - Writes `radius_sweep.csv`, `rhtso_sweep.csv`, `terminal_states.csv`,
    per-candidate phasewise diagnostics, raw terminal info, and `final_report.md`.

Run root:

- `results/load_factor_aware_rhtso_pu165/20260522_rhtso_pu165/`

Policy evaluated:

- `base = results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`
- `residual = results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2`
- inference residual scale: `0.20`
- no PPO training was run.

Radius sweep results on `pu165`:

- R15000:
  - timeout, phase `107.5`, CTE `7948.5`, effective_Gmax `7.76`
- R18000:
  - timeout, phase `139.8`, CTE `11097.8`, effective_Gmax `9.26`
- R20000:
  - low_speed, phase `160.9`, CTE `358.2`, effective_Gmax `7.32`,
    effective_vt_min `0.0`
- R25000:
  - low_speed, phase `124.0`, CTE `194.2`, effective_Gmax `6.25`,
    effective_vt_min `0.0`
- R30000:
  - low_speed, phase `105.8`, CTE `368.0`, effective_Gmax `6.25`,
    effective_vt_min `0.0`

RH-TSO / target-stream sweep highlights:

- `pitch_rate_limited_6dps`
  - timeout, phase `139.8`, CTE `7367.7`, effective_Gmax `7.92`
  - best scalar cost because it avoids terminal low-speed / overload, but it is
    not close to completion.
- `rhtso_lf_profile_R25000_v220_la700`
  - low_speed, phase `154.9`, CTE `1097.2`, effective_Gmax `7.92`,
    effective_vt_min `0.0`
  - useful diagnostic: better geometry/progress, but energy collapses.
- `local_radius_inflation_R25000_s100`
  - overload, phase `163.8`, CTE `6345.8`, effective_Gmax `10.56`
  - furthest phase progress, but invalid due overload and poor CTE.

Final decision:

- No candidate completed `pu165` with `success` and effective_Gmax `< 9`.
- Minimum successful radius was not found.
- Residual training must remain stopped.
- Nothing is ready for Claude ACMI regression.

Failure attribution:

- R15000 baseline is a controller/progress failure before the 145-170 bridge:
  timeout at phase `107.5` with large CTE.
- Easing curvature/global radius can make geometry and G acceptable near the
  bridge, especially R20000, but then the aircraft hits low-speed / energy
  collapse before valid completion.
- Local radius inflation can push phase to `163.8`, but it overloads at terminal
  effective_Gmax `10.56`.
- The blocker is coupled target-stream/controller/energy feasibility, not a
  residual-only PPO issue.

Recommended next direction:

- Keep `pu165` as the only hard objective.
- Do not reintroduce `pu170/175/180` or exit/recovery.
- Do not resume residual PPO yet.
- Next work should use stronger load-factor-aware RH-TSO or MPC-style
  target-stream co-design with explicit speed-floor, altitude-demand, and
  load-factor constraints, or a larger-radius bridge curriculum / reference
  reshaping before 165.

## 16. 2026-05-23 pu165 Energy Feasibility And Maneuver Reparameterization

User requested: do not resume residual PPO, do not test `pu170/175/180`, keep
current best as `epoch619 + residual_update_2`, and run an energy-feasibility /
reparameterized target-stream study for `pu165`.

Code changes:

- `run_half_loop_termination_trace.py`
  - Added `entry_vt` support by setting initial `plane_state.vt`, `vel_y`, and
    target speed before rollout.
  - Added `eval_altitude_gain_limit_m` and `target_altitude_gain_limit_m` via
    altitude-gain scaling of generated pull-up waypoints.
  - Kept terminal-state-aware `Gmax` / `vt_min` handling.
- `run_energy_feasibility_pu165.py`
  - New inference-only runner.
  - Produces analytical feasibility diagnostics for R15000/R18000/R20000/R25000/R30000,
    angles 150/165, and entry_vt 250/300/350/400.
  - Tests reparameterized pu165 target streams with entry speed, altitude cap,
    local radius profile, target_vt schedule, lookahead, pitch/roll rate limits,
    and load-factor-aware smoothing.

Final consolidated root:

- `results/energy_feasibility_pu165/20260523_energy_reparam_pu165_final/`

Files:

- `final_report.md`
- `feasibility_diagnostics.csv`
- `radius_sweep.csv`
- `rhtso_sweep.csv`
- `terminal_states.csv`
- `phasewise/`
- `raw_terminal_info/`

Analytical feasibility:

- All fixed-radius 150/165 deg arcs at radii 15000-30000 and entry_vt 250-400
  were classified `infeasible` under unpowered energy-height plus 190 m/s
  safe-exit-speed reserve.
- Fixed R15000 / 165 deg requires about `29488.9 m` altitude gain.
- Entry_vt 400 only provides about `8157.7 m` kinetic-energy height before
  reserving exit speed.

Rollout results:

- `fixed_R15000_entry_vt_250`
  - failed by `timeout`, phase `107.5`, CTE `7948.5`, effective_Gmax `7.76`.
- `fixed_R15000_entry_vt_300`
  - reached phase `165.0`, but failed by `overload`, effective_Gmax `10.84`.
- `entry_vt=350/400`
  - overloads almost immediately around phase `0.3-0.5`; controller/trim
    compatibility limit.

Passing reparameterized pu165 target streams:

- `altcap_20000_entry_vt_300`
  - success, phase `163.3`, CTE `101.7`, effective_Gmax `8.51`,
    effective_vt_min `200.7`.
- `altcap_8000_entry_vt_300`
  - success, phase `162.5`, CTE `193.9`, effective_Gmax `8.51`,
    effective_vt_min `200.7`.
- `mpc_altcap12000_v300_smooth`
  - success, phase `162.5`, CTE `133.0`, effective_Gmax `8.51`,
    effective_vt_min `204.3`.
- `mpc_altcap8000_v300_smooth`
  - success, phase `162.4`, CTE `160.7`, effective_Gmax `8.51`,
    effective_vt_min `204.3`.
- `profile_R20000_altcap12000_v300`
  - success, phase `162.7`, CTE `394.5`, effective_Gmax `8.51`,
    effective_vt_min `204.3`.
- `altcap_12000_entry_vt_300`
  - success, phase `162.4`, CTE `207.8`, effective_Gmax `8.51`,
    effective_vt_min `200.7`.

Best current target-stream candidate:

- `altcap_20000_entry_vt_300`

Important interpretation:

- This is not a new policy checkpoint and not a residual-training result.
- It proves `pu165` can be made energy-feasible by reparameterizing the target
  stream: entry_vt 300 plus capped altitude demand.
- Original fixed-radius R15000/pu165 remains energy-infeasible under current
  entry conditions and should be revised or explicitly labeled as such.

Next recommendation:

- Before any residual PPO resumes, run visual/ACMI validation and retention
  checks for the reparameterized target stream, especially
  `altcap_20000_entry_vt_300`.
- If training resumes after validation, keep it `pu165`-only and train around
  the passing capped-altitude target stream.
- Do not reintroduce `pu170/175/180` yet.

## 17. 2026-05-24 pu165 Tangent / Curvature Target-Stream Comparison

User reported Tacview inspection of `altcap_20000_entry_vt_300`: smooth through
80-100 deg, then cuts inside the reference arc after about 100 deg. TAS remains
high and AOA moderate, so the new issue is curvature / tangent tracking rather
than low-speed stall. User requested no residual PPO, no `pu170/175/180`, and a
pu165-only target-stream comparison.

Code changes:

- `run_half_loop_termination_trace.py`
  - Added target-stream modes:
    - `pure_pursuit`
    - `tangent_following`
    - `pursuit_tangent_blend`
    - `phase_scheduled_blend`
    - `curvature_aware`
  - Important implementation detail:
    - phase-scheduled / curvature-aware modes preserve the planner's blended
      pure-pursuit targets before 80 deg. Direct tangent override before 80 deg
      caused immediate overload.
- `run_pu165_tangent_curvature_target_stream.py`
  - New pu165-only inference runner.
  - Fixed policy: `epoch619 + residual_update_2`, residual scale `0.2`.
  - Fixed energy-feasible setup:
    - `entry_vt=300`
    - `eval_altitude_gain_limit_m=20000`
    - `target_altitude_gain_limit_m=20000`
  - Exports `summary.csv`, `phasewise/*.csv`, `acmi/*.acmi`, raw terminal info,
    and `final_report.md`.

Run root:

- `results/pu165_tangent_curvature_target_stream/20260524_pu165_tangent_curvature/`

ACMI files:

- `acmi/pure_pursuit_moving_lookahead.acmi`
- `acmi/tangent_following.acmi`
- `acmi/pursuit_tangent_blend_w050.acmi`
- `acmi/pursuit_tangent_blend_w025.acmi`
- `acmi/phase_scheduled_blend.acmi`
- `acmi/curvature_aware_smooth.acmi`

Results:

- `pure_pursuit_moving_lookahead`
  - success, phase `163.3`, CTE `101.7`, after100 CTE `124.7`,
    after100 velocity_tangent_error `6.3`, after100 nose_tangent_error `5.5`,
    Gmax `8.51`, vt_min `200.7`.
- `tangent_following`
  - overload almost immediately, phase `0.3`, Gmax `10.34`.
- `pursuit_tangent_blend_w050`
  - overload almost immediately, phase `0.3`, Gmax `10.34`.
- `pursuit_tangent_blend_w025`
  - overload almost immediately, phase `0.3`, Gmax `10.34`.
- `phase_scheduled_blend`
  - no early overload after fixing 0-80 deg pure-pursuit preservation, but
    timed out with phase `165.0`, huge CTE `28454.7`, after100 CTE `35237.4`.
- `curvature_aware_smooth`
  - success, phase `164.3`, CTE `95.1`, after100 CTE `179.0`,
    after100 velocity_tangent_error `10.8`, after100 nose_tangent_error `1.6`,
    Gmax `8.51`, vt_min `204.3`.

Interpretation:

- The tested tangent-aware streams do not fix the post-100 deg inside-cut.
- Direct tangent and constant pursuit/tangent blends are incompatible with the
  current controller at maneuver start and overload immediately.
- Phase-scheduled tangent avoids early overload but drifts far off the arc after
  tangent weighting is introduced.
- Curvature-aware smoothing is safe and slightly lowers global CTE, but its
  post-100 CTE and velocity-tangent error are worse than the pure-pursuit
  baseline.
- Best after-100 metrics remain the pure-pursuit moving-lookahead baseline.

Next recommendation:

- Do not train residual PPO yet.
- Do not test `pu170/175/180` yet.
- Inspect the exported ACMI files, especially baseline vs curvature-aware.
- Next target-stream refinement should avoid abrupt tangent target override and
  instead optimize the local reference path / lookahead / curvature with a
  closed-loop CTE penalty after 100 deg, because target tangent alignment alone
  is not enough.

## 18. 2026-05-24 pu165 Local Correction Search on Pure Pursuit

User requested a low-dimensional local-correction search on top of the current
best pure-pursuit moving-lookahead pu165 setup. Hard constraints were respected:
no residual PPO training, no `pu170/175/180`, fixed policy
`epoch619 + residual_update_2`, and pu165-only evaluation.

Code changes:

- `run_half_loop_termination_trace.py`
  - Added smooth phase-gated local corrections:
    - `local_pitch_bias_deg`
    - `local_lookahead_scale`
    - `local_target_vt_delta`
  - Corrections are applied only inside a configured phase band with smooth
    margins, preserving the pure-pursuit moving-lookahead backbone.
- `run_pu165_local_correction_search.py`
  - New pu165-only runner for local pitch/lookahead/target_vt correction
    sweeps.
  - Fixed setup:
    - base: `results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`
    - residual: `results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2`
    - residual scale: `0.20`
    - `entry_vt=300`
    - `eval_altitude_gain_limit_m=20000`
    - `target_altitude_gain_limit_m=20000`
  - Exports summary, terminal traces, per-candidate phasewise CSVs, phase-band
    summary, raw terminal info, and ACMI files.

Run root:

- `results/pu165_local_correction_search/20260524_pu165_local_corrections/`

Files:

- `summary.csv`
- `phase_band_summary.csv`
- `inward_cut_proxy.csv`
- `terminal_states.csv`
- `final_report.md`
- `phasewise/*.csv`
- `raw_terminal_info/*.json`
- `acmi/*.acmi`

Baseline recheck:

- `baseline_pure_pursuit`
  - success, phase `163.3`, after100 CTE `124.7`,
    after100 velocity_tangent_error `6.3`, after100 nose_tangent_error `5.5`,
    Gmax `8.51`, vt_min `200.7`.

Search coverage:

- 29 candidates total:
  - pitch bias bands `90-130` and `100-145`, bias `+2/+4/+6 deg`
  - smaller lookahead bands `90-130` and `100-145`, scale `0.8/0.6`
  - target_vt bands `90-130` and `100-145`, deltas `-20/-10/+10 m/s`
  - small joint combinations.

Strict numeric candidates:

- `pitch_p2_b100_145`
  - after100 CTE `110.6`, after100 velocity_tangent_error `6.334`,
    after100 nose_tangent_error `5.09`, Gmax `8.51`, vt_min `200.7`.
- `vt_p10_b100_145`
  - after100 CTE `116.9`, after100 velocity_tangent_error `6.274`,
    after100 nose_tangent_error `5.34`, Gmax `8.51`, vt_min `200.7`.
- `pitch_p2_vt_p10_b100_145`
  - after100 CTE `107.8`, after100 velocity_tangent_error `6.336`,
    after100 nose_tangent_error `5.04`, Gmax `8.51`, vt_min `200.7`.

Important non-promoted diagnostics:

- `lookahead_x06_b100_145`
  - best after100 CTE at `97.2`, but velocity_tangent_error worsened to
    `6.59`, so it failed the user decision rule.
- `90-130 deg` pitch bias often improved velocity-tangent error but worsened
  CTE badly, e.g. `pitch_p6_b90_130` CTE `173.8` with vtan `5.49`.
- `inward_cut_proxy.csv`
  - baseline mean signed radial error after 100 deg: `-123.3 m`,
    inside fraction `0.72`.
  - `pitch_p2_b100_145`: `-82.4 m`, inside fraction `0.57`.
  - `vt_p10_b100_145`: `-115.3 m`, inside fraction `0.72`.
  - `pitch_p2_vt_p10_b100_145`: `-74.5 m`, inside fraction `0.55`.
  - Interpretation: the best strict numeric candidates reduce the inward-cut
    proxy, but do not eliminate inward-cut behavior. Tacview is still required.

Decision:

- No candidate is promoted yet.
- Three candidates satisfy the numeric safety + after100 metric gate, but the
  user required visual confirmation that ACMI shows less inward cutting after
  100 deg.
- User Tacview inspection of the combined comparison found:
  - Magenta / Orange / Green are closer than Cyan around and after 100 deg.
  - Near 160 deg / close to the end, Cyan slightly overtakes and becomes
    marginally closer than the other three.
  - Without strong zoom, all four trajectories are nearly overlapping.
  - All four trajectories drift away from the yellow reference arc near the end.
- Decision after visual inspection:
  - The three strict numeric candidates are weak local improvements in the
    100-145 deg region.
  - The improvement is not decisive and not sustained through the terminal
    145-160 deg region.
  - No new target stream is promoted.
  - Do not resume residual PPO and do not reintroduce `pu170/175/180`.
  - If continuing target-stream-only work, focus on terminal 145-165 deg drift,
    not broad target-stream family replacement.

## 19. 2026-05-24 Final Stop Decision for Current Half-Loop Iteration

User concluded that continuing training iteration is no longer meaningful for
the current route. This section consolidates the state so future work does not
repeat the same failed experiments.

Current strongest baseline combination:

- base policy:
  - `results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`
- residual specialist:
  - `results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2`
- shorthand:
  - `epoch619 + residual_update_2`

Checkpoints / branches that must not be promoted or used as new starts:

- monolithic / mixed PPO diagnostic branches:
  - `checkpoint_epoch_658`
  - `checkpoint_epoch_628`
  - `checkpoint_epoch_632`
  - `checkpoint_epoch_622`
  - `checkpoint_epoch_621`
- failed exit-heavy / update_4 branches:
  - `results/half_loop_specialist_residual_exit_recovery_v1/20260519_1633/...`
  - `results/half_loop_specialist_residual_exit_recovery_v1_conservative/20260519_1741/...`
- failed residual auto-search branches:
  - `results/half_loop_residual_auto_search/20260520_1146/round_01/...`
  - `results/half_loop_residual_auto_search/20260520_1146/round_02/...`

Major experimental conclusions:

- Monolithic PPO / mixed fine-tuning is not viable for the current objective:
  - It can improve some vertical metrics but repeatedly damages horizontal
    behavior or fails Claude-style loop-quality promotion gates.
  - Do not resume ordinary mixed PPO from epoch619.
- Frozen base plus phase-gated residual was the correct architectural pivot:
  - It protects horizontal behavior by construction outside the gate.
  - `residual_update_2` remains the best specialist residual.
  - Later residual updates did not beat it under strict gates.
- Residual-only bridge training has hit a practical stop:
  - `pu165_R15000` could not be made into a robust no-crash / G-safe
    fixed-radius bridge by additional residual PPO.
  - Scale-only diagnostics showed lower residual scale can improve geometry,
    but either remains incomplete or runs into load-factor / bridge stability
    limits.
  - The bottleneck is not solved by more blind PPO updates.
- Termination tracing was fixed:
  - Earlier `done_unknown` cases were mostly timeouts or true terminal
    conditions, not hidden successes.
  - `update2_scale0.2 / pu165_R15000` was not close to real fixed-radius
    success: terminal phase was only around `107.5 deg`.
- Load-factor-aware / energy feasibility studies changed the interpretation:
  - Original fixed-radius R15000 / pu165 is energy-infeasible under the current
    entry assumptions if the reference altitude demand is kept literal.
  - Reparameterized target streams with entry_vt 300 and capped altitude demand
    can make a pu165-like maneuver complete safely.
  - Best reparameterized target stream so far:
    - `altcap_20000_entry_vt_300`
    - success, phase about `163.3`, CTE about `101.7`, effective_Gmax `8.51`,
      effective_vt_min `200.7`.
- Tangent / curvature target-stream replacements did not solve the visual
  problem:
  - Direct tangent-following and constant pursuit/tangent blends overloaded
    immediately.
  - Phase-scheduled tangent avoided early overload but diverged badly after
    tangent weighting was introduced.
  - Curvature-aware smoothing lowered global CTE but worsened after-100 CTE and
    velocity-tangent error.
  - Pure-pursuit moving-lookahead remains the best broad target-stream backbone.
- Local corrections on pure pursuit gave only weak improvements:
  - Three candidates improved numeric after-100 metrics:
    - `pitch_p2_b100_145`
    - `vt_p10_b100_145`
    - `pitch_p2_vt_p10_b100_145`
  - Tacview inspection confirmed they are slightly closer than baseline around
    and after 100 deg.
  - However, all four trajectories are nearly overlapping without strong zoom,
    and near 160 deg baseline becomes marginally closer again.
  - All candidates still drift away from the reference arc near the terminal
    145-165 deg region.
  - Therefore no target-stream candidate is promoted.

Current final decision:

- Stop current residual PPO / local target-stream iteration.
- Do not train more residual PPO candidates.
- Do not test `pu170/pu175/pu180` from the current candidate set.
- Do not send any new checkpoint or target stream to Claude as a promoted
  result.
- Keep the current best policy baseline as `epoch619 + residual_update_2`.
- Treat the current target-stream results as diagnostics only, not a solved
  full-loop or promotable pu165 baseline.

If this line of work is resumed later, the next work should not be "more of the
same". Reasonable directions would be:

- redefine the benchmark to an energy-feasible pu165 target stream rather than
  literal fixed-radius R15000 / 165 deg;
- design a load-factor-aware RH-TSO / MPC-style target-stream optimizer that
  explicitly handles energy, terminal 145-165 deg drift, curvature, tangent
  alignment, and target smoothness jointly;
- consider a separate higher-level maneuver reparameterization or reference
  reshaping module before attempting any more residual policy training;
- only resume residual PPO after a target stream is demonstrably feasible and
  visually close on pu165 with Gmax < 9 and vt_min > 190.

Artifacts most useful for future reference:

- `results/half_loop_bridge_termination_trace/`
- `results/load_factor_aware_bridge_target_stream/`
- `results/load_factor_aware_rhtso_pu165/`
- `results/energy_feasibility_pu165/20260523_energy_reparam_pu165_final/`
- `results/pu165_tangent_curvature_target_stream/20260524_pu165_tangent_curvature/`
- `results/pu165_local_correction_search/20260524_pu165_local_corrections/`

Final note:

- The current work produced useful diagnosis, not a new promotable policy.
- The remaining issue is a coupled energy / target-stream / controller
  feasibility problem, especially terminal 145-165 deg drift, rather than a
  problem likely to be solved by continued residual PPO iteration.
