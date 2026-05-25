# TODO: Numbers And Figures Before Submission

## Numbers To Verify Before Making Strong Claims

- Low-level skill tracking:
  - quaternion geodesic error on held-out target schedules
  - speed error
  - survival rate
  - alpha/beta/G safety violations
  - target-switch convergence rate

- Euler vs. quaternion ablation:
  - Euler baseline under matched training/evaluation conditions
  - quaternion baseline under the same target schedules
  - separate moderate-attitude and near-vertical/inverted subsets
  - only after this table exists should the paper claim quantitative superiority

- Energy-aware vertical extension:
  - base vs. fine-tuned results on 60, 90, 120, 150, and 180 degree arcs
  - completion/survival definitions
  - speed minimum, energy loss, alpha/beta/G envelopes
  - evidence that horizontal behavior is retained after fine-tuning

- Loop-quality frontier:
  - CTE mean/P90/max
  - velocity-tangent error
  - nose-tangent error
  - nose-velocity error
  - wing-plane error
  - quaternion geodesic error
  - grade thresholds for B and Fail

## Figures To Replace Current Schematic Placeholders

- Figure 1: System overview.
  - Current manuscript has a compile-safe schematic box.
  - Replace with a clean diagram: reference maneuver -> RH-TSO / target-stream generator -> frozen quaternion skill -> actuator commands -> dynamics -> geometry-aware evaluation.

- Figure 2: RH-TSO closed-loop selection.
  - Current manuscript has a compile-safe schematic box.
  - Replace with candidate streams, short closed-loop rollouts, cost evaluation, selected stream, execute first segment, replan.

- Figure 3: Representative maneuvers.
  - Add S-curve, figure-eight, helix/mild-3D, and 90 degree vertical pull-up.
  - Show reference path, flown path, selected target-stream parameters, and CTE over time.

- Figure 4: Vertical arc capability / CTE-vs-geometry.
  - Show why CTE-only is misleading.
  - Include phase-binned geometry metrics for 60-150 degree B-grade arcs and 180 degree Fail.

## Tables To Finalize

- Table 1: Policy interface.
  - Already included.

- Table 2: RH-TSO target-stream selection.
  - Included with verified values from `20260519_122354`.

- Table 3: Loop-quality frontier.
  - Included structurally, but numeric metric cells are placeholders.
  - Fill from verified loop-quality logs before submission.

## Wording To Preserve

- Do not claim full-loop success.
- Do not frame the paper as a Planax benchmark.
- Do not call RH-TSO globally optimal.
- Do not claim CEM/MPPI is better than lattice in current experiments.
- Keep 180 degree half-loop as Fail unless new verified evidence changes the result.
