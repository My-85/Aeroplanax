# Upper Trajectory Abstraction — Results Summary

**Date**: 2026-05-15
**Lower Policy**: `results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600`
**Planner**: PurePursuitPlanner (lookahead + pure pursuit + blend smoothing + roll=0)

---

## 1. Architecture

```
Reference trajectory → PathManager(lookahead) → PurePursuit subgoal → TargetBlender → target(H,P,R,Vt) → Frozen Baseline → F-16 dynamics
```

Key design findings:
- **Fixed spatial waypoint fails**: aircraft oscillates near target (min dist ≈653m, never converges)
- **Moving lookahead succeeds**: continuous local subgoal stream keeps observations within training distribution
- **Roll target = 0 preferred**: heuristic roll commands (tangent_plus_roll) destabilise the lower policy

---

## 2. Capability Map

### Feasible

| Maneuver | CTE mean | CTE p50 | Gmax | vt min | Status |
|----------|---------:|--------:|-----:|-------:|:------:|
| Circle R=5000m right | 251 m | 52 m | 9.1 | 191 | ✓ |
| Circle R=5000m left | 474 m | 385 m | 8.8 | 190 | ✓ |
| Circle R=3000m right | 365 m | 242 m | 9.8 | 187 | ✓ |
| Circle R=3000m left | 553 m | 442 m | 9.0 | 190 | ✓ |
| S-curve A=3000m | 328 m | 88 m | 9.0 | 191 | ✓ |
| Figure-8 R=5000m | 531 m | 429 m | 9.7 | 176 | ✓ |
| Climb +1000m/15km | 24 m | 8 m | 6.5 | 192 | ✓ |
| Climb +2000m/15km | 24 m | 9 m | 6.5 | 192 | ✓ |
| Descent -1000m/15km | 41 m | 11 m | 6.6 | 192 | ✓ |
| 15° Pull-up R=8000m | 72 m | — | 6.2 | 199 | ✓ |
| 30° Pull-up R=8000m | 100 m | 91 m | 6.2 | 187 | ✓ |
| 30° Pull-up R=10000m | 105 m | 105 m | 6.2 | 187 | ✓ |

### Infeasible / Boundary

| Maneuver | Limitation | Root Cause |
|----------|-----------|------------|
| Fixed single waypoint | Oscillation, never converges | Observation goes OOD without moving target |
| Tangent-only guidance | CTE diverges | No lateral error correction without pursuit term |
| Naive roll guidance (1.5×hdg_err) | Destabilises aircraft | Roll target enters qv, breaks attitude tracking |
| Pull-up R≤3000m | Speed drops to 147 m/s, stalls | Energy management deficit during steep climb |
| Descent -2000m/15km | Speed drops to 149 m/s, timeout | Energy management deficit during steep descent |

---

## 3. Key Scientific Findings

1. The frozen quaternion baseline requires a **continuous, gradually moving target stream** — not fixed spatial waypoints.
2. Pure pursuit with moving lookahead is the most robust upper-level guidance; tangent blending and learned roll provide no benefit at this stage.
3. The capability boundary is set by **low-level energy management**, not upper-level planning complexity.
4. Left-turn CTE is consistently higher than right-turn (asymmetry recorded, root cause pending Codex investigation).
5. Pitch ramp diagnostic confirms the lower policy can track ±20° pitch commands — climb failures are energy-related, not attitude-tracking-related.

---

## 4. Files

```
experiments/hierarchical_trajectory_tracking/
├── path_manager.py       — Waypoint/lookahead path tracking
├── subgoal_generator.py  — Pure pursuit, tangent, blend, roll guidance
├── target_blender.py     — Blend smoothing + rate limiting
├── trajectory_generators.py — 11 trajectory types (circle, S, fig8, helix, pull-up, etc.)
├── planner.py            — Unified planner interface (PurePursuit, ScheduledLA, EnergyAware)
├── demo_library.py        — 14-demo config library
├── path_utils.py          — CTE computation, arc length, segment projection
├── render_maneuver_demo_library.py — Full demo render (ACMI + metrics + plots)
└── results_summary.md     — This file
```

## 5. Next Steps (Codex)

1. Vertical energy management fine-tuning for the lower quaternion PPO baseline
2. Re-test small-radius pull-up (R=2000-3000m), full loop, and aggressive vertical arcs
3. Investigate left/right asymmetry root cause
4. Plug updated checkpoint into the same PurePursuitPlanner for re-evaluation
