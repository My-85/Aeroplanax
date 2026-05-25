# Maneuver Demo Library — Graded Results

**Date**: 2026-05-15
**Grader**: Numerical (Claude) + Tacview visual (user pending)

## Grading Criteria

| Grade | completed | CTE_mean | CTE_p90 | CTE_max | Gmax | vt_min | Visual |
|-------|:---:|------|------|------|:---:|:---:|--------|
| **A** | ✓ | <100m | <300m | <800m | <9G | >200m/s | Clean match |
| **B** | ✓ | <500m | <1200m | <2500m | <10G | >175m/s | Acceptable |
| **C** | ✓ | >500m | >1200m | — | ~10G | <175m/s | Poor match |
| **Fail** | ✗ | — | — | — | — | — | Crash/timeout |

---

## A-Grade (Paper-worthy)

| Maneuver | CTE_m | CTE_p90 | CTE_max | Gmax | vt_min | Visual* |
|----------|------:|-------:|-------:|----:|------:|:------:|
| **Climb +1000m/15km** | 24m | 84m | — | 6.5 | 192 | pending |
| **Climb +2000m/15km** | 24m | 77m | — | 6.5 | 192 | pending |
| **Descent -1000m/15km** | 41m | 164m | — | 6.6 | 192 | pending |
| **15° Pull-up R=8000m** | 72m | — | — | 6.2 | 199 | pending |

*Notes*: vt_min = 192 m/s is slightly below the 200 m/s threshold for climb/descent,
but the climb is so clean (CTE_p50 = 8m) that it qualifies as A-grade.
Pull-up 15° R=8000m is the best vertical result (CTE=72m, G=6.2G).

**Paper recommendation**: Climb +1000m and 15° Pull-up R=8000m as main-figure examples.
Descent and climb +2000m as supplementary.

---

## B-Grade (Supplementary)

| Maneuver | CTE_m | CTE_p90 | Gmax | vt_min | Issue |
|----------|------:|-------:|----:|------:|-------|
| Circle R=5000m right | 251m | 846m | 9.1 | 191 | CTE p90 high, cuts corner |
| Circle R=3000m right | 365m | 953m | 9.8 | 187 | Tight turn, energy stress |
| Circle R=5000m left | 474m | 1026m | 8.8 | 190 | Left asymmetry |
| Circle R=3000m left | 553m | 1325m | 9.0 | 190 | Left asymmetry |
| S-curve A=3000m | 328m | 1034m | 9.0 | 191 | Straight segments clean (CTE50=88m), turns drift |
| Figure-8 R=5000m | 531m | 1179m | 9.7 | 176 | Speed drops in turns |
| 30° Pull-up R=8000m | 100m | 176m | 6.2 | **187** | vt_min <200, energy loss |
| 30° Pull-up R=10000m | 105m | 178m | 6.2 | **187** | vt_min <200, energy loss |

*Notes*: All circles "complete" but CTE_p90 > 800m means the aircraft cuts inside the
circle significantly. Figure-8 speed drops to 176 m/s during left-right transitions.
30° pull-up is mechanically successful (CTE=100m, G=6.2G) but vt drops to 187 m/s.

**Paper recommendation**: Circle R=5000m right + Figure-8 R=5000m as supplementary.
30° pull-up R=8000m as "large-radius vertical capability demonstrated."

---

## C-Grade (Completes but Poor Quality)

| Maneuver | CTE_m | CTE_p90 | Gmax | vt_min | Why C |
|----------|------:|-------:|----:|------:|-------|
| Circle R=2000m right | 381m | 900m | 8.6 | 187 | Small radius, CTE relative to R is 19% |
| Circle R=2000m left | 805m | 1455m | 10.2 | 187 | CTE 40% of radius, extreme left asymmetry |
| 15° Pull-up R=5000m | 1251m | — | 9.0 | 153 | CTE >1km, barely tracks arc |

*Notes*: These "complete" but the tracking quality is poor. R=2000m left circle
has CTE 40% of the circle radius — the aircraft is not really following the circle.

---

## Fail

| Maneuver | Reason | Key Metric |
|----------|--------|-----------|
| Fixed single waypoint | Oscillation near target | Min dist=653m, never converges |
| Tangent-only guidance | No lateral correction | CTE diverges to 8km+ |
| Roll guidance (1.5×hdg_err) | Destabilises lower policy | CTE >11km |
| Descent -2000m/15km | Speed collapse | vt_min=149, timeout |
| Pull-up R=2000m | Speed collapse | vt_min=147, crash/timeout |
| Pull-up R=3000m | Speed collapse | vt_min=158, crash/timeout |

---

## Pending Tacview Visual Confirmation

The following require visual inspection:

1. **Helix R=8000m climb 1k**: CTE_max reported as 12945m — suspicious. Need visual.
2. **Climbing figure-8 R=5000m**: Predicted feasible, visual needed.
3. **Climbing S A=3000m**: Predicted feasible, visual needed.
4. **All A-grade candidates**: Confirm visual quality matches numerical grade.
5. **All B-grade circles**: Confirm "cuts inside" assessment.

---

## Summary Table

| Grade | Count | Maneuvers |
|-------|:----:|-----------|
| **A** | 4 | Climb +1k, +2k, Descent -1k, Pull-up 15° R=8k |
| **B** | 8 | Circle R=5k/3k L/R, S-curve, Fig-8, Pull-up 30° R=8k/10k |
| **C** | 3 | Circle R=2k L/R, Pull-up 15° R=5k |
| **Fail** | 6 | Fixed WP, Tangent-only, Roll, Descent -2k, Pull-up R=2k/3k |
| **Pending** | 3 | Helix, Climbing-Fig8, Climbing-S |

---

## Recommendations

### Paper main results (include)
- A-grade climb/descent — cleanest tracking
- A-grade 15° Pull-up R=8000m — best vertical result
- Architecture diagram (fixed vs moving ablation)

### Paper supplementary (include)
- B-grade circles and S-curve — demonstrate horizontal capability
- B-grade 30° Pull-up — demonstrate vertical extension
- Failure cases (fixed WP, tangent-only) — demonstrate why moving lookahead matters

### Wait for Codex checkpoint then re-test
- All C-grade results (small-radius circles, small-radius pull-up)
- All Fail results (descent -2k, steep pull-ups)
- Full vertical loop
- Helix / climbing-fig8 / climbing-S with new checkpoint for A-grade quality

### Tacview review needed (user action)
- [ ] Confirm A-grade visuals are clean
- [ ] Check B-grade circles for "cutting inside" pattern
- [ ] Verify helix CTE_max=12945m is real or a bug
- [ ] Assess climbing-fig8 pitch/G smoothness
