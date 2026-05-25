# Hierarchical Architecture for 3D Trajectory Tracking

**Date**: 2026-05-14
**Status**: Design proposal — not yet implemented

---

## 1. Motivation

Current quaternion baseline (epoch 600) has mastered attitude tracking:
it can follow arbitrary heading/pitch/roll/vt commands with high precision
(r_att=0.442, Gmax p95 < 7G). However, it cannot track spatial trajectories
because it was never trained with position information.

The fundamental gap: **attitude tracking is 3-DOF (heading, pitch, speed);
trajectory tracking is 6-DOF (position + attitude + speed).**

Rather than retraining a single large policy from scratch, we freeze the
proven attitude-tracking baseline and train a lightweight upper policy
that maps spatial errors to attitude targets.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  UPPER LAYER: Spatial Guidance Policy                     │
│                                                            │
│  Input (12 dims):                                          │
│    - ENU position error to target [dN, dE, dAlt] (norm)   │
│    - 3D distance to target (norm)                          │
│    - Current roll, pitch, yaw                              │
│    - Current vt                                             │
│    - sin/cos alpha, sin/cos beta                           │
│                                                            │
│  Network: MLP [64 → 64 → 3]  (no RNN — reactive control)  │
│    - No hidden state needed: spatial error is sufficient   │
│      statistic for guidance decisions                      │
│                                                            │
│  Output (3 dims, continuous):                              │
│    - Δheading  ∈ [-π/2, π/2]                              │
│    - Δpitch    ∈ [-π/4, π/4]                              │
│    - target_vt ∈ [120, 360] m/s                           │
│                                                            │
│  Reward: -(cross_track_error / R_track)                    │
│    + 0.1 * dt_reduction  (progress toward target)          │
└──────────────┬───────────────────────────────────────────┘
               │  target_heading = yaw + Δheading
               │  target_pitch   = pitch + Δpitch
               │  target_vt      = target_vt
               ▼
┌──────────────────────────────────────────────────────────┐
│  LOWER LAYER: Attitude Tracking Policy (FROZEN)            │
│                                                            │
│  From: baseline_quat_20260514 (epoch 600)                  │
│  Network: ActorCriticRNN (GRU-128, FC-128)                 │
│  Obs: 21 dims (quaternion err, body-frame target,          │
│       PQR, alpha/beta, past action)                        │
│  Output: discrete [thr, el, ail, rud, sb]                  │
│                                                            │
│  FROZEN — no gradient flows through this layer             │
└──────────────────────────────────────────────────────────┘
```

**Total trainable parameters**: ~5,000 (upper MLP only)
**Frozen parameters**: ~85,000 (lower baseline)

---

## 3. Upper-Layer Input Design

| Index | Feature | Normalisation | Rationale |
|:-----:|---------|:------------:|-----------|
| 0 | dNorth / 5000 | 5000 m | Position error forward |
| 1 | dEast / 5000 | 5000 m | Position error lateral |
| 2 | dAlt / 5000 | 5000 m | Position error vertical |
| 3 | dist_3d / 5000 | 5000 m | Total distance to target |
| 4 | roll / π | ±π rad | Current bank angle |
| 5 | pitch / π | ±π rad | Current pitch |
| 6 | yaw / π (wrapped) | ±π rad | Current heading |
| 7 | vt / 340 | 340 m/s | Normalised airspeed |
| 8 | sin(alpha) | [-1, 1] | AoA sine |
| 9 | cos(alpha) | [-1, 1] | AoA cosine |
| 10 | sin(beta) | [-1, 1] | Sideslip sine |
| 11 | cos(beta) | [-1, 1] | Sideslip cosine |

All values clipped to [-2, 2] after normalisation for numerical stability.

---

## 4. Upper-Layer Output Design

| Output | Range | Mapping |
|--------|:-----:|---------|
| Δheading | [-π/2, π/2] | `output[0] * π/2` |
| Δpitch | [-π/4, π/4] | `output[1] * π/4` |
| target_vt | [120, 360] | `120 + (output[2] + 1) / 2 * 240` |

Output uses `tanh` activation, scaled to physical ranges.

The guidance policy does NOT output roll commands. The lower baseline
handles turn coordination autonomously (it was trained with roll targets).

---

## 5. Training Setup

### 5.1 Environment

New env: `AeroPlanaxTrajectoryTrackingEnv`
- Inherits from `AeroPlanaxEnv`
- Uses FROZEN lower policy internally (loaded from checkpoint)
- Upper policy interacts through the attitude-target interface
- Generates spatial trajectory waypoints as training targets

### 5.2 Trajectory Types (Curriculum)

| Stage | Trajectories | Difficulty |
|:-----:|-------------|:----------:|
| 1 | Straight line (1 km, constant alt) | Trivial |
| 2 | Horizontal arc (90° turn, R=3000m) | Easy |
| 3 | Horizontal circle (R=2000m) | Medium |
| 4 | Vertical loop (R=2000m) | Hard |
| 5 | 3D helix (R=1500m, climb 500m/lap) | Very Hard |

Each trajectory = sequence of 20-60 waypoints.
Waypoints spaced ~200m apart (arc length).
Target switches when aircraft reaches waypoint (reach radius ~300m)
OR after timeout (level-dependent: 30-90s).

### 5.3 Reward Function

```python
def trajectory_tracking_reward(state, target_pos, R_track=100.0):
    # Cross-track error: distance from aircraft to nearest point on trajectory
    cross_track = distance_to_trajectory(state, target_pos)
    # Progress reward: reduction in distance to next waypoint
    prev_dist = state.pre_dist_to_wp
    curr_dist = distance_to_waypoint(state, target_pos)
    progress = prev_dist - curr_dist

    reward = -cross_track / R_track + 0.05 * clip(progress, 0, 100)
    return clip(reward, -5.0, 1.0)
```

- Cross-track penalty dominates for precision
- Progress bonus prevents stalling
- R_track=100m: being 100m off costs 1.0 reward per step

### 5.4 Training Algorithm

- **Algorithm**: PPO (continuous action)
- **Action space**: Box(3,) — [Δh, Δp, Δv]
- **Network**: MLP [64, 64, 3] — no RNN
- **LR**: 1e-4 (smaller than baseline, since network is small)
- **Parallel envs**: 500 (fewer than baseline, since a full episode needs to complete a trajectory)
- **Steps per episode**: 1500
- **Total timesteps**: 2e8 (estimate, ~200-400 epochs)

### 5.5 Anti-Forgetting

20% of training samples use original attitude-target distribution
(random heading/pitch/roll/vt deltas from current state, as in baseline
training). Upper policy must learn to output "neutral" corrections
(Δh≈0, Δp≈0, Δv≈250) for these samples, preserving baseline's
independent flight capability.

---

## 6. Implementation Plan

### Phase 1: Upper Policy Training (3-5 days)

| Step | Task | Effort |
|:----:|------|:------:|
| 1.1 | Create `trajectory_guidance_policy.py` — MLP network | 30 min |
| 1.2 | Create `aeroplanax_trajectory_tracking.py` — new env with frozen baseline | 2 hr |
| 1.3 | Implement trajectory generators (line, arc, circle, loop) | 1 hr |
| 1.4 | Implement cross-track error computation | 30 min |
| 1.5 | Create `train_trajectory_guidance.py` — PPO continuous | 1 hr |
| 1.6 | Run training (overnight, ~8 hr) | — |
| 1.7 | Evaluate on vertical loop render | 30 min |

### Phase 2: Iteration (if needed)

- Tune cross-track penalty coefficient R_track
- Add more diverse trajectory types
- Adjust curriculum progression speed

---

## 7. Success Criteria

| Criterion | Threshold | Measurement |
|-----------|:---------:|-------------|
| Vertical loop completion | 30/30 waypoints | render_vertical_loop_test.py |
| Cross-track error p50 | < 100 m | analyze_loop_tracking.py |
| Cross-track error p90 | < 250 m | analyze_loop_tracking.py |
| S-maneuver (regression) | 100/100 | render_waypoint_s_quat.py |
| Basic flight (regression) | No crash in 2000 steps | eval script |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Upper policy outputs infeasible Δpitch | Clip output to ±45° |
| Upper + lower interaction causes oscillation | Add rate limit on upper output (max Δh/step = 10°) |
| Lower baseline degrades (inference-only can still drift) | Monitor r_att during evaluation |
| Training takes too long | Start with stage 1-2 only, verify convergence, then add harder trajectories |
