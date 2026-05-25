# Quaternion Baseline — Full Envelope Flight Control

**Date**: 2026-05-14  
**Checkpoint**: `results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600`  
**Training steps**: 1.2e9 env steps (600 epochs)

---

## Performance Summary

### S-Maneuver Waypoint Tracking (Cruise / Small Maneuvers)

| Metric | Value |
|--------|:-----:|
| Waypoints reached | **100 / 100** |
| Flight time | 186 s (930 steps) |
| Altitude stability | ±30 m |
| Speed holding | 245–250 m/s |
| Roll range | ±10° |
| Pitch range | ±5° |

Smooth, stable cruise flight with small heading changes.

### L3 Full-Envelope Maneuver Test

| Phase | Maneuver | Steps | Result |
|-------|----------|:----:|--------|
| WP0 | Straight cruise | 38 | Stable |
| WP1 | Right turn 90° | 83 | Passed, roll max 83° |
| WP2 | Reversal 180° | 171 | Passed, roll max 136° |
| WP3 | Climb +2000 m | 154 | 4896→6888 m, passed |
| WP4 | Dive −1500 m | 82 | 6940→5936 m, passed |
| WP5 | Speed run (straight) | 95 | Speed 224–251 m/s |
| WP6 | Climb + 180° turn (compound) | 176 | Passed |
| WP7 | Descending turn | 130 | Passed |
| WP8 | Return to origin | 163 | Passed |

**9 / 9 waypoints reached.**

### Flight Quality

| Metric | Value | Assessment |
|--------|:-----:|------------|
| Waypoint completion | 9/9 L3 + 100/100 S | All difficulty levels covered |
| G-load p95 | 6.9 G | Within 9 G limit |
| G-load max | 9.0 G | No crash triggered |
| Airspeed range | 158–257 m/s | Low to high speed covered |
| Altitude range | 4040–6940 m | Dive to climb covered |
| Alpha range | −37.8° to +15.6° | Occasional excursion below −20° |
| Roll rate max | 209 °/s | Occasional aggressive roll |

---

## Capability Map

```
Covered:
  [x] Straight cruise + small heading changes (L0)
  [x] 90° turns + speed changes (L1)
  [x] 180° reversals + moderate pitch (L2)
  [x] Full-envelope compound maneuvers (L3) — climbing 180° reversal

Limitations:
  [!] Occasional roll overshoot (WP8 roll reached 180°)
  [!] Speed management during climb is suboptimal (min 158 m/s)
  [!] Alpha occasionally dips below aero-table lower bound (−20°)
  [!] Rare G-load spikes in training (max observed 1700 G, decreasing)
```

---

## Key Files

| File | Role |
|------|------|
| `envs/aeroplanax_heading_pitch_V_quaternion_version_add_full_roll.py` | Training environment (trim init, mixed-sampling curriculum, 21-dim obs with past action) |
| `envs/termination_conditions/unreach_heading_pitch_V_quat.py` | Adaptive target switching (11/24/42/50 s per level) |
| `envs/reward_functions/heading_pitch_V_reward_add_roll_target.py` | Attitude+speed tracking reward (Gaussian kernel, w_att=0.7, w_speed=0.3) |
| `envs/reward_functions/reward_nz_soft_penalty.py` | G-load soft penalty (nz_limit=9 G, coef=0.05, clip=5.0) |
| `train_heading_pitch_V_discrete_rnn_new_critic_no_fc2_quaternion_version_add_roll_target.py` | PPO training script (GRU-128, FC-128, 21-dim obs) |

### Training Configuration

| Parameter | Value |
|-----------|:-----:|
| Network | Actor-Critic RNN (GRU 128, FC 128) |
| Observation dim | 21 (16 base + 5 past action) |
| Action space | Discrete [31, 41, 41, 41, 5] |
| Reward scale | tracking=2.0, altitude=1.0, crash=−200, G-penalty=1.0 |
| Curriculum | Mixed sampling, progress denominator 300 |
| G-penalty coef/clip | 0.05 / 5.0 |
| Parallel envs | 1000 |
| Steps per episode | 2000 |

### Key Design Decisions

1. **Trim initialization** — Aircraft starts at roll=0, pitch=0, vt=250 m/s, alt=5000 m (eliminated 75% early crash rate from random init)
2. **Mixed-sampling curriculum** — Target difficulty sampled probabilistically per axis (L0:80%→5%, L3:0%→60%), no hard stage boundaries
3. **Earned-only curriculum advancement** — `heading_turn_counts` only increments when tracking quality is good (θ<5°, ΔV<15 m/s), preventing safety-timeout-driven curriculum runaway
4. **Adaptive check intervals** — Target switch intervals vary by level (L0=55, L1=120, L2=210, L3=250 RL steps)
5. **Past action in observation** — Previous step's normalized control surface commands appended to obs, reducing RNN hidden state dependence
