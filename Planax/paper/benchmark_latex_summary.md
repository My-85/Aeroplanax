# Planax Benchmark Suite — LaTeX Snippet Summary

**Generated for:** IEEE RA-L manuscript revision  
**Date:** 2026-05-04  
**Branch:** phase2b_add_obs  
**Purpose:** Overleaf-ready LaTeX for Section III-C / IV (before experiments)

---

## 1. What Was Generated

Three LaTeX blocks ready for insertion:

| Block | LaTeX environment | File / key |
|-------|------------------|-----------|
| Benchmark Suite paragraph | `\subsection{...}` prose | — |
| Table I: Benchmark Tasks | `table*` (double-column, `footnotesize`) | `tab:tasks` |
| Table II: Training & Evaluation Settings | `table` (single-column, `small`) | `tab:training` |
| Implementation paragraph | inline prose | — |

**Required preamble packages:** `booktabs`, `siunitx`, `multirow`, `array`

---

## 2. Verified Codebase Facts

### 2.1 Simulation Kernel

| Parameter | Value | Source |
|-----------|-------|--------|
| Dynamics model | 6-DoF F-16 | `envs/core/simulators/fighterplane/` |
| Aerodynamics | Tensor LUT (bilinear / trilinear interp., `.dat` tables) | `fighterplane/aero_data.py` |
| **No MLP surrogate** | ✓ JAX version uses data files, not `.pth` networks | `dynamics/F16_jax/` vs `dynamics/F16_torch/` |
| Simulation frequency | 50 Hz → dt_sim = 0.02 s | `aeroplanax.py:38` `sim_freq=50` |
| Control skip | 10 sim steps → dt_ctrl = 0.20 s | `agent_interaction_steps=10` |
| Internal state dim | 26 | `aeroplanax.py:340` |

### 2.2 Task Inventory

| Task class | File | Agents | max\_steps (s) | obs\_dim | action |
|-----------|------|--------|---------------|---------|--------|
| `AeroPlanaxHeadingEnv` | `aeroplanax_heading.py` | 1 | ≤30 s (unreach) | 16 | 4-ch disc. (31,41,41,41) |
| `AeroPlanaxHeading_Pitch_V_Env` | `aeroplanax_heading_pitch_V.py` | 1 | ≤5 s (unreach) | 16 | 4-ch disc. (31,41,41,41) |
| `AeroPlanaxSManeuverEnv` | `aeroplanax_s_maneuver.py` | 1 | 200 waypoints | 16 | 4-ch disc. (31,41,41,41) |
| `AeroPlanaxWaypointEnv` | `aeroplanax_waypoint.py` | 1 | 3000 s | 16 | 4-ch disc. or 3-ch HLA |
| `AeroPlanaxFullDomainEnv` | `aeroplanax_full_domain_maneuver.py` | 1 | ≤40 s (timeout) | 22 | 4-ch disc. (31,41,41,41) |
| `AeroPlanaxFormationEnv` (2-agent) | `aeroplanax_formation.py` | 2 | TODO | 23 | 3-ch HLA (30,30,30) |
| `AeroPlanaxFormationEnv` (5-agent) | `aeroplanax_reformation.py` | 5 | TODO | 23 | 3-ch HLA (30,30,30) |
| `AeroPlanaxCombatEnv` | `aeroplanax_combat.py` | 2 (1v1) | 100 s | 21 | 3-ch HLA |
| `AeroPlanaxCombatwithMissileEnv` | `aeroplanax_combat_with_missile.py` | 1+missile | 100 s | 10 | continuous 3-D Box[−1,1] |

> **Note on `max_steps` units:** `timeout.py:9` has source comment "这里其实是多少秒的意思"
> (= "this value actually means how many seconds"). Confirmed: `max_steps` is in **seconds**.
> Threshold = `max_steps × sim_freq / agent_interaction_steps` in state.time units.

### 2.3 Observation Spaces

**HeadingPitchV / S-Maneuver / WaypointNavigation — 16-D** (`aeroplanax_heading_pitch_V.py:354–396`)

| Index | Feature | Normalization |
|-------|---------|--------------|
| 0 | Δψ (heading error) | raw rad |
| 1 | Δθ (pitch error) | raw rad |
| 2 | ΔV_t | /340 |
| 3 | altitude h | /5000 |
| 4 | airspeed V_t | /340 |
| 5–6 | sin φ, cos φ | — |
| 7–8 | sin θ, cos θ | — |
| 9–10 | sin α, cos α | — |
| 11–12 | sin β, cos β | — |
| 13–15 | p, q, r (body rates) | rad/s |

**FullDomainManeuver — 22-D** (`aeroplanax_full_domain_maneuver.py:851–864`)

| Slice | Feature |
|-------|---------|
| 0–3 | Quaternion error (w,x,y,z) |
| 4–5 | Speed error /100, speed ratio |
| 6–8 | p/10, q/10, r/10 |
| 9–11 | v_x/100, v_y/100, v_z/100 |
| 12–13 | h/10000, ḣ/100 |
| 14 | Specific energy /1e5 |
| 15 | Load factor N_z /9 |
| 16 | Dynamic pressure /1e4 |
| 17 | Flight path angle /π |
| 18–19 | roll_norm/π, pitch_norm/(π/2) |
| 20 | curriculum\_level / 7 |
| 21 | on\_target\_ratio |

**FormationFlying / Re-Formation — 23-D per agent** (`aeroplanax_reformation.py:527–624`)
- Own (18-D): Δn, Δe, Δh (km); roll, pitch, ψ_rel; ΔV_t/340; h; V_t; overload; α, β; p, q, r; a_x, a_y, a_z
- Neighbour (5-D per agent, topK=1): Δn, Δe, Δh (km); ΔV_t/340; AO

**AdversarialPursuit — 21-D** (`aeroplanax_combat.py:258–265`)
- Own: altitude, sin/cos φ, sin/cos θ, V_t
- Relative: ΔV_t, Δh, AO, TA, range, side flag

### 2.4 Action Spaces

**4-channel discrete** (HeadingControl, HeadingPitchV, S-Maneuver, WaypointNavigation, FullDomain):
```
throttle : Discrete(31)  →  actions[0]/30          ∈ [0, 1]
elevator : Discrete(41)  →  actions[1]*2/40 - 1    ∈ [-1, 1]
aileron  : Discrete(41)  →  actions[2]*2/40 - 1    ∈ [-1, 1]
rudder   : Discrete(41)  →  actions[3]*2/40 - 1    ∈ [-1, 1]
```
Source: `aeroplanax.py:154–157`

**3-channel High-Level Action (HLA)** (FormationFlying, Re-Formation):
```python
norm_delta_pitch   = jnp.linspace(-π/6,  π/6,  30)   # ±30°
norm_delta_heading = jnp.linspace(-π/2,  π/2,  30)   # ±90°
norm_delta_vt      = jnp.linspace(-100., 100., 30)   # ±100 m/s
```
Source: `aeroplanax_reformation.py:185–187`
> HLA commands decoded by pre-trained RNN baseline controller → 4-channel deflections.

### 2.5 Reward Functions

**HeadingPitchV** (`heading_pitch_V_reward.py:39–90`)
```
θ_geo  = geodesic_angle(q_curr, q_target)       # quaternion error
r_att  = exp(-(θ_geo / deg2rad(5.0))²)          # σ = 5°
r_spd  = exp(-(ΔV_t / 24.0)²)                   # σ = 24 m/s
reward = clip(r_att^0.8 * r_spd^0.2, 0, 1)      # geometric mean
active when: is_alive OR is_locked
```

**FullDomainManeuver** (`full_domain_reward.py`, `REWARD_CONFIG` lines 63–115)
```
r_att_gaussian = 0.40·exp(-(θ/80°)²) + 0.35·exp(-(θ/20°)²) + 0.25·exp(-(θ/5°)²)
r_att_cosine   = (1 + cos θ) / 2
r_att          = 0.6·r_att_cosine + 0.4·r_att_gaussian
r_spd          = exp(-(ΔV_t/30)²)
r_tracking     = 2.0·(0.75·r_att + 0.25·r_spd)
r_progress     = clip(5·Δθ/π, -0.5, 1.5)            # Δθ = prev_θ - θ
on-target att  = +1.5  if θ ≤ 10°
on-target spd  = +1.0  if |ΔV_t|≤15 AND θ≤30°
on-target both = +2.0  if θ≤10° AND |ΔV_t|≤25
on-target close= +0.3  if 10°<θ≤30°
alive bonus    = +0.05
crash penalty  = -0.3
level scale    = 1 + 0.1·curriculum_level
total          = clip(r_alive_total·level_scale, -1.0, 8.0)
```

**FormationFlying** (`formation_reward.py:5–16`)
```
r = -||target_pos - current_pos|| / 1000   (m, normalized to km)
mask = is_alive OR is_locked
```

**WaypointNavigation** (`aeroplanax_waypoint.py:282–289`)
```
r = r_distance(scale=1.0) + r_alignment(scale=0.3) + r_speed_profile(scale=0.1)
  + r_reach_bonus(bonus=3.0) - penalty_crash(pen=5.0)
```

**AdversarialPursuit** (`posture_reward.py`, `event_driven_reward.py`)
```
orientation_reward = 1/(50·AO/π + 2) + 0.5 + min(arctanh(1-2·TA/π)/(2π), 0) + 0.5
range_reward       = piece-wise: +1 if R<5, quadratic if R∈[5,∞), +clip(exp(-0.16R),0,0.2)
event reward       = ±200 (win=+200, loss/crash=-200)
```

### 2.6 Termination Conditions

| Condition | Threshold | Source |
|-----------|-----------|--------|
| Low altitude (most) | < 750 ft | `low_altitude.py:9` |
| Low altitude (FullDomain) | < 1000 ft | `aeroplanax_full_domain_maneuver.py:124` |
| Overload | > 10 G | `overload.py:10`, `crashed.py:42–48` |
| FullDomain N_z hard limit | > 30 G | `aeroplanax_full_domain_maneuver.py:129` |
| Low speed | V_t/340 < 0.01 | `low_speed.py:10` |
| Extreme AoA | α ∉ [−20°, 45°] | `extreme_state.py:11–12` |
| Extreme sideslip | \|β\| > 30° | `extreme_state.py:13–14` |
| Default timeout | max\_steps = 400 s | `timeout.py:9` |
| HeadingPitchV unreach | 5 s window, tol 5°/5°/10 m/s | `unreach_heading_pitch_V.py` |
| FullDomain unreach | 12 s window, θ<10°, \|ΔV_t\|<15 | `unreach_full_domain.py:63–66` |
| Formation unreach | dist < 200 m | `unreach_formation.py:15` |
| S-Maneuver done | waypoints_reached ≥ 200 | `aeroplanax_s_maneuver.py:99` |
| Combat timeout | max\_steps = 100 s | `aeroplanax_combat.py:202` |

### 2.7 Training Hyperparameters

All values from config dicts in `train_*.py` files:

| Parameter | HeadingPitchV | FullDomain | Combat | Formation |
|-----------|--------------|------------|--------|-----------|
| Algorithm | PPO | PPO (extended) | IPPO | IPPO |
| LR | 3e-4 | 2e-4 | 3e-4 | 3e-4 |
| NUM_ENVS | 1000 | 1000 | 300 | 1000 |
| NUM_STEPS | 2000 | 1000 | 1000 | 3000 |
| TOTAL_TIMESTEPS | 3.2×10⁸ | 2.0×10⁹ | 3.0×10⁸ | 1.0×10⁹ |
| GAMMA | 0.99 | 0.99 | 0.99 | 0.99 |
| GAE_LAMBDA | 0.95 | 0.95 | 0.95 | 0.95 |
| CLIP_EPS | 0.20 | 0.20 | 0.20 | 0.20 |
| ENT_COEF | 1e-3 | 2e-3 | 1e-3 | 1e-3 |
| VF_COEF | 1.0 | 1.0 | 1.0 | 1.0 |
| MAX_GRAD_NORM | 2.0 | 2.0 | 2.0 | 2.0 |
| UPDATE_EPOCHS | 16 | 16 | 16 | 16 |
| NUM_MINIBATCHES | 5 | 5 | 5 | 5 |
| FC_DIM_SIZE | 128 | 128 | 128 | 128 |
| GRU_HIDDEN_DIM | 128 | 128 | 128 | 128 |
| ACTIVATION | relu | relu | relu | relu |
| SEED | 42 | 42 | 42 | 42 |

**FullDomainManeuver additional PPO defaults** (`train_full_domain_maneuver.py:104–121`):
```
VF_CLIP_EPS=0.20, HUBER_DELTA=1.0, TARGET_KL=0.02, KL_STOP_MULT=1.5
ENT_COEF_MIN=1e-3, ENT_COEF_MAX=5e-2, ENT_ADJ_RATE=1.05
LR_DECAY=0.999, MIN_LR_MULT=0.2, WARMUP_UPDATES=2000
KL_START_MULT=5.0, KL_RAMP_UPDATES=1000
FREEZE_ENTROPY_DURING_WARMUP=True, FREEZE_LR_DURING_WARMUP=True
DISABLE_KL_STOP_DURING_WARMUP=True
```

### 2.8 Evaluation Metrics (`baseline_evaluate.py`)

```
NUM_EPISODES     = 15
STEPS_LIMIT      = 1000
GREEDY_ACTION    = True
HEAD_LABELS      = ["throttle", "elevator", "aileron", "rudder"]
ACTION_DIMS      = [31, 41, 41, 41]
```

Logged per checkpoint:
- `return_sum_mean / std` — cumulative reward statistics
- `length_mean / std` — episode length
- `pmax_mean_per_head` — mean max probability per action head
- `pmax_ge_0.9_per_head` — fraction of steps with max-prob ≥ 0.9
- `margin_mean_per_head` — top-1 minus top-2 probability margin
- `entropy_mean_per_head` — Shannon entropy per head
- `mode_change_rate_per_head` — action switching frequency
- `dwell_steps_stats_per_head` — dwell length statistics (mean, median, p10, p90)

### 2.9 Throughput Benchmarks

**Standard benchmark** (`benchmark_results.json`, HeadingPitchV, 1 agent):

| num_envs | env_sps_mean | ms_per_call |
|----------|-------------|-------------|
| 1 | 171 | 1173 ms |
| 100 | 16,707 | 1197 ms |
| 1,000 | 151,273 | 1325 ms |
| 5,000 | 708,532 | 1414 ms |
| 10,000 | **1,480,715** | 1352 ms |

S-Maneuver and FullDomainManeuver are within ±5% of HeadingPitchV at all batch sizes.

**Scaling limits** (`benchmark_limits.json`, single GPU):

| num_envs | env_sps_mean | VRAM (MB) |
|----------|-------------|-----------|
| 10,000 | 1,305,332 | 463 |
| 100,000 | 11,046,126 | 583 |
| 1,000,000 | **23,515,507** | 2,505 |
| 2,000,000 | 23,693,546 | 2,505 |
| 5,000,000 | 22,562,035 | 8,651 |

**Scaling limits, dual GPU:**

| num_envs | env_sps_mean |
|----------|-------------|
| 200,000 | 22,026,978 |
| 500,000 | 29,332,652 |
| **1,000,000** | **38,711,385** |

---

## 3. Missing Information (TODOs for manual fill)

| # | Item | Where to check |
|---|------|---------------|
| 1 | **GPU model** | Run `python -c "import jax; print(jax.devices())"` in the benchmark environment; no hardware label in JSON |
| 2 | **FormationFlying / Re-Formation episode horizon** | `unreach_formation.py:13` has `max_check_interval=100` — unit (seconds vs policy steps) not confirmed by source comment. Check what `state.time` increments by in `aeroplanax.py`. |
| 3 | **HeadingControl episode horizon** | `unreach_heading.py:13` has `max_check_interval=30` — same unit ambiguity |
| 4 | **AdversarialPursuit HLA bin count** | Number of bins per channel not confirmed in combat HLA; verify `aeroplanax_combat.py:~250–265` |
| 5 | **WaypointNavigation `max_waypoints`** | Base `aeroplanax_waypoint.py` `WaypointTaskParams` value not confirmed (vertical loop variant has 100) |
| 6 | **S-Maneuver obs\_dim direct confirmation** | Reported as 16 by inference from shared structure; confirm `_get_obs_size()` in `aeroplanax_s_maneuver.py` |
| 7 | **FullDomain curriculum schedule** | `curriculum_level/7` in obs implies 0–7 levels; verify curriculum advancement logic in env reset |
| 8 | **Training wall-clock / convergence curves** | Not in any inspected file; add from W&B runs if available |

---

## 4. Key Constraints for the Paper

- **Aerodynamics:** "Tensor LUT aerodynamics" or "look-up table (LUT) aerodynamics"; do NOT use "MLP surrogate" or "neural aerodynamic model" — the JAX version uses `.dat` data files with interpolation, not `.pth` neural networks.
- **Combat language:** Use "adversarial pursuit-evasion", "competitive tracking", or "multi-agent interaction" — avoid "combat", "kill", "shoot down" as primary descriptors.
- **Throughput numbers:** All figures come from `benchmark_results.json` / `benchmark_limits.json`; do not round beyond 3 significant figures without noting it.
- **Overclaiming words to avoid:** "unprecedented", "eliminate sim-to-real gap", "guarantee", "strictly proves".

---

## 5. LaTeX Snippet Location

The complete Overleaf-ready LaTeX (paragraph + Table I + Table II + implementation paragraph)
is in the previous assistant message in this conversation.
Copy-paste directly into the manuscript after `\subsection{...}` in Section III-C or Section IV.
