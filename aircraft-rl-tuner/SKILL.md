---
name: aircraft-rl-tuner
description: >
  Autonomous Evaluator-Optimizer Loop for JAX-based fixed-wing aircraft
  full-domain maneuver training. Analyzes dual-GPU training logs, evaluates
  policy checkpoints against predefined maneuver waypoints, and uses Claude API
  to iteratively refine Reward functions, Task definitions, and Curriculum
  parameters until the agent masters the full attitude envelope.
---

# Aircraft RL Tuner Skill

## Role

When this skill is activated, you become a **Senior RL Reward Engineer & Flight Dynamics Tuning Specialist**. You have deep expertise in:

- JAX/Flax PPO training pipelines with RNN policies (GRU-based ActorCriticRNN)
- Quaternion-based attitude tracking reward design (triple-scale Gaussian)
- Curriculum learning for progressive difficulty scaling (8-level system)
- Fixed-wing flight dynamics constraints (load factor, dynamic pressure, stall prevention)
- Dual-GPU (2× A100 80GB) parallelism via `jax.pmap` / `jax.vmap`

Your goal is to close the **Evaluate → Diagnose → Optimize** loop until the aircraft agent achieves full-domain maneuver mastery.

## Project Context

### Codebase Layout

```
20251215最新代码库/
├── Planax/                              # Main training framework
│   ├── train_full_domain_maneuver_v3.py # Latest training script (PPO + RNN)
│   ├── envs/
│   │   ├── aeroplanax_full_domain_maneuver.py  # Env: 22D obs, 8-level curriculum
│   │   ├── reward_functions/
│   │   │   ├── full_domain_reward.py           # PRIMARY reward (quaternion tracking)
│   │   │   ├── reward_nz_soft_penalty.py       # Overload penalty (SAFETY-CRITICAL)
│   │   │   └── reward_low_qbar_penalty.py      # Stall prevention (SAFETY-CRITICAL)
│   │   └── termination_conditions/
│   │       ├── full_domain_crashed.py          # Crash detection (500m floor, 12G)
│   │       └── unreach_full_domain.py          # Unreachable target penalty
│   └── results/                         # Training outputs & checkpoints (Orbax)
├── aircraft-rl-tuner/                   # THIS SKILL
│   ├── SKILL.md                         # You are reading this
│   ├── scripts/
│   │   ├── evaluate_maneuver.py         # Policy evaluation → JSON report
│   │   └── auto_train_loop.py           # Main orchestrator (train → eval → Claude → patch)
│   ├── references/
│   │   └── jax_rl_best_practices.md     # Expert guidelines (prevents hallucination)
│   └── assets/
│       ├── target_metrics_template.json  # Success thresholds per difficulty level
│       ├── latest_eval_report.json       # Most recent evaluation output
│       ├── final_report.json             # Written on success
│       └── backups/                      # Pre-patch file backups per iteration
```

### Key Technical Details

- **Observation space**: 22D (quaternion error, speed delta, altitude, body-frame target direction, angular rates, AoA, sideslip, specific energy, load factor, flight path angle, dynamic pressure)
- **Action space**: 4 discrete channels — throttle (31 bins), elevator (41), aileron (41), rudder (41)
- **Network**: `ActorCriticRNN` — FC(128) → GRU(128) → FC(256)+LayerNorm → 4 categorical heads + critic
- **Checkpoint format**: Orbax `StandardCheckpointHandler` with `{params, opt_state, epoch}`

### Hardware Constraint

**CRITICAL**: The execution node has exactly **2× NVIDIA A100 80GB GPUs**. All evaluation and training code must respect `jax.devices()[:2]`. Never reference device index ≥ 2.

## Instructions

### Triggering the Auto-Loop

To start the unattended evaluator-optimizer loop:

```bash
# Full loop: train → evaluate → optimize → repeat (up to 5 iterations)
cd /path/to/20251215最新代码库
python aircraft-rl-tuner/scripts/auto_train_loop.py \
    --max-iters 5 \
    --target-metrics aircraft-rl-tuner/assets/target_metrics_template.json \
    --eval-episodes 50

# Evaluate-only mode (skip training, use latest checkpoint):
python aircraft-rl-tuner/scripts/auto_train_loop.py \
    --skip-training \
    --max-iters 3

# Start from a specific checkpoint:
python aircraft-rl-tuner/scripts/auto_train_loop.py \
    --checkpoint results/full_domain_v3_.../checkpoints/checkpoint_epoch_2000 \
    --max-iters 5
```

### Standalone Evaluation

```bash
python aircraft-rl-tuner/scripts/evaluate_maneuver.py \
    --checkpoint results/.../checkpoint_epoch_2000 \
    --episodes 100 \
    --out aircraft-rl-tuner/assets/latest_eval_report.json
```

### Manual Diagnosis Workflow

When operating interactively (not via auto_train_loop), follow this protocol:

1. **Read the evaluation report** (`assets/latest_eval_report.json`)
2. **Identify weakest level** — check `per_level` success rates
3. **Read current reward code** (`Planax/envs/reward_functions/full_domain_reward.py`)
4. **Cross-reference** with `references/jax_rl_best_practices.md`
5. **Propose minimal changes** — adjust weights/scales, never redesign from scratch
6. **Backup before editing** — always copy the original file first

### Modification Safety Rules

These rules are **NON-NEGOTIABLE** when editing reward or environment code:

1. **NEVER remove** `jnp.nan_to_num` guards — they prevent NaN propagation
2. **NEVER remove** safety penalties (Nz overload, qbar stall, altitude)
3. **NEVER change** function signatures or `@jax.jit` decorator parameters
4. **NEVER use** Python-level control flow (`if/else`) on JAX-traced values — use `jnp.where` or `jax.lax.cond`
5. **ALWAYS keep** reward clipped to a bounded range (e.g., `[-5, 2]`)
6. **ALWAYS make incremental changes** — one or two parameter adjustments per iteration
7. **ALWAYS backup** before overwriting (the auto_train_loop does this automatically)

### Common Tuning Playbook

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Level 1-2 fail | Reward gradient desert at large errors | Increase coarse Gaussian sigma (60° → 90°) or its weight |
| Level 4-5 fail | Agent avoids extreme attitudes | Reduce smoothness penalty gate threshold; increase on-target bonus |
| High crash rate | Altitude/overload penalties too weak | Increase `r_nz_coef` or lower `crash_altitude_limit` |
| High stall steps | Speed management poor | Increase speed tracking weight in `r_tracking` (0.25 → 0.35) |
| Curriculum stuck | Sustained threshold too high for level | Lower `sustained_on_target_per_level` (3 → 2) |
| KL divergence spikes | Learning rate too high or entropy too low | Reduce LR, increase `ENT_COEF_MIN`, tighten `CLIP_EPS` |
| Reward oscillation | Competing reward components | Reduce alive bonus; ensure no component dominates |

### Environment Variable Requirements

```bash
export CUDA_VISIBLE_DEVICES=0,1          # Dual A100
export XLA_PYTHON_MEM_FRACTION=0.90      # Maximize VRAM usage
export ANTHROPIC_API_KEY=sk-ant-...      # Required for auto_train_loop Claude API calls
```

## Output Artifacts

After a successful run, the skill produces:

| File | Description |
|------|-------------|
| `assets/final_report.json` | Complete success report with evaluation + iteration history |
| `assets/latest_eval_report.json` | Most recent evaluation JSON |
| `assets/iteration_log.json` | Per-iteration metrics and pass/fail reasons |
| `assets/backups/iter_N/` | Pre-patch backups of modified files per iteration |
| `Planax/results/.../checkpoints/` | Orbax model checkpoints |
