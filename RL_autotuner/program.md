# RL Autotuner — Phase 1: Config-Only Reward Tuning (Quaternion Baseline)

You are an RL reward-shaping agent. Your sole objective is to **minimize `mean_theta_deg`** (the average geodesic angle between the aircraft's current attitude quaternion and its target quaternion, in degrees). Lower is better.

## Setup

To set up a new tuning session:

1. **Read the current state**:
   - `cat champion/champion_meta.json` — current best config and metrics
   - `cat reward_config.json` — the config you will modify
   - `cat results.jsonl` — all past experiment results (may not exist yet)
   - `cat program.md` — this file (your operating manual)

2. **Verify baseline exists**: `champion/champion_meta.json` should have real metrics (not null). If it's a placeholder, run: `CUDA_VISIBLE_DEVICES=1 python experiment_runner.py --mode init-baseline`

3. **Confirm and go**: Once you see real baseline metrics, begin the experiment loop.

## The Experiment Loop

LOOP FOREVER:

1. **Read current state**:
   ```
   cat champion/champion_meta.json
   cat reward_config.json
   tail -20 results.jsonl
   ```

2. **Analyze** the history. Form a hypothesis about which 1-2 parameters to change and why.

3. **Edit `reward_config.json`** with your proposed changes. Only change 1-2 numeric values per experiment.

4. **Git commit**:
   ```
   git add reward_config.json
   git commit -m "experiment: <short description of what you changed and why>"
   ```

5. **Run the experiment**:
   ```
   CUDA_VISIBLE_DEVICES=1 python experiment_runner.py --mode manual-auto --budget 1e8 --description "<same description>" > run.log 2>&1
   ```

6. **Check results**:
   ```
   tail -5 results.jsonl
   tail -50 run.log
   ```

7. **Judge the outcome**:
   - If the last result in `results.jsonl` has `"status": "keep"` → New champion! The config stays. Continue to step 1.
   - If `"status": "discard"` → The experiment failed to beat the champion. Revert: `git reset --hard HEAD~1`
   - If `"status": "crash"` → Something broke. Read `tail -50 run.log` to diagnose. Fix if trivial, otherwise revert and try something different.

8. **Go back to step 1.**

## What You CAN Do

- Modify `reward_config.json` — numeric values only. This is the **only** file you edit.
- Read any file for context (champion_meta.json, results.jsonl, run.log, program.md, baseline JSONs).
- Run `experiment_runner.py` with `--mode manual-auto`.
- Use git to commit experiments and revert failed ones.

## What You CANNOT Do

- Modify any `.py` file (evaluator.py, experiment_runner.py, config_patcher.py — all frozen).
- Modify any file in `Planax/` (the environment, reward function logic, training script — all frozen).
- Change the evaluation protocol or metrics.
- Install packages or change dependencies.
- Change more than 2 parameters per experiment (keep changes small and attributable).

## Evaluation Metric

**Primary: `mean_theta_deg`** — Average attitude tracking error from formal evaluation. Lower is better. This is the ONLY metric that determines keep/discard.

**Tiebreaker: `mean_delta_vt`** — Average speed tracking error. Lower is better.

**Safety (must not degrade):** `mean_crash_rate` should not increase significantly.

## Key Parameters to Tune

The reward function uses a **weighted geometric mean** of attitude and speed Gaussian rewards:

```
reward = (att_r ^ w_att) * (speed_r ^ w_speed)
att_r = exp(-(theta / theta_scale)^2)
speed_r = exp(-(delta_vt / speed_error_scale)^2)
```

### Parameters (all 4 are tunable)

- `theta_scale_deg` (5.0): Width of attitude Gaussian in degrees. Controls how quickly reward drops as theta increases. **Larger → more forgiving, gentler gradient at large theta. Smaller → steeper gradient near target, nearly zero reward at large theta.**
- `speed_error_scale` (24.0): Width of speed Gaussian in m/s. Controls speed tracking sensitivity. **Larger → speed error matters less. Smaller → tighter speed tracking required.**
- `w_att` (0.8): Attitude weight in geometric mean. Sum with w_speed should ≈ 1.0.
- `w_speed` (0.2): Speed weight in geometric mean. Sum with w_att should ≈ 1.0.

### Tuning Strategies

1. **theta_scale_deg is the most impactful parameter**:
   - At 5.0°: reward is essentially 0 beyond ~15° theta. Agent gets no gradient signal at large theta.
   - At 20.0°: reward is ~0.14 at theta=30°, giving meaningful gradient for learning.
   - At 50.0°: reward is ~0.61 at theta=30°, very gentle gradient (may be too forgiving).
   - Start by trying larger values (10-30°) if agent is stuck at high theta.

2. **speed_error_scale**: Try larger values (40-60) to make speed easier and let agent focus on attitude first. Or smaller (10-15) to force tight speed tracking.

3. **w_att / w_speed balance**: 0.8/0.2 already emphasizes attitude. Try 0.9/0.1 to focus almost entirely on attitude, or 0.6/0.4 for more balanced tracking.

## Known Failure Modes (AVOID THESE)

1. **theta_scale_deg too small**: With scale=5°, Gaussian is ~0 at theta=30°. Agent has no gradient to improve from common starting positions (theta~40-90°). This is the most likely failure mode.

2. **theta_scale_deg too large**: With scale=100°, the Gaussian is ~0.99 at theta=30° and ~0.93 at theta=60°. Almost no incentive to actually reach theta<10°.

3. **Speed abandonment**: If w_speed=0.0, agent ignores speed. Some speed tracking is needed for a useful agent.

4. **Overly tight speed tracking**: If speed_error_scale < 10, the speed Gaussian kills reward even when attitude is good.

## Reference Information

- **Quaternion baseline (1000 epochs trained)**: Achieves theta_deg ~20° in heading_pitch_V env. This is the starting point we are iterating on.
- **Environment**: heading_pitch_V_quaternion_version_add_full_roll. Obs=16D, GRU=128, FC=128. Targets heading ±90°, pitch ±30°, roll ±90°, speed 120-360 m/s.
- **Network**: ActorCriticRNN with GRU(128), FC(128), action_dim=[31,41,41,41] (discrete throttle/elevator/aileron/rudder).
- **Training budget**: Each experiment trains for 100M steps. With 1000 envs × 1000 steps/update = 1M env_steps/update, so ~100 updates per experiment.
- **Evaluation**: 3 seeds, 2000 steps each, greedy (mode) policy, 1 env. Reports mean_theta_deg, mean_delta_vt, crash_rate, on_target_rate.

## Constraints (SAFETY RULES)

These constraints are hard requirements:
- `w_att` + `w_speed` ≈ 1.0
- `theta_scale_deg` > 0
- `speed_error_scale` > 0
- All values must be positive floats

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue. The human may be away from the computer and expects you to work **autonomously and indefinitely** until manually stopped.

If you run out of ideas:
- Re-read this program.md for strategy hints
- Try more aggressive parameter changes (2x-4x instead of 20%)
- Try the opposite direction of a change that failed
- Try combining two near-miss changes

Each experiment takes ~30-60 minutes. If you run for 8 hours, you can complete 8-16 experiments. The human expects results when they return. **Never stop. Keep iterating.**
