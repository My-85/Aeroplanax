# JAX RL Best Practices for Dual-GPU PPO Training & Reward Shaping

## 1. Dual-GPU (2× A100) JAX Parallelism

### Device Management
- Always verify available devices: `devices = jax.devices()[:2]`
- Never hardcode device indices beyond what's physically available
- Use `jax.local_device_count()` for runtime checks

### Parallelism Strategy (Single-node, 2 GPUs)
- **Data parallelism via `jax.pmap`**: Split `NUM_ENVS` across 2 GPUs. Each GPU runs `NUM_ENVS // 2` environments.
- **Vectorization via `jax.vmap`**: Within each GPU, vectorize across environments. `vmap` is free (no inter-device communication).
- **Avoid `pmap` for small batches**: If `NUM_ENVS < 64`, keep everything on one GPU with `vmap`. `pmap` overhead dominates at small scale.
- **Preferred pattern**: `pmap(vmap(env.step))` — outer pmap splits across GPUs, inner vmap vectorizes within each GPU.

### Memory Management
- A100 80GB can hold ~1000-2000 environments with 22D obs + RNN hidden state (128 dim)
- Set `XLA_PYTHON_MEM_FRACTION=0.90` to maximize usable VRAM
- Monitor with `jax.local_devices()[0].memory_stats()` if available

## 2. PPO Hyperparameter Guidelines

### Stable Defaults for Aircraft Control
| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `LR` | 1e-4 to 3e-4 | Start at 2e-4; reduce if KL divergence spikes |
| `CLIP_EPS` | 0.15 - 0.25 | 0.2 is standard; tighten to 0.15 for fine-tuning |
| `GAMMA` | 0.99 | Aircraft tasks have long horizons |
| `GAE_LAMBDA` | 0.95 | Standard; lower (0.9) reduces variance at cost of bias |
| `NUM_MINIBATCHES` | 4-8 | Must evenly divide NUM_ENVS |
| `UPDATE_EPOCHS` | 8-16 | More epochs = more sample efficiency but risk overfitting |
| `VF_COEF` | 0.5 - 1.0 | Higher if value loss dominates |
| `ENT_COEF` | 1e-3 to 5e-2 | Adaptive is better (see below) |
| `MAX_GRAD_NORM` | 1.0 - 5.0 | 2.0 works well; increase if gradients are frequently clipped |

### Adaptive Entropy Coefficient
- Track `approx_kl` per update
- If KL < 0.5 * target_kl: increase entropy (exploration needed)
- If KL > 1.5 * target_kl: decrease entropy (too much exploration)
- Bounds: [1e-3, 5e-2] to prevent collapse or chaos

### KL-based Early Stopping
- Monitor `approx_kl` across minibatch epochs
- Stop updating if `approx_kl > KL_STOP_MULT * TARGET_KL`
- Prevents catastrophic policy updates

### Warmup Phase
- First ~2000 updates: freeze LR decay, relax KL threshold (5× target)
- Allows initial exploration without premature convergence
- Gradually ramp KL threshold down over ~1000 updates

## 3. Reward Shaping for Full-Domain Aircraft Maneuvers

### Quaternion-based Attitude Tracking
- **Always use quaternion geodesic distance** — Euler angles have singularities (gimbal lock at ±90° pitch)
- Geodesic angle: `theta = 2 * arccos(|dot(q_curr, q_target)|)`
- **Triple-scale Gaussian reward** for smooth gradient across all error ranges:
  ```
  r_att = 0.15 * exp(-(theta/60°)²)   # coarse: gradient at 30-90°
        + 0.50 * exp(-(theta/20°)²)   # medium: gradient at 15-40°
        + 0.35 * exp(-(theta/5°)²)    # fine: precision at <10°
  ```
- Single-scale reward creates "gradient desert" — agent gets no signal at intermediate errors

### Combined Tracking Reward
- Use **weighted sum** (not product) of attitude and speed rewards:
  `r_tracking = 0.75 * r_attitude + 0.25 * r_speed`
- Product causes one bad component to zero out entire reward signal

### Safety Penalties (Must Keep)
- **Overload (Nz) penalty**: Quadratic penalty for |az| > 9G, hard cap at 15G
- **Low dynamic pressure (qbar)**: Penalty when qbar drops below threshold during high-pitch maneuvers (prevents stall)
- **Altitude safety**: Soft penalty below safe altitude, hard penalty below danger altitude
- These penalties MUST remain even when optimizing — removing them causes crashes

### On-target Bonus
- Give extra reward (+0.2 to +0.5) when both attitude error < 10° AND speed error < 15 m/s
- This creates a clear "attractor" in reward space

### Common Anti-patterns to Avoid
1. **Large alive bonus** (> 0.1): Encourages "just survive" strategy, agent avoids maneuvering
2. **Harsh crash penalty** (< -10): Causes extremely risk-averse behavior, agent refuses to pitch/roll
3. **Unscaled random targets at low curriculum**: Agent faces impossible targets early → learns nothing
4. **Fixed sustained_on_target threshold**: Too high at early curriculum → never advances
5. **Timeout counted as success**: Inflates curriculum level without real skill

## 4. Curriculum Learning Best Practices

### Progressive Difficulty (8-Level System)
- Level 0: ~15% of full range (small heading changes, level flight)
- Level 7: 100% of full range (arbitrary 3D attitude, full speed range)
- Scale factor: `scale = 0.15 + 0.85 * min(level / 7, 1.0)`

### Advancement Criteria
- Only count **sustained on-target** (N consecutive steps) as real success
- Sustained threshold scales with level: `threshold = base + per_level * level`
- Base = 3 steps, per_level = 3 → Level 7 requires 24 consecutive on-target steps
- Advancement requires `base_count + extra_per_level * level` real successes

### Target Generation
- **Dual mode**: 60% delta (relative to current), 40% random (curriculum-scaled)
- Both modes scale by curriculum level — prevents early overwhelming
- Include altitude-aware pitch bias: low altitude → bias upward (safety)

## 5. Numerical Stability in JAX

### Critical Guards
- `jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)` on ALL reward outputs
- Clip ratio to [1e-6, 1e6] in PPO loss
- Clip log-ratio to [-20, 20] before exp()
- Clip advantages after normalization
- Use Huber loss for value function (more robust than MSE)

### Gradient Safety
- `jnp.nan_to_num` on gradients before applying
- Compute global norm and scale if > MAX_GRAD_NORM
- Zero out gradients when KL early stopping is active

### Quaternion Normalization
- Always normalize quaternions before use: `q / (norm(q) + 1e-9)`
- Disambiguate sign: ensure `q[0] >= 0` (shortest rotation)
- Check for degenerate quaternions (norm near zero) → default to identity [1,0,0,0]

## 6. Checkpoint Management

### Orbax Pattern
```python
import orbax.checkpoint as ocp

# Save
state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": epoch}
ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
ckptr.save(path, args=ocp.args.StandardSave(state))
ckptr.wait_until_finished()

# Load
template = {"params": ..., "opt_state": ..., "epoch": jnp.array(0)}
checkpoint = ckptr.restore(path, args=ocp.args.StandardRestore(item=template))
```

### Best Practices
- Save after each `FOR_LOOP_EPOCH`, not just at the end
- Include epoch number in checkpoint path for versioning
- Always create a `state_template` with matching structure for restore

## 7. When Claude API Modifies Reward Code

### Safety Checklist
1. Never remove numerical stability guards (nan_to_num, clip)
2. Never change function signatures (breaks env integration)
3. Never remove safety penalties (Nz, qbar, altitude)
4. Keep reward clipped to reasonable range (e.g., [-5, 2])
5. Make incremental changes — adjust weights and scales, don't redesign
6. Always backup before overwriting

### Effective Modifications
- Adjust Gaussian scale parameters (sigma in exp(-(x/sigma)²))
- Tune weight ratios between reward components
- Add/modify on-target bonus thresholds
- Adjust curriculum advancement thresholds
- Modify altitude safety margins
- Scale smoothness penalty
