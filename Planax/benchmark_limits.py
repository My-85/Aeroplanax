"""
Planax Maximum Throughput Exploration
Goal: Find the REAL performance ceiling.
Tests:
  1. Single GPU: push num_envs from 10K → 1M (find saturation/OOM)
  2. Dual GPU (pmap): measure multi-GPU scaling
  3. Report theoretical max and what limits it (compute vs memory)
"""

import time, json, os
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import pmap

from envs.aeroplanax_heading_pitch_V import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
)

NUM_WARMUP = 3
NUM_TIMED  = 10
SCAN_LEN   = 100   # shorter to avoid JIT timeout at huge batches

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def random_actions(rng, agents, num_envs):
    keys = jax.random.split(rng, len(agents))
    return {a: jax.random.randint(k, (num_envs, 4), 0, 31)
            for k, a in zip(keys, agents)}


def gpu_mem_mb():
    """Current GPU-0 memory usage in MiB via nvidia-smi."""
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip().split("\n")
    return [int(x) for x in out]


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark A: Single GPU, sweep huge batch sizes
# ──────────────────────────────────────────────────────────────────────────────
def bench_single_gpu(env, params, batch_sizes):
    agents = env.agents
    interaction = params.agent_interaction_steps
    results = []

    step_fn = jax.jit(jax.vmap(env.step, (0, 0, 0, None)))
    reset_fn = jax.jit(jax.vmap(env.reset, (0, None)))

    print(f"\n{'='*72}")
    print("  Single GPU (A100 #0)  —  HeadingPitchV env")
    print(f"{'='*72}")
    print(f"  {'num_envs':>10}  {'env_sps':>12}  {'sim_sps':>14}  {'GPU MB':>8}  {'ms/call':>9}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*8}  {'-'*9}")

    prev_state = None
    prev_n = 0
    rng = jax.random.PRNGKey(7)

    for num_envs in batch_sizes:
        rng, rng_reset = jax.random.split(rng)
        try:
            # Reset
            rng_keys = jax.random.split(rng_reset, num_envs)
            obs_v, state_v = reset_fn(rng_keys, params)
            jax.block_until_ready(state_v)

            def carry_step(carry, _):
                state, rng = carry
                rng, ra, rs = jax.random.split(rng, 3)
                acts = random_actions(ra, agents, num_envs)
                skeys = jax.random.split(rs, num_envs)
                _, ns, _, _, _ = step_fn(skeys, state, acts, params)
                return (ns, rng), None

            @jax.jit
            def run_scan(state, rng):
                (fs, _), _ = jax.lax.scan(carry_step, (state, rng), None, SCAN_LEN)
                return fs

            # Warmup
            for _ in range(NUM_WARMUP):
                state_v = run_scan(state_v, rng)
                jax.block_until_ready(state_v)
                rng, _ = jax.random.split(rng)

            # Timed
            elapsed = []
            for _ in range(NUM_TIMED):
                rng, ru = jax.random.split(rng)
                t0 = time.perf_counter()
                state_v = run_scan(state_v, ru)
                jax.block_until_ready(state_v)
                elapsed.append(time.perf_counter() - t0)

            steps_per_call = num_envs * SCAN_LEN
            sps  = [steps_per_call / t for t in elapsed]
            ssps = [steps_per_call * interaction / t for t in elapsed]
            mem  = gpu_mem_mb()

            row = dict(
                mode="single_gpu",
                num_envs=num_envs,
                env_sps_mean=float(np.mean(sps)),
                env_sps_max=float(np.max(sps)),
                env_sps_std=float(np.std(sps)),
                sim_sps_mean=float(np.mean(ssps)),
                ms_per_call=float(np.mean(elapsed)*1000),
                gpu0_mb=mem[0],
            )
            results.append(row)
            print(f"  {num_envs:>10,}  {row['env_sps_mean']:>12,.0f}  "
                  f"{row['sim_sps_mean']:>14,.0f}  {mem[0]:>7}M  "
                  f"{row['ms_per_call']:>8.0f}ms")

        except Exception as e:
            err = str(e)[:80]
            print(f"  {num_envs:>10,}  FAILED: {err}")
            break

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark B: Dual GPU with pmap
# ──────────────────────────────────────────────────────────────────────────────
def bench_dual_gpu(env, params, batch_sizes_per_gpu):
    """Split envs evenly across 2 GPUs using pmap."""
    agents = env.agents
    interaction = params.agent_interaction_steps
    results = []
    n_devices = jax.device_count()

    print(f"\n{'='*72}")
    print(f"  Dual GPU (pmap, {n_devices} devices)  —  HeadingPitchV env")
    print(f"{'='*72}")
    print(f"  {'total_envs':>10}  {'env_sps':>12}  {'sim_sps':>14}  {'ms/call':>9}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*9}")

    # pmap-compatible step: each device handles N/n_devices envs
    p_reset = jax.pmap(jax.vmap(env.reset, (0, None)), in_axes=(0, None))
    p_step  = jax.pmap(jax.vmap(env.step, (0, 0, 0, None)), in_axes=(0, 0, 0, None))

    rng = jax.random.PRNGKey(42)

    for n_per_gpu in batch_sizes_per_gpu:
        total_envs = n_per_gpu * n_devices
        rng, rng_r = jax.random.split(rng)
        try:
            # Reset: shape (n_devices, n_per_gpu, ...)
            all_keys = jax.random.split(rng_r, total_envs).reshape(n_devices, n_per_gpu, -1)
            obs_v, state_v = p_reset(all_keys, params)
            jax.block_until_ready(state_v)

            def pmapped_scan_body(carry, _):
                state, rng = carry   # rng shape (n_devices, 2)
                rng, ra, rs = jax.vmap(lambda r: jax.random.split(r, 3))(rng).transpose(1, 0, 2)
                acts = {a: jax.vmap(lambda k: jax.random.randint(k, (n_per_gpu, 4), 0, 31))(ra)
                        for a in agents}
                skeys = jax.vmap(lambda k: jax.random.split(k, n_per_gpu))(rs)
                _, ns, _, _, _ = p_step(skeys, state, acts, params)
                return (ns, rng), None

            # Simpler approach: scan inside pmap
            def single_device_scan(state, rng):
                def body(carry, _):
                    st, rng = carry
                    rng, ra, rs = jax.random.split(rng, 3)
                    acts = {a: jax.random.randint(ra, (n_per_gpu, 4), 0, 31) for a in agents}
                    skeys = jax.random.split(rs, n_per_gpu)
                    vmap_step = jax.vmap(env.step, (0, 0, 0, None))
                    _, ns, _, _, _ = vmap_step(skeys, st, acts, params)
                    return (ns, rng), None
                (fs, _), _ = jax.lax.scan(body, (state, rng), None, SCAN_LEN)
                return fs

            p_scan = jax.pmap(single_device_scan)

            # Warmup
            dev_rngs = jax.random.split(rng, n_devices)
            for _ in range(NUM_WARMUP):
                state_v = p_scan(state_v, dev_rngs)
                jax.block_until_ready(state_v)
                rng, _ = jax.random.split(rng)
                dev_rngs = jax.random.split(rng, n_devices)

            # Timed
            elapsed = []
            for _ in range(NUM_TIMED):
                rng, ru = jax.random.split(rng)
                dev_rngs = jax.random.split(ru, n_devices)
                t0 = time.perf_counter()
                state_v = p_scan(state_v, dev_rngs)
                jax.block_until_ready(state_v)
                elapsed.append(time.perf_counter() - t0)

            steps_per_call = total_envs * SCAN_LEN
            sps  = [steps_per_call / t for t in elapsed]
            ssps = [steps_per_call * interaction / t for t in elapsed]

            row = dict(
                mode="dual_gpu",
                num_envs=total_envs,
                n_per_gpu=n_per_gpu,
                env_sps_mean=float(np.mean(sps)),
                env_sps_max=float(np.max(sps)),
                sim_sps_mean=float(np.mean(ssps)),
                ms_per_call=float(np.mean(elapsed)*1000),
            )
            results.append(row)
            print(f"  {total_envs:>10,}  {row['env_sps_mean']:>12,.0f}  "
                  f"{row['sim_sps_mean']:>14,.0f}  {row['ms_per_call']:>8.0f}ms")

        except Exception as e:
            print(f"  {total_envs:>10,}  FAILED: {str(e)[:80]}")
            break

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Planax MAXIMUM Throughput Exploration")
    print(f"JAX {jax.__version__}  |  devices: {jax.devices()}")
    print(f"State bytes/env: 240  |  A100 VRAM: 80 GB")
    print(f"Theoretical max envs by VRAM (×3 buffer): ~{80*1024**3 // (240*3):,}")
    print(f"Scan length: {SCAN_LEN}  |  Warmup: {NUM_WARMUP}  |  Timed: {NUM_TIMED}\n")

    params = Heading_Pitch_V_TaskParams()
    env    = AeroPlanaxHeading_Pitch_V_Env(params)

    # ── Single GPU: up to saturation ──────────────────────────────────────────
    single_sizes = [
        10_000, 20_000, 50_000,
        100_000, 200_000, 500_000,
        1_000_000, 2_000_000, 5_000_000,
    ]
    single_results = bench_single_gpu(env, params, single_sizes)

    # ── Dual GPU: match single-GPU batch sizes ────────────────────────────────
    # n_per_gpu list (total = 2×)
    dual_per_gpu = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]
    dual_results = bench_dual_gpu(env, params, dual_per_gpu)

    # ── Save & summarise ──────────────────────────────────────────────────────
    all_results = single_results + dual_results
    with open("benchmark_limits.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved → benchmark_limits.json")

    # Peak
    if single_results:
        best_s = max(single_results, key=lambda x: x["env_sps_mean"])
        print(f"\nSingle GPU peak:  {best_s['env_sps_mean']:>15,.0f} env_steps/s  "
              f"(sim: {best_s['sim_sps_mean']:>15,.0f})  @ num_envs={best_s['num_envs']:,}")
    if dual_results:
        best_d = max(dual_results, key=lambda x: x["env_sps_mean"])
        print(f"Dual GPU   peak:  {best_d['env_sps_mean']:>15,.0f} env_steps/s  "
              f"(sim: {best_d['sim_sps_mean']:>15,.0f})  @ num_envs={best_d['num_envs']:,}")

if __name__ == "__main__":
    main()
