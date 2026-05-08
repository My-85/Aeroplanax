"""
Fig.5 完整基准测量
统一 scan_len，覆盖 N=1 到 OOM，同步采集 VRAM
"""
import time, json, os, subprocess
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax, jax.numpy as jnp
from envs.aeroplanax_heading_pitch_V import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
)

SCAN_LEN   = 50    # 统一 scan 长度
NUM_WARMUP = 3
NUM_TIMED  = 8

# 覆盖从 1 到 2000 万的 N 值
BATCH_SIZES = [
    1, 2, 5, 10, 20, 50, 100, 200, 500,
    1_000, 2_000, 5_000,
    10_000, 20_000, 50_000,
    100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000,
    10_000_000, 20_000_000,
]

def gpu_mem_mb():
    out = subprocess.check_output(
        ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]
    ).decode().strip().split("\n")
    return int(out[0])

def main():
    params = Heading_Pitch_V_TaskParams()
    env    = AeroPlanaxHeading_Pitch_V_Env(params)
    agents = env.agents
    interaction = params.agent_interaction_steps   # 10

    reset_fn = jax.jit(jax.vmap(env.reset, (0, None)))
    step_fn  = jax.jit(jax.vmap(env.step,  (0, 0, 0, None)))

    print(f"scan_len={SCAN_LEN}, warmup={NUM_WARMUP}, timed={NUM_TIMED}")
    print(f"{'N':>12}  {'time/step(ms)':>14}  {'SPS':>12}  {'sim_SPS':>13}  {'VRAM(MB)':>10}")
    print("-"*65)

    results = []
    rng = jax.random.PRNGKey(0)

    # 预热 XLA 图（避免首次编译污染小 N 的计时）
    _k = jax.random.split(rng, 1)
    _, _s = reset_fn(_k, params)
    jax.block_until_ready(_s)

    for N in BATCH_SIZES:
        rng, rr = jax.random.split(rng)
        try:
            rng_keys = jax.random.split(rr, N)
            _, state = reset_fn(rng_keys, params)
            jax.block_until_ready(state)

            def body(carry, _):
                st, rng = carry
                rng, ra, rs = jax.random.split(rng, 3)
                acts = {a: jax.random.randint(ra, (N, 4), 0, 31) for a in agents}
                skeys = jax.random.split(rs, N)
                _, ns, _, _, _ = step_fn(skeys, st, acts, params)
                return (ns, rng), None

            @jax.jit
            def run(state, rng):
                (fs, _), _ = jax.lax.scan(body, (state, rng), None, SCAN_LEN)
                return fs

            for _ in range(NUM_WARMUP):
                state = run(state, rng)
                jax.block_until_ready(state)
                rng, _ = jax.random.split(rng)

            elapsed = []
            for _ in range(NUM_TIMED):
                rng, ru = jax.random.split(rng)
                t0 = time.perf_counter()
                state = run(state, ru)
                jax.block_until_ready(state)
                elapsed.append(time.perf_counter() - t0)

            vram = gpu_mem_mb()

            t_per_step_ms = np.mean(elapsed) / SCAN_LEN * 1000   # ms
            env_sps  = N * SCAN_LEN / np.mean(elapsed)
            sim_sps  = env_sps * interaction

            row = dict(N=N, t_per_step_ms=float(t_per_step_ms),
                       env_sps=float(env_sps), sim_sps=float(sim_sps),
                       vram_mb=int(vram))
            results.append(row)
            print(f"{N:>12,}  {t_per_step_ms:>14.3f}  {env_sps:>12,.0f}  "
                  f"{sim_sps:>13,.0f}  {vram:>9,}")

        except Exception as e:
            print(f"{N:>12,}  OOM / ERROR: {str(e)[:60]}")
            break

    with open("fig5_planax_data.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved fig5_planax_data.json  ({len(results)} points)")

if __name__ == "__main__":
    main()
