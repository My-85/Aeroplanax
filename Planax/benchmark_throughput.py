"""
Planax Platform Throughput Benchmark
Measures maximum steps/second under various parallelism settings.

Metrics:
  - env_steps/s : calls to env.step() per wall-clock second
  - sim_steps/s : physics substeps per second (env_steps × agent_interaction_steps)
  - total_agent_steps/s : env_steps × num_agents (what RL training counts)

Tests:
  1. HeadingPitchV  - simplest 1-agent task
  2. SManeuver      - 1-agent, more complex reward computation
  3. FullDomain     - 1-agent, heaviest reward/termination logic

Vectorization: jax.vmap(env.step) + jax.lax.scan, fully JIT compiled.
"""

import time
import json
import os
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

from envs.aeroplanax_heading_pitch_V import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
)
from envs.aeroplanax_s_maneuver import (
    AeroPlanaxSManeuverEnv, S_Maneuver_TaskParams,
)
from envs.aeroplanax_full_domain_maneuver import (
    AeroPlanaxFullDomainEnv, FullDomain_TaskParams,
)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_WARMUP = 5       # JIT warmup iterations
NUM_TIMED  = 20      # Timed iterations per setting
SCAN_LEN   = 200     # Steps per lax.scan call (trajectory length)

BATCH_SIZES = [1, 10, 50, 100, 300, 500, 1000, 2000, 5000, 10000]


def random_actions(rng, agents, num_envs):
    """Discrete actions: shape (num_envs, 4) int array per agent."""
    keys = jax.random.split(rng, len(agents))
    return {
        a: jax.random.randint(k, (num_envs, 4), 0, 31)
        for k, a in zip(keys, agents)
    }


def benchmark_env(label, env, params, batch_sizes):
    """Return list of result dicts for varying batch sizes."""
    agents = env.agents
    n_agents = len(agents)
    interaction = params.agent_interaction_steps
    results = []

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  agents={n_agents}, obs_dim={list(env.observation_space(agents[0], params).shape)}, "
          f"sim_freq={params.sim_freq}, interaction_steps={interaction}")
    print(f"{'='*65}")
    print(f"  {'num_envs':>8}  {'env_sps':>12}  {'sim_sps':>14}  {'agent_sps':>14}  {'ms/call':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*10}")

    # Precompile at batch=1 to avoid first-batch overhead affecting later runs
    _pre_keys = jax.random.split(jax.random.PRNGKey(0), 1)
    _pre_obs, _pre_state = jax.jit(jax.vmap(env.reset, (0, None)))(_pre_keys, params)
    jax.block_until_ready(_pre_state)

    for num_envs in batch_sizes:
        rng = jax.random.PRNGKey(num_envs)

        # ── Vectorized reset ──────────────────────────────────────────────
        try:
            reset_fn = jax.jit(jax.vmap(env.reset, (0, None)))
            step_fn  = jax.jit(jax.vmap(env.step,  (0, 0, 0, None)))

            rng_keys = jax.random.split(rng, num_envs)
            obs_v, state_v = reset_fn(rng_keys, params)
            jax.block_until_ready(state_v)
        except Exception as e:
            print(f"  {num_envs:>8}  RESET FAILED: {e}")
            break

        # ── Build scanned rollout ─────────────────────────────────────────
        def env_step_carry(carry, _):
            state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            acts = random_actions(rng_a, agents, num_envs)
            step_keys = jax.random.split(rng_s, num_envs)
            _, next_state, _, _, _ = step_fn(step_keys, state, acts, params)
            return (next_state, rng), None

        @jax.jit
        def run_scan(state, rng):
            (final_state, _), _ = jax.lax.scan(
                env_step_carry, (state, rng), None, SCAN_LEN
            )
            return final_state

        # ── JIT warmup ────────────────────────────────────────────────────
        try:
            for _ in range(NUM_WARMUP):
                state_v = run_scan(state_v, rng)
                jax.block_until_ready(state_v)
                rng, _ = jax.random.split(rng)
        except Exception as e:
            print(f"  {num_envs:>8}  WARMUP FAILED: {e}")
            break

        # ── Timed runs ────────────────────────────────────────────────────
        elapsed = []
        for _ in range(NUM_TIMED):
            rng, rng_use = jax.random.split(rng)
            t0 = time.perf_counter()
            state_v = run_scan(state_v, rng_use)
            jax.block_until_ready(state_v)
            elapsed.append(time.perf_counter() - t0)

        env_steps_per_call = num_envs * SCAN_LEN
        sim_steps_per_call = env_steps_per_call * interaction
        agent_steps_per_call = env_steps_per_call * n_agents

        env_sps   = [env_steps_per_call   / t for t in elapsed]
        sim_sps   = [sim_steps_per_call   / t for t in elapsed]
        agent_sps = [agent_steps_per_call / t for t in elapsed]
        ms_call   = [t * 1000 for t in elapsed]

        row = {
            "env":            label,
            "num_envs":       num_envs,
            "num_agents":     n_agents,
            "interaction":    interaction,
            "env_sps_mean":   float(np.mean(env_sps)),
            "env_sps_max":    float(np.max(env_sps)),
            "env_sps_std":    float(np.std(env_sps)),
            "sim_sps_mean":   float(np.mean(sim_sps)),
            "sim_sps_max":    float(np.max(sim_sps)),
            "agent_sps_mean": float(np.mean(agent_sps)),
            "agent_sps_max":  float(np.max(agent_sps)),
            "ms_per_call":    float(np.mean(ms_call)),
        }
        results.append(row)

        print(f"  {num_envs:>8}  {row['env_sps_mean']:>12,.0f}  "
              f"{row['sim_sps_mean']:>14,.0f}  {row['agent_sps_mean']:>14,.0f}  "
              f"{row['ms_per_call']:>9.1f}ms")

    return results


def main():
    print("Planax Throughput Benchmark")
    print(f"JAX {jax.__version__}  |  devices: {jax.devices()}")
    print(f"Scan length: {SCAN_LEN} steps  |  "
          f"Warmup: {NUM_WARMUP}  |  Timed: {NUM_TIMED}\n")

    all_results = []

    # ── Env A: HeadingPitchV (simplest) ──────────────────────────────────────
    env_a = AeroPlanaxHeading_Pitch_V_Env(Heading_Pitch_V_TaskParams())
    all_results += benchmark_env(
        "HeadingPitchV (1-agent)", env_a, Heading_Pitch_V_TaskParams(), BATCH_SIZES
    )

    # ── Env B: SManeuver ─────────────────────────────────────────────────────
    env_b = AeroPlanaxSManeuverEnv(S_Maneuver_TaskParams())
    all_results += benchmark_env(
        "SManeuver (1-agent)", env_b, S_Maneuver_TaskParams(), BATCH_SIZES
    )

    # ── Env C: FullDomainManeuver (most complex) ──────────────────────────────
    env_c = AeroPlanaxFullDomainEnv(FullDomain_TaskParams())
    all_results += benchmark_env(
        "FullDomainManeuver (1-agent)", env_c, FullDomain_TaskParams(), BATCH_SIZES
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'Env':<32} {'num_envs':>8}  {'env_sps':>12}  {'sim_sps':>14}  {'agent_sps':>14}")
    print("="*80)
    for r in all_results:
        print(f"{r['env']:<32} {r['num_envs']:>8}  "
              f"{r['env_sps_mean']:>12,.0f}  "
              f"{r['sim_sps_mean']:>14,.0f}  "
              f"{r['agent_sps_mean']:>14,.0f}")

    # Peak per env
    print("\n--- Peak throughput per environment ---")
    env_names = list(dict.fromkeys(r["env"] for r in all_results))
    for name in env_names:
        rows = [r for r in all_results if r["env"] == name]
        best = max(rows, key=lambda x: x["env_sps_mean"])
        print(f"  {name}")
        print(f"    env_steps/s : {best['env_sps_mean']:>15,.0f}  @ num_envs={best['num_envs']}")
        print(f"    sim_steps/s : {best['sim_sps_mean']:>15,.0f}")
        print(f"    agent_sps   : {best['agent_sps_mean']:>15,.0f}")


if __name__ == "__main__":
    main()
