#!/usr/bin/env python3
"""
validate_reward.py — Validate that a reward function can be JAX-traced.

Usage:
    python validate_reward.py

Exit codes:
    0 = reward function is JAX-traceable
    1 = reward function fails JAX tracing (error printed to stderr)

This script is called by experiment_runner.py as a subprocess to validate
Claude-generated reward code before committing to a full training run.
It avoids import cache pollution in the parent process.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["XLA_PYTHON_MEM_FRACTION"] = "0.30"

import sys
from pathlib import Path

AUTOTUNER_DIR = Path(__file__).resolve().parent
PLANAX_DIR = AUTOTUNER_DIR.parent / "Planax"
sys.path.insert(0, str(PLANAX_DIR))

import jax
import jax.numpy as jnp


def validate():
    """Try to JAX-trace the reward function. Returns True on success."""
    try:
        # Import env and create a dummy state via reset
        from envs.aeroplanax_quat_baseline_iter import (
            AeroPlanaxHeading_Pitch_V_Env as Env,
            Heading_Pitch_V_TaskParams as TaskParams,
        )
        from envs.reward_functions.quat_baseline_reward import quat_baseline_reward_fn

        import functools

        params = TaskParams()
        env = Env(params)
        rng = jax.random.PRNGKey(0)
        obsv, state = env.reset(rng, params)

        # Try to trace the reward function with jax.make_jaxpr
        reward_fn = functools.partial(quat_baseline_reward_fn, reward_scale=1.0)
        # The reward function is called per-agent via vmap in the env
        jaxpr = jax.make_jaxpr(
            lambda s: jax.vmap(reward_fn, in_axes=(None, None, 0))(s, params, jnp.arange(env.num_agents))
        )(state)

        print(f"OK: reward function traced successfully ({len(jaxpr.eqns)} equations)")
        return True

    except Exception as e:
        print(f"FAIL: JAX trace error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
