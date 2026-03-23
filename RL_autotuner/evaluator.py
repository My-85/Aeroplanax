#!/usr/bin/env python3
"""
evaluator.py — Fixed evaluation protocol for full-domain maneuver checkpoints.

DO NOT MODIFY this file. It defines the immutable evaluation metric.

Loads a trained checkpoint, runs standardized test episodes with fixed seeds,
and reports physics-based metrics:
  - mean_theta_deg: average geodesic angle error (lower = better, PRIMARY METRIC)
  - mean_delta_vt: average speed error (lower = better)
  - crash_rate: fraction of steps where crash occurred
  - on_target_rate: fraction of steps with theta<10° AND delta_vt<25
  - mean_episodic_return: average per-step reward
  - curriculum_level: curriculum level reached

Usage:
    python evaluator.py --checkpoint PATH [--num-envs 200] [--num-steps 500] [--output metrics.json]
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["XLA_PYTHON_MEM_FRACTION"] = "0.90"

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
import functools
from typing import Sequence, Dict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

# Add Planax to path
AUTOTUNER_DIR = Path(__file__).resolve().parent
PLANAX_DIR = AUTOTUNER_DIR.parent / "Planax"
sys.path.insert(0, str(PLANAX_DIR))

from envs.wrappers import LogWrapper
from envs.aeroplanax_quat_baseline_iter import (
    AeroPlanaxHeading_Pitch_V_Env as AeroPlanaxFullDomainEnv,
    Heading_Pitch_V_TaskParams as FullDomain_TaskParams,
)

# ======================== Quaternion Helpers (self-contained, do NOT import from env) ========================

def _quat_normalize(q):
    """Normalize quaternion. q shape: (..., 4)."""
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)

def _quat_conj(q):
    """Quaternion conjugate. q shape: (..., 4) as [w, x, y, z]."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])

def _quat_from_euler_bn(roll, pitch, yaw):
    """ZYX Euler angles to q_BN quaternion. Scalar inputs or batched."""
    cr, sr = jnp.cos(0.5 * roll),  jnp.sin(0.5 * roll)
    cp, sp = jnp.cos(0.5 * pitch), jnp.sin(0.5 * pitch)
    cy, sy = jnp.cos(0.5 * yaw),   jnp.sin(0.5 * yaw)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return jnp.stack([qw, qx, qy, qz], axis=-1)

def _quat_geodesic_angle(q_a, q_b):
    """Geodesic angle between quaternions. q shape: (..., 4). Returns scalar or batch."""
    q_a = _quat_normalize(q_a)
    q_b = _quat_normalize(q_b)
    cos_half = jnp.abs(jnp.sum(q_a * q_b, axis=-1))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half)


# ======================== Network Architecture (must match training) ========================

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(nn_fc2)
        actor_mean = activation(actor_mean)
        pi_throttle = distrax.Categorical(
            logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        )
        pi_elevator = distrax.Categorical(
            logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        )
        pi_aileron = distrax.Categorical(
            logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        )
        pi_rudder = distrax.Categorical(
            logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        )

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)


# ======================== Evaluation Config (IMMUTABLE) ========================

EVAL_CONFIG = {
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION": "relu",
    "NUM_ENVS": 1,         # single eval env (3 seeds for stability)
    "NUM_STEPS": 1000,     # 1000 decision steps = 200s sim time at sim_per_decision=5
    "NUM_EPISODES": 1,     # single seed for fast screening
    "EVAL_SEEDS": [42],
}


def load_checkpoint(checkpoint_path: str, config: dict, eval_level: int = None) -> dict:
    """Load trained checkpoint with partial_restore (handles opt_state mismatch).

    Args:
        eval_level: If set, lock curriculum at this level (no advancement).
                    If None, curriculum advances normally (legacy behavior).
    """
    # Lock curriculum: set threshold impossibly high so it never advances
    if eval_level is not None:
        env_params = FullDomain_TaskParams(curriculum_advance_threshold=999999)
    else:
        env_params = FullDomain_TaskParams()
    env = AeroPlanaxFullDomainEnv(env_params)
    env = LogWrapper(env)
    num_actors = env.num_agents

    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    rng = jax.random.PRNGKey(0)
    obs_shape = env.observation_space(env.agents[0], env_params).shape

    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * num_actors, *obs_shape)),
        jnp.zeros((1, config["NUM_ENVS"] * num_actors)),
    )
    init_hstate = ScannedRNN.initialize_carry(
        config["NUM_ENVS"] * num_actors, config["GRU_HIDDEN_DIM"]
    )
    network_params = network.init(rng, init_hstate, init_x)

    params_template = {
        "params": network_params,
        "epoch": jnp.array(0),
    }

    ckptr = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    checkpoint = ckptr.restore(
        checkpoint_path,
        args=ocp.args.PyTreeRestore(item=params_template, partial_restore=True),
    )

    return {
        "params": checkpoint["params"],
        "epoch": int(checkpoint.get("epoch", 0)),
        "network": network,
        "env": env,
        "env_params": env_params,
        "num_actors": num_actors,
        "eval_level": eval_level,
    }


def run_eval_episode(loaded: dict, config: dict, seed: int) -> dict:
    """Run one evaluation episode and return metrics."""
    network = loaded["network"]
    params = loaded["params"]
    env = loaded["env"]
    env_params = loaded["env_params"]
    num_actors = loaded["num_actors"]
    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]

    rng = jax.random.PRNGKey(seed)

    # Reset envs — LogWrapper.reset takes only (key), no env_params
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    obsv, state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

    # Override curriculum_level if eval_level is set
    eval_level = loaded.get("eval_level")
    if eval_level is not None:
        es = state.env_state
        new_curriculum = jnp.full_like(es.curriculum_level, eval_level)
        state = state.replace(env_state=es.replace(curriculum_level=new_curriculum))

    # Init RNN hidden state
    hstate = ScannedRNN.initialize_carry(num_envs * num_actors, config["GRU_HIDDEN_DIM"])

    # Metrics accumulators
    all_rewards = []
    all_theta_deg = []
    all_delta_vt = []
    all_crashes = []
    all_on_target = []
    all_curriculum = []

    for step in range(num_steps):
        rng, act_rng = jax.random.split(rng)

        # Get observations
        obs_batch = jnp.stack([obsv[a] for a in env.agents])
        obs_batch = obs_batch.reshape((num_actors * num_envs, -1))
        dones = jnp.zeros((num_actors * num_envs,))

        # Forward pass
        ac_in = (obs_batch[np.newaxis, :, :], dones[np.newaxis, :])
        hstate, pi, value = network.apply(params, hstate, ac_in)

        # Greedy action selection (mode)
        actions = jnp.stack([p.mode() for p in pi], axis=-1).squeeze(0)

        # Reshape to per-agent dict
        actions_dict = {}
        for i, agent in enumerate(env.agents):
            agent_actions = actions[i * num_envs : (i + 1) * num_envs]
            actions_dict[agent] = agent_actions

        # Step — LogWrapper.step takes (key, state, actions), no env_params
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_envs)
        obsv, state, rewards, dones_dict, infos = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(step_rngs, state, actions_dict)

        # Re-override curriculum_level after auto-reset (reset sets it back to 0)
        if eval_level is not None:
            es = state.env_state
            new_cl = jnp.full_like(es.curriculum_level, eval_level)
            state = state.replace(env_state=es.replace(curriculum_level=new_cl))

        # Collect metrics from info
        agent0 = env.agents[0]
        reward_vals = rewards[agent0]
        all_rewards.append(float(jnp.mean(reward_vals)))

        # Extract theta_deg and delta_vt from env_state
        # LogWrapper wraps state: state.env_state is the real env state
        es = state.env_state
        ps = es.plane_state
        agent_idx = 0  # single-agent env

        # Current quaternion q_NB: shape (num_envs,)
        q_curr = jnp.stack([
            jnp.nan_to_num(ps.q0[:, agent_idx], nan=1.0),
            jnp.nan_to_num(ps.q1[:, agent_idx], nan=0.0),
            jnp.nan_to_num(ps.q2[:, agent_idx], nan=0.0),
            jnp.nan_to_num(ps.q3[:, agent_idx], nan=0.0),
        ], axis=-1)  # (num_envs, 4)

        # Target euler angles
        tgt_heading = es.target_heading[:, agent_idx]
        tgt_pitch = es.target_pitch[:, agent_idx]
        tgt_roll = es.target_roll[:, agent_idx]
        tgt_vt = es.target_vt[:, agent_idx]

        # Target quaternion: q_NB = conj(q_BN)
        q_tgt = _quat_conj(_quat_from_euler_bn(tgt_roll, tgt_pitch, tgt_heading))

        # Geodesic angle
        theta = _quat_geodesic_angle(q_curr, q_tgt)  # (num_envs,)
        theta_deg_batch = theta * 180.0 / jnp.pi

        # Speed error
        vt = jnp.nan_to_num(ps.vt[:, agent_idx], nan=0.0)
        delta_vt_batch = jnp.abs(vt - tgt_vt)

        all_theta_deg.append(float(jnp.mean(theta_deg_batch)))
        all_delta_vt.append(float(jnp.mean(delta_vt_batch)))

        # On-target: theta<10° AND delta_vt<25
        on_target = (theta_deg_batch < 10.0) & (delta_vt_batch < 25.0)
        all_on_target.append(float(jnp.mean(on_target.astype(jnp.float32))))

        # Check for crashes (done + not success)
        done_vals = dones_dict[agent0]
        all_crashes.append(float(jnp.mean(done_vals)))

        # Reset RNN state for done envs
        done_broadcast = jnp.repeat(done_vals, num_actors)
        hstate = jnp.where(done_broadcast[:, None], 0.0, hstate)

    # Aggregate
    mean_reward = float(np.mean(all_rewards))
    crash_rate = float(np.mean(all_crashes))

    return {
        "seed": seed,
        "mean_theta_deg": float(np.mean(all_theta_deg)) if all_theta_deg else None,
        "mean_delta_vt": float(np.mean(all_delta_vt)) if all_delta_vt else None,
        "mean_per_step_reward": mean_reward,
        "crash_rate": crash_rate,
        "on_target_rate": float(np.mean(all_on_target)) if all_on_target else None,
        "total_steps": num_steps,
    }


def evaluate_checkpoint(checkpoint_path: str, config: dict = None, eval_level: int = None) -> dict:
    """Full evaluation: run multiple episodes and aggregate.

    Args:
        eval_level: If set, lock curriculum at this level. If None, curriculum advances normally.
    """
    if config is None:
        config = dict(EVAL_CONFIG)

    level_str = f" (Level {eval_level})" if eval_level is not None else " (curriculum auto-advance)"
    print(f"Loading checkpoint: {checkpoint_path}{level_str}")
    t0 = time.time()
    loaded = load_checkpoint(checkpoint_path, config, eval_level=eval_level)
    print(f"Loaded in {time.time()-t0:.1f}s (epoch={loaded['epoch']})")

    all_results = []
    for seed in config.get("EVAL_SEEDS", EVAL_CONFIG["EVAL_SEEDS"]):
        print(f"  Running eval seed={seed}...")
        result = run_eval_episode(loaded, config, seed)
        all_results.append(result)
        print(f"    theta={result.get('mean_theta_deg', '?')}deg, "
              f"delta_vt={result.get('mean_delta_vt', '?')}, "
              f"reward={result['mean_per_step_reward']:.3f}, "
              f"crash_rate={result['crash_rate']:.4f}")

    # Aggregate across seeds
    def _safe_mean(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def _safe_std(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return float(np.std(vals)) if vals else None

    aggregate = {
        "mean_theta_deg": _safe_mean("mean_theta_deg"),
        "std_theta_deg": _safe_std("mean_theta_deg"),
        "mean_delta_vt": _safe_mean("mean_delta_vt"),
        "std_delta_vt": _safe_std("mean_delta_vt"),
        "mean_per_step_reward": _safe_mean("mean_per_step_reward"),
        "std_per_step_reward": _safe_std("mean_per_step_reward"),
        "mean_crash_rate": _safe_mean("crash_rate"),
        "std_crash_rate": _safe_std("crash_rate"),
        "mean_on_target_rate": _safe_mean("on_target_rate"),
        "std_on_target_rate": _safe_std("on_target_rate"),
    }

    return {
        "checkpoint_path": checkpoint_path,
        "epoch": loaded["epoch"],
        "eval_config": {k: v for k, v in config.items() if k in ["NUM_ENVS", "NUM_STEPS", "NUM_EPISODES"]},
        "per_seed_results": all_results,
        "aggregate": aggregate,
        "timestamp": datetime.now().isoformat(),
    }


def extract_training_metrics(log_lines: list) -> dict:
    """Extract key training metrics from stdout log lines.

    This is used as a lightweight alternative to full checkpoint evaluation:
    parse the training stdout for theta_deg, delta_vt, episodic_return, etc.

    Returns dict of final values for key metrics.
    """
    metrics = {
        "final_theta_deg": None,
        "final_delta_vt": None,
        "final_episodic_return": None,
        "final_curriculum_level": None,
        "final_on_target_steps": None,
        "min_theta_deg": float("inf"),
        "total_env_steps": 0,
    }

    import re
    for line in log_lines:
        # Parse theta_deg=XX.X patterns
        m = re.search(r"theta_deg[=:\s]+([\d.]+)", line)
        if m:
            val = float(m.group(1))
            metrics["final_theta_deg"] = val
            metrics["min_theta_deg"] = min(metrics["min_theta_deg"], val)

        m = re.search(r"delta_vt[=:\s]+([\d.]+)", line)
        if m:
            metrics["final_delta_vt"] = float(m.group(1))

        m = re.search(r"episodic_return[=:\s]+([-\d.]+)", line)
        if m:
            metrics["final_episodic_return"] = float(m.group(1))

        m = re.search(r"curriculum_level[=:\s]+([\d.]+)", line)
        if m:
            metrics["final_curriculum_level"] = float(m.group(1))

        m = re.search(r"on_target_steps[=:\s]+([\d.]+)", line)
        if m:
            metrics["final_on_target_steps"] = float(m.group(1))

        m = re.search(r"env_step[=:\s]+(\d+)", line)
        if m:
            metrics["total_env_steps"] = max(metrics.get("total_env_steps", 0), int(m.group(1)))

    if metrics["min_theta_deg"] == float("inf"):
        metrics["min_theta_deg"] = None

    return metrics


CURRICULUM_LABELS = {
    0: "Level 0: H±90° P±30° R±90°",
    1: "Level 1: H±120° P±45° R±120°",
    2: "Level 2: H±180° P±60° R±150°",
    3: "Level 3: H±180° P±75° R±180°",
    4: "Level 4: H±180° P±89° R±180°",
    5: "Level 5: H±180° P±89° R±180°",
}


def evaluate_checkpoint_per_level(checkpoint_path: str, config: dict = None,
                                   levels: list = None,
                                   champion_per_level: dict = None) -> dict:
    """Evaluate checkpoint separately at each curriculum level.

    Args:
        champion_per_level: If provided, early-exit when challenger loses a level.
            Dict mapping level (str or int) to {"mean_theta_deg": float, ...}.

    Returns per-level results + weighted aggregate.
    """
    if config is None:
        config = dict(EVAL_CONFIG)
    if levels is None:
        levels = [0, 1, 2, 3, 5]  # Skip level 4 (identical params to level 5)

    print(f"\n{'='*60}")
    print(f"PER-LEVEL EVALUATION: {checkpoint_path}")
    if champion_per_level:
        print(f"  (early-exit enabled: must beat champion at each level)")
    print(f"{'='*60}")

    per_level = {}
    early_exit = False
    for level in levels:
        print(f"\n--- {CURRICULUM_LABELS.get(level, f'Level {level}')} ---")
        result = evaluate_checkpoint(checkpoint_path, config, eval_level=level)
        per_level[level] = result["aggregate"]
        agg = result["aggregate"]
        print(f"  => theta={agg['mean_theta_deg']:.2f}° ± {agg['std_theta_deg']:.2f}°, "
              f"delta_vt={agg['mean_delta_vt']:.2f}, "
              f"crash={agg['mean_crash_rate']:.4f}, "
              f"on_target={agg['mean_on_target_rate']:.3f}")

        # Early-exit: if challenger is worse than champion at this level, stop
        if champion_per_level:
            champ_key = str(level)
            champ_level = champion_per_level.get(champ_key)
            if champ_level and agg['mean_theta_deg'] is not None:
                champ_theta = champ_level.get('mean_theta_deg', 999)
                if agg['mean_theta_deg'] > champ_theta:
                    print(f"  EARLY EXIT: theta {agg['mean_theta_deg']:.2f}° > champion {champ_theta:.2f}° at Level {level}")
                    early_exit = True
                    break
                else:
                    print(f"  PASS: theta {agg['mean_theta_deg']:.2f}° <= champion {champ_theta:.2f}°")

    # Weighted average across evaluated levels
    all_theta = [per_level[l]["mean_theta_deg"] for l in per_level if per_level[l]["mean_theta_deg"] is not None]
    all_dvt = [per_level[l]["mean_delta_vt"] for l in per_level if per_level[l]["mean_delta_vt"] is not None]
    all_crash = [per_level[l]["mean_crash_rate"] for l in per_level if per_level[l]["mean_crash_rate"] is not None]
    all_on_target = [per_level[l]["mean_on_target_rate"] for l in per_level if per_level[l]["mean_on_target_rate"] is not None]

    overall = {
        "mean_theta_deg": float(np.mean(all_theta)) if all_theta else None,
        "mean_delta_vt": float(np.mean(all_dvt)) if all_dvt else None,
        "mean_crash_rate": float(np.mean(all_crash)) if all_crash else None,
        "mean_on_target_rate": float(np.mean(all_on_target)) if all_on_target else None,
        "early_exit": early_exit,
    }

    print(f"\n{'='*60}")
    if early_exit:
        print(f"EARLY EXIT after {len(per_level)} of {len(levels)} levels — DISCARD")
    else:
        print(f"OVERALL (avg across {len(per_level)} levels):")
        print(f"  theta={overall['mean_theta_deg']:.2f}°, delta_vt={overall['mean_delta_vt']:.2f}, "
              f"crash={overall['mean_crash_rate']:.4f}, on_target={overall['mean_on_target_rate']:.3f}")
    print(f"{'='*60}")

    return {
        "checkpoint_path": checkpoint_path,
        "per_level": {str(k): v for k, v in per_level.items()},
        "overall": overall,
        "levels_evaluated": levels,
        "timestamp": datetime.now().isoformat(),
    }


def evaluate_random_policy(config: dict = None) -> dict:
    """Run evaluation with randomly initialized network (no checkpoint).
    This establishes the "random policy" baseline that any trained agent must beat.
    """
    if config is None:
        config = dict(EVAL_CONFIG)

    print("Evaluating random policy baseline...")
    env_params = FullDomain_TaskParams()
    env = AeroPlanaxFullDomainEnv(env_params)
    env = LogWrapper(env)
    num_actors = env.num_agents

    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    rng = jax.random.PRNGKey(0)
    obs_shape = env.observation_space(env.agents[0], env_params).shape

    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * num_actors, *obs_shape)),
        jnp.zeros((1, config["NUM_ENVS"] * num_actors)),
    )
    init_hstate = ScannedRNN.initialize_carry(
        config["NUM_ENVS"] * num_actors, config["GRU_HIDDEN_DIM"]
    )
    network_params = network.init(rng, init_hstate, init_x)

    loaded = {
        "params": network_params,
        "epoch": 0,
        "network": network,
        "env": env,
        "env_params": env_params,
        "num_actors": num_actors,
    }

    all_results = []
    for seed in config.get("EVAL_SEEDS", EVAL_CONFIG["EVAL_SEEDS"]):
        print(f"  Running random policy eval seed={seed}...")
        result = run_eval_episode(loaded, config, seed)
        all_results.append(result)
        print(f"    theta={result.get('mean_theta_deg', '?')}deg, "
              f"crash_rate={result['crash_rate']:.4f}")

    # Aggregate (same logic as evaluate_checkpoint)
    def _safe_mean(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def _safe_std(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return float(np.std(vals)) if vals else None

    aggregate = {
        "mean_theta_deg": _safe_mean("mean_theta_deg"),
        "std_theta_deg": _safe_std("mean_theta_deg"),
        "mean_delta_vt": _safe_mean("mean_delta_vt"),
        "std_delta_vt": _safe_std("mean_delta_vt"),
        "mean_per_step_reward": _safe_mean("mean_per_step_reward"),
        "std_per_step_reward": _safe_std("mean_per_step_reward"),
        "mean_crash_rate": _safe_mean("crash_rate"),
        "std_crash_rate": _safe_std("crash_rate"),
        "mean_on_target_rate": _safe_mean("on_target_rate"),
        "std_on_target_rate": _safe_std("on_target_rate"),
    }

    return {
        "checkpoint_path": None,
        "epoch": 0,
        "eval_config": {k: v for k, v in config.items() if k in ["NUM_ENVS", "NUM_STEPS", "NUM_EPISODES"]},
        "per_seed_results": all_results,
        "aggregate": aggregate,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate full-domain maneuver checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint directory")
    parser.add_argument("--random-baseline", action="store_true", help="Evaluate random policy")
    parser.add_argument("--per-level", action="store_true", help="Evaluate each curriculum level separately")
    parser.add_argument("--level", type=int, default=None, help="Evaluate at a specific curriculum level (0-5)")
    parser.add_argument("--levels", type=str, default="0,1,2,3,4,5", help="Comma-separated levels for --per-level")
    parser.add_argument("--num-envs", type=int, default=EVAL_CONFIG["NUM_ENVS"])
    parser.add_argument("--num-steps", type=int, default=EVAL_CONFIG["NUM_STEPS"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    config = dict(EVAL_CONFIG)
    config["NUM_ENVS"] = args.num_envs
    config["NUM_STEPS"] = args.num_steps

    if args.random_baseline:
        results = evaluate_random_policy(config)
    elif args.checkpoint:
        if args.per_level:
            levels = [int(l) for l in args.levels.split(",")]
            results = evaluate_checkpoint_per_level(args.checkpoint, config, levels=levels)
        else:
            results = evaluate_checkpoint(args.checkpoint, config, eval_level=args.level)
    else:
        parser.error("Either --checkpoint or --random-baseline is required")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))
