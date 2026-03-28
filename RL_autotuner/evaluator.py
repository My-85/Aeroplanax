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
    # Reward component diagnostics
    all_att_r = []
    all_speed_r = []
    # Smoothness diagnostics (angular acceleration)
    all_angular_accel = []
    prev_P, prev_Q, prev_R = None, None, None

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

        # Angular rates for smoothness diagnostic (rad/s)
        P_batch = jnp.nan_to_num(ps.P[:, agent_idx], nan=0.0)
        Q_batch = jnp.nan_to_num(ps.Q[:, agent_idx], nan=0.0)
        R_batch = jnp.nan_to_num(ps.R[:, agent_idx], nan=0.0)
        if prev_P is not None:
            dt = 0.2  # 1 decision step = 0.2s
            dP = (P_batch - prev_P) / dt
            dQ = (Q_batch - prev_Q) / dt
            dR = (R_batch - prev_R) / dt
            ang_accel = jnp.sqrt(dP**2 + dQ**2 + dR**2)
            all_angular_accel.append(float(jnp.mean(ang_accel)))
        prev_P, prev_Q, prev_R = P_batch, Q_batch, R_batch

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

        # Reward component diagnostics (post-hoc, same math as reward fn)
        try:
            from envs.reward_functions.quat_baseline_reward import REWARD_CONFIG as _rc
        except Exception:
            _rc = {"theta_scale_deg_low": 30.0, "theta_scale_deg_mid": 45.0,
                   "theta_scale_deg_high": 70.0, "speed_error_scale": 40.0,
                   "w_att": 0.7, "w_speed": 0.3, "att_exponent": 4.0}
        cl = es.curriculum_level[:, agent_idx]
        _ts_low = _rc.get("theta_scale_deg_low", _rc.get("theta_scale_deg", 30.0))
        _ts_mid = _rc.get("theta_scale_deg_mid", _ts_low)
        _ts_high = _rc.get("theta_scale_deg_high", _ts_mid)
        theta_scale_deg = jnp.where(cl <= 1, _ts_low,
                          jnp.where(cl <= 3, _ts_mid, _ts_high))
        theta_scale_rad = jnp.deg2rad(theta_scale_deg)
        _exp = _rc.get("att_exponent", 2.0)
        att_r_batch = jnp.exp(-((theta / theta_scale_rad) ** _exp))
        speed_r_batch = jnp.exp(-((delta_vt_batch / _rc.get("speed_error_scale", 40.0)) ** 2))
        all_att_r.append(float(jnp.mean(att_r_batch)))
        all_speed_r.append(float(jnp.mean(speed_r_batch)))

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
        # Reward component diagnostics
        "diag_att_r_mean": float(np.mean(all_att_r)) if all_att_r else None,
        "diag_att_r_std": float(np.std(all_att_r)) if all_att_r else None,
        "diag_speed_r_mean": float(np.mean(all_speed_r)) if all_speed_r else None,
        "diag_speed_r_std": float(np.std(all_speed_r)) if all_speed_r else None,
        # Smoothness diagnostics
        "diag_angular_accel_mean": float(np.mean(all_angular_accel)) if all_angular_accel else None,
        "diag_angular_accel_std": float(np.std(all_angular_accel)) if all_angular_accel else None,
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
        # Reward component diagnostics
        "diag_att_r_mean": _safe_mean("diag_att_r_mean"),
        "diag_att_r_std": _safe_mean("diag_att_r_std"),
        "diag_speed_r_mean": _safe_mean("diag_speed_r_mean"),
        "diag_speed_r_std": _safe_mean("diag_speed_r_std"),
        # Smoothness diagnostics
        "diag_angular_accel_mean": _safe_mean("diag_angular_accel_mean"),
        "diag_angular_accel_std": _safe_mean("diag_angular_accel_std"),
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


def compute_composite_score(metrics: dict) -> float:
    """Composite score: lower = better.

    Uses waypoint evaluation metrics:
    - mean_ss_theta (steady-state attitude error, last 25% of each waypoint)
    - mean_ss_dvt (steady-state speed error)
    - settled_rate (fraction of waypoints where theta < 8° was achieved)

    Weights: attitude=0.5, speed=0.3, settling=0.2
    """
    # Support both waypoint-style and legacy keys
    ss_theta = metrics.get("mean_ss_theta") or metrics.get("mean_theta_deg", 90.0)
    ss_dvt = metrics.get("mean_ss_dvt") or metrics.get("mean_delta_vt", 100.0)
    settled_rate = metrics.get("settled_rate", 0.0)

    norm_theta = min(ss_theta / 90.0, 1.0)
    norm_dvt = min(ss_dvt / 100.0, 1.0)
    settle_penalty = 1.0 - settled_rate  # 0 if all settled, 1 if none

    return 0.5 * norm_theta + 0.3 * norm_dvt + 0.2 * settle_penalty


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

        # Early-exit: composite score comparison with 10% tolerance
        if champion_per_level:
            champ_key = str(level)
            champ_level = champion_per_level.get(champ_key)
            if champ_level and agg['mean_theta_deg'] is not None:
                champ_score = compute_composite_score(champ_level)
                new_score = compute_composite_score(agg)
                tolerance = champ_score * 0.10
                if new_score > champ_score + tolerance:
                    print(f"  EARLY EXIT: composite {new_score:.4f} > champion {champ_score:.4f} + tol {tolerance:.4f} at Level {level}")
                    early_exit = True
                    break
                else:
                    print(f"  PASS: composite {new_score:.4f} <= champion {champ_score:.4f} (+10% = {champ_score + tolerance:.4f})")

    # Weighted average across evaluated levels
    all_theta = [per_level[l]["mean_theta_deg"] for l in per_level if per_level[l]["mean_theta_deg"] is not None]
    all_dvt = [per_level[l]["mean_delta_vt"] for l in per_level if per_level[l]["mean_delta_vt"] is not None]
    all_crash = [per_level[l]["mean_crash_rate"] for l in per_level if per_level[l]["mean_crash_rate"] is not None]
    all_on_target = [per_level[l]["mean_on_target_rate"] for l in per_level if per_level[l]["mean_on_target_rate"] is not None]
    all_ang_accel = [per_level[l].get("diag_angular_accel_mean") for l in per_level if per_level[l].get("diag_angular_accel_mean") is not None]

    overall = {
        "mean_theta_deg": float(np.mean(all_theta)) if all_theta else None,
        "mean_delta_vt": float(np.mean(all_dvt)) if all_dvt else None,
        "mean_crash_rate": float(np.mean(all_crash)) if all_crash else None,
        "mean_on_target_rate": float(np.mean(all_on_target)) if all_on_target else None,
        "diag_angular_accel_mean": float(np.mean(all_ang_accel)) if all_ang_accel else None,
        "early_exit": early_exit,
    }
    overall["composite_score"] = compute_composite_score(overall)

    print(f"\n{'='*60}")
    if early_exit:
        print(f"EARLY EXIT after {len(per_level)} of {len(levels)} levels — DISCARD")
    else:
        print(f"OVERALL (avg across {len(per_level)} levels):")
        print(f"  theta={overall['mean_theta_deg']:.2f}°, delta_vt={overall['mean_delta_vt']:.2f}, "
              f"crash={overall['mean_crash_rate']:.4f}, on_target={overall['mean_on_target_rate']:.3f}")
        print(f"  composite_score={overall['composite_score']:.4f} (lower=better)")
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
        # Reward component diagnostics
        "diag_att_r_mean": _safe_mean("diag_att_r_mean"),
        "diag_att_r_std": _safe_mean("diag_att_r_std"),
        "diag_speed_r_mean": _safe_mean("diag_speed_r_mean"),
        "diag_speed_r_std": _safe_mean("diag_speed_r_std"),
        # Smoothness diagnostics
        "diag_angular_accel_mean": _safe_mean("diag_angular_accel_mean"),
        "diag_angular_accel_std": _safe_mean("diag_angular_accel_std"),
    }

    return {
        "checkpoint_path": None,
        "epoch": 0,
        "eval_config": {k: v for k, v in config.items() if k in ["NUM_ENVS", "NUM_STEPS", "NUM_EPISODES"]},
        "per_seed_results": all_results,
        "aggregate": aggregate,
        "timestamp": datetime.now().isoformat(),
    }


# ======================== Waypoint-based Evaluation ========================

WAYPOINTS = [
    # (name, heading_deg, pitch_deg, roll_deg, target_speed_mps)
    # 20 waypoints, each evaluated independently from level flight.
    # --- L0: easy (H±90° P±30° R±90°) ---
    ("H-15",        -15,   0,   0, 200),   # L0: heading easy
    ("P-8",           0,  -8,   0, 200),   # L0: pitch easy
    # --- L1: moderate (H±120° P±45° R±120°) ---
    ("H-30_P-10",   -30, -10,   0, 190),   # L1: H+P moderate
    ("R-45",          0,   0, -45, 200),   # L1: roll moderate
    # --- L2: hard (H±180° P±60° R±150°) ---
    ("H-60_P-20",   -60, -20,   0, 180),   # L2: H+P hard
    ("R+90",          0,   0,  90, 200),   # L2: roll hard
    ("combo_L2",     45,  15,  30, 215),   # L2: 3-axis moderate
    ("P+60",          0,  60,   0, 200),   # L2: pitch up extreme
    ("P-60",          0, -60,   0, 200),   # L2: pitch down extreme
    # --- L3: very hard (H±180° P±75° R±180°) ---
    ("R+135",         0,   0, 135, 200),   # L3: roll vhard
    ("combo_L3",     60,  25,  60, 220),   # L3: 3-axis hard
    ("R+180",         0,   0, 180, 200),   # L3: full roll inversion
    ("H+180",       180,   0,   0, 200),   # L3: heading reversal
    # --- L4/L5: extreme (H±180° P±89° R±180°) ---
    ("H-90_P-30",   -90, -30,   0, 175),   # L4: H+P extreme
    ("combo_L4",     90,  40,  90, 240),   # L4: 3-axis vhard
    ("full",        -120, -45, -90, 170),  # L4: all-axis extreme
    ("P+89",          0,  89,   0, 200),   # L5: near-vertical pitch up
    ("P-89",          0, -89,   0, 200),   # L5: near-vertical pitch down
    ("H-180_P-60",  -180, -60,   0, 170),  # L5: heading reversal + steep dive
    ("combo_L5",    150,  60, 150, 180),   # L5: all-axis maximum
]

STEPS_PER_WP = 250
SETTLE_THRESH_DEG = 8.0
WP_SIM_DT = 0.2  # 1 decision step = 0.2s

# Level flight initial state for reset between waypoints
LEVEL_VT = 150.0  # m/s — level flight reset speed (training range lower bound [150, 300])
LEVEL_ALT = 5000.0  # m — safe altitude

LEVEL_GROUPS = {
    "L0_easy":     (0,  2),   # WP 0-1:   H-15, P-8
    "L1_moderate": (2,  4),   # WP 2-3:   H-30_P-10, R-45
    "L2_hard":     (4,  9),   # WP 4-8:   H-60_P-20, R+90, combo_L2, P+60, P-60
    "L3_vhard":    (9,  13),  # WP 9-12:  R+135, combo_L3, R+180, H+180
    "L4_extreme":  (13, 16),  # WP 13-15: H-90_P-30, combo_L4, full
    "L5_max":      (16, 20),  # WP 16-19: P+89, P-89, H-180_P-60, combo_L5
}


def _reset_to_level_flight(state, num_envs, num_actors):
    """Force level flight state. Obs will be stale for 1 step but step() recomputes it."""
    ps = state.env_state.plane_state
    zeros = jnp.zeros_like(ps.roll)
    ones = jnp.ones_like(ps.q0)
    level_vt = jnp.full_like(ps.vt, LEVEL_VT)
    level_alt = jnp.full_like(ps.altitude, LEVEL_ALT)

    new_ps = ps.replace(
        roll=zeros, pitch=zeros, yaw=zeros,
        vt=level_vt, vel_y=level_vt,
        altitude=level_alt,
        q0=ones, q1=zeros, q2=zeros, q3=zeros,
    )
    return state.replace(env_state=state.env_state.replace(plane_state=new_ps))


def run_waypoint_eval(loaded: dict, config: dict) -> dict:
    """Run fixed waypoint evaluation. Returns per-waypoint and overall metrics.

    Each waypoint is evaluated independently from level flight (roll=pitch=yaw=0).
    This ensures no cross-contamination between waypoints and tests the agent's
    ability to track targets from a known stable baseline.
    """
    network = loaded["network"]
    params = loaded["params"]
    env = loaded["env"]
    num_envs = config["NUM_ENVS"]
    num_actors = loaded["num_actors"]

    rng = jax.random.PRNGKey(42)
    per_waypoint = []

    # Helper: set targets on state
    def _set_targets(st, h_rad, p_rad, r_rad, v_mps):
        es = st.env_state
        return st.replace(env_state=es.replace(
            target_heading=jnp.broadcast_to(h_rad, es.target_heading.shape),
            target_pitch=jnp.broadcast_to(p_rad, es.target_pitch.shape),
            target_roll=jnp.broadcast_to(r_rad, es.target_roll.shape),
            target_vt=jnp.broadcast_to(v_mps, es.target_vt.shape),
        ))

    for wp_idx, (wp_name, h_deg, p_deg, r_deg, vt_mps) in enumerate(WAYPOINTS):
        # Fresh reset → set WP target
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        obsv, state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
        # Force level flight (obs will be stale but step() recomputes it)
        state = _reset_to_level_flight(state, num_envs, num_actors)

        # Fresh RNN hidden state (no carry-over between WPs)
        hstate = ScannedRNN.initialize_carry(num_envs * num_actors, config["GRU_HIDDEN_DIM"])

        # Set WP target (also set as level flight first for sync steps)
        tgt_h = jnp.array(np.radians(h_deg), dtype=jnp.float32)
        tgt_p = jnp.array(np.radians(p_deg), dtype=jnp.float32)
        tgt_r = jnp.array(np.radians(r_deg), dtype=jnp.float32)
        tgt_v = jnp.array(float(vt_mps), dtype=jnp.float32)

        # First: set target to level flight and run sync steps to align obs with overridden state
        level_h = jnp.array(0.0, dtype=jnp.float32)
        level_p = jnp.array(0.0, dtype=jnp.float32)
        level_r = jnp.array(0.0, dtype=jnp.float32)
        level_v = jnp.array(LEVEL_VT, dtype=jnp.float32)
        state = _set_targets(state, level_h, level_p, level_r, level_v)

        SYNC_STEPS = 25  # 5s to let obs sync and agent settle at level flight
        for _ in range(SYNC_STEPS):
            obs_batch = jnp.stack([obsv[a] for a in env.agents]).reshape((num_actors * num_envs, -1))
            dones_in = jnp.zeros((num_actors * num_envs,))
            ac_in = (obs_batch[np.newaxis, :, :], dones_in[np.newaxis, :])
            hstate, pi, _ = network.apply(params, hstate, ac_in)
            actions = jnp.stack([p.mode() for p in pi], axis=-1).squeeze(0)
            actions_dict = {agent: actions[i * num_envs : (i + 1) * num_envs]
                           for i, agent in enumerate(env.agents)}
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_envs)
            obsv, state, _, _, _ = jax.vmap(env.step, in_axes=(0, 0, 0))(step_rngs, state, actions_dict)
            state = _set_targets(state, level_h, level_p, level_r, level_v)

        # Now switch to actual WP target for measurement
        state = _set_targets(state, tgt_h, tgt_p, tgt_r, tgt_v)
        tgt_v = jnp.array(float(vt_mps), dtype=jnp.float32)

        state = _set_targets(state, tgt_h, tgt_p, tgt_r, tgt_v)

        settle_step = None
        step_thetas = []
        step_dvts = []
        step_drolls = []
        step_dpitchs = []
        step_dyaws = []

        for step_in_wp in range(STEPS_PER_WP):
            obs_batch = jnp.stack([obsv[a] for a in env.agents]).reshape((num_actors * num_envs, -1))
            dones_in = jnp.zeros((num_actors * num_envs,))

            ac_in = (obs_batch[np.newaxis, :, :], dones_in[np.newaxis, :])
            hstate, pi, _ = network.apply(params, hstate, ac_in)
            actions = jnp.stack([p.mode() for p in pi], axis=-1).squeeze(0)

            actions_dict = {agent: actions[i * num_envs : (i + 1) * num_envs]
                           for i, agent in enumerate(env.agents)}

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_envs)
            obsv, state, rewards, dones_dict, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_rngs, state, actions_dict)

            # Skip this frame if env auto-reset (done=True causes state corruption)
            if dones_dict[env.agents[0]][0]:
                continue

            # Re-override targets after env auto-reset
            state = _set_targets(state, tgt_h, tgt_p, tgt_r, tgt_v)

            # Extract metrics
            ps = state.env_state.plane_state
            q_curr = jnp.stack([
                jnp.nan_to_num(ps.q0[0, 0], nan=1.0),
                jnp.nan_to_num(ps.q1[0, 0], nan=0.0),
                jnp.nan_to_num(ps.q2[0, 0], nan=0.0),
                jnp.nan_to_num(ps.q3[0, 0], nan=0.0),
            ])
            q_tgt = _quat_conj(_quat_from_euler_bn(tgt_r, tgt_p, tgt_h))
            theta_rad = float(_quat_geodesic_angle(q_curr, q_tgt))
            theta_deg = theta_rad * 180.0 / np.pi

            vt_now = float(jnp.nan_to_num(ps.vt[0, 0], nan=0.0))
            delta_vt = abs(vt_now - float(tgt_v))

            roll_now = float(ps.roll[0, 0]) * 180.0 / np.pi
            pitch_now = float(ps.pitch[0, 0]) * 180.0 / np.pi
            yaw_now = float(ps.yaw[0, 0]) * 180.0 / np.pi
            tgt_roll_deg = float(tgt_r) * 180.0 / np.pi
            tgt_pitch_deg = float(tgt_p) * 180.0 / np.pi
            tgt_yaw_deg = float(tgt_h) * 180.0 / np.pi

            step_thetas.append(theta_deg)
            step_dvts.append(delta_vt)
            step_drolls.append(abs(roll_now - tgt_roll_deg))
            step_dpitchs.append(abs(pitch_now - tgt_pitch_deg))
            step_dyaws.append(abs(yaw_now - tgt_yaw_deg))

            if settle_step is None and theta_deg < SETTLE_THRESH_DEG:
                settle_step = step_in_wp

            # Reset RNN state for done envs
            done_vals = dones_dict[env.agents[0]]
            done_bc = jnp.repeat(done_vals, num_actors)
            hstate = jnp.where(done_bc[:, None], 0.0, hstate)

        # Compute per-waypoint summary (from last unsettled point to end)
        # Find last step where theta >= SETTLE_THRESH_DEG (reverse search)
        last_unsettled = None
        for i in range(len(step_thetas)-1, -1, -1):
            if step_thetas[i] >= SETTLE_THRESH_DEG:
                last_unsettled = i
                break

        if last_unsettled is not None:
            tail_start = last_unsettled + 1
        else:
            tail_start = 0  # All steps settled

        # Fallback: if tail is empty, use last 25%
        if tail_start >= len(step_thetas):
            tail_start = int(STEPS_PER_WP * 0.75)

        tail_thetas = step_thetas[tail_start:]
        tail_dvts = step_dvts[tail_start:]
        tail_drolls = step_drolls[tail_start:]
        tail_dpitchs = step_dpitchs[tail_start:]
        tail_dyaws = step_dyaws[tail_start:]

        settled = settle_step is not None  # momentary entry (for settle_time_s record)
        settle_time_s = settle_step * WP_SIM_DT if settled else STEPS_PER_WP * WP_SIM_DT
        ss_theta = float(np.mean(tail_thetas))
        ss_dvt = float(np.mean(tail_dvts))

        # CRITICAL: settled status for composite_score must use steady-state error, not momentary entry
        settled = ss_theta < SETTLE_THRESH_DEG

        wp_result = {
            "name": wp_name,
            "wp_idx": wp_idx,
            "settled": settled,
            "settle_time_s": settle_time_s,
            "ss_theta_deg": ss_theta,
            "ss_dvt": ss_dvt,
            "ss_droll": float(np.mean(tail_drolls)),
            "ss_dpitch": float(np.mean(tail_dpitchs)),
            "ss_dyaw": float(np.mean(tail_dyaws)),
            "mean_theta_deg": float(np.mean(step_thetas)),
        }
        per_waypoint.append(wp_result)

        status = f"settled {settle_time_s:.1f}s" if settled else "NOT settled"
        print(f"  WP{wp_idx:02d} {wp_name:<20s} | ss_theta={ss_theta:6.1f}° | ss_dvt={ss_dvt:6.1f} | {status}")

    # Overall summary
    settled_count = sum(1 for wp in per_waypoint if wp["settled"])
    settled_rate = settled_count / len(WAYPOINTS)
    mean_ss_theta = float(np.mean([wp["ss_theta_deg"] for wp in per_waypoint]))
    mean_ss_dvt = float(np.mean([wp["ss_dvt"] for wp in per_waypoint]))

    # Level-grouped summary
    level_grouped = {}
    for group_name, (start, end) in LEVEL_GROUPS.items():
        group_wps = per_waypoint[start:end]
        level_grouped[group_name] = {
            "mean_ss_theta": float(np.mean([wp["ss_theta_deg"] for wp in group_wps])),
            "mean_ss_dvt": float(np.mean([wp["ss_dvt"] for wp in group_wps])),
            "settled_count": sum(1 for wp in group_wps if wp["settled"]),
        }

    overall = {
        "mean_ss_theta": mean_ss_theta,
        "mean_ss_dvt": mean_ss_dvt,
        "settled_count": settled_count,
        "settled_rate": settled_rate,
        "mean_theta_deg": mean_ss_theta,  # alias for compatibility
        "mean_delta_vt": mean_ss_dvt,     # alias for compatibility
    }
    overall["composite_score"] = compute_composite_score(overall)

    print(f"\n  OVERALL: ss_theta={mean_ss_theta:.2f}°, ss_dvt={mean_ss_dvt:.2f}, "
          f"settled={settled_count}/{len(WAYPOINTS)}, composite={overall['composite_score']:.4f}")

    return {
        "per_waypoint": per_waypoint,
        "overall": overall,
        "level_grouped": level_grouped,
    }


def evaluate_checkpoint_waypoint(checkpoint_path: str, config: dict = None,
                                  champion_metrics: dict = None) -> dict:
    """Waypoint-based evaluation with optional early-exit against champion.

    Args:
        champion_metrics: If provided, compare composite_score for early-exit.
    """
    if config is None:
        config = dict(EVAL_CONFIG)

    print(f"\n{'='*60}")
    print(f"WAYPOINT EVALUATION: {checkpoint_path}")
    print(f"{'='*60}")

    loaded = load_checkpoint(checkpoint_path, config)
    result = run_waypoint_eval(loaded, config)

    early_exit = False
    if champion_metrics:
        champ_score = champion_metrics.get("composite_score", 999)
        new_score = result["overall"]["composite_score"]
        tolerance = champ_score * 0.10
        if new_score > champ_score + tolerance:
            print(f"  EARLY EXIT: composite {new_score:.4f} > champion {champ_score:.4f} + tol {tolerance:.4f}")
            early_exit = True
        else:
            print(f"  PASS: composite {new_score:.4f} <= champion {champ_score:.4f} (+10% = {champ_score + tolerance:.4f})")

    result["overall"]["early_exit"] = early_exit
    result["checkpoint_path"] = checkpoint_path
    result["timestamp"] = datetime.now().isoformat()

    print(f"{'='*60}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate full-domain maneuver checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint directory")
    parser.add_argument("--random-baseline", action="store_true", help="Evaluate random policy")
    parser.add_argument("--per-level", action="store_true", help="Evaluate each curriculum level separately")
    parser.add_argument("--waypoint", action="store_true", help="Run fixed waypoint evaluation (12 waypoints from level flight, control quality)")
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
        if args.waypoint:
            results = evaluate_checkpoint_waypoint(args.checkpoint, config)
        elif args.per_level:
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


def compute_target_marker_position(plane_north, plane_east, plane_alt, 
                                   target_heading, target_pitch, distance=2000.0):
    """Compute target marker position in NED frame.
    
    Args:
        plane_north, plane_east, plane_alt: Current plane position
        target_heading: Target yaw in radians
        target_pitch: Target pitch in radians
        distance: Distance ahead in meters (default 2000m)
    
    Returns:
        (target_north, target_east, target_alt) in NED frame
    """
    # Forward direction vector in NED frame
    # North = distance * cos(pitch) * cos(heading)
    # East = distance * cos(pitch) * sin(heading)
    # Down = -distance * sin(pitch)  (negative because NED uses down as positive)
    dn = distance * np.cos(target_pitch) * np.cos(target_heading)
    de = distance * np.cos(target_pitch) * np.sin(target_heading)
    du = distance * np.sin(target_pitch)
    
    return (
        float(plane_north + dn),
        float(plane_east + de),
        float(plane_alt + du)
    )
