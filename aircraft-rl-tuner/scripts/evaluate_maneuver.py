#!/usr/bin/env python3
"""
evaluate_maneuver.py
====================
Load a JAX policy checkpoint trained with train_full_domain_maneuver_v3.py,
run parallel evaluation episodes on 2× A100 GPUs, and output a structured
JSON report measuring full-domain maneuver capability.

Usage:
    python evaluate_maneuver.py --checkpoint /path/to/checkpoint --episodes 200
    python evaluate_maneuver.py --checkpoint /path/to/checkpoint --out report.json

Environment: Single-node, 2× NVIDIA A100 80 GB (CUDA_VISIBLE_DEVICES=0,1).
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_MEM_FRACTION", "0.90")

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

# ---------------------------------------------------------------------------
# Add Planax to path so we can import the environment
# ---------------------------------------------------------------------------
PLANAX_DIR = str(Path(__file__).resolve().parents[2] / "Planax")
if PLANAX_DIR not in sys.path:
    sys.path.insert(0, PLANAX_DIR)

from envs.wrappers import LogWrapper
from envs.aeroplanax_full_domain_maneuver import (
    AeroPlanaxFullDomainEnv,
    FullDomain_TaskParams,
    _quat_from_euler_bn,
    _quat_conj,
    _quat_normalize,
)

# ======================== Model definition (mirrors train_v3) ========================

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0, out_axes=0,
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
        activation = nn.relu if self.config.get("ACTIVATION", "relu") == "relu" else nn.tanh
        obs, dones = x

        embedding = nn.Dense(self.config["FC_DIM_SIZE"],
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"],
                              kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)

        logit_thr = nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        logit_ele = nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        logit_ail = nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        logit_rud = nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi_thr = distrax.Categorical(logits=logit_thr)
        pi_ele = distrax.Categorical(logits=logit_ele)
        pi_ail = distrax.Categorical(logits=logit_ail)
        pi_rud = distrax.Categorical(logits=logit_rud)

        critic = nn.Dense(self.config["FC_DIM_SIZE"],
                          kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, (pi_thr, pi_ele, pi_ail, pi_rud), jnp.squeeze(critic, axis=-1)


# ======================== Helpers ========================

def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _f(x, i=0):
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    return float(a.reshape(-1)[min(i, a.size - 1)])


# ======================== Predefined test waypoints ========================
# 5 difficulty levels × 2-3 waypoints each — covers full attitude domain
# Format: (heading_deg, pitch_deg, roll_deg, vt_m_s)
TEST_WAYPOINTS = [
    # Level 1: gentle turns
    (30.0,    0.0,    0.0,  200.0),
    (-30.0,   0.0,    0.0,  200.0),
    # Level 2: moderate maneuvers
    (90.0,   10.0,    0.0,  220.0),
    (-90.0, -10.0,    0.0,  220.0),
    # Level 3: large turns + bank
    (180.0,   0.0,   30.0,  250.0),
    (0.0,    20.0,  -45.0,  250.0),
    # Level 4: aggressive pitch + roll
    (45.0,   45.0,   60.0,  280.0),
    (-45.0, -30.0,  -60.0,  200.0),
    # Level 5: extreme / inverted
    (0.0,    70.0,   90.0,  300.0),
    (180.0, -50.0,  120.0,  250.0),
    (90.0,    0.0,  180.0,  250.0),
]


# ======================== Single-episode evaluation ========================

def evaluate_single_episode(
    network,
    params,
    env,
    env_params,
    config,
    rng,
    waypoints_rad,
    max_steps=2000,
):
    """Run one episode through predefined waypoints, return per-waypoint metrics."""
    num_wps = len(waypoints_rad)
    num_actors = config["NUM_ACTORS"]

    # Reset
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

    # Set first waypoint
    wp_idx = 0
    hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
    core_state = env_state.env_state
    core_state = core_state.replace(
        target_heading=jnp.full_like(core_state.target_heading, hdg_t),
        target_pitch=jnp.full_like(core_state.target_pitch, pit_t),
        target_roll=jnp.full_like(core_state.target_roll, rol_t),
        target_vt=jnp.full_like(core_state.target_vt, vt_t),
    )
    env_state = env_state.replace(env_state=core_state)

    hstate = ScannedRNN.initialize_carry(num_actors * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], num_actors)
    done_arr = jnp.zeros((config["NUM_ENVS"] * num_actors), dtype=bool)

    wp_hold_steps = 0
    min_hold = 30
    success_theta_deg = 10.0
    success_vt_tol = 15.0
    total_reward = 0.0
    crash_count = 0
    stall_steps = 0
    wp_results = []

    for step in range(max_steps):
        ac_in = (obs_batch[np.newaxis, :], done_arr[np.newaxis, :])
        hstate, pi, value = network.apply(params, hstate, ac_in)
        pi_thr, pi_ele, pi_ail, pi_rud = pi

        # Greedy (mode) for evaluation
        a_thr = pi_thr.mode()
        a_ele = pi_ele.mode()
        a_ail = pi_ail.mode()
        a_rud = pi_rud.mode()

        action = jnp.concatenate([
            a_thr[:, :, np.newaxis], a_ele[:, :, np.newaxis],
            a_ail[:, :, np.newaxis], a_rud[:, :, np.newaxis],
        ], axis=-1).squeeze(0)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
            rng_step, env_state,
            unbatchify(action, env.agents, config["NUM_ENVS"], num_actors)
        )

        core_state = env_state.env_state
        reward_sum = float(batchify(reward, env.agents, config["NUM_ENVS"], num_actors).reshape(-1).sum())
        total_reward += reward_sum

        # Compute error to current waypoint
        hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
        q_curr = np.array([
            _f(core_state.plane_state.q0), _f(core_state.plane_state.q1),
            _f(core_state.plane_state.q2), _f(core_state.plane_state.q3),
        ])
        q_norm = np.linalg.norm(q_curr)
        if q_norm < 1e-9:
            theta_err = 180.0
        else:
            q_curr_n = q_curr / q_norm
            q_tgt = np.array(_quat_conj(_quat_from_euler_bn(
                jnp.float32(rol_t), jnp.float32(pit_t), jnp.float32(hdg_t)
            ))).reshape(-1)
            cos_half = abs(np.dot(q_curr_n, q_tgt))
            cos_half = np.clip(cos_half, 0.0, 1.0)
            theta_err = float(2.0 * np.degrees(np.arccos(cos_half)))

        vt_now = _f(core_state.plane_state.vt)
        alt_now = _f(core_state.plane_state.altitude)
        delta_vt = abs(vt_now - vt_t)

        # Track stall (very low speed)
        if vt_now < 90.0:
            stall_steps += 1

        # Check on-target
        if theta_err <= success_theta_deg and delta_vt <= success_vt_tol:
            wp_hold_steps += 1
        else:
            wp_hold_steps = 0

        if wp_hold_steps >= min_hold:
            wp_results.append({
                "wp_index": wp_idx,
                "heading_deg": float(np.degrees(hdg_t)),
                "pitch_deg": float(np.degrees(pit_t)),
                "roll_deg": float(np.degrees(rol_t)),
                "target_vt": float(vt_t),
                "reached": True,
                "steps_to_reach": step,
                "final_theta_err_deg": theta_err,
                "final_delta_vt": delta_vt,
            })
            wp_idx += 1
            wp_hold_steps = 0
            if wp_idx >= num_wps:
                break
            hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
            core_state = core_state.replace(
                target_heading=jnp.full_like(core_state.target_heading, hdg_t),
                target_pitch=jnp.full_like(core_state.target_pitch, pit_t),
                target_roll=jnp.full_like(core_state.target_roll, rol_t),
                target_vt=jnp.full_like(core_state.target_vt, vt_t),
                last_check_time=core_state.time,
            )
            env_state = env_state.replace(env_state=core_state)

        # Handle episode termination (crash)
        obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], num_actors)
        done_arr = batchify(done, env.agents, config["NUM_ENVS"], num_actors).reshape(-1)
        if bool(np.asarray(done_arr).any()):
            crash_count += 1
            hstate = ScannedRNN.initialize_carry(num_actors * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
            hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
            core_state = env_state.env_state
            core_state = core_state.replace(
                target_heading=jnp.full_like(core_state.target_heading, hdg_t),
                target_pitch=jnp.full_like(core_state.target_pitch, pit_t),
                target_roll=jnp.full_like(core_state.target_roll, rol_t),
                target_vt=jnp.full_like(core_state.target_vt, vt_t),
            )
            env_state = env_state.replace(env_state=core_state)

    # Mark unreached waypoints
    for i in range(len(wp_results), num_wps):
        hdg_t, pit_t, rol_t, vt_t = waypoints_rad[i]
        wp_results.append({
            "wp_index": i,
            "heading_deg": float(np.degrees(hdg_t)),
            "pitch_deg": float(np.degrees(pit_t)),
            "roll_deg": float(np.degrees(rol_t)),
            "target_vt": float(vt_t),
            "reached": False,
            "steps_to_reach": -1,
            "final_theta_err_deg": -1.0,
            "final_delta_vt": -1.0,
        })

    return {
        "waypoints_reached": wp_idx,
        "total_waypoints": num_wps,
        "success_rate": wp_idx / num_wps if num_wps > 0 else 0.0,
        "total_reward": total_reward,
        "crash_count": crash_count,
        "stall_steps": stall_steps,
        "total_steps": min(step + 1, max_steps),
        "per_waypoint": wp_results,
    }


# ======================== Multi-episode parallel evaluation ========================

def run_evaluation(checkpoint_path: str, num_episodes: int = 50, max_steps: int = 2000):
    """Run multiple evaluation episodes and aggregate metrics."""

    devices = jax.devices()
    print(f"[evaluate] JAX devices: {[str(d) for d in devices]}")
    print(f"[evaluate] Checkpoint: {checkpoint_path}")
    print(f"[evaluate] Episodes: {num_episodes}, Max steps/episode: {max_steps}")

    config = {
        "SEED": 42,  # Fixed seed for deterministic evaluation (reproducible results across runs)
        "LR": 2e-4,
        "NUM_ENVS": 1,
        "NUM_ACTORS": 1,
        "FC_DIM_SIZE": 256,
        "GRU_HIDDEN_DIM": 256,
        "ACTIVATION": "relu",
    }

    env_params = FullDomain_TaskParams()
    env_core = AeroPlanaxFullDomainEnv(env_params)
    env = LogWrapper(env_core)
    config["NUM_ACTORS"] = env.num_agents

    # Build network
    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    # Deterministic evaluation: fixed PRNGKey from SEED ensures identical episode
    # sequences. Actions use greedy mode() (not sample), so no stochasticity.
    rng = jax.random.PRNGKey(config["SEED"])
    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"],
                    *env.observation_space(env.agents[0], env_params).shape)),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    init_h = ScannedRNN.initialize_carry(
        config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
    )
    init_params = network.init(rng, init_h, init_x)

    # Load checkpoint — only params needed for inference.
    # Use PyTreeCheckpointHandler with partial_restore=True to skip opt_state
    # (opt_state structure varies with optimizer config like ANNEAL_LR).
    params_template = {
        "params": network.init(rng, init_h, init_x),
        "epoch": jnp.array(0),
    }
    ckptr = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    checkpoint = ckptr.restore(
        checkpoint_path,
        args=ocp.args.PyTreeRestore(item=params_template, partial_restore=True),
    )
    loaded_params = checkpoint["params"]
    loaded_epoch = int(checkpoint.get("epoch", 0))
    print(f"[evaluate] Loaded epoch: {loaded_epoch}")

    # Prepare waypoints in radians
    waypoints_rad = [
        (np.radians(h), np.radians(p), np.radians(r), float(v))
        for h, p, r, v in TEST_WAYPOINTS
    ]

    # Run episodes (sequential for correctness, each episode uses full GPU)
    all_results = []
    t0 = time.time()

    for ep in range(num_episodes):
        rng, ep_rng = jax.random.split(rng)
        result = evaluate_single_episode(
            network, loaded_params, env, env_params, config,
            ep_rng, waypoints_rad, max_steps=max_steps,
        )
        all_results.append(result)
        if (ep + 1) % 10 == 0:
            avg_sr = np.mean([r["success_rate"] for r in all_results])
            print(f"  Episode {ep+1}/{num_episodes}  avg_success_rate={avg_sr:.3f}")

    elapsed = time.time() - t0
    print(f"[evaluate] {num_episodes} episodes completed in {elapsed:.1f}s")

    # Aggregate metrics
    success_rates = [r["success_rate"] for r in all_results]
    wps_reached = [r["waypoints_reached"] for r in all_results]
    total_rewards = [r["total_reward"] for r in all_results]
    crash_counts = [r["crash_count"] for r in all_results]
    stall_steps_all = [r["stall_steps"] for r in all_results]

    # Per-level analysis (group waypoints by difficulty level)
    level_boundaries = [0, 2, 4, 6, 8, 11]  # matches TEST_WAYPOINTS structure
    level_success = {}
    for lvl in range(5):
        wp_start, wp_end = level_boundaries[lvl], level_boundaries[lvl + 1]
        reached_in_level = []
        for r in all_results:
            count = sum(1 for wp in r["per_waypoint"][wp_start:wp_end] if wp["reached"])
            reached_in_level.append(count / (wp_end - wp_start))
        level_success[f"level_{lvl+1}"] = {
            "mean_success_rate": float(np.mean(reached_in_level)),
            "waypoint_range": f"WP{wp_start}-WP{wp_end-1}",
        }

    # Pitch/roll coverage analysis
    pitch_coverage = set()
    roll_coverage = set()
    for r in all_results:
        for wp in r["per_waypoint"]:
            if wp["reached"]:
                p_deg = abs(wp["pitch_deg"])
                r_deg = abs(wp["roll_deg"])
                if p_deg <= 15:
                    pitch_coverage.add("mild_pitch")
                elif p_deg <= 45:
                    pitch_coverage.add("moderate_pitch")
                else:
                    pitch_coverage.add("extreme_pitch")
                if r_deg <= 30:
                    roll_coverage.add("mild_roll")
                elif r_deg <= 90:
                    roll_coverage.add("moderate_roll")
                else:
                    roll_coverage.add("extreme_roll")

    report = {
        "metadata": {
            "checkpoint_path": checkpoint_path,
            "checkpoint_epoch": loaded_epoch,
            "num_episodes": num_episodes,
            "max_steps_per_episode": max_steps,
            "num_test_waypoints": len(TEST_WAYPOINTS),
            "evaluation_time_sec": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
        "aggregate": {
            "mean_success_rate": float(np.mean(success_rates)),
            "std_success_rate": float(np.std(success_rates)),
            "mean_waypoints_reached": float(np.mean(wps_reached)),
            "mean_total_reward": float(np.mean(total_rewards)),
            "mean_crash_count": float(np.mean(crash_counts)),
            "mean_stall_steps": float(np.mean(stall_steps_all)),
            "full_completion_rate": float(np.mean([1.0 if r["success_rate"] >= 1.0 else 0.0 for r in all_results])),
        },
        "per_level": level_success,
        "coverage": {
            "pitch_ranges_reached": sorted(list(pitch_coverage)),
            "roll_ranges_reached": sorted(list(roll_coverage)),
            "full_domain_achieved": (
                "extreme_pitch" in pitch_coverage and "extreme_roll" in roll_coverage
            ),
        },
        "quality_assessment": _assess_quality(success_rates, level_success, crash_counts, stall_steps_all),
    }

    return report


def _assess_quality(success_rates, level_success, crash_counts, stall_steps_all):
    """Generate human-readable quality assessment."""
    mean_sr = float(np.mean(success_rates))
    mean_crash = float(np.mean(crash_counts))

    issues = []
    if mean_sr < 0.3:
        issues.append("CRITICAL: Overall success rate < 30%. Agent cannot track targets reliably.")
    elif mean_sr < 0.6:
        issues.append("WARNING: Success rate 30-60%. Agent struggles with harder maneuvers.")

    if mean_crash > 2.0:
        issues.append(f"WARNING: High crash rate ({mean_crash:.1f}/episode). Check altitude safety or overload limits.")

    if float(np.mean(stall_steps_all)) > 50:
        issues.append("WARNING: Frequent stall conditions. Speed management needs improvement.")

    # Per-level drill-down
    for lvl_key, lvl_data in level_success.items():
        if lvl_data["mean_success_rate"] < 0.5:
            issues.append(f"WEAK: {lvl_key} success rate = {lvl_data['mean_success_rate']:.2f}. "
                          f"Focus reward shaping on {lvl_data['waypoint_range']}.")

    if not issues:
        issues.append("GOOD: All metrics within acceptable range.")

    verdict = "PASS" if mean_sr >= 0.8 and mean_crash <= 1.0 else "FAIL"

    return {
        "verdict": verdict,
        "mean_success_rate": mean_sr,
        "issues": issues,
    }


# ======================== CLI entry point ========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate full-domain maneuver policy")
    parser.add_argument("--checkpoint", required=True, help="Path to Orbax checkpoint directory")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--out", default=None, help="Output JSON file path (default: stdout)")
    args = parser.parse_args()

    report = run_evaluation(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
    )

    json_str = json.dumps(report, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"\n[evaluate] Report saved to {args.out}")
    else:
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)
        print(json_str)

    return report


if __name__ == "__main__":
    main()
