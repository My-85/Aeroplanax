#!/usr/bin/env python3
"""
eval_quat_baseline.py — Evaluate the trained quaternion baseline checkpoint.

This uses the heading_pitch_V_quaternion_version_add_full_roll env (obs=16D, GRU=128, FC=128).
NOT compatible with full_domain env. Dedicated script for reference measurement only.

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval_quat_baseline.py
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["XLA_PYTHON_MEM_FRACTION"] = "0.90"

import sys
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import functools
from typing import Sequence, Dict
from flax.linen.initializers import constant, orthogonal
import orbax.checkpoint as ocp

PLANAX_DIR = Path(__file__).resolve().parent.parent / "Planax"
sys.path.insert(0, str(PLANAX_DIR))

from envs.wrappers import LogWrapper
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env,
    Heading_Pitch_V_TaskParams,
)

# ---- Quaternion helpers (self-contained) ----
def _quat_normalize(q):
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)

def _quat_conj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])

def _quat_from_euler_bn(roll, pitch, yaw):
    cr, sr = jnp.cos(0.5 * roll),  jnp.sin(0.5 * roll)
    cp, sp = jnp.cos(0.5 * pitch), jnp.sin(0.5 * pitch)
    cy, sy = jnp.cos(0.5 * yaw),   jnp.sin(0.5 * yaw)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return jnp.stack([qw, qx, qy, qz], axis=-1)

def _quat_geodesic_angle(q_a, q_b):
    q_a = _quat_normalize(q_a)
    q_b = _quat_normalize(q_b)
    cos_half = jnp.abs(jnp.sum(q_a * q_b, axis=-1))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half)

# ---- Network (must match training: GRU=128, FC=128) ----
class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state)
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
        embedding = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)
        pi_throttle = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_aileron  = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_rudder   = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)


# ---- Config ----
CHECKPOINT_PATH = "/home/dqy/aeroplanax/new/20251215最新代码库/results/baseline（四元数版本）/checkpoints/checkpoint_epoch_1000"

EVAL_CONFIG = {
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION": "relu",
    "NUM_ENVS": 1,
    "NUM_STEPS": 2000,   # 400 sim steps * 50/10 = 2000 decision steps = 40s
    "EVAL_SEEDS": [42, 137, 256],
}


def main():
    config = EVAL_CONFIG
    num_envs = config["NUM_ENVS"]

    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)
    env = LogWrapper(env)
    num_actors = env.num_agents
    obs_shape = env.observation_space(env.agents[0], env_params).shape

    # Init network
    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    rng = jax.random.PRNGKey(0)
    init_x = (
        jnp.zeros((1, num_envs * num_actors, *obs_shape)),
        jnp.zeros((1, num_envs * num_actors)),
    )
    init_hstate = ScannedRNN.initialize_carry(num_envs * num_actors, config["GRU_HIDDEN_DIM"])
    network_params = network.init(rng, init_hstate, init_x)

    # Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    t0 = time.time()
    params_template = {"params": network_params, "epoch": jnp.array(0)}
    ckptr = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    checkpoint = ckptr.restore(
        CHECKPOINT_PATH,
        args=ocp.args.PyTreeRestore(item=params_template, partial_restore=True),
    )
    loaded_params = checkpoint["params"]
    print(f"Loaded in {time.time()-t0:.1f}s (epoch={checkpoint.get('epoch', '?')})")

    # Run evaluation
    all_seed_results = []
    for seed in config["EVAL_SEEDS"]:
        print(f"\n--- Seed {seed} ---")
        rng = jax.random.PRNGKey(seed)

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        obsv, state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
        hstate = ScannedRNN.initialize_carry(num_envs * num_actors, config["GRU_HIDDEN_DIM"])

        all_theta = []
        all_dvt = []
        all_rewards = []
        all_crashes = []
        all_on_target = []
        num_task_switches = 0

        for step in range(config["NUM_STEPS"]):
            rng, act_rng = jax.random.split(rng)

            obs_batch = jnp.stack([obsv[a] for a in env.agents]).reshape((num_actors * num_envs, -1))
            dones_in = jnp.zeros((num_actors * num_envs,))
            ac_in = (obs_batch[np.newaxis, :, :], dones_in[np.newaxis, :])
            hstate, pi, value = network.apply(loaded_params, hstate, ac_in)

            actions = jnp.stack([p.mode() for p in pi], axis=-1).squeeze(0)
            actions_dict = {}
            for i, agent in enumerate(env.agents):
                actions_dict[agent] = actions[i * num_envs : (i + 1) * num_envs]

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_envs)
            obsv, state, rewards, dones_dict, infos = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_rngs, state, actions_dict)

            agent0 = env.agents[0]
            all_rewards.append(float(jnp.mean(rewards[agent0])))
            done_vals = dones_dict[agent0]
            all_crashes.append(float(jnp.mean(done_vals)))

            # Compute theta_deg and delta_vt
            es = state.env_state
            ps = es.plane_state
            q_curr = jnp.stack([
                jnp.nan_to_num(ps.q0[:, 0], nan=1.0),
                jnp.nan_to_num(ps.q1[:, 0], nan=0.0),
                jnp.nan_to_num(ps.q2[:, 0], nan=0.0),
                jnp.nan_to_num(ps.q3[:, 0], nan=0.0),
            ], axis=-1)

            tgt_h = es.target_heading[:, 0]
            tgt_p = es.target_pitch[:, 0]
            tgt_r = es.target_roll[:, 0]
            tgt_vt = es.target_vt[:, 0]

            q_tgt = _quat_conj(_quat_from_euler_bn(tgt_r, tgt_p, tgt_h))
            theta = _quat_geodesic_angle(q_curr, q_tgt)
            theta_deg_val = float(jnp.mean(theta * 180.0 / jnp.pi))

            vt = jnp.nan_to_num(ps.vt[:, 0], nan=0.0)
            dvt_val = float(jnp.mean(jnp.abs(vt - tgt_vt)))

            all_theta.append(theta_deg_val)
            all_dvt.append(dvt_val)

            on_target = (theta * 180.0 / jnp.pi < 10.0) & (jnp.abs(vt - tgt_vt) < 25.0)
            all_on_target.append(float(jnp.mean(on_target.astype(jnp.float32))))

            # Reset RNN on done
            done_broadcast = jnp.repeat(done_vals, num_actors)
            hstate = jnp.where(done_broadcast[:, None], 0.0, hstate)

            if step % 200 == 0:
                print(f"  step={step:4d}  theta={theta_deg_val:6.1f}deg  dvt={dvt_val:5.1f}  reward={all_rewards[-1]:.3f}")

        seed_result = {
            "seed": seed,
            "mean_theta_deg": float(np.mean(all_theta)),
            "mean_delta_vt": float(np.mean(all_dvt)),
            "mean_per_step_reward": float(np.mean(all_rewards)),
            "crash_rate": float(np.mean(all_crashes)),
            "on_target_rate": float(np.mean(all_on_target)),
        }
        all_seed_results.append(seed_result)
        print(f"  SUMMARY: theta={seed_result['mean_theta_deg']:.1f}deg  dvt={seed_result['mean_delta_vt']:.1f}  "
              f"on_target={seed_result['on_target_rate']:.3f}  crash={seed_result['crash_rate']:.4f}")

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATE ACROSS SEEDS:")
    agg = {
        "mean_theta_deg": float(np.mean([r["mean_theta_deg"] for r in all_seed_results])),
        "std_theta_deg": float(np.std([r["mean_theta_deg"] for r in all_seed_results])),
        "mean_delta_vt": float(np.mean([r["mean_delta_vt"] for r in all_seed_results])),
        "std_delta_vt": float(np.std([r["mean_delta_vt"] for r in all_seed_results])),
        "mean_per_step_reward": float(np.mean([r["mean_per_step_reward"] for r in all_seed_results])),
        "mean_crash_rate": float(np.mean([r["crash_rate"] for r in all_seed_results])),
        "mean_on_target_rate": float(np.mean([r["on_target_rate"] for r in all_seed_results])),
    }
    print(f"  theta_deg = {agg['mean_theta_deg']:.1f} +/- {agg['std_theta_deg']:.1f} deg")
    print(f"  delta_vt  = {agg['mean_delta_vt']:.1f} +/- {agg['std_delta_vt']:.1f}")
    print(f"  reward    = {agg['mean_per_step_reward']:.3f}")
    print(f"  crash     = {agg['mean_crash_rate']:.4f}")
    print(f"  on_target = {agg['mean_on_target_rate']:.3f}")

    # Save
    output_path = Path(__file__).parent / "baselines" / "quat_baseline_eval.json"
    with open(output_path, "w") as f:
        json.dump({"aggregate": agg, "per_seed": all_seed_results}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
