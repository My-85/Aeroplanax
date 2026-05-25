"""
Fair Euler vs Quaternion comparison — v2.

Uses the same task geometry for both policies, computes all metrics from
plane_state (identical fields in both envs), and tests aggressive vertical
loop-plane arcs (60°–150°) where quaternion representation should win.

Euler checkpoint: results/heading_pitch_V_discrete_rnn_2026-05-14-15-29/checkpoints/checkpoint_epoch_300
Quat checkpoint:   results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619
"""
import argparse
import csv
import functools
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.linen.initializers import constant, orthogonal

# Euler env
from envs.aeroplanax_heading_pitch_V import (
    AeroPlanaxHeading_Pitch_V_Env as EulerEnv,
    Heading_Pitch_V_TaskParams as EulerParams,
)
from envs.utils.utils import wrap_PI

# Quaternion env (for Euler baseline reference — not used for Quat epoch 619)
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as QuatEnvSimple,
    Heading_Pitch_V_TaskParams as QuatParamsSimple,
)
# Vertical-energy env (native env for Quat epoch 619 — has task_mode/task_duration)
from envs.aeroplanax_heading_pitch_V_quaternion_version_vertical_energy import (
    AeroPlanaxHeading_Pitch_V_Env as VertEnergyEnv,
    Heading_Pitch_V_TaskParams as VertEnergyParams,
)

# Loop-plane target generation
from experiments.hierarchical_trajectory_tracking.loop_attitude_target import loop_plane_hpr_jax

PLANAX_ROOT = Path(__file__).resolve().parent

EULER_CKPT = PLANAX_ROOT / "results/heading_pitch_V_discrete_rnn_2026-05-14-15-29/checkpoints/checkpoint_epoch_300"
QUAT_CKPT  = PLANAX_ROOT / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
OUT_DIR    = PLANAX_ROOT / "results/euler_vs_quat_v2"

NET_CONFIG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
DT_RL = 10.0 / 50.0
G = 9.80665


# ═══════════════════════════════════════════════════════════════════════
# Network
# ═══════════════════════════════════════════════════════════════════════

class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0,
                       split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis],
                              self.initialize_carry(*rnn_state.shape), rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        embedding = nn.Dense(self.config["FC_DIM_SIZE"],
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))
        fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                       bias_init=constant(0.0))(embedding)
        fc2 = nn.LayerNorm()(fc2)
        fc2 = activation(fc2)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"],
                              kernel_init=orthogonal(2),
                              bias_init=constant(0.0))(fc2)
        actor_mean = activation(actor_mean)
        pis = []
        for i, ad in enumerate(self.action_dim):
            if i == 4:
                pis.append(distrax.Categorical(logits=nn.Dense(
                    ad, kernel_init=constant(0.0),
                    bias_init=lambda key, shape, dtype=jnp.float32: jnp.array(
                        [0.0, -1.5, -1.5, -1.5, -1.5], dtype=dtype))(actor_mean)))
            else:
                pis.append(distrax.Categorical(logits=nn.Dense(
                    ad, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)))
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2),
                          bias_init=constant(0.0))(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, tuple(pis), jnp.squeeze(critic, axis=-1)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def restore_params(path: Path):
    ckpt = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        str(path), args=ocp.args.StandardRestore())
    return ckpt["params"], int(np.asarray(ckpt["epoch"]))


def wrap_pi_np(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def euler_to_quat_nb_np(roll, pitch, yaw):
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.stack([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy], axis=-1)


def quat_conj_np(q):
    out = np.array(q, copy=True); out[..., 1:] *= -1.0; return out


def quat_mul_np(q1, q2):
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1)


def quat_angle_deg_np(q_curr_bn, target_roll, target_pitch, target_yaw):
    """Geodesic angle between current and target attitude, in degrees."""
    q_tgt_bn = quat_conj_np(euler_to_quat_nb_np(target_roll, target_pitch, target_yaw))
    q_curr_bn = q_curr_bn / (np.linalg.norm(q_curr_bn, axis=-1, keepdims=True) + 1e-9)
    q_tgt_bn = q_tgt_bn / (np.linalg.norm(q_tgt_bn, axis=-1, keepdims=True) + 1e-9)
    q_err = quat_mul_np(q_tgt_bn, quat_conj_np(q_curr_bn))
    w = np.clip(np.abs(q_err[..., 0]), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def body_axes_from_euler_np(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    nose = np.stack([cp*cy, cp*sy, -sp], axis=-1)
    right = np.stack([-cr*sy + sr*sp*cy, cr*cy + sr*sp*sy, sr*cp], axis=-1)
    return nose, right


def angle_deg_np(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def np_mean_std(x):
    arr = np.asarray(x, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


# ═══════════════════════════════════════════════════════════════════════
# Task catalog — aggressive vertical arcs + horizontal baselines
# ═══════════════════════════════════════════════════════════════════════

def task_catalog():
    return [
        # ── Horizontal baselines ──
        {"name": "circle_R5000_right", "category": "horizontal",
         "kind": "circle", "radius": 5000.0, "direction": 1.0, "max_steps": 600},
        {"name": "circle_R5000_left", "category": "horizontal",
         "kind": "circle", "radius": 5000.0, "direction": -1.0, "max_steps": 600},
        {"name": "s_curve_A3000", "category": "horizontal",
         "kind": "s_curve", "amplitude": 3000.0, "max_steps": 600},
        {"name": "figure_eight_R5000", "category": "horizontal",
         "kind": "figure_eight", "radius": 5000.0, "max_steps": 600},
        # ── Mild vertical (for calibration) ──
        {"name": "pullup_15_R3000", "category": "vertical_mild",
         "kind": "pullup", "angle_deg": 15.0, "radius": 3000.0, "max_steps": 500},
        {"name": "pullup_30_R5000", "category": "vertical_mild",
         "kind": "pullup", "angle_deg": 30.0, "radius": 5000.0, "max_steps": 500},
        # ── Aggressive loop-plane arcs (the key comparison) ──
        {"name": "loop_arc_60_R10000", "category": "loop_arc",
         "kind": "loop_arc", "angle_deg": 60.0, "radius": 10000.0, "max_steps": 600},
        {"name": "loop_arc_90_R10000", "category": "loop_arc",
         "kind": "loop_arc", "angle_deg": 90.0, "radius": 10000.0, "max_steps": 600},
        {"name": "loop_arc_120_R15000", "category": "loop_arc",
         "kind": "loop_arc", "angle_deg": 120.0, "radius": 15000.0, "max_steps": 700},
        {"name": "loop_arc_150_R15000", "category": "loop_arc",
         "kind": "loop_arc", "angle_deg": 150.0, "radius": 15000.0, "max_steps": 800},
    ]


def task_horizon(task):
    kind = task["kind"]
    if kind in ("pullup", "loop_arc"):
        ramp_steps = max(10, int(np.ceil(np.deg2rad(task["angle_deg"]) * task["radius"] / 250.0 / DT_RL)))
        hold_steps = 60
        return ramp_steps + hold_steps, ramp_steps
    if kind in ("circle", "s_curve", "figure_eight"):
        return task.get("max_steps", 600), 1
    return task.get("max_steps", 500), 1


def build_target(task, step, ramp_steps, init, state):
    yaw0, pitch0, alt0 = init
    shape = yaw0.shape
    target_heading = jnp.asarray(yaw0)
    target_pitch = jnp.zeros(shape)
    target_roll = jnp.zeros(shape)
    target_vt = jnp.full(shape, 250.0)
    t = step * DT_RL
    kind = task["kind"]

    if kind == "pullup":
        frac = jnp.clip(step / jnp.maximum(ramp_steps - 1, 1), 0.0, 1.0)
        frac = frac * frac * (3.0 - 2.0 * frac)
        target_pitch = pitch0 + jnp.deg2rad(task["angle_deg"]) * frac
    elif kind == "loop_arc":
        frac = jnp.clip(step / jnp.maximum(ramp_steps - 1, 1), 0.0, 1.0)
        frac = frac * frac * (3.0 - 2.0 * frac)
        theta = jnp.deg2rad(task["angle_deg"]) * frac
        target_heading, target_pitch, target_roll = loop_plane_hpr_jax(theta, yaw0, 1.0)
    elif kind == "circle":
        omega = 250.0 / task["radius"]
        target_heading = jnp.asarray(wrap_PI(yaw0 + task.get("direction", 1.0) * omega * t))
    elif kind == "s_curve":
        period = 85.0; amp = jnp.deg2rad(32.0)
        target_heading = jnp.asarray(wrap_PI(yaw0 + amp * jnp.sin(2.0 * jnp.pi * t / period)))
    elif kind == "figure_eight":
        period = 120.0; amp = jnp.deg2rad(42.0)
        target_heading = jnp.asarray(wrap_PI(yaw0 + amp * jnp.sin(4.0 * jnp.pi * t / period)))

    return target_heading, target_pitch, target_roll, target_vt


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=20260520)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    num_seeds = args.seeds

    # ── Load envs ──
    # Euler policy → native Euler env (22-dim obs)
    euler_env_params = EulerParams()
    euler_env = EulerEnv(euler_env_params)
    euler_agent = euler_env.agents[0]

    # Quat epoch 619 → native vertical_energy env (21-dim obs + task context)
    vert_env_params = VertEnergyParams()
    vert_env = VertEnergyEnv(vert_env_params)
    vert_agent = vert_env.agents[0]

    action_dim = [31, 41, 41, 41, 5]

    # ── Load policies ──
    euler_network = ActorCriticRNN(action_dim, config=NET_CONFIG)
    euler_params, euler_epoch = restore_params(EULER_CKPT.resolve())

    quat_network = ActorCriticRNN(action_dim, config=NET_CONFIG)
    quat_params, quat_epoch = restore_params(QUAT_CKPT.resolve())

    print(f"Devices: {jax.devices()}")
    print(f"Euler epoch: {euler_epoch}  (obs_dim=22, env=Euler)")
    print(f"Quat  epoch: {quat_epoch}  (obs_dim=21, env=vertical_energy)")

    # ── JIT step functions ──
    @jax.jit
    def euler_step(params, hstate, state, done, key, th, tp, tr, tv):
        state = state.replace(target_heading=th, target_pitch=tp,
                              target_roll=tr, target_vt=tv,
                              last_check_time=state.time)
        obs_dict = jax.vmap(euler_env._get_obs, in_axes=(0, None))(state, euler_env_params)
        obs = obs_dict[euler_agent]
        hstate, pi, _ = euler_network.apply(params, hstate, (obs[None, :, :], done[None, :]))
        action = jnp.stack([p.mode()[0] for p in pi], axis=-1).astype(jnp.int32)
        step_keys = jax.random.split(key, num_seeds)
        obs_next, state_next, reward, done_dict, info = jax.vmap(
            euler_env.step, in_axes=(0, 0, 0, None))(step_keys, state, {euler_agent: action}, euler_env_params)
        del obs_next
        return state_next, done_dict[euler_agent], hstate, info, action

    @jax.jit
    def quat_step(params, hstate, state, done, key, th, tp, tr, tv):
        state = state.replace(target_heading=th, target_pitch=tp,
                              target_roll=tr, target_vt=tv,
                              task_mode=jnp.zeros_like(state.task_mode),
                              task_duration_steps=jnp.full_like(state.task_duration_steps, 10000.0),
                              last_check_time=state.time)
        obs_dict = jax.vmap(vert_env._get_obs, in_axes=(0, None))(state, vert_env_params)
        obs = obs_dict[vert_agent]
        hstate, pi, _ = quat_network.apply(params, hstate, (obs[None, :, :], done[None, :]))
        action = jnp.stack([p.mode()[0] for p in pi], axis=-1).astype(jnp.int32)
        step_keys = jax.random.split(key, num_seeds)
        obs_next, state_next, reward, done_dict, info = jax.vmap(
            vert_env.step, in_axes=(0, 0, 0, None))(step_keys, state, {vert_agent: action}, vert_env_params)
        del obs_next
        return state_next, done_dict[vert_agent], hstate, info, action

    # ── Evaluate ──
    policies = {
        "Euler_epoch300": (euler_params, euler_env, euler_env_params, euler_agent, euler_step),
        "Quat_epoch619": (quat_params, vert_env, vert_env_params, vert_agent, quat_step),
    }

    all_rows = []
    tasks = task_catalog()

    for policy_name, (net_params, env, env_params, agent, step_fn) in policies.items():
        print(f"\n{'='*60}\nEvaluating: {policy_name}\n{'='*60}")
        for task_idx, task in enumerate(tasks):
            max_steps, ramp_steps = task_horizon(task)
            reset_keys = jax.random.split(jax.random.PRNGKey(args.seed_base + task_idx), num_seeds)
            _, state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
            init = (state.plane_state.yaw, state.plane_state.pitch, state.plane_state.altitude)

            # Set extra state fields if they exist
            extra_kwargs = {}
            if hasattr(state, "task_start_heading"):
                extra_kwargs.update(
                    task_start_heading=state.plane_state.yaw,
                    task_start_pitch=state.plane_state.pitch,
                    task_start_roll=state.plane_state.roll,
                    task_start_vt=state.plane_state.vt,
                    task_start_altitude=state.plane_state.altitude,
                    task_start_energy=0.5 * state.plane_state.vt * state.plane_state.vt + G * state.plane_state.altitude,
                )
            if hasattr(state, "task_mode"):
                extra_kwargs["task_mode"] = jnp.zeros_like(state.task_mode)
            if hasattr(state, "task_duration_steps"):
                extra_kwargs["task_duration_steps"] = jnp.full_like(state.task_duration_steps, 10000.0)
            if hasattr(state, "last_check_time"):
                extra_kwargs["last_check_time"] = state.time
            state = state.replace(**extra_kwargs)

            hstate = ScannedRNN.initialize_carry(num_seeds, NET_CONFIG["GRU_HIDDEN_DIM"])
            done = jnp.zeros((num_seeds,), dtype=jnp.bool_)

            # ── Accumulators (all from plane_state — env-agnostic) ──
            alt0 = np.asarray(state.plane_state.altitude[:, 0])
            prev_north = np.asarray(state.plane_state.north[:, 0]).copy()
            prev_east = np.asarray(state.plane_state.east[:, 0]).copy()
            prev_altitude = alt0.copy()

            alt_min = alt0.copy(); alt_max = alt0.copy()
            vt_min = np.full(num_seeds, np.inf); vt_max = np.zeros(num_seeds)
            alpha_max = np.zeros(num_seeds)
            gmax = np.zeros(num_seeds)
            q_err_sum = np.zeros(num_seeds)
            heading_err_sum = np.zeros(num_seeds)
            pitch_err_sum = np.zeros(num_seeds)
            vel_tan_err_sum = np.zeros(num_seeds)
            nose_tan_err_sum = np.zeros(num_seeds)
            nose_vel_err_sum = np.zeros(num_seeds)
            wing_plane_err_sum = np.zeros(num_seeds)
            active_count = np.zeros(num_seeds)
            crash_count = np.zeros(num_seeds)
            stall_steps = np.zeros(num_seeds)
            overg_steps = np.zeros(num_seeds)
            cte_values = [[] for _ in range(num_seeds)]
            cte_p90_per_seed = np.zeros(num_seeds)

            reason = np.array(["none"] * num_seeds, dtype=object)

            for step in range(max_steps):
                th, tp, tr, tv = build_target(task, step, ramp_steps, init, state)
                key = jax.random.PRNGKey(args.seed_base + 100000 + task_idx * 1000 + step)
                state, done_step, hstate, info, action = step_fn(
                    net_params, hstate, state, done, key, th, tp, tr, tv)

                active = ~np.asarray(done)
                if not active.any():
                    break

                # ── All metrics from plane_state (identical fields for both envs) ──
                vt = np.asarray(state.plane_state.vt)[:, 0]
                alt = np.asarray(state.plane_state.altitude)[:, 0]
                pitch_rad = np.asarray(state.plane_state.pitch)[:, 0]
                yaw_rad = np.asarray(state.plane_state.yaw)[:, 0]
                roll_rad = np.asarray(state.plane_state.roll)[:, 0]
                alpha_rad = np.asarray(state.plane_state.alpha)[:, 0]
                beta_rad = np.asarray(state.plane_state.beta)[:, 0]
                az_nd = np.asarray(state.plane_state.az)[:, 0]  # G-load

                target_heading_rad = np.asarray(th)[:, 0]
                target_pitch_rad = np.asarray(tp)[:, 0]
                target_roll_rad = np.asarray(tr)[:, 0]

                # Attitude geodesic error
                q_vals = np.stack([
                    np.asarray(state.plane_state.q0)[:, 0],
                    np.asarray(state.plane_state.q1)[:, 0],
                    np.asarray(state.plane_state.q2)[:, 0],
                    np.asarray(state.plane_state.q3)[:, 0],
                ], axis=-1)
                q_err = quat_angle_deg_np(q_vals, target_roll_rad, target_pitch_rad, target_heading_rad)

                # Heading/pitch errors
                heading_err = np.abs(wrap_pi_np(yaw_rad - target_heading_rad)) * 180.0 / np.pi
                pitch_err = np.abs(np.degrees(pitch_rad) - np.degrees(target_pitch_rad))

                # Geometry-aware errors
                actual_nose, actual_right = body_axes_from_euler_np(roll_rad, pitch_rad, yaw_rad)
                target_nose, target_right = body_axes_from_euler_np(target_roll_rad, target_pitch_rad, target_heading_rad)
                north = np.asarray(state.plane_state.north)[:, 0]
                east = np.asarray(state.plane_state.east)[:, 0]
                displacement_n = north - prev_north
                displacement_e = east - prev_east
                displacement_d = -(alt - prev_altitude)
                velocity_n = np.stack([displacement_n, displacement_e, displacement_d], axis=-1)
                displacement_norm = np.linalg.norm(velocity_n, axis=-1, keepdims=True)
                velocity_n = np.where(displacement_norm > 1e-6, velocity_n, actual_nose)

                vel_tan_err = angle_deg_np(velocity_n, target_nose)
                nose_tan_err = angle_deg_np(actual_nose, target_nose)
                nose_vel_err = angle_deg_np(actual_nose, velocity_n)
                wing_plane_err = angle_deg_np(actual_right, target_right)

                # CTE (task-dependent)
                kind = task["kind"]
                if kind in ("circle", "s_curve", "figure_eight"):
                    cte = heading_err
                elif kind in ("pullup", "loop_arc"):
                    cte = np.sqrt(
                        (north - np.asarray(state.plane_state.north)[:, 0])**2 +
                        (east - np.asarray(state.plane_state.east)[:, 0])**2 +
                        (alt - alt0)**2
                    )
                    # Actually just use max of geometry errors for loop arcs
                    cte = np.maximum.reduce([vel_tan_err, nose_tan_err, wing_plane_err])
                else:
                    cte = pitch_err

                # Accumulate
                alt_min[active] = np.minimum(alt_min[active], alt[active])
                alt_max[active] = np.maximum(alt_max[active], alt[active])
                vt_min[active] = np.minimum(vt_min[active], vt[active])
                vt_max[active] = np.maximum(vt_max[active], vt[active])
                alpha_max[active] = np.maximum(alpha_max[active], np.abs(alpha_rad[active]) * 180.0 / np.pi)
                gmax[active] = np.maximum(gmax[active], np.abs(az_nd[active]))
                q_err_sum[active] += q_err[active]
                heading_err_sum[active] += heading_err[active]
                pitch_err_sum[active] += pitch_err[active]
                vel_tan_err_sum[active] += vel_tan_err[active]
                nose_tan_err_sum[active] += nose_tan_err[active]
                nose_vel_err_sum[active] += nose_vel_err[active]
                wing_plane_err_sum[active] += wing_plane_err[active]
                stall_steps[active] += (np.abs(alpha_rad[active]) * 180.0 / np.pi > 30.0).astype(float)
                overg_steps[active] += (np.abs(az_nd[active]) > 10.0).astype(float)
                active_count[active] += 1.0

                for s in np.where(active)[0]:
                    cte_values[s].append(float(cte[s]))

                prev_north[active] = north[active]
                prev_east[active] = east[active]
                prev_altitude[active] = alt[active]

                # Detect crashes (env-agnostic: altitude <= 0 or extreme alpha)
                done_np = np.asarray(done_step)
                newly_done = active & done_np
                for i in np.where(newly_done)[0]:
                    if alt[i] <= 0.0:
                        reason[i] = "crash_ground"
                        crash_count[i] = 1.0
                    elif np.abs(az_nd[i]) > 12.0:
                        reason[i] = "overload"
                    else:
                        reason[i] = "env_done"
                done = jnp.asarray(np.asarray(done) | done_np)

            # ── Aggregate metrics ──
            active_count = np.maximum(active_count, 1.0)
            for s in range(num_seeds):
                if len(cte_values[s]) > 0:
                    cte_p90_per_seed[s] = np.percentile(cte_values[s], 90)

            survivors = np.array([r in ("none",) for r in reason])
            survival_rate = float(survivors.mean())

            row = {
                "policy": policy_name,
                "task": task["name"],
                "category": task["category"],
                "survival_rate": survival_rate,
                "crash_rate": float(crash_count.mean()),
                "q_error_mean_deg": float((q_err_sum / active_count).mean()),
                "q_error_std_deg": float((q_err_sum / active_count).std()),
                "heading_err_mean_deg": float((heading_err_sum / active_count).mean()),
                "pitch_err_mean_deg": float((pitch_err_sum / active_count).mean()),
                "vel_tan_err_mean_deg": float((vel_tan_err_sum / active_count).mean()),
                "nose_tan_err_mean_deg": float((nose_tan_err_sum / active_count).mean()),
                "nose_vel_err_mean_deg": float((nose_vel_err_sum / active_count).mean()),
                "wing_plane_err_mean_deg": float((wing_plane_err_sum / active_count).mean()),
                "CTE_p90_mean": float(cte_p90_per_seed.mean()),
                "vt_min_mean": float(vt_min.mean()),
                "vt_max_mean": float(vt_max.mean()),
                "alpha_max_mean_deg": float(alpha_max.mean()),
                "gmax_mean": float(gmax.mean()),
                "stall_rate": float((stall_steps / active_count).mean()),
                "overg_rate": float((overg_steps / active_count).mean()),
                "mean_ep_length": float(active_count.mean()),
                "top_termination": Counter(reason).most_common(1)[0][0],
            }
            all_rows.append(row)
            print(f"  {task['name']:25s}  surv={survival_rate:.2f}  q_err={row['q_error_mean_deg']:.1f}°  "
                  f"vel_tan={row['vel_tan_err_mean_deg']:.1f}°  nose_tan={row['nose_tan_err_mean_deg']:.1f}°  "
                  f"wing={row['wing_plane_err_mean_deg']:.1f}°  stall={row['stall_rate']:.3f}  "
                  f"term={row['top_termination']}")

    # ── Write full CSV ──
    csv_path = args.out_dir / "full_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nFull CSV: {csv_path}")

    # ── Build head-to-head comparison ──
    euler_by_task = {r["task"]: r for r in all_rows if r["policy"] == "Euler_epoch300"}
    quat_by_task = {r["task"]: r for r in all_rows if r["policy"] == "Quat_epoch619"}

    print("\n" + "=" * 80)
    print("LATEX TABLE: Euler vs Quaternion (epoch 619) — Fair Comparison")
    print("=" * 80)

    for section, section_tasks in [
        ("Horizontal Maneuvers", ["circle_R5000_right", "circle_R5000_left",
                                   "s_curve_A3000", "figure_eight_R5000"]),
        ("Mild Vertical (calibration)", ["pullup_15_R3000", "pullup_30_R5000"]),
        ("Aggressive Loop-Plane Arcs", ["loop_arc_60_R10000", "loop_arc_90_R10000",
                                         "loop_arc_120_R15000", "loop_arc_150_R15000"]),
    ]:
        print(f"\n  % --- {section} ---")
        for tn in section_tasks:
            er = euler_by_task.get(tn, {})
            qr = quat_by_task.get(tn, {})
            if not er or not qr:
                continue
            e_q = er.get("q_error_mean_deg", 0)
            q_q = qr.get("q_error_mean_deg", 0)
            e_vt = er.get("vel_tan_err_mean_deg", 0)
            q_vt = qr.get("vel_tan_err_mean_deg", 0)
            e_nt = er.get("nose_tan_err_mean_deg", 0)
            q_nt = qr.get("nose_tan_err_mean_deg", 0)
            e_wp = er.get("wing_plane_err_mean_deg", 0)
            q_wp = qr.get("wing_plane_err_mean_deg", 0)
            e_surv = er.get("survival_rate", 0)
            q_surv = qr.get("survival_rate", 0)
            e_stall = er.get("stall_rate", 0)
            q_stall = qr.get("stall_rate", 0)
            ratio = e_q / max(q_q, 0.01)
            winner = "Q" if q_q < e_q else ("E" if e_q < q_q else "=")
            print(f"    {tn:28s} & {e_q:5.1f} & {q_q:5.1f} & {e_vt:5.1f} & {q_vt:5.1f} & "
                  f"{e_nt:5.1f} & {q_nt:5.1f} & {e_wp:5.1f} & {q_wp:5.1f} & "
                  f"{e_surv:.0%} & {q_surv:.0%} & {ratio:.1f}$\\times$ & {winner} \\\\")

    print("\nDone!")


if __name__ == "__main__":
    main()
