"""
Euler vs Quaternion comparison evaluator.
Compares Euler-angle baseline (heading_pitch_V_discrete_rnn_2026-05-14-15-29/epoch_300)
against quaternion baseline (heading_pitch_V_discrete_rnn_2026-05-13-21-17/epoch_600)
on the same task catalog.
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

# ── Euler env ──
from envs.aeroplanax_heading_pitch_V import (
    AeroPlanaxHeading_Pitch_V_Env as EulerEnv,
    Heading_Pitch_V_TaskParams as EulerParams,
)

# ── Quaternion env ──
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as QuatEnv,
    Heading_Pitch_V_TaskParams as QuatParams,
    _quat_conj,
    _quat_from_euler_nb,
)
from envs.utils.utils import wrap_PI

PLANAX_ROOT = Path(__file__).resolve().parent

# ── Checkpoints ──
EULER_CKPT = PLANAX_ROOT / "results/heading_pitch_V_discrete_rnn_2026-05-14-15-29/checkpoints/checkpoint_epoch_300"
QUAT_CKPT = PLANAX_ROOT / "results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600"
VERTICAL_CKPT = PLANAX_ROOT / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"

OUT_DIR = PLANAX_ROOT / "results/euler_vs_quat_comparison"

NET_CONFIG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
DT_RL = 10.0 / 50.0
G = 9.80665


# ═══════════════════════════════════════════════════════════════════════
# Network (same architecture for both, only obs_dim differs)
# ═══════════════════════════════════════════════════════════════════════

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
            resets[:, np.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


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

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            embedding
        )
        fc2 = nn.LayerNorm()(fc2)
        fc2 = activation(fc2)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(fc2)
        actor_mean = activation(actor_mean)

        pis = []
        for i, ad in enumerate(self.action_dim):
            if i == 4:  # speed brake — zero-biased
                pis.append(distrax.Categorical(
                    logits=nn.Dense(ad, kernel_init=constant(0.0), bias_init=lambda key, shape, dtype=jnp.float32: jnp.array([0.0, -1.5, -1.5, -1.5, -1.5], dtype=dtype))(actor_mean)
                ))
            else:
                pis.append(distrax.Categorical(
                    logits=nn.Dense(ad, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
                ))

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, tuple(pis), jnp.squeeze(critic, axis=-1)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def restore_params(path: Path):
    ckpt = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        str(path), args=ocp.args.StandardRestore()
    )
    return ckpt["params"], int(np.asarray(ckpt["epoch"]))


def wrap_pi_np(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def euler_to_quat_nb_np(roll, pitch, yaw):
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


def quat_conj_np(q):
    out = np.array(q, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_mul_np(q1, q2):
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_angle_deg_np(q_curr_bn, target_roll, target_pitch, target_yaw):
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
    nose = np.stack([cp * cy, cp * sy, -sp], axis=-1)
    right = np.stack([-cr * sy + sr * sp * cy, cr * cy + sr * sp * sy, sr * cp], axis=-1)
    return nose, right


def angle_deg_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def np_mean_std(x):
    arr = np.asarray(x, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


# ═══════════════════════════════════════════════════════════════════════
# Common task catalog
# ═══════════════════════════════════════════════════════════════════════

def task_catalog():
    return [
        # ── Basic tracking ──
        {"name": "level_flight", "category": "basic", "kind": "level", "max_steps": 500},
        {"name": "heading_p20", "category": "basic", "kind": "heading_step", "heading_deg": 20.0, "max_steps": 500},
        {"name": "heading_m20", "category": "basic", "kind": "heading_step", "heading_deg": -20.0, "max_steps": 500},
        {"name": "heading_p45", "category": "basic", "kind": "heading_step", "heading_deg": 45.0, "max_steps": 500},
        {"name": "heading_m45", "category": "basic", "kind": "heading_step", "heading_deg": -45.0, "max_steps": 500},
        {"name": "pitch_p10", "category": "basic", "kind": "pitch_step", "pitch_deg": 10.0, "max_steps": 500},
        {"name": "pitch_m10", "category": "basic", "kind": "pitch_step", "pitch_deg": -10.0, "max_steps": 500},
        # ── Vertical ──
        {"name": "pullup_15_R3000", "category": "vertical", "kind": "pullup", "angle_deg": 15.0, "radius": 3000.0, "max_steps": 500},
        {"name": "pullup_30_R5000", "category": "vertical", "kind": "pullup", "angle_deg": 30.0, "radius": 5000.0, "max_steps": 500},
        {"name": "pullup_30_R8000", "category": "vertical", "kind": "pullup", "angle_deg": 30.0, "radius": 8000.0, "max_steps": 500},
        # ── Horizontal maneuvers ──
        {"name": "circle_R5000_right", "category": "horizontal", "kind": "circle", "radius": 5000.0, "direction": 1.0, "max_steps": 600},
        {"name": "circle_R5000_left", "category": "horizontal", "kind": "circle", "radius": 5000.0, "direction": -1.0, "max_steps": 600},
        {"name": "circle_R3000_right", "category": "horizontal", "kind": "circle", "radius": 3000.0, "direction": 1.0, "max_steps": 600},
        {"name": "s_curve_A3000", "category": "horizontal", "kind": "s_curve", "amplitude": 3000.0, "max_steps": 600},
        {"name": "figure_eight_R5000", "category": "horizontal", "kind": "figure_eight", "radius": 5000.0, "max_steps": 600},
        # ── Altitude ──
        {"name": "climb_p1000m", "category": "altitude", "kind": "altitude_step", "alt_delta": 1000.0, "max_steps": 500},
        {"name": "descent_m1000m", "category": "altitude", "kind": "altitude_step", "alt_delta": -1000.0, "max_steps": 500},
    ]


def task_horizon(task):
    kind = task["kind"]
    if kind in ("pullup",):
        ramp_steps = max(3, int(np.ceil(np.deg2rad(task["angle_deg"]) * task["radius"] / 250.0 / DT_RL)))
        hold_steps = 85
        return ramp_steps + hold_steps, ramp_steps
    if kind in ("circle", "s_curve", "figure_eight"):
        return task.get("max_steps", 600), 1
    if kind == "altitude_step":
        return task.get("max_steps", 500), 1
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

    if kind == "heading_step":
        target_heading = jnp.asarray(wrap_PI(yaw0 + jnp.deg2rad(task["heading_deg"])))
    elif kind == "pitch_step":
        target_pitch = jnp.full(shape, jnp.deg2rad(task["pitch_deg"]))
    elif kind == "pullup":
        frac = jnp.clip(step / jnp.maximum(ramp_steps - 1, 1), 0.0, 1.0)
        frac = frac * frac * (3.0 - 2.0 * frac)
        target_pitch = pitch0 + jnp.deg2rad(task["angle_deg"]) * frac
    elif kind == "circle":
        omega = 250.0 / task["radius"]
        target_heading = jnp.asarray(wrap_PI(yaw0 + task.get("direction", 1.0) * omega * t))
    elif kind == "s_curve":
        period = 85.0
        amp_heading = jnp.deg2rad(32.0)
        target_heading = jnp.asarray(wrap_PI(yaw0 + amp_heading * jnp.sin(2.0 * jnp.pi * t / period)))
    elif kind == "figure_eight":
        period = 120.0
        amp_heading = jnp.deg2rad(42.0)
        target_heading = jnp.asarray(wrap_PI(yaw0 + amp_heading * jnp.sin(4.0 * jnp.pi * t / period)))
    elif kind == "altitude_step":
        target_alt = alt0 + task["alt_delta"]
        alt_err = target_alt - state.plane_state.altitude
        target_pitch = jnp.clip(jnp.arctan2(alt_err, 6500.0), jnp.deg2rad(-10.0), jnp.deg2rad(10.0))

    return target_heading, target_pitch, target_roll, target_vt


# ═══════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=20260520)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--euler-ckpt", type=Path, default=EULER_CKPT)
    parser.add_argument("--quat-ckpt", type=Path, default=QUAT_CKPT)
    parser.add_argument("--vertical-ckpt", type=Path, default=VERTICAL_CKPT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    num_seeds = args.seeds

    # ── Load both envs and policies ──
    euler_env_params = EulerParams()
    euler_env = EulerEnv(euler_env_params)
    euler_agent = euler_env.agents[0]

    quat_env_params = QuatParams()
    quat_env = QuatEnv(quat_env_params)
    quat_agent = quat_env.agents[0]

    # Same action dims for both
    action_dim = [31, 41, 41, 41, 5]

    # Euler network (obs_size=22)
    euler_network = ActorCriticRNN(action_dim, config=NET_CONFIG)
    euler_params, euler_epoch = restore_params(args.euler_ckpt.resolve())

    # Quaternion network (obs_size=21)
    quat_network = ActorCriticRNN(action_dim, config=NET_CONFIG)
    quat_params, quat_epoch = restore_params(args.quat_ckpt.resolve())

    # Vertical energy network (obs_size=21, same as quat)
    vert_network = ActorCriticRNN(action_dim, config=NET_CONFIG)
    vert_params, vert_epoch = restore_params(args.vertical_ckpt.resolve())

    print(f"devices: {jax.devices()}")
    print(f"Euler epoch: {euler_epoch}, Quat epoch: {quat_epoch}, Vertical epoch: {vert_epoch}")

    # ── JIT step functions ──
    @jax.jit
    def euler_step(params, hstate, state, done, key, th, tp, tr, tv):
        state = state.replace(
            target_heading=th, target_pitch=tp, target_roll=tr, target_vt=tv,
            last_check_time=state.time,
        )
        obs_dict = jax.vmap(euler_env._get_obs, in_axes=(0, None))(state, euler_env_params)
        obs = obs_dict[euler_agent]
        hstate, pi, _ = euler_network.apply(params, hstate, (obs[None, :, :], done[None, :]))
        action = jnp.stack([p.mode()[0] for p in pi], axis=-1).astype(jnp.int32)
        step_keys = jax.random.split(key, num_seeds)
        obs_next, state_next, reward, done_dict, info = jax.vmap(euler_env.step, in_axes=(0, 0, 0, None))(
            step_keys, state, {euler_agent: action}, euler_env_params
        )
        del obs_next
        return state_next, done_dict[euler_agent], hstate, reward[euler_agent], info, action

    @jax.jit
    def quat_step(params, hstate, state, done, key, th, tp, tr, tv):
        state = state.replace(
            target_heading=th, target_pitch=tp, target_roll=tr, target_vt=tv,
            last_check_time=state.time,
        )
        obs_dict = jax.vmap(quat_env._get_obs, in_axes=(0, None))(state, quat_env_params)
        obs = obs_dict[quat_agent]
        hstate, pi, _ = quat_network.apply(params, hstate, (obs[None, :, :], done[None, :]))
        action = jnp.stack([p.mode()[0] for p in pi], axis=-1).astype(jnp.int32)
        step_keys = jax.random.split(key, num_seeds)
        obs_next, state_next, reward, done_dict, info = jax.vmap(quat_env.step, in_axes=(0, 0, 0, None))(
            step_keys, state, {quat_agent: action}, quat_env_params
        )
        del obs_next
        return state_next, done_dict[quat_agent], hstate, reward[quat_agent], info, action

    # ── Evaluate ──
    policies = {
        "Euler_epoch300": (euler_params, euler_env, euler_env_params, euler_agent, euler_step, euler_network),
        "Quat_epoch600": (quat_params, quat_env, quat_env_params, quat_agent, quat_step, quat_network),
        "Vertical_epoch619": (vert_params, quat_env, quat_env_params, quat_agent, quat_step, vert_network),
    }

    all_rows = []
    summary_rows = []
    tasks = task_catalog()

    for policy_name, (net_params, env, env_params, agent, step_fn, network) in policies.items():
        print(f"\n{'='*60}\nEvaluating: {policy_name}\n{'='*60}")
        for task_idx, task in enumerate(tasks):
            max_steps, ramp_steps = task_horizon(task)
            reset_keys = jax.random.split(jax.random.PRNGKey(args.seed_base + task_idx), num_seeds)
            _, state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
            init = (state.plane_state.yaw, state.plane_state.pitch, state.plane_state.altitude)
            extra_kwargs = {}
            if hasattr(state, "task_mode"):
                extra_kwargs["task_mode"] = jnp.zeros_like(state.task_mode)
            if hasattr(state, "task_duration_steps"):
                extra_kwargs["task_duration_steps"] = jnp.full_like(state.task_duration_steps, 10000.0)
            if hasattr(state, "task_start_heading"):
                extra_kwargs["task_start_heading"] = state.plane_state.yaw
                extra_kwargs["task_start_pitch"] = state.plane_state.pitch
                extra_kwargs["task_start_roll"] = state.plane_state.roll
                extra_kwargs["task_start_vt"] = state.plane_state.vt
                extra_kwargs["task_start_altitude"] = state.plane_state.altitude
                extra_kwargs["task_start_energy"] = 0.5 * state.plane_state.vt * state.plane_state.vt + G * state.plane_state.altitude
            if hasattr(state, "last_check_time"):
                extra_kwargs["last_check_time"] = state.time
            state = state.replace(**extra_kwargs)
            hstate = ScannedRNN.initialize_carry(num_seeds, NET_CONFIG["GRU_HIDDEN_DIM"])
            done = jnp.zeros((num_seeds,), dtype=jnp.bool_)

            # ── Accumulators ──
            vt_min = np.full(num_seeds, np.inf)
            vt_sum = np.zeros(num_seeds)
            energy0 = np.asarray((0.5 * state.plane_state.vt * state.plane_state.vt + G * state.plane_state.altitude)[:, 0])
            energy_min = np.full(num_seeds, np.inf)
            alt0 = np.asarray(state.plane_state.altitude[:, 0])
            alt_final = alt0.copy()
            alt_min = alt0.copy()
            alt_max = alt0.copy()
            alpha_max = np.zeros(num_seeds)
            gmax = np.zeros(num_seeds)
            heading_err_sum = np.zeros(num_seeds)
            pitch_err_sum = np.zeros(num_seeds)
            q_error_sum = np.zeros(num_seeds)
            velocity_tangent_err_sum = np.zeros(num_seeds)
            nose_tangent_err_sum = np.zeros(num_seeds)
            nose_velocity_err_sum = np.zeros(num_seeds)
            wing_plane_err_sum = np.zeros(num_seeds)
            prev_north = np.asarray(state.plane_state.north[:, 0]).copy()
            prev_east = np.asarray(state.plane_state.east[:, 0]).copy()
            prev_altitude = alt0.copy()
            active_count = np.zeros(num_seeds)
            stall_count = np.zeros(num_seeds)
            g_violation_count = np.zeros(num_seeds)
            reason = np.array(["none"] * num_seeds, dtype=object)

            for step in range(max_steps):
                th, tp, tr, tv = build_target(task, step, ramp_steps, init, state)
                key = jax.random.PRNGKey(args.seed_base + 100000 + task_idx * 1000 + step)
                state, done_step, hstate, reward, info, action = step_fn(
                    net_params, hstate, state, done, key, th, tp, tr, tv
                )
                active = ~np.asarray(done)
                if not active.any():
                    break

                # ── Use plane_state directly for all metrics (works for both envs) ──
                vt = np.asarray(state.plane_state.vt)[:, 0]
                alt = np.asarray(state.plane_state.altitude)[:, 0]
                energy = 0.5 * vt * vt + G * alt
                pitch_deg = np.degrees(np.asarray(state.plane_state.pitch)[:, 0])
                yaw_deg = np.degrees(np.asarray(state.plane_state.yaw)[:, 0])
                target_pitch_deg = np.degrees(np.asarray(tp)[:, 0])
                target_heading_rad = np.asarray(th)[:, 0]
                pitch_err = np.abs(pitch_deg - target_pitch_deg)
                heading_err = np.abs(wrap_pi_np(np.radians(yaw_deg) - target_heading_rad)) * 180.0 / np.pi
                alpha_deg = np.degrees(np.asarray(state.plane_state.alpha)[:, 0])
                alpha = np.abs(alpha_deg)
                # G-load from nz (body-frame vertical load factor)
                g_load = np.abs(np.asarray(state.plane_state.az)[:, 0])

                # Quaternion error
                try:
                    q_vals = np.stack([
                        np.asarray(state.plane_state.q0)[:, 0],
                        np.asarray(state.plane_state.q1)[:, 0],
                        np.asarray(state.plane_state.q2)[:, 0],
                        np.asarray(state.plane_state.q3)[:, 0],
                    ], axis=-1)
                    tr_rad = np.asarray(tr)[:, 0]
                    tp_rad = np.asarray(tp)[:, 0]
                    th_rad = np.asarray(th)[:, 0]
                    q_error = quat_angle_deg_np(q_vals, tr_rad, tp_rad, th_rad)
                except Exception:
                    q_error = np.zeros(num_seeds)

                # Geometry errors
                actual_roll_rad = np.asarray(state.plane_state.roll)[:, 0]
                actual_pitch_rad = np.asarray(state.plane_state.pitch)[:, 0]
                actual_yaw_rad = np.asarray(state.plane_state.yaw)[:, 0]
                target_roll_rad = np.asarray(tr)[:, 0]
                target_pitch_rad = np.asarray(tp)[:, 0]
                actual_nose, actual_right = body_axes_from_euler_np(actual_roll_rad, actual_pitch_rad, actual_yaw_rad)
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

                vt_min[active] = np.minimum(vt_min[active], vt[active])
                vt_sum[active] += vt[active]
                energy_min[active] = np.minimum(energy_min[active], energy[active])
                alt_final[active] = alt[active]
                alt_min[active] = np.minimum(alt_min[active], alt[active])
                alt_max[active] = np.maximum(alt_max[active], alt[active])
                alpha_max[active] = np.maximum(alpha_max[active], alpha[active])
                gmax[active] = np.maximum(gmax[active], g_load[active])
                heading_err_sum[active] += heading_err[active]
                pitch_err_sum[active] += pitch_err[active]
                q_error_sum[active] += q_error[active]
                velocity_tangent_err_sum[active] += vel_tan_err[active]
                nose_tangent_err_sum[active] += nose_tan_err[active]
                nose_velocity_err_sum[active] += nose_vel_err[active]
                wing_plane_err_sum[active] += wing_plane_err[active]
                stall_count[active] += (alpha[active] > 30.0).astype(float)
                g_violation_count[active] += (g_load[active] > 10.0).astype(float)
                active_count[active] += 1.0
                prev_north[active] = north[active]
                prev_east[active] = east[active]
                prev_altitude[active] = alt[active]

                done_np = np.asarray(done_step)
                newly_done = active & done_np
                for i in np.where(newly_done)[0]:
                    if g_load[i] > 10.0:
                        reason[i] = "overload"
                    elif alt[i] <= 0.0:
                        reason[i] = "crash"
                    else:
                        reason[i] = "env_done"
                done = jnp.asarray(np.asarray(done) | done_np)

            # ── Aggregate ──
            active_count = np.maximum(active_count, 1.0)
            survivors = np.array([r == "none" for r in reason])
            crashed = np.array([r in ("crash", "overload", "env_done") for r in reason])
            survival_rate = float(survivors.mean())

            row = {
                "policy": policy_name,
                "task": task["name"],
                "category": task["category"],
                "survival_rate": survival_rate,
                "heading_err_mean": float(heading_err_sum.mean() / active_count.mean()),
                "pitch_err_mean": float(pitch_err_sum.mean() / active_count.mean()),
                "q_error_mean_deg": float(q_error_sum.mean() / active_count.mean()),
                "vt_min_mean": float(vt_min.mean()),
                "energy_loss_pct": float((1.0 - energy_min / energy0).mean() * 100),
                "alpha_max_mean": float(alpha_max.mean()),
                "gmax_mean": float(gmax.mean()),
                "stall_rate_per_step": float(stall_count.mean() / active_count.mean()),
                "g_violation_rate_per_step": float(g_violation_count.mean() / active_count.mean()),
                "vel_tan_err_mean": float(velocity_tangent_err_sum.mean() / active_count.mean()),
                "nose_tan_err_mean": float(nose_tangent_err_sum.mean() / active_count.mean()),
                "nose_vel_err_mean": float(nose_velocity_err_sum.mean() / active_count.mean()),
                "wing_plane_err_mean": float(wing_plane_err_sum.mean() / active_count.mean()),
                "mean_ep_length": float(active_count.mean()),
                "termination": Counter(reason).most_common(1)[0][0] if len(reason) > 0 else "unknown",
            }
            all_rows.append(row)
            print(f"  {task['name']:25s}  surv={survival_rate:.2f}  heading_err={row['heading_err_mean']:.1f}°  "
                  f"q_err={row['q_error_mean_deg']:.1f}°  stall={row['stall_rate_per_step']:.3f}  "
                  f"G_vio={row['g_violation_rate_per_step']:.4f}  term={row['termination']}")

    # ── Write CSV ──
    csv_path = args.out_dir / "euler_vs_quat_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV written: {csv_path}")

    # ── Build comparison table ──
    # Group by task, compare Euler vs Quat vs Vertical
    euler_by_task = {r["task"]: r for r in all_rows if r["policy"] == "Euler_epoch300"}
    quat_by_task = {r["task"]: r for r in all_rows if r["policy"] == "Quat_epoch600"}
    vert_by_task = {r["task"]: r for r in all_rows if r["policy"] == "Vertical_epoch619"}

    comparison_path = args.out_dir / "comparison_by_task.csv"
    with open(comparison_path, "w", newline="") as f:
        fields = ["task", "category",
                   "euler_survival", "quat_survival", "vert_survival",
                   "euler_q_err", "quat_q_err", "vert_q_err",
                   "euler_heading_err", "quat_heading_err",
                   "euler_stall_rate", "quat_stall_rate",
                   "euler_G_vio", "quat_G_vio",
                   "euler_vel_tan", "quat_vel_tan",
                   "euler_nose_tan", "quat_nose_tan",
                   "euler_wing_plane", "quat_wing_plane"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task_name in euler_by_task:
            er = euler_by_task[task_name]
            qr = quat_by_task.get(task_name, {})
            vr = vert_by_task.get(task_name, {})
            writer.writerow({
                "task": task_name,
                "category": er["category"],
                "euler_survival": er.get("survival_rate", ""),
                "quat_survival": qr.get("survival_rate", ""),
                "vert_survival": vr.get("survival_rate", ""),
                "euler_q_err": er.get("q_error_mean_deg", ""),
                "quat_q_err": qr.get("q_error_mean_deg", ""),
                "vert_q_err": vr.get("q_error_mean_deg", ""),
                "euler_heading_err": er.get("heading_err_mean", ""),
                "quat_heading_err": qr.get("heading_err_mean", ""),
                "euler_stall_rate": er.get("stall_rate_per_step", ""),
                "quat_stall_rate": qr.get("stall_rate_per_step", ""),
                "euler_G_vio": er.get("g_violation_rate_per_step", ""),
                "quat_G_vio": qr.get("g_violation_rate_per_step", ""),
                "euler_vel_tan": er.get("vel_tan_err_mean", ""),
                "quat_vel_tan": qr.get("vel_tan_err_mean", ""),
                "euler_nose_tan": er.get("nose_tan_err_mean", ""),
                "quat_nose_tan": qr.get("nose_tan_err_mean", ""),
                "euler_wing_plane": er.get("wing_plane_err_mean", ""),
                "quat_wing_plane": qr.get("wing_plane_err_mean", ""),
            })
    print(f"Comparison CSV written: {comparison_path}")

    # ── Print LaTeX-ready summary table ──
    print("\n" + "="*80)
    print("LaTeX Table: Euler vs Quaternion Comparison")
    print("="*80)
    print(r"\begin{table}[t]")
    print(r"  \caption{Euler vs. quaternion target encoding: tracking performance comparison.}")
    print(r"  \label{tab:euler-vs-quat}")
    print(r"  \centering")
    print(r"  \begin{tabular}{lcccccc}")
    print(r"    \toprule")
    print(r"    \textbf{Task} & \textbf{Survival} & \textbf{Att. Error} & \textbf{Heading Err} & \textbf{Stall Rate} & \textbf{G Vio} & \textbf{Grade} \\")
    print(r"    \midrule")
    for task in tasks:
        tn = task["name"]
        er = euler_by_task.get(tn, {})
        qr = quat_by_task.get(tn, {})
        e_surv = er.get("survival_rate", 0)
        q_surv = qr.get("survival_rate", 0)
        e_q = er.get("q_error_mean_deg", 0)
        q_q = qr.get("q_error_mean_deg", 0)
        e_h = er.get("heading_err_mean", 0)
        q_h = qr.get("heading_err_mean", 0)
        e_stall = er.get("stall_rate_per_step", 0)
        q_stall = qr.get("stall_rate_per_step", 0)
        e_g = er.get("g_violation_rate_per_step", 0)
        q_g = qr.get("g_violation_rate_per_step", 0)
        # Determine winner
        winner = "Q" if (q_surv >= e_surv and q_q <= e_q) else ("E" if e_surv > q_surv else "≈")
        print(f"    {tn:25s} & Euler: {e_surv:.0%} / Quat: {q_surv:.0%} & "
              f"E: {e_q:.1f}° / Q: {q_q:.1f}° & "
              f"E: {e_h:.1f}° / Q: {q_h:.1f}° & "
              f"E: {e_stall:.3f} / Q: {q_stall:.3f} & "
              f"E: {e_g:.4f} / Q: {q_g:.4f} & {winner} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")

    print("\nDone!")


if __name__ == "__main__":
    main()
