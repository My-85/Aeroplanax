#!/usr/bin/env python3
"""Zero-shot sim-to-real sensitivity evaluation for CoRL 2026 paper.

Loads the frozen energy-aware PPO checkpoint and evaluates the policy under
multiple perturbation settings on S-curve and 90deg vertical pull-up tasks.

Perturbations:
  1. Nominal (no perturbation)
  2. Aero coefficient scaling: 0.95, 1.05, 0.90, 1.10
  3. Mass/Inertia scaling: 0.95, 1.05
  4. Actuator delay: 1, 2, 3 control steps
  5. Wind disturbance: mild 5 m/s, moderate 10 m/s
  6. Observation noise: Gaussian std=0.01 on normalized obs

Usage:
  python scripts/eval_sim2real_sensitivity.py \
    --checkpoint results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619 \
    --tasks s_curve,pullup_90 \
    --num_seeds 10 \
    --output_dir results/sim2real_sensitivity
"""

import argparse
import csv
import functools
import json
import os
import sys
from collections import Counter
from datetime import datetime
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PLANAX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLANAX_ROOT))

from envs.aeroplanax_heading_pitch_V_quaternion_version_vertical_energy import (
    AeroPlanaxHeading_Pitch_V_Env,
    Heading_Pitch_V_TaskParams,
)
from envs.utils.utils import wrap_PI

# ── Constants ──────────────────────────────────────────────────────────
DEFAULT_CKPT = (
    PLANAX_ROOT
    / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
)
NET_CONFIG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
DT_RL = 10.0 / 50.0  # agent_interaction_steps / sim_freq
G = 9.80665
ACTION_DIMS = [31, 41, 41, 41, 5]  # throttle, elevator, aileron, rudder, speed_brake

# ── NumPy helpers ──────────────────────────────────────────────────────
def euler_to_quat_nb_np(roll, pitch, yaw):
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.stack(
        [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
         cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy], axis=-1)

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
        w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1)

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
    nose = np.stack([cp*cy, cp*sy, -sp], axis=-1)
    right = np.stack([-cr*sy + sr*sp*cy, cr*cy + sr*sp*sy, sr*cp], axis=-1)
    return nose, right

def angle_deg_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(a*b, axis=-1), -1.0, 1.0)))

def wrap_pi_np(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def np_mean_std(x):
    arr = np.asarray(x, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))

# ── Network definitions (mirrors eval_vertical_energy_checkpoints.py) ──
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan, variable_broadcast="params", in_axes=0, out_axes=0,
        split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state)
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
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))
        fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        fc2 = nn.LayerNorm()(fc2)
        fc2 = activation(fc2)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2)
        actor_mean = activation(actor_mean)
        pi_throttle = distrax.Categorical(logits=nn.Dense(
            self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(logits=nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_aileron = distrax.Categorical(logits=nn.Dense(
            self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_rudder = distrax.Categorical(logits=nn.Dense(
            self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_speed_brake = distrax.Categorical(logits=nn.Dense(
            self.action_dim[4],
            kernel_init=constant(0.0),
            bias_init=lambda key, shape, dtype=jnp.float32: jnp.array(
                [0.0, -1.5, -1.5, -1.5, -1.5], dtype=dtype))(actor_mean))
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return (hidden,
                (pi_throttle, pi_elevator, pi_aileron, pi_rudder, pi_speed_brake),
                jnp.squeeze(critic, axis=-1))

def restore_params(path: Path):
    ckpt = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        str(path), args=ocp.args.StandardRestore())
    return ckpt["params"], int(np.asarray(ckpt["epoch"]))

# ── Task definitions ───────────────────────────────────────────────────
def task_catalog(tasks_str: str):
    """Return list of task dicts for the requested tasks."""
    available = {
        "s_curve": {
            "name": "s_curve_A3000", "kind": "s_curve",
            "description": "S-curve (heading oscillation, amp=32deg, period=85s)",
            "max_steps": 240, "ramp_steps": 1,
        },
        "pullup_90": {
            "name": "90_vertical_pullup_R10000", "kind": "vertical_arc",
            "angle_deg": 90.0, "radius": 10000.0,
            "description": "90deg vertical pull-up, R=10000m",
            "max_steps": 250, "ramp_steps": 0,
        },
    }
    requested = [t.strip() for t in tasks_str.split(",")]
    tasks = []
    for key in requested:
        if key in available:
            t = available[key]
            # compute horizon
            if t["kind"] in ("vertical_arc", "pullup"):
                ramp = max(3, int(np.ceil(np.deg2rad(t["angle_deg"]) * t["radius"] / 250.0 / DT_RL)))
                hold = 85
                t["max_steps"] = ramp + hold
                t["ramp_steps"] = ramp
            tasks.append(t)
        else:
            print(f"Warning: unknown task '{key}', available: {list(available.keys())}")
    return tasks

def build_target(task, step, ramp_steps, init, state):
    yaw0, pitch0, alt0 = init
    shape = yaw0.shape
    kind = task["kind"]
    target_heading = yaw0
    target_pitch = jnp.zeros(shape)
    target_roll = jnp.zeros(shape)
    target_vt = jnp.full(shape, 250.0)
    t = step * DT_RL

    if kind == "s_curve":
        period = 85.0
        amp_heading = jnp.deg2rad(32.0)
        target_heading = wrap_PI(yaw0 + amp_heading * jnp.sin(2.0 * jnp.pi * t / period))
    elif kind in ("pullup", "vertical_arc"):
        frac = jnp.clip(step / jnp.maximum(ramp_steps - 1, 1), 0.0, 1.0)
        frac = frac * frac * (3.0 - 2.0 * frac)
        target_pitch = pitch0 + jnp.deg2rad(task["angle_deg"]) * frac

    return target_heading, target_pitch, target_roll, target_vt

# ── Perturbation definitions ───────────────────────────────────────────
def get_perturbations():
    """Return ordered list of (setting_name, perturbation_dict) tuples."""
    settings = [
        ("nominal", {
            "description": "Nominal (no perturbation)",
            "aero_scale": 1.0, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 0,
        }),
        ("aero_095", {
            "description": "Aero coeff ×0.95",
            "aero_scale": 0.95, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 0,
        }),
        ("aero_105", {
            "description": "Aero coeff ×1.05",
            "aero_scale": 1.05, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 0,
        }),
        ("aero_090", {
            "description": "Aero coeff ×0.90",
            "aero_scale": 0.90, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 0,
        }),
        ("aero_110", {
            "description": "Aero coeff ×1.10",
            "aero_scale": 1.10, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 0,
        }),
        ("mass_095", {
            "description": "Mass/Inertia ×0.95",
            "aero_scale": 1.0, "mass_scale": 0.95, "inertia_scale": 0.95,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 0,
        }),
        ("mass_105", {
            "description": "Mass/Inertia ×1.05",
            "aero_scale": 1.0, "mass_scale": 1.05, "inertia_scale": 1.05,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 0,
        }),
        ("delay_1", {
            "description": "Actuator delay 1 step (0.02s)",
            "aero_scale": 1.0, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 1,
        }),
        ("delay_2", {
            "description": "Actuator delay 2 steps (0.04s)",
            "aero_scale": 1.0, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 2,
        }),
        ("delay_3", {
            "description": "Actuator delay 3 steps (0.06s)",
            "aero_scale": 1.0, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 0.0, "obs_noise_std": 0.0, "actuator_delay": 3,
        }),
        ("wind_5ms", {
            "description": "Wind 5 m/s (mild)",
            "aero_scale": 1.0, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 5.0, "obs_noise_std": 0.0, "actuator_delay": 0,
        }),
        ("wind_10ms", {
            "description": "Wind 10 m/s (moderate)",
            "aero_scale": 1.0, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 10.0, "obs_noise_std": 0.0, "actuator_delay": 0,
        }),
        ("obs_noise_001", {
            "description": "Obs noise σ=0.01",
            "aero_scale": 1.0, "mass_scale": 1.0, "inertia_scale": 1.0,
            "wind_mag": 0.0, "obs_noise_std": 0.01, "actuator_delay": 0,
        }),
    ]
    return settings

# ── Per-step wind direction computation ────────────────────────────────
def compute_body_wind(wind_mag, aircraft_yaw):
    """Compute body-frame wind from NED wind magnitude.
    Wind blows from North, converted to body frame using aircraft yaw.
    Simplified: headwind component = -wind_mag * cos(yaw), crosswind = wind_mag * sin(yaw).
    """
    # Wind in NED: blowing towards South (from North)
    wind_n = -wind_mag
    wind_e = 0.0
    wind_d = 0.0
    # Rotate to body frame (simplified: only yaw rotation)
    cy, sy = jnp.cos(aircraft_yaw), jnp.sin(aircraft_yaw)
    wind_body_u = wind_n * cy + wind_e * sy
    wind_body_v = -wind_n * sy + wind_e * cy
    wind_body_w = -wind_d  # up-positive in body frame → same as NED down
    return wind_body_u, wind_body_v, wind_body_w

# ── Evaluation ─────────────────────────────────────────────────────────
def run_eval(args):
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PLANAX_ROOT / out_dir
    out_dir = out_dir.resolve() / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PLANAX_ROOT / ckpt_path
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    net_params, epoch = restore_params(ckpt_path)
    print(f"Loaded checkpoint epoch {epoch} from {ckpt_path}")

    # Setup env and network
    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)
    agent = env.agents[0]
    network = ActorCriticRNN(ACTION_DIMS, config=NET_CONFIG)

    # Get tasks and perturbations
    tasks = task_catalog(args.tasks)
    perturbations = get_perturbations()
    num_seeds = args.num_seeds

    print(f"Tasks: {[t['name'] for t in tasks]}")
    print(f"Perturbations: {[p[0] for p in perturbations]}")
    print(f"Seeds: {num_seeds}")
    print(f"Devices: {jax.devices()}")

    # JIT-compiled batched step function (perturbation fields already on state)
    @jax.jit
    def batched_step(net_params, hstate, state, done, key,
                     target_heading, target_pitch, target_roll, target_vt,
                     obs_noise_key, obs_noise_std):
        # Set targets
        state = state.replace(
            target_heading=target_heading,
            target_pitch=target_pitch,
            target_roll=target_roll,
            target_vt=target_vt,
            task_mode=jnp.zeros_like(state.task_mode),
            task_duration_steps=jnp.full_like(state.task_duration_steps, 10000.0),
            last_check_time=state.time,
        )
        # Get observation
        obs_dict = jax.vmap(env._get_obs, in_axes=(0, None))(state, env_params)
        obs = obs_dict[agent]
        # Add observation noise (before network forward pass)
        noise = jax.random.normal(obs_noise_key, obs.shape) * obs_noise_std
        obs = obs + noise
        # Network forward
        hstate, pi, _ = network.apply(net_params, hstate, (obs[None, :, :], done[None, :]))
        action = jnp.stack([p.mode()[0] for p in pi], axis=-1).astype(jnp.int32)
        # Step env
        step_keys = jax.random.split(key, num_seeds)
        obs_next, state_next, reward, done_dict, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None))(step_keys, state, {agent: action}, env_params)
        del obs_next
        return state_next, done_dict[agent], hstate, reward[agent], info, action

    # ── Run all combinations ──
    all_rows = []      # per-seed metrics
    all_summaries = []  # aggregated summaries

    for set_idx, (setting_name, pert) in enumerate(perturbations):
        aero_scale_val = pert["aero_scale"]
        mass_scale_val = pert["mass_scale"]
        inertia_scale_val = pert["inertia_scale"]
        wind_mag = pert["wind_mag"]
        obs_noise_std = pert["obs_noise_std"]
        delay_steps = pert["actuator_delay"]

        for task_idx, task in enumerate(tasks):
            max_steps = task["max_steps"]
            ramp_steps = task["ramp_steps"]
            print(f"\n{'='*60}")
            print(f"[{setting_name}] {task['name']} (max_steps={max_steps})")
            print(f"{'='*60}")

            # Reset env
            seed_base = args.seed_base + task_idx * 1000 + set_idx * 100
            reset_keys = jax.random.split(jax.random.PRNGKey(seed_base), num_seeds)
            _, state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
            init = (state.plane_state.yaw, state.plane_state.pitch, state.plane_state.altitude)
            # Set perturbation fields with proper batch dimensions
            state = state.replace(
                plane_state=state.plane_state.replace(
                    aero_scale=jnp.full_like(state.plane_state.vt, aero_scale_val),
                    mass_scale=jnp.full_like(state.plane_state.vt, mass_scale_val),
                    inertia_scale=jnp.full_like(state.plane_state.vt, inertia_scale_val),
                    wind_body_u=jnp.zeros_like(state.plane_state.vt),
                    wind_body_v=jnp.zeros_like(state.plane_state.vt),
                    wind_body_w=jnp.zeros_like(state.plane_state.vt),
                ),
                task_start_heading=state.plane_state.yaw,
                task_start_pitch=state.plane_state.pitch,
                task_start_roll=state.plane_state.roll,
                task_start_vt=state.plane_state.vt,
                task_start_altitude=state.plane_state.altitude,
                task_start_energy=0.5*state.plane_state.vt*state.plane_state.vt
                                   + G*state.plane_state.altitude,
                task_mode=jnp.zeros_like(state.task_mode),
                task_duration_steps=jnp.full_like(state.task_duration_steps, 10000.0),
                last_check_time=state.time,
            )
            hstate = ScannedRNN.initialize_carry(num_seeds, NET_CONFIG["GRU_HIDDEN_DIM"])
            done = jnp.zeros((num_seeds,), dtype=jnp.bool_)

            # Per-seed accumulators
            vt_min = np.full(num_seeds, np.inf)
            energy_min = np.full(num_seeds, np.inf)
            energy0 = np.asarray((
                0.5*state.plane_state.vt*state.plane_state.vt
                + G*state.plane_state.altitude)[:, 0])
            alt0 = np.asarray(state.plane_state.altitude[:, 0])
            alt_final = alt0.copy()
            prev_north = np.asarray(state.plane_state.north[:, 0]).copy()
            prev_east = np.asarray(state.plane_state.east[:, 0]).copy()
            prev_altitude = alt0.copy()
            alpha_max = np.zeros(num_seeds)
            gmax = np.zeros(num_seeds)
            pitch_err_sum = np.zeros(num_seeds)
            heading_err_sum = np.zeros(num_seeds)
            velocity_tangent_err_sum = np.zeros(num_seeds)
            nose_tangent_err_sum = np.zeros(num_seeds)
            nose_velocity_err_sum = np.zeros(num_seeds)
            wing_plane_err_sum = np.zeros(num_seeds)
            q_error_norm_sum = np.zeros(num_seeds)
            tracking_err_values = [[] for _ in range(num_seeds)]
            active_count = np.zeros(num_seeds)
            reason = np.array(["none"] * num_seeds, dtype=object)

            # Actuator delay buffer (initialize with trim-like neutral action)
            if delay_steps > 0:
                # neutral action: throttle=15 (mid), surfaces centered, speed_brake=0
                neutral = np.full((num_seeds, 5), [15, 20, 20, 20, 0], dtype=np.int32)
                action_buffer = [neutral] * (delay_steps + 1)

            for step in range(max_steps):
                th, tp, tr, tv = build_target(task, step, ramp_steps, init, state)
                key = jax.random.PRNGKey(seed_base + 100000 + step)

                # Update wind_body fields on state (wind direction depends on yaw)
                if wind_mag > 0.0:
                    yaw_arr = jnp.asarray(state.plane_state.yaw)
                    cy = jnp.cos(yaw_arr)
                    sy = jnp.sin(yaw_arr)
                    wbu = -wind_mag * cy
                    wbv = wind_mag * sy
                    wbw = jnp.zeros_like(yaw_arr)
                    state = state.replace(
                        plane_state=state.plane_state.replace(
                            wind_body_u=wbu, wind_body_v=wbv, wind_body_w=wbw))

                # Observation noise key
                obs_noise_key = jax.random.PRNGKey(seed_base + 200000 + step)

                if delay_steps > 0:
                    # Set targets
                    state = state.replace(
                        target_heading=th, target_pitch=tp,
                        target_roll=tr, target_vt=tv,
                        task_mode=jnp.zeros_like(state.task_mode),
                        task_duration_steps=jnp.full_like(
                            state.task_duration_steps, 10000.0),
                        last_check_time=state.time,
                    )
                    obs_dict = jax.vmap(env._get_obs, in_axes=(0, None))(
                        state, env_params)
                    obs = obs_dict[agent]
                    obs_noise = jax.random.normal(obs_noise_key, obs.shape) * obs_noise_std
                    obs = obs + obs_noise
                    hstate, pi, _ = network.apply(
                        net_params, hstate, (obs[None, :, :], done[None, :]))
                    new_action = jnp.stack([p.mode()[0] for p in pi], axis=-1).astype(jnp.int32)

                    # Feed into delay buffer, get delayed action
                    action_buffer.append(np.asarray(new_action))
                    if len(action_buffer) > delay_steps + 1:
                        action_buffer.pop(0)
                    delayed_action = action_buffer[0]
                    action_jnp = jnp.asarray(delayed_action)

                    # Step env with delayed action
                    step_keys = jax.random.split(key, num_seeds)
                    _, state_next, reward, done_dict, info = jax.vmap(
                        env.step, in_axes=(0, 0, 0, None))(
                        step_keys, state, {agent: action_jnp}, env_params)
                    state = state_next
                    done_step = done_dict[agent]
                    action = action_jnp
                else:
                    # Standard single step
                    state, done_step, hstate, reward, info, action = batched_step(
                        net_params, hstate, state, done, key,
                        th, tp, tr, tv,
                        obs_noise_key, obs_noise_std,
                    )

                active = ~np.asarray(done)
                if not active.any():
                    break

                # ── Collect per-step metrics ──
                vt = np.asarray(info["vt"])[:, 0]
                alt = np.asarray(info["altitude"])[:, 0]
                energy = np.asarray(info["energy_proxy"])[:, 0]
                pitch_deg = np.asarray(info["pitch_deg"])[:, 0]
                target_pitch_deg = np.asarray(info["target_pitch_deg"])[:, 0]
                target_heading = np.asarray(th)[:, 0]
                yaw = np.asarray(state.plane_state.yaw)[:, 0]
                pitch_err = np.abs(pitch_deg - target_pitch_deg)
                heading_err = np.abs(np.asarray(wrap_PI(yaw - target_heading))) * 180.0 / np.pi
                alpha_signed = np.asarray(info["alpha_deg"])[:, 0]
                alpha = np.abs(alpha_signed)
                g_load = np.asarray(info["g_load_max"])[:, 0]
                kind = task["kind"]

                # Quaternion / geometry errors
                q_curr_np = np.stack([
                    np.asarray(state.plane_state.q0)[:, 0],
                    np.asarray(state.plane_state.q1)[:, 0],
                    np.asarray(state.plane_state.q2)[:, 0],
                    np.asarray(state.plane_state.q3)[:, 0],
                ], axis=-1)
                target_roll_rad = np.asarray(tr)[:, 0]
                target_pitch_rad = np.asarray(tp)[:, 0]
                target_heading_rad = np.asarray(th)[:, 0]
                actual_roll_rad = np.asarray(state.plane_state.roll)[:, 0]
                actual_pitch_rad = np.asarray(state.plane_state.pitch)[:, 0]
                actual_yaw_rad = np.asarray(state.plane_state.yaw)[:, 0]
                q_error_norm = quat_angle_deg_np(
                    q_curr_np, target_roll_rad, target_pitch_rad, target_heading_rad)
                actual_nose, actual_right = body_axes_from_euler_np(
                    actual_roll_rad, actual_pitch_rad, actual_yaw_rad)
                target_nose, target_right = body_axes_from_euler_np(
                    target_roll_rad, target_pitch_rad, target_heading_rad)
                north = np.asarray(state.plane_state.north)[:, 0]
                east = np.asarray(state.plane_state.east)[:, 0]
                displacement_n = north - prev_north
                displacement_e = east - prev_east
                displacement_d = -(alt - prev_altitude)
                velocity_n = np.stack([displacement_n, displacement_e, displacement_d], axis=-1)
                displacement_norm = np.linalg.norm(velocity_n, axis=-1, keepdims=True)
                velocity_n = np.where(displacement_norm > 1e-6, velocity_n, actual_nose)
                velocity_tangent_err = angle_deg_np(velocity_n, target_nose)
                nose_tangent_err = angle_deg_np(actual_nose, target_nose)
                nose_velocity_err = angle_deg_np(actual_nose, velocity_n)
                wing_plane_err = angle_deg_np(actual_right, target_right)

                # Tracking error (paper-consistent task metric):
                #   s_curve → heading tracking error (deg)
                #   pullup/vertical_arc → pitch tracking error (deg)
                if kind in ("s_curve",):
                    tracking_err = heading_err
                elif kind in ("vertical_arc", "pullup"):
                    tracking_err = pitch_err
                else:
                    tracking_err = pitch_err

                # Update accumulators
                vt_min[active] = np.minimum(vt_min[active], vt[active])
                energy_min[active] = np.minimum(energy_min[active], energy[active])
                alt_final[active] = alt[active]
                alpha_max[active] = np.maximum(alpha_max[active], alpha[active])
                gmax[active] = np.maximum(gmax[active], g_load[active])
                pitch_err_sum[active] += pitch_err[active]
                heading_err_sum[active] += heading_err[active]
                velocity_tangent_err_sum[active] += velocity_tangent_err[active]
                nose_tangent_err_sum[active] += nose_tangent_err[active]
                nose_velocity_err_sum[active] += nose_velocity_err[active]
                wing_plane_err_sum[active] += wing_plane_err[active]
                q_error_norm_sum[active] += q_error_norm[active]
                active_count[active] += 1.0
                for te_seed in np.where(active)[0]:
                    tracking_err_values[te_seed].append(float(tracking_err[te_seed]))
                prev_north[active] = north[active]
                prev_east[active] = east[active]
                prev_altitude[active] = alt[active]

                # Track termination reasons
                done_np = np.asarray(done_step)
                newly_done = active & done_np
                for i in np.where(newly_done)[0]:
                    if g_load[i] > 10.0:
                        reason[i] = "overload"
                    elif np.asarray(info["r_crash"])[i, 0] < 0.0:
                        reason[i] = "crash"
                    else:
                        reason[i] = "env_done"
                done = jnp.asarray(np.asarray(done) | done_np)

            # ── Compute per-seed summary metrics ──
            active_count = np.maximum(active_count, 1.0)
            pitch_err_mean = pitch_err_sum / active_count
            heading_err_mean = heading_err_sum / active_count
            velocity_tangent_err_mean = velocity_tangent_err_sum / active_count
            nose_tangent_err_mean = nose_tangent_err_sum / active_count
            nose_velocity_err_mean = nose_velocity_err_sum / active_count
            wing_plane_err_mean = wing_plane_err_sum / active_count
            q_error_norm_mean = q_error_norm_sum / active_count
            energy_loss = energy0 - energy_min

            tracking_err_mean = np.zeros(num_seeds)
            tracking_err_p50 = np.zeros(num_seeds)
            tracking_err_p90 = np.zeros(num_seeds)
            tracking_err_max = np.zeros(num_seeds)
            for seed_idx, values in enumerate(tracking_err_values):
                arr = np.asarray(values, dtype=np.float64)
                if arr.size == 0:
                    arr = np.asarray([np.inf])
                tracking_err_mean[seed_idx] = float(np.mean(arr))
                tracking_err_p50[seed_idx] = float(np.percentile(arr, 50))
                tracking_err_p90[seed_idx] = float(np.percentile(arr, 90))
                tracking_err_max[seed_idx] = float(np.max(arr))

            # Success criteria
            kind = task["kind"]
            if kind in ("s_curve",):
                track_ok = heading_err_mean < 18.0
            elif kind in ("pullup", "vertical_arc"):
                track_ok = pitch_err_mean < 15.0
            else:
                track_ok = np.ones(num_seeds, dtype=bool)
            success = (reason == "none") & (vt_min > 170.0) & track_ok

            # Aggregate statistics
            surv_rate = success.mean()
            term_rate = (reason != "none").mean()
            vt_m, vt_s = np_mean_std(vt_min)
            al_m, al_s = np_mean_std(alpha_max)
            g_m, g_s = np_mean_std(gmax)
            track90_m, track90_s = np_mean_std(tracking_err_p90)
            vte_m, vte_s = np_mean_std(velocity_tangent_err_mean)
            wpe_m, wpe_s = np_mean_std(wing_plane_err_mean)
            reason_counts = Counter(reason)

            summary = {
                "setting": setting_name,
                "setting_desc": pert["description"],
                "task": task["name"],
                "task_kind": kind,
                "num_seeds": num_seeds,
                "survival_rate": f"{surv_rate:.4f}",
                "termination_rate": f"{term_rate:.4f}",
                "tracking_error_p90_mean": f"{track90_m:.4f}",
                "tracking_error_p90_std": f"{track90_s:.4f}",
                "tracking_error_metric": "heading_err_deg" if kind in ("s_curve",) else "pitch_err_deg",
                "vt_min_mean": f"{vt_m:.4f}",
                "vt_min_std": f"{vt_s:.4f}",
                "alpha_max_mean": f"{al_m:.4f}",
                "alpha_max_std": f"{al_s:.4f}",
                "gmax_mean": f"{g_m:.4f}",
                "gmax_std": f"{g_s:.4f}",
                "velocity_tangent_err_mean": f"{vte_m:.4f}",
                "velocity_tangent_err_std": f"{vte_s:.4f}",
                "wing_plane_err_mean": f"{wpe_m:.4f}",
                "wing_plane_err_std": f"{wpe_s:.4f}",
                "termination_reasons": ";".join(
                    f"{k}:{v}" for k, v in sorted(reason_counts.items())),
                "main_failure_mode": (
                    reason_counts.most_common(1)[0][0]
                    if reason_counts and reason_counts.most_common(1)[0][0] != "none"
                    else "none"),
            }
            all_summaries.append(summary)

            # Per-seed rows
            for seed_idx in range(num_seeds):
                all_rows.append({
                    "setting": setting_name,
                    "task": task["name"],
                    "seed": seed_idx + args.seed_base,
                    "success": int(success[seed_idx]),
                    "termination_reason": reason[seed_idx],
                    "vt_min": f"{vt_min[seed_idx]:.4f}",
                    "alpha_max": f"{alpha_max[seed_idx]:.4f}",
                    "gmax": f"{gmax[seed_idx]:.4f}",
                    "tracking_error_p90": f"{tracking_err_p90[seed_idx]:.4f}",
                    "tracking_error_mean": f"{tracking_err_mean[seed_idx]:.4f}",
                    "tracking_error_metric": "heading_err_deg" if task["kind"] in ("s_curve",) else "pitch_err_deg",
                    "velocity_tangent_err": f"{velocity_tangent_err_mean[seed_idx]:.4f}",
                    "nose_tangent_err": f"{nose_tangent_err_mean[seed_idx]:.4f}",
                    "nose_velocity_err": f"{nose_velocity_err_mean[seed_idx]:.4f}",
                    "wing_plane_err": f"{wing_plane_err_mean[seed_idx]:.4f}",
                    "q_error_norm": f"{q_error_norm_mean[seed_idx]:.4f}",
                })
            print(summary)

    # ── Save outputs ──
    # 1. per_seed_metrics.csv
    per_seed_path = out_dir / "per_seed_metrics.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"\nSaved: {per_seed_path}")

    # 2. summary_metrics.csv
    summary_path = out_dir / "summary_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        if all_summaries:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            w.writerows(all_summaries)
    print(f"Saved: {summary_path}")

    # 3. Generate tables and plots
    generate_latex_table(all_summaries, out_dir)
    generate_summary_table_md(all_summaries, out_dir)
    generate_plots(all_summaries, out_dir)

    # 4. final_report.md
    generate_report(args, all_summaries, out_dir, timestamp, ckpt_path, epoch)

    print(f"\nAll outputs saved to: {out_dir}")
    return out_dir

# ── Output generators ──────────────────────────────────────────────────
def generate_latex_table(summaries, out_dir):
    """Generate LaTeX table: Setting | Task | Survival | Track.Err.P90 | ..."""
    n_seeds = summaries[0]['num_seeds'] if summaries else "?"
    rows = []
    for s in summaries:
        metric_label = s.get("tracking_error_metric", "tracking_err")
        rows.append(
            f"    {s['setting']} & {s['task_kind']} & "
            f"{s['survival_rate']} & {s['tracking_error_p90_mean']} & "
            f"{s['vt_min_mean']} & {s['alpha_max_mean']} & "
            f"{s['gmax_mean']} & {s['termination_rate']} \\\\"
        )

    latex = r"""\begin{table}[t]
  \caption{Sim-to-real sensitivity evaluation. Zero-shot evaluation of the frozen energy-aware PPO policy under dynamics, actuation, wind, and observation perturbations.}
  \label{tab:sim2real-sensitivity}
  \centering
  \small
  \setlength{\tabcolsep}{3.5pt}
  \begin{tabular}{@{}llcccccc@{}}
    \toprule
    \textbf{Setting} & \textbf{Task} & \textbf{Survival} & \textbf{Track.Err.P90} & \textbf{Min $V_t$} & \textbf{Max $|\alpha|$} & \textbf{Max $G$} & \textbf{Term.} \\
    \midrule
"""
    latex += "\n".join(rows)
    latex += r"""
    \bottomrule
  \end{tabular}

  \vspace{3pt}
  \begin{minipage}{0.96\textwidth}
    \footnotesize
    \emph{Note.}
    Each setting evaluated with """ + f"{n_seeds}" + r""" seeds.
    Survival $=$ no crash, overload, or timeout.
    Track.Err.P90 $=$ P90 tracking error:
    heading error ($^\circ$) for S-curve, pitch error ($^\circ$) for $90^\circ$ pull-up.
    Min $V_t$ $=$ minimum airspeed (m/s).
    Max $|\alpha|$ $=$ maximum absolute angle of attack ($^\circ$).
    Max $G$ $=$ maximum load factor.
    Term. $=$ termination rate.
  \end{minipage}
\end{table}
"""
    latex_path = out_dir / "latex_table.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"Saved: {latex_path}")

def generate_summary_table_md(summaries, out_dir):
    """Generate aggregated summary markdown table."""
    lines = [
        "# Sim-to-Real Sensitivity — Summary",
        "",
        "Tracking error: heading error (deg) for S-curve, pitch error (deg) for 90° pull-up.",
        "",
        "## Aggregate Table",
        "",
        "| Setting | Avg Survival | Avg Track.Err.P90 | Worst Min Vt | Worst Max AoA | Worst Max G | Main Failure Mode |",
        "|---------|-------------|-------------------|-------------|--------------|-----------|------------------|",
    ]
    by_setting = {}
    for s in summaries:
        setting = s["setting"]
        if setting not in by_setting:
            by_setting[setting] = []
        by_setting[setting].append(s)

    for setting, items in by_setting.items():
        surv = np.mean([float(x["survival_rate"]) for x in items])
        te = np.mean([float(x["tracking_error_p90_mean"]) for x in items])
        vt = np.min([float(x["vt_min_mean"]) for x in items])
        aoa = np.max([float(x["alpha_max_mean"]) for x in items])
        gmax_val = np.max([float(x["gmax_mean"]) for x in items])
        failures = [x["main_failure_mode"] for x in items if x["main_failure_mode"] != "none"]
        main_failure = Counter(failures).most_common(1)[0][0] if failures else "none"
        lines.append(
            f"| {setting} | {surv:.3f} | {te:.2f} | {vt:.1f} | {aoa:.1f} | {gmax_val:.1f} | {main_failure} |")

    lines.append("")
    lines.append("## Detailed Table")
    lines.append("")
    lines.append("| Setting | Task | Survival | Track.Err.P90 | Min Vt | Max AoA | Max G | Term. Rate | Term. Reasons |")
    lines.append("|---------|------|----------|---------------|--------|---------|-------|------------|---------------|")
    for s in summaries:
        lines.append(
            f"| {s['setting']} | {s['task_kind']} | {s['survival_rate']} | {s['tracking_error_p90_mean']} | "
            f"{s['vt_min_mean']} | {s['alpha_max_mean']} | {s['gmax_mean']} | "
            f"{s['termination_rate']} | {s['termination_reasons']} |")

    md_path = out_dir / "summary_table.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {md_path}")

def generate_plots(summaries, out_dir):
    """Generate visualization plots."""
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})

    # Parse data
    settings = [s["setting"] for s in summaries]
    survivals = [float(s["survival_rate"]) for s in summaries]
    tracking_err_p90s = [float(s["tracking_error_p90_mean"]) for s in summaries]
    vt_mins = [float(s["vt_min_mean"]) for s in summaries]
    aoa_maxs = [float(s["alpha_max_mean"]) for s in summaries]
    gmaxs = [float(s["gmax_mean"]) for s in summaries]
    tasks = [s["task_kind"] for s in summaries]
    task_names = [s["task"] for s in summaries]

    unique_tasks = sorted(set(tasks))
    unique_settings = sorted(set(settings), key=lambda x: settings.index(x))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tasks)))

    # 1. robustness_bar_survival.png
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(unique_settings))
    width = 0.35
    for ti, task in enumerate(unique_tasks):
        vals = [survivals[i] for i, t in enumerate(tasks) if t == task
                and settings[i] in unique_settings]
        # align with settings
        task_vals = []
        for s in unique_settings:
            found = [survivals[i] for i in range(len(summaries))
                     if settings[i] == s and tasks[i] == task]
            task_vals.append(found[0] if found else 0)
        ax.bar(x + ti*width, task_vals, width, label=task, color=colors[ti])
    ax.set_xticks(x + width * (len(unique_tasks)-1)/2)
    ax.set_xticklabels(unique_settings, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Survival Rate")
    ax.set_title("Robustness: Survival Rate under Perturbations")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_bar_survival.png")
    fig.savefig(out_dir / "robustness_bar_survival.pdf")
    plt.close(fig)

    # 2. robustness_bar_cte.png
    fig, ax = plt.subplots(figsize=(12, 5))
    for ti, task in enumerate(unique_tasks):
        task_vals = []
        for s in unique_settings:
            found = [tracking_err_p90s[i] for i in range(len(summaries))
                     if settings[i] == s and tasks[i] == task]
            task_vals.append(found[0] if found else 0)
        ax.bar(x + ti*width, task_vals, width, label=task, color=colors[ti])
    ax.set_xticks(x + width * (len(unique_tasks)-1)/2)
    ax.set_xticklabels(unique_settings, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CTE-P90")
    ax.set_title("Robustness: CTE-P90 under Perturbations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_bar_cte.png")
    fig.savefig(out_dir / "robustness_bar_cte.pdf")
    plt.close(fig)

    # 3. robustness_safety_metrics.png (min speed / max AoA / max G)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metric_names = ["Min Speed (m/s)", "Max |AoA| (deg)", "Max G-load"]
    metric_keys = [vt_mins, aoa_maxs, gmaxs]
    for ax_i, (ax, name, vals) in enumerate(zip(axes, metric_names, metric_keys)):
        for ti, task in enumerate(unique_tasks):
            task_vals = []
            for s in unique_settings:
                found = [vals[i] for i in range(len(summaries))
                         if settings[i] == s and tasks[i] == task]
                task_vals.append(found[0] if found else 0)
            ax.bar(x + ti*width, task_vals, width, label=task, color=colors[ti])
        ax.set_xticks(x + width * (len(unique_tasks)-1)/2)
        ax.set_xticklabels(unique_settings, rotation=45, ha="right", fontsize=6)
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Robustness: Safety Metrics under Perturbations")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_safety_metrics.png")
    fig.savefig(out_dir / "robustness_safety_metrics.pdf")
    plt.close(fig)

    # 4. robustness_task_comparison.png (S-curve vs 90deg pull-up comparison)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_pairs = [
        ("Survival Rate", survivals),
        ("CTE-P90", tracking_err_p90s),
        ("Min Speed (m/s)", vt_mins),
        ("Max G-load", gmaxs),
    ]
    for (ax, (mname, mvals)) in zip(axes.flat, metric_pairs):
        for ti, task in enumerate(unique_tasks):
            task_x = []
            task_y = []
            for i in range(len(summaries)):
                if tasks[i] == task:
                    task_x.append(unique_settings.index(settings[i]))
                    task_y.append(mvals[i])
            ax.plot(task_x, task_y, "o-", label=task, color=colors[ti], markersize=5)
        ax.set_xticks(range(len(unique_settings)))
        ax.set_xticklabels(unique_settings, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(mname)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Task Comparison: S-curve vs 90° Pull-up Robustness")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_task_comparison.png")
    fig.savefig(out_dir / "robustness_task_comparison.pdf")
    plt.close(fig)

    print(f"Saved plots to {out_dir}")

def generate_report(args, summaries, out_dir, timestamp, ckpt_path, epoch):
    """Generate final_report.md."""
    # Compute relative paths for anonymity
    rel_ckpt = str(ckpt_path).split("results/")[-1] if "results/" in str(ckpt_path) else str(ckpt_path.name)
    out_rel = str(out_dir).split("results/")[-1] if "results/" in str(out_dir) else str(out_dir)

    lines = [
        "# Sim-to-Real Sensitivity Evaluation Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Checkpoint:** `results/{rel_ckpt}` (epoch {epoch})",
        f"**Number of seeds per setting:** {args.num_seeds}",
        f"**Tasks:** {args.tasks}",
        "",
        "## Files Modified",
        "",
        "1. `envs/core/simulators/fighterplane/dynamics.py` — Added perturbation fields ",
        "   (`aero_scale`, `mass_scale`, `inertia_scale`, `wind_body_u/v/w`) to ",
        "   `FighterPlaneState`. Modified `nlplant()` to accept and apply these parameters. ",
        "   Modified `update()` to propagate them. All default values preserve the original ",
        "   dynamics behavior exactly.",
        "2. `scripts/eval_sim2real_sensitivity.py` — New evaluation script.",
        "",
        "## Files NOT Modified",
        "",
        "- Training scripts (unchanged)",
        "- Environment reward/termination logic (unchanged)",
        "- Default environment behavior (unchanged — perturbations only applied during eval)",
        "",
        "## How to Run Smoke Test",
        "",
        "```bash",
        "conda activate aeroplanax",
        "python scripts/eval_sim2real_sensitivity.py \\",
        "  --checkpoint results/vertical_energy_finetune/<run>/checkpoint/checkpoint_epoch_619 \\",
        "  --tasks s_curve,pullup_90 \\",
        "  --num_seeds 2 \\",
        "  --output_dir results/sim2real_sensitivity",
        "```",
        "",
        "## How to Run Full Experiment",
        "",
        "```bash",
        "python scripts/eval_sim2real_sensitivity.py \\",
        "  --checkpoint results/vertical_energy_finetune/<run>/checkpoint/checkpoint_epoch_619 \\",
        "  --tasks s_curve,pullup_90 \\",
        "  --num_seeds 10 \\",
        "  --output_dir results/sim2real_sensitivity",
        "```",
        "",
        "## Perturbation Settings Evaluated",
        "",
        "| Setting | Description |",
        "|---------|-------------|",
    ]
    for name, pert in get_perturbations():
        lines.append(f"| {name} | {pert['description']} |")

    lines += [
        "",
        "## Implementation Notes",
        "",
        "- **Aero coefficient scaling**: Multiplies all aerodynamic coefficients ",
        "  (Cx, Cz, Cm, Cy, Cn, Cl and their delta terms) by the scale factor at the ",
        "  force/moment computation level. Does NOT scale speed brake effects.",
        "- **Mass/Inertia scaling**: Scales mass and all moments of inertia (Jx, Jy, Jz, Jxz) ",
        "  simultaneously by the same factor.",
        "- **Actuator delay**: Maintains a Python-level buffer of past actions and feeds ",
        "  delayed actions to `env.step`. The policy still runs at 5 Hz; the action reaches ",
        "  the dynamics with the specified delay (1 step = 0.02s).",
        "- **Wind disturbance**: Adds body-frame wind by subtracting wind velocity from ",
        "  airspeed before aero coefficient lookup. Wind is modeled as headwind from North.",
        "- **Observation noise**: Adds independent Gaussian noise (std=0.01) to the normalized ",
        "  observation vector AFTER `_get_obs()` and BEFORE the policy network forward pass.",
        "",
        "## Results",
        "",
        f"- **per_seed_metrics.csv**: `results/{out_rel}/per_seed_metrics.csv`",
        f"- **summary_metrics.csv**: `results/{out_rel}/summary_metrics.csv`",
        f"- **summary_table.md**: `results/{out_rel}/summary_table.md`",
        f"- **latex_table.tex**: `results/{out_rel}/latex_table.tex`",
        f"- **final_report.md**: `results/{out_rel}/final_report.md` (this file)",
        f"- **figure_sim2real_sensitivity.pdf/svg**: `results/{out_rel}/figure_sim2real_sensitivity.pdf`",
        "",
        "## Figures for Paper",
        "",
        "The following files can be directly used in the CoRL paper:",
        "",
        "1. `latex_table.tex` — LaTeX table (Setting | Task | Survival | Track.Err.P90 | ...)",
        "2. `figure_sim2real_sensitivity.pdf` / `.svg` — Grouped perturbation sensitivity (2-panel)",
    ]

    report_path = out_dir / "final_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")

# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot sim-to-real sensitivity evaluation")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT,
                        help="Path to frozen energy-aware PPO checkpoint")
    parser.add_argument("--tasks", type=str, default="s_curve,pullup_90",
                        help="Comma-separated task names: s_curve, pullup_90")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of random seeds per setting")
    parser.add_argument("--seed_base", type=int, default=20260521,
                        help="Base seed for reproducibility")
    parser.add_argument("--output_dir", type=Path,
                        default=PLANAX_ROOT / "results/sim2real_sensitivity",
                        help="Parent output directory (timestamp subdir created)")
    args = parser.parse_args()
    run_eval(args)

if __name__ == "__main__":
    main()
