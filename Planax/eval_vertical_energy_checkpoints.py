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

from envs.aeroplanax_heading_pitch_V_quaternion_version_vertical_energy import (
    AeroPlanaxHeading_Pitch_V_Env,
    Heading_Pitch_V_TaskParams,
)
from envs.utils.utils import wrap_PI
from experiments.hierarchical_trajectory_tracking.loop_attitude_target import loop_plane_hpr_jax
from half_loop_residual_policy import (
    ResidualActorCriticRNN,
    ResidualScannedRNN,
    augment_obs_flat,
    combine_base_and_residual_logits,
)


PLANAX_ROOT = Path(__file__).resolve().parent
BASELINE_CKPT = (
    PLANAX_ROOT
    / "results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600"
)
DEFAULT_NEW_CKPT = (
    PLANAX_ROOT
    / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
)
DEFAULT_OUT = PLANAX_ROOT / "results/vertical_energy_finetune/20260515_1615"

NET_CONFIG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
DT_RL = 10.0 / 50.0
G = 9.80665


def wrap_pi_np(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def euler_to_quat_nb_np(roll, pitch, yaw):
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )


def quat_conj_np(q):
    out = np.array(q, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_mul_np(q1, q2):
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ],
        axis=-1,
    )


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
    right = np.stack(
        [
            -cr * sy + sr * sp * cy,
            cr * cy + sr * sp * sy,
            sr * cp,
        ],
        axis=-1,
    )
    return nose, right


def angle_deg_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


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
        pi_throttle = distrax.Categorical(
            logits=nn.Dense(
                self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
        )
        pi_elevator = distrax.Categorical(
            logits=nn.Dense(
                self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
        )
        pi_aileron = distrax.Categorical(
            logits=nn.Dense(
                self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
        )
        pi_rudder = distrax.Categorical(
            logits=nn.Dense(
                self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
        )
        pi_speed_brake = distrax.Categorical(
            logits=nn.Dense(
                self.action_dim[4],
                kernel_init=constant(0.0),
                bias_init=lambda key, shape, dtype=jnp.float32: jnp.array(
                    [0.0, -1.5, -1.5, -1.5, -1.5], dtype=dtype
                ),
            )(actor_mean)
        )

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return (
            hidden,
            (pi_throttle, pi_elevator, pi_aileron, pi_rudder, pi_speed_brake),
            jnp.squeeze(critic, axis=-1),
        )


def restore_params(path: Path):
    ckpt = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        str(path), args=ocp.args.StandardRestore()
    )
    return ckpt["params"], int(np.asarray(ckpt["epoch"]))


def restore_residual_params(path: Path):
    ckpt = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        str(path), args=ocp.args.StandardRestore()
    )
    return ckpt["params"], int(np.asarray(ckpt.get("epoch", 0)))


def load_residual_config(path: Path | None):
    cfg = {
        "ACTIVATION": "relu",
        "RESIDUAL_FC_DIM_SIZE": 96,
        "RESIDUAL_GRU_HIDDEN_DIM": 64,
        "RESIDUAL_LOGIT_CLIP": 1.25,
        "RESIDUAL_GATE_START_DEG": 80.0,
        "RESIDUAL_GATE_END_DEG": 180.0,
    }
    if path is not None:
        with path.open("r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def task_catalog(suite: str = "all", include_90: bool = False):
    vertical = [
        {"name": "15_pullup_R3000", "category": "vertical", "kind": "pullup", "angle_deg": 15.0, "radius": 3000.0},
        {"name": "15_pullup_R2000", "category": "vertical", "kind": "pullup", "angle_deg": 15.0, "radius": 2000.0},
        {"name": "30_pullup_R8000", "category": "vertical", "kind": "pullup", "angle_deg": 30.0, "radius": 8000.0},
        {"name": "30_pullup_R5000", "category": "vertical", "kind": "pullup", "angle_deg": 30.0, "radius": 5000.0},
        {"name": "60_vertical_arc_R8000", "category": "vertical_arc", "kind": "vertical_arc", "angle_deg": 60.0, "radius": 8000.0},
        {"name": "60_vertical_arc_R10000", "category": "vertical_arc", "kind": "vertical_arc", "angle_deg": 60.0, "radius": 10000.0},
    ]
    quarter_loop = [
        {"name": "90_quarter_loop_R10000", "category": "vertical_arc", "kind": "vertical_arc", "angle_deg": 90.0, "radius": 10000.0},
    ]
    loop_retention = [
        {"name": "60_loop_plane_arc_R10000", "category": "vertical_retention", "kind": "loop_arc", "angle_deg": 60.0, "radius": 10000.0},
        {"name": "90_loop_plane_arc_R10000", "category": "vertical_retention", "kind": "loop_arc", "angle_deg": 90.0, "radius": 10000.0},
        {"name": "120_loop_plane_arc_R15000", "category": "vertical_retention", "kind": "loop_arc", "angle_deg": 120.0, "radius": 15000.0},
        {"name": "150_loop_plane_arc_R15000", "category": "vertical_retention", "kind": "loop_arc", "angle_deg": 150.0, "radius": 15000.0},
    ]
    half_loop = [
        {"name": "160_half_loop_arc_R15000", "category": "half_loop", "kind": "loop_arc", "angle_deg": 160.0, "radius": 15000.0},
        {"name": "165_half_loop_arc_R15000", "category": "half_loop", "kind": "loop_arc", "angle_deg": 165.0, "radius": 15000.0},
        {"name": "170_half_loop_arc_R15000", "category": "half_loop", "kind": "loop_arc", "angle_deg": 170.0, "radius": 15000.0},
        {"name": "175_half_loop_arc_R15000", "category": "half_loop", "kind": "loop_arc", "angle_deg": 175.0, "radius": 15000.0},
        {"name": "180_half_loop_R15000", "category": "half_loop", "kind": "loop_arc", "angle_deg": 180.0, "radius": 15000.0},
    ]
    if include_90:
        vertical = vertical + quarter_loop
    retention = [
        {"name": "level_flight", "category": "retention", "kind": "level"},
        {"name": "heading_p20", "category": "retention", "kind": "heading_step", "heading_deg": 20.0},
        {"name": "heading_m20", "category": "retention", "kind": "heading_step", "heading_deg": -20.0},
        {"name": "heading_p45", "category": "retention", "kind": "heading_step", "heading_deg": 45.0},
        {"name": "heading_m45", "category": "retention", "kind": "heading_step", "heading_deg": -45.0},
        {"name": "pitch_p10", "category": "retention", "kind": "pitch_step", "pitch_deg": 10.0},
        {"name": "pitch_m10", "category": "retention", "kind": "pitch_step", "pitch_deg": -10.0},
    ]
    old_skill = [
        {"name": "level_circle_R5000_right", "category": "old_skill_proxy", "kind": "circle", "radius": 5000.0, "direction": 1.0},
        {"name": "level_circle_R5000_left", "category": "old_skill_proxy", "kind": "circle", "radius": 5000.0, "direction": -1.0},
        {"name": "level_circle_R3000_right", "category": "old_skill_proxy", "kind": "circle", "radius": 3000.0, "direction": 1.0},
        {"name": "level_circle_R3000_left", "category": "old_skill_proxy", "kind": "circle", "radius": 3000.0, "direction": -1.0},
        {"name": "s_curve_A3000", "category": "old_skill_proxy", "kind": "s_curve", "amplitude": 3000.0},
        {"name": "figure_eight_R5000", "category": "old_skill_proxy", "kind": "figure_eight", "radius": 5000.0},
        {"name": "mild_climb_p1000m", "category": "old_skill_proxy", "kind": "altitude_step", "alt_delta": 1000.0},
        {"name": "mild_climb_p2000m", "category": "old_skill_proxy", "kind": "altitude_step", "alt_delta": 2000.0},
        {"name": "mild_descent_m1000m", "category": "old_skill_proxy", "kind": "altitude_step", "alt_delta": -1000.0},
    ]
    if suite == "target":
        return vertical[:4]
    if suite == "quarter_loop":
        return quarter_loop
    if suite == "horizontal_v2":
        return [
            old_skill[2],
            old_skill[3],
            old_skill[0],
            old_skill[1],
            old_skill[4],
            old_skill[5],
            old_skill[6],
            old_skill[8],
        ]
    if suite == "half_loop_search":
        return old_skill[:6] + loop_retention + half_loop
    if suite == "planner_proxy":
        return retention + old_skill + vertical
    return vertical + retention + old_skill


def task_horizon(task):
    kind = task["kind"]
    if kind in ("pullup", "vertical_arc", "loop_arc"):
        ramp_steps = max(3, int(np.ceil(np.deg2rad(task["angle_deg"]) * task["radius"] / 250.0 / DT_RL)))
        hold_steps = 45 if task["angle_deg"] <= 15.0 else 85
        return ramp_steps + hold_steps, ramp_steps
    if kind in ("circle", "s_curve", "figure_eight"):
        return 240, 1
    if kind == "altitude_step":
        return 180, 1
    return 70, 1


def build_target(task, step, ramp_steps, init, state):
    yaw0, pitch0, alt0 = init
    shape = yaw0.shape
    kind = task["kind"]
    target_heading = yaw0
    target_pitch = jnp.zeros(shape)
    target_roll = jnp.zeros(shape)
    target_vt = jnp.full(shape, 250.0)
    t = step * DT_RL

    if kind == "heading_step":
        target_heading = wrap_PI(yaw0 + jnp.deg2rad(task["heading_deg"]))
    elif kind == "pitch_step":
        target_pitch = jnp.full(shape, jnp.deg2rad(task["pitch_deg"]))
    elif kind in ("pullup", "vertical_arc"):
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
        target_heading = wrap_PI(yaw0 + task.get("direction", 1.0) * omega * t)
    elif kind == "s_curve":
        period = 85.0
        amp_heading = jnp.deg2rad(32.0)
        target_heading = wrap_PI(yaw0 + amp_heading * jnp.sin(2.0 * jnp.pi * t / period))
    elif kind == "figure_eight":
        period = 120.0
        amp_heading = jnp.deg2rad(42.0)
        target_heading = wrap_PI(yaw0 + amp_heading * jnp.sin(4.0 * jnp.pi * t / period))
    elif kind == "altitude_step":
        target_alt = alt0 + task["alt_delta"]
        alt_err = target_alt - state.plane_state.altitude
        target_pitch = jnp.clip(jnp.arctan2(alt_err, 6500.0), jnp.deg2rad(-10.0), jnp.deg2rad(10.0))
    return target_heading, target_pitch, target_roll, target_vt


def np_mean_std(x):
    arr = np.asarray(x, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE_CKPT)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260515)
    parser.add_argument(
        "--suite",
        choices=["all", "target", "planner_proxy", "quarter_loop", "half_loop_search", "horizontal_v2"],
        default="all",
    )
    parser.add_argument("--include-90", action="store_true")
    parser.add_argument("--residual-checkpoint", type=Path, default=None)
    parser.add_argument("--residual-config", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)
    agent = env.agents[0]
    network = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CONFIG)
    residual_cfg = load_residual_config(args.residual_config)
    residual_network = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=residual_cfg)
    residual_params = None
    residual_epoch = None
    if args.residual_checkpoint is not None:
        residual_params, residual_epoch = restore_residual_params(args.residual_checkpoint.resolve())

    checkpoints = {
        "baseline_epoch600": args.baseline,
        "candidate": args.new,
    }
    params_by_policy = {}
    epochs = {}
    for name, path in checkpoints.items():
        params_by_policy[name], epochs[name] = restore_params(path.resolve())

    print("devices", jax.devices())
    print("epochs", epochs)

    num_seeds = args.seeds

    @jax.jit
    def batched_step(net_params, hstate, state, done, key, target_heading, target_pitch, target_roll, target_vt):
        state = state.replace(
            target_heading=target_heading,
            target_pitch=target_pitch,
            target_roll=target_roll,
            target_vt=target_vt,
            task_mode=jnp.zeros_like(state.task_mode),
            task_duration_steps=jnp.full_like(state.task_duration_steps, 10000.0),
            last_check_time=state.time,
        )
        obs_dict = jax.vmap(env._get_obs, in_axes=(0, None))(state, env_params)
        obs = obs_dict[agent]
        hstate, pi, _ = network.apply(net_params, hstate, (obs[None, :, :], done[None, :]))
        action = jnp.stack([p.mode()[0] for p in pi], axis=-1).astype(jnp.int32)
        step_keys = jax.random.split(key, num_seeds)
        obs_next, state_next, reward, done_dict, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            step_keys, state, {agent: action}, env_params
        )
        del obs_next
        return state_next, done_dict[agent], hstate, reward[agent], info, action

    @jax.jit
    def batched_step_residual(
        net_params,
        residual_params,
        hstate,
        residual_hstate,
        state,
        done,
        key,
        target_heading,
        target_pitch,
        target_roll,
        target_vt,
    ):
        state = state.replace(
            target_heading=target_heading,
            target_pitch=target_pitch,
            target_roll=target_roll,
            target_vt=target_vt,
            task_mode=jnp.zeros_like(state.task_mode),
            task_duration_steps=jnp.full_like(state.task_duration_steps, 10000.0),
            last_check_time=state.time,
        )
        obs_dict = jax.vmap(env._get_obs, in_axes=(0, None))(state, env_params)
        obs = obs_dict[agent]
        hstate, base_pi, _ = network.apply(net_params, hstate, (obs[None, :, :], done[None, :]))
        obs_aug = augment_obs_flat(obs, state, residual_cfg)
        residual_hstate, residual_logits, _ = residual_network.apply(
            residual_params, residual_hstate, (obs_aug[None, :, :], done[None, :])
        )
        pi, _, _ = combine_base_and_residual_logits(base_pi, residual_logits, obs_aug, residual_cfg)
        action = jnp.stack([p.mode()[0] for p in pi], axis=-1).astype(jnp.int32)
        step_keys = jax.random.split(key, num_seeds)
        obs_next, state_next, reward, done_dict, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            step_keys, state, {agent: action}, env_params
        )
        del obs_next
        return state_next, done_dict[agent], hstate, residual_hstate, reward[agent], info, action

    per_seed_rows = []
    summary_rows = []

    for policy_name, net_params in params_by_policy.items():
        use_residual_policy = (
            args.residual_checkpoint is not None
            and residual_params is not None
            and policy_name == "candidate"
        )
        for task_idx, task in enumerate(task_catalog(args.suite, include_90=args.include_90)):
            max_steps, ramp_steps = task_horizon(task)
            reset_keys = jax.random.split(jax.random.PRNGKey(args.seed_base + task_idx), num_seeds)
            _, state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
            init = (
                state.plane_state.yaw,
                state.plane_state.pitch,
                state.plane_state.altitude,
            )
            state = state.replace(
                task_start_heading=state.plane_state.yaw,
                task_start_pitch=state.plane_state.pitch,
                task_start_roll=state.plane_state.roll,
                task_start_vt=state.plane_state.vt,
                task_start_altitude=state.plane_state.altitude,
                task_start_energy=0.5 * state.plane_state.vt * state.plane_state.vt + G * state.plane_state.altitude,
                task_mode=jnp.zeros_like(state.task_mode),
                task_duration_steps=jnp.full_like(state.task_duration_steps, 10000.0),
                last_check_time=state.time,
            )
            hstate = ScannedRNN.initialize_carry(num_seeds, NET_CONFIG["GRU_HIDDEN_DIM"])
            residual_hstate = ResidualScannedRNN.initialize_carry(
                num_seeds, int(residual_cfg.get("RESIDUAL_GRU_HIDDEN_DIM", 64))
            )
            done = jnp.zeros((num_seeds,), dtype=jnp.bool_)

            vt_min = np.full(num_seeds, np.inf)
            energy_min = np.full(num_seeds, np.inf)
            energy0 = np.asarray((0.5 * state.plane_state.vt * state.plane_state.vt + G * state.plane_state.altitude)[:, 0])
            alt0 = np.asarray(state.plane_state.altitude[:, 0])
            alt_final = alt0.copy()
            alt_min = alt0.copy()
            alt_max = alt0.copy()
            prev_north = np.asarray(state.plane_state.north[:, 0]).copy()
            prev_east = np.asarray(state.plane_state.east[:, 0]).copy()
            prev_altitude = alt0.copy()
            alpha_max = np.zeros(num_seeds)
            alpha_min_signed = np.full(num_seeds, np.inf)
            alpha_max_signed = np.full(num_seeds, -np.inf)
            gmax = np.zeros(num_seeds)
            g_sum = np.zeros(num_seeds)
            pitch_err_sum = np.zeros(num_seeds)
            pitch_err_count = np.zeros(num_seeds)
            heading_err_sum = np.zeros(num_seeds)
            velocity_tangent_err_sum = np.zeros(num_seeds)
            nose_tangent_err_sum = np.zeros(num_seeds)
            nose_velocity_err_sum = np.zeros(num_seeds)
            wing_plane_err_sum = np.zeros(num_seeds)
            q_error_norm_sum = np.zeros(num_seeds)
            cte_values = [[] for _ in range(num_seeds)]
            action_sat_sum = np.zeros(num_seeds)
            active_count = np.zeros(num_seeds)
            vt_sum = np.zeros(num_seeds)
            reward_sum = np.zeros(num_seeds)
            target_pitch_min = np.full(num_seeds, np.inf)
            target_pitch_max = np.full(num_seeds, -np.inf)
            actual_pitch_min = np.full(num_seeds, np.inf)
            actual_pitch_max = np.full(num_seeds, -np.inf)
            target_roll_min = np.full(num_seeds, np.inf)
            target_roll_max = np.full(num_seeds, -np.inf)
            actual_roll_min = np.full(num_seeds, np.inf)
            actual_roll_max = np.full(num_seeds, -np.inf)
            reason = np.array(["none"] * num_seeds, dtype=object)

            for step in range(max_steps):
                th, tp, tr, tv = build_target(task, step, ramp_steps, init, state)
                key = jax.random.PRNGKey(args.seed_base + 100000 + task_idx * 1000 + step)
                if use_residual_policy:
                    state, done_step, hstate, residual_hstate, reward, info, action = batched_step_residual(
                        net_params,
                        residual_params,
                        hstate,
                        residual_hstate,
                        state,
                        done,
                        key,
                        th,
                        tp,
                        tr,
                        tv,
                    )
                else:
                    state, done_step, hstate, reward, info, action = batched_step(
                        net_params, hstate, state, done, key, th, tp, tr, tv
                    )
                active = ~np.asarray(done)
                if not active.any():
                    break

                vt = np.asarray(info["vt"])[:, 0]
                alt = np.asarray(info["altitude"])[:, 0]
                energy = np.asarray(info["energy_proxy"])[:, 0]
                pitch_deg = np.asarray(info["pitch_deg"])[:, 0]
                target_pitch_deg = np.asarray(info["target_pitch_deg"])[:, 0]
                roll_deg = np.asarray(info.get("roll_deg", state.plane_state.roll * 180.0 / jnp.pi))[:, 0]
                target_roll_deg = np.asarray(info.get("target_roll_deg", tr * 180.0 / jnp.pi))[:, 0]
                target_heading = np.asarray(th)[:, 0]
                yaw = np.asarray(info.get("yaw", state.plane_state.yaw))[:, 0] if "yaw" in info else np.asarray(state.plane_state.yaw)[:, 0]
                pitch_err = np.abs(pitch_deg - target_pitch_deg)
                heading_err = np.abs(np.asarray(wrap_PI(jnp.asarray(yaw - target_heading)))) * 180.0 / np.pi
                alpha_signed = np.asarray(info["alpha_deg"])[:, 0]
                alpha = np.abs(alpha_signed)
                g_load = np.asarray(info["g_load_max"])[:, 0]
                rew = np.asarray(reward)
                kind = task["kind"]
                q_curr_np = np.stack(
                    [
                        np.asarray(state.plane_state.q0)[:, 0],
                        np.asarray(state.plane_state.q1)[:, 0],
                        np.asarray(state.plane_state.q2)[:, 0],
                        np.asarray(state.plane_state.q3)[:, 0],
                    ],
                    axis=-1,
                )
                target_roll_rad = np.asarray(tr)[:, 0]
                target_pitch_rad = np.asarray(tp)[:, 0]
                target_heading_rad = np.asarray(th)[:, 0]
                actual_roll_rad = np.asarray(state.plane_state.roll)[:, 0]
                actual_pitch_rad = np.asarray(state.plane_state.pitch)[:, 0]
                actual_yaw_rad = np.asarray(state.plane_state.yaw)[:, 0]
                q_error_norm = quat_angle_deg_np(q_curr_np, target_roll_rad, target_pitch_rad, target_heading_rad)
                actual_nose, actual_right = body_axes_from_euler_np(actual_roll_rad, actual_pitch_rad, actual_yaw_rad)
                target_nose, target_right = body_axes_from_euler_np(target_roll_rad, target_pitch_rad, target_heading_rad)
                north = np.asarray(state.plane_state.north)[:, 0]
                east = np.asarray(state.plane_state.east)[:, 0]
                displacement_n = north - prev_north
                displacement_e = east - prev_east
                displacement_d = -(alt - prev_altitude)
                velocity_n = np.stack(
                    [
                        displacement_n,
                        displacement_e,
                        displacement_d,
                    ],
                    axis=-1,
                )
                displacement_norm = np.linalg.norm(velocity_n, axis=-1, keepdims=True)
                velocity_n = np.where(displacement_norm > 1e-6, velocity_n, actual_nose)
                velocity_tangent_err = angle_deg_np(velocity_n, target_nose)
                nose_tangent_err = angle_deg_np(actual_nose, target_nose)
                nose_velocity_err = angle_deg_np(actual_nose, velocity_n)
                wing_plane_err = angle_deg_np(actual_right, target_right)
                if kind in ("heading_step", "circle", "s_curve", "figure_eight"):
                    cte = heading_err
                elif kind == "altitude_step":
                    cte = np.abs((alt0 + task["alt_delta"]) - alt)
                elif kind == "loop_arc":
                    cte = np.maximum.reduce([velocity_tangent_err, nose_tangent_err, wing_plane_err])
                else:
                    cte = pitch_err
                act_np = np.asarray(action)
                sat = np.mean(
                    (act_np <= np.array([0, 0, 0, 0, 0]))
                    | (act_np >= np.array([30, 40, 40, 40, 4])),
                    axis=1,
                )

                vt_min[active] = np.minimum(vt_min[active], vt[active])
                vt_sum[active] += vt[active]
                energy_min[active] = np.minimum(energy_min[active], energy[active])
                alt_final[active] = alt[active]
                alt_min[active] = np.minimum(alt_min[active], alt[active])
                alt_max[active] = np.maximum(alt_max[active], alt[active])
                alpha_max[active] = np.maximum(alpha_max[active], alpha[active])
                alpha_min_signed[active] = np.minimum(alpha_min_signed[active], alpha_signed[active])
                alpha_max_signed[active] = np.maximum(alpha_max_signed[active], alpha_signed[active])
                gmax[active] = np.maximum(gmax[active], g_load[active])
                g_sum[active] += g_load[active]
                pitch_err_sum[active] += pitch_err[active]
                heading_err_sum[active] += heading_err[active]
                velocity_tangent_err_sum[active] += velocity_tangent_err[active]
                nose_tangent_err_sum[active] += nose_tangent_err[active]
                nose_velocity_err_sum[active] += nose_velocity_err[active]
                wing_plane_err_sum[active] += wing_plane_err[active]
                q_error_norm_sum[active] += q_error_norm[active]
                action_sat_sum[active] += sat[active]
                reward_sum[active] += rew[active]
                target_pitch_min[active] = np.minimum(target_pitch_min[active], target_pitch_deg[active])
                target_pitch_max[active] = np.maximum(target_pitch_max[active], target_pitch_deg[active])
                actual_pitch_min[active] = np.minimum(actual_pitch_min[active], pitch_deg[active])
                actual_pitch_max[active] = np.maximum(actual_pitch_max[active], pitch_deg[active])
                target_roll_min[active] = np.minimum(target_roll_min[active], target_roll_deg[active])
                target_roll_max[active] = np.maximum(target_roll_max[active], target_roll_deg[active])
                actual_roll_min[active] = np.minimum(actual_roll_min[active], roll_deg[active])
                actual_roll_max[active] = np.maximum(actual_roll_max[active], roll_deg[active])
                for cte_seed in np.where(active)[0]:
                    cte_values[cte_seed].append(float(cte[cte_seed]))
                active_count[active] += 1.0
                prev_north[active] = north[active]
                prev_east[active] = east[active]
                prev_altitude[active] = alt[active]

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

            active_count = np.maximum(active_count, 1.0)
            pitch_err_mean = pitch_err_sum / active_count
            heading_err_mean = heading_err_sum / active_count
            velocity_tangent_err_mean = velocity_tangent_err_sum / active_count
            nose_tangent_err_mean = nose_tangent_err_sum / active_count
            nose_velocity_err_mean = nose_velocity_err_sum / active_count
            wing_plane_err_mean = wing_plane_err_sum / active_count
            q_error_norm_mean = q_error_norm_sum / active_count
            action_sat_mean = action_sat_sum / active_count
            vt_mean = vt_sum / active_count
            g_mean = g_sum / active_count
            reward_mean = reward_sum / active_count
            energy_loss = energy0 - energy_min
            altitude_gain = alt_final - alt0
            cte_mean = np.zeros(num_seeds)
            cte_p50 = np.zeros(num_seeds)
            cte_p90 = np.zeros(num_seeds)
            cte_max = np.zeros(num_seeds)
            for seed_idx, values in enumerate(cte_values):
                arr = np.asarray(values, dtype=np.float64)
                if arr.size == 0:
                    arr = np.asarray([np.inf])
                cte_mean[seed_idx] = float(np.mean(arr))
                cte_p50[seed_idx] = float(np.percentile(arr, 50))
                cte_p90[seed_idx] = float(np.percentile(arr, 90))
                cte_max[seed_idx] = float(np.max(arr))

            kind = task["kind"]
            if kind in ("heading_step", "circle", "s_curve", "figure_eight"):
                track_ok = heading_err_mean < (18.0 if kind != "heading_step" else 22.0)
            elif kind == "loop_arc":
                track_ok = (cte_mean < 45.0) & (q_error_norm_mean < 50.0)
            else:
                track_ok = pitch_err_mean < (15.0 if kind in ("pullup", "vertical_arc") else 9.0)
            if kind == "altitude_step":
                desired = np.sign(task["alt_delta"])
                alt_ok = desired * altitude_gain > 400.0
            else:
                alt_ok = np.ones(num_seeds, dtype=bool)
            success = (reason == "none") & (vt_min > 170.0) & track_ok & alt_ok
            grade = np.full(num_seeds, "F", dtype=object)
            if kind in ("pullup", "vertical_arc"):
                grade = np.where(success & (cte_p90 < 4.0) & (vt_min > 180.0) & (alpha_max < 26.0) & (gmax < 8.0), "A", grade)
                grade = np.where(success & (grade == "F") & (cte_p90 < 8.0) & (vt_min > 170.0) & (alpha_max < 32.0) & (gmax < 9.0), "B", grade)
                grade = np.where(success & (grade == "F") & (cte_p90 < 15.0) & (vt_min > 165.0) & (alpha_max < 38.0) & (gmax < 10.0), "C", grade)
            elif kind == "loop_arc":
                grade = np.where(success & (cte_p90 < 18.0) & (q_error_norm_mean < 14.0) & (vt_min > 180.0) & (alpha_max < 28.0) & (gmax < 8.5), "A", grade)
                grade = np.where(success & (grade == "F") & (cte_p90 < 35.0) & (q_error_norm_mean < 28.0) & (vt_min > 170.0) & (alpha_max < 36.0) & (gmax < 9.5), "B", grade)
                grade = np.where(success & (grade == "F") & (cte_p90 < 55.0) & (q_error_norm_mean < 45.0) & (vt_min > 165.0) & (alpha_max < 42.0) & (gmax < 10.0), "C", grade)
            elif kind in ("circle", "s_curve", "figure_eight", "level", "heading_step", "pitch_step"):
                grade = np.where(success & (cte_p90 < 6.0) & (np.abs(altitude_gain) < 150.0), "A", grade)
                grade = np.where(success & (grade == "F") & (cte_p90 < 12.0) & (np.abs(altitude_gain) < 300.0), "B", grade)
                grade = np.where(success & (grade == "F") & (cte_p90 < 20.0) & (np.abs(altitude_gain) < 500.0), "C", grade)
            else:
                grade = np.where(success, "B", grade)

            for seed_idx in range(num_seeds):
                per_seed_rows.append(
                    {
                        "policy": policy_name,
                        "checkpoint": str(checkpoints[policy_name]),
                        "epoch": epochs[policy_name],
                        "task": task["name"],
                        "category": task["category"],
                        "seed": args.seed_base + seed_idx,
                        "success": int(success[seed_idx]),
                        "completed": int(success[seed_idx]),
                        "grade": grade[seed_idx],
                        "termination_reason": reason[seed_idx],
                        "vt_min": f"{vt_min[seed_idx]:.6f}",
                        "vt_mean": f"{vt_mean[seed_idx]:.6f}",
                        "energy_loss": f"{energy_loss[seed_idx]:.6f}",
                        "alpha_max": f"{alpha_max[seed_idx]:.6f}",
                        "Gmax": f"{gmax[seed_idx]:.6f}",
                        "Gmean": f"{g_mean[seed_idx]:.6f}",
                        "CTE_mean": f"{cte_mean[seed_idx]:.6f}",
                        "CTE_p50": f"{cte_p50[seed_idx]:.6f}",
                        "CTE_p90": f"{cte_p90[seed_idx]:.6f}",
                        "CTE_max": f"{cte_max[seed_idx]:.6f}",
                        "velocity_tangent_error": f"{velocity_tangent_err_mean[seed_idx]:.6f}",
                        "nose_tangent_error": f"{nose_tangent_err_mean[seed_idx]:.6f}",
                        "nose_velocity_error": f"{nose_velocity_err_mean[seed_idx]:.6f}",
                        "wing_plane_error": f"{wing_plane_err_mean[seed_idx]:.6f}",
                        "q_error_norm": f"{q_error_norm_mean[seed_idx]:.6f}",
                        "pitch_tracking_error": f"{pitch_err_mean[seed_idx]:.6f}",
                        "heading_tracking_error": f"{heading_err_mean[seed_idx]:.6f}",
                        "altitude_gain": f"{altitude_gain[seed_idx]:.6f}",
                        "altitude_drift": f"{altitude_gain[seed_idx]:.6f}",
                        "altitude_min": f"{alt_min[seed_idx]:.6f}",
                        "altitude_max": f"{alt_max[seed_idx]:.6f}",
                        "env_alpha_range": f"{alpha_min_signed[seed_idx]:.6f}:{alpha_max_signed[seed_idx]:.6f}",
                        "target_pitch_range": f"{target_pitch_min[seed_idx]:.6f}:{target_pitch_max[seed_idx]:.6f}",
                        "actual_pitch_range": f"{actual_pitch_min[seed_idx]:.6f}:{actual_pitch_max[seed_idx]:.6f}",
                        "target_roll_range": f"{target_roll_min[seed_idx]:.6f}:{target_roll_max[seed_idx]:.6f}",
                        "actual_roll_range": f"{actual_roll_min[seed_idx]:.6f}:{actual_roll_max[seed_idx]:.6f}",
                        "action_saturation": f"{action_sat_mean[seed_idx]:.6f}",
                        "reward_mean": f"{reward_mean[seed_idx]:.6f}",
                    }
                )

            vt_m, vt_s = np_mean_std(vt_min)
            en_m, en_s = np_mean_std(energy_loss)
            al_m, al_s = np_mean_std(alpha_max)
            g_m, g_s = np_mean_std(gmax)
            gm_m, gm_s = np_mean_std(g_mean)
            pe_m, pe_s = np_mean_std(pitch_err_mean)
            ag_m, ag_s = np_mean_std(altitude_gain)
            ctem_m, ctem_s = np_mean_std(cte_mean)
            ctep50_m, ctep50_s = np_mean_std(cte_p50)
            ctep90_m, ctep90_s = np_mean_std(cte_p90)
            ctemax_m, ctemax_s = np_mean_std(cte_max)
            vte_m, vte_s = np_mean_std(velocity_tangent_err_mean)
            nte_m, nte_s = np_mean_std(nose_tangent_err_mean)
            nve_m, nve_s = np_mean_std(nose_velocity_err_mean)
            wpe_m, wpe_s = np_mean_std(wing_plane_err_mean)
            qn_m, qn_s = np_mean_std(q_error_norm_mean)
            vtmean_m, vtmean_s = np_mean_std(vt_mean)
            altmin_m, altmin_s = np_mean_std(alt_min)
            altmax_m, altmax_s = np_mean_std(alt_max)
            reason_counts = Counter(reason)
            grade_counts = Counter(grade)
            grade_order = {"A": 4, "B": 3, "C": 2, "F": 1}
            grade_label = max(grade_counts, key=lambda g: (grade_counts[g], grade_order.get(g, 0)))
            target_signature = (
                f"kind={kind};horizon={max_steps};ramp_steps={ramp_steps};"
                f"radius={task.get('radius','')};angle_deg={task.get('angle_deg','')}"
            )
            summary = {
                "policy": policy_name,
                "checkpoint": str(checkpoints[policy_name]),
                "epoch": epochs[policy_name],
                "task": task["name"],
                "category": task["category"],
                "num_seeds": num_seeds,
                "success_rate": f"{success.mean():.6f}",
                "completed_rate": f"{success.mean():.6f}",
                "grade": grade_label,
                "grade_counts": ";".join(f"{k}:{v}" for k, v in sorted(grade_counts.items())),
                "crash_rate": f"{np.mean(reason != 'none'):.6f}",
                "vt_min_mean": f"{vt_m:.6f}",
                "vt_min_std": f"{vt_s:.6f}",
                "vt_mean_mean": f"{vtmean_m:.6f}",
                "vt_mean_std": f"{vtmean_s:.6f}",
                "energy_loss_mean": f"{en_m:.6f}",
                "energy_loss_std": f"{en_s:.6f}",
                "alpha_max_mean": f"{al_m:.6f}",
                "alpha_max_std": f"{al_s:.6f}",
                "Gmax_mean": f"{g_m:.6f}",
                "Gmax_std": f"{g_s:.6f}",
                "Gmean_mean": f"{gm_m:.6f}",
                "Gmean_std": f"{gm_s:.6f}",
                "CTE_mean_mean": f"{ctem_m:.6f}",
                "CTE_mean_std": f"{ctem_s:.6f}",
                "CTE_p50_mean": f"{ctep50_m:.6f}",
                "CTE_p50_std": f"{ctep50_s:.6f}",
                "CTE_p90_mean": f"{ctep90_m:.6f}",
                "CTE_p90_std": f"{ctep90_s:.6f}",
                "CTE_max_mean": f"{ctemax_m:.6f}",
                "CTE_max_std": f"{ctemax_s:.6f}",
                "velocity_tangent_error_mean": f"{vte_m:.6f}",
                "velocity_tangent_error_std": f"{vte_s:.6f}",
                "nose_tangent_error_mean": f"{nte_m:.6f}",
                "nose_tangent_error_std": f"{nte_s:.6f}",
                "nose_velocity_error_mean": f"{nve_m:.6f}",
                "nose_velocity_error_std": f"{nve_s:.6f}",
                "wing_plane_error_mean": f"{wpe_m:.6f}",
                "wing_plane_error_std": f"{wpe_s:.6f}",
                "q_error_norm_mean": f"{qn_m:.6f}",
                "q_error_norm_std": f"{qn_s:.6f}",
                "pitch_tracking_error_mean": f"{pe_m:.6f}",
                "pitch_tracking_error_std": f"{pe_s:.6f}",
                "altitude_gain_mean": f"{ag_m:.6f}",
                "altitude_gain_std": f"{ag_s:.6f}",
                "altitude_drift_mean": f"{ag_m:.6f}",
                "altitude_drift_std": f"{ag_s:.6f}",
                "altitude_min_mean": f"{altmin_m:.6f}",
                "altitude_min_std": f"{altmin_s:.6f}",
                "altitude_max_mean": f"{altmax_m:.6f}",
                "altitude_max_std": f"{altmax_s:.6f}",
                "env_alpha_range": f"{float(np.min(alpha_min_signed)):.6f}:{float(np.max(alpha_max_signed)):.6f}",
                "target_pitch_range": f"{float(np.min(target_pitch_min)):.6f}:{float(np.max(target_pitch_max)):.6f}",
                "actual_pitch_range": f"{float(np.min(actual_pitch_min)):.6f}:{float(np.max(actual_pitch_max)):.6f}",
                "target_roll_range": f"{float(np.min(target_roll_min)):.6f}:{float(np.max(target_roll_max)):.6f}",
                "actual_roll_range": f"{float(np.min(actual_roll_min)):.6f}:{float(np.max(actual_roll_max)):.6f}",
                "termination_reason": ";".join(f"{k}:{v}" for k, v in sorted(reason_counts.items())),
                "target_signature": target_signature,
            }
            summary_rows.append(summary)
            print(summary)

    per_seed_path = args.out_dir / "eval_rollouts.csv"
    summary_path = args.out_dir / "eval_summary.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_seed_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    config_path = args.out_dir / "config.json"
    incremental_timesteps = None
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            incremental_timesteps = json.load(f).get("TOTAL_TIMESTEPS")

    manifest = {
        "checkpoint_path": str(args.new.resolve()),
        "baseline_checkpoint_path": str(args.baseline.resolve()),
        "residual_checkpoint_path": None
        if args.residual_checkpoint is None
        else str(args.residual_checkpoint.resolve()),
        "residual_epoch": residual_epoch,
        "training_config": str(config_path),
        "reward_weights": {
            "reward_scale": 2.0,
            "attitude": 0.68,
            "speed": 0.24,
            "low_speed_threshold": 180.0,
            "strong_low_speed_threshold": 170.0,
            "alpha_soft_deg": 15.0,
            "alpha_hard_deg": 18.0,
            "g_soft": 9.0,
            "g_hard": 10.0,
            "altitude_retention": "configured in training ENV_PARAMS",
            "altitude_drift_penalty": "configured in training ENV_PARAMS",
        },
        "incremental_timesteps": incremental_timesteps,
        "env_version": "envs/aeroplanax_heading_pitch_V_quaternion_version_vertical_energy.py",
        "eval_summary": str(summary_path),
        "eval_rollouts": str(per_seed_path),
        "suite": args.suite,
        "notes": "Planner proxy tasks here are bottom-policy target generators for old-skill and altitude-drift screening. Claude planner should still run full waypoint/planner rollouts.",
    }
    with (args.out_dir / "checkpoint_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
