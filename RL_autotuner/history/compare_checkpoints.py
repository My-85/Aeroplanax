#!/usr/bin/env python3
"""
compare_checkpoints.py — A/B comparison of two checkpoints on identical waypoint scenarios.

Design:
  - Pre-define a sequence of absolute attitude waypoints (heading/pitch/roll/vt),
    progressively harder. These are fixed, not relative to current state.
  - Each waypoint is held for STEPS_PER_WP steps (or until settled).
  - After every env.step, we re-inject our target into state.env_state so that
    the agent's observation always reflects the correct error.
  - ACMI: all waypoint markers are batch-written to the file header (like
    render_s_maneuver_3d.py), so TacView shows them all as static yellow markers.
    A green "CurrentWP" marker tracks which waypoint is active.
  - Plots: 4 subplots — theta curve, delta_vt curve, settling-time bar, SS-error bar.

Usage:
    cd RL_autotuner/
    python compare_checkpoints.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──
SCRIPT_DIR = Path(__file__).resolve().parent          # RL_autotuner/
PLANAX_DIR = SCRIPT_DIR.parent / "Planax"             # ../Planax/
sys.path.insert(0, str(PLANAX_DIR))

from envs.wrappers import LogWrapper
from envs.aeroplanax_quat_baseline_iter import (
    AeroPlanaxHeading_Pitch_V_Env as AeroPlanaxEnv,
    Heading_Pitch_V_TaskParams as TaskParams,
)
from envs.utils.utils import enu_to_geodetic

# ======================== Quaternion helpers ========================

def _quat_normalize(q):
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)

def _quat_conj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])

def _quat_from_euler_bn(roll, pitch, yaw):
    """ZYX Euler -> q_BN quaternion."""
    cr, sr = jnp.cos(0.5 * roll),  jnp.sin(0.5 * roll)
    cp, sp = jnp.cos(0.5 * pitch), jnp.sin(0.5 * pitch)
    cy, sy = jnp.cos(0.5 * yaw),   jnp.sin(0.5 * yaw)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return jnp.stack([qw, qx, qy, qz], axis=-1)

def _quat_geodesic_angle(q_a, q_b):
    q_a, q_b = _quat_normalize(q_a), _quat_normalize(q_b)
    cos_half = jnp.abs(jnp.sum(q_a * q_b, axis=-1))
    return 2.0 * jnp.arccos(jnp.clip(cos_half, 0.0, 1.0))

# ======================== Network (must match training) ========================

class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params",
                       in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(resets[:, np.newaxis],
                          self.initialize_carry(*carry.shape), carry)
        new_carry, y = nn.GRUCell(features=ins.shape[1])(carry, ins)
        return new_carry, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        act = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        emb = act(nn.Dense(self.config["FC_DIM_SIZE"],
                           kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(obs))
        hidden, emb = ScannedRNN()(hidden, (emb, dones))
        h = act(nn.LayerNorm()(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                                        bias_init=constant(0.0))(emb)))
        am = act(nn.Dense(self.config["GRU_HIDDEN_DIM"],
                          kernel_init=orthogonal(2), bias_init=constant(0.0))(h))
        pis = tuple(
            distrax.Categorical(logits=nn.Dense(d, kernel_init=orthogonal(0.01),
                                                bias_init=constant(0.0))(am))
            for d in self.action_dim
        )
        critic = act(nn.Dense(self.config["FC_DIM_SIZE"],
                              kernel_init=orthogonal(2), bias_init=constant(0.0))(h))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, pis, jnp.squeeze(critic, axis=-1)

# ======================== Config ========================

CONFIG = {
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION": "relu",
    "NUM_ENVS": 1,
    "SEED": 42,
}

# ======================== Waypoints (absolute, fixed) ========================
# Format: (name, heading_deg, pitch_deg, roll_deg, vt_mps)
# Heading 0° = North. Progressively harder.
# Initial state: heading≈0°, pitch≈0°, roll≈0°, vt≈200m/s
WAYPOINTS = [
    # Level 0 — easy single-axis
    ("WP00_H+15",        15,   0,   0, 200),
    ("WP01_H-15",       -15,   0,   0, 200),
    ("WP02_P+8",          0,   8,   0, 200),
    ("WP03_P-8",          0,  -8,   0, 200),
    # Level 1 — moderate
    ("WP04_H+30_P+10",   30,  10,   0, 210),
    ("WP05_H-30_P-10",  -30, -10,   0, 190),
    ("WP06_R+30",         0,   0,  30, 200),
    ("WP07_R-45",         0,   0, -45, 200),
    # Level 2 — hard
    ("WP08_H+60_P+20",   60,  20,   0, 220),
    ("WP09_H-60_P-20",  -60, -20,   0, 180),
    ("WP10_R+90",         0,   0,  90, 200),
    ("WP11_combo_L2",    45,  15,  30, 215),
    # Level 3 — very hard
    ("WP12_H+90_P+30",   90,  30,   0, 230),
    ("WP13_R+135",        0,   0, 135, 200),
    ("WP14_combo_L3",    60,  25,  60, 220),
    ("WP15_H-90_P-30",  -90, -30,   0, 175),
    # Level 4 — extreme
    ("WP16_H+120_P+45", 120,  45,   0, 240),
    ("WP17_R+180",        0,   0, 180, 200),
    ("WP18_combo_L4",    90,  40,  90, 240),
    ("WP19_full",       -120, -45, -90, 170),
]

STEPS_PER_WP = 250     # 250 steps × 0.2s = 50s per waypoint
SETTLE_THRESH_DEG = 8  # consider settled when theta < 8°

# ======================== Checkpoint loading ========================

def load_checkpoint(ckpt_path: str, config: dict):
    env_params = TaskParams()
    env = AeroPlanaxEnv(env_params)
    env = LogWrapper(env)
    num_actors = env.num_agents

    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    rng = jax.random.PRNGKey(0)
    obs_shape = env.observation_space(env.agents[0], env_params).shape

    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * num_actors, *obs_shape)),
        jnp.zeros((1, config["NUM_ENVS"] * num_actors)),
    )
    init_h = ScannedRNN.initialize_carry(
        config["NUM_ENVS"] * num_actors, config["GRU_HIDDEN_DIM"])
    net_params = network.init(rng, init_h, init_x)

    template = {"params": net_params, "epoch": jnp.array(0)}
    ckptr = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    ckpt = ckptr.restore(
        ckpt_path,
        args=ocp.args.PyTreeRestore(item=template, partial_restore=True),
    )
    epoch = int(np.asarray(ckpt.get("epoch", 0)).reshape(-1)[0])
    print(f"  Loaded epoch={epoch}: {ckpt_path}")
    return {
        "params": ckpt["params"], "network": network,
        "env": env, "env_params": env_params, "num_actors": num_actors,
    }

# ======================== ACMI writer ========================

class ACMIWriter:
    """Per-frame ACMI writer for attitude-tracking tasks.

    Objects:
      - ID 100 (Red F16): actual aircraft.
      - ID 1000 (Green Waypoint): current active target, moves each frame.
      - ID 5000+ (Yellow Waypoint): frozen past targets, written once when WP ends.
    """
    ID_PLANE     = 100
    ID_ACTIVE    = 1000   # green current-target waypoint marker
    ID_TGT_TRAIL = 2000   # blue trail marker (no Static → TacView draws trail line)
    ID_HIST_BASE = 5000
    DIST = 2000.0  # metres ahead for active target marker

    def __init__(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.filepath = filepath
        self._hist_count = 0
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.2\n")
            f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")

    def _target_geodetic(self, npos, epos, alt, tgt_heading_rad, tgt_pitch_rad):
        th, tp = tgt_heading_rad, tgt_pitch_rad
        dn = self.DIST * np.cos(tp) * np.cos(th)
        de = self.DIST * np.cos(tp) * np.sin(th)
        da = self.DIST * np.sin(tp)
        return enu_to_geodetic(epos + de, npos + dn, alt + da, 0, 0, 0)

    def write_frame(self, sim_time: float, plane_state,
                    tgt_heading_rad: float, tgt_pitch_rad: float,
                    tgt_roll_rad: float, wp_name: str,
                    skip_plane: bool = False):
        """Write one time-frame. skip_plane=True on done frames to avoid teleport."""
        npos  = float(plane_state.north[0, 0])
        epos  = float(plane_state.east[0, 0])
        alt   = float(plane_state.altitude[0, 0])
        roll  = float(plane_state.roll[0, 0])  * 180.0 / np.pi
        pitch = float(plane_state.pitch[0, 0]) * 180.0 / np.pi
        yaw   = float(plane_state.yaw[0, 0])   * 180.0 / np.pi
        lat, lon, alt_g = enu_to_geodetic(epos, npos, alt, 0, 0, 0)

        tlat, tlon, talt = self._target_geodetic(
            npos, epos, alt, tgt_heading_rad, tgt_pitch_rad)
        troll  = tgt_roll_rad    * 180.0 / np.pi
        tpitch = tgt_pitch_rad   * 180.0 / np.pi
        tyaw   = tgt_heading_rad * 180.0 / np.pi

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"#{sim_time:.2f}\n")
            if not skip_plane:
                f.write(f"{self.ID_PLANE},"
                        f"T={lon}|{lat}|{alt_g}|{roll}|{pitch}|{yaw},"
                        f"Type=Air+FixedWing,Name=F16,Color=Red\n")
            # Active target: green waypoint
            f.write(f"{self.ID_ACTIVE},"
                    f"T={tlon}|{tlat}|{talt},"
                    f"Name=Target[{wp_name}],Color=Green,"
                    f"Type=Navaid+Static+Waypoint\n")
            # Blue trail: same position, no Static tag → TacView connects frames into a line
            f.write(f"{self.ID_TGT_TRAIL},"
                    f"T={tlon}|{tlat}|{talt},"
                    f"Name=TgtTrail,Color=Blue,"
                    f"Type=Navaid+Marker\n")

    def freeze_target(self, sim_time: float, plane_state,
                      tgt_heading_rad: float, tgt_pitch_rad: float, wp_name: str):
        """Freeze the just-completed target as a permanent yellow marker."""
        npos = float(plane_state.north[0, 0])
        epos = float(plane_state.east[0, 0])
        alt  = float(plane_state.altitude[0, 0])
        tlat, tlon, talt = self._target_geodetic(
            npos, epos, alt, tgt_heading_rad, tgt_pitch_rad)
        oid = self.ID_HIST_BASE + self._hist_count
        self._hist_count += 1
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"#{sim_time:.2f}\n")
            f.write(f"{oid},"
                    f"T={tlon}|{tlat}|{talt},"
                    f"Name=Done[{wp_name}],Color=Yellow,"
                    f"Type=Navaid+Static+Waypoint\n")

# ======================== Run one checkpoint ========================

def run_checkpoint(label: str, loaded: dict, config: dict, acmi_path: str):
    """Run all waypoints sequentially. Returns per-step metrics list."""
    network    = loaded["network"]
    params     = loaded["params"]
    env        = loaded["env"]
    env_params = loaded["env_params"]
    num_actors = loaded["num_actors"]
    num_envs   = config["NUM_ENVS"]

    rng = jax.random.PRNGKey(config["SEED"])

    # Reset
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    obsv, state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

    # ── Force fixed initial state: roll=0, pitch=0, yaw=0, vt=200, alt=8000m ──
    # Override the random reset so every run starts from the same clean condition.
    INIT_VT  = 200.0
    INIT_ALT = 8000.0
    ps0 = state.env_state.plane_state
    zero_r = jnp.zeros_like(ps0.roll)
    zero_p = jnp.zeros_like(ps0.pitch)
    zero_h = jnp.zeros_like(ps0.yaw)
    vt0    = jnp.full_like(ps0.vt, INIT_VT)
    alt0   = jnp.full_like(ps0.altitude, INIT_ALT)
    # q_BN for roll=pitch=yaw=0 is identity quaternion [1,0,0,0]
    q0_init = jnp.ones_like(ps0.q0)
    q_zero  = jnp.zeros_like(ps0.q1)
    state = state.replace(env_state=state.env_state.replace(
        plane_state=ps0.replace(
            roll=zero_r, pitch=zero_p, yaw=zero_h,
            vt=vt0, vel_y=vt0, altitude=alt0,
            q0=q0_init, q1=q_zero, q2=q_zero, q3=q_zero,
        )
    ))
    # obsv from reset() will be stale for 1 step, but env.step() immediately
    # re-computes obs from the injected state, so there is no lasting effect.

    hstate = ScannedRNN.initialize_carry(
        num_envs * num_actors, config["GRU_HIDDEN_DIM"])

    acmi = ACMIWriter(acmi_path)

    sim_dt = env_params.agent_interaction_steps / env_params.sim_freq  # 0.2s
    all_metrics = []
    global_step = 0

    print(f"\n  {'Step':>5} | {'WP':>3} | {'Name':<22} | {'theta°':>7} | "
          f"{'dVt':>6} | {'alt':>6} | {'settled'}")
    print(f"  {'-'*72}")

    for wp_idx, (wp_name, h_deg, p_deg, r_deg, vt_mps) in enumerate(WAYPOINTS):
        # Absolute target (fixed, not relative to current state)
        tgt_h = jnp.array(np.radians(h_deg), dtype=jnp.float32)
        tgt_p = jnp.array(np.radians(p_deg), dtype=jnp.float32)
        tgt_r = jnp.array(np.radians(r_deg), dtype=jnp.float32)
        tgt_v = jnp.array(float(vt_mps),     dtype=jnp.float32)

        # Broadcast to env state shape: vmap gives (num_envs, num_agents)
        es_ref = state.env_state
        tgt_h_arr = jnp.broadcast_to(tgt_h, es_ref.target_heading.shape)
        tgt_p_arr = jnp.broadcast_to(tgt_p, es_ref.target_pitch.shape)
        tgt_r_arr = jnp.broadcast_to(tgt_r, es_ref.target_roll.shape)
        tgt_v_arr = jnp.broadcast_to(tgt_v, es_ref.target_vt.shape)

        # Inject target before first step of this waypoint
        state = state.replace(env_state=es_ref.replace(
            target_heading=tgt_h_arr,
            target_pitch=tgt_p_arr,
            target_roll=tgt_r_arr,
            target_vt=tgt_v_arr,
        ))

        settle_step = None  # first step where theta < SETTLE_THRESH_DEG
        last_ps = None  # track last valid plane_state for freeze_target

        for step_in_wp in range(STEPS_PER_WP):
            sim_time = global_step * sim_dt

            # Build obs
            obs_batch = jnp.stack([obsv[a] for a in env.agents])
            obs_batch = obs_batch.reshape((num_actors * num_envs, -1))
            dones_in  = jnp.zeros((num_actors * num_envs,))

            # Forward pass
            ac_in = (obs_batch[np.newaxis, :, :], dones_in[np.newaxis, :])
            hstate, pi, _ = network.apply(params, hstate, ac_in)
            actions = jnp.stack([p.mode() for p in pi], axis=-1).squeeze(0)

            # Per-agent action dict
            actions_dict = {
                agent: actions[i * num_envs : (i + 1) * num_envs]
                for i, agent in enumerate(env.agents)
            }

            # Env step
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_envs)
            obsv, state, rewards, dones_dict, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_rngs, state, actions_dict)

            # Check done BEFORE re-injecting (done means env has auto-reset)
            done_vals = dones_dict[env.agents[0]]
            is_done   = bool(np.asarray(done_vals).any())

            # CRITICAL: re-inject our fixed target after env step
            # (env._step_task may have replaced target on success)
            state = state.replace(env_state=state.env_state.replace(
                target_heading=tgt_h_arr,
                target_pitch=tgt_p_arr,
                target_roll=tgt_r_arr,
                target_vt=tgt_v_arr,
            ))

            # ── Compute metrics directly from physics state ──
            ps = state.env_state.plane_state
            q_curr = jnp.stack([
                jnp.nan_to_num(ps.q0[0, 0], nan=1.0),
                jnp.nan_to_num(ps.q1[0, 0], nan=0.0),
                jnp.nan_to_num(ps.q2[0, 0], nan=0.0),
                jnp.nan_to_num(ps.q3[0, 0], nan=0.0),
            ])
            # q_NB target = conj(q_BN)
            q_tgt = _quat_conj(_quat_from_euler_bn(tgt_r, tgt_p, tgt_h))
            theta_rad = float(_quat_geodesic_angle(q_curr, q_tgt))
            theta_deg = theta_rad * 180.0 / np.pi
            vt_now    = float(jnp.nan_to_num(ps.vt[0, 0], nan=0.0))
            delta_vt  = abs(vt_now - float(tgt_v))
            alt_now   = float(ps.altitude[0, 0])
            roll_now  = float(ps.roll[0, 0])  * 180.0 / np.pi
            pitch_now = float(ps.pitch[0, 0]) * 180.0 / np.pi
            yaw_now   = float(ps.yaw[0, 0])   * 180.0 / np.pi
            reward_v  = float(jnp.mean(rewards[env.agents[0]]))

            if settle_step is None and theta_deg < SETTLE_THRESH_DEG:
                settle_step = step_in_wp

            all_metrics.append({
                "global_step":    global_step,
                "sim_time":       float(sim_time),
                "wp_idx":         wp_idx,
                "wp_name":        wp_name,
                "step_in_wp":     step_in_wp,
                "theta_deg":      theta_deg,
                "delta_vt":       delta_vt,
                "vt":             vt_now,
                "reward":         reward_v,
                "altitude":       alt_now,
                "roll":           roll_now,
                "pitch":          pitch_now,
                "yaw":            yaw_now,
                "tgt_roll":       float(tgt_r) * 180.0 / np.pi,
                "tgt_pitch":      float(tgt_p) * 180.0 / np.pi,
                "tgt_yaw":        float(tgt_h) * 180.0 / np.pi,
                "tgt_vt":         float(tgt_v),
            })

            # ACMI: skip global_step==0 (reset state invalid) and done frames
            if global_step > 0 and global_step % 5 == 0:
                if not is_done:
                    acmi.write_frame(sim_time, ps,
                                     float(tgt_h), float(tgt_p), float(tgt_r),
                                     wp_name)
                    last_ps = ps

            # Reset RNN on env done
            done_bc = jnp.repeat(done_vals, num_actors)
            hstate  = jnp.where(done_bc[:, None], 0.0, hstate)

            global_step += 1

        # Freeze completed waypoint as permanent yellow marker
        if last_ps is not None:
            acmi.freeze_target(sim_time, last_ps,
                               float(tgt_h), float(tgt_p), wp_name)

        # Per-waypoint summary
        wp_data  = [m for m in all_metrics if m["wp_idx"] == wp_idx]
        tail     = wp_data[int(len(wp_data) * 0.75):]
        ss_err   = np.mean([m["theta_deg"] for m in tail]) if tail else 0
        settle_s = (settle_step * sim_dt) if settle_step is not None else STEPS_PER_WP * sim_dt
        settled  = settle_step is not None
        print(f"  {global_step:5d} | {wp_idx:3d} | {wp_name:<22} | "
              f"{ss_err:7.1f}° | {np.mean([m['delta_vt'] for m in tail]):6.1f} | "
              f"{np.mean([m['altitude'] for m in tail]):6.0f} | "
              f"{'YES '+str(round(settle_s,1))+'s' if settled else 'NO'}")

    return all_metrics

# ======================== Plotting ========================

def _add_wp_lines(ax, wp_starts, wp_labels, top=180):
    for xv, xl in zip(wp_starts, wp_labels):
        ax.axvline(x=xv, color="gray", linestyle=":", alpha=0.35, linewidth=0.7)
        ax.text(xv + 0.3, top * 0.97, xl, fontsize=5, rotation=90,
                va="top", color="gray")


def plot_comparison(metrics_a, metrics_b, label_a, label_b, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    COLOR_A = "#2196F3"
    COLOR_B = "#FF9800"

    def ts(m): return [x["sim_time"] for x in m]

    # WP boundaries
    wp_starts, wp_labels = [], []
    for m in metrics_a:
        if m["step_in_wp"] == 0:
            wp_starts.append(m["sim_time"])
            wp_labels.append(m["wp_name"].split("_", 1)[1])

    # Per-WP stats
    n_wp = len(WAYPOINTS)
    settle_a, settle_b, ss_a, ss_b = [], [], [], []
    for wi in range(n_wp):
        for metrics, slist, sslist in [
            (metrics_a, settle_a, ss_a),
            (metrics_b, settle_b, ss_b),
        ]:
            data = [m for m in metrics if m["wp_idx"] == wi]
            st = next((m["step_in_wp"] for m in data
                       if m["theta_deg"] < SETTLE_THRESH_DEG), STEPS_PER_WP)
            slist.append(st * 0.2)
            tail = data[int(len(data) * 0.75):]
            sslist.append(np.mean([m["theta_deg"] for m in tail]) if tail else 0.0)

    x_pos = np.arange(n_wp)
    w = 0.38

    # ── Fig 1: theta ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts(metrics_a), [m["theta_deg"] for m in metrics_a],
            color=COLOR_A, label=label_a, alpha=0.85, linewidth=0.9)
    ax.plot(ts(metrics_b), [m["theta_deg"] for m in metrics_b],
            color=COLOR_B, label=label_b, alpha=0.85, linewidth=0.9)
    ax.axhline(y=SETTLE_THRESH_DEG, color="green", linestyle="--", alpha=0.6,
               label=f"on-target ({SETTLE_THRESH_DEG}°)")
    _add_wp_lines(ax, wp_starts, wp_labels, top=185)
    ax.set_ylim(0, 185); ax.set_ylabel("Theta (deg)"); ax.set_xlabel("Sim time (s)")
    ax.set_title("Geodesic Attitude Error (theta_deg)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig1_theta.png", dpi=150); plt.close()

    # ── Fig 2: delta_vt ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts(metrics_a), [m["delta_vt"] for m in metrics_a],
            color=COLOR_A, label=label_a, alpha=0.85, linewidth=0.9)
    ax.plot(ts(metrics_b), [m["delta_vt"] for m in metrics_b],
            color=COLOR_B, label=label_b, alpha=0.85, linewidth=0.9)
    ax.axhline(y=25, color="green", linestyle="--", alpha=0.6, label="on-target (25 m/s)")
    _add_wp_lines(ax, wp_starts, wp_labels,
                  top=max(max(m["delta_vt"] for m in metrics_a),
                          max(m["delta_vt"] for m in metrics_b)) * 0.95)
    ax.set_ylim(bottom=0); ax.set_ylabel("Delta Vt (m/s)"); ax.set_xlabel("Sim time (s)")
    ax.set_title("Speed Tracking Error (delta_vt)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig2_delta_vt.png", dpi=150); plt.close()

    # ── Fig 3: settling time ──
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x_pos - w/2, settle_a, w, color=COLOR_A, alpha=0.75, label=label_a)
    ax.bar(x_pos + w/2, settle_b, w, color=COLOR_B, alpha=0.75, label=label_b)
    ax.axhline(y=STEPS_PER_WP * 0.2, color="red", linestyle="--", alpha=0.5,
               label=f"max ({STEPS_PER_WP*0.2:.0f}s)")
    ax.set_xticks(x_pos); ax.set_xticklabels(wp_labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Settling Time (s)")
    ax.set_title(f"Settling Time to theta < {SETTLE_THRESH_DEG}°")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig3_settling.png", dpi=150); plt.close()

    # ── Fig 4: steady-state error ──
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x_pos - w/2, ss_a, w, color=COLOR_A, alpha=0.75, label=label_a)
    ax.bar(x_pos + w/2, ss_b, w, color=COLOR_B, alpha=0.75, label=label_b)
    ax.axhline(y=SETTLE_THRESH_DEG, color="green", linestyle="--", alpha=0.5,
               label=f"target ({SETTLE_THRESH_DEG}°)")
    ax.set_xticks(x_pos); ax.set_xticklabels(wp_labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Steady-State Theta Error (deg)")
    ax.set_title("Steady-State Attitude Error (last 25% of WP)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig4_ss_error.png", dpi=150); plt.close()

    # ── Fig 5: yaw tracking ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts(metrics_a), [m["tgt_yaw"]  for m in metrics_a],
            color="black", linestyle="--", linewidth=1.2, label="Target Yaw")
    ax.plot(ts(metrics_a), [m["yaw"]      for m in metrics_a],
            color=COLOR_A, alpha=0.85, linewidth=0.9, label=f"Yaw {label_a}")
    ax.plot(ts(metrics_b), [m["yaw"]      for m in metrics_b],
            color=COLOR_B, alpha=0.85, linewidth=0.9, label=f"Yaw {label_b}")
    _add_wp_lines(ax, wp_starts, wp_labels, top=200)
    ax.set_ylabel("Yaw / Heading (deg)"); ax.set_xlabel("Sim time (s)")
    ax.set_title("Yaw (Heading) Tracking"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig5_yaw.png", dpi=150); plt.close()

    # ── Fig 6: pitch tracking ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts(metrics_a), [m["tgt_pitch"] for m in metrics_a],
            color="black", linestyle="--", linewidth=1.2, label="Target Pitch")
    ax.plot(ts(metrics_a), [m["pitch"]     for m in metrics_a],
            color=COLOR_A, alpha=0.85, linewidth=0.9, label=f"Pitch {label_a}")
    ax.plot(ts(metrics_b), [m["pitch"]     for m in metrics_b],
            color=COLOR_B, alpha=0.85, linewidth=0.9, label=f"Pitch {label_b}")
    _add_wp_lines(ax, wp_starts, wp_labels, top=100)
    ax.set_ylabel("Pitch (deg)"); ax.set_xlabel("Sim time (s)")
    ax.set_title("Pitch Tracking"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig6_pitch.png", dpi=150); plt.close()

    # ── Fig 7: roll tracking ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts(metrics_a), [m["tgt_roll"]  for m in metrics_a],
            color="black", linestyle="--", linewidth=1.2, label="Target Roll")
    ax.plot(ts(metrics_a), [m["roll"]      for m in metrics_a],
            color=COLOR_A, alpha=0.85, linewidth=0.9, label=f"Roll {label_a}")
    ax.plot(ts(metrics_b), [m["roll"]      for m in metrics_b],
            color=COLOR_B, alpha=0.85, linewidth=0.9, label=f"Roll {label_b}")
    _add_wp_lines(ax, wp_starts, wp_labels, top=200)
    ax.set_ylabel("Roll (deg)"); ax.set_xlabel("Sim time (s)")
    ax.set_title("Roll Tracking"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig7_roll.png", dpi=150); plt.close()

    # ── Fig 8: speed tracking ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts(metrics_a), [m["tgt_vt"]   for m in metrics_a],
            color="black", linestyle="--", linewidth=1.2, label="Target Vt")
    ax.plot(ts(metrics_a), [m["vt"]       for m in metrics_a],
            color=COLOR_A, alpha=0.85, linewidth=0.9, label=f"Vt {label_a}")
    ax.plot(ts(metrics_b), [m["vt"]       for m in metrics_b],
            color=COLOR_B, alpha=0.85, linewidth=0.9, label=f"Vt {label_b}")
    _add_wp_lines(ax, wp_starts, wp_labels,
                  top=max(max(m["vt"] for m in metrics_a),
                          max(m["vt"] for m in metrics_b)) * 0.95)
    ax.set_ylabel("Speed (m/s)"); ax.set_xlabel("Sim time (s)")
    ax.set_title("Speed Tracking"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig8_speed.png", dpi=150); plt.close()

    print(f"\nSaved 8 plots to {out_dir}/")

    # ── Console summary ──
    theta_a = [m["theta_deg"] for m in metrics_a]
    theta_b = [m["theta_deg"] for m in metrics_b]
    dvt_a   = [m["delta_vt"]  for m in metrics_a]
    dvt_b   = [m["delta_vt"]  for m in metrics_b]

    print("\n" + "=" * 88)
    print(f"{'WP Name':<24} {'Settle_A':>9} {'Settle_B':>9} {'SS_A':>8} {'SS_B':>8} {'Winner'}")
    print("-" * 88)
    for i, (name, *_) in enumerate(WAYPOINTS):
        short = name.split("_", 1)[1]
        winner = "B" if ss_b[i] < ss_a[i] else ("A" if ss_a[i] < ss_b[i] else "=")
        print(f"{short:<24} {settle_a[i]:>8.1f}s {settle_b[i]:>8.1f}s "
              f"{ss_a[i]:>7.1f}° {ss_b[i]:>7.1f}° {winner}")
    print("=" * 88)
    print(f"\nOverall mean theta:   {label_a}={np.mean(theta_a):.2f}°   "
          f"{label_b}={np.mean(theta_b):.2f}°")
    print(f"Overall mean delta_vt:{label_a}={np.mean(dvt_a):.1f}   "
          f"{label_b}={np.mean(dvt_b):.1f}")
    winner_overall = label_b if np.mean(theta_b) < np.mean(theta_a) else label_a
    print(f"\n★  Overall winner: {winner_overall}")

# ======================== Main ========================

if __name__ == "__main__":
    BASE = SCRIPT_DIR.parent  # 20251215最新代码库

    CKPT_A = str(BASE / "results" / "baseline（四元数版本）" / "checkpoints" / "checkpoint_epoch_1000")
    CKPT_B = str(BASE / "Planax" / "results" / "heading_pitch_V_discrete_rnn_2026-03-20-19-38" / "checkpoints" / "checkpoint_epoch_1350")

    LABEL_A = "baseline_θ24.88"
    LABEL_B = "autotuned_θ20.65"

    OUT_DIR  = str(SCRIPT_DIR / "comparison_results")
    ACMI_DIR = f"{OUT_DIR}/acmi"

    print("=" * 60)
    print("Checkpoint A/B Comparison — Waypoint Tracking")
    print("=" * 60)

    print(f"\nLoading checkpoint A: {LABEL_A}")
    loaded_a = load_checkpoint(CKPT_A, CONFIG)
    print(f"\nLoading checkpoint B: {LABEL_B}")
    loaded_b = load_checkpoint(CKPT_B, CONFIG)

    print(f"\n{'='*60}")
    print(f"Running {LABEL_A}  ({len(WAYPOINTS)} waypoints × {STEPS_PER_WP} steps)")
    t0 = time.time()
    metrics_a = run_checkpoint(LABEL_A, loaded_a, CONFIG,
                               f"{ACMI_DIR}/{LABEL_A}.acmi")
    print(f"  Finished in {time.time()-t0:.1f}s")

    print(f"\n{'='*60}")
    print(f"Running {LABEL_B}  ({len(WAYPOINTS)} waypoints × {STEPS_PER_WP} steps)")
    t0 = time.time()
    metrics_b = run_checkpoint(LABEL_B, loaded_b, CONFIG,
                               f"{ACMI_DIR}/{LABEL_B}.acmi")
    print(f"  Finished in {time.time()-t0:.1f}s")

    # Save raw metrics
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{OUT_DIR}/metrics_A.json", "w") as f:
        json.dump(metrics_a, f, indent=2)
    with open(f"{OUT_DIR}/metrics_B.json", "w") as f:
        json.dump(metrics_b, f, indent=2)
    print(f"\nRaw metrics → {OUT_DIR}/")

    # Plot
    plot_comparison(metrics_a, metrics_b, LABEL_A, LABEL_B, OUT_DIR)

    print(f"\nACMI files → {ACMI_DIR}/")
    print("  (Open either .acmi in TacView: Yellow marker = target attitude, Red = aircraft)")
    print("Done.")
