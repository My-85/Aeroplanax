#!/usr/bin/env python3
"""
evaluate_and_render_baseline.py — Complete evaluation with ACMI rendering and visualization.

Evaluates a checkpoint on 20 fixed waypoints, generates:
  - ACMI file for TacView visualization
  - 9 plots showing tracking performance
  - JSON metrics file

Usage:
    python evaluate_and_render_baseline.py \
        --checkpoint PATH \
        --output-dir OUTPUT_DIR \
        --label LABEL
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_MEM_FRACTION"] = "0.90"

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add Planax to path
SCRIPT_DIR = Path(__file__).resolve().parent
PLANAX_DIR = SCRIPT_DIR.parent / "Planax"
sys.path.insert(0, str(PLANAX_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from envs.utils.utils import enu_to_geodetic

# Import from evaluator
from evaluator import (
    WAYPOINTS, STEPS_PER_WP, SETTLE_THRESH_DEG, LEVEL_VT, LEVEL_ALT,
    EVAL_CONFIG,
    load_checkpoint, ScannedRNN,
    _quat_conj, _quat_from_euler_bn, _quat_geodesic_angle,
    _reset_to_level_flight,
    compute_target_marker_position,
)
import jax
import jax.numpy as jnp

# Constants
WP_SIM_DT = 0.2  # 1 decision step = 0.2s
SYNC_STEPS = 25  # 5s sync at level flight before each WP

# ======================== ACMI Writer ========================

class ACMIWriter:
    """Write ACMI format for TacView visualization."""
    
    ID_PLANE = 100      # Red F16
    ID_ACTIVE = 1000    # Green active target
    ID_TRAIL = 2000     # Blue target trail
    ID_HIST_BASE = 5000 # Yellow frozen WP markers
    
    def __init__(self, path: str):
        self.f = open(path, "w")
        self.ref_lat = 0.0
        self.ref_lon = 0.0
        self.ref_alt = 0.0
        self.hist_id = self.ID_HIST_BASE
        
        # Write header
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.f.write("FileType=text/acmi/tacview\n")
        self.f.write("FileVersion=2.2\n")
        self.f.write(f"0.0,ReferenceTime={now}\n")
        self.f.write(f"0.0,RecordingTime={now}\n")
        self.f.write("\n")
    
    def write_frame(self, sim_time: float, plane_state, tgt_h: float, tgt_p: float, tgt_r: float, wp_name: str):
        """Write one frame with plane and target."""
        self.f.write(f"#{sim_time:.2f}\n")

        # Plane position from real state
        north = float(plane_state.north[0,0])
        east = float(plane_state.east[0,0])
        alt = float(plane_state.altitude[0,0])
        lat, lon, alt_geo = enu_to_geodetic(east, north, alt,
                                            self.ref_lat, self.ref_lon, self.ref_alt)
        roll = float(plane_state.roll[0,0]) * 180.0 / np.pi
        pitch = float(plane_state.pitch[0,0]) * 180.0 / np.pi
        yaw = float(plane_state.yaw[0,0]) * 180.0 / np.pi

        self.f.write(f"{self.ID_PLANE},T={lon}|{lat}|{alt_geo}|{roll}|{pitch}|{yaw},"
                    f"Type=Air+FixedWing,Name=F16,Color=Red\n")

        # Target marker using correct forward direction
        tgt_north, tgt_east, tgt_alt = compute_target_marker_position(
            north, east, alt, tgt_h, tgt_p, distance=2000.0)
        tgt_lat, tgt_lon, tgt_alt_geo = enu_to_geodetic(tgt_east, tgt_north, tgt_alt,
                                                         self.ref_lat, self.ref_lon, self.ref_alt)

        self.f.write(f"{self.ID_ACTIVE},T={tgt_lon}|{tgt_lat}|{tgt_alt_geo},"
                    f"Name=Target[{wp_name}],Color=Green,Type=Navaid+Static+Waypoint\n")
        self.f.write(f"{self.ID_TRAIL},T={tgt_lon}|{tgt_lat}|{tgt_alt_geo},"
                    f"Name=TgtTrail,Color=Blue,Type=Navaid+Marker\n")
    
    def freeze_target(self, sim_time: float, plane_state, tgt_h: float, tgt_p: float, wp_name: str):
        """Freeze completed WP as yellow marker at target position."""
        north = float(plane_state.north[0,0])
        east = float(plane_state.east[0,0])
        alt = float(plane_state.altitude[0,0])

        # Freeze at target point, not plane position
        tgt_north, tgt_east, tgt_alt = compute_target_marker_position(
            north, east, alt, tgt_h, tgt_p, distance=2000.0)
        lat, lon, alt_geo = enu_to_geodetic(tgt_east, tgt_north, tgt_alt,
                                            self.ref_lat, self.ref_lon, self.ref_alt)

        self.f.write(f"#{sim_time:.2f}\n")
        self.f.write(f"{self.hist_id},T={lon}|{lat}|{alt_geo},"
                    f"Name=Done[{wp_name}],Color=Yellow,Type=Navaid+Static+Waypoint\n")
        self.hist_id += 1
    
    def close(self):
        self.f.close()


# ======================== Evaluation ========================

def run_evaluation(loaded: dict, config: dict, acmi_path: str) -> tuple:
    """Run evaluation on all waypoints, return (per-step metrics, per-waypoint summary)."""
    network = loaded["network"]
    params = loaded["params"]
    env = loaded["env"]
    num_actors = loaded["num_actors"]
    num_envs = config["NUM_ENVS"]

    acmi = ACMIWriter(acmi_path)
    all_metrics = []
    wp_summaries = []  # Store per-waypoint summary
    global_sim_time = 0.0
    
    print(f"\n  {'WP':>3} | {'Name':<22} | {'ss_theta°':>9} | {'ss_dvt':>7} | {'settled'}")
    print(f"  {'-'*62}")
    
    for wp_idx, (wp_name, h_deg, p_deg, r_deg, vt_mps) in enumerate(WAYPOINTS):
        tgt_h = jnp.array(np.radians(h_deg), dtype=jnp.float32)
        tgt_p = jnp.array(np.radians(p_deg), dtype=jnp.float32)
        tgt_r = jnp.array(np.radians(r_deg), dtype=jnp.float32)
        tgt_v = jnp.array(float(vt_mps), dtype=jnp.float32)
        
        # Reset to level flight (use fixed seed for reproducibility)
        rng = jax.random.PRNGKey(42)
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        obsv, state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
        # Force level flight (obs stale but step() recomputes it)
        state = _reset_to_level_flight(state, num_envs, num_actors)

        # Fresh RNN state
        hstate = ScannedRNN.initialize_carry(num_envs * num_actors, config["GRU_HIDDEN_DIM"])

        # Helper: set targets
        def _set_targets(st, th, tp, tr, tv):
            es = st.env_state
            return st.replace(env_state=es.replace(
                target_heading=jnp.broadcast_to(th, es.target_heading.shape),
                target_pitch=jnp.broadcast_to(tp, es.target_pitch.shape),
                target_roll=jnp.broadcast_to(tr, es.target_roll.shape),
                target_vt=jnp.broadcast_to(tv, es.target_vt.shape),
            ))

        # SYNC phase: 25 steps at level flight (0,0,0)
        level_h = jnp.array(0.0, dtype=jnp.float32)
        level_p = jnp.array(0.0, dtype=jnp.float32)
        level_r = jnp.array(0.0, dtype=jnp.float32)
        level_v = jnp.array(LEVEL_VT, dtype=jnp.float32)
        state = _set_targets(state, level_h, level_p, level_r, level_v)

        for sync_step in range(SYNC_STEPS):
            obs_batch = jnp.stack([obsv[a] for a in env.agents]).reshape((num_actors * num_envs, -1))
            dones_in = jnp.zeros((num_actors * num_envs,))
            ac_in = (obs_batch[np.newaxis, :, :], dones_in[np.newaxis, :])
            hstate, pi, _ = network.apply(params, hstate, ac_in)
            actions = jnp.stack([p.mode() for p in pi], axis=-1).squeeze(0)
            actions_dict = {agent: actions[i * num_envs : (i + 1) * num_envs]
                           for i, agent in enumerate(env.agents)}
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_envs)
            obsv, state, _, dones_dict, _ = jax.vmap(env.step, in_axes=(0, 0, 0))(step_rngs, state, actions_dict)
            state = _set_targets(state, level_h, level_p, level_r, level_v)

            if sync_step > 0 and sync_step % 5 == 0:
                ps = state.env_state.plane_state
                acmi.write_frame(global_sim_time, ps, 0.0, 0.0, 0.0, f"SYNC_{wp_name}")
            global_sim_time += WP_SIM_DT

        # Switch to WP target
        state = _set_targets(state, tgt_h, tgt_p, tgt_r, tgt_v)

        settle_step = None
        crashed = False
        crash_step = None
        crash_reason = None
        last_ps = None
        prev_ps = None
        wp_metrics = []
        prev_action = None  # Track previous action for change rate

        # Tracking phase
        for step_in_wp in range(STEPS_PER_WP):
            obs_batch = jnp.stack([obsv[a] for a in env.agents]).reshape((num_actors * num_envs, -1))
            dones_in = jnp.zeros((num_actors * num_envs,))
            ac_in = (obs_batch[np.newaxis, :, :], dones_in[np.newaxis, :])
            hstate, pi, _ = network.apply(params, hstate, ac_in)
            actions = jnp.stack([p.mode() for p in pi], axis=-1).squeeze(0)
            actions_dict = {agent: actions[i * num_envs : (i + 1) * num_envs]
                           for i, agent in enumerate(env.agents)}

            # Save state before step (for crash detection)
            prev_ps = state.env_state.plane_state

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_envs)
            obsv, state, rewards, dones_dict, _ = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_rngs, state, actions_dict)

            # Check for crash
            if dones_dict[env.agents[0]][0]:
                crashed = True
                crash_step = step_in_wp
                crash_reason = "substep_event(altitude<2000m or overload>10G)"
                break

            # Re-inject targets
            state = _set_targets(state, tgt_h, tgt_p, tgt_r, tgt_v)

            # Compute metrics
            ps = state.env_state.plane_state

            q_curr = jnp.stack([
                jnp.nan_to_num(ps.q0[0, 0], nan=1.0),
                jnp.nan_to_num(ps.q1[0, 0], nan=0.0),
                jnp.nan_to_num(ps.q2[0, 0], nan=0.0),
                jnp.nan_to_num(ps.q3[0, 0], nan=0.0),
            ])
            q_tgt = _quat_conj(_quat_from_euler_bn(tgt_r, tgt_p, tgt_h))
            theta_deg = float(_quat_geodesic_angle(q_curr, q_tgt)) * 180.0 / np.pi
            vt_now = float(jnp.nan_to_num(ps.vt[0, 0], nan=0.0))
            delta_vt = abs(vt_now - float(tgt_v))
            alt_now = float(ps.altitude[0, 0])
            
            if settle_step is None and theta_deg < SETTLE_THRESH_DEG:
                settle_step = step_in_wp

            # Extract action for this step
            current_action = actions[0]  # First env's action
            action_change = 0.0
            if prev_action is not None:
                action_change = float(jnp.mean(jnp.abs(current_action - prev_action)))
            prev_action = current_action

            m = {
                "wp_idx": wp_idx,
                "wp_name": wp_name,
                "step_in_wp": step_in_wp,
                "sim_time": global_sim_time,
                "theta_deg": theta_deg,
                "delta_vt": delta_vt,
                "action_change": action_change,
                "vt": vt_now,
                "altitude": alt_now,
                "roll": float(ps.roll[0, 0]) * 180.0 / np.pi,
                "pitch": float(ps.pitch[0, 0]) * 180.0 / np.pi,
                "yaw": float(ps.yaw[0, 0]) * 180.0 / np.pi,
                "tgt_roll": float(tgt_r) * 180.0 / np.pi,
                "tgt_pitch": float(tgt_p) * 180.0 / np.pi,
                "tgt_yaw": float(tgt_h) * 180.0 / np.pi,
                "tgt_vt": float(tgt_v),
            }
            wp_metrics.append(m)
            all_metrics.append(m)

            # Write ACMI every 5 steps (skip step 0)
            if step_in_wp > 0 and step_in_wp % 5 == 0:
                acmi.write_frame(global_sim_time, ps, float(tgt_h), float(tgt_p), float(tgt_r), wp_name)
                last_ps = ps
            
            # Reset RNN on done
            done_bc = jnp.repeat(dones_dict[env.agents[0]], num_actors)
            hstate = jnp.where(done_bc[:, None], 0.0, hstate)
            
            global_sim_time += WP_SIM_DT
        
        # Freeze completed WP
        if last_ps is not None:
            acmi.freeze_target(global_sim_time, last_ps, float(tgt_h), float(tgt_p), wp_name)
        
        # Compute ss_theta using last 50 steps (10s)
        # settled = ALL of last 50 steps have theta < 8°

        TAIL_STEPS = 50

        # Handle crash case: if crashed, ss_theta = 180°
        if crashed:
            ss_theta = 180.0
            ss_dvt = float(np.mean([m["delta_vt"] for m in wp_metrics])) if wp_metrics else 100.0
            theta_std = 0.0
            action_change_rate = 0.0
            settled = False
        elif len(wp_metrics) == 0:
            ss_theta = 180.0
            ss_dvt = 100.0
            theta_std = 0.0
            action_change_rate = 0.0
            settled = False
        elif len(wp_metrics) < TAIL_STEPS:
            # Less than 50 steps - use all available data
            ss_theta = float(np.mean([m["theta_deg"] for m in wp_metrics]))
            ss_dvt = float(np.mean([m["delta_vt"] for m in wp_metrics]))
            theta_std = float(np.std([m["theta_deg"] for m in wp_metrics]))
            action_change_rate = float(np.mean([m["action_change"] for m in wp_metrics]))
            # settled = 80% of steps have theta < SETTLE_THRESH_DEG
            on_target_count = sum(1 for m in wp_metrics if m["theta_deg"] < SETTLE_THRESH_DEG)
            settled = on_target_count >= 0.8 * len(wp_metrics)
        else:
            # Normal case: use last 50 steps
            tail = wp_metrics[-TAIL_STEPS:]
            ss_theta = float(np.mean([m["theta_deg"] for m in tail]))
            ss_dvt = float(np.mean([m["delta_vt"] for m in tail]))
            theta_std = float(np.std([m["theta_deg"] for m in tail]))
            action_change_rate = float(np.mean([m["action_change"] for m in tail]))

            # settled = 80% of last 50 steps have theta < SETTLE_THRESH_DEG
            on_target_count = sum(1 for m in tail if m["theta_deg"] < SETTLE_THRESH_DEG)
            settled = on_target_count >= 0.8 * TAIL_STEPS

        status_str = "✓" if settled else "✗"
        if crashed:
            status_str = f"CRASH({crash_reason})"

        print(f"  {wp_idx:3d} | {wp_name:<22} | {ss_theta:9.1f}° | {ss_dvt:7.1f} | {status_str}")

        # Save waypoint summary
        wp_summaries.append({
            "wp_idx": wp_idx,
            "wp_name": wp_name,
            "ss_theta": ss_theta,
            "ss_dvt": ss_dvt,
            "theta_std": theta_std,
            "action_change_rate": action_change_rate,
            "settled": settled,
            "crashed": crashed
        })

        # Mark all metrics for this waypoint with settled flag
        for m in wp_metrics:
            m["settled"] = settled

    acmi.close()
    return all_metrics, wp_summaries


# ======================== Plotting ========================

def _wp_lines(ax, metrics, top):
    """Draw WP boundary lines and labels."""
    seen = set()
    for m in metrics:
        if m["wp_idx"] not in seen:
            seen.add(m["wp_idx"])
            ax.axvline(x=m["sim_time"], color="gray", linestyle=":", alpha=0.35, linewidth=0.7)
            label = m["wp_name"]
            ax.text(m["sim_time"] + 0.3, top * 0.97, label,
                    fontsize=5, rotation=90, va="top", color="gray")


def plot_results(metrics: list, out_dir: str, label: str):
    """Generate 9 plots."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    C = "#2196F3"
    
    n_wp = len(WAYPOINTS)
    settle_times, ss_errors = [], []
    
    # Compute per-WP metrics
    for wi in range(n_wp):
        data = [m for m in metrics if m["wp_idx"] == wi]
        st = next((m["step_in_wp"] for m in data if m["theta_deg"] < SETTLE_THRESH_DEG), STEPS_PER_WP)
        settle_times.append(st * WP_SIM_DT)
        
        # Reverse search for ss_theta
        last_unsettled = None
        for i in range(len(data)-1, -1, -1):
            if data[i]["theta_deg"] >= SETTLE_THRESH_DEG:
                last_unsettled = i
                break
        
        if last_unsettled is not None:
            tail = data[last_unsettled+1:]
        else:
            tail = data
        
        if not tail:
            tail = data[int(len(data) * 0.75):]
        ss_errors.append(np.mean([m["theta_deg"] for m in tail]) if tail else 0.0)
    
    wp_labels = [wp[0] for wp in WAYPOINTS]
    x_pos = np.arange(n_wp)
    ts = [m["sim_time"] for m in metrics]
    
    # Fig1: theta
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(ts, [m["theta_deg"] for m in metrics], color=C, linewidth=0.9)
    ax.axhline(y=SETTLE_THRESH_DEG, color="green", linestyle="--", alpha=0.6,
               label=f"on-target ({SETTLE_THRESH_DEG}°)")
    _wp_lines(ax, metrics, 185)
    ax.set_ylim(0, 185); ax.set_ylabel("Theta (deg)"); ax.set_xlabel("Sim time (s)")
    ax.set_title(f"Geodesic Attitude Error — {label}"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig1_theta.png", dpi=150); plt.close()
    
    # Fig2: delta_vt
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(ts, [m["delta_vt"] for m in metrics], color=C, linewidth=0.9)
    ax.axhline(y=25, color="green", linestyle="--", alpha=0.6, label="on-target (25 m/s)")
    _wp_lines(ax, metrics, max(m["delta_vt"] for m in metrics) * 0.95)
    ax.set_ylim(bottom=0); ax.set_ylabel("Delta Vt (m/s)"); ax.set_xlabel("Sim time (s)")
    ax.set_title(f"Speed Tracking Error — {label}"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig2_delta_vt.png", dpi=150); plt.close()
    
    # Fig3: settling time
    fig, ax = plt.subplots(figsize=(16, 5))
    bar_heights = [STEPS_PER_WP * WP_SIM_DT if ss_errors[i] >= SETTLE_THRESH_DEG else settle_times[i]
                   for i in range(n_wp)]
    colors_bar = ["#F44336" if ss_errors[i] >= SETTLE_THRESH_DEG else C for i in range(n_wp)]
    ax.bar(x_pos, bar_heights, color=colors_bar, alpha=0.8)
    ax.axhline(y=STEPS_PER_WP * WP_SIM_DT, color="red", linestyle="--", alpha=0.5,
               label=f"not settled ({STEPS_PER_WP * WP_SIM_DT:.0f}s)")
    ax.set_xticks(x_pos); ax.set_xticklabels(wp_labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Settling Time (s)"); ax.set_title(f"Settling Time — {label}")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig3_settling.png", dpi=150); plt.close()
    
    # Fig4: ss_error
    fig, ax = plt.subplots(figsize=(16, 5))
    colors_bar = ["#F44336" if e >= SETTLE_THRESH_DEG else C for e in ss_errors]
    ax.bar(x_pos, ss_errors, color=colors_bar, alpha=0.8)
    ax.axhline(y=SETTLE_THRESH_DEG, color="green", linestyle="--", alpha=0.6)
    ax.set_xticks(x_pos); ax.set_xticklabels(wp_labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("SS Theta (deg)"); ax.set_title(f"Steady-State Error — {label}")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig4_ss_error.png", dpi=150); plt.close()

    
    # Fig5: roll tracking
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(ts, [m["roll"] for m in metrics], color=C, linewidth=0.9, label="Actual")
    ax.plot(ts, [m["tgt_roll"] for m in metrics], color="orange", linestyle="--", linewidth=0.9, label="Target")
    _wp_lines(ax, metrics, max(abs(m["roll"]) for m in metrics) * 0.95)
    ax.set_ylabel("Roll (deg)"); ax.set_xlabel("Sim time (s)")
    ax.set_title(f"Roll Tracking — {label}"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig5_roll.png", dpi=150); plt.close()
    
    # Fig6: pitch tracking
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(ts, [m["pitch"] for m in metrics], color=C, linewidth=0.9, label="Actual")
    ax.plot(ts, [m["tgt_pitch"] for m in metrics], color="orange", linestyle="--", linewidth=0.9, label="Target")
    _wp_lines(ax, metrics, max(abs(m["pitch"]) for m in metrics) * 0.95)
    ax.set_ylabel("Pitch (deg)"); ax.set_xlabel("Sim time (s)")
    ax.set_title(f"Pitch Tracking — {label}"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig6_pitch.png", dpi=150); plt.close()
    
    # Fig7: yaw tracking
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(ts, [m["yaw"] for m in metrics], color=C, linewidth=0.9, label="Actual")
    ax.plot(ts, [m["tgt_yaw"] for m in metrics], color="orange", linestyle="--", linewidth=0.9, label="Target")
    _wp_lines(ax, metrics, max(abs(m["yaw"]) for m in metrics) * 0.95)
    ax.set_ylabel("Yaw (deg)"); ax.set_xlabel("Sim time (s)")
    ax.set_title(f"Yaw Tracking — {label}"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig7_yaw.png", dpi=150); plt.close()
    
    # Fig8: speed tracking
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(ts, [m["vt"] for m in metrics], color=C, linewidth=0.9, label="Actual")
    ax.plot(ts, [m["tgt_vt"] for m in metrics], color="orange", linestyle="--", linewidth=0.9, label="Target")
    _wp_lines(ax, metrics, max(m["vt"] for m in metrics) * 0.95)
    ax.set_ylabel("Speed (m/s)"); ax.set_xlabel("Sim time (s)")
    ax.set_title(f"Speed Tracking — {label}"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig8_speed.png", dpi=150); plt.close()
    
    # Fig9: altitude
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(ts, [m["altitude"] for m in metrics], color=C, linewidth=0.9)
    _wp_lines(ax, metrics, max(m["altitude"] for m in metrics) * 0.95)
    ax.set_ylabel("Altitude (m)"); ax.set_xlabel("Sim time (s)")
    ax.set_title(f"Altitude — {label}"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig9_altitude.png", dpi=150); plt.close()


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate and render baseline checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", default="eval_output", help="Output directory")
    parser.add_argument("--label", default=None, help="Label for plots")
    args = parser.parse_args()
    
    label = args.label or Path(args.checkpoint).name
    out_dir = args.output_dir

    config = dict(EVAL_CONFIG)
    config["NUM_ENVS"] = 1

    print(f"Loading checkpoint: {args.checkpoint}")
    t0 = time.time()
    loaded = load_checkpoint(args.checkpoint, config)
    print(f"Loaded in {time.time()-t0:.1f}s (epoch={loaded['epoch']})")
    print(f"Running {len(WAYPOINTS)} waypoints × {STEPS_PER_WP} steps each")

    # Use label in filename to avoid overwriting
    acmi_path = str(Path(out_dir) / f"{label}_result.acmi")
    t0 = time.time()
    metrics, wp_summaries = run_evaluation(loaded, config, acmi_path)
    print(f"\nFinished in {time.time()-t0:.1f}s")
    print(f"ACMI saved to: {acmi_path}")

    # Save metrics
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = f"{out_dir}/{label}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Generate plots
    plot_results(metrics, out_dir, label)
    print(f"Plots saved to: {out_dir}/")
    
    # Print summary using wp_summaries
    ss_thetas = [wp["ss_theta"] for wp in wp_summaries]
    ss_dvts = [wp["ss_dvt"] for wp in wp_summaries]
    theta_stds = [wp["theta_std"] for wp in wp_summaries]
    action_change_rates = [wp["action_change_rate"] for wp in wp_summaries]
    settleds = [wp["settled"] for wp in wp_summaries]
    crasheds = [wp["crashed"] for wp in wp_summaries]

    mean_ss_theta = np.mean(ss_thetas)
    mean_ss_dvt = np.mean(ss_dvts)
    mean_theta_std = np.mean(theta_stds)
    mean_action_change_rate = np.mean(action_change_rates)
    settled_rate = sum(settleds) / len(settleds)
    crash_rate = sum(crasheds) / len(crasheds)

    print("\n" + "="*70)
    print(f"PRIMARY METRICS:")
    print(f"  mean_ss_theta      = {mean_ss_theta:.2f}° (lower is better)")
    print(f"  mean_ss_dvt        = {mean_ss_dvt:.2f} m/s (lower is better)")
    print(f"  crash_rate         = {crash_rate:.1%} (lower is better)")
    print(f"\nSTABILITY METRICS:")
    print(f"  settled_rate       = {settled_rate:.1%} (higher is better)")
    print(f"  mean_theta_std     = {mean_theta_std:.2f}° (lower is better)")
    print(f"  mean_action_change = {mean_action_change_rate:.4f} (lower is better)")
    print("="*70)

if __name__ == "__main__":
    main()
