"""
render_waypoint_s_quat.py — S-maneuver waypoint tracking for quaternion baseline.

Loads the trained quaternion-version heading/pitch/roll/V baseline, flies an
S-maneuver via waypoint tracking, and outputs:
  1. ACMI file (for Tacview 3D visualization, includes waypoint markers)
  2. PNG performance charts (altitude, speed, attitude tracking, etc.)

Usage:
    conda activate aeroplanax
    cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
    python render_waypoint_s_quat.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import functools
import distrax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
)
from envs.utils.utils import enu_to_geodetic

# ── Network (copied from train_...quaternion...add_roll_target.py) ──
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan, variable_broadcast="params",
        in_axes=0, out_axes=0, split_rngs={"params": False},
    )
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
        pi_speed_brake = distrax.Categorical(logits=nn.Dense(self.action_dim[4], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder, pi_speed_brake), jnp.squeeze(critic, axis=-1)

# ────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────
CKPT_PATH = os.path.abspath(
    "results/heading_pitch_V_discrete_rnn_2026-05-11-02-23/checkpoints/checkpoint_epoch_300"
)
OUTPUT_DIR = "results/waypoint_s_quat"
SEED = 42
MAX_STEPS = 5000

# S-maneuver waypoint params
S_AMPLITUDE        = 5000.0    # east-west half-circle radius (m)
S_HALF_PERIOD_NORTH = 20000.0  # north distance per half-cycle (m)
S_POINTS_PER_HALF  = 50        # waypoints per half-cycle
MAX_WAYPOINTS      = 100        # total waypoints
REACH_RADIUS       = 500.0     # waypoint capture radius (m)
CRUISE_VT          = 250.0     # target airspeed (m/s)

NET_CONFIG = {
    "FC_DIM_SIZE":    128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION":     "relu",
}

# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────
def _f(x, i=0):
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    return float(a.reshape(-1)[min(i, a.size - 1)])


def compute_s_waypoints(origin_n, origin_e, origin_alt):
    """Generate S-curve waypoints (north, east, alt)."""
    dn = S_HALF_PERIOD_NORTH / S_POINTS_PER_HALF
    waypoints = []
    for i in range(1, MAX_WAYPOINTS + 1):
        wp_n = origin_n + i * dn
        wp_e = origin_e + S_AMPLITUDE * np.sin(np.pi * i / S_POINTS_PER_HALF)
        wp_a = origin_alt  # level flight at origin altitude
        waypoints.append((wp_n, wp_e, wp_a))
    return waypoints


def write_acmi_header(acmi_path):
    with open(acmi_path, 'w', encoding='utf-8') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")


def write_acmi_waypoints(acmi_path, waypoints):
    """Batch-write all waypoint markers to ACMI file."""
    WP_ID_BASE = 5000
    with open(acmi_path, 'a', encoding='utf-8') as f:
        for k, (wp_n, wp_e, wp_a) in enumerate(waypoints):
            lat, lon, alt = enu_to_geodetic(wp_e, wp_n, wp_a, 0, 0, 0)
            oid = WP_ID_BASE + k
            f.write(
                f"{oid},Type=Navaid+Static+Waypoint,"
                f"Name=WP_{k},Label={k},Color=Yellow,"
                f"T={float(lon)}|{float(lat)}|{float(alt)}|0|0|0\n"
            )
    print(f"[ACMI] Wrote {len(waypoints)} waypoint markers")


def write_acmi_aircraft(acmi_path, t_sec, north, east, alt, roll_rad, pitch_rad, yaw_rad):
    """Write one aircraft frame to ACMI."""
    roll_d  = float(np.degrees(roll_rad))
    pitch_d = float(np.degrees(pitch_rad))
    yaw_d   = float(np.degrees(yaw_rad))
    lat, lon, alt_m = enu_to_geodetic(east, north, alt, 0, 0, 0)
    with open(acmi_path, 'a', encoding='utf-8') as f:
        f.write(f"#{t_sec:.2f}\n")
        f.write(f"100,T={float(lon)}|{float(lat)}|{float(alt_m)}|"
                f"{roll_d:.2f}|{pitch_d:.2f}|{yaw_d:.2f},"
                f"Type=Air+FixedWing,Name=F16,Color=Cyan\n")


def write_acmi_current_wp(acmi_path, t_sec, wp_n, wp_e, wp_a):
    """Write current target waypoint marker (green) to ACMI."""
    lat, lon, alt = enu_to_geodetic(wp_e, wp_n, wp_a, 0, 0, 0)
    with open(acmi_path, 'a', encoding='utf-8') as f:
        f.write(f"1000,T={float(lon)}|{float(lat)}|{float(alt)}|0|0|0,"
                f"Name=CurrentWP,Color=Green,Type=Navaid+Static+Waypoint\n")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    tag = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── build env ──
    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)

    # ── build network ──
    network = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CONFIG)
    rng = jax.random.PRNGKey(SEED)

    obs_shape = env.observation_space(env.agents[0], env_params).shape
    init_x = (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1)))
    h0 = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    net_params = network.init(rng, h0, init_x)

    # ── load checkpoint ──
    if os.path.isdir(CKPT_PATH):
        print(f"Loading checkpoint: {CKPT_PATH}")
        import orbax.checkpoint as ocp
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        ckpt = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore())
        net_params = ckpt["params"]
        print(f"Restored epoch {int(ckpt['epoch'])}")
    else:
        print(f"WARNING: Checkpoint not found at {CKPT_PATH}, using random network!")

    # ── reset env ──
    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, env_params)

    # ── generate waypoints ──
    origin_n = _f(state.plane_state.north)
    origin_e = _f(state.plane_state.east)
    origin_alt = _f(state.plane_state.altitude)
    waypoints = compute_s_waypoints(origin_n, origin_e, origin_alt)
    print(f"Generated {len(waypoints)} S-curve waypoints, origin=({origin_n:.0f}, {origin_e:.0f}, {origin_alt:.0f})")

    # ── ACMI output ──
    acmi_path = os.path.join(OUTPUT_DIR, f"s_maneuver_quat_{tag}.acmi")
    write_acmi_header(acmi_path)
    write_acmi_waypoints(acmi_path, waypoints)

    # ── init RNN hidden state ──
    hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    done_flag = jnp.zeros((1,))

    # ── record buffers ──
    rec_t       = []; rec_alt = []; rec_vt  = []; rec_roll = []
    rec_pitch   = []; rec_yaw = []; rec_alpha = []; rec_beta = []
    rec_P = []; rec_Q = []; rec_R = []
    rec_thr = []; rec_el = []; rec_ail = []; rec_rud = []
    rec_north = []; rec_east = []
    rec_wp_idx = []; rec_dist = []; rec_heading_err = []; rec_reached = []

    # ── waypoint tracking state ──
    current_wp = 0
    total_reached = 0
    wp_reached_times = []

    dt_rl = env_params.agent_interaction_steps / env_params.sim_freq

    print(f"\n{'Step':>6} | {'WP':>4} | {'Reach':>5} | {'Dist3D':>8} | "
          f"{'Alt':>7} | {'Vt':>6} | {'Roll':>7} | {'Pitch':>7} | {'Yaw':>7} | {'HdgErr':>7}")
    print("-" * 100)

    for step in range(MAX_STEPS):
        ps = state.plane_state
        t_phys = step * dt_rl

        north = _f(ps.north); east = _f(ps.east); alt = _f(ps.altitude)
        vt   = _f(ps.vt)
        roll = _f(ps.roll); pitch = _f(ps.pitch); yaw = _f(ps.yaw)
        alpha = _f(ps.alpha); beta = _f(ps.beta)
        P = _f(ps.P); Q = _f(ps.Q); R = _f(ps.R)

        # ── compute target heading to current waypoint ──
        wp_n, wp_e, wp_a = waypoints[min(current_wp, len(waypoints) - 1)]
        d_n = wp_n - north
        d_e = wp_e - east
        d_alt = wp_a - alt
        h_dist = float(np.sqrt(d_n**2 + d_e**2))
        dist_3d = float(np.sqrt(h_dist**2 + d_alt**2))
        target_heading_raw = float(np.arctan2(d_e, d_n))

        # Smooth approach: gradually move targets from current state toward waypoint
        # This keeps the observation within the training distribution
        # (training targets were always relative to current state via _step_task)
        blend = min(1.0, step / 200.0)  # smooth over first 200 steps
        hdg_err = float(np.arctan2(np.sin(target_heading_raw - yaw),
                                    np.cos(target_heading_raw - yaw)))
        target_heading = float(np.arctan2(np.sin(yaw + blend * hdg_err),
                                           np.cos(yaw + blend * hdg_err)))
        d_alt_clipped = float(np.clip(d_alt, -2000, 2000))
        target_pitch_raw = float(np.arctan2(d_alt_clipped, max(h_dist, 1e-6)))
        target_pitch = float(pitch + blend * (target_pitch_raw - pitch))
        target_roll_raw = float(np.clip(0.5 * hdg_err, -0.5, 0.5))
        roll_err = float(np.arctan2(
                        np.sin(target_roll_raw - roll),
                        np.cos(target_roll_raw - roll)
                    ))
        target_roll = float(np.arctan2(
                        np.sin(roll + blend * roll_err),
                        np.cos(roll + blend * roll_err)
                    ))
        # Smooth target velocity: prevents OOD speed error at episode start
        # (training _step_task always sets target_vt = current_vt + small_delta)
        target_vt = float(vt + blend * (CRUISE_VT - vt))

        # ── check waypoint reached ──
        if dist_3d < REACH_RADIUS and current_wp < len(waypoints):
            current_wp += 1
            total_reached += 1
            wp_reached_times.append(t_phys)

        # ── build observation via env._get_obs with overridden targets ──
        state_with_targets = state.replace(
            target_heading=jnp.array([target_heading]),
            target_pitch=jnp.array([target_pitch]),
            target_roll=jnp.array([target_roll]),
            target_vt=jnp.array([target_vt]),
        )
        obs_dict = env._get_obs(state_with_targets, env_params)
        obs_vec = obs_dict[env.agents[0]]  # (16,)
        obs_in = obs_vec[None, None, :]    # (1, 1, 16)
        done_in = done_flag[None, :]        # (1, 1)

        # ── policy forward (greedy) ──
        hstate, pi, _ = network.apply(net_params, hstate, (obs_in, done_in))
        pi_thr, pi_el, pi_ail, pi_rud, pi_sb = pi

        act_thr = int(pi_thr.mode()[0, 0])
        act_el  = int(pi_el.mode()[0, 0])
        act_ail = int(pi_ail.mode()[0, 0])
        act_rud = int(pi_rud.mode()[0, 0])
        act_sb  = int(pi_sb.mode()[0, 0])

        if step < 3:
            obs_arr = np.array(obs_vec)
            print(f"  [DEBUG step={step}] obs: qv=[{obs_arr[0]:.3f},{obs_arr[1]:.3f},{obs_arr[2]:.3f}] "
                  f"dvt={obs_arr[3]:.3f} v_b=[{obs_arr[6]:.3f},{obs_arr[7]:.3f},{obs_arr[8]:.3f}] "
                  f"PQR=[{obs_arr[9]:.2f},{obs_arr[10]:.2f},{obs_arr[11]:.2f}]")
            print(f"  [DEBUG step={step}] tgt: h={np.degrees(target_heading):.1f}° p={np.degrees(target_pitch):.1f}° "
                  f"r={np.degrees(target_roll):.1f}° v={CRUISE_VT:.0f}")
            print(f"  [DEBUG step={step}] cur: h={np.degrees(yaw):.1f}° p={np.degrees(pitch):.1f}° "
                  f"r={np.degrees(roll):.1f}° v={vt:.1f}")
            print(f"  [DEBUG step={step}] acts: thr={act_thr} el={act_el} ail={act_ail} rud={act_rud} sb={act_sb}")

        action_dict = {env.agents[0]: jnp.array([act_thr, act_el, act_ail, act_rud, act_sb])}

        # ── step env ──
        rng, step_key = jax.random.split(rng)
        obs_dict, state, rew_dict, done_dict, info = env.step(
            step_key, state, action_dict, env_params)

        done_flag = jnp.array([float(done_dict[env.agents[0]])])

        # ── record ──
        rec_t.append(t_phys)
        rec_alt.append(alt); rec_vt.append(vt)
        rec_roll.append(np.degrees(roll)); rec_pitch.append(np.degrees(pitch))
        rec_yaw.append(np.degrees(yaw))
        rec_alpha.append(np.degrees(alpha)); rec_beta.append(np.degrees(beta))
        rec_P.append(P); rec_Q.append(Q); rec_R.append(R)
        rec_north.append(north); rec_east.append(east)
        rec_wp_idx.append(current_wp); rec_dist.append(dist_3d)
        rec_heading_err.append(np.degrees(float(np.arctan2(np.sin(target_heading - yaw),
                                                            np.cos(target_heading - yaw)))))
        rec_reached.append(total_reached)

        # decode actions for recording
        thr_n = act_thr / 30.0
        el_n  = act_el * 2.0 / 40.0 - 1.0
        ail_n = act_ail * 2.0 / 40.0 - 1.0
        rud_n = act_rud * 2.0 / 40.0 - 1.0
        rec_thr.append(thr_n)
        rec_el.append(el_n * 45.0)
        rec_ail.append(ail_n * 45.0)
        rec_rud.append(rud_n * 45.0)

        # ── ACMI frame ──
        write_acmi_aircraft(acmi_path, t_phys, north, east, alt, roll, pitch, yaw)
        write_acmi_current_wp(acmi_path, t_phys, wp_n, wp_e, wp_a)

        # ── log ──
        if step % 100 == 0 or (step > 0 and current_wp > rec_wp_idx[max(0, step-2)]):
            print(f"{step:6d} | {current_wp:4d} | {total_reached:5d} | {dist_3d:8.1f} | "
                  f"{alt:7.0f} | {vt:6.1f} | {np.degrees(roll):+7.1f} | "
                  f"{np.degrees(pitch):+7.1f} | {np.degrees(yaw):+7.1f} | "
                  f"{rec_heading_err[-1]:+7.1f}")

        # ── termination ──
        if bool(done_dict["__all__"]):
            status = int(_f(ps.status))
            reason = "crashed" if status == 2 else "timeout"
            is_alive = bool(_f(ps.is_alive))
            is_crashed = bool(_f(ps.is_crashed))
            ax_val = _f(ps.ax)
            ay_val = _f(ps.ay)
            az_val = _f(ps.az)
            print(f"\n[TERMINATED] step={step}, reason={reason}, wp_reached={total_reached}")
            print(f"  status={status}, is_alive={is_alive}, is_crashed={is_crashed}")
            print(f"  ax={ax_val:.2f}, ay={ay_val:.2f}, az={az_val:.2f}")
            print(f"  alt={alt:.0f}, vt={vt:.1f}, roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")
            break

        if current_wp >= len(waypoints):
            print(f"\n[SUCCESS] All {len(waypoints)} waypoints reached at step {step}!")
            break

    # ────────────────────────────────────────────────────────────────────
    # Matplotlib performance charts
    # ────────────────────────────────────────────────────────────────────
    t = np.array(rec_t)
    n_steps = len(t)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"S-Maneuver Waypoint Tracking — Quaternion Baseline\n"
                 f"Waypoints reached: {total_reached}/{MAX_WAYPOINTS}  |  "
                 f"Duration: {t[-1]:.1f}s  |  Steps: {n_steps}",
                 fontsize=14)
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.5, wspace=0.35)

    # ── Trajectory (top-down) ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(rec_east, rec_north, 'b-', lw=0.8, alpha=0.7, label='Aircraft')
    wp_ns = [w[0] for w in waypoints]
    wp_es = [w[1] for w in waypoints]
    ax.scatter(wp_es, wp_ns, c='orange', s=8, alpha=0.6, label='Waypoints')
    ax.scatter(wp_es[0], wp_ns[0], c='green', s=80, marker='*', label='Start WP')
    ax.scatter(wp_es[-1], wp_ns[-1], c='red', s=80, marker='*', label='End WP')
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Trajectory (top-down view)")
    ax.legend(fontsize=7); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # ── Altitude ──
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, rec_alt, 'b-', lw=1.2, label='Altitude')
    ax.axhline(y=origin_alt, color='gray', ls='--', lw=0.8, label=f'Origin {origin_alt:.0f}m')
    ax.set_ylabel("Altitude (m)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Altitude"); ax.grid(True, alpha=0.3)

    # ── Airspeed ──
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, rec_vt, 'r-', lw=1.2, label='Vt')
    ax.axhline(y=CRUISE_VT, color='gray', ls='--', lw=0.8, label=f'Target {CRUISE_VT} m/s')
    ax.set_ylabel("Airspeed (m/s)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Airspeed"); ax.grid(True, alpha=0.3)

    # ── Roll / Pitch / Yaw ──
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, rec_roll, lw=1.2, label='Roll')
    ax.plot(t, rec_pitch, lw=1.2, label='Pitch')
    ax.plot(t, rec_yaw, lw=0.8, label='Yaw', alpha=0.6)
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_ylabel("Angle (°)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Attitude (Roll / Pitch / Yaw)")
    ax.grid(True, alpha=0.3)

    # ── Alpha / Beta ──
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, rec_alpha, lw=1.2, label='α')
    ax.plot(t, rec_beta, lw=1.2, label='β')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_ylabel("Angle (°)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Aerodynamic angles"); ax.grid(True, alpha=0.3)

    # ── Body rates ──
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, rec_P, lw=1.2, label='P')
    ax.plot(t, rec_Q, lw=1.2, label='Q')
    ax.plot(t, rec_R, lw=1.2, label='R')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_ylabel("Rate (rad/s)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Body angular rates"); ax.grid(True, alpha=0.3)

    # ── Controls ──
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, rec_thr, lw=1.2, label='Throttle')
    ax.set_ylabel("Throttle (0→1)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Throttle"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, rec_el, lw=1.2, label='Elevator')
    ax.plot(t, rec_ail, lw=1.2, label='Aileron')
    ax.plot(t, rec_rud, lw=1.2, label='Rudder')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_ylabel("Deflection (°)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Surface deflections"); ax.grid(True, alpha=0.3)

    # ── Distance to waypoint ──
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t, rec_dist, 'g-', lw=1.2, label='Dist to WP')
    ax.axhline(y=REACH_RADIUS, color='gray', ls='--', lw=0.8, label=f'Reach radius {REACH_RADIUS}m')
    ax.set_ylabel("Distance (m)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Distance to current waypoint"); ax.grid(True, alpha=0.3)

    # ── Waypoint progress ──
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, rec_wp_idx, 'b-', lw=1.5, label='WP index')
    ax.set_ylabel("Waypoint #"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Waypoint progress"); ax.grid(True, alpha=0.3)

    # ── Heading error ──
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, rec_heading_err, 'r-', lw=1.2, label='Heading error')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_ylabel("Error (°)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Heading error to waypoint"); ax.grid(True, alpha=0.3)

    # ── Reached count ──
    ax = fig.add_subplot(gs[3, 2])
    ax.plot(t, rec_reached, 'g-', lw=1.5, label='Total reached')
    if wp_reached_times:
        for rt in wp_reached_times:
            ax.axvline(x=rt, color='orange', ls=':', lw=0.5, alpha=0.5)
    ax.set_ylabel("Count"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.set_title("Cumulative waypoints reached"); ax.grid(True, alpha=0.3)

    # ── 3D trajectory ──
    ax = fig.add_subplot(gs[4, :2], projection='3d')
    ax.plot(rec_east, rec_north, rec_alt, 'b-', lw=0.8, alpha=0.7)
    ax.scatter(wp_es, wp_ns, [w[2] for w in waypoints], c='orange', s=5, alpha=0.4)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Alt (m)")
    ax.set_title("3D S-maneuver trajectory")

    # ── Summary text ──
    ax = fig.add_subplot(gs[4, 2])
    ax.axis('off')
    summary_lines = [
        f"=== S-Maneuver Summary ===",
        f"",
        f"Waypoints: {total_reached}/{MAX_WAYPOINTS} reached",
        f"Duration: {t[-1]:.1f} s",
        f"Steps: {n_steps}",
        f"",
        f"Altitude (m):",
        f"  Mean: {np.mean(rec_alt):.1f}",
        f"  Std:  {np.std(rec_alt):.1f}",
        f"  Min:  {np.min(rec_alt):.1f}",
        f"  Max:  {np.max(rec_alt):.1f}",
        f"",
        f"Vt (m/s):",
        f"  Mean: {np.mean(rec_vt):.1f}",
        f"  Std:  {np.std(rec_vt):.1f}",
        f"",
        f"Roll max: {np.max(np.abs(rec_roll)):.1f}°",
        f"Beta max: {np.max(np.abs(rec_beta)):.1f}°",
        f"Heading err mean: {np.mean(np.abs(rec_heading_err)):.1f}°",
        f"",
        f"Checkpoint: epoch_300",
    ]
    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    # ── Save ──
    png_path = os.path.join(OUTPUT_DIR, f"s_maneuver_quat_{tag}.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n{'='*80}")
    print(f"Render complete!")
    print(f"  ACMI:   {acmi_path}")
    print(f"  Charts: {png_path}")
    print(f"  Waypoints reached: {total_reached}/{MAX_WAYPOINTS}")
    print(f"  Duration: {t[-1]:.1f} s  ({n_steps} steps)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
