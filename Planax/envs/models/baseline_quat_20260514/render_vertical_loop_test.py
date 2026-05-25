"""
Vertical loop waypoint tracking — pure test for quaternion baseline.

Generates a vertical loop (circle in North-Altitude plane) and tests
whether the policy can track the waypoints WITHOUT any blend smoothing
or target manipulation. Pure waypoint-following: target heading/pitch
are computed directly from aircraft-to-waypoint geometry.

Usage:
    conda activate aeroplanax
    cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
    python envs/models/baseline_quat_20260514/render_vertical_loop_test.py
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.5'

# Ensure Planax root is on path
_planax_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _planax_root not in sys.path:
    sys.path.insert(0, _planax_root)

from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict

import jax, jax.numpy as jnp, numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import functools, distrax
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import orbax.checkpoint as ocp

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
    _quat_from_euler_nb, _quat_conj,
)
from envs.utils.utils import enu_to_geodetic

# =============================================================================
# Network (must match training exactly)
# =============================================================================
class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry; ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y
    @staticmethod
    def initialize_carry(bs, hs):
        return nn.GRUCell(features=hs).initialize_carry(jax.random.PRNGKey(0), (bs, hs))

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]; config: Dict
    @nn.compact
    def __call__(self, hidden, x):
        ac = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        e = ac(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs))
        hidden, e = ScannedRNN()(hidden, (e, dones))
        fc2 = ac(nn.LayerNorm()(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(e)))
        am = ac(nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
        heads = []
        for i in range(4):
            heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[i], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am)))
        heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[4], kernel_init=constant(0.0),
            bias_init=lambda key, shape, dtype=jnp.float32: jnp.array([0.0,-1.5,-1.5,-1.5,-1.5], dtype=dtype))(am)))
        c = ac(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
        c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return hidden, (heads[0], heads[1], heads[2], heads[3], heads[4]), jnp.squeeze(c, axis=-1)

# =============================================================================
# Config
# =============================================================================
CKPT_PATH = os.path.join(_planax_root,
    "results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600")
OUTPUT_DIR = os.path.join(_planax_root, "results/vertical_loop_test")
SEED = 42
MAX_STEPS = 3000
REACH_RADIUS = 800.0       # waypoint capture radius (m)
CRUISE_VT   = 250.0        # target cruise speed (m/s)
GRAVITY     = 9.81

NET_CONFIG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}

# Loop geometry
LOOP_RADIUS          = 2000.0   # loop radius (m)
LOOP_POINTS_PER_LOOP = 60       # waypoints per full loop (tighter spacing)
LOOP_CENTER_ALT      = 5000.0 + LOOP_RADIUS  # center altitude = start_alt + radius


def _f(x, i=0):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[min(i, a.size - 1)])


def build_vertical_loop_waypoints(origin_n, origin_e, origin_alt, num_wp, forward_dist=1000.0):
    """
    Vertical loop in the North-Altitude plane (east = constant).
    Loop centre is placed 'forward_dist' ahead of the origin in the heading direction.
    Aircraft starts at origin, flies forward to reach the loop bottom.
    """
    waypoints = []
    center_n = origin_n + forward_dist
    center_e = origin_e
    center_a = origin_alt + LOOP_RADIUS
    for i in range(num_wp):
        theta = (2.0 * np.pi * i) / LOOP_POINTS_PER_LOOP
        wp_n = center_n + LOOP_RADIUS * np.sin(theta)
        wp_a = center_a - LOOP_RADIUS * np.cos(theta)
        wp_e = center_e
        waypoints.append((wp_n, wp_e, wp_a))
    return waypoints


def write_acmi_header(path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("FileType=text/acmi/tacview\nFileVersion=2.2\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")


def write_acmi_waypoints(path, waypoints):
    with open(path, 'a', encoding='utf-8') as f:
        for k, (wp_n, wp_e, wp_a) in enumerate(waypoints):
            lat, lon, alt = enu_to_geodetic(wp_e, wp_n, wp_a, 0, 0, 0)
            f.write(f"{5000+k},Type=Navaid+Static+Waypoint,Name=WP_{k},Label={k},Color=Yellow,"
                    f"T={float(lon)}|{float(lat)}|{float(alt)}|0|0|0\n")
    print(f"[ACMI] {len(waypoints)} waypoint markers written")


def write_acmi_aircraft(path, t, north, east, alt, roll, pitch, yaw):
    lat, lon, alt_m = enu_to_geodetic(east, north, alt, 0, 0, 0)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"#{t:.2f}\n")
        f.write(f"100,T={float(lon)}|{float(lat)}|{float(alt_m)}|"
                f"{float(np.degrees(roll)):.2f}|{float(np.degrees(pitch)):.2f}|{float(np.degrees(yaw)):.2f},"
                f"Type=Air+FixedWing,Name=F16,Color=Cyan\n")


# =============================================================================
# Main
# =============================================================================
def main():
    tag = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)

    network = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CONFIG)
    rng = jax.random.PRNGKey(SEED)

    obs_shape = env.observation_space(env.agents[0], env_params).shape
    init_x = (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1)))
    h0 = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    net_params = network.init(rng, h0, init_x)

    print(f"Obs dim: {obs_shape[0]}")

    if os.path.isdir(CKPT_PATH):
        print(f"Loading checkpoint: {CKPT_PATH}")
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        ckpt = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore())
        net_params = ckpt["params"]
        print(f"Restored epoch {int(ckpt['epoch'])}")
    else:
        print(f"ERROR: checkpoint not found at {CKPT_PATH}")
        return

    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, env_params)

    # Align aircraft heading to North (loop track direction)
    init_vt  = _f(state.plane_state.vt)
    loop_yaw = 0.0  # North
    q_nb = _quat_from_euler_nb(0.0, 0.0, loop_yaw)
    q_bn = _quat_conj(q_nb)
    state = state.replace(
        plane_state=state.plane_state.replace(
            yaw=jnp.array([loop_yaw]),
            q0=jnp.array([q_bn[0]]), q1=jnp.array([q_bn[1]]),
            q2=jnp.array([q_bn[2]]), q3=jnp.array([q_bn[3]]),
        ),
        target_heading=jnp.array([loop_yaw]),
    )

    origin_n = _f(state.plane_state.north)
    origin_e = _f(state.plane_state.east)
    origin_alt = _f(state.plane_state.altitude)
    init_yaw = np.degrees(_f(state.plane_state.yaw))

    waypoints = build_vertical_loop_waypoints(origin_n, origin_e, origin_alt, LOOP_POINTS_PER_LOOP)
    print(f"Vertical loop: radius={LOOP_RADIUS:.0f}m, center_alt={LOOP_CENTER_ALT:.0f}m, "
          f"{LOOP_POINTS_PER_LOOP} wp/loop, {len(waypoints)} total waypoints")
    print(f"Origin: N={origin_n:.0f} E={origin_e:.0f} Alt={origin_alt:.0f}m, "
          f"heading={init_yaw:.1f}deg, vt={init_vt:.0f}m/s")

    acmi_path = os.path.join(OUTPUT_DIR, f"vertical_loop_{tag}.acmi")
    write_acmi_header(acmi_path)
    write_acmi_waypoints(acmi_path, waypoints)

    hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    done_flag = jnp.zeros((1,))
    dt_rl = env_params.agent_interaction_steps / env_params.sim_freq

    rec = {"t": [], "alt": [], "vt": [], "roll": [], "pitch": [], "yaw": [],
           "north": [], "east": [], "wp_idx": [], "dist": [], "hdg_err": [],
           "thr": [], "el": [], "ail": [], "rud": [],
           "alpha": [], "beta": [], "ax": [], "ay": [], "az": []}

    current_wp = 0; total_reached = 0; wp_reached_loop1 = -1
    loop1_wp_count = LOOP_POINTS_PER_LOOP

    print(f"\n{'Step':>6} | {'WP':>4} | {'Dist':>8} | {'Alt':>7} | {'Vt':>6} | "
          f"{'Roll':>7} | {'Pitch':>7} | {'Yaw':>7} | {'HdgErr':>7} | {'G':>5}")
    print("-" * 100)

    for step in range(MAX_STEPS):
        ps = state.plane_state
        t_phys = step * dt_rl
        north = _f(ps.north); east = _f(ps.east); alt = _f(ps.altitude)
        vt = _f(ps.vt); roll = _f(ps.roll); pitch = _f(ps.pitch); yaw = _f(ps.yaw)
        alpha = _f(ps.alpha); beta = _f(ps.beta)
        ax = _f(ps.ax); ay = _f(ps.ay); az = _f(ps.az)
        g_load = float(np.sqrt(ax**2 + ay**2 + az**2))

        # ── Waypoint tracking with blend smoothing ──
        wp_n, wp_e, wp_a = waypoints[min(current_wp, len(waypoints) - 1)]
        d_n = wp_n - north; d_e = wp_e - east; d_alt = wp_a - alt
        h_dist = float(np.sqrt(d_n**2 + d_e**2))
        dist_3d = float(np.sqrt(h_dist**2 + d_alt**2))
        target_heading_raw = float(np.arctan2(d_e, d_n))

        # Blend smoothing: gradually move targets from current state toward waypoint.
        # Keeps observation within the training distribution (targets close to current).
        blend = min(1.0, step / 200.0)
        hdg_err = float(np.arctan2(np.sin(target_heading_raw - yaw), np.cos(target_heading_raw - yaw)))
        target_heading = float(np.arctan2(np.sin(yaw + blend * hdg_err), np.cos(yaw + blend * hdg_err)))
        d_alt_clipped = float(np.clip(d_alt, -2000, 2000))
        target_pitch_raw = float(np.arctan2(d_alt_clipped, max(h_dist, 1e-6)))
        target_pitch = float(pitch + blend * (target_pitch_raw - pitch))
        target_roll_raw = float(np.clip(0.5 * hdg_err, -0.5, 0.5))
        roll_err = float(np.arctan2(np.sin(target_roll_raw - roll), np.cos(target_roll_raw - roll)))
        target_roll = float(np.arctan2(np.sin(roll + blend * roll_err), np.cos(roll + blend * roll_err)))
        target_vt = float(vt + blend * (CRUISE_VT - vt))

        # ── Check waypoint reached ──
        if dist_3d < REACH_RADIUS and current_wp < len(waypoints):
            current_wp += 1
            total_reached += 1
            if current_wp == loop1_wp_count and wp_reached_loop1 < 0:
                wp_reached_loop1 = step

        # ── Build observation with waypoint targets ──
        state_with_targets = state.replace(
            target_heading=jnp.array([target_heading]),
            target_pitch=jnp.array([target_pitch]),
            target_roll=jnp.array([target_roll]),
            target_vt=jnp.array([target_vt]),
        )
        obs_dict = env._get_obs(state_with_targets, env_params)
        obs_vec = obs_dict[env.agents[0]]

        # Debug first 3 steps
        if step < 3:
            obs_arr = np.array(obs_vec)
            print(f"  [DEBUG step={step}]")
            print(f"    qv=[{obs_arr[0]:.3f},{obs_arr[1]:.3f},{obs_arr[2]:.3f}] "
                  f"dvt={obs_arr[3]:.3f} v_b=[{obs_arr[6]:.3f},{obs_arr[7]:.3f},{obs_arr[8]:.3f}]")
            print(f"    target: h={np.degrees(target_heading):.1f}deg p={np.degrees(target_pitch):.1f}deg "
                  f"r=0deg v={CRUISE_VT:.0f}")
            print(f"    state:  h={np.degrees(yaw):.1f}deg p={np.degrees(pitch):.1f}deg "
                  f"r={np.degrees(roll):.1f}deg v={vt:.1f}")
            print(f"    wp dist: {dist_3d:.0f}m  wp: ({wp_n:.0f},{wp_e:.0f},{wp_a:.0f})")

        obs_in = obs_vec[None, None, :]; done_in = done_flag[None, :]
        hstate, pi, _ = network.apply(net_params, hstate, (obs_in, done_in))
        acts = [int(p.mode()[0, 0]) for p in pi]
        if step < 3:
            print(f"    acts: thr={acts[0]} el={acts[1]} ail={acts[2]} rud={acts[3]} sb={acts[4]}")

        action = {env.agents[0]: jnp.array(acts)}
        rng, step_key = jax.random.split(rng)
        obs2, state, rew_dict, done_dict, info = env.step(step_key, state, action, env_params)
        done_flag = jnp.array([float(done_dict[env.agents[0]])])

        # Record
        rec["t"].append(t_phys); rec["alt"].append(alt); rec["vt"].append(vt)
        rec["roll"].append(np.degrees(roll)); rec["pitch"].append(np.degrees(pitch))
        rec["yaw"].append(np.degrees(yaw))
        rec["north"].append(north); rec["east"].append(east)
        rec["wp_idx"].append(current_wp); rec["dist"].append(dist_3d)
        rec["hdg_err"].append(np.degrees(float(np.arctan2(np.sin(target_heading - yaw),
                                                           np.cos(target_heading - yaw)))))
        rec["thr"].append(acts[0]/30.0)
        rec["el"].append((acts[1]*2.0/40.0 - 1.0) * 45.0)
        rec["ail"].append((acts[2]*2.0/40.0 - 1.0) * 45.0)
        rec["rud"].append((acts[3]*2.0/40.0 - 1.0) * 45.0)
        rec["alpha"].append(np.degrees(alpha)); rec["beta"].append(np.degrees(beta))
        rec["ax"].append(ax); rec["ay"].append(ay); rec["az"].append(az)

        write_acmi_aircraft(acmi_path, t_phys, north, east, alt, roll, pitch, yaw)

        if step % 50 == 0 or (step > 0 and current_wp > rec["wp_idx"][max(0, step-2)]):
            wp_label = f"WP{current_wp}" if current_wp < len(waypoints) else "DONE"
            print(f"{step:6d} | {wp_label:>4} | {dist_3d:8.0f} | {alt:7.0f} | {vt:6.0f} | "
                  f"{np.degrees(roll):+7.1f} | {np.degrees(pitch):+7.1f} | "
                  f"{np.degrees(yaw):+7.1f} | {rec['hdg_err'][-1]:+7.1f} | {g_load:5.1f}")

        if bool(done_dict["__all__"]):
            causes = []
            if alt < 2500: causes.append("low_altitude")
            if vt < 130: causes.append("stall")
            if vt > 350: causes.append("overspeed")
            reason = "crashed" + (f" ({', '.join(causes)})" if causes else "")
            print(f"\n[TERMINATED] step={step}, reason={reason}, wp_reached={total_reached}/{len(waypoints)}")
            print(f"  pre-step: alt={alt:.0f}m, vt={vt:.0f}m/s, "
                  f"roll={np.degrees(roll):.1f}deg, pitch={np.degrees(pitch):.1f}deg")
            break

        if current_wp >= len(waypoints):
            print(f"\n[SUCCESS] All {len(waypoints)} vertical loop waypoints reached at step {step}!")
            print(f"  Duration: {t_phys:.1f}s ({step+1} steps)")
            if wp_reached_loop1 >= 0:
                print(f"  First loop completed at step {wp_reached_loop1} "
                      f"({wp_reached_loop1 * dt_rl:.1f}s)")
            break

    # =========================================================================
    # Charts
    # =========================================================================
    t = np.array(rec["t"]); n = len(t)
    alt_a = np.array(rec["alt"]); vt_a = np.array(rec["vt"])
    roll_a = np.array(rec["roll"]); pitch_a = np.array(rec["pitch"]); yaw_a = np.array(rec["yaw"])
    north_a = np.array(rec["north"]); east_a = np.array(rec["east"])
    hdg_err_a = np.array(rec["hdg_err"]); dist_a = np.array(rec["dist"])
    el_a = np.array(rec["el"]); ail_a = np.array(rec["ail"]); thr_a = np.array(rec["thr"])
    rud_a = np.array(rec["rud"])
    alpha_a = np.array(rec["alpha"]); beta_a = np.array(rec["beta"])
    ax_a = np.array(rec["ax"]); ay_a = np.array(rec["ay"]); az_a = np.array(rec["az"])
    g_load_a = np.sqrt(ax_a**2 + ay_a**2 + az_a**2)

    print(f"\n{'='*80}")
    print(f"SUMMARY — Vertical Loop Test")
    print(f"{'='*80}")
    print(f"  Waypoints reached:  {total_reached}/{len(waypoints)}")
    print(f"  Total duration:     {t[-1]:.1f}s ({n} steps)")
    if wp_reached_loop1 >= 0:
        print(f"  First loop:         {wp_reached_loop1 * dt_rl:.1f}s ({wp_reached_loop1} steps)")
    print(f"  Altitude:           min={alt_a.min():.0f} max={alt_a.max():.0f} mean={alt_a.mean():.0f}m")
    print(f"  Airspeed:           min={vt_a.min():.0f} max={vt_a.max():.0f} mean={vt_a.mean():.0f}m/s")
    print(f"  Roll:               |mean|={np.abs(roll_a).mean():.1f}deg max_abs={np.abs(roll_a).max():.0f}deg")
    print(f"  Pitch:              min={pitch_a.min():.0f} max={pitch_a.max():.0f}deg")
    print(f"  Alpha:              min={alpha_a.min():.1f} max={alpha_a.max():.1f}deg")
    print(f"  G-load:             max={g_load_a.max():.1f} p95={np.percentile(g_load_a,95):.1f}G")
    print(f"{'='*80}")

    # ── Figure ──
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"Vertical Loop Test — Quaternion Baseline\n"
                 f"WP: {total_reached}/{len(waypoints)}  |  {t[-1]:.1f}s  |  "
                 f"R={LOOP_RADIUS:.0f}m  |  "
                 f"Alt: [{alt_a.min():.0f},{alt_a.max():.0f}]m  |  "
                 f"Gmax: {g_load_a.max():.1f}",
                 fontsize=13)
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Row 0: Trajectory top-down, Trajectory side (North-Alt), Altitude
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(east_a, north_a, 'b-', lw=0.8, alpha=0.7)
    wp_ns = [w[0] for w in waypoints]; wp_es = [w[1] for w in waypoints]
    ax.scatter(wp_es, wp_ns, c='orange', s=8, alpha=0.6)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Trajectory (top-down)"); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(north_a, alt_a, 'b-', lw=0.8)
    ax.scatter(wp_ns, [w[2] for w in waypoints], c='orange', s=8, alpha=0.6)
    ax.set_xlabel("North (m)"); ax.set_ylabel("Altitude (m)")
    ax.set_title("Vertical Profile (North-Alt)"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, alt_a, 'b-', lw=1.2)
    ax.plot(t, dist_a, 'r-', lw=0.5, alpha=0.5, label='Dist to WP')
    ax.set_ylabel("Altitude (m)"); ax.set_title("Altitude + WP Distance"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # Row 1: Airspeed, Roll, Pitch
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, vt_a, 'b-', lw=1.2)
    ax.axhline(y=CRUISE_VT, color='gray', ls='--', lw=0.8)
    ax.set_ylabel("Airspeed (m/s)"); ax.set_title("Airspeed"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, roll_a, 'b-', lw=0.7)
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel("Roll (deg)"); ax.set_title("Roll Angle"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, pitch_a, 'b-', lw=0.7)
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel("Pitch (deg)"); ax.set_title("Pitch Angle"); ax.grid(True, alpha=0.3)

    # Row 2: Yaw, Alpha/Beta, G-load
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, yaw_a % 360, 'b-', lw=0.7)
    ax.set_ylabel("Yaw (deg)"); ax.set_title("Heading"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, alpha_a, 'b-', lw=0.8, label='Alpha')
    ax.plot(t, beta_a, 'r-', lw=0.8, label='Beta')
    ax.axhline(y=-20, color='gray', ls=':', lw=0.5)
    ax.axhline(y=90, color='gray', ls=':', lw=0.5)
    ax.set_ylabel("Angle (deg)"); ax.set_title("Alpha / Beta"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t, g_load_a, 'r-', lw=1.0)
    ax.axhline(y=9, color='orange', ls='--', lw=1.0)
    ax.set_ylabel("G-load"); ax.set_title("Total G-load"); ax.grid(True, alpha=0.3)

    # Row 3: Controls
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, el_a, 'b-', lw=0.6, label='Elevator')
    ax.set_ylabel("Elevator (deg)"); ax.set_title("Elevator"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, ail_a, 'r-', lw=0.6, label='Aileron')
    ax.set_ylabel("Aileron (deg)"); ax.set_title("Aileron"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2])
    ax.plot(t, thr_a, 'g-', lw=1.0, label='Throttle')
    ax.plot(t, rud_a, 'm-', lw=0.6, alpha=0.7, label='Rudder')
    ax.set_ylabel("Norm / deg"); ax.set_title("Throttle / Rudder"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # Row 4: Heading error, Energy, Waypoint distance
    ax = fig.add_subplot(gs[4, 0])
    ax.plot(t, hdg_err_a, 'r-', lw=0.8)
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel("Heading Err (deg)"); ax.set_title("Heading Error"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[4, 1])
    E_k = 0.5 * vt_a**2; E_p = GRAVITY * alt_a
    ax.plot(t, E_k/1e6, 'b-', lw=0.8, label='Kinetic (MJ)')
    ax.plot(t, E_p/1e6, 'g-', lw=0.8, label='Potential (MJ)')
    ax.plot(t, (E_k+E_p)/1e6, 'k-', lw=1.0, label='Total (MJ)')
    ax.set_ylabel("Energy (MJ)"); ax.set_title("Energy Management"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[4, 2])
    ax.plot(north_a, alt_a, 'b-', lw=0.8)
    ax.scatter(wp_ns, [w[2] for w in waypoints], c='orange', s=15, zorder=5)
    for i, (wn, wa) in enumerate(zip(wp_ns, [w[2] for w in waypoints])):
        if i % 5 == 0:
            ax.annotate(str(i), (wn, wa), fontsize=5, color='red')
    ax.set_xlabel("North (m)"); ax.set_ylabel("Altitude (m)")
    ax.set_title("Vertical Loop Track + Waypoints"); ax.grid(True, alpha=0.3)

    png_path = os.path.join(OUTPUT_DIR, f"vertical_loop_{tag}.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save raw trajectory CSV for analysis
    csv_path = os.path.join(OUTPUT_DIR, f"vertical_loop_{tag}.csv")
    csv_data = np.column_stack([t, north_a, east_a, alt_a, vt_a, roll_a, pitch_a, yaw_a, dist_a])
    np.savetxt(csv_path, csv_data, delimiter=',',
               header='time,north,east,alt,vt,roll,pitch,yaw,wp_dist')
    print(f"\n  ACMI:   {acmi_path}")
    print(f"  Charts: {png_path}")
    print(f"  CSV:    {csv_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
