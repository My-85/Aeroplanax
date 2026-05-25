"""
L3 full-envelope maneuver test for quaternion baseline.
Generates detailed debug output and charts for analysis.

Usage:
    python render_waypoint_L3_test.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict
import json

import jax, jax.numpy as jnp, numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import functools, distrax
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
)
from envs.utils.utils import enu_to_geodetic

# ── Network ──
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

# ── Config ──
CKPT_PATH = os.path.abspath(
    "results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600"
)
OUTPUT_DIR = "results/waypoint_L3_test"
SEED = 42
MAX_STEPS = 3000
REACH_RADIUS = 500.0
CRUISE_VT = 250.0
GRAVITY = 9.81

NET_CONFIG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}

def _f(x, i=0):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[min(i, a.size - 1)])

def build_L3_waypoints(origin_n, origin_e, origin_alt, init_yaw):
    """
    Waypoints defined by (turn_angle_deg, distance, alt_delta).
    turn_angle_deg = how many degrees to turn from previous approach heading.
    Positive = right turn, negative = left turn.

    init_yaw determines the direction of the first leg (WP0 is straight ahead).
    """
    yaw0 = float(init_yaw)  # rad

    legs = [
        # (turn_deg, dist_m, alt_delta_m, label)
        # WP0: straight ahead
        (0,     2000,    0, "WP0_cruise"),
        # WP1: RIGHT 90deg turn → perpendicular to approach
        (90,    3000,    0, "WP1_right_90"),
        # WP2: REVERSAL 180deg → fly back opposite direction
        (180,   4000,    0, "WP2_reversal_180"),
        # WP3: LEFT 135deg turn → new direction, CLIMB +2000m
        (-135,  4000, 2000, "WP3_climb"),
        # WP4: straight ahead, DIVE -1500m
        (0,     3000,-1500, "WP4_dive"),
        # WP5: straight ahead, SPEED RUN
        (0,     5000,    0, "WP5_speed"),
        # WP6: RIGHT 135deg turn, CLIMB +1000m (combined maneuver)
        (135,   5000, 1000, "WP6_climb_turn"),
        # WP7: LEFT 90deg turn, DESCEND -1000m
        (-90,   5000,-1000, "WP7_descend_turn"),
        # WP8: RIGHT 135deg turn, return toward origin
        (135,   4000,    0, "WP8_return"),
    ]

    waypoints = []; labels = []
    cur_n, cur_e, cur_a = origin_n, origin_e, origin_alt
    cur_hdg = yaw0  # current approach heading for next leg

    for turn_deg, dist, da, lab in legs:
        cur_hdg = cur_hdg + np.radians(turn_deg)
        cur_n += dist * np.cos(cur_hdg)
        cur_e += dist * np.sin(cur_hdg)
        cur_a += da
        cur_a = float(np.clip(cur_a, 2500, 20000))
        waypoints.append((float(cur_n), float(cur_e), float(cur_a)))
        labels.append(lab)

    return waypoints, labels

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
    print(f"[ACMI] {len(waypoints)} waypoints written")

def write_acmi_aircraft(path, t, north, east, alt, roll, pitch, yaw):
    lat, lon, alt_m = enu_to_geodetic(east, north, alt, 0, 0, 0)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"#{t:.2f}\n")
        f.write(f"100,T={float(lon)}|{float(lat)}|{float(alt_m)}|"
                f"{float(np.degrees(roll)):.2f}|{float(np.degrees(pitch)):.2f}|{float(np.degrees(yaw)):.2f},"
                f"Type=Air+FixedWing,Name=F16,Color=Cyan\n")

# ── Main ──
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
        import orbax.checkpoint as ocp
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        ckpt = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore())
        net_params = ckpt["params"]
        print(f"Restored epoch {int(ckpt['epoch'])}")
    else:
        print(f"WARNING: checkpoint not found at {CKPT_PATH}")

    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, env_params)

    origin_n = _f(state.plane_state.north); origin_e = _f(state.plane_state.east)
    origin_alt = _f(state.plane_state.altitude)
    init_yaw = _f(state.plane_state.yaw)
    waypoints, labels = build_L3_waypoints(origin_n, origin_e, origin_alt, init_yaw)
    print(f"L3 waypoints: {len(waypoints)}, origin=({origin_n:.0f}, {origin_e:.0f}, {origin_alt:.0f})")
    print(f"  Initial heading: {np.degrees(init_yaw):.1f}deg (waypoints rotated to align)")
    for i, (wp, lab) in enumerate(zip(waypoints, labels)):
        print(f"  {lab}: ({wp[0]:.0f}, {wp[1]:.0f}, {wp[2]:.0f})")

    acmi_path = os.path.join(OUTPUT_DIR, f"L3_test_{tag}.acmi")
    write_acmi_header(acmi_path)
    write_acmi_waypoints(acmi_path, waypoints)

    hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    done_flag = jnp.zeros((1,))

    rec = {"t": [], "alt": [], "vt": [], "roll": [], "pitch": [], "yaw": [],
           "north": [], "east": [], "wp_idx": [], "dist": [], "hdg_err": [],
           "thr": [], "el": [], "ail": [], "rud": [], "phase": [],
           "alpha": [], "beta": [], "P": [], "Q": [], "R": [],
           "ax": [], "ay": [], "az": []}

    current_wp = 0; total_reached = 0
    wp_start_step = 0; wp_start_t = 0.0
    wp_stats = []  # per-WP stats
    dt_rl = env_params.agent_interaction_steps / env_params.sim_freq

    print(f"\n{'Step':>6} | {'WP':>3} | {'Phase':>18} | {'Dist':>7} | "
          f"{'Alt':>6} | {'Vt':>5} | {'Roll':>6} | {'Pitch':>6} | "
          f"{'HdgErr':>6} | {'G-load':>6} | {'Energy':>8}")
    print("-" * 110)

    for step in range(MAX_STEPS):
        ps = state.plane_state
        t_phys = step * dt_rl
        north = _f(ps.north); east = _f(ps.east); alt = _f(ps.altitude)
        vt = _f(ps.vt); roll = _f(ps.roll); pitch = _f(ps.pitch); yaw = _f(ps.yaw)
        alpha = _f(ps.alpha); beta = _f(ps.beta)
        P = _f(ps.P); Q = _f(ps.Q); R = _f(ps.R)
        ax = _f(ps.ax); ay = _f(ps.ay); az = _f(ps.az)

        # Energy metrics
        kinetic = 0.5 * vt**2
        potential = GRAVITY * alt
        total_energy = kinetic + potential
        g_load = float(np.sqrt(ax**2 + ay**2 + az**2))

        wp_n, wp_e, wp_a = waypoints[min(current_wp, len(waypoints) - 1)]
        d_n = wp_n - north; d_e = wp_e - east; d_alt = wp_a - alt
        h_dist = float(np.sqrt(d_n**2 + d_e**2))
        dist_3d = float(np.sqrt(h_dist**2 + d_alt**2))
        target_heading_raw = float(np.arctan2(d_e, d_n))

        blend = min(1.0, step / 200.0)
        hdg_err = float(np.arctan2(np.sin(target_heading_raw - yaw), np.cos(target_heading_raw - yaw)))
        target_heading = float(np.arctan2(np.sin(yaw + blend * hdg_err), np.cos(yaw + blend * hdg_err)))
        target_pitch_raw = float(np.arctan2(np.clip(d_alt, -2000, 2000), max(h_dist, 1e-6)))
        target_pitch = float(pitch + blend * (target_pitch_raw - pitch))
        target_roll_raw = float(np.clip(0.5 * hdg_err, -0.5, 0.5))
        roll_err = float(np.arctan2(np.sin(target_roll_raw - roll), np.cos(target_roll_raw - roll)))
        target_roll = float(np.arctan2(np.sin(roll + blend * roll_err), np.cos(roll + blend * roll_err)))
        target_vt = float(vt + blend * (CRUISE_VT - vt))

        state_with_targets = state.replace(
            target_heading=jnp.array([target_heading]),
            target_pitch=jnp.array([target_pitch]),
            target_roll=jnp.array([target_roll]),
            target_vt=jnp.array([target_vt]),
        )
        obs_dict = env._get_obs(state_with_targets, env_params)

        # Debug: first 3 steps dump full obs
        if step < 3:
            obs_vec = np.array(obs_dict[env.agents[0]])
            print(f"  [DEBUG obs step={step}]")
            print(f"    qv=[{obs_vec[0]:.3f},{obs_vec[1]:.3f},{obs_vec[2]:.3f}] dvt={obs_vec[3]:.3f} "
                  f"alt_n={obs_vec[4]:.3f} vt_n={obs_vec[5]:.3f}")
            print(f"    v_b=[{obs_vec[6]:.3f},{obs_vec[7]:.3f},{obs_vec[8]:.3f}] "
                  f"PQR=[{obs_vec[9]:.2f},{obs_vec[10]:.2f},{obs_vec[11]:.2f}]")
            print(f"    alpha=[{obs_vec[12]:.3f},{obs_vec[13]:.3f}] beta=[{obs_vec[14]:.3f},{obs_vec[15]:.3f}]")
            print(f"    prev_act=[thr:{obs_vec[16]:.2f} el:{obs_vec[17]:.2f} ail:{obs_vec[18]:.2f} "
                  f"rud:{obs_vec[19]:.2f} sb:{obs_vec[20]:.2f}]")
            print(f"    step state: yaw={np.degrees(yaw):.1f} pitch={np.degrees(pitch):.1f} roll={np.degrees(roll):.1f} "
                  f"vt={vt:.1f} alt={alt:.0f} alpha={np.degrees(alpha):.1f} beta={np.degrees(beta):.1f}")

        obs_in = obs_dict[env.agents[0]][None, None, :]
        done_in = done_flag[None, :]

        hstate, pi, _ = network.apply(net_params, hstate, (obs_in, done_in))
        acts = [int(p.mode()[0, 0]) for p in pi]

        # Debug: first 5 steps dump logits
        if step < 5:
            thr_logits = np.array(pi[0].logits[0, 0])
            thr_top3 = np.argsort(thr_logits)[-3:][::-1]
            el_logits = np.array(pi[1].logits[0, 0])
            el_top3 = np.argsort(el_logits)[-3:][::-1]
            ail_logits = np.array(pi[2].logits[0, 0])
            ail_top3 = np.argsort(ail_logits)[-3:][::-1]
            rud_logits = np.array(pi[3].logits[0, 0])
            rud_top3 = np.argsort(rud_logits)[-3:][::-1]
            print(f"    acts: thr={acts[0]}({thr_top3}) el={acts[1]}({el_top3}) "
                  f"ail={acts[2]}({ail_top3}) rud={acts[3]}({rud_top3}) sb={acts[4]}")

        action = {env.agents[0]: jnp.array(acts)}
        rng, step_key = jax.random.split(rng)
        obs2, state, rew_dict, done_dict, info = env.step(step_key, state, action, env_params)
        done_flag = jnp.array([float(done_dict[env.agents[0]])])

        phase_label = labels[min(current_wp, len(labels) - 1)]
        # Check waypoint
        wp_reached_now = False
        if dist_3d < REACH_RADIUS and current_wp < len(waypoints):
            wp_reached_now = True
            steps_for_wp = step - wp_start_step
            time_for_wp = t_phys - wp_start_t
            wp_speed_min = float(np.min(rec["vt"][wp_start_step:]) if step > wp_start_step else vt)
            wp_speed_max = float(np.max(rec["vt"][wp_start_step:]) if step > wp_start_step else vt)
            wp_roll_max = float(np.max(np.abs(rec["roll"][wp_start_step:])) if step > wp_start_step else 0)
            wp_stats.append({
                "wp": current_wp, "label": phase_label,
                "steps": steps_for_wp, "time_s": round(time_for_wp, 1),
                "alt_min": float(np.min(rec["alt"][wp_start_step:]) if step > wp_start_step else alt),
                "alt_max": float(np.max(rec["alt"][wp_start_step:]) if step > wp_start_step else alt),
                "speed_min": wp_speed_min, "speed_max": wp_speed_max,
                "roll_max_deg": round(wp_roll_max, 1),
            })
            current_wp += 1
            total_reached += 1
            wp_start_step = step
            wp_start_t = t_phys

        rec["t"].append(t_phys); rec["alt"].append(alt); rec["vt"].append(vt)
        rec["roll"].append(np.degrees(roll)); rec["pitch"].append(np.degrees(pitch))
        rec["yaw"].append(np.degrees(yaw))
        rec["north"].append(north); rec["east"].append(east)
        rec["wp_idx"].append(current_wp); rec["dist"].append(dist_3d)
        rec["hdg_err"].append(np.degrees(hdg_err))
        rec["thr"].append(acts[0]/30.0)
        rec["el"].append((acts[1]*2.0/40.0 - 1.0) * 45.0)
        rec["ail"].append((acts[2]*2.0/40.0 - 1.0) * 45.0)
        rec["rud"].append((acts[3]*2.0/40.0 - 1.0) * 45.0)
        rec["phase"].append(phase_label)
        rec["alpha"].append(np.degrees(alpha)); rec["beta"].append(np.degrees(beta))
        rec["P"].append(np.degrees(P)); rec["Q"].append(np.degrees(Q)); rec["R"].append(np.degrees(R))
        rec["ax"].append(ax); rec["ay"].append(ay); rec["az"].append(az)

        write_acmi_aircraft(acmi_path, t_phys, north, east, alt, roll, pitch, yaw)

        if step % 50 == 0 or (step > 0 and current_wp > rec["wp_idx"][max(0, step-2)]):
            print(f"{step:6d} | {current_wp:3d} | {phase_label:>18} | {dist_3d:7.0f} | "
                  f"{alt:6.0f} | {vt:5.0f} | {np.degrees(roll):+6.1f} | "
                  f"{np.degrees(pitch):+6.1f} | {np.degrees(hdg_err):+6.1f} | "
                  f"{g_load:6.1f} | {total_energy/1e6:8.3f}")

        if bool(done_dict["__all__"]):
            causes = []
            if alt < 2500: causes.append("low_altitude")
            if vt < 130: causes.append("stall")
            if vt > 350: causes.append("overspeed")
            if abs(np.degrees(roll)) > 80: causes.append("extreme_roll")
            if abs(np.degrees(pitch)) > 55: causes.append("extreme_pitch")
            reason = "crashed" + (f" ({', '.join(causes)})" if causes else "")
            print(f"\n[TERMINATED] step={step}, reason={reason}, wp_reached={total_reached}/{len(waypoints)}")
            print(f"  pre-step: alt={alt:.0f}m, vt={vt:.0f}m/s, "
                  f"roll={np.degrees(roll):.1f}deg, pitch={np.degrees(pitch):.1f}deg")
            print(f"  phase: {phase_label}, dist_to_wp: {dist_3d:.0f}m")
            break

        if current_wp >= len(waypoints):
            print(f"\n[SUCCESS] All {len(waypoints)} L3 waypoints reached at step {step}!")
            print(f"  Duration: {t_phys:.1f}s ({step+1} steps)")
            break

    # ── Summary statistics ──
    t = np.array(rec["t"]); n = len(t)
    alt_a = np.array(rec["alt"]); vt_a = np.array(rec["vt"])
    roll_a = np.array(rec["roll"]); pitch_a = np.array(rec["pitch"])
    yaw_a = np.array(rec["yaw"]); hdg_err_a = np.array(rec["hdg_err"])
    thr_a = np.array(rec["thr"]); el_a = np.array(rec["el"]); ail_a = np.array(rec["ail"])
    rud_a = np.array(rec["rud"])
    alpha_a = np.array(rec["alpha"]); beta_a = np.array(rec["beta"])
    P_a = np.array(rec["P"]); Q_a = np.array(rec["Q"]); R_a = np.array(rec["R"])
    ax_a = np.array(rec["ax"]); ay_a = np.array(rec["ay"]); az_a = np.array(rec["az"])
    g_load_a = np.sqrt(ax_a**2 + ay_a**2 + az_a**2)

    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"  Waypoints reached:  {total_reached}/{len(waypoints)}")
    print(f"  Total duration:     {t[-1]:.1f}s ({n} steps)")
    print(f"  Altitude:           min={alt_a.min():.0f}  max={alt_a.max():.0f}  mean={alt_a.mean():.0f}  std={alt_a.std():.0f}m")
    print(f"  Airspeed:           min={vt_a.min():.0f}  max={vt_a.max():.0f}  mean={vt_a.mean():.0f}  std={vt_a.std():.0f}m/s")
    print(f"  Roll:               min={roll_a.min():.0f}  max={roll_a.max():.0f}  |mean|={np.abs(roll_a).mean():.1f}deg")
    print(f"  Pitch:              min={pitch_a.min():.0f}  max={pitch_a.max():.0f}  |mean|={np.abs(pitch_a).mean():.1f}deg")
    print(f"  Alpha:              min={alpha_a.min():.1f}  max={alpha_a.max():.1f}deg")
    print(f"  Beta:               min={beta_a.min():.1f}  max={beta_a.max():.1f}deg")
    print(f"  G-load:             min={g_load_a.min():.1f}  max={g_load_a.max():.1f}  p95={np.percentile(g_load_a,95):.1f}G")
    print(f"  Heading error:      mean={hdg_err_a.mean():.1f}  p95={np.percentile(hdg_err_a,95):.1f}deg")
    print(f"  Elevator:           |mean|={np.abs(el_a).mean():.1f}  p95={np.percentile(np.abs(el_a),95):.1f}deg")
    print(f"  Aileron:            |mean|={np.abs(ail_a).mean():.1f}  p95={np.percentile(np.abs(ail_a),95):.1f}deg")
    print(f"  Throttle:           mean={thr_a.mean():.2f}")
    print(f"  Roll rate P:        |mean|={np.abs(P_a).mean():.1f}  max={np.abs(P_a).max():.1f}deg/s")
    print(f"  Pitch rate Q:       |mean|={np.abs(Q_a).mean():.1f}  max={np.abs(Q_a).max():.1f}deg/s")
    print(f"  Yaw rate R:         |mean|={np.abs(R_a).mean():.1f}  max={np.abs(R_a).max():.1f}deg/s")

    if total_reached > 0:
        print(f"\n  Per-waypoint timing:")
        for s in wp_stats:
            print(f"    {s['label']:>20}: {s['steps']:4d} steps ({s['time_s']:5.1f}s)  "
                  f"alt=[{s['alt_min']:.0f},{s['alt_max']:.0f}]m  "
                  f"spd=[{s['speed_min']:.0f},{s['speed_max']:.0f}]m/s  "
                  f"|roll|_max={s['roll_max_deg']:.0f}deg")

    # Phase quality scores
    if total_reached >= 3 and n > 0:
        wp_idx_arr = np.array(rec["wp_idx"])
        cruise_mask = (wp_idx_arr == 0)
        cruise_alt_std = alt_a[cruise_mask].std() if cruise_mask.sum() > 1 else 0.0
        print(f"\n  Quality metrics (lower = better):")
        print(f"    Cruise alt stability (std): {cruise_alt_std:.1f}m")
        print(f"    Speed tracking error (|V-250| mean): {np.abs(vt_a - 250).mean():.1f}m/s")
        print(f"    Control smoothness (|delta_el| mean): {np.abs(np.diff(el_a)).mean():.1f}deg/step")
        print(f"    Coordinated turn (|beta| mean): {np.abs(beta_a).mean():.1f}deg")
        print(f"    Energy efficiency (dE/dt mean): {np.abs(np.diff(0.5*vt_a**2 + GRAVITY*alt_a)).mean():.0f}J/s")

    print(f"{'='*80}")

    # ── Charts ──
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(f"L3 Full-Envelope Maneuver Test — Quaternion Baseline\n"
                 f"WP reached: {total_reached}/{len(waypoints)}  |  {t[-1]:.1f}s  |  "
                 f"Alt: [{alt_a.min():.0f}, {alt_a.max():.0f}]m  |  "
                 f"Vt: [{vt_a.min():.0f}, {vt_a.max():.0f}]m/s  |  "
                 f"Gmax: {g_load_a.max():.1f}",
                 fontsize=13)

    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Row 0: Trajectory, Altitude, Speed
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(rec["east"], rec["north"], 'b-', lw=0.8, alpha=0.7)
    wp_ns = [w[0] for w in waypoints]; wp_es = [w[1] for w in waypoints]
    ax.scatter(wp_es, wp_ns, c='orange', s=30, zorder=5)
    ax.scatter(wp_es[0], wp_ns[0], c='green', s=120, marker='*', zorder=6, label='Start')
    for i, lab in enumerate(labels):
        ax.annotate(lab.replace('WP','').split('_')[0], (wp_es[i], wp_ns[i]), fontsize=5, color='red')
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Trajectory (top-down)"); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, alt_a, 'b-', lw=1.2)
    for wp, lab in zip(waypoints, labels):
        ax.axhline(y=wp[2], color='gray', ls=':', lw=0.4)
    ax.set_ylabel("Altitude (m)"); ax.set_title("Altitude Profile"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, vt_a, 'b-', lw=1.2)
    ax.axhline(y=250, color='gray', ls='--', lw=0.8, label='Target 250')
    ax.fill_between(t, 200, 300, alpha=0.1, color='green', label='Safe band')
    ax.set_ylabel("Airspeed (m/s)"); ax.set_title("Airspeed"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # Row 1: Attitude (Roll, Pitch, Yaw)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, roll_a, 'b-', lw=0.7)
    ax.fill_between(t, -80, 80, alpha=0.05, color='red')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel("Roll (deg)"); ax.set_title("Roll Angle (±80° safe)"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, pitch_a, 'b-', lw=0.7)
    ax.fill_between(t, -55, 55, alpha=0.05, color='red')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel("Pitch (deg)"); ax.set_title("Pitch Angle (±55° safe)"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, yaw_a, 'b-', lw=0.7)
    ax.set_ylabel("Yaw (deg)"); ax.set_title("Heading (Yaw)"); ax.grid(True, alpha=0.3)

    # Row 2: Alpha/Beta, Angular Rates, G-load
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, alpha_a, 'b-', lw=0.8, label='Alpha')
    ax.plot(t, beta_a, 'r-', lw=0.8, label='Beta')
    ax.fill_between(t, -20, 90, alpha=0.03, color='green', label='Aero table range')
    ax.axhline(y=-20, color='gray', ls=':', lw=0.5)
    ax.axhline(y=90, color='gray', ls=':', lw=0.5)
    ax.set_ylabel("Angle (deg)"); ax.set_title("Alpha / Beta (aero table limits)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, P_a, 'b-', lw=0.6, alpha=0.7, label='P (roll rate)')
    ax.plot(t, Q_a, 'r-', lw=0.6, alpha=0.7, label='Q (pitch rate)')
    ax.plot(t, R_a, 'g-', lw=0.6, alpha=0.7, label='R (yaw rate)')
    ax.set_ylabel("Rate (deg/s)"); ax.set_title("Angular Rates P/Q/R"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t, g_load_a, 'r-', lw=1.0)
    ax.axhline(y=9, color='orange', ls='--', lw=1.0, label='G-limit (9G)')
    ax.axhline(y=15, color='red', ls=':', lw=1.0, label='Hard cap (15G)')
    ax.set_ylabel("Total G-load"); ax.set_title("G-load (|accel|)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # Row 3: Control surfaces
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, el_a, 'b-', lw=0.6, label='Elevator')
    ax.fill_between(t, -45, 45, alpha=0.05, color='gray')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel("Elevator (deg)"); ax.set_title("Elevator"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, ail_a, 'r-', lw=0.6, label='Aileron')
    ax.fill_between(t, -45, 45, alpha=0.05, color='gray')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel("Aileron (deg)"); ax.set_title("Aileron"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2])
    ax.plot(t, thr_a, 'g-', lw=1.0, label='Throttle')
    ax.plot(t, rud_a, 'm-', lw=0.6, alpha=0.7, label='Rudder')
    ax.set_ylabel("Normalized"); ax.set_title("Throttle (0-1) / Rudder (deg)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    # Row 4: Heading Error, Energy, Combined attitude
    ax = fig.add_subplot(gs[4, 0])
    ax.plot(t, hdg_err_a, 'r-', lw=0.8)
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel("Heading Error (deg)"); ax.set_title("Heading Error (deg)"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[4, 1])
    E_k = 0.5 * vt_a**2
    E_p = GRAVITY * alt_a
    ax.plot(t, E_k/1e6, 'b-', lw=0.8, label='Kinetic (MJ)')
    ax.plot(t, E_p/1e6, 'g-', lw=0.8, label='Potential (MJ)')
    ax.plot(t, (E_k+E_p)/1e6, 'k-', lw=1.0, label='Total (MJ)')
    ax.set_ylabel("Energy (MJ)"); ax.set_title("Energy Management"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[4, 2])
    ax.plot(t, np.abs(el_a), 'b-', lw=0.4, alpha=0.5, label='|Elevator|')
    ax.plot(t, np.abs(ail_a), 'r-', lw=0.4, alpha=0.5, label='|Aileron|')
    ax.set_ylabel("|Control| (deg)"); ax.set_title("Control Effort"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # Row 5: Full timeline with phase markers
    ax = fig.add_subplot(gs[5, :])
    ax.plot(t, roll_a, lw=0.5, alpha=0.6, label='Roll')
    ax.plot(t, pitch_a, lw=0.5, alpha=0.6, label='Pitch')
    ax.plot(t, yaw_a % 360, lw=0.5, alpha=0.6, label='Yaw (mod 360)')
    # Phase transition lines
    prev_wp = -1
    for i in range(n):
        wp_i = rec["wp_idx"][i]
        if wp_i != prev_wp and wp_i < len(labels):
            ax.axvline(x=t[i], color='orange', ls=':', lw=0.8)
            ax.annotate(labels[wp_i].replace('WP',''), (t[i], ax.get_ylim()[1]*0.9),
                        fontsize=5, rotation=90, color='red')
            prev_wp = wp_i
    ax.set_ylabel("deg"); ax.set_title("Full Attitude Timeline with Phase Markers")
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)

    png_path = os.path.join(OUTPUT_DIR, f"L3_test_{tag}.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save stats as JSON for programmatic analysis
    stats_path = os.path.join(OUTPUT_DIR, f"L3_stats_{tag}.json")
    export = {
        "waypoints_total": len(waypoints),
        "waypoints_reached": total_reached,
        "total_steps": int(n),
        "total_time_s": float(t[-1]),
        "altitude": {"min": float(alt_a.min()), "max": float(alt_a.max()),
                      "mean": float(alt_a.mean()), "std": float(alt_a.std())},
        "airspeed": {"min": float(vt_a.min()), "max": float(vt_a.max()),
                      "mean": float(vt_a.mean()), "std": float(vt_a.std())},
        "roll": {"min": float(roll_a.min()), "max": float(roll_a.max()),
                  "abs_mean": float(np.abs(roll_a).mean())},
        "pitch": {"min": float(pitch_a.min()), "max": float(pitch_a.max()),
                   "abs_mean": float(np.abs(pitch_a).mean())},
        "alpha": {"min": float(alpha_a.min()), "max": float(alpha_a.max())},
        "beta": {"min": float(beta_a.min()), "max": float(beta_a.max())},
        "g_load": {"min": float(g_load_a.min()), "max": float(g_load_a.max()),
                    "p95": float(np.percentile(g_load_a, 95))},
        "heading_error": {"mean": float(hdg_err_a.mean()),
                           "p95": float(np.percentile(hdg_err_a, 95))},
        "angular_rates": {
            "P_abs_mean": float(np.abs(P_a).mean()), "P_max": float(np.abs(P_a).max()),
            "Q_abs_mean": float(np.abs(Q_a).mean()), "Q_max": float(np.abs(Q_a).max()),
            "R_abs_mean": float(np.abs(R_a).mean()), "R_max": float(np.abs(R_a).max()),
        },
        "per_waypoint": wp_stats,
    }
    with open(stats_path, 'w') as f:
        json.dump(export, f, indent=2)

    print(f"\n  ACMI:       {acmi_path}")
    print(f"  Charts:     {png_path}")
    print(f"  Stats JSON: {stats_path}")
    print(f"\n{'='*80}")
    print("HOW TO SHARE RESULTS:")
    print(f"  1. Send the PNG:  {png_path}")
    print(f"  2. Send the JSON: {stats_path}")
    print(f"  3. (Optional) Send the ACMI for Tacview: {acmi_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
