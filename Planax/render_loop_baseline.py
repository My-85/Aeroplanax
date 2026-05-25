"""
render_loop_baseline.py — Loop maneuver waypoint tracking for waypoint baseline.

Generates a vertical loop (radius 8000m) in the north-altitude plane,
loads the trained waypoint baseline, and flies through the waypoints.

Outputs:
  1. ACMI file (for Tacview 3D visualization)
  2. PNG performance charts
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import functools
import distrax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from envs.aeroplanax_waypoint import (
    AeroPlanaxWaypointEnv, WaypointTaskParams,
)
from envs.utils.utils import enu_to_geodetic, wrap_PI


# ── Network (same as training) ──
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan, variable_broadcast="params",
        in_axes=0, out_axes=0, split_rngs={"params": False},
    )
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
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(obs)
        embedding = activation(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        fc2 = nn.Dense(256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                       bias_init=nn.initializers.constant(0.0))(embedding)
        fc2 = nn.LayerNorm()(fc2)
        fc2 = activation(fc2)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=nn.initializers.orthogonal(2),
            bias_init=nn.initializers.constant(0.0),
        )(fc2)
        actor_mean = activation(actor_mean)
        pi_throttle = distrax.Categorical(logits=nn.Dense(
            self.action_dim[0], kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(logits=nn.Dense(
            self.action_dim[1], kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_aileron = distrax.Categorical(logits=nn.Dense(
            self.action_dim[2], kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_rudder = distrax.Categorical(logits=nn.Dense(
            self.action_dim[3], kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_speed_brake = distrax.Categorical(logits=nn.Dense(
            self.action_dim[4], kernel_init=nn.initializers.constant(0.0),
            bias_init=lambda key, shape, dtype=jnp.float32: jnp.array(
                [0.0, -1.5, -1.5, -1.5, -1.5], dtype=dtype))(actor_mean))
        critic = nn.Dense(self.config["FC_DIM_SIZE"],
                          kernel_init=nn.initializers.orthogonal(2),
                          bias_init=nn.initializers.constant(0.0))(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0),
                          bias_init=nn.initializers.constant(0.0))(critic)
        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder, pi_speed_brake), jnp.squeeze(critic, axis=-1)


# ── Config ──
CKPT_PATH = os.path.abspath(
    "results/waypoint_2026-05-15-00-47/checkpoints/checkpoint_epoch_600"
)
OUTPUT_DIR = "results/loop_eval"
SEED = 42
MAX_STEPS = 8000

# Loop maneuver params
LOOP_RADIUS    = 8000.0   # meters
LOOP_WAYPOINTS = 200      # waypoints around the circle
REACH_RADIUS   = 1500.0   # meters
CRUISE_VT      = 250.0    # m/s — entry speed for the loop

NET_CONFIG = {
    "FC_DIM_SIZE":    128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION":     "relu",
}


# ── Helpers ──
def _f(x, i=0):
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    return float(a.reshape(-1)[min(i, a.size - 1)])


def compute_loop_waypoints(origin_n, origin_e, origin_alt):
    """Generate waypoints along a vertical loop in the north-altitude plane."""
    center_n = origin_n + LOOP_RADIUS
    center_e = origin_e
    center_alt = origin_alt + LOOP_RADIUS

    waypoints = []
    for i in range(LOOP_WAYPOINTS):
        # Angle from bottom (-90°) through top (+90°) back to bottom (270°)
        angle = -np.pi / 2 + (i / LOOP_WAYPOINTS) * 2.0 * np.pi
        wp_n = center_n + LOOP_RADIUS * np.cos(angle)
        wp_alt = center_alt + LOOP_RADIUS * np.sin(angle)
        wp_e = center_e
        waypoints.append((wp_n, wp_e, wp_alt))
    return waypoints


def write_acmi_header(acmi_path):
    with open(acmi_path, 'w', encoding='utf-8') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")


def write_acmi_waypoints(acmi_path, waypoints):
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
    lat, lon, alt = enu_to_geodetic(wp_e, wp_n, wp_a, 0, 0, 0)
    with open(acmi_path, 'a', encoding='utf-8') as f:
        f.write(f"#{t_sec:.2f}\n")
        f.write(f"1000,T={float(lon)}|{float(lat)}|{float(alt)}|0|0|0,"
                f"Name=Target,Color=Green,Type=Marker\n")


# ── Main ──
def main():
    tag = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Build env
    env_params = WaypointTaskParams()
    env = AeroPlanaxWaypointEnv(env_params)

    # Build network
    network = ActorCriticRNN((31, 41, 41, 41, 5), config=NET_CONFIG)
    rng = jax.random.PRNGKey(SEED)

    obs_shape = env.observation_space(env.agents[0], env_params).shape
    print(f"Obs shape: {obs_shape}")
    init_x = (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1)))
    h0 = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    net_params = network.init(rng, h0, init_x)

    # Load checkpoint
    import orbax.checkpoint as ocp
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore())
    net_params = ckpt["params"]
    print(f"Restored epoch {int(ckpt['epoch'])}")

    # Reset env
    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, env_params)

    # Generate waypoints
    origin_n = _f(state.plane_state.north)
    origin_e = _f(state.plane_state.east)
    origin_alt = _f(state.plane_state.altitude)
    waypoints = compute_loop_waypoints(origin_n, origin_e, origin_alt)
    print(f"Generated {len(waypoints)} loop waypoints, "
          f"radius={LOOP_RADIUS:.0f}m, top_alt={origin_alt + 2*LOOP_RADIUS:.0f}m")

    # ACMI output
    acmi_path = os.path.join(OUTPUT_DIR, f"loop_{tag}.acmi")
    write_acmi_header(acmi_path)
    write_acmi_waypoints(acmi_path, waypoints)

    # Init RNN hidden state
    hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    done_flag = jnp.zeros((1,))

    # Record buffers
    rec_t = []; rec_alt = []; rec_vt = []; rec_roll = []
    rec_pitch = []; rec_yaw = []; rec_alpha = []; rec_beta = []
    rec_P = []; rec_Q = []; rec_R = []
    rec_thr = []; rec_el = []; rec_ail = []; rec_rud = []; rec_sb = []
    rec_north = []; rec_east = []
    rec_wp_idx = []; rec_dist = []; rec_heading_err = []; rec_reached = []

    # Waypoint tracking state
    current_wp = 0
    total_reached = 0
    wp_reached_times = []
    wp_switch_step = 0

    dt_rl = env_params.agent_interaction_steps / env_params.sim_freq

    print(f"\n{'Step':>6} | {'WP':>4} | {'Reach':>5} | {'Dist3D':>8} | "
          f"{'Alt':>7} | {'Vt':>6} | {'Roll':>7} | {'Pitch':>7} | {'Yaw':>7} | {'HdgErr':>7} | {'SB':>5}")
    print("-" * 100)

    for step in range(MAX_STEPS):
        ps = state.plane_state
        t_phys = step * dt_rl

        north = _f(ps.north); east = _f(ps.east); alt = _f(ps.altitude)
        vt   = _f(ps.vt)
        roll = _f(ps.roll); pitch = _f(ps.pitch); yaw = _f(ps.yaw)
        alpha = _f(ps.alpha); beta = _f(ps.beta)
        P = _f(ps.P); Q = _f(ps.Q); R = _f(ps.R)

        # Compute target to current waypoint
        wp_n, wp_e, wp_a = waypoints[min(current_wp, len(waypoints) - 1)]
        d_n = wp_n - north
        d_e = wp_e - east
        d_alt = wp_a - alt
        h_dist = float(np.sqrt(d_n**2 + d_e**2))
        dist_3d = float(np.sqrt(h_dist**2 + d_alt**2))

        # Per-step bearing (agent tracks precisely enough — no chasing)
        target_heading = float(np.arctan2(d_e, d_n))
        target_pitch = float(np.arctan2(d_alt, max(h_dist, 1e-6)))

        # Check waypoint reached (or timed out)
        steps_on_wp = step - wp_switch_step
        if dist_3d < REACH_RADIUS and current_wp < len(waypoints):
            current_wp += 1
            total_reached += 1
            wp_reached_times.append(t_phys)
            wp_switch_step = step
        elif steps_on_wp > 400 and current_wp < len(waypoints):
            current_wp += 1
            wp_switch_step = step

        # Build observation
        state_with_targets = state.replace(
            target_heading=jnp.array([target_heading]),
            target_pitch=jnp.array([target_pitch]),
            target_roll=jnp.array([0.0]),
            target_vt=jnp.array([CRUISE_VT]),
        )
        obs_dict = env._get_obs(state_with_targets, env_params)
        obs_vec = obs_dict[env.agents[0]]
        obs_in = obs_vec[None, None, :]
        done_in = done_flag[None, :]

        # Policy forward (greedy)
        hstate, pi, _ = network.apply(net_params, hstate, (obs_in, done_in))
        pi_thr, pi_el, pi_ail, pi_rud, pi_sb = pi

        act_thr = int(pi_thr.mode()[0, 0])
        act_el  = int(pi_el.mode()[0, 0])
        act_ail = int(pi_ail.mode()[0, 0])
        act_rud = int(pi_rud.mode()[0, 0])
        act_sb  = int(pi_sb.mode()[0, 0])

        action_dict = {env.agents[0]: jnp.array([act_thr, act_el, act_ail, act_rud, act_sb])}

        # Step env
        rng, step_key = jax.random.split(rng)
        obs_dict, state, rew_dict, done_dict, info = env.step(
            step_key, state, action_dict, env_params)
        done_flag = jnp.array([float(done_dict[env.agents[0]])])

        # Record
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

        # Decode actions
        thr_n = act_thr / 30.0
        el_n  = act_el * 2.0 / 40.0 - 1.0
        ail_n = act_ail * 2.0 / 40.0 - 1.0
        rud_n = act_rud * 2.0 / 40.0 - 1.0
        sb_n  = act_sb / 4.0
        rec_thr.append(thr_n)
        rec_el.append(el_n * 45.0)
        rec_ail.append(ail_n * 45.0)
        rec_rud.append(rud_n * 45.0)
        rec_sb.append(sb_n * 60.0)

        # ACMI
        write_acmi_aircraft(acmi_path, t_phys, north, east, alt, roll, pitch, yaw)
        write_acmi_current_wp(acmi_path, t_phys, wp_n, wp_e, wp_a)

        # Log
        if step % 200 == 0 or (step > 0 and current_wp > rec_wp_idx[max(0, step - 2)]):
            print(f"{step:6d} | {current_wp:4d} | {total_reached:5d} | {dist_3d:8.1f} | "
                  f"{alt:7.0f} | {vt:6.1f} | {np.degrees(roll):+7.1f} | "
                  f"{np.degrees(pitch):+7.1f} | {np.degrees(yaw):+7.1f} | "
                  f"{rec_heading_err[-1]:+7.1f} | {rec_sb[-1]:>5.0f}")

        # Termination
        if bool(done_dict["__all__"]):
            print(f"\n[TERMINATED] step={step}, wp_reached={total_reached}")
            break

        if total_reached >= len(waypoints):
            print(f"\n[SUCCESS] All {len(waypoints)} waypoints reached at step {step}!")
            break

    # ── Charts ──
    t = np.array(rec_t)
    fig = plt.figure(figsize=(20, 30))
    gs = gridspec.GridSpec(10, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(rec_east, rec_north, 'b-', lw=0.8, alpha=0.6, label='Flight path')
    wp_ns = [w[0] for w in waypoints]
    wp_es = [w[1] for w in waypoints]
    ax.scatter(wp_es, wp_ns, c='orange', s=5, alpha=0.7, label='Waypoints')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Top-down View (Loop)')
    ax.legend()
    ax.axis('equal')

    ax = fig.add_subplot(gs[1, :])
    ax.plot(rec_north, rec_alt, 'b-', lw=0.8, alpha=0.6, label='Flight path')
    ax.scatter(wp_ns, [w[2] for w in waypoints], c='orange', s=5, alpha=0.7, label='Waypoints')
    ax.set_xlabel('North (m)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Side View (North-Altitude)')
    ax.legend()
    ax.axis('equal')

    for idx, (title, ydata, ylabel, color) in enumerate([
        ('Altitude (m)', rec_alt, 'm', 'tab:blue'),
        ('Speed (m/s)', rec_vt, 'm/s', 'tab:red'),
        ('Roll (°)', rec_roll, '°', 'tab:green'),
        ('Pitch (°)', rec_pitch, '°', 'tab:orange'),
        ('Yaw (°)', rec_yaw, '°', 'tab:purple'),
        ('Alpha / Beta (°)', None, '°', None),
        ('Angular Rates (°/s)', None, '°/s', None),
        ('Heading Error (°)', rec_heading_err, '°', 'tab:red'),
        ('Distance to WP (m)', rec_dist, 'm', 'tab:brown'),
        ('Throttle', rec_thr, '0-1', 'tab:cyan'),
        ('Elevator (°)', rec_el, '°', 'tab:green'),
        ('Aileron (°)', rec_ail, '°', 'tab:orange'),
        ('Rudder (°)', rec_rud, '°', 'tab:purple'),
        ('Speed Brake (°)', rec_sb, '°', 'tab:pink'),
    ]):
        row = 2 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        if title == 'Alpha / Beta (°)':
            ax.plot(t, rec_alpha, 'r-', lw=0.8, label='Alpha')
            ax.plot(t, rec_beta, 'b-', lw=0.8, label='Beta')
            ax.legend()
        elif title == 'Angular Rates (°/s)':
            ax.plot(t, [np.degrees(p) for p in rec_P], 'r-', lw=0.8, label='P')
            ax.plot(t, [np.degrees(q) for q in rec_Q], 'g-', lw=0.8, label='Q')
            ax.plot(t, [np.degrees(r) for r in rec_R], 'b-', lw=0.8, label='R')
            ax.legend()
        else:
            ax.plot(t, ydata, color=color, lw=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    chart_path = os.path.join(OUTPUT_DIR, f"loop_{tag}.png")
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n{'='*80}")
    print(f"Render complete!")
    print(f"  ACMI:   {acmi_path}")
    print(f"  Charts: {chart_path}")
    print(f"  Waypoints reached: {total_reached}/{len(waypoints)}")
    print(f"  Duration: {t_phys:.1f} s  ({step} steps)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
