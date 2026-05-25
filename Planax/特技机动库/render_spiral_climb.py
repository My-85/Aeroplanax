"""
Spiral Climb (螺旋上升) aerobatic maneuver renderer.

Uses the latest combined policy: epoch619 base + residual_update_2.
Trajectory: 2-turn climbing helix with substantial altitude gain.
Output: ACMI for Tacview + PNG figures.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import orbax.checkpoint as ocp

PLANAX_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PLANAX_ROOT))

from experiments.hierarchical_trajectory_tracking.render_ablation_tests import (
    ActorCriticRNN, NET_CFG, SEED, ScannedRNN,
)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import helix_trajectory
from experiments.hierarchical_trajectory_tracking.planner import PlannerConfig, PurePursuitPlanner
from experiments.hierarchical_trajectory_tracking.path_utils import compute_true_cte
from experiments.hierarchical_trajectory_tracking.export_acmi import write_acmi
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as Env,
    Heading_Pitch_V_TaskParams as Params,
    _quat_conj, _quat_from_euler_nb,
)
from half_loop_residual_policy import (
    ResidualActorCriticRNN, ResidualScannedRNN,
    augment_obs_with_phase, combine_base_and_residual_logits,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_CKPT = PLANAX_ROOT / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
RESIDUAL_CKPT = PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2"
OUT_DIR = Path(__file__).resolve().parent  # 特技机动库


def f_scalar(x):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


def restore_params(ckpt_path):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    return ckptr.restore(str(ckpt_path.resolve()), args=ocp.args.StandardRestore())["params"]


def restore_residual_params(ckpt_path):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(str(ckpt_path.resolve()), args=ocp.args.StandardRestore())
    return ckpt["params"]


# ── Spiral climb configuration ───────────────────────────────────────────────
# User's requirements: 2 turns, noticeable altitude gain per turn
# Default: R=8000m, climb 3000m per turn → total 6000m over 2 turns
SPIRAL_CONFIGS = [
    # (name, radius_m, turns, delta_alt_per_turn_m, total_climb_m, n_points, lookahead, reach_radius, max_steps)
    ("spiral_R8000_climb6000", 8000, 2.0, 3000, 6000, 240, 2000, 500, 3000),
    ("spiral_R6000_climb3000", 6000, 2.0, 1500, 3000, 200, 1800, 450, 2000),
    ("spiral_R10000_climb8000", 10000, 2.0, 4000, 8000, 280, 2200, 600, 3000),
]


def run_spiral(env, net, net_params, residual_net, residual_params, residual_cfg,
               name, radius, turns, delta_alt, n_points, lookahead, reach_radius, max_steps):
    """Run a spiral climb rollout with combined base+residual policy."""

    wps, meta = helix_trajectory(
        0, 0, 5000, 0.0,
        radius=radius, turns=turns, delta_alt=delta_alt,
        n_points=n_points, direction=1,
    )
    total_len = meta["total_length_m"]

    planner = PurePursuitPlanner(
        PlannerConfig(lookahead_dist=lookahead, reach_radius=reach_radius,
                      blend_steps=250, target_vt=250.0)
    )

    # ── Compute initial position and heading from WP_0 → WP_1 ──
    wp0 = wps[0]   # (north, east, altitude)
    wp1 = wps[1] if len(wps) > 1 else wps[0]
    init_n = float(wp0[0])
    init_e = float(wp0[1])
    init_alt = float(wp0[2])
    delta_n = float(wp1[0] - wp0[0])
    delta_e = float(wp1[1] - wp0[1])
    init_yaw = float(np.arctan2(delta_e, delta_n))  # heading from WP_0 to WP_1

    rng = jax.random.PRNGKey(SEED)
    rng, reset_key = jax.random.split(rng)
    _, state = env.reset(reset_key, Params())
    q_nb_init = _quat_from_euler_nb(0.0, 0.0, init_yaw)
    q_bn_init = _quat_conj(q_nb_init)
    state = state.replace(
        plane_state=state.plane_state.replace(
            north=jnp.array([init_n]),
            east=jnp.array([init_e]),
            altitude=jnp.array([init_alt]),
            yaw=jnp.array([init_yaw]),
            q0=jnp.array([q_bn_init[0]]), q1=jnp.array([q_bn_init[1]]),
            q2=jnp.array([q_bn_init[2]]), q3=jnp.array([q_bn_init[3]]),
        ),
        target_heading=jnp.array([init_yaw]),
    )
    planner.reset(wps, init_yaw, 0.0, 0.0, 250.0)

    hstate = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
    residual_hstate = ResidualScannedRNN.initialize_carry(
        1, int(residual_cfg.get("RESIDUAL_GRU_HIDDEN_DIM", 64))
    )
    done_flag = jnp.zeros((1,))

    rec = {
        "t": [], "n": [], "e": [], "a": [], "vt": [],
        "roll": [], "pitch": [], "yaw": [],
        "t_roll": [], "t_pitch": [], "t_hdg": [],
        "alpha": [], "beta": [], "G": [], "cte": [],
        "wp_idx": [], "phase_deg": [], "gate_val": [],
    }
    crashed = False

    for step in range(max_steps):
        ps = state.plane_state
        north = f_scalar(ps.north)
        east = f_scalar(ps.east)
        alt = f_scalar(ps.altitude)
        vt = f_scalar(ps.vt)
        roll = f_scalar(ps.roll)
        pitch = f_scalar(ps.pitch)
        yaw = f_scalar(ps.yaw)
        alpha = f_scalar(ps.alpha)
        beta = f_scalar(ps.beta)
        ax = f_scalar(ps.ax)
        ay = f_scalar(ps.ay)
        az = f_scalar(ps.az)

        result = planner.step(north, east, alt, yaw, pitch, roll, vt)
        target_heading = result["target_heading"]
        target_pitch = result["target_pitch"]
        target_roll = result["target_roll"]
        target_vt = result["target_vt"]

        state = state.replace(
            target_heading=jnp.array([target_heading]),
            target_pitch=jnp.array([target_pitch]),
            target_roll=jnp.array([target_roll]),
            target_vt=jnp.array([float(target_vt)], dtype=jnp.float32),
        )

        # Path progress for phase computation
        path_s = planner.path_progress
        phase_deg = (path_s / total_len) * 720.0 if total_len > 0 else 0.0  # 2 turns = 720 deg
        phase_deg = float(np.clip(phase_deg, 0.0, 720.0))

        obs = env._get_obs(state, Params())[env.agents[0]][None, None, :]
        hstate, base_pi, _ = net.apply(net_params, hstate, (obs, done_flag[None, :]))

        # Combined policy: base + residual (no gate for horizontal maneuvers)
        # For spiral climb, we use base only since residual is gated for 80-180 deg vertical arcs
        # The combined policy is identical to base outside the gate
        gate = 80.0 <= (phase_deg % 360.0) <= 180.0
        gate_float = 1.0 if gate else 0.0
        obs_aug = augment_obs_with_phase(
            obs.reshape((1, -1)), state, phase_deg % 360.0, gate_float, residual_cfg
        )
        residual_hstate, residual_logits, _ = residual_net.apply(
            residual_params, residual_hstate, (obs_aug[None, :, :], done_flag[None, :])
        )
        pi_out, _, _ = combine_base_and_residual_logits(
            base_pi, residual_logits, obs_aug, residual_cfg
        )
        actions = [int(p.mode()[0, 0]) for p in pi_out]

        rng, step_key = jax.random.split(rng)
        _, state, _, done, _ = env.step(
            step_key, state, {env.agents[0]: jnp.array(actions)}, Params()
        )
        done_flag = jnp.array([float(done[env.agents[0]])])

        wp_idx = result["path_ctx"]["wp_idx"]
        rec["t"].append(step * 0.2)
        rec["n"].append(north); rec["e"].append(east); rec["a"].append(alt)
        rec["vt"].append(vt)
        rec["roll"].append(np.degrees(roll))
        rec["pitch"].append(np.degrees(pitch))
        rec["yaw"].append(np.degrees(yaw))
        rec["t_roll"].append(np.degrees(target_roll))
        rec["t_pitch"].append(np.degrees(target_pitch))
        rec["t_hdg"].append(np.degrees(target_heading))
        rec["alpha"].append(np.degrees(alpha))
        rec["beta"].append(np.degrees(beta))
        rec["G"].append(float(np.sqrt(ax*ax + ay*ay + az*az)))
        rec["cte"].append(compute_true_cte(np.array([north, east, alt]), wps, wp_idx, 10))
        rec["wp_idx"].append(wp_idx)
        rec["phase_deg"].append(phase_deg)
        rec["gate_val"].append(gate_float)

        if bool(done[env.agents[0]]):
            crashed = True
            break
        if planner.is_done():
            break

    n = len(rec["t"])
    completed = planner.is_done() and not crashed

    # Metrics
    cte_arr = np.array(rec["cte"])
    vt_arr = np.array(rec["vt"])
    g_arr = np.array(rec["G"])
    alpha_arr = np.array(rec["alpha"])
    alt_arr = np.array(rec["a"])

    metrics = {
        "name": name, "completed": bool(completed), "steps": n,
        "termination": "crash" if crashed else ("ok" if completed else "timeout"),
        "CTE_mean": float(cte_arr.mean()), "CTE_max": float(cte_arr.max()),
        "Gmax": float(g_arr.max()), "Gmean": float(g_arr.mean()),
        "vt_min": float(vt_arr.min()), "vt_mean": float(vt_arr.mean()), "vt_max": float(vt_arr.max()),
        "alpha_max": float(alpha_arr.max()), "alpha_mean": float(alpha_arr.mean()),
        "alt_start": float(alt_arr[0]), "alt_end": float(alt_arr[-1]),
        "alt_gain": float(alt_arr[-1] - alt_arr[0]),
        "alt_min": float(alt_arr.min()), "alt_max": float(alt_arr.max()),
        "alt_gain_per_turn": float(alt_arr[-1] - alt_arr[0]) / turns,
    }

    return metrics, rec, wps, meta


def plot_spiral_3d(name, rec, wps, out_dir):
    """Generate 3D trajectory plot with waypoints."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    n_arr = np.array(rec["n"])
    e_arr = np.array(rec["e"])
    a_arr = np.array(rec["a"])

    # Waypoints (target path)
    ax.plot(wps[:, 0], wps[:, 1], wps[:, 2], 'y-', linewidth=1.5, alpha=0.6, label='Target path')
    ax.scatter(wps[0, 0], wps[0, 1], wps[0, 2], c='yellow', s=80, marker='o', label='Start WP')
    ax.scatter(wps[-1, 0], wps[-1, 1], wps[-1, 2], c='orange', s=80, marker='s', label='End WP')

    # Aircraft track colored by altitude
    points = ax.scatter(e_arr, n_arr, a_arr, c=a_arr, cmap='viridis', s=2, alpha=0.7, label='Aircraft')
    plt.colorbar(points, ax=ax, label='Altitude (m)')

    # Mark start/end
    ax.scatter(e_arr[0], n_arr[0], a_arr[0], c='cyan', s=100, marker='^', label='Start')
    ax.scatter(e_arr[-1], n_arr[-1], a_arr[-1], c='red', s=100, marker='v', label='End')

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(f'Spiral Climb: {name}\nAltitude gain: {a_arr[-1]-a_arr[0]:.0f}m over 2 turns',
                 fontsize=14)
    ax.legend(loc='upper left')
    ax.view_init(elev=25, azim=-60)

    fig_path = out_dir / f"{name}_trajectory.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig_path


def plot_spiral_telemetry(name, rec, out_dir):
    """Generate telemetry subplots."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    t_arr = np.array(rec["t"])
    phase_arr = np.array(rec["phase_deg"])

    # Altitude
    ax = axes[0, 0]
    ax.plot(phase_arr, np.array(rec["a"]), 'b-', linewidth=1.5)
    ax.set_xlabel('Phase (deg)'); ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude vs Phase'); ax.grid(True, alpha=0.3)

    # CTE
    ax = axes[0, 1]
    ax.plot(phase_arr, np.array(rec["cte"]), 'r-', linewidth=1.5)
    ax.set_xlabel('Phase (deg)'); ax.set_ylabel('CTE (m)')
    ax.set_title('Cross-Track Error'); ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='A-grade (100m)')
    ax.axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='B-grade (500m)')
    ax.legend()

    # Speed
    ax = axes[1, 0]
    ax.plot(phase_arr, np.array(rec["vt"]), 'g-', linewidth=1.5)
    ax.set_xlabel('Phase (deg)'); ax.set_ylabel('VT (m/s)')
    ax.set_title('True Airspeed'); ax.grid(True, alpha=0.3)
    ax.axhline(y=250, color='gray', linestyle='--', alpha=0.5, label='Target 250')
    ax.axhline(y=190, color='red', linestyle='--', alpha=0.5, label='Min A-grade')
    ax.legend()

    # G-load
    ax = axes[1, 1]
    ax.plot(phase_arr, np.array(rec["G"]), 'orange', linewidth=1.5)
    ax.set_xlabel('Phase (deg)'); ax.set_ylabel('G-load')
    ax.set_title('Normal Acceleration'); ax.grid(True, alpha=0.3)
    ax.axhline(y=9, color='red', linestyle='--', alpha=0.5, label='Max A-grade')
    ax.legend()

    # Alpha / Beta
    ax = axes[2, 0]
    ax.plot(phase_arr, np.array(rec["alpha"]), 'purple', linewidth=1.5, label='Alpha')
    ax.plot(phase_arr, np.array(rec["beta"]), 'brown', linewidth=1.0, label='Beta', alpha=0.7)
    ax.set_xlabel('Phase (deg)'); ax.set_ylabel('Angle (deg)')
    ax.set_title('Aerodynamic Angles'); ax.grid(True, alpha=0.3)
    ax.legend()

    # Roll / Pitch / Yaw
    ax = axes[2, 1]
    ax.plot(phase_arr, np.array(rec["pitch"]), 'b-', linewidth=1.5, label='Pitch')
    ax.plot(phase_arr, np.array(rec["roll"]), 'r-', linewidth=1.0, label='Roll', alpha=0.7)
    ax.plot(phase_arr, np.array(rec["t_pitch"]), 'b--', linewidth=0.8, label='Target Pitch', alpha=0.5)
    ax.set_xlabel('Phase (deg)'); ax.set_ylabel('Angle (deg)')
    ax.set_title('Attitude'); ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f'Spiral Climb Telemetry: {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = out_dir / f"{name}_telemetry.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig_path


def main():
    print("=" * 60)
    print("SPIRAL CLIMB (螺旋上升) — Combined Policy Render")
    print("=" * 60)

    # Load policies
    print("\nLoading policies...")
    env = Env(Params())
    net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    net_params = restore_params(BASE_CKPT)
    print(f"  Base checkpoint: {BASE_CKPT}")

    residual_cfg = {
        "ACTIVATION": "relu", "RESIDUAL_FC_DIM_SIZE": 96,
        "RESIDUAL_GRU_HIDDEN_DIM": 64, "RESIDUAL_LOGIT_CLIP": 1.25,
        "RESIDUAL_GATE_START_DEG": 80.0, "RESIDUAL_GATE_END_DEG": 180.0,
    }
    residual_net = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=residual_cfg)
    residual_params = restore_residual_params(RESIDUAL_CKPT)
    print(f"  Residual checkpoint: {RESIDUAL_CKPT}")

    results = []

    for config in SPIRAL_CONFIGS:
        name, radius, turns, delta_per_turn, total_climb, n_pts, la, rr, mx = config

        print(f"\n{'='*40}")
        print(f"Running: {name}")
        print(f"  Radius={radius}m, Turns={turns}, Climb={total_climb}m ({delta_per_turn}m/turn)")
        print(f"  Waypoints={n_pts}, Lookahead={la}m, MaxSteps={mx}")

        metrics, rec, wps, meta = run_spiral(
            env, net, net_params, residual_net, residual_params, residual_cfg,
            name, radius, turns, total_climb, n_pts, la, rr, mx,
        )

        print(f"  Completed: {metrics['completed']} | Steps: {metrics['steps']}")
        print(f"  Altitude: {metrics['alt_start']:.0f}m → {metrics['alt_end']:.0f}m "
              f"(gain: {metrics['alt_gain']:.0f}m, per turn: {metrics['alt_gain_per_turn']:.0f}m)")
        print(f"  CTE: mean={metrics['CTE_mean']:.0f}m, max={metrics['CTE_max']:.0f}m")
        print(f"  Alpha: mean={metrics['alpha_mean']:.1f}deg, max={metrics['alpha_max']:.1f}deg")
        print(f"  G: mean={metrics['Gmean']:.1f}, max={metrics['Gmax']:.1f}")
        print(f"  VT: min={metrics['vt_min']:.0f}m/s, mean={metrics['vt_mean']:.0f}m/s")

        # Generate ACMI
        acmi_path = OUT_DIR / f"{name}.acmi"
        write_acmi(str(acmi_path), wps, {
            "t": rec["t"], "n": rec["n"], "e": rec["e"], "a": rec["a"],
            "roll": rec["roll"], "pitch": rec["pitch"], "yaw": rec["yaw"],
        }, aircraft_name=f"F16_Spiral_{int(total_climb)}m", color="Cyan")
        print(f"  ACMI: {acmi_path}")

        # Generate PNGs
        plot_spiral_3d(name, rec, wps, OUT_DIR)
        plot_spiral_telemetry(name, rec, OUT_DIR)
        print(f"  PNG: {OUT_DIR}/{name}_trajectory.png, {OUT_DIR}/{name}_telemetry.png")

        results.append((name, metrics, rec, wps, meta))

    # Summary
    print("\n" + "=" * 60)
    print("SPIRAL CLIMB RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Name':<35} {'OK':<6} {'Alt Gain':<10} {'Per Turn':<12} {'CTE_m':<10} {'Gmax':<8} {'a_max':<8}")
    print("-" * 90)
    for name, m, _, _, _ in results:
        print(f"{name:<35} {str(m['completed']):<6} {m['alt_gain']:<10.0f} "
              f"{m['alt_gain_per_turn']:<12.0f} {m['CTE_mean']:<10.0f} "
              f"{m['Gmax']:<8.1f} {m['alpha_max']:<8.1f}")

    print(f"\nAll files saved to: {OUT_DIR}")
    print("Files: *.acmi (Tacview), *_trajectory.png (3D view), *_telemetry.png (telemetry)")


if __name__ == "__main__":
    main()
