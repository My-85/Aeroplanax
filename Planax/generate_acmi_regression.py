"""
Generate ACMI files for residual candidate Claude regression.
Runs key scenarios and exports Tacview-compatible ACMI with waypoint markers.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from experiments.hierarchical_trajectory_tracking.render_ablation_tests import (
    ActorCriticRNN, NET_CFG, SEED, ScannedRNN,
)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import vertical_pullup_arc
from experiments.hierarchical_trajectory_tracking.planner import PlannerConfig, PurePursuitPlanner
from experiments.hierarchical_trajectory_tracking.path_utils import compute_true_cte
from experiments.hierarchical_trajectory_tracking.loop_attitude_target import (
    loop_plane_rotation_matrix, quaternion_to_euler, rotation_matrix_to_quaternion,
)
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

PLANAX_ROOT = Path(__file__).resolve().parent
OUT_DIR = PLANAX_ROOT / "results/residual_candidate_claude_regression/20260518_233806"
BASE_CKPT = PLANAX_ROOT / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
RESIDUAL_CKPT = PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2"


def f_scalar(x):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


def restore_params(checkpoint):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    return ckptr.restore(str(checkpoint.resolve()), args=ocp.args.StandardRestore())["params"]


def restore_residual_params(checkpoint):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(str(checkpoint.resolve()), args=ocp.args.StandardRestore())
    return ckpt["params"]


def loop_roll(theta_deg):
    rot = loop_plane_rotation_matrix(np.radians(theta_deg), 0.0, 1)
    q = rotation_matrix_to_quaternion(rot)
    roll_val, _, _ = quaternion_to_euler(q)
    return roll_val


def run_and_export_acmi(env, net, net_params, name, angle_deg, radius_m,
                         lookahead, reach_radius, max_steps,
                         residual_net=None, residual_params=None, residual_cfg=None,
                         acmi_dir=None):
    """Run rollout and save ACMI file."""
    wps, meta = vertical_pullup_arc(
        0, 0, 5000, 0.0, radius=radius_m, arc_angle_deg=angle_deg,
        n_points=max(80, int(angle_deg * 2 / 3)),
    )
    total_arc = meta["total_length_m"]
    planner = PurePursuitPlanner(
        PlannerConfig(lookahead_dist=lookahead, reach_radius=reach_radius,
                      blend_steps=250, target_vt=250.0)
    )

    rng = jax.random.PRNGKey(SEED)
    rng, reset_key = jax.random.split(rng)
    _, state = env.reset(reset_key, Params())
    q_nb_init = _quat_from_euler_nb(0.0, 0.0, 0.0)
    q_bn_init = _quat_conj(q_nb_init)
    state = state.replace(
        plane_state=state.plane_state.replace(
            yaw=jnp.array([0.0]),
            q0=jnp.array([q_bn_init[0]]), q1=jnp.array([q_bn_init[1]]),
            q2=jnp.array([q_bn_init[2]]), q3=jnp.array([q_bn_init[3]]),
        ),
        target_heading=jnp.array([0.0]),
    )
    planner.reset(wps, 0.0, 0.0, 0.0, 250.0)

    hstate = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
    residual_hstate = None
    if residual_net is not None:
        residual_hstate = ResidualScannedRNN.initialize_carry(
            1, int(residual_cfg.get("RESIDUAL_GRU_HIDDEN_DIM", 64))
        )
    done_flag = jnp.zeros((1,))

    traj = {"t": [], "n": [], "e": [], "a": [], "roll": [], "pitch": [], "yaw": []}
    crashed = False
    completed = False

    for step in range(max_steps):
        ps = state.plane_state
        north = f_scalar(ps.north)
        east = f_scalar(ps.east)
        alt = f_scalar(ps.altitude)
        vt = f_scalar(ps.vt)
        roll = f_scalar(ps.roll)
        pitch = f_scalar(ps.pitch)
        yaw = f_scalar(ps.yaw)

        result = planner.step(north, east, alt, yaw, pitch, roll, vt)
        target_heading = result["target_heading"]
        target_pitch = result["target_pitch"]
        target_roll = result["target_roll"]
        target_vt = result["target_vt"]

        path_s = planner.path_progress
        theta_deg = (path_s / total_arc) * angle_deg if total_arc > 0 else 0.0
        theta_deg = float(np.clip(theta_deg, 0.0, angle_deg))
        target_loop_roll = loop_roll(theta_deg)
        blend = min(1.0, step / 250.0)
        target_roll = float(np.arctan2(
            np.sin(roll + blend * (target_loop_roll - roll)),
            np.cos(roll + blend * (target_loop_roll - roll)),
        ))

        state = state.replace(
            target_heading=jnp.array([target_heading]),
            target_pitch=jnp.array([target_pitch]),
            target_roll=jnp.array([target_roll]),
            target_vt=jnp.array([float(target_vt)], dtype=jnp.float32),
        )

        obs = env._get_obs(state, Params())[env.agents[0]][None, None, :]
        hstate, base_pi, _ = net.apply(net_params, hstate, (obs, done_flag[None, :]))

        if residual_net is not None:
            gate = 80.0 <= theta_deg <= 180.0
            gate_float = 1.0 if gate else 0.0
            obs_aug = augment_obs_with_phase(
                obs.reshape((1, -1)), state, theta_deg, gate_float, residual_cfg
            )
            residual_hstate, residual_logits, _ = residual_net.apply(
                residual_params, residual_hstate, (obs_aug[None, :, :], done_flag[None, :])
            )
            pi_out, _, _ = combine_base_and_residual_logits(
                base_pi, residual_logits, obs_aug, residual_cfg
            )
        else:
            pi_out = base_pi

        actions = [int(p.mode()[0, 0]) for p in pi_out]

        rng, step_key = jax.random.split(rng)
        _, state, _, done, _ = env.step(
            step_key, state, {env.agents[0]: jnp.array(actions)}, Params()
        )
        done_flag = jnp.array([float(done[env.agents[0]])])

        traj["t"].append(step * 0.2)
        traj["n"].append(north)
        traj["e"].append(east)
        traj["a"].append(alt)
        traj["roll"].append(np.degrees(roll))
        traj["pitch"].append(np.degrees(pitch))
        traj["yaw"].append(np.degrees(yaw))

        if bool(done[env.agents[0]]):
            crashed = True
            break
        if planner.is_done():
            completed = True
            break

    acmi_name = f"{name}_{'base' if residual_net is None else 'base_plus_residual'}.acmi"
    acmi_path = acmi_dir / acmi_name
    write_acmi(str(acmi_path), wps, traj,
               aircraft_name=f"F16_{'base' if residual_net is None else 'residual'}",
               color="Cyan" if residual_net is None else "Red")
    print(f"  ACMI saved: {acmi_path} ({len(traj['t'])} frames, "
          f"{'crashed' if crashed else 'ok' if completed else 'timeout'})")
    return acmi_path, crashed, completed


def main():
    print("Generating ACMI regression files...")
    acmi_dir = OUT_DIR / "acmi"
    acmi_dir.mkdir(parents=True, exist_ok=True)

    env = Env(Params())
    net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    net_params = restore_params(BASE_CKPT)

    residual_cfg = {
        "ACTIVATION": "relu", "RESIDUAL_FC_DIM_SIZE": 96,
        "RESIDUAL_GRU_HIDDEN_DIM": 64, "RESIDUAL_LOGIT_CLIP": 1.25,
        "RESIDUAL_GATE_START_DEG": 80.0, "RESIDUAL_GATE_END_DEG": 180.0,
    }
    residual_net = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=residual_cfg)
    residual_params = restore_residual_params(RESIDUAL_CKPT)

    scenarios = [
        # Target loop scenarios
        ("pu150_R12000", 150, 12000, 1200, 500, 2000),
        ("pu175_R15000", 175, 15000, 1500, 500, 2500),
        ("pu180_R15000", 180, 15000, 1500, 500, 2500),
        # Loop retention
        ("pu090_R12000", 90, 12000, 1000, 400, 1500),
        ("pu120_R12000", 120, 12000, 1000, 400, 1800),
    ]

    for scenario in scenarios:
        name, angle_deg, radius_m, lookahead, reach_radius, max_steps = scenario
        print(f"\n{name} (base)...")
        run_and_export_acmi(
            env, net, net_params, name, angle_deg, radius_m,
            lookahead, reach_radius, max_steps,
            residual_net=None, acmi_dir=acmi_dir,
        )
        print(f"{name} (base+residual)...")
        run_and_export_acmi(
            env, net, net_params, name, angle_deg, radius_m,
            lookahead, reach_radius, max_steps,
            residual_net=residual_net, residual_params=residual_params,
            residual_cfg=residual_cfg, acmi_dir=acmi_dir,
        )

    print(f"\nAll ACMI files saved to: {acmi_dir}")


if __name__ == "__main__":
    main()
