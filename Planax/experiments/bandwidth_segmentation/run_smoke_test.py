"""
Smoke test: validates the entire pipeline on a single S-curve with 4 methods.

Goal: verify code runs end-to-end, not statistical significance.

Usage:
    conda activate aeroplanax
    cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
    python experiments/bandwidth_segmentation/run_smoke_test.py
"""

import os, sys, json
from datetime import datetime
from pathlib import Path
import numpy as np

GPU_ID = os.environ.get("PLANAX_GPU", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["XLA_PYTHON_MEM_FRACTION"] = "0.7"
print(f"[smoke_test] Using GPU {GPU_ID}")

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from experiments.bandwidth_segmentation.dp_solver import SegmentationConfig, solve, run_all_methods
from experiments.bandwidth_segmentation.rollout import run_rollout
from experiments.bandwidth_segmentation.metrics import compute_all_metrics
from experiments.bandwidth_segmentation.trajectories import generate_s_curve

CKPT_PATH = os.path.abspath(
    "results/heading_pitch_V_discrete_rnn_2026-05-09-16-53/checkpoints/checkpoint_epoch_300"
)
OUTPUT_DIR = "outputs/smoke_test"
REACH_RADIUS = 500.0
CRUISE_VT = 250.0
MAX_STEPS = 2000


def main():
    import jax, jax.numpy as jnp
    import orbax.checkpoint as ocp
    import flax.linen as nn
    from flax.linen.initializers import constant, orthogonal
    import functools, distrax
    from typing import Sequence, Dict

    from envs.aeroplanax_heading_pitch_V import (
        AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
    )

    class ScannedRNN(nn.Module):
        @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
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
        action_dim: Sequence[int]; config: Dict
        @nn.compact
        def __call__(self, hidden, x):
            act_fn = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
            obs, dones = x
            e = act_fn(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs))
            hidden, e = ScannedRNN()(hidden, (e, dones))
            fc2 = act_fn(nn.LayerNorm()(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(e)))
            am = act_fn(nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
            pi_thr = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am))
            pi_ele = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am))
            pi_ail = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am))
            pi_rud = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am))
            c = act_fn(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
            c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
            return hidden, (pi_thr, pi_ele, pi_ail, pi_rud), jnp.squeeze(c, axis=-1)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_DIR) / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)
    config_nn = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
    network = ActorCriticRNN([31, 41, 41, 41], config=config_nn)
    rng = jax.random.PRNGKey(42)
    obs_shape = env.observation_space(env.agents[0], env_params).shape
    init_x = (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1)))
    h0 = ScannedRNN.initialize_carry(1, 128)
    net_params = network.init(rng, h0, init_x)
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore())
    net_params = ckpt["params"]
    print(f"Loaded checkpoint, epoch={int(ckpt['epoch'])}\n")

    # Generate trajectory
    traj, meta = generate_s_curve(n_half_periods=4)
    print(f"Trajectory: {meta['name']}, {traj.shape[0]} pts, {meta['total_length_m']:.0f}m")
    print(f"  max_curvature={np.degrees(meta['max_curvature_rad']):.1f}°, "
          f"max_heading_rate={meta['max_heading_rate_proxy_deg_s']:.1f}°/s, "
          f"max_pitch={meta['max_pitch_deg']:.1f}°")
    np.savez(output_dir / "reference.npz", traj=traj, meta=meta)

    # Only 4 methods for smoke test
    cfg = SegmentationConfig()
    methods = {
        "uniform_N10": {"method": "uniform", "N": 10},
        "uniform_N40": {"method": "uniform", "N": 40},
    }
    # Add DP methods
    methods["dp_no_bandwidth"] = {"method": "dp_no_bandwidth"}
    methods["dp_with_bandwidth"] = {"method": "dp_with_bandwidth"}

    from experiments.bandwidth_segmentation import baselines as bl
    uniform_10 = bl.uniform_arc_length(traj, 10)
    uniform_40 = bl.uniform_arc_length(traj, 40)
    methods["uniform_N10"]["waypoints"] = traj[uniform_10]
    methods["uniform_N10"]["indices"] = uniform_10
    methods["uniform_N40"]["waypoints"] = traj[uniform_40]
    methods["uniform_N40"]["indices"] = uniform_40

    r_dp_no_bw = solve(SegmentationConfig(**{**cfg.__dict__, "w_rate": 0.0, "hard_reject_curvature": False,
        "psi_dot_max": 1e9, "theta_dot_max": 1e9, "phi_dot_max": 1e9, "max_turn_angle": 1e9, "tau_cmd": 0.0}),
        traj, verbose=True)
    r_dp_bw = solve(cfg, traj, verbose=True)

    methods["dp_no_bandwidth"]["waypoints"] = r_dp_no_bw.waypoints
    methods["dp_no_bandwidth"]["indices"] = r_dp_no_bw.waypoint_indices
    methods["dp_no_bandwidth"]["N"] = r_dp_no_bw.num_segments + 1
    methods["dp_with_bandwidth"]["waypoints"] = r_dp_bw.waypoints
    methods["dp_with_bandwidth"]["indices"] = r_dp_bw.waypoint_indices
    methods["dp_with_bandwidth"]["N"] = r_dp_bw.num_segments + 1

    summary = []
    for name, md in methods.items():
        N = md.get("N", len(md["waypoints"]))
        print(f"\n--- {name}: {N} waypoints ---")
        rollout = run_rollout(md["waypoints"], env, env_params, network, net_params, h0,
                              rng_seed=42, max_steps=MAX_STEPS, reach_radius=REACH_RADIUS, cruise_vt=CRUISE_VT)
        metrics = compute_all_metrics(traj, rollout["actual_traj"], md["waypoints"],
                                       rollout["actions"], rollout["state"],
                                       termination_reason=rollout["termination_reason"],
                                       steps_completed=rollout["steps"])
        metrics["method"] = name
        metrics["N_waypoints"] = N
        print(f"  Terminated: {rollout['termination_reason']}, steps={rollout['steps']}, wp_reached={rollout['waypoints_reached']}/{rollout['total_waypoints']}")
        print(f"  CTE_rms={metrics['cross_track_error_continuous_rms_m']:.1f}m, sat_rate={metrics['actuator_total_saturation_rate']:.3f}")
        summary.append(metrics)

        np.savez(output_dir / f"{name}_rollout.npz",
                 waypoints=md["waypoints"], actual_traj=rollout["actual_traj"],
                 actions=rollout["actions"], t=rollout["t"])
        json.dump(metrics, open(output_dir / f"{name}_metrics.json", "w"), indent=2)

    # Summary
    import csv
    if summary:
        fields = ["method", "N_waypoints", "cross_track_error_continuous_rms_m",
                  "cross_track_error_continuous_max_m", "actuator_total_saturation_rate",
                  "actuator_command_smoothness_rms", "trajectory_completion"]
        with open(output_dir / "smoke_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(summary)

    print(f"\n✓ Smoke test complete. Output: {output_dir}")
    print(f"  Summary: {output_dir}/smoke_summary.csv")
    for s in summary:
        print(f"  {s['method']:<25} N={s['N_waypoints']:>4}  CTE_rms={s['cross_track_error_continuous_rms_m']:>7.1f}m  "
              f"sat={s['actuator_total_saturation_rate']:.3f}  smooth={s['actuator_command_smoothness_rms']:.1f}")


if __name__ == "__main__":
    main()
