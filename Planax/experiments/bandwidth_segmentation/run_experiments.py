"""
Experiment runner: Bandwidth-Aware Hierarchical Trajectory Abstraction.

Runs closed-loop waypoint-tracking simulations with the trained Euler baseline,
comparing different waypoint selection methods on multiple reference trajectories.

Usage:
    conda activate aeroplanax
    cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
    python experiments/bandwidth_segmentation/run_experiments.py
"""

import os, sys, json, csv
from datetime import datetime
from pathlib import Path
import numpy as np

GPU_ID = os.environ.get("PLANAX_GPU", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["XLA_PYTHON_MEM_FRACTION"] = "0.7"
print(f"[run_experiments] Using GPU {GPU_ID}")

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from experiments.bandwidth_segmentation.dp_solver import SegmentationConfig, run_all_methods
from experiments.bandwidth_segmentation.rollout import run_rollout
from experiments.bandwidth_segmentation.metrics import compute_all_metrics
from experiments.bandwidth_segmentation.trajectories import ALL_TRAJECTORIES

CKPT_PATH = os.path.abspath(
    "results/heading_pitch_V_discrete_rnn_2026-05-09-16-53/checkpoints/checkpoint_epoch_300"
)
OUTPUT_BASE = "outputs/adaptive_segmentation"
REACH_RADIUS = 500.0
CRUISE_VT = 250.0
MAX_STEPS = 3000


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
    output_dir = Path(OUTPUT_BASE) / tag
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
    print(f"Loaded checkpoint, epoch={int(ckpt['epoch'])}")

    all_results = {}
    summary_rows = []
    cfg = SegmentationConfig()

    for traj_name, traj_fn in ALL_TRAJECTORIES.items():
        print(f"\n{'='*60}")
        traj, meta = traj_fn()
        print(f"Trajectory: {traj_name} ({meta['name']})")
        print(f"  {traj.shape[0]} pts, {meta['total_length_m']:.0f}m, "
              f"max_pitch={meta['max_pitch_deg']:.1f}°, "
              f"singularity_risk={meta['singularity_risk']}")

        np.savez(output_dir / f"{traj_name}_reference.npz", traj=traj, meta=meta)

        methods = run_all_methods(traj, cfg, verbose=True)
        print(f"  Methods: {list(methods.keys())}")
        traj_results = {}

        for method_name, method_data in methods.items():
            waypoints = method_data["waypoints"]
            N = method_data.get("N", len(waypoints))
            print(f"    {method_name}: {N} waypoints", end="", flush=True)

            rollout = run_rollout(waypoints, env, env_params, network, net_params, h0,
                                  rng_seed=42 + hash(method_name) % 1000,
                                  max_steps=MAX_STEPS, reach_radius=REACH_RADIUS, cruise_vt=CRUISE_VT)

            metrics = compute_all_metrics(traj, rollout["actual_traj"], waypoints,
                                           rollout["actions"], rollout["state"],
                                           termination_reason=rollout["termination_reason"],
                                           steps_completed=rollout["steps"])
            metrics["trajectory"] = traj_name
            metrics["method"] = method_name
            metrics["N_waypoints"] = N
            metrics["waypoints_reached"] = rollout["waypoints_reached"]
            metrics["total_waypoints"] = rollout["total_waypoints"]
            print(f" -> done={rollout['termination_reason']}, steps={rollout['steps']}, "
                  f"cte_rms={metrics['cross_track_error_continuous_rms_m']:.1f}m")

            traj_results[method_name] = {"metrics": metrics, "rollout": {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in rollout.items()},
                "waypoints": waypoints.tolist()}

            summary_rows.append({
                "trajectory": traj_name, "method": method_name, "N_waypoints": N,
                "cross_track_rms": metrics.get("cross_track_error_continuous_rms_m", np.nan),
                "cross_track_max": metrics.get("cross_track_error_continuous_max_m", np.nan),
                "total_saturation_rate": metrics.get("actuator_total_saturation_rate", np.nan),
                "actuator_smoothness": metrics.get("actuator_command_smoothness_rms", np.nan),
                "waypoints_reached": rollout["waypoints_reached"],
                "steps": rollout["steps"],
                "termination_reason": rollout["termination_reason"],
            })

        all_results[traj_name] = traj_results

        traj_dir = output_dir / traj_name
        traj_dir.mkdir(exist_ok=True)
        for mn, md in traj_results.items():
            np.savez(traj_dir / f"{mn}.npz", waypoints=np.array(md["waypoints"]),
                     actual_traj=np.array(md["rollout"]["actual_traj"]),
                     actions=np.array(md["rollout"]["actions"]),
                     t=np.array(md["rollout"]["t"]))
            with open(traj_dir / f"{mn}_metrics.json", "w") as f:
                json.dump(md["metrics"], f, indent=2)

    if summary_rows:
        with open(output_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader(); w.writerows(summary_rows)
    json.dump(all_results, open(output_dir / "full_results.json", "w"), indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Experiments complete. Output: {output_dir}")
    print(f"  Summary: {output_dir}/summary.csv")


if __name__ == "__main__":
    main()
