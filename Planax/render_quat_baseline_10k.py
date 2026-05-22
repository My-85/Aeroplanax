"""
Run a GPU large-batch Planax simulation with the quaternion attitude baseline.

Default run:
    python render_quat_baseline_10k.py

Useful overrides:
    python render_quat_baseline_10k.py --gpu 1 --num-envs 10000 --steps 300
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence


def _preconfigure_jax() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    known, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_MEM_FRACTION", "0.82")


_preconfigure_jax()

import distrax  # noqa: E402
import flax.linen as nn  # noqa: E402
import functools  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
from flax.linen.initializers import constant, orthogonal  # noqa: E402

from envs.aeroplanax_quat_large_batch import (  # noqa: E402
    AeroPlanaxQuatLargeBatchEnv,
    QuatLargeBatchTaskParams,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_CKPT = (
    "/home/dqy/aeroplanax/new/20251215最新代码库/Planax/"
    "envs/models/baseline_quat_20260514/"
    "heading_pitch_V_discrete_rnn_2026-05-13-21-17/"
    "checkpoints/checkpoint_epoch_600"
)

NET_CONFIG = {
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION": "relu",
}


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
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
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        fc2 = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        fc2 = nn.LayerNorm()(fc2)
        fc2 = activation(fc2)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(fc2)
        actor_mean = activation(actor_mean)

        pi_throttle = distrax.Categorical(
            logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        )
        pi_elevator = distrax.Categorical(
            logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        )
        pi_aileron = distrax.Categorical(
            logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        )
        pi_rudder = distrax.Categorical(
            logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        )
        pi_speed_brake = distrax.Categorical(
            logits=nn.Dense(
                self.action_dim[4],
                kernel_init=constant(0.0),
                bias_init=lambda key, shape, dtype=jnp.float32: jnp.array(
                    [0.0, -1.5, -1.5, -1.5, -1.5], dtype=dtype
                ),
            )(actor_mean)
        )

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder, pi_speed_brake), jnp.squeeze(critic, axis=-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--num-envs", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", default="results/quat_baseline_10k")
    parser.add_argument("--sample-trajectories", type=int, default=32)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def restore_params(network, env, env_params, checkpoint_path, seed):
    rng = jax.random.PRNGKey(seed)
    obs_shape = env.observation_space(env.agents[0], env_params).shape
    init_hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    init_x = (
        jnp.zeros((1, 1, *obs_shape), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
    )
    params = network.init(rng, init_hstate, init_x)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckptr = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(str(ckpt_path), args=ocp.args.StandardRestore())
    return ckpt["params"], int(np.asarray(ckpt.get("epoch", -1)))


def make_rollout(env, env_params, network, net_params, num_envs, steps, sample_n):
    agent = env.agents[0]
    sample_n = min(sample_n, num_envs)

    @jax.jit
    def rollout(seed_key):
        reset_keys = jax.random.split(seed_key, num_envs)
        obs_dict, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        obs = obs_dict[agent]
        done = jnp.zeros((num_envs,), dtype=jnp.float32)
        hstate = ScannedRNN.initialize_carry(num_envs, NET_CONFIG["GRU_HIDDEN_DIM"])

        def step_fn(carry, _):
            env_state, obs, done, hstate, key = carry
            hstate, pi, value = network.apply(net_params, hstate, (obs[None, :, :], done[None, :]))
            pi_throttle, pi_elevator, pi_aileron, pi_rudder, pi_speed_brake = pi
            action = jnp.stack(
                [
                    pi_throttle.mode()[0],
                    pi_elevator.mode()[0],
                    pi_aileron.mode()[0],
                    pi_rudder.mode()[0],
                    pi_speed_brake.mode()[0],
                ],
                axis=-1,
            ).astype(jnp.int32)

            key, split_key = jax.random.split(key)
            step_keys = jax.random.split(split_key, num_envs)

            def step_one(k, st, act):
                return env.step(k, st, {agent: act}, env_params)

            next_obs_dict, next_env_state, reward_dict, done_dict, info = jax.vmap(
                step_one, in_axes=(0, 0, 0)
            )(step_keys, env_state, action)

            next_obs = next_obs_dict[agent]
            next_done_bool = done_dict[agent]
            next_done = next_done_bool.astype(jnp.float32)
            reward = reward_dict[agent]
            ps = env_state.plane_state

            metric = {
                "alive_frac": jnp.mean(~next_done_bool),
                "done_frac": jnp.mean(next_done),
                "mean_reward": jnp.mean(reward),
                "mean_altitude": jnp.mean(ps.altitude),
                "min_altitude": jnp.min(ps.altitude),
                "max_altitude": jnp.max(ps.altitude),
                "mean_vt": jnp.mean(ps.vt),
                "min_vt": jnp.min(ps.vt),
                "max_vt": jnp.max(ps.vt),
                "mean_abs_roll_deg": jnp.mean(jnp.abs(ps.roll)) * 180.0 / jnp.pi,
                "mean_abs_pitch_deg": jnp.mean(jnp.abs(ps.pitch)) * 180.0 / jnp.pi,
                "mean_g_load": jnp.mean(jnp.maximum(jnp.maximum(jnp.abs(ps.ax), jnp.abs(ps.ay)), jnp.abs(ps.az))),
                "sample_north": ps.north[:sample_n],
                "sample_east": ps.east[:sample_n],
                "sample_altitude": ps.altitude[:sample_n],
                "sample_vt": ps.vt[:sample_n],
                "sample_roll_deg": ps.roll[:sample_n] * 180.0 / jnp.pi,
                "sample_pitch_deg": ps.pitch[:sample_n] * 180.0 / jnp.pi,
            }

            return (next_env_state, next_obs, next_done, hstate, key), metric

        final_carry, metrics = jax.lax.scan(
            step_fn,
            (env_state, obs, done, hstate, seed_key),
            xs=None,
            length=steps,
        )
        final_state = final_carry[0]
        final_ps = final_state.plane_state
        final_done = final_carry[2]
        summary = {
            "final_done_frac": jnp.mean(final_done),
            "final_mean_altitude": jnp.mean(final_ps.altitude),
            "final_mean_vt": jnp.mean(final_ps.vt),
            "final_min_altitude": jnp.min(final_ps.altitude),
            "final_max_altitude": jnp.max(final_ps.altitude),
            "final_min_vt": jnp.min(final_ps.vt),
            "final_max_vt": jnp.max(final_ps.vt),
            "final_mean_g_load": jnp.mean(jnp.maximum(jnp.maximum(jnp.abs(final_ps.ax), jnp.abs(final_ps.ay)), jnp.abs(final_ps.az))),
            "total_agent_steps": jnp.array(num_envs * steps, dtype=jnp.int32),
        }
        return metrics, summary

    return rollout


def to_numpy_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def write_outputs(out_dir, tag, args, devices, epoch, metrics, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    scalar_keys = [k for k in metrics.keys() if not k.startswith("sample_")]
    step_table = np.column_stack([np.arange(args.steps)] + [metrics[k] for k in scalar_keys])
    csv_path = out_dir / f"step_metrics_{tag}.csv"
    header = "step," + ",".join(scalar_keys)
    np.savetxt(csv_path, step_table, delimiter=",", header=header, comments="")

    summary_payload = {
        "tag": tag,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "gpu": str(args.gpu),
        "devices": [str(d) for d in devices],
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": epoch,
        "summary": {k: float(v) for k, v in summary.items() if k != "total_agent_steps"},
        "total_agent_steps": int(summary["total_agent_steps"]),
    }
    json_path = out_dir / f"summary_{tag}.json"
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    png_path = None
    if not args.no_plots:
        t = np.arange(args.steps) * (QuatLargeBatchTaskParams().agent_interaction_steps / QuatLargeBatchTaskParams().sim_freq)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
        axes[0, 0].plot(t, metrics["mean_altitude"])
        axes[0, 0].fill_between(t, metrics["min_altitude"], metrics["max_altitude"], alpha=0.2)
        axes[0, 0].set_title("Altitude")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("m")

        axes[0, 1].plot(t, metrics["mean_vt"])
        axes[0, 1].fill_between(t, metrics["min_vt"], metrics["max_vt"], alpha=0.2)
        axes[0, 1].set_title("Airspeed")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("m/s")

        axes[0, 2].plot(t, metrics["mean_reward"])
        axes[0, 2].set_title("Mean Reward")
        axes[0, 2].set_xlabel("Time (s)")

        axes[1, 0].plot(t, metrics["mean_abs_roll_deg"], label="roll")
        axes[1, 0].plot(t, metrics["mean_abs_pitch_deg"], label="pitch")
        axes[1, 0].set_title("Mean Abs Attitude")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("deg")
        axes[1, 0].legend()

        axes[1, 1].plot(t, metrics["mean_g_load"])
        axes[1, 1].set_title("Mean Max-Axis G Load")
        axes[1, 1].set_xlabel("Time (s)")

        for i in range(metrics["sample_east"].shape[1]):
            axes[1, 2].plot(metrics["sample_east"][:, i], metrics["sample_north"][:, i], lw=0.8, alpha=0.65)
        axes[1, 2].set_title(f"Sample Trajectories ({metrics['sample_east'].shape[1]})")
        axes[1, 2].set_xlabel("East (m)")
        axes[1, 2].set_ylabel("North (m)")
        axes[1, 2].axis("equal")

        fig.suptitle(f"Planax quaternion baseline large-batch rollout: {args.num_envs} envs x {args.steps} steps")
        png_path = out_dir / f"rollout_{tag}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

    return csv_path, json_path, png_path


def main():
    args = parse_args()
    tag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    gpu_devices = jax.devices("gpu")
    if not gpu_devices:
        raise RuntimeError("No JAX GPU device is visible. Check CUDA_VISIBLE_DEVICES and the NVIDIA driver.")

    print(f"[config] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[config] JAX devices: {gpu_devices}")
    print(f"[config] num_envs={args.num_envs}, steps={args.steps}, seed={args.seed}")

    env_params = QuatLargeBatchTaskParams()
    env = AeroPlanaxQuatLargeBatchEnv(env_params)
    network = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CONFIG)
    net_params, epoch = restore_params(network, env, env_params, args.checkpoint, args.seed)
    print(f"[checkpoint] restored epoch={epoch} from {args.checkpoint}")

    rollout = make_rollout(
        env=env,
        env_params=env_params,
        network=network,
        net_params=net_params,
        num_envs=args.num_envs,
        steps=args.steps,
        sample_n=args.sample_trajectories,
    )

    rng = jax.random.PRNGKey(args.seed)
    print("[rollout] compiling and running on GPU...")
    start = time.time()
    metrics, summary = rollout(rng)
    metrics, summary = to_numpy_tree((metrics, summary))
    elapsed = time.time() - start

    out_dir = Path(args.output_dir) / tag
    csv_path, json_path, png_path = write_outputs(out_dir, tag, args, gpu_devices, epoch, metrics, summary)
    total_agent_steps = int(summary["total_agent_steps"])

    print("[done] large-batch simulation complete")
    print(f"  wall_time_s: {elapsed:.2f}")
    print(f"  throughput_agent_steps_s: {total_agent_steps / max(elapsed, 1e-9):.0f}")
    print(f"  total_agent_steps: {total_agent_steps}")
    print(f"  final_done_frac: {float(summary['final_done_frac']):.4f}")
    print(f"  final_mean_altitude: {float(summary['final_mean_altitude']):.1f}")
    print(f"  final_mean_vt: {float(summary['final_mean_vt']):.1f}")
    print(f"  summary: {json_path}")
    print(f"  step_metrics: {csv_path}")
    if png_path is not None:
        print(f"  plot: {png_path}")


if __name__ == "__main__":
    main()
