"""
Visual showcase for Planax 10k-scale GPU simulation.

The experiment runs 10,000 independent Planax quaternion-baseline rollouts in
one JAX/GPU batch, then renders a more communicative figure:
  - every final dot is one aircraft rollout
  - translucent time-slice clouds show the swarm expanding over time
  - altitude/speed histograms show all 10k terminal states
  - a terminal-style panel preserves the CLI evidence

Example:
    /home/dqy/miniconda3/envs/aeroplanax/bin/python render_quat_baseline_10k_showcase.py --gpu 0
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence
import textwrap


def _preconfigure_jax() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    known, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_MEM_FRACTION", "0.82")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-planax-showcase")


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
from matplotlib.collections import LineCollection  # noqa: E402


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

        fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
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

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder, pi_speed_brake), jnp.squeeze(critic, axis=-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--num-envs", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", default="results/quat_baseline_10k_showcase")
    parser.add_argument("--trajectory-samples", type=int, default=500)
    return parser.parse_args()


def restore_params(network, env, env_params, checkpoint_path, seed):
    rng = jax.random.PRNGKey(seed)
    obs_shape = env.observation_space(env.agents[0], env_params).shape
    init_hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    init_x = (
        jnp.zeros((1, 1, *obs_shape), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
    )
    network.init(rng, init_hstate, init_x)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckptr = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(str(ckpt_path), args=ocp.args.StandardRestore())
    return ckpt["params"], int(np.asarray(ckpt.get("epoch", -1)))


def build_rollout(env, env_params, network, net_params, num_envs, steps, snapshot_every):
    agent = env.agents[0]
    snapshot_count = steps // snapshot_every + 1

    @jax.jit
    def rollout(seed_key):
        def one_agent(x):
            return jnp.squeeze(x, axis=-1)

        reset_keys = jax.random.split(seed_key, num_envs)
        obs_dict, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        obs = obs_dict[agent]
        done = jnp.zeros((num_envs,), dtype=jnp.float32)
        hstate = ScannedRNN.initialize_carry(num_envs, NET_CONFIG["GRU_HIDDEN_DIM"])

        initial_snapshot = jnp.stack(
            [
                one_agent(env_state.plane_state.north),
                one_agent(env_state.plane_state.east),
                one_agent(env_state.plane_state.altitude),
                one_agent(env_state.plane_state.vt),
            ],
            axis=1,
        )
        snapshots = jnp.zeros((snapshot_count, num_envs, 4), dtype=jnp.float32)
        snapshots = snapshots.at[0].set(initial_snapshot)

        def step_fn(carry, step_idx):
            env_state, obs, done, hstate, key, snapshots = carry
            hstate, pi, _ = network.apply(net_params, hstate, (obs[None, :, :], done[None, :]))
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

            next_done = done_dict[agent].astype(jnp.float32)
            ps = next_env_state.plane_state
            north = one_agent(ps.north)
            east = one_agent(ps.east)
            altitude = one_agent(ps.altitude)
            vt = one_agent(ps.vt)
            roll = one_agent(ps.roll)
            pitch = one_agent(ps.pitch)
            ax = one_agent(ps.ax)
            ay = one_agent(ps.ay)
            az = one_agent(ps.az)
            snapshot = jnp.stack([north, east, altitude, vt], axis=1)
            snap_idx = (step_idx + 1) // snapshot_every
            should_store = ((step_idx + 1) % snapshot_every) == 0
            snapshots = jax.lax.cond(
                should_store,
                lambda s: s.at[snap_idx].set(snapshot),
                lambda s: s,
                snapshots,
            )

            metric = jnp.array(
                [
                    jnp.mean(done_dict[agent].astype(jnp.float32)),
                    jnp.mean(reward_dict[agent]),
                    jnp.mean(altitude),
                    jnp.mean(vt),
                    jnp.mean(jnp.abs(roll)) * 180.0 / jnp.pi,
                    jnp.mean(jnp.abs(pitch)) * 180.0 / jnp.pi,
                    jnp.mean(jnp.maximum(jnp.maximum(jnp.abs(ax), jnp.abs(ay)), jnp.abs(az))),
                ],
                dtype=jnp.float32,
            )
            next_carry = (next_env_state, next_obs_dict[agent], next_done, hstate, key, snapshots)
            return next_carry, metric

        carry, metrics = jax.lax.scan(
            step_fn,
            (env_state, obs, done, hstate, seed_key, snapshots),
            jnp.arange(steps),
        )
        final_state, _, final_done, _, _, snapshots = carry
        ps = final_state.plane_state
        north = one_agent(ps.north)
        east = one_agent(ps.east)
        altitude = one_agent(ps.altitude)
        vt = one_agent(ps.vt)
        roll = one_agent(ps.roll)
        pitch = one_agent(ps.pitch)
        yaw = one_agent(ps.yaw)
        ax = one_agent(ps.ax)
        ay = one_agent(ps.ay)
        az = one_agent(ps.az)
        final = jnp.stack(
            [
                north,
                east,
                altitude,
                vt,
                roll * 180.0 / jnp.pi,
                pitch * 180.0 / jnp.pi,
                yaw * 180.0 / jnp.pi,
                jnp.maximum(jnp.maximum(jnp.abs(ax), jnp.abs(ay)), jnp.abs(az)),
                final_done,
            ],
            axis=1,
        )
        return metrics, snapshots, final

    return rollout


def _km(x):
    return x / 1000.0


def render_showcase(out_dir, tag, args, metrics, snapshots, final, cli_lines):
    out_dir.mkdir(parents=True, exist_ok=True)
    final_north = final[:, 0]
    final_east = final[:, 1]
    final_alt = final[:, 2]
    final_vt = final[:, 3]
    final_g = final[:, 7]

    snap_times = np.arange(snapshots.shape[0]) * args.snapshot_every
    traj_n = min(args.trajectory_samples, args.num_envs)
    rng = np.random.default_rng(args.seed)
    traj_idx = rng.choice(args.num_envs, size=traj_n, replace=False)

    fig = plt.figure(figsize=(20, 12), facecolor="#f6f7fb")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.0, 0.9], wspace=0.28, hspace=0.34)

    ax_cloud = fig.add_subplot(gs[:2, :2])
    colors = plt.cm.viridis(np.linspace(0.12, 0.95, snapshots.shape[0]))
    for i, color in enumerate(colors):
        alpha = 0.045 if i < snapshots.shape[0] - 1 else 0.24
        size = 2 if i < snapshots.shape[0] - 1 else 7
        ax_cloud.scatter(
            _km(snapshots[i, :, 1]),
            _km(snapshots[i, :, 0]),
            s=size,
            color=color,
            alpha=alpha,
            linewidths=0,
        )
    ax_cloud.scatter(_km(final_east), _km(final_north), s=4, c=final_alt, cmap="turbo", alpha=0.58, linewidths=0)
    ax_cloud.set_title("10,000 Planax Rollouts: Time-Slice Aircraft Clouds", fontsize=15, weight="bold")
    ax_cloud.set_xlabel("East (km)")
    ax_cloud.set_ylabel("North (km)")
    ax_cloud.grid(True, alpha=0.24)
    ax_cloud.axis("equal")

    ax_traj = fig.add_subplot(gs[0, 2:])
    xy = np.stack([_km(snapshots[:, traj_idx, 1]), _km(snapshots[:, traj_idx, 0])], axis=-1)
    segments = np.transpose(xy, (1, 0, 2))
    lc = LineCollection(segments, colors="#1f77b4", linewidths=0.55, alpha=0.22)
    ax_traj.add_collection(lc)
    ax_traj.scatter(_km(final_east), _km(final_north), s=3, c="#e4572e", alpha=0.20, linewidths=0)
    ax_traj.autoscale()
    ax_traj.set_title(f"{traj_n} Sample Paths + All 10,000 Final Points", fontsize=13, weight="bold")
    ax_traj.set_xlabel("East (km)")
    ax_traj.set_ylabel("North (km)")
    ax_traj.grid(True, alpha=0.24)
    ax_traj.axis("equal")

    ax_alt = fig.add_subplot(gs[1, 2])
    ax_alt.hist(final_alt, bins=70, color="#2b8cbe", alpha=0.88)
    ax_alt.axvline(final_alt.mean(), color="#111111", lw=1.4, ls="--")
    ax_alt.set_title("Final Altitude Distribution")
    ax_alt.set_xlabel("Altitude (m)")
    ax_alt.set_ylabel("Aircraft count")
    ax_alt.grid(True, axis="y", alpha=0.2)

    ax_vt = fig.add_subplot(gs[1, 3])
    ax_vt.hist(final_vt, bins=70, color="#f29e4c", alpha=0.88)
    ax_vt.axvline(final_vt.mean(), color="#111111", lw=1.4, ls="--")
    ax_vt.set_title("Final Airspeed Distribution")
    ax_vt.set_xlabel("Vt (m/s)")
    ax_vt.set_ylabel("Aircraft count")
    ax_vt.grid(True, axis="y", alpha=0.2)

    ax_ts = fig.add_subplot(gs[2, :2])
    dt = QuatLargeBatchTaskParams().agent_interaction_steps / QuatLargeBatchTaskParams().sim_freq
    t = np.arange(metrics.shape[0]) * dt
    ax_ts.plot(t, metrics[:, 2], label="mean altitude (m)", color="#2b8cbe")
    ax_ts_2 = ax_ts.twinx()
    ax_ts_2.plot(t, metrics[:, 3], label="mean Vt (m/s)", color="#f29e4c")
    ax_ts.set_title("Aggregate State Over All 10,000 Rollouts")
    ax_ts.set_xlabel("Sim time (s)")
    ax_ts.set_ylabel("Mean altitude (m)")
    ax_ts_2.set_ylabel("Mean Vt (m/s)")
    ax_ts.grid(True, alpha=0.24)

    ax_cli = fig.add_subplot(gs[2, 2:])
    ax_cli.set_facecolor("#111318")
    ax_cli.set_xticks([])
    ax_cli.set_yticks([])
    for spine in ax_cli.spines.values():
        spine.set_visible(False)
    def wrap_cli(lines, width):
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent="  ", replace_whitespace=False) or [""])
        return wrapped

    cli_text = "\n".join(wrap_cli(cli_lines[-14:], 90)[-16:])
    ax_cli.text(
        0.02,
        0.96,
        cli_text,
        transform=ax_cli.transAxes,
        va="top",
        ha="left",
        color="#d8dee9",
        fontsize=9.5,
        family="monospace",
        linespacing=1.2,
    )
    ax_cli.set_title("CLI Evidence Panel", loc="left", fontsize=13, weight="bold", color="#111111")

    fig.suptitle(
        "Planax GPU 10k-Scale Quaternion Baseline Showcase",
        fontsize=20,
        weight="bold",
        y=0.985,
    )
    png_path = out_dir / f"planax_10k_showcase_{tag}.png"
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    terminal_fig = plt.figure(figsize=(14, 7), facecolor="#111318")
    ax = terminal_fig.add_subplot(111)
    ax.set_facecolor("#111318")
    ax.set_axis_off()
    terminal_text = "\n".join(wrap_cli(cli_lines, 108))
    ax.text(
        0.025,
        0.965,
        terminal_text,
        va="top",
        ha="left",
        color="#d8dee9",
        fontsize=12,
        family="monospace",
        linespacing=1.18,
    )
    terminal_path = out_dir / f"cli_screenshot_{tag}.png"
    terminal_fig.savefig(terminal_path, dpi=170, bbox_inches="tight", facecolor="#111318")
    plt.close(terminal_fig)

    return png_path, terminal_path


def write_arrays(out_dir, tag, args, metrics, snapshots, final, summary):
    metrics_path = out_dir / f"step_metrics_{tag}.csv"
    header = "done_frac,mean_reward,mean_altitude,mean_vt,mean_abs_roll_deg,mean_abs_pitch_deg,mean_g_load"
    np.savetxt(metrics_path, metrics, delimiter=",", header=header, comments="")

    snapshot_path = out_dir / f"snapshots_{tag}.npz"
    np.savez_compressed(
        snapshot_path,
        snapshots=snapshots,
        final=final,
        snapshot_every=args.snapshot_every,
        columns=np.array(["north", "east", "altitude", "vt"]),
        final_columns=np.array(["north", "east", "altitude", "vt", "roll_deg", "pitch_deg", "yaw_deg", "g_load", "done"]),
    )

    summary_path = out_dir / f"summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_path, snapshot_path, summary_path


def main():
    args = parse_args()
    tag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cli_lines = []

    def log(line):
        print(line, flush=True)
        cli_lines.append(line)

    log("$ /home/dqy/miniconda3/envs/aeroplanax/bin/python render_quat_baseline_10k_showcase.py "
        f"--gpu {args.gpu} --num-envs {args.num_envs} --steps {args.steps} --snapshot-every {args.snapshot_every}")

    gpu_devices = jax.devices("gpu")
    if not gpu_devices:
        raise RuntimeError("No JAX GPU device is visible.")

    log(f"[planax] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log(f"[planax] JAX devices={[str(d) for d in gpu_devices]}")
    log(f"[planax] experiment=10k quaternion maneuver arena")
    log(f"[planax] envs={args.num_envs:,}, steps={args.steps:,}, snapshots every {args.snapshot_every} steps")

    env_params = QuatLargeBatchTaskParams()
    env = AeroPlanaxQuatLargeBatchEnv(env_params)
    network = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CONFIG)
    net_params, epoch = restore_params(network, env, env_params, args.checkpoint, args.seed)
    log(f"[checkpoint] epoch={epoch}")
    try:
        ckpt_display = str(Path(args.checkpoint).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        ckpt_display = Path(args.checkpoint).name
    log(f"[checkpoint] path={ckpt_display}")

    rollout = build_rollout(env, env_params, network, net_params, args.num_envs, args.steps, args.snapshot_every)
    rng = jax.random.PRNGKey(args.seed)

    log("[rollout] compiling + executing on GPU...")
    start = time.time()
    metrics, snapshots, final = rollout(rng)
    metrics, snapshots, final = jax.device_get((metrics, snapshots, final))
    metrics = np.asarray(metrics)
    snapshots = np.asarray(snapshots)
    final = np.asarray(final)
    elapsed = time.time() - start

    total_agent_steps = args.num_envs * args.steps
    throughput = total_agent_steps / max(elapsed, 1e-9)
    done_frac = float(final[:, 8].mean())
    final_alt = final[:, 2]
    final_vt = final[:, 3]
    final_range_km = float(np.sqrt(final[:, 0] ** 2 + final[:, 1] ** 2).max() / 1000.0)
    final_spread_km = float(np.std(final[:, 1] / 1000.0) + np.std(final[:, 0] / 1000.0))

    log("[done] Planax GPU 10k rollout finished")
    log(f"[done] wall_time_s={elapsed:.2f}")
    log(f"[done] throughput_agent_steps_s={throughput:,.0f}")
    log(f"[done] total_agent_steps={total_agent_steps:,}")
    log(f"[done] final_done_frac={done_frac:.5f}")
    log(f"[done] final_altitude_mean/min/max={final_alt.mean():.1f}/{final_alt.min():.1f}/{final_alt.max():.1f} m")
    log(f"[done] final_vt_mean/min/max={final_vt.mean():.1f}/{final_vt.min():.1f}/{final_vt.max():.1f} m/s")
    log(f"[done] final_range_max={final_range_km:.1f} km, spread_score={final_spread_km:.1f} km")

    summary = {
        "tag": tag,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "total_agent_steps": total_agent_steps,
        "gpu": str(args.gpu),
        "devices": [str(d) for d in gpu_devices],
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": epoch,
        "wall_time_s": elapsed,
        "throughput_agent_steps_s": throughput,
        "final_done_frac": done_frac,
        "final_altitude_mean": float(final_alt.mean()),
        "final_altitude_min": float(final_alt.min()),
        "final_altitude_max": float(final_alt.max()),
        "final_vt_mean": float(final_vt.mean()),
        "final_vt_min": float(final_vt.min()),
        "final_vt_max": float(final_vt.max()),
        "final_range_max_km": final_range_km,
        "final_spread_score_km": final_spread_km,
    }

    metrics_path, snapshot_path, summary_path = write_arrays(out_dir, tag, args, metrics, snapshots, final, summary)
    showcase_path, cli_png_path = render_showcase(out_dir, tag, args, metrics, snapshots, final, cli_lines)

    log(f"[output] showcase={showcase_path}")
    log(f"[output] cli_screenshot={cli_png_path}")
    log(f"[output] summary={summary_path}")
    log(f"[output] metrics={metrics_path}")
    log(f"[output] arrays={snapshot_path}")

    log_path = out_dir / f"cli_log_{tag}.txt"
    log_path.write_text("\n".join(cli_lines) + "\n", encoding="utf-8")

    # Refresh terminal screenshot after output paths are appended.
    render_showcase(out_dir, tag, args, metrics, snapshots, final, cli_lines)
    log(f"[output] cli_log={log_path}")


if __name__ == "__main__":
    main()
