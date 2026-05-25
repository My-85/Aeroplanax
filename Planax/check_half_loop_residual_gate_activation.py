import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from envs.aeroplanax_heading_pitch_V_quaternion_version_vertical_energy import (
    AeroPlanaxHeading_Pitch_V_Env,
    Heading_Pitch_V_TaskParams,
)
from envs.wrappers import LogWrapper
from half_loop_residual_policy import flatten_agent_axis, phase_features_from_state
from eval_vertical_energy_checkpoints import ActorCriticRNN, NET_CONFIG, ScannedRNN
from train_half_loop_specialist_residual_v1 import deep_update, restore_base_params


PLANAX_ROOT = Path(__file__).resolve().parent


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_transition_only(cfg: Dict[str, Any]) -> None:
    env_params = cfg.setdefault("ENV_PARAMS", {})
    env_params.update(
        {
            "original_task_prob": 0.0,
            "horizontal_proxy_task_prob": 0.0,
            "level_altitude_task_prob": 0.0,
            "half_loop_curriculum_prob": 1.0,
            "half_loop_pullup_retention_prob": 0.0,
            "half_loop_climb_retention_prob": 0.0,
            "half_loop_vertical_retention_prob": 0.0,
            "half_loop_transition_prob": 1.0,
            "half_loop_partial_prob": 0.0,
        }
    )


def scalar_rate(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=np.float32).mean()) if mask.size else 0.0


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def summarize_step(state, cfg: Dict[str, Any], batch_size: int) -> Dict[str, float]:
    inner = getattr(state, "env_state", state)
    phase_deg, gate = phase_features_from_state(
        state,
        batch_size,
        gate_start_deg=float(cfg.get("RESIDUAL_GATE_START_DEG", 80.0)),
        gate_end_deg=float(cfg.get("RESIDUAL_GATE_END_DEG", 180.0)),
    )
    mode = flatten_agent_axis(getattr(inner, "task_mode", 0), batch_size)
    duration = flatten_agent_axis(getattr(inner, "task_duration_steps", 1.0), batch_size)
    elapsed = flatten_agent_axis(getattr(inner, "time", 0.0), batch_size) - flatten_agent_axis(
        getattr(inner, "last_check_time", 0.0), batch_size
    )
    arc_start = jnp.rad2deg(flatten_agent_axis(getattr(inner, "task_arc_start_angle", 0.0), batch_size))
    arc_delta = jnp.rad2deg(flatten_agent_axis(getattr(inner, "task_arc_angle", 0.0), batch_size))
    arc_end = arc_start + arc_delta

    phase_np = np.asarray(phase_deg)
    gate_np = np.asarray(gate)
    mode_np = np.asarray(mode)
    loop_mask = ((mode_np == 5) | (mode_np == 9))
    window_mask = (phase_np >= float(cfg.get("RESIDUAL_GATE_START_DEG", 80.0))) & (
        phase_np <= float(cfg.get("RESIDUAL_GATE_END_DEG", 180.0))
    )

    if np.any(loop_mask):
        loop_phase = phase_np[loop_mask]
        loop_phase_min = float(np.min(loop_phase))
        loop_phase_mean = float(np.mean(loop_phase))
        loop_phase_max = float(np.max(loop_phase))
    else:
        loop_phase_min = 0.0
        loop_phase_mean = 0.0
        loop_phase_max = 0.0

    return {
        "gate_rate": float(np.mean(gate_np)),
        "loop_mode_rate": scalar_rate(loop_mask),
        "mode0_rate": scalar_rate(mode_np == 0),
        "mode5_rate": scalar_rate(mode_np == 5),
        "mode9_rate": scalar_rate(mode_np == 9),
        "phase_window_rate": scalar_rate(window_mask),
        "loop_phase_window_rate": scalar_rate(loop_mask & window_mask),
        "loop_phase_min": loop_phase_min,
        "loop_phase_mean": loop_phase_mean,
        "loop_phase_max": loop_phase_max,
        "elapsed_mean": float(np.mean(np.asarray(elapsed))),
        "duration_mean": float(np.mean(np.asarray(duration))),
        "arc_start_mean": float(np.mean(np.asarray(arc_start))),
        "arc_end_mean": float(np.mean(np.asarray(arc_end))),
    }


def force_initial_success_window(state):
    inner = getattr(state, "env_state", state)
    time = jnp.asarray(inner.time)
    duration = jnp.asarray(inner.task_duration_steps)
    while duration.ndim > time.ndim:
        duration = duration[..., 0]
    duration = jnp.ceil(duration).astype(time.dtype)
    inner = inner.replace(last_check_time=time - duration)
    if hasattr(state, "env_state"):
        return state.replace(env_state=inner)
    return inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="paper/second_paper/half_loop_specialist_residual_v1_config.json",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transition-only", action="store_true")
    parser.add_argument(
        "--force-initial-success",
        action="store_true",
        help="Move last_check_time backward after reset so the next env.step samples a curriculum task.",
    )
    parser.add_argument(
        "--action-source",
        choices=["neutral", "base"],
        default="base",
        help="Use neutral controls or the frozen base checkpoint to advance the environment.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/half_loop_specialist_residual_v1_gate_debug",
    )
    args = parser.parse_args()

    cfg = {
        "RESIDUAL_GATE_START_DEG": 80.0,
        "RESIDUAL_GATE_END_DEG": 180.0,
        "ENV_PARAMS": {},
    }
    deep_update(cfg, load_config(args.config))
    if args.transition_only:
        apply_transition_only(cfg)

    out_dir = PLANAX_ROOT / args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    env_params = Heading_Pitch_V_TaskParams(**cfg.get("ENV_PARAMS", {}))
    env = LogWrapper(AeroPlanaxHeading_Pitch_V_Env(env_params))
    batch_size = args.num_envs * env.num_agents

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, args.num_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
    if args.force_initial_success:
        state = force_initial_success_window(state)

    neutral = jnp.tile(jnp.array([[20, 20, 20, 20, 0]], dtype=jnp.int32), (args.num_envs, 1))
    neutral_action = {env.agents[0]: neutral}
    last_done = jnp.zeros((batch_size,), dtype=bool)
    base_net = None
    base_params = None
    base_hstate = None
    if args.action_source == "base":
        base_params, _ = restore_base_params(cfg["BASE_CHECKPOINT"])
        base_net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CONFIG)
        base_hstate = ScannedRNN.initialize_carry(batch_size, NET_CONFIG["GRU_HIDDEN_DIM"])
    rows = []

    for step in range(args.steps):
        row = {"step": step}
        row.update(summarize_step(state, cfg, batch_size))
        rows.append(row)

        if args.action_source == "base":
            obs_flat = batchify(obs, env.agents, args.num_envs, env.num_agents)
            base_hstate, pi, _ = base_net.apply(
                base_params,
                base_hstate,
                (obs_flat[None, :, :], last_done[None, :]),
            )
            action_flat = jnp.stack([p.mode()[0] for p in pi], axis=-1).astype(jnp.int32)
            action = unbatchify(action_flat, env.agents, args.num_envs, env.num_agents)
        else:
            action = neutral_action

        rng, step_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, args.num_envs)
        obs, state, _, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, state, action)
        last_done = batchify(done, env.agents, args.num_envs, env.num_agents).reshape(-1)
        if args.action_source == "base":
            base_hstate = jnp.where(last_done[:, None], jnp.zeros_like(base_hstate), base_hstate)

    csv_path = out_dir / "gate_activation_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gate_rates = np.asarray([r["gate_rate"] for r in rows], dtype=np.float32)
    loop_rates = np.asarray([r["loop_mode_rate"] for r in rows], dtype=np.float32)
    loop_window_rates = np.asarray([r["loop_phase_window_rate"] for r in rows], dtype=np.float32)
    first_gate = int(np.argmax(gate_rates > 0.0)) if np.any(gate_rates > 0.0) else None
    first_loop = int(np.argmax(loop_rates > 0.0)) if np.any(loop_rates > 0.0) else None
    first_loop_window = (
        int(np.argmax(loop_window_rates > 0.0)) if np.any(loop_window_rates > 0.0) else None
    )
    summary = {
        "config": str(Path(args.config).resolve()),
        "transition_only": bool(args.transition_only),
        "force_initial_success": bool(args.force_initial_success),
        "action_source": args.action_source,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "mean_gate_rate": float(np.mean(gate_rates)),
        "max_gate_rate": float(np.max(gate_rates)),
        "mean_loop_mode_rate": float(np.mean(loop_rates)),
        "max_loop_mode_rate": float(np.max(loop_rates)),
        "mean_loop_phase_window_rate": float(np.mean(loop_window_rates)),
        "max_loop_phase_window_rate": float(np.max(loop_window_rates)),
        "first_gate_step": first_gate,
        "first_loop_step": first_loop,
        "first_loop_phase_window_step": first_loop_window,
        "timeseries": str(csv_path.resolve()),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report = [
        "# Residual Gate Activation Smoke Test",
        "",
        f"- transition_only: `{args.transition_only}`",
        f"- force_initial_success: `{args.force_initial_success}`",
        f"- action_source: `{args.action_source}`",
        f"- num_envs: `{args.num_envs}`",
        f"- steps: `{args.steps}`",
        f"- mean_gate_rate: `{summary['mean_gate_rate']:.8f}`",
        f"- max_gate_rate: `{summary['max_gate_rate']:.8f}`",
        f"- first_gate_step: `{summary['first_gate_step']}`",
        f"- first_loop_step: `{summary['first_loop_step']}`",
        f"- first_loop_phase_window_step: `{summary['first_loop_phase_window_step']}`",
        "",
        "Use this before residual training. If `max_gate_rate` is zero, the training-side phase/gate path is broken.",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
