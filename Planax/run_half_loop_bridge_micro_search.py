import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-2c45b7fd-69c8-1697-23a0-fe7ce7a2a620")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import jax
import jax.numpy as jnp
import numpy as np

import eval_loop_quality_claude_aligned as ev
from envs.termination_conditions import crashed_fn, timeout_fn
from half_loop_residual_policy import (
    ResidualActorCriticRNN,
    ResidualScannedRNN,
    augment_obs_with_phase,
    combine_base_and_residual_logits,
)
from termination_trace_utils import classify_terminal_reason, terminal_state_from_info


PLANAX_ROOT = Path(__file__).resolve().parent
GPU_UUID = "GPU-2c45b7fd-69c8-1697-23a0-fe7ce7a2a620"
BASE_CKPT = PLANAX_ROOT / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
BEST_RESIDUAL = (
    PLANAX_ROOT
    / "results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2"
)
BEST_RESIDUAL_CONFIG = (
    PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/configs/round_01_config.json"
)
AUTO_SEARCH_ROOT = PLANAX_ROOT / "results/half_loop_residual_auto_search/20260520_1146"

TASKS = {
    "pu150_R12000": ("pu150_R12000", 150, 12000, 1200, 500, 2000),
    "pu165_R15000": ("pu165_R15000", 165, 15000, 1300, 500, 2300),
    "pu170_R15000": ("pu170_R15000", 170, 15000, 1400, 500, 2400),
}

SUMMARY_FIELDS = [
    "policy",
    "task",
    "scale",
    "gate_start",
    "gate_end",
    "smooth_margin",
    "target_vt",
    "lookahead_mode",
    "completed",
    "steps",
    "termination",
    "done_reason",
    "terminal_reason_classified",
    "terminal_reason_raw",
    "CTE_mean",
    "velocity_tangent_error_mean",
    "nose_tangent_error_mean",
    "nose_velocity_error_mean",
    "wing_plane_error_mean",
    "q_error_mean_rad",
    "env_alpha_max",
    "env_beta_max",
    "vt_min",
    "vt_max",
    "Gmax",
    "alt_min",
    "alt_max",
    "phase145_170_CTE_mean",
    "phase145_170_velocity_tangent_error_mean",
    "phase145_170_nose_tangent_error_mean",
    "phase145_170_wing_plane_error_mean",
    "phase145_170_Gmax",
    "phase145_170_alpha_max",
    "phase145_170_vt_min",
    "residual_logits_norm_mean",
    "final_base_logits_norm_mean",
    "action_diff_norm_mean",
    "action_diff_norm_max",
    "gate_jump_max",
    "jitter_action_diff_mean",
]

PHASE_FIELDS = [
    "policy",
    "task",
    "step",
    "time_sec",
    "phase",
    "CTE",
    "velocity_tangent_error",
    "nose_tangent_error",
    "nose_velocity_error",
    "wing_plane_error",
    "q_error_norm",
    "alpha",
    "beta",
    "G",
    "vt",
    "altitude",
    "pitch",
    "roll",
    "yaw",
    "north",
    "east",
    "target_pitch",
    "target_roll",
    "elevator_action",
    "aileron_action",
    "rudder_action",
    "throttle_action",
    "speedbrake_action",
    "base_logits_norm",
    "residual_logits_norm",
    "final_base_logits_norm",
    "residual_gate_value",
    "action_difference_from_base",
]


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def task_scale_key(row):
    try:
        scale = round(float(row.get("scale", "")), 6)
    except (TypeError, ValueError):
        return None
    return (scale, row.get("task", ""))


def copy_row_with_policy(row, policy):
    copied = dict(row)
    copied["policy"] = policy
    return copied


def f(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_windows(text):
    windows = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        start, end = item.split("-", 1)
        windows.append((int(float(start)), int(float(end))))
    return windows


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def continuous_action(action_idx):
    arr = np.asarray(action_idx, dtype=np.float64)
    return np.array(
        [
            arr[0] / 30.0,
            arr[1] * 2.0 / 40.0 - 1.0,
            arr[2] * 2.0 / 40.0 - 1.0,
            arr[3] * 2.0 / 40.0 - 1.0,
            arr[4] / 4.0,
        ],
        dtype=np.float64,
    )


def logits_norm(items):
    total = 0.0
    for item in items:
        logits = item.logits if hasattr(item, "logits") else item
        arr = np.asarray(logits, dtype=np.float64).reshape(-1)
        total += float(np.sum(arr * arr))
    return float(np.sqrt(total))


def array_tuple_norm(items):
    total = 0.0
    for item in items:
        arr = np.asarray(item, dtype=np.float64).reshape(-1)
        total += float(np.sum(arr * arr))
    return float(np.sqrt(total))


def termination_reason(state, params):
    crash, _ = crashed_fn(state, params, 0)
    timeout, _ = timeout_fn(state, params, 0)
    if bool(np.asarray(crash)):
        return "crash_or_overload"
    if bool(np.asarray(timeout)):
        return "env_timeout"
    return "done_unknown"


def make_residual_cfg(scale=1.0, gate_start=80.0, gate_end=180.0, smooth_margin=0.0, force_gate_off=False):
    cfg = ev.load_residual_config(BEST_RESIDUAL_CONFIG)
    cfg["RESIDUAL_SCALE"] = float(scale)
    cfg["RESIDUAL_GATE_START_DEG"] = float(gate_start)
    cfg["RESIDUAL_GATE_END_DEG"] = float(gate_end)
    cfg["RESIDUAL_PHASE_MAX_DEG"] = max(float(cfg.get("RESIDUAL_PHASE_MAX_DEG", 180.0)), float(gate_end), 180.0)
    cfg["RESIDUAL_SMOOTH_GATE_MARGIN_DEG"] = float(smooth_margin)
    cfg["RESIDUAL_FORCE_GATE_OFF"] = bool(force_gate_off)
    return cfg


def load_models():
    env = ev.Env(ev.Params())
    net = ev.ActorCriticRNN([31, 41, 41, 41, 5], config=ev.NET_CFG)
    obs_shape = env.observation_space(env.agents[0], ev.Params()).shape
    h0 = ev.ScannedRNN.initialize_carry(1, ev.NET_CFG["GRU_HIDDEN_DIM"])
    _ = net.init(
        jax.random.PRNGKey(ev.SEED),
        h0,
        (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1))),
    )
    net_params = ev.restore_params(BASE_CKPT)
    residual_cfg = make_residual_cfg()
    residual_net = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=residual_cfg)
    residual_params, residual_epoch = ev.restore_residual_params(BEST_RESIDUAL)
    return env, net, net_params, residual_net, residual_params, residual_epoch


def run_detailed_test(
    env,
    net,
    net_params,
    residual_net,
    residual_params,
    policy_name,
    task,
    residual_cfg=None,
    target_vt=250.0,
    lookahead_mode="default",
):
    name, angle_deg, radius_m, lookahead, reach_radius, max_steps = TASKS[task]
    if lookahead_mode == "conservative":
        lookahead = lookahead * 1.25
    elif lookahead_mode == "relaxed":
        lookahead = lookahead * 1.50
    elif lookahead_mode != "default":
        raise ValueError(f"unknown lookahead_mode={lookahead_mode}")

    wps, meta = ev.vertical_pullup_arc(
        0,
        0,
        5000,
        0.0,
        radius=radius_m,
        arc_angle_deg=angle_deg,
        n_points=max(80, int(angle_deg * 2 / 3)),
    )
    total_arc = meta["total_length_m"]
    planner = ev.PurePursuitPlanner(
        ev.PlannerConfig(
            lookahead_dist=lookahead,
            reach_radius=reach_radius,
            blend_steps=250,
            target_vt=float(target_vt),
        )
    )

    rng = jax.random.PRNGKey(ev.SEED)
    rng, reset_key = jax.random.split(rng)
    _, state = env.reset(reset_key, ev.Params())
    q_nb_init = ev._quat_from_euler_nb(0.0, 0.0, 0.0)
    q_bn_init = ev._quat_conj(q_nb_init)
    state = state.replace(
        plane_state=state.plane_state.replace(
            yaw=jnp.array([0.0]),
            q0=jnp.array([q_bn_init[0]]),
            q1=jnp.array([q_bn_init[1]]),
            q2=jnp.array([q_bn_init[2]]),
            q3=jnp.array([q_bn_init[3]]),
        ),
        target_heading=jnp.array([0.0]),
    )
    planner.reset(wps, 0.0, 0.0, 0.0, float(target_vt))

    hstate = ev.ScannedRNN.initialize_carry(1, ev.NET_CFG["GRU_HIDDEN_DIM"])
    residual_hstate = None
    if residual_cfg is not None:
        residual_hstate = ResidualScannedRNN.initialize_carry(
            1, int(residual_cfg.get("RESIDUAL_GRU_HIDDEN_DIM", 64))
        )
    done_flag = jnp.zeros((1,))
    rec = {key: [] for key in [
        "step", "t", "n", "e", "a", "vt", "roll", "pitch", "yaw", "t_roll",
        "t_pitch", "t_hdg", "alpha", "beta", "G", "cte", "q0", "q1", "q2", "q3",
        "wp_idx", "theta_deg", "throttle", "elevator", "aileron", "rudder",
        "speedbrake", "base_logits_norm", "residual_logits_norm",
        "final_base_logits_norm", "gate", "action_diff_norm",
    ]}
    done_reason = ""
    terminal_reason_raw = ""
    done_flag_bool = False

    for step in range(max_steps):
        ps = state.plane_state
        north = ev.f_scalar(ps.north)
        east = ev.f_scalar(ps.east)
        alt = ev.f_scalar(ps.altitude)
        vt = ev.f_scalar(ps.vt)
        roll = ev.f_scalar(ps.roll)
        pitch = ev.f_scalar(ps.pitch)
        yaw = ev.f_scalar(ps.yaw)
        alpha = ev.f_scalar(ps.alpha)
        beta = ev.f_scalar(ps.beta)
        ax = ev.f_scalar(ps.ax)
        ay = ev.f_scalar(ps.ay)
        az = ev.f_scalar(ps.az)

        result = planner.step(north, east, alt, yaw, pitch, roll, vt)
        target_heading = result["target_heading"]
        target_pitch = result["target_pitch"]
        target_roll = result["target_roll"]
        planner_target_vt = result["target_vt"]

        path_s = planner.path_progress
        theta_deg = (path_s / total_arc) * angle_deg if total_arc > 0 else 0.0
        theta_deg = float(np.clip(theta_deg, 0.0, angle_deg))
        target_loop_roll = ev.loop_roll(theta_deg)
        blend = min(1.0, step / 250.0)
        target_roll = float(
            np.arctan2(
                np.sin(roll + blend * (target_loop_roll - roll)),
                np.cos(roll + blend * (target_loop_roll - roll)),
            )
        )

        state = state.replace(
            target_heading=jnp.array([target_heading]),
            target_pitch=jnp.array([target_pitch]),
            target_roll=jnp.array([target_roll]),
            target_vt=jnp.array([float(planner_target_vt)], dtype=jnp.float32),
        )

        obs = env._get_obs(state, ev.Params())[env.agents[0]][None, None, :]
        hstate, base_pi, _ = net.apply(net_params, hstate, (obs, done_flag[None, :]))
        base_actions = [int(p.mode()[0, 0]) for p in base_pi]
        base_cont = continuous_action(base_actions)
        base_norm = logits_norm(base_pi)
        residual_norm = 0.0
        final_delta_norm = 0.0
        gate = 0.0

        if residual_cfg is not None:
            gate = ev.residual_gate_value(theta_deg, residual_cfg)
            obs_aug = augment_obs_with_phase(
                obs.reshape((1, -1)),
                state,
                float(theta_deg),
                gate,
                residual_cfg,
            )
            residual_hstate, residual_logits, _ = residual_net.apply(
                residual_params,
                residual_hstate,
                (obs_aug[None, :, :], done_flag[None, :]),
            )
            pi_out, clipped_delta, _ = combine_base_and_residual_logits(
                base_pi, residual_logits, obs_aug, residual_cfg
            )
            residual_norm = array_tuple_norm(clipped_delta)
            final_delta_norm = logits_norm(pi_out) - base_norm
            final_delta_norm = array_tuple_norm(
                [p.logits - b.logits for p, b in zip(pi_out, base_pi)]
            )
        else:
            pi_out = base_pi

        actions = [int(p.mode()[0, 0]) for p in pi_out]
        cont = continuous_action(actions)
        action_diff = float(np.linalg.norm(cont - base_cont))

        rng, step_key = jax.random.split(rng)
        _, next_state, _, done, info = env.step(
            step_key, state, {env.agents[0]: jnp.array(actions)}, ev.Params()
        )
        done_flag = jnp.array([float(done[env.agents[0]])])

        wp_idx = result["path_ctx"]["wp_idx"]
        rec["step"].append(step)
        rec["t"].append(step * 0.2)
        rec["n"].append(north)
        rec["e"].append(east)
        rec["a"].append(alt)
        rec["vt"].append(vt)
        rec["roll"].append(np.degrees(roll))
        rec["pitch"].append(np.degrees(pitch))
        rec["yaw"].append(np.degrees(yaw))
        rec["t_roll"].append(np.degrees(target_roll))
        rec["t_pitch"].append(np.degrees(target_pitch))
        rec["t_hdg"].append(np.degrees(target_heading))
        rec["alpha"].append(np.degrees(alpha))
        rec["beta"].append(np.degrees(beta))
        rec["G"].append(float(np.sqrt(ax * ax + ay * ay + az * az)))
        rec["cte"].append(ev.compute_true_cte(np.array([north, east, alt]), wps, wp_idx, 10))
        rec["q0"].append(ev.f_scalar(ps.q0))
        rec["q1"].append(ev.f_scalar(ps.q1))
        rec["q2"].append(ev.f_scalar(ps.q2))
        rec["q3"].append(ev.f_scalar(ps.q3))
        rec["wp_idx"].append(wp_idx)
        rec["theta_deg"].append(theta_deg)
        rec["throttle"].append(cont[0])
        rec["elevator"].append(cont[1])
        rec["aileron"].append(cont[2])
        rec["rudder"].append(cont[3])
        rec["speedbrake"].append(cont[4])
        rec["base_logits_norm"].append(base_norm)
        rec["residual_logits_norm"].append(residual_norm)
        rec["final_base_logits_norm"].append(final_delta_norm)
        rec["gate"].append(gate)
        rec["action_diff_norm"].append(action_diff)

        if bool(done[env.agents[0]]):
            terminal_state = terminal_state_from_info(info, next_state)
            trace = classify_terminal_reason(
                terminal_state,
                params=ev.Params(),
                done_flag=True,
                planner_completed=planner.is_done(),
                agent_id=0,
            )
            done_reason = trace["terminal_reason_classified"]
            terminal_reason_raw = trace["terminal_reason_raw"]
            done_flag_bool = True
            state = next_state
            break
        state = next_state
        if planner.is_done():
            break

    n = len(rec["t"])
    completed = planner.is_done() and not done_flag_bool
    geo = {
        "velocity_tangent_error": [],
        "nose_tangent_error": [],
        "nose_velocity_error": [],
        "wing_plane_error": [],
        "q_error_rad": [],
    }
    for i in range(n):
        q_bn = np.array(
            [rec["q0"][i], rec["q1"][i], rec["q2"][i], rec["q3"][i]], dtype=np.float64
        )
        q_bn = q_bn / (np.linalg.norm(q_bn) + 1e-12)
        x_body_neu = ev.ned_to_neu(ev.rotate_body_to_ned(q_bn, np.array([1.0, 0.0, 0.0])))
        y_body_neu = ev.ned_to_neu(ev.rotate_body_to_ned(q_bn, np.array([0.0, 1.0, 0.0])))
        alpha_rad = np.radians(rec["alpha"][i])
        beta_rad = np.radians(rec["beta"][i])
        ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
        cb, sb = np.cos(beta_rad), np.sin(beta_rad)
        v_body = np.array([rec["vt"][i] * ca * cb, rec["vt"][i] * sb, rec["vt"][i] * sa * cb])
        v_neu = ev.ned_to_neu(ev.rotate_body_to_ned(q_bn, v_body))
        v_hat_neu = v_neu / (np.linalg.norm(v_neu) + 1e-12)
        t_ref_neu, n_loop_neu = ev.compute_loop_reference(wps, rec["wp_idx"][i])
        geo["velocity_tangent_error"].append(ev.angle_between(v_hat_neu, t_ref_neu))
        geo["nose_tangent_error"].append(ev.angle_between(x_body_neu, t_ref_neu))
        geo["nose_velocity_error"].append(ev.angle_between(x_body_neu, v_hat_neu))
        geo["wing_plane_error"].append(ev.angle_between(y_body_neu, n_loop_neu))
        geo["q_error_rad"].append(
            ev.quat_error_angle(
                q_bn,
                np.radians(rec["t_hdg"][i]),
                np.radians(rec["t_pitch"][i]),
                np.radians(rec["t_roll"][i]),
            )
        )

    def arr(key):
        return np.asarray(rec[key], dtype=np.float64)

    def garr(key):
        return np.asarray(geo[key], dtype=np.float64)

    theta = arr("theta_deg")
    phase_mask = (theta >= 145.0) & (theta <= 170.0)
    if not np.any(phase_mask):
        phase_mask = np.ones_like(theta, dtype=bool)
    action_diff = arr("action_diff_norm")
    gate_arr = arr("gate")
    termination = "ok" if completed else (done_reason or "timeout")
    cte = arr("cte")
    summary = {
        "policy": policy_name,
        "task": task,
        "scale": "" if residual_cfg is None else residual_cfg.get("RESIDUAL_SCALE", ""),
        "gate_start": "" if residual_cfg is None else residual_cfg.get("RESIDUAL_GATE_START_DEG", ""),
        "gate_end": "" if residual_cfg is None else residual_cfg.get("RESIDUAL_GATE_END_DEG", ""),
        "smooth_margin": "" if residual_cfg is None else residual_cfg.get("RESIDUAL_SMOOTH_GATE_MARGIN_DEG", ""),
        "target_vt": target_vt,
        "lookahead_mode": lookahead_mode,
        "completed": bool(completed),
        "steps": n,
        "termination": termination,
        "done_reason": done_reason,
        "terminal_reason_classified": done_reason,
        "terminal_reason_raw": terminal_reason_raw,
        "CTE_mean": float(cte.mean()),
        "velocity_tangent_error_mean": float(garr("velocity_tangent_error").mean()),
        "nose_tangent_error_mean": float(garr("nose_tangent_error").mean()),
        "nose_velocity_error_mean": float(garr("nose_velocity_error").mean()),
        "wing_plane_error_mean": float(garr("wing_plane_error").mean()),
        "q_error_mean_rad": float(garr("q_error_rad").mean()),
        "env_alpha_max": float(arr("alpha").max()),
        "env_beta_max": float(arr("beta").max()),
        "vt_min": float(arr("vt").min()),
        "vt_max": float(arr("vt").max()),
        "Gmax": float(arr("G").max()),
        "alt_min": float(arr("a").min()),
        "alt_max": float(arr("a").max()),
        "phase145_170_CTE_mean": float(cte[phase_mask].mean()),
        "phase145_170_velocity_tangent_error_mean": float(garr("velocity_tangent_error")[phase_mask].mean()),
        "phase145_170_nose_tangent_error_mean": float(garr("nose_tangent_error")[phase_mask].mean()),
        "phase145_170_wing_plane_error_mean": float(garr("wing_plane_error")[phase_mask].mean()),
        "phase145_170_Gmax": float(arr("G")[phase_mask].max()),
        "phase145_170_alpha_max": float(arr("alpha")[phase_mask].max()),
        "phase145_170_vt_min": float(arr("vt")[phase_mask].min()),
        "residual_logits_norm_mean": float(arr("residual_logits_norm").mean()),
        "final_base_logits_norm_mean": float(arr("final_base_logits_norm").mean()),
        "action_diff_norm_mean": float(action_diff.mean()),
        "action_diff_norm_max": float(action_diff.max()),
        "gate_jump_max": float(np.max(np.abs(np.diff(gate_arr)))) if len(gate_arr) > 1 else 0.0,
        "jitter_action_diff_mean": float(np.mean(np.abs(np.diff(action_diff)))) if len(action_diff) > 1 else 0.0,
    }

    phase_rows = []
    for i in range(n):
        phase_rows.append(
            {
                "policy": policy_name,
                "task": task,
                "step": rec["step"][i],
                "time_sec": rec["t"][i],
                "phase": rec["theta_deg"][i],
                "CTE": rec["cte"][i],
                "velocity_tangent_error": geo["velocity_tangent_error"][i],
                "nose_tangent_error": geo["nose_tangent_error"][i],
                "nose_velocity_error": geo["nose_velocity_error"][i],
                "wing_plane_error": geo["wing_plane_error"][i],
                "q_error_norm": geo["q_error_rad"][i],
                "alpha": rec["alpha"][i],
                "beta": rec["beta"][i],
                "G": rec["G"][i],
                "vt": rec["vt"][i],
                "altitude": rec["a"][i],
                "pitch": rec["pitch"][i],
                "roll": rec["roll"][i],
                "target_pitch": rec["t_pitch"][i],
                "target_roll": rec["t_roll"][i],
                "elevator_action": rec["elevator"][i],
                "aileron_action": rec["aileron"][i],
                "rudder_action": rec["rudder"][i],
                "throttle_action": rec["throttle"][i],
                "speedbrake_action": rec["speedbrake"][i],
                "base_logits_norm": rec["base_logits_norm"][i],
                "residual_logits_norm": rec["residual_logits_norm"][i],
                "final_base_logits_norm": rec["final_base_logits_norm"][i],
                "residual_gate_value": rec["gate"][i],
                "action_difference_from_base": rec["action_diff_norm"][i],
            }
        )
    return summary, phase_rows


def cached_baselines():
    rows = {}
    path = AUTO_SEARCH_ROOT / "baseline_update_2/quick/loop_quality_summary.csv"
    for row in read_csv(path):
        rows[("update2_scale1", row["name"])] = row
    path = AUTO_SEARCH_ROOT / "round_01/scale_ablation/scale_0.25/quick/loop_quality_summary.csv"
    for row in read_csv(path):
        rows[("update2_scale0.25", row["name"])] = row
    return rows


def run_phase_diagnostics(root, env, net, net_params, residual_net, residual_params, tasks):
    policies = [
        ("update2_scale1.0", make_residual_cfg(scale=1.0)),
        ("update2_scale0.25", make_residual_cfg(scale=0.25)),
        ("update2_scale0.125", make_residual_cfg(scale=0.125)),
        ("base_epoch619_only", None),
    ]
    rows = []
    for policy_name, cfg in policies:
        for task in tasks:
            summary, phase_rows = run_detailed_test(
                env, net, net_params, residual_net, residual_params, policy_name, task, cfg
            )
            rows.append(summary)
            write_csv(root / "phasewise" / f"{policy_name}_{task}.csv", phase_rows, PHASE_FIELDS)
            print(
                f"{policy_name} {task} term={summary['termination']} CTE={summary['CTE_mean']:.1f} "
                f"Gmax={summary['Gmax']:.2f} wing={summary['wing_plane_error_mean']:.1f}",
                flush=True,
            )
    write_csv(root / "phasewise_diagnostics.csv", rows, SUMMARY_FIELDS)
    return rows


def run_scale_sweep(root, env, net, net_params, residual_net, residual_params, tasks, phase_rows=None):
    phase_rows = phase_rows or []
    existing_rows = read_csv(root / "scale_sweep.csv")
    existing_by_key = {task_scale_key(row): row for row in existing_rows if task_scale_key(row)}
    phase_by_key = {task_scale_key(row): row for row in phase_rows if task_scale_key(row)}
    rows = []
    for scale in [0.05, 0.10, 0.125, 0.20, 0.25, 0.35]:
        cfg = make_residual_cfg(scale=scale)
        for task in tasks:
            key = (round(float(scale), 6), task)
            if key in existing_by_key:
                row = copy_row_with_policy(existing_by_key[key], f"scale_{scale:g}")
                rows.append(row)
                print(
                    f"scale={scale:g} {task} cached term={row['termination']} "
                    f"CTE={f(row,'CTE_mean'):.1f} Gmax={f(row,'Gmax'):.2f} "
                    f"wing={f(row,'wing_plane_error_mean'):.1f}",
                    flush=True,
                )
                continue
            if key in phase_by_key:
                row = copy_row_with_policy(phase_by_key[key], f"scale_{scale:g}")
                rows.append(row)
                print(
                    f"scale={scale:g} {task} reused_phase term={row['termination']} "
                    f"CTE={f(row,'CTE_mean'):.1f} Gmax={f(row,'Gmax'):.2f} "
                    f"wing={f(row,'wing_plane_error_mean'):.1f}",
                    flush=True,
                )
                continue
            summary, phase_rows = run_detailed_test(
                env,
                net,
                net_params,
                residual_net,
                residual_params,
                f"scale_{scale:g}",
                task,
                cfg,
            )
            rows.append(summary)
            if task == "pu165_R15000":
                write_csv(root / "phasewise" / f"scale_{scale:g}_{task}.csv", phase_rows, PHASE_FIELDS)
            write_csv(root / "scale_sweep.csv", rows, SUMMARY_FIELDS)
            print(
                f"scale={scale:g} {task} term={summary['termination']} CTE={summary['CTE_mean']:.1f} "
                f"Gmax={summary['Gmax']:.2f} wing={summary['wing_plane_error_mean']:.1f}",
                flush=True,
            )
    write_csv(root / "scale_sweep.csv", rows, SUMMARY_FIELDS)
    return rows


def select_best_scale(scale_rows):
    baseline = {}
    cached = cached_baselines()
    for key, row in cached.items():
        if key[0] == "update2_scale1":
            baseline[key[1]] = row
    safe = []
    for row in scale_rows:
        if row["task"] != "pu165_R15000":
            continue
        b = baseline.get("pu165_R15000")
        if not b:
            continue
        improves = (
            f(row, "CTE_mean") < f(b, "CTE_mean")
            and f(row, "wing_plane_error_mean") <= f(b, "wing_plane_error_mean")
            and f(row, "nose_tangent_error_mean") <= f(b, "nose_tangent_error_mean")
        )
        if improves and f(row, "Gmax") < 9.0:
            safe.append(row)
    if safe:
        return max(safe, key=lambda r: f(r, "scale"))
    pu165_rows = [r for r in scale_rows if r["task"] == "pu165_R15000"]
    if not pu165_rows:
        return {"scale": 0.125}
    return min(pu165_rows, key=lambda r: (f(r, "Gmax") >= 9.0, f(r, "Gmax"), f(r, "CTE_mean")))


def run_gate_sweep(
    root,
    env,
    net,
    net_params,
    residual_net,
    residual_params,
    selected_scale,
    windows=None,
    margins=None,
    tasks=None,
):
    existing_rows = read_csv(root / "gate_sweep.csv")
    existing_by_key = {}
    for row in existing_rows:
        try:
            key = (
                round(float(row.get("scale", "")), 6),
                int(float(row.get("gate_start", ""))),
                int(float(row.get("gate_end", ""))),
                int(float(row.get("smooth_margin", ""))),
                row.get("task", ""),
            )
        except (TypeError, ValueError):
            continue
        existing_by_key[key] = row
    windows = windows or [(80, 180), (100, 180), (120, 180), (140, 175), (145, 170), (150, 170)]
    margins = margins or [0, 5, 10, 15, 20]
    tasks = tasks or ["pu150_R12000", "pu165_R15000"]
    rows = []
    for gate_start, gate_end in windows:
        for margin in margins:
            cfg = make_residual_cfg(
                scale=selected_scale,
                gate_start=gate_start,
                gate_end=gate_end,
                smooth_margin=margin,
            )
            for task in tasks:
                key = (
                    round(float(selected_scale), 6),
                    gate_start,
                    gate_end,
                    int(float(margin)),
                    task,
                )
                if key in existing_by_key:
                    row = existing_by_key[key]
                    rows.append(row)
                    print(
                        f"gate={gate_start}-{gate_end} margin={margin} {task} cached "
                        f"term={row['termination']} Gmax={f(row,'Gmax'):.2f} CTE={f(row,'CTE_mean'):.1f}",
                        flush=True,
                    )
                    continue
                summary, _ = run_detailed_test(
                    env,
                    net,
                    net_params,
                    residual_net,
                    residual_params,
                    f"gate_{gate_start}_{gate_end}_m{margin}",
                    task,
                    cfg,
                )
                rows.append(summary)
                write_csv(root / "gate_sweep.csv", rows, SUMMARY_FIELDS)
                print(
                    f"gate={gate_start}-{gate_end} margin={margin} {task} "
                    f"term={summary['termination']} Gmax={summary['Gmax']:.2f} CTE={summary['CTE_mean']:.1f}",
                    flush=True,
                )
    write_csv(root / "gate_sweep.csv", rows, SUMMARY_FIELDS)
    return rows


def run_target_stream_sanity(
    root,
    env,
    net,
    net_params,
    residual_net,
    residual_params,
    cfg,
    target_vts=None,
    lookahead_modes=None,
):
    existing_rows = read_csv(root / "target_stream_sweep.csv")
    existing_by_key = {}
    for row in existing_rows:
        try:
            key = (
                round(float(row.get("scale", "")), 6),
                round(float(row.get("target_vt", "")), 3),
                row.get("lookahead_mode", ""),
            )
        except (TypeError, ValueError):
            continue
        existing_by_key[key] = row
    rows = []
    manifest = {"tests": []}
    target_vts = target_vts or [220.0, 240.0, 250.0, 260.0]
    lookahead_modes = lookahead_modes or ["conservative", "default", "relaxed"]
    for target_vt in target_vts:
        for mode in lookahead_modes:
            key = (
                round(float(cfg.get("RESIDUAL_SCALE", 0.0)), 6),
                round(float(target_vt), 3),
                mode,
            )
            if key in existing_by_key:
                row = existing_by_key[key]
                rows.append(row)
                manifest["tests"].append(row)
                print(
                    f"target_vt={target_vt:g} lookahead={mode} pu165 cached "
                    f"term={row['termination']} Gmax={f(row,'Gmax'):.2f} CTE={f(row,'CTE_mean'):.1f}",
                    flush=True,
                )
                continue
            summary, _ = run_detailed_test(
                env,
                net,
                net_params,
                residual_net,
                residual_params,
                f"target_vt_{target_vt:g}_{mode}",
                "pu165_R15000",
                cfg,
                target_vt=target_vt,
                lookahead_mode=mode,
            )
            rows.append(summary)
            manifest["tests"].append(summary)
            write_csv(root / "target_stream_sweep.csv", rows, SUMMARY_FIELDS)
            write_json(root / "target_stream_sananifest.json", manifest)
            print(
                f"target_vt={target_vt:g} lookahead={mode} pu165 term={summary['termination']} "
                f"Gmax={summary['Gmax']:.2f} CTE={summary['CTE_mean']:.1f}",
                flush=True,
            )
    write_csv(root / "target_stream_sweep.csv", rows, SUMMARY_FIELDS)
    write_json(root / "target_stream_sananifest.json", manifest)
    return rows


def analyze_and_write_diagnosis(root, phase_rows, scale_rows, gate_rows, target_rows):
    scale025 = [r for r in scale_rows if str(r.get("scale")) in {"0.25", "0.250000"} and r["task"] == "pu165_R15000"]
    safe_scales = [r for r in scale_rows if r["task"] == "pu165_R15000" and f(r, "Gmax") < 9.0]
    no_crash_scales = [r for r in scale_rows if r["task"] == "pu165_R15000" and r["termination"] == "ok"]
    target_no_crash = [r for r in target_rows if r["termination"] == "ok" and f(r, "Gmax") < 9.0]
    best_scale = select_best_scale(scale_rows)
    text = [
        "# Half-Loop Bridge Diagnosis",
        "",
        "## Summary",
        "",
        f"- selected_low_scale: `{best_scale.get('scale')}`",
        f"- safe_scale_count_Gmax_lt_9: `{len(safe_scales)}`",
        f"- no_crash_scale_count: `{len(no_crash_scales)}`",
        f"- target_stream_no_crash_count: `{len(target_no_crash)}`",
        "",
        "## Diagnostic Questions",
        "",
        "1. Does G spike before geometry improves, during improvement, or after drift starts?",
        "   - In this diagnostic, compare `phasewise/*.csv` around 145-170 deg. The key marker is `phase145_170_Gmax` versus phase geometry errors.",
        "   - If low scale improves CTE/wing while Gmax rises, the spike occurs during the same bridge correction window, not after a completed recovery.",
        "",
        "2. Is over-G caused by elevator command, roll/aileron correction, speed increase, or target curvature?",
        "   - Use `elevator_action`, `aileron_action`, `vt`, and `G` columns in phasewise CSVs. A G spike with elevator saturation indicates pitch/load-factor authority; a spike with large aileron/rudder and roll error indicates roll-plane correction.",
        "",
        "3. Does scale=0.125 keep most geometry benefit while reducing G?",
        "   - See `scale_sweep.csv`; compare rows `scale=0.125` and `scale=0.25` on `pu165_R15000`.",
        "",
        "4. Is crash caused by over-G, alpha departure, path drift, timeout, or target/gate bug?",
        "   - `done_reason=crash_or_overload` indicates real crash/overload. `env_timeout` or `timeout` indicates non-completion rather than overload. Missing/empty rows should be treated as evaluator bugs.",
        "",
        "5. Is there a sharp residual-gate discontinuity near 145-170 deg?",
        "   - Check `gate_jump_max` in `phasewise_diagnostics.csv` and `gate_sweep.csv`.",
        "",
        "6. Does smooth gate margin reduce G/jitter?",
        "   - Compare `phase145_170_Gmax`, `gate_jump_max`, and `jitter_action_diff_mean` across margins in `gate_sweep.csv`.",
        "",
        "7. Is pu165 failing because the target is too aggressive for the current radius/speed, or because the residual response is unstable?",
        "   - If `target_stream_sweep.csv` finds no-crash cases by lowering target_vt or relaxing lookahead, target-stream shaping is required.",
        "",
        "8. Are timeout/missing cases evaluator issues or real non-completion?",
        "   - Empty/missing CSV rows are evaluator/runner issues. `termination=timeout` with many steps and no overload is real non-completion.",
        "",
        "## Current Interpretation",
        "",
    ]
    if target_no_crash:
        text.append("- Target-stream assisted bridge has at least one no-crash case; do not keep trying residual-only first.")
    elif no_crash_scales:
        text.append("- A residual-only inference setting reached no-crash; training may be justified around that scale/gate.")
    else:
        text.append("- No inference-only setting has yet proven pu165 no-crash; residual-only training should not proceed blindly.")
    if scale025:
        r = scale025[0]
        text.append(
            f"- scale=0.25 pu165: termination={r['termination']}, CTE={f(r,'CTE_mean'):.1f}, "
            f"wing={f(r,'wing_plane_error_mean'):.1f}, Gmax={f(r,'Gmax'):.2f}."
        )
    text.append("")
    (root / "diagnosis.md").write_text("\n".join(text), encoding="utf-8")


def write_final_report(root, scale_rows, gate_rows, target_rows):
    no_crash_scale = [r for r in scale_rows if r["task"] == "pu165_R15000" and r["termination"] == "ok" and f(r, "Gmax") < 9.0]
    no_crash_gate = [r for r in gate_rows if r["task"] == "pu165_R15000" and r["termination"] == "ok" and f(r, "Gmax") < 9.0]
    target_no_crash = [r for r in target_rows if r["termination"] == "ok" and f(r, "Gmax") < 9.0]
    promoted = False
    text = [
        "# Half-Loop Bridge Micro-Search Final Report",
        "",
        f"1. Did any candidate beat residual_update_2? `{promoted}`",
        f"2. Did pu165 become no-crash? `{bool(no_crash_scale or no_crash_gate or target_no_crash)}`",
        f"3. Did Gmax stay below 9? `{'yes for some diagnostics' if (no_crash_scale or no_crash_gate or target_no_crash) else 'no promotable no-crash case'}`",
        "4. Did pu150 remain ok/B or better? `checked in scale/gate diagnostics; see CSVs`",
        "5. Did scale/gate sweeps explain the bridge failure? `see diagnosis.md`",
        f"6. Is residual-only enough, or is target-stream assisted bridge needed? `{'target-stream assisted bridge likely needed' if target_no_crash else 'not proven by diagnostics'}`",
        "7. Did any training candidate deserve Claude ACMI regression? `False`",
        f"8. Current best combination: base `{BASE_CKPT}` + residual `{BEST_RESIDUAL}`",
        "9. Exact next phase: `continue pu165-only bridge diagnosis/training; do not reintroduce exit/recovery`",
        "10. Should 170/175 be reintroduced? `No; keep pu165 as the only hard objective until no-crash and Gmax<9`",
        "",
        f"- diagnosis: `{(root / 'diagnosis.md').resolve()}`",
        f"- scale sweep: `{(root / 'scale_sweep.csv').resolve()}`",
        f"- gate sweep: `{(root / 'gate_sweep.csv').resolve()}`",
        f"- target stream: `{(root / 'target_stream_sananifest.json').resolve()}`",
    ]
    (root / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    write_json(
        root / "best_candidate_manifest.json",
        {
            "promoted": False,
            "base_checkpoint": str(BASE_CKPT),
            "residual_checkpoint": str(BEST_RESIDUAL),
            "ready_for_claude_acmi": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--phase-tasks", default="pu150_R12000,pu165_R15000,pu170_R15000")
    parser.add_argument("--sweep-tasks", default="pu150_R12000,pu165_R15000,pu170_R15000")
    parser.add_argument("--skip-gate-sweep", action="store_true")
    parser.add_argument("--skip-target-stream", action="store_true")
    parser.add_argument("--phase-only", action="store_true")
    parser.add_argument("--skip-phase-diagnostics", action="store_true")
    parser.add_argument("--selected-scale", type=float, default=None)
    parser.add_argument("--gate-windows", default="80-180,100-180,120-180,140-175,145-170,150-170")
    parser.add_argument("--gate-margins", default="0,5,10,15,20")
    parser.add_argument("--gate-tasks", default="pu150_R12000,pu165_R15000")
    parser.add_argument("--target-vts", default="220,240,250,260")
    parser.add_argument("--target-lookahead-modes", default="conservative,default,relaxed")
    args = parser.parse_args()

    root = args.out_dir or PLANAX_ROOT / "results/half_loop_bridge_micro_search" / datetime.now().strftime("%Y%m%d_%H%M")
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "config.json",
        {
            "base": str(BASE_CKPT),
            "residual": str(BEST_RESIDUAL),
            "gpu_uuid": GPU_UUID,
            "hard_rule": "No training before diagnosis; no promotion unless pu165 no-crash and Gmax<9.",
        },
    )

    env, net, net_params, residual_net, residual_params, residual_epoch = load_models()
    phase_tasks = [x.strip() for x in args.phase_tasks.split(",") if x.strip()]
    sweep_tasks = [x.strip() for x in args.sweep_tasks.split(",") if x.strip()]
    if args.skip_phase_diagnostics:
        phase_rows = read_csv(root / "phasewise_diagnostics.csv")
        if not phase_rows:
            raise FileNotFoundError(
                f"--skip-phase-diagnostics requested but no cached phasewise_diagnostics.csv under {root}"
            )
        print(f"loaded_cached_phase_rows={len(phase_rows)}", flush=True)
    else:
        phase_rows = run_phase_diagnostics(root, env, net, net_params, residual_net, residual_params, phase_tasks)
    if args.phase_only:
        write_csv(root / "scale_sweep.csv", [], SUMMARY_FIELDS)
        write_csv(root / "gate_sweep.csv", [], SUMMARY_FIELDS)
        write_json(root / "target_stream_sananifest.json", {"tests": [], "skipped": True})
        analyze_and_write_diagnosis(root, phase_rows, [], [], [])
        write_final_report(root, [], [], [])
        return

    scale_rows = run_scale_sweep(root, env, net, net_params, residual_net, residual_params, sweep_tasks, phase_rows)
    selected = select_best_scale(scale_rows)
    selected_scale = float(args.selected_scale if args.selected_scale is not None else (selected.get("scale", 0.125) or 0.125))
    gate_rows = []
    if not args.skip_gate_sweep:
        gate_rows = run_gate_sweep(
            root,
            env,
            net,
            net_params,
            residual_net,
            residual_params,
            selected_scale,
            parse_windows(args.gate_windows),
            parse_float_list(args.gate_margins),
            parse_str_list(args.gate_tasks),
        )
    else:
        gate_rows = read_csv(root / "gate_sweep.csv")
        if not gate_rows:
            write_csv(root / "gate_sweep.csv", [], SUMMARY_FIELDS)
    target_rows = []
    if not args.skip_target_stream:
        target_rows = run_target_stream_sanity(
            root,
            env,
            net,
            net_params,
            residual_net,
            residual_params,
            make_residual_cfg(scale=selected_scale),
            parse_float_list(args.target_vts),
            parse_str_list(args.target_lookahead_modes),
        )
    else:
        target_rows = read_csv(root / "target_stream_sweep.csv")
        if not target_rows:
            write_json(root / "target_stream_sananifest.json", {"tests": [], "skipped": True})
    analyze_and_write_diagnosis(root, phase_rows, scale_rows, gate_rows, target_rows)
    write_final_report(root, scale_rows, gate_rows, target_rows)
    print(f"micro_search_dir={root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
