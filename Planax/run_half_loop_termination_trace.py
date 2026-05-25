import argparse
import json
import os
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

import run_half_loop_bridge_micro_search as bridge
from termination_trace_utils import (
    classify_terminal_reason,
    done_flag_from_info,
    scalar,
    terminal_state_from_info,
)


PLANAX_ROOT = Path(__file__).resolve().parent
TASKS = {
    "pu090_R12000": ("pu090_R12000", 90, 12000, 1000, 400, 1500),
    "pu120_R12000": ("pu120_R12000", 120, 12000, 1000, 400, 1800),
    "pu150_R12000": ("pu150_R12000", 150, 12000, 1200, 500, 2000),
    "pu165_R15000": ("pu165_R15000", 165, 15000, 1300, 500, 2300),
    "pu170_R15000": ("pu170_R15000", 170, 15000, 1400, 500, 2400),
}


def profile_pullup_arc(
    origin_n,
    origin_e,
    origin_alt,
    init_yaw,
    base_radius,
    arc_angle_deg,
    n_points,
    profile,
):
    """Generate a pull-up arc with phase-local radius inflation.

    The path parameter is the tangent angle theta.  For a constant radius R,
    dx/dtheta = R cos(theta) and dz/dtheta = R sin(theta), which matches
    vertical_pullup_arc.  A phase-local radius profile therefore changes the
    executable target curvature without changing the requested terminal angle.
    """
    theta = np.linspace(0.0, np.radians(arc_angle_deg), n_points)
    theta_deg = np.degrees(theta)
    radius = np.full_like(theta, float(base_radius), dtype=np.float64)
    for segment in profile:
        start = float(segment.get("start_deg", 0.0))
        end = float(segment.get("end_deg", arc_angle_deg))
        local_radius = float(segment.get("radius_m", base_radius))
        transition = max(float(segment.get("transition_deg", 0.0)), 1e-6)
        if end <= start:
            continue
        up = np.clip((theta_deg - start) / transition, 0.0, 1.0)
        down = np.clip((end - theta_deg) / transition, 0.0, 1.0)
        weight = np.minimum(up, down)
        radius = radius * (1.0 - weight) + local_radius * weight
    dtheta = np.gradient(theta)
    forward = np.cumsum(radius * np.cos(theta) * dtheta)
    altitude_gain = np.cumsum(radius * np.sin(theta) * dtheta)
    forward -= forward[0]
    altitude_gain -= altitude_gain[0]
    cy, sy = np.cos(init_yaw), np.sin(init_yaw)
    wp_n = origin_n + forward * cy
    wp_e = origin_e + forward * sy
    wp_a = origin_alt + altitude_gain
    waypoints = np.column_stack([wp_n, wp_e, wp_a])
    arc_len = float(np.sum(np.abs(radius * dtheta)))
    meta = {
        "name": f"profile_pullup_{int(arc_angle_deg)}deg_R{int(base_radius)}",
        "n_points": int(n_points),
        "total_length_m": arc_len,
        "radius": float(base_radius),
        "arc_angle_deg": float(arc_angle_deg),
        "start_alt": float(origin_alt),
        "end_alt": float(wp_a[-1]),
        "altitude_gain": float(wp_a[-1] - origin_alt),
        "forward_distance": float(forward[-1]),
        "altitude_range": (float(wp_a.min()), float(wp_a.max())),
        "max_tangent_pitch_deg": float(abs(arc_angle_deg)),
        "average_climb_angle_deg": float(
            np.degrees(np.arctan2(altitude_gain[-1], max(forward[-1], 1e-9)))
        ),
        "max_curvature": float(1.0 / max(radius.min(), 1e-9)),
        "radius_profile": profile,
    }
    return waypoints, meta


def limit_altitude_gain(waypoints, meta, altitude_gain_limit_m):
    limit = float(altitude_gain_limit_m)
    if limit <= 0.0:
        return waypoints, meta
    capped = np.asarray(waypoints, dtype=np.float64).copy()
    start_alt = float(capped[0, 2])
    gain = capped[:, 2] - start_alt
    max_gain = float(np.max(gain))
    if max_gain <= limit:
        limited_meta = dict(meta)
        limited_meta["altitude_gain_limit_m"] = limit
        limited_meta["altitude_gain_limited"] = False
        return capped, limited_meta
    scale = limit / max(max_gain, 1e-9)
    capped[:, 2] = start_alt + gain * scale
    seg = np.diff(capped, axis=0)
    total_length = float(np.sum(np.linalg.norm(seg, axis=1)))
    limited_meta = dict(meta)
    limited_meta.update(
        {
            "total_length_m": total_length,
            "end_alt": float(capped[-1, 2]),
            "altitude_gain": float(capped[-1, 2] - start_alt),
            "altitude_range": (float(capped[:, 2].min()), float(capped[:, 2].max())),
            "altitude_gain_limit_m": limit,
            "altitude_gain_limited": True,
            "altitude_gain_scale": float(scale),
        }
    )
    return capped, limited_meta


def heading_pitch_from_vec(vec):
    arr = np.asarray(vec, dtype=np.float64)
    h_dist = np.sqrt(arr[0] * arr[0] + arr[1] * arr[1]) + 1e-9
    return float(np.arctan2(arr[1], arr[0])), float(np.arctan2(arr[2], h_dist))


def pursuit_tangent_heading_pitch(path_ctx, w_pursuit):
    err = np.asarray(path_ctx["lookahead_error_world"], dtype=np.float64)
    tangent = np.asarray(path_ctx["tangent_world"], dtype=np.float64)
    pursuit_dir = err / (np.linalg.norm(err) + 1e-9)
    tangent_dir = tangent / (np.linalg.norm(tangent) + 1e-9)
    blended = float(w_pursuit) * pursuit_dir + (1.0 - float(w_pursuit)) * tangent_dir
    blended = blended / (np.linalg.norm(blended) + 1e-9)
    return heading_pitch_from_vec(blended)


def phase_scheduled_pursuit_weight(theta_deg):
    if theta_deg < 80.0:
        return 1.0
    if theta_deg < 130.0:
        return float(1.0 + (0.25 - 1.0) * ((theta_deg - 80.0) / 50.0))
    return 0.10


def apply_target_stream_mode(path_ctx, theta_deg, target_heading, target_pitch, variant):
    mode = variant.get("target_stream_mode", "pure_pursuit")
    if mode == "pure_pursuit":
        return target_heading, target_pitch
    if mode == "tangent_following":
        return heading_pitch_from_vec(path_ctx["tangent_world"])
    if mode == "pursuit_tangent_blend":
        return pursuit_tangent_heading_pitch(path_ctx, variant.get("w_pursuit", 0.5))
    if mode == "phase_scheduled_blend":
        if theta_deg < 80.0:
            return target_heading, target_pitch
        return pursuit_tangent_heading_pitch(path_ctx, phase_scheduled_pursuit_weight(theta_deg))
    if mode == "curvature_aware":
        if theta_deg < 80.0:
            return target_heading, target_pitch
        weight = phase_scheduled_pursuit_weight(theta_deg)
        if theta_deg >= 100.0:
            weight = min(weight, 0.20)
        if theta_deg >= 130.0:
            weight = min(weight, 0.05)
        return pursuit_tangent_heading_pitch(path_ctx, weight)
    raise ValueError(f"unknown target_stream_mode: {mode}")


def smoothstep01(x):
    y = float(np.clip(x, 0.0, 1.0))
    return y * y * (3.0 - 2.0 * y)


def phase_gate(theta_deg, band, margin=5.0):
    start, end = float(band[0]), float(band[1])
    if end <= start:
        return 0.0
    margin = max(float(margin), 1e-6)
    if theta_deg < start or theta_deg > end:
        return 0.0
    return smoothstep01((theta_deg - start) / margin) * smoothstep01((end - theta_deg) / margin)


def local_correction_gate(theta_deg, variant):
    band = variant.get("local_correction_band")
    if not band:
        return 0.0
    return phase_gate(theta_deg, band, variant.get("local_correction_margin_deg", 5.0))

SUMMARY_FIELDS = bridge.SUMMARY_FIELDS + [
    "terminal_step",
    "terminal_phase_deg",
    "terminal_reason_raw_flags",
]

TERMINAL_FIELDS = [
    "policy",
    "task",
    "scale",
    "completed",
    "terminal_step",
    "terminal_phase_deg",
    "terminal_reason_classified",
    "terminal_done_flags",
    "terminal_info_keys",
    "terminal_alpha",
    "terminal_beta",
    "terminal_G",
    "terminal_vt",
    "terminal_altitude",
    "terminal_pitch",
    "terminal_roll",
    "terminal_target_pitch",
    "terminal_target_roll",
    "terminal_q_error",
    "terminal_CTE",
    "terminal_velocity_tangent_error",
    "terminal_nose_tangent_error",
    "terminal_nose_velocity_error",
    "terminal_wing_plane_error",
    "terminal_residual_gate",
    "terminal_residual_norm",
    "terminal_action_diff_norm",
    "terminal_action",
    "terminal_target",
    "terminal_reason_raw",
]


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    try:
        arr = np.asarray(obj)
        if arr.ndim == 0:
            value = arr.item()
            if isinstance(value, (np.bool_, bool)):
                return bool(value)
            if isinstance(value, (np.integer, int)):
                return int(value)
            if isinstance(value, (np.floating, float)):
                return float(value)
            return str(value)
        if arr.dtype == object:
            return str(obj)
        if arr.size <= 16:
            return arr.reshape(-1).tolist()
        return {"shape": list(arr.shape), "sample": arr.reshape(-1)[:16].tolist()}
    except Exception:
        return str(obj)


def geometry_for_state(state, wps, wp_idx, target_heading, target_pitch, target_roll):
    ps = state.plane_state
    q_bn = np.array(
        [scalar(ps.q0), scalar(ps.q1), scalar(ps.q2), scalar(ps.q3)],
        dtype=np.float64,
    )
    q_bn = q_bn / (np.linalg.norm(q_bn) + 1e-12)
    x_body_neu = bridge.ev.ned_to_neu(
        bridge.ev.rotate_body_to_ned(q_bn, np.array([1.0, 0.0, 0.0]))
    )
    y_body_neu = bridge.ev.ned_to_neu(
        bridge.ev.rotate_body_to_ned(q_bn, np.array([0.0, 1.0, 0.0]))
    )
    alpha_rad = scalar(ps.alpha)
    beta_rad = scalar(ps.beta)
    ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
    cb, sb = np.cos(beta_rad), np.sin(beta_rad)
    vt = scalar(ps.vt)
    v_body = np.array([vt * ca * cb, vt * sb, vt * sa * cb], dtype=np.float64)
    v_neu = bridge.ev.ned_to_neu(bridge.ev.rotate_body_to_ned(q_bn, v_body))
    v_hat_neu = v_neu / (np.linalg.norm(v_neu) + 1e-12)
    t_ref_neu, n_loop_neu = bridge.ev.compute_loop_reference(wps, wp_idx)
    return {
        "velocity_tangent_error": bridge.ev.angle_between(v_hat_neu, t_ref_neu),
        "nose_tangent_error": bridge.ev.angle_between(x_body_neu, t_ref_neu),
        "nose_velocity_error": bridge.ev.angle_between(x_body_neu, v_hat_neu),
        "wing_plane_error": bridge.ev.angle_between(y_body_neu, n_loop_neu),
        "q_error_norm": bridge.ev.quat_error_angle(
            q_bn,
            float(target_heading),
            float(target_pitch),
            float(target_roll),
        ),
    }


def state_scalars(state):
    ps = state.plane_state
    ax = scalar(ps.ax)
    ay = scalar(ps.ay)
    az = scalar(ps.az)
    return {
        "alpha": float(np.degrees(scalar(ps.alpha))),
        "beta": float(np.degrees(scalar(ps.beta))),
        "G": float(np.sqrt(ax * ax + ay * ay + az * az)),
        "vt": scalar(ps.vt),
        "altitude": scalar(ps.altitude),
        "pitch": float(np.degrees(scalar(ps.pitch))),
        "roll": float(np.degrees(scalar(ps.roll))),
        "north": scalar(ps.north),
        "east": scalar(ps.east),
    }


def run_trace_test(
    env,
    net,
    net_params,
    residual_net,
    residual_params,
    policy_name,
    task,
    residual_cfg=None,
    variant=None,
):
    variant = variant or {}
    name, angle_deg, radius_m, lookahead, reach_radius, max_steps = TASKS[task]
    max_steps = int(variant.get("max_steps", max_steps))
    eval_radius = float(variant.get("eval_radius_m", radius_m))
    eval_wps, eval_meta = bridge.ev.vertical_pullup_arc(
        0,
        0,
        5000,
        0.0,
        radius=eval_radius,
        arc_angle_deg=angle_deg,
        n_points=max(80, int(angle_deg * 2 / 3)),
    )
    if "eval_altitude_gain_limit_m" in variant:
        eval_wps, eval_meta = limit_altitude_gain(
            eval_wps,
            eval_meta,
            variant["eval_altitude_gain_limit_m"],
        )
    target_radius = float(variant.get("target_radius_m", radius_m))
    if "target_radius_profile" in variant:
        target_wps, target_meta = profile_pullup_arc(
            0,
            0,
            5000,
            0.0,
            target_radius,
            angle_deg,
            max(80, int(angle_deg * 2 / 3)),
            variant["target_radius_profile"],
        )
    else:
        target_wps, target_meta = bridge.ev.vertical_pullup_arc(
            0,
            0,
            5000,
            0.0,
            radius=target_radius,
            arc_angle_deg=angle_deg,
            n_points=max(80, int(angle_deg * 2 / 3)),
        )
    if "target_altitude_gain_limit_m" in variant:
        target_wps, target_meta = limit_altitude_gain(
            target_wps,
            target_meta,
            variant["target_altitude_gain_limit_m"],
        )
    target_total_arc = target_meta["total_length_m"]
    initial_vt = float(variant.get("entry_vt", 250.0))
    planner = bridge.ev.PurePursuitPlanner(
        bridge.ev.PlannerConfig(
            lookahead_dist=float(variant.get("lookahead_dist", lookahead)),
            reach_radius=reach_radius,
            blend_steps=250,
            target_vt=float(variant.get("target_vt", initial_vt)),
        )
    )
    rng = jax.random.PRNGKey(bridge.ev.SEED)
    rng, reset_key = jax.random.split(rng)
    _, state = env.reset(reset_key, bridge.ev.Params())
    q_nb_init = bridge.ev._quat_from_euler_nb(0.0, 0.0, 0.0)
    q_bn_init = bridge.ev._quat_conj(q_nb_init)
    state = state.replace(
        plane_state=state.plane_state.replace(
            yaw=jnp.array([0.0]),
            vt=jnp.array([initial_vt], dtype=jnp.float32),
            vel_y=jnp.array([initial_vt], dtype=jnp.float32),
            q0=jnp.array([q_bn_init[0]]),
            q1=jnp.array([q_bn_init[1]]),
            q2=jnp.array([q_bn_init[2]]),
            q3=jnp.array([q_bn_init[3]]),
        ),
        target_heading=jnp.array([0.0]),
        target_vt=jnp.array([float(variant.get("target_vt", initial_vt))], dtype=jnp.float32),
    )
    planner.reset(target_wps, 0.0, 0.0, 0.0, float(variant.get("target_vt", initial_vt)))

    hstate = bridge.ev.ScannedRNN.initialize_carry(1, bridge.ev.NET_CFG["GRU_HIDDEN_DIM"])
    residual_hstate = None
    if residual_cfg is not None:
        residual_hstate = bridge.ResidualScannedRNN.initialize_carry(
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
    terminal = None
    last_context = None
    prev_target_pitch = 0.0
    prev_target_roll = 0.0
    for step in range(max_steps):
        ps = state.plane_state
        north = bridge.ev.f_scalar(ps.north)
        east = bridge.ev.f_scalar(ps.east)
        alt = bridge.ev.f_scalar(ps.altitude)
        vt = bridge.ev.f_scalar(ps.vt)
        roll = bridge.ev.f_scalar(ps.roll)
        pitch = bridge.ev.f_scalar(ps.pitch)
        yaw = bridge.ev.f_scalar(ps.yaw)
        alpha = bridge.ev.f_scalar(ps.alpha)
        beta = bridge.ev.f_scalar(ps.beta)
        ax = bridge.ev.f_scalar(ps.ax)
        ay = bridge.ev.f_scalar(ps.ay)
        az = bridge.ev.f_scalar(ps.az)
        phase_est = (planner.path_progress / target_total_arc) * angle_deg if target_total_arc > 0 else 0.0
        if "local_lookahead_scale" in variant and variant.get("local_correction_band"):
            g_local = local_correction_gate(phase_est, variant)
            scale = 1.0 + (float(variant["local_lookahead_scale"]) - 1.0) * g_local
            planner.path.lookahead_dist = float(lookahead) * scale
        if "bridge_lookahead_dist" in variant and 80.0 <= phase_est <= 170.0:
            planner.path.lookahead_dist = float(variant["bridge_lookahead_dist"])
        elif "lookahead_dist" in variant:
            planner.path.lookahead_dist = float(variant["lookahead_dist"])
        result = planner.step(north, east, alt, yaw, pitch, roll, vt)
        target_heading = result["target_heading"]
        target_pitch = result["target_pitch"]
        target_roll_pp = result["target_roll"]
        planner_target_vt = result["target_vt"]
        path_s = planner.path_progress
        theta_deg = (path_s / target_total_arc) * angle_deg if target_total_arc > 0 else 0.0
        theta_deg = float(np.clip(theta_deg, 0.0, angle_deg))
        target_heading, target_pitch = apply_target_stream_mode(
            result["path_ctx"],
            theta_deg,
            target_heading,
            target_pitch,
            variant,
        )
        if "bridge_target_vt" in variant and 80.0 <= theta_deg <= 170.0:
            planner_target_vt = float(variant["bridge_target_vt"])
        g_local = local_correction_gate(theta_deg, variant)
        if "local_target_vt_delta" in variant:
            planner_target_vt = float(planner_target_vt) + float(variant["local_target_vt_delta"]) * g_local
        if "local_pitch_bias_deg" in variant:
            target_pitch = float(target_pitch) + np.radians(float(variant["local_pitch_bias_deg"])) * g_local
        if "target_pitch_clip_deg" in variant and 80.0 <= theta_deg <= 170.0:
            lim = np.radians(float(variant["target_pitch_clip_deg"]))
            target_pitch = float(np.clip(target_pitch, -lim, lim))
        if "pitch_blend_with_current" in variant and 100.0 <= theta_deg <= 170.0:
            w = float(variant["pitch_blend_with_current"])
            target_pitch = float((1.0 - w) * target_pitch + w * pitch)
        if "pitch_rate_limit_deg_s" in variant:
            max_delta = np.radians(float(variant["pitch_rate_limit_deg_s"])) * 0.2
            target_pitch = float(np.clip(target_pitch, prev_target_pitch - max_delta, prev_target_pitch + max_delta))
        target_loop_roll = bridge.ev.loop_roll(theta_deg)
        blend = min(1.0, step / 250.0)
        target_roll = float(
            np.arctan2(
                np.sin(roll + blend * (target_loop_roll - roll)),
                np.cos(roll + blend * (target_loop_roll - roll)),
            )
        )
        if "roll_rate_limit_deg_s" in variant:
            max_delta = np.radians(float(variant["roll_rate_limit_deg_s"])) * 0.2
            delta = np.arctan2(np.sin(target_roll - prev_target_roll), np.cos(target_roll - prev_target_roll))
            target_roll = float(prev_target_roll + np.clip(delta, -max_delta, max_delta))
        prev_target_pitch = target_pitch
        prev_target_roll = target_roll
        state = state.replace(
            target_heading=jnp.array([target_heading]),
            target_pitch=jnp.array([target_pitch]),
            target_roll=jnp.array([target_roll]),
            target_vt=jnp.array([float(planner_target_vt)], dtype=jnp.float32),
        )
        obs = env._get_obs(state, bridge.ev.Params())[env.agents[0]][None, None, :]
        hstate, base_pi, _ = net.apply(net_params, hstate, (obs, done_flag[None, :]))
        base_actions = [int(p.mode()[0, 0]) for p in base_pi]
        base_cont = bridge.continuous_action(base_actions)
        base_norm = bridge.logits_norm(base_pi)
        residual_norm = 0.0
        final_delta_norm = 0.0
        gate = 0.0
        if residual_cfg is not None:
            gate = bridge.ev.residual_gate_value(theta_deg, residual_cfg)
            obs_aug = bridge.augment_obs_with_phase(
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
            pi_out, clipped_delta, _ = bridge.combine_base_and_residual_logits(
                base_pi, residual_logits, obs_aug, residual_cfg
            )
            residual_norm = bridge.array_tuple_norm(clipped_delta)
            final_delta_norm = bridge.array_tuple_norm(
                [p.logits - b.logits for p, b in zip(pi_out, base_pi)]
            )
        else:
            pi_out = base_pi
        actions = [int(p.mode()[0, 0]) for p in pi_out]
        cont = bridge.continuous_action(actions)
        action_diff = float(np.linalg.norm(cont - base_cont))
        wp_idx = result["path_ctx"]["wp_idx"]
        cte = bridge.ev.compute_true_cte(np.array([north, east, alt]), eval_wps, wp_idx, 10)
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
        rec["cte"].append(cte)
        rec["q0"].append(bridge.ev.f_scalar(ps.q0))
        rec["q1"].append(bridge.ev.f_scalar(ps.q1))
        rec["q2"].append(bridge.ev.f_scalar(ps.q2))
        rec["q3"].append(bridge.ev.f_scalar(ps.q3))
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
        last_context = {
            "step": step,
            "theta_deg": theta_deg,
            "wp_idx": wp_idx,
            "target_heading": target_heading,
            "target_pitch": target_pitch,
            "target_roll": target_roll,
            "action": actions,
            "cont_action": cont.tolist(),
            "gate": gate,
            "residual_norm": residual_norm,
            "action_diff": action_diff,
        }
        rng, step_key = jax.random.split(rng)
        _, next_state, _, done, info = env.step(
            step_key, state, {env.agents[0]: jnp.array(actions)}, bridge.ev.Params()
        )
        done_bool = bool(done[env.agents[0]])
        done_flag = jnp.array([float(done_bool)])
        if done_bool:
            terminal_state = terminal_state_from_info(info, next_state)
            trace = classify_terminal_reason(
                terminal_state,
                params=bridge.ev.Params(),
                done_flag=True,
                planner_completed=planner.is_done(),
                agent_id=0,
            )
            terminal = {
                "state": terminal_state,
                "info": info,
                "trace": trace,
                "context": last_context,
                "done_flag": True,
            }
            state = next_state
            break
        state = next_state
        if planner.is_done():
            trace = classify_terminal_reason(
                state,
                params=bridge.ev.Params(),
                done_flag=False,
                planner_completed=True,
                agent_id=0,
            )
            terminal = {
                "state": state,
                "info": info,
                "trace": trace,
                "context": last_context,
                "done_flag": False,
            }
            break

    if terminal is None:
        trace = classify_terminal_reason(
            state,
            params=bridge.ev.Params(),
            done_flag=False,
            planner_completed=False,
            agent_id=0,
        )
        if trace["terminal_reason_classified"] == "running":
            trace["terminal_reason_classified"] = "timeout"
        terminal = {
            "state": state,
            "info": {},
            "trace": trace,
            "context": last_context,
            "done_flag": False,
        }

    n = len(rec["t"])
    completed = planner.is_done() and terminal["trace"]["terminal_reason_classified"] == "success"
    geo = {
        "velocity_tangent_error": [],
        "nose_tangent_error": [],
        "nose_velocity_error": [],
        "wing_plane_error": [],
        "q_error_rad": [],
    }
    for i in range(n):
        q_bn = np.array([rec["q0"][i], rec["q1"][i], rec["q2"][i], rec["q3"][i]], dtype=np.float64)
        q_bn = q_bn / (np.linalg.norm(q_bn) + 1e-12)
        x_body_neu = bridge.ev.ned_to_neu(bridge.ev.rotate_body_to_ned(q_bn, np.array([1.0, 0.0, 0.0])))
        y_body_neu = bridge.ev.ned_to_neu(bridge.ev.rotate_body_to_ned(q_bn, np.array([0.0, 1.0, 0.0])))
        alpha_rad = np.radians(rec["alpha"][i])
        beta_rad = np.radians(rec["beta"][i])
        ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
        cb, sb = np.cos(beta_rad), np.sin(beta_rad)
        v_body = np.array([rec["vt"][i] * ca * cb, rec["vt"][i] * sb, rec["vt"][i] * sa * cb])
        v_neu = bridge.ev.ned_to_neu(bridge.ev.rotate_body_to_ned(q_bn, v_body))
        v_hat_neu = v_neu / (np.linalg.norm(v_neu) + 1e-12)
        t_ref_neu, n_loop_neu = bridge.ev.compute_loop_reference(eval_wps, rec["wp_idx"][i])
        geo["velocity_tangent_error"].append(bridge.ev.angle_between(v_hat_neu, t_ref_neu))
        geo["nose_tangent_error"].append(bridge.ev.angle_between(x_body_neu, t_ref_neu))
        geo["nose_velocity_error"].append(bridge.ev.angle_between(x_body_neu, v_hat_neu))
        geo["wing_plane_error"].append(bridge.ev.angle_between(y_body_neu, n_loop_neu))
        geo["q_error_rad"].append(
            bridge.ev.quat_error_angle(
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

    cte_arr = arr("cte")
    theta = arr("theta_deg")
    phase_mask = (theta >= 145.0) & (theta <= 170.0)
    if not np.any(phase_mask):
        phase_mask = np.ones_like(theta, dtype=bool)
    gate_arr = arr("gate")
    action_diff = arr("action_diff_norm")
    terminal_state = terminal["state"]
    terminal_ctx = terminal["context"] or {}
    terminal_geo = geometry_for_state(
        terminal_state,
        eval_wps,
        int(terminal_ctx.get("wp_idx", rec["wp_idx"][-1])),
        terminal_ctx.get("target_heading", np.radians(rec["t_hdg"][-1])),
        terminal_ctx.get("target_pitch", np.radians(rec["t_pitch"][-1])),
        terminal_ctx.get("target_roll", np.radians(rec["t_roll"][-1])),
    )
    terminal_scalars = state_scalars(terminal_state)
    effective_gmax = max(float(arr("G").max()), float(terminal_scalars["G"]))
    effective_vt_min = min(float(arr("vt").min()), float(terminal_scalars["vt"]))
    terminal_pos = np.array(
        [terminal_scalars["north"], terminal_scalars["east"], terminal_scalars["altitude"]],
        dtype=np.float64,
    )
    terminal_cte = bridge.ev.compute_true_cte(
        terminal_pos,
        eval_wps,
        int(terminal_ctx.get("wp_idx", rec["wp_idx"][-1])),
        10,
    )
    reason = terminal["trace"]["terminal_reason_classified"]
    summary = {
        "policy": policy_name,
        "task": task,
        "scale": "" if residual_cfg is None else residual_cfg.get("RESIDUAL_SCALE", ""),
        "gate_start": "" if residual_cfg is None else residual_cfg.get("RESIDUAL_GATE_START_DEG", ""),
        "gate_end": "" if residual_cfg is None else residual_cfg.get("RESIDUAL_GATE_END_DEG", ""),
        "smooth_margin": "" if residual_cfg is None else residual_cfg.get("RESIDUAL_SMOOTH_GATE_MARGIN_DEG", ""),
        "target_vt": float(variant.get("target_vt", initial_vt)),
        "lookahead_mode": "default",
        "completed": bool(completed),
        "steps": n,
        "termination": "ok" if completed else reason,
        "done_reason": reason,
        "terminal_reason_classified": reason,
        "terminal_reason_raw": terminal["trace"]["terminal_reason_raw"],
        "terminal_reason_raw_flags": terminal["trace"]["terminal_reason_raw"],
        "terminal_step": terminal_ctx.get("step", n - 1),
        "terminal_phase_deg": terminal_ctx.get("theta_deg", theta[-1]),
        "CTE_mean": float(cte_arr.mean()),
        "velocity_tangent_error_mean": float(garr("velocity_tangent_error").mean()),
        "nose_tangent_error_mean": float(garr("nose_tangent_error").mean()),
        "nose_velocity_error_mean": float(garr("nose_velocity_error").mean()),
        "wing_plane_error_mean": float(garr("wing_plane_error").mean()),
        "q_error_mean_rad": float(garr("q_error_rad").mean()),
        "env_alpha_max": float(arr("alpha").max()),
        "env_beta_max": float(arr("beta").max()),
        "vt_min": effective_vt_min,
        "vt_max": float(arr("vt").max()),
        "Gmax": effective_gmax,
        "alt_min": float(arr("a").min()),
        "alt_max": float(arr("a").max()),
        "phase145_170_CTE_mean": float(cte_arr[phase_mask].mean()),
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
    terminal_row = {
        "policy": policy_name,
        "task": task,
        "scale": summary["scale"],
        "completed": bool(completed),
        "terminal_step": summary["terminal_step"],
        "terminal_phase_deg": summary["terminal_phase_deg"],
        "terminal_reason_classified": reason,
        "terminal_done_flags": json.dumps(jsonable(terminal.get("info", {}).get("terminal_dones_before_reset", {})), sort_keys=True),
        "terminal_info_keys": ",".join(sorted(str(k) for k in terminal.get("info", {}).keys())),
        "terminal_alpha": terminal_scalars["alpha"],
        "terminal_beta": terminal_scalars["beta"],
        "terminal_G": terminal_scalars["G"],
        "terminal_vt": terminal_scalars["vt"],
        "terminal_altitude": terminal_scalars["altitude"],
        "terminal_pitch": terminal_scalars["pitch"],
        "terminal_roll": terminal_scalars["roll"],
        "terminal_target_pitch": float(np.degrees(terminal_ctx.get("target_pitch", 0.0))),
        "terminal_target_roll": float(np.degrees(terminal_ctx.get("target_roll", 0.0))),
        "terminal_q_error": terminal_geo["q_error_norm"],
        "terminal_CTE": terminal_cte,
        "terminal_velocity_tangent_error": terminal_geo["velocity_tangent_error"],
        "terminal_nose_tangent_error": terminal_geo["nose_tangent_error"],
        "terminal_nose_velocity_error": terminal_geo["nose_velocity_error"],
        "terminal_wing_plane_error": terminal_geo["wing_plane_error"],
        "terminal_residual_gate": terminal_ctx.get("gate", 0.0),
        "terminal_residual_norm": terminal_ctx.get("residual_norm", 0.0),
        "terminal_action_diff_norm": terminal_ctx.get("action_diff", 0.0),
        "terminal_action": json.dumps(terminal_ctx.get("action", [])),
        "terminal_target": json.dumps(
            {
                "heading_deg": float(np.degrees(terminal_ctx.get("target_heading", 0.0))),
                "pitch_deg": float(np.degrees(terminal_ctx.get("target_pitch", 0.0))),
                "roll_deg": float(np.degrees(terminal_ctx.get("target_roll", 0.0))),
            },
            sort_keys=True,
        ),
        "terminal_reason_raw": terminal["trace"]["terminal_reason_raw"],
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
                "yaw": rec["yaw"][i],
                "north": rec["n"][i],
                "east": rec["e"][i],
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
    raw_info = {
        "policy": policy_name,
        "task": task,
        "summary": summary,
        "terminal": terminal_row,
        "terminal_info": jsonable(terminal.get("info", {})),
    }
    return summary, terminal_row, phase_rows, raw_info


def write_report(root, rows, terminal_rows):
    by_key = {(row["policy"], row["task"]): row for row in rows}
    pu165_scale02 = by_key.get(("update2_scale0.2", "pu165_R15000"))
    true_reason = pu165_scale02["terminal_reason_classified"] if pu165_scale02 else "missing"
    terminal_phase = pu165_scale02["terminal_phase_deg"] if pu165_scale02 else "missing"
    completed = pu165_scale02["completed"] if pu165_scale02 else "missing"
    gmax = pu165_scale02["Gmax"] if pu165_scale02 else "missing"
    cte = pu165_scale02["CTE_mean"] if pu165_scale02 else "missing"
    text = [
        "# Half-Loop Bridge Termination Trace Report",
        "",
        "## Decision",
        "",
        "- Termination tracing is fixed for these diagnostics by reading `info['terminal_state_before_reset']` emitted before env auto-reset.",
        "- No residual training was run.",
        "",
        "## Required Answers",
        "",
        f"1. True termination reason for `pu165_R15000` at `scale=0.20`: `{true_reason}`.",
        f"2. It completed: `{completed}`; Gmax: `{gmax}`; CTE_mean: `{cte}`.",
        f"3. Terminal phase for `scale=0.20 / pu165`: `{terminal_phase}` deg.",
        "4. The terminal state is recorded in `terminal_states.csv` and `raw_terminal_info/`.",
        "5. Previous `done_unknown` rows are now classified where env flags expose the cause.",
        "",
        "## Summary Rows",
        "",
        "| policy | task | completed | termination | phase | CTE | Gmax | wing |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {row['policy']} | {row['task']} | {row['completed']} | "
            f"{row['terminal_reason_classified']} | {float(row['terminal_phase_deg']):.1f} | "
            f"{float(row['CTE_mean']):.1f} | {float(row['Gmax']):.2f} | "
            f"{float(row['wing_plane_error_mean']):.1f} |"
        )
    text.append("")
    root.joinpath("termination_trace_report.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--tasks", default="pu150_R12000,pu165_R15000,pu170_R15000")
    args = parser.parse_args()
    root = args.out_dir or PLANAX_ROOT / "results/half_loop_bridge_termination_trace" / datetime.now().strftime("%Y%m%d_%H%M")
    root.mkdir(parents=True, exist_ok=True)
    (root / "phasewise").mkdir(exist_ok=True)
    (root / "raw_terminal_info").mkdir(exist_ok=True)
    env, net, net_params, residual_net, residual_params, _ = bridge.load_models()
    policies = [
        ("base_only", None),
        ("update2_scale1.0", bridge.make_residual_cfg(scale=1.0)),
        ("update2_scale0.2", bridge.make_residual_cfg(scale=0.20)),
        ("update2_scale0.25", bridge.make_residual_cfg(scale=0.25)),
    ]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    rows = []
    terminal_rows = []
    for policy_name, cfg in policies:
        for task in tasks:
            summary, terminal_row, phase_rows, raw_info = run_trace_test(
                env,
                net,
                net_params,
                residual_net,
                residual_params,
                policy_name,
                task,
                cfg,
            )
            rows.append(summary)
            terminal_rows.append(terminal_row)
            bridge.write_csv(root / "scale_sweep_fixed.csv", rows, SUMMARY_FIELDS)
            bridge.write_csv(root / "terminal_states.csv", terminal_rows, TERMINAL_FIELDS)
            bridge.write_csv(root / "phasewise" / f"{policy_name}_{task}.csv", phase_rows, bridge.PHASE_FIELDS)
            bridge.write_json(root / "raw_terminal_info" / f"{policy_name}_{task}.json", raw_info)
            print(
                f"{policy_name} {task} term={summary['terminal_reason_classified']} "
                f"completed={summary['completed']} phase={float(summary['terminal_phase_deg']):.1f} "
                f"CTE={float(summary['CTE_mean']):.1f} Gmax={float(summary['Gmax']):.2f}",
                flush=True,
            )
    write_report(root, rows, terminal_rows)
    bridge.write_json(
        root / "manifest.json",
        {
            "base": str(bridge.BASE_CKPT),
            "residual": str(bridge.BEST_RESIDUAL),
            "training_ran": False,
            "rows": len(rows),
        },
    )
    print(f"termination_trace_dir={root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
