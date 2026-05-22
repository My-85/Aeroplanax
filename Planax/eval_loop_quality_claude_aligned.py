"""
Claude-aligned loop-quality evaluator.

This evaluator intentionally follows the planner-level loop-quality script that
produced `results/loop_quality_evaluation/20260517_010055`.  It is separate
from `eval_vertical_energy_checkpoints.py` because promotion for inverted/top
transition work must use planner-level loop geometry, not target-level CTE-only
grading.
"""
import argparse
import csv
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
    ActorCriticRNN,
    NET_CFG,
    SEED,
    ScannedRNN,
)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import (
    vertical_pullup_arc,
)
from experiments.hierarchical_trajectory_tracking.planner import (
    PlannerConfig,
    PurePursuitPlanner,
)
from experiments.hierarchical_trajectory_tracking.path_utils import compute_true_cte
from experiments.hierarchical_trajectory_tracking.loop_attitude_target import (
    loop_plane_rotation_matrix,
    quaternion_to_euler,
    rotation_matrix_to_quaternion,
)
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as Env,
    Heading_Pitch_V_TaskParams as Params,
    _quat_conj,
    _quat_from_euler_nb,
)
from half_loop_residual_policy import (
    ResidualActorCriticRNN,
    ResidualScannedRNN,
    augment_obs_with_phase,
    combine_base_and_residual_logits,
    smooth01,
)


PLANAX_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = (
    PLANAX_ROOT
    / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
)
DEFAULT_CLAUDE_SUMMARY = (
    PLANAX_ROOT
    / "results/loop_quality_evaluation/20260517_010055/loop_quality_summary.csv"
)

CLAUDE_KEY_METRICS = [
    "CTE_mean",
    "CTE_p90",
    "CTE_max",
    "velocity_tangent_error_mean",
    "velocity_tangent_error_p90",
    "nose_tangent_error_mean",
    "nose_tangent_error_p90",
    "nose_velocity_error_mean",
    "nose_velocity_error_p90",
    "wing_plane_error_mean",
    "wing_plane_error_p90",
    "q_error_mean_rad",
    "q_error_p90_rad",
    "roll_tracking_error_mean",
    "env_alpha_min",
    "env_alpha_max",
    "vt_min",
    "vt_mean",
    "Gmax",
    "Gmean",
    "alt_min",
    "alt_max",
    "phase150_180_velocity_tangent_error_mean",
    "phase150_180_nose_tangent_error_mean",
    "phase150_180_wing_plane_error_mean",
    "phase170_200_velocity_tangent_error_mean",
    "phase170_200_nose_tangent_error_mean",
    "phase170_200_wing_plane_error_mean",
    "phase180_200_velocity_tangent_error_mean",
    "phase180_200_nose_tangent_error_mean",
    "phase180_200_wing_plane_error_mean",
]

FIELDNAMES = [
    "name",
    "angle_deg",
    "radius_m",
    "completed",
    "steps",
    "termination",
    "grade_cte_only_deprecated",
    "grade_loop_quality",
    "CTE_mean",
    "CTE_p50",
    "CTE_p90",
    "CTE_max",
    "velocity_tangent_error_mean",
    "velocity_tangent_error_p90",
    "nose_tangent_error_mean",
    "nose_tangent_error_p90",
    "nose_velocity_error_mean",
    "nose_velocity_error_p90",
    "wing_plane_error_mean",
    "wing_plane_error_p90",
    "belly_error_mean",
    "q_error_mean_rad",
    "q_error_p90_rad",
    "roll_tracking_error_mean",
    "env_alpha_min",
    "env_alpha_max",
    "env_alpha_mean",
    "env_beta_min",
    "env_beta_max",
    "target_roll_min",
    "target_roll_max",
    "actual_roll_min",
    "actual_roll_max",
    "actual_roll_mean",
    "vt_min",
    "vt_mean",
    "vt_max",
    "Gmax",
    "Gmean",
    "alt_min",
    "alt_max",
    "phase150_180_velocity_tangent_error_mean",
    "phase150_180_nose_tangent_error_mean",
    "phase150_180_wing_plane_error_mean",
    "phase170_200_velocity_tangent_error_mean",
    "phase170_200_nose_tangent_error_mean",
    "phase170_200_wing_plane_error_mean",
    "phase180_200_velocity_tangent_error_mean",
    "phase180_200_nose_tangent_error_mean",
    "phase180_200_wing_plane_error_mean",
]


def f_scalar(x):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


def quat_conj_np(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def rotate_body_to_ned(q_bn, v_body):
    q_nb = quat_conj_np(q_bn)
    p = np.array([0.0, v_body[0], v_body[1], v_body[2]], dtype=np.float64)
    qpq = quat_mul_np(quat_mul_np(q_nb, p), quat_conj_np(q_nb))
    return qpq[1:]


def ned_to_neu(v_ned):
    return np.array([v_ned[0], v_ned[1], -v_ned[2]], dtype=np.float64)


def angle_between(v1, v2):
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    dot = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
    return float(np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0))))


def compute_loop_reference(wps, idx, look_ahead=3):
    n = len(wps)
    i0 = max(0, idx - look_ahead)
    i1 = min(n - 1, idx + look_ahead)
    if i1 > i0:
        tangent = wps[i1] - wps[i0]
    else:
        tangent = wps[min(idx + 1, n - 1)] - wps[max(idx - 1, 0)]
    tangent = tangent / (np.linalg.norm(tangent) + 1e-12)

    if n >= 3:
        nb = wps[max(0, idx - 5) : min(n, idx + 5)]
        if len(nb) >= 3:
            centroid = nb.mean(axis=0)
            _, _, vh = np.linalg.svd(nb - centroid)
            normal = vh[2]
            if normal[1] < 0:
                normal = -normal
        else:
            normal = np.array([0.0, 1.0, 0.0])
    else:
        normal = np.array([0.0, 1.0, 0.0])
    return tangent, normal


def quat_error_angle(q_curr_bn, yaw_t, pitch_t, roll_t):
    cr, sr = np.cos(0.5 * roll_t), np.sin(0.5 * roll_t)
    cp, sp = np.cos(0.5 * pitch_t), np.sin(0.5 * pitch_t)
    cy, sy = np.cos(0.5 * yaw_t), np.sin(0.5 * yaw_t)
    q_tgt_nb = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    q_tgt_bn = quat_conj_np(q_tgt_nb)
    q_tgt_bn = q_tgt_bn / (np.linalg.norm(q_tgt_bn) + 1e-12)
    q_curr_bn = q_curr_bn / (np.linalg.norm(q_curr_bn) + 1e-12)
    q_err = quat_mul_np(q_tgt_bn, quat_conj_np(q_curr_bn))
    if q_err[0] < 0:
        q_err = -q_err
    w = np.clip(abs(q_err[0]), 0.0, 1.0 - 1e-12)
    return float(2.0 * np.arccos(w))


def loop_roll(theta_deg):
    rot = loop_plane_rotation_matrix(np.radians(theta_deg), 0.0, 1)
    q = rotation_matrix_to_quaternion(rot)
    roll, _, _ = quaternion_to_euler(q)
    return roll


def residual_gate_value(theta_deg, residual_cfg):
    start = float(residual_cfg.get("RESIDUAL_GATE_START_DEG", 80.0))
    end = float(residual_cfg.get("RESIDUAL_GATE_END_DEG", 180.0))
    margin = float(residual_cfg.get("RESIDUAL_SMOOTH_GATE_MARGIN_DEG", 0.0))
    if theta_deg < start or theta_deg > end:
        return 0.0
    if margin <= 0.0:
        return 1.0
    theta = jnp.asarray(theta_deg, dtype=jnp.float32)
    start_w = smooth01((theta - start) / max(margin, 1e-3))
    end_w = smooth01((end - theta) / max(margin, 1e-3))
    return float(start_w * end_w)


def grade_loop(m, deprecated=False):
    if not bool(m["completed"]):
        return "Fail"
    cm = float(m["CTE_mean"])
    c90 = float(m["CTE_p90"])
    cmax = float(m["CTE_max"])
    gmax = float(m["Gmax"])
    vt_min = float(m["vt_min"])
    if deprecated:
        if cm < 100 and c90 < 300 and cmax < 800 and gmax < 9 and vt_min >= 190:
            return "A"
        if cm < 500 and c90 < 1200 and gmax < 10 and vt_min >= 175:
            return "B"
        return "C"

    vte = float(m["velocity_tangent_error_mean"])
    nte = float(m["nose_tangent_error_mean"])
    nve = float(m["nose_velocity_error_mean"])
    wpe = float(m["wing_plane_error_mean"])
    qe = float(m["q_error_mean_rad"])
    if (
        cm < 100
        and c90 < 300
        and cmax < 800
        and gmax < 9
        and vt_min >= 190
        and vte < 15
        and nte < 15
        and nve < 15
        and wpe < 15
        and qe < 0.5
    ):
        return "A"
    if (
        cm < 500
        and c90 < 1200
        and gmax < 10
        and vt_min >= 175
        and vte < 30
        and nte < 30
    ):
        return "B"
    return "C"


def test_grid(suite):
    official = [
        ("pu060_R12000", 60, 12000, 800, 300, 1200),
        ("pu090_R12000", 90, 12000, 1000, 400, 1500),
        ("pu105_R12000", 105, 12000, 1000, 400, 1500),
        ("pu120_R12000", 120, 12000, 1000, 400, 1800),
        ("pu135_R12000", 135, 12000, 1200, 500, 2000),
        ("pu150_R12000", 150, 12000, 1200, 500, 2000),
        ("pu180_R15000", 180, 15000, 1500, 500, 2500),
    ]
    if suite == "official":
        return official
    if suite == "v2":
        return [
            ("pu060_R12000", 60, 12000, 800, 300, 1200),
            ("pu090_R12000", 90, 12000, 1000, 400, 1500),
            ("pu120_R12000", 120, 12000, 1000, 400, 1800),
            ("pu150_R12000", 150, 12000, 1200, 500, 2000),
            ("pu165_R15000", 165, 15000, 1300, 500, 2300),
            ("pu170_R15000", 170, 15000, 1400, 500, 2400),
            ("pu175_R15000", 175, 15000, 1500, 500, 2500),
            ("pu180_R15000", 180, 15000, 1500, 500, 2500),
        ]
    if suite == "exit_v2":
        return [
            ("pu150_R12000", 150, 12000, 1200, 500, 2000),
            ("pu170_R15000", 170, 15000, 1400, 500, 2400),
            ("pu175_R15000", 175, 15000, 1500, 500, 2500),
            ("pu180_R15000", 180, 15000, 1500, 500, 2500),
            ("pu185_R15000", 185, 15000, 1600, 500, 2800),
            ("pu190_R15000", 190, 15000, 1700, 500, 3000),
            ("pu200_R15000", 200, 15000, 1800, 500, 3200),
            ("pu210_R15000", 210, 15000, 1900, 500, 3400),
        ]
    raise ValueError(f"Unknown suite: {suite}")


def restore_params(checkpoint):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    return ckptr.restore(str(checkpoint.resolve()), args=ocp.args.StandardRestore())["params"]


def restore_residual_params(checkpoint):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(str(checkpoint.resolve()), args=ocp.args.StandardRestore())
    return ckpt["params"], int(np.asarray(ckpt.get("epoch", 0)))


def load_residual_config(path):
    cfg = {
        "ACTIVATION": "relu",
        "RESIDUAL_FC_DIM_SIZE": 96,
        "RESIDUAL_GRU_HIDDEN_DIM": 64,
        "RESIDUAL_LOGIT_CLIP": 1.25,
        "RESIDUAL_GATE_START_DEG": 80.0,
        "RESIDUAL_GATE_END_DEG": 180.0,
        "RESIDUAL_PHASE_MAX_DEG": 180.0,
        "RESIDUAL_SCALE": 1.0,
        "RESIDUAL_FORCE_GATE_OFF": False,
    }
    if path:
        with path.open("r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def run_test(
    env,
    net,
    net_params,
    name,
    angle_deg,
    radius_m,
    lookahead,
    reach_radius,
    max_steps,
    residual_net=None,
    residual_params=None,
    residual_cfg=None,
):
    wps, meta = vertical_pullup_arc(
        0,
        0,
        5000,
        0.0,
        radius=radius_m,
        arc_angle_deg=angle_deg,
        n_points=max(80, int(angle_deg * 2 / 3)),
    )
    total_arc = meta["total_length_m"]
    planner = PurePursuitPlanner(
        PlannerConfig(
            lookahead_dist=lookahead,
            reach_radius=reach_radius,
            blend_steps=250,
            target_vt=250.0,
        )
    )

    rng = jax.random.PRNGKey(SEED)
    rng, reset_key = jax.random.split(rng)
    _, state = env.reset(reset_key, Params())
    q_nb_init = _quat_from_euler_nb(0.0, 0.0, 0.0)
    q_bn_init = _quat_conj(q_nb_init)
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
    planner.reset(wps, 0.0, 0.0, 0.0, 250.0)

    hstate = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
    residual_hstate = None
    if residual_net is not None:
        residual_hstate = ResidualScannedRNN.initialize_carry(
            1, int(residual_cfg.get("RESIDUAL_GRU_HIDDEN_DIM", 64))
        )
    done_flag = jnp.zeros((1,))
    rec = {
        "t": [],
        "n": [],
        "e": [],
        "a": [],
        "vt": [],
        "roll": [],
        "pitch": [],
        "yaw": [],
        "t_roll": [],
        "t_pitch": [],
        "t_hdg": [],
        "alpha": [],
        "beta": [],
        "G": [],
        "cte": [],
        "q0": [],
        "q1": [],
        "q2": [],
        "q3": [],
        "wp_idx": [],
        "theta_deg": [],
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

        path_s = planner.path_progress
        theta_deg = (path_s / total_arc) * angle_deg if total_arc > 0 else 0.0
        theta_deg = float(np.clip(theta_deg, 0.0, angle_deg))
        target_loop_roll = loop_roll(theta_deg)
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
            target_vt=jnp.array([float(target_vt)], dtype=jnp.float32),
        )

        obs = env._get_obs(state, Params())[env.agents[0]][None, None, :]
        hstate, base_pi, _ = net.apply(net_params, hstate, (obs, done_flag[None, :]))
        if residual_net is not None:
            gate = residual_gate_value(theta_deg, residual_cfg)
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

        wp_idx = result["path_ctx"]["wp_idx"]
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
        rec["cte"].append(compute_true_cte(np.array([north, east, alt]), wps, wp_idx, 10))
        rec["q0"].append(f_scalar(ps.q0))
        rec["q1"].append(f_scalar(ps.q1))
        rec["q2"].append(f_scalar(ps.q2))
        rec["q3"].append(f_scalar(ps.q3))
        rec["wp_idx"].append(wp_idx)
        rec["theta_deg"].append(theta_deg)

        if bool(done[env.agents[0]]):
            crashed = True
            break
        if planner.is_done():
            break

    n = len(rec["t"])
    completed = planner.is_done() and not crashed
    geo = {
        "velocity_tangent_error": [],
        "nose_tangent_error": [],
        "nose_velocity_error": [],
        "wing_plane_error": [],
        "belly_error": [],
        "q_error_rad": [],
        "roll_tracking_error": [],
    }

    for i in range(n):
        q_bn = np.array(
            [rec["q0"][i], rec["q1"][i], rec["q2"][i], rec["q3"][i]], dtype=np.float64
        )
        q_bn = q_bn / (np.linalg.norm(q_bn) + 1e-12)
        x_body_neu = ned_to_neu(rotate_body_to_ned(q_bn, np.array([1.0, 0.0, 0.0])))
        y_body_neu = ned_to_neu(rotate_body_to_ned(q_bn, np.array([0.0, 1.0, 0.0])))
        z_body_neu = ned_to_neu(rotate_body_to_ned(q_bn, np.array([0.0, 0.0, 1.0])))

        alpha = np.radians(rec["alpha"][i])
        beta = np.radians(rec["beta"][i])
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        vt = rec["vt"][i]
        v_body = np.array([vt * ca * cb, vt * sb, vt * sa * cb], dtype=np.float64)
        v_neu = ned_to_neu(rotate_body_to_ned(q_bn, v_body))
        v_hat_neu = v_neu / (np.linalg.norm(v_neu) + 1e-12)

        t_ref_neu, n_loop_neu = compute_loop_reference(wps, rec["wp_idx"][i])
        geo["velocity_tangent_error"].append(angle_between(v_hat_neu, t_ref_neu))
        geo["nose_tangent_error"].append(angle_between(x_body_neu, t_ref_neu))
        geo["nose_velocity_error"].append(angle_between(x_body_neu, v_hat_neu))
        geo["wing_plane_error"].append(angle_between(y_body_neu, n_loop_neu))

        z_expected = np.cross(t_ref_neu, n_loop_neu)
        z_expected = z_expected / (np.linalg.norm(z_expected) + 1e-12)
        geo["belly_error"].append(angle_between(z_body_neu, z_expected))
        geo["q_error_rad"].append(
            quat_error_angle(
                q_bn,
                np.radians(rec["t_hdg"][i]),
                np.radians(rec["t_pitch"][i]),
                np.radians(rec["t_roll"][i]),
            )
        )
        roll_err = abs(rec["roll"][i] - rec["t_roll"][i])
        geo["roll_tracking_error"].append(min(roll_err, 360.0 - roll_err))

    def arr(key):
        return np.asarray(rec[key], dtype=np.float64)

    def garr(key):
        return np.asarray(geo[key], dtype=np.float64)

    cte = arr("cte")
    vt = arr("vt")
    g = arr("G")
    alpha = arr("alpha")
    beta = arr("beta")
    roll = arr("roll")
    target_roll = arr("t_roll")
    metrics = {
        "name": name,
        "angle_deg": angle_deg,
        "radius_m": radius_m,
        "completed": bool(completed),
        "steps": n,
        "termination": "crash" if crashed else ("ok" if completed else "timeout"),
        "CTE_mean": float(cte.mean()),
        "CTE_p50": float(np.percentile(cte, 50)),
        "CTE_p90": float(np.percentile(cte, 90)),
        "CTE_max": float(cte.max()),
        "velocity_tangent_error_mean": float(garr("velocity_tangent_error").mean()),
        "velocity_tangent_error_p90": float(np.percentile(garr("velocity_tangent_error"), 90)),
        "nose_tangent_error_mean": float(garr("nose_tangent_error").mean()),
        "nose_tangent_error_p90": float(np.percentile(garr("nose_tangent_error"), 90)),
        "nose_velocity_error_mean": float(garr("nose_velocity_error").mean()),
        "nose_velocity_error_p90": float(np.percentile(garr("nose_velocity_error"), 90)),
        "wing_plane_error_mean": float(garr("wing_plane_error").mean()),
        "wing_plane_error_p90": float(np.percentile(garr("wing_plane_error"), 90)),
        "belly_error_mean": float(garr("belly_error").mean()),
        "q_error_mean_rad": float(garr("q_error_rad").mean()),
        "q_error_p90_rad": float(np.percentile(garr("q_error_rad"), 90)),
        "roll_tracking_error_mean": float(garr("roll_tracking_error").mean()),
        "env_alpha_min": float(alpha.min()),
        "env_alpha_max": float(alpha.max()),
        "env_alpha_mean": float(alpha.mean()),
        "env_beta_min": float(beta.min()),
        "env_beta_max": float(beta.max()),
        "target_roll_min": float(target_roll.min()),
        "target_roll_max": float(target_roll.max()),
        "actual_roll_min": float(roll.min()),
        "actual_roll_max": float(roll.max()),
        "actual_roll_mean": float(roll.mean()),
        "vt_min": float(vt.min()),
        "vt_mean": float(vt.mean()),
        "vt_max": float(vt.max()),
        "Gmax": float(g.max()),
        "Gmean": float(g.mean()),
        "alt_min": float(arr("a").min()),
        "alt_max": float(arr("a").max()),
    }

    theta_arr = arr("theta_deg")

    def phase_mean(key, start_deg, end_deg):
        values = garr(key)
        mask = (theta_arr >= start_deg) & (theta_arr <= end_deg)
        if not np.any(mask):
            return ""
        return float(values[mask].mean())

    for prefix, start_deg, end_deg in [
        ("phase150_180", 150.0, 180.0),
        ("phase170_200", 170.0, 200.0),
        ("phase180_200", 180.0, 200.0),
    ]:
        metrics[f"{prefix}_velocity_tangent_error_mean"] = phase_mean(
            "velocity_tangent_error", start_deg, end_deg
        )
        metrics[f"{prefix}_nose_tangent_error_mean"] = phase_mean(
            "nose_tangent_error", start_deg, end_deg
        )
        metrics[f"{prefix}_wing_plane_error_mean"] = phase_mean(
            "wing_plane_error", start_deg, end_deg
        )
    metrics["grade_cte_only_deprecated"] = grade_loop(metrics, deprecated=True)
    metrics["grade_loop_quality"] = grade_loop(metrics, deprecated=False)
    return metrics


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare_to_claude(rows, claude_csv):
    if not claude_csv:
        return [], "not_requested"
    if not claude_csv.exists():
        return [], "missing_reference"
    official = {r["name"]: r for r in read_csv(claude_csv)}
    comparison = []
    status = "pass"
    for row in rows:
        ref = official.get(row["name"])
        if not ref:
            continue
        for key in CLAUDE_KEY_METRICS:
            if key not in ref:
                continue
            got = float(row[key])
            exp = float(ref[key])
            delta = got - exp
            tol = max(1e-3, abs(exp) * 1e-5)
            ok = abs(delta) <= tol
            if not ok:
                status = "fail"
            comparison.append(
                {
                    "name": row["name"],
                    "metric": key,
                    "codex": f"{got:.12g}",
                    "claude": f"{exp:.12g}",
                    "delta": f"{delta:.12g}",
                    "tolerance": f"{tol:.12g}",
                    "aligned": ok,
                }
            )
        grade_ok = row["grade_loop_quality"] == ref.get("grade_loop_quality")
        if not grade_ok:
            status = "fail"
        comparison.append(
            {
                "name": row["name"],
                "metric": "grade_loop_quality",
                "codex": row["grade_loop_quality"],
                "claude": ref.get("grade_loop_quality", ""),
                "delta": "",
                "tolerance": "exact",
                "aligned": grade_ok,
            }
        )
    return comparison, status


def write_report(path, rows, comparison, status, checkpoint, claude_csv, suite):
    fails = [r for r in comparison if str(r["aligned"]) != "True"]
    by_angle = "\n".join(
        (
            f"| {r['angle_deg']} | {r['grade_loop_quality']} | "
            f"{float(r['CTE_mean']):.1f} | "
            f"{float(r['velocity_tangent_error_mean']):.2f} | "
            f"{float(r['nose_tangent_error_mean']):.2f} | "
            f"{float(r['nose_velocity_error_mean']):.2f} | "
            f"{float(r['wing_plane_error_mean']):.2f} | "
            f"{float(r['q_error_mean_rad']):.3f} | "
            f"{float(r['Gmax']):.2f} | "
            f"{float(r['vt_min']):.1f} | "
            f"{r['termination']} |"
        )
        for r in rows
    )
    text = [
        "# Codex/Claude Loop-Quality Evaluator Alignment",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- suite: `{suite}`",
        f"- Claude reference: `{claude_csv}`",
        f"- alignment_status: `{status}`",
        "",
        "## Codex Loop-Quality Results",
        "",
        "| angle | grade | CTE_mean | v_tang | n_tang | n_vel | wing_p | q_err | Gmax | vt_min | term |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        by_angle,
        "",
        "## Required Alignment Decision",
        "",
    ]
    if status == "pass":
        text.append("- Codex evaluator matches Claude official epoch619 loop-quality report within numerical tolerance.")
        text.append("- Training may proceed only if horizontal proxy checks are also used for promotion.")
    elif status == "not_requested":
        text.append("- Claude reference comparison was not requested for this run.")
        text.append("- Use this output for candidate-vs-baseline scoring, not evaluator alignment.")
    elif status == "missing_reference":
        text.append("- Claude reference CSV was not found. Treat alignment as incomplete and do not train.")
    else:
        text.append("- Codex evaluator disagrees with Claude official metrics. Do not train until this is fixed.")
        text.append("- First mismatches:")
        for row in fails[:20]:
            text.append(
                f"  - {row['name']} `{row['metric']}`: Codex={row['codex']} Claude={row['claude']} delta={row['delta']}"
            )
    text.append("")
    path.write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--suite", choices=["official", "v2", "exit_v2"], default="official")
    parser.add_argument("--compare-claude", type=Path, default=DEFAULT_CLAUDE_SUMMARY)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--residual-checkpoint", type=Path, default=None)
    parser.add_argument("--residual-config", type=Path, default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--gate-start", type=float, default=None)
    parser.add_argument("--gate-end", type=float, default=None)
    parser.add_argument("--force-gate-off", action="store_true")
    parser.add_argument(
        "--only-names",
        default="",
        help="Optional comma-separated test names to run from the selected suite.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PLANAX_ROOT / "results/codex_eval_alignment_epoch619" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    env = Env(Params())
    net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    obs_shape = env.observation_space(env.agents[0], Params()).shape
    h0 = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
    init_params = net.init(
        jax.random.PRNGKey(SEED),
        h0,
        (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1))),
    )
    del init_params
    net_params = restore_params(args.checkpoint)
    residual_net = None
    residual_params = None
    residual_cfg = None
    residual_epoch = None
    if args.residual_checkpoint is not None:
        residual_cfg = load_residual_config(args.residual_config)
        if args.residual_scale is not None:
            residual_cfg["RESIDUAL_SCALE"] = args.residual_scale
        if args.gate_start is not None:
            residual_cfg["RESIDUAL_GATE_START_DEG"] = args.gate_start
        if args.gate_end is not None:
            residual_cfg["RESIDUAL_GATE_END_DEG"] = args.gate_end
            residual_cfg["RESIDUAL_PHASE_MAX_DEG"] = max(
                float(residual_cfg.get("RESIDUAL_PHASE_MAX_DEG", 180.0)),
                float(args.gate_end),
            )
        if args.force_gate_off:
            residual_cfg["RESIDUAL_FORCE_GATE_OFF"] = True
        residual_net = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=residual_cfg)
        residual_params, residual_epoch = restore_residual_params(args.residual_checkpoint)

    rows = []
    tests = test_grid(args.suite)
    if args.only_names:
        wanted = {name.strip() for name in args.only_names.split(",") if name.strip()}
        tests = [test for test in tests if test[0] in wanted]
        missing = sorted(wanted - {test[0] for test in tests})
        if missing:
            raise ValueError(f"--only-names entries not in suite {args.suite}: {missing}")

    for test in tests:
        metrics = run_test(
            env,
            net,
            net_params,
            *test,
            residual_net=residual_net,
            residual_params=residual_params,
            residual_cfg=residual_cfg,
        )
        rows.append(metrics)
        print(
            f"{metrics['name']} grade={metrics['grade_loop_quality']} "
            f"CTE={metrics['CTE_mean']:.1f} "
            f"v_tang={metrics['velocity_tangent_error_mean']:.2f} "
            f"n_tang={metrics['nose_tangent_error_mean']:.2f} "
            f"wing={metrics['wing_plane_error_mean']:.2f} "
            f"term={metrics['termination']}",
            flush=True,
        )

    loop_quality_path = out_dir / "loop_quality_summary.csv"
    write_csv(loop_quality_path, rows, FIELDNAMES)
    eval_rows = [
        {
            "policy": "candidate",
            "checkpoint": str(args.checkpoint.resolve()),
            **row,
        }
        for row in rows
    ]
    write_csv(out_dir / "eval_summary.csv", eval_rows, ["policy", "checkpoint"] + FIELDNAMES)

    comparison, status = compare_to_claude(rows, None if args.no_compare else args.compare_claude)
    if comparison:
        write_csv(
            out_dir / "comparison_to_claude.csv",
            comparison,
            ["name", "metric", "codex", "claude", "delta", "tolerance", "aligned"],
        )
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "residual_checkpoint": None
        if args.residual_checkpoint is None
        else str(args.residual_checkpoint.resolve()),
        "residual_epoch": residual_epoch,
        "suite": args.suite,
        "loop_quality_summary": str(loop_quality_path),
        "eval_summary": str(out_dir / "eval_summary.csv"),
        "claude_reference": None if args.no_compare else str(args.compare_claude),
        "alignment_status": status,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(
        out_dir / "report.md",
        rows,
        comparison,
        status,
        args.checkpoint,
        None if args.no_compare else args.compare_claude,
        args.suite,
    )
    print(f"out_dir={out_dir}", flush=True)
    if status not in ("pass", "not_requested"):
        sys.exit(2)


if __name__ == "__main__":
    main()
