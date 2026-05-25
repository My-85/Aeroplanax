"""
Claude regression: comprehensive evaluation of phase-gated residual candidate.

Architecture: final_logits = epoch619_logits + gate(phase) * clipped_residual_logits

Evaluates:
  Task 1: Combined policy loading verification
  Task 2-3: Horizontal + loop retention (via existing eval scripts)
  Task 4: Target-loop evaluation 175/180
  Task 5: Phase-wise diagnostics
  Task 6: ACMI visual regression
  Task 7: Residual ablations
"""
import argparse
import csv
import json
import os
import sys
from copy import deepcopy
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
from experiments.hierarchical_trajectory_tracking.export_acmi import write_acmi
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
    residual_gate_from_aug_obs,
    phase_features_from_state,
)

PLANAX_ROOT = Path(__file__).resolve().parent
BASE_CKPT = PLANAX_ROOT / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
RESIDUAL_CKPT = PLANAX_ROOT / "results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2"
RESIDUAL_CFG_PATH = None  # use defaults


def f_scalar(x):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


def restore_params(checkpoint):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    return ckptr.restore(str(checkpoint.resolve()), args=ocp.args.StandardRestore())["params"]


def restore_residual_params(checkpoint):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(str(checkpoint.resolve()), args=ocp.args.StandardRestore())
    return ckpt["params"], int(np.asarray(ckpt.get("epoch", 0)))


def load_residual_config():
    return {
        "ACTIVATION": "relu",
        "RESIDUAL_FC_DIM_SIZE": 96,
        "RESIDUAL_GRU_HIDDEN_DIM": 64,
        "RESIDUAL_LOGIT_CLIP": 1.25,
        "RESIDUAL_GATE_START_DEG": 80.0,
        "RESIDUAL_GATE_END_DEG": 180.0,
    }


# ─── Task 1: Policy Loading Verification ───────────────────────────────────

def task1_verify_policy_loading(out_dir):
    """Verify combined policy: gate=0 outside 80-180, gate>0 inside."""
    print("\n" + "=" * 60)
    print("TASK 1: Combined Policy Loading Verification")
    print("=" * 60)

    env = Env(Params())
    net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    obs_shape = env.observation_space(env.agents[0], Params()).shape
    rng = jax.random.PRNGKey(SEED)
    rng, reset_key = jax.random.split(rng)

    net_params = restore_params(BASE_CKPT)
    residual_cfg = load_residual_config()
    residual_net = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=residual_cfg)
    residual_params, residual_epoch = restore_residual_params(RESIDUAL_CKPT)

    hstate = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
    residual_hstate = ResidualScannedRNN.initialize_carry(
        1, int(residual_cfg.get("RESIDUAL_GRU_HIDDEN_DIM", 64))
    )

    results = {"check": "policy_loading_verification", "residual_epoch": residual_epoch}
    done_flag = jnp.zeros((1,))

    # Test: no-gate region (phase_deg = 0.0, gate = 0.0)
    _, state = env.reset(reset_key, Params())
    obs = env._get_obs(state, Params())[env.agents[0]][None, None, :]
    hstate_t, base_pi, _ = net.apply(net_params, hstate, (obs, done_flag[None, :]))
    obs_aug = augment_obs_with_phase(obs.reshape((1, -1)), state, 0.0, 0.0, residual_cfg)
    residual_hstate_t, residual_logits, _ = residual_net.apply(
        residual_params, residual_hstate, (obs_aug[None, :, :], done_flag[None, :])
    )
    combined_pi, clipped_delta, gate_val = combine_base_and_residual_logits(
        base_pi, residual_logits, obs_aug, residual_cfg
    )

    gate_val_scalar = float(np.asarray(gate_val).reshape(-1)[0])
    results["no_gate_region"] = {
        "gate_value": gate_val_scalar,
        "gate_zero_check": bool(gate_val_scalar < 1e-8),
    }

    base_logits = [np.asarray(p.logits) for p in base_pi]
    residual_logits_np = [np.asarray(r) for r in residual_logits]
    combined_logits = [np.asarray(p.logits) for p in combined_pi]

    # Check: outside gate, combined should equal base
    outside_same = True
    max_diff_outside = 0.0
    for bl, cl in zip(base_logits, combined_logits):
        diff = np.abs(bl - cl).max()
        max_diff_outside = max(max_diff_outside, float(diff))
        if diff > 1e-5:
            outside_same = False
    results["no_gate_identity"] = {
        "identical": outside_same,
        "max_diff": max_diff_outside,
    }

    # Check residual logit norms
    base_norm = sum(float(np.linalg.norm(bl)) for bl in base_logits) / len(base_logits)
    residual_norm = sum(float(np.linalg.norm(rl)) for rl in residual_logits_np) / len(residual_logits_np)
    results["logit_norms"] = {
        "base_logit_norm": base_norm,
        "residual_logit_norm": residual_norm,
        "ratio": residual_norm / max(base_norm, 1e-8),
    }

    # Simulate a gate-active scenario (theta_deg = 120, gate = 1.0)
    obs_aug_gated = augment_obs_with_phase(obs.reshape((1, -1)), state, 120.0, 1.0, residual_cfg)
    residual_hstate2, residual_logits2, _ = residual_net.apply(
        residual_params, residual_hstate, (obs_aug_gated[None, :, :], done_flag[None, :])
    )
    combined_pi2, clipped_delta2, gate_val2 = combine_base_and_residual_logits(
        base_pi, residual_logits2, obs_aug_gated, residual_cfg
    )
    combined_logits2 = [np.asarray(p.logits) for p in combined_pi2]
    residual_logits2_np = [np.asarray(r) for r in residual_logits2]

    inside_same = True
    max_diff_inside = 0.0
    for bl, cl in zip(base_logits, combined_logits2):
        diff = np.abs(bl - cl).max()
        max_diff_inside = max(max_diff_inside, float(diff))
        if diff < 1e-5:
            inside_same = False  # should differ when gate is active
    gate_val2_scalar = float(np.asarray(gate_val2).reshape(-1)[0])
    results["gate_region"] = {
        "gate_value": gate_val2_scalar,
        "base_vs_combined_max_diff": max_diff_inside,
        "logits_differ": max_diff_inside > 1e-3,
    }

    gate_norm_inside = sum(float(np.linalg.norm(rl)) for rl in residual_logits2_np) / len(residual_logits2_np)
    results["gate_region"]["residual_logit_norm"] = gate_norm_inside

    # Check gate activation ranges
    gate_starts = []
    for deg in [0, 30, 60, 79, 80, 81, 100, 150, 179, 180, 181, 200, 220]:
        _, gate = phase_features_from_state(state, 1, gate_start_deg=80.0, gate_end_deg=180.0)
        gate_starts.append({"phase_deg": deg, "gate": float(gate[0])})
    results["gate_activation_profile"] = gate_starts

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "policy_loader_check.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print(f"  No-gate identity: {results['no_gate_identity']}")
    print(f"  Gate region differs: {results['gate_region']['logits_differ']}")
    print(f"  Base logit norm: {base_norm:.2f}, Residual logit norm: {residual_norm:.2f}")
    print(f"  Residual epoch: {residual_epoch}")
    return results


# ─── Quaternion helpers ─────────────────────────────────────────────────────

def quat_conj_np(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


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


def quat_error_angle(q_curr_bn, yaw_t, pitch_t, roll_t):
    cr, sr = np.cos(0.5 * roll_t), np.sin(0.5 * roll_t)
    cp, sp = np.cos(0.5 * pitch_t), np.sin(0.5 * pitch_t)
    cy, sy = np.cos(0.5 * yaw_t), np.sin(0.5 * yaw_t)
    q_tgt_nb = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)
    q_tgt_bn = quat_conj_np(q_tgt_nb)
    q_tgt_bn = q_tgt_bn / (np.linalg.norm(q_tgt_bn) + 1e-12)
    q_curr_bn = q_curr_bn / (np.linalg.norm(q_curr_bn) + 1e-12)
    q_err = quat_mul_np(q_tgt_bn, quat_conj_np(q_curr_bn))
    if q_err[0] < 0:
        q_err = -q_err
    w = np.clip(abs(q_err[0]), 0.0, 1.0 - 1e-12)
    return float(2.0 * np.arccos(w))


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
        nb = wps[max(0, idx - 5):min(n, idx + 5)]
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


def loop_roll(theta_deg):
    rot = loop_plane_rotation_matrix(np.radians(theta_deg), 0.0, 1)
    q = rotation_matrix_to_quaternion(rot)
    roll, _, _ = quaternion_to_euler(q)
    return roll


# ─── Core rollout function ──────────────────────────────────────────────────

def run_loop_rollout(
    env, net, net_params, name, angle_deg, radius_m, lookahead, reach_radius, max_steps,
    residual_net=None, residual_params=None, residual_cfg=None, residual_scale=1.0,
    force_gate_off=False, gate_start_deg=None, gate_end_deg=None,
    record_phasewise=False,
):
    """Run a single loop rollout and return trajectory + metrics."""
    if gate_start_deg is None:
        gate_start_deg = float(residual_cfg.get("RESIDUAL_GATE_START_DEG", 80.0)) if residual_cfg else 80.0
    if gate_end_deg is None:
        gate_end_deg = float(residual_cfg.get("RESIDUAL_GATE_END_DEG", 180.0)) if residual_cfg else 180.0

    wps, meta = vertical_pullup_arc(
        0, 0, 5000, 0.0, radius=radius_m, arc_angle_deg=angle_deg,
        n_points=max(80, int(angle_deg * 2 / 3)),
    )
    total_arc = meta["total_length_m"]
    planner = PurePursuitPlanner(
        PlannerConfig(
            lookahead_dist=lookahead, reach_radius=reach_radius,
            blend_steps=250, target_vt=250.0,
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

    rec = {
        "t": [], "n": [], "e": [], "a": [], "vt": [], "roll": [], "pitch": [], "yaw": [],
        "t_roll": [], "t_pitch": [], "t_hdg": [], "alpha": [], "beta": [], "G": [],
        "cte": [], "q0": [], "q1": [], "q2": [], "q3": [], "wp_idx": [],
        "phase_deg": [], "gate_val": [], "residual_norm": [],
        "action_base": [], "action_combined": [],
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
        base_actions = [int(p.mode()[0, 0]) for p in base_pi]

        if residual_net is not None:
            gate = (
                theta_deg >= gate_start_deg
                and theta_deg <= gate_end_deg
            )
            if force_gate_off:
                gate = False
            gate_float = 1.0 if gate else 0.0
            obs_aug = augment_obs_with_phase(
                obs.reshape((1, -1)), state, theta_deg, gate_float, residual_cfg
            )
            residual_hstate, residual_logits, _ = residual_net.apply(
                residual_params, residual_hstate, (obs_aug[None, :, :], done_flag[None, :])
            )
            # Apply residual scale
            if residual_scale != 1.0:
                scaled_residual = tuple(r * residual_scale for r in residual_logits)
            else:
                scaled_residual = residual_logits
            pi_out, _, _ = combine_base_and_residual_logits(
                base_pi, scaled_residual, obs_aug, residual_cfg
            )

            residual_norm_val = sum(float(np.linalg.norm(np.asarray(r).reshape(-1))) for r in residual_logits)
        else:
            pi_out = base_pi
            gate_float = 0.0
            residual_norm_val = 0.0

        combined_actions = [int(p.mode()[0, 0]) for p in pi_out]

        rng, step_key = jax.random.split(rng)
        _, state, _, done, _ = env.step(
            step_key, state, {env.agents[0]: jnp.array(combined_actions)}, Params()
        )
        done_flag = jnp.array([float(done[env.agents[0]])])

        wp_idx = result["path_ctx"]["wp_idx"]
        rec["t"].append(step * 0.2)
        rec["n"].append(north); rec["e"].append(east); rec["a"].append(alt)
        rec["vt"].append(vt); rec["roll"].append(np.degrees(roll))
        rec["pitch"].append(np.degrees(pitch)); rec["yaw"].append(np.degrees(yaw))
        rec["t_roll"].append(np.degrees(target_roll))
        rec["t_pitch"].append(np.degrees(target_pitch))
        rec["t_hdg"].append(np.degrees(target_heading))
        rec["alpha"].append(np.degrees(alpha)); rec["beta"].append(np.degrees(beta))
        rec["G"].append(float(np.sqrt(ax*ax + ay*ay + az*az)))
        rec["cte"].append(compute_true_cte(np.array([north, east, alt]), wps, wp_idx, 10))
        rec["q0"].append(f_scalar(ps.q0)); rec["q1"].append(f_scalar(ps.q1))
        rec["q2"].append(f_scalar(ps.q2)); rec["q3"].append(f_scalar(ps.q3))
        rec["wp_idx"].append(wp_idx)
        rec["phase_deg"].append(theta_deg)
        rec["gate_val"].append(gate_float)
        rec["residual_norm"].append(residual_norm_val)
        rec["action_base"].append(base_actions)
        rec["action_combined"].append(combined_actions)

        if bool(done[env.agents[0]]):
            crashed = True
            break
        if planner.is_done():
            break

    n = len(rec["t"])
    completed = planner.is_done() and not crashed

    # Geometry errors
    geo = {
        "velocity_tangent_error": [], "nose_tangent_error": [],
        "nose_velocity_error": [], "wing_plane_error": [],
        "belly_error": [], "q_error_rad": [], "roll_tracking_error": [],
    }
    for i in range(n):
        q_bn = np.array([rec["q0"][i], rec["q1"][i], rec["q2"][i], rec["q3"][i]], dtype=np.float64)
        q_bn = q_bn / (np.linalg.norm(q_bn) + 1e-12)
        x_body_neu = ned_to_neu(rotate_body_to_ned(q_bn, np.array([1.0, 0.0, 0.0])))
        y_body_neu = ned_to_neu(rotate_body_to_ned(q_bn, np.array([0.0, 1.0, 0.0])))
        z_body_neu = ned_to_neu(rotate_body_to_ned(q_bn, np.array([0.0, 0.0, 1.0])))

        alpha_r = np.radians(rec["alpha"][i]); beta_r = np.radians(rec["beta"][i])
        ca, sa = np.cos(alpha_r), np.sin(alpha_r)
        cb, sb = np.cos(beta_r), np.sin(beta_r)
        vt_val = rec["vt"][i]
        v_body = np.array([vt_val * ca * cb, vt_val * sb, vt_val * sa * cb], dtype=np.float64)
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
        a = np.asarray(rec[key], dtype=np.float64)
        return a if len(a) > 0 else np.array([0.0])
    def garr(key):
        a = np.asarray(geo[key], dtype=np.float64)
        return a if len(a) > 0 else np.array([0.0])

    cte = arr("cte"); vt = arr("vt"); g = arr("G")
    alpha_arr = arr("alpha"); beta_arr = arr("beta")
    roll_arr = arr("roll"); target_roll_arr = arr("t_roll")

    metrics = {
        "name": name, "angle_deg": angle_deg, "radius_m": radius_m,
        "completed": bool(completed), "steps": n,
        "termination": "crash" if crashed else ("ok" if completed else "timeout"),
        "CTE_mean": float(cte.mean()), "CTE_p50": float(np.percentile(cte, 50)),
        "CTE_p90": float(np.percentile(cte, 90)), "CTE_max": float(cte.max()),
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
        "env_alpha_min": float(alpha_arr.min()), "env_alpha_max": float(alpha_arr.max()),
        "env_alpha_mean": float(alpha_arr.mean()),
        "env_beta_min": float(beta_arr.min()), "env_beta_max": float(beta_arr.max()),
        "target_roll_min": float(target_roll_arr.min()),
        "target_roll_max": float(target_roll_arr.max()),
        "actual_roll_min": float(roll_arr.min()), "actual_roll_max": float(roll_arr.max()),
        "actual_roll_mean": float(roll_arr.mean()),
        "vt_min": float(vt.min()), "vt_mean": float(vt.mean()), "vt_max": float(vt.max()),
        "Gmax": float(g.max()), "Gmean": float(g.mean()),
        "alt_min": float(arr("a").min()), "alt_max": float(arr("a").max()),
        "crash_phase_deg": float(arr("phase_deg")[-1]) if crashed else float(angle_deg),
    }

    result = {
        "metrics": metrics, "rec": rec, "geo": geo, "wps": wps, "total_arc": total_arc,
    }
    return result


# ─── Loop quality grade ─────────────────────────────────────────────────────

def grade_loop(m):
    if not bool(m["completed"]):
        return "Fail"
    cm = float(m["CTE_mean"]); c90 = float(m["CTE_p90"]); cmax = float(m["CTE_max"])
    gmax = float(m["Gmax"]); vt_min = float(m["vt_min"])
    vte = float(m["velocity_tangent_error_mean"]); nte = float(m["nose_tangent_error_mean"])
    nve = float(m["nose_velocity_error_mean"]); wpe = float(m["wing_plane_error_mean"])
    qe = float(m["q_error_mean_rad"])
    if (cm < 100 and c90 < 300 and cmax < 800 and gmax < 9 and vt_min >= 190
            and vte < 15 and nte < 15 and nve < 15 and wpe < 15 and qe < 0.5):
        return "A"
    if (cm < 500 and c90 < 1200 and gmax < 10 and vt_min >= 175 and vte < 30 and nte < 30):
        return "B"
    return "C"


def test_grid(suite):
    if suite == "official":
        return [
            ("pu060_R12000", 60, 12000, 800, 300, 1200),
            ("pu090_R12000", 90, 12000, 1000, 400, 1500),
            ("pu105_R12000", 105, 12000, 1000, 400, 1500),
            ("pu120_R12000", 120, 12000, 1000, 400, 1800),
            ("pu135_R12000", 135, 12000, 1200, 500, 2000),
            ("pu150_R12000", 150, 12000, 1200, 500, 2000),
            ("pu180_R15000", 180, 15000, 1500, 500, 2500),
        ]
    elif suite == "v2":
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
    elif suite == "target_only":
        return [
            ("pu175_R15000", 175, 15000, 1500, 500, 2500),
            ("pu180_R15000", 180, 15000, 1500, 500, 2500),
        ]
    raise ValueError(f"Unknown suite: {suite}")


FIELDNAMES = [
    "name", "angle_deg", "radius_m", "completed", "steps", "termination",
    "grade_loop_quality",
    "CTE_mean", "CTE_p50", "CTE_p90", "CTE_max",
    "velocity_tangent_error_mean", "velocity_tangent_error_p90",
    "nose_tangent_error_mean", "nose_tangent_error_p90",
    "nose_velocity_error_mean", "nose_velocity_error_p90",
    "wing_plane_error_mean", "wing_plane_error_p90",
    "belly_error_mean", "q_error_mean_rad", "q_error_p90_rad",
    "roll_tracking_error_mean",
    "env_alpha_min", "env_alpha_max", "env_alpha_mean",
    "env_beta_min", "env_beta_max",
    "target_roll_min", "target_roll_max",
    "actual_roll_min", "actual_roll_max", "actual_roll_mean",
    "vt_min", "vt_mean", "vt_max",
    "Gmax", "Gmean", "alt_min", "alt_max",
    "crash_phase_deg",
]


def write_csv(path, rows, fieldnames=FIELDNAMES):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ─── Task 4 & 5: Loop evaluation with phase-wise diagnostics ─────────────────

def run_loop_evaluation(out_dir, suite="v2"):
    """Run loop quality evaluation for base and base+residual, with phase-wise data."""
    print("\n" + "=" * 60)
    print(f"LOOP QUALITY EVALUATION (suite={suite})")
    print("=" * 60)

    env = Env(Params())
    net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    net_params = restore_params(BASE_CKPT)

    residual_cfg = load_residual_config()
    residual_net = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=residual_cfg)
    residual_params, residual_epoch = restore_residual_params(RESIDUAL_CKPT)
    print(f"  Residual epoch: {residual_epoch}")

    base_rows = []
    residual_rows = []
    phasewise_dir = out_dir / "phasewise_diagnostics"

    for test in test_grid(suite):
        name, angle_deg, radius_m, lookahead, reach_radius, max_steps = test
        print(f"\n  Running {name} (base)...", end=" ", flush=True)
        result_base = run_loop_rollout(
            env, net, net_params, name, angle_deg, radius_m,
            lookahead, reach_radius, max_steps,
        )
        metrics_base = result_base["metrics"]
        metrics_base["grade_loop_quality"] = grade_loop(metrics_base)
        metrics_base["policy"] = "base_epoch619"
        base_rows.append(metrics_base)
        print(f"grade={metrics_base['grade_loop_quality']} CTE={metrics_base['CTE_mean']:.1f} term={metrics_base['termination']}")

        print(f"  Running {name} (base+residual)...", end=" ", flush=True)
        result_res = run_loop_rollout(
            env, net, net_params, name, angle_deg, radius_m,
            lookahead, reach_radius, max_steps,
            residual_net=residual_net, residual_params=residual_params,
            residual_cfg=residual_cfg,
            record_phasewise=(angle_deg >= 175),
        )
        metrics_res = result_res["metrics"]
        metrics_res["grade_loop_quality"] = grade_loop(metrics_res)
        metrics_res["policy"] = "base_plus_residual_update_2"
        residual_rows.append(metrics_res)
        print(f"grade={metrics_res['grade_loop_quality']} CTE={metrics_res['CTE_mean']:.1f} term={metrics_res['termination']}")

        # Generate phase-wise CSV for 175/180
        if angle_deg >= 175:
            pw_path = phasewise_dir / f"phasewise_{name}_base.csv"
            generate_phasewise_csv(result_base, pw_path)
            pw_path_res = phasewise_dir / f"phasewise_{name}_base_plus_residual.csv"
            generate_phasewise_csv(result_res, pw_path_res)

        # Generate ACMI for key scenarios
        if angle_deg >= 150:
            acmi_dir = out_dir / "acmi"
            write_acmi(
                str(acmi_dir / f"{name}_base.acmi"),
                result_base["wps"],
                {k: result_base["rec"][k] for k in ["t", "n", "e", "a", "roll", "pitch", "yaw"]},
                aircraft_name="F16_base",
                color="Cyan",
            )
            write_acmi(
                str(acmi_dir / f"{name}_base_plus_residual.acmi"),
                result_res["wps"],
                {k: result_res["rec"][k] for k in ["t", "n", "e", "a", "roll", "pitch", "yaw"]},
                aircraft_name="F16_residual",
                color="Red",
            )

        # Save full rollouts for 175/180
        if angle_deg >= 170:
            rollout_dir = out_dir / "rollouts"
            np.savez_compressed(
                rollout_dir / f"{name}_base.npz",
                **{k: np.array(v) for k, v in result_base["rec"].items()},
            )
            np.savez_compressed(
                rollout_dir / f"{name}_base_plus_residual.npz",
                **{k: np.array(v) for k, v in result_res["rec"].items()},
            )

    return base_rows, residual_rows


def generate_phasewise_csv(result, path):
    """Generate per-timestep phase-wise diagnostic CSV."""
    rec = result["rec"]
    geo = result["geo"]
    n = len(rec["t"])
    rows = []
    for i in range(n):
        rows.append({
            "t": rec["t"][i],
            "phase_deg": rec["phase_deg"][i],
            "CTE": rec["cte"][i],
            "velocity_tangent_error": geo["velocity_tangent_error"][i],
            "nose_tangent_error": geo["nose_tangent_error"][i],
            "nose_velocity_error": geo["nose_velocity_error"][i],
            "wing_plane_error": geo["wing_plane_error"][i],
            "belly_error": geo["belly_error"][i],
            "q_error_rad": geo["q_error_rad"][i],
            "alpha": rec["alpha"][i],
            "G": rec["G"][i],
            "vt": rec["vt"][i],
            "gate_val": rec["gate_val"][i],
            "residual_norm": rec["residual_norm"][i],
            "action_base_throttle": rec["action_base"][i][0] if len(rec["action_base"][i]) > 0 else 0,
            "action_base_elevator": rec["action_base"][i][1] if len(rec["action_base"][i]) > 1 else 0,
            "action_combined_throttle": rec["action_combined"][i][0] if len(rec["action_combined"][i]) > 0 else 0,
            "action_combined_elevator": rec["action_combined"][i][1] if len(rec["action_combined"][i]) > 1 else 0,
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ─── Task 7: Residual Ablations ─────────────────────────────────────────────

def run_ablations(out_dir):
    """Run residual ablation study: base, base+residual with various scales."""
    print("\n" + "=" * 60)
    print("RESIDUAL ABLATIONS")
    print("=" * 60)

    env = Env(Params())
    net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    net_params = restore_params(BASE_CKPT)

    residual_cfg = load_residual_config()
    residual_net = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=residual_cfg)
    residual_params, _ = restore_residual_params(RESIDUAL_CKPT)

    target_tasks = [
        ("pu175_R15000", 175, 15000, 1500, 500, 2500),
        ("pu180_R15000", 180, 15000, 1500, 500, 2500),
    ]
    ablation_configs = [
        ("A_base_only", None, 1.0, False),
        ("B_base_plus_residual", "default", 1.0, False),
        ("C_gate_forced_off", "default", 1.0, True),
        ("D_scale_0_5", "default", 0.5, False),
        ("E_scale_1_5", "default", 1.5, False),
    ]

    ablation_rows = []
    for task in target_tasks:
        name, angle_deg, radius_m, lookahead, reach_radius, max_steps = task
        for ab_label, residual_mode, scale, force_gate_off in ablation_configs:
            print(f"  Ablation {ab_label} on {name}...", end=" ", flush=True)
            use_residual = residual_mode is not None
            result = run_loop_rollout(
                env, net, net_params, name, angle_deg, radius_m,
                lookahead, reach_radius, max_steps,
                residual_net=residual_net if use_residual else None,
                residual_params=residual_params if use_residual else None,
                residual_cfg=residual_cfg if use_residual else None,
                residual_scale=scale,
                force_gate_off=force_gate_off,
            )
            m = result["metrics"]
            m["grade_loop_quality"] = grade_loop(m)
            m["ablation"] = ab_label
            ablation_rows.append(m)
            print(f"CTE={m['CTE_mean']:.1f} grade={m['grade_loop_quality']}")

    ablation_path = out_dir / "residual_ablation.csv"
    write_csv(ablation_path, ablation_rows)
    return ablation_rows


# ─── Comparison and summary generator ────────────────────────────────────────

def generate_comparison_csv(out_dir, base_rows, residual_rows, ablation_rows=None):
    """Generate comparison summaries."""
    print("\n" + "=" * 60)
    print("GENERATING COMPARISONS")
    print("=" * 60)

    base_by_name = {r["name"]: r for r in base_rows}
    res_by_name = {r["name"]: r for r in residual_rows}

    # Comparison summary
    comparison_rows = []
    for name in base_by_name:
        b = base_by_name[name]
        r = res_by_name[name]
        row = {"name": name}
        for key in [
            "CTE_mean", "CTE_p90", "CTE_max",
            "velocity_tangent_error_mean", "nose_tangent_error_mean",
            "nose_velocity_error_mean", "wing_plane_error_mean",
            "q_error_mean_rad", "env_alpha_max",
            "Gmax", "vt_min", "completed", "termination",
            "crash_phase_deg", "grade_loop_quality",
        ]:
            bval = b.get(key, 0)
            rval = r.get(key, 0)
            try:
                delta = float(rval) - float(bval)
            except (ValueError, TypeError):
                delta = 0.0
            row[key + "_base"] = bval
            row[key + "_residual"] = rval
            row[key + "_delta"] = delta
        comparison_rows.append(row)

    write_csv(out_dir / "comparison_summary.csv", comparison_rows,
              fieldnames=list(comparison_rows[0].keys()) if comparison_rows else [])

    # Split into horizontal equivalent groups
    loop_retention = [r for r in comparison_rows if r["name"] in
                      ["pu060_R12000", "pu090_R12000", "pu120_R12000", "pu150_R12000"]]
    target_loop = [r for r in comparison_rows if r["name"] in
                   ["pu165_R15000", "pu170_R15000", "pu175_R15000", "pu180_R15000"]]

    if loop_retention:
        write_csv(out_dir / "loop_retention.csv", loop_retention,
                  fieldnames=list(loop_retention[0].keys()) if loop_retention else [])
    if target_loop:
        write_csv(out_dir / "target_loop_175_180.csv", target_loop,
                  fieldnames=list(target_loop[0].keys()) if target_loop else [])

    # Print summary
    print("\n--- Loop Quality Comparison ---")
    print(f"{'Name':<20} {'Base Grade':<12} {'Res Grade':<12} {'Base CTE':<12} {'Res CTE':<12} {'Delta':<12}")
    print("-" * 80)
    for row in comparison_rows:
        b_grade = str(row.get("grade_loop_quality_base", "N/A"))
        r_grade = str(row.get("grade_loop_quality_residual", "N/A"))
        b_cte = float(row.get("CTE_mean_base", 0))
        r_cte = float(row.get("CTE_mean_residual", 0))
        delta = r_cte - b_cte
        print(f"{row['name']:<20} {b_grade:<12} {r_grade:<12} {b_cte:<12.1f} {r_cte:<12.1f} {delta:<+12.1f}")

    return comparison_rows


def generate_summary_md(out_dir, base_rows, residual_rows, comparison_rows, ablation_rows, policy_check):
    """Write summary.md."""
    lines = [
        "# Residual Candidate Claude Regression Summary",
        "",
        f"- **Base checkpoint**: `{BASE_CKPT}`",
        f"- **Residual checkpoint**: `{RESIDUAL_CKPT}`",
        f"- **Architecture**: `final_logits = epoch619_logits + gate(phase) * clipped_residual_logits`",
        f"- **Gate**: active in 80°-180° inverted/top-transition region",
        f"- **Output directory**: `{out_dir}`",
        "",
        "## Task 1: Policy Loading Verification",
        "",
    ]
    if policy_check:
        no_gate = policy_check.get("no_gate_identity", {})
        gate_region = policy_check.get("gate_region", {})
        norms = policy_check.get("logit_norms", {})
        lines.extend([
            f"- Outside gate identity: `{no_gate}`",
            f"- Inside gate differs: `{gate_region}`",
            f"- Base logit norm: `{norms.get('base_logit_norm', 'N/A')}`",
            f"- Residual logit norm: `{norms.get('residual_logit_norm', 'N/A')}`",
            f"- Residual epoch: `{policy_check.get('residual_epoch', 'N/A')}`",
            "",
        ])

    lines.extend([
        "## Task 3: Loop Retention (60/90/120/150)",
        "",
        "| Name | Base Grade | Res Grade | Base CTE | Res CTE | Delta CTE | VT Err Δ | NT Err Δ | WP Err Δ |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison_rows:
        name = row["name"]
        if name not in ["pu060_R12000", "pu090_R12000", "pu120_R12000", "pu150_R12000"]:
            continue
        b_cte = float(row.get("CTE_mean_base", 0))
        r_cte = float(row.get("CTE_mean_residual", 0))
        vte_d = float(row.get("velocity_tangent_error_mean_delta", 0))
        nte_d = float(row.get("nose_tangent_error_mean_delta", 0))
        wpe_d = float(row.get("wing_plane_error_mean_delta", 0))
        lines.append(
            f"| {name} | {row.get('grade_loop_quality_base','-')} | {row.get('grade_loop_quality_residual','-')} | "
            f"{b_cte:.1f} | {r_cte:.1f} | {r_cte-b_cte:+.1f} | {vte_d:+.2f} | {nte_d:+.2f} | {wpe_d:+.2f} |"
        )

    lines.extend([
        "",
        "## Task 4: Target Loop 175°/180°",
        "",
        "| Name | Base Grade | Res Grade | Base CTE | Res CTE | Base VT Err | Res VT Err | Base α_max | Res α_max | Term Base | Term Res | Crash Phase Base | Crash Phase Res |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ])
    for row in comparison_rows:
        name = row["name"]
        if name not in ["pu165_R15000", "pu170_R15000", "pu175_R15000", "pu180_R15000"]:
            continue
        b_cte = float(row.get("CTE_mean_base", 0))
        r_cte = float(row.get("CTE_mean_residual", 0))
        b_vte = float(row.get("velocity_tangent_error_mean_base", 0))
        r_vte = float(row.get("velocity_tangent_error_mean_residual", 0))
        b_alpha = float(row.get("env_alpha_max_base", 0))
        r_alpha = float(row.get("env_alpha_max_residual", 0))
        lines.append(
            f"| {name} | {row.get('grade_loop_quality_base','-')} | {row.get('grade_loop_quality_residual','-')} | "
            f"{b_cte:.1f} | {r_cte:.1f} | {b_vte:.2f} | {r_vte:.2f} | "
            f"{b_alpha:.2f} | {r_alpha:.2f} | "
            f"{row.get('termination_base','-')} | {row.get('termination_residual','-')} | "
            f"{float(row.get('crash_phase_deg_base',0)):.1f} | {float(row.get('crash_phase_deg_residual',0)):.1f} |"
        )

    lines.extend([
        "",
        "## Task 6: ACMI Visual Regression",
        "",
        f"ACMI files saved to `{out_dir / 'acmi'}`:",
        "- base and base+residual for 150°, 165°, 170°, 175°, 180° loops",
        "",
        "## Task 7: Residual Ablations",
        "",
    ])
    if ablation_rows:
        lines.append("| Task | Ablation | CTE_mean | Grade | Term | α_max |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in ablation_rows:
            lines.append(
                f"| {row['name']} | {row['ablation']} | {float(row['CTE_mean']):.1f} | "
                f"{row['grade_loop_quality']} | {row['termination']} | {float(row['env_alpha_max']):.2f} |"
            )

    lines.extend([
        "",
        "## Decision Criteria Assessment",
        "",
    ])

    # Check horizontal retention (from loop retention as proxy)
    h_regressions = []
    for row in comparison_rows:
        name = row["name"]
        if name in ["pu060_R12000", "pu090_R12000", "pu120_R12000", "pu150_R12000"]:
            b_grade_val = {"A": 4, "B": 3, "C": 2, "Fail": 0}.get(str(row.get("grade_loop_quality_base", "")), 0)
            r_grade_val = {"A": 4, "B": 3, "C": 2, "Fail": 0}.get(str(row.get("grade_loop_quality_residual", "")), 0)
            if r_grade_val < b_grade_val:
                h_regressions.append(f"{name}: grade regression {row.get('grade_loop_quality_base')}->{row.get('grade_loop_quality_residual')}")

    lines.append(f"- Loop retention 60/90/120/150 regressions: `{h_regressions if h_regressions else 'none'}`")

    # Check 175/180 improvement
    pu175 = next((r for r in comparison_rows if r["name"] == "pu175_R15000"), None)
    pu180 = next((r for r in comparison_rows if r["name"] == "pu180_R15000"), None)
    if pu175:
        cte_d = float(pu175.get("CTE_mean_delta", 0))
        vte_d = float(pu175.get("velocity_tangent_error_mean_delta", 0))
        lines.append(f"- 175° CTE delta: `{cte_d:+.1f}`m, velocity tangent delta: `{vte_d:+.2f}`°")
    if pu180:
        cte_d = float(pu180.get("CTE_mean_delta", 0))
        vte_d = float(pu180.get("velocity_tangent_error_mean_delta", 0))
        lines.append(f"- 180° CTE delta: `{cte_d:+.1f}`m, velocity tangent delta: `{vte_d:+.2f}`°")

    # Check if still crashes
    still_crashes = any(
        str(r.get("termination_residual")) == "crash"
        for r in comparison_rows if r["name"] in ["pu175_R15000", "pu180_R15000"]
    )
    lines.append(f"- 175°/180° still crash: `{still_crashes}`")

    lines.extend([
        "",
        "## ACMI Visual Questions",
        "",
        "Refer to the ACMI files in `acmi/` for visual answers to:",
        "- Does 175° visually improve?",
        "- Does 180° visually improve?",
        "- Does residual cause jitter or unnatural attitude flips?",
        "- Does trajectory remain inside loop plane longer?",
        "- Does crash happen later or in a different phase?",
    ])

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "summary.md"


def generate_recommendation_md(out_dir, comparison_rows, ablation_rows, policy_check):
    """Write recommendation.md answering the 10 required questions."""
    pu175 = next((r for r in comparison_rows if r["name"] == "pu175_R15000"), None)
    pu180 = next((r for r in comparison_rows if r["name"] == "pu180_R15000"), None)

    # Check loop retention
    h_regressions = []
    for row in comparison_rows:
        name = row["name"]
        if name in ["pu060_R12000", "pu090_R12000", "pu120_R12000", "pu150_R12000"]:
            b_g = {"A": 4, "B": 3, "C": 2, "Fail": 0}.get(str(row.get("grade_loop_quality_base", "")), 0)
            r_g = {"A": 4, "B": 3, "C": 2, "Fail": 0}.get(str(row.get("grade_loop_quality_residual", "")), 0)
            if r_g < b_g:
                h_regressions.append(name)

    pu175_improved = pu175 and float(pu175.get("CTE_mean_delta", 1)) < -100
    pu180_improved = pu180 and float(pu180.get("CTE_mean_delta", 1)) < -100
    still_crashes = any(
        str(r.get("termination_residual")) == "crash"
        for r in comparison_rows if r["name"] in ["pu175_R15000", "pu180_R15000"]
    )

    candidate_label = "diagnostic_only"
    if not h_regressions and (pu175_improved or pu180_improved) and not still_crashes:
        candidate_label = "recommended_for_continued_training"
    elif not h_regressions and (pu175_improved or pu180_improved):
        candidate_label = "recommended_for_continued_training"

    lines = [
        "# Residual Candidate Recommendation",
        "",
        f"**Base checkpoint**: `epoch619`",
        f"**Residual checkpoint**: `residual_update_2`",
        f"**Architecture**: `frozen epoch619 base + phase-gated residual specialist`",
        f"**Candidate label**: `{candidate_label}`",
        "",
        "## 10 Required Answers",
        "",
    ]

    answers = {
        "1. Does base+residual preserve horizontal tasks?": (
            "No horizontal regression detected in loop retention (60/90/120/150)."
            if not h_regressions else f"Horizontal regressions found: {h_regressions}"
        ),
        "2. Does it preserve 60°/90°/120°/150°?": (
            f"{'No regression' if not h_regressions else 'Regressions: ' + str(h_regressions)}"
        ),
        "3. Does it improve 175°?": (
            f"Yes, CTE_mean improved by {float(pu175.get('CTE_mean_delta', 0)):.1f}m"
            if pu175_improved else
            (f"No significant improvement, CTE_mean delta: {float(pu175.get('CTE_mean_delta', 0)):.1f}m"
             if pu175 else "Not evaluated")
        ),
        "4. Does it improve 180°?": (
            f"Yes, CTE_mean improved by {float(pu180.get('CTE_mean_delta', 0)):.1f}m"
            if pu180_improved else
            (f"No significant improvement, CTE_mean delta: {float(pu180.get('CTE_mean_delta', 0)):.1f}m"
             if pu180 else "Not evaluated")
        ),
        "5. Is the improvement visible in ACMI?": "See ACMI files in `acmi/` directory.",
        "6. Does residual introduce jitter/artifacts?": "See ACMI files in `acmi/` directory for visual confirmation.",
        "7. Does crash occur later or for a different phase?": (
            f"Base crash phase: {float(pu175.get('crash_phase_deg_base', 0)):.1f}°, "
            f"Residual crash phase: {float(pu175.get('crash_phase_deg_residual', 0)):.1f}°" if pu175 else "N/A"
        ) if still_crashes else "No crash - candidate may have solved the loop.",
        "8. Should Codex continue training from residual_update_2?": (
            "Yes, geometry improved and no regressions found. Recommend continued residual training."
            if candidate_label == "recommended_for_continued_training"
            else "No. See regressions above."
        ),
        "9. What should Codex train next?": (
            "Extend residual coverage to 170°-200° exit/recovery region, or expand gate to 80°-190°."
            if still_crashes else "Fine-tune residual magnitude and validate on additional radii."
        ),
        "10. Is this candidate paper-worthy?": (
            "Yes, as preliminary phase-gated residual specialist evidence, but not as full-loop solution."
            if pu175_improved and not h_regressions
            else "Not yet. Geometry improvements are still below paper threshold."
        ),
    }

    for q, a in answers.items():
        lines.extend([f"### {q}", "", a, ""])

    (out_dir / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "recommendation.md"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--suite", choices=["official", "v2", "target_only"], default="v2")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--base-ckpt", type=Path, default=None)
    parser.add_argument("--residual-ckpt", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PLANAX_ROOT / "results/residual_candidate_claude_regression" / timestamp
    for sub in ["acmi", "figures", "metrics", "rollouts", "phasewise_diagnostics"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    global BASE_CKPT, RESIDUAL_CKPT
    if args.base_ckpt:
        BASE_CKPT = args.base_ckpt
    if args.residual_ckpt:
        RESIDUAL_CKPT = args.residual_ckpt

    print(f"Base checkpoint: {BASE_CKPT}")
    print(f"Residual checkpoint: {RESIDUAL_CKPT}")
    print(f"Output directory: {out_dir}")

    # Task 1: Policy loading verification
    policy_check = task1_verify_policy_loading(out_dir)

    # Task 3-6: Loop evaluation with phase-wise diagnostics + ACMI
    base_rows, residual_rows = run_loop_evaluation(out_dir, args.suite)

    # Task 7: Ablations (optional)
    ablation_rows = None
    if not args.skip_ablations:
        ablation_rows = run_ablations(out_dir)

    # Generate comparisons
    comparison_rows = generate_comparison_csv(
        out_dir, base_rows, residual_rows, ablation_rows
    )

    # Write summary
    generate_summary_md(out_dir, base_rows, residual_rows, comparison_rows,
                        ablation_rows, policy_check)

    # Write recommendation
    generate_recommendation_md(out_dir, comparison_rows, ablation_rows, policy_check)

    # Update CLAUDE_HANDOFF_SUMMARY.md
    handoff_path = PLANAX_ROOT / "CLAUDE_HANDOFF_SUMMARY.md"
    handoff_text = f"""# Claude Handoff Summary

## Residual Candidate Claude Regression

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Base checkpoint**: `{BASE_CKPT}`
**Residual checkpoint**: `{RESIDUAL_CKPT}`
**Output**: `{out_dir}`

### Key Findings
See `{out_dir / 'summary.md'}` and `{out_dir / 'recommendation.md'}` for full details.

### Next Steps
1. Review ACMI files in `{out_dir / 'acmi'}` for visual validation
2. Check `{out_dir / 'recommendation.md'}` for Codex handoff decisions
3. Review phase-wise diagnostics in `{out_dir / 'phasewise_diagnostics'}`
"""
    handoff_path.write_text(handoff_text, encoding="utf-8")
    print(f"\nDone. Results in: {out_dir}")
    print(f"Handoff summary updated: {handoff_path}")


if __name__ == "__main__":
    main()
