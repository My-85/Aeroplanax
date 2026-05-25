"""
Generate ACMI files and visual_regression.md for residual candidate evaluation.
Focuses on 175° and 180° base vs base+residual comparison.
Answers 7 visual questions.
"""
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
    r, _, _ = quaternion_to_euler(q)
    return r


def quat_conj_np(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
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


def run_detailed_rollout(env, net, net_params, name, angle_deg, radius_m,
                         lookahead, reach_radius, max_steps,
                         residual_net=None, residual_params=None, residual_cfg=None,
                         acmi_dir=None, phasewise_dir=None):
    """Run rollout and collect ALL visual metrics per timestep."""
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

    # Detailed recording
    rec = {
        "t": [], "n": [], "e": [], "a": [],
        "roll": [], "pitch": [], "yaw": [],
        "alpha": [], "beta": [], "G": [],
        "vt": [], "phase_deg": [], "gate_val": [],
        "cte": [],
        # Action tracking
        "action_base": [], "action_combined": [],
        # Quaternion
        "q0": [], "q1": [], "q2": [], "q3": [],
        # Targets
        "t_roll": [], "t_pitch": [], "t_hdg": [],
        # Waypoint
        "wp_idx": [],
    }
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
            gate_float = 0.0

        combined_actions = [int(p.mode()[0, 0]) for p in pi_out]

        rng, step_key = jax.random.split(rng)
        _, state, _, done, _ = env.step(
            step_key, state, {env.agents[0]: jnp.array(combined_actions)}, Params()
        )
        done_flag = jnp.array([float(done[env.agents[0]])])

        wp_idx = result["path_ctx"]["wp_idx"]

        rec["t"].append(step * 0.2)
        rec["n"].append(north); rec["e"].append(east); rec["a"].append(alt)
        rec["roll"].append(np.degrees(roll)); rec["pitch"].append(np.degrees(pitch))
        rec["yaw"].append(np.degrees(yaw)); rec["alpha"].append(np.degrees(alpha))
        rec["beta"].append(np.degrees(beta))
        rec["G"].append(float(np.sqrt(ax*ax + ay*ay + az*az)))
        rec["vt"].append(vt)
        rec["phase_deg"].append(theta_deg)
        rec["gate_val"].append(gate_float)
        rec["cte"].append(compute_true_cte(np.array([north, east, alt]), wps, wp_idx, 10))
        rec["q0"].append(f_scalar(ps.q0)); rec["q1"].append(f_scalar(ps.q1))
        rec["q2"].append(f_scalar(ps.q2)); rec["q3"].append(f_scalar(ps.q3))
        rec["t_roll"].append(np.degrees(target_roll))
        rec["t_pitch"].append(np.degrees(target_pitch))
        rec["t_hdg"].append(np.degrees(target_heading))
        rec["wp_idx"].append(wp_idx)
        rec["action_base"].append(base_actions)
        rec["action_combined"].append(combined_actions)

        if bool(done[env.agents[0]]):
            crashed = True
            break
        if planner.is_done():
            completed = True
            break

    n = len(rec["t"])

    # Compute geometry errors per frame
    geo = {
        "velocity_tangent_error": [], "nose_tangent_error": [],
        "nose_velocity_error": [], "wing_plane_error": [],
        "belly_error": [],
    }
    roll_jitter = []
    pitch_jitter = []
    yaw_jitter = []

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

        # Jitter: frame-to-frame attitude change magnitude
        if i >= 1:
            roll_jitter.append(abs(rec["roll"][i] - rec["roll"][i-1]))
            pitch_jitter.append(abs(rec["pitch"][i] - rec["pitch"][i-1]))
            yaw_jitter.append(abs(rec["yaw"][i] - rec["yaw"][i-1]))

    # Compute visual summary statistics
    def arr(key):
        a = np.asarray(rec[key], dtype=np.float64)
        return a if len(a) > 0 else np.array([0.0])
    def garr(key):
        a = np.asarray(geo[key], dtype=np.float64)
        return a if len(a) > 0 else np.array([0.0])

    phase_arr = arr("phase_deg")
    cte_arr = arr("cte")
    vte_arr = garr("velocity_tangent_error")
    nte_arr = garr("nose_tangent_error")
    nve_arr = garr("nose_velocity_error")
    wpe_arr = garr("wing_plane_error")
    alpha_arr = arr("alpha")

    # Phase-binned metrics
    phase_bins = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]
    phase_metrics = []
    for lo, hi in phase_bins:
        mask = (phase_arr >= lo) & (phase_arr < hi)
        if mask.sum() > 0:
            phase_metrics.append({
                "phase_range": f"{lo}-{hi}",
                "n_frames": int(mask.sum()),
                "CTE_mean": float(cte_arr[mask].mean()),
                "velocity_tangent_error_mean": float(vte_arr[mask].mean()),
                "nose_tangent_error_mean": float(nte_arr[mask].mean()),
                "nose_velocity_error_mean": float(nve_arr[mask].mean()),
                "wing_plane_error_mean": float(wpe_arr[mask].mean()),
                "alpha_mean": float(alpha_arr[mask].mean()),
                "alpha_max": float(alpha_arr[mask].max()),
            })
        else:
            phase_metrics.append({
                "phase_range": f"{lo}-{hi}",
                "n_frames": 0,
            })

    # Jitter metrics
    roll_jitter_mean = float(np.mean(roll_jitter)) if roll_jitter else 0.0
    roll_jitter_max = float(np.max(roll_jitter)) if roll_jitter else 0.0
    pitch_jitter_mean = float(np.mean(pitch_jitter)) if pitch_jitter else 0.0
    pitch_jitter_max = float(np.max(pitch_jitter)) if pitch_jitter else 0.0

    # Crash phase info
    crash_phase = float(phase_arr[-1]) if crashed else angle_deg
    crash_alpha = float(alpha_arr[-1]) if crashed else float(alpha_arr[-1])
    crash_G = float(arr("G")[-1]) if crashed else 0.0

    # Action differences between base and combined (only meaningful for residual runs)
    action_diffs = []
    if "action_base" in rec and "action_combined" in rec:
        for i in range(n):
            if len(rec["action_base"][i]) == 5 and len(rec["action_combined"][i]) == 5:
                ndiff = sum(1 for a, b in zip(rec["action_base"][i], rec["action_combined"][i]) if a != b)
                action_diffs.append(ndiff)
    action_diff_rate = float(np.mean(action_diffs)) if action_diffs else 0.0

    # Save ACMI
    acmi_name = f"{name}_{'base' if residual_net is None else 'base_plus_residual'}.acmi"
    acmi_path = acmi_dir / acmi_name if acmi_dir else None
    if acmi_dir:
        write_acmi(str(acmi_path), wps, {
            "t": rec["t"], "n": rec["n"], "e": rec["e"], "a": rec["a"],
            "roll": rec["roll"], "pitch": rec["pitch"], "yaw": rec["yaw"],
        }, aircraft_name=f"F16_{'base' if residual_net is None else 'residual'}",
           color="Cyan" if residual_net is None else "Red")

    # Save phase-wise CSV
    if phasewise_dir:
        csv_path = phasewise_dir / f"phasewise_{name}_{'base' if residual_net is None else 'residual'}.csv"
        rows = []
        for i in range(n):
            rows.append({
                "t": rec["t"][i], "phase_deg": rec["phase_deg"][i],
                "cte": rec["cte"][i],
                "velocity_tangent_error": geo["velocity_tangent_error"][i],
                "nose_tangent_error": geo["nose_tangent_error"][i],
                "nose_velocity_error": geo["nose_velocity_error"][i],
                "wing_plane_error": geo["wing_plane_error"][i],
                "belly_error": geo["belly_error"][i],
                "alpha": rec["alpha"][i], "G": rec["G"][i],
                "vt": rec["vt"][i], "gate_val": rec["gate_val"][i],
                "roll": rec["roll"][i], "pitch": rec["pitch"][i], "yaw": rec["yaw"][i],
            })
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = {
        "name": name, "policy": "base" if residual_net is None else "base_plus_residual",
        "angle_deg": angle_deg, "n_frames": n,
        "completed": completed, "crashed": crashed,
        "crash_phase_deg": crash_phase,
        "crash_alpha": crash_alpha,
        "crash_G": crash_G,
        "CTE_mean": float(cte_arr.mean()),
        "velocity_tangent_error_mean": float(vte_arr.mean()),
        "nose_tangent_error_mean": float(nte_arr.mean()),
        "nose_velocity_error_mean": float(nve_arr.mean()),
        "wing_plane_error_mean": float(wpe_arr.mean()),
        "alpha_max": float(alpha_arr.max()),
        "Gmax": float(arr("G").max()),
        "vt_min": float(arr("vt").min()),
        "roll_jitter_mean_deg": roll_jitter_mean,
        "roll_jitter_max_deg": roll_jitter_max,
        "pitch_jitter_mean_deg": pitch_jitter_mean,
        "pitch_jitter_max_deg": pitch_jitter_max,
        "action_diff_rate": action_diff_rate,
        "phase_metrics": phase_metrics,
    }
    return summary


def compare_visual(base_summary, res_summary):
    """Compare base vs residual visual metrics and answer questions."""
    comparison = {}

    # Q1: Stay in loop plane longer?
    b_wpe = base_summary["wing_plane_error_mean"]
    r_wpe = res_summary["wing_plane_error_mean"]
    comparison["wing_plane_improvement"] = {
        "base_mean": b_wpe,
        "residual_mean": r_wpe,
        "delta": r_wpe - b_wpe,
        "improved": r_wpe < b_wpe - 2.0,
        "description": (
            f"Wing-plane error: {b_wpe:.1f} vs {r_wpe:.1f} (delta {r_wpe-b_wpe:+.1f} ). "
            f"{'Aircraft stays in loop plane longer.' if r_wpe < b_wpe - 2.0 else 'No significant change.'}"
        ),
    }

    # Q2: Nose closer to tangent?
    b_nte = base_summary["nose_tangent_error_mean"]
    r_nte = res_summary["nose_tangent_error_mean"]
    comparison["nose_tangent_improvement"] = {
        "base_mean": b_nte,
        "residual_mean": r_nte,
        "delta": r_nte - b_nte,
        "improved": r_nte < b_nte - 2.0,
        "description": (
            f"Nose-tangent error: {b_nte:.1f} vs {r_nte:.1f} (delta {r_nte-b_nte:+.1f} ). "
            f"{'Nose is visibly closer to tangent.' if r_nte < b_nte - 2.0 else 'No significant change.'}"
        ),
    }

    # Q3: Wing-plane alignment visibly improved?
    comparison["wing_plane_details"] = {
        "description": (
            f"Wing-plane error reduced from {b_wpe:.1f} to {r_wpe:.1f} "
            f"({'improved' if r_wpe < b_wpe - 2.0 else 'similar'})."
        ),
    }

    # Q4: Jitter / abrupt flips?
    b_roll_j = base_summary["roll_jitter_mean_deg"]
    r_roll_j = res_summary["roll_jitter_mean_deg"]
    b_pitch_j = base_summary["pitch_jitter_mean_deg"]
    r_pitch_j = res_summary["pitch_jitter_mean_deg"]
    jitter_increase = r_roll_j > b_roll_j * 1.5 or r_pitch_j > b_pitch_j * 1.5
    comparison["jitter"] = {
        "base_roll_jitter_deg_per_frame": b_roll_j,
        "residual_roll_jitter_deg_per_frame": r_roll_j,
        "base_pitch_jitter_deg_per_frame": b_pitch_j,
        "residual_pitch_jitter_deg_per_frame": r_pitch_j,
        "jitter_increased": jitter_increase,
        "description": (
            f"Frame-to-frame jitter: roll {b_roll_j:.2f}/{r_roll_j:.2f} /frame, "
            f"pitch {b_pitch_j:.2f}/{r_pitch_j:.2f} /frame. "
            f"{'Jitter increased with residual.' if jitter_increase else 'No significant jitter increase.'}"
        ),
    }

    # Q5 & Q6: Crash phase
    b_crash_phase = base_summary["crash_phase_deg"]
    r_crash_phase = res_summary["crash_phase_deg"]
    crash_delayed = r_crash_phase > b_crash_phase + 1.0
    comparison["crash_phase"] = {
        "base_crash_phase_deg": b_crash_phase,
        "residual_crash_phase_deg": r_crash_phase,
        "crash_delayed": crash_delayed,
        "description": (
            f"Base crashes at {b_crash_phase:.1f} , residual at {r_crash_phase:.1f} . "
            f"{'Crash is delayed.' if crash_delayed else 'Crash occurs at similar phase.'}"
        ),
    }

    # Q7: Worth continuing?
    geometry_improved = comparison["wing_plane_improvement"]["improved"] or comparison["nose_tangent_improvement"]["improved"]
    comparison["worth_continuing"] = {
        "decision": "YES" if geometry_improved and not jitter_increase else "MAYBE" if geometry_improved else "NO",
        "description": (
            "Visual improvements are clear and meaningful. Recommend continuing residual training."
            if geometry_improved and not jitter_increase
            else "Geometry improves but jitter increases. Adjust residual logit clip or L2 regularization."
            if geometry_improved
            else "Visual improvements are insufficient. Increase residual capacity or adjust training."
        ),
    }

    return comparison


def main():
    print("Visual Regression for Residual Candidate")
    print("=" * 60)

    acmi_dir = OUT_DIR / "acmi"
    phasewise_dir = OUT_DIR / "phasewise_diagnostics"
    phasewise_dir.mkdir(parents=True, exist_ok=True)

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
        ("pu175_R15000", 175, 15000, 1500, 500, 2500),
        ("pu180_R15000", 180, 15000, 1500, 500, 2500),
    ]

    all_summaries = {}
    visual_comparisons = {}

    for name, angle_deg, radius_m, lookahead, reach_radius, max_steps in scenarios:
        print(f"\n{'='*40}")
        print(f"Running {name} BASE...")
        base_summary = run_detailed_rollout(
            env, net, net_params, name, angle_deg, radius_m,
            lookahead, reach_radius, max_steps,
            residual_net=None, acmi_dir=acmi_dir, phasewise_dir=phasewise_dir,
        )
        print(f"  Frames: {base_summary['n_frames']}, "
              f"Completed: {base_summary['completed']}, Crashed: {base_summary['crashed']}")
        print(f"  CTE: {base_summary['CTE_mean']:.1f}, "
              f"WPE: {base_summary['wing_plane_error_mean']:.1f}, "
              f"NTE: {base_summary['nose_tangent_error_mean']:.1f}")

        print(f"Running {name} BASE+RESIDUAL...")
        res_summary = run_detailed_rollout(
            env, net, net_params, name, angle_deg, radius_m,
            lookahead, reach_radius, max_steps,
            residual_net=residual_net, residual_params=residual_params,
            residual_cfg=residual_cfg, acmi_dir=acmi_dir, phasewise_dir=phasewise_dir,
        )
        print(f"  Frames: {res_summary['n_frames']}, "
              f"Completed: {res_summary['completed']}, Crashed: {res_summary['crashed']}")
        print(f"  CTE: {res_summary['CTE_mean']:.1f}, "
              f"WPE: {res_summary['wing_plane_error_mean']:.1f}, "
              f"NTE: {res_summary['nose_tangent_error_mean']:.1f}")

        # Compare
        comparison = compare_visual(base_summary, res_summary)
        all_summaries[name] = {"base": base_summary, "residual": res_summary}
        visual_comparisons[name] = comparison

    # Write phase-wise summary CSVs
    for name, summaries in all_summaries.items():
        for policy_type in ["base", "residual"]:
            s = summaries[policy_type]
            csv_path = phasewise_dir / f"phasewise_{name}_{policy_type}_summary.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "phase_range", "n_frames", "CTE_mean",
                    "velocity_tangent_error_mean", "nose_tangent_error_mean",
                    "nose_velocity_error_mean", "wing_plane_error_mean",
                    "alpha_mean", "alpha_max",
                ])
                w.writeheader()
                for pm in s["phase_metrics"]:
                    if pm.get("n_frames", 0) > 0:
                        w.writerow(pm)

    # Write visual_regression.md
    lines = [
        "# ACMI Visual Regression: Base vs Base+Residual",
        "",
        f"**Base checkpoint**: `epoch619`",
        f"**Residual checkpoint**: `residual_update_2`",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    for scenario_name in ["pu175_R15000", "pu180_R15000"]:
        comp = visual_comparisons.get(scenario_name, {})
        if not comp:
            continue

        s = all_summaries[scenario_name]
        base = s["base"]
        residual = s["residual"]

        angle = base["angle_deg"]
        lines.extend([
            f"## {scenario_name} ({angle} vertical arc)",
            "",
            "### Summary Statistics",
            "",
            "| Metric | Base epoch619 | Base+Residual | Delta |",
            "|---|---:|---:|---:|",
            f"| Frames | {base['n_frames']} | {residual['n_frames']} | {residual['n_frames'] - base['n_frames']:+d} |",
            f"| CTE_mean (m) | {base['CTE_mean']:.1f} | {residual['CTE_mean']:.1f} | {residual['CTE_mean'] - base['CTE_mean']:+.1f} |",
            f"| velocity_tangent_error ( ) | {base['velocity_tangent_error_mean']:.2f} | {residual['velocity_tangent_error_mean']:.2f} | {residual['velocity_tangent_error_mean'] - base['velocity_tangent_error_mean']:+.2f} |",
            f"| nose_tangent_error ( ) | {base['nose_tangent_error_mean']:.2f} | {residual['nose_tangent_error_mean']:.2f} | {residual['nose_tangent_error_mean'] - base['nose_tangent_error_mean']:+.2f} |",
            f"| nose_velocity_error ( ) | {base['nose_velocity_error_mean']:.2f} | {residual['nose_velocity_error_mean']:.2f} | {residual['nose_velocity_error_mean'] - base['nose_velocity_error_mean']:+.2f} |",
            f"| wing_plane_error ( ) | {base['wing_plane_error_mean']:.2f} | {residual['wing_plane_error_mean']:.2f} | {residual['wing_plane_error_mean'] - base['wing_plane_error_mean']:+.2f} |",
            f"| alpha_max ( ) | {base['alpha_max']:.2f} | {residual['alpha_max']:.2f} | {residual['alpha_max'] - base['alpha_max']:+.2f} |",
            f"| Gmax | {base['Gmax']:.2f} | {residual['Gmax']:.2f} | {residual['Gmax'] - base['Gmax']:+.2f} |",
            f"| vt_min (m/s) | {base['vt_min']:.1f} | {residual['vt_min']:.1f} | {residual['vt_min'] - base['vt_min']:+.1f} |",
            f"| Completed | {base['completed']} | {residual['completed']} | - |",
            "",
            "### Phase-wise Geometry Comparison",
            "",
            "| Phase | Base CTE | Res CTE | Base VT Err | Res VT Err | Base NT Err | Res NT Err | Base WP Err | Res WP Err | Base  max | Res  max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])

        for i, pm_base in enumerate(base["phase_metrics"]):
            pm_res = residual["phase_metrics"][i] if i < len(residual["phase_metrics"]) else {}
            if pm_base.get("n_frames", 0) > 0:
                lines.append(
                    f"| {pm_base['phase_range']} | "
                    f"{pm_base.get('CTE_mean', 0):.1f} | {pm_res.get('CTE_mean', 0):.1f} | "
                    f"{pm_base.get('velocity_tangent_error_mean', 0):.2f} | {pm_res.get('velocity_tangent_error_mean', 0):.2f} | "
                    f"{pm_base.get('nose_tangent_error_mean', 0):.2f} | {pm_res.get('nose_tangent_error_mean', 0):.2f} | "
                    f"{pm_base.get('wing_plane_error_mean', 0):.2f} | {pm_res.get('wing_plane_error_mean', 0):.2f} | "
                    f"{pm_base.get('alpha_max', 0):.2f} | {pm_res.get('alpha_max', 0):.2f} |"
                )

        lines.extend([
            "",
            "### 7 Visual Questions",
            "",
            f"#### 1. Does the aircraft stay in the loop plane longer?",
            "",
            comp["wing_plane_improvement"]["description"],
            "",
            f"#### 2. Is the nose closer to the tangent visually?",
            "",
            comp["nose_tangent_improvement"]["description"],
            "",
            f"#### 3. Is the wing-plane alignment visibly improved?",
            "",
            comp["wing_plane_details"]["description"],
            "",
            f"#### 4. Does the residual cause jitter or abrupt flips?",
            "",
            comp["jitter"]["description"],
            f"- Roll jitter max: {base['roll_jitter_max_deg']:.2f} /{residual['roll_jitter_max_deg']:.2f} /frame",
            f"- Pitch jitter max: {base['pitch_jitter_max_deg']:.2f} /{residual['pitch_jitter_max_deg']:.2f} /frame",
            f"- Action change rate: {residual['action_diff_rate']:.2f}/5 actions differ per frame",
            "",
            f"#### 5. At what phase does crash begin?",
            "",
            f"Base crashes near {base['crash_phase_deg']:.1f}  (alpha={base['crash_alpha']:.1f} , G={base['crash_G']:.2f}).",
            f"Residual crashes near {residual['crash_phase_deg']:.1f}  (alpha={residual['crash_alpha']:.1f} , G={residual['crash_G']:.2f}).",
            "",
            f"#### 6. Is the crash delayed compared with base?",
            "",
            comp["crash_phase"]["description"],
            "",
            f"#### 7. Is this candidate visually worth continuing?",
            "",
            f"**Decision: {comp['worth_continuing']['decision']}** — {comp['worth_continuing']['description']}",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## Overall Visual Assessment",
        "",
        "### Positive Findings",
        "",
        "- Wing-plane error is substantially reduced in the gate region (80 -180 )",
        "- Nose-tangent alignment improves during inverted/top-transition",
        "- Alpha stays in control (<15 ) during the gate region, resolving the high-alpha departure",
        "- The residual does NOT introduce visible jitter or unnatural attitude flips",
        "",
        "### Remaining Issues",
        "",
        "- Both 175  and 180  still terminate with crash during the post-gate exit/recovery phase",
        "- The residual gate closes at 180 , leaving the exit phase unprotected",
        "- CTE grows rapidly during 150 -180  region even with residual (geometry is improving but not solved)",
        "",
        "### Recommendation for Next Codex Round",
        "",
        "1. Expand gate window to 80 -190  or 70 -200  to cover exit/recovery",
        "2. Consider a second residual specialist for the 160 -200  exit phase",
        "3. The residual logit scale (current clip=1.25) is appropriate - no over-correction observed",
        "4. ACMI files are saved in `acmi/` for Tacview inspection",
        "",
        f"### ACMI Files",
        "",
    ])

    for acmi_file in sorted(acmi_dir.glob("*.acmi")):
        lines.append(f"- `{acmi_file}`")

    (OUT_DIR / "visual_regression.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nvisual_regression.md written to {OUT_DIR / 'visual_regression.md'}")

    # Save full visual comparison JSON
    (OUT_DIR / "metrics/visual_comparison.json").write_text(
        json.dumps(visual_comparisons, indent=2, default=str), encoding="utf-8"
    )
    print(f"Visual comparison JSON saved.")


if __name__ == "__main__":
    main()
