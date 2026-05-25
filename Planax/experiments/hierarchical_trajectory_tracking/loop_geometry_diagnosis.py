"""
Comprehensive loop geometry/attitude consistency diagnosis.
Tests 150°/180° vertical arcs with old (roll=0) and new (loop-plane roll) targets.

Usage:
    python experiments/hierarchical_trajectory_tracking/loop_geometry_diagnosis.py
"""

import os, sys, json, csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.3'

# Project root is 3 levels up from this file
_px = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _px)

import jax, jax.numpy as jnp, numpy as np
import orbax.checkpoint as ocp
from datetime import datetime

from experiments.hierarchical_trajectory_tracking.render_ablation_tests import (
    ScannedRNN, ActorCriticRNN, NET_CFG, SEED,
)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import (
    vertical_pullup_arc,
)
from experiments.hierarchical_trajectory_tracking.planner import (
    PurePursuitPlanner, PlannerConfig,
)
from experiments.hierarchical_trajectory_tracking.path_utils import (
    compute_true_cte, arc_length,
)
from experiments.hierarchical_trajectory_tracking.loop_attitude_target import (
    loop_plane_rotation_matrix, rotation_matrix_to_quaternion, quaternion_to_euler,
)
from experiments.hierarchical_trajectory_tracking.export_acmi import write_acmi
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as Env,
    Heading_Pitch_V_TaskParams as Params,
    _quat_from_euler_nb as qe,
    _quat_conj as qc,
    _quat_mul,
    _quat_normalize,
)

CKPT = os.path.join(
    _px,
    'results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619',
)


def _f(x):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


# ═══════════════════════ Quaternion / geometry helpers ═══════════════════════


def _quat_conj_np(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotate_body_to_ned(q_bn, v_body):
    q_nb = _quat_conj_np(q_bn)
    p = np.array([0.0, v_body[0], v_body[1], v_body[2]])
    qpq = _quat_mul_np(_quat_mul_np(q_nb, p), _quat_conj_np(q_nb))
    return qpq[1:]


def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    dot = np.clip(dot / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def ned_to_neu(v_ned):
    return np.array([v_ned[0], v_ned[1], -v_ned[2]])


def compute_loop_reference(wps, idx, look_ahead=3):
    n = len(wps)
    i0 = max(0, idx - look_ahead)
    i1 = min(n - 1, idx + look_ahead)
    if i1 > i0:
        t = wps[i1] - wps[i0]
    else:
        t = wps[min(idx + 1, n - 1)] - wps[max(idx - 1, 0)]
    t_ref = t / (np.linalg.norm(t) + 1e-12)

    if n >= 3:
        neighbours = wps[max(0, idx - 5):min(n, idx + 5)]
        if len(neighbours) >= 3:
            centroid = neighbours.mean(axis=0)
            _, _, vh = np.linalg.svd(neighbours - centroid)
            n_loop = vh[2]
            if n_loop[1] < 0:
                n_loop = -n_loop
        else:
            n_loop = np.array([0.0, 1.0, 0.0])
    else:
        n_loop = np.array([0.0, 1.0, 0.0])

    return t_ref, n_loop


def get_loop_roll(theta_deg, init_yaw=0.0):
    R = loop_plane_rotation_matrix(np.radians(theta_deg), init_yaw, 1)
    q = rotation_matrix_to_quaternion(R)
    r, p, h = quaternion_to_euler(q)
    return r


# ═══════════════════════ Run single test ═══════════════════════


def run_test(name, ang, rad, la, rr, mx, use_loop_roll, rng, net, net_params, env, out_root):
    wps, meta = vertical_pullup_arc(
        0, 0, 5000, 0.0, radius=rad, arc_angle_deg=ang,
        n_points=max(80, int(ang * 2 / 3)),
    )
    total_arc = meta['total_length_m']
    cfg = PlannerConfig(
        lookahead_dist=la, reach_radius=rr, blend_steps=250, target_vt=250.0,
    )
    planner = PurePursuitPlanner(cfg)

    rng2, rk = jax.random.split(rng)
    obs_dict, state = env.reset(rk, Params())
    q_nb = qe(0.0, 0.0, 0.0)
    q_bn = qc(q_nb)
    state = state.replace(
        plane_state=state.plane_state.replace(
            yaw=jnp.array([0.0]),
            q0=jnp.array([q_bn[0]]), q1=jnp.array([q_bn[1]]),
            q2=jnp.array([q_bn[2]]), q3=jnp.array([q_bn[3]]),
        ),
        target_heading=jnp.array([0.0]),
    )
    planner.reset(wps, 0.0, 0.0, 0.0, 250.0)

    hstate = ScannedRNN.initialize_carry(1, NET_CFG['GRU_HIDDEN_DIM'])
    df = jnp.zeros((1,))

    rec = {
        't': [], 'n': [], 'e': [], 'a': [], 'vt': [],
        'roll': [], 'pitch': [], 'yaw': [],
        't_roll': [], 't_pitch': [], 't_hdg': [],
        'alpha': [], 'beta': [], 'G': [], 'cte': [],
        'q0': [], 'q1': [], 'q2': [], 'q3': [],
        'wp_idx': [],
    }
    crashed = False

    for step in range(mx):
        ps = state.plane_state
        no = _f(ps.north); ea = _f(ps.east); al = _f(ps.altitude)
        vt = _f(ps.vt); ro = _f(ps.roll); pi = _f(ps.pitch); ya = _f(ps.yaw)
        alph = _f(ps.alpha); bet = _f(ps.beta)
        ax = _f(ps.ax); ay = _f(ps.ay); az = _f(ps.az)

        result = planner.step(no, ea, al, ya, pi, ro, vt)
        th, tp, tr, tv = (
            result['target_heading'], result['target_pitch'],
            result['target_roll'], result['target_vt'],
        )

        if use_loop_roll:
            path_s = planner.path_progress
            loop_theta_deg = (path_s / total_arc) * ang if total_arc > 0 else 0.0
            loop_theta_deg = np.clip(loop_theta_deg, 0, ang)
            loop_roll = get_loop_roll(loop_theta_deg)
            blend = min(1.0, step / 250.0)
            tr = float(np.arctan2(
                np.sin(ro + blend * (loop_roll - ro)),
                np.cos(ro + blend * (loop_roll - ro)),
            ))

        state = state.replace(
            target_heading=jnp.array([th]),
            target_pitch=jnp.array([tp]),
            target_roll=jnp.array([tr]),
            target_vt=jnp.array([float(tv)], dtype=jnp.float32),
        )

        obs_in = env._get_obs(state, Params())[env.agents[0]][None, None, :]
        hstate, pi_out, _ = net.apply(net_params, hstate, (obs_in, df[None, :]))
        acts = [int(p.mode()[0, 0]) for p in pi_out]

        rng2, sk = jax.random.split(rng2)
        obs2, state, rew, done, info = env.step(
            sk, state, {env.agents[0]: jnp.array(acts)}, Params(),
        )
        df = jnp.array([float(done[env.agents[0]])])

        wp_idx = result['path_ctx']['wp_idx']
        rec['t'].append(step * 0.2)
        rec['n'].append(no); rec['e'].append(ea); rec['a'].append(al)
        rec['vt'].append(vt)
        rec['roll'].append(np.degrees(ro)); rec['pitch'].append(np.degrees(pi))
        rec['yaw'].append(np.degrees(ya))
        rec['t_roll'].append(np.degrees(tr)); rec['t_pitch'].append(np.degrees(tp))
        rec['t_hdg'].append(np.degrees(th))
        rec['alpha'].append(np.degrees(alph)); rec['beta'].append(np.degrees(bet))
        rec['G'].append(float(np.sqrt(ax ** 2 + ay ** 2 + az ** 2)))
        rec['cte'].append(compute_true_cte(
            np.array([no, ea, al]), wps, wp_idx, 10,
        ))
        rec['q0'].append(_f(ps.q0)); rec['q1'].append(_f(ps.q1))
        rec['q2'].append(_f(ps.q2)); rec['q3'].append(_f(ps.q3))
        rec['wp_idx'].append(wp_idx)

        if bool(done[env.agents[0]]):
            crashed = True
            break
        if planner.is_done():
            break

    n = len(rec['t']); ok = planner.is_done() and not crashed

    # ═══ Compute per-frame geometry metrics ═══
    geo = {
        'velocity_tangent_error': [], 'nose_tangent_error': [],
        'nose_velocity_error': [], 'wing_plane_error': [],
        'belly_error': [], 'env_alpha_deg': [], 'tacview_aoa_approx': [],
        'fpa_deg': [],
    }

    for i in range(n):
        q_bn_i = np.array([rec['q0'][i], rec['q1'][i], rec['q2'][i], rec['q3'][i]])
        q_bn_i = q_bn_i / (np.linalg.norm(q_bn_i) + 1e-12)

        x_body_ned = rotate_body_to_ned(q_bn_i, np.array([1.0, 0.0, 0.0]))
        y_body_ned = rotate_body_to_ned(q_bn_i, np.array([0.0, 1.0, 0.0]))
        z_body_ned = rotate_body_to_ned(q_bn_i, np.array([0.0, 0.0, 1.0]))

        x_body_neu = ned_to_neu(x_body_ned)
        y_body_neu = ned_to_neu(y_body_ned)
        z_body_neu = ned_to_neu(z_body_ned)

        vt_i = rec['vt'][i]
        alpha_i = np.radians(rec['alpha'][i])
        beta_i = np.radians(rec['beta'][i])
        ca, sa = np.cos(alpha_i), np.sin(alpha_i)
        cb, sb = np.cos(beta_i), np.sin(beta_i)

        u_body = vt_i * ca * cb
        v_body = vt_i * sb
        w_body = vt_i * sa * cb

        v_ned = rotate_body_to_ned(q_bn_i, np.array([u_body, v_body, w_body]))
        v_neu = ned_to_neu(v_ned)
        v_hat_neu = v_neu / (np.linalg.norm(v_neu) + 1e-12)

        t_ref_neu, n_loop_neu = compute_loop_reference(wps, rec['wp_idx'][i])

        geo['velocity_tangent_error'].append(angle_between(v_hat_neu, t_ref_neu))
        geo['nose_tangent_error'].append(angle_between(x_body_neu, t_ref_neu))
        geo['nose_velocity_error'].append(angle_between(x_body_neu, v_hat_neu))
        geo['wing_plane_error'].append(angle_between(y_body_neu, n_loop_neu))

        z_body_expected = np.cross(t_ref_neu, n_loop_neu)
        z_body_expected = z_body_expected / (np.linalg.norm(z_body_expected) + 1e-12)
        geo['belly_error'].append(angle_between(z_body_neu, z_body_expected))

        geo['env_alpha_deg'].append(rec['alpha'][i])

        fpa = np.degrees(np.arctan2(v_neu[2], np.sqrt(v_neu[0] ** 2 + v_neu[1] ** 2) + 1e-12))
        geo['fpa_deg'].append(fpa)
        geo['tacview_aoa_approx'].append(rec['pitch'][i] - fpa)

    # ═══ Segment analysis ═══
    segments = [(0, 0.44), (0.44, 0.56), (0.56, 0.72), (0.72, 0.88), (0.88, 1.0)]
    seg_names = ['0-80°', '80-100°', '100-130°', '130-160°', '160-180°']
    if ang <= 150:
        segments = segments[:4]
        seg_names = seg_names[:4]

    seg_metrics = {}
    for seg_name, (lo, hi) in zip(seg_names, segments):
        indices = []
        for i in range(n):
            frac = i / max(n - 1, 1)
            if lo <= frac < hi or (hi >= 0.99 and frac >= lo):
                indices.append(i)
        if indices:
            seg_metrics[seg_name] = {
                'velocity_tangent_error': np.mean([geo['velocity_tangent_error'][i] for i in indices]),
                'nose_tangent_error': np.mean([geo['nose_tangent_error'][i] for i in indices]),
                'nose_velocity_error': np.mean([geo['nose_velocity_error'][i] for i in indices]),
                'wing_plane_error': np.mean([geo['wing_plane_error'][i] for i in indices]),
                'belly_error': np.mean([geo['belly_error'][i] for i in indices]),
                'n_frames': len(indices),
            }

    # ═══ Summary metrics ═══
    ca_arr = np.array(rec['cte'])
    va_arr = np.array(rec['vt'])
    aa_arr = np.array(rec['alpha'])
    ga_arr = np.array(rec['G'])

    summary = {
        'name': name, 'angle_deg': ang, 'radius_m': rad,
        'use_loop_roll': use_loop_roll,
        'completed': bool(ok), 'steps': n,
        'CTE_mean': float(ca_arr.mean()), 'CTE_p50': float(np.percentile(ca_arr, 50)),
        'CTE_p90': float(np.percentile(ca_arr, 90)), 'CTE_max': float(ca_arr.max()),
        'vt_min': float(va_arr.min()), 'vt_mean': float(va_arr.mean()),
        'Gmax': float(ga_arr.max()), 'Gmean': float(ga_arr.mean()),
        'env_alpha_min': float(aa_arr.min()), 'env_alpha_max': float(aa_arr.max()),
        'env_alpha_mean': float(aa_arr.mean()),
        'velocity_tangent_error_mean': float(np.mean(geo['velocity_tangent_error'])),
        'velocity_tangent_error_p90': float(np.percentile(geo['velocity_tangent_error'], 90)),
        'nose_tangent_error_mean': float(np.mean(geo['nose_tangent_error'])),
        'nose_tangent_error_p90': float(np.percentile(geo['nose_tangent_error'], 90)),
        'nose_velocity_error_mean': float(np.mean(geo['nose_velocity_error'])),
        'nose_velocity_error_p90': float(np.percentile(geo['nose_velocity_error'], 90)),
        'wing_plane_error_mean': float(np.mean(geo['wing_plane_error'])),
        'wing_plane_error_p90': float(np.percentile(geo['wing_plane_error'], 90)),
        'belly_error_mean': float(np.mean(geo['belly_error'])),
        'tacview_aoa_min': float(np.min(geo['tacview_aoa_approx'])),
        'tacview_aoa_max': float(np.max(geo['tacview_aoa_approx'])),
        'target_roll_min': float(np.min(rec['t_roll'])),
        'target_roll_max': float(np.max(rec['t_roll'])),
        'actual_roll_min': float(np.min(rec['roll'])),
        'actual_roll_max': float(np.max(rec['roll'])),
        'actual_roll_mean': float(np.mean(rec['roll'])),
        'termination': 'crash' if crashed else ('ok' if ok else 'timeout'),
    }

    # ═══ Loop-quality grade ═══
    cm = summary['CTE_mean']; c90 = summary['CTE_p90']; cmax = summary['CTE_max']
    g = summary['Gmax']; v = summary['vt_min']
    vte = summary['velocity_tangent_error_mean']
    nte = summary['nose_tangent_error_mean']
    nve = summary['nose_velocity_error_mean']
    wpe = summary['wing_plane_error_mean']

    if not ok:
        grade = 'Fail'
    elif (cm < 100 and c90 < 300 and cmax < 800 and g < 9 and v >= 190
          and vte < 15 and nte < 15 and nve < 15 and wpe < 15):
        grade = 'A'
    elif (cm < 500 and c90 < 1200 and g < 10 and v >= 175
          and vte < 30 and nte < 30):
        grade = 'B'
    elif ok:
        grade = 'C'
    else:
        grade = 'Fail'
    summary['loop_grade'] = grade

    # ═══ Save data ═══
    np.savez(
        os.path.join(out_root, 'rollouts', f'{name}.npz'),
        waypoints=wps, total_arc=total_arc, **rec,
    )
    with open(os.path.join(out_root, 'metrics', f'{name}_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ═══ ACMI via reusable exporter ═══
    acmi_path = os.path.join(out_root, 'acmi', f'{name}.acmi')
    write_acmi(acmi_path, wps, rec)

    return summary, geo, seg_metrics, rec


# ═══════════════════════ Main ═══════════════════════


def main(checkpoint_path=None, output_root=None, tests=None):
    if checkpoint_path is None:
        checkpoint_path = CKPT

    env = Env(Params())
    net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    rng = jax.random.PRNGKey(SEED)
    obs_shape = env.observation_space(env.agents[0], Params()).shape
    h0 = ScannedRNN.initialize_carry(1, NET_CFG['GRU_HIDDEN_DIM'])
    net_params = net.init(
        rng, h0, (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1))),
    )
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(checkpoint_path, args=ocp.args.StandardRestore())
    net_params = ckpt['params']

    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_root is None:
        out_root = os.path.join(_px, 'results/loop_geometry_diagnosis', tag)
    else:
        out_root = os.path.join(output_root, tag)
    for sub in ['rollouts', 'metrics', 'figures', 'acmi']:
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    print(f"Output: {out_root}")

    if tests is None:
        tests = [
            ('pu150_old', 150, 12000, 1200, 500, 2000, False),
            ('pu150_new', 150, 12000, 1200, 500, 2000, True),
            ('pu180_old', 180, 15000, 1500, 500, 2500, False),
            ('pu180_new', 180, 15000, 1500, 500, 2500, True),
        ]

    all_summaries = []
    all_geos = {}
    all_segs = {}
    all_recs = {}

    for name, ang, rad, la, rr, mx, use_lr in tests:
        print(f"\n{'=' * 70}\nRunning: {name} (angle={ang}°, R={rad}m, "
              f"{'loop-roll' if use_lr else 'roll=0'})\n{'=' * 70}")
        s, geo, seg, rec = run_test(
            name, ang, rad, la, rr, mx, use_lr, rng, net, net_params, env, out_root,
        )
        all_summaries.append(s)
        all_geos[name] = geo
        all_segs[name] = seg
        all_recs[name] = rec

        print(f"  Result: {s['loop_grade']} | steps={s['steps']} | {s['termination']}")
        print(f"  CTE: mean={s['CTE_mean']:.0f} p90={s['CTE_p90']:.0f} max={s['CTE_max']:.0f}")
        print(f"  Geometry: vel_tang={s['velocity_tangent_error_mean']:.1f}° "
              f"nose_tang={s['nose_tangent_error_mean']:.1f}° "
              f"nose_vel={s['nose_velocity_error_mean']:.1f}° "
              f"wing_plane={s['wing_plane_error_mean']:.1f}°")
        print(f"  Alpha: env=[{s['env_alpha_min']:.1f},{s['env_alpha_max']:.1f}] "
              f"tacview_aoa=[{s['tacview_aoa_min']:.1f},{s['tacview_aoa_max']:.1f}]")
        print(f"  Roll: target=[{s['target_roll_min']:.0f},{s['target_roll_max']:.0f}] "
              f"actual=[{s['actual_roll_min']:.0f},{s['actual_roll_max']:.0f}]")

    # ═══ Generate reports ═══
    _generate_reports(all_summaries, all_geos, all_segs, out_root, tests)

    print(f"\n{'=' * 70}")
    print(f"ALL OUTPUTS: {out_root}")
    print(f"  geometry_metrics.csv")
    print(f"  target_frame_check.md")
    print(f"  alpha_check.md")
    print(f"  actual_vs_target_report.md")
    print(f"  loop_quality_grades.csv")
    print(f"  old_vs_new_comparison.md")
    print(f"  codex_recommendation.md")
    print(f"  acmi/  rollouts/  metrics/")
    print(f"\nDONE")
    return out_root, all_summaries


def _generate_reports(all_summaries, all_geos, all_segs, out_root, tests):
    """Generate all diagnosis reports."""

    # ── Target frame check ──
    print(f"\n\n{'=' * 70}")
    print("TASK 1: TARGET FRAME CHECK")
    print(f"{'=' * 70}")

    print(f"\n{'theta':>8} {'t_hdg':>8} {'t_pitch':>8} {'t_roll':>8} "
          f"{'body_x·tang':>12} {'body_y·norm':>12} {'body_z_correct':>14}")
    print("-" * 70)

    target_frame_results = []
    for theta_deg in [0, 30, 60, 90, 120, 150, 180]:
        R = loop_plane_rotation_matrix(np.radians(theta_deg), 0.0, 1)
        q = rotation_matrix_to_quaternion(R)
        r, p, h = quaternion_to_euler(q)

        body_x = R[:, 0]
        body_y = R[:, 1]
        body_z = R[:, 2]

        ct, st = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
        f0 = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, f0)
        tangent_ref = ct * f0 + st * np.cross(right, f0)
        tangent_ref = tangent_ref / np.linalg.norm(tangent_ref)
        n_loop = right / np.linalg.norm(right)

        dot_bx_tang = np.dot(body_x, tangent_ref)
        dot_by_norm = np.dot(body_y, n_loop)
        z_expected = np.cross(tangent_ref, n_loop)
        dot_bz_exp = np.dot(body_z, z_expected)

        print(f"{theta_deg:8.0f} {np.degrees(h):8.1f} {np.degrees(p):8.1f} {np.degrees(r):8.1f} "
              f"{dot_bx_tang:12.6f} {dot_by_norm:12.6f} {dot_bz_exp:14.6f}")

        target_frame_results.append({
            'theta_deg': theta_deg,
            't_hdg': np.degrees(h), 't_pitch': np.degrees(p), 't_roll': np.degrees(r),
            'body_x_dot_tangent': dot_bx_tang,
            'body_y_dot_normal': dot_by_norm,
            'body_z_dot_expected': dot_bz_exp,
        })

    # ── Segment geometry ──
    print(f"\n\n{'=' * 70}")
    print("TASK 2 & 3: ACTUAL VS TARGET & SEGMENT GEOMETRY")
    print(f"{'=' * 70}")

    for name in [t[0] for t in tests]:
        s = [x for x in all_summaries if x['name'] == name][0]
        seg = all_segs[name]
        print(f"\n── {name} (grade={s['loop_grade']}) ──")
        print(f"  {'Segment':<16} {'n':>4} {'vel_tang':>9} {'nose_tang':>9} "
              f"{'nose_vel':>9} {'wing_plane':>10} {'belly':>8}")
        print(f"  {'─' * 16} {'─' * 4} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 10} {'─' * 8}")
        for seg_name, sm in seg.items():
            print(f"  {seg_name:<16} {sm['n_frames']:4d} {sm['velocity_tangent_error']:9.1f} "
                  f"{sm['nose_tangent_error']:9.1f} {sm['nose_velocity_error']:9.1f} "
                  f"{sm['wing_plane_error']:10.1f} {sm['belly_error']:8.1f}")

    # ── AOA/alpha cross-check ──
    print(f"\n\n{'=' * 70}")
    print("TASK 4: AOA/ALPHA CROSS-CHECK")
    print(f"{'=' * 70}")

    alpha_check = []
    for name in [t[0] for t in tests]:
        s = [x for x in all_summaries if x['name'] == name][0]
        geo = all_geos[name]
        env_a_range = f"[{s['env_alpha_min']:.1f}, {s['env_alpha_max']:.1f}]"
        geo_a_range = f"[{np.min(geo['nose_velocity_error']):.1f}, {np.max(geo['nose_velocity_error']):.1f}]"
        tacview_a_range = f"[{s['tacview_aoa_min']:.1f}, {s['tacview_aoa_max']:.1f}]"

        env_a = np.array(geo['env_alpha_deg'])
        nve = np.array(geo['nose_velocity_error'])
        corr = np.corrcoef(env_a, nve)[0, 1] if len(env_a) > 1 else 0
        match = "YES" if abs(corr) > 0.7 else "PARTIAL" if abs(corr) > 0.3 else "NO"

        print(f"{name:<20} {env_a_range:>20} {geo_a_range:>20} {tacview_a_range:>20} {match:>10} (r={corr:.3f})")
        alpha_check.append({
            'name': name, 'env_alpha_range': env_a_range,
            'geo_nose_vel_range': geo_a_range, 'tacview_aoa_range': tacview_a_range,
            'correlation': float(corr), 'match': match,
        })

    # ── Loop-quality grades ──
    print(f"\n\n{'=' * 70}")
    print("TASK 5: LOOP-QUALITY GRADES (NEW CRITERIA)")
    print(f"{'=' * 70}")
    print(f"{'Name':<20} {'CTE':>6} {'v_tang':>8} {'n_tang':>8} {'n_vel':>8} "
          f"{'wing_p':>8} {'Gmax':>6} {'vt_min':>6} {'Grade':>6}")
    print("-" * 90)
    for s in all_summaries:
        print(f"{s['name']:<20} {s['CTE_mean']:6.0f} {s['velocity_tangent_error_mean']:8.1f} "
              f"{s['nose_tangent_error_mean']:8.1f} {s['nose_velocity_error_mean']:8.1f} "
              f"{s['wing_plane_error_mean']:8.1f} {s['Gmax']:6.1f} {s['vt_min']:6.0f} "
              f"{s['loop_grade']:>6}")

    # ── Old vs New comparison ──
    print(f"\n\n{'=' * 70}")
    print("TASK 6: OLD vs NEW COMPARISON")
    print(f"{'=' * 70}")

    comparison_fields = [
        'CTE_mean', 'velocity_tangent_error_mean', 'nose_tangent_error_mean',
        'nose_velocity_error_mean', 'wing_plane_error_mean',
        'env_alpha_min', 'env_alpha_max', 'tacview_aoa_min', 'tacview_aoa_max',
        'target_roll_min', 'target_roll_max', 'actual_roll_min', 'actual_roll_max',
        'Gmax', 'vt_min', 'loop_grade',
    ]

    for ang in [150, 180]:
        print(f"\n── {ang}° Comparison ──")
        old_name = f'pu{ang}_old'; new_name = f'pu{ang}_new'
        old_s = [x for x in all_summaries if x['name'] == old_name][0]
        new_s = [x for x in all_summaries if x['name'] == new_name][0]
        for field in comparison_fields:
            ov = old_s.get(field, 'N/A'); nv = new_s.get(field, 'N/A')
            if isinstance(ov, float):
                print(f"{field:<35} {ov:20.1f} {nv:20.1f}")
            else:
                print(f"{field:<35} {str(ov):>20} {str(nv):>20}")

    # ═══ Write report files ═══
    # Master CSV
    csv_path = os.path.join(out_root, 'geometry_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        w.writeheader()
        w.writerows(all_summaries)
    print(f"\nCSV: {csv_path}")

    # Target frame check
    tf_path = os.path.join(out_root, 'target_frame_check.md')
    with open(tf_path, 'w') as f:
        f.write("# Target Frame Check\n\n")
        f.write("Verifies that `loop_attitude_target.py` produces correct body-frame alignment.\n\n")
        f.write("| theta | t_hdg | t_pitch | t_roll | body_x·tang | body_y·norm | body_z·expected |\n")
        f.write("|-------|-------|---------|--------|-------------|-------------|------------------|\n")
        for r in target_frame_results:
            f.write(f"| {r['theta_deg']:.0f} | {r['t_hdg']:.1f} | {r['t_pitch']:.1f} | {r['t_roll']:.1f} | "
                    f"{r['body_x_dot_tangent']:.6f} | {r['body_y_dot_normal']:.6f} | "
                    f"{r['body_z_dot_expected']:.6f} |\n")
        f.write("\n## Verdict\n\n")
        max_bx_err = max(abs(1 - r['body_x_dot_tangent']) for r in target_frame_results)
        max_by_err = max(abs(1 - r['body_y_dot_normal']) for r in target_frame_results)
        f.write(f"- body_x alignment with tangent: max error = {max_bx_err:.2e}\n")
        f.write(f"- body_y alignment with loop normal: max error = {max_by_err:.2e}\n")
        if max_bx_err < 1e-10 and max_by_err < 1e-10:
            f.write("- **Target frame is CORRECT.** body_x perfectly aligns with loop tangent, body_y with loop plane normal.\n")
        else:
            f.write("- **Target frame has ISSUES.**\n")

    # Alpha check
    ac_path = os.path.join(out_root, 'alpha_check.md')
    with open(ac_path, 'w') as f:
        f.write("# AOA/Alpha Cross-Check\n\n")
        f.write("Compares env alpha (from dynamics), geometric nose-velocity error, and Tacview-approximated AOA.\n\n")
        f.write("| Name | env_alpha | geo_nose_vel_err | tacview_aoa | correlation | match? |\n")
        f.write("|------|-----------|------------------|-------------|-------------|--------|\n")
        for r in alpha_check:
            f.write(f"| {r['name']} | {r['env_alpha_range']} | {r['geo_nose_vel_range']} | "
                    f"{r['tacview_aoa_range']} | {r['correlation']:.3f} | {r['match']} |\n")

    # Actual vs target
    avt_path = os.path.join(out_root, 'actual_vs_target_report.md')
    with open(avt_path, 'w') as f:
        f.write("# Actual vs Target Report\n\n")
        f.write("## Roll tracking\n\n")
        f.write("| Name | t_roll range | actual_roll range | tracking OK? |\n")
        f.write("|------|-------------|-------------------|---------------|\n")
        for s in all_summaries:
            tr = abs(s['target_roll_max'] - s['target_roll_min'])
            ar = abs(s['actual_roll_max'] - s['actual_roll_min'])
            ok = "YES" if abs(tr - ar) < 60 else "PARTIAL" if abs(tr - ar) < 120 else "NO"
            f.write(f"| {s['name']} | [{s['target_roll_min']:.0f},{s['target_roll_max']:.0f}] | "
                    f"[{s['actual_roll_min']:.0f},{s['actual_roll_max']:.0f}] | {ok} |\n")
        f.write("\n## Geometry tracking\n\n")
        f.write("| Name | vel_tang_err | nose_tang_err | nose_vel_err | wing_plane_err | belly_err |\n")
        f.write("|------|-------------|--------------|-------------|----------------|----------|\n")
        for s in all_summaries:
            f.write(f"| {s['name']} | {s['velocity_tangent_error_mean']:.1f}° | "
                    f"{s['nose_tangent_error_mean']:.1f}° | {s['nose_velocity_error_mean']:.1f}° | "
                    f"{s['wing_plane_error_mean']:.1f}° | {s['belly_error_mean']:.1f}° |\n")

    # Loop quality grades CSV
    lq_path = os.path.join(out_root, 'loop_quality_grades.csv')
    with open(lq_path, 'w', newline='') as f:
        grade_fields = [
            'name', 'loop_grade', 'CTE_mean', 'CTE_p90',
            'velocity_tangent_error_mean', 'nose_tangent_error_mean',
            'nose_velocity_error_mean', 'wing_plane_error_mean',
            'belly_error_mean', 'Gmax', 'vt_min', 'completed',
        ]
        w = csv.DictWriter(f, fieldnames=grade_fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_summaries)

    # Old vs new comparison
    old_vs_new_path = os.path.join(out_root, 'old_vs_new_comparison.md')
    with open(old_vs_new_path, 'w') as f:
        f.write("# Old (roll=0) vs New (loop-plane roll) Comparison\n\n")
        for ang in [150, 180]:
            f.write(f"## {ang}° Vertical Arc\n\n")
            f.write("| Metric | OLD (roll=0) | NEW (loop-plane) |\n")
            f.write("|--------|-------------|------------------|\n")
            old_name = f'pu{ang}_old'; new_name = f'pu{ang}_new'
            old_s = [x for x in all_summaries if x['name'] == old_name][0]
            new_s = [x for x in all_summaries if x['name'] == new_name][0]
            for field in comparison_fields:
                ov = old_s.get(field, 'N/A'); nv = new_s.get(field, 'N/A')
                if isinstance(ov, float):
                    f.write(f"| {field} | {ov:.1f} | {nv:.1f} |\n")
                else:
                    f.write(f"| {field} | {ov} | {nv} |\n")
            f.write("\n")

    # Codex recommendation
    cr_path = os.path.join(out_root, 'codex_recommendation.md')
    with open(cr_path, 'w') as f:
        f.write("# Codex Training Recommendation\n\n")
        f.write("## Diagnosis Summary\n\n")
        for ang in [150, 180]:
            new_name = f'pu{ang}_new'
            new_s = [x for x in all_summaries if x['name'] == new_name][0]
            f.write(f"### {ang}° (loop-plane roll target)\n\n")
            f.write(f"- Loop grade: **{new_s['loop_grade']}**\n")
            f.write(f"- CTE mean: {new_s['CTE_mean']:.0f}m\n")
            f.write(f"- Velocity-tangent error: {new_s['velocity_tangent_error_mean']:.1f}°\n")
            f.write(f"- Nose-tangent error: {new_s['nose_tangent_error_mean']:.1f}°\n")
            f.write(f"- Nose-velocity error: {new_s['nose_velocity_error_mean']:.1f}°\n")
            f.write(f"- Wing-plane error: {new_s['wing_plane_error_mean']:.1f}°\n")
            f.write(f"- Belly error: {new_s['belly_error_mean']:.1f}°\n")
            f.write(f"- Env alpha: [{new_s['env_alpha_min']:.1f}, {new_s['env_alpha_max']:.1f}]°\n")
            f.write(f"- Tacview AOA approx: [{new_s['tacview_aoa_min']:.1f}, {new_s['tacview_aoa_max']:.1f}]°\n\n")
        f.write("## Key Questions\n\n")
        f.write("1. Does the new version truly align the nose with the loop tangent?\n")
        f.write("2. Is the new version just a roll flip?\n")
        f.write("3. Is the Tacview AOA anomaly an export issue or real attitude problem?\n")
        f.write("4. Can 150° still be called loop A-grade?\n")
        f.write("5. What is the root cause of 180° failure?\n")
        f.write("6. Train Codex or fix target/export first?\n")
        f.write("\n*(See actual_vs_target_report.md and geometry_metrics.csv for data to answer these.)*\n")


if __name__ == '__main__':
    main()
