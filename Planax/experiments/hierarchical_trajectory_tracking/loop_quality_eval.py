"""
Comprehensive loop-quality evaluation for all vertical arc angles (60°-180°).
Uses loop-plane roll targets and full geometry metrics.
Outputs: loop_quality_summary.csv, loop_quality_report.md, paper_ready_vertical_arc_table.tex

Usage:
    python experiments/hierarchical_trajectory_tracking/loop_quality_eval.py
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
from experiments.hierarchical_trajectory_tracking.path_utils import compute_true_cte
from experiments.hierarchical_trajectory_tracking.loop_attitude_target import (
    loop_plane_rotation_matrix, rotation_matrix_to_quaternion, quaternion_to_euler,
)
from experiments.hierarchical_trajectory_tracking.export_acmi import (
    write_acmi, enu_to_geodetic,
)
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as Env,
    Heading_Pitch_V_TaskParams as Params,
    _quat_from_euler_nb,
    _quat_conj,
    _quat_mul,
)

CKPT = os.path.join(
    _px,
    'results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619',
)


def _f(x):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


# ═══════════════════════ Quaternion / geometry helpers ═══════════════════════


def quat_conj_np(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotate_body_to_ned(q_bn, v_body):
    q_nb = quat_conj_np(q_bn)
    p = np.array([0.0, v_body[0], v_body[1], v_body[2]])
    qpq = quat_mul_np(quat_mul_np(q_nb, p), quat_conj_np(q_nb))
    return qpq[1:]


def ned_to_neu(v_ned):
    return np.array([v_ned[0], v_ned[1], -v_ned[2]])


def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    dot = np.clip(dot / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


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
        nb = wps[max(0, idx - 5):min(n, idx + 5)]
        if len(nb) >= 3:
            centroid = nb.mean(axis=0)
            _, _, vh = np.linalg.svd(nb - centroid)
            n_loop = vh[2]
            if n_loop[1] < 0:
                n_loop = -n_loop
        else:
            n_loop = np.array([0.0, 1.0, 0.0])
    else:
        n_loop = np.array([0.0, 1.0, 0.0])
    return t_ref, n_loop


def quat_error_angle(q_curr_bn, yaw_t, pitch_t, roll_t):
    """Compute angular error (rad) between current and target attitude."""
    cr, sr = np.cos(0.5 * roll_t), np.sin(0.5 * roll_t)
    cp, sp = np.cos(0.5 * pitch_t), np.sin(0.5 * pitch_t)
    cy, sy = np.cos(0.5 * yaw_t), np.sin(0.5 * yaw_t)
    q_tgt_nb = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
    q_tgt_bn = quat_conj_np(q_tgt_nb)
    q_tgt_bn = q_tgt_bn / (np.linalg.norm(q_tgt_bn) + 1e-12)
    q_curr_bn = q_curr_bn / (np.linalg.norm(q_curr_bn) + 1e-12)
    q_err = quat_mul_np(q_tgt_bn, quat_conj_np(q_curr_bn))
    if q_err[0] < 0:
        q_err = -q_err
    w = np.clip(np.abs(q_err[0]), 0.0, 1.0 - 1e-12)
    return float(2.0 * np.arccos(w))


def get_loop_roll(theta_deg, init_yaw=0.0):
    R = loop_plane_rotation_matrix(np.radians(theta_deg), init_yaw, 1)
    q = rotation_matrix_to_quaternion(R)
    r, p, h = quaternion_to_euler(q)
    return r


# ═══════════════════════ Grading ═══════════════════════


def grade_loop(m, deprecated=False):
    """Loop-quality grade with full geometry criteria.

    A: CTE OK + all geometry < 15° + Gmax < 9 + vt ≥ 190
    B: CTE < 500 + geometry < 30° + G < 10 + vt ≥ 175
    C: completed but exceeds B thresholds
    Fail: crashed or timed out
    """
    if deprecated:
        if not m['completed']:
            return 'Fail'
        cm = m['CTE_mean']; c90 = m['CTE_p90']; cmax = m['CTE_max']
        g = m['Gmax']; v = m['vt_min']
        if cm < 100 and c90 < 300 and cmax < 800 and g < 9 and v >= 190:
            return 'A'
        if cm < 500 and c90 < 1200 and g < 10 and v >= 175:
            return 'B'
        if m['completed']:
            return 'C'
        return 'Fail'

    if not m['completed']:
        return 'Fail'

    cm = m['CTE_mean']; c90 = m['CTE_p90']; cmax = m['CTE_max']
    g = m['Gmax']; v = m['vt_min']
    vte = m['velocity_tangent_error_mean']
    nte = m['nose_tangent_error_mean']
    nve = m['nose_velocity_error_mean']
    wpe = m['wing_plane_error_mean']
    qe = m.get('q_error_mean_rad', 999)

    if (cm < 100 and c90 < 300 and cmax < 800
        and g < 9 and v >= 190
        and vte < 15 and nte < 15 and nve < 15 and wpe < 15
        and qe < 0.5):
        return 'A'
    if (cm < 500 and c90 < 1200
        and g < 10 and v >= 175
        and vte < 30 and nte < 30):
        return 'B'
    if m['completed']:
        return 'C'
    return 'Fail'


# ═══════════════════════ Run single test ═══════════════════════


def run_test(name, ang, rad, la, rr, mx, rng, net, net_params, env, out_root):
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

        # Loop-plane roll override
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
            target_heading=jnp.array([th]), target_pitch=jnp.array([tp]),
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
        rec['t'].append(step * 0.2); rec['n'].append(no); rec['e'].append(ea)
        rec['a'].append(al); rec['vt'].append(vt)
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

    # ═══ Per-frame geometry ═══
    geo = {
        'velocity_tangent_error': [], 'nose_tangent_error': [],
        'nose_velocity_error': [], 'wing_plane_error': [],
        'belly_error': [], 'q_error_rad': [], 'roll_tracking_error': [],
    }

    for i in range(n):
        q_bn_i = np.array([rec['q0'][i], rec['q1'][i], rec['q2'][i], rec['q3'][i]])
        q_bn_i = q_bn_i / (np.linalg.norm(q_bn_i) + 1e-12)

        x_body_ned = rotate_body_to_ned(q_bn_i, np.array([1., 0., 0.]))
        y_body_ned = rotate_body_to_ned(q_bn_i, np.array([0., 1., 0.]))
        z_body_ned = rotate_body_to_ned(q_bn_i, np.array([0., 0., 1.]))
        x_body_neu = ned_to_neu(x_body_ned)
        y_body_neu = ned_to_neu(y_body_ned)
        z_body_neu = ned_to_neu(z_body_ned)

        vt_i = rec['vt'][i]; alpha_i = np.radians(rec['alpha'][i])
        beta_i = np.radians(rec['beta'][i])
        ca, sa = np.cos(alpha_i), np.sin(alpha_i)
        cb, sb = np.cos(beta_i), np.sin(beta_i)
        u_body = vt_i * ca * cb; v_body = vt_i * sb; w_body = vt_i * sa * cb
        v_ned = rotate_body_to_ned(q_bn_i, np.array([u_body, v_body, w_body]))
        v_neu = ned_to_neu(v_ned)
        v_hat_neu = v_neu / (np.linalg.norm(v_neu) + 1e-12)

        t_ref_neu, n_loop_neu = compute_loop_reference(wps, rec['wp_idx'][i])

        geo['velocity_tangent_error'].append(angle_between(v_hat_neu, t_ref_neu))
        geo['nose_tangent_error'].append(angle_between(x_body_neu, t_ref_neu))
        geo['nose_velocity_error'].append(angle_between(x_body_neu, v_hat_neu))
        geo['wing_plane_error'].append(angle_between(y_body_neu, n_loop_neu))

        z_exp = np.cross(t_ref_neu, n_loop_neu)
        z_exp = z_exp / (np.linalg.norm(z_exp) + 1e-12)
        geo['belly_error'].append(angle_between(z_body_neu, z_exp))

        qe = quat_error_angle(
            q_bn_i,
            np.radians(rec['t_hdg'][i]),
            np.radians(rec['t_pitch'][i]),
            np.radians(rec['t_roll'][i]),
        )
        geo['q_error_rad'].append(qe)

        dr = abs(rec['roll'][i] - rec['t_roll'][i])
        dr = min(dr, 360 - dr)
        geo['roll_tracking_error'].append(dr)

    # ═══ Summary ═══
    ca = np.array(rec['cte']); va = np.array(rec['vt']); ga = np.array(rec['G'])
    aa = np.array(rec['alpha']); ba = np.array(rec['beta'])
    ra = np.array(rec['roll']); tra = np.array(rec['t_roll'])

    vte_a = np.array(geo['velocity_tangent_error'])
    nte_a = np.array(geo['nose_tangent_error'])
    nve_a = np.array(geo['nose_velocity_error'])
    wpe_a = np.array(geo['wing_plane_error'])
    be_a = np.array(geo['belly_error'])
    qe_a = np.array(geo['q_error_rad'])
    rte_a = np.array(geo['roll_tracking_error'])

    m = {
        'name': name, 'angle_deg': ang, 'radius_m': rad,
        'completed': bool(ok), 'steps': n,
        'CTE_mean': float(ca.mean()), 'CTE_p50': float(np.percentile(ca, 50)),
        'CTE_p90': float(np.percentile(ca, 90)), 'CTE_max': float(ca.max()),
        'velocity_tangent_error_mean': float(vte_a.mean()),
        'velocity_tangent_error_p90': float(np.percentile(vte_a, 90)),
        'nose_tangent_error_mean': float(nte_a.mean()),
        'nose_tangent_error_p90': float(np.percentile(nte_a, 90)),
        'nose_velocity_error_mean': float(nve_a.mean()),
        'nose_velocity_error_p90': float(np.percentile(nve_a, 90)),
        'wing_plane_error_mean': float(wpe_a.mean()),
        'wing_plane_error_p90': float(np.percentile(wpe_a, 90)),
        'belly_error_mean': float(be_a.mean()),
        'q_error_mean_rad': float(qe_a.mean()),
        'q_error_p90_rad': float(np.percentile(qe_a, 90)),
        'roll_tracking_error_mean': float(rte_a.mean()),
        'env_alpha_min': float(aa.min()), 'env_alpha_max': float(aa.max()),
        'env_alpha_mean': float(aa.mean()),
        'env_beta_min': float(ba.min()), 'env_beta_max': float(ba.max()),
        'target_roll_min': float(tra.min()), 'target_roll_max': float(tra.max()),
        'actual_roll_min': float(ra.min()), 'actual_roll_max': float(ra.max()),
        'actual_roll_mean': float(ra.mean()),
        'vt_min': float(va.min()), 'vt_mean': float(va.mean()),
        'Gmax': float(ga.max()), 'Gmean': float(ga.mean()),
        'alt_min': float(np.array(rec['a']).min()),
        'alt_max': float(np.array(rec['a']).max()),
        'termination': 'crash' if crashed else ('ok' if ok else 'timeout'),
    }
    m['grade_cte_only_deprecated'] = grade_loop(m, deprecated=True)
    m['grade_loop_quality'] = grade_loop(m, deprecated=False)

    # Save NPZ rollout
    np.savez(
        os.path.join(out_root, 'rollouts', f'{name}.npz'),
        waypoints=wps, total_arc=total_arc, **rec,
    )
    # Save JSON metrics
    with open(os.path.join(out_root, 'metrics', f'{name}_metrics.json'), 'w') as f:
        json.dump(m, f, indent=2, default=str)

    # ACMI via reusable exporter
    acmi_path = os.path.join(out_root, 'acmi', f'{name}.acmi')
    write_acmi(acmi_path, wps, rec)

    return m


# ═══════════════════════ Main ═══════════════════════


def main(checkpoint_path=None, output_root=None, tests=None):
    """Run loop-quality evaluation.

    Args:
        checkpoint_path: Override default checkpoint path.
        output_root: Override default output root.
        tests: Override default test grid (list of tuples).
    """
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
        out_root = os.path.join(_px, 'results/loop_quality_evaluation', tag)
    else:
        out_root = os.path.join(output_root, tag)
    for sub in ['rollouts', 'metrics', 'acmi', 'figures']:
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    print(f"Output: {out_root}")

    if tests is None:
        tests = [
            ('pu060_R8000',   60,  8000,  600, 200,  900),
            ('pu060_R10000',  60, 10000,  800, 300, 1200),
            ('pu090_R10000',  90, 10000, 1000, 400, 1500),
            ('pu090_R12000',  90, 12000, 1000, 400, 1500),
            ('pu105_R10000', 105, 10000, 1000, 400, 1500),
            ('pu105_R12000', 105, 12000, 1000, 400, 1500),
            ('pu120_R10000', 120, 10000, 1000, 400, 1800),
            ('pu120_R12000', 120, 12000, 1000, 400, 1800),
            ('pu135_R12000', 135, 12000, 1200, 500, 2000),
            ('pu150_R12000', 150, 12000, 1200, 500, 2000),
            ('pu180_R15000', 180, 15000, 1500, 500, 2500),
            ('pu180_R12000', 180, 12000, 1500, 500, 2500),
        ]

    all_metrics = []
    header = (
        f"{'Name':<20} {'Loop':>5} {'CTE-only':>9} {'St':>4} {'CTEm':>6} "
        f"{'v_tang':>7} {'n_tang':>7} {'n_vel':>7} {'wing_p':>7} "
        f"{'q_err':>6} {'Gmax':>5} {'vt_min':>6} {'alpha':>12} {'term':>8}"
    )
    print(header)
    print("-" * 120)

    for name, ang, rad, la, rr, mx in tests:
        m = run_test(name, ang, rad, la, rr, mx, rng, net, net_params, env, out_root)
        all_metrics.append(m)
        print(
            f"{name:<20} {m['grade_loop_quality']:>5} {m['grade_cte_only_deprecated']:>9} "
            f"{m['steps']:4d} {m['CTE_mean']:6.0f} {m['velocity_tangent_error_mean']:7.1f} "
            f"{m['nose_tangent_error_mean']:7.1f} {m['nose_velocity_error_mean']:7.1f} "
            f"{m['wing_plane_error_mean']:7.1f} {m['q_error_mean_rad']:6.3f} "
            f"{m['Gmax']:5.1f} {m['vt_min']:6.0f} "
            f"[{m['env_alpha_min']:.0f},{m['env_alpha_max']:.0f}] {m['termination']:>8}"
        )

    # ── CSV ──
    csv_path = os.path.join(out_root, 'loop_quality_summary.csv')
    fieldnames = [
        'name', 'angle_deg', 'radius_m', 'completed', 'steps', 'termination',
        'grade_cte_only_deprecated', 'grade_loop_quality',
        'CTE_mean', 'CTE_p50', 'CTE_p90', 'CTE_max',
        'velocity_tangent_error_mean', 'velocity_tangent_error_p90',
        'nose_tangent_error_mean', 'nose_tangent_error_p90',
        'nose_velocity_error_mean', 'nose_velocity_error_p90',
        'wing_plane_error_mean', 'wing_plane_error_p90',
        'belly_error_mean',
        'q_error_mean_rad', 'q_error_p90_rad',
        'roll_tracking_error_mean',
        'env_alpha_min', 'env_alpha_max', 'env_alpha_mean',
        'env_beta_min', 'env_beta_max',
        'target_roll_min', 'target_roll_max',
        'actual_roll_min', 'actual_roll_max', 'actual_roll_mean',
        'vt_min', 'vt_mean', 'Gmax', 'Gmean',
        'alt_min', 'alt_max',
    ]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_metrics)
    print(f"\nCSV: {csv_path}")

    # ── Markdown report ──
    _write_markdown_report(all_metrics, tag, out_root)
    # ── LaTeX table ──
    _write_latex_table(all_metrics, out_root)

    print(f"\nDONE. Output directory: {out_root}")
    return out_root, all_metrics


def _write_markdown_report(all_metrics, tag, out_root):
    md_path = os.path.join(out_root, 'loop_quality_report.md')
    with open(md_path, 'w') as f:
        f.write("# Vertical Arc Loop-Quality Evaluation\n\n")
        f.write(f"**Checkpoint:** epoch 619 (vertical-energy fine-tuned)\n")
        f.write(f"**Date:** {tag}\n")
        f.write(f"**Target:** loop-plane full-attitude (roll computed from rotation matrix)\n\n")

        f.write("## Grading Criteria\n\n")
        f.write("### Loop-Quality Grade (NEW)\n\n")
        f.write("| Criterion | A | B | C | Fail |\n")
        f.write("|-----------|---|---|---|------|\n")
        f.write("| CTE_mean | <100m | <500m | - | - |\n")
        f.write("| CTE_p90 | <300m | <1200m | - | - |\n")
        f.write("| CTE_max | <800m | - | - | - |\n")
        f.write("| velocity_tangent_error | <15° | <30° | - | - |\n")
        f.write("| nose_tangent_error | <15° | <30° | - | - |\n")
        f.write("| nose_velocity_error | <15° | - | - | - |\n")
        f.write("| wing_plane_error | <15° | - | - | - |\n")
        f.write("| q_error (attitude) | <0.5 rad | - | - | - |\n")
        f.write("| Gmax | <9 | <10 | - | - |\n")
        f.write("| vt_min | ≥190 m/s | ≥175 m/s | - | - |\n")
        f.write("| Completed | yes | yes | yes | no |\n\n")

        f.write("### CTE-Only Grade (DEPRECATED)\n\n")
        f.write("| Criterion | A | B | C | Fail |\n")
        f.write("|-----------|---|---|---|------|\n")
        f.write("| CTE_mean | <100m | <500m | - | - |\n")
        f.write("| CTE_p90 | <300m | <1200m | - | - |\n")
        f.write("| CTE_max | <800m | - | - | - |\n")
        f.write("| Gmax | <9 | <10 | - | - |\n")
        f.write("| vt_min | ≥190 m/s | ≥175 m/s | - | - |\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Angle | R | CTE_m | v_tang | n_tang | n_vel | wing_p | belly | q_err | Gmax | vt_min | α range | Loop Grade | CTE-Grade (depr.) |\n")
        f.write("|-------|---|-------|--------|--------|-------|--------|-------|-------|------|--------|---------|------------|-------------------|\n")
        for m in all_metrics:
            f.write(
                f"| {m['angle_deg']}° | {m['radius_m']} | {m['CTE_mean']:.0f} | "
                f"{m['velocity_tangent_error_mean']:.1f}° | "
                f"{m['nose_tangent_error_mean']:.1f}° | {m['nose_velocity_error_mean']:.1f}° | "
                f"{m['wing_plane_error_mean']:.1f}° | {m['belly_error_mean']:.1f}° | "
                f"{m['q_error_mean_rad']:.3f} | {m['Gmax']:.1f} | {m['vt_min']:.0f} | "
                f"[{m['env_alpha_min']:.0f},{m['env_alpha_max']:.0f}] | "
                f"**{m['grade_loop_quality']}** | {m['grade_cte_only_deprecated']} |\n"
            )

        f.write("\n## Per-Angle Detailed Metrics\n\n")
        for m in all_metrics:
            f.write(f"### {m['angle_deg']}° Vertical Arc (R={m['radius_m']}m)\n\n")
            f.write(f"- **Loop Grade:** {m['grade_loop_quality']} | CTE-only (depr.): {m['grade_cte_only_deprecated']}\n")
            f.write(f"- Steps: {m['steps']} | Termination: {m['termination']}\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key in [
                'CTE_mean', 'CTE_p90', 'CTE_max',
                'velocity_tangent_error_mean', 'velocity_tangent_error_p90',
                'nose_tangent_error_mean', 'nose_tangent_error_p90',
                'nose_velocity_error_mean', 'nose_velocity_error_p90',
                'wing_plane_error_mean', 'wing_plane_error_p90',
                'belly_error_mean', 'q_error_mean_rad', 'q_error_p90_rad',
                'roll_tracking_error_mean',
                'env_alpha_min', 'env_alpha_max', 'env_alpha_mean',
                'target_roll_min', 'target_roll_max',
                'actual_roll_min', 'actual_roll_max',
                'Gmax', 'vt_min',
            ]:
                v = m.get(key, 'N/A')
                if isinstance(v, float):
                    f.write(f"| {key} | {v:.3f} |\n")
                else:
                    f.write(f"| {key} | {v} |\n")
            f.write("\n")

        # Demo categories
        f.write("## Demo Categories\n\n")
        f.write("### Main Demo\n\n")
        main_demo = [m for m in all_metrics if m['angle_deg'] in [60, 90] and m['grade_loop_quality'] in ('A', 'B')]
        if main_demo:
            f.write("| Angle | R | CTE_m | wing_p | Grade |\n")
            f.write("|-------|---|-------|--------|-------|\n")
            for m in main_demo:
                f.write(f"| {m['angle_deg']}° | {m['radius_m']} | {m['CTE_mean']:.0f} | {m['wing_plane_error_mean']:.1f}° | **{m['grade_loop_quality']}** |\n")
        else:
            f.write("No configurations meet main-demo criteria yet (loop-quality A or B for 60°/90°).\n")

        f.write("\n### Boundary Demo\n\n")
        boundary = [m for m in all_metrics if m['angle_deg'] in [105, 120, 135, 150]]
        if boundary:
            f.write("| Angle | R | CTE_m | wing_p | Grade |\n")
            f.write("|-------|---|-------|--------|-------|\n")
            for m in boundary:
                f.write(f"| {m['angle_deg']}° | {m['radius_m']} | {m['CTE_mean']:.0f} | {m['wing_plane_error_mean']:.1f}° | **{m['grade_loop_quality']}** |\n")

        f.write("\n### Failure Diagnosis\n\n")
        failures = [m for m in all_metrics if m['grade_loop_quality'] == 'Fail']
        if failures:
            f.write("| Angle | R | CTE_m | v_tang | n_tang | wing_p | q_err | Gmax | vt_min |\n")
            f.write("|-------|---|-------|--------|--------|--------|-------|------|--------|\n")
            for m in failures:
                f.write(
                    f"| {m['angle_deg']}° | {m['radius_m']} | {m['CTE_mean']:.0f} | "
                    f"{m['velocity_tangent_error_mean']:.1f}° | {m['nose_tangent_error_mean']:.1f}° | "
                    f"{m['wing_plane_error_mean']:.1f}° | {m['q_error_mean_rad']:.3f} | "
                    f"{m['Gmax']:.1f} | {m['vt_min']:.0f} |\n"
                )

        f.write("\n## Key Findings\n\n")
        grades = {}
        for m in all_metrics:
            g = m['grade_loop_quality']
            grades[g] = grades.get(g, 0) + 1
        f.write(f"- A: {grades.get('A', 0)}, B: {grades.get('B', 0)}, C: {grades.get('C', 0)}, Fail: {grades.get('Fail', 0)}\n")

        f.write("\n### Grade Transitions vs CTE-only\n\n")
        f.write("| Angle | R | CTE-only | Loop-Quality | Key Regressor |\n")
        f.write("|-------|---|----------|--------------|---------------|\n")
        for m in all_metrics:
            old_g = m['grade_cte_only_deprecated']
            new_g = m['grade_loop_quality']
            if old_g != new_g:
                reasons = []
                if m['velocity_tangent_error_mean'] >= 15: reasons.append('v_tang')
                if m['nose_tangent_error_mean'] >= 15: reasons.append('n_tang')
                if m['nose_velocity_error_mean'] >= 15: reasons.append('n_vel')
                if m['wing_plane_error_mean'] >= 15: reasons.append('wing_p')
                if m.get('q_error_mean_rad', 0) >= 0.5: reasons.append('q_err')
                if m['Gmax'] >= 9: reasons.append('Gmax')
                if m['vt_min'] < 190: reasons.append('vt')
                if m['CTE_mean'] >= 100: reasons.append('CTE')
                reg = ', '.join(reasons) if reasons else 'N/A'
                f.write(f"| {m['angle_deg']}° | {m['radius_m']} | {old_g} | **{new_g}** | {reg} |\n")

    print(f"MD: {md_path}")


def _write_latex_table(all_metrics, out_root):
    tex_path = os.path.join(out_root, 'paper_ready_vertical_arc_table.tex')
    with open(tex_path, 'w') as f:
        f.write("% Paper-ready vertical arc evaluation table\n")
        f.write("% Loop-quality grade with full geometry criteria\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Vertical arc loop-quality evaluation. ")
        f.write("Grade A requires CTE$_\\text{mean}<100$m, CTE$_{p90}<300$m, ")
        f.write("all geometry errors $<15^\\circ$, $G_\\text{max}<9$, ")
        f.write("and $v_t\\geq190$m/s. ")
        f.write("CTE-only grades (deprecated) shown in parentheses.}\n")
        f.write("\\label{tab:vertical_arc_loop_quality}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{rrrcccccccc}\n")
        f.write("\\toprule\n")
        f.write("Arc & R & CTE$_\\text{mean}$ & $\\varepsilon_{v\\parallel}$ & ")
        f.write("$\\varepsilon_{x\\parallel}$ & $\\varepsilon_{x,v}$ & ")
        f.write("$\\varepsilon_\\text{wing}$ & ")
        f.write("$\\|q_\\text{err}\\|$ & $G_\\text{max}$ & $v_{t,\\min}$ & ")
        f.write("$\\alpha$ range & Grade \\\\\n")
        f.write("($^\\circ$) & (m) & (m) & ($^\\circ$) & ($^\\circ$) & ($^\\circ$) & ")
        f.write("($^\\circ$) & (rad) & (g) & (m/s) & ($^\\circ$) & \\\\\n")
        f.write("\\midrule\n")

        for m in all_metrics:
            g = m['grade_loop_quality']
            g_old = m['grade_cte_only_deprecated']
            grade_str = g if g == g_old else f"{g} ({g_old})"
            f.write(
                f"{m['angle_deg']} & {m['radius_m']} & "
                f"{m['CTE_mean']:.0f} & "
                f"{m['velocity_tangent_error_mean']:.1f} & "
                f"{m['nose_tangent_error_mean']:.1f} & "
                f"{m['nose_velocity_error_mean']:.1f} & "
                f"{m['wing_plane_error_mean']:.1f} & "
                f"{m['q_error_mean_rad']:.3f} & "
                f"{m['Gmax']:.1f} & "
                f"{m['vt_min']:.0f} & "
                f"[{m['env_alpha_min']:.0f},{m['env_alpha_max']:.0f}] & "
                f"{grade_str} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"TEX: {tex_path}")


if __name__ == '__main__':
    main()
