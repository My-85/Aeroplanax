"""
Checkpoint regression testing — compare a candidate checkpoint against epoch619 baseline.

Tests horizontal retention (circles, S-curve, figure-8, climb/descent) and
loop-quality retention (60°-180° vertical arcs). Flags regressions automatically.

Usage:
    python experiments/hierarchical_trajectory_tracking/checkpoint_regression.py \
        --candidate /path/to/candidate/checkpoint

    python experiments/hierarchical_trajectory_tracking/checkpoint_regression.py \
        --baseline /path/to/baseline --candidate /path/to/candidate
"""

import os, sys, json, csv, argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.3'

_px = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _px)

import jax, jax.numpy as jnp, numpy as np
import orbax.checkpoint as ocp
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from experiments.hierarchical_trajectory_tracking.render_ablation_tests import (
    ScannedRNN, ActorCriticRNN, NET_CFG, SEED,
)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import (
    level_circle, s_curve, figure_eight, mild_climb, vertical_pullup_arc,
)
from experiments.hierarchical_trajectory_tracking.planner import (
    PurePursuitPlanner, PlannerConfig,
)
from experiments.hierarchical_trajectory_tracking.path_utils import compute_true_cte
from experiments.hierarchical_trajectory_tracking.loop_attitude_target import (
    loop_plane_rotation_matrix, rotation_matrix_to_quaternion, quaternion_to_euler,
)
from experiments.hierarchical_trajectory_tracking.export_acmi import write_acmi
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as Env,
    Heading_Pitch_V_TaskParams as Params,
    _quat_from_euler_nb,
    _quat_conj,
    _quat_mul,
)

DEFAULT_BASELINE = os.path.join(
    _px,
    'results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619',
)


def _f(x):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


# ═══════════════════════ Geometry helpers ═══════════════════════

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


def get_loop_roll(theta_deg, init_yaw=0.0):
    R = loop_plane_rotation_matrix(np.radians(theta_deg), init_yaw, 1)
    q = rotation_matrix_to_quaternion(R)
    r, p, h = quaternion_to_euler(q)
    return r


# ═══════════════════════ Model loading ═══════════════════════

def load_checkpoint(checkpoint_path):
    """Load model parameters from an orbax checkpoint."""
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
    return env, net, net_params, ckpt['params']


# ═══════════════════════ Horizontal test ═══════════════════════

HORIZONTAL_TESTS = [
    # (name, traj_fn, traj_kwargs, lookahead, reach_r, max_steps)
    ('circle_R3000_right', level_circle,
     {'radius': 3000, 'direction': 1, 'n_points': 60}, 1000, 500, 1500),
    ('circle_R3000_left', level_circle,
     {'radius': 3000, 'direction': -1, 'n_points': 60}, 1000, 500, 1500),
    ('circle_R5000_right', level_circle,
     {'radius': 5000, 'direction': 1, 'n_points': 60}, 1000, 500, 1500),
    ('circle_R5000_left', level_circle,
     {'radius': 5000, 'direction': -1, 'n_points': 60}, 1000, 500, 1500),
    ('s_curve_A3000', s_curve,
     {'amplitude': 3000, 'half_period': 10000, 'n_points': 80}, 1000, 500, 1500),
    ('figure8_R5000', figure_eight,
     {'radius': 5000, 'n_points': 100}, 1000, 500, 2000),
    ('climb_1000m', mild_climb,
     {'length': 15000, 'delta_alt': 1000, 'n_points': 30}, 500, 200, 800),
    ('descent_1000m', mild_climb,
     {'length': 15000, 'delta_alt': -1000, 'n_points': 30}, 500, 200, 800),
]


def run_horizontal_test(name, traj_fn, traj_kwargs, lookahead, reach_r, max_steps,
                        rng, net, net_params, env, out_root):
    """Run a horizontal trajectory tracking test. No loop-plane roll — just pure pursuit."""
    origin_n, origin_e, origin_alt = 0.0, 0.0, 5000.0
    init_yaw = traj_kwargs.get('init_yaw', 0.0)
    wps, meta = traj_fn(origin_n, origin_e, origin_alt, init_yaw, **{
        k: v for k, v in traj_kwargs.items() if k != 'init_yaw'
    })
    total_arc = meta.get('total_length_m', 0.0)
    cfg = PlannerConfig(
        lookahead_dist=lookahead, reach_radius=reach_r,
        blend_steps=250, target_vt=250.0,
    )
    planner = PurePursuitPlanner(cfg)

    rng2, rk = jax.random.split(rng)
    obs_dict, state = env.reset(rk, Params())
    q_nb_init = _quat_from_euler_nb(0.0, 0.0, init_yaw)
    q_bn_init = _quat_conj(q_nb_init)
    state = state.replace(
        plane_state=state.plane_state.replace(
            yaw=jnp.array([init_yaw]),
            q0=jnp.array([q_bn_init[0]]), q1=jnp.array([q_bn_init[1]]),
            q2=jnp.array([q_bn_init[2]]), q3=jnp.array([q_bn_init[3]]),
        ),
        target_heading=jnp.array([init_yaw]),
    )
    planner.reset(wps, origin_n, origin_e, origin_alt, 250.0)

    hstate = ScannedRNN.initialize_carry(1, NET_CFG['GRU_HIDDEN_DIM'])
    df = jnp.zeros((1,))

    rec = {
        't': [], 'n': [], 'e': [], 'a': [], 'vt': [],
        'roll': [], 'pitch': [], 'yaw': [],
        'alpha': [], 'G': [], 'cte': [], 'wp_idx': [],
    }
    crashed = False

    for step in range(max_steps):
        ps = state.plane_state
        no = _f(ps.north); ea = _f(ps.east); al = _f(ps.altitude)
        vt = _f(ps.vt); ro = _f(ps.roll); pi = _f(ps.pitch); ya = _f(ps.yaw)
        alph = _f(ps.alpha)
        ax = _f(ps.ax); ay = _f(ps.ay); az = _f(ps.az)

        result = planner.step(no, ea, al, ya, pi, ro, vt)
        th, tp, tr, tv = (
            result['target_heading'], result['target_pitch'],
            result['target_roll'], result['target_vt'],
        )

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
        rec['alpha'].append(np.degrees(alph))
        rec['G'].append(float(np.sqrt(ax ** 2 + ay ** 2 + az ** 2)))
        rec['cte'].append(compute_true_cte(
            np.array([no, ea, al]), wps, wp_idx, 10,
        ))
        rec['wp_idx'].append(wp_idx)

        if bool(done[env.agents[0]]):
            crashed = True
            break
        if planner.is_done():
            break

    n = len(rec['t']); ok = planner.is_done() and not crashed

    ca = np.array(rec['cte']); va = np.array(rec['vt']); ga = np.array(rec['G'])

    m = {
        'name': name, 'category': 'horizontal',
        'completed': bool(ok), 'steps': n,
        'CTE_mean': float(ca.mean()), 'CTE_p50': float(np.percentile(ca, 50)),
        'CTE_p90': float(np.percentile(ca, 90)), 'CTE_max': float(ca.max()),
        'vt_min': float(va.min()), 'vt_mean': float(va.mean()),
        'Gmax': float(ga.max()), 'Gmean': float(ga.mean()),
        'alt_min': float(np.array(rec['a']).min()),
        'alt_max': float(np.array(rec['a']).max()),
        'termination': 'crash' if crashed else ('ok' if ok else 'timeout'),
    }
    # Horizontal grade: pass if CTE ok, no crash, G < 9, vt ≥ 190
    if not ok:
        m['grade'] = 'Fail'
    elif m['CTE_mean'] < 200 and m['Gmax'] < 9 and m['vt_min'] >= 190:
        m['grade'] = 'Pass'
    elif m['CTE_mean'] < 500 and m['Gmax'] < 10 and m['vt_min'] >= 175:
        m['grade'] = 'Marginal'
    elif ok:
        m['grade'] = 'Degraded'
    else:
        m['grade'] = 'Fail'

    # Save NPZ
    np.savez(os.path.join(out_root, 'rollouts', f'{name}.npz'), waypoints=wps, **rec)
    # ACMI
    write_acmi(os.path.join(out_root, 'acmi', f'{name}.acmi'), wps, rec)

    return m


# ═══════════════════════ Loop-quality test (adapted from loop_quality_eval) ═══════════════════════

LOOP_TESTS = [
    ('pu060_R12000',  60, 12000,  800, 300, 1200),
    ('pu090_R12000',  90, 12000, 1000, 400, 1500),
    ('pu120_R12000', 120, 12000, 1000, 400, 1800),
    ('pu150_R12000', 150, 12000, 1200, 500, 2000),
    ('pu180_R15000', 180, 15000, 1500, 500, 2500),
]


def run_loop_test(name, ang, rad, la, rr, mx, rng, net, net_params, env, out_root):
    """Run a single vertical arc test with full geometry metrics."""
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

    # Geometry metrics
    geo = {
        'velocity_tangent_error': [], 'nose_tangent_error': [],
        'nose_velocity_error': [], 'wing_plane_error': [],
        'belly_error': [],
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

    ca = np.array(rec['cte']); va = np.array(rec['vt']); ga = np.array(rec['G'])
    vte_a = np.array(geo['velocity_tangent_error'])
    nte_a = np.array(geo['nose_tangent_error'])
    wpe_a = np.array(geo['wing_plane_error'])
    nve_a = np.array(geo['nose_velocity_error'])

    m = {
        'name': name, 'category': 'loop', 'angle_deg': ang, 'radius_m': rad,
        'completed': bool(ok), 'steps': n,
        'CTE_mean': float(ca.mean()), 'CTE_p90': float(np.percentile(ca, 90)),
        'CTE_max': float(ca.max()),
        'velocity_tangent_error_mean': float(vte_a.mean()),
        'nose_tangent_error_mean': float(nte_a.mean()),
        'nose_velocity_error_mean': float(nve_a.mean()),
        'wing_plane_error_mean': float(wpe_a.mean()),
        'vt_min': float(va.min()), 'Gmax': float(ga.max()),
        'alt_min': float(np.array(rec['a']).min()),
        'alt_max': float(np.array(rec['a']).max()),
        'termination': 'crash' if crashed else ('ok' if ok else 'timeout'),
    }

    # Loop-quality grade
    cm = m['CTE_mean']; c90 = m['CTE_p90']; cmax = m['CTE_max']
    g = m['Gmax']; v = m['vt_min']
    vte = m['velocity_tangent_error_mean']
    nte = m['nose_tangent_error_mean']
    nve = m['nose_velocity_error_mean']
    wpe = m['wing_plane_error_mean']

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
    m['grade'] = grade

    # Save
    np.savez(os.path.join(out_root, 'rollouts', f'{name}.npz'), waypoints=wps, **rec)
    write_acmi(os.path.join(out_root, 'acmi', f'{name}.acmi'), wps, rec)

    return m


# ═══════════════════════ Comparison logic ═══════════════════════

REGRESSION_THRESHOLDS = {
    'CTE_mean': (1.5, 'relative'),   # 50% worse
    'CTE_p90': (1.5, 'relative'),
    'velocity_tangent_error_mean': (5.0, 'absolute'),  # 5° worse
    'nose_tangent_error_mean': (5.0, 'absolute'),
    'wing_plane_error_mean': (5.0, 'absolute'),
    'vt_min': (-20.0, 'absolute'),  # 20 m/s slower
    'Gmax': (1.0, 'absolute'),      # 1g higher
}


def compare_metrics(baseline: Dict, candidate: Dict) -> Dict:
    """Compare two metric dicts and flag regressions."""
    result = {'name': baseline.get('name', 'unknown')}
    regressions = []
    for key, (threshold, mode) in REGRESSION_THRESHOLDS.items():
        if key not in baseline or key not in candidate:
            continue
        bv = baseline[key]; cv = candidate[key]
        result[f'baseline_{key}'] = bv
        result[f'candidate_{key}'] = cv
        delta = cv - bv
        result[f'delta_{key}'] = delta
        if mode == 'relative' and bv != 0:
            ratio = cv / bv if bv != 0 else float('inf')
            result[f'ratio_{key}'] = ratio
            if ratio > threshold:
                regressions.append(f'{key}: {bv:.2f}→{cv:.2f} ({ratio:.2f}x)')
        elif mode == 'absolute':
            if delta > threshold:
                regressions.append(f'{key}: {bv:.2f}→{cv:.2f} (Δ{delta:+.2f})')
    # Grade change
    result['baseline_grade'] = baseline.get('grade', 'N/A')
    result['candidate_grade'] = candidate.get('grade', 'N/A')
    if baseline.get('grade') != candidate.get('grade'):
        regressions.append(f"grade: {baseline.get('grade')}→{candidate.get('grade')}")
    result['regressions'] = regressions
    result['has_regression'] = len(regressions) > 0
    return result


def can_recommend(comparisons: List[Dict]) -> Tuple[bool, List[str]]:
    """Check if candidate can be recommended over baseline."""
    reasons = []
    # Check horizontal retention
    horizontal = [c for c in comparisons if c.get('name', '').startswith(('circle', 's_curve', 'figure8', 'climb', 'descent'))]
    h_regressions = [c for c in horizontal if c['has_regression']]
    if h_regressions:
        reasons.append(f"Horizontal regression in: {[c['name'] for c in h_regressions]}")

    # Check loop-quality retention
    loop = [c for c in comparisons if c['name'].startswith('pu')]
    l_regressions = [c for c in loop if c['has_regression']]
    if l_regressions:
        reasons.append(f"Loop regression in: {[c['name'] for c in l_regressions]}")

    # Check 180° improvement
    pu180 = [c for c in loop if c['name'].startswith('pu180')]
    if pu180:
        c = pu180[0]
        if c.get('baseline_grade') == 'Fail' and c.get('candidate_grade') == 'Fail':
            reasons.append('180° still fails — no improvement')
        elif c.get('candidate_grade') == 'Fail':
            reasons.append('180° still fails')

    # Check for crashes/overload
    crashes = [c for c in comparisons if c.get('termination') == 'crash']
    if crashes:
        reasons.append(f"Crashes: {[c['name'] for c in crashes]}")
    overloads = [c for c in comparisons if c.get('Gmax', 0) > 9]
    if overloads:
        reasons.append(f"Overload >9g: {[c['name'] for c in overloads]}")

    recommend = len(reasons) == 0
    return recommend, reasons


# ═══════════════════════ Main ═══════════════════════


def main(baseline_path=None, candidate_path=None, output_root=None):
    if baseline_path is None:
        baseline_path = DEFAULT_BASELINE
    if candidate_path is None:
        raise ValueError("Must provide --candidate checkpoint path")

    print(f"Baseline: {baseline_path}")
    print(f"Candidate: {candidate_path}")

    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_root is None:
        out_root = os.path.join(_px, 'results/checkpoint_loop_regression', tag)
    else:
        out_root = os.path.join(output_root, tag)

    for sub in ['rollouts', 'metrics', 'acmi', 'figures']:
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    print(f"Output: {out_root}")

    # Load both checkpoints
    print("\nLoading baseline...")
    env, net, _, baseline_params = load_checkpoint(baseline_path)
    print("Loading candidate...")
    _, _, _, candidate_params = load_checkpoint(candidate_path)

    rng = jax.random.PRNGKey(SEED)

    all_comparisons = []

    # ═══ Horizontal retention ═══
    print("\n" + "=" * 70)
    print("HORIZONTAL RETENTION TESTS")
    print("=" * 70)

    for name, traj_fn, traj_kwargs, la, rr, mx in HORIZONTAL_TESTS:
        print(f"\n  {name}...")
        # Run baseline
        b_m = run_horizontal_test(
            f'{name}_baseline', traj_fn, traj_kwargs, la, rr, mx,
            rng, net, baseline_params, env, out_root,
        )
        # Run candidate
        c_m = run_horizontal_test(
            f'{name}_candidate', traj_fn, traj_kwargs, la, rr, mx,
            rng, net, candidate_params, env, out_root,
        )
        comp = compare_metrics(b_m, c_m)
        comp['name'] = name
        all_comparisons.append(comp)
        status = "REGRESSION" if comp['has_regression'] else "OK"
        print(f"    Baseline: {b_m['grade']} (CTE={b_m['CTE_mean']:.0f}m, Gmax={b_m['Gmax']:.1f}, vt_min={b_m['vt_min']:.0f})")
        print(f"    Candidate: {c_m['grade']} (CTE={c_m['CTE_mean']:.0f}m, Gmax={c_m['Gmax']:.1f}, vt_min={c_m['vt_min']:.0f})")
        print(f"    → {status}")

    # ═══ Loop-quality retention ═══
    print("\n" + "=" * 70)
    print("LOOP-QUALITY RETENTION TESTS")
    print("=" * 70)

    for name, ang, rad, la, rr, mx in LOOP_TESTS:
        print(f"\n  {name} ({ang}°, R={rad}m)...")
        b_m = run_loop_test(
            f'{name}_baseline', ang, rad, la, rr, mx,
            rng, net, baseline_params, env, out_root,
        )
        c_m = run_loop_test(
            f'{name}_candidate', ang, rad, la, rr, mx,
            rng, net, candidate_params, env, out_root,
        )
        comp = compare_metrics(b_m, c_m)
        comp['name'] = name
        all_comparisons.append(comp)
        status = "REGRESSION" if comp['has_regression'] else "OK"
        print(f"    Baseline: {b_m['grade']} (CTE={b_m['CTE_mean']:.0f}m, v_tang={b_m['velocity_tangent_error_mean']:.1f}°, "
              f"wing_p={b_m['wing_plane_error_mean']:.1f}°)")
        print(f"    Candidate: {c_m['grade']} (CTE={c_m['CTE_mean']:.0f}m, v_tang={c_m['velocity_tangent_error_mean']:.1f}°, "
              f"wing_p={c_m['wing_plane_error_mean']:.1f}°)")
        print(f"    → {status}")

    # ═══ Recommendation ═══
    recommend, reasons = can_recommend(all_comparisons)

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    if recommend:
        print("CANDIDATE CAN BE RECOMMENDED — no regressions detected.")
    else:
        print("CANDIDATE NOT RECOMMENDED:")
        for r in reasons:
            print(f"  - {r}")

    # ═══ Write outputs ═══
    # Comparison summary CSV
    csv_path = os.path.join(out_root, 'comparison_summary.csv')
    fieldnames = ['name', 'baseline_grade', 'candidate_grade', 'has_regression',
                  'baseline_CTE_mean', 'candidate_CTE_mean', 'delta_CTE_mean',
                  'baseline_velocity_tangent_error_mean', 'candidate_velocity_tangent_error_mean',
                  'baseline_wing_plane_error_mean', 'candidate_wing_plane_error_mean',
                  'baseline_Gmax', 'candidate_Gmax',
                  'baseline_vt_min', 'candidate_vt_min',
                  'regressions']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for c in all_comparisons:
            row = {k: c.get(k, '') for k in fieldnames}
            if isinstance(row.get('regressions'), list):
                row['regressions'] = '; '.join(row['regressions'])
            w.writerow(row)
    print(f"\nCSV: {csv_path}")

    # Loop-quality comparison CSV
    lq_path = os.path.join(out_root, 'loop_quality_comparison.csv')
    loop_comps = [c for c in all_comparisons if c['name'].startswith('pu')]
    if loop_comps:
        lq_fields = ['name', 'baseline_grade', 'candidate_grade',
                      'baseline_CTE_mean', 'candidate_CTE_mean',
                      'baseline_velocity_tangent_error_mean', 'candidate_velocity_tangent_error_mean',
                      'baseline_nose_tangent_error_mean', 'candidate_nose_tangent_error_mean',
                      'baseline_wing_plane_error_mean', 'candidate_wing_plane_error_mean',
                      'regressions']
        with open(lq_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=lq_fields, extrasaction='ignore')
            w.writeheader()
            for c in loop_comps:
                row = {k: c.get(k, '') for k in lq_fields}
                if isinstance(row.get('regressions'), list):
                    row['regressions'] = '; '.join(row['regressions'])
                w.writerow(row)
        print(f"Loop-quality CSV: {lq_path}")

    # Horizontal proxy comparison CSV
    hp_path = os.path.join(out_root, 'horizontal_proxy_comparison.csv')
    h_comps = [c for c in all_comparisons if not c['name'].startswith('pu')]
    if h_comps:
        hp_fields = ['name', 'baseline_grade', 'candidate_grade',
                      'baseline_CTE_mean', 'candidate_CTE_mean', 'delta_CTE_mean',
                      'baseline_Gmax', 'candidate_Gmax',
                      'baseline_vt_min', 'candidate_vt_min',
                      'regressions']
        with open(hp_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hp_fields, extrasaction='ignore')
            w.writeheader()
            for c in h_comps:
                row = {k: c.get(k, '') for k in hp_fields}
                if isinstance(row.get('regressions'), list):
                    row['regressions'] = '; '.join(row['regressions'])
                w.writerow(row)
        print(f"Horizontal proxy CSV: {hp_path}")

    # Markdown report
    report_path = os.path.join(out_root, 'report.md')
    with open(report_path, 'w') as f:
        f.write("# Checkpoint Regression Report\n\n")
        f.write(f"**Date:** {tag}\n\n")
        f.write(f"**Baseline:** `{baseline_path}`\n")
        f.write(f"**Candidate:** `{candidate_path}`\n\n")

        f.write("## Recommendation\n\n")
        if recommend:
            f.write("**CANDIDATE CAN BE RECOMMENDED** — no regressions detected.\n\n")
        else:
            f.write("**CANDIDATE NOT RECOMMENDED:**\n\n")
            for r in reasons:
                f.write(f"- {r}\n")
            f.write("\n")

        f.write("## Horizontal Retention\n\n")
        f.write("| Task | BL Grade | CL Grade | BL CTE | CL CTE | Δ CTE | BL Gmax | CL Gmax | BL vt_min | CL vt_min | Status |\n")
        f.write("|------|----------|----------|--------|--------|-------|---------|---------|-----------|-----------|--------|\n")
        for c in h_comps:
            status = "FAIL" if c['has_regression'] else "OK"
            f.write(f"| {c['name']} | {c['baseline_grade']} | {c['candidate_grade']} | "
                    f"{c.get('baseline_CTE_mean', 'N/A'):.0f} | {c.get('candidate_CTE_mean', 'N/A'):.0f} | "
                    f"{c.get('delta_CTE_mean', 0):+.0f} | "
                    f"{c.get('baseline_Gmax', 0):.1f} | {c.get('candidate_Gmax', 0):.1f} | "
                    f"{c.get('baseline_vt_min', 0):.0f} | {c.get('candidate_vt_min', 0):.0f} | "
                    f"**{status}** |\n")

        f.write("\n## Loop-Quality Retention\n\n")
        f.write("| Angle | BL Grade | CL Grade | BL CTE | CL CTE | BL v_tang | CL v_tang | BL wing_p | CL wing_p | Status |\n")
        f.write("|-------|----------|----------|--------|--------|-----------|-----------|-----------|-----------|--------|\n")
        for c in loop_comps:
            status = "FAIL" if c['has_regression'] else "OK"
            f.write(f"| {c['name']} | {c['baseline_grade']} | {c['candidate_grade']} | "
                    f"{c.get('baseline_CTE_mean', 0):.0f} | {c.get('candidate_CTE_mean', 0):.0f} | "
                    f"{c.get('baseline_velocity_tangent_error_mean', 0):.1f}° | {c.get('candidate_velocity_tangent_error_mean', 0):.1f}° | "
                    f"{c.get('baseline_wing_plane_error_mean', 0):.1f}° | {c.get('candidate_wing_plane_error_mean', 0):.1f}° | "
                    f"**{status}** |\n")

        f.write("\n## Regression Details\n\n")
        for c in all_comparisons:
            if c['has_regression']:
                f.write(f"### {c['name']}\n\n")
                f.write(f"- Baseline grade: **{c['baseline_grade']}** → Candidate grade: **{c['candidate_grade']}**\n")
                for r in c.get('regressions', []):
                    f.write(f"- {r}\n")
                f.write("\n")

        f.write("\n## Criteria for Recommendation\n\n")
        f.write("- Horizontal tasks do not regress\n")
        f.write("- 60°/90°/150° do not regress\n")
        f.write("- 180° improves in loop-quality metrics\n")
        f.write("- No new crash / overload / altitude drift\n")

    print(f"Report: {report_path}")
    print(f"\nDONE. Output directory: {out_root}")
    return out_root, all_comparisons, recommend


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checkpoint regression testing')
    parser.add_argument('--baseline', default=None, help='Baseline checkpoint path')
    parser.add_argument('--candidate', required=True, help='Candidate checkpoint path')
    parser.add_argument('--output', default=None, help='Output root directory')
    args = parser.parse_args()

    main(
        baseline_path=args.baseline or DEFAULT_BASELINE,
        candidate_path=args.candidate,
        output_root=args.output,
    )
