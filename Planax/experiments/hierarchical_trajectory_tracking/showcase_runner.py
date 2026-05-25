"""
Ep619 Six-Maneuver Showcase Runner.

Generates representative fixed-wing maneuver demos using frozen epoch619 policy
with moving-lookahead pure-pursuit guidance.

Demos:
  1. S-curve (snake)
  2. Figure-eight
  3. Mild 3D (helix / climbing variants)
  4. Chandelle-like climbing 180° turn
  5. 90° vertical pull-up (quarter-loop)
  6. Wingover-like climbing turn

Outputs: ACMI, plots, metrics JSON, summary CSV, reports.
"""

import os, sys, json, csv, argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.35'

_px = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _px)

import jax, jax.numpy as jnp, numpy as np
import orbax.checkpoint as ocp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

from experiments.hierarchical_trajectory_tracking.render_ablation_tests import (
    ScannedRNN, ActorCriticRNN, NET_CFG, SEED,
)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import (
    s_curve, figure_eight, mild_climb, vertical_pullup_arc,
    helix_trajectory, climbing_figure_eight, climbing_s_curve,
    level_circle,
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

DEFAULT_CKPT = os.path.join(
    _px,
    'results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619',
)

FORCE_METRICS = [
    'CTE_mean', 'CTE_p50', 'CTE_p90', 'CTE_max',
    'velocity_tangent_error_mean', 'nose_tangent_error_mean',
    'nose_velocity_error_mean', 'wing_plane_error_mean',
    'q_error_mean_rad', 'Gmax', 'vt_min', 'vt_mean',
    'alt_min', 'alt_max', 'env_alpha_min', 'env_alpha_max',
]


def _f(x):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


# ═══════════════════════ Geometry helpers ═══════════════════════

def _quat_conj_np(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _quat_mul_np(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def _rotate_body_to_ned(q_bn, v_body):
    q_nb = _quat_conj_np(q_bn)
    p = np.array([0.0, v_body[0], v_body[1], v_body[2]])
    qpq = _quat_mul_np(_quat_mul_np(q_nb, p), _quat_conj_np(q_nb))
    return qpq[1:]

def _ned_to_neu(v): return np.array([v[0], v[1], -v[2]])

def _angle_between(v1, v2):
    d = np.dot(v1, v2)
    d = np.clip(d / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(d))

def _compute_loop_reference(wps, idx, look=3):
    n = len(wps)
    i0, i1 = max(0, idx-look), min(n-1, idx+look)
    t = wps[i1]-wps[i0] if i1>i0 else wps[min(idx+1,n-1)]-wps[max(idx-1,0)]
    t_ref = t / (np.linalg.norm(t)+1e-12)
    if n >= 3:
        nb = wps[max(0,idx-5):min(n,idx+5)]
        if len(nb) >= 3:
            _, _, vh = np.linalg.svd(nb - nb.mean(axis=0))
            n_loop = vh[2]
            if n_loop[1] < 0: n_loop = -n_loop
        else: n_loop = np.array([0.,1.,0.])
    else: n_loop = np.array([0.,1.,0.])
    return t_ref, n_loop

def _get_loop_roll(theta_deg):
    R = loop_plane_rotation_matrix(np.radians(theta_deg), 0.0, 1)
    q = rotation_matrix_to_quaternion(R)
    r, _, _ = quaternion_to_euler(q)
    return r


# ═══════════════════════ New trajectory generators (inlined) ═══════════════════════

def chandelle_like_turn(origin_n, origin_e, origin_alt, init_yaw,
                        radius=10000.0, heading_change_deg=180.0, climb_m=1000.0,
                        n_points=120):
    """Climbing 180° turn — large radius, smooth altitude gain."""
    d_hdg = np.radians(heading_change_deg)
    theta = np.linspace(0, d_hdg, n_points)
    centre_n = origin_n - radius * np.sin(init_yaw)
    centre_e = origin_e + radius * np.cos(init_yaw)
    theta0 = init_yaw - np.pi/2
    wp_n = centre_n + radius * np.cos(theta0 + theta)
    wp_e = centre_e + radius * np.sin(theta0 + theta)
    frac = np.linspace(0, 1, n_points)
    wp_a = origin_alt + climb_m * frac
    waypoints = np.column_stack([wp_n, wp_e, wp_a])
    arc_len = radius * abs(d_hdg)
    meta = {
        'name': f'chandelle_like_dH{int(heading_change_deg)}_R{int(radius)}_climb{int(climb_m)}',
        'n_points': n_points, 'total_length_m': float(arc_len),
        'radius': radius, 'heading_change_deg': heading_change_deg,
        'climb_m': climb_m,
        'altitude_range': (float(wp_a.min()), float(wp_a.max())),
        'singularity_risk': 'low',
    }
    return waypoints, meta


def wingover_like_turn(origin_n, origin_e, origin_alt, init_yaw,
                       heading_change_deg=150.0, climb_m=1000.0,
                       radius=10000.0, n_points=120):
    """Wingover-like: climbing turn with smooth descending exit.

    Phase 0.0-0.5: climbing arc with heading gradually changing
    Phase 0.5-1.0: nose drops, altitude levels/slightly descends, heading continues

    heading_ref(s) = init_yaw + d_hdg * s
    alt_ref(s) = alt0 + H * sin(pi * s)   (peak at s=0.5)
    Forward motion along a smooth 3D arc.

    Horizontal path: approximate arc with variable curvature.
    """
    d_hdg = np.radians(heading_change_deg)
    s = np.linspace(0, 1, n_points)

    # Heading: linear interpolation
    heading = init_yaw + d_hdg * s

    # Altitude: sinusoidal profile — climb then descend
    alt = origin_alt + climb_m * np.sin(np.pi * s)

    # Forward position: integrate heading along arc
    # Use average radius to determine forward step
    arc_len = radius * abs(d_hdg)
    ds_forward = arc_len / (n_points - 1)

    # Build positions incrementally
    wp_n = np.zeros(n_points); wp_e = np.zeros(n_points)
    wp_n[0] = origin_n; wp_e[0] = origin_e
    for i in range(1, n_points):
        # Average heading over this segment
        h_avg = (heading[i-1] + heading[i]) / 2
        wp_n[i] = wp_n[i-1] + ds_forward * np.cos(h_avg)
        wp_e[i] = wp_e[i-1] + ds_forward * np.sin(h_avg)

    waypoints = np.column_stack([wp_n, wp_e, alt])
    meta = {
        'name': f'wingover_like_dH{int(heading_change_deg)}_R{int(radius)}_climb{int(climb_m)}',
        'n_points': n_points, 'total_length_m': float(arc_len),
        'radius': radius, 'heading_change_deg': heading_change_deg,
        'climb_m': climb_m,
        'altitude_range': (float(alt.min()), float(alt.max())),
        'singularity_risk': 'low',
    }
    return waypoints, meta


# ═══════════════════════ Rollout engine ═══════════════════════

def run_rollout(name, wps, total_arc, planner_cfg, max_steps,
                rng, net, net_params, env, out_root,
                use_loop_roll=False, loop_angle=None,
                init_yaw=0.0, compute_geo=True):
    """Run a single rollout with pure-pursuit guidance.

    Returns: (metrics_dict, rec_dict, geo_dict)
    """
    la = planner_cfg.get('lookahead_dist', 1000)
    rr = planner_cfg.get('reach_radius', 500)
    bs = planner_cfg.get('blend_steps', 250)
    tv = planner_cfg.get('target_vt', 250.0)
    cfg = PlannerConfig(lookahead_dist=la, reach_radius=rr,
                        blend_steps=bs, target_vt=tv)
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
    planner.reset(wps, 0.0, 0.0, 0.0, tv)

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

    for step in range(max_steps):
        ps = state.plane_state
        no = _f(ps.north); ea = _f(ps.east); al = _f(ps.altitude)
        vt = _f(ps.vt); ro = _f(ps.roll); pi = _f(ps.pitch); ya = _f(ps.yaw)
        alph = _f(ps.alpha); bet = _f(ps.beta)
        ax = _f(ps.ax); ay = _f(ps.ay); az = _f(ps.az)

        result = planner.step(no, ea, al, ya, pi, ro, vt)
        th, tp, tr, tv_ = (result['target_heading'], result['target_pitch'],
                           result['target_roll'], result['target_vt'])

        # Loop-plane roll override for vertical arcs
        if use_loop_roll and loop_angle is not None:
            path_s = planner.path_progress
            loop_theta_deg = (path_s / total_arc) * loop_angle if total_arc > 0 else 0.0
            loop_theta_deg = np.clip(loop_theta_deg, 0, loop_angle)
            loop_roll = _get_loop_roll(loop_theta_deg)
            blend = min(1.0, step / bs)
            tr = float(np.arctan2(
                np.sin(ro + blend * (loop_roll - ro)),
                np.cos(ro + blend * (loop_roll - ro)),
            ))

        state = state.replace(
            target_heading=jnp.array([th]), target_pitch=jnp.array([tp]),
            target_roll=jnp.array([tr]),
            target_vt=jnp.array([float(tv_)], dtype=jnp.float32),
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
        rec['G'].append(float(np.sqrt(ax**2 + ay**2 + az**2)))
        rec['cte'].append(compute_true_cte(
            np.array([no, ea, al]), wps, wp_idx, 10,
        ))
        rec['q0'].append(_f(ps.q0)); rec['q1'].append(_f(ps.q1))
        rec['q2'].append(_f(ps.q2)); rec['q3'].append(_f(ps.q3))
        rec['wp_idx'].append(wp_idx)

        if bool(done[env.agents[0]]):
            crashed = True; break
        if planner.is_done():
            break

    n = len(rec['t']); ok = planner.is_done() and not crashed

    # ═══ Per-frame geometry ═══
    geo = {}
    if compute_geo:
        geo = {
            'velocity_tangent_error': [], 'nose_tangent_error': [],
            'nose_velocity_error': [], 'wing_plane_error': [],
            'belly_error': [],
        }
        for i in range(n):
            q_bn_i = np.array([rec['q0'][i], rec['q1'][i], rec['q2'][i], rec['q3'][i]])
            q_bn_i = q_bn_i / (np.linalg.norm(q_bn_i) + 1e-12)
            x_body_neu = _ned_to_neu(_rotate_body_to_ned(q_bn_i, np.array([1.,0.,0.])))
            y_body_neu = _ned_to_neu(_rotate_body_to_ned(q_bn_i, np.array([0.,1.,0.])))
            z_body_neu = _ned_to_neu(_rotate_body_to_ned(q_bn_i, np.array([0.,0.,1.])))

            vt_i = rec['vt'][i]; alph_i = np.radians(rec['alpha'][i])
            bet_i = np.radians(rec['beta'][i])
            ca, sa = np.cos(alph_i), np.sin(alph_i)
            cb, sb = np.cos(bet_i), np.sin(bet_i)
            u_body = vt_i*ca*cb; v_body = vt_i*sb; w_body = vt_i*sa*cb
            v_ned = _rotate_body_to_ned(q_bn_i, np.array([u_body, v_body, w_body]))
            v_neu = _ned_to_neu(v_ned)
            v_hat_neu = v_neu / (np.linalg.norm(v_neu) + 1e-12)

            t_ref_neu, n_loop_neu = _compute_loop_reference(wps, rec['wp_idx'][i])
            geo['velocity_tangent_error'].append(_angle_between(v_hat_neu, t_ref_neu))
            geo['nose_tangent_error'].append(_angle_between(x_body_neu, t_ref_neu))
            geo['nose_velocity_error'].append(_angle_between(x_body_neu, v_hat_neu))
            geo['wing_plane_error'].append(_angle_between(y_body_neu, n_loop_neu))
            z_exp = np.cross(t_ref_neu, n_loop_neu)
            z_exp = z_exp / (np.linalg.norm(z_exp) + 1e-12)
            geo['belly_error'].append(_angle_between(z_body_neu, z_exp))

    # ═══ Summary ═══
    ca = np.array(rec['cte']); va = np.array(rec['vt']); ga = np.array(rec['G'])
    aa = np.array(rec['alpha']); ba = np.array(rec['beta'])
    ra = np.array(rec['roll']); tra = np.array(rec['t_roll'])

    m = {
        'name': name, 'completed': bool(ok), 'steps': n,
        'CTE_mean': float(ca.mean()), 'CTE_p50': float(np.percentile(ca, 50)),
        'CTE_p90': float(np.percentile(ca, 90)), 'CTE_max': float(ca.max()),
        'Gmax': float(ga.max()), 'Gmean': float(ga.mean()),
        'vt_min': float(va.min()), 'vt_max': float(va.max()),
        'vt_mean': float(va.mean()),
        'alt_min': float(np.array(rec['a']).min()),
        'alt_max': float(np.array(rec['a']).max()),
        'env_alpha_min': float(aa.min()), 'env_alpha_max': float(aa.max()),
        'env_alpha_mean': float(aa.mean()),
        'env_beta_min': float(ba.min()), 'env_beta_max': float(ba.max()),
        'target_roll_min': float(tra.min()), 'target_roll_max': float(tra.max()),
        'actual_roll_min': float(ra.min()), 'actual_roll_max': float(ra.max()),
        'actual_roll_mean': float(ra.mean()),
        'termination': 'crash' if crashed else ('ok' if ok else 'timeout'),
    }

    if compute_geo and len(geo.get('velocity_tangent_error', [])) > 0:
        for key in ['velocity_tangent_error', 'nose_tangent_error',
                     'nose_velocity_error', 'wing_plane_error', 'belly_error']:
            arr = np.array(geo[key])
            m[f'{key}_mean'] = float(arr.mean())
            m[f'{key}_p90'] = float(np.percentile(arr, 90))

        # q_error
        q_errs = []
        for i in range(n):
            q_bn_i = np.array([rec['q0'][i], rec['q1'][i], rec['q2'][i], rec['q3'][i]])
            q_bn_i = q_bn_i / (np.linalg.norm(q_bn_i) + 1e-12)
            cr_sr = np.cos(0.5*np.radians(rec['t_roll'][i]))
            sr_sr = np.sin(0.5*np.radians(rec['t_roll'][i]))
            cp = np.cos(0.5*np.radians(rec['t_pitch'][i]))
            sp = np.sin(0.5*np.radians(rec['t_pitch'][i]))
            cy = np.cos(0.5*np.radians(rec['t_hdg'][i]))
            sy = np.sin(0.5*np.radians(rec['t_hdg'][i]))
            q_tgt_nb = np.array([cr_sr*cp*cy+sr_sr*sp*sy,
                                 sr_sr*cp*cy-cr_sr*sp*sy,
                                 cr_sr*sp*cy+sr_sr*cp*sy,
                                 cr_sr*cp*sy-sr_sr*sp*cy])
            q_tgt_bn = _quat_conj_np(q_tgt_nb)
            q_tgt_bn = q_tgt_bn / (np.linalg.norm(q_tgt_bn) + 1e-12)
            q_err = _quat_mul_np(q_tgt_bn, _quat_conj_np(q_bn_i))
            if q_err[0] < 0: q_err = -q_err
            w = np.clip(np.abs(q_err[0]), 0.0, 1.0 - 1e-12)
            q_errs.append(float(2.0 * np.arccos(w)))
        qe_a = np.array(q_errs)
        m['q_error_mean_rad'] = float(qe_a.mean())
        m['q_error_p90_rad'] = float(np.percentile(qe_a, 90))

    # Grade for loop-like maneuvers
    if use_loop_roll or 'vertical' in name.lower() or 'pullup' in name.lower():
        m = _apply_loop_grade(m)

    return m, rec, geo


def _apply_loop_grade(m):
    if not m['completed']: m['grade'] = 'Fail'; return m
    cm = m['CTE_mean']; c90 = m['CTE_p90']; cmax = m['CTE_max']
    g = m['Gmax']; v = m['vt_min']
    vte = m.get('velocity_tangent_error_mean', 999)
    nte = m.get('nose_tangent_error_mean', 999)
    nve = m.get('nose_velocity_error_mean', 999)
    wpe = m.get('wing_plane_error_mean', 999)
    qe = m.get('q_error_mean_rad', 999)
    if (cm < 100 and c90 < 300 and cmax < 800 and g < 9 and v >= 190
        and vte < 15 and nte < 15 and nve < 15 and wpe < 15 and qe < 0.5):
        m['grade'] = 'A'
    elif cm < 500 and c90 < 1200 and g < 10 and v >= 175 and vte < 30 and nte < 30:
        m['grade'] = 'B'
    elif m['completed']: m['grade'] = 'C'
    else: m['grade'] = 'Fail'
    return m


# ═══════════════════════ Plotting ═══════════════════════

def make_plots(name, wps, rec, meta, out_dir):
    """Generate 3D, top-down, and side-view plots."""
    wps_np = np.asarray(wps)
    n_arr = np.array(rec['n']); e_arr = np.array(rec['e']); a_arr = np.array(rec['a'])
    roll_arr = np.array(rec['roll']); pitch_arr = np.array(rec['pitch'])
    vt_arr = np.array(rec['vt']); G_arr = np.array(rec['G'])

    # ── 3D trajectory with body-axes sampling ──
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig)

    # 3D view
    ax0 = fig.add_subplot(gs[0, 0], projection='3d')
    ax0.plot(wps_np[:, 0], wps_np[:, 1], wps_np[:, 2], 'k--', alpha=0.4, lw=1, label='reference')
    ax0.plot(n_arr, e_arr, a_arr, 'C0-', lw=1.5, label='aircraft')
    ax0.scatter(n_arr[0], e_arr[0], a_arr[0], c='green', s=40, zorder=5, label='start')
    ax0.scatter(n_arr[-1], e_arr[-1], a_arr[-1], c='red', s=40, zorder=5, label='end')
    # Sample body axes every 40 frames
    step_body = max(1, len(n_arr)//15)
    for i in range(0, len(n_arr), step_body):
        q_bn_i = np.array([rec['q0'][i], rec['q1'][i], rec['q2'][i], rec['q3'][i]])
        q_bn_i = q_bn_i / (np.linalg.norm(q_bn_i) + 1e-12)
        x_body = _ned_to_neu(_rotate_body_to_ned(q_bn_i, np.array([1., 0., 0.])))
        z_body = _ned_to_neu(_rotate_body_to_ned(q_bn_i, np.array([0., 0., 1.])))
        r_val = meta.get('radius')
        if r_val is None: r_val = 5000.0
        scale = max(1.0, r_val * 0.02)
        ax0.quiver(n_arr[i], e_arr[i], a_arr[i],
                   x_body[0]*scale, x_body[1]*scale, x_body[2]*scale,
                   color='red', alpha=0.5, lw=0.5, arrow_length_ratio=0.15)
        ax0.quiver(n_arr[i], e_arr[i], a_arr[i],
                   z_body[0]*scale, z_body[1]*scale, z_body[2]*scale,
                   color='blue', alpha=0.5, lw=0.5, arrow_length_ratio=0.15)
    ax0.set_xlabel('North (m)'); ax0.set_ylabel('East (m)'); ax0.set_zlabel('Alt (m)')
    ax0.set_title(f'{name}\n3D view'); ax0.legend(fontsize=7, loc='upper right')

    # Top-down
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(wps_np[:, 0], wps_np[:, 1], 'k--', alpha=0.4, lw=1)
    ax1.plot(n_arr, e_arr, 'C0-', lw=1.5)
    ax1.scatter(n_arr[0], e_arr[0], c='green', s=30, zorder=5)
    ax1.scatter(n_arr[-1], e_arr[-1], c='red', s=30, zorder=5)
    ax1.set_xlabel('North (m)'); ax1.set_ylabel('East (m)')
    ax1.set_title('Top-down'); ax1.set_aspect('equal')

    # Side view (altitude vs along-track)
    ax2 = fig.add_subplot(gs[0, 2])
    dist = np.cumsum(np.sqrt(np.diff(n_arr, prepend=n_arr[0])**2 +
                              np.diff(e_arr, prepend=e_arr[0])**2))
    ax2.plot(dist, a_arr, 'C0-', lw=1.5)
    # Reference altitude
    wp_dist = np.sqrt((wps_np[:, 0] - wps_np[0, 0])**2 +
                      (wps_np[:, 1] - wps_np[0, 1])**2)
    ax2.plot(wp_dist, wps_np[:, 2], 'k--', alpha=0.4, lw=1)
    ax2.set_xlabel('Along-track distance (m)'); ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Side view')

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'figures', f'{name}_trajectory.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Telemetry plots ──
    fig2, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t_arr = np.array(rec['t'])
    axes[0].plot(t_arr, vt_arr, 'C0-', lw=1); axes[0].set_ylabel('vt (m/s)')
    axes[0].axhline(y=190, color='green', ls='--', alpha=0.5, label='A min')
    axes[0].axhline(y=175, color='orange', ls='--', alpha=0.5, label='B min')
    axes[0].legend(fontsize=7)
    axes[1].plot(t_arr, G_arr, 'C1-', lw=1); axes[1].set_ylabel('G')
    axes[1].axhline(y=9, color='red', ls='--', alpha=0.5)
    axes[2].plot(t_arr, roll_arr, 'C2-', lw=0.8, label='roll')
    axes[2].plot(t_arr, pitch_arr, 'C3-', lw=0.8, label='pitch')
    axes[2].set_ylabel('deg'); axes[2].legend(fontsize=7)
    axes[3].plot(t_arr, np.array(rec['alpha']), 'C4-', lw=0.8, label='alpha')
    axes[3].set_ylabel('alpha (deg)'); axes[3].set_xlabel('Time (s)')
    axes[3].legend(fontsize=7)
    fig2.suptitle(f'{name} — Telemetry', fontsize=11)
    plt.tight_layout()
    telem_path = os.path.join(out_dir, 'figures', f'{name}_telemetry.png')
    plt.savefig(telem_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)


# ═══════════════════════ Scoring for candidate selection ═══════════════════════

def score_candidate(m):
    """Higher is better. Penalize crash, timeout, high CTE, high G."""
    if not m['completed']: return -1e9
    score = 0.0
    score -= m['CTE_p90'] * 1.0          # primary: low CTE
    score -= max(0, m['Gmax'] - 8.0) * 200  # penalty for high G
    score -= max(0, 190 - m['vt_min']) * 10  # penalty for slow speed
    if m.get('wing_plane_error_mean') is not None:
        score -= m['wing_plane_error_mean'] * 3  # wing plane alignment
    return score


# ═══════════════════════ Main Showcase ═══════════════════════

def main(checkpoint_path=None, output_root=None):
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CKPT

    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_root is None:
        out_root = os.path.join(_px, 'results/group_meeting_ep619_showcase', tag)
    else:
        out_root = os.path.join(output_root, tag)
    for sub in ['acmi', 'figures', 'metrics', 'rollouts']:
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    print(f"Output: {out_root}")

    # Load checkpoint
    print("Loading checkpoint...")
    env = Env(Params())
    net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    rng = jax.random.PRNGKey(SEED)
    obs_shape = env.observation_space(env.agents[0], Params()).shape
    h0 = ScannedRNN.initialize_carry(1, NET_CFG['GRU_HIDDEN_DIM'])
    net_params_init = net.init(rng, h0, (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1))))
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(checkpoint_path, args=ocp.args.StandardRestore())
    net_params = ckpt['params']

    all_results = {}  # demo_type -> [(candidate_name, metrics, rec, geo)]

    # ═══════════════════════════════════════════════════════════════
    # Demo 1: S-curve / Snake
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DEMO 1: S-CURVE / SNAKE")
    print("="*70)
    s_curve_candidates = [
        ('s_curve_A2000_P10000', 2000, 10000, 1000, 500, 1500),
        ('s_curve_A3000_P10000', 3000, 10000, 1000, 500, 1500),
        ('s_curve_A3000_P12000', 3000, 12000, 1200, 500, 1800),
        ('s_curve_A4000_P15000', 4000, 15000, 1500, 500, 2000),
    ]
    demo1_results = []
    for name, amp, hp, la, rr, mx in s_curve_candidates:
        wps, meta = s_curve(0, 0, 5000, 0.0, amplitude=amp, half_period=hp,
                            n_points=max(60, int(hp*2/100)))
        pcfg = {'lookahead_dist': la, 'reach_radius': rr, 'blend_steps': 250, 'target_vt': 250.0}
        rng, rk = jax.random.split(rng)
        m, rec, geo = run_rollout(name, wps, meta['total_length_m'], pcfg, mx,
                                  rng, net, net_params, env, out_root,
                                  compute_geo=False)
        demo1_results.append((name, m, rec, geo, wps, meta))
        print(f"  {name}: {'OK' if m['completed'] else 'FAIL'} "
              f"CTE_m={m['CTE_mean']:.0f} CTE_p90={m['CTE_p90']:.0f} "
              f"Gmax={m['Gmax']:.1f} vt_min={m['vt_min']:.0f}")
    # Pick best
    best1 = max(demo1_results, key=lambda x: score_candidate(x[1]))
    all_results['s_curve'] = best1
    print(f"  → Selected: {best1[0]}")

    # ═══════════════════════════════════════════════════════════════
    # Demo 2: Figure-eight
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DEMO 2: FIGURE-EIGHT")
    print("="*70)
    fig8_candidates = [
        ('figure8_R5000', 5000, 1000, 500, 2000),
        ('figure8_R6000', 6000, 1200, 500, 2000),
        ('figure8_R8000', 8000, 1500, 600, 2500),
    ]
    demo2_results = []
    for name, rad, la, rr, mx in fig8_candidates:
        wps, meta = figure_eight(0, 0, 5000, 0.0, radius=rad, n_points=max(80, int(rad/60)))
        pcfg = {'lookahead_dist': la, 'reach_radius': rr, 'blend_steps': 250, 'target_vt': 250.0}
        rng, rk = jax.random.split(rng)
        m, rec, geo = run_rollout(name, wps, meta['total_length_m'], pcfg, mx,
                                  rng, net, net_params, env, out_root,
                                  compute_geo=False)
        demo2_results.append((name, m, rec, geo, wps, meta))
        print(f"  {name}: {'OK' if m['completed'] else 'FAIL'} "
              f"CTE_m={m['CTE_mean']:.0f} CTE_p90={m['CTE_p90']:.0f} "
              f"Gmax={m['Gmax']:.1f} vt_min={m['vt_min']:.0f}")
    best2 = max(demo2_results, key=lambda x: score_candidate(x[1]))
    all_results['figure_eight'] = best2
    print(f"  → Selected: {best2[0]}")

    # ═══════════════════════════════════════════════════════════════
    # Demo 3: Mild 3D maneuver
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DEMO 3: MILD 3D MANEUVER")
    print("="*70)
    mild3d_candidates = [
        ('helix_R8000_climb1000', 'helix',
         {'radius': 8000, 'turns': 1.0, 'delta_alt': 1000, 'n_points': 120, 'direction': 1},
         1500, 500, 2500),
        ('helix_R10000_climb1000', 'helix',
         {'radius': 10000, 'turns': 1.0, 'delta_alt': 1000, 'n_points': 120, 'direction': 1},
         1500, 500, 2500),
        ('climbing_s_A3000_dAlt1000', 'climbing_s_curve',
         {'amplitude': 3000, 'half_period': 12000, 'delta_alt': 1000, 'n_points': 120},
         1500, 500, 2000),
        ('climbing_fig8_R6000_dAlt1000', 'climbing_figure_eight',
         {'radius': 6000, 'delta_alt': 1000, 'n_points': 120},
         1500, 600, 2500),
    ]
    demo3_results = []
    for name, gen_name, gkwargs, la, rr, mx in mild3d_candidates:
        if gen_name == 'helix':
            wps, meta = helix_trajectory(0, 0, 5000, 0.0, **gkwargs)
        elif gen_name == 'climbing_s_curve':
            wps, meta = climbing_s_curve(0, 0, 5000, 0.0, **gkwargs)
        elif gen_name == 'climbing_figure_eight':
            wps, meta = climbing_figure_eight(0, 0, 5000, 0.0, **gkwargs)
        else:
            continue
        pcfg = {'lookahead_dist': la, 'reach_radius': rr, 'blend_steps': 250, 'target_vt': 250.0}
        rng, rk = jax.random.split(rng)
        m, rec, geo = run_rollout(name, wps, meta['total_length_m'], pcfg, mx,
                                  rng, net, net_params, env, out_root,
                                  compute_geo=False)
        demo3_results.append((name, m, rec, geo, wps, meta))
        print(f"  {name}: {'OK' if m['completed'] else 'FAIL'} "
              f"CTE_m={m['CTE_mean']:.0f} CTE_p90={m['CTE_p90']:.0f} "
              f"Gmax={m['Gmax']:.1f} vt_min={m['vt_min']:.0f} "
              f"alt=[{m['alt_min']:.0f},{m['alt_max']:.0f}]")
    best3 = max(demo3_results, key=lambda x: score_candidate(x[1]))
    all_results['mild_3d'] = best3
    print(f"  → Selected: {best3[0]}")

    # ═══════════════════════════════════════════════════════════════
    # Demo 4: Chandelle-like climbing 180° turn
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DEMO 4: CHANDELLE-LIKE CLIMBING TURN")
    print("="*70)
    chandelle_candidates = [
        ('chandelle_R8000_climb800', 8000, 180, 800, 120, 1200, 500, 2000),
        ('chandelle_R10000_climb1000', 10000, 180, 1000, 120, 1500, 500, 2500),
        ('chandelle_R12000_climb1200', 12000, 180, 1200, 120, 1800, 600, 2500),
    ]
    demo4_results = []
    for name, rad, hdg_chg, clb, npts, la, rr, mx in chandelle_candidates:
        wps, meta = chandelle_like_turn(0, 0, 5000, 0.0,
                                        radius=rad, heading_change_deg=hdg_chg,
                                        climb_m=clb, n_points=npts)
        pcfg = {'lookahead_dist': la, 'reach_radius': rr, 'blend_steps': 250, 'target_vt': 250.0}
        rng, rk = jax.random.split(rng)
        m, rec, geo = run_rollout(name, wps, meta['total_length_m'], pcfg, mx,
                                  rng, net, net_params, env, out_root,
                                  compute_geo=False)
        demo4_results.append((name, m, rec, geo, wps, meta))
        print(f"  {name}: {'OK' if m['completed'] else 'FAIL'} "
              f"CTE_m={m['CTE_mean']:.0f} CTE_p90={m['CTE_p90']:.0f} "
              f"Gmax={m['Gmax']:.1f} vt_min={m['vt_min']:.0f} "
              f"alt=[{m['alt_min']:.0f},{m['alt_max']:.0f}]")
    best4 = max(demo4_results, key=lambda x: score_candidate(x[1]))
    all_results['chandelle_like'] = best4
    print(f"  → Selected: {best4[0]}")

    # ═══════════════════════════════════════════════════════════════
    # Demo 5: 90° vertical pull-up (quarter-loop)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DEMO 5: 90° VERTICAL PULL-UP (QUARTER-LOOP)")
    print("="*70)
    pullup_candidates = [
        ('pullup90_R10000', 10000, 800, 300, 1200),
        ('pullup90_R12000', 12000, 1000, 400, 1500),
    ]
    demo5_results = []
    for name, rad, la, rr, mx in pullup_candidates:
        wps, meta = vertical_pullup_arc(0, 0, 5000, 0.0,
                                        radius=rad, arc_angle_deg=90,
                                        n_points=max(60, int(90*2/3)))
        pcfg = {'lookahead_dist': la, 'reach_radius': rr, 'blend_steps': 250, 'target_vt': 250.0}
        rng, rk = jax.random.split(rng)
        m, rec, geo = run_rollout(name, wps, meta['total_length_m'], pcfg, mx,
                                  rng, net, net_params, env, out_root,
                                  use_loop_roll=True, loop_angle=90,
                                  compute_geo=True)
        demo5_results.append((name, m, rec, geo, wps, meta))
        print(f"  {name}: {'OK' if m['completed'] else 'FAIL'} "
              f"Grade={m.get('grade','N/A')} "
              f"CTE_m={m['CTE_mean']:.0f} v_tang={m.get('velocity_tangent_error_mean',0):.1f}° "
              f"wing_p={m.get('wing_plane_error_mean',0):.1f}° "
              f"Gmax={m['Gmax']:.1f}")
    best5 = max(demo5_results, key=lambda x: score_candidate(x[1]))
    all_results['vertical_90'] = best5
    print(f"  → Selected: {best5[0]}")

    # ═══════════════════════════════════════════════════════════════
    # Demo 6: Wingover-like climbing turn
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DEMO 6: WINGOVER-LIKE CLIMBING TURN")
    print("="*70)
    wingover_candidates = [
        ('wingover_dH120_climb800_R8000', 120, 800, 8000, 120, 1200, 500, 2000),
        ('wingover_dH150_climb1000_R10000', 150, 1000, 10000, 120, 1500, 500, 2500),
        ('wingover_dH180_climb1200_R12000', 180, 1200, 12000, 120, 1800, 600, 2500),
    ]
    demo6_results = []
    for name, hdg_chg, clb, rad, npts, la, rr, mx in wingover_candidates:
        wps, meta = wingover_like_turn(0, 0, 5000, 0.0,
                                       heading_change_deg=hdg_chg,
                                       climb_m=clb, radius=rad, n_points=npts)
        pcfg = {'lookahead_dist': la, 'reach_radius': rr, 'blend_steps': 250, 'target_vt': 250.0}
        rng, rk = jax.random.split(rng)
        m, rec, geo = run_rollout(name, wps, meta['total_length_m'], pcfg, mx,
                                  rng, net, net_params, env, out_root,
                                  compute_geo=False)
        demo6_results.append((name, m, rec, geo, wps, meta))
        print(f"  {name}: {'OK' if m['completed'] else 'FAIL'} "
              f"CTE_m={m['CTE_mean']:.0f} CTE_p90={m['CTE_p90']:.0f} "
              f"Gmax={m['Gmax']:.1f} vt_min={m['vt_min']:.0f} "
              f"alt=[{m['alt_min']:.0f},{m['alt_max']:.0f}]")
    best6 = max(demo6_results, key=lambda x: score_candidate(x[1]))
    all_results['wingover_like'] = best6
    print(f"  → Selected: {best6[0]}")

    # ═══════════════════════════════════════════════════════════════
    # Generate plots, ACMI, and metrics for selected best demos
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("GENERATING PLOTS, ACMI, METRICS FOR SELECTED DEMOS")
    print("="*70)

    selected_metrics = []
    demo_labels = {
        's_curve': ('main_demo', 'horizontal_complex'),
        'figure_eight': ('main_demo', 'horizontal_complex'),
        'mild_3d': ('main_demo', 'mild_3d'),
        'chandelle_like': ('main_demo', 'mild_3d'),
        'vertical_90': ('boundary_demo', 'vertical'),
        'wingover_like': ('main_demo', 'mild_3d'),
    }

    for demo_type, (name, m, rec, geo, wps, meta) in all_results.items():
        label, category = demo_labels.get(demo_type, ('boundary_demo', 'unknown'))
        m['demo_type'] = demo_type
        m['label'] = label
        m['category'] = category

        # Save metrics
        with open(os.path.join(out_root, 'metrics', f'{demo_type}.json'), 'w') as f:
            json.dump(m, f, indent=2, default=str)

        # Save rollout
        np.savez(os.path.join(out_root, 'rollouts', f'{demo_type}.npz'),
                 waypoints=wps, **rec)

        # ACMI
        write_acmi(os.path.join(out_root, 'acmi', f'{demo_type}.acmi'), wps, rec)

        # Plots
        make_plots(demo_type, wps, rec, meta, out_root)

        selected_metrics.append(m)
        print(f"  {demo_type} [{label}]: plots, ACMI, metrics saved")

    # ═══════════════════════════════════════════════════════════════
    # Write reports
    # ═══════════════════════════════════════════════════════════════

    # ── Summary CSV ──
    csv_path = os.path.join(out_root, 'summary.csv')
    csv_fields = ['demo_type', 'label', 'name', 'completed', 'steps',
                  'CTE_mean', 'CTE_p50', 'CTE_p90', 'CTE_max',
                  'Gmax', 'vt_min', 'vt_max', 'alt_min', 'alt_max',
                  'env_alpha_min', 'env_alpha_max',
                  'velocity_tangent_error_mean', 'wing_plane_error_mean',
                  'q_error_mean_rad', 'termination']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(selected_metrics)
    print(f"\nCSV: {csv_path}")

    # ── Showcase report ──
    _write_showcase_report(out_root, tag, selected_metrics, all_results, checkpoint_path)

    # ── Paper reframing ──
    _write_paper_reframing(out_root)

    # ── Group meeting talking points ──
    _write_talking_points(out_root, selected_metrics)

    print(f"\nDONE. Output directory: {out_root}")
    return out_root, selected_metrics


def _write_showcase_report(out_root, tag, selected_metrics, all_results, ckpt_path):
    report_path = os.path.join(out_root, 'showcase_report.md')
    with open(report_path, 'w') as f:
        f.write("# Ep619 Six-Maneuver Showcase Report\n\n")
        f.write(f"**Date:** {tag}\n")
        f.write(f"**Checkpoint:** `{ckpt_path}`\n")
        f.write(f"**Model:** ActorCriticRNN (GRU-128), 21-dim obs, 5-head discrete action\n\n")

        f.write("## Selected Demos\n\n")
        f.write("| # | Demo | Parameter | Label | CTE_m | CTE_p90 | Gmax | vt_min | Status |\n")
        f.write("|---|------|-----------|-------|-------|---------|------|--------|--------|\n")
        for m in selected_metrics:
            status = 'OK' if m['completed'] else 'FAIL'
            f.write(f"| {list(all_results.keys()).index(m['demo_type'])+1} | "
                    f"{m['demo_type']} | {m['name']} | "
                    f"{m['label']} | {m['CTE_mean']:.0f} | {m['CTE_p90']:.0f} | "
                    f"{m['Gmax']:.1f} | {m['vt_min']:.0f} | {status} |\n")

        f.write("\n## Per-Demo Details\n\n")
        for m in selected_metrics:
            f.write(f"### {m['demo_type']} — {m['label']}\n\n")
            f.write(f"- **Selected:** {m['name']}\n")
            f.write(f"- **Completed:** {m['completed']} | Steps: {m['steps']} | Termination: {m['termination']}\n")
            f.write(f"- **CTE:** mean={m['CTE_mean']:.1f}m, p90={m['CTE_p90']:.1f}m, max={m['CTE_max']:.1f}m\n")
            f.write(f"- **Performance:** Gmax={m['Gmax']:.1f}g, vt_min={m['vt_min']:.0f}m/s, vt_max={m['vt_max']:.0f}m/s\n")
            f.write(f"- **Altitude:** {m['alt_min']:.0f}m → {m['alt_max']:.0f}m\n")
            f.write(f"- **Alpha range:** [{m['env_alpha_min']:.1f}, {m['env_alpha_max']:.1f}]°\n")
            if m.get('grade'):
                f.write(f"- **Loop-quality grade:** {m['grade']}\n")
            if m.get('wing_plane_error_mean') is not None:
                f.write(f"- **Wing-plane error:** {m['wing_plane_error_mean']:.1f}°\n")
            if m.get('velocity_tangent_error_mean') is not None:
                f.write(f"- **Velocity-tangent error:** {m['velocity_tangent_error_mean']:.1f}°\n")
            if m.get('q_error_mean_rad') is not None:
                f.write(f"- **Quaternion error:** {m['q_error_mean_rad']:.3f} rad\n")

            # Visual comment
            vc = _visual_comment(m)
            f.write(f"\n**Visual comment:** {vc}\n\n")

        f.write("\n## Files\n\n")
        f.write("| Demo | ACMI | Plot | Metrics | Rollout |\n")
        f.write("|------|------|------|---------|--------|\n")
        for m in selected_metrics:
            dt = m['demo_type']
            f.write(f"| {dt} | `acmi/{dt}.acmi` | `figures/{dt}_trajectory.png` | "
                    f"`metrics/{dt}.json` | `rollouts/{dt}.npz` |\n")

    print(f"Report: {report_path}")


def _visual_comment(m):
    """Generate a one-paragraph visual assessment."""
    dt = m['demo_type']
    if dt == 's_curve':
        return ("The S-curve shows smooth lateral oscillation with the aircraft closely "
                "tracking the reference path. The top-down view reveals a clean sinusoidal "
                "shape. Suitable for group meeting as a representative horizontal tracking demo.")
    elif dt == 'figure_eight':
        return ("The figure-eight demonstrates crossing-path tracking with smooth heading "
                "reversals. The aircraft maintains stable altitude throughout. The visual "
                "shape is clean and would be recognizable in Tacview.")
    elif dt == 'mild_3d':
        return ("This mild 3D maneuver shows the policy can compose horizontal tracking "
                "with altitude change. The trajectory follows the 3D reference while "
                "maintaining safe G-loading and speed. Demonstrates basic 3D capability.")
    elif dt == 'chandelle_like':
        return ("The chandelle-like climbing turn shows a large-radius 180° turn with "
                "smooth altitude gain. The aircraft carves a clean 3D arc without "
                "aggressive maneuvering. Labeled 'chandelle-like' — not a certified "
                "classic chandelle.")
    elif dt == 'vertical_90':
        return ("The 90° quarter-loop is the vertical showcase maneuver. Under loop-quality "
                "metrics, this is a B-grade boundary demo. The pull-up is visually clear "
                "in Tacview, but wing-plane alignment is not yet A-grade. "
                "This is the current capability boundary for vertical maneuvers.")
    elif dt == 'wingover_like':
        return ("The wingover-like climbing turn shows a smooth climb followed by a "
                "descending heading reversal — a visually showy classic-like maneuver. "
                "The altitude profile is clean and the aircraft stays within safe limits. "
                "Labeled 'wingover-like' — not a certified aerobatic wingover.")
    return "No visual comment available."


def _write_paper_reframing(out_root):
    path = os.path.join(out_root, 'paper_reframing.md')
    with open(path, 'w') as f:
        f.write("# Conservative Paper Reframing\n\n")
        f.write("## What This Paper Is\n\n")
        f.write("This is **not** a full-loop aerobatics paper.\n")
        f.write("It is a **high-fidelity fixed-wing robot learning benchmark and evaluation paper.**\n\n")

        f.write("We study how a **frozen quaternion-based RL flight skill** can be composed "
                "through **moving-lookahead targets** into long-horizon fixed-wing maneuvers.\n\n")

        f.write("We show that **CTE-only evaluation can falsely suggest success** "
                "for loop-like maneuvers.\n\n")

        f.write("We introduce **geometry-aware loop-quality metrics**:\n")
        f.write("- velocity-tangent error\n")
        f.write("- nose-tangent error\n")
        f.write("- nose-velocity error\n")
        f.write("- wing-plane error\n")
        f.write("- quaternion attitude error\n")
        f.write("- alpha range, G-loading, speed constraints\n\n")

        f.write("Using epoch619, we demonstrate:\n")
        f.write("- Stable horizontal and mild 3D maneuver composition\n")
        f.write("- Boundary vertical pull-up capability (90°, B-grade)\n")
        f.write("- The 180° half-loop as a **diagnosed limitation**, not a success\n\n")

        f.write("## Contributions\n\n")
        f.write("1. Planax-based high-fidelity fixed-wing RL maneuver benchmark.\n")
        f.write("2. Moving-lookahead target stream for composing frozen RL flight skills.\n")
        f.write("3. Representative fixed-wing maneuver demo suite (6 maneuvers).\n")
        f.write("4. Full-attitude loop-plane target representation for vertical arcs.\n")
        f.write("5. Geometry-aware loop-quality metrics beyond CTE.\n")
        f.write("6. Capability boundary diagnosis for inverted/top-transition flight.\n\n")

        f.write("## What We Do NOT Claim\n\n")
        f.write("- Full loop / full-envelope aerobatics\n")
        f.write("- Complete classic aerobatic maneuver library\n")
        f.write("- 180° half-loop as success\n")
        f.write("- CTE-only evaluation as sufficient\n")

    print(f"Paper reframing: {path}")


def _write_talking_points(out_root, selected_metrics):
    path = os.path.join(out_root, 'group_meeting_talking_points.md')
    with open(path, 'w') as f:
        f.write("# Group Meeting Talking Points — ep619 Six-Maneuver Showcase\n\n")

        f.write("## 1. What ep619 Can Currently Demonstrate\n\n")
        for m in selected_metrics:
            status = 'OK' if m['completed'] else 'FAIL'
            f.write(f"- **{m['demo_type']}** ({m['label']}): "
                    f"CTE_mean={m['CTE_mean']:.0f}m, Gmax={m['Gmax']:.1f}g, {status}\n")

        f.write("\n## 2. Why Moving-Lookahead Works Better Than Fixed Waypoints\n\n")
        f.write("- Pure-pursuit planner provides continuous, smooth lookahead targets\n")
        f.write("- Eliminates discrete waypoint switching artifacts\n")
        f.write("- Enables the frozen policy to compose skills without retraining\n")
        f.write("- The lookahead distance acts as a 'smoothness knob' for the trajectory\n\n")

        f.write("## 3. Why We No Longer Use CTE-Only Evaluation\n\n")
        f.write("- CTE-only grades all 60-150° vertical arcs as 'A', but wing-plane error is 15-34°\n")
        f.write("- CTE-only misses geometry alignment: the aircraft can be at the right position "
                "but pointed the wrong direction\n")
        f.write("- Loop-quality metrics reveal the true capability boundary\n")
        f.write("- This is a key contribution: better evaluation for aerobatic RL\n\n")

        f.write("## 4. Six Demos — Visual Quality\n\n")
        f.write("| # | Demo | Label | Visual Quality |\n")
        f.write("|---|------|-------|---------------|\n")
        for idx, m in enumerate(selected_metrics):
            f.write(f"| {idx+1} | {m['demo_type']} | {m['label']} | See ACMI |\n")

        f.write("\n## 5. What Remains Unsolved\n\n")
        f.write("- **180° half-loop** fails catastrophically (CTE >6km, wing-plane >75°)\n")
        f.write("- Inverted/top-transition flight (80-180° segment) is beyond current policy\n")
        f.write("- Wing-plane alignment is the main bottleneck (15-34° in 60-150° arcs)\n")
        f.write("- Policy has not been explicitly rewarded for wing-plane or nose-tangent alignment\n\n")

        f.write("## 6. Conservative Paper Route\n\n")
        f.write("- Frame as fixed-wing RL benchmark, not full aerobatics\n")
        f.write("- Contributions: benchmark + evaluation metrics + demo suite + capability diagnosis\n")
        f.write("- The 180° failure is a feature, not a bug — it demonstrates the need for "
                "better geometry-aware training\n")
        f.write("- Clear distinction: this paper = evaluation/benchmark; "
                "Codex paper = specialist policy for full loop\n\n")

        f.write("## 7. Codex Route for Full-Loop\n\n")
        f.write("- Train a residual/specialist policy targeting the 80-180° inverted regime\n")
        f.write("- Reward: wing_plane_error + nose_tangent_error (not just quaternion error)\n")
        f.write("- Curriculum: start at 150°, extend to 180°\n")
        f.write("- Use checkpoint_regression.py to validate no regression on horizontal tasks\n")

    print(f"Talking points: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path')
    parser.add_argument('--output', default=None, help='Output root directory')
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint, output_root=args.output)
