"""
Online RH-TSO with JAX PurePursuitPlanner clone.
jit + scan + vmap for horizon evaluation; PurePursuitPlanner for execution.
"""
import os, sys, json, csv, time
_px = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _px)
import jax, jax.numpy as jnp, numpy as np
from jax import lax
from functools import partial
import orbax.checkpoint as ocp
from datetime import datetime

from experiments.hierarchical_trajectory_tracking.render_ablation_tests import (
    ScannedRNN, ActorCriticRNN, NET_CFG, SEED)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import (
    s_curve, figure_eight, helix_trajectory, vertical_pullup_arc)
from experiments.hierarchical_trajectory_tracking.planner import (
    PurePursuitPlanner, PlannerConfig)
from experiments.hierarchical_trajectory_tracking.path_utils import compute_true_cte
from experiments.hierarchical_trajectory_tracking.jax_planner import (
    precompute_path, jax_planner_step)
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as Env, Heading_Pitch_V_TaskParams as Params,
    _quat_from_euler_nb, _quat_conj)

CKPT = os.path.join(_px, 'results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619')


def _f(x):
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


# ═══ JIT-scanned single-candidate horizon rollout ═══
@partial(jax.jit, static_argnames=['H'])
def scan_horizon_jit(carry_init, wps, arc, L, vt, H, reach_radius=500.0, blend_steps=250):
    """JIT-scanned H-step rollout with jax_planner_step. Returns total_cost, crashed."""
    state_array, hstate, rng_key = carry_init

    def body(carry, _):
        state_array_i, hstate_i, rng_i, path_idx_i, path_prog_i, wp_cnt_i, step_cnt_i, total_cost = carry

        # Extract position from state
        ps = state_array_i.plane_state
        pos = jnp.array([ps.north[0], ps.east[0], ps.altitude[0]])
        cur_yaw = ps.yaw[0]; cur_pitch = ps.pitch[0]; cur_roll = ps.roll[0]; cur_vt = ps.vt[0]

        # JAX planner step
        t_h, t_p, t_r, t_v, new_pp_idx, new_pp, new_wc, is_done_path = \
            jax_planner_step(pos, cur_yaw, cur_pitch, cur_roll, cur_vt,
                             wps, arc, path_idx_i, path_prog_i, wp_cnt_i,
                             L, vt, reach_radius, step_cnt_i, blend_steps)

        # Set target on state
        state_array_i = state_array_i.replace(
            target_heading=jnp.array([t_h]),
            target_pitch=jnp.array([t_p]),
            target_roll=jnp.array([t_r]),
            target_vt=jnp.array([t_v], dtype=jnp.float32))

        # Network
        oi = env._get_obs(state_array_i, Params())[env.agents[0]][None, None, :]
        df = jnp.zeros((1, 1), dtype=jnp.float32)
        hstate_i, pi_out, _ = net.apply(net_params, hstate_i, (oi, df))
        acts = jnp.array([jnp.argmax(p.logits[0, 0]) for p in pi_out])

        # Env step
        rng_i, sk = jax.random.split(rng_i)
        _, state_array_new, _, done, _ = env.step(
            sk, state_array_i, {env.agents[0]: acts}, Params())

        # Cost
        ps2 = state_array_new.plane_state
        cte = jnp.sqrt((ps2.north[0] - pos[0]) ** 2 +
                        (ps2.east[0] - pos[1]) ** 2 +
                        (ps2.altitude[0] - pos[2]) ** 2)
        g = jnp.sqrt(ps2.ax[0] ** 2 + ps2.ay[0] ** 2 + ps2.az[0] ** 2)
        step_cost = (cte +
                     jnp.maximum(0.0, g - 8.5) * 500.0 +
                     jnp.maximum(0.0, 250.0 - ps2.vt[0]) * 3.0 +
                     jnp.maximum(0.0, jnp.abs(ps2.alpha[0]) - 0.35) * 100.0)
        total_cost += step_cost
        step_cnt_i += 1

        next_carry = (state_array_new, hstate_i, rng_i,
                      new_pp_idx, new_pp, new_wc, step_cnt_i, total_cost)
        return next_carry, done[env.agents[0]]

    # Parse carry_init
    state_array_0, hstate_0, rng_0 = carry_init
    init_path_idx = 0; init_path_prog = 0.0; init_wp_cnt = 0; init_step_cnt = 0
    init_cost = 0.0
    init = (state_array_0, hstate_0, rng_0,
            init_path_idx, init_path_prog, init_wp_cnt, init_step_cnt, init_cost)

    (final_state, final_h, final_rk, _, _, _, _, total_cost), _ = lax.scan(
        body, init, xs=None, length=H)
    crashed = False
    return total_cost, crashed


# ═══ Vmapped multi-candidate evaluation ═══
CANDIDATES = jnp.array([
    [600.0, 220.0],
    [1000.0, 250.0],
    [1500.0, 280.0],
], dtype=jnp.float32)


def eval_candidates_jit(state, hstate, wps_j, arc, rng_key, H):
    """Vmapped parallel evaluation of all (L,vt) candidates."""
    # Build carry for each candidate scan
    # vmap over CANDIDATES axis
    def eval_single(L_vt):
        L = L_vt[0]; vt = L_vt[1]
        rk = jax.random.split(rng_key)[0]
        cost, _ = scan_horizon_jit((state, hstate, rk), wps_j, arc, L, vt, H)
        return cost

    costs = jax.vmap(eval_single)(CANDIDATES)
    best_idx = jnp.argmin(costs)
    return best_idx, costs


# ═══ Execution: PurePursuitPlanner for real rollout ═══
def run_moving_lookahead(state, hstate, wps, net_params_local, env, rng_key, max_steps, L, vt, rr=500):
    cs, ch, df = state, hstate, jnp.zeros((1,))
    rk = rng_key
    cfg = PlannerConfig(lookahead_dist=L, reach_radius=rr, blend_steps=250, target_vt=float(vt))
    planner = PurePursuitPlanner(cfg)
    planner.reset(wps, 0.0, 0.0, 0.0, float(vt))
    rec = {'t': [], 'n': [], 'e': [], 'a': [], 'vt': [], 'roll': [], 'pitch': [], 'yaw': [],
           'alpha': [], 'beta': [], 'G': [], 'cte': []}
    crashed = False
    for step in range(max_steps):
        ps = cs.plane_state
        no, ea, al = _f(ps.north), _f(ps.east), _f(ps.altitude)
        vt2, ro, pi, ya = _f(ps.vt), _f(ps.roll), _f(ps.pitch), _f(ps.yaw)
        r = planner.step(no, ea, al, ya, pi, ro, vt2)
        cs = cs.replace(target_heading=jnp.array([r['target_heading']]),
                        target_pitch=jnp.array([r['target_pitch']]),
                        target_roll=jnp.array([r['target_roll']]),
                        target_vt=jnp.array([float(vt)], dtype=jnp.float32))
        oi = env._get_obs(cs, Params())[env.agents[0]][None, None, :]
        ch, po, _ = net.apply(net_params_local, ch, (oi, df[None, :]))
        acts = [int(p.mode()[0, 0]) for p in po]
        rk, sk = jax.random.split(rk)
        _, cs, _, done, _ = env.step(sk, cs, {env.agents[0]: jnp.array(acts)}, Params())
        df = jnp.array([float(done[env.agents[0]])])
        ps2 = cs.plane_state; wi = r['path_ctx']['wp_idx']
        rec['t'].append(step * 0.2); rec['n'].append(_f(ps2.north)); rec['e'].append(_f(ps2.east))
        rec['a'].append(_f(ps2.altitude)); rec['vt'].append(_f(ps2.vt))
        rec['roll'].append(np.degrees(_f(ps2.roll))); rec['pitch'].append(np.degrees(_f(ps2.pitch)))
        rec['yaw'].append(np.degrees(_f(ps2.yaw))); rec['alpha'].append(np.degrees(_f(ps2.alpha)))
        rec['beta'].append(np.degrees(_f(ps2.beta)))
        rec['G'].append(float(np.sqrt(_f(ps2.ax) ** 2 + _f(ps2.ay) ** 2 + _f(ps2.az) ** 2)))
        rec['cte'].append(compute_true_cte(
            np.array([_f(ps2.north), _f(ps2.east), _f(ps2.altitude)]), wps, wi, 10))
        if bool(done[env.agents[0]]): crashed = True; break
        if planner.is_done(): break
    ok = planner.is_done() and not crashed
    return rec, crashed, ok


# ═══ Static waypoint ═══
class StaticWaypointPlanner:
    def __init__(self, wps, vt=250.0): self.wps = wps; self.vt = vt; self.wi = 0

    def reset(self, wps, n, e, a, vt): self.wps = wps; self.wi = 0

    def step(self, n, e, a, yaw, pitch, roll, vt):
        nw = len(self.wps); bi = min(self.wi + 1, nw - 1)
        for k in range(self.wi, min(self.wi + 3, nw)):
            wp = self.wps[k]; dn, de = wp[0] - n, wp[1] - e
            if dn * dn + de * de > 100: bi = k; break
        wp = self.wps[bi]; self.wi = bi; dn, de, da = wp[0] - n, wp[1] - e, wp[2] - a
        hd = np.sqrt(dn ** 2 + de ** 2) + 1e-9
        return {'target_heading': float(np.arctan2(de, dn)),
                'target_pitch': float(np.arctan2(da, hd)), 'target_roll': 0.0,
                'target_vt': float(self.vt), 'path_ctx': {'wp_idx': bi}}

    def is_done(self): return self.wi >= len(self.wps) - 1


def run_static(state, hstate, wps, net_params_local, env, rng_key, max_steps, vt=250):
    cs, ch, df = state, hstate, jnp.zeros((1,)); rk = rng_key
    planner = StaticWaypointPlanner(wps, vt)
    rec = {'t': [], 'n': [], 'e': [], 'a': [], 'vt': [], 'roll': [], 'pitch': [], 'yaw': [],
           'alpha': [], 'beta': [], 'G': [], 'cte': []}; crashed = False
    for step in range(max_steps):
        ps = cs.plane_state; no, ea, al = _f(ps.north), _f(ps.east), _f(ps.altitude)
        vt2, ro, pi, ya = _f(ps.vt), _f(ps.roll), _f(ps.pitch), _f(ps.yaw)
        r = planner.step(no, ea, al, ya, pi, ro, vt2)
        cs = cs.replace(target_heading=jnp.array([r['target_heading']]),
                        target_pitch=jnp.array([r['target_pitch']]),
                        target_roll=jnp.array([r['target_roll']]),
                        target_vt=jnp.array([float(vt)], dtype=jnp.float32))
        oi = env._get_obs(cs, Params())[env.agents[0]][None, None, :]
        ch, po, _ = net.apply(net_params_local, ch, (oi, df[None, :]))
        acts = [int(p.mode()[0, 0]) for p in po]
        rk, sk = jax.random.split(rk)
        _, cs, _, done, _ = env.step(sk, cs, {env.agents[0]: jnp.array(acts)}, Params())
        df = jnp.array([float(done[env.agents[0]])])
        ps2 = cs.plane_state; wi = r['path_ctx']['wp_idx']
        rec['t'].append(step * 0.2); rec['n'].append(_f(ps2.north)); rec['e'].append(_f(ps2.east))
        rec['a'].append(_f(ps2.altitude)); rec['vt'].append(_f(ps2.vt))
        rec['roll'].append(np.degrees(_f(ps2.roll))); rec['pitch'].append(np.degrees(_f(ps2.pitch)))
        rec['yaw'].append(np.degrees(_f(ps2.yaw))); rec['alpha'].append(np.degrees(_f(ps2.alpha)))
        rec['beta'].append(np.degrees(_f(ps2.beta)))
        rec['G'].append(float(np.sqrt(_f(ps2.ax) ** 2 + _f(ps2.ay) ** 2 + _f(ps2.az) ** 2)))
        rec['cte'].append(compute_true_cte(
            np.array([_f(ps2.north), _f(ps2.east), _f(ps2.altitude)]), wps, wi, 10))
        if bool(done[env.agents[0]]): crashed = True; break
        if planner.is_done(): break
    ok = planner.is_done() and not crashed
    return rec, crashed, ok


# ═══ Online RH-TSO with jit+vmap ═══
def online_rhtso_jax(state, hstate, wps, wps_j, arc, net_params_local, env, rng_key,
                     max_steps, replan=10, H=20, rr=500):
    """Online RH-TSO: jit+vmap horizon eval + PurePursuitPlanner execution."""
    cs, ch, df = state, hstate, jnp.zeros((1,)); rk = rng_key; ts = 0
    rec = {'t': [], 'n': [], 'e': [], 'a': [], 'vt': [], 'roll': [], 'pitch': [], 'yaw': [],
           'alpha': [], 'beta': [], 'G': [], 'cte': []}; crashed = False
    cur_L, cur_vt = 1000, 250; replan_times = []; param_history = []
    cfg = PlannerConfig(lookahead_dist=cur_L, reach_radius=rr, blend_steps=250, target_vt=float(cur_vt))
    planner = PurePursuitPlanner(cfg)
    planner.reset(wps, 0.0, 0.0, 0.0, float(cur_vt))

    while ts < max_steps:
        # ── JIT+vmap RH-TSO ──
        t_r = time.time()
        rk2, _ = jax.random.split(rk)
        best_idx, costs = eval_candidates_jit(cs, ch, wps_j, arc, rk2, H)
        cur_L = int(CANDIDATES[best_idx][0])
        cur_vt = float(CANDIDATES[best_idx][1])
        replan_times.append(time.time() - t_r)
        param_history.append((ts, cur_L, cur_vt))

        # ── Execute segment ──
        cfg = PlannerConfig(lookahead_dist=cur_L, reach_radius=rr, blend_steps=250, target_vt=cur_vt)
        planner = PurePursuitPlanner(cfg)
        for s in range(replan):
            if ts >= max_steps: break
            ps = cs.plane_state; no, ea, al = _f(ps.north), _f(ps.east), _f(ps.altitude)
            vt2, ro, pi, ya = _f(ps.vt), _f(ps.roll), _f(ps.pitch), _f(ps.yaw)
            planner.reset(wps, no, ea, al, cur_vt)
            r = planner.step(no, ea, al, ya, pi, ro, vt2)
            cs = cs.replace(target_heading=jnp.array([r['target_heading']]),
                            target_pitch=jnp.array([r['target_pitch']]),
                            target_roll=jnp.array([r['target_roll']]),
                            target_vt=jnp.array([cur_vt], dtype=jnp.float32))
            oi = env._get_obs(cs, Params())[env.agents[0]][None, None, :]
            ch, po, _ = net.apply(net_params_local, ch, (oi, df[None, :]))
            acts = [int(p.mode()[0, 0]) for p in po]
            rk, sk = jax.random.split(rk)
            _, cs, _, done, _ = env.step(sk, cs, {env.agents[0]: jnp.array(acts)}, Params())
            df = jnp.array([float(done[env.agents[0]])])
            ps2 = cs.plane_state; wi = r['path_ctx']['wp_idx']
            rec['t'].append(ts * 0.2); rec['n'].append(_f(ps2.north)); rec['e'].append(_f(ps2.east))
            rec['a'].append(_f(ps2.altitude)); rec['vt'].append(_f(ps2.vt))
            rec['roll'].append(np.degrees(_f(ps2.roll))); rec['pitch'].append(np.degrees(_f(ps2.pitch)))
            rec['yaw'].append(np.degrees(_f(ps2.yaw))); rec['alpha'].append(np.degrees(_f(ps2.alpha)))
            rec['beta'].append(np.degrees(_f(ps2.beta)))
            rec['G'].append(float(np.sqrt(_f(ps2.ax) ** 2 + _f(ps2.ay) ** 2 + _f(ps2.az) ** 2)))
            rec['cte'].append(compute_true_cte(
                np.array([_f(ps2.north), _f(ps2.east), _f(ps2.altitude)]), wps, wi, 10))
            if bool(done[env.agents[0]]): crashed = True; break
            if planner.is_done(): break
            ts += 1
        if crashed or planner.is_done(): break
    ok = planner.is_done() and not crashed
    avg_replan = np.mean(replan_times) * 1000 if replan_times else 0
    return rec, crashed, ok, avg_replan, param_history


# ═══ Metrics ═══
def make_metrics(rec, crashed, ok, task, method, rt, params='', replan_ms=0):
    ca = np.array(rec['cte']); va = np.array(rec['vt']); ga = np.array(rec['G']); aa = np.array(rec['alpha'])
    return {'task': task, 'method': method, 'completed': bool(ok), 'steps': len(rec['t']),
            'CTE_mean': float(ca.mean()), 'CTE_p50': float(np.percentile(ca, 50)),
            'CTE_p90': float(np.percentile(ca, 90)), 'CTE_max': float(ca.max()),
            'Gmax': float(ga.max()), 'vt_min': float(va.min()), 'vt_max': float(va.max()),
            'vt_mean': float(va.mean()),
            'alt_min': float(np.array(rec['a']).min()), 'alt_max': float(np.array(rec['a']).max()),
            'env_alpha_min': float(aa.min()), 'env_alpha_max': float(aa.max()),
            'termination': 'crash' if crashed else ('ok' if ok else 'timeout'),
            'runtime_sec': rt, 'params': params, 'replan_avg_ms': replan_ms}


BEST = {'s_curve': (600, 220), 'figure_eight': (600, 220), 'mild_3d': (600, 220), 'vertical_90': (1500, 280)}

TASKS = [
    ('s_curve', s_curve, {'origin_n': 0, 'origin_e': 0, 'origin_alt': 5000, 'init_yaw': 0.0,
                           'amplitude': 3000, 'half_period': 10000, 'n_points': 60}, 500),
    ('figure_eight', figure_eight, {'origin_n': 0, 'origin_e': 0, 'origin_alt': 5000, 'init_yaw': 0.0,
                                     'radius': 5000, 'n_points': 80}, 600),
    ('mild_3d', helix_trajectory, {'origin_n': 0, 'origin_e': 0, 'origin_alt': 5000, 'init_yaw': 0.0,
                                    'radius': 10000, 'turns': 1, 'delta_alt': 1000, 'n_points': 80, 'direction': 1}, 600),
    ('vertical_90', vertical_pullup_arc, {'origin_n': 0, 'origin_e': 0, 'origin_alt': 5000, 'init_yaw': 0.0,
                                           'radius': 10000, 'arc_angle_deg': 90, 'n_points': 40}, 400),
]

HORIZONS = [20, 30, 40]


def main():
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = os.path.join(_px, 'results/online_rhtso_jax', tag)
    for sub in ['acmi', 'figures', 'metrics', 'rollouts']:
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    print(f'Output: {out_root}')

    print("Loading checkpoint...")
    global env, net, net_params
    env = Env(Params()); net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    rng = jax.random.PRNGKey(SEED)
    obs_shape = env.observation_space(env.agents[0], Params()).shape
    h0 = ScannedRNN.initialize_carry(1, NET_CFG['GRU_HIDDEN_DIM'])
    net_params_init = net.init(rng, h0, (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1))))
    ckpt = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()).restore(
        CKPT, args=ocp.args.StandardRestore())
    net_params = ckpt['params']
    print('Loaded.\n')

    all_rows = []

    for task_name, gen_fn, gen_kw, mx in TASKS:
        wps, meta = gen_fn(**gen_kw)
        wps_j, arc, total_arc_j = precompute_path(wps)
        print(f"\n{'=' * 60}\nTask: {task_name}\n{'=' * 60}")

        # Shared init
        rng, rk = jax.random.split(rng)
        _, base_state = env.reset(rk, Params())
        q_nb = _quat_from_euler_nb(0.0, 0.0, 0.0); q_bn = _quat_conj(q_nb)
        base_state = base_state.replace(plane_state=base_state.plane_state.replace(
            yaw=jnp.array([0.0]), q0=jnp.array([q_bn[0]]), q1=jnp.array([q_bn[1]]),
            q2=jnp.array([q_bn[2]]), q3=jnp.array([q_bn[3]])), target_heading=jnp.array([0.0]))
        base_hstate = ScannedRNN.initialize_carry(1, NET_CFG['GRU_HIDDEN_DIM'])

        # A: static_waypoint
        t0 = time.time(); rng, rk = jax.random.split(rng)
        rec, cr, ok = run_static(base_state, base_hstate, wps, net_params, env, rk, mx)
        m = make_metrics(rec, cr, ok, task_name, 'static_waypoint', time.time() - t0)
        all_rows.append(m)
        print(f"  static_waypoint:     {'OK' if ok else 'FAIL'} CTE_p90={m['CTE_p90']:.0f} Gmax={m['Gmax']:.1f}")

        # B: fixed_default
        t0 = time.time(); rng, rk = jax.random.split(rng)
        rec, cr, ok = run_moving_lookahead(base_state, base_hstate, wps, net_params, env, rk, mx, L=1000, vt=250)
        m = make_metrics(rec, cr, ok, task_name, 'fixed_default', time.time() - t0, params='L=1000,vt=250')
        all_rows.append(m)
        print(f"  fixed_default:       {'OK' if ok else 'FAIL'} CTE_p90={m['CTE_p90']:.0f} Gmax={m['Gmax']:.1f}")

        # C: offline_selected
        Lb, vtb = BEST[task_name]
        t0 = time.time(); rng, rk = jax.random.split(rng)
        rec, cr, ok = run_moving_lookahead(base_state, base_hstate, wps, net_params, env, rk, mx, L=Lb, vt=vtb)
        m = make_metrics(rec, cr, ok, task_name, 'offline_selected', time.time() - t0,
                         params=f'L={Lb},vt={vtb}')
        all_rows.append(m)
        print(f"  offline_selected:    {'OK' if ok else 'FAIL'} CTE_p90={m['CTE_p90']:.0f} Gmax={m['Gmax']:.1f}")

        # D: online_rhtso for each H
        for H in HORIZONS:
            method_name = f'online_rhtso_H{H}'
            t0 = time.time(); rng, rk = jax.random.split(rng)
            rec, cr, ok, avg_rp, param_hist = online_rhtso_jax(
                base_state, base_hstate, wps, wps_j, arc, net_params, env, rk, mx, replan=10, H=H)
            m = make_metrics(rec, cr, ok, task_name, method_name, time.time() - t0,
                             params=f'H={H}, replan={avg_rp:.0f}ms', replan_ms=avg_rp)
            m['param_history'] = param_hist
            all_rows.append(m)
            print(f"  {method_name:20s}: {'OK' if ok else 'FAIL'} CTE_p90={m['CTE_p90']:.0f} "
                  f"Gmax={m['Gmax']:.1f} replan={avg_rp:.0f}ms")

    # ═══ Save ═══
    csv_path = os.path.join(out_root, 'online_rhtso_results.csv')
    fields = ['task', 'method', 'completed', 'steps', 'CTE_mean', 'CTE_p50', 'CTE_p90', 'CTE_max',
              'Gmax', 'vt_min', 'vt_max', 'vt_mean', 'alt_min', 'alt_max',
              'env_alpha_min', 'env_alpha_max', 'termination', 'runtime_sec', 'params', 'replan_avg_ms']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(all_rows)

    # Parameter history CSV
    for r in all_rows:
        if 'param_history' in r and r['param_history']:
            hist_path = os.path.join(out_root, 'metrics', f"{r['task']}_{r['method']}_params.csv")
            with open(hist_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['step', 'L', 'vt'])
                w.writerows(r['param_history'])

    # Final table
    print(f"\n{'=' * 100}")
    print("ONLINE RH-TSO JAX FINAL TABLE")
    print(f"{'=' * 100}")
    print(f"{'Task':<16} {'Method':<24} {'CTE_p90':>8} {'Gmax':>6} {'vt_min':>6} {'Status':>6} {'Replan':>10}")
    print("-" * 100)
    for r in all_rows:
        st = 'OK' if r['completed'] else 'FAIL'
        rt = f"{r.get('replan_avg_ms', 0):.0f}ms" if r.get('replan_avg_ms', 0) > 0 else '-'
        print(f"{r['task']:<16} {r['method']:<24} {r['CTE_p90']:8.0f} {r['Gmax']:6.1f} "
              f"{r['vt_min']:6.0f} {st:>6} {rt:>10}")
    print(f"\nCSV: {csv_path}")
    print("DONE")
    return out_root, all_rows


if __name__ == '__main__':
    main()
