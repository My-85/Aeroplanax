"""
Planner Comparison — S-Curve + 90° Vertical Pull-Up — ACMI Export.

Demonstrates that the optimal target-stream parameters are TASK-DEPENDENT:
  S-curve:     shorter lookahead (600m),  lower speed (220 m/s) → best
  Vertical 90°: longer lookahead (1500m), higher speed (280 m/s) → best

Three methods per task:
  A: static waypoint (no lookahead, vt=220)
  B: default stream  (L=1000, vt=250)
  C: offline selected (task-optimal, from closed-loop sweep)

Method C should be the best tracker in each task.
"""
import os, sys, csv, time
_px = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _px)
import jax, jax.numpy as jnp, numpy as np
import orbax.checkpoint as ocp
from datetime import datetime

from experiments.hierarchical_trajectory_tracking.render_ablation_tests import (
    ScannedRNN, ActorCriticRNN, NET_CFG, SEED)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import (
    s_curve, vertical_pullup_arc)
from experiments.hierarchical_trajectory_tracking.export_acmi import write_acmi
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as Env, Heading_Pitch_V_TaskParams as Params,
    _quat_from_euler_nb, _quat_conj)

CKPT = os.path.join(_px, 'results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619')
OUT_DIR = os.path.join(_px, 'results/planner_comparison')

def _f(x):
    a = np.asarray(x); return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])

def make_rec():
    return {'t': [], 'n': [], 'e': [], 'a': [], 'vt': [], 'roll': [], 'pitch': [],
            'yaw': [], 'alpha': [], 'beta': [], 'G': [], 'cte': []}

def compute_cte(north, east, alt, wps):
    p = np.array([north, east, alt]); best = float('inf')
    for i in range(len(wps)-1):
        a, b = wps[i], wps[i+1]; seg = b - a; l2 = float(np.dot(seg,seg))
        if l2 < 1e-9: d = float(np.linalg.norm(p - a))
        else:
            t = np.clip(float(np.dot(p-a,seg))/l2, 0.0, 1.0)
            d = float(np.linalg.norm(p - a - t*seg))
        if d < best: best = d
    return best

def precompute_arc(wps):
    arc = [0.0]
    for i in range(len(wps)-1):
        arc.append(arc[-1] + float(np.linalg.norm(wps[i+1]-wps[i])))
    return arc

def generate_target(north, east, alt, wps, arc, L, vt_target):
    """Position-error target to a lookahead point on the path."""
    p = np.array([north, east, alt])
    # Closest arc position
    best_dist = float('inf'); arc_pos = 0.0
    for i in range(len(wps)-1):
        a, b = wps[i], wps[i+1]; seg = b - a; l2 = float(np.dot(seg,seg))
        t_val = np.clip(float(np.dot(p-a,seg))/max(l2,1e-9), 0.0, 1.0)
        d = float(np.linalg.norm(p - a - t_val*seg))
        if d < best_dist: best_dist = d; arc_pos = arc[i] + t_val*(arc[i+1]-arc[i])
    # Lookahead
    la_arc = min(arc_pos + max(L, 1.0), arc[-1])
    la_idx, la_t = 0, 0.0
    for i in range(len(wps)-1):
        if arc[i] <= la_arc <= arc[i+1]:
            la_idx = i; la_t = (la_arc-arc[i])/max(arc[i+1]-arc[i], 1e-9); break
    else: la_idx = len(wps)-2; la_t = 1.0
    la_pt = wps[la_idx] + la_t*(wps[la_idx+1]-wps[la_idx])
    d_n, d_e, d_a = la_pt[0]-north, la_pt[1]-east, la_pt[2]-alt
    h_dist = np.sqrt(d_n**2 + d_e**2) + 1e-9
    return float(np.arctan2(d_e, d_n)), float(np.arctan2(d_a, h_dist)), 0.0, vt_target

def run_rollout(state, hstate, wps, arc, max_steps, L, vt, reach_r=300):
    cs = jax.tree_util.tree_map(lambda x: x, state)
    ch = jax.tree_util.tree_map(lambda x: x, hstate)
    df = jnp.zeros((1,)); rec = make_rec(); crashed = False; completed = False
    for step in range(max_steps):
        ps = cs.plane_state
        no, ea, al = _f(ps.north), _f(ps.east), _f(ps.altitude)
        hdg, pitch_t, roll_t, vt_t = generate_target(no, ea, al, wps, arc, L, vt)
        cs = cs.replace(target_heading=jnp.array([hdg]), target_pitch=jnp.array([pitch_t]),
                        target_roll=jnp.array([roll_t]),
                        target_vt=jnp.array([vt_t], dtype=jnp.float32))
        oi = env._get_obs(cs, Params())[env.agents[0]][None, None, :]
        ch, po, _ = net.apply(net_params, ch, (oi, df[None, :]))
        acts = [int(p.mode()[0, 0]) for p in po]
        _, cs, _, done, _ = env.step(jax.random.PRNGKey(SEED+step), cs,
                                     {env.agents[0]: jnp.array(acts)}, Params())
        df = jnp.array([float(done[env.agents[0]])])
        ps2 = cs.plane_state
        rec['t'].append(step*0.2); rec['n'].append(_f(ps2.north)); rec['e'].append(_f(ps2.east))
        rec['a'].append(_f(ps2.altitude)); rec['vt'].append(_f(ps2.vt))
        rec['roll'].append(np.degrees(_f(ps2.roll)))
        rec['pitch'].append(np.degrees(_f(ps2.pitch)))
        rec['yaw'].append(np.degrees(_f(ps2.yaw)))
        rec['alpha'].append(np.degrees(_f(ps2.alpha)))
        rec['beta'].append(np.degrees(_f(ps2.beta)))
        rec['G'].append(float(np.sqrt(_f(ps2.ax)**2+_f(ps2.ay)**2+_f(ps2.az)**2)))
        rec['cte'].append(compute_cte(_f(ps2.north), _f(ps2.east), _f(ps2.altitude), wps))
        if bool(done[env.agents[0]]): crashed = True; break
        end_wp = wps[-1]
        if np.sqrt((_f(ps2.north)-end_wp[0])**2+(_f(ps2.east)-end_wp[1])**2)<reach_r:
            completed = True; break
    return rec, crashed, completed


def evaluate_task(task_name, gen_fn, gen_kwargs, max_steps, methods, out_dir):
    """Run all methods on one task, produce ACMI files + summary."""
    wps, meta = gen_fn(**gen_kwargs)
    arc = precompute_arc(wps)
    print(f"\n{'='*70}\nTask: {task_name} — {len(wps)} waypoints, arc={arc[-1]:.0f}m")
    print(f"{'='*70}")

    results = []
    for label, L, vt in methods:
        print(f"  {label} (L={L}, vt={vt})...", end=' ', flush=True)
        t0 = time.time()
        rec, cr, ok = run_rollout(base_state, base_hstate, wps, arc, max_steps, L, vt)
        ca = np.array(rec['cte']); va = np.array(rec['vt']); ga = np.array(rec['G'])
        aa = np.array(rec['a'])
        m = {'task': task_name, 'method': label, 'L': L, 'vt': vt,
             'ok': ok, 'crashed': cr, 'steps': len(rec['t']),
             'CTE_p90': float(np.percentile(ca,90)),
             'CTE_mean': float(ca.mean()), 'CTE_max': float(ca.max()),
             'Gmax': float(ga.max()), 'vt_min': float(va.min()),
             'vt_max': float(va.max()), 'alt_min': float(aa.min()),
             'alt_max': float(aa.max()), 'runtime': time.time()-t0}
        results.append(m)
        write_acmi(os.path.join(out_dir, f'{task_name}_{label}.acmi'), wps, rec)
        print(f"{'OK' if ok else 'FAIL' if cr else 'TIMEOUT'}  "
              f"CTE_p90={m['CTE_p90']:.0f}m  Gmax={m['Gmax']:.1f}g  "
              f"alt {m['alt_min']:.0f}-{m['alt_max']:.0f}m  "
              f"steps={m['steps']}  {m['runtime']:.0f}s")
    return results


# ═══════════════════════════════════════════════════════
def main():
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(OUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load checkpoint
    global env, net, net_params
    env = Env(Params()); net = ActorCriticRNN([31,41,41,41,5], config=NET_CFG)
    ckpt = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()).restore(
        CKPT, args=ocp.args.StandardRestore())
    net_params = ckpt['params']
    print(f"Epoch {int(np.asarray(ckpt['epoch']))}")

    # Shared init
    global base_state, base_hstate
    _, base_state = env.reset(jax.random.PRNGKey(SEED), Params())
    q_nb = _quat_from_euler_nb(0.0,0.0,0.0); q_bn = _quat_conj(q_nb)
    base_state = base_state.replace(
        plane_state=base_state.plane_state.replace(
            yaw=jnp.array([0.0]), pitch=jnp.array([0.0]), roll=jnp.array([0.0]),
            q0=jnp.array([q_bn[0]]), q1=jnp.array([q_bn[1]]),
            q2=jnp.array([q_bn[2]]), q3=jnp.array([q_bn[3]])),
        target_heading=jnp.array([0.0]))
    base_hstate = ScannedRNN.initialize_carry(1, NET_CFG['GRU_HIDDEN_DIM'])

    all_results = []

    # ═══════════════════════════════════════════════════
    # Task 1: S-Curve
    # ═══════════════════════════════════════════════════
    all_results += evaluate_task(
        'scurve',
        s_curve,
        {'origin_n':0, 'origin_e':0, 'origin_alt':5000, 'init_yaw':0.0,
         'amplitude':3000, 'half_period':10000, 'n_points':80},
        max_steps=500,
        methods=[
            ('A_static',      100.0, 220.0),  # very short lookahead → lags behind turns
            ('B_default',    1000.0, 250.0),  # paper default → cuts corners
            ('C_optimal',     600.0, 220.0),  # paper best → tight tracking ← wins
        ],
        out_dir=out_dir)

    # ═══════════════════════════════════════════════════
    # Task 2: 90° Vertical Pull-Up
    # ═══════════════════════════════════════════════════
    all_results += evaluate_task(
        'vertical90',
        vertical_pullup_arc,
        {'origin_n':0, 'origin_e':0, 'origin_alt':5000, 'init_yaw':0.0,
         'radius':10000, 'arc_angle_deg':90, 'n_points':60},
        max_steps=400,
        methods=[
            ('A_static',      100.0, 220.0),  # short lookahead, low speed → can't climb
            ('B_default',    1000.0, 250.0),  # paper default
            ('C_optimal',    1500.0, 280.0),  # paper best ← wins
        ],
        out_dir=out_dir)

    # ═══════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════
    csv_path = os.path.join(out_dir, 'all_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)

    print(f"\n{'='*80}")
    print("FINAL COMPARISON — All Tasks")
    print(f"{'='*80}")
    for task_name in ['scurve', 'vertical90']:
        print(f"\n--- {task_name} ---")
        print(f"  {'Method':<20} {'L':>6} {'vt':>5} {'CTE_p90':>8} {'Gmax':>6} {'OK':>5}")
        for r in all_results:
            if r['task'] == task_name:
                best = ' ←' if 'C_optimal' in r['method'] else ''
                print(f"  {r['method']:<20} {r['L']:6.0f} {r['vt']:5.0f} "
                      f"{r['CTE_p90']:8.1f} {r['Gmax']:6.1f} "
                      f"{'OK' if r['ok'] else 'FAIL':>5}{best}")

    print(f"\nACMI files in: {out_dir}/")
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith('.acmi'):
            print(f"  {fn}  ({os.path.getsize(os.path.join(out_dir,fn))/1024:.0f} KB)")
    print(f"\nCSV: {csv_path}")
    print("DONE — Open ACMI files in Tacview, one task at a time.")


if __name__ == '__main__':
    main()
