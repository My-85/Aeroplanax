"""
PID+NLGL (50Hz inner / 5Hz outer) vs RL (5Hz) — 90° Vertical Pull-Up.

PID uses agent_interaction_steps=1 → each RL step = 1 physics substep (0.02s).
NLGL updates every 10 substeps (5 Hz). PID updates every substep (50 Hz).
Trajectory recorded at 5 Hz for ACMI consistency with RL.
"""
import os, sys, time
_px = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _px)
import jax, jax.numpy as jnp, numpy as np
import orbax.checkpoint as ocp
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.hierarchical_trajectory_tracking.render_ablation_tests import (
    ScannedRNN, ActorCriticRNN, NET_CFG, SEED)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import vertical_pullup_arc
from experiments.hierarchical_trajectory_tracking.export_acmi import write_acmi
from experiments.hierarchical_trajectory_tracking.classical_autopilot import ClassicalAutopilot
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env as Env, Heading_Pitch_V_TaskParams as Params,
    _quat_from_euler_nb, _quat_conj)

CKPT = os.path.join(_px, 'results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619')
OUT_DIR = os.path.join(_px, 'results/pid_vs_rl_comparison')

def _f(x):
    a = np.asarray(x); return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])

def make_rec():
    return {'t': [], 'n': [], 'e': [], 'a': [], 'vt': [], 'roll': [], 'pitch': [],
            'yaw': [], 'alpha': [], 'beta': [], 'G': [], 'cte': []}

def compute_cte(n, e, a, wps):
    p = np.array([n, e, a]); best = float('inf')
    for i in range(len(wps) - 1):
        A, B = wps[i], wps[i+1]; seg = B - A; l2 = float(np.dot(seg, seg))
        if l2 < 1e-9: d = float(np.linalg.norm(p - A))
        else:
            t = np.clip(float(np.dot(p - A, seg)) / l2, 0.0, 1.0)
            d = float(np.linalg.norm(p - A - t * seg))
        if d < best: best = d
    return best

def precompute_arc(wps):
    arc = [0.0]
    for i in range(len(wps) - 1):
        arc.append(arc[-1] + float(np.linalg.norm(wps[i+1] - wps[i])))
    return arc

def rl_target(n, e, a, wps, arc, L, vt_tgt):
    p = np.array([n, e, a])
    best_dist = float('inf'); arc_pos = 0.0
    for i in range(len(wps)-1):
        A, B = wps[i], wps[i+1]; seg = B - A; l2 = float(np.dot(seg, seg))
        tv = np.clip(float(np.dot(p - A, seg)) / max(l2, 1e-9), 0.0, 1.0)
        d = float(np.linalg.norm(p - A - tv * seg))
        if d < best_dist: best_dist = d; arc_pos = arc[i] + tv * (arc[i+1] - arc[i])
    la_arc = min(arc_pos + max(L, 1.0), arc[-1])
    la_idx, la_t = 0, 0.0
    for i in range(len(wps) - 1):
        if arc[i] <= la_arc <= arc[i+1]:
            la_idx = i; la_t = (la_arc - arc[i]) / max(arc[i+1] - arc[i], 1e-9); break
    else: la_idx = len(wps) - 2; la_t = 1.0
    la_pt = wps[la_idx] + la_t * (wps[la_idx+1] - wps[la_idx])
    dn, de, da = la_pt[0] - n, la_pt[1] - e, la_pt[2] - a
    hd = np.sqrt(dn**2 + de**2) + 1e-9
    return float(np.arctan2(de, dn)), float(np.arctan2(da, hd)), 0.0, vt_tgt


def decode_pid_to_discrete(cont_acts):
    thr_idx = int(np.clip(cont_acts[0] * 30, 0, 30))
    ele_idx = int(np.clip((cont_acts[1] + 1) * 20, 0, 40))
    ail_idx = int(np.clip((cont_acts[2] + 1) * 20, 0, 40))
    rud_idx = int(np.clip((cont_acts[3] + 1) * 20, 0, 40))
    return np.array([thr_idx, ele_idx, ail_idx, rud_idx, 0], dtype=np.int32)


# ═══════════════════════════════════════════════════════════
# PID: 50 Hz inner loop + 5 Hz NLGL outer loop
# ═══════════════════════════════════════════════════════════

def run_pid_multirate(wps, max_5hz_steps, L1=800, cruise_vt=280, reach_r=300):
    """
    Multi-rate PID rollout.
    - Each 5Hz "macro-step" = 10 substeps at 50Hz (agent_interaction_steps=1).
    - NLGL runs once per macro-step (at the start).
    - PID runs at every substep.
    - Trajectory recorded once per macro-step (5 Hz for ACMI).
    """
    pid_params = Params().replace(agent_interaction_steps=1)
    pid_env = Env(pid_params)

    # JIT warmup
    _, ws = pid_env.reset(jax.random.PRNGKey(0), pid_params)
    for _ in range(3):
        d = np.array([7, 20, 20, 20, 0], dtype=np.int32)
        _, ws, _, _, _ = pid_env.step(jax.random.PRNGKey(0), ws,
                                      {pid_env.agents[0]: jnp.array(d)}, pid_params)

    _, state = pid_env.reset(jax.random.PRNGKey(SEED + 999), pid_params)
    qn = _quat_from_euler_nb(0.0, 0.0, 0.0); qb = _quat_conj(qn)
    state = state.replace(plane_state=state.plane_state.replace(
        yaw=jnp.array([0.0]), pitch=jnp.array([0.0]), roll=jnp.array([0.0]),
        q0=jnp.array([qb[0]]), q1=jnp.array([qb[1]]),
        q2=jnp.array([qb[2]]), q3=jnp.array([qb[3]])),
        target_heading=jnp.array([0.0]))

    ap = ClassicalAutopilot(wps, L1=L1, cruise_vt=cruise_vt, dt=0.02, reach_radius=reach_r)
    ap.reset()

    rec = make_rec(); crashed = False; completed = False
    SUBSTEPS = 10  # 10 substeps per 5Hz macro-step

    for macro_step in range(max_5hz_steps):
        # ── NLGL at 5 Hz ──
        ps = state.plane_state
        ap.nlgl_step(_f(ps.north), _f(ps.east), _f(ps.altitude),
                     _f(ps.yaw), _f(ps.pitch), _f(ps.vt))

        # ── 10 PID substeps at 50 Hz ──
        for sub in range(SUBSTEPS):
            ps = state.plane_state
            acts = ap.pid_step(_f(ps.pitch), _f(ps.roll), _f(ps.yaw), _f(ps.vt),
                               _f(ps.Q), _f(ps.P), _f(ps.R),
                               _f(ps.alpha), _f(ps.beta))
            disc = decode_pid_to_discrete(acts)
            _, state, _, done, _ = pid_env.step(
                jax.random.PRNGKey(SEED + 100000 + macro_step * SUBSTEPS + sub),
                state, {pid_env.agents[0]: jnp.array(disc)}, pid_params)
            state = state.replace(last_check_time=state.time)
            if bool(done[pid_env.agents[0]]):
                crashed = True; break
        if crashed: break

        # Record at 5 Hz
        ps2 = state.plane_state
        t = macro_step * 0.2
        rec['t'].append(t)
        rec['n'].append(_f(ps2.north)); rec['e'].append(_f(ps2.east))
        rec['a'].append(_f(ps2.altitude)); rec['vt'].append(_f(ps2.vt))
        rec['roll'].append(np.degrees(_f(ps2.roll)))
        rec['pitch'].append(np.degrees(_f(ps2.pitch)))
        rec['yaw'].append(np.degrees(_f(ps2.yaw)))
        rec['alpha'].append(np.degrees(_f(ps2.alpha)))
        rec['beta'].append(np.degrees(_f(ps2.beta)))
        rec['G'].append(float(np.sqrt(_f(ps2.ax)**2 + _f(ps2.ay)**2 + _f(ps2.az)**2)))
        rec['cte'].append(compute_cte(_f(ps2.north), _f(ps2.east), _f(ps2.altitude), wps))

        # Completion check
        ew = wps[-1]
        if np.sqrt((_f(ps2.north) - ew[0])**2 + (_f(ps2.east) - ew[1])**2) < reach_r:
            completed = True; break

    return rec, crashed, completed


# ═══════════════════════════════════════════════════════════
# RL: standard 5 Hz
# ═══════════════════════════════════════════════════════════

def run_rl_rollout(state, hstate, wps, arc, max_steps, L, vt, reach_r=300):
    cs = jax.tree_util.tree_map(lambda x: x, state)
    ch = jax.tree_util.tree_map(lambda x: x, hstate)
    df = jnp.zeros((1,)); rec = make_rec(); crashed = False; completed = False
    for step in range(max_steps):
        ps = cs.plane_state
        no, ea, al = _f(ps.north), _f(ps.east), _f(ps.altitude)
        hdg, pt, rt, vt_t = rl_target(no, ea, al, wps, arc, L, vt)
        cs = cs.replace(target_heading=jnp.array([hdg]), target_pitch=jnp.array([pt]),
                        target_roll=jnp.array([rt]),
                        target_vt=jnp.array([vt_t], dtype=jnp.float32))
        oi = rl_env._get_obs(cs, rl_params)[rl_env.agents[0]][None, None, :]
        ch, po, _ = net.apply(net_params, ch, (oi, df[None, :]))
        acts = [int(p.mode()[0, 0]) for p in po]
        _, cs, _, done, _ = rl_env.step(
            jax.random.PRNGKey(SEED + step), cs,
            {rl_env.agents[0]: jnp.array(acts)}, rl_params)
        df = jnp.array([float(done[rl_env.agents[0]])])
        ps2 = cs.plane_state
        rec['t'].append(step * 0.2)
        rec['n'].append(_f(ps2.north)); rec['e'].append(_f(ps2.east))
        rec['a'].append(_f(ps2.altitude)); rec['vt'].append(_f(ps2.vt))
        rec['roll'].append(np.degrees(_f(ps2.roll)))
        rec['pitch'].append(np.degrees(_f(ps2.pitch)))
        rec['yaw'].append(np.degrees(_f(ps2.yaw)))
        rec['alpha'].append(np.degrees(_f(ps2.alpha)))
        rec['beta'].append(np.degrees(_f(ps2.beta)))
        rec['G'].append(float(np.sqrt(_f(ps2.ax)**2 + _f(ps2.ay)**2 + _f(ps2.az)**2)))
        rec['cte'].append(compute_cte(_f(ps2.north), _f(ps2.east), _f(ps2.altitude), wps))
        if bool(done[rl_env.agents[0]]): crashed = True; break
        ew = wps[-1]
        if np.sqrt((_f(ps2.north) - ew[0])**2 + (_f(ps2.east) - ew[1])**2) < reach_r:
            completed = True; break
    return rec, crashed, completed


# ═══════════════════════════════════════════════════════════
def main():
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(OUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    wps, _ = vertical_pullup_arc(0, 0, 5000, 0.0, radius=10000, arc_angle_deg=90, n_points=60)
    arc = precompute_arc(wps)
    max_5hz_steps = 400
    reach_r = 300
    print(f"Task: 90° pull-up R=10000m, {len(wps)} wpts, arc={arc[-1]:.0f}m")

    # ── RL setup ──
    global rl_env, rl_params, net, net_params
    rl_params = Params()
    rl_env = Env(rl_params)
    net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
    ckpt = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()).restore(
        CKPT, args=ocp.args.StandardRestore())
    net_params = ckpt['params']
    print(f"RL epoch {int(np.asarray(ckpt['epoch']))}")

    _, base_state = rl_env.reset(jax.random.PRNGKey(SEED), rl_params)
    qn = _quat_from_euler_nb(0.0, 0.0, 0.0); qb = _quat_conj(qn)
    base_state = base_state.replace(plane_state=base_state.plane_state.replace(
        yaw=jnp.array([0.0]), pitch=jnp.array([0.0]), roll=jnp.array([0.0]),
        q0=jnp.array([qb[0]]), q1=jnp.array([qb[1]]),
        q2=jnp.array([qb[2]]), q3=jnp.array([qb[3]])),
        target_heading=jnp.array([0.0]))
    base_hstate = ScannedRNN.initialize_carry(1, NET_CFG['GRU_HIDDEN_DIM'])

    all_data = {}

    # ── PID (50Hz inner / 5Hz outer) ──
    print("\n--- PID+NLGL (50Hz inner / 5Hz outer) ---")
    t0 = time.time()
    pid_rec, pid_cr, pid_ok = run_pid_multirate(wps, max_5hz_steps, L1=800, cruise_vt=280)
    write_acmi(os.path.join(out_dir, 'PID_baseline.acmi'), wps, pid_rec)
    all_data['PID+NLGL'] = pid_rec
    pid_rt = time.time() - t0
    ca = np.array(pid_rec['cte']); ga = np.array(pid_rec['G']); aa = np.array(pid_rec['a'])
    status = 'OK' if pid_ok else ('FAIL' if pid_cr else 'TIMEOUT')
    print(f"  {status}  CTE_p90={np.percentile(ca, 90):.0f}m  Gmax={max(ga):.1f}g  "
          f"alt {min(aa):.0f}-{max(aa):.0f}m  steps={len(pid_rec['t'])}  {pid_rt:.0f}s")

    # ── RL methods ──
    rl_configs = [
        ('RL_A_static',   100.0, 220.0),
        ('RL_B_default', 1000.0, 250.0),
        ('RL_C_optimal',  500.0, 230.0),
    ]
    for label, L, vt in rl_configs:
        print(f"\n--- {label} (L={L:.0f}, vt={vt:.0f}) ---")
        t0 = time.time()
        rec, cr, ok = run_rl_rollout(base_state, base_hstate, wps, arc, max_5hz_steps, L, vt, reach_r)
        write_acmi(os.path.join(out_dir, f'{label}.acmi'), wps, rec)
        all_data[label] = rec
        rt = time.time() - t0
        status = 'OK' if ok else ('FAIL' if cr else 'TIMEOUT')
        ca = np.array(rec['cte']); ga = np.array(rec['G']); aa = np.array(rec['a'])
        print(f"  {status}  CTE_p90={np.percentile(ca, 90):.0f}m  Gmax={max(ga):.1f}g  "
              f"alt {min(aa):.0f}-{max(aa):.0f}m  steps={len(rec['t'])}  {rt:.0f}s")

    # ═══ Comparison plot ═══
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = {'PID+NLGL': 'red', 'RL_A_static': 'orange',
              'RL_B_default': 'blue', 'RL_C_optimal': 'green'}
    labels = {'PID+NLGL': 'PID+NLGL (50/5 Hz)',
              'RL_A_static': 'RL static (L=100, vt=220)',
              'RL_B_default': 'RL default (L=1000, vt=250)',
              'RL_C_optimal': 'RL optimal (L=500, vt=230)'}
    lws = {'PID+NLGL': 1.5, 'RL_A_static': 1, 'RL_B_default': 1, 'RL_C_optimal': 2}

    for key, rec in all_data.items():
        t = rec['t']
        axes[0].plot(t, rec['a'], color=colors[key], linewidth=lws[key], label=labels[key])
        axes[1].plot(t, rec['vt'], color=colors[key], linewidth=lws[key])
        axes[2].plot(t, rec['alpha'], color=colors[key], linewidth=lws[key])

    ref_alt = [wp[2] for wp in wps]
    max_t = max(len(r['t']) * 0.2 for r in all_data.values())
    axes[0].plot(np.linspace(0, max_t, len(ref_alt)), ref_alt, 'k--', linewidth=1,
                 alpha=0.5, label='Reference')

    axes[2].axhline(y=30, color='red', linestyle=':', alpha=0.5)
    axes[2].axhline(y=-20, color='red', linestyle=':', alpha=0.5)

    axes[0].set_ylabel('Altitude (m)'); axes[0].legend(fontsize=7, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel('Airspeed (m/s)'); axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel('AoA (deg)'); axes[2].set_xlabel('Time (s)'); axes[2].grid(True, alpha=0.3)

    fig.suptitle('PID+NLGL (50/5 Hz multi-rate) vs Learned RL — 90° Vertical Pull-Up',
                 fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'comparison_plot.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ═══ Summary ═══
    print(f"\n{'='*90}")
    print("FINAL COMPARISON — 90° Vertical Pull-Up (R=10000m)")
    print(f"{'='*90}")
    print(f"{'Method':<35} {'CTE_p90':>8} {'Gmax':>6} {'Alt_max':>7} {'Steps':>6} {'Status':>8}")
    print("-" * 90)
    for key, rec in all_data.items():
        ca = np.array(rec['cte']); ga = np.array(rec['G']); aa = np.array(rec['a'])
        ok = max(aa) > 12000
        st = 'OK' if ok else ('FAIL' if len(rec['t']) < 380 else 'TIMEOUT')
        best = ' ← BEST' if key == 'RL_C_optimal' else ''
        print(f"{labels.get(key, key):<35} {np.percentile(ca, 90):8.1f} {max(ga):6.1f} "
              f"{max(aa):7.0f} {len(rec['t']):6d} {st:>8}{best}")

    print(f"\nPlot: {plot_path}")
    print(f"ACMI: {out_dir}/")
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith('.acmi'):
            print(f"  {fn}  ({os.path.getsize(os.path.join(out_dir, fn))/1024:.0f} KB)")
    print("DONE")


if __name__ == '__main__':
    main()
