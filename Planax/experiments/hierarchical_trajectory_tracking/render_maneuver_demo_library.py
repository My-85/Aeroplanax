"""
Render entire maneuver demo library. Outputs: ACMI, metrics, plots, summary CSV.

Usage:
    python experiments/hierarchical_trajectory_tracking/render_maneuver_demo_library.py
"""
import os, sys, json, csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'; os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.3'
_planax_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _planax_root)

import jax, jax.numpy as jnp, numpy as np
import flax.linen as nn; from flax.linen.initializers import constant, orthogonal
import functools, distrax; import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import matplotlib.gridspec as gridspec
import orbax.checkpoint as ocp
from typing import Sequence, Dict
from datetime import datetime

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
    _quat_from_euler_nb, _quat_conj)
from envs.utils.utils import enu_to_geodetic
from experiments.hierarchical_trajectory_tracking.trajectory_generators import TRAJECTORY_REGISTRY
from experiments.hierarchical_trajectory_tracking.planner import PurePursuitPlanner, PlannerConfig
from experiments.hierarchical_trajectory_tracking.path_utils import compute_true_cte
from experiments.hierarchical_trajectory_tracking.demo_library import MANEUVER_DEMOS

# ── Network ──
class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry; ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y
    @staticmethod
    def initialize_carry(bs, hs):
        return nn.GRUCell(features=hs).initialize_carry(jax.random.PRNGKey(0), (bs, hs))

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]; config: Dict
    @nn.compact
    def __call__(self, hidden, x):
        ac = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh; obs, dones = x
        e = ac(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs))
        hidden, e = ScannedRNN()(hidden, (e, dones))
        fc2 = ac(nn.LayerNorm()(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(e)))
        am = ac(nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
        heads = [distrax.Categorical(logits=nn.Dense(self.action_dim[i], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am)) for i in range(4)]
        heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[4], kernel_init=constant(0.0),
            bias_init=lambda k, s, d=jnp.float32: jnp.array([0.0, -1.5, -1.5, -1.5, -1.5], dtype=d))(am)))
        c = ac(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
        c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return hidden, (heads[0], heads[1], heads[2], heads[3], heads[4]), jnp.squeeze(c, axis=-1)

# ── Setup ──
CKPT = os.path.join(_planax_root, "results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600")
NET_CFG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
SEED = 42
def _f(x): a = np.asarray(x); return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])

tag = datetime.now().strftime('%Y%m%d_%H%M%S')
out_root = os.path.join(_planax_root, "results/maneuver_demo_library", tag)
os.makedirs(out_root, exist_ok=True)
for sub in ["acmi", "figures", "rollouts", "metrics"]:
    os.makedirs(os.path.join(out_root, sub), exist_ok=True)

# Create env ONCE
env = AeroPlanaxHeading_Pitch_V_Env(Heading_Pitch_V_TaskParams())
net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CFG)
rng = jax.random.PRNGKey(SEED)
obs_shape = env.observation_space(env.agents[0], Heading_Pitch_V_TaskParams()).shape
h0 = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
net_params = net.init(rng, h0, (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1))))
ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
ckpt = ckptr.restore(CKPT, args=ocp.args.StandardRestore())
net_params = ckpt["params"]
print(f"Loaded checkpoint epoch {int(ckpt['epoch'])}, {len(MANEUVER_DEMOS)} demos to render\n", flush=True)


def run_demo(demo):
    gen_fn = TRAJECTORY_REGISTRY[demo["traj"]]
    wps, meta = gen_fn(0, 0, 5000, 0.0, **demo["params"])
    total_arc = meta['total_length_m']
    cfg = PlannerConfig(lookahead_dist=demo["lookahead"], reach_radius=demo["reach_r"],
                        blend_steps=200, target_vt=250.0)
    planner = PurePursuitPlanner(cfg)

    rng2, rk = jax.random.split(rng)
    obs_dict, state = env.reset(rk, Heading_Pitch_V_TaskParams())
    q_nb = _quat_from_euler_nb(0.0, 0.0, 0.0); q_bn = _quat_conj(q_nb)
    state = state.replace(plane_state=state.plane_state.replace(
        yaw=jnp.array([0.0]), q0=jnp.array([q_bn[0]]), q1=jnp.array([q_bn[1]]),
        q2=jnp.array([q_bn[2]]), q3=jnp.array([q_bn[3]])), target_heading=jnp.array([0.0]))
    planner.reset(wps, 0.0, 0.0, 0.0, 250.0)

    hstate = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
    df = jnp.zeros((1,))
    rec = {"t": [], "n": [], "e": [], "a": [], "vt": [], "roll": [], "pitch": [], "yaw": [],
           "t_hdg": [], "t_pitch": [], "t_roll": [], "cte": [], "G": [],
           "thr": [], "el": [], "ail": [], "alpha": []}
    crashed = False

    for step in range(demo["max_steps"]):
        ps = state.plane_state
        no = _f(ps.north); ea = _f(ps.east); al = _f(ps.altitude)
        vt = _f(ps.vt); ro = _f(ps.roll); pi = _f(ps.pitch); ya = _f(ps.yaw)
        alph = _f(ps.alpha); ax = _f(ps.ax); ay = _f(ps.ay); az = _f(ps.az)

        result = planner.step(no, ea, al, ya, pi, ro, vt)
        th, tp, tr, tv = result["target_heading"], result["target_pitch"], result["target_roll"], result["target_vt"]
        state = state.replace(target_heading=jnp.array([th]), target_pitch=jnp.array([tp]),
                              target_roll=jnp.array([tr]), target_vt=jnp.array([float(tv)], dtype=jnp.float32))
        obs_in = env._get_obs(state, Heading_Pitch_V_TaskParams())[env.agents[0]][None, None, :]
        hstate, pi_out, _ = net.apply(net_params, hstate, (obs_in, df[None, :]))
        acts = [int(p.mode()[0, 0]) for p in pi_out]
        rng2, sk = jax.random.split(rng2)
        obs2, state, rew, done, info = env.step(sk, state, {env.agents[0]: jnp.array(acts)}, Heading_Pitch_V_TaskParams())
        df = jnp.array([float(done[env.agents[0]])])

        rec["t"].append(step * 0.2); rec["n"].append(no); rec["e"].append(ea); rec["a"].append(al)
        rec["vt"].append(vt); rec["roll"].append(np.degrees(ro)); rec["pitch"].append(np.degrees(pi))
        rec["yaw"].append(np.degrees(ya))
        rec["t_hdg"].append(np.degrees(th)); rec["t_pitch"].append(np.degrees(tp))
        rec["t_roll"].append(np.degrees(tr))
        rec["cte"].append(compute_true_cte(np.array([no, ea, al]), wps, result["path_ctx"]["wp_idx"], 10))
        rec["G"].append(float(np.sqrt(ax**2 + ay**2 + az**2)))
        rec["thr"].append(acts[0] / 30.0); rec["el"].append((acts[1] * 2.0 / 40.0 - 1.0) * 45.0)
        rec["ail"].append((acts[2] * 2.0 / 40.0 - 1.0) * 45.0); rec["alpha"].append(np.degrees(alph))

        if bool(done[env.agents[0]]): crashed = True; break
        if planner.is_done(): break

    n = len(rec["t"]); ok = planner.is_done() and not crashed
    ca = np.array(rec["cte"]); pa = np.array(rec["a"]); va = np.array(rec["vt"])
    Ga = np.array(rec["G"]); ra = np.array(rec["roll"]); pia = np.array(rec["pitch"])
    tpa = np.array(rec["t_pitch"]); ala = np.array(rec["alpha"])

    metrics = {
        "name": demo["name"], "category": demo["category"],
        "trajectory_type": demo["traj"], "params": demo["params"],
        "planner": demo["planner"],
        "completed": ok, "steps": n,
        "completion_ratio": float(planner.path_progress / total_arc) if total_arc > 0 else 0.0,
        "CTE_mean": float(ca.mean()), "CTE_p50": float(np.percentile(ca, 50)),
        "CTE_p90": float(np.percentile(ca, 90)), "CTE_max": float(ca.max()),
        "Gmax": float(Ga.max()), "Gmean": float(Ga.mean()),
        "vt_min": float(va.min()), "vt_mean": float(va.mean()), "vt_max": float(va.max()),
        "altitude_min": float(pa.min()), "altitude_max": float(pa.max()),
        "roll_min": float(ra.min()), "roll_max": float(ra.max()),
        "pitch_min": float(pia.min()), "pitch_max": float(pia.max()),
        "target_pitch_min": float(tpa.min()), "target_pitch_max": float(tpa.max()),
        "alpha_max": float(ala.max()), "alpha_min": float(ala.min()),
        "termination_reason": "crash" if crashed else ("ok" if ok else "timeout"),
    }

    # ── Save ACMI ──
    acmi_path = os.path.join(out_root, "acmi", f"{demo['name']}.acmi")
    with open(acmi_path, 'w') as f:
        f.write("FileType=text/acmi/tacview\nFileVersion=2.2\n0,ReferenceTime=2023-04-01T00:00:00Z\n")
        # Write waypoint markers
        for k, (wn, we, wa) in enumerate(wps):
            lat, lon, alt_m = enu_to_geodetic(we, wn, wa, 0, 0, 0)
            f.write(f"{5000+k},Type=Navaid+Static+Waypoint,Name=WP_{k},Color=Yellow,"
                    f"T={float(lon)}|{float(lat)}|{float(alt_m)}|0|0|0\n")
        # Write aircraft frames
        for i in range(n):
            lat, lon, alt_m = enu_to_geodetic(rec["e"][i], rec["n"][i], rec["a"][i], 0, 0, 0)
            f.write(f"#{rec['t'][i]:.2f}\n")
            f.write(f"100,T={float(lon)}|{float(lat)}|{float(alt_m)}|"
                    f"{rec['roll'][i]:.2f}|{rec['pitch'][i]:.2f}|{rec['yaw'][i]:.2f},"
                    f"Type=Air+FixedWing,Name=F16,Color=Cyan\n")

    # ── Save rollout NPZ ──
    npz_path = os.path.join(out_root, "rollouts", f"{demo['name']}.npz")
    np.savez(npz_path, **rec, waypoints=wps)

    # ── Save metrics JSON ──
    json_path = os.path.join(out_root, "metrics", f"{demo['name']}_metrics.json")
    metrics["acmi_path"] = acmi_path; metrics["npz_path"] = npz_path
    metrics["figure_path"] = os.path.join(out_root, "figures", f"{demo['name']}.png")
    with open(json_path, 'w') as f: json.dump(metrics, f, indent=2)

    # ── Plot ──
    t_a = np.array(rec["t"])
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"{demo['name']}  {'OK' if ok else 'FAIL'}  "
                 f"CTE_m={ca.mean():.0f}m Gmax={Ga.max():.1f} "
                 f"alt=[{pa.min():.0f},{pa.max():.0f}]m", fontsize=11)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0]); ax.plot(rec["e"], rec["n"], 'b-', lw=0.8, alpha=0.7)
    ax.scatter(wps[:, 1], wps[:, 0], c='orange', s=3, alpha=0.4)
    ax.set_title("Top-down"); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(gs[0, 1]); ax.plot(t_a, rec["a"]); ax.set_title("Altitude"); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(gs[0, 2]); ax.plot(t_a, ca, 'r-', lw=0.8)
    ax.set_title(f"CTE (mean={ca.mean():.0f}m)"); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(gs[1, 0]); ax.plot(t_a, rec["vt"]); ax.axhline(y=250, c='gray', ls='--')
    ax.set_title("Airspeed"); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(gs[1, 1]); ax.plot(t_a, Ga, 'r-', lw=0.8); ax.axhline(y=9, c='orange', ls='--')
    ax.set_title(f"G-load (max={Ga.max():.1f})"); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(gs[1, 2]); ax.plot(t_a, rec["pitch"], label='pitch')
    ax.plot(t_a, rec["t_pitch"], '--', label='tgt'); ax.set_title("Pitch"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.savefig(metrics["figure_path"], dpi=100, bbox_inches='tight'); plt.close(fig)

    status = "✓" if ok else "✗"
    print(f"  {status} {demo['name']:<35} st={n:4d} CTE_m={ca.mean():6.0f} CTE50={np.percentile(ca,50):6.0f} "
          f"Gmax={Ga.max():4.1f} vt=[{va.min():.0f},{va.max():.0f}] "
          f"alt=[{pa.min():.0f},{pa.max():.0f}]")
    return metrics


# ── Run all ──
print(f"{'Status':>6} {'Maneuver':<35} {'St':>4} {'CTE_m':>7} {'CTE50':>7} {'Gmax':>5} {'vt_range':>14} {'alt_range':>20}")
print("-" * 115)
all_metrics = []
for demo in MANEUVER_DEMOS:
    m = run_demo(demo); all_metrics.append(m)

# Summary CSV
csv_path = os.path.join(out_root, "summary.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[k for k in all_metrics[0].keys() if not k.endswith('_path')])
    w.writeheader()
    for m in all_metrics:
        w.writerow({k: v for k, v in m.items() if not k.endswith('_path')})
print(f"\nCSV: {csv_path}")
print(f"ACMI: {out_root}/acmi/")
print(f"Figures: {out_root}/figures/")
print(f"DA:NE")
