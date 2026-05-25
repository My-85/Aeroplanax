"""
Circle capability boundary test: R=5000/3000/2000, guidance comparison, w_pursuit sweep.

Usage:
    python experiments/hierarchical_trajectory_tracking/render_circle_debug.py R5000
    python experiments/hierarchical_trajectory_tracking/render_circle_debug.py all
"""
import os, sys, csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.3'

_planax_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _planax_root)

import jax, jax.numpy as jnp, numpy as np
import flax.linen as nn; from flax.linen.initializers import constant, orthogonal
import functools, distrax; import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import matplotlib.gridspec as gridspec
import orbax.checkpoint as ocp
from typing import Sequence, Dict

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
    _quat_from_euler_nb, _quat_conj,
)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import level_circle
from experiments.hierarchical_trajectory_tracking.path_manager import PathManager
from experiments.hierarchical_trajectory_tracking.subgoal_generator import (
    pure_pursuit_subgoal, tangent_following_subgoal,
    pursuit_tangent_blend,
)
from experiments.hierarchical_trajectory_tracking.target_blender import TargetBlender

# ── Network (same as training) ──
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
        ac = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        e = ac(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs))
        hidden, e = ScannedRNN()(hidden, (e, dones))
        fc2 = ac(nn.LayerNorm()(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(e)))
        am = ac(nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
        heads = []
        for i in range(4):
            heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[i], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am)))
        heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[4], kernel_init=constant(0.0),
            bias_init=lambda k,s,d=jnp.float32: jnp.array([0.0,-1.5,-1.5,-1.5,-1.5],dtype=d))(am)))
        c = ac(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
        c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return hidden, (heads[0], heads[1], heads[2], heads[3], heads[4]), jnp.squeeze(c, axis=-1)

CKPT = os.path.join(_planax_root, "results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600")
NET_CFG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
SEED, REACH_R = 42, 500.0
LOOKAHEAD, BLEND_STEPS = 1000.0, 200
def _f(x): a=np.asarray(x); return float(a) if a.ndim==0 else float(a.reshape(-1)[0])


def lowgain_roll_subgoal(path_ctx, yaw, pitch, roll, vt, k_roll=0.3, max_roll_deg=30.0):
    """Low-gain roll: tangent heading error → mild bank angle."""
    import numpy as _np
    G = 9.81
    t = path_ctx["tangent_world"]
    t_hdg = _np.arctan2(t[1], t[0])
    t_pitch = _np.arctan2(t[2], _np.sqrt(t[0]**2 + t[1]**2) + 1e-9)
    hdg_err = _np.arctan2(_np.sin(t_hdg - yaw), _np.cos(t_hdg - yaw))
    t_roll = _np.clip(k_roll * hdg_err, -_np.radians(max_roll_deg), _np.radians(max_roll_deg))
    return t_hdg, t_pitch, t_roll, 250.0, {"mode": "lowgain_roll", "k_roll": k_roll}


def run_circle(radius: float, direction: int, guidance: str,
               n_points: int = 60, max_steps: int = 1500,
               w_pursuit: float = 0.6, k_roll: float = 0.0, max_roll_deg: float = 30.0):
    env = AeroPlanaxHeading_Pitch_V_Env(Heading_Pitch_V_TaskParams())
    net = ActorCriticRNN([31,41,41,41,5], config=NET_CFG)
    rng = jax.random.PRNGKey(SEED)
    obs_shape = env.observation_space(env.agents[0], Heading_Pitch_V_TaskParams()).shape
    h0 = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
    net_params = net.init(rng, h0, (jnp.zeros((1,1,*obs_shape)), jnp.zeros((1,1))))
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(CKPT, args=ocp.args.StandardRestore())
    net_params = ckpt["params"]

    rng, rk = jax.random.split(rng)
    obs_dict, state = env.reset(rk, Heading_Pitch_V_TaskParams())
    origin_n=_f(state.plane_state.north); origin_e=_f(state.plane_state.east)
    origin_alt=_f(state.plane_state.altitude)

    init_yaw = 0.0
    q_nb=_quat_from_euler_nb(0.0,0.0,init_yaw); q_bn=_quat_conj(q_nb)
    state=state.replace(plane_state=state.plane_state.replace(
        yaw=jnp.array([init_yaw]),q0=jnp.array([q_bn[0]]),q1=jnp.array([q_bn[1]]),
        q2=jnp.array([q_bn[2]]),q3=jnp.array([q_bn[3]])), target_heading=jnp.array([init_yaw]))

    wps, _ = level_circle(origin_n, origin_e, origin_alt, init_yaw,
                          radius=radius, n_points=n_points, direction=direction)
    pm = PathManager(wps, mode="lookahead", lookahead_dist=LOOKAHEAD, reach_radius=REACH_R)
    total_arc = pm.arc[-1]

    gname = guidance
    if guidance == "pursuit_tangent_blend":
        gname = f"ptb_w{w_pursuit:.1f}"
    if guidance == "lowgain_roll":
        gname = f"lgr_k{k_roll:.1f}_mx{max_roll_deg:.0f}"
    tag = f"R{int(radius)}_{'R' if direction>0 else 'L'}_{gname}"

    blender = TargetBlender(blend_steps=BLEND_STEPS)
    hstate = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
    done_flag=jnp.zeros((1,))
    iy=_f(state.plane_state.yaw); ip=_f(state.plane_state.pitch)
    ir=_f(state.plane_state.roll); iv=_f(state.plane_state.vt)
    blender.reset(iy, ip, ir, iv)

    rec={"t":[],"n":[],"e":[],"a":[],"vt":[],"roll":[],"pitch":[],"yaw":[],
         "t_hdg":[],"t_pitch":[],"t_roll":[],"t_vt":[],"cte":[],"path_s":[],
         "thr":[],"el":[],"ail":[],"rud":[],"alpha":[],"beta":[],"G":[]}

    crashed = False
    for step in range(max_steps):
        ps=state.plane_state
        north=_f(ps.north); east=_f(ps.east); alt=_f(ps.altitude)
        vt=_f(ps.vt); roll=_f(ps.roll); pitch=_f(ps.pitch); yaw=_f(ps.yaw)
        alpha=_f(ps.alpha); beta=_f(ps.beta)
        ax=_f(ps.ax); ay=_f(ps.ay); az=_f(ps.az)

        path_ctx = pm.update(north, east, alt)
        if pm.just_reached:
            blender.reset(yaw, pitch, roll, vt)

        if guidance == "pure_pursuit":
            rh,rp,rr,rv,_ = pure_pursuit_subgoal(path_ctx, yaw, pitch, vt)
        elif guidance == "tangent_following":
            rh,rp,rr,rv,_ = tangent_following_subgoal(path_ctx, yaw, pitch, vt)
        elif guidance == "pursuit_tangent_blend":
            rh,rp,rr,rv,_ = pursuit_tangent_blend(path_ctx, yaw, pitch, vt, w_pursuit=w_pursuit)
        elif guidance == "lowgain_roll":
            rh,rp,rr,rv,_ = lowgain_roll_subgoal(path_ctx, yaw, pitch, roll, vt,
                                                   k_roll=k_roll, max_roll_deg=max_roll_deg)
        else:
            raise ValueError(guidance)

        th,tp,tr,tv = blender.blend(rh, rp, rr, rv, yaw, pitch, roll, vt)

        centre_n = origin_n + radius*np.sin(init_yaw)*direction
        centre_e = origin_e - radius*np.cos(init_yaw)*direction
        radial = np.sqrt((north-centre_n)**2 + (east-centre_e)**2)
        cte = abs(radial - radius)

        state=state.replace(target_heading=jnp.array([th]),target_pitch=jnp.array([tp]),
                            target_roll=jnp.array([tr]),target_vt=jnp.array([tv]))
        obs_in=env._get_obs(state,Heading_Pitch_V_TaskParams())[env.agents[0]][None,None,:]
        hstate,pi,_=net.apply(net_params,hstate,(obs_in,done_flag[None,:]))
        acts=[int(p.mode()[0,0]) for p in pi]
        rng,sk=jax.random.split(rng)
        obs2,state,rew,done,info=env.step(sk,state,{env.agents[0]:jnp.array(acts)},Heading_Pitch_V_TaskParams())
        done_flag=jnp.array([float(done[env.agents[0]])])

        rec["t"].append(step*0.2); rec["n"].append(north); rec["e"].append(east); rec["a"].append(alt)
        rec["vt"].append(vt)
        rec["roll"].append(np.degrees(roll)); rec["pitch"].append(np.degrees(pitch)); rec["yaw"].append(np.degrees(yaw))
        rec["t_hdg"].append(np.degrees(th)); rec["t_pitch"].append(np.degrees(tp))
        rec["t_roll"].append(np.degrees(tr)); rec["t_vt"].append(tv)
        rec["cte"].append(cte); rec["path_s"].append(path_ctx["path_progress"])
        rec["thr"].append(acts[0]/30.0); rec["el"].append((acts[1]*2.0/40.0-1.0)*45.0)
        rec["ail"].append((acts[2]*2.0/40.0-1.0)*45.0); rec["rud"].append((acts[3]*2.0/40.0-1.0)*45.0)
        rec["alpha"].append(np.degrees(alpha)); rec["beta"].append(np.degrees(beta))
        rec["G"].append(float(np.sqrt(ax**2+ay**2+az**2)))

        if bool(done[env.agents[0]]):
            crashed = True; break
        if pm.is_done():
            break

    t_a=np.array(rec["t"]); n=len(t_a)
    completed = pm.is_done() and not crashed
    cte_a=np.array(rec["cte"]); alt_a=np.array(rec["a"]); vt_a=np.array(rec["vt"])
    roll_a=np.array(rec["roll"]); pitch_a=np.array(rec["pitch"])
    t_roll_a=np.array(rec["t_roll"]); t_pitch_a=np.array(rec["t_pitch"])
    thr_a=np.array(rec["thr"]); el_a=np.array(rec["el"]); ail_a=np.array(rec["ail"])
    alpha_a=np.array(rec["alpha"]); beta_a=np.array(rec["beta"]); G_a=np.array(rec["G"])
    path_s=np.array(rec["path_s"])
    completion_ratio = path_s[-1]/total_arc if n>0 else 0.0

    result = {
        "radius": radius, "direction": "R" if direction>0 else "L",
        "guidance": gname, "w_pursuit": w_pursuit if guidance=="pursuit_tangent_blend" else None,
        "k_roll": k_roll if guidance=="lowgain_roll" else None,
        "completed": completed, "steps": n, "completion_ratio": completion_ratio,
        "cte_mean": cte_a.mean(), "cte_p50": np.percentile(cte_a,50),
        "cte_p90": np.percentile(cte_a,90), "cte_max": cte_a.max(),
        "alt_min": alt_a.min(), "alt_max": alt_a.max(),
        "vt_min": vt_a.min(), "vt_max": vt_a.max(),
        "roll_min": roll_a.min(), "roll_max": roll_a.max(),
        "t_roll_min": t_roll_a.min(), "t_roll_max": t_roll_a.max(),
        "pitch_min": pitch_a.min(), "pitch_max": pitch_a.max(),
        "t_pitch_min": t_pitch_a.min(), "t_pitch_max": t_pitch_a.max(),
        "alpha_max": alpha_a.max(), "beta_max": np.abs(beta_a).max(),
        "G_max": G_a.max(), "G_p95": np.percentile(G_a,95),
        "thr_mean": thr_a.mean(), "el_abs_mean": np.abs(el_a).mean(),
        "ail_abs_mean": np.abs(ail_a).mean(),
        "el_saturation": float((np.abs(el_a)>40).mean()),
        "ail_saturation": float((np.abs(ail_a)>40).mean()),
        "termination": "crash" if crashed else ("ok" if completed else "timeout"),
    }

    # Plot (only for working configs to save time)
    if completed or completion_ratio > 0.5:
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f"Circle R={int(radius)}m {result['direction']} {gname}  "
                     f"{'OK' if completed else 'FAIL'}  CTE mean={cte_a.mean():.0f}m", fontsize=11)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
        ax=fig.add_subplot(gs[0,0]); ax.plot(rec["e"],rec["n"],'b-',lw=0.8,alpha=0.7)
        ax.scatter(wps[:,1],wps[:,0],c='orange',s=3,alpha=0.4)
        th_c=np.linspace(0,2*np.pi,200)
        cn=origin_n+radius*np.sin(init_yaw)*direction+radius*np.cos(th_c)
        ce=origin_e-radius*np.cos(init_yaw)*direction+radius*np.sin(th_c)
        ax.plot(ce,cn,'gray',lw=0.5,ls='--'); ax.set_title("Top-down"); ax.set_aspect('equal'); ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[0,1]); ax.plot(t_a,cte_a,'r-',lw=0.8); ax.set_title("CTE (m)"); ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[0,2]); ax.plot(t_a,path_s); ax.axhline(y=total_arc,c='gray',ls='--')
        ax.set_title("Path progress"); ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[1,0]); ax.plot(t_a,rec["a"]); ax.set_title("Altitude"); ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[1,1]); ax.plot(t_a,rec["vt"]); ax.axhline(y=250,c='gray',ls='--')
        ax.set_title("Airspeed"); ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[1,2]); ax.plot(t_a,rec["t_hdg"],label='tgt'); ax.plot(t_a,rec["yaw"],label='cur')
        ax.set_title("Heading"); ax.grid(True,alpha=0.3); ax.legend(fontsize=7)
        ax=fig.add_subplot(gs[2,0]); ax.plot(t_a,rec["roll"],label='cur')
        ax.plot(t_a,rec["t_roll"],label='tgt',alpha=0.5); ax.set_title("Roll"); ax.grid(True,alpha=0.3); ax.legend(fontsize=7)
        ax=fig.add_subplot(gs[2,1]); ax.plot(t_a,G_a,'r-',lw=0.8); ax.axhline(y=9,c='orange',ls='--')
        ax.set_title(f"G-load (max={G_a.max():.1f})"); ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[2,2]); ax.plot(t_a,rec["el"],lw=0.6,label='El')
        ax.plot(t_a,rec["ail"],'r-',lw=0.6,label='Ail'); ax.set_title("Controls (deg)")
        ax.grid(True,alpha=0.3); ax.legend(fontsize=7)
        outdir=os.path.join(_planax_root,"results/hierarchical_trajectory_tracking/circle_debug")
        os.makedirs(outdir,exist_ok=True)
        fig.savefig(os.path.join(outdir,f"circle_{tag}.png"),dpi=100,bbox_inches='tight')
        plt.close(fig)

    print(f"  {result['direction']:>3} {gname:>25} {'OK' if completed else 'FAIL':>5} "
          f"st={n:4d} comp={completion_ratio:.0%} "
          f"CTE: m={cte_a.mean():.0f} p50={np.percentile(cte_a,50):.0f} p90={np.percentile(cte_a,90):.0f} "
          f"Gmax={G_a.max():.1f} alt=[{alt_a.min():.0f},{alt_a.max():.0f}]")
    return result


GUIDANCES_BASE = ["pure_pursuit", "tangent_following", "pursuit_tangent_blend"]
W_SWEEP = [0.6, 0.7, 0.8, 0.9]
ROLL_CONFIGS = [(0.2, 20), (0.3, 30), (0.4, 30), (0.4, 45)]
RADII = [5000, 3000, 2000]

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv)>1 else "R5000"

    all_results = []

    if target == "all":
        for r in RADII:
            for d in [1, -1]:
                for g in GUIDANCES_BASE:
                    all_results.append(run_circle(r, d, g))
                for wp in W_SWEEP:
                    all_results.append(run_circle(r, d, "pursuit_tangent_blend", w_pursuit=wp))
                for kr, mr in ROLL_CONFIGS:
                    all_results.append(run_circle(r, d, "lowgain_roll", k_roll=kr, max_roll_deg=mr))
    elif target.startswith("R"):
        r = int(target.replace("R",""))
        for d in [1, -1]:
            for g in GUIDANCES_BASE:
                all_results.append(run_circle(r, d, g))
            for wp in W_SWEEP:
                all_results.append(run_circle(r, d, "pursuit_tangent_blend", w_pursuit=wp))
            for kr, mr in ROLL_CONFIGS:
                all_results.append(run_circle(r, d, "lowgain_roll", k_roll=kr, max_roll_deg=mr))
    else:
        print(f"Usage: render_circle_debug.py [R5000|R3000|R2000|all]")
        sys.exit(1)

    # Summary
    print(f"\n{'='*130}")
    print(f"CIRCLE BOUNDARY SUMMARY")
    print(f"{'='*130}")
    header = (f"{'R':>5} {'D':>3} {'Guidance':>28} {'OK':>5} {'St':>5} {'Comp%':>6} "
              f"{'CTE_m':>7} {'CTE50':>7} {'CTE90':>7} {'CTEmax':>7} "
              f"{'Alt[min,max]':>18} {'Roll[min,max]':>18} "
              f"{'Gmax':>5} {'|el|':>5} {'elSat':>5} {'ailSat':>5}")
    print("-"*130)
    for r in all_results:
        print(f"{r['radius']:5.0f} {r['direction']:>3} {r['guidance']:>28} {str(r['completed']):>5} "
              f"{r['steps']:5d} {r['completion_ratio']*100:5.0f}% "
              f"{r['cte_mean']:7.0f} {r['cte_p50']:7.0f} {r['cte_p90']:7.0f} {r['cte_max']:7.0f} "
              f"[{r['alt_min']:5.0f},{r['alt_max']:5.0f}] [{r['roll_min']:5.0f},{r['roll_max']:5.0f}] "
              f"{r['G_max']:5.1f} {r['el_abs_mean']:5.1f} {r['el_saturation']:5.2f} {r['ail_saturation']:5.2f}")
    print(f"{'='*130}")

    # Save CSV
    csv_path = os.path.join(_planax_root, "results/hierarchical_trajectory_tracking/circle_debug",
                            f"summary_{target}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if all_results:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader(); w.writerows(all_results)
        print(f"\nCSV saved: {csv_path}")
    print("DONE")
