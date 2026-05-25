"""
S-curve / figure-eight debug with pure_pursuit guidance.

Usage:
    python experiments/hierarchical_trajectory_tracking/render_horizontal_complex_debug.py
"""
import os, sys, csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'; os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.3'
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
from experiments.hierarchical_trajectory_tracking.trajectory_generators import (
    s_curve, figure_eight, mild_climb,
)
from experiments.hierarchical_trajectory_tracking.path_manager import PathManager
from experiments.hierarchical_trajectory_tracking.subgoal_generator import pure_pursuit_subgoal
from experiments.hierarchical_trajectory_tracking.target_blender import TargetBlender
from experiments.hierarchical_trajectory_tracking.path_utils import arc_length

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
        ac=nn.relu if self.config["ACTIVATION"]=="relu" else nn.tanh; obs,dones=x
        e=ac(nn.Dense(self.config["FC_DIM_SIZE"],kernel_init=orthogonal(np.sqrt(2)),bias_init=constant(0.0))(obs))
        hidden,e=ScannedRNN()(hidden,(e,dones))
        fc2=ac(nn.LayerNorm()(nn.Dense(256,kernel_init=orthogonal(np.sqrt(2)),bias_init=constant(0.0))(e)))
        am=ac(nn.Dense(self.config["GRU_HIDDEN_DIM"],kernel_init=orthogonal(2),bias_init=constant(0.0))(fc2))
        heads=[distrax.Categorical(logits=nn.Dense(self.action_dim[i],kernel_init=orthogonal(0.01),bias_init=constant(0.0))(am)) for i in range(4)]
        heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[4],kernel_init=constant(0.0),
            bias_init=lambda k,s,d=jnp.float32:jnp.array([0.0,-1.5,-1.5,-1.5,-1.5],dtype=d))(am)))
        c=ac(nn.Dense(self.config["FC_DIM_SIZE"],kernel_init=orthogonal(2),bias_init=constant(0.0))(fc2))
        c=nn.Dense(1,kernel_init=orthogonal(1.0),bias_init=constant(0.0))(c)
        return hidden,(heads[0],heads[1],heads[2],heads[3],heads[4]),jnp.squeeze(c,axis=-1)

CKPT=os.path.join(_planax_root,"results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600")
NET_CFG={"FC_DIM_SIZE":128,"GRU_HIDDEN_DIM":128,"ACTIVATION":"relu"}
SEED,REACH_R,LOOKAHEAD,BLEND_STEPS=42,500.0,1000.0,200
def _f(x):a=np.asarray(x);return float(a)if a.ndim==0 else float(a.reshape(-1)[0])

def run_trajectory(traj_name,wps,total_arc,init_yaw=0.0,max_steps=2000):
    env=AeroPlanaxHeading_Pitch_V_Env(Heading_Pitch_V_TaskParams())
    net=ActorCriticRNN([31,41,41,41,5],config=NET_CFG)
    rng=jax.random.PRNGKey(SEED)
    obs_shape=env.observation_space(env.agents[0],Heading_Pitch_V_TaskParams()).shape
    h0=ScannedRNN.initialize_carry(1,NET_CFG["GRU_HIDDEN_DIM"])
    net_params=net.init(rng,h0,(jnp.zeros((1,1,*obs_shape)),jnp.zeros((1,1))))
    ckptr=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt=ckptr.restore(CKPT,args=ocp.args.StandardRestore());net_params=ckpt["params"]

    rng,rk=jax.random.split(rng)
    obs_dict,state=env.reset(rk,Heading_Pitch_V_TaskParams())
    origin_n=_f(state.plane_state.north);origin_e=_f(state.plane_state.east)
    origin_alt=_f(state.plane_state.altitude)
    q_nb=_quat_from_euler_nb(0.0,0.0,init_yaw);q_bn=_quat_conj(q_nb)
    state=state.replace(plane_state=state.plane_state.replace(
        yaw=jnp.array([init_yaw]),q0=jnp.array([q_bn[0]]),q1=jnp.array([q_bn[1]]),
        q2=jnp.array([q_bn[2]]),q3=jnp.array([q_bn[3]])),target_heading=jnp.array([init_yaw]))

    pm=PathManager(wps,mode="lookahead",lookahead_dist=LOOKAHEAD,reach_radius=REACH_R)
    blender=TargetBlender(blend_steps=BLEND_STEPS)
    hstate=ScannedRNN.initialize_carry(1,NET_CFG["GRU_HIDDEN_DIM"]);done_flag=jnp.zeros((1,))
    iy=_f(state.plane_state.yaw);ip=_f(state.plane_state.pitch)
    ir=_f(state.plane_state.roll);iv=_f(state.plane_state.vt)
    blender.reset(iy,ip,ir,iv)

    rec={"t":[],"n":[],"e":[],"a":[],"vt":[],"roll":[],"pitch":[],"yaw":[],
         "t_hdg":[],"t_pitch":[],"t_roll":[],"cte":[],"path_s":[],
         "thr":[],"el":[],"ail":[],"rud":[],"alpha":[],"beta":[],"G":[]}
    crashed=False

    for step in range(max_steps):
        ps=state.plane_state
        north=_f(ps.north);east=_f(ps.east);alt=_f(ps.altitude)
        vt=_f(ps.vt);roll=_f(ps.roll);pitch=_f(ps.pitch);yaw=_f(ps.yaw)
        alpha=_f(ps.alpha);beta=_f(ps.beta);ax=_f(ps.ax);ay=_f(ps.ay);az=_f(ps.az)

        path_ctx=pm.update(north,east,alt)
        if pm.just_reached:blender.reset(yaw,pitch,roll,vt)
        rh,rp,rr,rv,_=pure_pursuit_subgoal(path_ctx,yaw,pitch,vt)
        th,tp,tr,tv=blender.blend(rh,rp,rr,rv,yaw,pitch,roll,vt)

        # CTE: distance to nearest segment (simplified)
        cte=path_ctx["dist_to_wp"]

        state=state.replace(target_heading=jnp.array([th]),target_pitch=jnp.array([tp]),
                            target_roll=jnp.array([tr]),target_vt=jnp.array([tv]))
        obs_in=env._get_obs(state,Heading_Pitch_V_TaskParams())[env.agents[0]][None,None,:]
        hstate,pi,_=net.apply(net_params,hstate,(obs_in,done_flag[None,:]))
        acts=[int(p.mode()[0,0]) for p in pi]
        rng,sk=jax.random.split(rng)
        obs2,state,rew,done,info=env.step(sk,state,{env.agents[0]:jnp.array(acts)},Heading_Pitch_V_TaskParams())
        done_flag=jnp.array([float(done[env.agents[0]])])

        rec["t"].append(step*0.2);rec["n"].append(north);rec["e"].append(east);rec["a"].append(alt)
        rec["vt"].append(vt);rec["roll"].append(np.degrees(roll));rec["pitch"].append(np.degrees(pitch))
        rec["yaw"].append(np.degrees(yaw));rec["t_hdg"].append(np.degrees(th));rec["t_pitch"].append(np.degrees(tp))
        rec["t_roll"].append(np.degrees(tr));rec["cte"].append(cte);rec["path_s"].append(path_ctx["path_progress"])
        rec["thr"].append(acts[0]/30.0);rec["el"].append((acts[1]*2.0/40.0-1.0)*45.0)
        rec["ail"].append((acts[2]*2.0/40.0-1.0)*45.0);rec["rud"].append((acts[3]*2.0/40.0-1.0)*45.0)
        rec["alpha"].append(np.degrees(alpha));rec["beta"].append(np.degrees(beta))
        rec["G"].append(float(np.sqrt(ax**2+ay**2+az**2)))

        if bool(done[env.agents[0]]):crashed=True;break
        if pm.is_done():break

    t_a=np.array(rec["t"]);n=len(t_a);completed=pm.is_done() and not crashed
    cte_a=np.array(rec["cte"]);alt_a=np.array(rec["a"]);vt_a=np.array(rec["vt"])
    roll_a=np.array(rec["roll"]);pitch_a=np.array(rec["pitch"]);yaw_a=np.array(rec["yaw"])
    t_hdg_a=np.array(rec["t_hdg"]);t_pitch_a=np.array(rec["t_pitch"]);t_roll_a=np.array(rec["t_roll"])
    thr_a=np.array(rec["thr"]);el_a=np.array(rec["el"]);ail_a=np.array(rec["ail"])
    alpha_a=np.array(rec["alpha"]);beta_a=np.array(rec["beta"]);G_a=np.array(rec["G"])
    path_s=np.array(rec["path_s"]);comp_ratio=path_s[-1]/total_arc if n>0 else 0.0

    result={"trajectory":traj_name,"completed":completed,"steps":n,"completion_ratio":comp_ratio,
        "cte_mean":cte_a.mean(),"cte_p50":np.percentile(cte_a,50),"cte_p90":np.percentile(cte_a,90),"cte_max":cte_a.max(),
        "alt_min":alt_a.min(),"alt_max":alt_a.max(),"vt_min":vt_a.min(),"vt_max":vt_a.max(),
        "roll_min":roll_a.min(),"roll_max":roll_a.max(),"pitch_min":pitch_a.min(),"pitch_max":pitch_a.max(),
        "yaw_range":yaw_a.max()-yaw_a.min(),"t_hdg_range":t_hdg_a.max()-t_hdg_a.min(),
        "t_pitch_min":t_pitch_a.min(),"t_pitch_max":t_pitch_a.max(),
        "alpha_max":alpha_a.max(),"beta_max":np.abs(beta_a).max(),"G_max":G_a.max(),"G_p95":np.percentile(G_a,95),
        "el_sat":float((np.abs(el_a)>40).mean()),"ail_sat":float((np.abs(ail_a)>40).mean()),
        "termination":"crash" if crashed else ("ok" if completed else "timeout")}

    # Plot (only for interesting configs)
    if completed or comp_ratio>0.5:
        fig=plt.figure(figsize=(18,12))
        fig.suptitle(f"{traj_name} {'OK' if completed else 'FAIL'} CTE_m={cte_a.mean():.0f}m Gmax={G_a.max():.1f}",fontsize=11)
        gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.5,wspace=0.35)
        ax=fig.add_subplot(gs[0,0]);ax.plot(rec["e"],rec["n"],'b-',lw=0.8,alpha=0.7)
        ax.scatter(wps[:,1],wps[:,0],c='orange',s=3,alpha=0.4);ax.set_title("Top-down");ax.set_aspect('equal');ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[0,1]);ax.plot(t_a,cte_a,'r-',lw=0.8);ax.set_title("CTE (m)");ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[0,2]);ax.plot(t_a,path_s);ax.axhline(y=total_arc,c='gray',ls='--');ax.set_title("Path progress");ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[1,0]);ax.plot(t_a,rec["a"]);ax.set_title("Altitude");ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[1,1]);ax.plot(t_a,rec["vt"]);ax.axhline(y=250,c='gray',ls='--');ax.set_title("Airspeed");ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[1,2]);ax.plot(t_a,t_hdg_a,label='tgt');ax.plot(t_a,yaw_a,label='cur');ax.set_title("Heading");ax.grid(True,alpha=0.3);ax.legend(fontsize=7)
        ax=fig.add_subplot(gs[2,0]);ax.plot(t_a,roll_a);ax.set_title("Roll");ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[2,1]);ax.plot(t_a,G_a,'r-',lw=0.8);ax.axhline(y=9,c='orange',ls='--');ax.set_title(f"G-load max={G_a.max():.1f}");ax.grid(True,alpha=0.3)
        ax=fig.add_subplot(gs[2,2]);ax.plot(t_a,el_a,lw=0.6,label='El');ax.plot(t_a,ail_a,'r-',lw=0.6,label='Ail');ax.set_title("Controls");ax.grid(True,alpha=0.3);ax.legend(fontsize=7)
        outdir=os.path.join(_planax_root,"results/hierarchical_trajectory_tracking/horizontal_complex")
        os.makedirs(outdir,exist_ok=True)
        fig.savefig(os.path.join(outdir,f"{traj_name.replace(' ','_')}.png"),dpi=100,bbox_inches='tight')
        plt.close(fig)

    return result


if __name__=="__main__":
    results=[]
    # S-curves
    for amp in [2000,3000,5000]:
        for hp in [10000,15000]:
            wps,meta=s_curve(0,0,5000,0.0,amplitude=amp,half_period=hp,n_points=80)
            arc_c=arc_length(wps)
            name=f"S_A{amp}_P{hp//1000}"
            print(f"\n{name}: {meta['n_points']}wp {arc_c[-1]:.0f}m max_curv={meta['max_curvature']:.5f}")
            res=run_trajectory(name,wps,arc_c[-1],max_steps=1500)
            results.append(res)
            print(f"  {'OK' if res['completed'] else 'FAIL'} st={res['steps']} comp={res['completion_ratio']:.0%} "
                  f"CTE_m={res['cte_mean']:.0f}p50={res['cte_p50']:.0f}p90={res['cte_p90']:.0f} Gmax={res['G_max']:.1f}")

    # Figure-eights
    for rad in [5000,3000]:
        wps,meta=figure_eight(0,0,5000,0.0,radius=rad,n_points=100)
        arc_c=arc_length(wps)
        name=f"Fig8_R{rad}"
        print(f"\n{name}: {meta['n_points']}wp {arc_c[-1]:.0f}m")
        res=run_trajectory(name,wps,arc_c[-1],max_steps=2000)
        results.append(res)
        print(f"  {'OK' if res['completed'] else 'FAIL'} st={res['steps']} comp={res['completion_ratio']:.0%} "
              f"CTE_m={res['cte_mean']:.0f}p50={res['cte_p50']:.0f}p90={res['cte_p90']:.0f} Gmax={res['G_max']:.1f}")

    # Mild climb/descent
    for da in [1000,-1000,2000,-2000]:
        wps,meta=mild_climb(0,0,5000,0.0,length=15000,delta_alt=da,n_points=20)
        arc_c=arc_length(wps)
        name=f"Climb_{da:+}m"
        print(f"\n{name}: {meta['n_points']}wp {arc_c[-1]:.0f}m gamma={meta['max_pitch_proxy_deg']:.0f}deg")
        res=run_trajectory(name,wps,arc_c[-1],max_steps=800)
        results.append(res)
        print(f"  {'OK' if res['completed'] else 'FAIL'} st={res['steps']} comp={res['completion_ratio']:.0%} "
              f"CTE_m={res['cte_mean']:.0f}p50={res['cte_p50']:.0f}p90={res['cte_p90']:.0f} Gmax={res['G_max']:.1f}")

    # Summary
    print(f"\n{'='*120}")
    print(f"{'Trajectory':<25} {'OK':>5} {'St':>5} {'Comp%':>6} {'CTE_m':>7} {'CTE50':>7} {'CTE90':>7} "
          f"{'Alt[min,max]':>18} {'Gmax':>5} {'Gp95':>5} {'Term':>8}")
    print("-"*120)
    for r in results:
        print(f"{r['trajectory']:<25} {str(r['completed']):>5} {r['steps']:5d} {r['completion_ratio']*100:5.0f}% "
              f"{r['cte_mean']:7.0f} {r['cte_p50']:7.0f} {r['cte_p90']:7.0f} "
              f"[{r['alt_min']:5.0f},{r['alt_max']:5.0f}] {r['G_max']:5.1f} {r['G_p95']:5.1f} {r['termination']:>8}")
    print(f"{'='*120}")

    outdir=os.path.join(_planax_root,"results/hierarchical_trajectory_tracking/horizontal_complex")
    os.makedirs(outdir,exist_ok=True)
    with open(os.path.join(outdir,"summary.csv"),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
    print(f"\nCSV: {outdir}/summary.csv")
    print("DONE")
