"""
Validate: straight line waypoints + pure pursuit guidance → lower baseline.
Uses blend smoothing per waypoint (reset on each WP reach).

Usage:
    python experiments/hierarchical_trajectory_tracking/render_straight_debug.py
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.3'

_planax_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _planax_root)

import jax, jax.numpy as jnp, numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import functools, distrax
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from typing import Sequence, Dict

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
    _quat_from_euler_nb, _quat_conj,
)
from experiments.hierarchical_trajectory_tracking.trajectory_generators import straight_line
from experiments.hierarchical_trajectory_tracking.guidance_baselines import pure_pursuit
from experiments.hierarchical_trajectory_tracking.path_utils import cross_track_error_2d

# Network (same as training)
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
NET_CONFIG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
SEED = 42; MAX_STEPS = 1000; REACH_RADIUS = 500.0; BLEND_STEPS = 200
def _f(x): a=np.asarray(x); return float(a) if a.ndim==0 else float(a.reshape(-1)[0])

def main():
    env = AeroPlanaxHeading_Pitch_V_Env(Heading_Pitch_V_TaskParams())
    net = ActorCriticRNN([31,41,41,41,5], config=NET_CONFIG)
    rng = jax.random.PRNGKey(SEED)
    obs_shape = env.observation_space(env.agents[0], Heading_Pitch_V_TaskParams()).shape
    h0 = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    net_params = net.init(rng, h0, (jnp.zeros((1,1,*obs_shape)), jnp.zeros((1,1))))

    print(f"Loading checkpoint: {CKPT}")
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(CKPT, args=ocp.args.StandardRestore())
    net_params = ckpt["params"]

    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, Heading_Pitch_V_TaskParams())
    # Align heading North
    q_nb = _quat_from_euler_nb(0.0,0.0,0.0); q_bn=_quat_conj(q_nb)
    state = state.replace(plane_state=state.plane_state.replace(
        yaw=jnp.array([0.0]),q0=jnp.array([q_bn[0]]),q1=jnp.array([q_bn[1]]),
        q2=jnp.array([q_bn[2]]),q3=jnp.array([q_bn[3]])),
        target_heading=jnp.array([0.0]))

    origin_n=_f(state.plane_state.north); origin_e=_f(state.plane_state.east)
    origin_alt=_f(state.plane_state.altitude)
    waypoints, meta = straight_line(origin_n, origin_e, origin_alt, 0.0, length=15000.0, n_points=10)
    print(f"Trajectory: {meta['name']}, {meta['n_points']} WP, {meta['total_length_m']:.0f}m")
    arc_cum = np.cumsum([0]+[np.linalg.norm(waypoints[i+1]-waypoints[i]) for i in range(len(waypoints)-1)])

    hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    done_flag=jnp.zeros((1,)); current_wp=0; wp_blend_start=0
    rec_t,rec_n,rec_e,rec_a,rec_vt=[],[],[],[],[]
    rec_cte=[]; reached_steps=[]

    print(f"\n{'Step':>5} | {'WP':>3} | {'Dist':>7} | {'CTE':>7} | {'Alt':>6} | {'Vt':>5} | {'Roll':>6}")
    print("-"*65)

    for step in range(MAX_STEPS):
        ps=state.plane_state
        north=_f(ps.north); east=_f(ps.east); alt=_f(ps.altitude)
        vt=_f(ps.vt); roll=_f(ps.roll); pitch=_f(ps.pitch); yaw=_f(ps.yaw)

        raw_h,raw_p,raw_r,raw_v,_ = pure_pursuit(north,east,alt,vt,yaw,pitch,roll,waypoints,current_wp)

        dist=np.sqrt((waypoints[current_wp,0]-north)**2+(waypoints[current_wp,1]-east)**2+(waypoints[current_wp,2]-alt)**2)
        if dist < REACH_RADIUS and current_wp < len(waypoints)-1:
            current_wp+=1; wp_blend_start=step; reached_steps.append(step)

        # Blend smoothing per-WP
        blend = min(1.0, (step-wp_blend_start)/BLEND_STEPS)
        hdg_err=float(np.arctan2(np.sin(raw_h-yaw),np.cos(raw_h-yaw)))
        t_hdg=float(np.arctan2(np.sin(yaw+blend*hdg_err),np.cos(yaw+blend*hdg_err)))
        t_pitch=float(pitch+blend*(raw_p-pitch))
        t_roll=float(roll+blend*(raw_r-roll))
        t_vt=float(vt+blend*(raw_v-vt))

        cte,_=cross_track_error_2d(north,east,waypoints[max(0,current_wp-1)],waypoints[current_wp])

        state=state.replace(target_heading=jnp.array([t_hdg]),target_pitch=jnp.array([t_pitch]),
                            target_roll=jnp.array([t_roll]),target_vt=jnp.array([t_vt]))
        obs_in=env._get_obs(state,Heading_Pitch_V_TaskParams())[env.agents[0]][None,None,:]

        hstate,pi,_=net.apply(net_params,hstate,(obs_in,done_flag[None,:]))
        acts=[int(p.mode()[0,0]) for p in pi]
        action={env.agents[0]:jnp.array(acts)}
        rng,sk=jax.random.split(rng)
        obs2,state,rew,done,info=env.step(sk,state,action,Heading_Pitch_V_TaskParams())
        done_flag=jnp.array([float(done[env.agents[0]])])

        rec_t.append(step*0.2);rec_n.append(north);rec_e.append(east);rec_a.append(alt)
        rec_vt.append(vt);rec_cte.append(cte)

        if step%30==0 or (step>0 and current_wp>(rec_t[max(0,step-2)] and len(reached_steps)>0)):
            print(f"{step:5d} | {current_wp:3d} | {dist:7.0f} | {cte:7.0f} | {alt:6.0f} | {vt:5.0f} | {np.degrees(roll):+6.1f}")

        if bool(done[env.agents[0]]):
            print(f"  CRASHED at step {step}"); break
        if current_wp>=len(waypoints)-1 and dist<REACH_RADIUS:
            print(f"\n[SUCCESS] All {len(waypoints)} waypoints reached at step {step}!")
            break

    # Summary
    t_a=np.array(rec_t); n=len(t_a)
    cte_a=np.array(rec_cte)
    print(f"\nSteps: {n}, WP reached: {current_wp+1}/{len(waypoints)}")
    print(f"Cross-track: mean={cte_a.mean():.0f}m p95={np.percentile(cte_a,95):.0f}m max={cte_a.max():.0f}m")
    print(f"Alt range: {np.array(rec_a).min():.0f}-{np.array(rec_a).max():.0f}m")
    print(f"Vt range: {np.array(rec_vt).min():.0f}-{np.array(rec_vt).max():.0f}m/s")

    # Quick plot
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(rec_e,rec_n,'b-',lw=0.8); axes[0].scatter(waypoints[:,1],waypoints[:,0],c='orange',s=15)
    axes[0].set_title("Top-down"); axes[0].set_aspect('equal')
    axes[1].plot(t_a,rec_a); axes[1].set_title("Altitude")
    axes[2].plot(t_a,cte_a); axes[2].set_title("Cross-track error (m)")
    for ax in axes: ax.grid(True,alpha=0.3)
    outdir=os.path.join(_planax_root,"results/vertical_loop_test")
    os.makedirs(outdir,exist_ok=True)
    fig.savefig(os.path.join(outdir,"straight_debug.png"),dpi=120,bbox_inches='tight')
    plt.close(fig)
    print("DONE")

if __name__=="__main__":
    main()
