"""
Ablation v2: fixed, moving, multi + blend_steps sweep + diagnostic log.

Usage:
    python experiments/hierarchical_trajectory_tracking/render_ablation_tests.py fixed
    python experiments/hierarchical_trajectory_tracking/render_ablation_tests.py moving
    python experiments/hierarchical_trajectory_tracking/render_ablation_tests.py multi
    python experiments/hierarchical_trajectory_tracking/render_ablation_tests.py blend_sweep
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('XLA_PYTHON_MEM_FRACTION', '0.3')

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
from experiments.hierarchical_trajectory_tracking.trajectory_generators import straight_line, single_waypoint
from experiments.hierarchical_trajectory_tracking.path_manager import PathManager
from experiments.hierarchical_trajectory_tracking.subgoal_generator import pure_pursuit_subgoal
from experiments.hierarchical_trajectory_tracking.target_blender import TargetBlender

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
SEED, MAX_STEPS, REACH_R = 42, 600, 500.0
def _f(x): a=np.asarray(x); return float(a) if a.ndim==0 else float(a.reshape(-1)[0])


def run_test(mode: str, blend_steps: int = 200):
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
    q_nb=_quat_from_euler_nb(0.0,0.0,0.0); q_bn=_quat_conj(q_nb)
    state=state.replace(plane_state=state.plane_state.replace(
        yaw=jnp.array([0.0]),q0=jnp.array([q_bn[0]]),q1=jnp.array([q_bn[1]]),
        q2=jnp.array([q_bn[2]]),q3=jnp.array([q_bn[3]])), target_heading=jnp.array([0.0]))

    origin_n=_f(state.plane_state.north); origin_e=_f(state.plane_state.east)
    origin_alt=_f(state.plane_state.altitude)

    if mode == "fixed":
        wps,_ = single_waypoint(origin_n, origin_e, origin_alt, 0.0, distance=5000.0)
        pm = PathManager(wps, mode="waypoint", reach_radius=REACH_R)
    elif mode == "moving":
        wps,_ = single_waypoint(origin_n, origin_e, origin_alt, 0.0, distance=15000.0)
        wps = np.array([[origin_n, origin_e, origin_alt], [wps[0,0], wps[0,1], wps[0,2]]])
        pm = PathManager(wps, mode="lookahead", lookahead_dist=1000.0, reach_radius=REACH_R)
    elif mode == "multi":
        wps, meta = straight_line(origin_n, origin_e, origin_alt, 0.0, length=15000.0, n_points=10)
        pm = PathManager(wps, mode="waypoint", reach_radius=REACH_R)
    else:
        raise ValueError(mode)

    tag = f"{mode}_b{blend_steps}"
    print(f"\n{'='*65}")
    print(f"Mode: {mode}, blend_steps: {blend_steps}")
    print(f"  WPs: {len(wps)}, total arc: {pm.arc[-1]:.0f}m")

    blender = TargetBlender(blend_steps=blend_steps)
    hstate = ScannedRNN.initialize_carry(1, NET_CFG["GRU_HIDDEN_DIM"])
    done_flag=jnp.zeros((1,))
    init_yaw=_f(state.plane_state.yaw); init_pitch=_f(state.plane_state.pitch)
    init_roll=_f(state.plane_state.roll); init_vt=_f(state.plane_state.vt)
    blender.reset(init_yaw, init_pitch, init_roll, init_vt)

    rec={"t":[],"n":[],"e":[],"a":[],"vt":[],"roll":[],"pitch":[],"yaw":[],
         "t_hdg":[],"t_pitch":[],"t_roll":[],"t_vt":[],"dist":[]}
    diag_step=0

    for step in range(MAX_STEPS):
        ps=state.plane_state
        north=_f(ps.north); east=_f(ps.east); alt=_f(ps.altitude)
        vt=_f(ps.vt); roll=_f(ps.roll); pitch=_f(ps.pitch); yaw=_f(ps.yaw)

        path_ctx = pm.update(north, east, alt)
        if pm.just_reached:
            blender.reset(yaw, pitch, roll, vt)

        raw_h, raw_p, raw_r, raw_v, _ = pure_pursuit_subgoal(path_ctx, yaw, pitch, vt)
        t_hdg, t_pitch, t_roll, t_vt = blender.blend(raw_h, raw_p, raw_r, raw_v, yaw, pitch, roll, vt)

        # Diagnostic log
        if step < 20:
            la = path_ctx["lookahead_wp_abs"]
            err = path_ctx["lookahead_error_world"]
            print(f"  [{step:2d}] pos=({north:6.0f},{east:6.0f},{alt:6.0f})  "
                  f"LA_abs=({la[0]:6.0f},{la[1]:6.0f},{la[2]:6.0f})  "
                  f"err=({err[0]:6.0f},{err[1]:6.0f},{err[2]:6.0f})  "
                  f"tgt=(h:{np.degrees(t_hdg):+6.1f} p:{np.degrees(t_pitch):+6.1f} r:{np.degrees(t_roll):+5.1f})  "
                  f"ypr=({np.degrees(yaw):+6.1f},{np.degrees(pitch):+6.1f},{np.degrees(roll):+6.1f})  "
                  f"dist={path_ctx['dist_to_wp']:.0f}m")
            diag_step+=1

        state=state.replace(target_heading=jnp.array([t_hdg]),target_pitch=jnp.array([t_pitch]),
                            target_roll=jnp.array([t_roll]),target_vt=jnp.array([t_vt]))
        obs_in=env._get_obs(state,Heading_Pitch_V_TaskParams())[env.agents[0]][None,None,:]
        hstate,pi,_=net.apply(net_params,hstate,(obs_in,done_flag[None,:]))
        acts=[int(p.mode()[0,0]) for p in pi]
        rng,sk=jax.random.split(rng)
        obs2,state,rew,done,info=env.step(sk,state,{env.agents[0]:jnp.array(acts)},Heading_Pitch_V_TaskParams())
        done_flag=jnp.array([float(done[env.agents[0]])])

        rec["t"].append(step*0.2);rec["n"].append(north);rec["e"].append(east);rec["a"].append(alt)
        rec["vt"].append(vt);rec["roll"].append(np.degrees(roll));rec["pitch"].append(np.degrees(pitch))
        rec["yaw"].append(np.degrees(yaw))
        rec["t_hdg"].append(np.degrees(t_hdg));rec["t_pitch"].append(np.degrees(t_pitch))
        rec["t_roll"].append(np.degrees(t_roll));rec["t_vt"].append(t_vt)
        rec["dist"].append(path_ctx["dist_to_wp"])

        if step%50==0 and step>0:
            print(f"  {step:4d}: dist={path_ctx['dist_to_wp']:.0f}m alt={alt:.0f}m vt={vt:.0f}m/s")

        if bool(done[env.agents[0]]):
            print(f"  CRASHED at step {step}"); break
        if pm.is_done():
            print(f"\n  [SUCCESS] Reached end at step {step}!")
            break

    # Summary
    t_a=np.array(rec["t"]); n=len(t_a)
    completed = pm.is_done()
    final_dist = rec["dist"][-1] if n>0 else -1
    alt_a=np.array(rec["a"]); vt_a=np.array(rec["vt"])

    print(f"\n  Completion: {completed}  Steps: {n}  Final dist: {final_dist:.0f}m")
    print(f"  Alt: [{alt_a.min():.0f},{alt_a.max():.0f}]m  Vt: [{vt_a.min():.0f},{vt_a.max():.0f}]m/s")
    print(f"  WP reached: {pm.wp_reached_count}")

    # Plot
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Ablation: mode={mode} blend_steps={blend_steps}  "
                 f"{'COMPLETED' if completed else 'FAILED'}  "
                 f"alt=[{alt_a.min():.0f},{alt_a.max():.0f}]m", fontsize=12)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    ax=fig.add_subplot(gs[0,0]); ax.plot(rec["e"],rec["n"],'b-',lw=0.8)
    ax.scatter(wps[:,1],wps[:,0],c='orange',s=15); ax.set_title("Top-down"); ax.set_aspect('equal'); ax.grid(True,alpha=0.3)
    ax=fig.add_subplot(gs[0,1]); ax.plot(t_a,rec["a"]); ax.set_title("Altitude"); ax.grid(True,alpha=0.3)
    ax=fig.add_subplot(gs[0,2]); ax.plot(t_a,rec["vt"]); ax.axhline(y=250,c='gray',ls='--'); ax.set_title("Airspeed"); ax.grid(True,alpha=0.3)
    ax=fig.add_subplot(gs[0,3]); ax.plot(t_a,rec["dist"]); ax.set_title("Dist to target"); ax.grid(True,alpha=0.3)

    ax=fig.add_subplot(gs[1,0]); ax.plot(t_a,rec["t_hdg"],label='tgt_h'); ax.plot(t_a,rec["yaw"],label='yaw')
    ax.set_title("Heading"); ax.legend(fontsize=6); ax.grid(True,alpha=0.3)
    ax=fig.add_subplot(gs[1,1]); ax.plot(t_a,rec["t_pitch"],label='tgt_p'); ax.plot(t_a,rec["pitch"],label='pitch')
    ax.set_title("Pitch"); ax.legend(fontsize=6); ax.grid(True,alpha=0.3)
    ax=fig.add_subplot(gs[1,2]); ax.plot(t_a,rec["t_roll"],label='tgt_r'); ax.plot(t_a,rec["roll"],label='roll')
    ax.set_title("Roll"); ax.legend(fontsize=6); ax.grid(True,alpha=0.3)
    ax=fig.add_subplot(gs[1,3]); ax.plot(t_a,rec["t_vt"]); ax.set_title("Target Vt"); ax.grid(True,alpha=0.3)

    outdir=os.path.join(_planax_root,"results/vertical_loop_test")
    os.makedirs(outdir,exist_ok=True)
    fig.savefig(os.path.join(outdir,f"ablation_{tag}.png"),dpi=120,bbox_inches='tight')
    plt.close(fig)

    return {"mode": mode, "blend_steps": blend_steps, "completed": completed,
            "steps": n, "final_dist": final_dist, "alt_min": alt_a.min(),
            "alt_max": alt_a.max(), "vt_min": vt_a.min(), "vt_max": vt_a.max(),
            "wp_reached": pm.wp_reached_count}


if __name__=="__main__":
    mode = sys.argv[1] if len(sys.argv)>1 else "multi"

    if mode == "blend_sweep":
        for bs in [25, 50, 100, 200]:
            for m in ["fixed", "moving", "multi"]:
                run_test(m, bs)
    else:
        run_test(mode, 200)

    print("\nDONE")
