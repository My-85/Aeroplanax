"""Test: does the quat policy crash on different initial states?"""
import os;os.environ['CUDA_VISIBLE_DEVICES']='0'
import jax,jax.numpy as jnp,numpy as np
import flax.linen as nn
from flax.linen.initializers import constant,orthogonal
import functools,distrax
from typing import Sequence,Dict
import orbax.checkpoint as ocp

class ScannedRNN(nn.Module):
    @functools.partial(nn.scan,variable_broadcast='params',in_axes=0,out_axes=0,split_rngs={'params':False})
    @nn.compact
    def __call__(self,carry,x):
        rnn_state=carry;ins,resets=x
        rnn_state=jnp.where(resets[:,np.newaxis],self.initialize_carry(*rnn_state.shape),rnn_state)
        new_rnn_state,y=nn.GRUCell(features=ins.shape[1])(rnn_state,ins)
        return new_rnn_state,y
    @staticmethod
    def initialize_carry(bs,hs):
        return nn.GRUCell(features=hs).initialize_carry(jax.random.PRNGKey(0),(bs,hs))

class ActorCriticRNN(nn.Module):
    action_dim:Sequence[int];config:Dict
    @nn.compact
    def __call__(self,hidden,x):
        ac=nn.relu if self.config['ACTIVATION']=='relu' else nn.tanh;obs,dones=x
        e=ac(nn.Dense(self.config['FC_DIM_SIZE'],kernel_init=orthogonal(np.sqrt(2)),bias_init=constant(0.0))(obs))
        hidden,e=ScannedRNN()(hidden,(e,dones))
        fc2=ac(nn.LayerNorm()(nn.Dense(256,kernel_init=orthogonal(np.sqrt(2)),bias_init=constant(0.0))(e)))
        am=ac(nn.Dense(self.config['GRU_HIDDEN_DIM'],kernel_init=orthogonal(2),bias_init=constant(0.0))(fc2))
        heads=[]
        for i in range(4):
            heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[i],kernel_init=orthogonal(0.01),bias_init=constant(0.0))(am)))
        heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[4],kernel_init=constant(0.0),
            bias_init=lambda key,shape,dtype=jnp.float32:jnp.array([0.0,-1.5,-1.5,-1.5,-1.5],dtype=dtype))(am)))
        c=ac(nn.Dense(self.config['FC_DIM_SIZE'],kernel_init=orthogonal(2),bias_init=constant(0.0))(fc2))
        c=nn.Dense(1,kernel_init=orthogonal(1.0),bias_init=constant(0.0))(c)
        return hidden,(heads[0],heads[1],heads[2],heads[3],heads[4]),jnp.squeeze(c,axis=-1)

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import AeroPlanaxHeading_Pitch_V_Env,Heading_Pitch_V_TaskParams
env=AeroPlanaxHeading_Pitch_V_Env(Heading_Pitch_V_TaskParams())
cfg={'FC_DIM_SIZE':128,'GRU_HIDDEN_DIM':128,'ACTIVATION':'relu'}
net=ActorCriticRNN([31,41,41,41,5],config=cfg)
rng=jax.random.PRNGKey(111)
obs_shape=env.observation_space(env.agents[0],Heading_Pitch_V_TaskParams()).shape
h0=ScannedRNN.initialize_carry(1,128)
CKPT=os.path.abspath('results/heading_pitch_V_discrete_rnn_2026-05-11-17-40/checkpoints/checkpoint_epoch_1000')
ckptr=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
ckpt=ckptr.restore(CKPT,args=ocp.args.StandardRestore())
ckpt_params=ckpt['params']

print(f"{'seed':>5} | {'yaw':>7} | {'pitch':>7} | {'roll':>7} | {'vt':>6} | {'alt':>7} | {'done':>5} | {'status_post':>12} | {'az_post':>8} | {'actions'}")
print("-"*120)

crash_count=0
ok_count=0
for seed in range(20):
    rng=rp=jax.random.PRNGKey(seed)
    rng,reset_key=jax.random.split(rng)
    obs_dict,state=env.reset(reset_key,Heading_Pitch_V_TaskParams())
    ps=state.plane_state
    yaw=float(np.asarray(ps.yaw).reshape(-1)[0])
    pitch=float(np.asarray(ps.pitch).reshape(-1)[0])
    roll=float(np.asarray(ps.roll).reshape(-1)[0])
    vt=float(np.asarray(ps.vt).reshape(-1)[0])
    alt=float(np.asarray(ps.altitude).reshape(-1)[0])

    obs_vec=obs_dict[env.agents[0]]
    obs_in=obs_vec[None,None,:];done_in=jnp.zeros((1,1))
    h_ckpt,pi_ckpt,_=net.apply(ckpt_params,h0,(obs_in,done_in))
    acts=[int(p.mode()[0,0]) for p in pi_ckpt]

    action={env.agents[0]:jnp.array(acts)}
    rng,key=jax.random.split(rng)
    obs2,state2,rew,done,info=env.step(key,state,action,Heading_Pitch_V_TaskParams())
    d=bool(np.asarray(done[env.agents[0]]).item())
    ps2=state2.plane_state
    status_post=int(np.asarray(ps2.status).reshape(-1)[0])
    az_post=float(np.asarray(ps2.az).reshape(-1)[0])
    crash_flag='CRASH' if d else 'OK'
    if d:crash_count+=1
    else:ok_count+=1
    al=float(np.asarray(ps2.altitude).reshape(-1)[0])
    print(f"{seed:5d} | {np.degrees(yaw):+7.1f} | {np.degrees(pitch):+7.1f} | {np.degrees(roll):+7.1f} | {vt:6.1f} | {al:7.0f} | {str(d):>5} | {status_post:>12} | {az_post:+8.2f} | {acts}")

print(f"\nCrash: {crash_count}/{20}, OK: {ok_count}/{20}")
print('DONE')
