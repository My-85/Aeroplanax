"""Debug: compare CKPT vs FRESH policy actions on env.reset() observation."""
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
        pi=[distrax.Categorical(logits=nn.Dense(self.action_dim[i],kernel_init=orthogonal(0.01),bias_init=constant(0.0))(am)) for i in range(4)]
        c=ac(nn.Dense(self.config['FC_DIM_SIZE'],kernel_init=orthogonal(2),bias_init=constant(0.0))(fc2))
        c=nn.Dense(1,kernel_init=orthogonal(1.0),bias_init=constant(0.0))(c)
        return hidden,(pi[0],pi[1],pi[2],pi[3]),jnp.squeeze(c,axis=-1)

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import AeroPlanaxHeading_Pitch_V_Env,Heading_Pitch_V_TaskParams
env=AeroPlanaxHeading_Pitch_V_Env(Heading_Pitch_V_TaskParams())
cfg={'FC_DIM_SIZE':128,'GRU_HIDDEN_DIM':128,'ACTIVATION':'relu'}
net=ActorCriticRNN([31,41,41,41],config=cfg)
rng=jax.random.PRNGKey(42)
obs_shape=env.observation_space(env.agents[0],Heading_Pitch_V_TaskParams()).shape
h0=ScannedRNN.initialize_carry(1,128)
fresh_params=net.init(rng,h0,(jnp.zeros((1,1,*obs_shape)),jnp.zeros((1,1))))
CKPT=os.path.abspath('results/heading_pitch_V_discrete_rnn_2026-05-10-16-49/checkpoints/checkpoint_epoch_300')
ckptr=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
ckpt=ckptr.restore(CKPT,args=ocp.args.StandardRestore())
ckpt_params=ckpt['params']

rng,reset_key=jax.random.split(rng)
obs_dict,state=env.reset(reset_key,Heading_Pitch_V_TaskParams())
obs_vec=obs_dict[env.agents[0]]
ps=state.plane_state
yaw=float(np.asarray(ps.yaw).reshape(-1)[0]);pitch=float(np.asarray(ps.pitch).reshape(-1)[0])
roll=float(np.asarray(ps.roll).reshape(-1)[0]);vt=float(np.asarray(ps.vt).reshape(-1)[0])
print(f'RESET obs: qv=[{obs_vec[0]:.3f},{obs_vec[1]:.3f},{obs_vec[2]:.3f}] dvt={obs_vec[3]:.3f} v_b=[{obs_vec[6]:.3f},{obs_vec[7]:.3f},{obs_vec[8]:.3f}]')
print(f'RESET state: yaw={np.degrees(yaw):.1f}° pitch={np.degrees(pitch):.1f}° roll={np.degrees(roll):.1f}° vt={vt:.1f}')
print(f'RESET targets: h={np.degrees(float(np.asarray(state.target_heading).reshape(-1)[0])):.1f}° p={np.degrees(float(np.asarray(state.target_pitch).reshape(-1)[0])):.1f}° r={np.degrees(float(np.asarray(state.target_roll).reshape(-1)[0])):.1f}° v={float(np.asarray(state.target_vt).reshape(-1)[0]):.1f}')

obs_in=obs_vec[None,None,:];done_in=jnp.zeros((1,1))
h_ckpt,pi_ckpt,_=net.apply(ckpt_params,h0,(obs_in,done_in))
acts_ckpt=[int(p.mode()[0,0]) for p in pi_ckpt]
h_fresh,pi_fresh,_=net.apply(fresh_params,h0,(obs_in,done_in))
acts_fresh=[int(p.mode()[0,0]) for p in pi_fresh]
print(f'CKPT acts: thr={acts_ckpt[0]} el={acts_ckpt[1]} ail={acts_ckpt[2]} rud={acts_ckpt[3]}')
print(f'FRESH acts: thr={acts_fresh[0]} el={acts_fresh[1]} ail={acts_fresh[2]} rud={acts_fresh[3]}')
print('DONE')
