"""Debug: compare CKPT vs FRESH policy actions on env.reset() observation (with speed brake)."""
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
rng=jax.random.PRNGKey(42)
obs_shape=env.observation_space(env.agents[0],Heading_Pitch_V_TaskParams()).shape
h0=ScannedRNN.initialize_carry(1,128)
fresh_params=net.init(rng,h0,(jnp.zeros((1,1,*obs_shape)),jnp.zeros((1,1))))
CKPT=os.path.abspath('results/heading_pitch_V_discrete_rnn_2026-05-11-17-40/checkpoints/checkpoint_epoch_1000')
ckptr=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
ckpt=ckptr.restore(CKPT,args=ocp.args.StandardRestore())
ckpt_params=ckpt['params']

# Compare shapes
fresh_lvs=jax.tree_util.tree_leaves(fresh_params)
ckpt_lvs=jax.tree_util.tree_leaves(ckpt_params)
print(f'Fresh leaves: {len(fresh_lvs)}, CKPT leaves: {len(ckpt_lvs)}')
all_match=all(a.shape==b.shape for a,b in zip(fresh_lvs,ckpt_lvs)) and len(fresh_lvs)==len(ckpt_lvs)
print(f'All shapes match: {all_match}')
if not all_match:
    for i,(a,b) in enumerate(zip(fresh_lvs,ckpt_lvs)):
        if a.shape!=b.shape:
            print(f'  MISMATCH {i}: fresh={a.shape} ckpt={b.shape}')

# Key structure
def key_paths(tree):
    return ['.'.join(str(p.key) for p in path if hasattr(p,'key')) for path,_ in jax.tree_util.tree_flatten_with_path(tree)[0]]
fp=key_paths(fresh_params);cp=key_paths(ckpt_params)
print(f'Key structure match: {fp==cp}')
if fp!=cp:
    for i,(f,c) in enumerate(zip(fp,cp)):
        if f!=c: print(f'  KEY MISMATCH {i}: fresh={f} ckpt={c}')

# Forward pass comparison
rng,reset_key=jax.random.split(rng)
obs_dict,state=env.reset(reset_key,Heading_Pitch_V_TaskParams())
obs_vec=obs_dict[env.agents[0]]
obs_in=obs_vec[None,None,:];done_in=jnp.zeros((1,1))

h_ckpt,pi_ckpt,_=net.apply(ckpt_params,h0,(obs_in,done_in))
h_fresh,pi_fresh,_=net.apply(fresh_params,h0,(obs_in,done_in))

print(f'\nObs qv=[{obs_vec[0]:.3f},{obs_vec[1]:.3f},{obs_vec[2]:.3f}] dvt={obs_vec[3]:.3f} v_b=[{obs_vec[6]:.3f},{obs_vec[7]:.3f},{obs_vec[8]:.3f}]')
print(f'CKPT  acts: thr={int(pi_ckpt[0].mode()[0,0])} el={int(pi_ckpt[1].mode()[0,0])} ail={int(pi_ckpt[2].mode()[0,0])} rud={int(pi_ckpt[3].mode()[0,0])} sb={int(pi_ckpt[4].mode()[0,0])}')
print(f'FRESH acts: thr={int(pi_fresh[0].mode()[0,0])} el={int(pi_fresh[1].mode()[0,0])} ail={int(pi_fresh[2].mode()[0,0])} rud={int(pi_fresh[3].mode()[0,0])} sb={int(pi_fresh[4].mode()[0,0])}')
print('DONE')
