"""Debug: compare .mode() vs .sample() on quat checkpoint."""
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

rng,reset_key=jax.random.split(rng)
obs_dict,state=env.reset(reset_key,Heading_Pitch_V_TaskParams())
obs_vec=obs_dict[env.agents[0]]
obs_in=obs_vec[None,None,:];done_in=jnp.zeros((1,1))

# MODE (what render uses)
h_m,pi_m,_=net.apply(ckpt_params,h0,(obs_in,done_in))
acts_mode=[int(p.mode()[0,0]) for p in pi_m]

# SAMPLE (what training uses) - try 10 samples
print("MODE actions:", acts_mode)
print("\n10 SAMPLE actions:")
all_samples=[]
for seed in range(10):
    rng,k1,k2,k3,k4,k5=jax.random.split(jax.random.PRNGKey(seed),6)
    h_s,pi_s,_=net.apply(ckpt_params,h0,(obs_in,done_in))
    s=[int(pi_s[0].sample(seed=k1)[0,0]),int(pi_s[1].sample(seed=k2)[0,0]),
       int(pi_s[2].sample(seed=k3)[0,0]),int(pi_s[3].sample(seed=k4)[0,0]),
       int(pi_s[4].sample(seed=k5)[0,0])]
    all_samples.append(s)
    print(f"  seed={seed}: {s}")

# Entropy check
h_final,pi_final,_=net.apply(ckpt_params,h0,(obs_in,done_in))
for i,name in enumerate(['thr','el','ail','rud','sb']):
    ent=float(pi_final[i].entropy().mean())
    max_ent=float(np.log(pi_final[i].num_categories))
    print(f"  {name}: entropy={ent:.3f} / max={max_ent:.3f} = {100*ent/max_ent:.0f}%")
print('\nDONE')
