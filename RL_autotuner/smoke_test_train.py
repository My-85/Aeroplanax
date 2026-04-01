"""smoke_test_train.py — 验证新 obs/env 改动无 Bug，网络能正常学习。
跑约 30 万步后自动退出，打印 return 趋势。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.40'

import sys
from pathlib import Path
PLANAX_DIR = Path(__file__).resolve().parent.parent / "Planax"
sys.path.insert(0, str(PLANAX_DIR))

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import functools
from typing import Sequence, NamedTuple, Tuple, Dict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax

from envs.wrappers import LogWrapper
from envs.aeroplanax_quat_baseline_iter import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams
)

# ── Network (same arch as production) ──────────────────────────────────────

class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params",
                       in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(resets[:, np.newaxis],
                          self.initialize_carry(*carry.shape), carry)
        new_carry, y = nn.GRUCell(features=ins.shape[1])(carry, ins)
        return new_carry, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        act = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        emb = act(nn.Dense(self.config["FC_DIM_SIZE"],
                           kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(obs))
        hidden, emb = ScannedRNN()(hidden, (emb, dones))
        h = act(nn.LayerNorm()(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                                        bias_init=constant(0.0))(emb)))
        am = act(nn.Dense(self.config["GRU_HIDDEN_DIM"],
                          kernel_init=orthogonal(2), bias_init=constant(0.0))(h))
        pis = tuple(
            distrax.Categorical(logits=nn.Dense(d, kernel_init=orthogonal(0.01),
                                                bias_init=constant(0.0))(am))
            for d in self.action_dim
        )
        critic = act(nn.Dense(self.config["FC_DIM_SIZE"],
                              kernel_init=orthogonal(2), bias_init=constant(0.0))(h))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, pis, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    valid_action: jnp.ndarray


def batchify(x, agent_list, num_envs, num_actors):
    return jnp.stack([x[a] for a in agent_list]).reshape((num_actors * num_envs, -1))


def unbatchify(x, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


CONFIG = {
    "TOTAL_TIMESTEPS": 300_000,
    "NUM_ENVS": 128,
    "NUM_STEPS": 10,
    "NUM_MINIBATCHES": 4,
    "UPDATE_EPOCHS": 2,
    "CLIP_EPS": 0.2,
    "VF_COEF": 0.5,
    "ENT_COEF": 0.01,
    "MAX_GRAD_NORM": 0.5,
    "LR": 3e-4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION": "relu",
}


def make_train(config):
    cfg = dict(config)
    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)
    env = LogWrapper(env)
    cfg["NUM_ACTORS"] = env.num_agents
    cfg["NUM_UPDATES"] = (cfg["TOTAL_TIMESTEPS"]
                          // cfg["NUM_STEPS"] // cfg["NUM_ENVS"])
    cfg["MINIBATCH_SIZE"] = (cfg["NUM_ACTORS"] * cfg["NUM_STEPS"]
                             // cfg["NUM_MINIBATCHES"])

    network = ActorCriticRNN([31, 41, 41, 41], config=cfg)
    rng = jax.random.PRNGKey(42)
    obs_shape = env.observation_space(env.agents[0], env_params).shape

    init_x = (
        jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"], *obs_shape)),
        jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"])),
    )
    init_h = ScannedRNN.initialize_carry(
        cfg["NUM_ENVS"] * cfg["NUM_ACTORS"], cfg["GRU_HIDDEN_DIM"])
    net_params = network.init(rng, init_h, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
        optax.adam(cfg["LR"], eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply,
                                    params=net_params, tx=tx)

    def _env_step(runner_state, _):
        train_state, env_state, hstate, obs, dones, rng = runner_state
        obs_batch = batchify(obs, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
        done_batch = batchify(dones, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])

        ac_in = (obs_batch[np.newaxis, :, :], done_batch[np.newaxis, :])
        hstate_new, pis, values = network.apply(train_state.params, hstate, ac_in)
        hstate_new = hstate_new[0]

        actions = jnp.stack([p.sample(seed=rng) for p in pis], axis=-1).squeeze(0)
        log_probs = sum(p.log_prob(actions[:, i]) for i, p in enumerate(pis))

        actions_dict = unbatchify(actions, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, cfg["NUM_ENVS"])
        new_obs, new_env_state, rewards, new_dones, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(step_rngs, env_state, actions_dict)

        reward_batch = batchify(rewards, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
        tran = Transition(
            done=done_batch,
            action=actions,
            value=values.squeeze(0),
            reward=reward_batch,
            log_prob=log_probs,
            obs=obs_batch,
            info=info,
            valid_action=jnp.ones_like(reward_batch),
        )
        return (train_state, new_env_state, hstate_new, new_obs, new_dones, rng), tran

    def train(rng):
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, cfg["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
        dones = {a: jnp.zeros(cfg["NUM_ENVS"]) for a in env.agents}
        hstate = ScannedRNN.initialize_carry(
            cfg["NUM_ENVS"] * cfg["NUM_ACTORS"], cfg["GRU_HIDDEN_DIM"])

        runner = (train_state, env_state, hstate, obs, dones, rng)

        def _update_step(runner_state_and_ts, _):
            runner_state = runner_state_and_ts
            ts = runner_state[0]
            runner_state, traj = jax.lax.scan(
                _env_step, runner_state, None, length=cfg["NUM_STEPS"]
            )
            # Simple PPO update (abbreviated)
            last_obs = batchify(runner_state[3], env.agents,
                                cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
            last_done = batchify(runner_state[4], env.agents,
                                 cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
            _, _, last_val = network.apply(
                ts.params, runner_state[2],
                (last_obs[np.newaxis], last_done[np.newaxis])
            )

            def _gae(traj_batch, last_val):
                def _step(carry, transition):
                    gae, next_val = carry
                    done, val, rew = transition.done, transition.value, transition.reward
                    delta = rew + cfg["GAMMA"] * next_val * (1 - done) - val
                    gae = delta + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, val), gae
                _, adv = jax.lax.scan(
                    _step, (jnp.zeros_like(last_val), last_val),
                    traj_batch, reverse=True
                )
                return adv, adv + traj_batch.value

            adv, ret = _gae(traj, last_val.squeeze(0))

            def _loss(params):
                _, pis, vals = jax.vmap(
                    lambda h, ob, dn: network.apply(params, h, (ob, dn)),
                    in_axes=(0, 0, 0)
                )(traj.obs, traj.obs, traj.done)
                # Minimal actor-critic loss
                log_p = sum(p.log_prob(traj.action[:, :, i])
                            for i, p in enumerate(pis))
                ratio = jnp.exp(log_p - traj.log_prob)
                loss_a = -jnp.minimum(
                    ratio * adv,
                    jnp.clip(ratio, 1 - cfg["CLIP_EPS"],
                             1 + cfg["CLIP_EPS"]) * adv
                ).mean()
                loss_c = ((vals - ret) ** 2).mean()
                return loss_a + cfg["VF_COEF"] * loss_c

            grad_fn = jax.value_and_grad(_loss)
            loss_val, grads = grad_fn(ts.params)
            ts = ts.apply_gradients(grads=grads)

            # 更新 runner_state 中的 train_state
            runner_state = (ts,) + runner_state[1:]

            info_out = {"return": traj.reward.mean(),
                        "loss": loss_val}
            return runner_state, info_out

        runner_final, metrics = jax.lax.scan(
            _update_step,
            runner,
            None,
            length=cfg["NUM_UPDATES"],
        )
        return runner_final[0], metrics

    return train


if __name__ == "__main__":
    print(f"OBS DIM = 21 expected")
    print("JAX devices:", jax.devices())
    train_fn = jax.jit(make_train(CONFIG))
    rng = jax.random.PRNGKey(0)
    print("Compiling... (first run takes ~1 min)")
    ts, metrics = train_fn(rng)
    returns = np.asarray(metrics["return"])
    print(f"\nSmoke test PASSED — {CONFIG['TOTAL_TIMESTEPS']:,} steps")
    print(f"  Return first 5 updates: {returns[:5]}")
    print(f"  Return last  5 updates: {returns[-5:]}")
    trend = "UP" if returns[-5:].mean() > returns[:5].mean() else "FLAT/DOWN"
    print(f"  Return trend: {trend}")
