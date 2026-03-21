"""
train_full_domain_maneuver_v3.py
=================================
Full-domain maneuver training v3: fixes from v2 analysis.

Key fixes (env/reward modified in-place):
  1. Random heading/vt targets NOW scaled by curriculum (was full [0,2pi] range!)
  2. Triple-scale attitude reward: coarse(60°)+medium(20°)+fine(5°) for smooth gradient
     - v2 had gradient desert at 20-40° (70% weight on 5° fine term was dead there)
  3. Combined tracking: weighted sum (0.75*att + 0.25*spd) instead of product
  4. Curriculum-dependent sustained_on_target: 3+3*level (was fixed 25, unreachable early)
  5. Terminal printing of all reward components for debugging
  6. End-of-training summary with all key metrics
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.95'
os.environ['WANDB_API_KEY'] = '4c0cc04699296bed768adea4824fbaecea35dc59'

import jax
import wandb
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import optax
from flax.linen.initializers import constant, orthogonal
import functools
from typing import Sequence, NamedTuple, Tuple, Optional, Union, Any, Dict
from flax.training.train_state import TrainState
import distrax
import tensorboardX
import jax.experimental
from envs.wrappers import LogWrapper
from envs.aeroplanax_full_domain_maneuver import AeroPlanaxFullDomainEnv, FullDomain_TaskParams
import orbax.checkpoint as ocp

# ======================== Global tracking for end-of-training summary ========================
_training_history = {
    "env_steps": [],
    "return": [],
    "theta_deg": [],
    "delta_vt": [],
    "r_main": [],
    "r_nz": [],
    "r_qbar": [],
    "curriculum_level": [],
    "on_target_steps": [],
    "success_times": [],
    "timeout_count": [],
    "actor_loss": [],
    "value_loss": [],
    "entropy": [],
    "approx_kl": [],
}

def _clip_scalar(x, lo, hi):
    return jnp.minimum(jnp.maximum(x, lo), hi)

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        embedding = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)
        actor_throttle_mean = nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_elevator_mean = nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_aileron_mean  = nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_rudder_mean   = nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi_throttle = distrax.Categorical(logits=actor_throttle_mean)
        pi_elevator = distrax.Categorical(logits=actor_elevator_mean)
        pi_aileron  = distrax.Categorical(logits=actor_aileron_mean)
        pi_rudder   = distrax.Categorical(logits=actor_rudder_mean)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    valid_action: jnp.ndarray

def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    cfg = dict(config)
    cfg.setdefault("VF_CLIP_EPS", 0.20)
    cfg.setdefault("HUBER_DELTA", 1.0)
    cfg.setdefault("TARGET_KL", 0.015)
    cfg.setdefault("KL_STOP_MULT", 1.5)
    cfg.setdefault("ENT_COEF_MIN", 1e-3)
    cfg.setdefault("ENT_COEF_MAX", 5e-2)
    cfg.setdefault("ENT_ADJ_RATE", 1.05)
    cfg.setdefault("LR_DECAY", 0.999)
    cfg.setdefault("MIN_LR_MULT", 0.2)

    cfg.setdefault("WARMUP_UPDATES",     2000)
    cfg.setdefault("KL_START_MULT",      5.0)
    cfg.setdefault("KL_RAMP_UPDATES",    1000)

    cfg.setdefault("FREEZE_ENTROPY_DURING_WARMUP", True)
    cfg.setdefault("FREEZE_LR_DURING_WARMUP",      True)
    cfg.setdefault("DISABLE_KL_STOP_DURING_WARMUP", True)

    env_params = FullDomain_TaskParams()
    env = AeroPlanaxFullDomainEnv(env_params)
    env = LogWrapper(env)
    cfg["NUM_ACTORS"] = env.num_agents
    cfg["NUM_UPDATES"] = cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"]
    cfg["MINIBATCH_SIZE"] = cfg["NUM_ACTORS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]

    if "LOADDIR" in cfg:
        network = ActorCriticRNN([31, 41, 41, 41], config=cfg)
        rng = jax.random.PRNGKey(42)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"]))
        )
        init_hstate = ScannedRNN.initialize_carry(cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])
        network_params = network.init(rng, init_hstate, init_x)
        tx = optax.adam(cfg["LR"])
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(cfg['LOADDIR'], args=ocp.args.StandardRestore(item=state))
    else:
        checkpoint = None

    def linear_schedule(count):
        frac = 1.0 - (count // (cfg["NUM_MINIBATCHES"] * cfg["UPDATE_EPOCHS"])) / cfg["NUM_UPDATES"]
        return cfg["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN([31, 41, 41, 41], config=cfg)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"]))
        )
        init_hstate = ScannedRNN.initialize_carry(cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        tx = optax.adam(cfg["LR"]) if not cfg["ANNEAL_LR"] else optax.adam(learning_rate=linear_schedule, eps=1e-5)
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        if checkpoint is not None:
            params = checkpoint["params"]
            opt_state = checkpoint["opt_state"]
            train_state = train_state.replace(params=params, opt_state=opt_state)
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = 0

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])

        # INIT Tensorboard
        if cfg.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(cfg["LOGDIR"])

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

            rng, _rng = jax.random.split(rng)
            action_throttle = pi_throttle.sample(seed=_rng)
            rng, _rng = jax.random.split(rng)
            action_elevator = pi_elevator.sample(seed=_rng)
            rng, _rng = jax.random.split(rng)
            action_aileron = pi_aileron.sample(seed=_rng)
            rng, _rng = jax.random.split(rng)
            action_rudder = pi_rudder.sample(seed=_rng)

            log_prob_throttle = pi_throttle.log_prob(action_throttle)
            log_prob_elevator = pi_elevator.log_prob(action_elevator)
            log_prob_aileron  = pi_aileron.log_prob(action_aileron)
            log_prob_rudder   = pi_rudder.log_prob(action_rudder)
            log_prob = log_prob_throttle + log_prob_elevator + log_prob_aileron + log_prob_rudder

            action = jnp.concatenate([action_throttle[:, :, np.newaxis],
                                      action_elevator[:, :, np.newaxis],
                                      action_aileron[:, :, np.newaxis],
                                      action_rudder[:, :, np.newaxis]], axis=-1)

            value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, cfg["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state, unbatchify(action, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
            )
            reward = batchify(reward, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1)
            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, info,
                valid_action=jnp.logical_not(jnp.logical_and(last_done, jnp.reshape(batchify(done, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1), last_done.shape)))
            )
            obsv = batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
            done = batchify(done, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1)

            def _reset_h(h):
                zeros = jnp.zeros_like(h)
                return jnp.where(done[:, None], jax.lax.stop_gradient(zeros), h)
            hstate = _reset_h(hstate)

            runner_state = (train_state, env_state, obsv, done, hstate, rng)
            return runner_state, transition

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = transition.done, transition.value, transition.reward
                reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
                value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                next_value = jnp.nan_to_num(next_value, nan=0.0, posinf=0.0, neginf=0.0)
                delta = reward + cfg["GAMMA"] * next_value * (1 - done) - value
                gae = delta + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae
            _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
            advantages_raw = advantages
            targets = advantages_raw + traj_batch.value
            mask = traj_batch.valid_action.astype(jnp.float32)
            count = mask.sum() + 1e-8
            adv_mean = (advantages_raw * mask).sum() / count
            adv_var  = ((advantages_raw - adv_mean) ** 2 * mask).sum() / count
            adv_std  = jnp.sqrt(adv_var + 1e-8)
            advantages = (advantages_raw - adv_mean) / (adv_std + 1e-8)
            return advantages, targets

        def _loss_and_aux(params, init_hstate, traj_batch, gae, targets, ent_coef):
            _, pi, value = network.apply(params, init_hstate.squeeze(0), (traj_batch.obs, traj_batch.done))
            mask = traj_batch.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8

            min_log_prob = jnp.log(1e-6)
            log_probs = [
                jnp.maximum(p.log_prob(traj_batch.action[:, :, idx]), min_log_prob)
                for idx, p in enumerate(pi)
            ]
            log_prob = jnp.array(log_probs).sum(axis=0)
            old_log = traj_batch.log_prob
            logratio = log_prob - old_log
            logratio = jnp.where(jnp.isfinite(logratio), logratio, 0.0)
            logratio = jnp.clip(logratio, -20.0, 20.0)
            ratio = jnp.exp(logratio)
            ratio = jnp.where(jnp.isfinite(ratio), ratio, 1.0)
            ratio = jnp.clip(ratio, 1e-6, 1e6)

            loss_actor1 = ratio * gae
            loss_actor2 = jnp.clip(ratio, 1.0 - cfg["CLIP_EPS"], 1.0 + cfg["CLIP_EPS"]) * gae
            loss_actor  = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor  = (loss_actor * mask).sum() / denom

            entropys = [p.entropy() for p in pi]
            entropy  = ((jnp.array(entropys).sum(axis=0)) * mask).sum() / denom

            value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            vf_clip = cfg["VF_CLIP_EPS"]
            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-vf_clip, vf_clip)
            err      = value - targets
            err_clip = value_pred_clipped - targets
            delta    = cfg["HUBER_DELTA"]
            def huber(x, d): ax = jnp.abs(x); quad = jnp.minimum(ax, d); lin = ax - quad; return 0.5 * quad * quad + d * lin
            vloss      = huber(err,      delta)
            vloss_clip = huber(err_clip, delta)
            vloss_comb = jnp.maximum(vloss, vloss_clip)
            value_loss = (0.5 * vloss_comb * mask).sum() / denom

            approx_kl = (((ratio - 1.0) - logratio) * mask).sum() / denom
            clip_frac = ((jnp.abs(ratio - 1.0) > cfg["CLIP_EPS"]) * mask).sum() / denom

            total_loss = loss_actor + cfg["VF_COEF"] * value_loss - ent_coef * entropy
            aux = (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)
            return total_loss, aux

        def _update_minbatch(carry, minibatch):
            train_state, ent_coef, lr_mult, do_update = carry
            init_hstate, traj_batch, advantages, targets = minibatch

            grad_fn = jax.value_and_grad(_loss_and_aux, has_aux=True)
            (total_loss, aux), grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets, ent_coef)

            grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
            gn = optax.global_norm(grads)
            scale = jnp.minimum(1.0, cfg["MAX_GRAD_NORM"] / (gn + 1e-9))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
            grads = jax.tree_util.tree_map(lambda g: g * lr_mult, grads)

            update_mask = jnp.asarray(do_update, dtype=jnp.float32)
            grads = jax.tree_util.tree_map(lambda g: g * update_mask, grads)

            train_state = train_state.apply_gradients(grads=grads)

            loss_info = {
                "total_loss": total_loss,
                "value_loss": aux[0],
                "actor_loss": aux[1],
                "entropy":    aux[2],
                "ratio":      aux[3],
                "approx_kl":  aux[4],
                "clip_frac":  aux[5],
                "grad_norm":  gn,
            }
            loss_info = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), loss_info)
            return (train_state, ent_coef, lr_mult, do_update), loss_info

        def _update_epoch(update_state, unused):
            (train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
            ent_coef,
            lr_mult,
            stop_flag,
            target_kl_eff,
            allow_ent_adapt,
            apply_lr_decay,
            allow_kl_stop) = update_state

            rng, _rng = jax.random.split(rng)

            batch = (init_hstate, traj_batch, advantages, targets)
            permutation = jax.random.permutation(_rng, cfg["NUM_ENVS"])
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], cfg["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0),
                shuffled_batch,
            )

            do_update = jnp.logical_not(stop_flag)
            (train_state, ent_coef, lr_mult, _), loss_stack = jax.lax.scan(
                _update_minbatch, (train_state, ent_coef, lr_mult, do_update), minibatches
            )

            kl_mean = jnp.mean(loss_stack["approx_kl"])
            new_stop = jnp.logical_and(
                allow_kl_stop,
                kl_mean > (target_kl_eff * cfg["KL_STOP_MULT"])
            )
            stop_flag = jnp.logical_or(stop_flag, new_stop)

            ent_lo = jnp.asarray(cfg["ENT_COEF_MIN"], dtype=jnp.float32)
            ent_hi = jnp.asarray(cfg["ENT_COEF_MAX"], dtype=jnp.float32)
            ent_adj = jnp.asarray(cfg["ENT_ADJ_RATE"], dtype=jnp.float32)

            ent_down = _clip_scalar(ent_coef / ent_adj, ent_lo, ent_hi)
            ent_up   = _clip_scalar(ent_coef * ent_adj, ent_lo, ent_hi)

            ent_new = jnp.where(kl_mean < (0.5 * target_kl_eff), ent_up, ent_coef)
            ent_new = jnp.where(kl_mean > (1.5 * target_kl_eff), ent_down, ent_new)
            ent_coef = jnp.where(allow_ent_adapt, ent_new, ent_coef)

            lr_decay = jnp.asarray(cfg["LR_DECAY"], dtype=jnp.float32)
            lr_min   = jnp.asarray(cfg["MIN_LR_MULT"], dtype=jnp.float32)
            lr_next  = jnp.maximum(lr_min, lr_mult * lr_decay)
            lr_mult  = jnp.where(apply_lr_decay, lr_next, lr_mult)

            update_state = (train_state, init_hstate, traj_batch, advantages, targets,
                            rng, ent_coef, lr_mult, stop_flag,
                            target_kl_eff, allow_ent_adapt, apply_lr_decay, allow_kl_stop)
            return update_state, loss_stack

        def _update_step(update_runner_state, _):
            (runner_state, sched_state), update_steps = update_runner_state
            ent_coef, lr_mult, stop_flag = sched_state

            initial_h = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, cfg["NUM_STEPS"])

            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[None, :], last_done[None, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            h0 = jax.lax.stop_gradient(initial_h)[None, :]

            u = update_steps
            in_warmup = u < cfg["WARMUP_UPDATES"]
            post = jnp.maximum(u - cfg["WARMUP_UPDATES"], 0)
            ramp = jnp.minimum(post / jnp.maximum(cfg["KL_RAMP_UPDATES"], 1), 1.0)

            target_kl_hi  = cfg["TARGET_KL"] * cfg["KL_START_MULT"]
            target_kl_eff = target_kl_hi - (target_kl_hi - cfg["TARGET_KL"]) * ramp

            allow_ent_adapt = jnp.array(not cfg["FREEZE_ENTROPY_DURING_WARMUP"], dtype=jnp.bool_)
            allow_ent_adapt = jnp.where(in_warmup, allow_ent_adapt, jnp.array(True, dtype=jnp.bool_))

            apply_lr_decay = jnp.array(not cfg["FREEZE_LR_DURING_WARMUP"], dtype=jnp.bool_)
            apply_lr_decay = jnp.where(in_warmup, apply_lr_decay, jnp.array(True, dtype=jnp.bool_))

            allow_kl_stop = jnp.array(not cfg["DISABLE_KL_STOP_DURING_WARMUP"], dtype=jnp.bool_)
            allow_kl_stop = jnp.where(in_warmup, allow_kl_stop, jnp.array(True, dtype=jnp.bool_))

            stop_flag = jnp.array(False, dtype=jnp.bool_)

            update_state = (train_state, h0, traj_batch, advantages, targets, rng,
                            ent_coef, lr_mult, stop_flag,
                            target_kl_eff, allow_ent_adapt, apply_lr_decay, allow_kl_stop)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, cfg["UPDATE_EPOCHS"])
            train_state = update_state[0]

            ent_coef = update_state[6]
            lr_mult  = update_state[7]
            stop_flag= update_state[8]

            loss_mean = jax.tree.map(lambda x: x.mean(), loss_info)
            ratio_0 = loss_info["ratio"].at[0, 0].get().mean()

            metric = traj_batch.info
            metric["loss"] = loss_mean
            metric["loss"]["ratio_0"] = ratio_0
            metric["ent_coef"] = ent_coef
            metric["lr_mult"]  = lr_mult
            metric["kl_mean_epoch"] = jnp.mean(loss_info["approx_kl"])
            metric["kl_stop"]  = stop_flag.astype(jnp.float32)
            metric["target_kl_eff"] = jnp.asarray(target_kl_eff, dtype=jnp.float32)

            update_steps = update_steps + 1
            metric["update_steps"] = update_steps

            if cfg.get("DEBUG"):
                def callback(m):
                    env_steps = int(m["update_steps"]) * cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
                    for k, v in m["loss"].items():
                        v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        writer.add_scalar(f"loss/{k}", float(v), env_steps)

                    ep_mask = m["returned_episode"]
                    ep_mask_sq = m["returned_episode"].squeeze()

                    val_return = float(m["returned_episode_returns"][ep_mask].mean())
                    val_length = float(m["returned_episode_lengths"][ep_mask].mean())
                    val_success = float(m["success_times"][ep_mask_sq].mean()) if "success_times" in m else 0.0

                    writer.add_scalar('eval/episodic_return', val_return, env_steps)
                    writer.add_scalar('eval/episodic_length', val_length, env_steps)
                    writer.add_scalar('eval/success_times', val_success, env_steps)

                    # Eval metrics
                    val_cl = 0.0
                    if "curriculum_level" in m:
                        val_cl = float(m["curriculum_level"][ep_mask_sq].mean())
                        writer.add_scalar('eval/curriculum_level', val_cl, env_steps)
                    val_ot = 0.0
                    if "on_target_steps" in m:
                        val_ot = float(m["on_target_steps"][ep_mask_sq].mean())
                        writer.add_scalar('eval/on_target_steps', val_ot, env_steps)
                    val_to = 0.0
                    if "timeout_count" in m:
                        val_to = float(m["timeout_count"][ep_mask_sq].mean())
                        writer.add_scalar('eval/timeout_count', val_to, env_steps)
                    if "curriculum_success_counts" in m:
                        writer.add_scalar('eval/curriculum_success_counts',
                                          float(m["curriculum_success_counts"][ep_mask_sq].mean()), env_steps)

                    # --- Individual reward components ---
                    reward_vals = {}
                    for rn in ["r_main", "r_nz", "r_qbar", "theta_deg", "delta_vt", "alt_km"]:
                        if rn in m:
                            rv = float(jnp.mean(m[rn]))
                            reward_vals[rn] = rv
                            writer.add_scalar(f'reward/{rn}', rv, env_steps)

                    # --- Schedule metrics ---
                    writer.add_scalar('sched/target_kl_eff', float(m["target_kl_eff"]), env_steps)
                    writer.add_scalar('sched/ent_coef',      float(m["ent_coef"]),      env_steps)
                    writer.add_scalar('sched/lr_mult',       float(m["lr_mult"]),       env_steps)
                    writer.add_scalar('sched/kl_stop',       float(m["kl_stop"]),       env_steps)

                    # ============================================================
                    # TERMINAL PRINT: all key metrics including rewards
                    # ============================================================
                    print(
                        f"env_step={env_steps:<10d} "
                        f"return={val_return:<7.2f} "
                        f"episode_length={val_length:<6.0f} "
                        f"success_times={val_success:.2f} "
                        f"curriculum_level={val_cl:.2f} "
                        f"on_target_steps={val_ot:.1f} "
                        f"timeout_count={val_to:.1f}"
                    )
                    print(
                        f"  reward: r_main={reward_vals.get('r_main', 0.0):.4f}"
                        f"  theta_deg={reward_vals.get('theta_deg', 0.0):.1f}"
                        f"  delta_vt={reward_vals.get('delta_vt', 0.0):.1f}"
                        f"  r_nz={reward_vals.get('r_nz', 0.0):.6f}"
                        f"  r_qbar={reward_vals.get('r_qbar', 0.0):.8f}"
                        f"  alt_km={reward_vals.get('alt_km', 0.0):.1f}"
                    )
                    print(
                        f"  loss:   actor_loss={float(m['loss']['actor_loss']):.4f}"
                        f"  value_loss={float(m['loss']['value_loss']):.4f}"
                        f"  entropy={float(m['loss']['entropy']):.3f}"
                        f"  approx_kl={float(m['loss']['approx_kl']):.5f}"
                        f"  grad_norm={float(m['loss']['grad_norm']):.3f}"
                    )

                    # Store in global history for end-of-training summary
                    _training_history["env_steps"].append(env_steps)
                    _training_history["return"].append(val_return)
                    _training_history["theta_deg"].append(reward_vals.get('theta_deg', 0.0))
                    _training_history["delta_vt"].append(reward_vals.get('delta_vt', 0.0))
                    _training_history["r_main"].append(reward_vals.get('r_main', 0.0))
                    _training_history["r_nz"].append(reward_vals.get('r_nz', 0.0))
                    _training_history["r_qbar"].append(reward_vals.get('r_qbar', 0.0))
                    _training_history["curriculum_level"].append(val_cl)
                    _training_history["on_target_steps"].append(val_ot)
                    _training_history["success_times"].append(val_success)
                    _training_history["timeout_count"].append(val_to)
                    _training_history["actor_loss"].append(float(m['loss']['actor_loss']))
                    _training_history["value_loss"].append(float(m['loss']['value_loss']))
                    _training_history["entropy"].append(float(m['loss']['entropy']))
                    _training_history["approx_kl"].append(float(m['loss']['approx_kl']))

                jax.experimental.io_callback(callback, None, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            minimal_metric = {
                "loss": {k: v.mean() for k, v in loss_mean.items()},
                "update_steps": update_steps,
            }
            return ((runner_state, (ent_coef, lr_mult, jnp.array(False, dtype=jnp.bool_))), update_steps), minimal_metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]),
            jnp.zeros((cfg["NUM_ENVS"] * cfg["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )

        ent_coef0 = jnp.array(cfg.get("ENT_COEF_INIT", cfg.get("ENT_COEF", 2e-3)), dtype=jnp.float32)
        lr_mult0  = jnp.array(1.0, dtype=jnp.float32)
        stop_flag0 = jnp.array(False)

        ((runner_state, sched_state), epoch), metric = jax.lax.scan(
            _update_step,
            ((runner_state, (ent_coef0, lr_mult0, stop_flag0)), start_epoch),
            None,
            cfg["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "sched_state": sched_state, "epoch": epoch, "metric": metric, "rng": runner_state[5]}

    return train


def print_training_summary():
    """Print comprehensive end-of-training summary for easy copy-paste debugging."""
    h = _training_history
    if not h["env_steps"]:
        print("\n[WARNING] No training history recorded.\n")
        return

    n = len(h["env_steps"])
    # Use last 10% of training for "final" metrics
    tail = max(1, n // 10)

    def _avg(lst, start=0, end=None):
        sub = lst[start:end]
        return sum(sub) / max(len(sub), 1)

    def _trend(lst):
        """Simple trend: compare first quarter to last quarter."""
        q = max(1, n // 4)
        first = _avg(lst, 0, q)
        last = _avg(lst, -q, None)
        return first, last

    print("\n" + "=" * 80)
    print("  TRAINING SUMMARY (v3)")
    print("=" * 80)
    print(f"  total_env_steps:  {h['env_steps'][-1]:,}")
    print(f"  total_updates:    {n}")
    print()

    print("  --- Final Metrics (last 10% of training) / 最终指标（训练最后10%的均值） ---")
    print(f"  return (累计回报):              {_avg(h['return'], -tail):.2f}")
    print(f"  theta_deg (姿态误差角度):       {_avg(h['theta_deg'], -tail):.2f}")
    print(f"  delta_vt (速度误差 m/s):        {_avg(h['delta_vt'], -tail):.2f}")
    print(f"  r_main (主奖励):                {_avg(h['r_main'], -tail):.4f}")
    print(f"  r_nz (过载惩罚):                {_avg(h['r_nz'], -tail):.6f}")
    print(f"  r_qbar (低动压惩罚):            {_avg(h['r_qbar'], -tail):.8f}")
    print(f"  curriculum_level (课程等级):     {_avg(h['curriculum_level'], -tail):.2f}")
    print(f"  on_target_steps (连续命中步数):  {_avg(h['on_target_steps'], -tail):.2f}")
    print(f"  success_times (成功切换次数):    {_avg(h['success_times'], -tail):.2f}")
    print(f"  timeout_count (超时次数):        {_avg(h['timeout_count'], -tail):.2f}")
    print()

    print("  --- Trends (first 25% -> last 25%) / 趋势（前25% -> 后25%） ---")
    trend_names_cn = {
        "return": "累计回报",
        "theta_deg": "姿态误差",
        "r_main": "主奖励",
        "delta_vt": "速度误差",
        "on_target_steps": "连续命中步数",
        "curriculum_level": "课程等级",
        "success_times": "成功切换次数",
    }
    for name in ["return", "theta_deg", "r_main", "delta_vt", "on_target_steps", "curriculum_level", "success_times"]:
        first, last = _trend(h[name])
        arrow = "↑" if last > first else ("↓" if last < first else "→")
        cn = trend_names_cn.get(name, "")
        print(f"  {name:<20s} ({cn}): {first:>8.3f} -> {last:>8.3f}  {arrow}")
    print()

    print("  --- Loss Metrics (final) / 损失指标（最终值） ---")
    print(f"  actor_loss (策略损失):   {_avg(h['actor_loss'], -tail):.5f}")
    print(f"  value_loss (价值损失):   {_avg(h['value_loss'], -tail):.5f}")
    print(f"  entropy (策略熵):        {_avg(h['entropy'], -tail):.4f}")
    print(f"  approx_kl (近似KL散度):  {_avg(h['approx_kl'], -tail):.6f}")
    print()

    # Quality assessment
    theta_final = _avg(h['theta_deg'], -tail)
    on_tgt_final = _avg(h['on_target_steps'], -tail)
    succ_final = _avg(h['success_times'], -tail)

    print("  --- Quality Assessment / 训练质量评估 ---")
    if theta_final < 10:
        print("  [GOOD/好] theta_deg < 10: agent tracks targets well / 智能体姿态跟踪良好")
    elif theta_final < 20:
        print("  [OK/一般] theta_deg 10-20: agent is learning, needs more training / 智能体在学习中，还需更多训练")
    elif theta_final < 30:
        print("  [WARN/较差] theta_deg 20-30: agent struggles with attitude tracking / 智能体姿态跟踪困难")
    else:
        print("  [BAD/很差] theta_deg > 30: agent cannot track attitude targets / 智能体无法跟踪姿态目标")

    if on_tgt_final > 10:
        print("  [GOOD/好] on_target_steps > 10: agent sustains on-target / 智能体能持续命中目标")
    elif on_tgt_final > 3:
        print("  [OK/一般] on_target_steps 3-10: some sustained tracking / 有一定的持续跟踪能力")
    else:
        print("  [BAD/很差] on_target_steps < 3: no sustained on-target tracking / 无法持续命中目标")

    if succ_final > 15:
        print("  [GOOD/好] success_times > 15: frequent target switches / 频繁成功切换目标")
    elif succ_final > 5:
        print("  [OK/一般] success_times 5-15: moderate success rate / 中等成功率")
    else:
        print("  [WARN/较差] success_times < 5: few successful completions / 成功完成目标次数少")

    print("=" * 80 + "\n")


# ======================== Main ========================

import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--resume-checkpoint", type=str, default=None,
                     help="Path to checkpoint directory for warm-start resume")
_args = _parser.parse_args()

str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "GROUP": "full_domain_maneuver_v3",
    "SEED": 42,
    "FOR_LOOP_EPOCHS": 1,
    "LR": 0.0003,
    "NUM_ENVS": 1000,
    "NUM_ACTORS": 1,
    "NUM_STEPS": 1000,
    "TOTAL_TIMESTEPS": 3e8,
    "FC_DIM_SIZE": 256,
    "GRU_HIDDEN_DIM": 256,
    "UPDATE_EPOCHS": 8,
    "NUM_MINIBATCHES": 5,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.005,
    "VF_COEF": 0.25,
    "MAX_GRAD_NORM": 5,
    "ACTIVATION": "relu",
    "ANNEAL_LR": True,
    "DEBUG": True,
    "WANDB_API_KEY": "4c0cc04699296bed768adea4824fbaecea35dc59",
    "OUTPUTDIR": "results/" + "full_domain_maneuver_v3" + "_" + str_date_time,
    "LOGDIR": "results/" + "full_domain_maneuver_v3" + "_" + str_date_time + "/logs",
    "SAVEDIR": "results/" + "full_domain_maneuver_v3" + "_" + str_date_time + "/checkpoints",
    # 从头训练，不加载旧 checkpoint
    # "LOADDIR": "/path/to/checkpoint"
}
# Allow dry-run override: set DRYRUN_TIMESTEPS env var to run minimal steps for validation
config["TOTAL_TIMESTEPS"] = float(os.environ.get("DRYRUN_TIMESTEPS", config["TOTAL_TIMESTEPS"]))

seed = config['SEED']
wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
wandb.init(
    project="AeroPlanax",
    config=config,
    name=config['GROUP'],
    group=config['GROUP'],
    notes='v3: scaled random targets, triple-scale reward, curriculum-dependent sustained threshold',
    reinit=True,
)

output_dir = config["OUTPUTDIR"]
Path(output_dir).mkdir(parents=True, exist_ok=True)
save_dir = config["SAVEDIR"]
Path(save_dir).mkdir(parents=True, exist_ok=True)

rng = jax.random.PRNGKey(seed)

latest_checkpoint_path = config.get("LOADDIR", None)

# If --resume-checkpoint was provided via CLI, use it as the LOADDIR
if _args.resume_checkpoint:
    config["LOADDIR"] = _args.resume_checkpoint
    latest_checkpoint_path = _args.resume_checkpoint
    print(f"[resume] Warm-starting from checkpoint: {_args.resume_checkpoint}")

for i in range(config["FOR_LOOP_EPOCHS"]):
    if latest_checkpoint_path is not None:
        config["LOADDIR"] = latest_checkpoint_path
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    rng = out['rng']

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpoint = {
        "params": out['runner_state'][0].params,
        "opt_state": out['runner_state'][0].opt_state,
        "epoch": jnp.array(out['epoch'])
    }
    latest_checkpoint_path = os.path.abspath(os.path.join(config["SAVEDIR"], f"checkpoint_epoch_{out['epoch']}"))
    ckptr.save(latest_checkpoint_path, args=ocp.args.StandardSave(checkpoint))
    ckptr.wait_until_finished()
    print(f"Checkpoint saved at epoch {out['epoch']}, iteration {i+1}/{config['FOR_LOOP_EPOCHS']}")

# ======================== End-of-training summary ========================
print_training_summary()

wandb.finish()

plt.plot(out.get("metric", {"loss":{"total_loss": jnp.array([0.0])}})["loss"].get("total_loss", jnp.array([0.0])).reshape(-1))
plt.xlabel("Update Step")
plt.ylabel("Total Loss")
plt.savefig(output_dir + '/loss_curve.png')
plt.cla()
