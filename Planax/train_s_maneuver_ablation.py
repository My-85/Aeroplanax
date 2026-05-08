"""
S-Maneuver Ablation Training Script
=====================================
Train a PPO+GRU policy on the S-maneuver ablation environment.

Usage:
  # Low-fidelity run (on GPU 0):
  CUDA_VISIBLE_DEVICES=0 python train_s_maneuver_ablation.py --fidelity_mode low

  # High-fidelity run (on GPU 1):
  CUDA_VISIBLE_DEVICES=1 python train_s_maneuver_ablation.py --fidelity_mode high

To change the heading-switch period from the default 10 s:
  --s_switch_steps 25   →  5 s   (25 * 10/50 = 5 s)
  --s_switch_steps 100  →  20 s  (100 * 10/50 = 20 s)

Checkpoints are saved under:
  results/ablation_<lofi|hifi>_<timestamp>/checkpoints/
"""

import os
import argparse
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.experimental
import numpy as np
import flax.linen as nn
import optax
import wandb
import tensorboardX
import orbax.checkpoint as ocp
import distrax

from typing import Sequence, NamedTuple, Tuple, Optional, Union, Any, Dict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import functools

from envs.wrappers import LogWrapper
from envs.aeroplanax_s_maneuver_ablation import (
    AeroPlanaxSManeuverAblationEnv,
    SManeuverTaskParams,
)


# =============================================================================
# Network (identical to baseline: GRU + discrete action heads)
# =============================================================================

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0, out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
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

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2), bias_init=constant(0.0),
        )(nn_fc2)
        actor_mean = activation(actor_mean)

        pi_throttle = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_aileron  = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_rudder   = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2), bias_init=constant(0.0),
        )(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done:         jnp.ndarray
    action:       jnp.ndarray
    value:        jnp.ndarray
    reward:       jnp.ndarray
    log_prob:     jnp.ndarray
    obs:          jnp.ndarray
    info:         jnp.ndarray
    valid_action: jnp.ndarray


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _clip_scalar(x, lo, hi):
    return jnp.minimum(jnp.maximum(x, lo), hi)


# =============================================================================
# Training function
# =============================================================================

def make_train(config):
    cfg = dict(config)
    cfg.setdefault("VF_CLIP_EPS",    0.20)
    cfg.setdefault("HUBER_DELTA",    1.0)
    cfg.setdefault("TARGET_KL",      0.02)
    cfg.setdefault("KL_STOP_MULT",   1.5)
    cfg.setdefault("ENT_COEF_MIN",   5e-4)
    cfg.setdefault("ENT_COEF_MAX",   2e-2)
    cfg.setdefault("ENT_ADJ_RATE",   1.05)
    cfg.setdefault("LR_DECAY",       0.999)
    cfg.setdefault("MIN_LR_MULT",    0.2)
    cfg.setdefault("WARMUP_UPDATES", 1500)
    cfg.setdefault("KL_START_MULT",  5.0)
    cfg.setdefault("KL_RAMP_UPDATES",1000)
    cfg.setdefault("FREEZE_ENTROPY_DURING_WARMUP",  True)
    cfg.setdefault("FREEZE_LR_DURING_WARMUP",       True)
    cfg.setdefault("DISABLE_KL_STOP_DURING_WARMUP", True)

    env_params = SManeuverTaskParams(
        fidelity_mode  = cfg["FIDELITY_MODE"],
        s_switch_steps = cfg.get("S_SWITCH_STEPS", 50),
    )
    env = AeroPlanaxSManeuverAblationEnv(env_params)
    env = LogWrapper(env)

    cfg["NUM_ACTORS"]    = env.num_agents
    cfg["NUM_UPDATES"]   = int(cfg["TOTAL_TIMESTEPS"]) // cfg["NUM_STEPS"] // cfg["NUM_ENVS"]
    cfg["MINIBATCH_SIZE"]= cfg["NUM_ACTORS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]

    # Optional checkpoint resume
    checkpoint_init = None
    if "LOADDIR" in cfg and cfg["LOADDIR"]:
        network = ActorCriticRNN([31, 41, 41, 41], config=cfg)
        rng = jax.random.PRNGKey(42)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"],
                        *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"])),
        )
        init_h = ScannedRNN.initialize_carry(cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])
        params = network.init(rng, init_h, init_x)
        tx = optax.adam(cfg["LR"])
        ts = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        state_template = {"params": ts.params, "opt_state": ts.opt_state, "epoch": jnp.array(0)}
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint_init = ckptr.restore(cfg["LOADDIR"], args=ocp.args.StandardRestore(item=state_template))

    def linear_schedule(count):
        frac = 1.0 - (count // (cfg["NUM_MINIBATCHES"] * cfg["UPDATE_EPOCHS"])) / cfg["NUM_UPDATES"]
        return cfg["LR"] * frac

    def train(rng):
        network = ActorCriticRNN([31, 41, 41, 41], config=cfg)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"],
                        *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"])),
        )
        init_h = ScannedRNN.initialize_carry(cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])
        net_params = network.init(_rng, init_h, init_x)
        tx = (optax.adam(cfg["LR"]) if not cfg["ANNEAL_LR"]
              else optax.adam(learning_rate=linear_schedule, eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=net_params, tx=tx)

        if checkpoint_init is not None:
            train_state = train_state.replace(
                params=checkpoint_init["params"],
                opt_state=checkpoint_init["opt_state"],
            )
            start_epoch = checkpoint_init["epoch"]
        else:
            start_epoch = 0

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_h = ScannedRNN.initialize_carry(cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])

        if cfg.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(cfg["LOGDIR"])

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            pi_t, pi_e, pi_a, pi_r = pi

            rng, r1, r2, r3, r4 = jax.random.split(rng, 5)
            a_t = pi_t.sample(seed=r1); a_e = pi_e.sample(seed=r2)
            a_a = pi_a.sample(seed=r3); a_r = pi_r.sample(seed=r4)

            log_prob = (pi_t.log_prob(a_t) + pi_e.log_prob(a_e)
                        + pi_a.log_prob(a_a) + pi_r.log_prob(a_r))

            action = jnp.concatenate([
                a_t[:, :, np.newaxis], a_e[:, :, np.newaxis],
                a_a[:, :, np.newaxis], a_r[:, :, np.newaxis],
            ], axis=-1)
            value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, cfg["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state,
                unbatchify(action, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]),
            )
            reward = batchify(reward, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1)
            done_b = batchify(done, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1)

            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, info,
                valid_action=jnp.logical_not(
                    jnp.logical_and(last_done, jnp.reshape(done_b, last_done.shape))
                ),
            )

            obsv = batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
            def _reset_h(h):
                return jnp.where(done_b[:, None], jax.lax.stop_gradient(jnp.zeros_like(h)), h)
            hstate = _reset_h(hstate)

            return (train_state, env_state, obsv, done_b, hstate, rng), transition

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_nv, t):
                gae, nv = gae_and_nv
                done, value, reward = t.done, t.value, t.reward
                reward    = jnp.nan_to_num(reward,    nan=0.0, posinf=0.0, neginf=0.0)
                value     = jnp.nan_to_num(value,     nan=0.0, posinf=0.0, neginf=0.0)
                nv        = jnp.nan_to_num(nv,        nan=0.0, posinf=0.0, neginf=0.0)
                delta = reward + cfg["GAMMA"] * nv * (1 - done) - value
                gae   = delta + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages_raw = jax.lax.scan(
                _get_advantages, (jnp.zeros_like(last_val), last_val),
                traj_batch, reverse=True, unroll=16,
            )
            targets = advantages_raw + traj_batch.value
            mask    = traj_batch.valid_action.astype(jnp.float32)
            count   = mask.sum() + 1e-8
            adv_mean = (advantages_raw * mask).sum() / count
            adv_var  = ((advantages_raw - adv_mean)**2 * mask).sum() / count
            advantages = (advantages_raw - adv_mean) / (jnp.sqrt(adv_var + 1e-8))
            return advantages, targets

        def _loss_and_aux(params, init_hstate, traj_batch, gae, targets, ent_coef):
            _, pi, value = network.apply(
                params, init_hstate.squeeze(0), (traj_batch.obs, traj_batch.done)
            )
            mask  = traj_batch.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8

            min_lp = jnp.log(1e-6)
            log_probs = [
                jnp.maximum(p.log_prob(traj_batch.action[:, :, i]), min_lp)
                for i, p in enumerate(pi)
            ]
            log_prob  = jnp.array(log_probs).sum(axis=0)
            logratio  = jnp.clip(log_prob - traj_batch.log_prob, -20.0, 20.0)
            logratio  = jnp.where(jnp.isfinite(logratio), logratio, 0.0)
            ratio     = jnp.clip(jnp.exp(logratio), 1e-6, 1e6)
            ratio     = jnp.where(jnp.isfinite(ratio), ratio, 1.0)

            loss_actor = -jnp.minimum(ratio * gae,
                                      jnp.clip(ratio, 1 - cfg["CLIP_EPS"], 1 + cfg["CLIP_EPS"]) * gae)
            loss_actor = (loss_actor * mask).sum() / denom
            entropy    = ((jnp.array([p.entropy() for p in pi]).sum(axis=0)) * mask).sum() / denom

            value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            vf_c  = cfg["VF_CLIP_EPS"]
            vp_c  = traj_batch.value + (value - traj_batch.value).clip(-vf_c, vf_c)
            def huber(x, d): ax = jnp.abs(x); q = jnp.minimum(ax, d); l = ax - q; return 0.5*q*q + d*l
            d = cfg["HUBER_DELTA"]
            value_loss = (0.5 * jnp.maximum(huber(value - targets, d), huber(vp_c - targets, d)) * mask).sum() / denom

            approx_kl  = (((ratio - 1.0) - logratio) * mask).sum() / denom
            clip_frac  = ((jnp.abs(ratio - 1.0) > cfg["CLIP_EPS"]) * mask).sum() / denom

            total_loss = loss_actor + cfg["VF_COEF"] * value_loss - ent_coef * entropy
            return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

        def _update_minbatch(carry, mb):
            train_state, ent_coef, lr_mult, do_update = carry
            init_h, traj, adv, tgt = mb
            grad_fn = jax.value_and_grad(_loss_and_aux, has_aux=True)
            (total_loss, aux), grads = grad_fn(train_state.params, init_h, traj, adv, tgt, ent_coef)
            grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
            gn    = optax.global_norm(grads)
            scale = jnp.minimum(1.0, cfg["MAX_GRAD_NORM"] / (gn + 1e-9))
            grads = jax.tree_util.tree_map(lambda g: g * scale * lr_mult * do_update.astype(jnp.float32), grads)
            train_state = train_state.apply_gradients(grads=grads)
            loss_info = {
                "total_loss": total_loss, "value_loss": aux[0], "actor_loss": aux[1],
                "entropy": aux[2], "ratio": aux[3], "approx_kl": aux[4],
                "clip_frac": aux[5], "grad_norm": gn,
            }
            loss_info = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), loss_info)
            return (train_state, ent_coef, lr_mult, do_update), loss_info

        def _update_epoch(update_state, unused):
            (train_state, init_h, traj, adv, tgt, rng,
             ent_coef, lr_mult, stop_flag,
             target_kl_eff, allow_ent, apply_lr, allow_kl) = update_state

            rng, _rng = jax.random.split(rng)
            perm = jax.random.permutation(_rng, cfg["NUM_ENVS"])
            batch = (init_h, traj, adv, tgt)
            shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=1), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(x, [x.shape[0], cfg["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0
                ),
                shuffled,
            )
            do_update = jnp.logical_not(stop_flag)
            (train_state, ent_coef, lr_mult, _), loss_stack = jax.lax.scan(
                _update_minbatch, (train_state, ent_coef, lr_mult, do_update), minibatches
            )

            kl_mean  = jnp.mean(loss_stack["approx_kl"])
            stop_flag= jnp.logical_or(stop_flag,
                           jnp.logical_and(allow_kl, kl_mean > target_kl_eff * cfg["KL_STOP_MULT"]))

            elo, ehi, erate = (jnp.asarray(cfg["ENT_COEF_MIN"], jnp.float32),
                               jnp.asarray(cfg["ENT_COEF_MAX"], jnp.float32),
                               jnp.asarray(cfg["ENT_ADJ_RATE"], jnp.float32))
            ent_new  = jnp.where(kl_mean < 0.5 * target_kl_eff,
                                 _clip_scalar(ent_coef * erate, elo, ehi), ent_coef)
            ent_new  = jnp.where(kl_mean > 1.5 * target_kl_eff,
                                 _clip_scalar(ent_coef / erate, elo, ehi), ent_new)
            ent_coef = jnp.where(allow_ent, ent_new, ent_coef)

            lr_next  = jnp.maximum(jnp.asarray(cfg["MIN_LR_MULT"], jnp.float32),
                                   lr_mult * jnp.asarray(cfg["LR_DECAY"], jnp.float32))
            lr_mult  = jnp.where(apply_lr, lr_next, lr_mult)

            return (train_state, init_h, traj, adv, tgt, rng,
                    ent_coef, lr_mult, stop_flag,
                    target_kl_eff, allow_ent, apply_lr, allow_kl), loss_stack

        def _update_step(update_runner_state, _):
            (runner_state, sched_state), update_steps = update_runner_state
            ent_coef, lr_mult, stop_flag = sched_state

            init_h  = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, cfg["NUM_STEPS"])

            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in   = (last_obs[None, :], last_done[None, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val= last_val.squeeze(0)

            advantages, targets = _calculate_gae(traj_batch, last_val)
            h0 = jax.lax.stop_gradient(init_h)[None, :]

            u         = update_steps
            in_warmup = u < cfg["WARMUP_UPDATES"]
            post      = jnp.maximum(u - cfg["WARMUP_UPDATES"], 0)
            ramp      = jnp.minimum(post / jnp.maximum(cfg["KL_RAMP_UPDATES"], 1), 1.0)
            kl_hi     = cfg["TARGET_KL"] * cfg["KL_START_MULT"]
            target_kl_eff = kl_hi - (kl_hi - cfg["TARGET_KL"]) * ramp

            def _bool(flag, freeze): return jnp.where(in_warmup, jnp.array(not freeze, jnp.bool_), jnp.array(True, jnp.bool_))
            allow_ent = _bool(in_warmup, cfg["FREEZE_ENTROPY_DURING_WARMUP"])
            apply_lr  = _bool(in_warmup, cfg["FREEZE_LR_DURING_WARMUP"])
            allow_kl  = _bool(in_warmup, cfg["DISABLE_KL_STOP_DURING_WARMUP"])

            stop_flag = jnp.array(False, jnp.bool_)
            update_state = (train_state, h0, traj_batch, advantages, targets, rng,
                            ent_coef, lr_mult, stop_flag,
                            target_kl_eff, allow_ent, apply_lr, allow_kl)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, cfg["UPDATE_EPOCHS"])

            train_state = update_state[0]
            ent_coef    = update_state[6]
            lr_mult     = update_state[7]

            loss_mean = jax.tree.map(lambda x: x.mean(), loss_info)
            metric    = traj_batch.info
            metric["loss"]           = loss_mean
            metric["ent_coef"]       = ent_coef
            metric["lr_mult"]        = lr_mult
            metric["kl_mean_epoch"]  = jnp.mean(loss_info["approx_kl"])
            metric["target_kl_eff"]  = jnp.asarray(target_kl_eff, jnp.float32)
            update_steps = update_steps + 1
            metric["update_steps"]   = update_steps

            if cfg.get("DEBUG"):
                def callback(m):
                    u      = int(m["update_steps"])
                    steps  = u * int(cfg["NUM_ENVS"]) * int(cfg["NUM_STEPS"])
                    for k, v in m["loss"].items():
                        v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        writer.add_scalar(f"loss/{k}", float(v), steps)
                    writer.add_scalar("eval/episodic_return",
                        float(m["returned_episode_returns"][m["returned_episode"]].mean()), steps)
                    writer.add_scalar("eval/episodic_length",
                        float(m["returned_episode_lengths"][m["returned_episode"]].mean()), steps)
                    writer.add_scalar("eval/heading_switches",
                        float(m["heading_turn_counts"][m["returned_episode"].squeeze()].mean()), steps)
                    writer.add_scalar("sched/ent_coef",      float(m["ent_coef"]),      steps)
                    writer.add_scalar("sched/lr_mult",       float(m["lr_mult"]),       steps)
                    writer.add_scalar("sched/target_kl_eff", float(m["target_kl_eff"]), steps)
                    print(
                        f"[{cfg['FIDELITY_MODE']}] "
                        f"Step={steps:<10} "
                        f"EpLen={float(m['returned_episode_lengths'][m['returned_episode']].mean()):<6.1f} "
                        f"Ret={float(m['returned_episode_returns'][m['returned_episode']].mean()):<7.3f} "
                        f"Switches={float(m['heading_turn_counts'][m['returned_episode'].squeeze()].mean()):.1f}"
                    )
                jax.experimental.io_callback(callback, None, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return ((runner_state, (ent_coef, lr_mult, jnp.array(False, jnp.bool_))), update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]),
            jnp.zeros((cfg["NUM_ENVS"] * cfg["NUM_ACTORS"],), dtype=bool),
            init_h,
            _rng,
        )
        ent_coef0  = jnp.array(cfg.get("ENT_COEF_INIT", cfg.get("ENT_COEF", 1e-3)), jnp.float32)
        lr_mult0   = jnp.array(1.0, jnp.float32)
        stop_flag0 = jnp.array(False)

        ((runner_state, sched_state), epoch), metric = jax.lax.scan(
            _update_step,
            ((runner_state, (ent_coef0, lr_mult0, stop_flag0)), start_epoch),
            None,
            cfg["NUM_UPDATES"],
        )
        return {
            "runner_state": runner_state,
            "sched_state":  sched_state,
            "epoch":        epoch,
            "metric":       metric,
            "rng":          runner_state[5],
        }

    return train


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="S-Maneuver Ablation Training")
    p.add_argument("--fidelity_mode", type=str, default="high",
                   choices=["high", "low"],
                   help="Aerodynamic fidelity: 'high' (F-16 hifi) or 'low' (linear lofi)")
    p.add_argument("--s_switch_steps", type=int, default=50,
                   help=(
                       "Heading-switch period in agent-interaction steps. "
                       "Real time = s_switch_steps * agent_interaction_steps / sim_freq. "
                       "Default 50 → 10 s.  E.g. 25→5s, 100→20s."
                   ))
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--total_steps", type=int,   default=int(2e9))
    p.add_argument("--num_envs",    type=int,   default=1000)
    p.add_argument("--num_steps",   type=int,   default=1000)
    p.add_argument("--loaddir",     type=str,   default="",
                   help="Resume from existing checkpoint directory")
    p.add_argument("--no_wandb",    action="store_true")
    p.add_argument("--gpu",         type=int,   default=-1,
                   help="CUDA device index (ignored if CUDA_VISIBLE_DEVICES is set)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # GPU selection (if not already set via CUDA_VISIBLE_DEVICES)
    if args.gpu >= 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("XLA_PYTHON_MEM_FRACTION", "0.95")

    tag  = "lofi" if args.fidelity_mode == "low" else "hifi"
    ts   = datetime.now().strftime("%Y-%m-%d-%H-%M")
    name = f"ablation_{tag}_{ts}"

    config = {
        "GROUP":           f"s_maneuver_ablation_{tag}",
        "FIDELITY_MODE":   args.fidelity_mode,
        "S_SWITCH_STEPS":  args.s_switch_steps,
        "SEED":            args.seed,
        "FOR_LOOP_EPOCHS": 1,
        "LR":              3e-4,
        "NUM_ENVS":        args.num_envs,
        "NUM_STEPS":       args.num_steps,
        "TOTAL_TIMESTEPS": args.total_steps,
        "FC_DIM_SIZE":     128,
        "GRU_HIDDEN_DIM":  128,
        "UPDATE_EPOCHS":   16,
        "NUM_MINIBATCHES": 5,
        "GAMMA":           0.99,
        "GAE_LAMBDA":      0.95,
        "CLIP_EPS":        0.2,
        "ENT_COEF":        1e-3,
        "VF_COEF":         1,
        "MAX_GRAD_NORM":   2,
        "ACTIVATION":      "relu",
        "ANNEAL_LR":       False,
        "DEBUG":           True,
        "OUTPUTDIR":       f"results/{name}",
        "LOGDIR":          f"results/{name}/logs",
        "SAVEDIR":         f"results/{name}/checkpoints",
    }
    if args.loaddir:
        config["LOADDIR"] = args.loaddir

    Path(config["OUTPUTDIR"]).mkdir(parents=True, exist_ok=True)
    Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)

    if not args.no_wandb:
        os.environ.setdefault("WANDB_API_KEY", "4c0cc04699296bed768adea4824fbaecea35dc59")
        wandb.init(
            project="AeroPlanax",
            config=config,
            name=name,
            group=config["GROUP"],
            notes=f"S-maneuver ablation [{tag}], switch_steps={args.s_switch_steps}",
            reinit=True,
            sync_tensorboard=True,
        )

    rng = jax.random.PRNGKey(args.seed)

    latest_ckpt = config.get("LOADDIR", None)
    for i in range(config["FOR_LOOP_EPOCHS"]):
        if latest_ckpt:
            config["LOADDIR"] = latest_ckpt
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)
        rng = out["rng"]

        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = {
            "params":    out["runner_state"][0].params,
            "opt_state": out["runner_state"][0].opt_state,
            "epoch":     jnp.array(out["epoch"]),
        }
        latest_ckpt = os.path.abspath(
            os.path.join(config["SAVEDIR"], f"checkpoint_epoch_{out['epoch']}")
        )
        ckptr.save(latest_ckpt, args=ocp.args.StandardSave(checkpoint))
        ckptr.wait_until_finished()
        print(f"[{tag}] Checkpoint saved: {latest_ckpt}  (epoch {out['epoch']})")

    if not args.no_wandb:
        wandb.finish()

    print(f"\n[{tag}] Training complete.")
    print(f"  Checkpoint dir : {config['SAVEDIR']}")
    print(f"  Switch period  : {args.s_switch_steps} steps = "
          f"{args.s_switch_steps * 10 / 50:.1f} s")
    print(f"\nFor zero-shot evaluation, run:")
    print(f"  python eval_s_maneuver_sim2real.py "
          f"  --lofi_ckpt <lofi_checkpoint_dir> "
          f"  --hifi_ckpt <hifi_checkpoint_dir>")
