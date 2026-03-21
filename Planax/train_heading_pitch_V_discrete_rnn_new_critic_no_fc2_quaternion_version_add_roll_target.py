import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
# from envs.aeroplanax_heading_pitch_V import AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams
# from envs.aeroplanax_heading_pitch_V_quaternion_version import AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams
# from envs.aeroplanax_heading_pitch_V_quaternion_version_add_roll_target import AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams
import orbax.checkpoint as ocp

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

        # critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
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
    # 兼容 5v5 的稳健配置（若未提供则填默认）
    cfg = dict(config)
    cfg.setdefault("VF_CLIP_EPS", 0.20)
    cfg.setdefault("HUBER_DELTA", 1.0)
    cfg.setdefault("TARGET_KL", 0.02)
    cfg.setdefault("KL_STOP_MULT", 1.5)
    cfg.setdefault("ENT_COEF_MIN", 5e-4)
    cfg.setdefault("ENT_COEF_MAX", 2e-2)
    cfg.setdefault("ENT_ADJ_RATE", 1.05)
    cfg.setdefault("LR_DECAY", 0.999)
    cfg.setdefault("MIN_LR_MULT", 0.2)

    # === 放在 make_train(config) 里，紧邻你原来的 cfg.setdefault(...) 那一段 ===
    cfg.setdefault("WARMUP_UPDATES",     1500)  # 前期“旧版风格”训练的 update 数（不等于 env step）
    cfg.setdefault("KL_START_MULT",      5.0)   # 暖启动后 KL 阈值从 TARGET_KL*5 线性下降到 TARGET_KL
    cfg.setdefault("KL_RAMP_UPDATES",    1000)  # KL 阈值下降所需的 update 数

    # 暖启动阶段是否冻结这些稳定化机制（默认全冻结）
    cfg.setdefault("FREEZE_ENTROPY_DURING_WARMUP", True)   # 不做熵系数自适应
    cfg.setdefault("FREEZE_LR_DURING_WARMUP",      True)   # 不做学习率衰减（lr_mult 始终 1.0）
    cfg.setdefault("DISABLE_KL_STOP_DURING_WARMUP", True)  # KL 超阈不提前停（不打断 epoch）


    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)
    env = LogWrapper(env)
    cfg["NUM_ACTORS"] = env.num_agents
    cfg["NUM_UPDATES"] = cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"]
    cfg["MINIBATCH_SIZE"] = cfg["NUM_ACTORS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]

    # 可选：从 checkpoint 恢复
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

            # 在 done 处重置隐藏态（断梯度）
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
            # 前向
            _, pi, value = network.apply(params, init_hstate.squeeze(0), (traj_batch.obs, traj_batch.done))
            mask = traj_batch.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8

            # log_prob 加最小保护，ratio 数值安全
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

            # Actor loss（掩码平均）
            loss_actor1 = ratio * gae
            loss_actor2 = jnp.clip(ratio, 1.0 - cfg["CLIP_EPS"], 1.0 + cfg["CLIP_EPS"]) * gae
            loss_actor  = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor  = (loss_actor * mask).sum() / denom

            # Entropy（掩码平均）
            entropys = [p.entropy() for p in pi]
            entropy  = ((jnp.array(entropys).sum(axis=0)) * mask).sum() / denom

            # Value loss：Huber + 独立 clip + 数值安全 + 掩码平均
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

            # 清洗 + 全局梯度裁剪 + lr_mult
            grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
            gn = optax.global_norm(grads)
            scale = jnp.minimum(1.0, cfg["MAX_GRAD_NORM"] / (gn + 1e-9))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
            grads = jax.tree_util.tree_map(lambda g: g * lr_mult, grads)

            # 早停 mask
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
            """
            单个 epoch 的 PPO 更新（带“后期稳定化、前期兼容旧版”的调度骨架）：
            - 允许按标志控制：是否做 KL-stop、是否做熵系数自适应、是否做 LR 衰减
            - TARGET_KL 允许动态传入（post-warmup 线性从高阈值退火到原阈值）
            """
            (train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
            ent_coef,
            lr_mult,
            stop_flag,
            target_kl_eff,          # 动态 KL 目标
            allow_ent_adapt,        # 暖启动后才允许熵自适应
            apply_lr_decay,         # 暖启动后才做 LR 衰减
            allow_kl_stop) = update_state  # 暖启动后才启用 KL-stop

            rng, _rng = jax.random.split(rng)

            # === 打乱 & 划分小批 ===
            batch = (init_hstate, traj_batch, advantages, targets)
            permutation = jax.random.permutation(_rng, cfg["NUM_ENVS"])
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], cfg["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0),
                shuffled_batch,
            )

            # === 本 epoch 的若干 minibatch 迭代（可能被 KL-stop 提前打断） ===
            do_update = jnp.logical_not(stop_flag)
            (train_state, ent_coef, lr_mult, _), loss_stack = jax.lax.scan(
                _update_minbatch, (train_state, ent_coef, lr_mult, do_update), minibatches
            )

            # === 统计本 epoch 的 KL，决定是否触发 KL-stop ===
            kl_mean = jnp.mean(loss_stack["approx_kl"])
            new_stop = jnp.logical_and(
                allow_kl_stop,
                kl_mean > (target_kl_eff * cfg["KL_STOP_MULT"])
            )
            stop_flag = jnp.logical_or(stop_flag, new_stop)

            # === 熵系数自适应（仅在允许时启用） ===
            ent_lo = jnp.asarray(cfg["ENT_COEF_MIN"], dtype=jnp.float32)
            ent_hi = jnp.asarray(cfg["ENT_COEF_MAX"], dtype=jnp.float32)
            ent_adj = jnp.asarray(cfg["ENT_ADJ_RATE"], dtype=jnp.float32)

            ent_down = _clip_scalar(ent_coef / ent_adj, ent_lo, ent_hi)
            ent_up   = _clip_scalar(ent_coef * ent_adj, ent_lo, ent_hi)

            # 低于 0.5*target_kl → 提高熵；高于 1.5*target_kl → 降低熵
            ent_new = jnp.where(kl_mean < (0.5 * target_kl_eff), ent_up, ent_coef)
            ent_new = jnp.where(kl_mean > (1.5 * target_kl_eff), ent_down, ent_new)
            ent_coef = jnp.where(allow_ent_adapt, ent_new, ent_coef)

            # === 学习率衰减（仅在允许时启用） ===
            lr_decay = jnp.asarray(cfg["LR_DECAY"], dtype=jnp.float32)
            lr_min   = jnp.asarray(cfg["MIN_LR_MULT"], dtype=jnp.float32)
            lr_next  = jnp.maximum(lr_min, lr_mult * lr_decay)
            lr_mult  = jnp.where(apply_lr_decay, lr_next, lr_mult)

            update_state = (train_state, init_hstate, traj_batch, advantages, targets,
                            rng, ent_coef, lr_mult, stop_flag,
                            target_kl_eff, allow_ent_adapt, apply_lr_decay, allow_kl_stop)
            return update_state, loss_stack

        # ----- 一个 update：rollout -> 计算GAE -> 多个 epoch 更新（带调度） -----
        def _update_step(update_runner_state, _):
            (runner_state, sched_state), update_steps = update_runner_state
            ent_coef, lr_mult, stop_flag = sched_state

            # 采样一段轨迹
            initial_h = runner_state[-2]  # (B,H)
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, cfg["NUM_STEPS"])

            # bootstrapped value
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[None, :], last_done[None, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # BPTT 截断：把隐藏态“向后”断开梯度；同时扩一维变成 (1,B,H) 以适配 scan->minibatch 维度
            h0 = jax.lax.stop_gradient(initial_h)[None, :]

            # 调度（暖启动 + 线性退火）
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

            # 暖启动阶段不允许 KL-stop 打断
            stop_flag = jnp.array(False, dtype=jnp.bool_)

            update_state = (train_state, h0, traj_batch, advantages, targets, rng,
                            ent_coef, lr_mult, stop_flag,
                            target_kl_eff, allow_ent_adapt, apply_lr_decay, allow_kl_stop)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, cfg["UPDATE_EPOCHS"])
            train_state = update_state[0]

            # 取出调度后的 ent_coef/lr_mult/kl 止损标志
            ent_coef = update_state[6]
            lr_mult  = update_state[7]
            stop_flag= update_state[8]

            # ====== 统计 + 日志 ======
            loss_mean = jax.tree.map(lambda x: x.mean(), loss_info)
            ratio_0 = loss_info["ratio"].at[0, 0].get().mean()

            metric = traj_batch.info  # 环境返回的 episodic/计数等
            metric["loss"] = loss_mean
            metric["loss"]["ratio_0"] = ratio_0
            metric["ent_coef"] = ent_coef
            metric["lr_mult"]  = lr_mult
            metric["kl_mean_epoch"] = jnp.mean(loss_info["approx_kl"])
            metric["kl_stop"]  = stop_flag.astype(jnp.float32)
            metric["target_kl_eff"] = jnp.asarray(target_kl_eff, dtype=jnp.float32)

            # ====== 奖励裁剪统计（计数 & 比例）—— 与 LSTM 版一致的键名 ======
            clip_alt = traj_batch.info.get("clipped_altitude_reward_count",
                                           jnp.zeros_like(traj_batch.valid_action)).astype(jnp.float32)
            clip_hpv = traj_batch.info.get("clipped_heading_pitch_V_reward_count",
                                           jnp.zeros_like(traj_batch.valid_action)).astype(jnp.float32)
            clip_any = traj_batch.info.get("clipped_any_reward_count",
                                           jnp.zeros_like(traj_batch.valid_action)).astype(jnp.float32)

            mask = traj_batch.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8

            metric["clipped_altitude_reward_count"] = (clip_alt * mask).sum()
            metric["clipped_heading_pitch_V_reward_count"] = (clip_hpv * mask).sum()
            metric["clipped_any_reward_count"] = (clip_any * mask).sum()

            metric["clipped_altitude_reward_count_rate"] = (clip_alt * mask).sum() / denom
            metric["clipped_heading_pitch_V_reward_count_rate"] = (clip_hpv * mask).sum() / denom
            metric["clipped_any_reward_count_rate"] = (clip_any * mask).sum() / denom
            # ---------------------------------------------------------------

            # update step +1
            update_steps = update_steps + 1
            metric["update_steps"] = update_steps

            if cfg.get("DEBUG"):
                def callback(m):
                    # 直接用 Python int 做乘法，完全绕开 JAX 的类型限制。这样先把 JAX 数组转成 Python int，后续乘法都是 Python 原生整数运算，不会溢出也不会有警告。
                    env_steps = int(m["update_steps"]) * cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
                    # 损失/比率
                    for k, v in m["loss"].items():
                        v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        writer.add_scalar(f"loss/{k}", float(v), env_steps)
                    # 评估曲线（LogWrapper里累计的）
                    writer.add_scalar('eval/episodic_return',
                                      float(m["returned_episode_returns"][m["returned_episode"]].mean()), env_steps)
                    writer.add_scalar('eval/episodic_length',
                                      float(m["returned_episode_lengths"][m["returned_episode"]].mean()), env_steps)
                    writer.add_scalar('eval/success_times',
                                      float(m["heading_turn_counts"][m["returned_episode"].squeeze()].mean()), env_steps)
                    # 调度
                    writer.add_scalar('sched/target_kl_eff', float(m["target_kl_eff"]), env_steps)
                    writer.add_scalar('sched/ent_coef',      float(m["ent_coef"]),      env_steps)
                    writer.add_scalar('sched/lr_mult',       float(m["lr_mult"]),       env_steps)
                    writer.add_scalar('sched/kl_stop',       float(m["kl_stop"]),       env_steps)
                    # 奖励裁剪打点（计数 & 比例）
                    writer.add_scalar('reward_clip/clipped_altitude_reward_count',
                                      float(m["clipped_altitude_reward_count"]), env_steps)
                    writer.add_scalar('reward_clip/clipped_heading_pitch_V_reward_count',
                                      float(m["clipped_heading_pitch_V_reward_count"]), env_steps)
                    writer.add_scalar('reward_clip/clipped_any_reward_count',
                                      float(m["clipped_any_reward_count"]), env_steps)

                    writer.add_scalar('reward_clip/clipped_altitude_reward_count_rate',
                                      float(m["clipped_altitude_reward_count_rate"]), env_steps)
                    writer.add_scalar('reward_clip/clipped_heading_pitch_V_reward_count_rate',
                                      float(m["clipped_heading_pitch_V_reward_count_rate"]), env_steps)
                    writer.add_scalar('reward_clip/clipped_any_reward_count_rate',
                                      float(m["clipped_any_reward_count_rate"]), env_steps)

                    print("EnvStep={:<10} EpisodeLength={:<6.2f} Return={:<7.2f} SuccessTimes={:.3f}".format(
                        env_steps,
                        float(m["returned_episode_lengths"][m["returned_episode"]].mean()),
                        float(m["returned_episode_returns"][m["returned_episode"]].mean()),
                        float(m["heading_turn_counts"][m["returned_episode"].squeeze()].mean()),
                    ))
                jax.experimental.io_callback(callback, None, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return ((runner_state, (ent_coef, lr_mult, jnp.array(False, dtype=jnp.bool_))), update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]),
            jnp.zeros((cfg["NUM_ENVS"] * cfg["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )

        # 初始化调度器
        ent_coef0 = jnp.array(cfg.get("ENT_COEF_INIT", cfg.get("ENT_COEF", 1e-3)), dtype=jnp.float32)
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

str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "GROUP": "baseline_quat_add_roll_control_add_obs",
    # "GROUP": "baseline_quat_add_roll_control",
    "SEED": 42,
    "FOR_LOOP_EPOCHS": 1,
    "LR": 3e-4,
    "NUM_ENVS": 1000,
    "NUM_ACTORS": 1,
    "NUM_STEPS": 1000,
    "TOTAL_TIMESTEPS": 1e9,
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "UPDATE_EPOCHS": 16,
    "NUM_MINIBATCHES": 5,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 1e-3,
    "VF_COEF": 1,
    "MAX_GRAD_NORM": 2,
    "ACTIVATION": "relu",
    "ANNEAL_LR": False,
    "DEBUG": True,
    "WANDB_API_KEY" : "4c0cc04699296bed768adea4824fbaecea35dc59",
    "OUTPUTDIR": "results/" + "heading_pitch_V_discrete_rnn" + "_" + str_date_time,
    "LOGDIR": "results/" + "heading_pitch_V_discrete_rnn" + "_" + str_date_time + "/logs",
    "SAVEDIR": "results/" + "heading_pitch_V_discrete_rnn" + "_" + str_date_time + "/checkpoints",
    # "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-11-20-16-42/checkpoints/checkpoint_epoch_1200"
    # "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-06-16-31/checkpoints/checkpoint_epoch_1300" # -89°~89°并随机初始化pitch的baseline
    # "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-10-13-17/checkpoints/checkpoint_epoch_900" # roll、pitch、yaw都是随机初始化，并且目标也是的baseline
    # "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-11-00-38/checkpoints/checkpoint_epoch_1700" # roll、pitch、yaw都是随机初始化，并且目标也是的baseline(trained twice)
    # "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-11-20-24/checkpoints/checkpoint_epoch_2700" # roll、pitch、yaw都是随机初始化，并且目标也是的baseline(trained third)
    # "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-12-14-12/checkpoints/checkpoint_epoch_2500" # roll、pitch、yaw都是随机初始化，并且目标也是的baseline(trained third)
    # "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-13-16-15/checkpoints/checkpoint_epoch_1500" # roll、pitch、yaw都是随机初始化，并且目标也是，扩展了obs
    # "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-14-01-47/checkpoints/checkpoint_epoch_2500" # roll、pitch、yaw都是随机初始化，并且目标也是，扩展了obs(trained twice)
}

seed = config['SEED']
wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
wandb.init(
    project="AeroPlanax",
    config=config,
    name=config['GROUP'],
    group=config['GROUP'],
    notes='multi tasks and discrete action, RNN version',
    reinit=True,
)

output_dir = config["OUTPUTDIR"]
Path(output_dir).mkdir(parents=True, exist_ok=True)
save_dir = config["SAVEDIR"]
Path(save_dir).mkdir(parents=True, exist_ok=True)

rng = jax.random.PRNGKey(seed)

latest_checkpoint_path = config.get("LOADDIR", None)

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
    ################
    # GPT给的意见，暂时没管。训练脚本里打印最好用 out['epoch']，避免索引错位：
    # print(f"Checkpoint saved at epoch {out['epoch']}, iteration {i+1}/{config['FOR_LOOP_EPOCHS']}")

wandb.finish()

plt.plot(out.get("metric", {"loss":{}})["loss"].get("total_loss", jnp.array([0.0])).reshape(-1))
plt.xlabel("Update Step")
plt.ylabel("Total Loss")
plt.savefig(output_dir + '/loss_curve.png')
plt.cla()