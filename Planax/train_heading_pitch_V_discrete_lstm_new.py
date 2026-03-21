#=========================================
# 想更像旧版：把 WARMUP_UPDATES 调大一点（比如 2000~3000）。

# 想更快进入“稳定化”：把 KL_RAMP_UPDATES 调小一点（比如 500）。

# 过早 KL 停更：把 KL_START_MULT 调大（比如 8.0）。

# 仍有后期熵过低/过高：微调 ENT_COEF_MIN/MAX 或 ENT_ADJ_RATE
#=========================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
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
from envs.aeroplanax_heading_pitch_V import AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams
import orbax.checkpoint as ocp

def _clip_scalar(x, lo, hi):
    return jnp.minimum(jnp.maximum(x, lo), hi)

class ScannedLSTM(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        lstm_state = carry  # (h, c)
        ins, resets = x
        h, c = lstm_state
        h = jnp.where(resets[:, np.newaxis], self.initialize_carry(*h.shape)[0], h)
        c = jnp.where(resets[:, np.newaxis], self.initialize_carry(*c.shape)[1], c)
        new_lstm_state, y = nn.LSTMCell(features=ins.shape[1])((h, c), ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.LSTMCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorCriticLSTM(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        embedding = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedLSTM()(hidden, rnn_in)

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

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        # critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
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

    if "LOADDIR" in cfg:
        network = ActorCriticLSTM([31, 41, 41, 41], config=cfg)
        rng = jax.random.PRNGKey(42)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"]))
        )
        init_hstate = ScannedLSTM.initialize_carry(cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])
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
        network = ActorCriticLSTM([31, 41, 41, 41], config=cfg)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"]))
        )
        init_hstate = ScannedLSTM.initialize_carry(cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])
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

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        init_hstate = ScannedLSTM.initialize_carry(cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])

        if cfg.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(cfg["LOGDIR"])

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            (h, c), pi, value = network.apply(train_state.params, hstate, ac_in)
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
            done_vec = batchify(done, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1)
            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, info,
                valid_action=jnp.logical_not(jnp.logical_and(last_done, done_vec))
            )
            obsv = batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])

            # 在 done 处重置隐藏态（断梯度）
            def _reset_h(hs):
                (h, c) = hs
                zh, zc = jnp.zeros_like(h), jnp.zeros_like(c)
                h = jnp.where(done_vec[:, None], jax.lax.stop_gradient(zh), h)
                c = jnp.where(done_vec[:, None], jax.lax.stop_gradient(zc), c)
                return (h, c)
            hstate = _reset_h((h, c))

            runner_state = (train_state, env_state, obsv, done_vec, hstate, rng)
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
            (h, c) = init_hstate
            h = h.squeeze(0); c = c.squeeze(0)
            _, pi, value = network.apply(params, (h, c), (traj_batch.obs, traj_batch.done))
            mask = traj_batch.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8

            min_log_prob = jnp.log(1e-6)
            log_probs = [jnp.maximum(p.log_prob(traj_batch.action[:, :, idx]), min_log_prob) for idx, p in enumerate(pi)]
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


        def _update_step(update_runner_state, unused):
            """
            一个 update：
            - 先 roll out NUM_STEPS，算 GAE/target
            - 再做 UPDATE_EPOCHS 次 epoch 更新
            - 期间按“暖启动 → 退火”的策略调度 target_kl/熵/学习率 & KL-stop
            """
            (runner_state, sched_state), update_steps = update_runner_state
            ent_coef, lr_mult, stop_flag = sched_state

            # === 采样一段轨迹 ===
            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, cfg["NUM_STEPS"])
            metric = traj_batch.info

            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            (h, c), _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # BPTT 截断 + 批维（LSTM 两个态）
            (h0, c0) = initial_hstate
            h0 = jax.lax.stop_gradient(h0)[None, :]
            c0 = jax.lax.stop_gradient(c0)[None, :]
            initial_hstate = (h0, c0)

            # === 计算调度（暖启动 & 退火） ===
            u = update_steps  # 已完成的 update 次数
            in_warmup = u < cfg["WARMUP_UPDATES"]

            post = jnp.maximum(u - cfg["WARMUP_UPDATES"], 0)
            ramp = jnp.minimum(post / jnp.maximum(cfg["KL_RAMP_UPDATES"], 1), 1.0)

            # KL 动态阈值：从 TARGET_KL * KL_START_MULT 线性退火到 TARGET_KL
            target_kl_hi  = cfg["TARGET_KL"] * cfg["KL_START_MULT"]
            target_kl_eff = target_kl_hi - (target_kl_hi - cfg["TARGET_KL"]) * ramp

            allow_ent_adapt = jnp.array(not cfg["FREEZE_ENTROPY_DURING_WARMUP"], dtype=jnp.bool_)
            allow_ent_adapt = jnp.where(in_warmup, allow_ent_adapt, jnp.array(True, dtype=jnp.bool_))

            apply_lr_decay = jnp.array(not cfg["FREEZE_LR_DURING_WARMUP"], dtype=jnp.bool_)
            apply_lr_decay = jnp.where(in_warmup, apply_lr_decay, jnp.array(True, dtype=jnp.bool_))

            allow_kl_stop = jnp.array(not cfg["DISABLE_KL_STOP_DURING_WARMUP"], dtype=jnp.bool_)
            allow_kl_stop = jnp.where(in_warmup, allow_kl_stop, jnp.array(True, dtype=jnp.bool_))

            # 暖启动时，不让 KL-stop 把 epoch 停掉
            stop_flag = jnp.array(False, dtype=jnp.bool_)

            update_state = (train_state, initial_hstate, traj_batch, advantages, targets,
                            rng, ent_coef, lr_mult, stop_flag,
                            target_kl_eff, allow_ent_adapt, apply_lr_decay, allow_kl_stop)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, cfg["UPDATE_EPOCHS"])
            train_state = update_state[0]

            # === 取出调度状态，做日志 ===
            ent_coef, lr_mult, stop_flag = update_state[6], update_state[7], update_state[8]

            loss_info_mean = jax.tree.map(lambda x: x.mean(), loss_info)
            ratio_0 = loss_info["ratio"].at[0, 0].get().mean()

            metric["loss"] = loss_info_mean
            metric["loss"]["ratio_0"] = ratio_0
            metric["ent_coef"] = ent_coef
            metric["lr_mult"] = lr_mult
            metric["kl_mean_epoch"] = jnp.mean(loss_info["approx_kl"])
            metric["kl_stop"] = stop_flag.astype(jnp.float32)
            metric["target_kl_eff"] = jnp.asarray(target_kl_eff, dtype=jnp.float32)  # 动态 KL 也打点

            # debug 指标（保持你原逻辑）
            mask = traj_batch.valid_action.astype(jnp.float32)
            count = mask.sum() + 1e-8
            adv_mean = (advantages * mask).sum() / count
            adv_var  = (((advantages - adv_mean) ** 2) * mask).sum() / count
            metric["debug_adv_std"] = jnp.sqrt(adv_var + 1e-8)
            metric["debug_valid_action_sum"] = traj_batch.valid_action.sum()
            metric["debug_valid_action_ratio"] = traj_batch.valid_action.sum() / traj_batch.valid_action.size
            metric["debug_valid_action_size"] = jnp.array(traj_batch.valid_action.size, dtype=jnp.float32)

            #================================================================#
            # 在 metric 中记录“奖励被裁剪”的标志
            #================================================================#
            # ===== 聚合奖励“裁剪”统计（计数与比例） =====
            # 形状：traj_batch.info["clip_xxx"] 是 (T, B) 布尔；与 valid_action 对齐
            clip_alt  = traj_batch.info.get("clipped_altitude_reward_count", jnp.zeros_like(traj_batch.valid_action)).astype(jnp.float32)
            clip_hpv  = traj_batch.info.get("clipped_heading_pitch_V_reward_count",       jnp.zeros_like(traj_batch.valid_action)).astype(jnp.float32)
            clip_any  = traj_batch.info.get("clipped_any_reward_count",       jnp.zeros_like(traj_batch.valid_action)).astype(jnp.float32)

            mask = traj_batch.valid_action.astype(jnp.float32)   # 只统计有效步
            denom = mask.sum() + 1e-8

            metric["clipped_altitude_reward_count"] = (clip_alt * mask).sum()
            metric["clipped_heading_pitch_V_reward_count"] = (clip_hpv * mask).sum()
            metric["clipped_any_reward_count"] = (clip_any * mask).sum()

            metric["clipped_altitude_reward_count_rate"] = (clip_alt * mask).sum() / denom
            metric["clipped_heading_pitch_V_reward_count_rate"] = (clip_hpv * mask).sum() / denom
            metric["clipped_any_reward_count_rate"] = (clip_any * mask).sum() / denom
            #================================================================#

            # 递增 update 计数
            update_steps = update_steps + 1
            metric["update_steps"] = update_steps

            if cfg.get("DEBUG"):
                def callback(metric):
                    env_steps = metric["update_steps"] * cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
                    for k, v in metric["loss"].items():
                        v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        writer.add_scalar('loss/{}'.format(k), v, env_steps)
                    writer.add_scalar('eval/episodic_return',  metric["returned_episode_returns"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/episodic_length',  metric["returned_episode_lengths"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/success_times',    metric["heading_turn_counts"][metric["returned_episode"].squeeze()].mean(), env_steps)

                    # 记录调度相关
                    writer.add_scalar('sched/target_kl_eff', metric["target_kl_eff"], env_steps)
                    writer.add_scalar('sched/ent_coef',      metric["ent_coef"],      env_steps)
                    writer.add_scalar('sched/lr_mult',       metric["lr_mult"],       env_steps)
                    writer.add_scalar('sched/kl_stop',       metric["kl_stop"],       env_steps)

                    # ====== 奖励裁剪统计（计数/比例），会同步到 W&B ======
                    writer.add_scalar('reward_clip/clipped_altitude_reward_count',        metric["clipped_altitude_reward_count"], env_steps)
                    writer.add_scalar('reward_clip/clipped_heading_pitch_V_reward_count', metric["clipped_heading_pitch_V_reward_count"], env_steps)
                    writer.add_scalar('reward_clip/clipped_any_reward_count',             metric["clipped_any_reward_count"], env_steps)

                    writer.add_scalar('reward_clip/clipped_altitude_reward_count_rate',        metric["clipped_altitude_reward_count_rate"], env_steps)
                    writer.add_scalar('reward_clip/clipped_heading_pitch_V_reward_count_rate', metric["clipped_heading_pitch_V_reward_count_rate"], env_steps)
                    writer.add_scalar('reward_clip/clipped_any_reward_count_rate',             metric["clipped_any_reward_count_rate"], env_steps)


                    print("EnvStep={:<10} EpisodeLength={:<4.2f} Return={:<4.2f} SuccessTimes={:.3f}".format(
                        metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                        metric["returned_episode_lengths"][metric["returned_episode"]].mean(),
                        metric["returned_episode_returns"][metric["returned_episode"]].mean(),
                        metric["heading_turn_counts"][metric["returned_episode"].squeeze()].mean(),
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
    "GROUP": "lczh_baseline(lstm_new(critic_fc2)_6e8_seed20)",
    "SEED": 20,
    "FOR_LOOP_EPOCHS": 1,
    "LR": 3e-4,
    "NUM_ENVS": 1000,
    "NUM_ACTORS": 1,
    "NUM_STEPS": 1000,
    "TOTAL_TIMESTEPS": 6e8,
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
    "OUTPUTDIR": "results/" + "heading_pitch_V_discrete_lstm" + "_" + str_date_time,
    "LOGDIR": "results/" + "heading_pitch_V_discrete_lstm" + "_" + str_date_time + "/logs",
    "SAVEDIR": "results/" + "heading_pitch_V_discrete_lstm" + "_" + str_date_time + "/checkpoints",
    # "LOADDIR": "/path/to/your/checkpoint"
}

seed = config['SEED']
wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
wandb.init(
    project="AeroPlanax",
    config=config,
    name=config['GROUP'],
    group=config['GROUP'],
    notes='multi tasks and discrete action, LSTM version',
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

wandb.finish()

plt.plot(out.get("metric", {"loss":{}})["loss"].get("total_loss", jnp.array([0.0])).reshape(-1))
plt.xlabel("Update Step")
plt.ylabel("Total Loss")
plt.savefig(output_dir + '/loss_curve.png')
plt.cla()