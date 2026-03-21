# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.95'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 1) 禁止一次性预分配整块显存；2) 使用 cudaMallocAsync 降低碎片；3) 限制占用 80%
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']    = 'cuda_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'

# 可去掉旧变量（避免冲突）
os.environ.pop('XLA_PYTHON_MEM_FRACTION', None)

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
from envs.aeroplanax_pitch_curriculum import AeroPlanaxPitchCurriculumEnv as AeroPlanaxPitchCurriculumEnv, Pitch_Curriculum_TaskParams as Pitch_Curriculum_TaskParams
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
        obs, dones = x  # obs:(T,B,D), dones:(T,B)

        # 前端 MLP
        emb = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        emb = activation(emb)

        # GRU 时序
        hidden, emb = ScannedRNN()(hidden, (emb, dones))

        # trunk
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        # 轻量预测头（t -> t+1），预测: vt_norm(4), pitch(rad)（由7/8反算）, az(16)
        pred_h = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(nn_fc2)
        pred_h = activation(pred_h)
        pred = nn.Dense(3, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(pred_h)

        # stop-grad 的前视特征拼回
        pred_sg = jax.lax.stop_gradient(pred)
        obs_aug = jnp.concatenate([obs, pred_sg], axis=-1)          # (T,B,22+3)
        obs_aug = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs_aug)
        obs_aug = nn.relu(obs_aug) if self.config["ACTIVATION"] == "relu" else nn.tanh(obs_aug)

        aug = jnp.concatenate([nn_fc2, obs_aug, pred_sg], axis=-1)
        aug = nn.LayerNorm()(aug)

        # actor
        actor_h = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2.0), bias_init=constant(0.0))(aug)
        actor_h = activation(actor_h)
        logits = [nn.Dense(n, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_h) for n in self.action_dim]
        pi = tuple(distrax.Categorical(logits=l) for l in logits)

        # critic
        critic_h = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2.0), bias_init=constant(0.0))(aug)
        critic_h = activation(critic_h)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic_h)
        value = jnp.squeeze(value, axis=-1)

        return hidden, pi, value, pred

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
    # 兼容 5v5 的稳健配置 ( 若未提供则填默认 ) 
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

    # === 放在 make_train(config) 里, 紧邻你原来的 cfg.setdefault(...) 那一段 ===
    cfg.setdefault("WARMUP_UPDATES",     1500)  # 前期“旧版风格”训练的 update 数 ( 不等于 env step ) 
    cfg.setdefault("KL_START_MULT",      5.0)   # 暖启动后 KL 阈值从 TARGET_KL*5 线性下降到 TARGET_KL
    cfg.setdefault("KL_RAMP_UPDATES",    1000)  # KL 阈值下降所需的 update 数

    # 暖启动阶段是否冻结这些稳定化机制 ( 默认全冻结 ) 
    cfg.setdefault("FREEZE_ENTROPY_DURING_WARMUP", True)   # 不做熵系数自适应
    cfg.setdefault("FREEZE_LR_DURING_WARMUP",      True)   # 不做学习率衰减 ( lr_mult 始终 1.0 ) 
    cfg.setdefault("DISABLE_KL_STOP_DURING_WARMUP", True)  # KL 超阈不提前停 ( 不打断 epoch ) 

    # 预测头损失权重
    cfg.setdefault("PRED_LOSS_COEF", 0.05)  # 预测损失权重；0.05~0.1 均可

    env_params = Pitch_Curriculum_TaskParams(
        pitch_only_mode=True,
        pitch_bin_deg=10,
        curriculum_target_rate=0.75,   # 0.80 -> 0.75
        curriculum_min_trials=60,
        target_max_jump_deg=20.0,
        use_vt_in_reward=False,
        only_negative=True,  # 只训负俯仰

        # 你要的“更快飞机”
        max_vt=1000.0, min_vt=120.0,

        # 判门/归一化参考保持稳定
        qbar_ref_vt=360.0,
        energy_ref_vt=360.0,
        energy_ref_alt=20000.0,

        # 升桶门（可先用较稳的值）
        promote_min_qbar=0.40, # 0.45->0.40
        promote_min_energy=0.28, # 降低安全门阈值 0.30->0.28

    )
    env = AeroPlanaxPitchCurriculumEnv(env_params)
    env = LogWrapper(env)
    cfg["NUM_ACTORS"] = env.num_agents
    cfg["NUM_UPDATES"] = cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"]
    cfg["MINIBATCH_SIZE"] = cfg["NUM_ACTORS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]

    # 可选: 从 checkpoint 恢复
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
            hstate, pi, value, _ = network.apply(train_state.params, hstate, ac_in)
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

            # 在 done 处重置隐藏态 ( 断梯度 ) 
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
            _, pi, value, pred = network.apply(params, init_hstate.squeeze(0), (traj_batch.obs, traj_batch.done))
            mask = traj_batch.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8

            # log_prob ( 四头 ) 
            min_log_prob = jnp.log(1e-6)
            log_probs = [
                jnp.maximum(p.log_prob(traj_batch.action[:, :, idx]), min_log_prob)
                for idx, p in enumerate(pi)
            ]
            log_prob = jnp.array(log_probs).sum(axis=0)
            old_log = traj_batch.log_prob
            logratio = jnp.clip(jnp.where(jnp.isfinite(log_prob - old_log), log_prob - old_log, 0.0), -20.0, 20.0)
            ratio = jnp.clip(jnp.exp(logratio), 1e-6, 1e6)

            # Actor loss
            loss_actor1 = ratio * gae
            loss_actor2 = jnp.clip(ratio, 1.0 - cfg["CLIP_EPS"], 1.0 + cfg["CLIP_EPS"]) * gae
            loss_actor  = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor  = (loss_actor * mask).sum() / denom

            # Entropy
            entropys = jnp.sum(jnp.stack([p.entropy() for p in pi], axis=-1), axis=-1)
            entropy  = (entropys * mask).sum() / denom

            # Value loss
            value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            vf_clip = cfg["VF_CLIP_EPS"]
            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-vf_clip, vf_clip)
            err      = value - targets
            err_clip = value_pred_clipped - targets
            delta    = cfg["HUBER_DELTA"]
            def huber(x, d):
                ax = jnp.abs(x); quad = jnp.minimum(ax, d); lin = ax - quad; return 0.5 * quad * quad + d * lin
            vloss      = huber(err,      delta)
            vloss_clip = huber(err_clip, delta)
            vloss_comb = jnp.maximum(vloss, vloss_clip)
            value_loss = (0.5 * vloss_comb * mask).sum() / denom

            # 预测辅助损失 ( t -> t+1 ) : 从 22 维 obs 恢复目标
            obs = traj_batch.obs  # (T,B,D=22)
            obs_tp1 = jnp.concatenate([obs[1:], obs[-1:]], axis=0)
            done_t = traj_batch.done.astype(jnp.float32)
            valid_tp1 = mask * (1.0 - done_t)

            vt_norm_tp1   = obs_tp1[:, :, 4]
            pitch_sin_tp1 = obs_tp1[:, :, 7]
            pitch_cos_tp1 = obs_tp1[:, :, 8]
            pitch_tp1     = jnp.arctan2(pitch_sin_tp1, pitch_cos_tp1)
            az_tp1        = obs_tp1[:, :, 16]

            target_pred = jnp.stack([vt_norm_tp1, pitch_tp1, az_tp1], axis=-1)
            pred_loss = ((pred - target_pred) ** 2 * valid_tp1[:, :, None]).sum() / (valid_tp1.sum() + 1e-8)

            # Pred MAE 指标
            denom_valid = valid_tp1.sum() + 1e-8
            abs_err = jnp.abs(pred - target_pred)
            pred_mae_all = (abs_err * valid_tp1[:, :, None]).sum() / (denom_valid * 3.0)
            mae_vt_norm   = (jnp.abs(pred[..., 0] - target_pred[..., 0]) * valid_tp1).sum() / denom_valid
            mae_pitch_deg = (jnp.abs(pred[..., 1] - target_pred[..., 1]) * valid_tp1).sum() / denom_valid * (180.0/jnp.pi)
            mae_az        = (jnp.abs(pred[..., 2] - target_pred[..., 2]) * valid_tp1).sum() / denom_valid

            approx_kl = (((ratio - 1.0) - logratio) * mask).sum() / denom
            clip_frac = ((jnp.abs(ratio - 1.0) > cfg["CLIP_EPS"]) * mask).sum() / denom

            total_loss = loss_actor + cfg["VF_COEF"] * value_loss - ent_coef * entropy + cfg["PRED_LOSS_COEF"] * pred_loss
            aux = (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac,
                   pred_loss, pred_mae_all, mae_vt_norm, mae_pitch_deg, mae_az)
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

                # 预测头
                "pred_loss":          aux[6],
                "pred_mae_all":       aux[7],
                "pred_mae_vt_norm":   aux[8],
                "pred_mae_pitch_deg": aux[9],
                "pred_mae_az":        aux[10],
            }
            loss_info = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), loss_info)
            return (train_state, ent_coef, lr_mult, do_update), loss_info

        def _update_epoch(update_state, unused):
            """
            单个 epoch 的 PPO 更新 ( 带“后期稳定化、前期兼容旧版”的调度骨架 ) : 
            - 允许按标志控制: 是否做 KL-stop、是否做熵系数自适应、是否做 LR 衰减
            - TARGET_KL 允许动态传入 ( post-warmup 线性从高阈值退火到原阈值 ) 
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

            # === 本 epoch 的若干 minibatch 迭代 ( 可能被 KL-stop 提前打断 )  ===
            do_update = jnp.logical_not(stop_flag)
            (train_state, ent_coef, lr_mult, _), loss_stack = jax.lax.scan(
                _update_minbatch, (train_state, ent_coef, lr_mult, do_update), minibatches
            )

            # === 统计本 epoch 的 KL, 决定是否触发 KL-stop ===
            kl_mean = jnp.mean(loss_stack["approx_kl"])
            new_stop = jnp.logical_and(
                allow_kl_stop,
                kl_mean > (target_kl_eff * cfg["KL_STOP_MULT"])
            )
            stop_flag = jnp.logical_or(stop_flag, new_stop)

            # === 熵系数自适应 ( 仅在允许时启用 )  ===
            ent_lo = jnp.asarray(cfg["ENT_COEF_MIN"], dtype=jnp.float32)
            ent_hi = jnp.asarray(cfg["ENT_COEF_MAX"], dtype=jnp.float32)
            ent_adj = jnp.asarray(cfg["ENT_ADJ_RATE"], dtype=jnp.float32)

            ent_down = _clip_scalar(ent_coef / ent_adj, ent_lo, ent_hi)
            ent_up   = _clip_scalar(ent_coef * ent_adj, ent_lo, ent_hi)

            # 低于 0.5*target_kl → 提高熵；高于 1.5*target_kl → 降低熵
            ent_new = jnp.where(kl_mean < (0.5 * target_kl_eff), ent_up, ent_coef)
            ent_new = jnp.where(kl_mean > (1.5 * target_kl_eff), ent_down, ent_new)
            ent_coef = jnp.where(allow_ent_adapt, ent_new, ent_coef)

            # === 学习率衰减 ( 仅在允许时启用 )  ===
            lr_decay = jnp.asarray(cfg["LR_DECAY"], dtype=jnp.float32)
            lr_min   = jnp.asarray(cfg["MIN_LR_MULT"], dtype=jnp.float32)
            lr_next  = jnp.maximum(lr_min, lr_mult * lr_decay)
            lr_mult  = jnp.where(apply_lr_decay, lr_next, lr_mult)

            update_state = (train_state, init_hstate, traj_batch, advantages, targets,
                            rng, ent_coef, lr_mult, stop_flag,
                            target_kl_eff, allow_ent_adapt, apply_lr_decay, allow_kl_stop)
            return update_state, loss_stack

        # ----- 一个 update: rollout -> 计算GAE -> 多个 epoch 更新 ( 带调度 )  -----
        def _update_step(update_runner_state, _):
            (runner_state, sched_state), update_steps = update_runner_state
            ent_coef, lr_mult, stop_flag = sched_state

            # 采样一段轨迹
            initial_h = runner_state[-2]  # (B,H)
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, cfg["NUM_STEPS"])

            # bootstrapped value
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[None, :], last_done[None, :])
            _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # BPTT 截断: 把隐藏态“向后”断开梯度；同时扩一维变成 (1,B,H) 以适配 scan->minibatch 维度
            h0 = jax.lax.stop_gradient(initial_h)[None, :]

            # 调度 ( 暖启动 + 线性退火 ) 
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

            # 预测指标
            metric["pred_loss"]          = jnp.mean(loss_info["pred_loss"])
            metric["pred_mae_all"]       = jnp.mean(loss_info["pred_mae_all"])
            metric["pred_mae_vt_norm"]   = jnp.mean(loss_info["pred_mae_vt_norm"])
            metric["pred_mae_pitch_deg"] = jnp.mean(loss_info["pred_mae_pitch_deg"])
            metric["pred_mae_az"]        = jnp.mean(loss_info["pred_mae_az"])

            #==================================================================================================#
            # === 训练质量指标 ( 按整个 rollout 聚合 ) ===
            """
            指标含义与计算方式
            track/err_deg_mean
            含义: 俯仰跟踪的平均绝对误差 ( 度 ) , 越小越好。
            计算: pitch_err_deg = abs(obs[:, :, 1]) x 180/π, 其中 obs[:, :, 1] 是标准化的俯仰差 wrap_PI(pitch - target_pitch)。对 (T,B) 全部样本取均值。
            
            track/err_deg_p95
            含义: 俯仰误差的 95 分位 ( 度 ) , 反映“最差 5%”的跟踪质量, 越小越好。
            计算: 将 pitch_err_deg 展平成一维, 排序后取第 floor(0.95x(N-1)) 个。
            
            track/in_band_2deg
            含义: 误差≤2°的样本占比, 越高越好。
            计算: mean((pitch_err_deg ≤ 2.0).astype(float)), 在 (T,B) 上平均。
            
            track/in_band_5deg
            含义: 误差≤5°的样本占比, 越高越好。
            计算: 同上, 阈值改为 5.0。
            
            switch/success_rate
            含义: 一次“目标切换”事件中, 由“成功达成目标”触发的比例, 越高越好。
            计算: env info 提供 did_switch/switch_success/switch_timeout ( 逐步 ) 。在一个 rollout 内: 
            success_rate = sum(switch_success) / (sum(did_switch)+1e-8)
            
            switch/timeout_rate
            含义: 一次“目标切换”事件中, 由“驻留超时”触发的比例, 越低越好。
            计算: timeout_rate = sum(switch_timeout) / (sum(did_switch)+1e-8)
            
            safety/low_qbar_rate
            含义: 低动压状态占比 ( qbar_norm<0.30 ) , 越低越好。
            计算: mean((qbar_norm<0.30).astype(float)), 在 (T,B) 上平均, qbar_norm=obs[:, :, 18]。
            
            safety/qbar_norm_p05
            含义: 动压归一的 5 分位, 越高越安全 ( 远离失速边缘 ) 。
            计算: 展平 qbar_norm 排序后取第 floor(0.05x(N-1)) 个。
            
            curr/bin_lo_deg、curr/bin_hi_deg
            含义: 当前课程桶的上下界 ( 度 ) , 用最后一个时间步的均值展示课程难度窗口。
            计算: mean(info["pitch_bin_lo_deg"][-1])、mean(info["pitch_bin_hi_deg"][-1]) ( 跨并行环境/智能体 ) 。
            
            curr/rate_cur
            含义: 当前桶的成功率 ( 累计成功/累计尝试 ) , 衡量“当前难度”掌握情况, 越高越好。
            计算: mean(info["pitch_bin_rate_cur"][-1])。
            
            curr/bin_idx、curr/max_bin
            含义: 当前桶索引、历史最高解锁桶索引 ( 0..8 ) , 反映课程进度, 越高代表学得更难的角度。
            计算: mean(info["pitch_current_bin"][-1])、mean(info["curriculum_max_bin"][-1])。
            """
            # 1) 俯仰跟踪误差 ( 来自 obs[:, :, 1] 的归一化俯仰差, 单位 rad ) 
            pitch_err_deg = jnp.abs(traj_batch.obs[:, :, 1]) * (180.0 / jnp.pi)  # (T,B)
            flat_err = pitch_err_deg.reshape(-1)
            flat_err_sorted = jnp.sort(flat_err)
            n = flat_err_sorted.shape[0]
            p95_idx = jnp.maximum(0, jnp.minimum(n - 1, jnp.int32(jnp.floor(0.95 * (n - 1)))))
            metric["track/err_deg_mean"] = jnp.mean(flat_err)
            metric["track/err_deg_p95"]  = flat_err_sorted[p95_idx]
            metric["track/in_band_2deg"] = jnp.mean((pitch_err_deg <= 2.0).astype(jnp.float32))
            metric["track/in_band_5deg"] = jnp.mean((pitch_err_deg <= 5.0).astype(jnp.float32))

            # 2) 切换成功/超时比例 ( 环境提供的 info ) 
            did  = traj_batch.info["did_switch"]       # (T,B)
            succ = traj_batch.info["switch_success"]
            tout = traj_batch.info["switch_timeout"]
            denom_sw = jnp.sum(did) + 1e-8
            metric["switch/success_rate"] = jnp.sum(succ) / denom_sw
            metric["switch/timeout_rate"] = jnp.sum(tout) / denom_sw

            # 3) 安全性: 低动压占比、动压分位
            qbar_norm = traj_batch.obs[:, :, 18]
            metric["safety/low_qbar_rate"] = jnp.mean((qbar_norm < 0.30).astype(jnp.float32))
            q_flat = qbar_norm.reshape(-1)
            q_sorted = jnp.sort(q_flat)
            q05_idx = jnp.maximum(0, jnp.minimum(q_sorted.shape[0] - 1, jnp.int32(jnp.floor(0.05 * (q_sorted.shape[0] - 1)))))
            metric["safety/qbar_norm_p05"] = q_sorted[q05_idx]

            # 4) 课程进度 ( 沿用 rollouts 的“最后一步均值”或你已有字段 ) 
            last = -1
            metric["curr/bin_lo_deg"]   = jnp.mean(traj_batch.info["pitch_bin_lo_deg"][last])
            metric["curr/bin_hi_deg"]   = jnp.mean(traj_batch.info["pitch_bin_hi_deg"][last])
            metric["curr/rate_cur"]     = jnp.mean(traj_batch.info["pitch_bin_rate_cur"][last])
            metric["curr/bin_idx"]      = jnp.mean(traj_batch.info["pitch_current_bin"][last])
            metric["curr/max_bin"]      = jnp.mean(traj_batch.info["curriculum_max_bin"][last])

            # 在 metric 计算中添加 t_cur (大约第627行后)
            metric["curr/t_cur"] = jnp.mean(traj_batch.info["curr_t_cur"][-1])  # 假设在 env 添加 info["curr_t_cur"]

            # NEW: 奖励分量均值（跨 T 和 B 取整体均值）
            metric["reward/reward_heading_pitch_mean"]     = jnp.mean(traj_batch.info["reward_heading_pitch"])
            metric["reward/reward_altitude_mean"]          = jnp.mean(traj_batch.info["reward_altitude"])
            metric["reward/reward_low_qbar_penalty_mean"]  = jnp.mean(traj_batch.info["reward_low_qbar_penalty"])
            metric["reward/reward_nz_soft_penalty_mean"]   = jnp.mean(traj_batch.info["reward_nz_soft_penalty"])
            metric["reward/reward_energy_track_mean"]      = jnp.mean(traj_batch.info["reward_energy_track"])

            # 新指标：安全门通过率 + 海拔均值
            metric["curr/promote_qbar_gate_pass"]   = jnp.mean((traj_batch.info["qbar_norm_cur"]   >= env_params.promote_min_qbar).astype(jnp.float32))
            metric["curr/promote_energy_gate_pass"] = jnp.mean((traj_batch.info["energy_norm_cur"] >= env_params.promote_min_energy).astype(jnp.float32))
            alt_km = traj_batch.obs[:, :, 3] * 5.0
            metric["alt/mean_km"] = jnp.mean(alt_km)

            # 新指标：当前桶晋升统计
            metric["curr/promote_at_switch_mean"] = jnp.mean(traj_batch.info["promote_at_switch"])
            metric["curr/bin_trials_cur_mean"]    = jnp.mean(traj_batch.info["bin_trials_cur"])
            metric["curr/bin_success_cur_mean"]   = jnp.mean(traj_batch.info["bin_success_cur"])

            # 新指标：晋升成功率
            promote_hits = jnp.sum(traj_batch.info["promote_at_switch"])
            switch_cnt   = jnp.sum(traj_batch.info["did_switch"])
            metric["curr/promote_at_switch_rate"] = promote_hits / (switch_cnt + 1e-8)

            # 新指标：当前桶奖励均值
            metric["reward/reward_heading_pitch_cur_bin_mean"] = jnp.mean(traj_batch.info["reward_heading_pitch_cur_bin"])

            """
            新监控用途：
            curr/promote_on_success_rate：成功瞬间的真实升桶命中率（越高越好）。
            curr/qbar_gate_on_success_rate、curr/energy_gate_on_success_rate：成功瞬间门通过是否稳定。
            curr/ready_cur_bin_rate：并行环境中“已满足 rate_cur/t_cur”的占比，反映“是否真的都准备好升”。
            """
            # 以“成功”为条件的晋升成功率
            promote_hits_succ = jnp.sum(traj_batch.info["promote_on_success"])
            succ_cnt          = jnp.sum(traj_batch.info["switch_success"])
            metric["curr/promote_on_success_rate"] = promote_hits_succ / (succ_cnt + 1e-8)

            # 成功步上的门通过率
            metric["curr/qbar_gate_on_success_rate"]   = jnp.sum(traj_batch.info["qbar_gate_on_success"])   / (succ_cnt + 1e-8)
            metric["curr/energy_gate_on_success_rate"] = jnp.sum(traj_batch.info["energy_gate_on_success"]) / (succ_cnt + 1e-8)

            # 就绪度（达成 rate_cur/t_cur 的占比）
            metric["curr/ready_cur_bin_rate"] = jnp.mean(traj_batch.info["ready_cur_bin"])

            #==================================================================================================#


            # update step +1
            update_steps = update_steps + 1
            metric["update_steps"] = update_steps

            if cfg.get("DEBUG"):
                def callback(m):
                    env_steps = int(m["update_steps"] * cfg["NUM_ENVS"] * cfg["NUM_STEPS"])
                    # 损失/比率
                    for k, v in m["loss"].items():
                        v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        writer.add_scalar(f"loss/{k}", float(v), env_steps)
                    # 评估曲线 ( LogWrapper里累计的 ) 
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

                    # 训练质量: 跟踪/切换/安全/课程
                    writer.add_scalar('track/err_deg_mean',  float(m["track/err_deg_mean"]),  env_steps)
                    writer.add_scalar('track/err_deg_p95',   float(m["track/err_deg_p95"]),   env_steps)
                    writer.add_scalar('track/in_band_2deg',  float(m["track/in_band_2deg"]),  env_steps)
                    writer.add_scalar('track/in_band_5deg',  float(m["track/in_band_5deg"]),  env_steps)

                    writer.add_scalar('switch/success_rate', float(m["switch/success_rate"]), env_steps)
                    writer.add_scalar('switch/timeout_rate', float(m["switch/timeout_rate"]), env_steps)

                    writer.add_scalar('safety/low_qbar_rate', float(m["safety/low_qbar_rate"]), env_steps)
                    writer.add_scalar('safety/qbar_norm_p05', float(m["safety/qbar_norm_p05"]), env_steps)

                    writer.add_scalar('curr/bin_lo_deg',   float(m["curr/bin_lo_deg"]),   env_steps)
                    writer.add_scalar('curr/bin_hi_deg',   float(m["curr/bin_hi_deg"]),   env_steps)
                    writer.add_scalar('curr/rate_cur',     float(m["curr/rate_cur"]),     env_steps)
                    writer.add_scalar('curr/bin_idx',      float(m["curr/bin_idx"]),      env_steps)
                    writer.add_scalar('curr/max_bin',      float(m["curr/max_bin"]),      env_steps)

                    writer.add_scalar('curr/t_cur', float(m["curr/t_cur"]), env_steps)

                    # NEW: 奖励分量均值
                    writer.add_scalar('reward/reward_heading_pitch_mean',    float(m["reward/reward_heading_pitch_mean"]),   env_steps)
                    writer.add_scalar('reward/reward_altitude_mean',         float(m["reward/reward_altitude_mean"]),        env_steps)
                    writer.add_scalar('reward/reward_low_qbar_penalty_mean', float(m["reward/reward_low_qbar_penalty_mean"]),env_steps)
                    writer.add_scalar('reward/reward_nz_soft_penalty_mean',  float(m["reward/reward_nz_soft_penalty_mean"]), env_steps)
                    writer.add_scalar('reward/reward_energy_track_mean',     float(m["reward/reward_energy_track_mean"]),    env_steps)

                    # 新指标：当前桶奖励均值
                    writer.add_scalar('reward/reward_heading_pitch_cur_bin_mean', float(m["reward/reward_heading_pitch_cur_bin_mean"]), env_steps)

                    # 新指标：安全门通过率 + 海拔均值
                    writer.add_scalar('curr/promote_qbar_gate_pass',   float(m["curr/promote_qbar_gate_pass"]),   env_steps)
                    writer.add_scalar('curr/promote_energy_gate_pass', float(m["curr/promote_energy_gate_pass"]), env_steps)
                    writer.add_scalar('alt/mean_km',                   float(m["alt/mean_km"]),                   env_steps)

                    # 新指标：当前桶晋升统计
                    writer.add_scalar('curr/promote_at_switch_mean', float(m["curr/promote_at_switch_mean"]), env_steps)
                    writer.add_scalar('curr/bin_trials_cur_mean',    float(m["curr/bin_trials_cur_mean"]),    env_steps)
                    writer.add_scalar('curr/bin_success_cur_mean',   float(m["curr/bin_success_cur_mean"]),   env_steps)
                    writer.add_scalar('curr/promote_at_switch_rate', float(m["curr/promote_at_switch_rate"]), env_steps)

                    # 新增
                    writer.add_scalar('curr/promote_on_success_rate',     float(m["curr/promote_on_success_rate"]),     env_steps)
                    writer.add_scalar('curr/qbar_gate_on_success_rate',   float(m["curr/qbar_gate_on_success_rate"]),   env_steps)
                    writer.add_scalar('curr/energy_gate_on_success_rate', float(m["curr/energy_gate_on_success_rate"]), env_steps)
                    writer.add_scalar('curr/ready_cur_bin_rate',          float(m["curr/ready_cur_bin_rate"]),          env_steps)

                    # print("EnvStep={:<10} EpisodeLength={:<6.2f} Return={:<7.2f} heading_turn_counts={:.3f}".format(
                    #     env_steps,
                    #     float(m["returned_episode_lengths"][m["returned_episode"]].mean()),
                    #     float(m["returned_episode_returns"][m["returned_episode"]].mean()),
                    #     float(m["heading_turn_counts"][m["returned_episode"].squeeze()].mean()),
                    # ))

                    #==================================================================================================#
                    print(
                        (
                            "EnvStep={:<10} EpLen={:<6.2f} Return={:<7.2f} heading_turn_counts(success | force_switch)={:.3f}\n"
                            "Track: err_deg_mean={:.2f} err_deg_p95={:.2f} in_band_2deg={:.1%} in_band_5deg={:.1%}\n"
                            "Switch: success_rate={:.1%} timeout_rate={:.1%}\n"
                            "Safety: low_qbar_rate={:.1%} qbar_p05={:.2f}\n"
                            "Curr: bin[{:>2.0f},{:>2.0f}] cur_success_rate={:.1%} cur_bin_idx={:.0f} history_max_bin_idx={:.0f} t_cur={:.0f}\n"
                            "Gates: qbar_pass={:.1%} energy_pass={:.1%} | Alt mean={:.2f}km\n"
                            "Promote: at_switch={:.1%} trials={:.0f} success={:.0f}"
                        ).format(
                            env_steps,
                            float(m["returned_episode_lengths"][m["returned_episode"]].mean()),
                            float(m["returned_episode_returns"][m["returned_episode"]].mean()),
                            float(m["heading_turn_counts"][m["returned_episode"].squeeze()].mean()),
                            float(m["track/err_deg_mean"]),
                            float(m["track/err_deg_p95"]),
                            float(m["track/in_band_2deg"]),
                            float(m["track/in_band_5deg"]),
                            float(m["switch/success_rate"]),
                            float(m["switch/timeout_rate"]),
                            float(m["safety/low_qbar_rate"]),
                            float(m["safety/qbar_norm_p05"]),
                            float(m["curr/bin_lo_deg"]),
                            float(m["curr/bin_hi_deg"]),
                            float(m["curr/rate_cur"]),
                            float(m["curr/bin_idx"]),
                            float(m["curr/max_bin"]),
                            float(m["curr/t_cur"]),
                            float(m["curr/promote_qbar_gate_pass"]),
                            float(m["curr/promote_energy_gate_pass"]),
                            float(m["alt/mean_km"]),
                            float(m["curr/promote_at_switch_mean"]),
                            float(m["curr/bin_trials_cur_mean"]),
                            float(m["curr/bin_success_cur_mean"]),
                        ),
                        flush=True,
                    )
                    #==================================================================================================#

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
    "GROUP": "pitch_curriculum_down_only",
    "SEED": 42,
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
    "OUTPUTDIR": "results/" + "pitch_only_curriculum_task" + "_" + str_date_time,
    "LOGDIR": "results/" + "pitch_only_curriculum_task" + "_" + str_date_time + "/logs",
    "SAVEDIR": "results/" + "pitch_only_curriculum_task" + "_" + str_date_time + "/checkpoints",
    "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_add_obs_and_pred_2025-10-10-23-57/checkpoints/checkpoint_epoch_600" # 带22维obs和pred头的rnn baseline
    # "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/pitch_only_curriculum_task_2025-10-14-20-46/checkpoints/checkpoint_epoch_1200" # 俯仰单桶推进

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
    # ckptr.save(latest_checkpoint_path, args=ocp.args.StandardSave(checkpoint))
    ckptr.save(latest_checkpoint_path, args=ocp.args.StandardSave(checkpoint), force=True)  # <- 加 force=True
    ckptr.wait_until_finished()
    print(f"Checkpoint saved at epoch {out['epoch']}, iteration {i+1}/{config['FOR_LOOP_EPOCHS']}")
    ################
    # GPT给的意见, 暂时没管。训练脚本里打印最好用 out['epoch'], 避免索引错位: 
    # print(f"Checkpoint saved at epoch {out['epoch']}, iteration {i+1}/{config['FOR_LOOP_EPOCHS']}")

wandb.finish()

plt.plot(out.get("metric", {"loss":{}})["loss"].get("total_loss", jnp.array([0.0])).reshape(-1))
plt.xlabel("Update Step")
plt.ylabel("Total Loss")
plt.savefig(output_dir + '/loss_curve.png')
plt.cla()