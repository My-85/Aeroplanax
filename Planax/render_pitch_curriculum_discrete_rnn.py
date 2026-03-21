import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import functools
from typing import Sequence, Dict, Any
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import orbax.checkpoint as ocp

from envs.wrappers import LogWrapper
from envs.aeroplanax_pitch_curriculum import (
    AeroPlanaxPitchCurriculumEnv,
    Pitch_Curriculum_TaskParams,
)

# ====== 配置 ======
LOGDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/tracks/"
BASELINE_LOADDIR = "/absolute/path/to/your/checkpoint_dir"  # 必填：训练保存的 checkpoint 目录
SEED = 42
NUM_STEPS = 200000

# 与训练脚本保持一致的 RNN 与头部
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
        rnn_state = jnp.where(
            resets[:, None],
            self.initialize_carry(*rnn_state.shape),
            rnn_state
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
        obs, dones = x  # obs:(T,B,D), dones:(T,B)

        # 前端 MLP
        emb = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs)
        emb = activation(emb)

        # GRU
        hidden, emb = ScannedRNN()(hidden, (emb, dones))

        # trunk
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(emb)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        # 预测头（t->t+1），与训练保持一致
        pred_h = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(nn_fc2)
        pred_h = activation(pred_h)
        pred = nn.Dense(3, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(pred_h)

        # stop-grad 的前视特征
        pred_sg = jax.lax.stop_gradient(pred)
        obs_aug = jnp.concatenate([obs, pred_sg], axis=-1)          # (T,B,22+3)
        obs_aug = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs_aug)
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


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def main():
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)

    # 环境（直接用默认参数即可；如需严格复现训练超参，可手动传 Pitch_Curriculum_TaskParams(...)）
    env_params = Pitch_Curriculum_TaskParams(
        # 可与训练时保持一致的基础项……
        pitch_only_mode=True,
        # 渲染固定模式
        render_no_switch=True,
        # render_fixed_bin_idx=3,         # 第 4 桶：30°~40°
        # 或者直接指定角度（度），二选一：
        render_fixed_pitch_deg=35.0,
    )
    env = LogWrapper(AeroPlanaxPitchCurriculumEnv(env_params))

    # 模型
    cfg = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
    network = ActorCriticRNN([31, 41, 41, 41], config=cfg)

    rng = jax.random.PRNGKey(SEED)
    B = 1 * env.num_agents
    obs_dim = env.observation_space(env.agents[0], env_params).shape[0]
    init_x = (jnp.zeros((1, B, obs_dim)), jnp.zeros((1, B)))
    init_h = ScannedRNN.initialize_carry(B, cfg["GRU_HIDDEN_DIM"])
    params = network.init(rng, init_h, init_x)

    # 为了用标准恢复接口，构造一个 TrainState
    tx = optax.adam(3e-4)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    state_item = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(BASELINE_LOADDIR, args=ocp.args.StandardRestore(item=state_item))
    params = ckpt["params"]

    # reset
    rng, key = jax.random.split(rng)
    obs, log_state = env.reset(key)

    # 渲染首帧
    env.render(log_state.env_state, env_params, {'__all__': False}, LOGDIR)

    # 推理循环（贪心动作，单环境）
    last_done = jnp.zeros((B,), dtype=bool)
    h = ScannedRNN.initialize_carry(B, cfg["GRU_HIDDEN_DIM"])
    for t in range(NUM_STEPS):
        ac_in = (batchify(obs, env.agents, 1, env.num_agents)[None, :], last_done[None, :])
        h, pi, value, _ = network.apply(params, h, ac_in)

        # 贪心推理（mode）
        a0 = pi[0].mode(); a1 = pi[1].mode(); a2 = pi[2].mode(); a3 = pi[3].mode()
        action = jnp.concatenate([a0[:, :, None], a1[:, :, None], a2[:, :, None], a3[:, :, None]], axis=-1).squeeze(0)
        action = unbatchify(action, env.agents, 1, env.num_agents)

        rng, key = jax.random.split(rng)
        obs, log_state, reward, done, info = env.step(key, log_state, action)

        # 写入 acmi
        env.render(log_state.env_state, env_params, {'__all__': bool(done["__all__"])}, LOGDIR)

        last_done = jnp.array(done["__all__"])[None].repeat(B, axis=0).astype(bool).squeeze()
        if bool(done["__all__"]):
            break

    print("Tacview ACMI 保存路径:", env.filename)

if __name__ == "__main__":
    main()