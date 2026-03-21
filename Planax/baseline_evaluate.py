# -*- coding: utf-8 -*-
# baseline_evaluate.py
# 用法：直接运行 python baseline_evaluate.py
# 改参数：见“用户可改参数”一节
#
# ========================= 设计说明（中文） =========================
# 本脚本在“策略评估”的基础上，新增：
# 1) 导出 CSV：summary_metrics.csv（汇总指标）
# 2) 画总览图：
#    - bar_pmax.png       ：各动作头的 pmax_mean 对比条形图
#    - bar_entropy.png    ：各动作头的 entropy_ratio=H/ln(A) 对比条形图（越低越好）
#    - hist_dwell_steps.png：各动作头的驻留长度（步）直方图对比
#    - hist_dwell_seconds.png：各动作头的驻留时间（秒）直方图对比
# 3) 画 4 张“动作时序图”（来自一个代表性的 episode）：
#    - action_series_throttle.png / elevator.png / aileron.png / rudder.png
# 4) 导出 Excel（默认 xlsx 格式）：
#    - full_series      ：代表 episode 全步长的四个动作值（每行一时刻）
#    - changes_throttle / changes_elevator / changes_aileron / changes_rudder
#      仅包含“有变更”的时刻（包含步号、变更前与变更后）
#
# “代表性 episode”的选择策略：
#   - 默认：选择“第一个跑满 STEPS_LIMIT 的 episode”（更稳定、可对齐步轴）。
#   - 若无满步回合：自动回退到“第一个 episode”。
#
# ===================================================================
# # ------------------------------
# # 指标总览（中文解释）：
# # ------------------------------
# # 1) Return(sum): 单次评估回合（episode）内的总回报，越大越好。报告里给出“均值±标准差”，用于衡量策略在固定长度下的平均表现与稳定性。
# #    - 如果 ONLY_FULL_EPISODES=True，则只统计“恰好跑满 STEPS_LIMIT 步”的回合（避免早终止拉低均值，但注意存在生存者偏差）。
# #
# # 2) pmax_mean_per_head（每个动作头的最大概率均值）：
# #    - 定义：对每一步的动作分布 p(a|s)，取 max_a p(a)，再在时间与回合上取均值。
# #    - 范围：[1/A, 1]，A 为该动作头的离散档位数（如 31/41）。1/A 接近“完全不确定（均匀）”，1 表示“完全确定（尖分布）”。
# #    - 解读：越大代表策略越“自信/确定”。但过高也可能导致“僵硬/易振荡”，需结合后面的熵、变更率一起看。
# #
# # 3) pmax_ge_0.9_per_head（pmax>=0.9 的时间占比）：
# #    - 定义：统计有多少时间步的 max_a p(a)≥0.9 的比例。
# #    - 范围：[0, 1]。越高代表“极度自信”的时刻越多。
# #
# # 4) margin_mean_per_head（top1-top2 概率间隔）：
# #    - 定义：每步取 p_top1 - p_top2，再做时间与回合均值。
# #    - 范围：[0, 1]。越大代表“第一名”和“第二名”差距越大（更有把握）。
# #
# # 5) entropy_mean_per_head（熵，使用自然对数）：
# #    - 定义：对每步的分布计算熵 H(p)=-∑ p log p，再做时间与回合均值。
# #    - 范围：[0, ln(A)]。越小代表分布越集中（越确定）。报告会同时给出 entropy_ratio=H/ln(A) 与 confidence≈1-H/ln(A)。
# #
# # 6) mode_change_rate_per_head（相邻控制步 argmax 是否变化的比例）：
# #    - 定义：统计相邻两步 argmax(a) 是否不同，不同记 1，相同记 0；在时间与回合上取均值。
# #    - 范围：[0, 1]。越低越“平滑”（抖动少）。但注意这是“是否变了”的统计，不考虑“变了多少”。
# #    - 重要：实际控制频率 = sim_freq / agent_interaction_steps 的倒数（dt_control）。
# #            我们同时给出 flips/sec = mode_change_rate / dt_control（单位 Hz），用于直观对比飞控可接受的翻档频率。
# #
# # 7) dwell（驻留时间统计）：
# #    - 定义：同一档位连续保持的步数（或秒）。给出均值、Median、中位数、10%分位、90%分位。
# #    - 步单位转秒单位用 dt_control（例如 sim=50Hz, 交互=10步 → 0.2s/次控制）。
# #    - 解读：p10 若为 1 步（0.2s），说明有大量“一闪即变”的微抖。p90 和 median 越大越平滑。
# #
# # 8) step_change_norm_mean（步幅大小的归一化均值）：
# #    - 定义：|a_t - a_{t-1}| / (A-1) 的均值，范围 [0, 1]。越小越平滑。
# #    - 与 mode_change_rate 区别：不仅看“是否变了”，还看“跨了多少档位”。
# #
# # 整体建议：对 elevator/aileron（俯仰/副翼）经常会看到 mode_change 及 flips/sec 较大，
# # 可考虑：动作平滑正则（KL-to-previous / TV penalty），动作重复(dwell约束)，离散档重映射（中段密、端段疏），
# # 或在训练中对这两个头单独调节熵系数/抖动惩罚。
#########################################################################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # 可选

from typing import Sequence, Dict, Any, List, Tuple
import functools
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

# 新增：可视化 & 表格导出依赖
import matplotlib
matplotlib.use("Agg")  # 无显示服务器也能保存图片
import matplotlib.pyplot as plt
import pandas as pd

from envs.wrappers import LogWrapper
from envs.aeroplanax_heading_pitch_V import (
    AeroPlanaxHeading_Pitch_V_Env,
    Heading_Pitch_V_TaskParams,
)

# ======================
# 用户可改参数（无需命令行）
# ======================

# 1.PPO+RNN(no_fc2_no_layer_norm)
# CKPT_PATH     = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline(no_fc2_no_layer_norm)/checkpoints/checkpoint_epoch_1000"

# 2.PPO+RNN(add_fc2_and_layer_norm)
# CKPT_PATH     = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline_new(add_fc2_and_layer_norm)/checkpoints/checkpoint_epoch_500"

# 3.PPO+LSTM(add_fc2_and_layer_norm)
CKPT_PATH     = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_lstm_2025-08-28-14-51/checkpoints/checkpoint_epoch_600"

NUM_EPISODES  = 15             # 评估回合数（越多方差越小）
STEPS_LIMIT   = 1000         # 每回合最大步数（固定时长评估）
SEED          = 42            # 随机种子
GREEDY_ACTION = True          # True=贪心（更稳定）；False=采样（更随机）
NUM_ENVS      = 1             # 评估建议用 1；并行评估>1时代码也兼容
ONLY_FULL_EPISODES = True     # 只统计“跑满 STEPS_LIMIT”的回合（避免早终止干扰）
HEAD_LABELS   = ["throttle", "elevator", "aileron", "rudder"]
ACTION_DIMS   = [31, 41, 41, 41]  # 四个头离散档数，必须与训练一致

# 输出目录（图片/CSV/Excel 都会保存在此处）
# 1.PPO+RNN(no_fc2_no_layer_norm)
# OUTPUT_DIR    = "./baseline_evaluate_outputs/PPO+RNN(origin)"
# EXCEL_FILENAME= "action_series(origin).xlsx"     # Excel 文件名（含全时序 + 变更时刻）
# CSV_SUMMARY   = "summary_metrics(origin).csv"    # 汇总指标 CSV

# 2.PPO+RNN(add_fc2_and_layer_norm)
# OUTPUT_DIR    = "./baseline_evaluate_outputs/PPO+RNN(add_fc2_and_layer_norm)"
# EXCEL_FILENAME= "action_series(add_fc2_and_layer_norm).xlsx"     # Excel 文件名（含全时序 + 变更时刻）
# CSV_SUMMARY   = "summary_metrics(add_fc2_and_layer_norm).csv"    # 汇总指标 CSV

# 3.PPO+LSTM(add_fc2_and_layer_norm)
OUTPUT_DIR    = "./baseline_evaluate_outputs/PPO+LSTM(add_fc2_and_layer_norm)"
EXCEL_FILENAME= "action_series(add_fc2_and_layer_norm).xlsx"     # Excel 文件名（含全时序 + 变更时刻）
CSV_SUMMARY   = "summary_metrics(add_fc2_and_layer_norm).csv"    # 汇总指标 CSV


# ===================================================================


# ==============
# 网络定义（与训练一致：GRU + scan）
# ==============
# # 1. RNN
# class ScannedRNN(nn.Module):
#     """按时间维（T）扫描的 GRU：
#        - carry: (B, H)  为批内隐藏态
#        - x=(ins,resets)：ins:(B,D) 当前步输入；resets:(B,) 步内哪些样本要重置隐藏态
#     """
#     @functools.partial(
#         nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False}
#     )
#     @nn.compact
#     def __call__(self, carry, x):
#         rnn_state = carry
#         ins, resets = x  # ins:(B,D), resets:(B,)
#         # 在 episode 边界（done）处重置隐藏态，避免“跨回合泄漏”
#         rnn_state = jnp.where(resets[:, jnp.newaxis],
#                               self.initialize_carry(*rnn_state.shape),
#                               rnn_state)
#         new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
#         return new_rnn_state, y

#     @staticmethod
#     def initialize_carry(batch_size, hidden_size):
#         """构造零初始化隐藏态（与训练脚本一致）"""
#         cell = nn.GRUCell(features=hidden_size)
#         return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# class ActorCriticRNN(nn.Module):
#     """Actor-Critic 带 GRU 的 RNN 架构（与训练一致）"""
#     action_dim: Sequence[int]
#     config: Dict

#     @nn.compact
#     def __call__(self, hidden, x):
#         act = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
#         obs, dones = x  # obs:(T,B,ObsDim), dones:(T,B)

#         # 前端 MLP（与训练相同初始化）
#         emb = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
#         emb = act(emb)

#         # GRU（按 T 扫描）
#         hidden, emb = ScannedRNN()(hidden, (emb, dones))  # hidden:(B,H) emb:(T,B,H)

#         ###############################################################################################
#         # 新增：补回瓶颈层(PPO+RNN(add_fc2_and_layer_norm))
#         nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb)
#         nn_fc2 = nn.LayerNorm()(nn_fc2) # LayerNorm 是“归一化层”：对每个样本在特征维上做标准化，使均值≈0、方差≈1，并带有可学习的缩放/偏置参数（gamma/beta）。作用是稳定分布、减小梯度震荡，特别适合小 batch、RNN/Transformer。它不引入非线性。
#         nn_fc2 = act(nn_fc2) # activation 是“非线性激活函数”（如 ReLU/tanh）：逐元素变换，引入非线性表达能力，不做归一化
#         ###############################################################################################

#         # 策略分支
#         actor = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(emb)
#         actor = act(actor)
#         def head(n): return nn.Dense(n, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)
#         pi_th = distrax.Categorical(logits=head(self.action_dim[0]))
#         pi_el = distrax.Categorical(logits=head(self.action_dim[1]))
#         pi_ai = distrax.Categorical(logits=head(self.action_dim[2]))
#         pi_ru = distrax.Categorical(logits=head(self.action_dim[3]))

#         # 价值分支
#         critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(emb)
#         critic = act(critic)
#         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)  # (T,B,1)
#         return hidden, (pi_th, pi_el, pi_ai, pi_ru), jnp.squeeze(critic, -1)                # (T,B)

# 2. LSTM
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
        """Applies the module."""
        lstm_state = carry  # (h, c)
        ins, resets = x
        h, c = lstm_state
        h = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*h.shape)[0],
            h,
        )
        c = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*c.shape)[1],
            c,
        )
        new_lstm_state, y = nn.LSTMCell(features=ins.shape[1])((h, c), ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.LSTMCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticLSTM(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedLSTM()(hidden, rnn_in)

        # 新增一层全连接
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2) # 试一下加LayerNorm
        nn_fc2 = activation(nn_fc2)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(nn_fc2)
        actor_mean = activation(actor_mean)
        actor_throttle_mean = nn.Dense(
            self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_elevator_mean = nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_aileron_mean = nn.Dense(
            self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_rudder_mean = nn.Dense(
            self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi_throttle = distrax.Categorical(logits=actor_throttle_mean)
        pi_elevator = distrax.Categorical(logits=actor_elevator_mean)
        pi_aileron = distrax.Categorical(logits=actor_aileron_mean)
        pi_rudder = distrax.Categorical(logits=actor_rudder_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)


# =========
# 与训练一致的打包/解包
# =========
def batchify(x: dict, agent_list, num_envs, num_actors):
    """把 {agent: (num_envs, dim)} 堆叠为 (B,dim)，B=num_envs*num_actors"""
    x = jnp.stack([x[a] for a in agent_list])          # (num_actors,num_envs,dim)
    return x.reshape((num_actors * num_envs, -1))      # (B,dim)

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """把 (B,4) 的离散动作拆回 {agent: (num_envs,4)}"""
    x = x.reshape((num_actors, num_envs, -1))          # (num_actors,num_envs,dim)
    return {a: x[i] for i, a in enumerate(agent_list)}


# =========
# 指标范围/归一化 + 控制频率与平滑性工具
# =========
def _ranges_for_head(A: int):
    """给定动作档数 A，返回各指标的理论范围（用于归一化/解释）"""
    return {
        "pmax_mean":        (1.0 / A, 1.0),         # 最低≈均匀，最高=完全确定
        "pmax_ge_0.9":      (0.0, 1.0),             # 比例
        "margin_mean":      (0.0, 1.0),             # top1-top2，理论上界接近1
        "entropy_mean":     (0.0, float(np.log(A))),# 熵上界 ln(A)
        "mode_change_rate": (0.0, 1.0),             # 变更率
        "step_change_norm": (0.0, 1.0),             # 步幅（|Δ|/(A-1)）
    }

def _norm(x, lo, hi):
    """将 x 归一化到 [0,1]（便于不同 A 比较）"""
    return float(np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0))

def _run_lengths(arr_1d: np.ndarray) -> np.ndarray:
    """计算离散序列的驻留段长度（相同值连续出现的长度）。长度之和 = 序列总长。"""
    if arr_1d.size == 0:
        return np.zeros(0, dtype=int)
    change = np.diff(arr_1d) != 0
    idx = np.nonzero(change)[0] + 1
    runs = np.diff(np.r_[0, idx, arr_1d.size])
    return runs  # e.g., [3,1,5,...] 表示 3步相同、1步相同、5步相同……

def _per_head_dwell_stats(modes_seq: np.ndarray, A: int, dt_control: float):
    """
    计算单个动作头的“驻留时间统计 + 步幅均值”：
    - modes_seq: 整个评估期间拼接后的离散动作序列（(T,) 或 (T,B)；B>1时会拼接）。
    - 返回：
        * dwell_steps:  {'mean','median','p10','p90'}（单位：步）
        * dwell_seconds:{'mean','median','p10','p90'}（单位：秒，=步*dt_control）
        * step_change_norm_mean: |a_t - a_{t-1}|/(A-1) 的均值，范围[0,1]，越小越平滑。
    """
    if modes_seq.ndim == 2:
        modes_seq = modes_seq.reshape(-1,)
    runs = _run_lengths(modes_seq.astype(int))
    if runs.size == 0:
        dwell_steps = dict(mean=0.0, median=0.0, p10=0.0, p90=0.0)
    else:
        dwell_steps = dict(
            mean=float(runs.mean()),
            median=float(np.median(runs)),
            p10=float(np.percentile(runs, 10)),
            p90=float(np.percentile(runs, 90)),
        )
    dwell_seconds = {k: v * dt_control for k, v in dwell_steps.items()}
    if modes_seq.size <= 1:
        step_change_norm_mean = 0.0
    else:
        delta = np.abs(np.diff(modes_seq.astype(float)))
        step_change_norm_mean = float((delta / max(A - 1, 1)).mean())
    return dwell_steps, dwell_seconds, step_change_norm_mean


# =========
# 评估主流程
# =========
def evaluate_checkpoint() -> Dict[str, Any]:
    # —— 与训练关键超参一致（影响参数树/初始化的部分务必匹配）
    config = {
        "SEED": SEED, "NUM_ENVS": NUM_ENVS, "NUM_ACTORS": 1,
        "FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128,
        "MAX_GRAD_NORM": 2, "ACTIVATION": "relu", "LR": 3e-4,
    }

    # 构建环境与控制周期（从环境参数读取真实 sim 频率与智能体交互步数）
    env_params = Heading_Pitch_V_TaskParams()
    env = LogWrapper(AeroPlanaxHeading_Pitch_V_Env(env_params))
    dt_control = float(env_params.agent_interaction_steps) / float(env_params.sim_freq)  # 例如 10/50=0.2s

    # 构图与参数初始化（shape 必须与训练一致）
    # net = ActorCriticRNN(ACTION_DIMS, config=config)
    net = ActorCriticLSTM(ACTION_DIMS, config=config)
    rng = jax.random.PRNGKey(config["SEED"])
    obs_dim = env.observation_space(env.agents[0], env_params).shape
    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], *obs_dim)),  # (T=1,B,Obs)
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),            # (T=1,B) 的 done
    )
    # init_h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    init_h = ScannedLSTM.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    params = net.init(rng, init_h, init_x)

    # 恢复 checkpoint（用 target 结构避免拓扑不匹配）
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    state_item = {"params": ts.params, "opt_state": ts.opt_state, "epoch": jnp.array(0)}
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore(item=state_item))
    params = restored["params"]
    restored_epoch = int(restored.get("epoch", jnp.array(-1)))
    print(f"[Evaluate] Restored epoch: {restored_epoch}")

    rng, _ = jax.random.split(rng)

    # 跨 episode 汇总容器
    ep_returns, ep_lengths = [], []
    ep_pmax_means, ep_margin_means, ep_entropy_means, ep_change_rates, ep_pmax_ge09 = [], [], [], [], []
    # 记录每个头的离散动作序列（每个 episode 一段序列，用于后续可视化/导出）
    ep_modes_per_head: List[List[np.ndarray]] = [ [] for _ in range(4) ]

    for ep in range(NUM_EPISODES):
        rng, ep_key = jax.random.split(rng)
        reset_keys = jax.random.split(ep_key, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
        obs = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        done = jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool)
        # h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        h = ScannedLSTM.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        ret_sum, steps_count = 0.0, 0
        pmax_list, margin_list, entropy_list, pmax_ge09_list = [], [], [], []
        change_count = jnp.zeros((4,), dtype=jnp.float32)
        prev_modes = None
        modes_buffer = [ [] for _ in range(4) ]  # 暂存本回合的离散动作序列

        for _ in range(STEPS_LIMIT):
            # 前向推理（T=1 的小批）
            ac_in = (obs[None, :], done[None, :])           # (1,B,Obs), (1,B)
            h, pis, _ = net.apply(params, h, ac_in)
            pi_th, pi_el, pi_ai, pi_ru = pis

            # 评估用的分布形态指标
            def head_metrics(pi):
                probs = jax.nn.softmax(pi.logits, axis=-1)  # (1,B,A)
                probs = jnp.clip(probs, 1e-9, 1.0)
                pmax = probs.max(axis=-1)                   # (1,B)
                top2 = jnp.sort(probs, axis=-1)[..., -2:]   # (1,B,2)
                margin = top2[..., 1] - top2[..., 0]        # (1,B)
                ent = pi.entropy()                          # (1,B)
                ge09 = (pmax >= 0.9).astype(jnp.float32)    # (1,B)
                return pmax.mean(), margin.mean(), ent.mean(), ge09.mean()

            p_m, m_m, e_m, ge_m = zip(*[head_metrics(p) for p in [pi_th, pi_el, pi_ai, pi_ru]])
            pmax_list.append(jnp.stack(p_m))
            margin_list.append(jnp.stack(m_m))
            entropy_list.append(jnp.stack(e_m))
            pmax_ge09_list.append(jnp.stack(ge_m))

            # 决策：贪心/采样（评估建议用贪心）
            if GREEDY_ACTION:
                a_th, a_el, a_ai, a_ru = pi_th.mode(), pi_el.mode(), pi_ai.mode(), pi_ru.mode()
            else:
                rng, sk = jax.random.split(rng); a_th = pi_th.sample(seed=sk)
                rng, sk = jax.random.split(rng); a_el = pi_el.sample(seed=sk)
                rng, sk = jax.random.split(rng); a_ai = pi_ai.sample(seed=sk)
                rng, sk = jax.random.split(rng); a_ru = pi_ru.sample(seed=sk)

            # 去掉 T 维，得到 (B,)
            a_th = a_th.squeeze(0); a_el = a_el.squeeze(0); a_ai = a_ai.squeeze(0); a_ru = a_ru.squeeze(0)
            actions = jnp.stack([a_th, a_el, a_ai, a_ru], axis=-1)  # (B,4)

            # 保存该步动作（用于时序可视化/导出；评估时 B=1，为标量）
            modes_buffer[0].append(np.array(a_th))
            modes_buffer[1].append(np.array(a_el))
            modes_buffer[2].append(np.array(a_ai))
            modes_buffer[3].append(np.array(a_ru))

            # mode 变更率（只看“变没变”，不看“变了多少”）
            if prev_modes is not None:
                change_count = change_count + jnp.mean((actions != prev_modes).astype(jnp.float32), axis=0)
            prev_modes = actions

            # 环境交互一步（一次控制输出对应 agent_interaction_steps 个仿真步）
            rng, step_key = jax.random.split(rng)
            step_keys = jax.random.split(step_key, config["NUM_ENVS"])
            action_dict = unbatchify(actions, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
            ob, env_state, rew, dn, _ = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_keys, env_state, action_dict
            )

            # 整理输出
            r = jnp.stack([rew[a] for a in env.agents]).reshape(-1)
            d = jnp.stack([dn[a]  for a in env.agents]).reshape(-1)
            obs = batchify(ob, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])

            ret_sum += float(r.mean())
            done = d
            steps_count += 1

            if bool(d.any()):  # 单环境下回合结束
                break

        # 本回合聚合
        pmax_arr    = jnp.stack(pmax_list)       # (T,4)
        margin_arr  = jnp.stack(margin_list)     # (T,4)
        entropy_arr = jnp.stack(entropy_list)    # (T,4)
        ge09_arr    = jnp.stack(pmax_ge09_list)  # (T,4)
        change_rate = change_count / max(steps_count - 1, 1)  # (4,)

        ep_returns.append(ret_sum)
        ep_lengths.append(steps_count)
        ep_pmax_means.append(np.array(pmax_arr.mean(axis=0)))
        ep_margin_means.append(np.array(margin_arr.mean(axis=0)))
        ep_entropy_means.append(np.array(entropy_arr.mean(axis=0)))
        ep_change_rates.append(np.array(change_rate))
        ep_pmax_ge09.append(np.array(ge09_arr.mean(axis=0)))

        # 保存该回合的离散动作序列（形状 (T,B)；评估时 B=1）
        for hidx in range(4):
            ep_modes_per_head[hidx].append(np.stack(modes_buffer[hidx], axis=0))

        print(f"[Episode {ep+1}/{NUM_EPISODES}] steps={steps_count}  return(sum)={ret_sum:.4f}")

    # 选择进入统计的 episode（是否只保留“满步”）
    ep_returns = np.array(ep_returns)
    ep_lengths = np.array(ep_lengths)
    ep_pmax_means    = np.stack(ep_pmax_means)
    ep_margin_means  = np.stack(ep_margin_means)
    ep_entropy_means = np.stack(ep_entropy_means)
    ep_change_rates  = np.stack(ep_change_rates)
    ep_pmax_ge09     = np.stack(ep_pmax_ge09)

    if ONLY_FULL_EPISODES:
        mask = (ep_lengths == STEPS_LIMIT)
        dropped = np.where(~mask)[0]
        if dropped.size > 0:
            print(f"[Info] Dropped episodes (not full length): {dropped.tolist()}")
        used = mask
        if int(used.sum()) == 0:
            print("[Warn] 没有任何 episode 跑满 STEPS_LIMIT，改为使用全部 episode 做统计（避免空结果）。")
            used = np.ones_like(mask, dtype=bool)
    else:
        used = np.ones_like(ep_lengths, dtype=bool)

    # —— 基础指标输出（与之前一致）——
    used_returns = ep_returns[used]
    used_lengths = ep_lengths[used]
    result = {
        "episodes_total": int(len(ep_returns)),
        "episodes_used":  int(used.sum()),
        "used_full_length_only": bool(ONLY_FULL_EPISODES),

        "return_sum_mean": float(used_returns.mean()),
        "return_sum_std":  float(used_returns.std()),
        "length_mean":     float(used_lengths.mean()),
        "length_std":      float(used_lengths.std()),

        "pmax_mean_per_head":           ep_pmax_means[used].mean(axis=0),
        "pmax_ge_0.9_per_head":         ep_pmax_ge09[used].mean(axis=0),
        "margin_mean_per_head":         ep_margin_means[used].mean(axis=0),
        "entropy_mean_per_head":        ep_entropy_means[used].mean(axis=0),
        "mode_change_rate_per_head":    ep_change_rates[used].mean(axis=0),

        # 控制频率信息（用于换算 Hz）
        "sim_freq": int(env_params.sim_freq),
        "agent_interaction_steps": int(env_params.agent_interaction_steps),
        "dt_control": dt_control,

        # 供可视化/导出使用的原始序列 & 选择的回合掩码
        "used_mask": used,
        "ep_modes_per_head": ep_modes_per_head,  # List[4][E]，每个元素是 (T,B) 的 numpy 数组（B=1）
        "ep_lengths": ep_lengths,
    }

    # —— 时间感知的“平滑性”统计：驻留 + 步幅 ——（在“被使用的 episode”上合并统计）
    dwell_steps_stats, dwell_seconds_stats, step_change_norm_mean = [], [], []
    # 为直方图准备“所有驻留段”的拼接数据（用于画直方图）
    dwell_all_steps_per_head: List[np.ndarray] = []

    for hidx, A in enumerate(ACTION_DIMS):
        seqs = [ ep_modes_per_head[hidx][i] for i in range(len(ep_modes_per_head[hidx])) if used[i] ]
        if len(seqs) == 0:
            dws = dict(mean=0.0, median=0.0, p10=0.0, p90=0.0)
            dws_sec = dict(mean=0.0, median=0.0, p10=0.0, p90=0.0)
            scn = 0.0
            dwell_all = np.zeros(0, dtype=int)
        else:
            cat = np.concatenate(seqs, axis=0)  # (sum_T, B)；评估时 B=1
            dws, dws_sec, scn = _per_head_dwell_stats(cat, A, dt_control)
            # 直方图用的“所有驻留段”
            runs = _run_lengths(cat.reshape(-1,))
            dwell_all = runs

        dwell_steps_stats.append(dws)
        dwell_seconds_stats.append(dws_sec)
        step_change_norm_mean.append(scn)
        dwell_all_steps_per_head.append(dwell_all)

    result["dwell_steps_stats_per_head"]     = dwell_steps_stats
    result["dwell_seconds_stats_per_head"]   = dwell_seconds_stats
    result["step_change_norm_mean_per_head"] = step_change_norm_mean
    result["dwell_all_steps_per_head"]       = dwell_all_steps_per_head  # 直方图使用

    return result


# =========
# 可视化与导出
# =========
def ensure_outdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_summary_csv(report: Dict[str, Any], out_dir: str, csv_name: str):
    """导出汇总指标到 CSV，便于长期追踪/对比"""
    ensure_outdir(out_dir)
    df = pd.DataFrame({
        "head": HEAD_LABELS,
        "A": ACTION_DIMS,
        "pmax_mean": report["pmax_mean_per_head"],
        "pmax_ge_0.9": report["pmax_ge_0.9_per_head"],
        "margin_mean": report["margin_mean_per_head"],
        "entropy_mean": report["entropy_mean_per_head"],
        "mode_change_rate": report["mode_change_rate_per_head"],
        "dwell_mean_steps": [d["mean"]   for d in report["dwell_steps_stats_per_head"]],
        "dwell_median_steps":[d["median"] for d in report["dwell_steps_stats_per_head"]],
        "step_change_norm_mean": report["step_change_norm_mean_per_head"],
    })
    # 在 CSV 头部附加总体信息（单行 DataFrame 另存）
    meta = pd.DataFrame({
        "episodes_used": [report["episodes_used"]],
        "episodes_total":[report["episodes_total"]],
        "only_full":     [report["used_full_length_only"]],
        "return_mean":   [report["return_sum_mean"]],
        "return_std":    [report["return_sum_std"]],
        "length_mean":   [report["length_mean"]],
        "length_std":    [report["length_std"]],
        "sim_freq":      [report["sim_freq"]],
        "agent_interaction_steps":[report["agent_interaction_steps"]],
        "dt_control":    [report["dt_control"]],
    })
    meta.to_csv(os.path.join(out_dir, "summary_meta.csv"), index=False)
    df.to_csv(os.path.join(out_dir, csv_name), index=False)
    print(f"[Save] Summary CSV -> {os.path.join(out_dir, csv_name)}")
    print(f"[Save] Meta CSV    -> {os.path.join(out_dir, 'summary_meta.csv')}")

def plot_bar_pmax(report: Dict[str, Any], out_dir: str):
    """各动作头 pmax_mean 对比条形图（越高越“自信”）"""
    ensure_outdir(out_dir)
    vals = report["pmax_mean_per_head"]
    plt.figure(figsize=(7,4))
    plt.bar(HEAD_LABELS, vals)
    plt.ylabel("pmax_mean (higher is more certain)")
    plt.title("pmax_mean Comparison per Action Head")
    plt.grid(True)
    plt.tight_layout()
    fn = os.path.join(out_dir, "bar_pmax.png")
    plt.savefig(fn, dpi=150); plt.close()
    print(f"[Save] {fn}")

def plot_bar_entropy(report: Dict[str, Any], out_dir: str):
    """各动作头 entropy_ratio=H/ln(A) 对比（越低越好；越低表示越集中/自信）"""
    ensure_outdir(out_dir)
    ent = np.array(report["entropy_mean_per_head"])
    lnA = np.log(np.array(ACTION_DIMS))
    ratio = ent / lnA
    plt.figure(figsize=(7,4))
    plt.bar(HEAD_LABELS, ratio)
    plt.ylabel("entropy_ratio = H/ln(A) (lower is better)")
    plt.title("Entropy Ratio Comparison per Action Head")
    plt.grid(True)
    plt.tight_layout()
    fn = os.path.join(out_dir, "bar_entropy.png")
    plt.savefig(fn, dpi=150); plt.close()
    print(f"[Save] {fn}")

def plot_hist_dwell(report: Dict[str, Any], out_dir: str):
    """各动作头驻留长度Dwell Length (steps)直方图（步 & 秒）"""
    ensure_outdir(out_dir)
    dwell_all_steps = report["dwell_all_steps_per_head"]
    dt = report["dt_control"]

    # 步直方图
    plt.figure(figsize=(8,5))
    bins = np.arange(1, 51)  # 1~50步，足够看近端抖动；若驻留很长可增大
    for arr, label in zip(dwell_all_steps, HEAD_LABELS):
        if arr.size > 0:
            plt.hist(arr, bins=bins, alpha=0.5, label=label, density=True)
    plt.grid(True)
    plt.xlabel("Dwell Length (steps)")
    plt.ylabel("Probability Density") # 概率密度
    plt.title("Dwell Length Histogram (steps) per Action Head")
    plt.legend()
    plt.tight_layout()
    fn = os.path.join(out_dir, "hist_dwell_steps.png")
    plt.savefig(fn, dpi=150); plt.close()
    print(f"[Save] {fn}")

    # 秒直方图（把步长乘以 dt_control）
    plt.figure(figsize=(8,5))
    for arr, label in zip(dwell_all_steps, HEAD_LABELS):
        if arr.size > 0:
            plt.hist(arr * dt, bins=50, alpha=0.5, label=label, density=True)
    plt.grid(True)
    plt.xlabel("Dwell Time (seconds)")
    plt.ylabel("Probability Density")
    plt.title("Dwell Time Histogram (seconds) per Action Head")
    plt.legend()
    plt.tight_layout()
    fn = os.path.join(out_dir, "hist_dwell_seconds.png")
    plt.savefig(fn, dpi=150); plt.close()
    print(f"[Save] {fn}")

def choose_rep_episode(report: Dict[str, Any], steps_limit: int) -> int:
    """选择一个代表性的 episode 用于时序可视化/导出：
       - 优先选择“第一个跑满步数的 episode”
       - 若不存在，则选择第一个 episode（used_mask 中的第一个 True；若全 False，则 0）
    """
    used = report["used_mask"]
    ep_lengths = report["ep_lengths"]
    # 满步优先
    idx_full = np.where((used == True) & (ep_lengths == steps_limit))[0]
    if idx_full.size > 0:
        return int(idx_full[0])
    # 回退到第一个 used 的 episode
    idx_used = np.where(used == True)[0]
    if idx_used.size > 0:
        return int(idx_used[0])
    # 最后兜底：0
    return 0

def plot_action_series_for_episode(report: Dict[str, Any], ep_index: int, out_dir: str):
    """画 4 张动作时序图（每个头一张），并导出 Excel（full_series + changes_*）"""
    ensure_outdir(out_dir)
    seqs_per_head: List[List[np.ndarray]] = report["ep_modes_per_head"]  # 4 x E 列表
    # 取该 episode 的 (T,B)；评估时 B=1
    series: List[np.ndarray] = [ seqs_per_head[h][ep_index].reshape(-1,) for h in range(4) ]
    T = len(series[0])
    steps = np.arange(T)

    # 1) 画 4 张时序图（每张一个动作头）
    for h, name in enumerate(HEAD_LABELS):
        plt.figure(figsize=(10,3.2))
        plt.plot(steps, series[h], linewidth=1.0)
        plt.grid(True)
        plt.xlabel("Step (control steps)")
        plt.ylabel(f"{name}(discrete bin index 0~{ACTION_DIMS[h]-1})") #离散档位索引
        plt.title(f"Episode {ep_index} - {name} Action Sequence")
        plt.tight_layout()
        fn = os.path.join(out_dir, f"action_series_{name}.png")
        plt.savefig(fn, dpi=150); plt.close()
        print(f"[Save] {fn}")

    # 2) 导出 Excel：full_series + changes_*（只记录发生变更的时刻）
    excel_path = os.path.join(out_dir, EXCEL_FILENAME)
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # full_series：每一步四个动作值
        df_full = pd.DataFrame({
            "step": steps,
            "throttle": series[0],
            "elevator": series[1],
            "aileron":  series[2],
            "rudder":   series[3],
        })
        df_full.to_excel(writer, sheet_name="full_series", index=False)

        # changes_*：仅记录“有变更”的时刻（含变更前/后）
        for h, name in enumerate(HEAD_LABELS):
            vals = series[h]
            if len(vals) <= 1:
                df_changes = pd.DataFrame(columns=["step", "prev", "new"])
            else:
                prev = vals[:-1]
                new  = vals[1:]
                changed_idx = np.where(prev != new)[0] + 1  # 变更发生在 t（看 new）
                df_changes = pd.DataFrame({
                    "step": changed_idx,
                    "prev": vals[changed_idx - 1],
                    "new":  vals[changed_idx],
                })
            df_changes.to_excel(writer, sheet_name=f"changes_{name}", index=False)

    print(f"[Save] Excel -> {excel_path}")


def main():
    # 1) 评估，得到 report
    report = evaluate_checkpoint()

    # 2) 打印总览（终端）
    print("\n=== 策略评估报告（固定时长） ===")
    print(f"Episodes used / total: {report['episodes_used']} / {report['episodes_total']}"
          + ("  (full-length only)" if report.get("used_full_length_only") else ""))
    print(f"Return(sum) 均值 ± 标准差: {report['return_sum_mean']:.4f} ± {report['return_sum_std']:.4f}")
    print(f"Length 步数 均值 ± 标准差:  {report['length_mean']:.2f} ± {report['length_std']:.2f}")
    print("pmax_mean_per_head [throttle, elevator, aileron, rudder]:", report["pmax_mean_per_head"])
    print("pmax>=0.9 fraction per head:", report["pmax_ge_0.9_per_head"])
    print("margin_mean_per_head:", report["margin_mean_per_head"])
    print("entropy_mean_per_head:", report["entropy_mean_per_head"])
    print("mode_change_rate_per_head:", report["mode_change_rate_per_head"])

    # 3) 导出汇总 CSV
    save_summary_csv(report, OUTPUT_DIR, CSV_SUMMARY)

    # 4) 画总览图（pmax/entropy 的条形图 + 驻留直方图）
    plot_bar_pmax(report, OUTPUT_DIR)
    plot_bar_entropy(report, OUTPUT_DIR)
    plot_hist_dwell(report, OUTPUT_DIR)

    # 5) 选择一个代表 episode，画 4 张动作时序图，并导出 Excel（含“每次变化”的记录）
    rep_ep = choose_rep_episode(report, STEPS_LIMIT)
    print(f"[Info] 代表性 episode index = {rep_ep}")
    plot_action_series_for_episode(report, rep_ep, OUTPUT_DIR)

    print("\n[Done] 所有图片与表格已导出到：", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()



#########################################################################################################


# # -*- coding: utf-8 -*-
# # baseline_evaluate.py
# # 用法：直接运行 python baseline_evaluate.py
# # 改参数：见“用户可改参数”一节
# #
# # ------------------------------
# # 指标总览（中文解释）：
# # ------------------------------
# # 1) Return(sum): 单次评估回合（episode）内的总回报，越大越好。报告里给出“均值±标准差”，用于衡量策略在固定长度下的平均表现与稳定性。
# #    - 如果 ONLY_FULL_EPISODES=True，则只统计“恰好跑满 STEPS_LIMIT 步”的回合（避免早终止拉低均值，但注意存在生存者偏差）。
# #
# # 2) pmax_mean_per_head（每个动作头的最大概率均值）：
# #    - 定义：对每一步的动作分布 p(a|s)，取 max_a p(a)，再在时间与回合上取均值。
# #    - 范围：[1/A, 1]，A 为该动作头的离散档位数（如 31/41）。1/A 接近“完全不确定（均匀）”，1 表示“完全确定（尖分布）”。
# #    - 解读：越大代表策略越“自信/确定”。但过高也可能导致“僵硬/易振荡”，需结合后面的熵、变更率一起看。
# #
# # 3) pmax_ge_0.9_per_head（pmax>=0.9 的时间占比）：
# #    - 定义：统计有多少时间步的 max_a p(a)≥0.9 的比例。
# #    - 范围：[0, 1]。越高代表“极度自信”的时刻越多。
# #
# # 4) margin_mean_per_head（top1-top2 概率间隔）：
# #    - 定义：每步取 p_top1 - p_top2，再做时间与回合均值。
# #    - 范围：[0, 1]。越大代表“第一名”和“第二名”差距越大（更有把握）。
# #
# # 5) entropy_mean_per_head（熵，使用自然对数）：
# #    - 定义：对每步的分布计算熵 H(p)=-∑ p log p，再做时间与回合均值。
# #    - 范围：[0, ln(A)]。越小代表分布越集中（越确定）。报告会同时给出 entropy_ratio=H/ln(A) 与 confidence≈1-H/ln(A)。
# #
# # 6) mode_change_rate_per_head（相邻控制步 argmax 是否变化的比例）：
# #    - 定义：统计相邻两步 argmax(a) 是否不同，不同记 1，相同记 0；在时间与回合上取均值。
# #    - 范围：[0, 1]。越低越“平滑”（抖动少）。但注意这是“是否变了”的统计，不考虑“变了多少”。
# #    - 重要：实际控制频率 = sim_freq / agent_interaction_steps 的倒数（dt_control）。
# #            我们同时给出 flips/sec = mode_change_rate / dt_control（单位 Hz），用于直观对比飞控可接受的翻档频率。
# #
# # 7) dwell（驻留时间统计）：
# #    - 定义：同一档位连续保持的步数（或秒）。给出均值、Median、中位数、10%分位、90%分位。
# #    - 步单位转秒单位用 dt_control（例如 sim=50Hz, 交互=10步 → 0.2s/次控制）。
# #    - 解读：p10 若为 1 步（0.2s），说明有大量“一闪即变”的微抖。p90 和 median 越大越平滑。
# #
# # 8) step_change_norm_mean（步幅大小的归一化均值）：
# #    - 定义：|a_t - a_{t-1}| / (A-1) 的均值，范围 [0, 1]。越小越平滑。
# #    - 与 mode_change_rate 区别：不仅看“是否变了”，还看“跨了多少档位”。
# #
# # 整体建议：对 elevator/aileron（俯仰/副翼）经常会看到 mode_change 及 flips/sec 较大，
# # 可考虑：动作平滑正则（KL-to-previous / TV penalty），动作重复(dwell约束)，离散档重映射（中段密、端段疏），
# # 或在训练中对这两个头单独调节熵系数/抖动惩罚。
# # ------------------------------

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
# # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # 可选：避免显存预分配

# from typing import Sequence, Dict, Any, List
# import functools
# import numpy as np
# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# from flax.linen.initializers import constant, orthogonal
# import distrax
# import optax
# from flax.training.train_state import TrainState
# import orbax.checkpoint as ocp

# from envs.wrappers import LogWrapper
# from envs.aeroplanax_heading_pitch_V import (
#     AeroPlanaxHeading_Pitch_V_Env,
#     Heading_Pitch_V_TaskParams,
# )

# # ======================
# # 用户可改参数（无需命令行）
# # ======================
# CKPT_PATH     = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline(no_fc2_no_layer_norm)/checkpoints/checkpoint_epoch_1000"
# NUM_EPISODES  = 15            # 评估回合数（越多方差越小）
# STEPS_LIMIT   = 1000          # 每个回合的最大步数（固定时长评估）
# SEED          = 42
# GREEDY_ACTION = True          # True=贪心（推荐用于评估；更稳定）；False=按分布采样（用于考察探索期表现）
# NUM_ENVS      = 1             # 评估建议用 1；并行评估>1时代码也兼容
# ONLY_FULL_EPISODES = True     # 只统计“跑满 STEPS_LIMIT”的回合（避免早终止干扰）
# HEAD_LABELS   = ["throttle", "elevator", "aileron", "rudder"]
# ACTION_DIMS   = [31, 41, 41, 41]  # 四个头的离散档数，必须与训练一致
# # ======================


# # ==============
# # 网络定义（与训练一致：GRU + scan）
# # ==============
# class ScannedRNN(nn.Module):
#     """按时间维（T）扫描的 GRU：
#        - carry: (B, H)  为批内隐藏态
#        - x=(ins,resets)：ins:(B,D) 是当前步的输入特征；resets:(B,) 指示哪些 env 在该步要重置隐藏态
#     """
#     @functools.partial(
#         nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False}
#     )
#     @nn.compact
#     def __call__(self, carry, x):
#         rnn_state = carry
#         ins, resets = x  # ins:(B,D), resets:(B,)
#         # 在 episode 边界（done）处重置隐藏态，避免“跨回合泄漏”
#         rnn_state = jnp.where(resets[:, jnp.newaxis],
#                               self.initialize_carry(*rnn_state.shape),
#                               rnn_state)
#         new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
#         return new_rnn_state, y

#     @staticmethod
#     def initialize_carry(batch_size, hidden_size):
#         """构造零初始化隐藏态（与训练脚本一致）"""
#         cell = nn.GRUCell(features=hidden_size)
#         return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# class ActorCriticRNN(nn.Module):
#     """Actor-Critic 带 GRU 的 RNN 架构（与训练一致）"""
#     action_dim: Sequence[int]
#     config: Dict

#     @nn.compact
#     def __call__(self, hidden, x):
#         act = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
#         obs, dones = x  # obs:(T,B,ObsDim), dones:(T,B)

#         # 前端 MLP（与训练相同初始化）
#         emb = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
#         emb = act(emb)

#         # GRU（按 T 扫描）
#         hidden, emb = ScannedRNN()(hidden, (emb, dones))  # hidden:(B,H) emb:(T,B,H)

#         # 策略分支
#         actor = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(emb)
#         actor = act(actor)
#         def head(n): return nn.Dense(n, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)
#         pi_th = distrax.Categorical(logits=head(self.action_dim[0]))
#         pi_el = distrax.Categorical(logits=head(self.action_dim[1]))
#         pi_ai = distrax.Categorical(logits=head(self.action_dim[2]))
#         pi_ru = distrax.Categorical(logits=head(self.action_dim[3]))

#         # 价值分支
#         critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(emb)
#         critic = act(critic)
#         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)  # (T,B,1)
#         return hidden, (pi_th, pi_el, pi_ai, pi_ru), jnp.squeeze(critic, -1)                # (T,B)


# # =========
# # 与训练一致的打包函数
# # =========
# def batchify(x: dict, agent_list, num_envs, num_actors):
#     """把 {agent: (num_envs, dim)} 堆叠为 (B,dim)，B=num_envs*num_actors"""
#     x = jnp.stack([x[a] for a in agent_list])          # (num_actors,num_envs,dim)
#     return x.reshape((num_actors * num_envs, -1))      # (B,dim)

# def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
#     """把 (B,4) 的离散动作拆回 {agent: (num_envs,4)}"""
#     x = x.reshape((num_actors, num_envs, -1))          # (num_actors,num_envs,dim)
#     return {a: x[i] for i, a in enumerate(agent_list)}


# # =========
# # 指标范围/归一化 + 控制频率与平滑性工具
# # =========
# def _ranges_for_head(A: int):
#     """给定动作档数 A，返回各指标的理论范围（用于归一化/解释）"""
#     return {
#         "pmax_mean":        (1.0 / A, 1.0),         # 最低值≈均匀分布，最高值=完全确定
#         "pmax_ge_0.9":      (0.0, 1.0),             # 比例
#         "margin_mean":      (0.0, 1.0),             # top1-top2，理论上界接近1
#         "entropy_mean":     (0.0, float(np.log(A))),# 熵的上界 ln(A)
#         "mode_change_rate": (0.0, 1.0),             # 变更率
#         "step_change_norm": (0.0, 1.0),             # 步幅（|Δ|/(A-1)）
#     }

# def _norm(x, lo, hi):
#     """将 x 归一化到 [0,1]（便于不同 A 比较）"""
#     return float(np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0))

# def _run_lengths(arr_1d: np.ndarray) -> np.ndarray:
#     """计算离散序列的驻留段长度（相同值连续出现的长度）。长度之和 = 序列总长。"""
#     if arr_1d.size == 0:
#         return np.zeros(0, dtype=int)
#     change = np.diff(arr_1d) != 0
#     idx = np.nonzero(change)[0] + 1
#     runs = np.diff(np.r_[0, idx, arr_1d.size])
#     return runs  # e.g., [3,1,5,...] 表示 3步相同、1步相同、5步相同……

# def _per_head_dwell_stats(modes_seq: np.ndarray, A: int, dt_control: float):
#     """
#     计算单个动作头的“驻留时间统计 + 步幅均值”：
#     - modes_seq: 整个评估期间拼接后的离散动作序列（形状 (T,) 或 (T,B)；B>1时会拼接）。
#     - 返回：
#         * dwell_steps:  {'mean','median','p10','p90'}（单位：步）
#         * dwell_seconds:{'mean','median','p10','p90'}（单位：秒，=步*dt_control）
#         * step_change_norm_mean: |a_t - a_{t-1}|/(A-1) 的均值，范围[0,1]，越小越平滑。
#     """
#     # 将 (T,B) 拼接为 (T*B,) 近似总体分布（评估期合并）
#     if modes_seq.ndim == 2:
#         modes_seq = modes_seq.reshape(-1,)
#     # 驻留统计
#     runs = _run_lengths(modes_seq.astype(int))
#     if runs.size == 0:
#         dwell_steps = dict(mean=0.0, median=0.0, p10=0.0, p90=0.0)
#     else:
#         dwell_steps = dict(
#             mean=float(runs.mean()),
#             median=float(np.median(runs)),
#             p10=float(np.percentile(runs, 10)),
#             p90=float(np.percentile(runs, 90)),
#         )
#     dwell_seconds = {k: v * dt_control for k, v in dwell_steps.items()}
#     # 步幅均值（归一化）
#     if modes_seq.size <= 1:
#         step_change_norm_mean = 0.0
#     else:
#         delta = np.abs(np.diff(modes_seq.astype(float)))
#         step_change_norm_mean = float((delta / max(A - 1, 1)).mean())
#     return dwell_steps, dwell_seconds, step_change_norm_mean

# def print_head_metrics_with_ranges(report: Dict[str, Any], action_dims, labels,
#                                    dwell_steps, dwell_seconds, step_change_norm_mean, dt_control):
#     """打印每个动作头的指标，并标注理论范围与时间尺度（Hz / 秒）解释"""
#     pm, pm90 = report["pmax_mean_per_head"], report["pmax_ge_0.9_per_head"]
#     mg, ent = report["margin_mean_per_head"], report["entropy_mean_per_head"]
#     mcr     = report["mode_change_rate_per_head"]

#     print("\n--- 每个动作头的指标（含理论范围、归一化分数与控制频率解释） ---")
#     print(f"[control] sim_freq={report['sim_freq']} Hz, agent_interaction_steps={report['agent_interaction_steps']}, dt_control={dt_control:.3f} s/步")
#     for i, (A, name) in enumerate(zip(action_dims, labels)):
#         r = _ranges_for_head(A); lnA = r["entropy_mean"][1]
#         flip_hz = mcr[i] / dt_control  # 变更率换算到次/秒，更直观
#         # 归一化解释：用于快速比较不同 A 的头（不同离散粒度）在“确定性/自信度”上的相对水平
#         print(f"[{name}] (A={A})")
#         print(f"  pmax_mean = {pm[i]:.3f}  范围[{r['pmax_mean'][0]:.3f}, {r['pmax_mean'][1]:.3f}]  归一化={_norm(pm[i], *r['pmax_mean']):.3f}")
#         print(f"  pmax>=0.9 = {pm90[i]:.3f}  范围[0, 1]（越高代表极度自信的时刻越多）")
#         print(f"  margin    = {mg[i]:.3f}  范围[0, 1]  归一化={_norm(mg[i], *r['margin_mean']):.3f}（top1-top2 概率差）")
#         print(f"  entropy   = {ent[i]:.3f} / ln(A)={lnA:.3f}  比例={ent[i]/lnA:.3f}  置信度≈{1.0 - ent[i]/lnA:.3f}")
#         print(f"  mode_change_rate = {mcr[i]:.3f}（0~1，越低越平滑）  ≈ {flip_hz:.2f} 次/秒")
#         ds, dsec = dwell_steps[i], dwell_seconds[i]
#         print(f"  dwell_steps  (步) 均值/中位/10%/90% = {ds['mean']:.1f}/{ds['median']:.1f}/{ds['p10']:.1f}/{ds['p90']:.1f}")
#         print(f"  dwell_time   (秒) 均值/中位/10%/90% = {dsec['mean']:.2f}/{dsec['median']:.2f}/{dsec['p10']:.2f}/{dsec['p90']:.2f}")
#         print(f"  step_change_norm_mean = {step_change_norm_mean[i]:.3f}（0~1，平均“跨档幅度”；越小越平滑）")


# # =========
# # 评估主流程
# # =========
# def evaluate_checkpoint() -> Dict[str, Any]:
#     # —— 与训练关键超参一致（影响参数树/初始化的部分务必匹配）
#     config = {
#         "SEED": SEED, "NUM_ENVS": NUM_ENVS, "NUM_ACTORS": 1,
#         "FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128,
#         "MAX_GRAD_NORM": 2, "ACTIVATION": "relu", "LR": 3e-4,
#     }

#     # 构建环境与控制周期（从环境参数读取真实 sim 频率与智能体交互步数）
#     env_params = Heading_Pitch_V_TaskParams()
#     env = LogWrapper(AeroPlanaxHeading_Pitch_V_Env(env_params))
#     # 控制周期：dt_control = agent_interaction_steps / sim_freq（例如 10/50=0.2s），
#     # 评估代码里每一步 env.step 就对应一次控制输出。
#     dt_control = float(env_params.agent_interaction_steps) / float(env_params.sim_freq)

#     # 构图与参数初始化（shape 必须与训练一致）
#     net = ActorCriticRNN(ACTION_DIMS, config=config)
#     rng = jax.random.PRNGKey(config["SEED"])
#     obs_dim = env.observation_space(env.agents[0], env_params).shape
#     init_x = (
#         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], *obs_dim)),  # (T=1,B,Obs)
#         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),            # (T=1,B) 的 done
#     )
#     init_h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
#     params = net.init(rng, init_h, init_x)

#     # 恢复 checkpoint（用 target 结构避免拓扑不匹配）
#     tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
#     ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
#     state_item = {"params": ts.params, "opt_state": ts.opt_state, "epoch": jnp.array(0)}
#     ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
#     restored = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore(item=state_item))
#     params = restored["params"]
#     restored_epoch = int(restored.get("epoch", jnp.array(-1)))
#     print(f"[Evaluate] Restored epoch: {restored_epoch}")

#     rng, _ = jax.random.split(rng)

#     # 跨 episode 汇总容器
#     ep_returns, ep_lengths = [], []
#     ep_pmax_means, ep_margin_means, ep_entropy_means, ep_change_rates, ep_pmax_ge09 = [], [], [], [], []
#     # 记录每个头的离散动作序列，用于驻留与步幅分析（每个 episode 一段序列）
#     ep_modes_per_head: List[List[np.ndarray]] = [ [] for _ in range(4) ]  # 4 个动作头

#     for ep in range(NUM_EPISODES):
#         # 每个 episode 独立 seed，保证复现实验更清晰
#         rng, ep_key = jax.random.split(rng)
#         reset_keys = jax.random.split(ep_key, config["NUM_ENVS"])
#         obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
#         obs = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
#         done = jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool)
#         h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

#         ret_sum, steps_count = 0.0, 0
#         pmax_list, margin_list, entropy_list, pmax_ge09_list = [], [], [], []
#         change_count = jnp.zeros((4,), dtype=jnp.float32)
#         prev_modes = None
#         modes_buffer = [ [] for _ in range(4) ]  # 暂存本回合的离散动作序列

#         for _ in range(STEPS_LIMIT):
#             # 前向推理（T=1 的小批）
#             ac_in = (obs[None, :], done[None, :])           # (1,B,Obs), (1,B)
#             h, pis, _ = net.apply(params, h, ac_in)
#             pi_th, pi_el, pi_ai, pi_ru = pis

#             # 计算每个头的“分布形态”指标（pmax/margin/entropy/ge09）
#             def head_metrics(pi):
#                 probs = jax.nn.softmax(pi.logits, axis=-1)  # (1,B,A)
#                 probs = jnp.clip(probs, 1e-9, 1.0)
#                 pmax = probs.max(axis=-1)                   # (1,B)
#                 top2 = jnp.sort(probs, axis=-1)[..., -2:]   # (1,B,2)
#                 margin = top2[..., 1] - top2[..., 0]        # (1,B)
#                 ent = pi.entropy()                          # (1,B)
#                 ge09 = (pmax >= 0.9).astype(jnp.float32)    # (1,B)
#                 return pmax.mean(), margin.mean(), ent.mean(), ge09.mean()

#             p_m, m_m, e_m, ge_m = zip(*[head_metrics(p) for p in [pi_th, pi_el, pi_ai, pi_ru]])
#             pmax_list.append(jnp.stack(p_m))
#             margin_list.append(jnp.stack(m_m))
#             entropy_list.append(jnp.stack(e_m))
#             pmax_ge09_list.append(jnp.stack(ge_m))

#             # 决策：贪心/采样（评估建议用贪心）
#             if GREEDY_ACTION:
#                 a_th, a_el, a_ai, a_ru = pi_th.mode(), pi_el.mode(), pi_ai.mode(), pi_ru.mode()
#             else:
#                 rng, sk = jax.random.split(rng); a_th = pi_th.sample(seed=sk)
#                 rng, sk = jax.random.split(rng); a_el = pi_el.sample(seed=sk)
#                 rng, sk = jax.random.split(rng); a_ai = pi_ai.sample(seed=sk)
#                 rng, sk = jax.random.split(rng); a_ru = pi_ru.sample(seed=sk)

#             # 去掉 T 维，得到 (B,)
#             a_th = a_th.squeeze(0); a_el = a_el.squeeze(0); a_ai = a_ai.squeeze(0); a_ru = a_ru.squeeze(0)
#             actions = jnp.stack([a_th, a_el, a_ai, a_ru], axis=-1)  # (B,4)

#             # 记录离散动作（用于驻留/步幅分析）；评估时 B=1，因此每步是标量
#             modes_buffer[0].append(np.array(a_th))
#             modes_buffer[1].append(np.array(a_el))
#             modes_buffer[2].append(np.array(a_ai))
#             modes_buffer[3].append(np.array(a_ru))

#             # mode 变更率（是否翻档）统计：只看“变没变”，不看“变了多少”
#             if prev_modes is not None:
#                 change_count = change_count + jnp.mean((actions != prev_modes).astype(jnp.float32), axis=0)
#             prev_modes = actions

#             # 交互一步环境（一次控制输出对应 agent_interaction_steps 个仿真步）
#             rng, step_key = jax.random.split(rng)
#             step_keys = jax.random.split(step_key, config["NUM_ENVS"])
#             action_dict = unbatchify(actions, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
#             ob, env_state, rew, dn, _ = jax.vmap(env.step, in_axes=(0, 0, 0))(
#                 step_keys, env_state, action_dict
#             )

#             # 整理输出
#             r = jnp.stack([rew[a] for a in env.agents]).reshape(-1)
#             d = jnp.stack([dn[a]  for a in env.agents]).reshape(-1)
#             obs = batchify(ob, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])

#             ret_sum += float(r.mean())
#             done = d
#             steps_count += 1

#             if bool(d.any()):  # 单环境下回合结束
#                 break

#         # 本回合聚合
#         pmax_arr    = jnp.stack(pmax_list)       # (T,4)
#         margin_arr  = jnp.stack(margin_list)     # (T,4)
#         entropy_arr = jnp.stack(entropy_list)    # (T,4)
#         ge09_arr    = jnp.stack(pmax_ge09_list)  # (T,4)
#         change_rate = change_count / max(steps_count - 1, 1)  # (4,)

#         ep_returns.append(ret_sum)
#         ep_lengths.append(steps_count)
#         ep_pmax_means.append(np.array(pmax_arr.mean(axis=0)))
#         ep_margin_means.append(np.array(margin_arr.mean(axis=0)))
#         ep_entropy_means.append(np.array(entropy_arr.mean(axis=0)))
#         ep_change_rates.append(np.array(change_rate))
#         ep_pmax_ge09.append(np.array(ge09_arr.mean(axis=0)))

#         # 保存该回合的离散动作序列（形状 (T,B)；评估时 B=1）
#         for hidx in range(4):
#             ep_modes_per_head[hidx].append(np.stack(modes_buffer[hidx], axis=0))

#         print(f"[Episode {ep+1}/{NUM_EPISODES}] steps={steps_count}  return(sum)={ret_sum:.4f}")

#     # 选择进入统计的 episode（是否只保留“满步”）
#     ep_returns = np.array(ep_returns)
#     ep_lengths = np.array(ep_lengths)
#     ep_pmax_means    = np.stack(ep_pmax_means)
#     ep_margin_means  = np.stack(ep_margin_means)
#     ep_entropy_means = np.stack(ep_entropy_means)
#     ep_change_rates  = np.stack(ep_change_rates)
#     ep_pmax_ge09     = np.stack(ep_pmax_ge09)

#     if ONLY_FULL_EPISODES:
#         mask = (ep_lengths == STEPS_LIMIT)
#         dropped = np.where(~mask)[0]
#         if dropped.size > 0:
#             print(f"[Info] Dropped episodes (not full length): {dropped.tolist()}")
#         used = mask
#         if int(used.sum()) == 0:
#             print("[Warn] 没有任何 episode 跑满 STEPS_LIMIT，改为使用全部 episode 做统计（避免空结果）。")
#             used = np.ones_like(mask, dtype=bool)
#     else:
#         used = np.ones_like(ep_lengths, dtype=bool)

#     # —— 基础指标输出（与之前一致）——
#     used_returns = ep_returns[used]
#     used_lengths = ep_lengths[used]
#     result = {
#         "episodes_total": int(len(ep_returns)),
#         "episodes_used":  int(used.sum()),
#         "used_full_length_only": bool(ONLY_FULL_EPISODES),

#         "return_sum_mean": float(used_returns.mean()),
#         "return_sum_std":  float(used_returns.std()),
#         "length_mean":     float(used_lengths.mean()),
#         "length_std":      float(used_lengths.std()),

#         "pmax_mean_per_head":           ep_pmax_means[used].mean(axis=0),
#         "pmax_ge_0.9_per_head":         ep_pmax_ge09[used].mean(axis=0),
#         "margin_mean_per_head":         ep_margin_means[used].mean(axis=0),
#         "entropy_mean_per_head":        ep_entropy_means[used].mean(axis=0),
#         "mode_change_rate_per_head":    ep_change_rates[used].mean(axis=0),

#         # 控制频率信息（用于换算 Hz）
#         "sim_freq": int(env_params.sim_freq),
#         "agent_interaction_steps": int(env_params.agent_interaction_steps),
#         "dt_control": dt_control,
#     }

#     # —— 时间感知的“平滑性”统计：驻留 + 步幅 ——（在“被使用的 episode”上合并统计）
#     dwell_steps_stats, dwell_seconds_stats, step_change_norm_mean = [], [], []
#     for hidx, A in enumerate(ACTION_DIMS):
#         seqs = [ ep_modes_per_head[hidx][i] for i in range(len(ep_modes_per_head[hidx])) if used[i] ]
#         if len(seqs) == 0:
#             dws = dict(mean=0.0, median=0.0, p10=0.0, p90=0.0)
#             dws_sec = dict(mean=0.0, median=0.0, p10=0.0, p90=0.0)
#             scn = 0.0
#         else:
#             cat = np.concatenate(seqs, axis=0)  # (sum_T, B)；评估时 B=1
#             dws, dws_sec, scn = _per_head_dwell_stats(cat, A, dt_control)
#         dwell_steps_stats.append(dws)
#         dwell_seconds_stats.append(dws_sec)
#         step_change_norm_mean.append(scn)

#     result["dwell_steps_stats_per_head"]     = dwell_steps_stats
#     result["dwell_seconds_stats_per_head"]   = dwell_seconds_stats
#     result["step_change_norm_mean_per_head"] = step_change_norm_mean

#     return result


# def main():
#     # 跑评估，打印总览
#     report = evaluate_checkpoint()
#     print("\n=== 策略评估报告（固定时长） ===")
#     print(f"Episodes used / total: {report['episodes_used']} / {report['episodes_total']}"
#           + ("  (full-length only)" if report.get("used_full_length_only") else ""))
#     print(f"Return(sum) 均值 ± 标准差: {report['return_sum_mean']:.4f} ± {report['return_sum_std']:.4f}")
#     print(f"Length 步数 均值 ± 标准差:  {report['length_mean']:.2f} ± {report['length_std']:.2f}")
#     print("pmax_mean_per_head [throttle, elevator, aileron, rudder]:", report["pmax_mean_per_head"])
#     print("pmax>=0.9 fraction per head:", report["pmax_ge_0.9_per_head"])
#     print("margin_mean_per_head:", report["margin_mean_per_head"])
#     print("entropy_mean_per_head:", report["entropy_mean_per_head"])
#     print("mode_change_rate_per_head:", report["mode_change_rate_per_head"])

#     # 打印每个头的细化解释（含范围/归一化 + 控制频率 + 驻留/步幅）
#     print_head_metrics_with_ranges(
#         report,
#         ACTION_DIMS,
#         HEAD_LABELS,
#         report["dwell_steps_stats_per_head"],
#         report["dwell_seconds_stats_per_head"],
#         report["step_change_norm_mean_per_head"],
#         report["dt_control"],
#     )


# if __name__ == "__main__":
#     main()



# #########################################################################################################
# # # baseline_evaluate.py
# # # 直接运行：python baseline_evaluate.py
# # # 改参数：见“用户可改参数”一节

# # import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
# # # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # 可选：避免预分配

# # from typing import Sequence, Dict, Any
# # import functools
# # import numpy as np
# # import jax
# # import jax.numpy as jnp
# # import flax.linen as nn
# # from flax.linen.initializers import constant, orthogonal
# # import distrax
# # import optax
# # from flax.training.train_state import TrainState
# # import orbax.checkpoint as ocp

# # from envs.wrappers import LogWrapper
# # from envs.aeroplanax_heading_pitch_V import (
# #     AeroPlanaxHeading_Pitch_V_Env,
# #     Heading_Pitch_V_TaskParams,
# # )

# # # ======================
# # # 用户可改参数（无需命令行）
# # # ======================
# # CKPT_PATH     = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline(no_fc2_no_layer_norm)/checkpoints/checkpoint_epoch_1000"
# # NUM_EPISODES  = 15
# # STEPS_LIMIT   = 1000
# # SEED          = 42
# # GREEDY_ACTION = True   # True=贪心 mode()；False=按分布采样
# # NUM_ENVS      = 1      # 建议评估用1，后续想并行可以改大（代码已兼容）
# # ONLY_FULL_EPISODES = True  # 只统计跑满 STEPS_LIMIT 的 episode；其余丢弃

# # HEAD_LABELS = ["throttle", "elevator", "aileron", "rudder"]
# # ACTION_DIMS = [31, 41, 41, 41]  # 与训练一致
# # # ======================


# # # ==============
# # # 网络定义（与训练一致：GRU + scan）
# # # ==============
# # class ScannedRNN(nn.Module):
# #     @functools.partial(
# #         nn.scan,
# #         variable_broadcast="params",
# #         in_axes=0,
# #         out_axes=0,
# #         split_rngs={"params": False},
# #     )
# #     @nn.compact
# #     def __call__(self, carry, x):
# #         # carry: (B, H)
# #         # x: (ins, resets) 其中 ins: (B, D)，resets: (B,)
# #         rnn_state = carry
# #         ins, resets = x
# #         rnn_state = jnp.where(
# #             resets[:, jnp.newaxis],                     # (B,1)
# #             self.initialize_carry(*rnn_state.shape),    # (B,H)
# #             rnn_state
# #         )
# #         new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
# #         return new_rnn_state, y

# #     @staticmethod
# #     def initialize_carry(batch_size, hidden_size):
# #         cell = nn.GRUCell(features=hidden_size)
# #         return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# # class ActorCriticRNN(nn.Module):
# #     action_dim: Sequence[int]
# #     config: Dict

# #     @nn.compact
# #     def __call__(self, hidden, x):
# #         # x: (obs, dones)  obs: (T,B,ObsDim)  dones: (T,B)
# #         act = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
# #         obs, dones = x

# #         # 前端 MLP（与训练相同）
# #         embedding = nn.Dense(
# #             self.config["FC_DIM_SIZE"],
# #             kernel_init=orthogonal(np.sqrt(2)),
# #             bias_init=constant(0.0),
# #         )(obs)
# #         embedding = act(embedding)

# #         # GRU（时间维T在最前，使用scan）
# #         hidden, embedding = ScannedRNN()(hidden, (embedding, dones))  # hidden: (B,H); embedding: (T,B,H)

# #         # 策略头（四个离散动作头）
# #         actor_mean = nn.Dense(
# #             self.config["GRU_HIDDEN_DIM"],
# #             kernel_init=orthogonal(2),
# #             bias_init=constant(0.0),
# #         )(embedding)
# #         actor_mean = act(actor_mean)

# #         def head(n):
# #             return nn.Dense(n, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

# #         pi_throttle = distrax.Categorical(logits=head(self.action_dim[0]))
# #         pi_elevator = distrax.Categorical(logits=head(self.action_dim[1]))
# #         pi_aileron  = distrax.Categorical(logits=head(self.action_dim[2]))
# #         pi_rudder   = distrax.Categorical(logits=head(self.action_dim[3]))

# #         # 价值头
# #         critic = nn.Dense(
# #             self.config["FC_DIM_SIZE"],
# #             kernel_init=orthogonal(2),
# #             bias_init=constant(0.0),
# #         )(embedding)
# #         critic = act(critic)
# #         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)  # (T,B,1)

# #         return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)  # (T,B)


# # # =========
# # # 与训练一致的打包函数
# # # =========
# # def batchify(x: dict, agent_list, num_envs, num_actors):
# #     x = jnp.stack([x[a] for a in agent_list])         # (num_actors, num_envs, dim)
# #     return x.reshape((num_actors * num_envs, -1))     # (B, dim)

# # def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
# #     x = x.reshape((num_actors, num_envs, -1))         # (num_actors, num_envs, dim)
# #     return {a: x[i] for i, a in enumerate(agent_list)}


# # # =========
# # # 工具：指标范围与归一化
# # # =========
# # def _ranges_for_head(A: int):
# #     # 理论范围
# #     return {
# #         "pmax_mean":       (1.0 / A, 1.0),
# #         "pmax_ge_0.9":     (0.0, 1.0),
# #         "margin_mean":     (0.0, 1.0),     # 理想上界 1（top1=1, top2=0）
# #         "entropy_mean":    (0.0, float(np.log(A))),  # 自然对数
# #         "mode_change_rate":(0.0, 1.0),
# #     }

# # def _norm(x, lo, hi):
# #     return float(np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0))

# # def print_head_metrics_with_ranges(report: Dict[str, Any], action_dims, labels):
# #     pm   = report["pmax_mean_per_head"]
# #     pm90 = report["pmax_ge_0.9_per_head"]
# #     mg   = report["margin_mean_per_head"]
# #     ent  = report["entropy_mean_per_head"]
# #     mcr  = report["mode_change_rate_per_head"]

# #     print("\n--- Per-Head Metrics (with ranges & normalized scores) ---")
# #     for i, (A, name) in enumerate(zip(action_dims, labels)):
# #         r = _ranges_for_head(A)
# #         lnA = r["entropy_mean"][1]
# #         print(f"[{name}] (A={A})")
# #         print(f"  pmax_mean = {pm[i]:.3f} (range {r['pmax_mean'][0]:.3f}–{r['pmax_mean'][1]:.3f}, "
# #               f"normalized {_norm(pm[i], *r['pmax_mean']):.3f})")
# #         print(f"  pmax>=0.9 = {pm90[i]:.3f} (range 0–1)")
# #         print(f"  margin    = {mg[i]:.3f} (range 0–1, normalized {_norm(mg[i], *r['margin_mean']):.3f})")
# #         print(f"  entropy   = {ent[i]:.3f} / ln(A)={lnA:.3f} "
# #               f"(entropy_ratio={ent[i]/lnA:.3f}, confidence≈{1.0 - ent[i]/lnA:.3f})")
# #         print(f"  mode_change_rate = {mcr[i]:.3f} (range 0–1, lower is smoother)")


# # # =========
# # # 评估函数
# # # =========
# # def evaluate_checkpoint() -> Dict[str, Any]:
# #     # —— 与训练关键超参一致（影响参数树/初始化的部分一定要匹配）
# #     config = {
# #         "SEED": SEED,
# #         "NUM_ENVS": NUM_ENVS,
# #         "NUM_ACTORS": 1,        # 本任务单智能体
# #         "FC_DIM_SIZE": 128,
# #         "GRU_HIDDEN_DIM": 128,
# #         "MAX_GRAD_NORM": 2,
# #         "ACTIVATION": "relu",
# #         "LR": 3e-4,             # 仅用于构建 TrainState 作为 restore 的 target
# #     }

# #     # 环境
# #     env_params = Heading_Pitch_V_TaskParams()
# #     env = LogWrapper(AeroPlanaxHeading_Pitch_V_Env(env_params))

# #     # 构图并初始化参数（shape 要与训练完全一致）
# #     net = ActorCriticRNN(ACTION_DIMS, config=config)
# #     rng = jax.random.PRNGKey(config["SEED"])
# #     obs_dim = env.observation_space(env.agents[0], env_params).shape

# #     init_x = (
# #         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], *obs_dim)),  # (T=1,B,ObsDim)
# #         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),            # (T=1,B)
# #     )
# #     init_h = ScannedRNN.initialize_carry(
# #         config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
# #     )
# #     params = net.init(rng, init_h, init_x)

# #     # 用 target 恢复，避免不安全警告
# #     tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
# #                      optax.adam(config["LR"], eps=1e-5))
# #     ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
# #     state_item = {"params": ts.params, "opt_state": ts.opt_state, "epoch": jnp.array(0)}

# #     ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
# #     restored = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore(item=state_item))
# #     params = restored["params"]
# #     restored_epoch = int(restored.get("epoch", jnp.array(-1)))
# #     print(f"[Evaluate] Restored epoch: {restored_epoch}")

# #     rng, _ = jax.random.split(rng)

# #     # 跨 episode 统计
# #     ep_returns, ep_lengths = [], []
# #     ep_pmax_means, ep_margin_means, ep_entropy_means, ep_change_rates, ep_pmax_ge09 = [], [], [], [], []

# #     for ep in range(NUM_EPISODES):
# #         # reset
# #         rng, ep_key = jax.random.split(rng)
# #         reset_keys = jax.random.split(ep_key, config["NUM_ENVS"])
# #         obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
# #         obs = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])  # (B,ObsDim)
# #         done = jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool)
# #         h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

# #         # per-episode 累积
# #         ret_sum = 0.0
# #         steps_count = 0

# #         pmax_list, margin_list, entropy_list, pmax_ge09_list = [], [], [], []
# #         change_count = jnp.zeros((4,), dtype=jnp.float32)
# #         prev_modes = None

# #         for _ in range(STEPS_LIMIT):
# #             ac_in = (obs[None, :], done[None, :])           # (T=1,B,Obs) & (1,B)
# #             h, pis, value = net.apply(params, h, ac_in)
# #             pi_th, pi_el, pi_ai, pi_ru = pis

# #             # 指标
# #             def head_metrics(pi):
# #                 probs = jax.nn.softmax(pi.logits, axis=-1)  # (1,B,A)
# #                 probs = jnp.clip(probs, 1e-9, 1.0)
# #                 pmax = probs.max(axis=-1)                   # (1,B)
# #                 top2 = jnp.sort(probs, axis=-1)[..., -2:]   # (1,B,2)
# #                 margin = top2[..., 1] - top2[..., 0]        # (1,B)
# #                 ent = pi.entropy()                          # (1,B)
# #                 ge09 = (pmax >= 0.9).astype(jnp.float32)    # (1,B)
# #                 return pmax.mean(), margin.mean(), ent.mean(), ge09.mean()

# #             m = [head_metrics(p) for p in [pi_th, pi_el, pi_ai, pi_ru]]
# #             p_m, m_m, e_m, ge_m = zip(*m)
# #             pmax_list.append(jnp.stack(p_m))       # (4,)
# #             margin_list.append(jnp.stack(m_m))     # (4,)
# #             entropy_list.append(jnp.stack(e_m))    # (4,)
# #             pmax_ge09_list.append(jnp.stack(ge_m)) # (4,)

# #             # 动作（贪心/采样）
# #             if GREEDY_ACTION:
# #                 a_th = pi_th.mode(); a_el = pi_el.mode(); a_ai = pi_ai.mode(); a_ru = pi_ru.mode()
# #             else:
# #                 rng, sk = jax.random.split(rng); a_th = pi_th.sample(seed=sk)
# #                 rng, sk = jax.random.split(rng); a_el = pi_el.sample(seed=sk)
# #                 rng, sk = jax.random.split(rng); a_ai = pi_ai.sample(seed=sk)
# #                 rng, sk = jax.random.split(rng); a_ru = pi_ru.sample(seed=sk)

# #             # 去掉时间维 -> (B,)
# #             a_th = a_th.squeeze(0); a_el = a_el.squeeze(0); a_ai = a_ai.squeeze(0); a_ru = a_ru.squeeze(0)
# #             actions = jnp.stack([a_th, a_el, a_ai, a_ru], axis=-1)  # (B,4)

# #             # 模式变更率
# #             if prev_modes is not None:
# #                 change_count = change_count + jnp.mean((actions != prev_modes).astype(jnp.float32), axis=0)
# #             prev_modes = actions

# #             # env.step
# #             rng, step_key = jax.random.split(rng)
# #             step_keys = jax.random.split(step_key, config["NUM_ENVS"])
# #             action_dict = unbatchify(actions, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
# #             ob, env_state, rew, dn, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
# #                 step_keys, env_state, action_dict
# #             )

# #             r = jnp.stack([rew[a] for a in env.agents]).reshape(-1)     # (B,)
# #             d = jnp.stack([dn[a]  for a in env.agents]).reshape(-1)     # (B,)
# #             obs = batchify(ob, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])

# #             ret_sum += float(r.mean())
# #             done = d
# #             steps_count += 1

# #             if bool(d.any()):  # 单环境下 episode 结束
# #                 break

# #         # 本 episode 汇总
# #         pmax_arr    = jnp.stack(pmax_list)       # (T,4)
# #         margin_arr  = jnp.stack(margin_list)     # (T,4)
# #         entropy_arr = jnp.stack(entropy_list)    # (T,4)
# #         ge09_arr    = jnp.stack(pmax_ge09_list)  # (T,4)
# #         change_rate = change_count / max(steps_count - 1, 1)  # (4,)

# #         ep_returns.append(ret_sum)
# #         ep_lengths.append(steps_count)
# #         ep_pmax_means.append(np.array(pmax_arr.mean(axis=0)))
# #         ep_margin_means.append(np.array(margin_arr.mean(axis=0)))
# #         ep_entropy_means.append(np.array(entropy_arr.mean(axis=0)))
# #         ep_change_rates.append(np.array(change_rate))
# #         ep_pmax_ge09.append(np.array(ge09_arr.mean(axis=0)))

# #         print(f"[Episode {ep+1}/{NUM_EPISODES}] steps={steps_count}  return(sum)={ret_sum:.4f}")

# #     # 跨 episode 汇总（只统计跑满 STEPS_LIMIT 的）
# #     ep_returns = np.array(ep_returns)               # (E,)
# #     ep_lengths = np.array(ep_lengths)               # (E,)
# #     ep_pmax_means    = np.stack(ep_pmax_means)      # (E,4)
# #     ep_margin_means  = np.stack(ep_margin_means)    # (E,4)
# #     ep_entropy_means = np.stack(ep_entropy_means)   # (E,4)
# #     ep_change_rates  = np.stack(ep_change_rates)    # (E,4)
# #     ep_pmax_ge09     = np.stack(ep_pmax_ge09)       # (E,4)

# #     if ONLY_FULL_EPISODES:
# #         mask = (ep_lengths == STEPS_LIMIT)
# #         dropped = np.where(~mask)[0]
# #         if dropped.size > 0:
# #             print(f"[Info] Dropped episodes (not full length): {dropped.tolist()}")
# #         used = mask
# #         used_count = int(mask.sum())
# #         if used_count == 0:
# #             print("[Warn] 没有任何 episode 跑满 STEPS_LIMIT，改为使用全部 episode 做统计（避免空结果）。")
# #             used = np.ones_like(mask, dtype=bool)
# #     else:
# #         used = np.ones_like(ep_lengths, dtype=bool)
# #         used_count = int(used.sum())

# #     used_returns = ep_returns[used]
# #     used_lengths = ep_lengths[used]

# #     result = {
# #         "episodes_total": int(len(ep_returns)),
# #         "episodes_used":  used_count,
# #         "used_full_length_only": bool(ONLY_FULL_EPISODES),

# #         "return_sum_mean": float(used_returns.mean()),
# #         "return_sum_std":  float(used_returns.std()),
# #         "length_mean":     float(used_lengths.mean()),
# #         "length_std":      float(used_lengths.std()),

# #         "pmax_mean_per_head":           ep_pmax_means[used].mean(axis=0),
# #         "pmax_ge_0.9_per_head":         ep_pmax_ge09[used].mean(axis=0),
# #         "margin_mean_per_head":         ep_margin_means[used].mean(axis=0),
# #         "entropy_mean_per_head":        ep_entropy_means[used].mean(axis=0),
# #         "mode_change_rate_per_head":    ep_change_rates[used].mean(axis=0),
# #     }
# #     return result


# # def main():
# #     report = evaluate_checkpoint()
# #     print("\n=== Policy Evaluation Report ===")
# #     print(f"Episodes used / total: {report['episodes_used']} / {report['episodes_total']}"
# #           + ("  (full-length only)" if report.get("used_full_length_only") else ""))
# #     print(f"Return(sum) mean ± std: {report['return_sum_mean']:.4f} ± {report['return_sum_std']:.4f}")
# #     print(f"Length mean ± std:      {report['length_mean']:.2f} ± {report['length_std']:.2f}")
# #     print("pmax_mean_per_head [throttle, elevator, aileron, rudder]:", report["pmax_mean_per_head"])
# #     print("pmax>=0.9 fraction per head:", report["pmax_ge_0.9_per_head"])
# #     print("margin_mean_per_head:", report["margin_mean_per_head"])
# #     print("entropy_mean_per_head:", report["entropy_mean_per_head"])
# #     print("mode_change_rate_per_head:", report["mode_change_rate_per_head"])

# #     # 额外打印：每个指标的理论范围 + 归一化分数
# #     print_head_metrics_with_ranges(report, ACTION_DIMS, HEAD_LABELS)


# # if __name__ == "__main__":
# #     main()


# # # # baseline_evaluate.py
# # # # 直接运行：python baseline_evaluate.py
# # # # 改参数：见“用户可改参数”一节

# # # import os
# # # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
# # # from typing import Sequence, Dict, Any, Tuple

# # # import jax
# # # import jax.numpy as jnp
# # # import numpy as np
# # # import flax.linen as nn
# # # from flax.linen.initializers import constant, orthogonal
# # # import distrax
# # # import optax
# # # from flax.training.train_state import TrainState
# # # import orbax.checkpoint as ocp
# # # import functools

# # # from envs.wrappers import LogWrapper
# # # from envs.aeroplanax_heading_pitch_V import (
# # #     AeroPlanaxHeading_Pitch_V_Env,
# # #     Heading_Pitch_V_TaskParams,
# # # )

# # # # ======================
# # # # 用户可改参数（无需命令行）
# # # # ======================
# # # CKPT_PATH     = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline(no_fc2_no_layer_norm)/checkpoints/checkpoint_epoch_1000"
# # # NUM_EPISODES  = 15
# # # STEPS_LIMIT   = 1000
# # # SEED          = 42
# # # GREEDY_ACTION = True   # True=贪心 mode()；False=按分布采样。把 GREEDY_ACTION = False（采样）。这会让“是否跑满 2000 步”高度依赖随机性，导致很多 episode 被丢弃。评估期通常设为：True
# # # NUM_ENVS      = 1      # 建议评估用1，后续想并行可以改大（代码已兼容）
# # # ONLY_FULL_EPISODES = True  # 只统计跑满 STEPS_LIMIT 的 episode；其余丢弃

# # # # ======================


# # # # ==============
# # # # 网络定义（与训练一致：GRU + scan）
# # # # ==============
# # # class ScannedRNN(nn.Module):
# # #     @functools.partial(
# # #         nn.scan,
# # #         variable_broadcast="params",
# # #         in_axes=0,
# # #         out_axes=0,
# # #         split_rngs={"params": False},
# # #     )
# # #     @nn.compact
# # #     def __call__(self, carry, x):
# # #         # carry: (B, H)
# # #         # x: (ins, resets) 其中 ins: (B, D)，resets: (B,)
# # #         rnn_state = carry
# # #         ins, resets = x
# # #         rnn_state = jnp.where(
# # #             resets[:, jnp.newaxis],                     # (B,1)
# # #             self.initialize_carry(*rnn_state.shape),    # (B,H)
# # #             rnn_state
# # #         )
# # #         new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
# # #         return new_rnn_state, y

# # #     @staticmethod
# # #     def initialize_carry(batch_size, hidden_size):
# # #         cell = nn.GRUCell(features=hidden_size)
# # #         return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# # # class ActorCriticRNN(nn.Module):
# # #     action_dim: Sequence[int]
# # #     config: Dict

# # #     @nn.compact
# # #     def __call__(self, hidden, x):
# # #         # x: (obs, dones)  obs: (T,B,ObsDim)  dones: (T,B)
# # #         act = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
# # #         obs, dones = x

# # #         # 前端 MLP（与训练相同）
# # #         embedding = nn.Dense(
# # #             self.config["FC_DIM_SIZE"],
# # #             kernel_init=orthogonal(np.sqrt(2)),
# # #             bias_init=constant(0.0),
# # #         )(obs)
# # #         embedding = act(embedding)

# # #         # GRU（时间维T在最前，使用scan）
# # #         hidden, embedding = ScannedRNN()(hidden, (embedding, dones))  # hidden: (B,H); embedding: (T,B,H)

# # #         # 策略头（四个离散动作头）
# # #         actor_mean = nn.Dense(
# # #             self.config["GRU_HIDDEN_DIM"],
# # #             kernel_init=orthogonal(2),
# # #             bias_init=constant(0.0),
# # #         )(embedding)
# # #         actor_mean = act(actor_mean)

# # #         def head(n):
# # #             return nn.Dense(n, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

# # #         pi_throttle = distrax.Categorical(logits=head(self.action_dim[0]))
# # #         pi_elevator = distrax.Categorical(logits=head(self.action_dim[1]))
# # #         pi_aileron  = distrax.Categorical(logits=head(self.action_dim[2]))
# # #         pi_rudder   = distrax.Categorical(logits=head(self.action_dim[3]))

# # #         # 价值头
# # #         critic = nn.Dense(
# # #             self.config["FC_DIM_SIZE"],
# # #             kernel_init=orthogonal(2),
# # #             bias_init=constant(0.0),
# # #         )(embedding)
# # #         critic = act(critic)
# # #         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)  # (T,B,1)

# # #         return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)  # (T,B)


# # # # =========
# # # # 与训练一致的打包函数
# # # # =========
# # # def batchify(x: dict, agent_list, num_envs, num_actors):
# # #     x = jnp.stack([x[a] for a in agent_list])         # (num_actors, num_envs, dim)
# # #     return x.reshape((num_actors * num_envs, -1))     # (B, dim)

# # # def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
# # #     x = x.reshape((num_actors, num_envs, -1))         # (num_actors, num_envs, dim)
# # #     return {a: x[i] for i, a in enumerate(agent_list)}


# # # # =========
# # # # 评估函数
# # # # =========
# # # def evaluate_checkpoint() -> Dict[str, Any]:
# # #     # —— 与训练关键超参一致（影响参数树/初始化的部分一定要匹配）
# # #     config = {
# # #         "SEED": SEED,
# # #         "NUM_ENVS": NUM_ENVS,
# # #         "NUM_ACTORS": 1,        # 本任务单智能体
# # #         "FC_DIM_SIZE": 128,
# # #         "GRU_HIDDEN_DIM": 128,
# # #         "MAX_GRAD_NORM": 2,
# # #         "ACTIVATION": "relu",
# # #         "LR": 3e-4,             # 仅用于构建 TrainState 作为 restore 的 target
# # #     }
# # #     action_dims = [31, 41, 41, 41]  # 与训练时一致

# # #     # 环境
# # #     env_params = Heading_Pitch_V_TaskParams()
# # #     env = LogWrapper(AeroPlanaxHeading_Pitch_V_Env(env_params))

# # #     # 构图并初始化参数（shape 要与训练完全一致）
# # #     net = ActorCriticRNN(action_dims, config=config)
# # #     rng = jax.random.PRNGKey(config["SEED"])
# # #     obs_dim = env.observation_space(env.agents[0], env_params).shape

# # #     init_x = (
# # #         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], *obs_dim)),  # (T=1,B,ObsDim)
# # #         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),            # (T=1,B)
# # #     )
# # #     init_h = ScannedRNN.initialize_carry(
# # #         config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
# # #     )
# # #     params = net.init(rng, init_h, init_x)

# # #     # 用 target 恢复，避免不安全警告
# # #     tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
# # #                      optax.adam(config["LR"], eps=1e-5))
# # #     ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
# # #     state_item = {"params": ts.params, "opt_state": ts.opt_state, "epoch": jnp.array(0)}

# # #     ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
# # #     restored = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore(item=state_item))
# # #     params = restored["params"]
# # #     restored_epoch = int(restored.get("epoch", jnp.array(-1)))
# # #     print(f"[Evaluate] Restored epoch: {restored_epoch}")

# # #     rng, _rng = jax.random.split(rng)

# # #     # 跨 episode 统计
# # #     ep_returns, ep_lengths = [], []
# # #     ep_pmax_means, ep_margin_means, ep_entropy_means, ep_change_rates, ep_pmax_ge09 = [], [], [], [], []

# # #     for ep in range(NUM_EPISODES):
# # #         # reset
# # #         rng, ep_key = jax.random.split(rng)                       # ← 新加
# # #         reset_keys = jax.random.split(ep_key, config["NUM_ENVS"])
# # #         obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
# # #         obs = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])  # (B,ObsDim)
# # #         done = jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool)
# # #         h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

# # #         # per-episode 累积
# # #         ret_sum = 0.0
# # #         steps_count = 0

# # #         pmax_list, margin_list, entropy_list, pmax_ge09_list = [], [], [], []
# # #         change_count = jnp.zeros((4,), dtype=jnp.float32)
# # #         prev_modes = None

# # #         for t in range(STEPS_LIMIT):
# # #             ac_in = (obs[None, :], done[None, :])           # (T=1,B,Obs) & (1,B)
# # #             h, pis, value = net.apply(params, h, ac_in)
# # #             pi_th, pi_el, pi_ai, pi_ru = pis

# # #             # 指标
# # #             def head_metrics(pi):
# # #                 probs = jax.nn.softmax(pi.logits, axis=-1)  # (1,B,A)
# # #                 probs = jnp.clip(probs, 1e-9, 1.0)
# # #                 pmax = probs.max(axis=-1)                   # (1,B)
# # #                 top2 = jnp.sort(probs, axis=-1)[..., -2:]   # (1,B,2)
# # #                 margin = top2[..., 1] - top2[..., 0]        # (1,B)
# # #                 ent = pi.entropy()                          # (1,B)
# # #                 ge09 = (pmax >= 0.9).astype(jnp.float32)    # (1,B)
# # #                 return pmax.mean(), margin.mean(), ent.mean(), ge09.mean()

# # #             m = [head_metrics(p) for p in [pi_th, pi_el, pi_ai, pi_ru]]
# # #             p_m, m_m, e_m, ge_m = zip(*m)
# # #             pmax_list.append(jnp.stack(p_m))       # (4,)
# # #             margin_list.append(jnp.stack(m_m))     # (4,)
# # #             entropy_list.append(jnp.stack(e_m))    # (4,)
# # #             pmax_ge09_list.append(jnp.stack(ge_m)) # (4,)

# # #             # 动作（贪心/采样）
# # #             if GREEDY_ACTION:
# # #                 a_th = pi_th.mode(); a_el = pi_el.mode(); a_ai = pi_ai.mode(); a_ru = pi_ru.mode()
# # #             else:
# # #                 rng, sk = jax.random.split(rng); a_th = pi_th.sample(seed=sk)
# # #                 rng, sk = jax.random.split(rng); a_el = pi_el.sample(seed=sk)
# # #                 rng, sk = jax.random.split(rng); a_ai = pi_ai.sample(seed=sk)
# # #                 rng, sk = jax.random.split(rng); a_ru = pi_ru.sample(seed=sk)

# # #             # 去掉时间维 -> (B,)
# # #             a_th = a_th.squeeze(0); a_el = a_el.squeeze(0); a_ai = a_ai.squeeze(0); a_ru = a_ru.squeeze(0)
# # #             actions = jnp.stack([a_th, a_el, a_ai, a_ru], axis=-1)  # (B,4)

# # #             # 模式变更率
# # #             if prev_modes is not None:
# # #                 change_count = change_count + jnp.mean((actions != prev_modes).astype(jnp.float32), axis=0)
# # #             prev_modes = actions

# # #             # env.step（注意：第三个参数要 unbatchify 成 {agent: (NUM_ENVS, 4)}）
# # #             rng, step_key = jax.random.split(rng)
# # #             step_keys = jax.random.split(step_key, config["NUM_ENVS"])
# # #             action_dict = unbatchify(actions, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
# # #             ob, env_state, rew, dn, info = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, env_state, action_dict)

# # #             r = jnp.stack([rew[a] for a in env.agents]).reshape(-1)     # (B,)
# # #             d = jnp.stack([dn[a]  for a in env.agents]).reshape(-1)     # (B,)
# # #             obs = batchify(ob, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])

# # #             ret_sum += float(r.mean())
# # #             done = d
# # #             steps_count += 1

# # #             if bool(d.any()):  # 单环境下 episode 结束
# # #                 break

# # #         # 本 episode 汇总
# # #         pmax_arr    = jnp.stack(pmax_list)       # (T,4)
# # #         margin_arr  = jnp.stack(margin_list)     # (T,4)
# # #         entropy_arr = jnp.stack(entropy_list)    # (T,4)
# # #         ge09_arr    = jnp.stack(pmax_ge09_list)  # (T,4)
# # #         change_rate = change_count / max(steps_count - 1, 1)  # (4,)

# # #         ep_returns.append(ret_sum)
# # #         ep_lengths.append(steps_count)
# # #         ep_pmax_means.append(np.array(pmax_arr.mean(axis=0)))
# # #         ep_margin_means.append(np.array(margin_arr.mean(axis=0)))
# # #         ep_entropy_means.append(np.array(entropy_arr.mean(axis=0)))
# # #         ep_change_rates.append(np.array(change_rate))
# # #         ep_pmax_ge09.append(np.array(ge09_arr.mean(axis=0)))

# # #         print(f"[Episode {ep+1}/{NUM_EPISODES}] steps={steps_count}  return(sum)={ret_sum:.4f}")

# # #     # 跨 episode 汇总（只统计跑满 STEPS_LIMIT 的）
# # #     ep_returns = np.array(ep_returns)               # (E,)
# # #     ep_lengths = np.array(ep_lengths)               # (E,)
# # #     ep_pmax_means    = np.stack(ep_pmax_means)      # (E,4)
# # #     ep_margin_means  = np.stack(ep_margin_means)    # (E,4)
# # #     ep_entropy_means = np.stack(ep_entropy_means)   # (E,4)
# # #     ep_change_rates  = np.stack(ep_change_rates)    # (E,4)
# # #     ep_pmax_ge09     = np.stack(ep_pmax_ge09)       # (E,4)

# # #     if ONLY_FULL_EPISODES:
# # #         mask = (ep_lengths == STEPS_LIMIT)
# # #         dropped = np.where(~mask)[0]
# # #         if dropped.size > 0:
# # #             print(f"[Info] Dropped episodes (not full length): {dropped.tolist()}")
# # #         used = mask
# # #         used_count = int(mask.sum())
# # #         if used_count == 0:
# # #             print("[Warn] 没有任何 episode 跑满 STEPS_LIMIT，改为使用全部 episode 做统计（避免空结果）。")
# # #             used = np.ones_like(mask, dtype=bool)
# # #     else:
# # #         used = np.ones_like(ep_lengths, dtype=bool)
# # #         used_count = int(used.sum())

# # #     used_returns = ep_returns[used]
# # #     used_lengths = ep_lengths[used]

# # #     result = {
# # #         "episodes_total": int(len(ep_returns)),
# # #         "episodes_used":  used_count,
# # #         "used_full_length_only": bool(ONLY_FULL_EPISODES),

# # #         "return_sum_mean": float(used_returns.mean()),
# # #         "return_sum_std":  float(used_returns.std()),
# # #         "length_mean":     float(used_lengths.mean()),
# # #         "length_std":      float(used_lengths.std()),

# # #         # "pmax_mean_per_head":           ep_pmax_means[used].mean(axis=0),
# # #         # "pmax_ge_0.9_per_head":         ep_pmax_ge09[used].mean(axis=0),
# # #         # "margin_mean_per_head":         ep_margin_means[used].mean(axis=0),
# # #         # "entropy_mean_per_head":        ep_entropy_means[used].mean(axis=0),
# # #         # "mode_change_rate_per_head":    ep_change_rates[used].mean(axis=0),

# # #         "pmax_mean_per_head":           ep_pmax_means[used].mean(axis=0),        # 每个动作头（throttle/elevator/aileron/rudder）的最大概率（p_max）平均值；越高表示分布越尖（更确定）
# # #         "pmax_ge_0.9_per_head":         ep_pmax_ge09[used].mean(axis=0),         # 每个头 p_max >= 0.9 的比例（fraction）；越高表示分布尖到90%以上的时间越多（非常确定）
# # #         "margin_mean_per_head":         ep_margin_means[used].mean(axis=0),       # 每个头 top1 - top2 概率的平均间隔；越大表示最优动作与其他动作的“自信度差距”越大（更自信）
# # #         "entropy_mean_per_head":        ep_entropy_means[used].mean(axis=0),      # 每个头的平均熵（entropy）；值越低表示分布越集中（更确定/少随机性）
# # #         "mode_change_rate_per_head":    ep_change_rates[used].mean(axis=0),       # 每个头连续步间 mode（argmax动作）变更率；值越低表示动作序列越平滑稳定（少抖动）
# # #     }
# # #     return result



# # # def main():
# # #     report = evaluate_checkpoint()
# # #     print("\n=== Policy Evaluation Report ===")
# # #     print(f"Episodes used / total: {report['episodes_used']} / {report['episodes_total']}"
# # #           + ("  (full-length only)" if report.get("used_full_length_only") else ""))
# # #     print(f"Return(sum) mean ± std: {report['return_sum_mean']:.4f} ± {report['return_sum_std']:.4f}")
# # #     print(f"Length mean ± std:      {report['length_mean']:.2f} ± {report['length_std']:.2f}")
# # #     print("pmax_mean_per_head [throttle, elevator, aileron, rudder]:", report["pmax_mean_per_head"])
# # #     print("pmax>=0.9 fraction per head:", report["pmax_ge_0.9_per_head"])
# # #     print("margin_mean_per_head:", report["margin_mean_per_head"])
# # #     print("entropy_mean_per_head:", report["entropy_mean_per_head"])
# # #     print("mode_change_rate_per_head:", report["mode_change_rate_per_head"])



# # # if __name__ == "__main__":
# # #     main()
