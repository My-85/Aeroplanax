"""
diagnose_full_domain_maneuver.py
================================
诊断 full_domain_maneuver 训练效果:
  1) 加载 checkpoint，在环境中运行多个 episode，统计成功率
  2) 分析课程学习指标 (curriculum_level) 为何下降
  3) 检查 reward / termination / observation 是否存在设计问题
  4) 给出具体改进建议

用法:
  python diagnose_full_domain_maneuver.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax.linen.initializers import constant, orthogonal
import functools
from typing import Sequence, Dict
from flax.training.train_state import TrainState
import distrax
import optax
import orbax.checkpoint as ocp

from envs.wrappers import LogWrapper
from envs.aeroplanax_full_domain_maneuver import (
    AeroPlanaxFullDomainEnv,
    FullDomain_TaskParams,
    _quat_from_euler_bn,
    _quat_normalize,
)

# ═══════════════════════════════ 模型定义 (与训练完全一致) ═══════════════════════════════

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan, variable_broadcast="params",
        in_axes=0, out_axes=0, split_rngs={"params": False},
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
        activation = nn.relu if self.config.get("ACTIVATION", "relu") == "relu" else nn.tanh
        obs, dones = x
        embedding = nn.Dense(self.config["FC_DIM_SIZE"],
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"],
                              kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)
        logit_thr = nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        logit_ele = nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        logit_ail = nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        logit_rud = nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi_thr = distrax.Categorical(logits=logit_thr)
        pi_ele = distrax.Categorical(logits=logit_ele)
        pi_ail = distrax.Categorical(logits=logit_ail)
        pi_rud = distrax.Categorical(logits=logit_rud)
        critic = nn.Dense(self.config["FC_DIM_SIZE"],
                          kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, (pi_thr, pi_ele, pi_ail, pi_rud), jnp.squeeze(critic, axis=-1)


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _f(x, i=0):
    a = np.asarray(x)
    if a.ndim == 0: return float(a)
    return float(a.reshape(-1)[min(i, a.size - 1)])


def quat_geodesic_deg(q0, q1, q2, q3, yaw_t, pitch_t, roll_t):
    q_curr = np.array([q0, q1, q2, q3])
    norm = np.linalg.norm(q_curr)
    if norm < 1e-9: return 180.0
    q_curr = q_curr / norm
    q_tgt = np.array(_quat_from_euler_bn(
        jnp.float32(roll_t), jnp.float32(pitch_t), jnp.float32(yaw_t)
    )).reshape(-1)
    cos_half = abs(np.dot(q_curr, q_tgt))
    cos_half = np.clip(cos_half, 0.0, 1.0)
    return float(2.0 * np.degrees(np.arccos(cos_half)))


# ═══════════════════════════════ 诊断运行 ═══════════════════════════════

def run_diagnostic(config):
    print("=" * 100)
    print("FULL-DOMAIN MANEUVER TRAINING DIAGNOSTIC")
    print("=" * 100)

    # -------- 环境 --------
    env_params = FullDomain_TaskParams()
    env_core = AeroPlanaxFullDomainEnv(env_params)
    env = LogWrapper(env_core)
    config["NUM_ACTORS"] = env.num_agents

    # -------- 网络 + checkpoint --------
    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    rng = jax.random.PRNGKey(config['SEED'])
    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"],
                    *env.observation_space(env.agents[0], env_params).shape)),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    init_h = ScannedRNN.initialize_carry(
        config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
    )
    params_init = network.init(rng, init_h, init_x)
    tx = optax.adam(config["LR"], eps=1e-5)
    train_state = TrainState.create(apply_fn=network.apply, params=params_init, tx=tx)

    params = params_init
    if config.get("LOADDIR"):
        state_template = {
            "params": train_state.params,
            "opt_state": train_state.opt_state,
            "epoch": jnp.array(0),
        }
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(
            config['LOADDIR'], args=ocp.args.StandardRestore(item=state_template)
        )
        print(f"[restore] epoch = {int(checkpoint['epoch'])}")
        params = checkpoint["params"]

    # ═══════════ PART 1: 统计性评估 (多 episode 贪心推理) ═══════════
    print("\n" + "=" * 100)
    print("PART 1: STATISTICAL EVALUATION (Greedy rollout)")
    print("=" * 100)

    num_eval_episodes = config.get("NUM_EVAL_EPISODES", 20)
    max_steps_per_ep = 2000  # timeout_fn 默认 400s ≈ 400*5=2000 decision steps

    all_ep_returns = []
    all_ep_lengths = []
    all_success_counts = []
    all_crash_types = []
    all_theta_history = []
    all_vt_err_history = []
    all_alt_history = []

    for ep in range(num_eval_episodes):
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        done_arr = jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool)
        obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])

        ep_return = 0.0
        ep_length = 0
        ep_thetas = []
        ep_vt_errs = []
        ep_alts = []
        ep_success_count = 0

        rng_ep = jax.random.PRNGKey(config["SEED"] + ep * 1000)

        for step in range(max_steps_per_ep):
            ac_in = (obs_batch[np.newaxis, :], done_arr[np.newaxis, :])
            hstate, pi, value = network.apply(params, hstate, ac_in)
            pi_thr, pi_ele, pi_ail, pi_rud = pi

            a_thr = pi_thr.mode()
            a_ele = pi_ele.mode()
            a_ail = pi_ail.mode()
            a_rud = pi_rud.mode()

            action = jnp.concatenate(
                [a_thr[:, :, np.newaxis], a_ele[:, :, np.newaxis],
                 a_ail[:, :, np.newaxis], a_rud[:, :, np.newaxis]], axis=-1
            ).squeeze(0)

            rng_ep, _rng = jax.random.split(rng_ep)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state,
                unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
            )

            core = env_state.env_state
            r = float(batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1).sum())
            ep_return += r
            ep_length = step + 1

            # 姿态误差
            q0_ = _f(core.plane_state.q0)
            q1_ = _f(core.plane_state.q1)
            q2_ = _f(core.plane_state.q2)
            q3_ = _f(core.plane_state.q3)
            hdg_t = _f(core.target_heading)
            pit_t = _f(core.target_pitch)
            rol_t = _f(core.target_roll)
            vt_t = _f(core.target_vt)
            vt_now = _f(core.plane_state.vt)
            alt_now = _f(core.plane_state.altitude)

            theta = quat_geodesic_deg(q0_, q1_, q2_, q3_, hdg_t, pit_t, rol_t)
            ep_thetas.append(theta)
            ep_vt_errs.append(abs(vt_now - vt_t))
            ep_alts.append(alt_now)

            # 成功次数
            sc = int(np.asarray(core.heading_turn_counts).reshape(-1)[0])
            if sc > ep_success_count:
                ep_success_count = sc

            obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
            done_arr = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)

            if bool(np.asarray(done_arr).any()):
                # 判断崩溃类型
                status = int(np.asarray(core.plane_state.status).reshape(-1)[0])
                if status == 2:
                    all_crash_types.append("crashed")
                elif status == 4:
                    all_crash_types.append("success")
                else:
                    all_crash_types.append(f"status_{status}")
                break
        else:
            all_crash_types.append("timeout")

        all_ep_returns.append(ep_return)
        all_ep_lengths.append(ep_length)
        all_success_counts.append(ep_success_count)
        all_theta_history.append(ep_thetas)
        all_vt_err_history.append(ep_vt_errs)
        all_alt_history.append(ep_alts)

        print(f"  EP {ep:2d}: steps={ep_length:5d}  R={ep_return:+8.1f}  "
              f"success={ep_success_count:3d}  "
              f"mean_θ={np.mean(ep_thetas):6.1f}°  "
              f"mean_Δvt={np.mean(ep_vt_errs):5.1f}  "
              f"end={all_crash_types[-1]}")

    # -------- 汇总 --------
    print(f"\n{'='*60}")
    print(f"SUMMARY ({num_eval_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Return:       mean={np.mean(all_ep_returns):+.1f}  std={np.std(all_ep_returns):.1f}")
    print(f"  Length:        mean={np.mean(all_ep_lengths):.0f}  std={np.std(all_ep_lengths):.0f}")
    print(f"  Success count: mean={np.mean(all_success_counts):.1f}  max={np.max(all_success_counts)}")
    crash_counts = {}
    for ct in all_crash_types:
        crash_counts[ct] = crash_counts.get(ct, 0) + 1
    print(f"  Termination types: {crash_counts}")

    # ═══════════ PART 2: 静态代码审查 ═══════════
    print("\n" + "=" * 100)
    print("PART 2: CODE REVIEW & ISSUE ANALYSIS")
    print("=" * 100)

    issues = []
    suggestions = []

    # --- Issue 1: curriculum_level 震荡下降 ---
    # 原因分析: _step_task 中 unreach_full_domain_fn 的 success 条件包含 timeout auto-switch
    # (elapsed >= max_interval_sec=20s 则 success=True)。但 _step_task 中的
    # _check_real_success 只在真正达标时才推进 curriculum。然而 heading_turn_counts
    # 在每次 success（包括 timeout）时都 +1，而 curriculum 只在 real success 时才 +1。
    # 这说明: 大多数 "success" 其实是 20s timeout auto-switch，不是真正的姿态到位。
    #
    # 但更关键的问题: curriculum_level 是 PER-EPISODE state，不是全局 state!
    # 每次 episode 结束 (crashed / 400s timeout)，环境 auto-reset，
    # curriculum_level 被重置为 0! 这意味着:
    #   - curriculum 无法在 episode 之间积累
    #   - 每个 episode 都从 level 0 开始
    #   - "先升后降" 是因为 episode 内部先快速积累，然后 reset 归零

    issues.append(
        "CRITICAL: curriculum_level 存储在 FullDomain_TaskState 中，是 per-episode state。\n"
        "  每次 episode 结束 (crashed/timeout/400s max) 后 auto-reset 会将其重置为 0。\n"
        "  这意味着 curriculum 无法跨 episode 积累，每个 episode 都从 level 0 重新开始。\n"
        "  训练日志中看到的 curriculum_level 变化只是 episode 内部的短期波动。"
    )

    suggestions.append(
        "FIX: 将 curriculum_level 和 curriculum_success_counts 从 TaskState 移到\n"
        "  全局训练循环中 (如 runner_state 的一部分)，或在 _reset_task 中保留\n"
        "  curriculum_level 不重置 (需修改 FullDomain_TaskState.create 使其接受\n"
        "  上一个 episode 的 curriculum_level)。"
    )

    # --- Issue 2: success 条件过严 ---
    issues.append(
        "SUCCESS条件过严: 需要同时满足 θ<=10° (全姿态四元数误差) 且 |Δvt|<=15m/s，\n"
        "  且至少等待 8s。对于 full-domain 姿态控制 (包含 roll 维度)，\n"
        "  10° 是非常严格的阈值，特别是在 roll 方向。"
    )

    suggestions.append(
        "SUGGESTION: 考虑将 theta_tol_deg 放宽到 15° 或 20° (至少在低 curriculum level)，\n"
        "  或者分离 heading/pitch 和 roll 的容差 (roll 容差可以更宽松)。"
    )

    # --- Issue 3: 每 episode 400s timeout 太短 ---
    # timeout_fn 默认 max_steps=400 (即 400 秒)
    # 每次 target switch 需 8-20s，所以 400s 最多 20-50 次 switch
    # 但 curriculum advance 需要 5+level*3 次 real success
    # level 0 需 5 次, level 1 需 8 次, level 7 需 26 次
    # 如果只有部分 switch 是 real success，很难在一个 episode 内积累足够
    ep_decisions_per_sec = env_params.sim_freq / env_params.agent_interaction_steps  # 50/10=5
    ep_timeout_sec = 400.0
    ep_max_decisions = ep_timeout_sec * ep_decisions_per_sec  # 2000
    switch_interval_sec = 20.0  # max_interval_sec
    max_switches = ep_timeout_sec / switch_interval_sec  # 20
    min_switches_fast = ep_timeout_sec / 8.0  # 50 (if all succeed in 8s)

    issues.append(
        f"TIMING: 每个 episode 最长 {ep_timeout_sec:.0f}s ({ep_max_decisions:.0f} decisions)。\n"
        f"  每次 target switch 间隔 8-20s → 一个 episode 最多 {max_switches:.0f}-{min_switches_fast:.0f} 次 switch。\n"
        f"  但 curriculum 从 level 0→1 需要 5 次 real success，level 6→7 需要 26 次。\n"
        f"  如果 real success rate < 50%，一个 episode 内很难推进 curriculum。\n"
        f"  加上 curriculum_level 每 episode reset (Issue #1)，问题更加严重。"
    )

    # --- Issue 4: episodic_return 上升但 success_times 低 ---
    issues.append(
        "REWARD 分析: episodic_return=720 上升说明策略学到了 '存活' 和 '部分靠近目标'。\n"
        "  reward 最大值 = 1.05/step * 2000 steps = 2100，当前 720 ≈ 0.36/step。\n"
        "  success_times=13 意味着一个 episode 内只有 ~13 次 target switch，\n"
        "  其中大部分可能是 20s timeout auto-switch (非 real success)。\n"
        "  策略可能在学习 '保持飞行稳定获取 alive bonus + 部分跟踪奖励'，\n"
        "  而不是 '快速精确到达目标姿态'。"
    )

    suggestions.append(
        "SUGGESTION: 增加 real success 的额外奖励 (bonus on success)，\n"
        "  或在 timeout auto-switch 时给予轻微惩罚，以区分 real vs timeout。"
    )

    # --- Issue 5: 40% random mode targets 可能太激进 ---
    issues.append(
        "TARGET GENERATION: 40% random mode 直接从全范围采样目标姿态，\n"
        "  在低 curriculum level (scale=0.15-0.27) 时 random mode 的 pitch 范围\n"
        "  为 [-90°*scale, 90°*scale]，但 random heading 是全范围 [0,360°]，\n"
        "  这可能导致突然的大角度转弯让策略难以学习。"
    )

    suggestions.append(
        "SUGGESTION: 在低 curriculum level 时减少 random mode 比例 (如 level<3 时 80% delta)，\n"
        "  或让 random mode 的范围也受 scale 约束 (heading 也乘 scale)。"
    )

    # --- Issue 6: observation 空间缺少时间信息 ---
    issues.append(
        "OBS 空间 (22D) 缺少: 距上次 target switch 的时间信息。\n"
        "  策略无法知道 '还有多久会 timeout auto-switch'，\n"
        "  这可能影响策略在时间压力下的行为决策。"
    )

    # --- 打印诊断报告 ---
    print("\n--- ISSUES FOUND ---")
    for i, issue in enumerate(issues, 1):
        print(f"\n  [{i}] {issue}")

    print("\n--- SUGGESTIONS ---")
    for i, sug in enumerate(suggestions, 1):
        print(f"\n  [{i}] {sug}")

    # ═══════════ PART 3: 生成诊断图表 ═══════════
    print("\n" + "=" * 100)
    print("PART 3: GENERATING DIAGNOSTIC PLOTS")
    print("=" * 100)

    output_dir = config.get("DIAG_OUTPUT", "./diag_full_domain/")
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: 每 episode 姿态误差分布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Episode returns
    ax = axes[0, 0]
    ax.bar(range(num_eval_episodes), all_ep_returns, alpha=0.7)
    ax.axhline(np.mean(all_ep_returns), color='r', linestyle='--', label=f'mean={np.mean(all_ep_returns):.0f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Episode Returns')
    ax.legend()

    # (b) Episode lengths
    ax = axes[0, 1]
    ax.bar(range(num_eval_episodes), all_ep_lengths, alpha=0.7, color='orange')
    ax.axhline(np.mean(all_ep_lengths), color='r', linestyle='--', label=f'mean={np.mean(all_ep_lengths):.0f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length (steps)')
    ax.set_title('Episode Lengths')
    ax.legend()

    # (c) Success counts
    ax = axes[1, 0]
    ax.bar(range(num_eval_episodes), all_success_counts, alpha=0.7, color='green')
    ax.axhline(np.mean(all_success_counts), color='r', linestyle='--', label=f'mean={np.mean(all_success_counts):.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Count')
    ax.set_title('Target Switch Count per Episode')
    ax.legend()

    # (d) Termination types
    ax = axes[1, 1]
    types = list(crash_counts.keys())
    counts = [crash_counts[t] for t in types]
    ax.bar(types, counts, alpha=0.7, color='red')
    ax.set_ylabel('Count')
    ax.set_title('Episode Termination Types')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episode_summary.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/episode_summary.png")

    # Plot 2: 典型 episode 的姿态误差 + 速度误差时间线
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    # 选择一个有代表性的 episode (最长的)
    best_ep = int(np.argmax(all_ep_lengths))

    ax = axes[0]
    ax.plot(all_theta_history[best_ep], linewidth=0.5, alpha=0.8)
    ax.axhline(10.0, color='g', linestyle='--', linewidth=1, label='success threshold (10°)')
    ax.set_ylabel('Quaternion Error (deg)')
    ax.set_title(f'Episode {best_ep}: Attitude Error Over Time')
    ax.legend()
    ax.set_ylim(0, 180)

    ax = axes[1]
    ax.plot(all_vt_err_history[best_ep], linewidth=0.5, alpha=0.8, color='orange')
    ax.axhline(15.0, color='g', linestyle='--', linewidth=1, label='success threshold (15 m/s)')
    ax.set_ylabel('Speed Error (m/s)')
    ax.set_title(f'Episode {best_ep}: Speed Error Over Time')
    ax.legend()

    ax = axes[2]
    ax.plot(all_alt_history[best_ep], linewidth=0.5, alpha=0.8, color='purple')
    ax.axhline(500, color='r', linestyle='--', linewidth=1, label='crash floor (500m)')
    ax.axhline(2500, color='orange', linestyle='--', linewidth=1, label='safe altitude (2500m)')
    ax.set_ylabel('Altitude (m)')
    ax.set_xlabel('Step')
    ax.set_title(f'Episode {best_ep}: Altitude Over Time')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episode_detail.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/episode_detail.png")

    # Plot 3: 所有 episode 的姿态误差分布 (histogram)
    all_thetas_flat = []
    for h in all_theta_history:
        all_thetas_flat.extend(h)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_thetas_flat, bins=100, range=(0, 180), alpha=0.7)
    ax.axvline(10.0, color='r', linestyle='--', label='success threshold (10°)')
    ax.set_xlabel('Quaternion Error (deg)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Attitude Error Across All Steps')
    below_10 = sum(1 for t in all_thetas_flat if t <= 10.0)
    below_20 = sum(1 for t in all_thetas_flat if t <= 20.0)
    total = len(all_thetas_flat)
    ax.legend([f'<10°: {below_10/total*100:.1f}%  <20°: {below_20/total*100:.1f}%'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'theta_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/theta_distribution.png")

    # ═══════════ PART 4: 综合结论 ═══════════
    print("\n" + "=" * 100)
    print("PART 4: CONCLUSION & RECOMMENDATIONS")
    print("=" * 100)

    mean_theta = np.mean(all_thetas_flat) if all_thetas_flat else 180.0
    pct_below_10 = below_10 / max(total, 1) * 100
    pct_below_20 = below_20 / max(total, 1) * 100

    print(f"""
  训练指标汇总:
    - 平均姿态误差: {mean_theta:.1f}°
    - 姿态误差 <10°: {pct_below_10:.1f}%
    - 姿态误差 <20°: {pct_below_20:.1f}%
    - 平均 episode 回报: {np.mean(all_ep_returns):.1f}
    - 平均 success count: {np.mean(all_success_counts):.1f}

  核心问题:
    1. [CRITICAL] curriculum_level 每 episode reset 归零，导致无法跨 episode
       积累课程进度。修复方法: 在 _reset_task 中保持 curriculum_level 不重置，
       或将其作为全局状态。
    2. [HIGH] 大多数 target switch 是 20s timeout auto-switch (非真实达标)，
       导致 success_times 虚高但 curriculum 不进步。
    3. [MEDIUM] 40% random mode 在低 level 时目标差异过大。
    4. [LOW] 缺少 real success bonus 和 timeout penalty 的区分。

  改进优先级:
    P0: 修复 curriculum_level 持久化 (跨 episode 保持)
    P1: 增加 real success bonus (如 +5.0 per real success)
    P2: 放宽低 level 的 success 容差 (如 level<3 时 θ<=20°)
    P3: 低 level 时增加 delta mode 比例 (如 90% delta)
    P4: 在 obs 中增加时间信息 (距 target switch 的时间)
""")

    print(f"诊断图表已保存到: {output_dir}")
    print("=" * 100)


# ═══════════════════════════════ 入口 ═══════════════════════════════

if __name__ == "__main__":
    config = {
        "SEED": 42,
        "LR": 2e-4,
        "NUM_ENVS": 1,
        "NUM_ACTORS": 1,
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "UPDATE_EPOCHS": 16,
        "NUM_MINIBATCHES": 5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 2e-3,
        "VF_COEF": 1,
        "MAX_GRAD_NORM": 2,
        "ACTIVATION": "relu",
        "ANNEAL_LR": False,
        "NUM_EVAL_EPISODES": 20,
        "DIAG_OUTPUT": "./diag_full_domain/",
        "LOADDIR": os.path.abspath(
            "results/full_domain_maneuver_2026-03-04-23-59/checkpoints/checkpoint_epoch_2000"
        ),
    }
    run_diagnostic(config)
