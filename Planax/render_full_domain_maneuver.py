"""
render_full_domain_maneuver.py
==============================
加载 full_domain_maneuver baseline，在预定义航点序列上逐级增加难度地
测试姿态控制效果。每个航点定义为目标 heading / pitch / roll / vt，
飞机到达后自动切换下一个航点。

输出:
  - ACMI (.txt.acmi) 文件，可在 Tacview 中回放
  - 终端逐步打印飞行状态 (roll/pitch/yaw/vt/误差角/航点序号)

用法:
  python render_full_domain_maneuver.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
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
from envs.utils.utils import enu_to_geodetic

# ═══════════════════════════════ 预定义航点 ═══════════════════════════════
# 每个航点: (target_heading_deg, target_pitch_deg, target_roll_deg, target_vt_m_s)
# 难度从 Level-1 (简单水平转弯) → Level-5 (全姿态域极端机动)

WAYPOINTS = [
    # ---- Level 1: 简单水平转弯 ----
    (  0.0,   0.0,   0.0, 200.0),   # WP0: 直飞
    ( 30.0,   0.0,   0.0, 200.0),   # WP1: 小幅右转 30°
    (-30.0,   0.0,   0.0, 200.0),   # WP2: 小幅左转 30°

    # ---- Level 2: 中等转弯 + 轻微俯仰 ----
    ( 90.0,  10.0,   0.0, 220.0),   # WP3: 右转 90° + 小爬升
    (-90.0, -10.0,   0.0, 220.0),   # WP4: 左转 90° + 小俯冲

    # ---- Level 3: 大角度转弯 + 横滚 ----
    (180.0,   0.0,  30.0, 250.0),   # WP5: 掉头 + 30° 横滚
    (  0.0,  20.0, -45.0, 250.0),   # WP6: 爬升 20° + 左倾 45°

    # ---- Level 4: 激进俯仰 + 大横滚 ----
    ( 45.0,  45.0,  60.0, 280.0),   # WP7: 大爬升 + 大横滚
    (-45.0, -30.0, -60.0, 200.0),   # WP8: 俯冲 + 左横滚

    # ---- Level 5: 全姿态域极端机动 ----
    (  0.0,  70.0,  90.0, 300.0),   # WP9: 近垂直拉起 + 90° 横滚
    (180.0, -50.0, 120.0, 250.0),   # WP10: 大俯冲 + 超大横滚
    ( 90.0,   0.0, 180.0, 250.0),   # WP11: 倒飞 (roll=180°)
]


# ═══════════════════════════════ 工具函数 ═══════════════════════════════

def _f(x, i=0):
    """安全取标量 float"""
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    return float(a.reshape(-1)[min(i, a.size - 1)])


def quat_geodesic_deg(q_curr, yaw_t, pitch_t, roll_t):
    """计算四元数测地距离 (角度)"""
    q_curr = np.asarray(q_curr).reshape(-1)[:4]
    norm = np.linalg.norm(q_curr)
    if norm < 1e-9:
        return 180.0
    q_curr = q_curr / norm
    q_tgt = np.array(_quat_from_euler_bn(
        jnp.float32(roll_t), jnp.float32(pitch_t), jnp.float32(yaw_t)
    )).reshape(-1)
    cos_half = abs(np.dot(q_curr, q_tgt))
    cos_half = np.clip(cos_half, 0.0, 1.0)
    return float(2.0 * np.degrees(np.arccos(cos_half)))


# ═══════════════════════════════ 模型定义 ═══════════════════════════════
# 必须与 train_full_domain_maneuver.py 完全一致

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


# ═══════════════════════════════ 辅助 ═══════════════════════════════

def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


# ═══════════════════════════════ 主渲染 ═══════════════════════════════

def render_episode(config):
    # ---------- 1) 环境 ----------
    env_params = FullDomain_TaskParams()
    env_core = AeroPlanaxFullDomainEnv(env_params)
    env = LogWrapper(env_core)
    config["NUM_ACTORS"] = env.num_agents
    assert config["NUM_ENVS"] == 1

    # ---------- 2) 网络 ----------
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
    params = network.init(rng, init_h, init_x)

    tx = optax.adam(config["LR"], eps=1e-5)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # ---------- 3) 加载 checkpoint ----------
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

    # ---------- 4) 预定义航点 → 弧度 ----------
    waypoints_rad = []
    for hdg_d, pit_d, rol_d, vt in WAYPOINTS:
        waypoints_rad.append((
            np.radians(hdg_d),
            np.radians(pit_d),
            np.radians(rol_d),
            float(vt),
        ))
    num_wps = len(waypoints_rad)

    # ---------- 5) Reset ----------
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

    # 覆盖初始目标为第一个航点
    wp_idx = 0
    hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
    core_state = env_state.env_state
    core_state = core_state.replace(
        target_heading=jnp.full_like(core_state.target_heading, hdg_t),
        target_pitch=jnp.full_like(core_state.target_pitch, pit_t),
        target_roll=jnp.full_like(core_state.target_roll, rol_t),
        target_vt=jnp.full_like(core_state.target_vt, vt_t),
    )
    env_state = env_state.replace(env_state=core_state)

    # ---------- 6) 渲染初始帧 + 写航点到 ACMI ----------
    logdir = config.get("LOGDIR", "./tracks/")
    os.makedirs(logdir, exist_ok=True)
    try:
        env_core.render(core_state, env_params, {'__all__': False}, logdir)
    except Exception:
        pass

    # 把预定义航点作为 Waypoint 标记写入 ACMI
    # 航点位置: 以飞机初始位置为基准，在 NED 空间中扇形展开 (仅作视觉参考)
    init_north = _f(core_state.plane_state.north)
    init_east = _f(core_state.plane_state.east)
    init_alt = _f(core_state.plane_state.altitude)

    try:
        with open(env_core.filename, mode='a', encoding='utf-8') as f:
            for k, (hdg_d, pit_d, rol_d, vt) in enumerate(WAYPOINTS):
                # 用 heading 和 pitch 计算一个参考位置 (距初始点 2km)
                hdg_r = np.radians(hdg_d)
                pit_r = np.radians(pit_d)
                dist = 2000.0 * (k + 1)  # 每个航点远 2km
                wp_n = init_north + dist * np.cos(hdg_r) * np.cos(pit_r)
                wp_e = init_east + dist * np.sin(hdg_r) * np.cos(pit_r)
                wp_a = init_alt + dist * np.sin(pit_r)
                wp_a = max(wp_a, 1000.0)  # 航点标记最低 1km
                lat, lon, alt = enu_to_geodetic(wp_e, wp_n, wp_a, 0, 0, 0)
                oid = env_core.ID_BASE_WP + k
                f.write(
                    f"{oid},"
                    f"Type=Navaid+Static+Waypoint,Name=WP_{k},Label=L{k//3+1}_WP{k},Color=Yellow,"
                    f"T={float(lon)}|{float(lat)}|{float(alt)}|0|0|0\n"
                )
    except Exception as e:
        print(f"[warn] 写航点标记失败: {e}")

    # ---------- 7) RNN 状态 ----------
    init_h = ScannedRNN.initialize_carry(
        config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
    )

    # ---------- 8) 仿真循环 ----------
    hstate = init_h
    done_arr = jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool)
    obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])

    max_steps = config.get("MAX_STEPS", 3000)
    success_theta_deg = 10.0    # 姿态误差 ≤10° 判为到达
    success_vt_tol = 15.0       # 速度误差 ≤15 m/s
    wp_hold_steps = 0           # 在当前航点保持的步数
    min_hold = 40               # 至少保持 40 步 (约 8 秒 at 5Hz decision)
    cumulative_reward = 0.0

    print(f"\n{'='*90}")
    print(f"Full-domain maneuver render: {num_wps} waypoints, max {max_steps} steps")
    print(f"{'='*90}")
    print(f"{'Step':>6} | {'WP':>3} | {'θ_err°':>7} | {'Δvt':>6} | {'Roll°':>7} | {'Pitch°':>7} | {'Yaw°':>7} | {'Vt':>6} | {'Alt':>7} | {'R':>7}")
    print(f"{'-'*90}")

    rng_run = jax.random.PRNGKey(config["SEED"] + 100)

    for step in range(max_steps):
        # --- 前向推理 (贪心) ---
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
        )
        action = action.squeeze(0)

        # --- env step ---
        rng_run, _rng = jax.random.split(rng_run)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
            rng_step, env_state,
            unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        )

        # 提取内部环境状态
        core_state = env_state.env_state
        reward_sum = float(batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1).sum())
        cumulative_reward += reward_sum

        # --- 飞行状态 ---
        roll_deg = _f(core_state.plane_state.roll) * 180 / np.pi
        pitch_deg = _f(core_state.plane_state.pitch) * 180 / np.pi
        yaw_deg = _f(core_state.plane_state.yaw) * 180 / np.pi
        vt_now = _f(core_state.plane_state.vt)
        alt_now = _f(core_state.plane_state.altitude)

        # --- 计算与当前航点的误差 ---
        hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
        q_curr = np.array([
            _f(core_state.plane_state.q0),
            _f(core_state.plane_state.q1),
            _f(core_state.plane_state.q2),
            _f(core_state.plane_state.q3),
        ])
        theta_err = quat_geodesic_deg(q_curr, hdg_t, pit_t, rol_t)
        delta_vt = abs(vt_now - vt_t)

        # --- 渲染到 ACMI ---
        try:
            env_core.render(core_state, env_params, done, logdir)
        except Exception:
            pass

        # --- 日志 ---
        if step % 10 == 0 or theta_err < success_theta_deg:
            print(
                f"{step:6d} | {wp_idx:3d} | {theta_err:7.2f} | {delta_vt:6.1f} | "
                f"{roll_deg:7.1f} | {pitch_deg:7.1f} | {yaw_deg:7.1f} | "
                f"{vt_now:6.1f} | {alt_now:7.0f} | {reward_sum:+7.3f}"
            )

        # --- 检查是否到达当前航点 ---
        if theta_err <= success_theta_deg and delta_vt <= success_vt_tol:
            wp_hold_steps += 1
        else:
            wp_hold_steps = 0

        if wp_hold_steps >= min_hold:
            wp_idx += 1
            wp_hold_steps = 0
            if wp_idx >= num_wps:
                print(f"\n*** 所有 {num_wps} 个航点均已到达! 累计回报={cumulative_reward:.1f} ***")
                break
            # 切换到下一个航点
            hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
            core_state = core_state.replace(
                target_heading=jnp.full((env.num_agents,), hdg_t),
                target_pitch=jnp.full((env.num_agents,), pit_t),
                target_roll=jnp.full((env.num_agents,), rol_t),
                target_vt=jnp.full((env.num_agents,), vt_t),
                last_check_time=core_state.time,
            )
            env_state = env_state.replace(env_state=core_state)
            print(f"\n>>> 切换到航点 WP_{wp_idx}: heading={np.degrees(hdg_t):.0f}° "
                  f"pitch={np.degrees(pit_t):.0f}° roll={np.degrees(rol_t):.0f}° vt={vt_t:.0f} m/s\n")

        # --- 检查 done ---
        obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        done_arr = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)

        done_any = bool(np.asarray(done_arr).any())
        if done_any:
            print(f"\n[DONE] Episode terminated at step {step}, WP={wp_idx}/{num_wps}, cumR={cumulative_reward:.1f}")
            # reset 后继续（环境会自动 reset）
            hstate = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            # 重新设置目标航点
            hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
            core_state = env_state.env_state
            core_state = core_state.replace(
                target_heading=jnp.full((env.num_agents,), hdg_t),
                target_pitch=jnp.full((env.num_agents,), pit_t),
                target_roll=jnp.full((env.num_agents,), rol_t),
                target_vt=jnp.full((env.num_agents,), vt_t),
            )
            env_state = env_state.replace(env_state=core_state)

    print(f"\n{'='*90}")
    print(f"渲染完成: 到达 {wp_idx}/{num_wps} 个航点, 累计回报 = {cumulative_reward:.1f}")
    print(f"ACMI 文件: {getattr(env_core, 'filename', 'N/A')}")
    print(f"{'='*90}")


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
        "MAX_STEPS": 3000,
        "LOGDIR": "./tracks/full_domain/",
        # checkpoint 路径
        "LOADDIR": os.path.abspath(
            "results/full_domain_maneuver_2026-03-04-23-59/checkpoints/checkpoint_epoch_2000"
        ),
    }
    render_episode(config)
