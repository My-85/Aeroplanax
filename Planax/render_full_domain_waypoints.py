"""
render_full_domain_waypoints.py
================================
加载 full_domain_maneuver baseline，逐级增加难度地定义目标航点，
测试训练出的飞行控制效果。

航点以 (target_heading_deg, target_pitch_deg, target_roll_deg, target_vt)
形式预定义，分 5 个难度级别。飞机到达一个航点后自动切换下一个。
所有航点一次性写入 ACMI 文件，可在 Tacview 中渲染。

用法:
  cd Planax && python render_full_domain_waypoints.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from pathlib import Path
from datetime import datetime
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


# ============================================================
# 预定义航点 (5 个难度级别)
# (heading_deg, pitch_deg, roll_deg, vt_m_s)
# ============================================================
WAYPOINTS = [
    # ---- Level 1: 简单水平转弯 / 直飞 ----
    (  0.0,   0.0,   0.0, 200.0),   # WP0:  直飞保持
    ( 30.0,   0.0,   0.0, 200.0),   # WP1:  小幅右转 30
    (-30.0,   0.0,   0.0, 200.0),   # WP2:  小幅左转 30

    # ---- Level 2: 中等转弯 + 轻微俯仰 ----
    ( 90.0,   5.0,   0.0, 220.0),   # WP3:  右转 90 + 微爬
    (-90.0,  -5.0,   0.0, 220.0),   # WP4:  左转 90 + 微俯

    # ---- Level 3: 大角度转弯 + 小横滚 ----
    (180.0,   0.0,  20.0, 240.0),   # WP5:  掉头 + 20 横滚
    (  0.0,  10.0, -20.0, 240.0),   # WP6:  爬升 10 + 左倾 20

    # ---- Level 4: 较大俯仰 + 中等横滚 ----
    ( 45.0,  30.0,  40.0, 260.0),   # WP7:  爬升 30 + 40 横滚
    (-45.0, -20.0, -40.0, 220.0),   # WP8:  俯冲 20 + 左横滚

    # ---- Level 5: 全姿态域 ----
    (  0.0,  60.0,  60.0, 300.0),   # WP9:  大拉起 + 大横滚
    (180.0, -40.0,  90.0, 250.0),   # WP10: 大俯冲 + 90 横滚
    ( 90.0,   0.0, 150.0, 250.0),   # WP11: 接近倒飞
]


# ============================================================
# 网络定义 (与 train_full_domain_maneuver.py 完全一致)
# ============================================================

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


# ============================================================
# 辅助函数
# ============================================================

def _f(x, i=0):
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    return float(a.reshape(-1)[min(i, a.size - 1)])


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def quat_geodesic_deg(q_curr, yaw_t, pitch_t, roll_t):
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


# ============================================================
# 主渲染函数
# ============================================================

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

    # ---------- 4) 航点 → 弧度 ----------
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

    # ---------- 6) 手动创建 ACMI 文件 + 写入所有航点标记 ----------
    # 不使用 env_core.render()，因为它在 episode done 时会创建新文件，导致航点丢失
    logdir = config.get("LOGDIR", "./tracks/full_domain_wp/")
    os.makedirs(logdir, exist_ok=True)
    acmi_filename = logdir + datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + '.txt.acmi'

    # 写 ACMI header
    with open(acmi_filename, mode='w', encoding='utf-8') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")

    # 以飞机初始位置为基准，沿目标 heading/pitch 方向放置航点标记
    init_north = _f(core_state.plane_state.north)
    init_east = _f(core_state.plane_state.east)
    init_alt = _f(core_state.plane_state.altitude)

    # 一次性写入所有航点到 ACMI 文件
    WP_ID_BASE = 5000
    with open(acmi_filename, mode='a', encoding='utf-8') as f:
        for k, (hdg_d, pit_d, rol_d, vt) in enumerate(WAYPOINTS):
            hdg_r = np.radians(hdg_d)
            pit_r = np.radians(pit_d)
            dist = 2000.0 * (k + 1)
            wp_n = init_north + dist * np.cos(hdg_r) * np.cos(pit_r)
            wp_e = init_east + dist * np.sin(hdg_r) * np.cos(pit_r)
            wp_a = init_alt + dist * np.sin(pit_r)
            wp_a = max(wp_a, 1000.0)
            lat, lon, alt = enu_to_geodetic(wp_e, wp_n, wp_a, 0, 0, 0)
            oid = WP_ID_BASE + k
            f.write(
                f"{oid},"
                f"Type=Navaid+Static+Waypoint,Name=WP_{k},Label={k},Color=Yellow,"
                f"T={float(lon)}|{float(lat)}|{float(alt)}|0|0|0\n"
            )
            print(f"  WP_{k} written: oid={oid} lat={float(lat):.6f} lon={float(lon):.6f} alt={float(alt):.1f}")
    print(f"ACMI file: {acmi_filename}")

    # ---------- 7) RNN 状态 ----------
    hstate = ScannedRNN.initialize_carry(
        config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
    )

    # ---------- 8) 仿真循环 ----------
    done_arr = jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool)
    obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])

    max_steps = config.get("MAX_STEPS", 5000)
    success_theta_deg = 10.0
    success_vt_tol = 15.0
    wp_hold_steps = 0
    min_hold = 40
    cumulative_reward = 0.0
    reset_count = 0

    print(f"\n{'='*100}")
    print(f"Full-domain maneuver waypoint render: {num_wps} waypoints, max {max_steps} steps")
    print(f"Checkpoint: {config.get('LOADDIR', 'N/A')}")
    print(f"{'='*100}")

    # 打印航点列表
    for k, (hdg_d, pit_d, rol_d, vt) in enumerate(WAYPOINTS):
        level = k // 3 + 1
        print(f"  WP{k:2d} [Level {level}]: heading={hdg_d:+7.1f}  pitch={pit_d:+6.1f}  roll={rol_d:+7.1f}  vt={vt:.0f} m/s")

    print(f"\n{'Step':>6} | {'WP':>3} | {'Lvl':>3} | {'theta_err':>9} | {'dVt':>6} | {'Roll':>7} | {'Pitch':>7} | {'Yaw':>7} | {'Vt':>6} | {'Alt':>7} | {'R':>7} | {'Hold':>4}")
    print(f"{'-'*100}")

    rng_run = jax.random.PRNGKey(config["SEED"] + 100)
    wp_results = []  # 记录每个航点的到达情况

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

        # --- 手动渲染到 ACMI (不使用 env_core.render，避免 done 时创建新文件) ---
        with open(acmi_filename, mode='a', encoding='utf-8') as f:
            timestamp = core_state.time * env_params.agent_interaction_steps / env_params.sim_freq
            ts = float(jnp.ravel(jnp.asarray(timestamp))[0])
            f.write(f"#{ts:.2f}\n")
            npos = _f(core_state.plane_state.north)
            epos = _f(core_state.plane_state.east)
            alt_v = _f(core_state.plane_state.altitude)
            roll_v = _f(core_state.plane_state.roll) * 180 / np.pi
            pitch_v = _f(core_state.plane_state.pitch) * 180 / np.pi
            yaw_v = _f(core_state.plane_state.yaw) * 180 / np.pi
            lat_v, lon_v, alt_v = enu_to_geodetic(epos, npos, alt_v, 0, 0, 0)
            f.write(f"100,T={lon_v}|{lat_v}|{alt_v}|{roll_v}|{pitch_v}|{yaw_v},Type=Air+FixedWing,Name=F16,Color=Red\n")
            # 目标指示器
            target_heading_v = _f(core_state.target_heading)
            target_pitch_v = _f(core_state.target_pitch)
            tgt_dist = 1000.0
            tgt_dn = tgt_dist * np.cos(target_pitch_v) * np.cos(target_heading_v)
            tgt_de = tgt_dist * np.cos(target_pitch_v) * np.sin(target_heading_v)
            tgt_da = tgt_dist * np.sin(target_pitch_v)
            tgt_lat, tgt_lon, tgt_alt = enu_to_geodetic(
                epos + tgt_de, npos + tgt_dn, _f(core_state.plane_state.altitude) + tgt_da, 0, 0, 0
            )
            f.write(
                f"1000,T={tgt_lon}|{tgt_lat}|{tgt_alt}|0|{target_pitch_v*180/np.pi}|{target_heading_v*180/np.pi},"
                f"Name=Target_0,Color=Yellow,Type=Marker\n"
            )

        # --- 日志 ---
        level = wp_idx // 3 + 1
        if step % 20 == 0 or theta_err < success_theta_deg:
            print(
                f"{step:6d} | {wp_idx:3d} | {level:3d} | {theta_err:9.2f}deg | {delta_vt:6.1f} | "
                f"{roll_deg:+7.1f} | {pitch_deg:+7.1f} | {yaw_deg:+7.1f} | "
                f"{vt_now:6.1f} | {alt_now:7.0f} | {reward_sum:+7.3f} | {wp_hold_steps:4d}"
            )

        # --- 检查是否到达当前航点 ---
        if theta_err <= success_theta_deg and delta_vt <= success_vt_tol:
            wp_hold_steps += 1
        else:
            wp_hold_steps = 0

        if wp_hold_steps >= min_hold:
            wp_results.append({
                "wp_idx": wp_idx,
                "level": level,
                "step": step,
                "theta_err": theta_err,
                "delta_vt": delta_vt,
                "resets": reset_count,
            })
            wp_idx += 1
            wp_hold_steps = 0
            if wp_idx >= num_wps:
                print(f"\n*** 所有 {num_wps} 个航点均已到达! 累计回报={cumulative_reward:.1f} ***")
                break
            # 切换到下一个航点
            hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
            core_state = core_state.replace(
                target_heading=jnp.full_like(core_state.target_heading, hdg_t),
                target_pitch=jnp.full_like(core_state.target_pitch, pit_t),
                target_roll=jnp.full_like(core_state.target_roll, rol_t),
                target_vt=jnp.full_like(core_state.target_vt, vt_t),
                last_check_time=core_state.time,
            )
            env_state = env_state.replace(env_state=core_state)
            new_level = wp_idx // 3 + 1
            print(f"\n>>> 到达 WP_{wp_idx-1}! 切换到 WP_{wp_idx} [Level {new_level}]: "
                  f"heading={np.degrees(hdg_t):+.0f} pitch={np.degrees(pit_t):+.0f} "
                  f"roll={np.degrees(rol_t):+.0f} vt={vt_t:.0f} m/s\n")

        # --- 检查 done ---
        obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        done_arr = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)

        done_any = bool(np.asarray(done_arr).any())
        if done_any:
            reset_count += 1
            print(f"\n[RESET #{reset_count}] Episode terminated at step {step}, "
                  f"WP={wp_idx}/{num_wps}, cumR={cumulative_reward:.1f}")
            # reset RNN hidden state
            hstate = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            # 重新设置目标航点
            hdg_t, pit_t, rol_t, vt_t = waypoints_rad[wp_idx]
            core_state = env_state.env_state
            core_state = core_state.replace(
                target_heading=jnp.full_like(core_state.target_heading, hdg_t),
                target_pitch=jnp.full_like(core_state.target_pitch, pit_t),
                target_roll=jnp.full_like(core_state.target_roll, rol_t),
                target_vt=jnp.full_like(core_state.target_vt, vt_t),
            )
            env_state = env_state.replace(env_state=core_state)

    # ---------- 9) 总结 ----------
    print(f"\n{'='*100}")
    print(f"渲染完成: 到达 {wp_idx}/{num_wps} 个航点")
    print(f"累计回报 = {cumulative_reward:.1f}, 环境重置次数 = {reset_count}")
    print(f"ACMI 文件: {acmi_filename}")
    print()

    if wp_results:
        print("已到达航点:")
        for r in wp_results:
            print(f"  WP{r['wp_idx']:2d} [Level {r['level']}] @ step {r['step']:5d}  "
                  f"theta_err={r['theta_err']:.2f}deg  dVt={r['delta_vt']:.1f}m/s  resets={r['resets']}")
    else:
        print("未能到达任何航点!")

    # 评估等级
    if wp_idx >= num_wps:
        grade = "A+ (全部通过)"
    elif wp_idx >= 9:
        grade = "A (Level 4 通过)"
    elif wp_idx >= 7:
        grade = "B (Level 3 通过)"
    elif wp_idx >= 5:
        grade = "C (Level 2 通过)"
    elif wp_idx >= 3:
        grade = "D (Level 1 通过)"
    else:
        grade = "F (基础能力不足)"

    print(f"\n综合评级: {grade}")
    print(f"{'='*100}")


# ============================================================
# 入口
# ============================================================

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
        "MAX_STEPS": 5000,
        "LOGDIR": "./tracks/full_domain_wp/",
        "LOADDIR": os.path.abspath(
            "/home/dqy/aeroplanax/new/20251215最新代码库/Planax/results/full_domain_maneuver_v2_2026-03-06-16-04/checkpoints/checkpoint_epoch_2000"
        ),
    }
    render_episode(config)
