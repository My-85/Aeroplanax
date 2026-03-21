# Planax/render_waypoint.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
from pathlib import Path

from envs.aeroplanax_waypoint_vertical_loop_add_pred_baseline import AeroPlanaxWaypointEnv, WaypointTaskParams
from envs.wrappers import LogWrapper
from envs.utils.utils import enu_to_geodetic

# ====== 常量 ======
LOGDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/tracks/"
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/2v2_lczh_mine/AeroPlanax_multi_combat_2v2/envs/models/RNN新策略/PPO+RNN(actor与critic均加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed10/heading_pitch_V_discrete_rnn_2025-08-29-12-46/checkpoints/checkpoint_epoch_600"
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-09-17-01-44/checkpoints/checkpoint_epoch_1000"
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-09-17-17-12/checkpoints/checkpoint_epoch_1400"
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-09-18-17-07/checkpoints/checkpoint_epoch_1600"
BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline/new_baseline/带预测头的版本/heading_pitch_V_discrete_rnn_add_obs_and_pred_2025-10-10-23-57/checkpoints/checkpoint_epoch_600"
BASELINE_TYPE = "rnn"  # or "lstm"
SEED = 42
NUM_STEPS = 200000
WP_BASE_ID = 610000  # 航点对象的高位 ID 段，避免与飞机/导弹冲突

# ====== 小工具 ======
def scalar(x):
    return float(jnp.ravel(jnp.asarray(x))[0])

def debug_print(st):
    n = scalar(st.plane_state.north)
    e = scalar(st.plane_state.east)
    a = scalar(st.plane_state.altitude)
    vt = scalar(st.plane_state.vt)
    yaw = scalar(st.plane_state.yaw)
    t = scalar(st.time)
    print(f"[t={t:6.2f}] N={n:9.2f} E={e:9.2f} Alt={a:7.2f} VT={vt:6.2f} Yaw={yaw:7.3f}")

def write_wp_marker(filename: str, t_sec: float, idx: int, n: float, e: float, alt: float):
    # 统一：先创建对象（Type=ReferencePoint），再写位置
    wlat, wlon, walt = enu_to_geodetic(e, n, alt, 0, 0, 0)
    oid = WP_BASE_ID + idx
    with open(filename, mode="a", encoding="utf-8") as f:
        f.write(f"{t_sec:.2f},{oid},Type=ReferencePoint,Name=WP_{idx},Color=Yellow\n")
        f.write(f"{t_sec:.2f},{oid},T={wlon}|{wlat}|{walt}|0|0|0\n")

def main():
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)

    # 1) 建环境（构造时传参；reset/step 不要再传 env_params）
    
    env_params = WaypointTaskParams(
        baseline_loaddir=BASELINE_LOADDIR,
        baseline_type=BASELINE_TYPE,
        use_internal_baseline=True,

        # 评测/可视化参数
        max_steps=50000000,
        sim_freq=50,
        agent_interaction_steps=2,   # 由 10 → 2，提高竖直段的控制带宽（25Hz）

        # 关键：开启筋斗
        use_vertical_loop=True,
        loop_radius=15000.0,
        loop_points_per_circle=30,     # 400 -> 180（每步角度 ~2°）
        loop_forward_north=1000.0,

        # 正确的首点与方向
        loop_phase0_deg=180.0,   # 首航点在圆的“正前方” (与飞机同高，俯仰≈0)
        loop_direction=-1,       # 顺时针推进：φ = π, π-dφ, π-2dφ, ...（先“前方”，再“前上方”，逐步抬升）

        loop_pitch_limit_deg=85.0, # 俯仰限幅

        # 航点到达判定（可按需调小/调大）
        reach_radius_init=2000.0,        # 2000 -> 1000    
        reach_radius_decay=1.0,

        # 筋斗需要跟随高度变化，建议关闭锁高（如果你的 env 里还有 s_altitude_lock 判断）
        s_altitude_lock=False,

        max_waypoints=2000,
    )

    env = LogWrapper(AeroPlanaxWaypointEnv(env_params))

    # 2) reset（单环境，不 vmap）
    rng = jax.random.PRNGKey(SEED)
    obs, log_state = env.reset(rng)

    # 评测模式外部动作会被忽略，但接口仍需传（保持 API 完整）
    dummy_action = {env.agents[0]: jnp.array([0, 0, 0, 0], dtype=jnp.int32)}

    # 3) 先写一帧
    env.render(log_state.env_state, env_params, {'__all__': False}, LOGDIR)
    debug_print(log_state.env_state)

    # 写入首个航点 Marker（Tacview）
    st = log_state.env_state
    w = jnp.ravel(st.waypoint)
    wp_n, wp_e, wp_alt = float(w[0]), float(w[1]), float(w[2])
    t0 = scalar(st.time)
    write_wp_marker(env.filename, t0, 0, wp_n, wp_e, wp_alt)

    # === 诊断日志文件 ===
    diag_path = env.filename + ".diag.txt"
    diag_f = open(diag_path, "w", encoding="utf-8")
    diag_f.write("t,dist,hdist,vt,alt,yaw_deg,d_pitch_deg,pitch_deg,roll_deg,"
                 "nz,qbar_norm,low_qbar,energy_norm,gamma_deg,vt_target,vert_phase,"
                 "cmd_head_deg,cmd_pitch_deg,cmd_vt\n")
    # prev_reached = int(scalar(log_state.env_state.reached))

    def _safe_has(obj, name):
        return hasattr(obj, name)

    def _scalar_any(x):
        try:
            arr = jnp.asarray(x)
            if arr.size == 1:
                return float(jnp.ravel(arr)[0])
            # 多元素：返回 Python list（少量字段也行）
            return [float(v) for v in jnp.ravel(arr).tolist()]
        except Exception:
            try:
                return float(x)
            except Exception:
                return x

    def dump_plane_block(tag, t, ps, cs, extra_info=None):
        # ps: plane_state, cs: control_state
        deg = 180.0 / jnp.pi
        lines = [f"=== {tag} @ t={t:.2f} ==="]

        def add(name, value, post=""):
            if value is None: return
            v = _scalar_any(value)
            lines.append(f"{name}: {v}{post}")

        # Plane kinematics / attitude
        add("status", getattr(ps, "status", None))
        add("north", getattr(ps, "north", None), " m")
        add("east", getattr(ps, "east", None), " m")
        add("altitude", getattr(ps, "altitude", None), " m")

        add("vel_x", getattr(ps, "vel_x", None), " m/s")
        add("vel_y", getattr(ps, "vel_y", None), " m/s")
        add("vel_z", getattr(ps, "vel_z", None), " m/s")
        add("vt", getattr(ps, "vt", None), " m/s")

        # Attitude (rad → deg)
        r = getattr(ps, "roll", None);   add("roll(deg)", None if r is None else _scalar_any(r)*deg)
        p = getattr(ps, "pitch", None);  add("pitch(deg)", None if p is None else _scalar_any(p)*deg)
        y = getattr(ps, "yaw", None);    add("yaw(deg)", None if y is None else _scalar_any(y)*deg)

        a = getattr(ps, "alpha", None);  add("alpha(deg)", None if a is None else _scalar_any(a)*deg)
        b = getattr(ps, "beta(deg)", None); add("beta(deg)", None if b is None else _scalar_any(b)*deg)

        add("P(rad/s)", getattr(ps, "P", None))
        add("Q(rad/s)", getattr(ps, "Q", None))
        add("R(rad/s)", getattr(ps, "R", None))

        add("ax(g)", getattr(ps, "ax", None))
        add("ay(g)", getattr(ps, "ay", None))
        add("az(g)", getattr(ps, "az", None))
        # g-load
        try:
            gz = _scalar_any(getattr(ps, "az", 0.0))
            lines.append(f"Nz(g): {gz:.3f}")
        except Exception:
            pass

        # Quaternions (如有)
        add("q0", getattr(ps, "q0", None))
        add("q1", getattr(ps, "q1", None))
        add("q2", getattr(ps, "q2", None))
        add("q3", getattr(ps, "q3", None))

        # Control surfaces / throttle（如有）
        if cs is not None:
            add("ctrl.throttle", getattr(cs, "throttle", None))
            add("ctrl.elevator", getattr(cs, "elevator", None))
            add("ctrl.aileron",  getattr(cs, "aileron", None))
            add("ctrl.rudder",   getattr(cs, "rudder", None))

        # 额外来自 info 的调参/目标值
        if extra_info:
            for k in sorted(extra_info.keys()):
                lines.append(f"{k}: {_scalar_any(extra_info[k])}")

        block = "\n".join(lines)
        # print(block)

        # 落到旁路日志
        with open(env.filename + ".precrash.log", "a", encoding="utf-8") as f:
            f.write(block + "\n")
        return block

    prev_reached = int(scalar(st.reached))

    for _ in range(NUM_STEPS):
        # ===== 步前快照 =====
        st_prev = log_state.env_state
        ps_prev = st_prev.plane_state
        cs_prev = getattr(st_prev, "control_state", None)  # 如果没有也能容错
        prev_t   = scalar(st_prev.time)
        prev_n   = scalar(ps_prev.north)
        prev_e   = scalar(ps_prev.east)
        prev_alt = scalar(ps_prev.altitude)

        rng, key_step = jax.random.split(rng)
        obs, log_state, reward, done, info = env.step(key_step, log_state, dummy_action)

        # 渲染
        env.render(log_state.env_state, env_params, {'__all__': bool(done["__all__"])}, LOGDIR)

        # 常规打印（步后）
        st = log_state.env_state
        def i0(name, default=0.0):
            v = info.get(name, default)
            try: return float(jnp.ravel(jnp.asarray(v))[0])
            except Exception: return float(jnp.ravel(jnp.asarray(v)))

        dist        = i0('dist_to_wp')
        hdist       = i0('hdist_to_wp')
        reach_flag  = bool(i0('reached_this_step'))
        reach_r     = i0('reach_radius')
        reached_cnt = int(i0('reached_count'))

        vt = scalar(st.plane_state.vt)
        alt = scalar(st.plane_state.altitude)
        yaw_deg = scalar(st.plane_state.yaw) * 180.0 / jnp.pi

        print(f"dist={dist:.1f}m  hdist={hdist:.1f}m  vt={vt:.1f}  alt={alt:.1f}  "
            f"yaw={yaw_deg:.2f}°  reach_r={reach_r:.1f}  reached_cnt={reached_cnt} "
            f"{'REACHED' if reach_flag else ''}")
        debug_print(st)

        # === 关键诊断 ===
        qn  = i0('mon_qbar_norm');  low = int(i0('mon_low_qbar'));  nz = i0('mon_nz')
        gdg = i0('mon_gamma_deg');  rdeg = i0('mon_roll_deg');      pdeg = i0('mon_pitch_deg')
        vtt = i0('mon_vt');         vttgt = i0('mon_target_vt');    En = i0('mon_energy_norm')
        dpg = i0('mon_desired_pitch_deg');  vphase = int(i0('mon_is_vertical_phase'))

        # 平滑后的控制指令（来自 env info）
        cmd_h = i0('cmd_heading', 0.0) * (180.0/jnp.pi)
        cmd_p = i0('cmd_pitch', 0.0)   * (180.0/jnp.pi)
        cmd_v = i0('cmd_vt', 0.0)
        print(f"[diag] qbar_n={qn:.2f} low={low}  nz={nz:.2f}  gamma={gdg:.1f}°  "
              f"roll={rdeg:.1f}°  pitch={pdeg:.1f}°  vt={vtt:.1f}/{vttgt:.1f}  "
              f"En={En:.2f}  d_pitch={dpg:.1f}°  vert={vphase}")

        # === 写入诊断文件 ===
        tnow = scalar(st.time)
        diag_f.write(f"{tnow:.2f},{dist:.2f},{hdist:.2f},{vtt:.2f},{alt:.2f},{yaw_deg:.2f},"
                     f"{dpg:.2f},{pdeg:.2f},{rdeg:.2f},{nz:.2f},{qn:.3f},{low},"
                     f"{En:.3f},{gdg:.2f},{vttgt:.2f},{vphase},{cmd_h:.2f},{cmd_p:.2f},{cmd_v:.2f}\n")
        if (_ % 100) == 0:
            diag_f.flush()
        # 判 reset（时间回绕）
        cur_t = scalar(st.time)
        just_reset = bool(done["__all__"]) and (cur_t < prev_t)

        # 汇总 info 里对定位问题最关键的目标/判据（如果环境里已写到 info）
        dbg_pack = {
            "dbg_desired_pitch(deg)": info.get("dbg_desired_pitch", None),
            "dbg_desired_heading(deg)": info.get("dbg_desired_heading", None),
            "dbg_target_vt": info.get("dbg_target_vt", None),
            "dbg_dist3d": info.get("dbg_dist3d", None),
            "dbg_hdist": info.get("dbg_hdist", None),
            "dbg_reach_radius": info.get("dbg_reach_radius", None),
            "dbg_reach_now": info.get("dbg_reach_now", None),
            "dbg_reached_count": info.get("dbg_reached_count", None),
            "plane_status_before": info.get("plane_status_before", None),
            "time_before": info.get("time_before", None),
        }
        # 角度字段换成度（更易读）
        if dbg_pack["dbg_desired_pitch(deg)"] is not None:
            dbg_pack["dbg_desired_pitch(deg)"] = _scalar_any(dbg_pack["dbg_desired_pitch(deg)"]) * (180.0/jnp.pi)
        if dbg_pack["dbg_desired_heading(deg)"] is not None:
            dbg_pack["dbg_desired_heading(deg)"] = _scalar_any(dbg_pack["dbg_desired_heading(deg)"]) * (180.0/jnp.pi)

        # 额外把平滑后的指令写入 precrash 辅助块
        dbg_pack["cmd_heading_deg"] = cmd_h
        dbg_pack["cmd_pitch_deg"] = cmd_p
        dbg_pack["cmd_vt"] = cmd_v

        if just_reset:
            # 终止原因（可选字段）
            reason = "DONE"
            if int(i0('dbg_crashed', 0.0)) == 1: reason = "CRASH"
            elif int(i0('dbg_timeout', 0.0)) == 1: reason = "TIMEOUT"
            elif int(i0('dbg_enough', 0.0)) == 1: reason = "SUCCESS"

            print(f"--- TERMINATED: {reason} ---")

            # 打印 + 落档：步前完整状态
            dump_plane_block("PRE-STEP SNAPSHOT", prev_t, ps_prev, cs_prev, extra_info=dbg_pack)

            # Tacview 标注崩溃点
            wlat, wlon, walt = enu_to_geodetic(prev_e, prev_n, prev_alt, 0, 0, 0)
            with open(env.filename, mode="a", encoding="utf-8") as f:
                f.write(f"9900,T={wlon}|{wlat}|{walt}|0|0|0,Name={reason},Color=Red,Type=ReferencePoint\n")

        # 一般状态跟踪（步后）
        reached_cnt_now = int(scalar(st.reached))
        status_now = int(scalar(st.plane_state.status))
        ax = scalar(st.plane_state.ax); ay = scalar(st.plane_state.ay); az = scalar(st.plane_state.az)
        print("status=", status_now, "ax/ay/az=", ax, ay, az, "reached_cnt=", reached_cnt_now)

        if reached_cnt_now > prev_reached:
            w = jnp.ravel(st.waypoint)
            wp_n, wp_e, wp_alt = float(w[0]), float(w[1]), float(w[2])
            t_now = scalar(st.time)
            write_wp_marker(env.filename, t_now, reached_cnt_now, wp_n, wp_e, wp_alt)
            prev_reached = reached_cnt_now

        if bool(done["__all__"]):
            break


    print("Done. Tacview file:", env.filename)
    diag_f.close()
    print("Diag saved:", diag_path)
if __name__ == "__main__":
    main()
