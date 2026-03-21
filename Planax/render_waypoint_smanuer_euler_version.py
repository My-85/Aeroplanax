# Planax/render_waypoint.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
from pathlib import Path

from envs.aeroplanax_waypoint_smanuer_euler_version import AeroPlanaxWaypointEnv, WaypointTaskParams
from envs.wrappers import LogWrapper
from envs.utils.utils import enu_to_geodetic

# ====== 常量 ======
LOGDIR = "/home/dqy/aeroplanax/new/20251215最新代码库/tracks/"
BASELINE_LOADDIR = "/home/dqy/aeroplanax/new/20251215最新代码库/results/s机动baseline（欧拉角版本）/checkpoints/checkpoint_epoch_600"
BASELINE_TYPE = "rnn"  # or "lstm"
SEED = 42
NUM_STEPS = 2000
WP_BASE_ID = 500000  # 航点对象的高位 ID 段，避免与飞机/导弹冲突

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
        max_steps=5000,
        sim_freq=50,
        agent_interaction_steps=10,

        # 航点更密集
        reach_radius_init=500.0, # 航点判定半径
        reach_radius_decay=1.0,

        # ——S 形机动——
        use_s_curve=True,
        s_amplitude=5000.0,     # 左右半圆的半径大小
        # s_step_north=2000.0,    # 沿北向的步进（米），越小越密

        s_half_period_north=20000.0,         # 北向半周期长度（米）← 半径更大就把它调大
        s_points_per_half=50,                 # 每个半周期上取多少个航点（越大越平滑）

        # s_altitude_lock=True,   # 俯仰期望=0°
        s_altitude_lock=False,   # 不进行高度锁定，当前 S 形航点生成里，航点高度每次都设为 state.plane_state.altitude[0]，因此即使不锁高，da≈0，俯仰仍接近 0，轨迹基本是“平面 S”。
        max_waypoints=100,       # 达成目标航点个数就终止（到够数就 SUCCESS）
        s_target_vt=250.0,      # 恒定巡航速度
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

        add("ax(m/s^2)", getattr(ps, "ax", None))
        add("ay(m/s^2)", getattr(ps, "ay", None))
        add("az(m/s^2)", getattr(ps, "az", None))
        # g-load
        try:
            gz = _scalar_any(getattr(ps, "az", 0.0)) / 9.80665
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
        print(block)

        # 落到旁路日志
        with open(env.filename + ".precrash.log", "a", encoding="utf-8") as f:
            f.write(block + "\n")
        return block

    prev_reached = int(scalar(log_state.env_state.reached))

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

if __name__ == "__main__":
    main()
