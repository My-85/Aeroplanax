# /home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/render_waypoint_vertical_loop.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
from pathlib import Path

# from envs.aeroplanax_waypoint_vertical_loop import AeroPlanaxWaypointEnv, WaypointTaskParams
from envs.aeroplanax_waypoint_vertical_loop_quaternion_version_new import (
# from envs.aeroplanax_waypoint_vertical_loop_quaternion_version import (
    AeroPlanaxWaypointEnv, WaypointTaskParams
)

from envs.wrappers import LogWrapper
from envs.utils.utils import enu_to_geodetic

import matplotlib.pyplot as plt

# ====== 常量 ======
LOGDIR = "/home/dqy/aeroplanax/new/20251215最新代码库/tracks/"
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-11-19-16-12/checkpoints/checkpoint_epoch_600" # -45°~45°的baseline
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-06-01-30/checkpoints/checkpoint_epoch_800" # -80°~80°的baseline
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-06-11-11/checkpoints/checkpoint_epoch_1300" # -80°~80°并随机初始化pitch的baseline
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-06-16-31/checkpoints/checkpoint_epoch_1300" # -89°~89°并随机初始化pitch的baseline
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-08-18-30/checkpoints/checkpoint_epoch_1600" # -89°~89° + 随机初始化pitch + 加大最大速度和最小速度阈值的baseline
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-14-01-47/checkpoints/checkpoint_epoch_2500" # roll、pitch、yaw都是随机初始化，并且目标也是，扩展了obs(trained third)
# BASELINE_LOADDIR = "/home/dqy/aeroplanax/new/20251215最新代码库/results/baseline_quat_increase_pitch_to_89_random_pitch_decrease_min_vt_to_50_2026-02-08-01-26/checkpoints/checkpoint_epoch_2000" # -89°~89°并随机初始化pitch的baseline
BASELINE_LOADDIR = "/home/dqy/aeroplanax/new/20251215最新代码库/results/baseline_quat_increase_pitch_to_89_random_pitch_decrease_min_vt_to_50_trained_third_2026-02-08-20-39/checkpoints/checkpoint_epoch_3200" # trained third baseline
BASELINE_TYPE = "rnn"  # or "lstm"
SEED = 42
NUM_STEPS = 1100
WP_BASE_ID = 610000  # 航点对象的高位 ID 段

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
    wlat, wlon, walt = enu_to_geodetic(e, n, alt, 0, 0, 0)
    oid = WP_BASE_ID + idx
    with open(filename, mode="a", encoding="utf-8") as f:
        f.write(f"{t_sec:.2f},{oid},Type=ReferencePoint,Name=WP_{idx},Label=WP_{idx},Color=Yellow,Locked=1\n")
        f.write(f"{t_sec:.2f},{oid},T={wlon}|{wlat}|{walt}|0|0|0\n")

def write_wp_polyline(filename: str, t_sec: float, wps: jnp.ndarray):
    pts = []
    for i in range(int(wps.shape[0])):
        n, e, a = float(wps[i, 0]), float(wps[i, 1]), float(wps[i, 2])
        wlat, wlon, walt = enu_to_geodetic(e, n, a, 0, 0, 0)
        pts.append(f"{wlon}|{wlat}|{walt}")
    oid = WP_BASE_ID + 900000
    with open(filename, mode="a", encoding="utf-8") as f:
        f.write(f"{t_sec:.2f},{oid},Type=Drawing,SubType=Polyline,Label=LoopWPs,Color=Yellow,LineStyle=Solid\n")
        f.write(f"{t_sec:.2f},{oid},Points=" + ";".join(pts) + "\n")

def main():
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)

    # 环境参数（开启筋斗；指令链路与 env 内一致）
    env_params = WaypointTaskParams(
        baseline_loaddir=BASELINE_LOADDIR,
        baseline_type=BASELINE_TYPE,
        use_internal_baseline=True,

        ##################################
        start_north=0.0,
        start_east=0.0,
        start_alt=9000.0,      # 精确控制初始高度 # 飞机初始高度 = 筋斗圈最低点高度
        start_yaw_deg=0.0,     # 初始朝向（度）
        start_vt=300.0,        # 初始空速（m/s）
        ##################################

        max_steps=50000000,
        sim_freq=50,
        agent_interaction_steps=10,   # 提高控制带宽

        use_vertical_loop=True,
        loop_radius=8000.0, # 圆心高度将是 9000 + 8000 = 17000m
        loop_points_per_circle=20,
        loop_forward_north=1000.0,
        # loop_target_vt=250.0,

        # # 选择“正前方首点+逆时针推进”（示例）
        # loop_phase0_deg=180.0,
        # loop_direction=-1,

        # 选择"最低点首点+逆时针推进"
        loop_phase0_deg=180.0,    # 改为 -90，让第一个航点是圆的最低点
        loop_direction=-1,         # 改为 +1 顺时针（从最低点向上爬升）

        loop_enter_offset=8000.0,     # 圆心在飞机前方 16000m (R + offset)
        loop_floor_margin=500.0,      # 降低地面安全余量（因为现在最低点=初始高度）
        # loop_pitch_limit_deg=85.0,

        reach_radius_init=2000.0,
        reach_radius_decay=1.0,

        s_altitude_lock=False,

        max_waypoints=500,
        loop_tilt_deg=0.0,
    )

    env = LogWrapper(AeroPlanaxWaypointEnv(env_params))

    rng = jax.random.PRNGKey(SEED)
    obs, log_state = env.reset(rng)

    dummy_action = {env.agents[0]: jnp.array([0, 0, 0, 0], dtype=jnp.int32)}

    # 写首帧
    env.render(log_state.env_state, env_params, {'__all__': False}, LOGDIR)
    debug_print(log_state.env_state)

    acmi_path = Path(env.filename)
    base_no_acmi = acmi_path.with_suffix("")
    run_dir = base_no_acmi.with_name(base_no_acmi.name + "_logs")
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_dir / acmi_path.stem

    # 若需要，也可一次性把整圈航点写入 Tacview
    # st = log_state.env_state
    # t0 = scalar(st.time)
    # wps = jnp.asarray(st.loop_wps) if hasattr(st, "loop_wps") else None
    # if wps is not None and wps.size > 0:
    #     for i in range(int(wps.shape[0])):
    #         write_wp_marker(env.filename, t0, i, float(wps[i,0]), float(wps[i,1]), float(wps[i,2]))
    #     write_wp_polyline(env.filename, t0, wps)

    # 诊断文件
    diag_path = str(prefix) + ".diag.txt"
    diag_f = open(diag_path, "w", encoding="utf-8")
    diag_f.write("t,dist,hdist,vt,alt,yaw_deg,d_pitch_deg,pitch_deg,roll_deg,"
                 "nz,qbar_norm,low_qbar,energy_norm,gamma_deg,vt_target,vert_phase,"
                 "cmd_head_deg,cmd_pitch_deg,cmd_vt\n")

    # 舵量 & 姿态等
    aer_f    = open(str(prefix) + ".aer.txt",   "w", encoding="utf-8")
    thr_f    = open(str(prefix) + ".thr.txt",   "w", encoding="utf-8")
    alpha_f  = open(str(prefix) + ".alpha.txt", "w", encoding="utf-8")
    beta_f   = open(str(prefix) + ".beta.txt",  "w", encoding="utf-8")
    gamma_f  = open(str(prefix) + ".gamma.txt", "w", encoding="utf-8")
    roll_f   = open(str(prefix) + ".roll.txt",  "w", encoding="utf-8")
    pitch_f  = open(str(prefix) + ".pitch.txt", "w", encoding="utf-8")
    yaw_f    = open(str(prefix) + ".yaw.txt",   "w", encoding="utf-8")

    nx_f    = open(str(prefix) + ".nx.txt", "w", encoding="utf-8")
    ny_f    = open(str(prefix) + ".ny.txt", "w", encoding="utf-8")
    nz_f    = open(str(prefix) + ".nz.txt", "w", encoding="utf-8")
    vt_f    = open(str(prefix) + ".vt.txt", "w", encoding="utf-8")
    vt_tgt_f = open(str(prefix) + ".vt_target.txt", "w", encoding="utf-8")
    roll_tgt_f = open(str(prefix) + ".roll_target.txt", "w", encoding="utf-8")
    pitch_tgt_f = open(str(prefix) + ".pitch_target.txt", "w", encoding="utf-8")
    yaw_tgt_f   = open(str(prefix) + ".yaw_target.txt",   "w", encoding="utf-8")

    yaw_err_f   = open(str(prefix) + ".yaw_err.txt",   "w", encoding="utf-8")
    pitch_err_f = open(str(prefix) + ".pitch_err.txt", "w", encoding="utf-8")
    vt_err_f    = open(str(prefix) + ".vt_err.txt",    "w", encoding="utf-8")

    # los_geom_f  = open(str(prefix) + ".los_geom.txt",  "w", encoding="utf-8"); los_geom_f.write("t_s,hdist_raw,hdist_sat,da\n")
    # pitch_cmd_f = open(str(prefix) + ".pitch_cmd_chain.txt", "w", encoding="utf-8"); pitch_cmd_f.write(
    #     "t_s,gamma_raw_deg,gamma_clip89_deg,cmd_preclip_deg,cmd_clip_deg,cmd_rate_deg,cmd_final_deg\n")

    # 表头
    aer_f.write("t_s,aileron,elevator,rudder\n")
    thr_f.write("t_s,throttle\n")
    alpha_f.write("t_s,alpha_deg\n")
    beta_f.write("t_s,beta_deg\n")
    gamma_f.write("t_s,gamma_deg\n")
    roll_f.write("t_s,roll_deg\n")
    pitch_f.write("t_s,pitch_deg\n")
    yaw_f.write("t_s,yaw_deg\n")
    nx_f.write("t_s,nx\n")
    ny_f.write("t_s,ny\n")
    nz_f.write("t_s,nz\n")
    vt_f.write("t_s,vt\n")
    vt_tgt_f.write("t_s,vt_target\n")
    roll_tgt_f.write("t_s,roll_target_deg\n")
    pitch_tgt_f.write("t_s,pitch_target_deg\n")
    yaw_tgt_f.write("t_s,yaw_target_deg\n")

    yaw_err_f.write("t_s,yaw_err_deg_from_obs\n")
    pitch_err_f.write("t_s,pitch_err_deg_from_obs\n")
    vt_err_f.write("t_s,vt_err_from_obs\n")

    # 内存缓存（画图）
    t_list = []
    aileron_list, elevator_list, rudder_list, throttle_list = [], [], [], []
    alpha_list, beta_list, gamma_list = [], [], []
    roll_list,  pitch_list,  yaw_list  = [], [], []
    nx_list, ny_list, nz_list = [], [], []
    vt_list, vt_target_list = [], []

    roll_target_list, pitch_target_list, yaw_target_list = [], [], []
    yaw_err_list, pitch_err_list, vt_err_list = [], [], []

    # los_hdist_raw_list, los_hdist_sat_list, los_da_list = [], [], []
    # los_gamma_raw_list, los_gamma_c89_list = [], []
    # cmd_preclip_list, cmd_clip_list, cmd_rate_list, cmd_final_list = [], [], [], []

    def i0(info, name, default=0.0):
        v = info.get(name, default)
        try:
            return float(jnp.ravel(jnp.asarray(v))[0])
        except Exception:
            return float(v)

    prev_reached = int(scalar(log_state.env_state.reached))

    for step in range(NUM_STEPS):
        st_prev = log_state.env_state
        prev_t   = scalar(st_prev.time)

        rng, key_step = jax.random.split(rng)
        obs, log_state, reward, done, info = env.step(key_step, log_state, dummy_action)

        env.render(log_state.env_state, env_params, {'__all__': bool(done["__all__"])}, LOGDIR)

        st = log_state.env_state
        cur_t = scalar(st.time)
        just_reset = bool(done["__all__"]) and (cur_t < prev_t)
        if just_reset:
            print("--- auto-reset detected, stop logging current episode ---")
            break

        ps = st.plane_state
        cs = st.control_state
        tnow = scalar(st.time) * env_params.agent_interaction_steps / env_params.sim_freq

        # 舵量
        try:
            thr = float(jnp.ravel(jnp.asarray(cs.throttle))[0])
            ele = float(jnp.ravel(jnp.asarray(cs.elevator))[0])
            ail = float(jnp.ravel(jnp.asarray(cs.aileron))[0])
            rud = float(jnp.ravel(jnp.asarray(cs.rudder))[0])
        except Exception:
            thr = ele = ail = rud = 0.0

        deg = 180.0 / jnp.pi
        roll_deg  = scalar(ps.roll)  * deg
        pitch_deg = scalar(ps.pitch) * deg
        yaw_deg   = scalar(ps.yaw)   * deg

        alpha_deg = scalar(ps.alpha) * deg
        beta_deg  = scalar(ps.beta)  * deg

        vx, vy, vz = scalar(ps.vel_x), scalar(ps.vel_y), scalar(ps.vel_z)
        vh = max((vx ** 2 + vy ** 2) ** 0.5, 1e-6)
        gamma_deg = float(jnp.arctan2(-vz, vh) * deg)

        nx = scalar(ps.ax)
        ny = scalar(ps.ay)
        nz = scalar(ps.az)
        vt_now = scalar(ps.vt)

        aer_f.write(f"{tnow:.4f},{ail:.6f},{ele:.6f},{rud:.6f}\n")
        thr_f.write(f"{tnow:.4f},{thr:.6f}\n")
        alpha_f.write(f"{tnow:.4f},{alpha_deg:.6f}\n")
        beta_f.write(f"{tnow:.4f},{beta_deg:.6f}\n")
        gamma_f.write(f"{tnow:.4f},{gamma_deg:.6f}\n")
        roll_f.write(f"{tnow:.4f},{roll_deg:.6f}\n")
        pitch_f.write(f"{tnow:.4f},{pitch_deg:.6f}\n")
        yaw_f.write(f"{tnow:.4f},{yaw_deg:.6f}\n")
        nx_f.write(f"{tnow:.4f},{nx:.6f}\n")
        ny_f.write(f"{tnow:.4f},{ny:.6f}\n")
        nz_f.write(f"{tnow:.4f},{nz:.6f}\n")
        vt_f.write(f"{tnow:.4f},{vt_now:.6f}\n")

        t_list.append(tnow)
        aileron_list.append(ail); elevator_list.append(ele); rudder_list.append(rud); throttle_list.append(thr)
        alpha_list.append(alpha_deg); beta_list.append(beta_deg); gamma_list.append(gamma_deg)
        roll_list.append(roll_deg); pitch_list.append(pitch_deg); yaw_list.append(yaw_deg)
        nx_list.append(nx); ny_list.append(ny); nz_list.append(nz); vt_list.append(vt_now)

        dist        = i0(info, 'dist_to_wp')
        hdist       = i0(info, 'hdist_to_wp')
        reach_flag  = bool(i0(info, 'reached_this_step'))
        reach_r     = i0(info, 'reach_radius')
        reached_cnt = int(i0(info, 'reached_count'))

        vt = scalar(st.plane_state.vt)
        alt = scalar(st.plane_state.altitude)
        print(f"dist={dist:.1f}m  hdist={hdist:.1f}m  vt={vt:.1f}  alt={alt:.1f}  "
              f"yaw={yaw_deg:.2f}°  reach_r={reach_r:.1f}  reached_cnt={reached_cnt} "
              f"{'REACHED' if reach_flag else ''}")
        debug_print(st)

        # 指令
        cmd_h = i0(info, 'cmd_heading') * (180.0/jnp.pi)
        cmd_p = i0(info, 'cmd_pitch')   * (180.0/jnp.pi)
        cmd_r = i0(info, 'cmd_roll')   * (180.0/jnp.pi)
        cmd_v = i0(info, 'cmd_vt')

        # —— 读取 env.info 里的 LOS/指令链路调试量 —— #
        def i0_nan(name):
            return info.get(name, float('nan'))

        # los_hdist_raw_list.append(float(i0_nan('dbg_hdist_raw_m')))
        # los_hdist_sat_list.append(float(i0_nan('dbg_hdist_sat_m')))
        # los_da_list.append(float(i0_nan('dbg_da_m')))

        # los_gamma_raw_list.append(float(i0_nan('dbg_gamma_los_raw_deg')))
        # los_gamma_c89_list.append(float(i0_nan('dbg_gamma_los_clip89_deg')))
        # cmd_preclip_list.append(float(i0_nan('dbg_gamma_cmd_preclip_deg')))
        # cmd_clip_list.append(float(i0_nan('dbg_gamma_cmd_clip_deg')))
        # cmd_rate_list.append(float(i0_nan('dbg_gamma_cmd_rate_deg')))
        # cmd_final_list.append(float(i0_nan('dbg_gamma_cmd_deg')))

        # los_geom_f.write(f"{tnow:.4f},{los_hdist_raw_list[-1]:.6f},{los_hdist_sat_list[-1]:.6f},{los_da_list[-1]:.6f}\n")
        # pitch_cmd_f.write(
        #     f"{tnow:.4f},{los_gamma_raw_list[-1]:.6f},{los_gamma_c89_list[-1]:.6f},"
        #     f"{cmd_preclip_list[-1]:.6f},{cmd_clip_list[-1]:.6f},{cmd_rate_list[-1]:.6f},{cmd_final_list[-1]:.6f}\n")

        # 目标写入
        vt_tgt = cmd_v
        vt_tgt_f.write(f"{tnow:.4f},{vt_tgt:.6f}\n")
        vt_target_list.append(vt_tgt)

        # roll/pitch/yaw target（roll 这里没有单独指令，先记录实际 roll）
        roll_tgt_f.write(f"{tnow:.4f},{cmd_r:.6f}\n")
        roll_target_list.append(cmd_r)
        pitch_tgt_f.write(f"{tnow:.4f},{cmd_p:.6f}\n")
        pitch_target_list.append(cmd_p)
        yaw_tgt_f.write(f"{tnow:.4f},{cmd_h:.6f}\n")
        yaw_target_list.append(cmd_h)

        # baseline 观测误差（从 env.info）
        if 'obs_norm_dheading' in info:
            yaw_err = float(jnp.ravel(jnp.asarray(info['obs_norm_dheading']))[0]) * (180.0 / jnp.pi)
            yaw_err_f.write(f"{tnow:.4f},{yaw_err:.6f}\n")
            yaw_err_list.append(yaw_err)
        if 'obs_norm_dpitch' in info:
            pitch_err = float(jnp.ravel(jnp.asarray(info['obs_norm_dpitch']))[0]) * (180.0 / jnp.pi)
            pitch_err_f.write(f"{tnow:.4f},{pitch_err:.6f}\n")
            pitch_err_list.append(pitch_err)
        if 'obs_norm_dvt' in info:
            vt_err = float(jnp.ravel(jnp.asarray(info['obs_norm_dvt']))[0]) * 340.0
            vt_err_f.write(f"{tnow:.4f},{vt_err:.6f}\n")
            vt_err_list.append(vt_err)

        # 写入诊断
        tnow2 = scalar(st.time)
        diag_f.write(f"{tnow2:.2f},{dist:.2f},{hdist:.2f},{vt:.2f},{alt:.2f},{yaw_deg:.2f},"
                     f"{cmd_p:.2f},{pitch_deg:.2f},{roll_deg:.2f},{nz:.2f},{0.0:.3f},{0},"
                     f"{0.0:.3f},{gamma_deg:.2f},{vt_tgt:.2f},{1 if env_params.use_vertical_loop else 0},"
                     f"{cmd_h:.2f},{cmd_p:.2f},{cmd_v:.2f}\n")
        if (step % 100) == 0:
            diag_f.flush()

        # reached_cnt_now = int(scalar(st.reached))
        # if reached_cnt_now > prev_reached:
        #     w = jnp.ravel(st.waypoint)
        #     write_wp_marker(env.filename, scalar(st.time), reached_cnt_now, float(w[0]), float(w[1]), float(w[2]))
        #     prev_reached = reached_cnt_now

        reached_cnt_now = int(scalar(st.reached))
        if reached_cnt_now > prev_reached:
            n = float(info['reached_wp_n'])
            e = float(info['reached_wp_e'])
            a = float(info['reached_wp_a'])
            t = float(info['time_before'])
            write_wp_marker(env.filename, t, reached_cnt_now, n, e, a)
            prev_reached = reached_cnt_now


        if bool(done["__all__"]):
            break

    # 关闭句柄
    for f in (aer_f, thr_f, alpha_f, beta_f, gamma_f, roll_f, pitch_f, yaw_f,
              nx_f, ny_f, nz_f, vt_f, vt_tgt_f, roll_tgt_f, pitch_tgt_f, yaw_tgt_f,
              yaw_err_f, pitch_err_f, vt_err_f):
        f.close()
    # los_geom_f.close()
    # pitch_cmd_f.close()
    diag_f.close()

    # 画图
    if t_list:
        # 舵量
        plt.figure(); plt.plot(t_list, aileron_list, label="aileron")
        plt.plot(t_list, elevator_list, label="elevator")
        plt.plot(t_list, rudder_list,   label="rudder")
        plt.xlabel("Time (s)"); plt.ylabel("Deflection"); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(str(prefix) + ".aer.png", dpi=200); plt.close()

        plt.figure(); plt.plot(t_list, throttle_list, label="throttle")
        plt.xlabel("Time (s)"); plt.ylabel("Throttle"); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(str(prefix) + ".thr.png", dpi=200); plt.close()

        # 姿态/角
        for name, lst, ylab in [
            ("alpha", alpha_list, "Alpha (deg)"),
            ("beta",  beta_list,  "Beta (deg)"),
            ("gamma", gamma_list, "Gamma (deg)"),
            ("roll",  roll_list,  "Roll (deg)"),
            ("pitch", pitch_list, "Pitch (deg)"),
            ("yaw",   yaw_list,   "Yaw (deg)"),
        ]:
            plt.figure(); plt.plot(t_list, lst); plt.xlabel("Time (s)"); plt.ylabel(ylab)
            plt.grid(True); plt.tight_layout(); plt.savefig(str(prefix) + f".{name}.png", dpi=200); plt.close()

        # 载荷/速度
        plt.figure(); plt.plot(t_list, nx_list, label="Nx"); plt.plot(t_list, ny_list, label="Ny"); plt.plot(t_list, nz_list, label="Nz")
        plt.xlabel("Time (s)"); plt.ylabel("Load factor / accel"); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(str(prefix) + ".nxyz.png", dpi=200); plt.close()

        plt.figure(); plt.plot(t_list, vt_list, label="vt"); plt.plot(t_list, vt_target_list, label="vt_target")
        plt.xlabel("Time (s)"); plt.ylabel("VT"); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(str(prefix) + ".vt.png", dpi=200); plt.close()

        # 姿态 vs 目标
        for name, act, tgt, ylab in [
            ("roll_target",  roll_list,  roll_target_list,  "Roll (deg)"),
            ("pitch_target", pitch_list, pitch_target_list, "Pitch (deg)"),
            ("yaw_target",   yaw_list,   yaw_target_list,   "Yaw (deg)"),
        ]:
            plt.figure(); plt.plot(t_list, act, label=name.replace("_target",""))
            plt.plot(t_list, tgt, label=name)
            plt.xlabel("Time (s)"); plt.ylabel(ylab); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(str(prefix) + f".{name}.png", dpi=200); plt.close()

        # 误差
        if yaw_err_list:
            plt.figure(); plt.plot(t_list[:len(yaw_err_list)], yaw_err_list)
            plt.xlabel("Time (s)"); plt.ylabel("Yaw error (deg)"); plt.grid(True); plt.tight_layout()
            plt.savefig(str(prefix) + ".yaw_err.png", dpi=200); plt.close()
        if pitch_err_list:
            plt.figure(); plt.plot(t_list[:len(pitch_err_list)], pitch_err_list)
            plt.xlabel("Time (s)"); plt.ylabel("Pitch error (deg)"); plt.grid(True); plt.tight_layout()
            plt.savefig(str(prefix) + ".pitch_err.png", dpi=200); plt.close()
        if vt_err_list:
            plt.figure(); plt.plot(t_list[:len(vt_err_list)], vt_err_list)
            plt.xlabel("Time (s)"); plt.ylabel("VT error (m/s)"); plt.grid(True); plt.tight_layout()
            plt.savefig(str(prefix) + ".vt_err.png", dpi=200); plt.close()

        # # Pitch 家族
        # plt.figure(figsize=(10,7))
        # plt.plot(t_list, pitch_list,           label="pitch (deg)")
        # plt.plot(t_list, pitch_target_list,    label="pitch_target/cmd (deg)")
        # plt.plot(t_list, los_gamma_raw_list,   linestyle="--", label="gamma_los_raw (deg)")
        # plt.plot(t_list, los_gamma_c89_list,   linestyle="-.", label="gamma_los_clip89 (deg)")
        # plt.plot(t_list, cmd_preclip_list,     label="cmd_preclip (deg)")
        # plt.plot(t_list, cmd_clip_list,        label="cmd_clip (deg)")
        # plt.xlabel("Time (s)"); plt.ylabel("Deg"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        # plt.savefig(str(prefix) + ".debug_pitch_family.png", dpi=200); plt.close()

        # # LOS 几何
        # plt.figure(figsize=(10,6))
        # plt.plot(t_list, los_hdist_raw_list, label="hdist_raw (m)")
        # plt.plot(t_list, los_hdist_sat_list, label="hdist_sat (m)")
        # plt.plot(t_list, los_da_list,        label="da = dz (m)")
        # plt.xlabel("Time (s)"); plt.ylabel("Meters"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        # plt.savefig(str(prefix) + ".debug_los_geometry.png", dpi=200); plt.close()

    print("Done. Tacview file:", env.filename)
    print("Diag saved:", diag_path)

if __name__ == "__main__":
    main()





##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################


# # /home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/render_waypoint_vertical_loop.py
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

# import jax
# import jax.numpy as jnp
# from pathlib import Path

# # from envs.aeroplanax_waypoint_vertical_loop import AeroPlanaxWaypointEnv, WaypointTaskParams
# from envs.aeroplanax_waypoint_vertical_loop_quaternion_version_new import (
# # from envs.aeroplanax_waypoint_vertical_loop_quaternion_version import (
#     AeroPlanaxWaypointEnv, WaypointTaskParams
# )

# from envs.wrappers import LogWrapper
# from envs.utils.utils import enu_to_geodetic

# import matplotlib.pyplot as plt

# # ====== 常量 ======
# LOGDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/tracks/"
# # BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-11-19-16-12/checkpoints/checkpoint_epoch_600" # -45°~45°的baseline
# # BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-06-01-30/checkpoints/checkpoint_epoch_800" # -80°~80°的baseline
# # BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-06-11-11/checkpoints/checkpoint_epoch_1300" # -80°~80°并随机初始化pitch的baseline
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-12-06-16-31/checkpoints/checkpoint_epoch_1300" # -89°~89°并随机初始化pitch的baseline
# BASELINE_TYPE = "rnn"  # or "lstm"
# SEED = 42
# NUM_STEPS = 900
# WP_BASE_ID = 610000  # 航点对象的高位 ID 段

# def scalar(x):
#     return float(jnp.ravel(jnp.asarray(x))[0])

# def debug_print(st):
#     n = scalar(st.plane_state.north)
#     e = scalar(st.plane_state.east)
#     a = scalar(st.plane_state.altitude)
#     vt = scalar(st.plane_state.vt)
#     yaw = scalar(st.plane_state.yaw)
#     t = scalar(st.time)
#     print(f"[t={t:6.2f}] N={n:9.2f} E={e:9.2f} Alt={a:7.2f} VT={vt:6.2f} Yaw={yaw:7.3f}")

# def write_wp_marker(filename: str, t_sec: float, idx: int, n: float, e: float, alt: float):
#     wlat, wlon, walt = enu_to_geodetic(e, n, alt, 0, 0, 0)
#     oid = WP_BASE_ID + idx
#     with open(filename, mode="a", encoding="utf-8") as f:
#         f.write(f"{t_sec:.2f},{oid},Type=ReferencePoint,Name=WP_{idx},Label=WP_{idx},Color=Yellow,Locked=1\n")
#         f.write(f"{t_sec:.2f},{oid},T={wlon}|{wlat}|{walt}|0|0|0\n")

# def write_wp_polyline(filename: str, t_sec: float, wps: jnp.ndarray):
#     pts = []
#     for i in range(int(wps.shape[0])):
#         n, e, a = float(wps[i, 0]), float(wps[i, 1]), float(wps[i, 2])
#         wlat, wlon, walt = enu_to_geodetic(e, n, a, 0, 0, 0)
#         pts.append(f"{wlon}|{wlat}|{walt}")
#     oid = WP_BASE_ID + 900000
#     with open(filename, mode="a", encoding="utf-8") as f:
#         f.write(f"{t_sec:.2f},{oid},Type=Drawing,SubType=Polyline,Label=LoopWPs,Color=Yellow,LineStyle=Solid\n")
#         f.write(f"{t_sec:.2f},{oid},Points=" + ";".join(pts) + "\n")

# def main():
#     Path(LOGDIR).mkdir(parents=True, exist_ok=True)

#     # 环境参数（开启筋斗；指令链路与 env 内一致）
#     env_params = WaypointTaskParams(
#         baseline_loaddir=BASELINE_LOADDIR,
#         baseline_type=BASELINE_TYPE,
#         use_internal_baseline=True,

#         ##################################
#         start_north=0.0,
#         start_east=0.0,
#         start_alt=10000.0,      # 精确控制初始高度
#         start_yaw_deg=0.0,     # 初始朝向（度）
#         start_vt=100.0,        # 初始空速（m/s）
#         ##################################

#         max_steps=50000000,
#         sim_freq=50,
#         agent_interaction_steps=10,   # 提高控制带宽

#         use_vertical_loop=True,
#         loop_radius=8000.0,
#         loop_points_per_circle=20,
#         loop_forward_north=1000.0,
#         loop_target_vt=250.0,

#         # 选择“正前方首点+逆时针推进”（示例）
#         loop_phase0_deg=180.0,
#         loop_direction=-1,

#         loop_enter_offset=8000.0,
#         loop_floor_margin=5000.0,
#         loop_pitch_limit_deg=85.0,

#         reach_radius_init=2000.0,
#         reach_radius_decay=1.0,

#         s_altitude_lock=False,

#         max_waypoints=500,
#         loop_tilt_deg=0.0,
#     )

#     env = LogWrapper(AeroPlanaxWaypointEnv(env_params))

#     rng = jax.random.PRNGKey(SEED)
#     obs, log_state = env.reset(rng)

#     dummy_action = {env.agents[0]: jnp.array([0, 0, 0, 0], dtype=jnp.int32)}

#     # 写首帧
#     env.render(log_state.env_state, env_params, {'__all__': False}, LOGDIR)
#     debug_print(log_state.env_state)

#     acmi_path = Path(env.filename)
#     base_no_acmi = acmi_path.with_suffix("")
#     run_dir = base_no_acmi.with_name(base_no_acmi.name + "_logs")
#     run_dir.mkdir(parents=True, exist_ok=True)
#     prefix = run_dir / acmi_path.stem

#     # 若需要，也可一次性把整圈航点写入 Tacview
#     # st = log_state.env_state
#     # t0 = scalar(st.time)
#     # wps = jnp.asarray(st.loop_wps) if hasattr(st, "loop_wps") else None
#     # if wps is not None and wps.size > 0:
#     #     for i in range(int(wps.shape[0])):
#     #         write_wp_marker(env.filename, t0, i, float(wps[i,0]), float(wps[i,1]), float(wps[i,2]))
#     #     write_wp_polyline(env.filename, t0, wps)

#     # 诊断文件
#     diag_path = str(prefix) + ".diag.txt"
#     diag_f = open(diag_path, "w", encoding="utf-8")
#     diag_f.write("t,dist,hdist,vt,alt,yaw_deg,d_pitch_deg,pitch_deg,roll_deg,"
#                  "nz,qbar_norm,low_qbar,energy_norm,gamma_deg,vt_target,vert_phase,"
#                  "cmd_head_deg,cmd_pitch_deg,cmd_vt\n")

#     # 舵量 & 姿态等
#     aer_f    = open(str(prefix) + ".aer.txt",   "w", encoding="utf-8")
#     thr_f    = open(str(prefix) + ".thr.txt",   "w", encoding="utf-8")
#     alpha_f  = open(str(prefix) + ".alpha.txt", "w", encoding="utf-8")
#     beta_f   = open(str(prefix) + ".beta.txt",  "w", encoding="utf-8")
#     gamma_f  = open(str(prefix) + ".gamma.txt", "w", encoding="utf-8")
#     roll_f   = open(str(prefix) + ".roll.txt",  "w", encoding="utf-8")
#     pitch_f  = open(str(prefix) + ".pitch.txt", "w", encoding="utf-8")
#     yaw_f    = open(str(prefix) + ".yaw.txt",   "w", encoding="utf-8")

#     nx_f    = open(str(prefix) + ".nx.txt", "w", encoding="utf-8")
#     ny_f    = open(str(prefix) + ".ny.txt", "w", encoding="utf-8")
#     nz_f    = open(str(prefix) + ".nz.txt", "w", encoding="utf-8")
#     vt_f    = open(str(prefix) + ".vt.txt", "w", encoding="utf-8")
#     vt_tgt_f = open(str(prefix) + ".vt_target.txt", "w", encoding="utf-8")
#     roll_tgt_f = open(str(prefix) + ".roll_target.txt", "w", encoding="utf-8")
#     pitch_tgt_f = open(str(prefix) + ".pitch_target.txt", "w", encoding="utf-8")
#     yaw_tgt_f   = open(str(prefix) + ".yaw_target.txt",   "w", encoding="utf-8")

#     yaw_err_f   = open(str(prefix) + ".yaw_err.txt",   "w", encoding="utf-8")
#     pitch_err_f = open(str(prefix) + ".pitch_err.txt", "w", encoding="utf-8")
#     vt_err_f    = open(str(prefix) + ".vt_err.txt",    "w", encoding="utf-8")

#     # los_geom_f  = open(str(prefix) + ".los_geom.txt",  "w", encoding="utf-8"); los_geom_f.write("t_s,hdist_raw,hdist_sat,da\n")
#     # pitch_cmd_f = open(str(prefix) + ".pitch_cmd_chain.txt", "w", encoding="utf-8"); pitch_cmd_f.write(
#     #     "t_s,gamma_raw_deg,gamma_clip89_deg,cmd_preclip_deg,cmd_clip_deg,cmd_rate_deg,cmd_final_deg\n")

#     # 表头
#     aer_f.write("t_s,aileron,elevator,rudder\n")
#     thr_f.write("t_s,throttle\n")
#     alpha_f.write("t_s,alpha_deg\n")
#     beta_f.write("t_s,beta_deg\n")
#     gamma_f.write("t_s,gamma_deg\n")
#     roll_f.write("t_s,roll_deg\n")
#     pitch_f.write("t_s,pitch_deg\n")
#     yaw_f.write("t_s,yaw_deg\n")
#     nx_f.write("t_s,nx\n")
#     ny_f.write("t_s,ny\n")
#     nz_f.write("t_s,nz\n")
#     vt_f.write("t_s,vt\n")
#     vt_tgt_f.write("t_s,vt_target\n")
#     roll_tgt_f.write("t_s,roll_target_deg\n")
#     pitch_tgt_f.write("t_s,pitch_target_deg\n")
#     yaw_tgt_f.write("t_s,yaw_target_deg\n")

#     yaw_err_f.write("t_s,yaw_err_deg_from_obs\n")
#     pitch_err_f.write("t_s,pitch_err_deg_from_obs\n")
#     vt_err_f.write("t_s,vt_err_from_obs\n")

#     # 内存缓存（画图）
#     t_list = []
#     aileron_list, elevator_list, rudder_list, throttle_list = [], [], [], []
#     alpha_list, beta_list, gamma_list = [], [], []
#     roll_list,  pitch_list,  yaw_list  = [], [], []
#     nx_list, ny_list, nz_list = [], [], []
#     vt_list, vt_target_list = [], []

#     roll_target_list, pitch_target_list, yaw_target_list = [], [], []
#     yaw_err_list, pitch_err_list, vt_err_list = [], [], []

#     # los_hdist_raw_list, los_hdist_sat_list, los_da_list = [], [], []
#     # los_gamma_raw_list, los_gamma_c89_list = [], []
#     # cmd_preclip_list, cmd_clip_list, cmd_rate_list, cmd_final_list = [], [], [], []

#     def i0(info, name, default=0.0):
#         v = info.get(name, default)
#         try:
#             return float(jnp.ravel(jnp.asarray(v))[0])
#         except Exception:
#             return float(v)

#     prev_reached = int(scalar(log_state.env_state.reached))

#     for step in range(NUM_STEPS):
#         st_prev = log_state.env_state
#         prev_t   = scalar(st_prev.time)

#         rng, key_step = jax.random.split(rng)
#         obs, log_state, reward, done, info = env.step(key_step, log_state, dummy_action)

#         env.render(log_state.env_state, env_params, {'__all__': bool(done["__all__"])}, LOGDIR)

#         st = log_state.env_state
#         cur_t = scalar(st.time)
#         just_reset = bool(done["__all__"]) and (cur_t < prev_t)
#         if just_reset:
#             print("--- auto-reset detected, stop logging current episode ---")
#             break

#         ps = st.plane_state
#         cs = st.control_state
#         tnow = scalar(st.time) * env_params.agent_interaction_steps / env_params.sim_freq

#         # 舵量
#         try:
#             thr = float(jnp.ravel(jnp.asarray(cs.throttle))[0])
#             ele = float(jnp.ravel(jnp.asarray(cs.elevator))[0])
#             ail = float(jnp.ravel(jnp.asarray(cs.aileron))[0])
#             rud = float(jnp.ravel(jnp.asarray(cs.rudder))[0])
#         except Exception:
#             thr = ele = ail = rud = 0.0

#         deg = 180.0 / jnp.pi
#         roll_deg  = scalar(ps.roll)  * deg
#         pitch_deg = scalar(ps.pitch) * deg
#         yaw_deg   = scalar(ps.yaw)   * deg

#         alpha_deg = scalar(ps.alpha) * deg
#         beta_deg  = scalar(ps.beta)  * deg

#         vx, vy, vz = scalar(ps.vel_x), scalar(ps.vel_y), scalar(ps.vel_z)
#         vh = max((vx ** 2 + vy ** 2) ** 0.5, 1e-6)
#         gamma_deg = float(jnp.arctan2(-vz, vh) * deg)

#         nx = scalar(ps.ax)
#         ny = scalar(ps.ay)
#         nz = scalar(ps.az)
#         vt_now = scalar(ps.vt)

#         aer_f.write(f"{tnow:.4f},{ail:.6f},{ele:.6f},{rud:.6f}\n")
#         thr_f.write(f"{tnow:.4f},{thr:.6f}\n")
#         alpha_f.write(f"{tnow:.4f},{alpha_deg:.6f}\n")
#         beta_f.write(f"{tnow:.4f},{beta_deg:.6f}\n")
#         gamma_f.write(f"{tnow:.4f},{gamma_deg:.6f}\n")
#         roll_f.write(f"{tnow:.4f},{roll_deg:.6f}\n")
#         pitch_f.write(f"{tnow:.4f},{pitch_deg:.6f}\n")
#         yaw_f.write(f"{tnow:.4f},{yaw_deg:.6f}\n")
#         nx_f.write(f"{tnow:.4f},{nx:.6f}\n")
#         ny_f.write(f"{tnow:.4f},{ny:.6f}\n")
#         nz_f.write(f"{tnow:.4f},{nz:.6f}\n")
#         vt_f.write(f"{tnow:.4f},{vt_now:.6f}\n")

#         t_list.append(tnow)
#         aileron_list.append(ail); elevator_list.append(ele); rudder_list.append(rud); throttle_list.append(thr)
#         alpha_list.append(alpha_deg); beta_list.append(beta_deg); gamma_list.append(gamma_deg)
#         roll_list.append(roll_deg); pitch_list.append(pitch_deg); yaw_list.append(yaw_deg)
#         nx_list.append(nx); ny_list.append(ny); nz_list.append(nz); vt_list.append(vt_now)

#         dist        = i0(info, 'dist_to_wp')
#         hdist       = i0(info, 'hdist_to_wp')
#         reach_flag  = bool(i0(info, 'reached_this_step'))
#         reach_r     = i0(info, 'reach_radius')
#         reached_cnt = int(i0(info, 'reached_count'))

#         vt = scalar(st.plane_state.vt)
#         alt = scalar(st.plane_state.altitude)
#         print(f"dist={dist:.1f}m  hdist={hdist:.1f}m  vt={vt:.1f}  alt={alt:.1f}  "
#               f"yaw={yaw_deg:.2f}°  reach_r={reach_r:.1f}  reached_cnt={reached_cnt} "
#               f"{'REACHED' if reach_flag else ''}")
#         debug_print(st)

#         # 指令
#         cmd_h = i0(info, 'cmd_heading') * (180.0/jnp.pi)
#         cmd_p = i0(info, 'cmd_pitch')   * (180.0/jnp.pi)
#         cmd_v = i0(info, 'cmd_vt')

#         # —— 读取 env.info 里的 LOS/指令链路调试量 —— #
#         def i0_nan(name):
#             return info.get(name, float('nan'))

#         # los_hdist_raw_list.append(float(i0_nan('dbg_hdist_raw_m')))
#         # los_hdist_sat_list.append(float(i0_nan('dbg_hdist_sat_m')))
#         # los_da_list.append(float(i0_nan('dbg_da_m')))

#         # los_gamma_raw_list.append(float(i0_nan('dbg_gamma_los_raw_deg')))
#         # los_gamma_c89_list.append(float(i0_nan('dbg_gamma_los_clip89_deg')))
#         # cmd_preclip_list.append(float(i0_nan('dbg_gamma_cmd_preclip_deg')))
#         # cmd_clip_list.append(float(i0_nan('dbg_gamma_cmd_clip_deg')))
#         # cmd_rate_list.append(float(i0_nan('dbg_gamma_cmd_rate_deg')))
#         # cmd_final_list.append(float(i0_nan('dbg_gamma_cmd_deg')))

#         # los_geom_f.write(f"{tnow:.4f},{los_hdist_raw_list[-1]:.6f},{los_hdist_sat_list[-1]:.6f},{los_da_list[-1]:.6f}\n")
#         # pitch_cmd_f.write(
#         #     f"{tnow:.4f},{los_gamma_raw_list[-1]:.6f},{los_gamma_c89_list[-1]:.6f},"
#         #     f"{cmd_preclip_list[-1]:.6f},{cmd_clip_list[-1]:.6f},{cmd_rate_list[-1]:.6f},{cmd_final_list[-1]:.6f}\n")

#         # 目标写入
#         vt_tgt = cmd_v
#         vt_tgt_f.write(f"{tnow:.4f},{vt_tgt:.6f}\n")
#         vt_target_list.append(vt_tgt)

#         # roll/pitch/yaw target（roll 这里没有单独指令，先记录实际 roll）
#         roll_tgt_f.write(f"{tnow:.4f},{roll_deg:.6f}\n")
#         roll_target_list.append(roll_deg)
#         pitch_tgt_f.write(f"{tnow:.4f},{cmd_p:.6f}\n")
#         pitch_target_list.append(cmd_p)
#         yaw_tgt_f.write(f"{tnow:.4f},{cmd_h:.6f}\n")
#         yaw_target_list.append(cmd_h)

#         # baseline 观测误差（从 env.info）
#         if 'obs_norm_dheading' in info:
#             yaw_err = float(jnp.ravel(jnp.asarray(info['obs_norm_dheading']))[0]) * (180.0 / jnp.pi)
#             yaw_err_f.write(f"{tnow:.4f},{yaw_err:.6f}\n")
#             yaw_err_list.append(yaw_err)
#         if 'obs_norm_dpitch' in info:
#             pitch_err = float(jnp.ravel(jnp.asarray(info['obs_norm_dpitch']))[0]) * (180.0 / jnp.pi)
#             pitch_err_f.write(f"{tnow:.4f},{pitch_err:.6f}\n")
#             pitch_err_list.append(pitch_err)
#         if 'obs_norm_dvt' in info:
#             vt_err = float(jnp.ravel(jnp.asarray(info['obs_norm_dvt']))[0]) * 340.0
#             vt_err_f.write(f"{tnow:.4f},{vt_err:.6f}\n")
#             vt_err_list.append(vt_err)

#         # 写入诊断
#         tnow2 = scalar(st.time)
#         diag_f.write(f"{tnow2:.2f},{dist:.2f},{hdist:.2f},{vt:.2f},{alt:.2f},{yaw_deg:.2f},"
#                      f"{cmd_p:.2f},{pitch_deg:.2f},{roll_deg:.2f},{nz:.2f},{0.0:.3f},{0},"
#                      f"{0.0:.3f},{gamma_deg:.2f},{vt_tgt:.2f},{1 if env_params.use_vertical_loop else 0},"
#                      f"{cmd_h:.2f},{cmd_p:.2f},{cmd_v:.2f}\n")
#         if (step % 100) == 0:
#             diag_f.flush()

#         # reached_cnt_now = int(scalar(st.reached))
#         # if reached_cnt_now > prev_reached:
#         #     w = jnp.ravel(st.waypoint)
#         #     write_wp_marker(env.filename, scalar(st.time), reached_cnt_now, float(w[0]), float(w[1]), float(w[2]))
#         #     prev_reached = reached_cnt_now

#         reached_cnt_now = int(scalar(st.reached))
#         if reached_cnt_now > prev_reached:
#             n = float(info['reached_wp_n'])
#             e = float(info['reached_wp_e'])
#             a = float(info['reached_wp_a'])
#             t = float(info['time_before'])
#             write_wp_marker(env.filename, t, reached_cnt_now, n, e, a)
#             prev_reached = reached_cnt_now


#         if bool(done["__all__"]):
#             break

#     # 关闭句柄
#     for f in (aer_f, thr_f, alpha_f, beta_f, gamma_f, roll_f, pitch_f, yaw_f,
#               nx_f, ny_f, nz_f, vt_f, vt_tgt_f, roll_tgt_f, pitch_tgt_f, yaw_tgt_f,
#               yaw_err_f, pitch_err_f, vt_err_f):
#         f.close()
#     # los_geom_f.close()
#     # pitch_cmd_f.close()
#     diag_f.close()

#     # 画图
#     if t_list:
#         # 舵量
#         plt.figure(); plt.plot(t_list, aileron_list, label="aileron")
#         plt.plot(t_list, elevator_list, label="elevator")
#         plt.plot(t_list, rudder_list,   label="rudder")
#         plt.xlabel("Time (s)"); plt.ylabel("Deflection"); plt.legend(); plt.grid(True); plt.tight_layout()
#         plt.savefig(str(prefix) + ".aer.png", dpi=200); plt.close()

#         plt.figure(); plt.plot(t_list, throttle_list, label="throttle")
#         plt.xlabel("Time (s)"); plt.ylabel("Throttle"); plt.legend(); plt.grid(True); plt.tight_layout()
#         plt.savefig(str(prefix) + ".thr.png", dpi=200); plt.close()

#         # 姿态/角
#         for name, lst, ylab in [
#             ("alpha", alpha_list, "Alpha (deg)"),
#             ("beta",  beta_list,  "Beta (deg)"),
#             ("gamma", gamma_list, "Gamma (deg)"),
#             ("roll",  roll_list,  "Roll (deg)"),
#             ("pitch", pitch_list, "Pitch (deg)"),
#             ("yaw",   yaw_list,   "Yaw (deg)"),
#         ]:
#             plt.figure(); plt.plot(t_list, lst); plt.xlabel("Time (s)"); plt.ylabel(ylab)
#             plt.grid(True); plt.tight_layout(); plt.savefig(str(prefix) + f".{name}.png", dpi=200); plt.close()

#         # 载荷/速度
#         plt.figure(); plt.plot(t_list, nx_list, label="Nx"); plt.plot(t_list, ny_list, label="Ny"); plt.plot(t_list, nz_list, label="Nz")
#         plt.xlabel("Time (s)"); plt.ylabel("Load factor / accel"); plt.legend(); plt.grid(True); plt.tight_layout()
#         plt.savefig(str(prefix) + ".nxyz.png", dpi=200); plt.close()

#         plt.figure(); plt.plot(t_list, vt_list, label="vt"); plt.plot(t_list, vt_target_list, label="vt_target")
#         plt.xlabel("Time (s)"); plt.ylabel("VT"); plt.legend(); plt.grid(True); plt.tight_layout()
#         plt.savefig(str(prefix) + ".vt.png", dpi=200); plt.close()

#         # 姿态 vs 目标
#         for name, act, tgt, ylab in [
#             ("roll_target",  roll_list,  roll_target_list,  "Roll (deg)"),
#             ("pitch_target", pitch_list, pitch_target_list, "Pitch (deg)"),
#             ("yaw_target",   yaw_list,   yaw_target_list,   "Yaw (deg)"),
#         ]:
#             plt.figure(); plt.plot(t_list, act, label=name.replace("_target",""))
#             plt.plot(t_list, tgt, label=name)
#             plt.xlabel("Time (s)"); plt.ylabel(ylab); plt.legend(); plt.grid(True); plt.tight_layout()
#             plt.savefig(str(prefix) + f".{name}.png", dpi=200); plt.close()

#         # 误差
#         if yaw_err_list:
#             plt.figure(); plt.plot(t_list[:len(yaw_err_list)], yaw_err_list)
#             plt.xlabel("Time (s)"); plt.ylabel("Yaw error (deg)"); plt.grid(True); plt.tight_layout()
#             plt.savefig(str(prefix) + ".yaw_err.png", dpi=200); plt.close()
#         if pitch_err_list:
#             plt.figure(); plt.plot(t_list[:len(pitch_err_list)], pitch_err_list)
#             plt.xlabel("Time (s)"); plt.ylabel("Pitch error (deg)"); plt.grid(True); plt.tight_layout()
#             plt.savefig(str(prefix) + ".pitch_err.png", dpi=200); plt.close()
#         if vt_err_list:
#             plt.figure(); plt.plot(t_list[:len(vt_err_list)], vt_err_list)
#             plt.xlabel("Time (s)"); plt.ylabel("VT error (m/s)"); plt.grid(True); plt.tight_layout()
#             plt.savefig(str(prefix) + ".vt_err.png", dpi=200); plt.close()

#         # # Pitch 家族
#         # plt.figure(figsize=(10,7))
#         # plt.plot(t_list, pitch_list,           label="pitch (deg)")
#         # plt.plot(t_list, pitch_target_list,    label="pitch_target/cmd (deg)")
#         # plt.plot(t_list, los_gamma_raw_list,   linestyle="--", label="gamma_los_raw (deg)")
#         # plt.plot(t_list, los_gamma_c89_list,   linestyle="-.", label="gamma_los_clip89 (deg)")
#         # plt.plot(t_list, cmd_preclip_list,     label="cmd_preclip (deg)")
#         # plt.plot(t_list, cmd_clip_list,        label="cmd_clip (deg)")
#         # plt.xlabel("Time (s)"); plt.ylabel("Deg"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#         # plt.savefig(str(prefix) + ".debug_pitch_family.png", dpi=200); plt.close()

#         # # LOS 几何
#         # plt.figure(figsize=(10,6))
#         # plt.plot(t_list, los_hdist_raw_list, label="hdist_raw (m)")
#         # plt.plot(t_list, los_hdist_sat_list, label="hdist_sat (m)")
#         # plt.plot(t_list, los_da_list,        label="da = dz (m)")
#         # plt.xlabel("Time (s)"); plt.ylabel("Meters"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
#         # plt.savefig(str(prefix) + ".debug_los_geometry.png", dpi=200); plt.close()

#     print("Done. Tacview file:", env.filename)
#     print("Diag saved:", diag_path)

# if __name__ == "__main__":
#     main()
