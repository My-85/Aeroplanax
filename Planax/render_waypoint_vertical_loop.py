# Planax/render_waypoint.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
from pathlib import Path

from envs.aeroplanax_waypoint_vertical_loop import AeroPlanaxWaypointEnv, WaypointTaskParams
from envs.wrappers import LogWrapper
from envs.utils.utils import enu_to_geodetic

import matplotlib.pyplot as plt  # 新增：用于画图


# ====== 常量 ======
LOGDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/tracks/"
BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/2v2_lczh_mine/AeroPlanax_multi_combat_2v2/envs/models/RNN新策略/PPO+RNN(actor与critic均加了layer_norm且改进训练过程（暖启动+防梯度消失+reward_clip)/seed42/heading_pitch_V_discrete_rnn_2025-08-29-15-55/checkpoints/checkpoint_epoch_600"
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-09-17-01-44/checkpoints/checkpoint_epoch_1000"
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-09-17-17-12/checkpoints/checkpoint_epoch_1400"
# BASELINE_LOADDIR = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-09-18-17-07/checkpoints/checkpoint_epoch_1600"
BASELINE_TYPE = "rnn"  # or "lstm"
SEED = 42
NUM_STEPS = 10000
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

# def write_wp_marker(filename: str, t_sec: float, idx: int, n: float, e: float, alt: float):
#     # 统一：先创建对象（Type=ReferencePoint），再写位置
#     wlat, wlon, walt = enu_to_geodetic(e, n, alt, 0, 0, 0)
#     oid = WP_BASE_ID + idx
#     with open(filename, mode="a", encoding="utf-8") as f:
#         f.write(f"{t_sec:.2f},{oid},Type=ReferencePoint,Name=WP_{idx},Label=WP_{idx},Color=Yellow\n")
#         f.write(f"{t_sec:.2f},{oid},T={wlon}|{wlat}|{walt}|0|0|0\n")


def write_wp_marker(filename: str, t_sec: float, idx: int, n: float, e: float, alt: float):
    wlat, wlon, walt = enu_to_geodetic(e, n, alt, 0, 0, 0)
    oid = WP_BASE_ID + idx
    with open(filename, mode="a", encoding="utf-8") as f:
        # 明确创建“参考点”并加标签
        f.write(f"{t_sec:.2f},{oid},Type=ReferencePoint,Name=WP_{idx},Label=WP_{idx},Color=Yellow,Locked=1\n")
        f.write(f"{t_sec:.2f},{oid},T={wlon}|{wlat}|{walt}|0|0|0\n")

def write_wp_polyline(filename: str, t_sec: float, wps: jnp.ndarray):
    # 把所有航点连成一条可见的折线
    pts = []
    for i in range(int(wps.shape[0])):
        n, e, a = float(wps[i, 0]), float(wps[i, 1]), float(wps[i, 2])
        wlat, wlon, walt = enu_to_geodetic(e, n, a, 0, 0, 0)
        pts.append(f"{wlon}|{wlat}|{walt}")
    oid = WP_BASE_ID + 900000  # 单独的绘图对象 ID
    with open(filename, mode="a", encoding="utf-8") as f:
        f.write(f"{t_sec:.2f},{oid},Type=Drawing,SubType=Polyline,Label=LoopWPs,Color=Yellow,LineStyle=Solid\n")
        f.write(f"{t_sec:.2f},{oid},Points=" + ";".join(pts) + "\n")

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
        loop_points_per_circle=20,     # 400 -> 180（每步角度 ~2°）
        loop_forward_north=1000.0,
        loop_target_vt=270.0,

        # 正确的首点与方向
        loop_phase0_deg=180.0,   # 首航点在圆的“正前方” (与飞机同高，俯仰≈0)
        loop_direction=-1,       # 顺时针推进：φ = π, π-dφ, π-2dφ, ...（先“前方”，再“前上方”，逐步抬升）

        # # 正确的首点与方向
        # loop_phase0_deg=0.0,    # φ=0 → 首航点就在机头正前方（n=n0，alt=a0）
        # loop_direction=+1,      # 相位正向推进：先前上方，再到顶点

        loop_enter_offset=3000.0,     # ← 新增：进入前移 3km（可按你图上需要调大/调小）
        loop_floor_margin=6000.0,     # ← 新增：最低点 ≥ min_altitude + 1000m + 5000m

        loop_pitch_limit_deg=85.0, # 俯仰限幅
       
        # 航点到达判定（可按需调小/调大）
        reach_radius_init=500.0,        # 2000 -> 1000    
        reach_radius_decay=1.0,

        # 筋斗需要跟随高度变化，建议关闭锁高（如果你的 env 里还有 s_altitude_lock 判断）
        s_altitude_lock=False,

        max_waypoints=2000,
        loop_tilt_deg=0.0, # 让筋斗左倾 45°
        
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

    # === 新增：本次运行的所有 txt/png 放入专用子目录 ===
    acmi_path = Path(env.filename)              # .../xxx.txt.acmi
    base_no_acmi = acmi_path.with_suffix("")    # 去掉 .acmi
    run_dir = base_no_acmi.with_name(base_no_acmi.name + "_logs")
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_dir / acmi_path.stem           # 子目录内公共前缀


    # # 写入首个航点 Marker（Tacview）
    # st = log_state.env_state
    # w = jnp.ravel(st.waypoint)
    # wp_n, wp_e, wp_alt = float(w[0]), float(w[1]), float(w[2])
    # t0 = scalar(st.time)
    # write_wp_marker(env.filename, t0, 0, wp_n, wp_e, wp_alt)

    # 写入预计算的全部航点（如无 loop_wps，则退化为只写当前航点）
    st = log_state.env_state
    t0 = scalar(st.time)
    # if hasattr(st, "loop_wps"):
    #     wps = jnp.asarray(st.loop_wps)
    #     N = int(wps.shape[0])
    #     for i in range(N):
    #         wp_n, wp_e, wp_alt = float(wps[i, 0]), float(wps[i, 1]), float(wps[i, 2])
    #         write_wp_marker(env.filename, t0, i, wp_n, wp_e, wp_alt)
    #     write_wp_polyline(env.filename, t0, wps)  # 可选：把所有航点连成线
    # else:
    #     w = jnp.ravel(st.waypoint)
    #     wp_n, wp_e, wp_alt = float(w[0]), float(w[1]), float(w[2])
    #     write_wp_marker(env.filename, t0, 0, wp_n, wp_e, wp_alt)

    # === 诊断日志文件 ===
    diag_path = str(prefix) + ".diag.txt"
    diag_f = open(diag_path, "w", encoding="utf-8")
    diag_f.write("t,dist,hdist,vt,alt,yaw_deg,d_pitch_deg,pitch_deg,roll_deg,"
                 "nz,qbar_norm,low_qbar,energy_norm,gamma_deg,vt_target,vert_phase,"
                 "cmd_head_deg,cmd_pitch_deg,cmd_vt\n")
    # prev_reached = int(scalar(log_state.env_state.reached))
    # === 新增：各量单独日志文件 ===
    aer_path    = str(prefix) + ".aer.txt"      # aileron / elevator / rudder
    thr_path    = str(prefix) + ".thr.txt"      # throttle
    alpha_path  = str(prefix) + ".alpha.txt"
    beta_path   = str(prefix) + ".beta.txt"
    gamma_path  = str(prefix) + ".gamma.txt"    # 轨迹倾角 gamma
    roll_path   = str(prefix) + ".roll.txt"
    pitch_path  = str(prefix) + ".pitch.txt"
    yaw_path    = str(prefix) + ".yaw.txt"

    aer_f    = open(aer_path,   "w", encoding="utf-8")
    thr_f    = open(thr_path,   "w", encoding="utf-8")
    alpha_f  = open(alpha_path, "w", encoding="utf-8")
    beta_f   = open(beta_path,  "w", encoding="utf-8")
    gamma_f  = open(gamma_path, "w", encoding="utf-8")
    roll_f   = open(roll_path,  "w", encoding="utf-8")
    pitch_f  = open(pitch_path, "w", encoding="utf-8")
    yaw_f    = open(yaw_path,   "w", encoding="utf-8")

    nx_path    = str(prefix) + ".nx.txt"
    ny_path    = str(prefix) + ".ny.txt"
    nz_path    = str(prefix) + ".nz.txt"
    vt_path    = str(prefix) + ".vt.txt"
    vt_tgt_path = str(prefix) + ".vt_target.txt"
    roll_tgt_path = str(prefix) + ".roll_target.txt"
    pitch_tgt_path = str(prefix) + ".pitch_target.txt"
    yaw_tgt_path   = str(prefix) + ".yaw_target.txt"

    nx_f    = open(nx_path, "w", encoding="utf-8")
    ny_f    = open(ny_path, "w", encoding="utf-8")
    nz_f    = open(nz_path, "w", encoding="utf-8")
    vt_f    = open(vt_path, "w", encoding="utf-8")
    vt_tgt_f = open(vt_tgt_path, "w", encoding="utf-8")
    roll_tgt_f = open(roll_tgt_path, "w", encoding="utf-8")
    pitch_tgt_f = open(pitch_tgt_path, "w", encoding="utf-8")
    yaw_tgt_f   = open(yaw_tgt_path, "w", encoding="utf-8")

    yaw_err_path   = str(prefix) + ".yaw_err.txt"
    pitch_err_path = str(prefix) + ".pitch_err.txt"
    vt_err_path    = str(prefix) + ".vt_err.txt"

    yaw_err_f   = open(yaw_err_path, "w", encoding="utf-8")
    pitch_err_f = open(pitch_err_path, "w", encoding="utf-8")
    vt_err_f    = open(vt_err_path, "w", encoding="utf-8")

    los_geom_path  = str(prefix) + ".los_geom.txt"
    pitch_cmd_path = str(prefix) + ".pitch_cmd_chain.txt"
    los_geom_f  = open(los_geom_path,  "w", encoding="utf-8"); los_geom_f.write("t_s,hdist_raw,hdist_sat,da\n")
    pitch_cmd_f = open(pitch_cmd_path, "w", encoding="utf-8"); pitch_cmd_f.write(
        "t_s,gamma_raw_deg,gamma_clip89_deg,cmd_preclip_deg,cmd_clip_deg,cmd_rate_deg,cmd_final_deg\n")


    yaw_err_f.write("t_s,yaw_err_deg_from_obs\n")
    pitch_err_f.write("t_s,pitch_err_deg_from_obs\n")
    vt_err_f.write("t_s,vt_err_from_obs\n")


    # 写表头（时间统一用实际物理时间：time * agent_interaction_steps / sim_freq）
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

    # === 新增：为画图准备的内存缓存 ===
    t_list = []

    aileron_list  = []
    elevator_list = []
    rudder_list   = []
    throttle_list = []

    alpha_list = []
    beta_list  = []
    gamma_list = []

    roll_list  = []
    pitch_list = []
    yaw_list   = []
    nx_list, ny_list, nz_list = [], [], []
    vt_list, vt_target_list = [], []
    roll_target_list, pitch_target_list, yaw_target_list = [], [], []
    yaw_err_list, pitch_err_list, vt_err_list = [], [], []
    # —— 新增：LOS/指令链路调试曲线 —— #
    los_hdist_raw_list = []
    los_hdist_sat_list = []
    los_da_list        = []
    los_gamma_raw_list = []
    los_gamma_c89_list = []
    cmd_preclip_list   = []
    cmd_clip_list      = []
    cmd_rate_list      = []
    cmd_final_list     = []

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
        with open(str(prefix) + ".precrash.log", "a", encoding="utf-8") as f:
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

        #############################################################
        # ===== 判定是否为 auto-reset 的第一步 =====
        # 若 done=True 且 当前 time 比上一步小，说明底层环境已经自动 reset，
        # 此时 st 是新一轮的初始状态，这一帧不应写入本次轨迹日志和画图数据。
        cur_t = scalar(st.time)
        just_reset = bool(done["__all__"]) and (cur_t < prev_t)
        if just_reset:
            print("--- auto-reset detected, stop logging current episode ---")
            break

        #############################################################

        # === 新增：从状态中取舵量和姿态，实时记录到 txt ===
        ps = st.plane_state
        cs = st.control_state

        # 物理时间：与 Tacview 保持一致
        tnow = scalar(st.time) * env_params.agent_interaction_steps / env_params.sim_freq

        # 舵量（单机体：取索引 0）
        try:
            thr = scalar(cs.throttle)
            ele = scalar(cs.elevator)
            ail = scalar(cs.aileron)
            rud = scalar(cs.rudder)
        except Exception:
            # 兜底：control_state 形状异常时不写
            thr = ele = ail = rud = 0.0

        # 姿态 & 气动角
        deg = 180.0 / jnp.pi
        roll_deg  = scalar(ps.roll)  * deg
        pitch_deg = scalar(ps.pitch) * deg
        yaw_deg   = scalar(ps.yaw)   * deg

        alpha_deg = scalar(ps.alpha) * deg
        beta_deg  = scalar(ps.beta)  * deg

        # 轨迹倾角 gamma（作为“攻角”第三条）：gamma = atan2(-vz, sqrt(vx^2+vy^2))
        vx = scalar(ps.vel_x)
        vy = scalar(ps.vel_y)
        vz = scalar(ps.vel_z)
        vh = max((vx ** 2 + vy ** 2) ** 0.5, 1e-6)
        gamma_deg = float(jnp.arctan2(-vz, vh) * deg)

        # nx, ny, nz（这里直接用 ax/ay/az；若你那边是 m/s^2 想转 g，自行除以 9.81）
        nx = scalar(ps.ax)
        ny = scalar(ps.ay)
        nz = scalar(ps.az)

        # 空速
        vt_now = scalar(ps.vt)

        # --- 写入各自 txt ---
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

        # --- 同步到内存列表，用于画图 ---
        t_list.append(tnow)

        aileron_list.append(ail)
        elevator_list.append(ele)
        rudder_list.append(rud)
        throttle_list.append(thr)

        alpha_list.append(alpha_deg)
        beta_list.append(beta_deg)
        gamma_list.append(gamma_deg)

        roll_list.append(roll_deg)
        pitch_list.append(pitch_deg)
        yaw_list.append(yaw_deg)

        nx_list.append(nx)
        ny_list.append(ny)
        nz_list.append(nz)
        vt_list.append(vt_now)

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

        #=============================================================================================#
        # === 写 vt_target & 姿态 target 到 txt + 缓存 ===

        # —— 读取 env.info 里的 LOS/指令链路调试量 —— #
        def i0_nan(name):
            return i0(name) if name in info else float('nan')

        los_hdist_raw_list.append(i0_nan('dbg_hdist_raw_m'))
        los_hdist_sat_list.append(i0_nan('dbg_hdist_sat_m'))
        los_da_list.append(i0_nan('dbg_da_m'))

        los_gamma_raw_list.append(i0_nan('dbg_gamma_los_raw_deg'))
        los_gamma_c89_list.append(i0_nan('dbg_gamma_los_clip89_deg'))

        cmd_preclip_list.append(i0_nan('dbg_gamma_cmd_preclip_deg'))
        cmd_clip_list.append(i0_nan('dbg_gamma_cmd_clip_deg'))
        cmd_rate_list.append(i0_nan('dbg_gamma_cmd_rate_deg'))
        cmd_final_list.append(i0_nan('dbg_gamma_cmd_deg'))

        los_geom_f.write(f"{tnow:.4f},{los_hdist_raw_list[-1]:.6f},{los_hdist_sat_list[-1]:.6f},{los_da_list[-1]:.6f}\n")
        pitch_cmd_f.write(
            f"{tnow:.4f},{los_gamma_raw_list[-1]:.6f},{los_gamma_c89_list[-1]:.6f},"
            f"{cmd_preclip_list[-1]:.6f},{cmd_clip_list[-1]:.6f},{cmd_rate_list[-1]:.6f},{cmd_final_list[-1]:.6f}\n")


        # vt_target：直接用 mon_target_vt（若缺省则退化为当前 vt）
        vt_target = i0('cmd_vt')
        vt_tgt_f.write(f"{tnow:.4f},{vt_target:.6f}\n")
        vt_target_list.append(vt_target)

        # roll_target：优先 dbg_desired_bank / cmd_roll，缺省则用当前 roll_deg（不至于空）
        roll_target_deg = rdeg
        if 'dbg_desired_bank' in info and info['dbg_desired_bank'] is not None:
            roll_target_deg = float(jnp.ravel(jnp.asarray(info['dbg_desired_bank']))[0]) * (180.0/jnp.pi)
        elif 'cmd_roll' in info and info['cmd_roll'] is not None:
            roll_target_deg = float(jnp.ravel(jnp.asarray(info['cmd_roll']))[0]) * (180.0/jnp.pi)
        roll_tgt_f.write(f"{tnow:.4f},{roll_target_deg:.6f}\n")
        roll_target_list.append(roll_target_deg)

        # pitch_target：优先 mon_desired_pitch_deg，其次 cmd_pitch
        pitch_target_deg = dpg if dpg != 0.0 else cmd_p
        pitch_tgt_f.write(f"{tnow:.4f},{pitch_target_deg:.6f}\n")
        pitch_target_list.append(pitch_target_deg)

        # yaw_target：直接用 cmd_heading（度）
        yaw_target_deg = cmd_h
        yaw_tgt_f.write(f"{tnow:.4f},{yaw_target_deg:.6f}\n")
        yaw_target_list.append(yaw_target_deg)

        # === 新增：用 env info 里的 obs_norm_* 记录 baseline 实际看到的误差 ===
        # 这三个量由 _step_task 写入，定义等同于 _controller_obs：
        #   obs_norm_dheading = _wrap_pi(yaw - cmd_heading)
        #   obs_norm_dpitch   = _wrap_pi(pitch - cmd_pitch)
        #   obs_norm_dvt      = (vt - cmd_vt) / 340.0

        if 'obs_norm_dheading' in info:
            ndh = i0('obs_norm_dheading')
            yaw_err = ndh * (180.0 / jnp.pi)  # rad -> deg
            yaw_err_f.write(f"{tnow:.4f},{yaw_err:.6f}\n")
            yaw_err_list.append(yaw_err)

        if 'obs_norm_dpitch' in info:
            ndp = i0('obs_norm_dpitch')
            pitch_err = ndp * (180.0 / jnp.pi)
            pitch_err_f.write(f"{tnow:.4f},{pitch_err:.6f}\n")
            pitch_err_list.append(pitch_err)

        if 'obs_norm_dvt' in info:
            ndvt = i0('obs_norm_dvt')
            vt_err = ndvt * 340.0
            vt_err_f.write(f"{tnow:.4f},{vt_err:.6f}\n")
            vt_err_list.append(vt_err)


        #=============================================================================================#

        # === 写入诊断文件 ===
        tnow = scalar(st.time)
        diag_f.write(f"{tnow:.2f},{dist:.2f},{hdist:.2f},{vtt:.2f},{alt:.2f},{yaw_deg:.2f},"
                     f"{dpg:.2f},{pdeg:.2f},{rdeg:.2f},{nz:.2f},{qn:.3f},{low},"
                     f"{En:.3f},{gdg:.2f},{vttgt:.2f},{vphase},{cmd_h:.2f},{cmd_p:.2f},{cmd_v:.2f}\n")
        if (_ % 100) == 0:
            diag_f.flush()


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

    # ===== 画图 =====
    # === 新增：收尾，关闭文件句柄 ===
    for f in (
        aer_f, thr_f,
        alpha_f, beta_f, gamma_f,
        roll_f, pitch_f, yaw_f,
        nx_f, ny_f, nz_f,
        vt_f, vt_tgt_f,
        roll_tgt_f, pitch_tgt_f, yaw_tgt_f,
        yaw_err_f, pitch_err_f, vt_err_f,  # 新增
    ):
        f.close()
    los_geom_f.close()
    pitch_cmd_f.close()


    # === 新增：画图并保存到和 acmi 同名前缀 ===
    # 1) AER 三舵量同一张图
    if t_list:
        # aileron / elevator / rudder
        plt.figure()
        plt.plot(t_list, aileron_list, label="aileron")
        plt.plot(t_list, elevator_list, label="elevator")
        plt.plot(t_list, rudder_list,   label="rudder")
        plt.xlabel("Time (s)")
        plt.ylabel("Deflection")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".aer.png", dpi=200)
        plt.close()

        # throttle 单独
        plt.figure()
        plt.plot(t_list, throttle_list, label="throttle")
        plt.xlabel("Time (s)")
        plt.ylabel("Throttle")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".thr.png", dpi=200)
        plt.close()

        # 迎角 alpha
        plt.figure()
        plt.plot(t_list, alpha_list)
        plt.xlabel("Time (s)")
        plt.ylabel("Alpha (deg)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".alpha.png", dpi=200)
        plt.close()

        # 侧滑角 beta
        plt.figure()
        plt.plot(t_list, beta_list)
        plt.xlabel("Time (s)")
        plt.ylabel("Beta (deg)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".beta.png", dpi=200)
        plt.close()

        # 攻角：这里用轨迹倾角 gamma
        plt.figure()
        plt.plot(t_list, gamma_list)
        plt.xlabel("Time (s)")
        plt.ylabel("Gamma (deg)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".gamma.png", dpi=200)
        plt.close()

        # 滚转角
        plt.figure()
        plt.plot(t_list, roll_list)
        plt.xlabel("Time (s)")
        plt.ylabel("Roll (deg)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".roll.png", dpi=200)
        plt.close()

        # 俯仰角
        plt.figure()
        plt.plot(t_list, pitch_list)
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (deg)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".pitch.png", dpi=200)
        plt.close()

        # 偏航角
        plt.figure()
        plt.plot(t_list, yaw_list)
        plt.xlabel("Time (s)")
        plt.ylabel("Yaw (deg)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".yaw.png", dpi=200)
        plt.close()

        # Nx / Ny / Nz 共一张图
        plt.figure()
        plt.plot(t_list, nx_list, label="Nx")
        plt.plot(t_list, ny_list, label="Ny")
        plt.plot(t_list, nz_list, label="Nz")
        plt.xlabel("Time (s)")
        plt.ylabel("Load factor / accel")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".nxyz.png", dpi=200)
        plt.close()

        # vt & vt_target 一张图
        plt.figure()
        plt.plot(t_list, vt_list, label="vt")
        plt.plot(t_list, vt_target_list, label="vt_target")
        plt.xlabel("Time (s)")
        plt.ylabel("VT")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".vt.png", dpi=200)
        plt.close()

        # roll & roll_target 一张图
        plt.figure()
        plt.plot(t_list, roll_list, label="roll")
        plt.plot(t_list, roll_target_list, label="roll_target")
        plt.xlabel("Time (s)")
        plt.ylabel("Roll (deg)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".roll_target.png", dpi=200)
        plt.close()

        # pitch & pitch_target 一张图
        plt.figure()
        plt.plot(t_list, pitch_list, label="pitch")
        plt.plot(t_list, pitch_target_list, label="pitch_target")
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (deg)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".pitch_target.png", dpi=200)
        plt.close()

        # yaw & yaw_target 一张图
        plt.figure()
        plt.plot(t_list, yaw_list, label="yaw")
        plt.plot(t_list, yaw_target_list, label="yaw_target")
        plt.xlabel("Time (s)")
        plt.ylabel("Yaw (deg)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(prefix) + ".yaw_target.png", dpi=200)
        plt.close()

        # yaw 误差（来自 baseline obs）
        if yaw_err_list:
            plt.figure()
            # plt.plot(t_list[:len(yaw_err_list)], yaw_err_list)
            plt.plot(t_list, yaw_err_list)
            plt.xlabel("Time (s)")
            plt.ylabel("Yaw error (deg)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(str(prefix) + ".yaw_err.png", dpi=200)
            plt.close()

        # pitch 误差
        if pitch_err_list:
            plt.figure()
            # plt.plot(t_list[:len(pitch_err_list)], pitch_err_list)
            plt.plot(t_list, pitch_err_list)
            plt.xlabel("Time (s)")
            plt.ylabel("Pitch error (deg)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(str(prefix) + ".pitch_err.png", dpi=200)
            plt.close()

        # vt 误差
        if vt_err_list:
            plt.figure()
            # plt.plot(t_list[:len(vt_err_list)], vt_err_list)
            plt.plot(t_list, vt_err_list)
            plt.xlabel("Time (s)")
            plt.ylabel("VT error (m/s)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(str(prefix) + ".vt_err.png", dpi=200)
            plt.close()

        # ===== 新图 1：Pitch 家族（实姿态 vs 最终目标 vs RAW/裁剪）===== #
        plt.figure(figsize=(10,7))
        plt.plot(t_list, pitch_list,           label="pitch (deg)")
        plt.plot(t_list, pitch_target_list,    label="pitch_target/cmd (deg)")
        plt.plot(t_list, los_gamma_raw_list,   linestyle="--", label="gamma_los_raw (deg)")
        plt.plot(t_list, los_gamma_c89_list,   linestyle="-.", label="gamma_los_clip89 (deg)")
        plt.plot(t_list, cmd_preclip_list,     label="cmd_preclip (deg)")
        plt.plot(t_list, cmd_clip_list,        label="cmd_clip (deg)")
        # cmd_rate 与 cmd_final 目前等同，如以后加了限速/滤波再看
        plt.xlabel("Time (s)"); plt.ylabel("Deg"); plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(str(prefix) + ".debug_pitch_family.png", dpi=200)
        plt.close()

        # ===== 新图 2：LOS 几何（hdist 变小 → atan2 接近 ±90°）===== #
        plt.figure(figsize=(10,6))
        plt.plot(t_list, los_hdist_raw_list, label="hdist_raw (m)")
        plt.plot(t_list, los_hdist_sat_list, label="hdist_sat (m)")
        plt.plot(t_list, los_da_list,        label="da = dz (m)")
        plt.xlabel("Time (s)"); plt.ylabel("Meters"); plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(str(prefix) + ".debug_los_geometry.png", dpi=200)
        plt.close()



    print("Done. Tacview file:", env.filename)
    diag_f.close()
    print("Diag saved:", diag_path)
if __name__ == "__main__":
    main()
