# 文件：Planax/utils/rollout_txt_logger.py
# 作用：将 JAX 训练中的 rollout（traj_batch）以可读 txt 形式落盘
# 新增：支持写入 obs 英文名称与反归一化缩放，并真正按缩放输出 obs_raw
# 用法：
#   from Planax.utils.rollout_txt_logger import save_rollout_with_io_callback
#   save_rollout_with_io_callback(
#       traj_batch, update_steps,
#       log_dir="./rollout_logs",
#       obs_names=[...],                 # 可选：长度=obs_dim 的英文名称列表
#       denorm_scales=[...],             # 可选：长度=obs_dim 的每维反归一化缩放系数
#       legacy=None                      # 可选：'combat' 使用你之前的分块式反归一化；默认 None
#   )

import os
import io
import json
import numpy as np

def _denorm_obs_by_scales(vec_norm, denorm_scales):
    """逐维缩放反归一化：raw[i] = norm[i] * scale[i]（若某维为 None 则不缩放）"""
    if denorm_scales is None:
        return list(vec_norm)
    out = []
    for x, s in zip(vec_norm, denorm_scales):
        if s is None:
            out.append(float(x))
        else:
            out.append(float(x) * float(s))
    return out

def _denorm_obs_legacy_combat(vec_norm):
    """
    兼容原 combat 的“分块式”反归一化逻辑（你旧版的写法）。
    结构假设：若干个 6 维相对特征块 + 末尾 9 维自身特征。
    """
    raw = list(vec_norm)
    cur = 0
    # 相对块：Δvt, Δaltitude, AO, TA, distance, side_flag
    while len(raw) - cur > 9:
        # 只对有量纲的几维缩放，角度与 side_flag 保持
        raw[cur + 0] *= 340      # Δvt
        raw[cur + 1] *= 1000     # Δaltitude
        raw[cur + 4] *= 10000    # distance
        cur += 6
    # 自身 9 维：altitude, roll_sin, roll_cos, pitch_sin, pitch_cos, vel_x, vel_y, vel_z, vt
    if len(raw) - cur == 9:
        raw[cur + 0] *= 5000     # altitude
        raw[cur + 5] *= 340      # vel_x
        raw[cur + 6] *= 340      # vel_y
        raw[cur + 7] *= 340      # vel_z
        raw[cur + 8] *= 340      # vt
    else:
        print(f"[warn] unexpected own-feature length {len(raw)-cur}")
    return raw

def make_rollout_txt_saver(log_dir="./rollout_logs",
                           obs_names=None,
                           denorm_scales=None,
                           legacy=None):
    """
    返回一个可用于 jax.experimental.io_callback 的回调函数。
    - obs_names: list[str]，长度必须等于 obs_dim（否则自动回退为 ['feat_0', ...]）
    - denorm_scales: list[float|None]，长度=obs_dim；None 表示该维不缩放
    - legacy: 若为 'combat'，使用旧的分块式反归一化（忽略 denorm_scales）
    """
    os.makedirs(log_dir, exist_ok=True)

    def _save_traj_callback(params):
        traj, step_idx = params
        # 将 step_idx 统一成 int
        try:
            step_i = int(np.asarray(step_idx).reshape(()).item())
        except Exception:
            try:
                step_i = int(step_idx)
            except Exception:
                step_i = 0

        # 基本形状
        obs_mat0 = np.asarray(traj.obs[0])          # (N_agent, obs_dim)
        n_agents = int(obs_mat0.shape[0])
        obs_dim  = int(obs_mat0.shape[1])

        # 兜底 obs_names
        names = obs_names
        if not isinstance(names, (list, tuple)) or len(names) != obs_dim:
            names = [f"feat_{i}" for i in range(obs_dim)]

        # 兜底 denorm_scales：None 表示不缩放
        scales = denorm_scales
        if scales is not None and len(scales) != obs_dim:
            print(f"[warn] denorm_scales length {len(scales)} != obs_dim {obs_dim}, ignore scales.")
            scales = None

        # 写文件
        fname = os.path.join(log_dir, f"trajectory_update_{step_i}.txt")
        with io.open(fname, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Transition sequence for one PPO update\n")
            f.write("# Columns: t, obs_norm(list), obs_raw(list), action(list), reward(list), done(list), value(list), log_prob(list)\n")
            # —— 紧跟第二行，补充观测字段与反归一化说明 —— #
            f.write(f"# Obs_Dim: {obs_dim}; Agents: {n_agents}\n")
            f.write("# Obs_Names (per index): " + ", ".join([f"{i}:{n}" for i, n in enumerate(names)]) + "\n")
            if legacy == "combat":
                f.write("# Denorm: legacy 'combat' block-wise rule (Δvt*340, Δalt*1000, dist*10000; own: alt*5000, vel*340)\n")
            elif scales is not None:
                f.write("# Denorm_Scales: " + json.dumps(scales) + "  # raw[i] = norm[i] * scale[i] (None = no-scale)\n")
            else:
                f.write("# Denorm: none (obs_raw equals obs_norm)\n")

            T = int(np.asarray(traj.obs).shape[0])
            for t in range(T):
                # obs_norm：保持你原有的 “每步一个 (N_agent, obs_dim) 矩阵->列表”
                obs_norm_mat = np.asarray(traj.obs[t])                # (N_agent, obs_dim)
                obs_norm_flat = obs_norm_mat.tolist()

                # obs_raw：逐 agent 反归一化
                obs_raw_flat = []
                for a in range(obs_norm_mat.shape[0]):
                    vec = obs_norm_mat[a].tolist()
                    if legacy == "combat":
                        obs_raw_flat.extend(_denorm_obs_legacy_combat(vec))
                    else:
                        obs_raw_flat.extend(_denorm_obs_by_scales(vec, scales))

                line = (
                    f"{t}, "
                    f"{obs_norm_flat}, "
                    f"{obs_raw_flat}, "
                    f"{np.asarray(traj.action[t]).tolist()}, "
                    f"{np.asarray(traj.reward[t]).tolist()}, "
                    f"{np.asarray(traj.done[t]).tolist()}, "
                    f"{np.asarray(traj.value[t]).tolist()}, "
                    f"{np.asarray(traj.log_prob[t]).tolist()}\n"
                )
                f.write(line)
        print(f"[logger] rollout saved -> {fname}")
        return None

    return _save_traj_callback


def save_rollout_with_io_callback(traj_batch, update_steps, log_dir="./rollout_logs",
                                  obs_names=None, denorm_scales=None, legacy=None):
    """便捷封装：训练脚本里直接调用即可。"""
    import jax
    cb = make_rollout_txt_saver(log_dir, obs_names=obs_names, denorm_scales=denorm_scales, legacy=legacy)
    jax.experimental.io_callback(cb, None, (traj_batch, update_steps), ordered=True)



#=============================================================================================================#

# # 文件：Planax/utils/rollout_txt_logger.py
# # 作用：将 JAX 训练中的 rollout（traj_batch）以与 AeroPlanax 相同的格式，写入 txt
# # 用法（示例见下文“调用示例”部分）：
# #   from Planax.utils.rollout_txt_logger import save_rollout_with_io_callback
# #   save_rollout_with_io_callback(traj_batch, update_steps, log_dir="./rollout_logs_norm_and_denorm")
# import os
# import io
# import numpy as np

# def denorm_obs(vec_norm):
# 	"""
# 	与 xiangmu/AeroPlanax/train_combat_selfplay_hierarchy.py 中相同逻辑的反归一化:
# 	- 多个 6 维相对特征块（Δvt, Δaltitude, AO, TA, distance, side_flag）
# 	- 剩余 9 维自身特征（altitude, roll/pitch sin cos, vel_x/vel_y/vel_z, vt）
# 	"""
# 	raw = list(vec_norm)
# 	cur = 0
# 	while len(raw) - cur > 9:
# 		raw[cur + 0] *= 340      # Δvt
# 		raw[cur + 1] *= 1000     # Δaltitude
# 		# AO, TA, side_flag 保持
# 		raw[cur + 4] *= 10000    # distance
# 		cur += 6
# 	if len(raw) - cur == 9:
# 		raw[cur + 0] *= 5000     # altitude
# 		# roll/pitch sin cos 保持
# 		raw[cur + 5] *= 340      # vel_x
# 		raw[cur + 6] *= 340      # vel_y
# 		raw[cur + 7] *= 340      # vel_z
# 		raw[cur + 8] *= 340      # vt
# 	else:
# 		print(f"[warn] unexpected own-feature length {len(raw)-cur}")
# 	return raw

# def make_rollout_txt_saver(log_dir="./rollout_logs_norm_and_denorm"):
# 	"""
# 	返回一个可用于 jax.experimental.io_callback 的回调函数：
# 	入参为 (traj_batch, update_steps)，输出 None。
# 	写入内容与原脚本完全一致：
# 	- 文件名：trajectory_update_{update_steps}.txt
# 	- 表头与列顺序保持一致
# 	"""
# 	os.makedirs(log_dir, exist_ok=True)

# 	def _save_traj_callback(params):
# 		traj, step_idx = params
# 		# step_idx 可能是 jnp scalar，统一成 python int
# 		try:
# 			step_i = int(np.asarray(step_idx).reshape(()).item())
# 		except Exception:
# 			try:
# 				step_i = int(step_idx)
# 			except Exception:
# 				step_i = 0

# 		fname = os.path.join(log_dir, f"trajectory_update_{step_i}.txt")
# 		with io.open(fname, "w", encoding="utf-8", newline="\n") as f:
# 			f.write(
# 				"# Transition sequence for one PPO update\n"
# 				"# Columns: t, obs_norm(list), obs_raw(list), action(list), "
# 				"reward(list), done(list), value(list), log_prob(list)\n"
# 			)
# 			# 注意：在 host 回调里，JAX 会把数组物化为 numpy
# 			T = int(np.asarray(traj.obs).shape[0])
# 			for t in range(T):
# 				obs_norm_flat = np.asarray(traj.obs[t]).tolist()

# 				obs_mat = np.asarray(traj.obs[t])         # (N_agent, obs_dim)
# 				obs_flat = obs_mat.ravel().tolist()
# 				obs_per_agent = int(obs_mat.shape[1])

# 				obs_raw = []
# 				for i in range(0, len(obs_flat), obs_per_agent):
# 					obs_raw.extend(denorm_obs(obs_flat[i:i + obs_per_agent]))

# 				line = (
# 					f"{t}, "
# 					f"{obs_norm_flat}, "
# 					f"{obs_raw}, "
# 					f"{np.asarray(traj.action[t]).tolist()}, "
# 					f"{np.asarray(traj.reward[t]).tolist()}, "
# 					f"{np.asarray(traj.done[t]).tolist()}, "
# 					f"{np.asarray(traj.value[t]).tolist()}, "
# 					f"{np.asarray(traj.log_prob[t]).tolist()}\n"
# 				)
# 				f.write(line)
# 		print(f"[logger] rollout saved -> {fname}")
# 		return None

# 	return _save_traj_callback

# def save_rollout_with_io_callback(traj_batch, update_steps, log_dir="./rollout_logs_norm_and_denorm"):
# 	"""
# 	便捷封装：直接在训练脚本里调用这一个函数即可触发落盘。
# 	"""
# 	import jax
# 	cb = make_rollout_txt_saver(log_dir)
# 	jax.experimental.io_callback(
# 		cb,
# 		None,                           # 无需返回到计算图
# 		(traj_batch, update_steps),     # 传入 rollout 与当前 update 序号
# 		ordered=True
# 	)