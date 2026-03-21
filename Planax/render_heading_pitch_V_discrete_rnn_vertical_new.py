# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import numpy as np
# from flax.linen.initializers import constant, orthogonal
# import functools
# from typing import Sequence, NamedTuple, Dict
# import distrax
# import optax
# from flax.training.train_state import TrainState
# import orbax.checkpoint as ocp

# from envs.wrappers import LogWrapper
# from envs.aeroplanax_heading_pitch_V_vertical import (
#     AeroPlanaxHeading_Pitch_V_Env,
#     Heading_Pitch_V_TaskParams,
# )


# # ================== RNN & Policy ==================
# class ScannedRNN(nn.Module):
#     @functools.partial(
#         nn.scan,
#         variable_broadcast="params",
#         in_axes=0,
#         out_axes=0,
#         split_rngs={"params": False},
#     )
#     @nn.compact
#     def __call__(self, carry, x):
#         rnn_state = carry
#         ins, resets = x
#         rnn_state = jnp.where(
#             resets[:, np.newaxis],
#             self.initialize_carry(*rnn_state.shape),
#             rnn_state,
#         )
#         new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
#         return new_rnn_state, y

#     @staticmethod
#     def initialize_carry(batch_size, hidden_size):
#         cell = nn.GRUCell(features=hidden_size)
#         return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# class ActorCriticRNN(nn.Module):
#     action_dim: Sequence[int]
#     config: Dict

#     @nn.compact
#     def __call__(self, hidden, x):
#         activation = nn.relu if self.config.get("ACTIVATION", "relu") == "relu" else nn.tanh
#         obs, dones = x

#         # encoder
#         emb = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
#         emb = activation(emb)

#         # GRU
#         hidden, emb = ScannedRNN()(hidden, (emb, dones))

#         # 瓶颈
#         h = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb)
#         h = nn.LayerNorm()(h)
#         h = activation(h)

#         # 轻量预测头(可选)
#         ph = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
#         ph = activation(ph)
#         pred = nn.Dense(3, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(ph)

#         aug = jnp.concatenate([h, jax.lax.stop_gradient(pred)], axis=-1)

#         # actor
#         am = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(aug)
#         am = activation(am)
#         th_logits = nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am)
#         el_logits = nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am)
#         ai_logits = nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am)
#         ru_logits = nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am)

#         pi_throttle = distrax.Categorical(logits=th_logits)
#         pi_elevator = distrax.Categorical(logits=el_logits)
#         pi_aileron  = distrax.Categorical(logits=ai_logits)
#         pi_rudder   = distrax.Categorical(logits=ru_logits)

#         # critic
#         v = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(aug)
#         v = activation(v)
#         v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)

#         return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(v, axis=-1), pred


# class Transition(NamedTuple):
#     done: jnp.ndarray
#     action: jnp.ndarray
#     value: jnp.ndarray
#     reward: jnp.ndarray
#     log_prob: jnp.ndarray
#     obs: jnp.ndarray
#     info: jnp.ndarray


# def batchify(x: dict, agent_list, num_envs, num_actors):
#     x = jnp.stack([x[a] for a in agent_list])
#     return x.reshape((num_actors * num_envs, -1))


# def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
#     x = x.reshape((num_actors, num_envs, -1))
#     return {a: x[i] for i, a in enumerate(agent_list)}


# # ================== RENDER ==================
# def render_episode(config):
#     # env
#     env_params = Heading_Pitch_V_TaskParams()
#     env = AeroPlanaxHeading_Pitch_V_Env(env_params)
#     env = LogWrapper(env)
#     config["NUM_ACTORS"] = env.num_agents

#     # model
#     net = ActorCriticRNN([31, 41, 41, 41], config=config)
#     rng = jax.random.PRNGKey(config["SEED"])
#     init_x = (
#         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)),
#         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
#     )
#     init_h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
#     params = net.init(rng, init_h, init_x)

#     tx = optax.chain(
#         optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#         optax.adam(config["LR"], eps=1e-5),
#     )
#     state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

#     # restore
#     if "LOADDIR" in config and config["LOADDIR"]:
#         to_restore = {"params": state.params, "opt_state": state.opt_state, "epoch": jnp.array(0)}
#         ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
#         ckpt = ckptr.restore(config["LOADDIR"], args=ocp.args.StandardRestore(item=to_restore))
#         print("[restore] epoch =", int(np.asarray(ckpt["epoch"])))
#         params = ckpt["params"]

#     # reset
#     rng, _rng = jax.random.split(rng)
#     reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
#     obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
#     env.render(env_state.env_state, env_params, {'__all__': False}, './tracks/')
#     h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

#     def _env_step(test_state):
#         env_state, last_obs, last_done, hst, rng = test_state
#         ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
#         hst, pi, value, _ = net.apply(params, hst, ac_in)
#         pi_th, pi_el, pi_ai, pi_ru = pi

#         # greedy for rendering
#         a_th = pi_th.mode()
#         a_el = pi_el.mode()
#         a_ai = pi_ai.mode()
#         a_ru = pi_ru.mode()

#         lp = (pi_th.log_prob(a_th) + pi_el.log_prob(a_el) + pi_ai.log_prob(a_ai) + pi_ru.log_prob(a_ru))

#         act = jnp.concatenate(
#             [a_th[:, :, np.newaxis], a_el[:, :, np.newaxis], a_ai[:, :, np.newaxis], a_ru[:, :, np.newaxis]],
#             axis=-1
#         )

#         value, act, lp = value.squeeze(0), act.squeeze(0), lp.squeeze(0)

#         rng, _rng = jax.random.split(rng)
#         rng_step = jax.random.split(_rng, config["NUM_ENVS"])
#         obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
#             rng_step, env_state, unbatchify(act, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
#         )
#         env.render(env_state.env_state, env_params, done, './tracks/')

#         reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
#         trans = Transition(last_done, act, value, reward, lp, last_obs, info)
#         obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
#         done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
#         return (env_state, obsv, done, hst, rng), trans

#     rng, _rng = jax.random.split(rng)
#     test_state = (
#         env_state,
#         batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
#         jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
#         h,
#         _rng,
#     )

#     # run
#     for _ in range(1000):
#         test_state, traj = _env_step(test_state)

#         # ---- safe scalar helpers ----
#         def to_float(x):
#             return float(np.asarray(x).reshape(-1)[0])

#         def to_mean(x):
#             return float(np.asarray(x).mean())

#         env_core = test_state[0].env_state  # unwrap LogWrapper
#         t  = to_float(env_core.time)
#         dn = bool(np.any(np.asarray(test_state[2])))

#         info = traj.info
#         # heading turn counts（逐 agent 向量）→ 取均值或第0个
#         succ_cnt = to_mean(info.get('heading_turn_counts', 0))
#         vert_succ = to_mean(info.get('vertical_success_counts', 0))

#         # 是否竖直：优先 info['is_vertical_target']；没有就按 |target_pitch|>70°
#         if 'is_vertical_target' in info:
#             is_vert = to_mean(info['is_vertical_target'])
#         else:
#             # 退化判断：读 target_pitch（若没有 info 里直接从 env_state 里取）
#             if hasattr(test_state[0], 'target_pitch'):
#                 tp = np.asarray(test_state[0].target_pitch)
#             else:
#                 tp = np.asarray(env_core.target_pitch) if hasattr(env_core, 'target_pitch') else 0.0
#             is_vert = float(np.mean((np.abs(np.rad2deg(tp)) >= 70.0).astype(np.float32)))

#         Rm = to_mean(traj.reward)
#         vt = to_float(env_core.plane_state.vt)
#         vx = to_float(env_core.plane_state.vel_x)
#         vy = to_float(env_core.plane_state.vel_y)
#         vz = to_float(env_core.plane_state.vel_z)

#         print(
#             f"[t={t:6.2f}] done={dn} "
#             f"succ_cnt={succ_cnt:.0f} vert_succ={vert_succ:.0f} "
#             f"is_vertical={is_vert:.2f} R_mean={Rm:+.3f} "
#             f"vt={vt:.2f} vel=({vx:.2f},{vy:.2f},{vz:.2f})"
#         )

#         if dn:
#             break

#     return {"state": test_state, "traj": traj}


# if __name__ == "__main__":
#     config = {
#         "SEED": 42,
#         "LR": 3e-4,
#         "NUM_ENVS": 1,
#         "NUM_ACTORS": 1,
#         "FC_DIM_SIZE": 128,
#         "GRU_HIDDEN_DIM": 128,
#         "UPDATE_EPOCHS": 16,
#         "NUM_MINIBATCHES": 5,
#         "GAMMA": 0.99,
#         "GAE_LAMBDA": 0.95,
#         "CLIP_EPS": 0.2,
#         "ENT_COEF": 1e-3,
#         "VF_COEF": 1,
#         "MAX_GRAD_NORM": 2,
#         "ACTIVATION": "relu",
#         "ANNEAL_LR": False,

#         # === 改成你的 checkpoint 路径 ===
#         # 例如:
#         # "LOADDIR": "/home/dqy/.../results/heading_pitch_V_discrete_rnn_2025-09-18-17-07/checkpoints/checkpoint_epoch_1600",
#         "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-09-18-17-07/checkpoints/checkpoint_epoch_1600",
#     }
#     _ = render_episode(config)


#========================================================================================#
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import numpy as np
# from flax.linen.initializers import constant, orthogonal
# import functools
# from typing import Sequence, NamedTuple, Any, Dict
# from flax.training.train_state import TrainState
# import distrax
# import optax
# from envs.wrappers import LogWrapper
# from envs.aeroplanax_heading_pitch_V_vertical import (
#     AeroPlanaxHeading_Pitch_V_Env,
#     Heading_Pitch_V_TaskParams,
# )
# import orbax.checkpoint as ocp

# # ------------------------ utils ------------------------
# def _first_float(x, default=0.0) -> float:
#     if x is None:
#         return float(default)
#     arr = np.asarray(x)
#     if arr.size == 0:
#         return float(default)
#     return float(arr.reshape(-1)[0])

# def _mean_float(x, default=0.0) -> float:
#     arr = np.asarray(x)
#     if arr.size == 0:
#         return float(default)
#     return float(arr.mean())

# def _any_bool(x) -> bool:
#     arr = np.asarray(x)
#     if arr.size == 0:
#         return False
#     return bool(arr.astype(bool).any())

# def _sum_int(x) -> int:
#     arr = np.asarray(x)
#     if arr.size == 0:
#         return 0
#     return int(arr.astype(np.int64).sum())

# def _wrap_deg(d):
#     # wrap to [-180, 180)
#     return (d + 180.0) % 360.0 - 180.0

# def _ang_diff_deg(a_new, a_old):
#     if a_old is None:
#         return 0.0
#     return _wrap_deg(a_new - a_old)

# # ------------------------ Model ------------------------
# class ScannedRNN(nn.Module):
#     @functools.partial(
#         nn.scan,
#         variable_broadcast="params",
#         in_axes=0,
#         out_axes=0,
#         split_rngs={"params": False},
#     )
#     @nn.compact
#     def __call__(self, carry, x):
#         rnn_state = carry
#         ins, resets = x
#         rnn_state = jnp.where(
#             resets[:, np.newaxis],
#             self.initialize_carry(*rnn_state.shape),
#             rnn_state,
#         )
#         new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
#         return new_rnn_state, y

#     @staticmethod
#     def initialize_carry(batch_size, hidden_size):
#         cell = nn.GRUCell(features=hidden_size)
#         return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

# class ActorCriticRNN(nn.Module):
#     action_dim: Sequence[int]
#     config: Dict

#     @nn.compact
#     def __call__(self, hidden, x):
#         activation = nn.relu if self.config.get("ACTIVATION", "relu") == "relu" else nn.tanh
#         obs, dones = x

#         emb = nn.Dense(self.config["FC_DIM_SIZE"],
#                        kernel_init=orthogonal(np.sqrt(2)),
#                        bias_init=constant(0.0))(obs)
#         emb = activation(emb)

#         hidden, emb = ScannedRNN()(hidden, (emb, dones))

#         nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb)
#         nn_fc2 = nn.LayerNorm()(nn_fc2)
#         nn_fc2 = activation(nn_fc2)

#         # 轻量预测头（和训练一致，保证能加载权重）
#         pred_h = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(nn_fc2)
#         pred_h = activation(pred_h)
#         pred   = nn.Dense(3,   kernel_init=orthogonal(0.01),      bias_init=constant(0.0))(pred_h)
#         aug    = jnp.concatenate([nn_fc2, jax.lax.stop_gradient(pred)], axis=-1)

#         # actor
#         ah = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(aug)
#         ah = activation(ah)
#         logit_thr = nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(ah)
#         logit_ele = nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(ah)
#         logit_ail = nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(ah)
#         logit_rud = nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(ah)
#         pi_thr = distrax.Categorical(logits=logit_thr)
#         pi_ele = distrax.Categorical(logits=logit_ele)
#         pi_ail = distrax.Categorical(logits=logit_ail)
#         pi_rud = distrax.Categorical(logits=logit_rud)

#         # critic
#         v = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(aug)
#         v = activation(v)
#         v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)

#         return hidden, (pi_thr, pi_ele, pi_ail, pi_rud), jnp.squeeze(v, axis=-1), pred

# # ------------------------ Types ------------------------
# class Transition(NamedTuple):
#     done: jnp.ndarray
#     action: jnp.ndarray
#     value: jnp.ndarray
#     reward: jnp.ndarray
#     log_prob: jnp.ndarray
#     obs: jnp.ndarray
#     info: jnp.ndarray

# # ------------------------ helpers ------------------------
# def batchify(x: dict, agent_list, num_envs, num_actors):
#     x = jnp.stack([x[a] for a in agent_list])
#     return x.reshape((num_actors * num_envs, -1))

# def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
#     x = x.reshape((num_actors, num_envs, -1))
#     return {a: x[i] for i, a in enumerate(agent_list)}

# # ------------------------ main render ------------------------
# def render_episode(config):
#     # env
#     env_params = Heading_Pitch_V_TaskParams()
#     env = AeroPlanaxHeading_Pitch_V_Env(env_params)
#     env = LogWrapper(env)
#     config["NUM_ACTORS"] = env.num_agents
#     assert config["NUM_ENVS"] == 1, "For rendering, please set NUM_ENVS = 1."

#     # net
#     network = ActorCriticRNN([31, 41, 41, 41], config=config)
#     rng = jax.random.PRNGKey(config['SEED'])
#     init_x = (
#         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"],
#                   *env.observation_space(env.agents[0], env_params).shape)),
#         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
#     )
#     init_h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"],
#                                          config["GRU_HIDDEN_DIM"])
#     params = network.init(rng, init_h, init_x)

#     tx = optax.adam(config["LR"], eps=1e-5)
#     train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

#     # restore
#     if config.get("LOADDIR"):
#         state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
#         ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
#         checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))
#         print(f"[restore] epoch = {int(checkpoint['epoch'])}")
#         params = checkpoint["params"]

#     # reset
#     rng, _rng = jax.random.split(rng)
#     reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
#     obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
#     try:
#         env.render(env_state.env_state, env_params, {'__all__': False}, './tracks/')
#     except Exception:
#         pass

#     init_h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"],
#                                          config["GRU_HIDDEN_DIM"])

#     def _env_step(test_state):
#         env_state, last_obs, last_done, hstate, rng = test_state
#         ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
#         hstate, pi, value, _ = network.apply(params, hstate, ac_in)
#         pi_thr, pi_ele, pi_ail, pi_rud = pi

#         # greedy for rendering
#         a_thr = pi_thr.mode()
#         a_ele = pi_ele.mode()
#         a_ail = pi_ail.mode()
#         a_rud = pi_rud.mode()

#         log_prob = (
#             pi_thr.log_prob(a_thr)
#             + pi_ele.log_prob(a_ele)
#             + pi_ail.log_prob(a_ail)
#             + pi_rud.log_prob(a_rud)
#         )
#         action = jnp.concatenate(
#             [a_thr[:, :, np.newaxis], a_ele[:, :, np.newaxis],
#              a_ail[:, :, np.newaxis], a_rud[:, :, np.newaxis]], axis=-1
#         )
#         value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

#         rng, _rng = jax.random.split(rng)
#         rng_step = jax.random.split(_rng, config["NUM_ENVS"])
#         obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
#             rng_step, env_state, unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
#         )
#         try:
#             env.render(env_state.env_state, env_params, done, './tracks/')
#         except Exception:
#             pass

#         reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
#         transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
#         obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
#         done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
#         test_state = (env_state, obsv, done, hstate, rng)
#         return test_state, transition

#     rng, _rng = jax.random.split(rng)
#     test_state = (
#         env_state,
#         batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
#         jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
#         init_h,
#         _rng,
#     )

#     # 保存上一次的目标（用于计算 Δ）
#     last_cmd = {"psi_deg": None, "theta_deg": None, "vt": None}

#     for _ in range(config.get("MAX_STEPS", 1000)):
#         test_state, traj = _env_step(test_state)

#         info = traj.info
#         core = test_state[0].env_state

#         # 是否竖直（用于观测）
#         if 'is_vertical_target' in info:
#             is_vert = _any_bool(info['is_vertical_target'])
#         else:
#             # 没有 flag 就按阈值判断
#             theta_deg_now = np.degrees(_first_float(core.target_pitch))
#             is_vert = abs(theta_deg_now) >= 70.0

#         switched = _any_bool(info.get('switch_event', 0))
#         succ_cnt  = _sum_int(info.get('heading_turn_counts', 0))
#         vert_succ = _sum_int(info.get('vertical_success_counts', 0))

#         t   = _first_float(core.time)
#         vt  = _first_float(core.plane_state.vt)
#         vx  = _first_float(core.plane_state.vel_x)
#         vy  = _first_float(core.plane_state.vel_y)
#         vz  = _first_float(core.plane_state.vel_z)
#         done_any = _any_bool(test_state[2])
#         R_mean   = _mean_float(traj.reward)

#         # 常规一行（每步）
#         print(
#             f"[t={t:.2f}] done={int(done_any)} succ_cnt={succ_cnt} vert_succ={vert_succ} "
#             f"is_vertical={int(is_vert)} R_mean={R_mean:+.3f} vt={vt:.2f} "
#             f"vel=({vx:.2f},{vy:.2f},{vz:.2f})"
#         )

#         # ----- 只在“切换发生”的那一帧，打印新的目标指令与变化量 -----
#         if switched:
#             # 目标俯仰(°) / 速度
#             theta_cmd_deg = _first_float(info.get('target_pitch_cmd_deg', None))
#             if theta_cmd_deg is None:
#                 theta_cmd_deg = np.degrees(_first_float(core.target_pitch))

#             vt_cmd = _first_float(info.get('target_vt_cmd', None))
#             if vt_cmd is None:
#                 vt_cmd = _first_float(core.target_vt)

#             # 目标航向(°)：优先 info，如果没有就从 state 取
#             psi_cmd_deg = _first_float(info.get('target_heading_cmd_deg', None))
#             if psi_cmd_deg is None:
#                 psi_cmd_deg = np.degrees(_first_float(core.target_heading))

#             d_theta = _ang_diff_deg(theta_cmd_deg, last_cmd["theta_deg"])
#             d_psi   = _ang_diff_deg(psi_cmd_deg,   last_cmd["psi_deg"])
#             d_vt    = 0.0 if last_cmd["vt"] is None else (vt_cmd - last_cmd["vt"])

#             print(
#                 f"  [SWITCH] target -> psi={psi_cmd_deg:+.2f}°, theta={theta_cmd_deg:+.2f}°, "
#                 f"Vt={vt_cmd:.2f}  |  Δpsi={d_psi:+.2f}°, Δtheta={d_theta:+.2f}°, ΔVt={d_vt:+.2f}"
#             )

#             # 记录本次目标
#             last_cmd["psi_deg"]   = psi_cmd_deg
#             last_cmd["theta_deg"] = theta_cmd_deg
#             last_cmd["vt"]        = vt_cmd

#         if done_any:
#             break

#     return {"test_state": test_state, "trajectory": traj}

# # ------------------------ config & run ------------------------
# if __name__ == "__main__":
#     config = {
#         "SEED": 42,
#         "LR": 3e-4,
#         "NUM_ENVS": 1,          # 渲染务必=1
#         "NUM_ACTORS": 1,
#         "FC_DIM_SIZE": 128,
#         "GRU_HIDDEN_DIM": 128,
#         "UPDATE_EPOCHS": 16,
#         "NUM_MINIBATCHES": 5,
#         "GAMMA": 0.99,
#         "GAE_LAMBDA": 0.95,
#         "CLIP_EPS": 0.2,
#         "ENT_COEF": 1e-3,
#         "VF_COEF": 1,
#         "MAX_GRAD_NORM": 2,
#         "ACTIVATION": "relu",
#         "ANNEAL_LR": False,
#         "MAX_STEPS": 1000,
#         "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-09-26-20-08/checkpoints/checkpoint_epoch_2600",
#     }
#     _ = render_episode(config)


#=================================================================#
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
import functools
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import optax
import orbax.checkpoint as ocp

from envs.wrappers import LogWrapper
from envs.aeroplanax_heading_pitch_V_vertical_new import (
    AeroPlanaxHeading_Pitch_V_Env,
    Heading_Pitch_V_TaskParams,
)

# ------------------------ utils ------------------------
def _first_float(x, default=0.0) -> float:
    if x is None:
        return float(default)
    arr = np.asarray(x)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])

def _mean_float(x, default=0.0) -> float:
    arr = np.asarray(x)
    if arr.size == 0:
        return float(default)
    return float(arr.mean())

def _any_bool(x) -> bool:
    arr = np.asarray(x)
    if arr.size == 0:
        return False
    return bool(arr.astype(bool).any())

def _sum_int(x) -> int:
    arr = np.asarray(x)
    if arr.size == 0:
        return 0
    return int(arr.astype(np.int64).sum())

# ------------------------ Model ------------------------
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

        emb = nn.Dense(self.config["FC_DIM_SIZE"],
                       kernel_init=orthogonal(np.sqrt(2)),
                       bias_init=constant(0.0))(obs)
        emb = activation(emb)

        hidden, emb = ScannedRNN()(hidden, (emb, dones))

        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        # 与训练一致的轻量预测头（预测 [vt, pitch(rad), nz]）
        pred_h = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(nn_fc2)
        pred_h = activation(pred_h)
        pred   = nn.Dense(3,   kernel_init=orthogonal(0.01),      bias_init=constant(0.0))(pred_h)
        aug    = jnp.concatenate([nn_fc2, jax.lax.stop_gradient(pred)], axis=-1)

        # actor
        ah = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(aug)
        ah = activation(ah)
        logit_thr = nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(ah)
        logit_ele = nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(ah)
        logit_ail = nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(ah)
        logit_rud = nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(ah)
        pi_thr = distrax.Categorical(logits=logit_thr)
        pi_ele = distrax.Categorical(logits=logit_ele)
        pi_ail = distrax.Categorical(logits=logit_ail)
        pi_rud = distrax.Categorical(logits=logit_rud)

        # critic
        v = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(aug)
        v = activation(v)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)

        return hidden, (pi_thr, pi_ele, pi_ail, pi_rud), jnp.squeeze(v, axis=-1), pred

# ------------------------ Types ------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

# ------------------------ helpers ------------------------
def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

# ------------------------ main render ------------------------
def render_episode(config):
    # 1) env（新环境）
    env_params = Heading_Pitch_V_TaskParams()
    env_core = AeroPlanaxHeading_Pitch_V_Env(env_params)
    env = LogWrapper(env_core)
    config["NUM_ACTORS"] = env.num_agents
    assert config["NUM_ENVS"] == 1, "渲染请设置 NUM_ENVS = 1。"

    # 2) 网络（与训练完全一致）
    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    rng = jax.random.PRNGKey(config['SEED'])
    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"],
                  *env.observation_space(env.agents[0], env_params).shape)),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    init_h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"],
                                         config["GRU_HIDDEN_DIM"])
    params = network.init(rng, init_h, init_x)

    tx = optax.adam(config["LR"], eps=1e-5)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # 3) 恢复 checkpoint
    if config.get("LOADDIR"):
        state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))
        print(f"[restore] epoch = {int(checkpoint['epoch'])}")
        params = checkpoint["params"]

    # 4) reset + 初帧渲染（写 acmi 头）
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    try:
        env.render(env_state.env_state, env_params, {'__all__': False}, './tracks/')
    except Exception:
        pass

    init_h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"],
                                         config["GRU_HIDDEN_DIM"])

    def _env_step(test_state):
        env_state, last_obs, last_done, hstate, rng = test_state
        ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
        hstate, pi, value, _ = network.apply(params, hstate, ac_in)
        pi_thr, pi_ele, pi_ail, pi_rud = pi

        # 渲染用贪心
        a_thr = pi_thr.mode()
        a_ele = pi_ele.mode()
        a_ail = pi_ail.mode()
        a_rud = pi_rud.mode()

        log_prob = (
            pi_thr.log_prob(a_thr)
            + pi_ele.log_prob(a_ele)
            + pi_ail.log_prob(a_ail)
            + pi_rud.log_prob(a_rud)
        )
        action = jnp.concatenate(
            [a_thr[:, :, np.newaxis], a_ele[:, :, np.newaxis],
             a_ail[:, :, np.newaxis], a_rud[:, :, np.newaxis]], axis=-1
        )
        value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
            rng_step, env_state, unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        )
        try:
            env.render(env_state.env_state, env_params, done, './tracks/')
        except Exception:
            pass

        reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
        obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        test_state = (env_state, obsv, done, hstate, rng)
        return test_state, transition

    rng, _rng = jax.random.split(rng)
    test_state = (
        env_state,
        batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
        jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
        init_h,
        _rng,
    )

    # 5) roll out
    for _ in range(config.get("MAX_STEPS", 2000)):
        test_state, traj = _env_step(test_state)
        info = traj.info
        core = test_state[0].env_state

        # 简要日志
        t   = _first_float(core.time)
        vt  = _first_float(core.plane_state.vt)
        done_any = _any_bool(test_state[2])
        R_mean   = _mean_float(traj.reward)
        vert_up_succ   = _sum_int(info.get('vertical_up_success_counts', 0))
        vert_down_succ = _sum_int(info.get('vertical_down_success_counts', 0))
        is_vert = _any_bool(info.get('is_vertical_target', 0))
        print(
            f"[t={t:7.2f}] done={int(done_any)} R_mean={R_mean:+.3f} vt={vt:6.2f} "
            f"vert={int(is_vert)} up_succ={vert_up_succ} down_succ={vert_down_succ}"
        )

        if done_any:
            break

    print("渲染结束，Tacview 文件已写入 tracks/ 目录。")
    return {"test_state": test_state, "trajectory": traj}

if __name__ == "__main__":
    config = {
        "SEED": 42,
        "LR": 3e-4,
        "NUM_ENVS": 1,          # 渲染务必=1
        "NUM_ACTORS": 1,
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "UPDATE_EPOCHS": 16,
        "NUM_MINIBATCHES": 5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 1e-3,
        "VF_COEF": 1,
        "MAX_GRAD_NORM": 2,
        "ACTIVATION": "relu",
        "ANNEAL_LR": False,
        "MAX_STEPS": 1000,
        # 修改为你的 checkpoint 路径
        "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/heading_pitch_V_discrete_rnn_2025-09-27-16-27/checkpoints/checkpoint_epoch_2600",
    }
    _ = render_episode(config)