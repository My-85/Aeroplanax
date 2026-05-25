"""
train_hold.py — PPO + GRU training for the Altitude/Speed/Attitude Hold task.

Usage:
    conda activate aeroplanax
    cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
    python train_hold.py

Checkpoint is saved to  results/hold_<datetime>/checkpoints/checkpoint_epoch_<N>
TensorBoard logs go to  results/hold_<datetime>/logs/
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.90'
os.environ['WANDB_API_KEY'] = '4c0cc04699296bed768adea4824fbaecea35dc59'

import wandb

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from pathlib import Path
from datetime import datetime
from typing import Sequence, NamedTuple, Dict
import functools
import distrax
import tensorboardX
import orbax.checkpoint as ocp

from envs.aeroplanax_hold import AeroPlanaxHoldEnv, HoldTaskParams
from envs.wrappers import LogWrapper

# ──────────────────────────────────────────────────────────────────────────
# Network  (identical to train_quat_baseline_iter.py)
# ──────────────────────────────────────────────────────────────────────────
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
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        fc2 = nn.Dense(256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                       bias_init=nn.initializers.constant(0.0))(embedding)
        fc2 = nn.LayerNorm()(fc2)
        fc2 = activation(fc2)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=nn.initializers.orthogonal(2),
            bias_init=nn.initializers.constant(0.0),
        )(fc2)
        actor_mean = activation(actor_mean)

        pi_throttle = distrax.Categorical(
            logits=nn.Dense(self.action_dim[0],
                            kernel_init=nn.initializers.orthogonal(0.01),
                            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(
            logits=nn.Dense(self.action_dim[1],
                            kernel_init=nn.initializers.orthogonal(0.01),
                            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_aileron = distrax.Categorical(
            logits=nn.Dense(self.action_dim[2],
                            kernel_init=nn.initializers.orthogonal(0.01),
                            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_rudder = distrax.Categorical(
            logits=nn.Dense(self.action_dim[3],
                            kernel_init=nn.initializers.orthogonal(0.01),
                            bias_init=nn.initializers.constant(0.0))(actor_mean))

        critic = nn.Dense(self.config["FC_DIM_SIZE"],
                          kernel_init=nn.initializers.orthogonal(2),
                          bias_init=nn.initializers.constant(0.0))(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0),
                          bias_init=nn.initializers.constant(0.0))(critic)

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done:         jnp.ndarray
    action:       jnp.ndarray
    value:        jnp.ndarray
    reward:       jnp.ndarray
    log_prob:     jnp.ndarray
    obs:          jnp.ndarray
    info:         jnp.ndarray
    valid_action: jnp.ndarray


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


# ──────────────────────────────────────────────────────────────────────────
# make_train
# ──────────────────────────────────────────────────────────────────────────
def make_train(config):
    cfg = dict(config)
    cfg.setdefault("VF_CLIP_EPS",   0.20)
    cfg.setdefault("HUBER_DELTA",   1.0)
    cfg.setdefault("LR_DECAY",      0.999)
    cfg.setdefault("MIN_LR_MULT",   0.2)
    cfg.setdefault("ENT_COEF_MIN",  5e-4)
    cfg.setdefault("ENT_COEF_MAX",  2e-2)
    cfg.setdefault("ENT_ADJ_RATE",  1.05)
    cfg.setdefault("TARGET_KL",     0.02)
    cfg.setdefault("KL_STOP_MULT",  1.5)

    env_params = HoldTaskParams()
    env = AeroPlanaxHoldEnv(env_params)
    env = LogWrapper(env)
    cfg["NUM_ACTORS"]    = env.num_agents
    cfg["NUM_UPDATES"]   = int(cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"])
    cfg["MINIBATCH_SIZE"] = cfg["NUM_ACTORS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]

    checkpoint = None
    if "LOADDIR" in cfg:
        # restore a previous run
        network = ActorCriticRNN([31, 41, 41, 41], config=cfg)
        rng = jax.random.PRNGKey(0)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"],
                        *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"])),
        )
        init_h = ScannedRNN.initialize_carry(
            cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])
        net_params = network.init(rng, init_h, init_x)
        tx = optax.adam(cfg["LR"])
        ts = TrainState.create(apply_fn=network.apply, params=net_params, tx=tx)
        state = {"params": ts.params, "opt_state": ts.opt_state, "epoch": jnp.array(0)}
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(cfg["LOADDIR"],
                                   args=ocp.args.StandardRestore(item=state))

    def train(rng):
        network = ActorCriticRNN([31, 41, 41, 41], config=cfg)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"],
                        *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"])),
        )
        init_h = ScannedRNN.initialize_carry(
            cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])
        net_params = network.init(_rng, init_h, init_x)
        tx = optax.adam(cfg["LR"])
        train_state = TrainState.create(apply_fn=network.apply, params=net_params, tx=tx)

        if checkpoint is not None:
            train_state = train_state.replace(
                params=checkpoint["params"],
                opt_state=checkpoint["opt_state"],
            )
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = 0

        # ── init envs ──
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_h = ScannedRNN.initialize_carry(
            cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["GRU_HIDDEN_DIM"])

        if cfg.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(cfg["LOGDIR"])

        # ── env step (inside scan) ──
        def _env_step(runner_state, _):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            pi_thr, pi_el, pi_ail, pi_rud = pi

            rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
            act_thr = pi_thr.sample(seed=k1)
            act_el  = pi_el.sample(seed=k2)
            act_ail = pi_ail.sample(seed=k3)
            act_rud = pi_rud.sample(seed=k4)

            lp = (pi_thr.log_prob(act_thr)
                + pi_el.log_prob(act_el)
                + pi_ail.log_prob(act_ail)
                + pi_rud.log_prob(act_rud))

            action = jnp.concatenate([
                act_thr[:, :, np.newaxis],
                act_el[:, :, np.newaxis],
                act_ail[:, :, np.newaxis],
                act_rud[:, :, np.newaxis],
            ], axis=-1)
            value, action, lp = value.squeeze(0), action.squeeze(0), lp.squeeze(0)

            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, cfg["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state,
                unbatchify(action, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]),
            )
            reward = batchify(reward, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1)

            transition = Transition(
                last_done, action, value, reward, lp, last_obs, info,
                valid_action=jnp.logical_not(
                    jnp.logical_and(
                        last_done,
                        jnp.reshape(
                            batchify(done, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1),
                            last_done.shape,
                        ),
                    )
                ),
            )
            obsv   = batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
            done_b = batchify(done, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1)

            def _reset_h(h):
                return jnp.where(done_b[:, None], jax.lax.stop_gradient(jnp.zeros_like(h)), h)
            hstate = _reset_h(hstate)

            runner_state = (train_state, env_state, obsv, done_b, hstate, rng)
            return runner_state, transition

        # ── GAE ──
        def _calc_gae(traj, last_val):
            def _step(gae_nv, t):
                gae, nv = gae_nv
                r    = jnp.nan_to_num(t.reward)
                v    = jnp.nan_to_num(t.value)
                nv   = jnp.nan_to_num(nv)
                d    = r + cfg["GAMMA"] * nv * (1 - t.done) - v
                gae  = d + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1 - t.done) * gae
                return (gae, v), gae
            _, advantages = jax.lax.scan(_step, (jnp.zeros_like(last_val), last_val),
                                         traj, reverse=True, unroll=16)
            targets = advantages + traj.value
            mask    = traj.valid_action.astype(jnp.float32)
            cnt     = mask.sum() + 1e-8
            adv_m   = (advantages * mask).sum() / cnt
            adv_s   = jnp.sqrt(((advantages - adv_m)**2 * mask).sum() / cnt + 1e-8)
            return (advantages - adv_m) / (adv_s + 1e-8), targets

        # ── loss ──
        def _loss(params, h0, traj, gae, targets, ent_coef):
            _, pi, value = network.apply(params, h0.squeeze(0), (traj.obs, traj.done))
            mask  = traj.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8

            lps = [jnp.maximum(p.log_prob(traj.action[:, :, i]), jnp.log(1e-6))
                   for i, p in enumerate(pi)]
            lp     = jnp.array(lps).sum(axis=0)
            logratio = jnp.clip(lp - traj.log_prob, -20.0, 20.0)
            ratio    = jnp.exp(logratio)

            pg1 = ratio * gae
            pg2 = jnp.clip(ratio, 1 - cfg["CLIP_EPS"], 1 + cfg["CLIP_EPS"]) * gae
            actor_loss = -(jnp.minimum(pg1, pg2) * mask).sum() / denom

            entropy = (sum(p.entropy() for p in pi) * mask).sum() / denom

            vf   = cfg["VF_CLIP_EPS"]
            vclp = traj.value + (value - traj.value).clip(-vf, vf)
            def huber(x): ax = jnp.abs(x); q = jnp.minimum(ax, 1.0); l = ax - q; return 0.5*q*q + l
            vloss = (jnp.maximum(huber(value - targets), huber(vclp - targets)) * mask).sum() / denom

            kl = (((ratio - 1.0) - logratio) * mask).sum() / denom
            cf = ((jnp.abs(ratio - 1) > cfg["CLIP_EPS"]).astype(jnp.float32) * mask).sum() / denom

            total = actor_loss + cfg["VF_COEF"] * vloss - ent_coef * entropy
            return total, (vloss, actor_loss, entropy, ratio, kl, cf)

        # ── minibatch update ──
        def _update_mb(carry, mb):
            ts, ent_coef = carry
            h0, traj, adv, tgt = mb
            (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
                ts.params, h0, traj, adv, tgt, ent_coef)
            grads = jax.tree_util.tree_map(
                lambda g: jnp.nan_to_num(g), grads)
            gn   = optax.global_norm(grads)
            scl  = jnp.minimum(1.0, cfg["MAX_GRAD_NORM"] / (gn + 1e-9))
            grads = jax.tree_util.tree_map(lambda g: g * scl, grads)
            ts   = ts.apply_gradients(grads=grads)
            info = {"total_loss": loss, "value_loss": aux[0], "actor_loss": aux[1],
                    "entropy": aux[2], "approx_kl": aux[4], "clip_frac": aux[5],
                    "grad_norm": gn}
            return (ts, ent_coef), info

        # ── epoch ──
        def _epoch(update_state, _):
            ts, h0, traj, adv, tgt, rng, ent_coef = update_state
            rng, _rng = jax.random.split(rng)
            perm = jax.random.permutation(_rng, cfg["NUM_ENVS"])
            batch = (h0, traj, adv, tgt)
            sh = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=1), batch)
            mbs = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(x, [x.shape[0], cfg["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0),
                sh)
            (ts, ent_coef), loss_stack = jax.lax.scan(_update_mb, (ts, ent_coef), mbs)

            kl_mean = jnp.mean(loss_stack["approx_kl"])
            ent_lo, ent_hi = cfg["ENT_COEF_MIN"], cfg["ENT_COEF_MAX"]
            adj = cfg["ENT_ADJ_RATE"]
            ent_coef = jnp.where(kl_mean < 0.5 * cfg["TARGET_KL"],
                                 jnp.clip(ent_coef * adj, ent_lo, ent_hi),
                                 ent_coef)
            ent_coef = jnp.where(kl_mean > 1.5 * cfg["TARGET_KL"],
                                 jnp.clip(ent_coef / adj, ent_lo, ent_hi),
                                 ent_coef)
            update_state = (ts, h0, traj, adv, tgt, rng, ent_coef)
            return update_state, loss_stack

        # ── update step ──
        def _update_step(carry, _):
            runner_state, ent_coef, update_steps = carry
            h0 = runner_state[-2]  # hstate is 5th element in the 6-tuple
            runner_state, traj = jax.lax.scan(_env_step, runner_state, None, cfg["NUM_STEPS"])
            ts, env_state, last_obs, last_done, hstate, rng = runner_state
            _, _, last_val = network.apply(ts.params, hstate,
                                           (last_obs[None, :], last_done[None, :]))
            last_val = last_val.squeeze(0)
            adv, tgt = _calc_gae(traj, last_val)
            h0_stop = jax.lax.stop_gradient(h0)[None, :]

            update_state = (ts, h0_stop, traj, adv, tgt, rng, ent_coef)
            update_state, loss_info = jax.lax.scan(_epoch, update_state, None, cfg["UPDATE_EPOCHS"])
            ts       = update_state[0]
            ent_coef = update_state[6]

            metric = traj.info
            metric["loss"] = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric["ent_coef"]     = ent_coef
            metric["update_steps"] = update_steps + 1

            if cfg.get("DEBUG"):
                def callback(m):
                    steps = int(m["update_steps"]) * cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
                    ep_done = m["returned_episode"].squeeze()
                    rets = m["returned_episode_returns"][ep_done]
                    lens = m["returned_episode_lengths"][ep_done]
                    for k, v in m["loss"].items():
                        writer.add_scalar(f"loss/{k}", float(jnp.nan_to_num(v)), steps)
                    writer.add_scalar("eval/ep_return", float(rets.mean()) if rets.size else 0.0, steps)
                    writer.add_scalar("eval/ep_length", float(lens.mean()) if lens.size else 0.0, steps)
                    writer.add_scalar("sched/ent_coef", float(m["ent_coef"]), steps)
                    print(f"steps={steps:>10}  ep_ret={float(rets.mean()) if rets.size else 0.0:7.3f}"
                          f"  ep_len={float(lens.mean()) if lens.size else 0.0:6.1f}"
                          f"  ent={float(m['ent_coef']):.4f}")
                jax.experimental.io_callback(callback, None, metric)

            runner_state = (ts, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, ent_coef, update_steps + 1), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]),
            jnp.zeros((cfg["NUM_ENVS"] * cfg["NUM_ACTORS"],), dtype=bool),
            init_h,
            _rng,
        )
        ent_coef0    = jnp.array(cfg.get("ENT_COEF_INIT", cfg.get("ENT_COEF", 1e-3)), dtype=jnp.float32)
        update_steps0 = jnp.array(int(start_epoch), dtype=jnp.int32)

        (runner_state, ent_coef, update_steps), metric = jax.lax.scan(
            _update_step,
            (runner_state, ent_coef0, update_steps0),
            None,
            cfg["NUM_UPDATES"],
        )
        return {"runner_state": runner_state, "metric": metric, "rng": runner_state[5]}

    return train


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint helper
# ──────────────────────────────────────────────────────────────────────────
def save_checkpoint(train_state, epoch: int, save_dir: str):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    state = {
        "params":    train_state.params,
        "opt_state": train_state.opt_state,
        "epoch":     jnp.array(epoch),
    }
    save_path = os.path.abspath(os.path.join(save_dir, f"checkpoint_epoch_{epoch}"))
    ckptr.save(save_path, args=ocp.args.StandardSave(state))
    ckptr.wait_until_finished()


# ──────────────────────────────────────────────────────────────────────────
# Config & entry point
# ──────────────────────────────────────────────────────────────────────────
str_dt = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    # ── training scale ──
    "SEED":            42,
    "NUM_ENVS":        2000,      # parallel envs
    "NUM_STEPS":       2000,       # rollout length per env (= 1 full episode max)
    "TOTAL_TIMESTEPS": 1e8,
    "FOR_LOOP_EPOCHS": 1,         # outer for-loop (set >1 to continue training)
    # ── PPO hyperparams ──
    "LR":              3e-4,
    "ANNEAL_LR":       False,
    "UPDATE_EPOCHS":   16,
    "NUM_MINIBATCHES": 5,
    "GAMMA":           0.99,
    "GAE_LAMBDA":      0.95,
    "CLIP_EPS":        0.2,
    "VF_COEF":         1.0,
    "MAX_GRAD_NORM":   2.0,
    "ENT_COEF":        1e-3,
    "ENT_COEF_INIT":   1e-3,
    # ── network ──
    "FC_DIM_SIZE":     128,
    "GRU_HIDDEN_DIM":  128,
    "ACTIVATION":      "relu",
    # ── I/O ──
    "DEBUG":   True,
    "WANDB_API_KEY": "4c0cc04699296bed768adea4824fbaecea35dc59",
    "GROUP":         f"hold_{str_dt}",
    "OUTPUTDIR": f"results/hold_{str_dt}",
    "LOGDIR":    f"results/hold_{str_dt}/logs",
    "SAVEDIR":   f"results/hold_{str_dt}/checkpoints",
    # "LOADDIR": "/path/to/previous/checkpoint",
}

if __name__ == "__main__":
    import jax.experimental  # noqa: F401

    wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
    wandb.init(
        project="AeroPlanax",
        config=config,
        name=config['GROUP'],
        group=config['GROUP'],
        notes="Altitude/Speed/Attitude Hold task — PPO + GRU",
        reinit=True,
    )

    Path(config["OUTPUTDIR"]).mkdir(parents=True, exist_ok=True)
    Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = make_train(config)

    latest_ckpt = config.get("LOADDIR", None)

    for i in range(config["FOR_LOOP_EPOCHS"]):
        if latest_ckpt:
            config["LOADDIR"] = latest_ckpt
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)
        rng = out["rng"]

        # Save checkpoint after every outer epoch
        runner_state = out["runner_state"]
        train_state_saved = runner_state[0]
        epoch_num = int(out["metric"]["update_steps"].ravel()[-1])
        save_checkpoint(train_state_saved, epoch_num, config["SAVEDIR"])
        latest_ckpt = os.path.abspath(os.path.join(config["SAVEDIR"], f"checkpoint_epoch_{epoch_num}"))
        print(f"[Saved] checkpoint_epoch_{epoch_num}")

    wandb.finish()
    print("Training complete.")
