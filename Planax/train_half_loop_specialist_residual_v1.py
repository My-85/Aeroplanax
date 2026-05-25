import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, NamedTuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from envs.aeroplanax_heading_pitch_V_quaternion_version_vertical_energy import (
    AeroPlanaxHeading_Pitch_V_Env,
    Heading_Pitch_V_TaskParams,
)
from envs.wrappers import LogWrapper
from eval_vertical_energy_checkpoints import ActorCriticRNN, NET_CONFIG, ScannedRNN
from half_loop_residual_policy import (
    RESIDUAL_EXTRA_DIM,
    ResidualActorCriticRNN,
    ResidualScannedRNN,
    augment_obs_flat,
    combine_base_and_residual_logits,
    flatten_agent_axis,
    residual_phase_deg_from_aug_obs,
    residual_gate_from_aug_obs,
    residual_regularization,
)


PLANAX_ROOT = Path(__file__).resolve().parent
BASELINE_CKPT = (
    PLANAX_ROOT
    / "results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619"
)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


class ResidualTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Dict[str, Any]
    valid_action: jnp.ndarray
    base_logits: Any
    gate: jnp.ndarray
    phase_deg: jnp.ndarray
    task_mode: jnp.ndarray


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def restore_base_params(path: str):
    ckpt = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        str(Path(path).resolve()), args=ocp.args.StandardRestore()
    )
    return ckpt["params"], int(np.asarray(ckpt.get("epoch", 0)))


def restore_residual(path: str, state):
    ckpt = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        str(Path(path).resolve()), args=ocp.args.StandardRestore(item=state)
    )
    return ckpt


def make_train(config):
    cfg = dict(config)
    cfg.setdefault("VF_CLIP_EPS", 0.20)
    cfg.setdefault("HUBER_DELTA", 1.0)
    cfg.setdefault("RESIDUAL_L2_COEF", 0.02)
    cfg.setdefault("NON_LOOP_RESIDUAL_L2_COEF", 0.20)
    cfg.setdefault("RESIDUAL_SATURATION_COEF", 0.02)
    cfg.setdefault("RESIDUAL_LOGIT_CLIP", 1.25)
    cfg.setdefault("RESIDUAL_GATE_START_DEG", 80.0)
    cfg.setdefault("RESIDUAL_GATE_END_DEG", 180.0)
    cfg.setdefault("RESIDUAL_SMOOTH_GATE_MARGIN_DEG", 0.0)
    cfg.setdefault("RESIDUAL_FC_DIM_SIZE", 96)
    cfg.setdefault("RESIDUAL_GRU_HIDDEN_DIM", 64)
    cfg.setdefault("ANCHOR_RESIDUAL_LOADDIR", "")
    cfg.setdefault("ANCHOR_BC_COEF", 0.0)
    cfg.setdefault("ANCHOR_PHASE_START_DEG", 90.0)
    cfg.setdefault("ANCHOR_PHASE_END_DEG", 165.0)

    env_params = Heading_Pitch_V_TaskParams(**cfg.get("ENV_PARAMS", {}))
    env = LogWrapper(AeroPlanaxHeading_Pitch_V_Env(env_params))
    cfg["NUM_ACTORS"] = env.num_agents
    cfg["NUM_UPDATES"] = cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"]
    cfg["MINIBATCH_SIZE"] = cfg["NUM_ACTORS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]

    base_checkpoint = cfg["BASE_CHECKPOINT"]
    base_params, base_epoch = restore_base_params(base_checkpoint)
    del base_epoch

    base_net = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CONFIG)
    residual_net = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=cfg)
    obs_shape = env.observation_space(env.agents[0], env_params).shape
    aug_obs_dim = obs_shape[0] + RESIDUAL_EXTRA_DIM

    def residual_restore_item():
        rng = jax.random.PRNGKey(23)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"], aug_obs_dim)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"])),
        )
        init_h = ResidualScannedRNN.initialize_carry(
            cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["RESIDUAL_GRU_HIDDEN_DIM"]
        )
        params = residual_net.init(rng, init_h, init_x)
        tx = optax.adam(cfg["LR"], eps=1e-5)
        train_state = TrainState.create(apply_fn=residual_net.apply, params=params, tx=tx)
        restore_item = {
            "params": train_state.params,
            "opt_state": train_state.opt_state,
            "epoch": jnp.array(0),
        }
        return restore_item

    if cfg.get("RESIDUAL_LOADDIR"):
        restore_item = residual_restore_item()
        residual_checkpoint = restore_residual(cfg["RESIDUAL_LOADDIR"], restore_item)
    else:
        residual_checkpoint = None

    if cfg.get("ANCHOR_RESIDUAL_LOADDIR"):
        anchor_checkpoint = restore_residual(cfg["ANCHOR_RESIDUAL_LOADDIR"], residual_restore_item())
        anchor_params = anchor_checkpoint["params"]
    else:
        anchor_params = None

    def train(rng):
        rng, init_rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"], aug_obs_dim)),
            jnp.zeros((1, cfg["NUM_ENVS"] * cfg["NUM_ACTORS"])),
        )
        init_h = ResidualScannedRNN.initialize_carry(
            cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["RESIDUAL_GRU_HIDDEN_DIM"]
        )
        residual_params = residual_net.init(init_rng, init_h, init_x)
        tx = optax.adam(cfg["LR"], eps=1e-5)
        train_state = TrainState.create(
            apply_fn=residual_net.apply, params=residual_params, tx=tx
        )
        if residual_checkpoint is not None:
            train_state = train_state.replace(
                params=residual_checkpoint["params"],
                opt_state=residual_checkpoint["opt_state"],
            )
            start_epoch = residual_checkpoint["epoch"]
        else:
            start_epoch = jnp.array(0)

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, cfg["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_keys)
        last_obs = batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])
        last_done = jnp.zeros((cfg["NUM_ENVS"] * cfg["NUM_ACTORS"],), dtype=bool)
        residual_hstate = ResidualScannedRNN.initialize_carry(
            cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], cfg["RESIDUAL_GRU_HIDDEN_DIM"]
        )
        base_hstate = ScannedRNN.initialize_carry(
            cfg["NUM_ACTORS"] * cfg["NUM_ENVS"], NET_CONFIG["GRU_HIDDEN_DIM"]
        )

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, residual_hstate, base_hstate, rng = runner_state

            base_in = (last_obs[None, :, :], last_done[None, :])
            base_hstate, base_pi, _ = base_net.apply(base_params, base_hstate, base_in)
            base_logits = tuple(p.logits.squeeze(0) for p in base_pi)

            obs_aug = augment_obs_flat(last_obs, env_state, cfg)
            gate_pre = residual_gate_from_aug_obs(obs_aug)
            phase_deg_pre = residual_phase_deg_from_aug_obs(obs_aug)
            inner_env_state = getattr(env_state, "env_state", env_state)
            task_mode_pre = flatten_agent_axis(
                getattr(inner_env_state, "task_mode", 0),
                cfg["NUM_ENVS"] * cfg["NUM_ACTORS"],
            ).astype(jnp.float32)
            residual_in = (obs_aug[None, :, :], last_done[None, :])
            residual_hstate, residual_logits, value = residual_net.apply(
                train_state.params, residual_hstate, residual_in
            )
            pi, _, _ = combine_base_and_residual_logits(base_pi, residual_logits, obs_aug, cfg)

            action_heads = []
            log_probs = []
            for policy in pi:
                rng, sample_rng = jax.random.split(rng)
                sampled = policy.sample(seed=sample_rng)
                action_heads.append(sampled[:, :, None])
                log_probs.append(policy.log_prob(sampled))
            action = jnp.concatenate(action_heads, axis=-1)
            log_prob = jnp.array(log_probs).sum(axis=0)
            value = value.squeeze(0)
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

            rng, step_rng = jax.random.split(rng)
            step_keys = jax.random.split(step_rng, cfg["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_keys,
                env_state,
                unbatchify(action, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]),
            )
            reward = batchify(reward, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1)
            done_flat = batchify(done, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"]).reshape(-1)
            valid_action = jnp.logical_not(jnp.logical_and(last_done, done_flat))
            transition = ResidualTransition(
                last_done,
                action,
                value,
                reward,
                log_prob,
                obs_aug,
                info,
                valid_action,
                base_logits,
                gate_pre,
                phase_deg_pre,
                task_mode_pre,
            )
            last_obs = batchify(obsv, env.agents, cfg["NUM_ENVS"], cfg["NUM_ACTORS"])

            def reset_h(h):
                return jnp.where(done_flat[:, None], jnp.zeros_like(h), h)

            residual_hstate = reset_h(residual_hstate)
            base_hstate = reset_h(base_hstate)
            runner_state = (
                train_state,
                env_state,
                last_obs,
                done_flat,
                residual_hstate,
                base_hstate,
                rng,
            )
            return runner_state, transition

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(carry, transition):
                gae, next_value = carry
                reward = jnp.nan_to_num(transition.reward, nan=0.0, posinf=0.0, neginf=0.0)
                value = jnp.nan_to_num(transition.value, nan=0.0, posinf=0.0, neginf=0.0)
                next_value = jnp.nan_to_num(next_value, nan=0.0, posinf=0.0, neginf=0.0)
                delta = reward + cfg["GAMMA"] * next_value * (1.0 - transition.done) - value
                gae = delta + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1.0 - transition.done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            targets = advantages + traj_batch.value
            mask = traj_batch.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8
            mean = (advantages * mask).sum() / denom
            var = (((advantages - mean) ** 2) * mask).sum() / denom
            advantages = (advantages - mean) / (jnp.sqrt(var + 1e-8) + 1e-8)
            return advantages, targets

        def _loss_and_aux(params, init_hstate, traj_batch, advantages, targets):
            _, residual_logits, value = residual_net.apply(
                params,
                init_hstate.squeeze(0),
                (traj_batch.obs, traj_batch.done),
            )
            pi, clipped_delta, gate = combine_base_and_residual_logits(
                traj_batch.base_logits, residual_logits, traj_batch.obs, cfg
            )
            mask = traj_batch.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8

            min_log_prob = jnp.log(1e-6)
            log_probs = [
                jnp.maximum(policy.log_prob(traj_batch.action[:, :, idx]), min_log_prob)
                for idx, policy in enumerate(pi)
            ]
            log_prob = jnp.array(log_probs).sum(axis=0)
            logratio = jnp.clip(log_prob - traj_batch.log_prob, -20.0, 20.0)
            ratio = jnp.clip(jnp.exp(logratio), 1e-6, 1e6)

            loss_actor1 = ratio * advantages
            loss_actor2 = (
                jnp.clip(ratio, 1.0 - cfg["CLIP_EPS"], 1.0 + cfg["CLIP_EPS"])
                * advantages
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = (loss_actor * mask).sum() / denom

            entropy = ((sum(policy.entropy() for policy in pi)) * mask).sum() / denom
            value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
            ).clip(-cfg["VF_CLIP_EPS"], cfg["VF_CLIP_EPS"])
            err = value - targets
            err_clip = value_pred_clipped - targets
            delta = cfg["HUBER_DELTA"]

            def huber(x):
                ax = jnp.abs(x)
                quad = jnp.minimum(ax, delta)
                lin = ax - quad
                return 0.5 * quad * quad + delta * lin

            value_loss = (
                0.5 * jnp.maximum(huber(err), huber(err_clip)) * mask
            ).sum() / denom
            approx_kl = (((ratio - 1.0) - logratio) * mask).sum() / denom
            clip_frac = ((jnp.abs(ratio - 1.0) > cfg["CLIP_EPS"]) * mask).sum() / denom
            gated_l2, non_loop_l2, residual_sat = residual_regularization(
                clipped_delta, gate, mask, cfg
            )
            gate_rate = (gate * mask).sum() / denom
            anchor_bc = jnp.asarray(0.0)
            if anchor_params is not None and float(cfg.get("ANCHOR_BC_COEF", 0.0)) > 0.0:
                _, anchor_logits, _ = residual_net.apply(
                    anchor_params,
                    init_hstate.squeeze(0),
                    (traj_batch.obs, traj_batch.done),
                )
                loop_mode = (
                    ((traj_batch.task_mode > 4.5) & (traj_batch.task_mode < 5.5))
                    | ((traj_batch.task_mode > 8.5) & (traj_batch.task_mode < 9.5))
                )
                anchor_window = (
                    loop_mode
                    & (traj_batch.phase_deg >= cfg["ANCHOR_PHASE_START_DEG"])
                    & (traj_batch.phase_deg <= cfg["ANCHOR_PHASE_END_DEG"])
                )
                anchor_mask = anchor_window.astype(jnp.float32) * mask
                anchor_denom = anchor_mask.sum() + 1e-8
                head_losses = [
                    (((res - jax.lax.stop_gradient(anchor)) ** 2) * anchor_mask[..., None]).sum()
                    / (anchor_denom * res.shape[-1] + 1e-8)
                    for res, anchor in zip(residual_logits, anchor_logits)
                ]
                anchor_bc = sum(head_losses) / max(len(head_losses), 1)

            total_loss = (
                loss_actor
                + cfg["VF_COEF"] * value_loss
                - cfg["ENT_COEF"] * entropy
                + cfg["RESIDUAL_L2_COEF"] * gated_l2
                + cfg["NON_LOOP_RESIDUAL_L2_COEF"] * non_loop_l2
                + cfg["RESIDUAL_SATURATION_COEF"] * residual_sat
                + cfg["ANCHOR_BC_COEF"] * anchor_bc
            )
            aux = {
                "total_loss": total_loss,
                "value_loss": value_loss,
                "actor_loss": loss_actor,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "clip_frac": clip_frac,
                "gate_rate": gate_rate,
                "residual_l2": gated_l2,
                "non_loop_l2": non_loop_l2,
                "residual_sat": residual_sat,
                "anchor_bc": anchor_bc,
            }
            return total_loss, aux

        def _update_minbatch(train_state, minibatch):
            init_hstate, traj_batch, advantages, targets = minibatch
            grad_fn = jax.value_and_grad(_loss_and_aux, has_aux=True)
            (total_loss, aux), grads = grad_fn(
                train_state.params, init_hstate, traj_batch, advantages, targets
            )
            grads = jax.tree_util.tree_map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            grads, _ = optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]).update(grads, None)
            train_state = train_state.apply_gradients(grads=grads)
            aux = jax.tree_util.tree_map(
                lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), aux
            )
            return train_state, aux

        def _update_epoch(update_state, unused):
            train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, cfg["NUM_ENVS"])
            batch = (init_hstate, traj_batch, advantages, targets)
            shuffled = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], cfg["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled,
            )
            train_state, loss_info = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            return (train_state, init_hstate, traj_batch, advantages, targets, rng), loss_info

        def _update_step(update_runner_state, update_steps):
            runner_state, rng = update_runner_state
            train_state, env_state, last_obs, last_done, residual_hstate, base_hstate, step_rng = runner_state
            initial_h = residual_hstate
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, cfg["NUM_STEPS"]
            )
            train_state, env_state, last_obs, last_done, residual_hstate, base_hstate, step_rng = runner_state
            obs_aug = augment_obs_flat(last_obs, env_state, cfg)
            _, _, last_val = residual_net.apply(
                train_state.params,
                residual_hstate,
                (obs_aug[None, :, :], last_done[None, :]),
            )
            advantages, targets = _calculate_gae(traj_batch, last_val.squeeze(0))
            h0 = jax.lax.stop_gradient(initial_h)[None, :]
            update_state = (train_state, h0, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, cfg["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]
            loss_mean = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            mask = traj_batch.valid_action.astype(jnp.float32)
            denom = mask.sum() + 1e-8
            loop_mode = ((traj_batch.task_mode > 4.5) & (traj_batch.task_mode < 5.5)) | (
                (traj_batch.task_mode > 8.5) & (traj_batch.task_mode < 9.5)
            )
            mode5 = (traj_batch.task_mode > 4.5) & (traj_batch.task_mode < 5.5)
            mode9 = (traj_batch.task_mode > 8.5) & (traj_batch.task_mode < 9.5)
            phase_window = (
                (traj_batch.phase_deg >= cfg["RESIDUAL_GATE_START_DEG"])
                & (traj_batch.phase_deg <= cfg["RESIDUAL_GATE_END_DEG"])
            )
            loop_mask = loop_mode.astype(jnp.float32) * mask
            loop_denom = loop_mask.sum() + 1e-8
            phase_loop = traj_batch.phase_deg * loop_mask
            phase_min = jnp.min(jnp.where(loop_mode & (mask > 0.0), traj_batch.phase_deg, 1e9))
            phase_max = jnp.max(jnp.where(loop_mode & (mask > 0.0), traj_batch.phase_deg, -1e9))
            phase_min = jnp.where(loop_mask.sum() > 0.0, phase_min, 0.0)
            phase_max = jnp.where(loop_mask.sum() > 0.0, phase_max, 0.0)
            metric = {
                "loss": loss_mean,
                "reward_mean": jnp.mean(traj_batch.reward),
                "gate_rate": (traj_batch.gate * mask).sum() / denom,
                "loop_mode_rate": (loop_mode.astype(jnp.float32) * mask).sum() / denom,
                "mode0_rate": (((traj_batch.task_mode > -0.5) & (traj_batch.task_mode < 0.5)).astype(jnp.float32) * mask).sum() / denom,
                "mode5_rate": (mode5.astype(jnp.float32) * mask).sum() / denom,
                "mode9_rate": (mode9.astype(jnp.float32) * mask).sum() / denom,
                "phase_window_rate": (phase_window.astype(jnp.float32) * mask).sum() / denom,
                "loop_phase_window_rate": ((phase_window & loop_mode).astype(jnp.float32) * mask).sum() / denom,
                "mode5_gate_rate": (traj_batch.gate * mode5.astype(jnp.float32) * mask).sum()
                / ((mode5.astype(jnp.float32) * mask).sum() + 1e-8),
                "mode9_gate_rate": (traj_batch.gate * mode9.astype(jnp.float32) * mask).sum()
                / ((mode9.astype(jnp.float32) * mask).sum() + 1e-8),
                "loop_phase_mean": phase_loop.sum() / loop_denom,
                "loop_phase_min": phase_min,
                "loop_phase_max": phase_max,
                "update_steps": update_steps + 1,
            }
            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                residual_hstate,
                base_hstate,
                step_rng,
            )
            return (runner_state, rng), metric

        runner_state = (
            train_state,
            env_state,
            last_obs,
            last_done,
            residual_hstate,
            base_hstate,
            rng,
        )
        (runner_state, rng), metric = jax.lax.scan(
            _update_step,
            (runner_state, rng),
            jnp.arange(cfg["NUM_UPDATES"]) + start_epoch,
        )
        return {
            "runner_state": runner_state,
            "epoch": start_epoch + cfg["NUM_UPDATES"],
            "metric": metric,
            "rng": rng,
        }

    return train


def default_config():
    run_root = (
        PLANAX_ROOT
        / os.environ.get("OUTPUT_ROOT", "results/half_loop_specialist_residual_v1")
        / datetime.now().strftime("%Y%m%d_%H%M")
    )
    return {
        "GROUP": "half_loop_specialist_residual_v1",
        "SEED": 42,
        "BASE_CHECKPOINT": str(BASELINE_CKPT),
        "RESIDUAL_LOADDIR": os.environ.get("RESIDUAL_LOADDIR", ""),
        "LR": float(os.environ.get("LR", 1e-5)),
        "NUM_ENVS": int(os.environ.get("NUM_ENVS", 500)),
        "NUM_ACTORS": 1,
        "NUM_STEPS": int(os.environ.get("NUM_STEPS", 512)),
        "TOTAL_TIMESTEPS": int(float(os.environ.get("TOTAL_TIMESTEPS", 512000))),
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "RESIDUAL_FC_DIM_SIZE": 96,
        "RESIDUAL_GRU_HIDDEN_DIM": 64,
        "UPDATE_EPOCHS": int(os.environ.get("UPDATE_EPOCHS", 8)),
        "NUM_MINIBATCHES": int(os.environ.get("NUM_MINIBATCHES", 5)),
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.15,
        "ENT_COEF": 7.5e-4,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "relu",
        "RESIDUAL_LOGIT_CLIP": 1.25,
        "RESIDUAL_L2_COEF": 0.02,
        "NON_LOOP_RESIDUAL_L2_COEF": 0.20,
        "RESIDUAL_SATURATION_COEF": 0.02,
        "RESIDUAL_GATE_START_DEG": 80.0,
        "RESIDUAL_GATE_END_DEG": 180.0,
        "RESIDUAL_SMOOTH_GATE_MARGIN_DEG": 0.0,
        "ANCHOR_RESIDUAL_LOADDIR": "",
        "ANCHOR_BC_COEF": 0.0,
        "ANCHOR_PHASE_START_DEG": 90.0,
        "ANCHOR_PHASE_END_DEG": 165.0,
        "OUTPUTDIR": str(run_root),
        "LOGDIR": str(run_root / "logs"),
        "SAVEDIR": str(run_root / "checkpoint"),
        "ENV_PARAMS": {
            "original_task_prob": 0.08,
            "horizontal_proxy_task_prob": 0.0,
            "level_altitude_task_prob": 0.0,
            "vertical_stage_successes": 10,
            "vertical_stage_offset": 8,
            "vertical_cruise_vt": 250.0,
            "use_loop_plane_targets_for_vertical_arc": 1.0,
            "half_loop_curriculum_prob": 1.0,
            "half_loop_pullup_retention_prob": 0.0,
            "half_loop_climb_retention_prob": 0.0,
            "half_loop_vertical_retention_prob": 0.30,
            "half_loop_transition_prob": 0.48,
            "half_loop_partial_prob": 0.22,
            "min_vertical_duration_sec": 8.0,
            "max_vertical_duration_sec": 38.0,
            "ve_low_speed_threshold": 182.0,
            "ve_strong_low_speed_threshold": 172.0,
            "ve_alpha_soft_deg": 14.0,
            "ve_alpha_hard_deg": 18.0,
            "ve_beta_soft_deg": 10.0,
            "ve_g_soft": 8.4,
            "ve_g_hard": 9.4,
            "ve_loop_geom_weight": 0.025,
            "ve_loop_roll_weight": 0.015,
            "ve_loop_nose_tangent_weight": 0.030,
            "ve_loop_wing_plane_weight": 0.035,
            "ve_loop_velocity_tangent_weight": 0.025,
            "ve_loop_nose_velocity_weight": 0.020,
            "ve_high_speed_alpha_weight": 0.030,
            "ve_action_saturation_weight": 0.025,
        },
    }


def main():
    config = default_config()
    config_json = os.environ.get("CONFIG_JSON")
    if config_json:
        with open(config_json, "r", encoding="utf-8") as f:
            deep_update(config, json.load(f))
        config["CONFIG_JSON"] = config_json

    if config["NUM_ENVS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError("NUM_ENVS must be divisible by NUM_MINIBATCHES")
    if float(config["LR"]) > 1e-5:
        raise ValueError("Residual specialist LR must stay <= 1e-5")

    output_dir = Path(config["OUTPUTDIR"])
    save_dir = Path(config["SAVEDIR"])
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpoint = {
        "params": out["runner_state"][0].params,
        "opt_state": out["runner_state"][0].opt_state,
        "epoch": jnp.array(out["epoch"]),
    }
    checkpoint_path = save_dir / f"residual_checkpoint_update_{int(np.asarray(out['epoch']))}"
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    ckptr.save(str(checkpoint_path), args=ocp.args.StandardSave(checkpoint))
    ckptr.wait_until_finished()

    metric = out["metric"]
    loss_total = np.asarray(metric["loss"]["total_loss"]).reshape(-1)
    plt.figure(figsize=(8, 4))
    plt.plot(loss_total)
    plt.xlabel("update")
    plt.ylabel("total_loss")
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png", dpi=120)
    plt.close()

    final_loss = float(loss_total[-1]) if loss_total.size else 0.0
    with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "updates",
                "final_total_loss",
                "final_anchor_bc",
                "base_checkpoint",
                "residual_load_checkpoint",
                "anchor_residual_checkpoint",
                "saved_checkpoint",
                "gate_rate_mean",
                "loop_mode_rate_mean",
                "mode0_rate_mean",
                "mode5_rate_mean",
                "mode9_rate_mean",
                "phase_window_rate_mean",
                "loop_phase_window_rate_mean",
                "mode5_gate_rate_mean",
                "mode9_gate_rate_mean",
                "loop_phase_min",
                "loop_phase_mean",
                "loop_phase_max",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": int(np.asarray(out["epoch"])),
                "updates": int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]),
                "final_total_loss": f"{final_loss:.8f}",
                "final_anchor_bc": f"{float(np.asarray(metric['loss'].get('anchor_bc', 0.0)).reshape(-1)[-1]):.8f}",
                "base_checkpoint": config["BASE_CHECKPOINT"],
                "residual_load_checkpoint": config.get("RESIDUAL_LOADDIR", ""),
                "anchor_residual_checkpoint": config.get("ANCHOR_RESIDUAL_LOADDIR", ""),
                "saved_checkpoint": str(checkpoint_path.resolve()),
                "gate_rate_mean": f"{float(np.asarray(metric['gate_rate']).mean()):.8f}",
                "loop_mode_rate_mean": f"{float(np.asarray(metric['loop_mode_rate']).mean()):.8f}",
                "mode0_rate_mean": f"{float(np.asarray(metric['mode0_rate']).mean()):.8f}",
                "mode5_rate_mean": f"{float(np.asarray(metric['mode5_rate']).mean()):.8f}",
                "mode9_rate_mean": f"{float(np.asarray(metric['mode9_rate']).mean()):.8f}",
                "phase_window_rate_mean": f"{float(np.asarray(metric['phase_window_rate']).mean()):.8f}",
                "loop_phase_window_rate_mean": f"{float(np.asarray(metric['loop_phase_window_rate']).mean()):.8f}",
                "mode5_gate_rate_mean": f"{float(np.asarray(metric['mode5_gate_rate']).mean()):.8f}",
                "mode9_gate_rate_mean": f"{float(np.asarray(metric['mode9_gate_rate']).mean()):.8f}",
                "loop_phase_min": f"{float(np.asarray(metric['loop_phase_min']).min()):.4f}",
                "loop_phase_mean": f"{float(np.asarray(metric['loop_phase_mean']).mean()):.4f}",
                "loop_phase_max": f"{float(np.asarray(metric['loop_phase_max']).max()):.4f}",
            }
        )

    manifest = {
        "architecture": "frozen_epoch619_plus_phase_gated_residual_logits_v1",
        "base_checkpoint": config["BASE_CHECKPOINT"],
        "residual_checkpoint": str(checkpoint_path.resolve()),
        "config": str((output_dir / "config.json").resolve()),
        "gate": [config["RESIDUAL_GATE_START_DEG"], config["RESIDUAL_GATE_END_DEG"]],
        "notes": "Base policy params are frozen. Residual logits are added only inside the phase gate.",
    }
    with (output_dir / "residual_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    report = [
        "# Half-Loop Specialist Residual V1 Training Report",
        "",
        f"- base checkpoint: `{config['BASE_CHECKPOINT']}`",
        f"- residual load checkpoint: `{config.get('RESIDUAL_LOADDIR', '')}`",
        f"- saved residual checkpoint: `{checkpoint_path.resolve()}`",
        f"- total timesteps: `{config['TOTAL_TIMESTEPS']}`",
        f"- learning rate: `{config['LR']}`",
        f"- residual gate: `{config['RESIDUAL_GATE_START_DEG']}..{config['RESIDUAL_GATE_END_DEG']} deg`",
        "",
        "The checkpoint contains only residual policy parameters. It must be evaluated with the frozen base checkpoint.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"saved_checkpoint={checkpoint_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
