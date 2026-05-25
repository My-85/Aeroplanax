import functools
from typing import Dict, Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


RESIDUAL_EXTRA_DIM = 7


def wrap_pi(x):
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def smooth01(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def flatten_agent_axis(x, batch_size):
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return jnp.full((batch_size,), arr, dtype=jnp.float32)
    arr = jnp.reshape(arr, (-1,))
    if arr.shape[0] == batch_size:
        return arr
    if arr.shape[0] == 1:
        return jnp.full((batch_size,), arr[0], dtype=arr.dtype)
    return arr[:batch_size]


def residual_gate_from_aug_obs(obs_aug):
    return obs_aug[..., -RESIDUAL_EXTRA_DIM]


def residual_phase_deg_from_aug_obs(obs_aug):
    return obs_aug[..., -RESIDUAL_EXTRA_DIM + 1] * 180.0


def phase_features_from_state(
    state,
    batch_size,
    gate_start_deg=80.0,
    gate_end_deg=180.0,
    phase_max_deg=None,
    smooth_gate_margin_deg=0.0,
):
    state = getattr(state, "env_state", state)
    time = flatten_agent_axis(getattr(state, "time", 0.0), batch_size)
    last_check = flatten_agent_axis(getattr(state, "last_check_time", 0.0), batch_size)
    duration = flatten_agent_axis(getattr(state, "task_duration_steps", 1.0), batch_size)
    frac = smooth01((time - last_check) / jnp.maximum(duration, 1.0))

    arc_start = flatten_agent_axis(getattr(state, "task_arc_start_angle", 0.0), batch_size)
    arc_delta = flatten_agent_axis(getattr(state, "task_arc_angle", 0.0), batch_size)
    if phase_max_deg is None:
        phase_max_deg = max(float(gate_end_deg), 180.0)
    theta_max = jnp.deg2rad(jnp.asarray(phase_max_deg, dtype=jnp.float32))
    theta = jnp.clip(arc_start + arc_delta * frac, 0.0, theta_max)
    theta_deg = jnp.rad2deg(theta)

    mode = flatten_agent_axis(getattr(state, "task_mode", 0), batch_size)
    loop_mode = ((mode > 4.5) & (mode < 5.5)) | ((mode > 8.5) & (mode < 9.5))
    hard_gate = loop_mode & (theta_deg >= gate_start_deg) & (theta_deg <= gate_end_deg)
    if smooth_gate_margin_deg > 0.0:
        margin = jnp.asarray(smooth_gate_margin_deg, dtype=jnp.float32)
        start_w = smooth01((theta_deg - gate_start_deg) / jnp.maximum(margin, 1e-3))
        end_w = smooth01((gate_end_deg - theta_deg) / jnp.maximum(margin, 1e-3))
        gate = loop_mode.astype(jnp.float32) * start_w * end_w
    else:
        gate = hard_gate.astype(jnp.float32)
    return theta_deg, gate.astype(jnp.float32)


def residual_extra_features(obs_flat, state, phase_deg, gate):
    state = getattr(state, "env_state", state)
    batch_size = obs_flat.shape[0]
    roll = flatten_agent_axis(state.plane_state.roll, batch_size)
    pitch = flatten_agent_axis(state.plane_state.pitch, batch_size)
    vt = flatten_agent_axis(state.plane_state.vt, batch_size)
    target_roll = flatten_agent_axis(state.target_roll, batch_size)
    target_pitch = flatten_agent_axis(state.target_pitch, batch_size)
    target_vt = flatten_agent_axis(state.target_vt, batch_size)

    phase_norm = jnp.clip(phase_deg / 180.0, 0.0, 2.0)
    phase_rad = jnp.deg2rad(phase_deg)
    roll_err = wrap_pi(roll - target_roll) / jnp.pi
    pitch_err = wrap_pi(pitch - target_pitch) / jnp.pi
    vt_err = jnp.clip((vt - target_vt) / 100.0, -3.0, 3.0)

    return jnp.stack(
        [
            gate.astype(jnp.float32),
            phase_norm.astype(jnp.float32),
            jnp.sin(phase_rad).astype(jnp.float32),
            jnp.cos(phase_rad).astype(jnp.float32),
            roll_err.astype(jnp.float32),
            pitch_err.astype(jnp.float32),
            vt_err.astype(jnp.float32),
        ],
        axis=-1,
    )


def augment_obs_flat(obs_flat, state, config: Dict):
    batch_size = obs_flat.shape[0]
    phase_deg, gate = phase_features_from_state(
        state,
        batch_size,
        gate_start_deg=float(config.get("RESIDUAL_GATE_START_DEG", 80.0)),
        gate_end_deg=float(config.get("RESIDUAL_GATE_END_DEG", 180.0)),
        phase_max_deg=float(
            config.get(
                "RESIDUAL_PHASE_MAX_DEG",
                max(
                    float(config.get("RESIDUAL_GATE_END_DEG", 180.0)),
                    180.0,
                ),
            )
        ),
        smooth_gate_margin_deg=float(config.get("RESIDUAL_SMOOTH_GATE_MARGIN_DEG", 0.0)),
    )
    extra = residual_extra_features(obs_flat, state, phase_deg, gate)
    return jnp.concatenate([obs_flat, extra], axis=-1)


def augment_obs_with_phase(obs_flat, state, phase_deg, gate, config: Dict):
    batch_size = obs_flat.shape[0]
    phase = jnp.full((batch_size,), jnp.asarray(phase_deg, dtype=jnp.float32))
    gate_arr = jnp.full((batch_size,), jnp.asarray(gate, dtype=jnp.float32))
    extra = residual_extra_features(obs_flat, state, phase, gate_arr)
    return jnp.concatenate([obs_flat, extra], axis=-1)


class ResidualScannedRNN(nn.Module):
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
            resets[:, np.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ResidualActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.config.get("ACTIVATION", "relu") == "relu" else nn.tanh
        obs, dones = x
        hidden_size = int(self.config.get("RESIDUAL_GRU_HIDDEN_DIM", 64))
        fc_size = int(self.config.get("RESIDUAL_FC_DIM_SIZE", 96))

        embedding = nn.Dense(
            fc_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)
        hidden, embedding = ResidualScannedRNN()(hidden, (embedding, dones))

        fc2 = nn.Dense(
            fc_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        fc2 = nn.LayerNorm()(fc2)
        fc2 = activation(fc2)

        actor = nn.Dense(
            hidden_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(fc2)
        actor = activation(actor)
        residual_logits = tuple(
            nn.Dense(dim, kernel_init=constant(0.0), bias_init=constant(0.0))(actor)
            for dim in self.action_dim
        )

        critic = nn.Dense(
            fc_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, residual_logits, jnp.squeeze(critic, axis=-1)


def _logits_from_dist_or_array(item):
    return item.logits if hasattr(item, "logits") else item


def combine_base_and_residual_logits(base_items, residual_logits, obs_aug, config: Dict):
    gate = residual_gate_from_aug_obs(obs_aug)
    if bool(config.get("RESIDUAL_FORCE_GATE_OFF", False)):
        gate = jnp.zeros_like(gate)
    clip = float(config.get("RESIDUAL_LOGIT_CLIP", 1.5))
    scale = float(config.get("RESIDUAL_SCALE", 1.0))
    base_logits = tuple(_logits_from_dist_or_array(item) for item in base_items)
    clipped_delta = tuple(jnp.clip(delta, -clip, clip) for delta in residual_logits)
    gate_expanded = gate[..., None]
    combined_logits = tuple(
        base + scale * gate_expanded * delta for base, delta in zip(base_logits, clipped_delta)
    )
    policies = tuple(distrax.Categorical(logits=logits) for logits in combined_logits)
    return policies, clipped_delta, gate


def residual_regularization(clipped_delta: Tuple[jnp.ndarray, ...], gate, mask, config: Dict):
    per_head = [jnp.mean(delta * delta, axis=-1) for delta in clipped_delta]
    delta_l2 = sum(per_head)
    denom = mask.sum() + 1e-8
    gated = ((delta_l2 * gate) * mask).sum() / denom
    non_loop = ((delta_l2 * (1.0 - gate)) * mask).sum() / denom
    clip = float(config.get("RESIDUAL_LOGIT_CLIP", 1.5))
    sat = sum(jnp.mean(jnp.clip(jnp.abs(delta) - 0.8 * clip, 0.0, 1e6) ** 2, axis=-1) for delta in clipped_delta)
    sat_loss = ((sat * gate) * mask).sum() / denom
    return gated, non_loop, sat_loss
