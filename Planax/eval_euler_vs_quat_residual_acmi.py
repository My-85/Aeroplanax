"""
Generate ACMI files comparing Euler baseline vs Quat+Residual on a loop-plane arc.

Euler:       results/heading_pitch_V_discrete_rnn_2026-05-14-15-29/checkpoints/checkpoint_epoch_300
Quat base:   results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619
Residual:    results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/checkpoint/residual_checkpoint_update_2

Task: 150° loop-plane arc (R=15000m) — where Euler representation degrades due to gimbal-lock
      near inverted attitudes, while quaternion + residual specialist handles the transition.
"""
import os, sys, json, time
_px = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _px)
import jax, jax.numpy as jnp, numpy as np
import orbax.checkpoint as ocp
from datetime import datetime

# ── Euler env ──
from envs.aeroplanax_heading_pitch_V import (
    AeroPlanaxHeading_Pitch_V_Env as EulerEnv,
    Heading_Pitch_V_TaskParams as EulerParams,
)
from envs.utils.utils import wrap_PI

# ── Quat env (vertical energy — native for 619 + residual) ──
from envs.aeroplanax_heading_pitch_V_quaternion_version_vertical_energy import (
    AeroPlanaxHeading_Pitch_V_Env as VertEnv,
    Heading_Pitch_V_TaskParams as VertParams,
)

# ── Loop-plane targets ──
from experiments.hierarchical_trajectory_tracking.loop_attitude_target import loop_plane_hpr_jax
from experiments.hierarchical_trajectory_tracking.export_acmi import write_acmi

# ── Networks ──
import functools, distrax, flax.linen as nn
from typing import Dict, Sequence
from flax.linen.initializers import constant, orthogonal
from half_loop_residual_policy import (
    ResidualActorCriticRNN, ResidualScannedRNN,
    augment_obs_flat, combine_base_and_residual_logits,
)

OUT_DIR = os.path.join(_px, 'results/euler_vs_quat_residual_acmi')

# Checkpoints
EULER_CKPT = os.path.join(
    _px, 'results/heading_pitch_V_discrete_rnn_2026-05-14-15-29/checkpoints/checkpoint_epoch_300')
QUAT_CKPT  = os.path.join(
    _px, 'results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619')
RESIDUAL_CKPT = os.path.join(
    _px, 'results/half_loop_specialist_residual_v1/20260518_1803/checkpoint/round_01/'
    'checkpoint/residual_checkpoint_update_2')

NET_CONFIG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
RESIDUAL_CFG = {
    "ACTIVATION": "relu", "RESIDUAL_FC_DIM_SIZE": 96,
    "RESIDUAL_GRU_HIDDEN_DIM": 64, "RESIDUAL_LOGIT_CLIP": 1.25,
    "RESIDUAL_GATE_START_DEG": 80.0, "RESIDUAL_GATE_END_DEG": 180.0,
}
DT_RL = 10.0 / 50.0
G = 9.80665


class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0,
                       split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry; ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis],
                              self.initialize_carry(*rnn_state.shape), rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence; config: Dict
    @nn.compact
    def __call__(self, hidden, x):
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x
        embedding = nn.Dense(self.config["FC_DIM_SIZE"],
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))
        fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                       bias_init=constant(0.0))(embedding)
        fc2 = nn.LayerNorm()(fc2); fc2 = activation(fc2)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"],
                              kernel_init=orthogonal(2),
                              bias_init=constant(0.0))(fc2)
        actor_mean = activation(actor_mean)
        pis = []
        for i, ad in enumerate(self.action_dim):
            if i == 4:
                pis.append(distrax.Categorical(logits=nn.Dense(
                    ad, kernel_init=constant(0.0),
                    bias_init=lambda key, shape, dtype=jnp.float32: jnp.array(
                        [0.0, -1.5, -1.5, -1.5, -1.5], dtype=dtype))(actor_mean)))
            else:
                pis.append(distrax.Categorical(logits=nn.Dense(
                    ad, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)))
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2),
                          bias_init=constant(0.0))(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, tuple(pis), jnp.squeeze(critic, axis=-1)


def restore_params(path):
    ckpt = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        str(path), args=ocp.args.StandardRestore())
    return ckpt["params"], int(np.asarray(ckpt["epoch"]))


def _f(x):
    a = np.asarray(x); return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


# ═══════════════════════════════════════════════════════════
def run_euler(task_angle_deg=150.0, radius=15000.0, max_steps=700):
    """Run Euler baseline on a loop-plane arc."""
    print(f'\n=== Euler baseline — {task_angle_deg:.0f}° loop-plane arc ===')
    env_params = EulerParams()
    env = EulerEnv(env_params)
    agent = env.agents[0]
    network = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CONFIG)
    params, epoch = restore_params(EULER_CKPT)
    print(f'  Epoch {epoch}')

    _, state = env.reset(jax.random.PRNGKey(42), env_params)
    state = state.replace(plane_state=state.plane_state.replace(
        yaw=jnp.array([0.0]), pitch=jnp.array([0.0]), roll=jnp.array([0.0]),
        q0=jnp.array([1.0]), q1=jnp.array([0.0]), q2=jnp.array([0.0]), q3=jnp.array([0.0])),
        target_heading=jnp.array([0.0]), last_check_time=state.time)
    hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    done = jnp.zeros((1,), dtype=jnp.bool_)

    ramp_steps = max(10, int(np.ceil(np.deg2rad(task_angle_deg) * radius / 250.0 / DT_RL)))
    max_steps_total = min(ramp_steps + 200, max_steps)

    rec = {'t': [], 'n': [], 'e': [], 'a': [], 'vt': [], 'roll': [], 'pitch': [],
           'yaw': [], 'alpha': [], 'beta': [], 'G': [], 'cte': []}
    crashed = False

    for step in range(max_steps_total):
        frac = np.clip(step / max(ramp_steps - 1, 1), 0.0, 1.0)
        frac = frac * frac * (3.0 - 2.0 * frac)
        theta = np.deg2rad(task_angle_deg) * frac
        th, tp, tr = loop_plane_hpr_jax(theta, 0.0, 1.0)
        th_np = float(np.asarray(th)); tp_np = float(np.asarray(tp)); tr_np = float(np.asarray(tr))
        tv = jnp.full((1,), 250.0)

        state = state.replace(target_heading=jnp.array([th_np]),
                              target_pitch=jnp.array([tp_np]),
                              target_roll=jnp.array([tr_np]),
                              target_vt=tv, last_check_time=state.time)
        obs_dict = env._get_obs(state, env_params)
        obs = obs_dict[agent]
        hstate, pi, _ = network.apply(params, hstate, (obs[None, None, :], done[None, :]))
        acts = jnp.array([jnp.argmax(p.logits[0, 0]) for p in pi])
        _, state, _, done_dict, _ = env.step(
            jax.random.PRNGKey(42 + step), state, {agent: acts}, env_params)
        crashed = bool(done_dict[agent])

        if crashed: break

        ps = state.plane_state
        rec['t'].append(step * 0.2); rec['n'].append(_f(ps.north)); rec['e'].append(_f(ps.east))
        rec['a'].append(_f(ps.altitude)); rec['vt'].append(_f(ps.vt))
        rec['roll'].append(np.degrees(_f(ps.roll))); rec['pitch'].append(np.degrees(_f(ps.pitch)))
        rec['yaw'].append(np.degrees(_f(ps.yaw)))
        rec['alpha'].append(np.degrees(_f(ps.alpha))); rec['beta'].append(np.degrees(_f(ps.beta)))
        rec['G'].append(float(np.sqrt(_f(ps.ax)**2 + _f(ps.ay)**2 + _f(ps.az)**2)))
        rec['cte'].append(0.0)

    write_acmi(os.path.join(out_dir, 'Euler_baseline.acmi'),
               _generate_loop_wps(task_angle_deg, radius), rec)
    print(f'  {"CRASH" if crashed else "OK"}  steps={len(rec["t"])}  '
          f'alt {min(rec["a"]):.0f}-{max(rec["a"]):.0f}m  Gmax={max(rec["G"]):.1f}g')
    return rec, crashed


# ═══════════════════════════════════════════════════════════
def _generate_loop_wps(angle_deg, radius, n_points=60):
    """Generate waypoints for a loop-plane arc (for ACMI export)."""
    theta = np.linspace(0, np.radians(angle_deg), n_points)
    forward = radius * np.sin(theta)
    up = radius * (1.0 - np.cos(theta))
    wps = np.column_stack([forward, np.zeros(n_points), 5000.0 + up])
    return wps


def run_quat_residual(task_angle_deg=150.0, radius=15000.0, max_steps=700):
    """Run Quat base (epoch 619) + residual specialist on a loop-plane arc."""
    print(f'\n=== Quat+Residual — {task_angle_deg:.0f}° loop-plane arc ===')
    env_params = VertParams()
    env = VertEnv(env_params)
    agent = env.agents[0]

    base_network = ActorCriticRNN([31, 41, 41, 41, 5], config=NET_CONFIG)
    base_params, base_epoch = restore_params(QUAT_CKPT)
    print(f'  Base epoch {base_epoch}')

    residual_network = ResidualActorCriticRNN([31, 41, 41, 41, 5], config=RESIDUAL_CFG)
    residual_params, residual_epoch = restore_params(RESIDUAL_CKPT)
    print(f'  Residual epoch {residual_epoch}')

    _, state = env.reset(jax.random.PRNGKey(42), env_params)
    state = state.replace(plane_state=state.plane_state.replace(
        yaw=jnp.array([0.0]), pitch=jnp.array([0.0]), roll=jnp.array([0.0]),
        q0=jnp.array([1.0]), q1=jnp.array([0.0]), q2=jnp.array([0.0]), q3=jnp.array([0.0])),
        task_mode=jnp.array([0], dtype=jnp.int32),
        task_duration_steps=jnp.array([10000.0], dtype=jnp.float32),
        last_check_time=state.time)

    hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    residual_hstate = ResidualScannedRNN.initialize_carry(
        1, RESIDUAL_CFG["RESIDUAL_GRU_HIDDEN_DIM"])
    done = jnp.zeros((1,), dtype=jnp.bool_)

    ramp_steps = max(10, int(np.ceil(np.deg2rad(task_angle_deg) * radius / 250.0 / DT_RL)))
    max_steps_total = min(ramp_steps + 200, max_steps)

    rec = {'t': [], 'n': [], 'e': [], 'a': [], 'vt': [], 'roll': [], 'pitch': [],
           'yaw': [], 'alpha': [], 'beta': [], 'G': [], 'cte': []}
    crashed = False

    for step in range(max_steps_total):
        frac = np.clip(step / max(ramp_steps - 1, 1), 0.0, 1.0)
        frac = frac * frac * (3.0 - 2.0 * frac)
        theta = np.deg2rad(task_angle_deg) * frac
        th, tp, tr = loop_plane_hpr_jax(theta, 0.0, 1.0)
        th_np = float(np.asarray(th)); tp_np = float(np.asarray(tp)); tr_np = float(np.asarray(tr))
        tv = jnp.full((1,), 250.0)

        state = state.replace(target_heading=jnp.array([th_np]),
                              target_pitch=jnp.array([tp_np]),
                              target_roll=jnp.array([tr_np]),
                              target_vt=tv, last_check_time=state.time)
        obs_dict = env._get_obs(state, env_params)
        obs = obs_dict[agent]

        # Base policy forward
        hstate, base_pi, _ = base_network.apply(
            base_params, hstate, (obs[None, None, :], done[None, :]))

        # Residual specialist forward
        obs_flat = obs[None, :]  # add batch dim: (obs_dim,) → (1, obs_dim)
        obs_aug = augment_obs_flat(obs_flat, state, RESIDUAL_CFG)
        residual_hstate, residual_logits, _ = residual_network.apply(
            residual_params, residual_hstate,
            (obs_aug[None, :, :], done[None, :]))

        # Combine
        pi, _, _ = combine_base_and_residual_logits(
            base_pi, residual_logits, obs_aug, RESIDUAL_CFG)
        acts = jnp.array([jnp.argmax(p.logits[0, 0]) for p in pi])

        _, state, _, done_dict, _ = env.step(
            jax.random.PRNGKey(42 + step), state, {agent: acts}, env_params)
        crashed = bool(done_dict[agent])

        if crashed: break

        ps = state.plane_state
        rec['t'].append(step * 0.2); rec['n'].append(_f(ps.north)); rec['e'].append(_f(ps.east))
        rec['a'].append(_f(ps.altitude)); rec['vt'].append(_f(ps.vt))
        rec['roll'].append(np.degrees(_f(ps.roll))); rec['pitch'].append(np.degrees(_f(ps.pitch)))
        rec['yaw'].append(np.degrees(_f(ps.yaw)))
        rec['alpha'].append(np.degrees(_f(ps.alpha))); rec['beta'].append(np.degrees(_f(ps.beta)))
        rec['G'].append(float(np.sqrt(_f(ps.ax)**2 + _f(ps.ay)**2 + _f(ps.az)**2)))
        rec['cte'].append(0.0)

    write_acmi(os.path.join(out_dir, 'Quat_Residual.acmi'),
               _generate_loop_wps(task_angle_deg, radius), rec)
    print(f'  {"CRASH" if crashed else "OK"}  steps={len(rec["t"])}  '
          f'alt {min(rec["a"]):.0f}-{max(rec["a"]):.0f}m  Gmax={max(rec["G"]):.1f}g')
    return rec, crashed


# ═══════════════════════════════════════════════════════════
def main():
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    global out_dir
    out_dir = os.path.join(OUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output: {out_dir}')

    # 150° loop-plane arc — where Euler gimbal-lock becomes visible
    # 90° arc — milder, both should handle
    for angle, radius, max_steps in [(150, 15000, 800)]:
        run_euler(angle, radius, max_steps)
        run_quat_residual(angle, radius, max_steps)

    print(f'\nACMI files: {out_dir}/')
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith('.acmi'):
            print(f'  {fn}  ({os.path.getsize(os.path.join(out_dir, fn))/1024:.0f} KB)')
    print('\nDONE — Open both ACMI files in Tacview side-by-side.')


if __name__ == '__main__':
    main()
