"""
render_ablation_comparison.py
==============================
Render a single ACMI file containing TWO aircraft flying the S-maneuver
simultaneously in the HIGH-FIDELITY environment:
  - Aircraft 100 (Red):  LoFi-trained policy  (expected to crash quickly)
  - Aircraft 200 (Blue): HiFi-trained policy  (expected to track well)

Each aircraft has a corresponding target heading indicator that moves with
the S-maneuver schedule.

Output: tracks/ablation_comparison/<timestamp>.txt.acmi

Usage:
  cd Planax
  CUDA_VISIBLE_DEVICES=0 python render_ablation_comparison.py \
    --lofi_ckpt results/ablation_lofi_2026-04-20-02-04/checkpoints/checkpoint_epoch_2000 \
    --hifi_ckpt results/heading_pitch_V_discrete_rnn_2026-03-30-00-57/checkpoints/checkpoint_epoch_1375 \
    [--max_steps 2000] [--seed 0]
"""

import os
import argparse
import functools
from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import optax
import orbax.checkpoint as ocp
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from envs.aeroplanax_s_maneuver_ablation import (
    AeroPlanaxSManeuverAblationEnv,
    SManeuverTaskParams,
)
from envs.wrappers import LogWrapper
from envs.utils.utils import enu_to_geodetic, wrap_PI


# =============================================================================
# Network (identical to training)
# =============================================================================

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
        embedding = nn.Dense(self.config["FC_DIM_SIZE"],
                             kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"],
                              kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)
        pi_t = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_e = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_a = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_r = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, (pi_t, pi_e, pi_a, pi_r), jnp.squeeze(critic, axis=-1)


NET_CONFIG = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}


# =============================================================================
# Helpers
# =============================================================================

def _f(x, i=0):
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    return float(a.reshape(-1)[min(i, a.size - 1)])


def load_params(ckpt_dir: str):
    ckpt_dir = os.path.abspath(ckpt_dir)
    network = ActorCriticRNN([31, 41, 41, 41], config=NET_CONFIG)
    rng = jax.random.PRNGKey(0)
    dummy_obs  = jnp.zeros((1, 1, 16))
    dummy_done = jnp.zeros((1, 1))
    init_h     = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    net_params = network.init(rng, init_h, (dummy_obs, dummy_done))
    tx         = optax.adam(3e-4)
    ts         = TrainState.create(apply_fn=network.apply, params=net_params, tx=tx)
    template   = {"params": ts.params, "opt_state": ts.opt_state, "epoch": jnp.array(0)}
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    try:
        restored = ckptr.restore(ckpt_dir, args=ocp.args.StandardRestore(item=template))
    except Exception:
        restored = ckptr.restore(ckpt_dir, item=template)
    return restored["params"]


def make_greedy_step(network, params):
    @jax.jit
    def step(obs, done, hstate):
        # obs: (1, 16), done: (1,)
        new_h, pi, _ = network.apply(params, hstate, (obs[None, :], done[None, :]))
        a = jnp.stack([jnp.argmax(p.logits, axis=-1)[0, 0] for p in pi], axis=-1)
        return a, new_h
    return step


def target_indicator_pos(north, east, altitude, target_heading, dist=2000.0):
    """Place a marker 2 km ahead in the target heading direction at same altitude."""
    tn = north + dist * np.cos(target_heading)
    te = east  + dist * np.sin(target_heading)
    return tn, te, altitude


# =============================================================================
# Main render
# =============================================================================

def render(args):
    os.environ.setdefault("XLA_PYTHON_MEM_FRACTION", "0.80")

    # ── Build two independent HiFi environments (same params, different seeds) ──
    env_params = SManeuverTaskParams(
        fidelity_mode  = "high",
        s_switch_steps = args.s_switch_steps,
        s_heading_tol  = float(np.deg2rad(args.s_heading_tol)) if args.s_heading_tol > 0 else 0.0,
    )

    def make_env():
        e = AeroPlanaxSManeuverAblationEnv(env_params)
        return LogWrapper(e)

    env_lo = make_env()
    env_hi = make_env()

    # ── Load policies ──────────────────────────────────────────────────────────
    print(f"Loading LoFi policy: {args.lofi_ckpt}")
    params_lo = load_params(args.lofi_ckpt)
    print(f"Loading HiFi policy: {args.hifi_ckpt}")
    params_hi = load_params(args.hifi_ckpt)

    network = ActorCriticRNN([31, 41, 41, 41], config=NET_CONFIG)
    step_lo = make_greedy_step(network, params_lo)
    step_hi = make_greedy_step(network, params_hi)

    # ── Reset both envs ────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(args.seed)
    rng, r1, r2 = jax.random.split(rng, 3)

    obs_lo, state_lo = env_lo.reset(r1)
    obs_hi, state_hi = env_hi.reset(r2)

    obs_lo_flat = jnp.stack([obs_lo[a] for a in env_lo.agents], axis=0).reshape(1, 16)
    obs_hi_flat = jnp.stack([obs_hi[a] for a in env_hi.agents], axis=0).reshape(1, 16)
    done_lo = jnp.zeros((1,), dtype=bool)
    done_hi = jnp.zeros((1,), dtype=bool)
    h_lo = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    h_hi = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])

    crashed_lo = False
    crashed_hi = False

    # ── ACMI output ────────────────────────────────────────────────────────────
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    acmi_path = outdir / f"ablation_comparison_{ts_str}.txt.acmi"

    with open(acmi_path, "w", encoding="utf-8") as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")
        # Object definitions (static properties written once)
        f.write("100,Type=Air+FixedWing,Name=F16_LoFi,Color=Red,Pilot=LoFi-policy\n")
        f.write("200,Type=Air+FixedWing,Name=F16_HiFi,Color=Blue,Pilot=HiFi-policy\n")
        f.write("1100,Type=Navaid+Static+Waypoint,Name=Target_LoFi,Color=Red\n")
        f.write("1200,Type=Navaid+Static+Waypoint,Name=Target_HiFi,Color=Blue\n")

    print(f"ACMI: {acmi_path}")
    print(f"Running {args.max_steps} steps ...")

    # ── Simulation loop ────────────────────────────────────────────────────────
    for t in range(args.max_steps):
        sim_time = t * env_params.agent_interaction_steps / env_params.sim_freq

        # --- LoFi step ---
        if not crashed_lo:
            action_lo, h_lo = step_lo(obs_lo_flat, done_lo, h_lo)
            rng, _r = jax.random.split(rng)
            obs_lo, state_lo, _, done_lo_dict, _ = env_lo.step(
                _r, state_lo, {a: action_lo for a in env_lo.agents}
            )
            obs_lo_flat = jnp.stack([obs_lo[a] for a in env_lo.agents], axis=0).reshape(1, 16)
            done_lo = jnp.array([done_lo_dict[env_lo.agents[0]]])
            if bool(np.asarray(done_lo)[0]) and t + 1 < args.max_steps:
                crashed_lo = True
                print(f"  [LoFi]  crashed at t={sim_time:.1f}s (step {t+1})")

        # --- HiFi step ---
        if not crashed_hi:
            action_hi, h_hi = step_hi(obs_hi_flat, done_hi, h_hi)
            rng, _r = jax.random.split(rng)
            obs_hi, state_hi, _, done_hi_dict, _ = env_hi.step(
                _r, state_hi, {a: action_hi for a in env_hi.agents}
            )
            obs_hi_flat = jnp.stack([obs_hi[a] for a in env_hi.agents], axis=0).reshape(1, 16)
            done_hi = jnp.array([done_hi_dict[env_hi.agents[0]]])
            if bool(np.asarray(done_hi)[0]) and t + 1 < args.max_steps:
                crashed_hi = True
                print(f"  [HiFi]  crashed at t={sim_time:.1f}s (step {t+1})")

        # --- Write ACMI frame ---
        with open(acmi_path, "a", encoding="utf-8") as f:
            f.write(f"#{sim_time:.2f}\n")

            # LoFi aircraft
            ps_lo = state_lo.env_state.plane_state
            n_lo  = _f(ps_lo.north);    e_lo  = _f(ps_lo.east)
            a_lo  = _f(ps_lo.altitude); ro_lo = _f(ps_lo.roll)  * 180/np.pi
            pi_lo = _f(ps_lo.pitch)   * 180/np.pi
            ya_lo = _f(ps_lo.yaw)     * 180/np.pi
            lat, lon, alt = enu_to_geodetic(e_lo, n_lo, a_lo, 0, 0, 0)
            f.write(f"100,T={float(lon)}|{float(lat)}|{float(alt)}|{ro_lo:.2f}|{pi_lo:.2f}|{ya_lo:.2f}\n")

            # LoFi target indicator (2 km ahead in target heading direction)
            tgt_hdg_lo = _f(state_lo.env_state.target_heading)
            tn, te, ta = target_indicator_pos(n_lo, e_lo, a_lo, tgt_hdg_lo)
            tlat, tlon, talt = enu_to_geodetic(te, tn, ta, 0, 0, 0)
            f.write(f"1100,T={float(tlon)}|{float(tlat)}|{float(talt)}|0|0|{float(np.degrees(tgt_hdg_lo)):.2f}\n")

            # HiFi aircraft (offset east by 5 km to avoid overlap in Tacview)
            ps_hi = state_hi.env_state.plane_state
            n_hi  = _f(ps_hi.north);    e_hi  = _f(ps_hi.east) + 5000.0
            a_hi  = _f(ps_hi.altitude); ro_hi = _f(ps_hi.roll)  * 180/np.pi
            pi_hi = _f(ps_hi.pitch)   * 180/np.pi
            ya_hi = _f(ps_hi.yaw)     * 180/np.pi
            lat, lon, alt = enu_to_geodetic(e_hi, n_hi, a_hi, 0, 0, 0)
            f.write(f"200,T={float(lon)}|{float(lat)}|{float(alt)}|{ro_hi:.2f}|{pi_hi:.2f}|{ya_hi:.2f}\n")

            # HiFi target indicator
            tgt_hdg_hi = _f(state_hi.env_state.target_heading)
            tn, te, ta = target_indicator_pos(n_hi, e_hi, a_hi, tgt_hdg_hi)
            tlat, tlon, talt = enu_to_geodetic(te, tn, ta, 0, 0, 0)
            f.write(f"1200,T={float(tlon)}|{float(tlat)}|{float(talt)}|0|0|{float(np.degrees(tgt_hdg_hi)):.2f}\n")

        if (t + 1) % 200 == 0:
            print(f"  step {t+1:>5}/{args.max_steps}  t={sim_time:.0f}s  "
                  f"lofi={'crashed' if crashed_lo else 'alive'}  "
                  f"hifi={'crashed' if crashed_hi else 'alive'}")

        if crashed_lo and crashed_hi:
            print("Both aircraft crashed, stopping early.")
            break

    print(f"\nDone. ACMI saved to: {acmi_path}")


# =============================================================================
# Entry
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lofi_ckpt",     type=str, required=True)
    p.add_argument("--hifi_ckpt",     type=str, required=True)
    p.add_argument("--max_steps",     type=int, default=2000,
                   help="Max simulation steps (2000 = 400s)")
    p.add_argument("--s_switch_steps",type=int, default=50,
                   help="Heading switch period in agent steps (50=10s)")
    p.add_argument("--s_heading_tol", type=float, default=0.0,
                   help="Heading tolerance gate in degrees (0=time-only)")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--outdir",        type=str, default="tracks/ablation_comparison")
    p.add_argument("--gpu",           type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gpu >= 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    render(args)
