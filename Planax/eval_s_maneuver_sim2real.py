"""
S-Maneuver Zero-Shot Sim-to-Real Evaluation
============================================
Loads BOTH the lofi-trained and hifi-trained policies, evaluates them in a
FORCED HIGH-FIDELITY environment, and outputs:
  - Survival rate & mean episode length
  - Physical constraint violation rates (stall α > 20°, G-overload |nz| > 9)
  - Heading tracking RMSE during S-maneuver

Results are saved to:
  eval_output/ablation_results.json
  eval_output/ablation_results.csv

Usage:
  python eval_s_maneuver_sim2real.py \
    --lofi_ckpt results/ablation_lofi_<timestamp>/checkpoints/checkpoint_epoch_<N> \
    --hifi_ckpt results/ablation_hifi_<timestamp>/checkpoints/checkpoint_epoch_<N> \
    [--n_envs 200] [--episode_steps 2000] [--s_switch_steps 50]
"""

import os
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import orbax.checkpoint as ocp
import functools
import distrax

from typing import Sequence, Dict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from envs.aeroplanax_s_maneuver_ablation import (
    AeroPlanaxSManeuverAblationEnv,
    SManeuverTaskParams,
)
from envs.wrappers import LogWrapper


# =============================================================================
# Network (must match training architecture exactly)
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
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)
        pi_t = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_e = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_a = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_r = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, (pi_t, pi_e, pi_a, pi_r), jnp.squeeze(critic, axis=-1)


NET_CONFIG = {
    "FC_DIM_SIZE":    128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION":     "relu",
}


# =============================================================================
# Checkpoint loader
# =============================================================================

def load_policy_params(ckpt_dir: str):
    """Restore policy params from an orbax checkpoint directory."""
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


# =============================================================================
# Single-step inference (JIT-compiled)
# =============================================================================

def make_step_fn(network: ActorCriticRNN, policy_params):
    """Returns a JIT-compiled (obs, done, hstate) → (action, new_hstate) function."""

    @jax.jit
    def step_fn(obs, done, hstate):
        ac_in  = (obs[None, :], done[None, :])          # (1, B, 16), (1, B)
        new_h, pi, _ = network.apply(policy_params, hstate, ac_in)
        pi_t, pi_e, pi_a, pi_r = pi
        # Greedy (mode) action for evaluation
        a_t = jnp.argmax(pi_t.logits, axis=-1).squeeze(0)
        a_e = jnp.argmax(pi_e.logits, axis=-1).squeeze(0)
        a_a = jnp.argmax(pi_a.logits, axis=-1).squeeze(0)
        a_r = jnp.argmax(pi_r.logits, axis=-1).squeeze(0)
        action = jnp.stack([a_t, a_e, a_a, a_r], axis=-1)   # (B, 4)
        return action, new_h

    return step_fn


# =============================================================================
# Evaluation loop
# =============================================================================

def wrap_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def evaluate_policy(
    policy_label:  str,
    policy_params,
    env:           AeroPlanaxSManeuverAblationEnv,  # MUST be hifi
    env_params:    SManeuverTaskParams,
    n_envs:        int,
    episode_steps: int,
    seed:          int = 0,
) -> dict:
    """
    Run `n_envs` parallel episodes for `episode_steps` steps.
    Returns a dict of scalar metric statistics.

    Metrics collected per step per env:
      - survived:    env still alive (not done)
      - heading_err: |wrap(yaw - target_heading)| in degrees
      - stall_event: |alpha| > stall_threshold (20°)
      - g_event:     |nz| > g_threshold (9 G)
      - ep_length:   steps until done (filled at episode end)
    """
    network   = ActorCriticRNN([31, 41, 41, 41], config=NET_CONFIG)
    step_fn   = make_step_fn(network, policy_params)
    agents    = env.agents

    # env uses LogWrapper — reset returns obs dict
    rng       = jax.random.PRNGKey(seed)
    reset_rng = jax.random.split(rng, n_envs)

    # Vectorised reset
    obs_v, state_v = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

    # Flatten obs: (n_envs, obs_dim)
    obs_flat = jnp.stack([obs_v[a] for a in agents], axis=1).reshape(n_envs, 16)
    done_flat = jnp.zeros((n_envs,), dtype=bool)
    hstate    = ScannedRNN.initialize_carry(n_envs, NET_CONFIG["GRU_HIDDEN_DIM"])

    # Accumulators (numpy for efficiency)
    heading_sq_errs    = np.zeros(n_envs)  # sum of squared heading errors
    heading_sq_count   = np.zeros(n_envs)  # number of alive steps counted
    stall_events       = np.zeros(n_envs)  # count of stall violations
    g_events           = np.zeros(n_envs)  # count of G violations
    ep_lengths         = np.full(n_envs, episode_steps, dtype=float)  # default = max
    ep_done            = np.zeros(n_envs, dtype=bool)
    ep_crashed         = np.zeros(n_envs, dtype=bool)  # True = crashed (not timeout)

    STALL_ALPHA_DEG = 20.0
    G_LIMIT         = 9.0

    print(f"  [{policy_label}] Evaluating {n_envs} envs × {episode_steps} steps ...")

    for t in range(episode_steps):
        action, hstate = step_fn(obs_flat, done_flat, hstate)

        # Build action dict for vmap env.step
        action_dict = {a: action[:, i:i+1].repeat(1, axis=1)
                       for i, a in enumerate(agents)}
        # action shape per agent: (n_envs, 4) split out per head via unbatchify convention
        # Actually the env.step expects Dict[agent -> (n_envs, 4)]
        action_dict = {a: action for a, _ in zip(agents, range(len(agents)))}

        rng, _rng = jax.random.split(rng)
        rng_step  = jax.random.split(_rng, n_envs)
        obs_v, state_v, reward_v, done_v, info_v = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, state_v, {a: action for a in agents})

        obs_flat  = jnp.stack([obs_v[a] for a in agents], axis=1).reshape(n_envs, 16)
        done_flat = jnp.stack([done_v[a] for a in agents], axis=1).any(axis=1)

        # Collect metrics from plane_state
        ps       = state_v.env_state.plane_state   # LogWrapper wraps env_state
        yaw_np   = np.asarray(ps.yaw[:, 0])         # (n_envs,)
        tgt_h_np = np.asarray(state_v.env_state.target_heading[:, 0])
        alpha_np = np.asarray(ps.alpha[:, 0]) * 180.0 / np.pi   # degrees
        nz_np    = np.asarray(ps.az[:, 0])                       # G

        alive_np  = ~np.asarray(done_flat)
        not_done  = ~ep_done

        # Heading RMSE accumulation (only for alive episodes)
        hdg_err_deg = np.abs(np.degrees(
            np.arctan2(np.sin(yaw_np - tgt_h_np), np.cos(yaw_np - tgt_h_np))
        ))
        mask = not_done & alive_np
        heading_sq_errs  += np.where(mask, hdg_err_deg ** 2,  0.0)
        heading_sq_count += np.where(mask, 1.0,               0.0)

        # Constraint violations (only for alive, not-done episodes)
        stall_now  = np.abs(alpha_np) > STALL_ALPHA_DEG
        g_now      = np.abs(nz_np)    > G_LIMIT
        stall_events += np.where(not_done, stall_now.astype(float), 0.0)
        g_events     += np.where(not_done, g_now.astype(float),     0.0)

        # Record episode length when first done
        # An episode ending before the last step is a crash (timeout fires at step episode_steps)
        newly_done = np.asarray(done_flat) & not_done
        is_crash   = newly_done & (t + 1 < episode_steps)
        ep_lengths  = np.where(newly_done, float(t + 1), ep_lengths)
        ep_done    |= newly_done
        ep_crashed |= is_crash

        if (t + 1) % 200 == 0:
            n_not_done   = int(np.sum(~ep_done))
            n_crashed    = int(np.sum(ep_crashed))
            n_timeout    = int(np.sum(ep_done & ~ep_crashed))
            n_not_crashed = n_envs - n_crashed   # still running + already timed out
            mean_hdg  = np.sqrt(np.where(heading_sq_count > 0,
                                         heading_sq_errs / np.maximum(heading_sq_count, 1), 0.0)).mean()
            print(f"    step {t+1:>5}/{episode_steps}  "
                  f"still_running={n_not_done}  crashed={n_crashed}  timeout={n_timeout}  "
                  f"not_crashed={n_not_crashed}({n_not_crashed/n_envs:.1%})  "
                  f"mean_hdg_RMSE={mean_hdg:.2f}°")

    # Final metrics
    n_crashed          = int(np.sum(ep_crashed))
    n_timeout          = int(np.sum(~ep_crashed))   # survived to episode end
    crash_rate         = float(n_crashed / n_envs)
    timeout_rate       = float(n_timeout / n_envs)
    survival_rate      = timeout_rate               # "survived" = reached timeout without crashing
    mean_ep_length     = float(np.mean(ep_lengths))
    per_env_hdg_rmse   = np.sqrt(np.where(heading_sq_count > 0,
                                           heading_sq_errs / np.maximum(heading_sq_count, 1),
                                           0.0))
    mean_hdg_rmse      = float(per_env_hdg_rmse.mean())
    mean_stall_rate    = float((stall_events / np.maximum(ep_lengths, 1)).mean())
    mean_g_rate        = float((g_events     / np.maximum(ep_lengths, 1)).mean())
    total_stall_events = int(stall_events.sum())
    total_g_events     = int(g_events.sum())

    results = {
        "policy":              policy_label,
        "eval_env":            "high_fidelity",
        "n_envs":              n_envs,
        "episode_steps":       episode_steps,
        "survival_rate":       survival_rate,
        "crash_rate":          crash_rate,
        "n_crashed":           n_crashed,
        "n_timeout":           n_timeout,
        "mean_ep_length_steps":mean_ep_length,
        "mean_ep_length_sec":  mean_ep_length * 10 / 50,
        "heading_rmse_deg":    mean_hdg_rmse,
        "stall_violation_rate_per_step": mean_stall_rate,
        "g_violation_rate_per_step":     mean_g_rate,
        "total_stall_events":  total_stall_events,
        "total_g_events":      total_g_events,
    }

    print(f"\n  [{policy_label}] Results:")
    print(f"    Crashed          : {n_crashed}/{n_envs} ({crash_rate:.2%})")
    print(f"    Survived (timeout): {n_timeout}/{n_envs} ({timeout_rate:.2%})")
    print(f"    Mean ep length   : {mean_ep_length:.1f} steps  ({mean_ep_length*10/50:.1f} s)")
    print(f"    Heading RMSE     : {mean_hdg_rmse:.2f}°")
    print(f"    Stall rate/step  : {mean_stall_rate:.4f}")
    print(f"    G-overload/step  : {mean_g_rate:.4f}")
    return results


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="S-Maneuver Zero-Shot Sim2Real Evaluation")
    p.add_argument("--lofi_ckpt",     type=str, required=True,
                   help="Path to lofi-trained checkpoint directory")
    p.add_argument("--hifi_ckpt",     type=str, required=True,
                   help="Path to hifi-trained checkpoint directory")
    p.add_argument("--n_envs",        type=int, default=200,
                   help="Number of parallel evaluation environments")
    p.add_argument("--episode_steps", type=int, default=2000,
                   help="Max steps per episode (200s at 0.2s/step)")
    p.add_argument("--s_switch_steps",type=int,   default=50,
                   help="Minimum steps before heading switch (time gate)")
    p.add_argument("--s_heading_tol", type=float, default=0.0,
                   help="Also require heading error < this (deg) before switching. "
                        "0 = time-only (original). e.g. 15 → wait until within 15°.")
    p.add_argument("--outdir",        type=str, default="eval_output",
                   help="Output directory for result files")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--gpu",           type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.gpu >= 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("XLA_PYTHON_MEM_FRACTION", "0.90")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # ---- Build HIGH-FIDELITY environment (locked, never lofi) ----
    s_heading_tol_rad = float(np.deg2rad(args.s_heading_tol)) if args.s_heading_tol > 0 else 0.0
    env_params = SManeuverTaskParams(
        fidelity_mode  = "high",
        s_switch_steps = args.s_switch_steps,
        s_heading_tol  = s_heading_tol_rad,
    )
    env = AeroPlanaxSManeuverAblationEnv(env_params)
    env = LogWrapper(env)
    tol_str = f"  heading_tol={args.s_heading_tol:.1f}°" if args.s_heading_tol > 0 else "  (time-only switching)"
    print(f"Evaluation environment: fidelity_mode=high  s_switch_steps={args.s_switch_steps} "
          f"(= {args.s_switch_steps*10/50:.1f}s){tol_str}")

    # ---- Load both policies ----
    print(f"\nLoading lofi policy from: {args.lofi_ckpt}")
    lofi_params = load_policy_params(args.lofi_ckpt)

    print(f"Loading hifi policy from: {args.hifi_ckpt}")
    hifi_params = load_policy_params(args.hifi_ckpt)

    # ---- Evaluate ----
    all_results = []
    for label, params in [("lofi_policy_hifi_env", lofi_params),
                           ("hifi_policy_hifi_env", hifi_params)]:
        print(f"\n{'='*60}")
        res = evaluate_policy(
            policy_label  = label,
            policy_params = params,
            env           = env,
            env_params    = env_params,
            n_envs        = args.n_envs,
            episode_steps = args.episode_steps,
            seed          = args.seed,
        )
        all_results.append(res)

    # ---- Print comparison ----
    print(f"\n{'='*60}")
    print("SUMMARY (both policies evaluated in HIGH-FIDELITY environment)")
    print(f"{'='*60}")
    header = ["Metric", "LoFi-trained policy", "HiFi-trained policy"]
    rows   = []
    keys   = [
        ("survival_rate",                "Survival rate (timeout)"),
        ("crash_rate",                   "Crash rate"),
        ("n_crashed",                    "# Crashed"),
        ("n_timeout",                    "# Survived to timeout"),
        ("mean_ep_length_sec",           "Mean episode length (s)"),
        ("heading_rmse_deg",             "Heading RMSE (deg)"),
        ("stall_violation_rate_per_step","Stall violation rate/step"),
        ("g_violation_rate_per_step",    "G-overload rate/step"),
        ("total_stall_events",           "Total stall events"),
        ("total_g_events",               "Total G-overload events"),
    ]
    for k, display in keys:
        v_lofi = all_results[0][k]
        v_hifi = all_results[1][k]
        if isinstance(v_lofi, float):
            row = [display, f"{v_lofi:.4f}", f"{v_hifi:.4f}"]
        else:
            row = [display, str(v_lofi), str(v_hifi)]
        rows.append(row)
        print(f"  {display:<40} lofi={row[1]:>10}   hifi={row[2]:>10}")

    # ---- Save JSON ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.outdir, f"ablation_results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp":    ts,
            "eval_env":     "high_fidelity",
            "s_switch_steps": args.s_switch_steps,
            "n_envs":       args.n_envs,
            "episode_steps":args.episode_steps,
            "lofi_ckpt":    args.lofi_ckpt,
            "hifi_ckpt":    args.hifi_ckpt,
            "results":      all_results,
        }, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # ---- Save CSV ----
    csv_path = os.path.join(args.outdir, f"ablation_results_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"CSV  saved: {csv_path}")

    # Also write a "latest" symlink for convenience
    for path, ext in [(json_path, ".json"), (csv_path, ".csv")]:
        latest = os.path.join(args.outdir, f"ablation_results_latest{ext}")
        try:
            if os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.abspath(path), latest)
        except OSError:
            pass  # symlink creation can fail in some environments

    print("\nDone.")
