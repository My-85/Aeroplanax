"""
render_hold.py — Load a trained Hold policy and render a 20-second evaluation.

Outputs:
  1. Matplotlib PNG:  results/hold_render/hold_<datetime>.png
  2. Tacview ACMI:    results/hold_render/hold_<datetime>.acmi

Usage:
    conda activate aeroplanax
    cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
    python render_hold.py --ckpt results/hold_<datetime>/checkpoints/checkpoint_epoch_<N>

If --ckpt is omitted the script runs with a random (untrained) policy so you
can verify the pipeline end-to-end even before training is complete.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import orbax.checkpoint as ocp

from envs.aeroplanax_hold import AeroPlanaxHoldEnv, HoldTaskParams, TARGET_ALT_M, TARGET_VT_MS
from envs.utils.utils import enu_to_geodetic
# reuse network definition from train_hold
from train_hold import ActorCriticRNN, ScannedRNN

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
EVAL_STEPS   = 1000          # 1000 RL steps × dt_RL(0.2s) = 200 s physical
                              # but 1000 RL steps is ≤ max_steps(500) so we
                              # run up to episode end; let's just cap at 1000
RENDER_DIR   = "results/hold_render"

NET_CONFIG = {
    "FC_DIM_SIZE":    128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION":     "relu",
}


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def decode_discrete(act_vec):
    """Convert discrete index vector (4,) → normalised floats."""
    thr  = float(act_vec[0]) / 30.0
    el   = float(act_vec[1]) * 2.0 / 40.0 - 1.0
    ail  = float(act_vec[2]) * 2.0 / 40.0 - 1.0
    rud  = float(act_vec[3]) * 2.0 / 40.0 - 1.0
    return thr, el, ail, rud


def _scalar(x):
    """Extract a Python float from any 0-d or 1-d JAX/numpy array."""
    a = np.asarray(x).ravel()
    return float(a[0]) if a.size else 0.0


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to orbax checkpoint directory (optional)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=EVAL_STEPS)
    args = parser.parse_args()

    Path(RENDER_DIR).mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # ── build env (single environment, no wrapper) ──
    env_params = HoldTaskParams()
    env = AeroPlanaxHoldEnv(env_params)

    # ── build network ──
    network = ActorCriticRNN([31, 41, 41, 41], config=NET_CONFIG)
    rng = jax.random.PRNGKey(args.seed)

    # dummy init
    obs_shape = env.observation_space(env.agents[0], env_params).shape
    init_x = (
        jnp.zeros((1, 1, *obs_shape)),
        jnp.zeros((1, 1)),
    )
    h0 = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    net_params = network.init(rng, h0, init_x)

    # ── load checkpoint ──
    if args.ckpt:
        ckpt_path = os.path.abspath(args.ckpt)
        print(f"Loading checkpoint: {ckpt_path}")
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        ckpt  = ckptr.restore(ckpt_path, args=ocp.args.StandardRestore())
        net_params = ckpt["params"]
        print(f"Restored epoch {int(ckpt['epoch'])}")
    else:
        print("No checkpoint given — running with random (untrained) network.")

    # ── reset ──
    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, env_params)

    # Convert obs to batch-of-1 for the RNN
    # obs_dict["ally_0"] has shape (obs_size,)
    hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    done_flag = jnp.zeros((1,))

    # ── record buffers ──
    rec_t       = []
    rec_alt     = []
    rec_vt      = []
    rec_roll    = []
    rec_pitch   = []
    rec_yaw     = []
    rec_alpha   = []
    rec_beta    = []
    rec_P       = []
    rec_Q       = []
    rec_R       = []
    rec_thr     = []
    rec_el      = []
    rec_ail     = []
    rec_rud     = []
    # rl step counter
    rec_reward  = []

    # Tacview file
    acmi_path = os.path.join(RENDER_DIR, f"hold_{tag}.acmi")
    with open(acmi_path, "w", encoding="utf-8") as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")
        # target marker — static, written once at t=0
        # place it at target altitude directly above origin
        tgt_lat, tgt_lon, tgt_alt_m = enu_to_geodetic(0.0, 0.0, TARGET_ALT_M, 0, 0, 0)
        f.write(
            f"1000,T={tgt_lon}|{tgt_lat}|{tgt_alt_m}|0|0|0,"
            f"Type=Navaid+Static+Waypoint,Name=Target,Color=Yellow,"
            f"Label=H={TARGET_ALT_M:.0f}m_V={TARGET_VT_MS:.0f}ms\n"
        )

    dt_rl = env_params.agent_interaction_steps / env_params.sim_freq   # 0.2 s

    for step in range(args.steps):
        # Time
        t_phys = step * dt_rl
        ps     = state.plane_state

        # Scalars (squeeze out the agent dim)
        alt   = _scalar(ps.altitude)
        vt    = _scalar(ps.vt)
        roll  = _scalar(ps.roll)
        pitch = _scalar(ps.pitch)
        yaw   = _scalar(ps.yaw)
        alpha = _scalar(ps.alpha)
        beta  = _scalar(ps.beta)
        P     = _scalar(ps.P)
        Q     = _scalar(ps.Q)
        R     = _scalar(ps.R)
        north = _scalar(ps.north)
        east  = _scalar(ps.east)

        rec_t.append(t_phys)
        rec_alt.append(alt)
        rec_vt.append(vt)
        rec_roll.append(np.degrees(roll))
        rec_pitch.append(np.degrees(pitch))
        rec_yaw.append(np.degrees(yaw))
        rec_alpha.append(np.degrees(alpha))
        rec_beta.append(np.degrees(beta))
        rec_P.append(P); rec_Q.append(Q); rec_R.append(R)

        # ── policy forward ──
        obs_arr = obs_dict[env.agents[0]]          # (obs_size,)
        obs_in  = obs_arr[None, None, :]            # (1, 1, obs_size)
        done_in = done_flag[None, :]                # (1, 1)

        hstate, pi, _ = network.apply(net_params, hstate, (obs_in, done_in))
        pi_thr, pi_el, pi_ail, pi_rud = pi

        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        act_t = int(pi_thr.mode()[0, 0])
        act_e = int(pi_el.mode()[0, 0])
        act_a = int(pi_ail.mode()[0, 0])
        act_r = int(pi_rud.mode()[0, 0])

        thr_n, el_n, ail_n, rud_n = decode_discrete([act_t, act_e, act_a, act_r])
        rec_thr.append(thr_n)
        rec_el.append(el_n * 45.0)    # deg
        rec_ail.append(ail_n * 45.0)  # deg
        rec_rud.append(rud_n * 45.0)  # deg

        action_dict = {env.agents[0]: jnp.array([act_t, act_e, act_a, act_r])}

        # ── step env ──
        rng, step_key = jax.random.split(rng)
        obs_dict, state, rew_dict, done_dict, info = env.step(
            step_key, state, action_dict, env_params)

        done_flag = jnp.array([float(done_dict[env.agents[0]])])
        rec_reward.append(float(rew_dict[env.agents[0]]))

        # ── write Tacview frame ──
        roll_d  = np.degrees(roll)
        pitch_d = np.degrees(pitch)
        yaw_d   = np.degrees(yaw)
        lat, lon, alt_m = enu_to_geodetic(east, north, alt, 0, 0, 0)

        with open(acmi_path, "a", encoding="utf-8") as f:
            f.write(f"#{t_phys:.2f}\n")
            f.write(
                f"100,T={lon}|{lat}|{alt_m}|{roll_d:.2f}|{pitch_d:.2f}|{yaw_d:.2f},"
                f"Type=Air+FixedWing,Name=F16,Color=Cyan\n"
            )
            # Update target indicator (stays at origin horizontally but at target alt)
            f.write(
                f"1000,T={tgt_lon}|{tgt_lat}|{tgt_alt_m}|0|0|0\n"
            )

        # Episode ended?
        if bool(done_dict["__all__"]):
            print(f"Episode ended at step {step} (t={t_phys:.1f} s)")
            break

    print(f"Tacview ACMI: {acmi_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Matplotlib figure
    # ─────────────────────────────────────────────────────────────────────
    t = np.array(rec_t)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Planax Hold Task — {'Policy' if args.ckpt else 'Random'}", fontsize=14)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Altitude ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, rec_alt, color="royalblue", lw=1.5, label="altitude")
    ax.axhline(TARGET_ALT_M, color="tomato", ls="--", lw=1.2, label=f"target {TARGET_ALT_M:.0f} m")
    ax.axhline(env_params.min_altitude, color="gray", ls=":", lw=0.8)
    ax.axhline(env_params.max_altitude, color="gray", ls=":", lw=0.8)
    ax.set_ylabel("Altitude (m)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.set_title("Altitude tracking")

    # ── Airspeed ──
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, rec_vt, color="darkorange", lw=1.5, label="Vt")
    ax.axhline(TARGET_VT_MS, color="tomato", ls="--", lw=1.2, label=f"target {TARGET_VT_MS:.0f} m/s")
    ax.set_ylabel("Airspeed (m/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.set_title("Airspeed tracking")

    # ── Roll / Pitch / Yaw ──
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, rec_roll,  lw=1.2, label="Roll (°)")
    ax.plot(t, rec_pitch, lw=1.2, label="Pitch (°)")
    ax.plot(t, rec_yaw,   lw=1.0, label="Yaw (°)", alpha=0.6)
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("Angle (°)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.set_title("Attitude (Roll / Pitch / Yaw)")

    # ── Alpha / Beta ──
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, rec_alpha, lw=1.2, label="α (°)")
    ax.plot(t, rec_beta,  lw=1.2, label="β (°)")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("Flow angle (°)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.set_title("Aerodynamic angles")

    # ── Body rates ──
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, rec_P, lw=1.2, label="P (rad/s)")
    ax.plot(t, rec_Q, lw=1.2, label="Q (rad/s)")
    ax.plot(t, rec_R, lw=1.2, label="R (rad/s)")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("Rate (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.set_title("Body angular rates")

    # ── Controls ──
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, rec_thr,         lw=1.2, label="Throttle (0→1)")
    ax.set_ylabel("Throttle")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.set_title("Throttle command")

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(t, rec_el,  lw=1.2, label="Elevator (°)")
    ax2.plot(t, rec_ail, lw=1.2, label="Aileron (°)")
    ax2.plot(t, rec_rud, lw=1.2, label="Rudder (°)")
    ax2.axhline(0, color="black", ls="--", lw=0.8)
    ax2.set_ylabel("Deflection (°)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=8)
    ax2.set_title("Surface deflections")

    # ── Reward ──
    ax3 = fig.add_subplot(gs[3, 1])
    ax3.plot(t[:len(rec_reward)], rec_reward, color="green", lw=1.2, label="step reward")
    ax3.set_ylabel("Reward")
    ax3.set_xlabel("Time (s)")
    ax3.legend(fontsize=8)
    ax3.set_title("Step reward")

    png_path = os.path.join(RENDER_DIR, f"hold_{tag}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {png_path}")

    # brief stats
    print(f"\n=== Evaluation summary ({len(t)} steps = {t[-1]:.1f} s) ===")
    print(f"  Alt : mean={np.mean(rec_alt):.1f}  std={np.std(rec_alt):.1f}  target={TARGET_ALT_M}")
    print(f"  Vt  : mean={np.mean(rec_vt):.1f}  std={np.std(rec_vt):.1f}  target={TARGET_VT_MS}")
    print(f"  Roll: mean={np.mean(np.abs(rec_roll)):.2f}°  max={np.max(np.abs(rec_roll)):.2f}°")
    print(f"  Beta: mean={np.mean(np.abs(rec_beta)):.2f}°  max={np.max(np.abs(rec_beta)):.2f}°")
    print(f"  Reward: mean={np.mean(rec_reward):.3f}  sum={np.sum(rec_reward):.2f}")


if __name__ == "__main__":
    main()
