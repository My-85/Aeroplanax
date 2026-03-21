"""
render_s_maneuver_3d.py
=======================
Load heading_pitch_V baseline checkpoint, fly 3D S-maneuver via waypoint tracking,
output .acmi file with all waypoints batch-written for Tacview visualization.

Waypoints include altitude variation:
  wp_alt = origin_alt + 1000 * sin(pi * idx / 60)

Usage:
  cd Planax && python render_s_maneuver_3d.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from flax.linen.initializers import constant, orthogonal
import functools
from typing import Sequence, Dict
from flax.training.train_state import TrainState
import distrax
import optax
import orbax.checkpoint as ocp

from envs.wrappers import LogWrapper
from envs.aeroplanax_s_maneuver_3d import (
    AeroPlanaxSManeuver3DEnv,
    S_Maneuver_3D_TaskParams,
)
from envs.utils.utils import enu_to_geodetic, wrap_PI


# ============================================================
# Network (identical to train_heading_pitch_V_discrete_rnn_new_critic_no_fc2.py)
# ============================================================

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
        activation = nn.relu if self.config.get("ACTIVATION", "relu") == "relu" else nn.tanh
        obs, dones = x

        embedding = nn.Dense(self.config["FC_DIM_SIZE"],
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(obs)
        embedding = activation(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = nn.LayerNorm()(nn_fc2)
        nn_fc2 = activation(nn_fc2)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"],
                              kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        actor_mean = activation(actor_mean)

        pi_thr = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_ele = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_ail = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        pi_rud = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))

        # critic from nn_fc2 (matching no_fc2 training script)
        critic = nn.Dense(self.config["FC_DIM_SIZE"],
                          kernel_init=orthogonal(2), bias_init=constant(0.0))(nn_fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, (pi_thr, pi_ele, pi_ail, pi_rud), jnp.squeeze(critic, axis=-1)


# ============================================================
# Helpers
# ============================================================

def _f(x, i=0):
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    return float(a.reshape(-1)[min(i, a.size - 1)])


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def compute_s_waypoints_3d(origin_n, origin_e, origin_alt, params):
    """Pre-compute all 3D S-curve waypoints (numpy, for ACMI batch write)."""
    dn = params.s_half_period_north / params.s_points_per_half
    total_points = params.max_waypoints
    waypoints = []
    for i in range(1, total_points + 1):
        wp_n = origin_n + i * dn
        wp_e = origin_e + params.s_amplitude * np.sin(np.pi * i / params.s_points_per_half)
        wp_a = origin_alt + params.s_altitude_amplitude * np.sin(np.pi * i / params.s_alt_points_per_half)
        waypoints.append((wp_n, wp_e, wp_a))
    return waypoints


# ============================================================
# Main render
# ============================================================

def render_episode(config):
    # ---------- 1) Environment ----------
    env_params = S_Maneuver_3D_TaskParams()
    env_core = AeroPlanaxSManeuver3DEnv(env_params)
    env = LogWrapper(env_core)
    config["NUM_ACTORS"] = env.num_agents
    assert config["NUM_ENVS"] == 1

    # ---------- 2) Network ----------
    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    rng = jax.random.PRNGKey(config['SEED'])
    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"],
                    *env.observation_space(env.agents[0], env_params).shape)),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    init_h = ScannedRNN.initialize_carry(
        config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
    )
    params = network.init(rng, init_h, init_x)

    tx = optax.adam(config["LR"], eps=1e-5)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # ---------- 3) Load checkpoint ----------
    if config.get("LOADDIR"):
        state_template = {
            "params": train_state.params,
            "opt_state": train_state.opt_state,
            "epoch": jnp.array(0),
        }
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(
            config['LOADDIR'], args=ocp.args.StandardRestore(item=state_template)
        )
        print(f"[restore] epoch = {int(checkpoint['epoch'])}")
        params = checkpoint["params"]

    # ---------- 4) Reset ----------
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    core_state = env_state.env_state

    # ---------- 5) Pre-compute 3D S-curve waypoints & write ACMI ----------
    origin_n = _f(core_state.s_origin_n)
    origin_e = _f(core_state.s_origin_e)
    origin_alt = _f(core_state.s_origin_alt)
    s_waypoints = compute_s_waypoints_3d(origin_n, origin_e, origin_alt, env_params)

    logdir = config.get("LOGDIR", "./tracks/s_maneuver_3d/")
    os.makedirs(logdir, exist_ok=True)
    acmi_filename = logdir + datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + '.txt.acmi'

    # Write ACMI header
    with open(acmi_filename, mode='w', encoding='utf-8') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")

    # Batch-write all waypoints
    WP_ID_BASE = 5000
    with open(acmi_filename, mode='a', encoding='utf-8') as f:
        for k, (wp_n, wp_e, wp_a) in enumerate(s_waypoints):
            lat, lon, alt = enu_to_geodetic(wp_e, wp_n, wp_a, 0, 0, 0)
            oid = WP_ID_BASE + k
            f.write(
                f"{oid},"
                f"Type=Navaid+Static+Waypoint,Name=WP_{k},Label={k},Color=Yellow,"
                f"T={float(lon)}|{float(lat)}|{float(alt)}|0|0|0\n"
            )
    print(f"Wrote {len(s_waypoints)} waypoints to ACMI")
    print(f"ACMI file: {acmi_filename}")

    # ---------- 6) RNN state ----------
    hstate = ScannedRNN.initialize_carry(
        config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
    )

    # ---------- 7) Simulation loop ----------
    done_arr = jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool)
    obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])

    max_steps = config.get("MAX_STEPS", 5000)
    cumulative_reward = 0.0
    reset_count = 0

    print(f"\n{'='*110}")
    print(f"3D S-maneuver render: amplitude={env_params.s_amplitude}m, "
          f"alt_amplitude={env_params.s_altitude_amplitude}m, "
          f"half_period={env_params.s_half_period_north}m, "
          f"points_per_half={env_params.s_points_per_half}, "
          f"reach_radius={env_params.reach_radius}m")
    print(f"Checkpoint: {config.get('LOADDIR', 'N/A')}")
    print(f"{'='*110}")

    print(f"\n{'Step':>6} | {'WP_idx':>6} | {'Reached':>7} | {'Dist3D':>8} | "
          f"{'Heading':>8} | {'TgtHdg':>8} | {'dH':>8} | {'Vt':>6} | {'Alt':>7} | {'TgtAlt':>7} | {'R':>7}")
    print(f"{'-'*110}")

    rng_run = jax.random.PRNGKey(config["SEED"] + 100)

    for step in range(max_steps):
        # --- Forward (greedy) ---
        ac_in = (obs_batch[np.newaxis, :], done_arr[np.newaxis, :])
        hstate, pi, value = network.apply(params, hstate, ac_in)
        pi_thr, pi_ele, pi_ail, pi_rud = pi

        a_thr = pi_thr.mode()
        a_ele = pi_ele.mode()
        a_ail = pi_ail.mode()
        a_rud = pi_rud.mode()

        action = jnp.concatenate(
            [a_thr[:, :, np.newaxis], a_ele[:, :, np.newaxis],
             a_ail[:, :, np.newaxis], a_rud[:, :, np.newaxis]], axis=-1
        )
        action = action.squeeze(0)

        # --- env step ---
        rng_run, _rng = jax.random.split(rng_run)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
            rng_step, env_state,
            unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        )

        core_state = env_state.env_state
        reward_sum = float(batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1).sum())
        cumulative_reward += reward_sum

        # --- Flight state ---
        yaw_now = _f(core_state.plane_state.yaw)
        vt_now = _f(core_state.plane_state.vt)
        alt_now = _f(core_state.plane_state.altitude)
        tgt_hdg = _f(core_state.target_heading)
        wp_idx = int(_f(core_state.current_wp_idx))
        wp_reached = int(_f(core_state.waypoints_reached))

        # 3D distance to current waypoint
        dn_step = env_params.s_half_period_north / env_params.s_points_per_half
        cur_wp_n = origin_n + wp_idx * dn_step
        cur_wp_e = origin_e + env_params.s_amplitude * np.sin(np.pi * wp_idx / env_params.s_points_per_half)
        cur_wp_alt = origin_alt + env_params.s_altitude_amplitude * np.sin(np.pi * wp_idx / env_params.s_alt_points_per_half)
        dist3d_val = float(np.sqrt(
            (_f(core_state.plane_state.north) - cur_wp_n) ** 2 +
            (_f(core_state.plane_state.east) - cur_wp_e) ** 2 +
            (alt_now - cur_wp_alt) ** 2
        ))
        delta_hdg = float(np.degrees(wrap_PI(jnp.array(yaw_now - tgt_hdg))))

        # --- Write ACMI frame ---
        with open(acmi_filename, mode='a', encoding='utf-8') as f:
            timestamp = core_state.time * env_params.agent_interaction_steps / env_params.sim_freq
            ts = float(jnp.ravel(jnp.asarray(timestamp))[0])
            f.write(f"#{ts:.2f}\n")

            npos = _f(core_state.plane_state.north)
            epos = _f(core_state.plane_state.east)
            alt_v = _f(core_state.plane_state.altitude)
            roll_v = _f(core_state.plane_state.roll) * 180 / np.pi
            pitch_v = _f(core_state.plane_state.pitch) * 180 / np.pi
            yaw_v = _f(core_state.plane_state.yaw) * 180 / np.pi
            lat_v, lon_v, alt_v = enu_to_geodetic(epos, npos, alt_v, 0, 0, 0)
            f.write(f"100,T={lon_v}|{lat_v}|{alt_v}|{roll_v}|{pitch_v}|{yaw_v},"
                    f"Type=Air+FixedWing,Name=F16,Color=Red\n")

            # Current target waypoint indicator (green marker)
            if wp_idx - 1 < len(s_waypoints):
                twp_n, twp_e, twp_a = s_waypoints[min(wp_idx - 1, len(s_waypoints) - 1)]
                tlat, tlon, talt = enu_to_geodetic(twp_e, twp_n, twp_a, 0, 0, 0)
                f.write(f"1000,T={float(tlon)}|{float(tlat)}|{float(talt)}|0|0|0,"
                        f"Name=CurrentWP,Color=Green,Type=Navaid+Static+Waypoint\n")

        # --- Log ---
        if step % 50 == 0 or wp_reached > 0 and step % 20 == 0:
            print(
                f"{step:6d} | {wp_idx:6d} | {wp_reached:7d} | {dist3d_val:8.1f} | "
                f"{np.degrees(yaw_now):+8.1f} | {np.degrees(tgt_hdg):+8.1f} | {delta_hdg:+8.1f} | "
                f"{vt_now:6.1f} | {alt_now:7.0f} | {cur_wp_alt:7.0f} | {reward_sum:+7.3f}"
            )

        # --- Check done ---
        obs_batch = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        done_arr = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)

        done_any = bool(np.asarray(done_arr).any())
        if done_any:
            reset_count += 1
            status = int(_f(core_state.plane_state.status))
            reason = "crashed" if status == 2 else "timeout/success"
            print(f"\n[RESET #{reset_count}] step={step}, reason={reason}, "
                  f"wp_reached={wp_reached}, cumR={cumulative_reward:.1f}")

            # Reset RNN hidden
            hstate = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            # Update origin for new S-curve segment
            core_state = env_state.env_state
            origin_n = _f(core_state.s_origin_n)
            origin_e = _f(core_state.s_origin_e)
            origin_alt = _f(core_state.s_origin_alt)

            # Check if all waypoints done
            if wp_reached >= env_params.max_waypoints:
                print(f"\n*** All {env_params.max_waypoints} waypoints reached! ***")
                break

    # ---------- 8) Summary ----------
    print(f"\n{'='*110}")
    print(f"Render complete: reached {wp_reached} waypoints")
    print(f"Cumulative reward = {cumulative_reward:.1f}, resets = {reset_count}")
    print(f"ACMI file: {acmi_filename}")
    print(f"{'='*110}")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    config = {
        "SEED": 42,
        "LR": 3e-4,
        "NUM_ENVS": 1,
        "NUM_ACTORS": 1,
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "ACTIVATION": "relu",
        "MAX_STEPS": 5000,
        "LOGDIR": "./tracks/s_maneuver_3d/",
        "LOADDIR": os.path.abspath(
            "/home/dqy/aeroplanax/new/20251215最新代码库/results/baseline（欧拉角版本）/checkpoints/checkpoint_epoch_600"
        ),
    }
    render_episode(config)
