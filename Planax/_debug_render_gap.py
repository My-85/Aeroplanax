"""Diagnose the training-vs-render gap.

Runs the trained agent in the SAME environment it was trained on
(random targets, 90s interval).  If tracking works here but fails
in the waypoint render, the gap is in the render setup.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import functools
import distrax

from envs.aeroplanax_heading_pitch_V import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
)
from envs.utils.utils import wrap_PI

# ── Network (same as training) ──
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan, variable_broadcast="params",
        in_axes=0, out_axes=0, split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis],
                              self.initialize_carry(*rnn_state.shape), rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: tuple
    config: dict

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
        pi_throttle = distrax.Categorical(logits=nn.Dense(
            self.action_dim[0], kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_elevator = distrax.Categorical(logits=nn.Dense(
            self.action_dim[1], kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_aileron = distrax.Categorical(logits=nn.Dense(
            self.action_dim[2], kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_rudder = distrax.Categorical(logits=nn.Dense(
            self.action_dim[3], kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0))(actor_mean))
        pi_speed_brake = distrax.Categorical(logits=nn.Dense(
            self.action_dim[4], kernel_init=nn.initializers.constant(0.0),
            bias_init=lambda key, shape, dtype=jnp.float32: jnp.array(
                [0.0, -1.5, -1.5, -1.5, -1.5], dtype=dtype))(actor_mean))
        critic = nn.Dense(self.config["FC_DIM_SIZE"],
                          kernel_init=nn.initializers.orthogonal(2),
                          bias_init=nn.initializers.constant(0.0))(fc2)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0),
                          bias_init=nn.initializers.constant(0.0))(critic)
        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder, pi_speed_brake), jnp.squeeze(critic, axis=-1)


NET_CONFIG = {
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "ACTIVATION": "relu",
}

def main():
    # ── Env ──
    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)

    # ── Network ──
    network = ActorCriticRNN((31, 41, 41, 41, 5), config=NET_CONFIG)
    rng = jax.random.PRNGKey(42)

    obs_shape = env.observation_space(env.agents[0], env_params).shape
    print(f"Obs shape: {obs_shape}")
    init_x = (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1)))
    h0 = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    net_params = network.init(rng, h0, init_x)

    # ── Load checkpoint ──
    import pathlib
    ckpt_path = str(pathlib.Path(
        "results/heading_pitch_V_discrete_rnn_2026-05-14-15-29/"
        "checkpoints/checkpoint_epoch_300").resolve())
    import orbax.checkpoint as ocp
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(ckpt_path, args=ocp.args.StandardRestore())
    net_params = ckpt["params"]
    print(f"Loaded epoch {int(ckpt['epoch'])}")

    # ── Reset ──
    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, env_params)
    hstate = ScannedRNN.initialize_carry(1, NET_CONFIG["GRU_HIDDEN_DIM"])
    done_flag = jnp.zeros((1,))

    dt_rl = env_params.agent_interaction_steps / env_params.sim_freq

    print(f"\n{'Step':>6} | {'alt':>7} | {'vt':>6} | {'roll':>7} | {'pitch':>7} | "
          f"{'yaw':>7} | {'tgt_h':>7} | {'h_err':>7} | {'tgt_p':>7} | {'p_err':>7} | "
          f"{'tgt_r':>7} | {'r_err':>7} | {'tgt_v':>7} | {'v_err':>7}")
    print("-" * 130)

    for step in range(600):
        ps = state.plane_state
        t_phys = step * dt_rl

        def _f(x):
            return float(jnp.nan_to_num(x).squeeze())

        alt = _f(ps.altitude)
        vt = _f(ps.vt)
        roll = np.degrees(_f(ps.roll))
        pitch = np.degrees(_f(ps.pitch))
        yaw = np.degrees(_f(ps.yaw))
        tgt_h = np.degrees(_f(state.target_heading))
        tgt_p = np.degrees(_f(state.target_pitch))
        tgt_r = np.degrees(_f(state.target_roll))
        tgt_v = _f(state.target_vt)

        h_err = np.degrees(float(wrap_PI(
            jnp.nan_to_num(ps.yaw).squeeze() - jnp.nan_to_num(state.target_heading).squeeze())))
        p_err = np.degrees(float(wrap_PI(
            jnp.nan_to_num(ps.pitch).squeeze() - jnp.nan_to_num(state.target_pitch).squeeze())))
        r_err = np.degrees(float(wrap_PI(
            jnp.nan_to_num(ps.roll).squeeze() - jnp.nan_to_num(state.target_roll).squeeze())))
        v_err = vt - tgt_v

        # ── Policy forward ──
        obs_vec = obs_dict[env.agents[0]]
        obs_in = obs_vec[None, None, :]
        done_in = done_flag[None, :]
        hstate, pi, value = network.apply(net_params, hstate, (obs_in, done_in))

        pi_thr, pi_el, pi_ail, pi_rud, pi_sb = pi
        act_thr = int(pi_thr.mode()[0, 0])
        act_el = int(pi_el.mode()[0, 0])
        act_ail = int(pi_ail.mode()[0, 0])
        act_rud = int(pi_rud.mode()[0, 0])
        act_sb = int(pi_sb.mode()[0, 0])

        thr_n = act_thr / 30.0
        el_n = act_el * 2.0 / 40.0 - 1.0
        ail_n = act_ail * 2.0 / 40.0 - 1.0
        rud_n = act_rud * 2.0 / 40.0 - 1.0
        sb_n = act_sb / 4.0

        action_dict = {env.agents[0]: jnp.array([act_thr, act_el, act_ail, act_rud, act_sb])}

        # ── Step ──
        rng, step_key = jax.random.split(rng)
        obs_dict, state, rew_dict, done_dict, info = env.step(
            step_key, state, action_dict, env_params)
        done_flag = jnp.array([float(done_dict[env.agents[0]])])

        if step % 50 == 0 or step < 10:
            print(f"{step:6d} | {alt:7.0f} | {vt:6.1f} | {roll:+7.1f} | {pitch:+7.1f} | "
                  f"{yaw:+7.1f} | {tgt_h:+7.1f} | {h_err:+7.1f} | {tgt_p:+7.1f} | {p_err:+7.1f} | "
                  f"{tgt_r:+7.1f} | {r_err:+7.1f} | {tgt_v:7.0f} | {v_err:+7.1f}")

        if bool(done_dict["__all__"]):
            print(f"\n[DONE] step={step}, reason={'CRASH' if not state.success else 'TIMEOUT'}")
            break


if __name__ == "__main__":
    main()
