"""
Calibrate closed-loop command bandwidth of the trained Euler baseline.

Runs step-command tests: applies a sudden change in heading, pitch, or speed
target and measures the RL policy's response time, overshoot, settling time,
and actuator saturation.

Output is used to set tau_cmd, yaw_rate_max, pitch_rate_max in DP solver.
"""

import os, sys, json
from datetime import datetime
from pathlib import Path
import numpy as np

GPU_ID = os.environ.get("PLANAX_GPU", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["XLA_PYTHON_MEM_FRACTION"] = "0.7"

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))


CKPT_PATH = os.path.abspath(
    "results/heading_pitch_V_discrete_rnn_2026-05-09-16-53/checkpoints/checkpoint_epoch_300"
)
OUTPUT_DIR = "outputs/bandwidth_calibration"
CRUISE_VT = 250.0
MAX_STEPS = 1000
SETTLING_THRESHOLD = 0.1  # 10% of step magnitude = settled


def run_step_response(env, env_params, network, net_params, hstate_init,
                      step_type: str, step_magnitude: float, rng_seed: int):
    """
    Run a step-command test.

    step_type: "heading" (rad), "pitch" (rad), or "speed" (m/s)
    step_magnitude: size of the step change
    """
    import jax, jax.numpy as jnp

    rng = jax.random.PRNGKey(rng_seed)
    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, env_params)
    hstate = hstate_init

    ps = state.plane_state
    yaw0 = float(np.asarray(ps.yaw).reshape(-1)[0])
    pitch0 = float(np.asarray(ps.pitch).reshape(-1)[0])
    vt0 = float(np.asarray(ps.vt).reshape(-1)[0])
    alt0 = float(np.asarray(ps.altitude).reshape(-1)[0])

    # Baseline targets (hold current state for first 50 steps to stabilise)
    tgt_h = yaw0
    tgt_p = pitch0
    tgt_v = vt0

    rec_t, rec_yaw, rec_pitch, rec_vt, rec_alt, rec_roll = [], [], [], [], [], []
    rec_thr, rec_el, rec_ail, rec_rud = [], [], [], []
    rec_tgt_h, rec_tgt_p, rec_tgt_v = [], [], []
    step_applied_at = None

    for step in range(MAX_STEPS):
        t = step * 0.2

        # Apply step after 50 steps (10 seconds) of stabilisation
        if step == 50:
            if step_type == "heading":
                tgt_h = yaw0 + step_magnitude
            elif step_type == "pitch":
                tgt_p = pitch0 + step_magnitude
            elif step_type == "speed":
                tgt_v = vt0 + step_magnitude
            step_applied_at = t

        ps = state.plane_state
        yaw = float(np.asarray(ps.yaw).reshape(-1)[0])
        pitch = float(np.asarray(ps.pitch).reshape(-1)[0])
        vt = float(np.asarray(ps.vt).reshape(-1)[0])
        alt = float(np.asarray(ps.altitude).reshape(-1)[0])
        roll = float(np.asarray(ps.roll).reshape(-1)[0])

        state_w = state.replace(
            target_heading=jnp.array([tgt_h]),
            target_pitch=jnp.array([tgt_p]),
            target_vt=jnp.array([tgt_v]),
        )
        obs_dict_w = env._get_obs(state_w, env_params)
        obs_vec = obs_dict_w[env.agents[0]]
        obs_in = obs_vec[None, None, :]
        done_in = jnp.zeros((1, 1))

        hstate, pi, _ = network.apply(net_params, hstate, (obs_in, done_in))
        acts = [int(p.mode()[0, 0]) for p in pi]
        action_dict = {env.agents[0]: jnp.array(acts)}

        rng, step_key = jax.random.split(rng)
        obs_dict2, state, rew_dict, done_dict, info = env.step(
            step_key, state, action_dict, env_params)

        rec_t.append(t)
        rec_yaw.append(np.degrees(yaw)); rec_pitch.append(np.degrees(pitch))
        rec_vt.append(vt); rec_alt.append(alt); rec_roll.append(np.degrees(roll))
        rec_thr.append(acts[0]); rec_el.append(acts[1])
        rec_ail.append(acts[2]); rec_rud.append(acts[3])
        rec_tgt_h.append(np.degrees(tgt_h))
        rec_tgt_p.append(np.degrees(tgt_p))
        rec_tgt_v.append(tgt_v)

        if bool(done_dict["__all__"]):
            break

    # ── Analyse step response ──
    if step_type == "heading":
        signal = np.array(rec_yaw)
        target = np.degrees(tgt_h)
    elif step_type == "pitch":
        signal = np.array(rec_pitch)
        target = np.degrees(tgt_p)
    else:
        signal = np.array(rec_vt)
        target = tgt_v

    # Unwrap heading for analysis
    if step_type == "heading":
        signal = np.unwrap(np.radians(signal), period=360)
        signal = np.degrees(signal)
        target_unwrapped = np.degrees(yaw0) + np.degrees(step_magnitude)
    else:
        target_unwrapped = target

    # Rise time: time from step to first reach target ±10% tolerance
    pre_step = signal[50]
    step_size = abs(target_unwrapped - pre_step)
    tol = step_size * SETTLING_THRESHOLD

    rise_idx = None
    settling_idx = None
    overshoot = 0.0
    if step_applied_at is not None:
        for k in range(50, len(signal)):
            err = abs(signal[k] - target_unwrapped)
            if rise_idx is None and err < step_size * 0.9:
                rise_idx = k
            if err < tol:
                # Check sustained: next 5 samples also within tolerance
                sustained = all(abs(signal[m] - target_unwrapped) < tol
                               for m in range(k, min(k + 5, len(signal))))
                if sustained:
                    settling_idx = k
                    break
        peak = np.max(np.abs(np.array(signal[50:]) - target_unwrapped))
        overshoot = peak / max(step_size, 0.01) - 1.0

    actions_arr = np.column_stack([rec_thr, rec_el, rec_ail, rec_rud])
    ele_norm = actions_arr[:, 1].astype(float) * 2.0 / 40.0 - 1.0
    ail_norm = actions_arr[:, 2].astype(float) * 2.0 / 40.0 - 1.0
    rud_norm = actions_arr[:, 3].astype(float) * 2.0 / 40.0 - 1.0
    post_step = slice(50, min(50 + 100, len(ele_norm)))
    sat_ele = np.mean(np.abs(ele_norm[post_step]) > 0.95)
    sat_ail = np.mean(np.abs(ail_norm[post_step]) > 0.95)
    sat_rud = np.mean(np.abs(rud_norm[post_step]) > 0.95)

    return {
        "step_type": step_type,
        "step_magnitude": step_magnitude,
        "step_magnitude_deg": np.degrees(step_magnitude) if step_type != "speed" else step_magnitude,
        "rise_time_s": (rise_idx - 50) * 0.2 if rise_idx else None,
        "settling_time_s": (settling_idx - 50) * 0.2 if settling_idx else None,
        "overshoot_ratio": float(overshoot),
        "peak_error": float(np.max(np.abs(np.array(signal[50:]) - target_unwrapped))),
        "steady_state_error": float(np.mean(np.abs(np.array(signal[-20:]) - target_unwrapped))),
        "post_step_saturation_elevator": float(sat_ele),
        "post_step_saturation_aileron": float(sat_ail),
        "post_step_saturation_rudder": float(sat_rud),
        "data": {
            "t": rec_t, "yaw_deg": rec_yaw, "pitch_deg": rec_pitch,
            "vt": rec_vt, "alt": rec_alt, "roll_deg": rec_roll,
            "target_h_deg": rec_tgt_h, "target_p_deg": rec_tgt_p,
            "target_vt": rec_tgt_v,
            "throttle": rec_thr, "elevator": rec_el,
            "aileron": rec_ail, "rudder": rec_rud,
        }
    }


def main():
    import jax, jax.numpy as jnp
    import orbax.checkpoint as ocp
    import flax.linen as nn
    from flax.linen.initializers import constant, orthogonal
    import functools, distrax
    from typing import Sequence, Dict

    from envs.aeroplanax_heading_pitch_V import (
        AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
    )

    # Inline network (same as training script)
    class ScannedRNN(nn.Module):
        @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
        @nn.compact
        def __call__(self, carry, x):
            rnn_state = carry
            ins, resets = x
            rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state)
            new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
            return new_rnn_state, y
        @staticmethod
        def initialize_carry(batch_size, hidden_size):
            cell = nn.GRUCell(features=hidden_size)
            return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

    class ActorCriticRNN(nn.Module):
        action_dim: Sequence[int]; config: Dict
        @nn.compact
        def __call__(self, hidden, x):
            act_fn = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
            obs, dones = x
            e = act_fn(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs))
            hidden, e = ScannedRNN()(hidden, (e, dones))
            fc2 = act_fn(nn.LayerNorm()(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(e)))
            am = act_fn(nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
            pi_thr = distrax.Categorical(logits=nn.Dense(self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am))
            pi_ele = distrax.Categorical(logits=nn.Dense(self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am))
            pi_ail = distrax.Categorical(logits=nn.Dense(self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am))
            pi_rud = distrax.Categorical(logits=nn.Dense(self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am))
            c = act_fn(nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
            c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
            return hidden, (pi_thr, pi_ele, pi_ail, pi_rud), jnp.squeeze(c, axis=-1)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_DIR) / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)
    config = {"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128, "ACTIVATION": "relu"}
    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    rng = jax.random.PRNGKey(42)
    obs_shape = env.observation_space(env.agents[0], env_params).shape
    init_x = (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1)))
    h0 = ScannedRNN.initialize_carry(1, 128)
    net_params = network.init(rng, h0, init_x)
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore())
    net_params = ckpt["params"]
    print(f"Loaded checkpoint, epoch={int(ckpt['epoch'])}")

    # ── Run step tests ──
    test_cases = [
        # (type, magnitude_rad_or_ms)
        ("heading", np.radians(10)), ("heading", np.radians(20)),
        ("heading", np.radians(45)), ("heading", np.radians(90)),
        ("pitch", np.radians(5)), ("pitch", np.radians(10)),
        ("pitch", np.radians(20)), ("pitch", np.radians(30)),
        ("speed", 10.0), ("speed", 20.0), ("speed", 50.0),
    ]

    all_results = []
    for step_type, mag in test_cases:
        print(f"Testing {step_type} step={np.degrees(mag) if step_type!='speed' else mag:.0f}")
        result = run_step_response(env, env_params, network, net_params, h0,
                                   step_type, mag, rng_seed=42)
        all_results.append(result)
        label = f"{step_type}_{np.degrees(mag) if step_type!='speed' else mag:.0f}"
        with open(output_dir / f"{label}.json", "w") as f:
            json.dump({k: v for k, v in result.items() if k != "data"}, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Step':<20} {'Rise(s)':>8} {'Settle(s)':>10} {'Overshoot':>10} {'Sat%':>8}")
    print("-" * 70)
    for r in all_results:
        name = f"{r['step_type']} {r['step_magnitude_deg']:.0f}°" if r['step_type'] != 'speed' else f"speed {r['step_magnitude']:.0f} m/s"
        rise = f"{r['rise_time_s']:.2f}" if r['rise_time_s'] else "N/A"
        settle = f"{r['settling_time_s']:.2f}" if r['settling_time_s'] else "N/A"
        sat = r['post_step_saturation_elevator'] + r['post_step_saturation_aileron']
        print(f"{name:<20} {rise:>8} {settle:>10} {r['overshoot_ratio']:>9.2f} {sat:>7.2f}")

    # Recommended parameters
    valid = [r for r in all_results if r['settling_time_s'] is not None]
    if valid:
        max_settle = max(r['settling_time_s'] for r in valid)
        print(f"\nRecommended DP parameters:")
        print(f"  tau_cmd >= {max_settle * 2.0:.1f}s (2× max settling time)")
        print(f"  psi_dot_max <= {np.degrees(np.radians(90) / max_settle):.0f}°/s (from 90° step)")
        print(f"  pitch_dot_max <= {np.degrees(np.radians(20) / max_settle):.0f}°/s (from 20° step)")

    print(f"\nOutput: {output_dir}")
    json.dump(all_results, open(output_dir / "all_results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
