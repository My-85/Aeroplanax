"""
Calibrate min_check_interval per difficulty level.

Uses a physically-plausible PD controller (bank-to-turn + pitch + speed)
to measure how many RL steps it takes for the F-16 to converge to random
targets of varying difficulty.

Each RL step = AGENT_INTERACTION_STEPS / sim_freq = 10/50 = 0.2s.
"""
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.15'

import jax, jax.numpy as jnp, numpy as np
from functools import partial

from envs.core.simulators.fighterplane.dynamics import (
    FighterPlaneState, FighterPlaneControlState, update,
)
from envs.utils.utils import wrap_PI

SIM_FREQ = 50
AGENT_INTERACTION_STEPS = 10
DT_RL = AGENT_INTERACTION_STEPS / SIM_FREQ
GRAVITY = 9.81

HEADING_TOL = jnp.pi / 72   # 2.5°
PITCH_TOL   = jnp.pi / 72   # 2.5°
SPEED_TOL   = 10.0

LEVELS = {
    0: {"hdg": jnp.pi/6,     "pitch": jnp.pi/18,    "speed": 20.0,   "max_steps": 100},
    1: {"hdg": jnp.pi/2,     "pitch": jnp.pi/6,     "speed": 50.0,   "max_steps": 200},
    2: {"hdg": jnp.pi,       "pitch": jnp.pi/3,     "speed": 100.0,  "max_steps": 400},
    3: {"hdg": jnp.pi,       "pitch": 89*jnp.pi/180,"speed": 120.0,  "max_steps": 600},
}


def make_trim_state():
    s_arr = jnp.zeros(26, dtype=jnp.float32)
    s_arr = s_arr.at[2].set(5000.0)                # altitude
    s_arr = s_arr.at[7].set(250.0)                 # vel_y = vt
    s_arr = s_arr.at[9].set(250.0)                 # vt
    s_arr = s_arr.at[10].set(1.0)                  # q0 = 1 (identity quat)
    s_arr = s_arr.at[14].set(jnp.radians(2.0))     # alpha ~2° (cruise trim)
    return FighterPlaneState.create(s_arr)


def controller(state: FighterPlaneState, target_heading, target_pitch, target_vt):
    """Physically-plausible bank-to-turn PD controller."""
    vt_safe = jnp.clip(jnp.abs(state.vt), 80.0, 400.0)

    # Heading → coordinated turn bank angle
    hdg_err = wrap_PI(target_heading - state.yaw)
    desired_rate = jnp.clip(0.3 * hdg_err, -0.12, 0.12)  # max ~7°/s
    target_bank = jnp.arctan(vt_safe * desired_rate / GRAVITY)
    target_bank = jnp.clip(target_bank, -jnp.radians(70), jnp.radians(70))

    # Roll PD
    roll_err = wrap_PI(target_bank - state.roll)
    ail = jnp.clip(0.8 * roll_err - 0.15 * state.P, -1.0, 1.0)

    # Pitch PD
    pitch_err = wrap_PI(target_pitch - state.pitch)
    el = jnp.clip(1.5 * pitch_err - 0.3 * state.Q, -1.0, 1.0)

    # Speed P (slow)
    thr = jnp.clip(0.55 + 0.003 * (target_vt - state.vt), 0.05, 1.0)

    # Rudder — light coordination
    rud = jnp.clip(0.1 * hdg_err - 0.05 * state.R, -1.0, 1.0)

    return FighterPlaneControlState.create(jnp.array([thr, el, ail, rud]))


# JIT-compiled single RL step (10 physics sub-steps)
@partial(jax.jit, static_argnames=("substeps",))
def rl_step(action: FighterPlaneControlState, state: FighterPlaneState, *, substeps: int = AGENT_INTERACTION_STEPS):
    def sub_step(s, _):
        return update(s, action, 1.0 / SIM_FREQ), None
    s, _ = jax.lax.scan(sub_step, state, None, length=substeps)
    return s


@partial(jax.jit, static_argnames=("max_steps",))
def run_one_trial(rng_key, target_heading, target_pitch, target_vt, *, max_steps: int):
    """JIT-compiled single trial: track one set of targets, return first-success step."""
    state0 = make_trim_state()
    sentinel = max_steps + 1

    def scan_fn(carry, _):
        s, step, hit_step, done = carry
        step = step + 1

        act = controller(s, target_heading, target_pitch, target_vt)
        s_next = rl_step(act, s)

        converged = (jnp.abs(wrap_PI(s_next.yaw - target_heading)) < HEADING_TOL) & \
                    (jnp.abs(wrap_PI(s_next.pitch - target_pitch)) < PITCH_TOL) & \
                    (jnp.abs(s_next.vt - target_vt) < SPEED_TOL)

        newly_done = converged & (~done)
        hit_step = jnp.where(newly_done, step, hit_step)
        done = done | converged

        return (s_next, step, hit_step, done), None

    init = (state0, 0, sentinel, False)
    (_, _, hit_step, _), _ = jax.lax.scan(scan_fn, init, None, length=max_steps)
    return hit_step


def run_batch(rng_key, level: int, num_trials: int = 200):
    """Run many trials for one difficulty level."""
    params = LEVELS[level]
    max_steps = params["max_steps"]
    hdg_range = params["hdg"]
    pitch_range = params["pitch"]
    speed_range = params["speed"]

    # One key per trial
    keys = jax.random.split(rng_key, num_trials)

    def one_trial(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        if level == 3:
            th = jax.random.uniform(k1, minval=-jnp.pi, maxval=jnp.pi)
        else:
            dh = jax.random.uniform(k1, minval=-hdg_range, maxval=hdg_range)
            th = wrap_PI(dh)  # trim yaw=0

        tp = jnp.clip(jax.random.uniform(k2, minval=-pitch_range, maxval=pitch_range),
                      jnp.radians(-89), jnp.radians(89))
        tv = jnp.clip(250.0 + jax.random.uniform(k3, minval=-speed_range, maxval=speed_range),
                      120.0, 360.0)

        hit_step = run_one_trial(k4, th, tp, tv, max_steps=max_steps)
        init_h = jnp.abs(wrap_PI(th - 0.0))
        init_p = jnp.abs(tp - 0.0)
        init_v = jnp.abs(tv - 250.0)
        return hit_step, init_h, init_p, init_v

    hit_steps, init_h, init_p, init_v = jax.vmap(one_trial)(keys)
    success = hit_steps <= max_steps
    return hit_steps, success, init_h, init_p, init_v


if __name__ == "__main__":
    print("=" * 80)
    print("Calibrating min_check_interval per difficulty level")
    print(f"  RL step = {AGENT_INTERACTION_STEPS}/{SIM_FREQ}Hz = {DT_RL:.2f}s")
    print(f"  Tolerances: heading=±{np.degrees(HEADING_TOL):.1f}°, "
          f"pitch=±{np.degrees(PITCH_TOL):.1f}°, speed=±{SPEED_TOL:.0f}m/s")
    print()

    NUM_TRIALS = 200
    rng = jax.random.PRNGKey(42)

    for level in sorted(LEVELS.keys()):
        params = LEVELS[level]
        print(f"Level {level}: hdg±{np.degrees(params['hdg']):.0f}°  "
              f"pitch±{np.degrees(params['pitch']):.0f}°  "
              f"speed±{params['speed']:.0f}m/s  (max {params['max_steps']} steps)")

        rng, batch_key = jax.random.split(rng)
        hit_steps, success, init_h, init_p, init_v = run_batch(batch_key, level, NUM_TRIALS)

        steps_arr = np.array(hit_steps)
        succ_arr = np.array(success)

        rate = float(np.mean(succ_arr))
        print(f"  Success rate: {rate*100:.0f}%")

        hit_success = steps_arr[succ_arr]
        if len(hit_success) > 0:
            p50 = int(np.percentile(hit_success, 50))
            p90 = int(np.percentile(hit_success, 90))
            p95 = int(np.percentile(hit_success, 95))
            p99 = int(np.percentile(hit_success, 99))
            mx  = int(np.max(hit_success))
            print(f"  Steps: p50={p50}  p90={p90}  p95={p95}  p99={p99}  max={mx}")
            # Recommended: p95 + 20%, rounded up to nearest 5
            rec = int(np.ceil(p95 * 1.2 / 5) * 5)
            # But never exceed max_steps
            rec = min(rec, params["max_steps"])
            # Floor at 20 steps
            rec = max(rec, 20)
            print(f"  → Recommended check_interval: {rec} steps ({rec * DT_RL:.1f}s)")
        else:
            # Show what prevented convergence
            print(f"  WARNING: 0% success. Likely controller not aggressive enough "
                  f"or tolerances too tight for this level.")

        init_h_deg = np.degrees(np.array(init_h))
        init_p_deg = np.degrees(np.array(init_p))
        init_v_ms  = np.array(init_v)
        print(f"  Initial errors (mean): hdg={np.mean(init_h_deg):.1f}°  "
              f"pitch={np.mean(init_p_deg):.1f}°  speed={np.mean(init_v_ms):.0f}m/s")
        print()

    print("DONE")
