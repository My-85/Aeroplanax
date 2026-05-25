"""
Diagnose why training return is high (886) but render waypoint tracking is terrible (14/100 WP).

Hypotheses to test:
1. Does the policy output sensible actions for "perfect alignment" (zero obs error)?
2. Does the policy survive when run in training mode (targets = current + small delta)?
3. What happens with the RNN hidden state over the first few steps?
4. Observation distribution: training vs render
5. Action distribution: what does the policy typically output?
"""
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.3'
import jax, jax.numpy as jnp, numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import functools, distrax
from typing import Sequence, Dict
import orbax.checkpoint as ocp

# ── Network (same as render) ──
class ScannedRNN(nn.Module):
    @functools.partial(nn.scan, variable_broadcast='params', in_axes=0, out_axes=0, split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry; ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y
    @staticmethod
    def initialize_carry(bs, hs):
        return nn.GRUCell(features=hs).initialize_carry(jax.random.PRNGKey(0), (bs, hs))

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]; config: Dict
    @nn.compact
    def __call__(self, hidden, x):
        ac = nn.relu if self.config['ACTIVATION'] == 'relu' else nn.tanh
        obs, dones = x
        e = ac(nn.Dense(self.config['FC_DIM_SIZE'], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs))
        hidden, e = ScannedRNN()(hidden, (e, dones))
        fc2 = ac(nn.LayerNorm()(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(e)))
        am = ac(nn.Dense(self.config['GRU_HIDDEN_DIM'], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
        heads = []
        for i in range(4):
            heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[i], kernel_init=orthogonal(0.01), bias_init=constant(0.0))(am)))
        heads.append(distrax.Categorical(logits=nn.Dense(self.action_dim[4], kernel_init=constant(0.0),
            bias_init=lambda key, shape, dtype=jnp.float32: jnp.array([0.0,-1.5,-1.5,-1.5,-1.5], dtype=dtype))(am)))
        c = ac(nn.Dense(self.config['FC_DIM_SIZE'], kernel_init=orthogonal(2), bias_init=constant(0.0))(fc2))
        c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return hidden, (heads[0], heads[1], heads[2], heads[3], heads[4]), jnp.squeeze(c, axis=-1)

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams,
    _quat_from_euler_nb, _quat_conj, _quat_err_bn,
)

env = AeroPlanaxHeading_Pitch_V_Env(Heading_Pitch_V_TaskParams())
cfg = {'FC_DIM_SIZE': 128, 'GRU_HIDDEN_DIM': 128, 'ACTIVATION': 'relu'}
net = ActorCriticRNN([31, 41, 41, 41, 5], config=cfg)
rng = jax.random.PRNGKey(111)

CKPT = os.path.abspath('results/heading_pitch_V_discrete_rnn_2026-05-12-12-35/checkpoints/checkpoint_epoch_300')
ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
ckpt = ckptr.restore(CKPT, args=ocp.args.StandardRestore())
ckpt_params = ckpt['params']

h0 = ScannedRNN.initialize_carry(1, 128)
obs_shape = env.observation_space(env.agents[0], Heading_Pitch_V_TaskParams()).shape

print("=" * 100)
print("TEST 1: Policy action on 'perfect alignment' observation (zero error)")
print("=" * 100)
# Construct a zero-error observation (q_err=[1,0,0,0] → qv=[0,0,0])
# obs layout: qv[0],qv[1],qv[2], dvt, alt/5000, vt/340, v_b[0],v_b[1],v_b[2], P,Q,R, sin(a),cos(a), sin(b),cos(b)
zero_obs = jnp.array([0.0, 0.0, 0.0,   # qv = [0,0,0] (perfect attitude match)
                       0.0,              # dvt = 0
                       4680/5000,        # alt normalized
                       285.5/340,        # vt normalized
                       1.0, 0.0, 0.0,    # v_b = [1,0,0] (target straight ahead)
                       0.0, 0.0, 0.0,    # PQR = [0,0,0]
                       0.0, 1.0,          # sin/cos alpha (alpha≈0)
                       0.0, 1.0])         # sin/cos beta (beta≈0)
obs_in = zero_obs[None, None, :]
done_in = jnp.zeros((1, 1))

h, pi, val = net.apply(ckpt_params, h0, (obs_in, done_in))
for name, p in [("throttle", pi[0]), ("elevator", pi[1]), ("aileron", pi[2]), ("rudder", pi[3]), ("sb", pi[4])]:
    logits = np.array(p.logits[0, 0])
    probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    mode = int(p.mode()[0, 0])
    sample = int(p.sample(seed=jax.random.PRNGKey(0))[0, 0])
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
    top3_idx = np.argsort(logits)[-3:][::-1]
    top3_str = ", ".join([f"{i}:{logits[i]:+.1f}({probs[i]:.3f})" for i in top3_idx])
    print(f"  {name:>12}: mode={mode:3d}, sample={sample:3d}, entropy={entropy:.4f}, value={float(val[0,0]):.2f}")
    print(f"               top3: [{top3_str}]")

print()
print("=" * 100)
print("TEST 2: Multi-seed test — run policy in training mode (targets close to current)")
print("=" * 100)
# In training, targets change every ~25 steps to current_state + small delta
# The aircraft starts with random attitude and targets = current state
# Then targets change when success=True (every 25 steps)

crash_count = 0
ok_count = 0
max_ok_steps = 0

for seed in range(50):
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    obs_dict, state = env.reset(reset_key, Heading_Pitch_V_TaskParams())
    ps = state.plane_state

    obs_vec = obs_dict[env.agents[0]]
    obs_in = obs_vec[None, None, :]
    done_in = jnp.zeros((1, 1))
    hstate = ScannedRNN.initialize_carry(1, 128)

    crash_early = False
    survived_steps = 0
    for step in range(500):
        hstate, pi, _ = net.apply(ckpt_params, hstate, (obs_in, done_in))
        acts = [int(p.mode()[0, 0]) for p in pi]
        action = {env.agents[0]: jnp.array(acts)}
        rng, key = jax.random.split(rng)
        obs2, state2, rew, done, info = env.step(key, state, action, Heading_Pitch_V_TaskParams())
        d = bool(np.asarray(done[env.agents[0]]).item())

        if d and step < 10:
            crash_early = True
            break

        if d:
            break

        survived_steps += 1
        obs_in = obs2[env.agents[0]][None, None, :]
        state = state2

    if crash_early:
        crash_count += 1
    else:
        ok_count += 1
        max_ok_steps = max(max_ok_steps, survived_steps)

    if seed < 10:
        yaw = float(np.asarray(ps.yaw).reshape(-1)[0])
        pitch = float(np.asarray(ps.pitch).reshape(-1)[0])
        roll = float(np.asarray(ps.roll).reshape(-1)[0])
        vt = float(np.asarray(ps.vt).reshape(-1)[0])
        status = "CRASH_EARLY" if crash_early else f"OK({survived_steps}st)"
        print(f"  seed={seed:3d}: yaw={np.degrees(yaw):+.0f}° pitch={np.degrees(pitch):+.0f}° "
              f"roll={np.degrees(roll):+.0f}° vt={vt:.0f} → {status}")

print(f"\n  Early crash (<10 steps): {crash_count}/{50}  ({crash_count/50*100:.1f}%)")
print(f"  OK: {ok_count}/{50}, max steps among OK: {max_ok_steps}")

print()
print("=" * 100)
print("TEST 3: Action sensitivity — how does policy respond to different observations?")
print("=" * 100)

# Test action changes as we vary key observation components
def get_action(obs_vec):
    obs_in = jnp.array(obs_vec)[None, None, :]
    done_in = jnp.zeros((1, 1))
    h_test = ScannedRNN.initialize_carry(1, 128)
    _, pi, _ = net.apply(ckpt_params, h_test, (obs_in, done_in))
    return [int(p.mode()[0, 0]) for p in pi]

base_obs = [0.0, 0.0, 0.0, 0.0, 4680/5000, 285.5/340, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]

print("\n  Varying qv[0] (quaternion error x-component):")
for qv0 in [-0.5, -0.2, 0.0, 0.2, 0.5]:
    o = base_obs.copy(); o[0] = qv0
    acts = get_action(o)
    print(f"    qv_x={qv0:+.1f} → thr={acts[0]:2d}, el={acts[1]:2d}, ail={acts[2]:2d}, rud={acts[3]:2d}, sb={acts[4]}")

print("\n  Varying dvt (speed error / 340):")
for dvt in [-0.5, -0.1, 0.0, 0.1, 0.5]:
    o = base_obs.copy(); o[3] = dvt
    acts = get_action(o)
    print(f"    dvt={dvt:+.1f} → thr={acts[0]:2d}, el={acts[1]:2d}, ail={acts[2]:2d}, rud={acts[3]:2d}, sb={acts[4]}")

print("\n  Varying P (roll rate):")
for P_val in [-2.0, -0.5, 0.0, 0.5, 2.0]:
    o = base_obs.copy(); o[9] = P_val
    acts = get_action(o)
    print(f"    P={P_val:+.1f} → thr={acts[0]:2d}, el={acts[1]:2d}, ail={acts[2]:2d}, rud={acts[3]:2d}, sb={acts[4]}")

print("\n  Varying Q (pitch rate):")
for Q_val in [-2.0, -0.5, 0.0, 0.5, 2.0]:
    o = base_obs.copy(); o[10] = Q_val
    acts = get_action(o)
    print(f"    Q={Q_val:+.1f} → thr={acts[0]:2d}, el={acts[1]:2d}, ail={acts[2]:2d}, rud={acts[3]:2d}, sb={acts[4]}")

print("\n  Varying R (yaw rate):")
for R_val in [-2.0, -0.5, 0.0, 0.5, 2.0]:
    o = base_obs.copy(); o[11] = R_val
    acts = get_action(o)
    print(f"    R={R_val:+.1f} → thr={acts[0]:2d}, el={acts[1]:2d}, ail={acts[2]:2d}, rud={acts[3]:2d}, sb={acts[4]}")

print("\n  Varying v_b[0] (target direction in body x):")
for vbx in [-1.0, -0.5, 0.0, 0.5]:
    o = base_obs.copy(); o[6] = vbx
    acts = get_action(o)
    print(f"    v_bx={vbx:+.1f} → thr={acts[0]:2d}, el={acts[1]:2d}, ail={acts[2]:2d}, rud={acts[3]:2d}, sb={acts[4]}")

print()
print("=" * 100)
print("TEST 4: Check what obs the render seed (42) produces at step 0")
print("=" * 100)
rng = jax.random.PRNGKey(42)
rng, reset_key = jax.random.split(rng)
obs_dict, state = env.reset(reset_key, Heading_Pitch_V_TaskParams())
ps = state.plane_state
obs_vec = np.array(obs_dict[env.agents[0]])
yaw = float(np.asarray(ps.yaw).reshape(-1)[0])
pitch = float(np.asarray(ps.pitch).reshape(-1)[0])
roll = float(np.asarray(ps.roll).reshape(-1)[0])
vt = float(np.asarray(ps.vt).reshape(-1)[0])
alt = float(np.asarray(ps.altitude).reshape(-1)[0])

print(f"  Seed 42 initial state: yaw={np.degrees(yaw):.1f}° pitch={np.degrees(pitch):.1f}° "
      f"roll={np.degrees(roll):.1f}° vt={vt:.1f} alt={alt:.0f}")
print(f"  Observation vector (16 dims):")
labels = ['qv_x', 'qv_y', 'qv_z', 'dvt', 'alt/5k', 'vt/340',
          'vb_x', 'vb_y', 'vb_z', 'P', 'Q', 'R',
          'sin(a)', 'cos(a)', 'sin(b)', 'cos(b)']
for i in range(16):
    print(f"    [{i:2d}] {labels[i]:>10s} = {obs_vec[i]:+.6f}")

# Check target values
print(f"  Target heading: {np.degrees(float(np.asarray(state.target_heading).reshape(-1)[0])):.1f}°")
print(f"  Target pitch:   {np.degrees(float(np.asarray(state.target_pitch).reshape(-1)[0])):.1f}°")
print(f"  Target roll:    {np.degrees(float(np.asarray(state.target_roll).reshape(-1)[0])):.1f}°")
print(f"  Target vt:      {float(np.asarray(state.target_vt).reshape(-1)[0]):.1f}")

print()
print("=" * 100)
print("TEST 5: Render-mode simulation with detailed step-by-step logging")
print("=" * 100)
# Simulate the render logic more carefully
from envs.utils.utils import wrap_PI

CRUISE_VT = 250.0
rng = jax.random.PRNGKey(42)
rng, reset_key = jax.random.split(rng)
obs_dict, state = env.reset(reset_key, Heading_Pitch_V_TaskParams())
hstate = ScannedRNN.initialize_carry(1, 128)
done_flag = jnp.zeros((1,))

for step in range(30):
    ps = state.plane_state
    north = float(np.asarray(ps.north).reshape(-1)[0])
    east = float(np.asarray(ps.east).reshape(-1)[0])
    alt = float(np.asarray(ps.altitude).reshape(-1)[0])
    vt = float(np.asarray(ps.vt).reshape(-1)[0])
    roll = float(np.asarray(ps.roll).reshape(-1)[0])
    pitch = float(np.asarray(ps.pitch).reshape(-1)[0])
    yaw = float(np.asarray(ps.yaw).reshape(-1)[0])

    # Simulate waypoint target (first waypoint roughly ahead)
    wp_n, wp_e, wp_a = 5000, 0, 4680  # arbitrary WP ahead
    d_n = wp_n - north
    d_e = wp_e - east
    d_alt = wp_a - alt
    h_dist = float(np.sqrt(d_n**2 + d_e**2))
    target_heading_raw = float(np.arctan2(d_e, d_n))

    blend = min(1.0, step / 200.0)
    hdg_err = float(np.arctan2(np.sin(target_heading_raw - yaw), np.cos(target_heading_raw - yaw)))
    target_heading = float(np.arctan2(np.sin(yaw + blend * hdg_err), np.cos(yaw + blend * hdg_err)))
    target_pitch_raw = float(np.arctan2(np.clip(d_alt, -2000, 2000), max(h_dist, 1e-6)))
    target_pitch = float(pitch + blend * (target_pitch_raw - pitch))
    target_roll_raw = float(np.clip(0.5 * hdg_err, -0.5, 0.5))
    roll_err = float(np.arctan2(np.sin(target_roll_raw - roll), np.cos(target_roll_raw - roll)))
    target_roll = float(np.arctan2(np.sin(roll + blend * roll_err), np.cos(roll + blend * roll_err)))
    target_vt = float(vt + blend * (CRUISE_VT - vt))

    state_with_targets = state.replace(
        target_heading=jnp.array([target_heading]),
        target_pitch=jnp.array([target_pitch]),
        target_roll=jnp.array([target_roll]),
        target_vt=jnp.array([target_vt]),
    )
    obs_dict = env._get_obs(state_with_targets, Heading_Pitch_V_TaskParams())
    obs_vec = obs_dict[env.agents[0]]

    if step < 5:
        obs_arr = np.array(obs_vec)
        print(f"  [Step {step}] blend={blend:.3f}")
        print(f"    target: h={np.degrees(target_heading):.1f}° p={np.degrees(target_pitch):.1f}° "
              f"r={np.degrees(target_roll):.1f}° v={target_vt:.0f}")
        print(f"    obs: qv=[{obs_arr[0]:.3f},{obs_arr[1]:.3f},{obs_arr[2]:.3f}] "
              f"dvt={obs_arr[3]:.3f} v_b=[{obs_arr[6]:.3f},{obs_arr[7]:.3f},{obs_arr[8]:.3f}] "
              f"PQR=[{obs_arr[9]:.2f},{obs_arr[10]:.2f},{obs_arr[11]:.2f}]")

    obs_in = obs_vec[None, None, :]
    done_in = done_flag[None, :]
    hstate, pi, val = net.apply(ckpt_params, hstate, (obs_in, done_in))
    acts = [int(p.mode()[0, 0]) for p in pi]
    action = {env.agents[0]: jnp.array(acts)}

    rng, key = jax.random.split(rng)
    obs2, state, rew, done, info = env.step(key, state, action, Heading_Pitch_V_TaskParams())
    done_flag = jnp.array([float(done[env.agents[0]])])

    d = bool(np.asarray(done[env.agents[0]]).item())
    ps2 = state.plane_state
    new_alt = float(np.asarray(ps2.altitude).reshape(-1)[0])
    new_vt = float(np.asarray(ps2.vt).reshape(-1)[0])
    new_roll = np.degrees(float(np.asarray(ps2.roll).reshape(-1)[0]))
    new_pitch = np.degrees(float(np.asarray(ps2.pitch).reshape(-1)[0]))
    new_yaw = np.degrees(float(np.asarray(ps2.yaw).reshape(-1)[0]))

    if step < 5:
        print(f"    action: thr={acts[0]}, el={acts[1]}, ail={acts[2]}, rud={acts[3]}, sb={acts[4]}")
        print(f"    result: alt={new_alt:.0f} vt={new_vt:.1f} roll={new_roll:.1f}° pitch={new_pitch:.1f}° yaw={new_yaw:.1f}° "
              f"done={d} rew={float(rew[env.agents[0]]):.2f}")
        print()

    if d:
        print(f"  CRASH at step {step}!")
        break

print()
print("=" * 100)
print("TEST 6: Check value function (critic) — does it predict the crash?")
print("=" * 100)
# If the critic predicts low values, the policy knows it's in a bad state
# If the critic predicts high values, the policy is delusional
rng = jax.random.PRNGKey(42)
rng, reset_key = jax.random.split(rng)
obs_dict, state = env.reset(reset_key, Heading_Pitch_V_TaskParams())
obs_vec = obs_dict[env.agents[0]]
obs_in = obs_vec[None, None, :]
done_in = jnp.zeros((1, 1))
h_test = ScannedRNN.initialize_carry(1, 128)
_, pi, val = net.apply(ckpt_params, h_test, (obs_in, done_in))
print(f"  Step 0 value estimate: {float(val[0,0]):.3f}")
print(f"  (Values typically range [-200, 900] — low value = policy expects bad outcome)")

print()
print("DONE")
