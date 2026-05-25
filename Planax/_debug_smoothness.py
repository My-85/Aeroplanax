"""Debug: why is r_sm saturated at -0.986?"""
import os; os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['XLA_PYTHON_MEM_FRACTION']='0.15'
import jax, jax.numpy as jnp, numpy as np
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams
)

env = AeroPlanaxHeading_Pitch_V_Env(Heading_Pitch_V_TaskParams())
rng = jax.random.PRNGKey(0)
rng, reset_key = jax.random.split(rng)
obs, state = env.reset(reset_key, Heading_Pitch_V_TaskParams())

# Print initial plane_state control values
ps = state.plane_state
print("=== After reset ===")
print(f"  T={float(np.asarray(ps.T).reshape(-1)[0]):.1f}")
print(f"  el={float(np.asarray(ps.el).reshape(-1)[0]):.4f}")
print(f"  ail={float(np.asarray(ps.ail).reshape(-1)[0]):.4f}")
print(f"  rud={float(np.asarray(ps.rud).reshape(-1)[0]):.4f}")
print(f"  control_state.throttle={float(np.asarray(state.control_state.throttle).reshape(-1)[0]):.4f}")
print(f"  control_state.elevator={float(np.asarray(state.control_state.elevator).reshape(-1)[0]):.4f}")

# Step 1: neutral-ish discrete action
act1 = {env.agents[0]: jnp.array([15, 20, 20, 20, 0])}
thr_norm = 15/30; el_norm = 0.0; ail_norm = 0.0; rud_norm = 0.0
print(f"\nStep 1: discrete=[15,20,20,20,0] → thr={thr_norm:.3f}, el={el_norm:.3f}, ail={ail_norm:.3f}, rud={rud_norm:.3f}")

rng, step_key = jax.random.split(rng)
obs2, state2, rew, done, info = env.step(step_key, state, act1, Heading_Pitch_V_TaskParams())
ps2 = state2.plane_state; cs2 = state2.control_state
print(f"  After step 1:")
print(f"    T={float(np.asarray(ps2.T).reshape(-1)[0]):.1f}  (expected {thr_norm*19000:.1f})")
print(f"    el={float(np.asarray(ps2.el).reshape(-1)[0]):.4f}  (expected {el_norm:.4f})")
print(f"    ail={float(np.asarray(ps2.ail).reshape(-1)[0]):.4f}  (expected {ail_norm:.4f})")
print(f"    rud={float(np.asarray(ps2.rud).reshape(-1)[0]):.4f}  (expected {rud_norm:.4f})")
print(f"    cs.thr={float(np.asarray(cs2.throttle).reshape(-1)[0]):.4f}")
print(f"    cs.el={float(np.asarray(cs2.elevator).reshape(-1)[0]):.4f}")
print(f"    cs.ail={float(np.asarray(cs2.aileron).reshape(-1)[0]):.4f}")
print(f"    cs.rud={float(np.asarray(cs2.rudder).reshape(-1)[0]):.4f}")

# Compute action_jerk manually
d_thr = abs(float(np.asarray(cs2.throttle).reshape(-1)[0]) - float(np.asarray(ps2.T).reshape(-1)[0])/19000)
d_el  = abs(float(np.asarray(cs2.elevator).reshape(-1)[0]) - float(np.asarray(ps2.el).reshape(-1)[0])) / 2.0
d_ail = abs(float(np.asarray(cs2.aileron).reshape(-1)[0]) - float(np.asarray(ps2.ail).reshape(-1)[0])) / 2.0
d_rud = abs(float(np.asarray(cs2.rudder).reshape(-1)[0]) - float(np.asarray(ps2.rud).reshape(-1)[0])) / 2.0
action_jerk = (d_thr**2 + d_el**2 + d_ail**2 + d_rud**2) / 4.0
r_sm_computed = max(-1.0, -0.3 * action_jerk)
print(f"  Computed: d_thr={d_thr:.6f} d_el={d_el:.6f} d_ail={d_ail:.6f} d_rud={d_rud:.6f}")
print(f"  action_jerk={action_jerk:.6f}  r_sm={r_sm_computed:.6f}")
print(f"  r_sm from info: {float(np.asarray(info.get('r_action_smooth',[0])).reshape(-1)[0]):.6f}")

# Step 2: SAME action
print(f"\nStep 2: SAME action [15,20,20,20,0]")
rng, step_key = jax.random.split(rng)
obs3, state3, rew2, done2, info2 = env.step(step_key, state2, act1, Heading_Pitch_V_TaskParams())
ps3 = state3.plane_state; cs3 = state3.control_state
d_thr2 = abs(float(np.asarray(cs3.throttle).reshape(-1)[0]) - float(np.asarray(ps3.T).reshape(-1)[0])/19000)
d_el2  = abs(float(np.asarray(cs3.elevator).reshape(-1)[0]) - float(np.asarray(ps3.el).reshape(-1)[0])) / 2.0
d_ail2 = abs(float(np.asarray(cs3.aileron).reshape(-1)[0]) - float(np.asarray(ps3.ail).reshape(-1)[0])) / 2.0
d_rud2 = abs(float(np.asarray(cs3.rudder).reshape(-1)[0]) - float(np.asarray(ps3.rud).reshape(-1)[0])) / 2.0
jerk2 = (d_thr2**2 + d_el2**2 + d_ail2**2 + d_rud2**2) / 4.0
r_sm2 = max(-1.0, -0.3 * jerk2)
print(f"  d_thr={d_thr2:.6f} d_el={d_el2:.6f} d_ail={d_ail2:.6f} d_rud={d_rud2:.6f}")
print(f"  action_jerk={jerk2:.6f}  r_sm_computed={r_sm2:.6f}")
print(f"  r_sm from info2: {float(np.asarray(info2.get('r_action_smooth',[0])).reshape(-1)[0]):.6f}")

# Step 3: DIFFERENT action
act2 = {env.agents[0]: jnp.array([30, 0, 40, 0, 0])}
thr2 = 30/30; el2 = 0*2/40-1; ail2 = 40*2/40-1; rud2 = 0*2/40-1
print(f"\nStep 3: DIFFERENT [30,0,40,0,0] → thr={thr2:.3f}, el={el2:.3f}, ail={ail2:.3f}, rud={rud2:.3f}")
rng, step_key = jax.random.split(rng)
obs4, state4, rew3, done3, info3 = env.step(step_key, state3, act2, Heading_Pitch_V_TaskParams())
ps4 = state4.plane_state; cs4 = state4.control_state
d_thr3 = abs(float(np.asarray(cs4.throttle).reshape(-1)[0]) - float(np.asarray(ps4.T).reshape(-1)[0])/19000)
d_el3  = abs(float(np.asarray(cs4.elevator).reshape(-1)[0]) - float(np.asarray(ps4.el).reshape(-1)[0])) / 2.0
d_ail3 = abs(float(np.asarray(cs4.aileron).reshape(-1)[0]) - float(np.asarray(ps4.ail).reshape(-1)[0])) / 2.0
d_rud3 = abs(float(np.asarray(cs4.rudder).reshape(-1)[0]) - float(np.asarray(ps4.rud).reshape(-1)[0])) / 2.0
jerk3 = (d_thr3**2 + d_el3**2 + d_ail3**2 + d_rud3**2) / 4.0
r_sm3 = max(-1.0, -0.3 * jerk3)
print(f"  d_thr={d_thr3:.6f} d_el={d_el3:.6f} d_ail={d_ail3:.6f} d_rud={d_rud3:.6f}")
print(f"  action_jerk={jerk3:.6f}  r_sm_computed={r_sm3:.6f}")
print(f"  r_sm from info3: {float(np.asarray(info3.get('r_action_smooth',[0])).reshape(-1)[0]):.6f}")

print("\nDONE")
