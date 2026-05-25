"""
Test script to investigate RL training return drop after adding speed brake + LEF auto-scheduling.
Run: python _debug_training_issue.py
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

from envs.core.simulators.fighterplane.dynamics import (
    FighterPlaneState, FighterPlaneControlState, update, atmos
)

def make_state(north=0., east=0., alt_m=5000., roll=0., pitch=0., yaw=0.,
               vt_ms=250., alpha=0.03, beta=0., P=0., Q=0., R=0.,
               T=10000., el=0., ail=0., rud=0., lef=0., sb=0.):
    s = FighterPlaneState(
        north=north, east=east, altitude=alt_m,
        roll=roll, pitch=pitch, yaw=yaw, vel_x=0., vel_y=vt_ms, vel_z=0., vt=vt_ms,
        q0=1., q1=0., q2=0., q3=0.,
        alpha=alpha, beta=beta, P=P, Q=Q, R=R,
        T=T, el=el, ail=ail, rud=rud, lef=lef, sb=sb,
        ax=0., ay=0., az=0.,
        blood=100., status=0,
    )
    return s

def make_action(throttle=0.5, elevator=0., aileron=0., rudder=0., speed_brake=0.):
    return FighterPlaneControlState(
        throttle=throttle, elevator=elevator, aileron=aileron,
        rudder=rudder, speed_brake=speed_brake
    )

print("=" * 70)
print("TEST 1: Basic dynamics sanity check (level flight, no brake)")
print("=" * 70)

s = make_state(vt_ms=250., alpha=0.03, T=20000.)
a = make_action(throttle=0.5, speed_brake=0.)
s_new = update(s, a, 0.02)
print(f"  vt: {s_new.vt:.1f} m/s  alpha: {np.degrees(s_new.alpha):.2f}°  "
      f"alt: {s_new.altitude:.1f}m  ax: {s_new.ax:.3f}g  az: {s_new.az:.3f}g")
print(f"  lef: {s_new.lef:.1f}°  sb: {np.degrees(s_new.sb):.1f}°")
assert not jnp.isnan(s_new.vt), "NaN in vt!"
assert not jnp.isnan(s_new.altitude), "NaN in altitude!"
print("  ✅ Basic dynamics OK")

print()
print("=" * 70)
print("TEST 2: Speed brake drag effect at various alpha")
print("=" * 70)

alphas = [-0.175, 0.0, 0.087, 0.175, 0.262, 0.349, 0.436, 0.524]
for alpha_rad in alphas:
    s = make_state(vt_ms=300., alpha=alpha_rad)

    # No brake
    a_off = make_action(throttle=0.3, speed_brake=0.)
    s_off = update(s, a_off, 0.02)

    # Full brake
    a_on = make_action(throttle=0.3, speed_brake=1.0)
    s_on = update(s, a_on, 0.02)

    dv = s_on.vt - s_off.vt
    dax = s_on.ax - s_off.ax
    print(f"  α={np.degrees(alpha_rad):5.1f}°  vt_no_brk={s_off.vt:.1f}  vt_brk={s_on.vt:.1f}  "
          f"Δvt={dv:+.2f}m/s  Δax={dax:+.4f}g  sb_filtered={np.degrees(s_on.sb):.0f}°")

print()
print("=" * 70)
print("TEST 3: Speed brake LIFT effect (CLDsb)")
print("=" * 70)

for alpha_rad in alphas:
    s = make_state(vt_ms=300., alpha=alpha_rad)
    a_off = make_action(throttle=0.3, speed_brake=0.)
    s_off = update(s, a_off, 0.02)
    a_on = make_action(throttle=0.3, speed_brake=1.0)
    s_on = update(s, a_on, 0.02)

    daz = s_on.az - s_off.az
    print(f"  α={np.degrees(alpha_rad):5.1f}°  az_no_brk={s_off.az:.3f}g  az_brk={s_on.az:.3f}g  Δaz={daz:+.4f}g")

print()
print("=" * 70)
print("TEST 4: LEF auto-scheduling effect (new vs old dlef=1.0)")
print("=" * 70)

# Old behavior: manually set lef=0 (dlef=1.0), overwrite after update
# New behavior: auto-scheduled LEF
test_alphas = [0.03, 0.10, 0.20, 0.30]  # ~1.7°, 5.7°, 11.5°, 17.2°

for alpha_rad in test_alphas:
    s = make_state(vt_ms=300., alpha=alpha_rad, lef=0.)  # start with lef=0
    a = make_action(throttle=0.3, speed_brake=0.)
    s_new = update(s, a, 0.02)

    # The auto-scheduled LEF takes effect through the filter
    # After 1 step with lef starting at 0, lef ≈ (1-0.9608)*lef_cmd
    # After many steps, lef converges to lef_cmd
    lef_steady = s_new.lef  # approximate

    print(f"  α={np.degrees(alpha_rad):5.1f}°  lef_after_1step={lef_steady:.1f}°  "
          f"(cmd: {'25°' if alpha_rad>0.262 else '15°' if alpha_rad>0.087 else '0°'})")

print()
print("=" * 70)
print("TEST 5: Random action trajectory (100 episodes × 20 steps)")
print("=" * 70)

key = jax.random.PRNGKey(42)
crashes = 0
episode_lengths = []
for ep in range(100):
    key, k_alt, k_vt, k_hdg = jax.random.split(key, 4)
    alt = jax.random.uniform(k_alt, minval=2000., maxval=18000.)
    vt = jax.random.uniform(k_vt, minval=150., maxval=340.)
    yaw = jax.random.uniform(k_hdg, minval=0., maxval=2*jnp.pi)

    s = make_state(alt_m=alt, vt_ms=vt, yaw=yaw, T=20000.)

    for step in range(20):
        key, k_thr, k_el, k_ail, k_rud, k_sb = jax.random.split(key, 6)
        thr = jax.random.uniform(k_thr, minval=0., maxval=1.)
        el = jax.random.uniform(k_el, minval=-1., maxval=1.)
        ail = jax.random.uniform(k_ail, minval=-1., maxval=1.)
        rud = jax.random.uniform(k_rud, minval=-1., maxval=1.)
        sb = jax.random.uniform(k_sb, minval=0., maxval=1.)

        a = make_action(throttle=thr, elevator=el, aileron=ail, rudder=rud, speed_brake=sb)
        s = update(s, a, 0.02)

        # Check altitude bounds (same as env termination)
        if s.altitude < 2000. or s.altitude > 20000.:
            crashes += 1
            episode_lengths.append(step + 1)
            break
        if s.vt < 50. or s.vt > 600.:
            crashes += 1
            episode_lengths.append(step + 1)
            break
    else:
        episode_lengths.append(20)

crash_rate = crashes / 100
mean_len = np.mean(episode_lengths)
print(f"  Crash rate: {crash_rate*100:.0f}%  Mean episode length: {mean_len:.1f} steps")
print(f"  Length distribution: {np.histogram(episode_lengths, bins=[0,5,10,15,21])[0]}")

if crash_rate > 0.2:
    print("  ⚠️  High crash rate with random speed brake!")
    alt_crashes = sum(1 for l in episode_lengths if l < 20)
    print(f"  Altitude/vt crashes: {alt_crashes}/100")

print()
print("=" * 70)
print("TEST 6: Speed brake 0% vs 100% deployment trajectory comparison")
print("=" * 70)

# Fly for 100 steps with and without speed brake
for label, sb_val in [("brake=0 (retracted)", 0.0), ("brake=1 (full 60°)", 1.0)]:
    key = jax.random.PRNGKey(123)
    s = make_state(vt_ms=300., alpha=0.03, T=20000.)
    vts = []
    alts = []
    for step in range(100):
        a = make_action(throttle=0.3, elevator=0., speed_brake=sb_val)
        s = update(s, a, 0.02)
        vts.append(float(s.vt))
        alts.append(float(s.altitude))

    print(f"  {label}:")
    print(f"    vt: {vts[0]:.0f} → {vts[-1]:.0f} m/s (Δ={vts[-1]-vts[0]:+.0f})")
    print(f"    alt: {alts[0]:.0f} → {alts[-1]:.0f} m (Δ={alts[-1]-alts[0]:+.0f})")
    print(f"    sb steady-state: {np.degrees(s.sb):.0f}°")

print()
print("=" * 70)
print("TEST 7: Action decode shape check")
print("=" * 70)

from envs.aeroplanax import AeroPlanaxEnv
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams
)

params = Heading_Pitch_V_TaskParams()
env = AeroPlanaxHeading_Pitch_V_Env(params)

# Simulate what happens during training: 5-element discrete action
test_action_5 = jnp.array([15, 20, 20, 20, 2])  # throttle=15, el=20, ail=20, rud=20, sb=2
decoded = env._decode_discrete_actions(test_action_5)
print(f"  Input (5-elt): {test_action_5}")
print(f"  Decoded: {decoded}  (shape: {decoded.shape})")
assert decoded.shape == (5,), f"Expected (5,), got {decoded.shape}"
assert -1.0 <= decoded[1] <= 1.0, "Elevator out of range!"
assert 0.0 <= decoded[4] <= 1.0, "Speed brake out of range!"

# Simulate backward compat: 4-element action
test_action_4 = jnp.array([15, 20, 20, 20])
decoded_4 = env._decode_discrete_actions(test_action_4)
print(f"  Input (4-elt): {test_action_4}")
print(f"  Decoded: {decoded_4}  (shape: {decoded_4.shape})")
assert decoded_4.shape == (5,), f"Expected (5,), got {decoded_4.shape}"
assert decoded_4[4] == 0.0, "Speed brake should be 0 for 4-elt input!"

print("  ✅ Action decode OK")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key findings to interpret:
- TEST 2: How much does speed brake reduce speed? Expect ~2-5 m/s per step at 300 m/s.
- TEST 3: Speed brake LIFT surge — large Δaz means ballooning/climbing.
- TEST 4: LEF scheduling changes dlef from 1.0 to 0 at high alpha.
- TEST 5: Random action crash rate — high rate suggests speed brake causes altitude violations.
- TEST 6: Long-term effect of sustained speed brake deployment.
""")
