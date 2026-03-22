#!/usr/bin/env python3
"""
Quaternion convention verification script.

4 test scenarios:
  A: Aligned — aircraft at target attitude → geodesic ≈ 0
  B: Misaligned — aircraft heading 0°, target heading 90° → geodesic ≈ 90°
  C: Speed-only error — attitude matched, speed differs → geodesic ≈ 0
  D: Vector rotation — NED North [1,0,0] rotated to Body for yaw=90° → Body [0,-1,0]

Tests are run against:
  1. Full-domain env helpers (after fix)
  2. Reference env helpers (known correct)
  3. Dynamics-derived DCM (ground truth)
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

# Ensure Planax is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Planax"))

# ======== Import full-domain helpers (under test) ========
from envs.aeroplanax_full_domain_maneuver import (
    _quat_from_euler_nb as fd_quat_from_euler,
    _quat_conj as fd_quat_conj,
    _quat_normalize as fd_quat_normalize,
    _target_q_bn_from_euler as fd_target_q,
    _quat_err_bn as fd_quat_err,
    _rotate_ned_to_body as fd_rotate_ned_to_body,
)

# ======== Import reference helpers (known correct) ========
from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    _quat_from_euler_nb as ref_quat_from_euler,
    _quat_conj as ref_quat_conj,
    _target_q_bn_from_heading_pitch as ref_target_q,
    _quat_err_bn as ref_quat_err,
    _rotate_ned_to_body as ref_rotate_ned_to_body,
)

# ======== Import reward helpers (under test) ========
from envs.reward_functions.full_domain_reward import (
    _quat_from_euler_nb as rw_euler_to_quat,
    _quat_conj as rw_quat_conj,
    _quat_geodesic_angle as rw_geodesic,
    _quat_normalize as rw_normalize,
)

# ======== Import termination helpers (under test) ========
from envs.termination_conditions.unreach_full_domain import (
    _quat_from_euler_nb as tm_euler_to_quat,
    _quat_conj as tm_quat_conj,
    _quat_geodesic_angle as tm_geodesic,
    _quat_normalize as tm_normalize,
)

# ======== Ground truth from dynamics ========
from envs.core.simulators.fighterplane.dynamics import quaternion_to_rpy


def geodesic_angle(q_a, q_b):
    """Geodesic angle between two quaternions."""
    q_a = q_a / (jnp.linalg.norm(q_a) + 1e-9)
    q_b = q_b / (jnp.linalg.norm(q_b) + 1e-9)
    cos_half = jnp.abs(jnp.dot(q_a, q_b))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half)


def dcm_from_euler_ned_to_body(roll, pitch, yaw):
    """Ground-truth DCM: NED -> Body (from standard aerospace convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R = Rx(roll) * Ry(pitch) * Rz(yaw)  (NED->Body)
    R = np.array([
        [cp*cy,                cp*sy,               -sp],
        [sr*sp*cy - cr*sy,     sr*sp*sy + cr*cy,     sr*cp],
        [cr*sp*cy + sr*sy,     cr*sp*sy - sr*cy,     cr*cp],
    ])
    return R


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_pass_fail(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    mark = "[+]" if condition else "[X]"
    extra = f"  ({detail})" if detail else ""
    print(f"  {mark} {name}: {status}{extra}")
    return condition


# ============================================================
all_pass = True

# ---- Scenario A: Aligned (yaw=45°, pitch=10°, roll=20°) ----
print_header("Scenario A: Aligned — aircraft AT target attitude")
yaw_a, pitch_a, roll_a = np.radians(45.0), np.radians(10.0), np.radians(20.0)

# Simulate: aircraft state q = _quat_from_euler_nb (same as init/reset)
q_state_a = fd_quat_from_euler(roll_a, pitch_a, yaw_a)  # q_{BN} from init
print(f"  State q (from init): {np.array(q_state_a)}")

# Full-domain target q (after fix — should be conjugated)
q_tgt_fd_a = fd_target_q(yaw_a, pitch_a, roll_a)
print(f"  FD target q (fixed): {np.array(q_tgt_fd_a)}")

# Reference target q
q_tgt_ref_a = ref_target_q(yaw_a, pitch_a, roll_a)
print(f"  Ref target q:        {np.array(q_tgt_ref_a)}")

# Check: FD target should match reference target
fd_ref_match_a = jnp.allclose(q_tgt_fd_a, q_tgt_ref_a, atol=1e-6) | jnp.allclose(q_tgt_fd_a, -q_tgt_ref_a, atol=1e-6)
all_pass &= print_pass_fail("FD target == Ref target", fd_ref_match_a)

# Geodesic between state and FD target
# NOTE: at init, state q is q_{BN} but target is now q_{NB} = conj(q_{BN}).
# So geodesic won't be 0 at init. But after dynamics runs, state q evolves to q_{NB}.
# For this test, simulate "after dynamics" by using q_{NB} = conj(q_{BN}) as state.
q_state_nb_a = fd_quat_conj(q_state_a)  # Simulate post-dynamics state
theta_fd_a = float(geodesic_angle(q_state_nb_a, q_tgt_fd_a))
theta_ref_a = float(geodesic_angle(q_state_nb_a, q_tgt_ref_a))
print(f"  Geodesic FD  (post-dynamics state vs FD target):  {np.degrees(theta_fd_a):.2f} deg")
print(f"  Geodesic Ref (post-dynamics state vs Ref target): {np.degrees(theta_ref_a):.2f} deg")
all_pass &= print_pass_fail("Geodesic FD ≈ 0° (aligned)", theta_fd_a < np.radians(1.0), f"{np.degrees(theta_fd_a):.2f}°")
all_pass &= print_pass_fail("Geodesic Ref ≈ 0° (aligned)", theta_ref_a < np.radians(1.0), f"{np.degrees(theta_ref_a):.2f}°")

# Also test q_err
q_err_fd_a = fd_quat_err(q_state_nb_a, yaw_a, pitch_a, roll_a)
q_err_ref_a = ref_quat_err(q_state_nb_a, yaw_a, pitch_a, roll_a)
print(f"  q_err FD:  {np.array(q_err_fd_a)}")
print(f"  q_err Ref: {np.array(q_err_ref_a)}")
err_match_a = jnp.allclose(q_err_fd_a, q_err_ref_a, atol=1e-5) | jnp.allclose(q_err_fd_a, -q_err_ref_a, atol=1e-5)
all_pass &= print_pass_fail("q_err FD ≈ q_err Ref", err_match_a)
all_pass &= print_pass_fail("q_err ≈ [1,0,0,0]", float(jnp.abs(q_err_fd_a[0])) > 0.99, f"w={float(q_err_fd_a[0]):.4f}")


# ---- Scenario B: Misaligned (state yaw=0°, target yaw=90°) ----
print_header("Scenario B: Misaligned — 90° yaw difference")
yaw_state_b, pitch_b, roll_b = 0.0, 0.0, 0.0
yaw_target_b = np.radians(90.0)

q_state_b_bn = fd_quat_from_euler(roll_b, pitch_b, yaw_state_b)
q_state_b_nb = fd_quat_conj(q_state_b_bn)  # post-dynamics convention
print(f"  State q_{'{NB}'} (yaw=0°): {np.array(q_state_b_nb)}")

q_tgt_fd_b = fd_target_q(yaw_target_b, pitch_b, roll_b)
q_tgt_ref_b = ref_target_q(yaw_target_b, pitch_b, roll_b)
print(f"  FD target q (yaw=90°):  {np.array(q_tgt_fd_b)}")
print(f"  Ref target q (yaw=90°): {np.array(q_tgt_ref_b)}")

theta_fd_b = float(geodesic_angle(q_state_b_nb, q_tgt_fd_b))
theta_ref_b = float(geodesic_angle(q_state_b_nb, q_tgt_ref_b))
print(f"  Geodesic FD:  {np.degrees(theta_fd_b):.2f} deg")
print(f"  Geodesic Ref: {np.degrees(theta_ref_b):.2f} deg")
all_pass &= print_pass_fail("Geodesic FD ≈ 90°", abs(np.degrees(theta_fd_b) - 90.0) < 2.0, f"{np.degrees(theta_fd_b):.2f}°")
all_pass &= print_pass_fail("Geodesic Ref ≈ 90°", abs(np.degrees(theta_ref_b) - 90.0) < 2.0, f"{np.degrees(theta_ref_b):.2f}°")
all_pass &= print_pass_fail("FD == Ref geodesic", abs(theta_fd_b - theta_ref_b) < 1e-5)

# Reward function geodesic check
q_tgt_rw_b = rw_quat_conj(rw_euler_to_quat(roll_b, pitch_b, yaw_target_b))
theta_rw_b = float(rw_geodesic(q_state_b_nb, q_tgt_rw_b))
print(f"  Reward fn geodesic: {np.degrees(theta_rw_b):.2f} deg")
all_pass &= print_pass_fail("Reward geodesic ≈ 90°", abs(np.degrees(theta_rw_b) - 90.0) < 2.0, f"{np.degrees(theta_rw_b):.2f}°")

# Termination geodesic check
q_tgt_tm_b = tm_quat_conj(tm_euler_to_quat(roll_b, pitch_b, yaw_target_b))
theta_tm_b = float(tm_geodesic(q_state_b_nb, q_tgt_tm_b))
print(f"  Termination fn geodesic: {np.degrees(theta_tm_b):.2f} deg")
all_pass &= print_pass_fail("Termination geodesic ≈ 90°", abs(np.degrees(theta_tm_b) - 90.0) < 2.0, f"{np.degrees(theta_tm_b):.2f}°")


# ---- Scenario C: Speed-only error ----
print_header("Scenario C: Speed-only error — attitude matched, speed differs")
yaw_c, pitch_c, roll_c = np.radians(30.0), np.radians(5.0), np.radians(-10.0)
q_state_c_nb = fd_quat_conj(fd_quat_from_euler(roll_c, pitch_c, yaw_c))
q_tgt_fd_c = fd_target_q(yaw_c, pitch_c, roll_c)

theta_fd_c = float(geodesic_angle(q_state_c_nb, q_tgt_fd_c))
print(f"  Geodesic (same attitude): {np.degrees(theta_fd_c):.4f} deg")
all_pass &= print_pass_fail("Geodesic ≈ 0° (attitude matched)", theta_fd_c < np.radians(0.5), f"{np.degrees(theta_fd_c):.4f}°")
print(f"  (Speed error would be in vt difference, not in quaternion)")


# ---- Scenario D: Vector rotation NED→Body ----
print_header("Scenario D: Vector rotation — NED North → Body for yaw=90°")
yaw_d = np.radians(90.0)
q_nb_d = fd_quat_conj(fd_quat_from_euler(0.0, 0.0, yaw_d))  # q_{NB}
v_ned = jnp.array([1.0, 0.0, 0.0])  # North

# DCM ground truth
R_nb = dcm_from_euler_ned_to_body(0.0, 0.0, float(yaw_d))
v_body_dcm = R_nb @ np.array([1.0, 0.0, 0.0])
print(f"  DCM ground truth:  v_body = {v_body_dcm}")

# Full-domain rotation
v_body_fd = fd_rotate_ned_to_body(q_nb_d, v_ned)
print(f"  FD rotation:       v_body = {np.array(v_body_fd)}")

# Reference rotation
v_body_ref = ref_rotate_ned_to_body(q_nb_d, v_ned)
print(f"  Ref rotation:      v_body = {np.array(v_body_ref)}")

# For yaw=90°: North [1,0,0] in NED → [0,-1,0] in Body
# (right wing points South, so North maps to -y_body)
fd_match_dcm = jnp.allclose(jnp.array(v_body_fd), jnp.array(v_body_dcm), atol=1e-4)
ref_match_dcm = jnp.allclose(jnp.array(v_body_ref), jnp.array(v_body_dcm), atol=1e-4)
all_pass &= print_pass_fail("FD rotation matches DCM", fd_match_dcm, f"expect [0,-1,0], got {np.array(v_body_fd).round(4)}")
all_pass &= print_pass_fail("Ref rotation matches DCM", ref_match_dcm, f"expect [0,-1,0], got {np.array(v_body_ref).round(4)}")

# Also test: East [0,1,0] → should be forward [1,0,0] in body for yaw=90°
v_east = jnp.array([0.0, 1.0, 0.0])
v_body_east = fd_rotate_ned_to_body(q_nb_d, v_east)
v_body_east_dcm = R_nb @ np.array([0.0, 1.0, 0.0])
print(f"  East [0,1,0] → Body: FD={np.array(v_body_east).round(4)}, DCM={v_body_east_dcm.round(4)}")
all_pass &= print_pass_fail("East→Body matches DCM", jnp.allclose(jnp.array(v_body_east), jnp.array(v_body_east_dcm), atol=1e-4))


# ---- Cross-consistency: all three modules produce same target quaternion ----
print_header("Cross-consistency: env, reward, termination target quaternions")
yaw_x, pitch_x, roll_x = np.radians(60.0), np.radians(25.0), np.radians(-45.0)

q_fd = fd_target_q(yaw_x, pitch_x, roll_x)
q_rw = rw_quat_conj(rw_euler_to_quat(roll_x, pitch_x, yaw_x))
q_tm = tm_quat_conj(tm_euler_to_quat(roll_x, pitch_x, yaw_x))
q_ref = ref_target_q(yaw_x, pitch_x, roll_x)

print(f"  FD env:      {np.array(q_fd).round(6)}")
print(f"  Reward:      {np.array(q_rw).round(6)}")
print(f"  Termination: {np.array(q_tm).round(6)}")
print(f"  Reference:   {np.array(q_ref).round(6)}")

def quat_equal(a, b):
    return bool(jnp.allclose(a, b, atol=1e-5) | jnp.allclose(a, -b, atol=1e-5))

all_pass &= print_pass_fail("FD env == Reward", quat_equal(q_fd, q_rw))
all_pass &= print_pass_fail("FD env == Termination", quat_equal(q_fd, q_tm))
all_pass &= print_pass_fail("FD env == Reference", quat_equal(q_fd, q_ref))


# ---- Final verdict ----
print_header("FINAL VERDICT")
if all_pass:
    print("  ALL TESTS PASSED")
else:
    print("  SOME TESTS FAILED — review output above")
    sys.exit(1)
