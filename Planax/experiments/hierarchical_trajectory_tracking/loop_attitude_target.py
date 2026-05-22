"""
Full-attitude loop-plane target generation.

For a vertical loop in the North-Altitude plane (init_yaw=0):
  body_x = loop tangent (varies with theta)
  body_y = East (loop plane normal = right wing direction, constant)
  body_z = body_x x body_y

At theta=0:   heading=N, pitch=0,  roll=0   (normal)
At theta=90:  heading=N, pitch=90, roll=0   (nose up)
At theta=180: heading=S, pitch=0,  roll=180 (INVERTED!)
"""
import numpy as np
from typing import Tuple, List

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover - numpy-only diagnostic use
    jnp = None


def loop_plane_rotation_matrix(theta: float, init_yaw: float = 0.0,
                                loop_direction: int = 1) -> np.ndarray:
    """
    Compute the 3x3 rotation matrix for a point on the vertical loop.

    theta: angle along the loop (0=bottom, pi=top, 2pi=bottom)
    init_yaw: initial heading (rad)
    loop_direction: 1 = right turn (standard pull-up), -1 = left

    Returns R such that v_world = R @ v_body.
    """
    c_yaw, s_yaw = np.cos(init_yaw), np.sin(init_yaw)

    # Forward direction at start (horizontal tangent at bottom)
    f0 = np.array([c_yaw, s_yaw, 0.0])

    # Loop plane normal = right wing direction (perpendicular to loop plane)
    # For vertical loop in (f0, up) plane, normal = cross(up, f0)
    up = np.array([0.0, 0.0, 1.0])  # positive = up in ENU/NED depending on convention
    right = np.cross(up, f0) * loop_direction
    right = right / np.linalg.norm(right)

    # Tangent at angle theta: rotates f0 around the right axis by theta
    ct, st = np.cos(theta), np.sin(theta)
    tangent = ct * f0 + st * np.cross(right, f0)

    # body_x = forward (tangent)
    # body_y = right wing direction (perpendicular to loop plane)
    # body_z = body_x x body_y (belly direction, completes right-handed frame)
    body_x = tangent / np.linalg.norm(tangent)
    body_y = right
    body_z = np.cross(body_x, body_y)

    # R maps body-frame vectors to world-frame: v_world = [body_x|body_y|body_z] @ v_body
    R = np.column_stack([body_x, body_y, body_z])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion [w,x,y,z] to Euler angles (roll, pitch, yaw) in radians.
    Uses Z-Y'-X'' (yaw-pitch-roll) convention matching the project's _quat_from_euler_nb.
    """
    w, x, y, z = q
    # Roll (X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # Yaw (Z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def generate_loop_hpr_sequence(arc_angle_deg: float, n_points: int = 80,
                                 init_yaw: float = 0.0,
                                 loop_direction: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate continuous HPR target sequence for a vertical loop arc.

    Returns (headings, pitches, rolls, quaternions) arrays of shape (n_points,).
    All angles in radians. Quaternions are [w,x,y,z].
    """
    theta_seq = np.linspace(0, np.radians(arc_angle_deg), n_points)
    h_seq = np.zeros(n_points)
    p_seq = np.zeros(n_points)
    r_seq = np.zeros(n_points)
    q_seq = np.zeros((n_points, 4))
    prev_q = None

    for i, theta in enumerate(theta_seq):
        R = loop_plane_rotation_matrix(theta, init_yaw, loop_direction)
        q = rotation_matrix_to_quaternion(R)

        # Ensure quaternion continuity: dot(q_i, q_{i-1}) > 0
        if prev_q is not None and np.dot(q, prev_q) < 0:
            q = -q

        r, p, h = quaternion_to_euler(q)
        q_seq[i] = q
        r_seq[i] = r
        p_seq[i] = p
        h_seq[i] = h
        prev_q = q

    return h_seq, p_seq, r_seq, q_seq


def _require_jax():
    if jnp is None:
        raise RuntimeError("JAX is required for loop-plane target generation inside training/eval.")


def wrap_pi_jax(x):
    _require_jax()
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def loop_plane_hpr_jax(theta, init_yaw=0.0, loop_direction=1.0):
    """
    JAX-compatible loop-plane target using the same geometry as
    `loop_plane_rotation_matrix` / `generate_loop_hpr_sequence`.

    Returns heading, pitch, roll in radians.  For theta > 90deg the target is
    the inverted loop-plane attitude: heading flips by 180deg and roll is 180deg.
    """
    _require_jax()
    theta = jnp.asarray(theta, dtype=jnp.float32)
    init_yaw = jnp.asarray(init_yaw, dtype=jnp.float32)
    theta, init_yaw = jnp.broadcast_arrays(theta, init_yaw)
    theta = jnp.clip(theta, 0.0, 2.0 * jnp.pi)
    post_vertical = theta > (0.5 * jnp.pi)
    heading = wrap_pi_jax(init_yaw + jnp.where(post_vertical, jnp.pi, 0.0))
    pitch = jnp.where(post_vertical, jnp.pi - theta, theta)
    roll_sign = jnp.where(loop_direction >= 0.0, 1.0, -1.0)
    roll = jnp.where(post_vertical, roll_sign * jnp.pi, 0.0)
    return heading, pitch, roll


def loop_plane_tangent_jax(theta, init_yaw=0.0):
    """JAX target tangent/nose vector in the NED convention used by Planax."""
    _require_jax()
    theta = jnp.asarray(theta, dtype=jnp.float32)
    init_yaw = jnp.asarray(init_yaw, dtype=jnp.float32)
    return jnp.stack(
        [
            jnp.cos(theta) * jnp.cos(init_yaw),
            jnp.cos(theta) * jnp.sin(init_yaw),
            -jnp.sin(theta),
        ],
        axis=0,
    )


# ── Test ──
if __name__ == "__main__":
    print("Loop attitude target test for 180° half-loop, init_yaw=0:")
    print(f"{'theta':>8} {'heading':>9} {'pitch':>8} {'roll':>8} {'quat(w,x,y,z)':>45}")
    print("-" * 85)

    for ang in [0, 30, 60, 90, 120, 150, 180]:
        R = loop_plane_rotation_matrix(np.radians(ang), 0.0, 1)
        q = rotation_matrix_to_quaternion(R)
        r, p, h = quaternion_to_euler(q)
        print(f"{ang:8.0f} {np.degrees(h):9.1f} {np.degrees(p):8.1f} {np.degrees(r):8.1f}   [{q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}]")

    # Verify continuity for full 180° sequence
    h, p, r, q = generate_loop_hpr_sequence(180, 80)
    dh = np.abs(np.diff(np.degrees(h)))
    dp = np.abs(np.diff(np.degrees(p)))
    dr = np.abs(np.diff(np.degrees(r)))
    dq = np.array([np.linalg.norm(q[i+1] - q[i]) for i in range(len(q)-1)])

    print(f"\nContinuity check (180°, 80 points):")
    print(f"  max dh: {dh.max():.1f}deg, max dp: {dp.max():.1f}deg, max dr: {dr.max():.1f}deg")
    print(f"  max dq: {dq.max():.4f}")
    print(f"  VERDICT: {'CONTINUOUS' if dh.max()<5 and dp.max()<5 and dq.max()<0.1 else 'DISCONTINUOUS'}")
    print(f"  At theta=180: heading={np.degrees(h[-1]):.1f}, pitch={np.degrees(p[-1]):.1f}, roll={np.degrees(r[-1]):.1f}")
    print(f"    Expected: heading≈180, pitch≈0, roll≈180 (inverted)")
