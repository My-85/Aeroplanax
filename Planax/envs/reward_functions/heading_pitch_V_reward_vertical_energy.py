import jax.numpy as jnp

from ..aeroplanax import TEnvState, TEnvParams, AgentID


def _euler_to_quat_nb(roll: float, pitch: float, yaw: float):
    cr, sr = jnp.cos(0.5 * roll), jnp.sin(0.5 * roll)
    cp, sp = jnp.cos(0.5 * pitch), jnp.sin(0.5 * pitch)
    cy, sy = jnp.cos(0.5 * yaw), jnp.sin(0.5 * yaw)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return jnp.stack([qw, qx, qy, qz], axis=0)


def _quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def _quat_normalize(q):
    return q / (jnp.linalg.norm(q) + 1e-6)


def _quat_geodesic_angle(q_a, q_b):
    q_a = _quat_normalize(q_a)
    q_b = _quat_normalize(q_b)
    cos_half = jnp.abs(jnp.dot(q_a, q_b))
    cos_half = jnp.clip(cos_half, 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _rotate_body_to_ned(q_bn, v_b):
    q_bn = _quat_normalize(q_bn)
    q_nb = _quat_conj(q_bn)
    p = jnp.array([0.0, v_b[0], v_b[1], v_b[2]])
    return _quat_mul(_quat_mul(q_nb, p), q_bn)[1:]


def _unit_vec(v):
    return v / (jnp.linalg.norm(v) + 1e-6)


def _axis_angle_error(a, b):
    a = _unit_vec(a)
    b = _unit_vec(b)
    return jnp.arccos(jnp.clip(jnp.dot(a, b), -1.0, 1.0))


def _wrap_pi(x):
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _get_state_vec(state: TEnvState, name: str, agent_id: AgentID, default: float = 0.0):
    value = getattr(state, name, None)
    if value is None:
        return jnp.array(default, dtype=jnp.float32)
    return jnp.asarray(value[agent_id], dtype=jnp.float32)


def heading_pitch_V_reward_fn_vertical_energy(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
    ) -> float:
    """
    Quaternion attitude tracking reward with vertical-energy shaping.

    This keeps the checkpoint-compatible attitude/speed surface from the
    quaternion baseline, then adds asymmetric low-speed, energy retention,
    climb progress, alpha/beta/G safety, and action smoothness terms.
    """
    vt = jnp.nan_to_num(state.plane_state.vt[agent_id], nan=0.0, posinf=1e6, neginf=0.0)
    alt = jnp.nan_to_num(state.plane_state.altitude[agent_id], nan=0.0, posinf=1e6, neginf=-1e6)

    q_curr = jnp.array([
        state.plane_state.q0[agent_id],
        state.plane_state.q1[agent_id],
        state.plane_state.q2[agent_id],
        state.plane_state.q3[agent_id],
    ])
    q_curr = _quat_normalize(jnp.nan_to_num(q_curr, nan=0.0, posinf=0.0, neginf=0.0))

    yaw_t = state.target_heading[agent_id]
    pitch_t = state.target_pitch[agent_id]
    roll_t = state.target_roll[agent_id]
    q_tgt_bn = _quat_conj(_euler_to_quat_nb(roll_t, pitch_t, yaw_t))

    theta = _quat_geodesic_angle(q_curr, q_tgt_bn)
    theta = jnp.nan_to_num(theta, nan=jnp.pi, posinf=jnp.pi, neginf=jnp.pi)
    theta_scale = jnp.deg2rad(getattr(params, "ve_theta_scale_deg", 7.5))
    r_att = jnp.exp(-((theta / theta_scale) ** 2))

    vt_target = state.target_vt[agent_id]
    dv = jnp.clip(jnp.nan_to_num(vt - vt_target, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
    dv_under = jnp.clip(-dv, 0.0, 1e3)
    dv_over = jnp.clip(dv, 0.0, 1e3)
    low_speed_scale = getattr(params, "ve_speed_under_scale", 18.0)
    high_speed_scale = getattr(params, "ve_speed_over_scale", 35.0)
    r_speed = jnp.exp(-((dv_under / low_speed_scale) ** 2) - ((dv_over / high_speed_scale) ** 2))

    low_speed_threshold = getattr(params, "ve_low_speed_threshold", 180.0)
    strong_low_speed_threshold = getattr(params, "ve_strong_low_speed_threshold", 170.0)
    low_speed_lack = jnp.clip((low_speed_threshold - vt) / 40.0, 0.0, 4.0)
    strong_lack = jnp.clip((strong_low_speed_threshold - vt) / 30.0, 0.0, 4.0)
    r_low_speed = -(0.30 * low_speed_lack * low_speed_lack + 0.70 * strong_lack * strong_lack)

    mode = _get_state_vec(state, "task_mode", agent_id, 0.0)
    vertical_gate = (mode > 0.5).astype(jnp.float32)
    pullup_gate = (((mode > 3.5) & (mode < 5.5))).astype(jnp.float32)
    level_hold_gate = (
        ((mode > 0.5) & (mode < 1.5))
        | ((mode > 5.5) & (mode < 8.5))
        | ((mode < 0.5) & (jnp.abs(pitch_t) < jnp.deg2rad(3.0)))
    ).astype(jnp.float32)
    climb_gate = (state.target_pitch[agent_id] > jnp.deg2rad(2.0)).astype(jnp.float32) * vertical_gate

    g = 9.80665
    energy = 0.5 * vt * vt + g * alt
    prev_energy = _get_state_vec(state, "prev_energy", agent_id, energy)
    start_energy = _get_state_vec(state, "task_start_energy", agent_id, energy)
    energy_drop_step = jnp.clip((prev_energy - energy) / 2500.0, 0.0, 8.0)
    energy_drop_task = jnp.clip((start_energy - energy) / 12000.0, 0.0, 8.0)
    r_energy = -(0.06 * energy_drop_step * energy_drop_step + 0.04 * energy_drop_task * energy_drop_task) * vertical_gate

    start_alt = _get_state_vec(state, "task_start_altitude", agent_id, alt)
    alt_gain = alt - start_alt
    speed_safe = jnp.clip((vt - low_speed_threshold) / 40.0, 0.0, 1.0)
    r_climb = 0.20 * jnp.tanh(alt_gain / 350.0) * speed_safe * climb_gate
    r_climb = jnp.where(pullup_gate > 0.5, r_climb * 0.75, r_climb)

    alt_hold_err = jnp.clip(
        jnp.abs(alt_gain) - getattr(params, "ve_altitude_retention_deadband_m", 80.0),
        0.0,
        1e6,
    )
    alt_hold_scale = getattr(params, "ve_altitude_retention_scale_m", 220.0)
    alt_hold_weight = getattr(params, "ve_altitude_retention_weight", 0.14)
    vz = jnp.abs(jnp.nan_to_num(state.plane_state.vel_z[agent_id], nan=0.0, posinf=1e6, neginf=-1e6))
    vz_weight = getattr(params, "ve_altitude_retention_vz_weight", 0.03)
    r_altitude_hold = -(
        alt_hold_weight * (alt_hold_err / jnp.maximum(alt_hold_scale, 1.0)) ** 2
        + vz_weight * (vz / 80.0) ** 2
    ) * level_hold_gate
    drift_weight = getattr(params, "ve_altitude_drift_weight", 0.04)
    drift_scale = getattr(params, "ve_altitude_drift_scale_m", 500.0)
    r_altitude_drift = -drift_weight * (jnp.abs(alt_gain) / jnp.maximum(drift_scale, 1.0)) ** 2 * level_hold_gate

    alpha = jnp.abs(jnp.nan_to_num(state.plane_state.alpha[agent_id], nan=0.0, posinf=1e3, neginf=-1e3))
    beta = jnp.abs(jnp.nan_to_num(state.plane_state.beta[agent_id], nan=0.0, posinf=1e3, neginf=-1e3))
    alpha_soft = jnp.deg2rad(getattr(params, "ve_alpha_soft_deg", 15.0))
    alpha_hard = jnp.deg2rad(getattr(params, "ve_alpha_hard_deg", 18.0))
    beta_soft = jnp.deg2rad(getattr(params, "ve_beta_soft_deg", 10.0))
    alpha_excess = jnp.clip((alpha - alpha_soft) / jnp.maximum(alpha_hard - alpha_soft, 1e-3), 0.0, 6.0)
    beta_excess = jnp.clip((beta - beta_soft) / jnp.deg2rad(10.0), 0.0, 6.0)
    r_alpha_beta = -(0.08 * alpha_excess * alpha_excess + 0.04 * beta_excess * beta_excess)

    nx_g = jnp.abs(jnp.nan_to_num(state.plane_state.ax[agent_id], nan=0.0, posinf=0.0, neginf=0.0))
    ny_g = jnp.abs(jnp.nan_to_num(state.plane_state.ay[agent_id], nan=0.0, posinf=0.0, neginf=0.0))
    nz_g = jnp.abs(jnp.nan_to_num(state.plane_state.az[agent_id], nan=0.0, posinf=0.0, neginf=0.0))
    load_max = jnp.max(jnp.array([nx_g, ny_g, nz_g]))
    g_soft = getattr(params, "ve_g_soft", 9.0)
    g_hard = getattr(params, "ve_g_hard", 10.0)
    g_excess = jnp.clip((load_max - g_soft) / jnp.maximum(g_hard - g_soft, 1e-3), 0.0, 6.0)
    r_g = -0.05 * g_excess * g_excess

    ctrl = state.control_state
    d_thr = jnp.abs(jnp.nan_to_num(ctrl.throttle[agent_id] - _get_state_vec(state, "prev_throttle", agent_id, 0.0), nan=0.0))
    d_el = jnp.abs(jnp.nan_to_num(ctrl.elevator[agent_id] - _get_state_vec(state, "prev_elevator", agent_id, 0.0), nan=0.0))
    d_ail = jnp.abs(jnp.nan_to_num(ctrl.aileron[agent_id] - _get_state_vec(state, "prev_aileron", agent_id, 0.0), nan=0.0))
    d_rud = jnp.abs(jnp.nan_to_num(ctrl.rudder[agent_id] - _get_state_vec(state, "prev_rudder", agent_id, 0.0), nan=0.0))
    d_sb = jnp.abs(jnp.nan_to_num(ctrl.speed_brake[agent_id] - _get_state_vec(state, "prev_speed_brake", agent_id, 0.0), nan=0.0))
    smooth_gate = (theta < jnp.deg2rad(35.0)).astype(jnp.float32)
    r_smooth = -0.025 * (0.5 * d_thr + d_el + 0.35 * d_ail + 0.35 * d_rud + 0.5 * d_sb) * smooth_gate

    loop_mode_gate = (((mode > 4.5) & (mode < 5.5)) | ((mode > 8.5) & (mode < 9.5))).astype(jnp.float32)
    inverted_gate = (jnp.abs(roll_t) > jnp.deg2rad(90.0)).astype(jnp.float32)
    loop_gate = loop_mode_gate * jnp.maximum(inverted_gate, (theta > jnp.deg2rad(20.0)).astype(jnp.float32))
    body_x = jnp.array([1.0, 0.0, 0.0])
    body_y = jnp.array([0.0, 1.0, 0.0])
    actual_nose = _rotate_body_to_ned(q_curr, body_x)
    target_nose = _rotate_body_to_ned(q_tgt_bn, body_x)
    actual_right = _rotate_body_to_ned(q_curr, body_y)
    target_right = _rotate_body_to_ned(q_tgt_bn, body_y)
    vel_n = jnp.array([
        jnp.nan_to_num(state.plane_state.vel_x[agent_id], nan=0.0, posinf=0.0, neginf=0.0),
        jnp.nan_to_num(state.plane_state.vel_y[agent_id], nan=0.0, posinf=0.0, neginf=0.0),
        jnp.nan_to_num(state.plane_state.vel_z[agent_id], nan=0.0, posinf=0.0, neginf=0.0),
    ])
    nose_tangent_err = _axis_angle_error(actual_nose, target_nose)
    wing_plane_err = _axis_angle_error(actual_right, target_right)
    velocity_tangent_err = _axis_angle_error(vel_n, target_nose)
    nose_velocity_err = _axis_angle_error(actual_nose, vel_n)
    roll_err = jnp.abs(_wrap_pi(jnp.nan_to_num(state.plane_state.roll[agent_id], nan=0.0) - roll_t))
    geom_scale = jnp.deg2rad(25.0)
    r_loop_geom = -loop_gate * (
        getattr(params, "ve_loop_geom_weight", 0.0) * (theta / jnp.maximum(geom_scale, 1e-3)) ** 2
        + getattr(params, "ve_loop_roll_weight", 0.0) * (roll_err / jnp.maximum(geom_scale, 1e-3)) ** 2
        + getattr(params, "ve_loop_nose_tangent_weight", 0.0) * (nose_tangent_err / jnp.maximum(geom_scale, 1e-3)) ** 2
        + getattr(params, "ve_loop_wing_plane_weight", 0.0) * (wing_plane_err / jnp.maximum(geom_scale, 1e-3)) ** 2
        + getattr(params, "ve_loop_velocity_tangent_weight", 0.0) * (velocity_tangent_err / jnp.maximum(geom_scale, 1e-3)) ** 2
        + getattr(params, "ve_loop_nose_velocity_weight", 0.0) * (nose_velocity_err / jnp.maximum(geom_scale, 1e-3)) ** 2
    )
    speed_alpha_gate = jnp.clip((vt - 250.0) / 100.0, 0.0, 2.0)
    r_high_speed_alpha = -loop_gate * getattr(params, "ve_high_speed_alpha_weight", 0.0) * speed_alpha_gate * (
        alpha / jnp.maximum(alpha_soft, 1e-3)
    ) ** 2
    action_sat = (
        jnp.clip(jnp.abs(ctrl.elevator[agent_id]) - 0.85, 0.0, 1.0) ** 2
        + 0.5 * jnp.clip(jnp.abs(ctrl.aileron[agent_id]) - 0.85, 0.0, 1.0) ** 2
        + 0.5 * jnp.clip(jnp.abs(ctrl.rudder[agent_id]) - 0.85, 0.0, 1.0) ** 2
        + 0.4 * jnp.clip(ctrl.speed_brake[agent_id] - 0.80, 0.0, 1.0) ** 2
    )
    r_action_saturation = -loop_gate * getattr(params, "ve_action_saturation_weight", 0.0) * action_sat

    r_alive = 0.02
    reward = (
        0.68 * r_att
        + 0.24 * r_speed
        + r_low_speed * vertical_gate
        + r_energy
        + r_climb
        + r_altitude_hold
        + r_altitude_drift
        + r_alpha_beta
        + r_g
        + r_smooth
        + r_loop_geom
        + r_high_speed_alpha
        + r_action_saturation
        + r_alive
    )
    reward = jnp.clip(jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), -2.0, 1.25)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask
