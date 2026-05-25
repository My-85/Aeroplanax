"""
Classical Autopilot: Multi-rate (50Hz inner PID + 5Hz outer NLGL).

Architecture:
  NLGL (5 Hz): path → target pitch, roll, vt
  PID  (50 Hz): attitude error → rate + integral → actuator commands

Sign convention (verified on Planax F-16):
  negative elevator → nose-UP    positive → nose-DOWN
  formula: ele = trim - kp*err + kd*Q + ki*∫err

Gain scaling for 50 Hz (dt=0.02 vs 5Hz dt=0.2):
  kp: same (doesn't depend on dt)
  ki: ×10 (integral accumulates 10x slower per step at 50 Hz)
  kd: ÷10 (derivative divided by 10x smaller dt)
"""
import numpy as np
from typing import Tuple


class PIDController:
    def __init__(self, kp, ki=0.0, kd=0.0, out_min=-1.0, out_max=1.0, integrator_max=0.5):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.out_min = out_min; self.out_max = out_max
        self.integrator_max = integrator_max
        self._integral = 0.0; self._prev_error = 0.0; self._first = True

    def reset(self):
        self._integral = 0.0; self._prev_error = 0.0; self._first = True

    def step(self, error, dt):
        if self._first: self._prev_error = error; self._first = False
        p = self.kp * error
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integrator_max, self.integrator_max)
        i = self.ki * self._integral
        d = self.kd * (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error
        raw = p + i + d
        out = np.clip(raw, self.out_min, self.out_max)
        if (raw > self.out_max and error > 0) or (raw < self.out_min and error < 0):
            self._integral -= error * dt
        return out


class ClassicalAutopilot:
    """
    Multi-rate: call nlgl_step() at 5 Hz, pid_step() at 50 Hz.
    """

    def __init__(self, waypoints, L1=800.0, cruise_vt=250.0, dt=0.02, reach_radius=300.0):
        self.waypoints = waypoints; self.n_wp = len(waypoints)
        self.L1 = L1; self.cruise_vt = cruise_vt
        self.dt = dt; self.reach_radius = reach_radius

        self.arc = [0.0]
        for i in range(self.n_wp - 1):
            self.arc.append(self.arc[-1] + float(np.linalg.norm(waypoints[i+1] - waypoints[i])))

        # ── PID gains: auto-scale based on dt ──
        # At 5 Hz (dt=0.2): kp=0.3, ki=0.03, kd=0.15
        # At 50 Hz (dt=0.02): kp same, ki same, kd/10
        kd_scale = 0.15 if dt >= 0.1 else 0.015  # auto-detect 5Hz vs 50Hz
        self.pid_pitch = PIDController(
            kp=0.3, ki=0.03, kd=kd_scale,
            out_min=-1.0, out_max=1.0, integrator_max=0.4)
        self.ele_trim = -0.025

        self.pid_roll = PIDController(
            kp=0.4, ki=0.02, kd=0.1 if dt >= 0.1 else 0.01,
            out_min=-1.0, out_max=1.0, integrator_max=0.2)

        self.pid_speed = PIDController(
            kp=0.015, ki=0.005, kd=0.0,
            out_min=-0.3, out_max=1.0, integrator_max=0.3)
        self.thr_trim = 0.22

        self.pid_yaw = PIDController(
            kp=0.3, ki=0.0, kd=0.05 if dt >= 0.1 else 0.005,
            out_min=-1.0, out_max=1.0, integrator_max=0.1)

        # NLGL targets (updated at 5 Hz, held between updates)
        self.target_pitch = 0.0
        self.target_roll = 0.0
        self.target_vt = 250.0

        self._current_wp_idx = 0; self._wp_reached = 0
        self._prev_elevator = 0.0

    def reset(self):
        self.pid_pitch.reset(); self.pid_roll.reset()
        self.pid_speed.reset(); self.pid_yaw.reset()
        self.target_pitch = 0.0; self.target_roll = 0.0; self.target_vt = 250.0
        self._current_wp_idx = 0; self._wp_reached = 0
        self._prev_elevator = 0.0

    # ── NLGL (call at 5 Hz) ────────────────────────────────

    def _closest_arc_position(self, north, east, alt):
        p = np.array([north, east, alt])
        best_dist = float('inf'); best_arc = 0.0
        best_idx = max(0, self._current_wp_idx - 2)
        for i in range(best_idx, min(self.n_wp - 1, self._current_wp_idx + 10)):
            a = self.waypoints[i]; b = self.waypoints[min(i + 1, self.n_wp - 1)]
            seg = b - a; l2 = float(np.dot(seg, seg))
            t = np.clip(float(np.dot(p - a, seg)) / max(l2, 1e-9), 0.0, 1.0)
            d = float(np.linalg.norm(p - a - t * seg))
            if d < best_dist:
                best_dist = d
                best_arc = self.arc[i] + t * (self.arc[min(i+1, self.n_wp-1)] - self.arc[i])
                best_idx = i
        return best_arc, best_idx

    def _lookahead_point(self, arc_pos):
        la_arc = min(arc_pos + self.L1, self.arc[-1])
        for i in range(self.n_wp - 1):
            if self.arc[i] <= la_arc <= self.arc[i + 1]:
                la_idx = i; la_t = (la_arc - self.arc[i]) / max(self.arc[i+1] - self.arc[i], 1e-9)
                break
        else: la_idx = self.n_wp - 2; la_t = 1.0
        a = self.waypoints[la_idx]; b = self.waypoints[min(la_idx + 1, self.n_wp - 1)]
        return a + la_t * (b - a)

    def nlgl_step(self, north, east, alt, yaw, pitch, vt):
        """Call at 5 Hz to refresh guidance targets."""
        arc_pos, seg_idx = self._closest_arc_position(north, east, alt)
        self._current_wp_idx = seg_idx

        dist_end = np.sqrt((north - self.waypoints[-1][0])**2 +
                           (east - self.waypoints[-1][1])**2 +
                           (alt - self.waypoints[-1][2])**2)
        if dist_end < self.reach_radius:
            self._wp_reached += 1

        la_pt = self._lookahead_point(arc_pos)
        d_n = la_pt[0] - north; d_e = la_pt[1] - east; d_a = la_pt[2] - alt
        h_dist = np.sqrt(d_n**2 + d_e**2) + 1e-9
        target_heading = float(np.arctan2(d_e, d_n))
        tg_pitch_raw = float(np.arctan2(d_a, h_dist))
        alt_error = la_pt[2] - alt
        alt_corr = np.clip(0.01 * alt_error / max(self.L1, 100.0), -0.03, 0.03)
        self.target_pitch = np.clip(tg_pitch_raw + alt_corr, np.radians(-15), np.radians(70))

        heading_err = float(np.arctan2(np.sin(target_heading - yaw),
                                       np.cos(target_heading - yaw)))
        self.target_roll = np.clip(1.0 * heading_err, np.radians(-50), np.radians(50))
        self.target_vt = self.cruise_vt

    # ── PID (call at 50 Hz) ─────────────────────────────────

    def pid_step(self, pitch, roll, yaw, vt, q, p, r, alpha, beta) -> np.ndarray:
        """Call at 50 Hz. Returns [thr, ele, ail, rud] in [-1,1]."""

        # Pitch: ele = trim - PID(err) + FF
        pitch_err = float(np.arctan2(np.sin(self.target_pitch - pitch),
                                     np.cos(self.target_pitch - pitch)))
        ele_pid = self.pid_pitch.step(pitch_err, self.dt)
        gravity_ff = -0.03 * np.sin(max(self.target_pitch, 0.0))
        elevator = self.ele_trim - ele_pid + gravity_ff
        max_slew = 0.15  # per 0.02s = 7.5/s
        elevator = np.clip(elevator, self._prev_elevator - max_slew,
                           self._prev_elevator + max_slew)
        self._prev_elevator = elevator
        elevator = np.clip(elevator, -1.0, 1.0)

        # Roll
        roll_err = float(np.arctan2(np.sin(self.target_roll - roll),
                                    np.cos(self.target_roll - roll)))
        aileron = -self.pid_roll.step(roll_err, self.dt)
        aileron = np.clip(aileron, -1.0, 1.0)

        # Yaw damper
        rudder = -self.pid_yaw.kp * beta - self.pid_yaw.kd * r
        rudder = np.clip(rudder, -1.0, 1.0)

        # Speed
        speed_error = self.target_vt - vt
        throttle = self.pid_speed.step(speed_error, self.dt)
        throttle = np.clip(throttle + self.thr_trim, -1.0, 1.0)

        return np.array([throttle, elevator, aileron, rudder])

    def is_done(self):
        return self._wp_reached > 0 and self._current_wp_idx >= self.n_wp - 2
