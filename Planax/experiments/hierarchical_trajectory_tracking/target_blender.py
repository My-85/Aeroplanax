"""
TargetBlender: smooths raw (H,P,R,Vt) targets toward current aircraft state.

Handles:
  - Angle wrap (heading via arctan2(sin,cos))
  - Rate limiting
  - Exponential smoothing
  - Reset on waypoint reached
"""

import numpy as np
from typing import Tuple


class TargetBlender:
    def __init__(self, blend_steps: int = 200,
                 max_hdg_rate_deg_s: float = 30.0,
                 max_pitch_rate_deg_s: float = 20.0,
                 max_roll_rate_deg_s: float = 60.0,
                 max_vt_rate: float = 50.0,
                 smoothing_alpha: float = 0.0,
                 dt: float = 0.2):
        self.blend_steps = blend_steps
        self.max_hdg_rate = np.radians(max_hdg_rate_deg_s)
        self.max_pitch_rate = np.radians(max_pitch_rate_deg_s)
        self.max_roll_rate = np.radians(max_roll_rate_deg_s)
        self.max_vt_rate = max_vt_rate
        self.alpha = smoothing_alpha
        self.dt = dt

        self.step_counter = 0
        self._prev_hdg = None
        self._prev_pitch = None
        self._prev_roll = None
        self._prev_vt = None

    def reset(self, current_yaw: float, current_pitch: float,
              current_roll: float, current_vt: float):
        """Call on waypoint reached or episode start."""
        self.step_counter = 0
        self._prev_hdg = current_yaw
        self._prev_pitch = current_pitch
        self._prev_roll = current_roll
        self._prev_vt = current_vt

    def blend(self, raw_hdg: float, raw_pitch: float, raw_roll: float,
              raw_vt: float, yaw: float, pitch: float, roll: float,
              vt: float) -> Tuple[float, float, float, float]:
        """Apply blend + rate limit + smoothing. Returns smoothed targets."""
        self.step_counter += 1

        # Blend factor: 0 at reset, → 1 after blend_steps
        blend = min(1.0, self.step_counter / self.blend_steps)

        # Heading: handle wrap-around
        hdg_err = float(np.arctan2(np.sin(raw_hdg - yaw), np.cos(raw_hdg - yaw)))
        t_hdg = float(np.arctan2(np.sin(yaw + blend * hdg_err), np.cos(yaw + blend * hdg_err)))

        # Pitch: clip to ±89°
        t_pitch = float(pitch + blend * (np.clip(raw_pitch, -np.radians(89), np.radians(89)) - pitch))

        # Roll
        roll_err = float(np.arctan2(np.sin(raw_roll - roll), np.cos(raw_roll - roll)))
        t_roll = float(np.arctan2(np.sin(roll + blend * roll_err), np.cos(roll + blend * roll_err)))

        # Speed
        t_vt = float(vt + blend * (raw_vt - vt))

        # Rate limiting
        if self._prev_hdg is not None:
            t_hdg = self._rate_limit_angle(t_hdg, self._prev_hdg, self.max_hdg_rate)
            t_pitch = self._rate_limit_scalar(t_pitch, self._prev_pitch, self.max_pitch_rate)
            t_roll = self._rate_limit_angle(t_roll, self._prev_roll, self.max_roll_rate)
            t_vt = self._rate_limit_scalar(t_vt, self._prev_vt, self.max_vt_rate)

        # Smoothing
        if self.alpha > 0 and self._prev_hdg is not None:
            t_hdg = self._smooth_angle(t_hdg, self._prev_hdg)
            t_pitch = self._smooth_scalar(t_pitch, self._prev_pitch)
            t_roll = self._smooth_angle(t_roll, self._prev_roll)
            t_vt = self._smooth_scalar(t_vt, self._prev_vt)

        self._prev_hdg = t_hdg
        self._prev_pitch = t_pitch
        self._prev_roll = t_roll
        self._prev_vt = t_vt

        return t_hdg, t_pitch, t_roll, t_vt

    def _rate_limit_angle(self, val, prev, max_rate):
        err = float(np.arctan2(np.sin(val - prev), np.cos(val - prev)))
        clipped = np.clip(err, -max_rate * self.dt, max_rate * self.dt)
        return float(np.arctan2(np.sin(prev + clipped), np.cos(prev + clipped)))

    def _rate_limit_scalar(self, val, prev, max_rate):
        return prev + np.clip(val - prev, -max_rate * self.dt, max_rate * self.dt)

    def _smooth_angle(self, val, prev):
        err = float(np.arctan2(np.sin(val - prev), np.cos(val - prev)))
        return float(np.arctan2(np.sin(prev + self.alpha * err), np.cos(prev + self.alpha * err)))

    def _smooth_scalar(self, val, prev):
        return prev + self.alpha * (val - prev)
