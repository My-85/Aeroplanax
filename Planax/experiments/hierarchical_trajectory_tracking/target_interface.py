"""
Target interface: converts upper-layer action into (H, P, R, Vt) targets.

Three modes:
  Mode 0 (HPv):  Minimal — Δheading, Δpitch, target_vt.  Roll = 0.
  Mode 1 (HPRv): Learned roll — Δheading, Δpitch, Δroll, target_vt.
  Mode 2 (HPRv-H): Heuristic roll — Δheading, Δpitch, target_vt,
                   roll computed from heading error or turn rate.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import jax.numpy as jnp
import numpy as np


@dataclass
class UpperActionConfig:
    mode: str = "HPRv-H"  # "HPv" | "HPRv" | "HPRv-H"

    # Output ranges
    max_delta_heading_deg: float = 90.0
    max_delta_pitch_deg: float = 45.0
    max_delta_roll_deg: float = 90.0
    min_vt: float = 120.0
    max_vt: float = 360.0
    cruise_vt: float = 250.0

    # Rate limits (deg/s)
    max_heading_rate_deg_s: float = 30.0
    max_pitch_rate_deg_s: float = 20.0
    max_roll_rate_deg_s: float = 60.0

    # Smoothing
    smoothing_alpha: float = 0.5  # 0 = no smoothing, 1 = full prior

    # Heuristic roll params (Mode 2)
    heading_error_roll_gain: float = 1.5  # k_roll
    max_heuristic_roll_deg: float = 75.0

    # State for rate limiting / smoothing
    _prev_target_heading: float = field(default=0.0, repr=False)
    _prev_target_pitch: float = field(default=0.0, repr=False)
    _prev_target_roll: float = field(default=0.0, repr=False)
    _prev_target_vt: float = field(default=250.0, repr=False)
    _dt: float = field(default=0.2, repr=False)  # RL step duration


def apply_rate_limit(value, prev, max_rate, dt):
    """Clip value change to max_rate * dt."""
    delta = value - prev
    clipped_delta = np.clip(delta, -max_rate * dt, max_rate * dt)
    return prev + clipped_delta


def apply_smoothing(value, prev, alpha):
    """Exponential moving average."""
    return alpha * prev + (1.0 - alpha) * value


def compute_targets(
    upper_action: np.ndarray,
    current_yaw: float,
    current_pitch: float,
    current_roll: float,
    current_vt: float,
    heading_error: Optional[float] = None,
    config: Optional[UpperActionConfig] = None,
) -> Tuple[float, float, float, float, dict]:
    """
    Convert upper action to target (heading, pitch, roll, vt).

    Args:
        upper_action:  np array of shape (3,) or (4,) depending on mode
        current_yaw:   current aircraft yaw (rad)
        current_pitch: current aircraft pitch (rad)
        current_roll:  current aircraft roll (rad)
        current_vt:    current aircraft airspeed (m/s)
        heading_error: current heading error to waypoint (rad), for Mode 2
        config:        UpperActionConfig

    Returns:
        target_heading, target_pitch, target_roll, target_vt, info_dict
    """
    if config is None:
        config = UpperActionConfig()

    dt = config._dt

    # ── Decode upper action ──
    if config.mode == "HPv":
        delta_heading = float(upper_action[0]) * np.radians(config.max_delta_heading_deg)
        delta_pitch   = float(upper_action[1]) * np.radians(config.max_delta_pitch_deg)
        delta_roll    = 0.0
        vt_norm       = float(upper_action[2])
    elif config.mode == "HPRv":
        delta_heading = float(upper_action[0]) * np.radians(config.max_delta_heading_deg)
        delta_pitch   = float(upper_action[1]) * np.radians(config.max_delta_pitch_deg)
        delta_roll    = float(upper_action[2]) * np.radians(config.max_delta_roll_deg)
        vt_norm       = float(upper_action[3])
    elif config.mode == "HPRv-H":
        delta_heading = float(upper_action[0]) * np.radians(config.max_delta_heading_deg)
        delta_pitch   = float(upper_action[1]) * np.radians(config.max_delta_pitch_deg)
        vt_norm       = float(upper_action[2])
        # Roll computed below from heading error
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    # ── Compute raw targets ──
    raw_target_heading = current_yaw + delta_heading
    raw_target_pitch   = np.clip(current_pitch + delta_pitch,
                                 np.radians(-89), np.radians(89))
    raw_target_vt      = config.min_vt + (vt_norm + 1.0) / 2.0 * (config.max_vt - config.min_vt)
    raw_target_vt      = np.clip(raw_target_vt, config.min_vt, config.max_vt)

    # ── Compute roll target ──
    if config.mode == "HPRv-H":
        if heading_error is not None:
            raw_target_roll = np.clip(
                config.heading_error_roll_gain * heading_error,
                -np.radians(config.max_heuristic_roll_deg),
                np.radians(config.max_heuristic_roll_deg),
            )
        else:
            raw_target_roll = 0.0
    else:
        raw_target_roll = current_roll + delta_roll

    # ── Rate limiting ──
    target_heading = apply_rate_limit(raw_target_heading,
                                      config._prev_target_heading,
                                      np.radians(config.max_heading_rate_deg_s), dt)
    target_pitch = apply_rate_limit(raw_target_pitch,
                                    config._prev_target_pitch,
                                    np.radians(config.max_pitch_rate_deg_s), dt)
    target_roll = apply_rate_limit(raw_target_roll,
                                   config._prev_target_roll,
                                   np.radians(config.max_roll_rate_deg_s), dt)
    target_vt = apply_rate_limit(raw_target_vt,
                                 config._prev_target_vt,
                                 50.0, dt)  # 50 m/s/s thrust rate limit

    # ── Smoothing ──
    if config.smoothing_alpha > 0:
        target_heading = apply_smoothing(target_heading,
                                         config._prev_target_heading,
                                         config.smoothing_alpha)
        target_pitch = apply_smoothing(target_pitch,
                                       config._prev_target_pitch,
                                       config.smoothing_alpha)
        target_roll = apply_smoothing(target_roll,
                                      config._prev_target_roll,
                                      config.smoothing_alpha)

    # ── Update state ──
    config._prev_target_heading = target_heading
    config._prev_target_pitch = target_pitch
    config._prev_target_roll = target_roll
    config._prev_target_vt = target_vt

    info = {
        "raw_target_heading": raw_target_heading,
        "raw_target_pitch": raw_target_pitch,
        "raw_target_roll": raw_target_roll,
        "target_heading": target_heading,
        "target_pitch": target_pitch,
        "target_roll": target_roll,
        "target_vt": target_vt,
        "delta_heading": delta_heading,
        "delta_pitch": delta_pitch,
        "delta_roll": delta_roll if config.mode != "HPv" else 0.0,
    }

    return target_heading, target_pitch, target_roll, target_vt, info


def reset_target_state(config: UpperActionConfig, yaw=0.0, pitch=0.0, roll=0.0, vt=250.0):
    """Reset rate-limit / smoothing state (call at episode start)."""
    config._prev_target_heading = yaw
    config._prev_target_pitch = pitch
    config._prev_target_roll = roll
    config._prev_target_vt = vt
