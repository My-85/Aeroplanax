#!/usr/bin/env python3
"""
param_registry.py
=================
Central registry of all tunable parameters for the auto-train loop.

Each entry defines:
  - default: current default value
  - min / max: allowed range (apply_params rejects out-of-range)
  - type: one of "reward_config", "training_config", "func_default", "dataclass_default"
  - file: which source file this parameter lives in
  - description: human-readable explanation for Claude prompt
"""

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_PLANAX_DIR = _PROJECT_ROOT / "Planax"

REWARD_FILE = _PLANAX_DIR / "envs" / "reward_functions" / "full_domain_reward.py"
TRAIN_SCRIPT = _PLANAX_DIR / "train_full_domain_maneuver_v3.py"
TERMINATION_FILE = _PLANAX_DIR / "envs" / "termination_conditions" / "unreach_full_domain.py"
ENV_FILE = _PLANAX_DIR / "envs" / "aeroplanax_full_domain_maneuver.py"


def _r(default, lo, hi, desc):
    """Shorthand for reward_config entry."""
    return {"default": default, "min": lo, "max": hi,
            "type": "reward_config", "file": str(REWARD_FILE), "description": desc}


def _t(default, lo, hi, desc):
    """Shorthand for training_config entry."""
    return {"default": default, "min": lo, "max": hi,
            "type": "training_config", "file": str(TRAIN_SCRIPT), "description": desc}


def _f(default, lo, hi, arg_name, desc):
    """Shorthand for func_default entry (function signature default value)."""
    return {"default": default, "min": lo, "max": hi,
            "type": "func_default", "file": str(TERMINATION_FILE),
            "arg_name": arg_name, "description": desc}


def _d(default, lo, hi, field_name, desc):
    """Shorthand for dataclass_default entry."""
    return {"default": default, "min": lo, "max": hi,
            "type": "dataclass_default", "file": str(ENV_FILE),
            "field_name": field_name, "description": desc}


PARAM_REGISTRY = {
    # ---- reward.* (31 params) — REWARD_CONFIG dict in full_domain_reward.py ----
    "reward.gaussian_scale_coarse_deg":  _r(40.0,  5.0, 120.0, "Gaussian sigma for coarse attitude tracking (degrees)"),
    "reward.gaussian_scale_medium_deg":  _r(12.0,  2.0, 50.0,  "Gaussian sigma for medium attitude tracking (degrees)"),
    "reward.gaussian_scale_fine_deg":    _r(4.0,   0.5, 20.0,  "Gaussian sigma for fine attitude tracking (degrees)"),
    "reward.gaussian_weight_coarse":     _r(0.40,  0.0, 1.0,   "Weight for coarse Gaussian component"),
    "reward.gaussian_weight_medium":     _r(0.40,  0.0, 1.0,   "Weight for medium Gaussian component"),
    "reward.gaussian_weight_fine":       _r(0.20,  0.0, 1.0,   "Weight for fine Gaussian component"),
    "reward.speed_sigma":                _r(30.0,  5.0, 100.0, "Speed tracking Gaussian sigma (m/s)"),
    "reward.track_weight_att":           _r(0.75,  0.0, 1.0,   "Weight of attitude in combined tracking reward"),
    "reward.track_weight_spd":           _r(0.25,  0.0, 1.0,   "Weight of speed in combined tracking reward"),
    "reward.progress_coeff":             _r(0.4,   0.0, 2.0,   "Coefficient for theta-progress reward"),
    "reward.progress_clip_lo":           _r(-0.3, -2.0, 0.0,   "Lower clip for progress reward"),
    "reward.progress_clip_hi":           _r(0.3,   0.0, 2.0,   "Upper clip for progress reward"),
    "reward.on_target_full_bonus":       _r(3.0,   0.0, 10.0,  "Bonus when fully on-target (theta<thresh AND speed<thresh)"),
    "reward.on_target_full_thresh_deg":  _r(10.0,  1.0, 30.0,  "Theta threshold for full on-target bonus (degrees)"),
    "reward.on_target_full_vt_thresh":   _r(15.0,  5.0, 50.0,  "Speed threshold for full on-target bonus (m/s)"),
    "reward.on_target_close_bonus":      _r(1.0,   0.0, 5.0,   "Bonus when close to target (theta<thresh)"),
    "reward.on_target_close_thresh_deg": _r(20.0,  5.0, 45.0,  "Theta threshold for close on-target bonus (degrees)"),
    "reward.on_target_near_bonus":       _r(0.4,   0.0, 3.0,   "Bonus when approaching target (theta<thresh)"),
    "reward.on_target_near_thresh_deg":  _r(35.0, 10.0, 60.0,  "Theta threshold for near on-target bonus (degrees)"),
    "reward.error_penalty_coeff":        _r(-0.15, -1.0, 0.0,  "Penalty coefficient for large attitude error"),
    "reward.error_penalty_thresh_deg":   _r(60.0, 20.0, 120.0, "Theta threshold to start error penalty (degrees)"),
    "reward.error_extreme_coeff":        _r(-0.15, -1.0, 0.0,  "Penalty coefficient for extreme attitude error"),
    "reward.error_extreme_thresh_deg":   _r(90.0, 45.0, 150.0, "Theta threshold to start extreme error penalty (degrees)"),
    "reward.smoothness_omega_thresh":    _r(5.0,   1.0, 20.0,  "Angular velocity threshold before smoothness penalty (rad/s)"),
    "reward.smoothness_coeff":           _r(-0.008, -0.1, 0.0, "Smoothness penalty coefficient"),
    "reward.smoothness_gate_deg":        _r(30.0,  5.0, 60.0,  "Gate smoothness penalty: only apply when theta < this (degrees)"),
    "reward.alive_bonus":                _r(0.005, 0.0, 0.1,   "Per-step alive bonus"),
    "reward.crash_penalty":              _r(-3.0, -10.0, 0.0,  "Penalty on crash/death"),
    "reward.reward_clip_lo":             _r(-5.0, -20.0, 0.0,  "Lower clip bound for total reward"),
    "reward.reward_clip_hi":             _r(5.0,   1.0, 20.0,  "Upper clip bound for total reward"),

    # ---- training.* (12 params) — config dict in train_full_domain_maneuver_v3.py ----
    "training.LR":              _t(2e-4,  1e-6, 1e-2, "Learning rate"),
    "training.ENT_COEF":        _t(2e-3,  1e-5, 0.1,  "Entropy coefficient"),
    "training.CLIP_EPS":        _t(0.2,   0.05, 0.5,  "PPO clipping epsilon"),
    "training.MAX_GRAD_NORM":   _t(2.0,   0.1,  10.0, "Max gradient norm for clipping"),
    "training.GAE_LAMBDA":      _t(0.95,  0.8,  1.0,  "GAE lambda"),
    "training.GAMMA":           _t(0.99,  0.9,  0.999, "Discount factor"),
    "training.NUM_MINIBATCHES": _t(5,     1,    20,   "Number of minibatches per update"),
    "training.UPDATE_EPOCHS":   _t(16,    1,    32,   "PPO update epochs per rollout"),
    "training.VF_COEF":         _t(1.0,   0.1,  5.0,  "Value function loss coefficient"),
    "training.NUM_STEPS":       _t(1000,  64,   4096, "Rollout steps per environment"),
    "training.TOTAL_TIMESTEPS": _t(8e8,   1e7,  5e9,  "Total training timesteps"),
    "training.ANNEAL_LR":       _t(False, False, True, "Whether to anneal learning rate"),

    # ---- termination.* (4 params) — func defaults in unreach_full_domain.py ----
    "termination.min_elapsed_sec":  _f(3.0,   0.5, 10.0,  "min_elapsed_sec",  "Min seconds before success allowed"),
    "termination.theta_tol_deg":    _f(10.0,  2.0, 30.0,  "theta_tol_deg",    "Attitude tolerance for success (degrees)"),
    "termination.vt_tol":           _f(15.0,  5.0, 50.0,  "vt_tol",           "Speed tolerance for success (m/s)"),
    "termination.max_interval_sec": _f(20.0,  5.0, 60.0,  "max_interval_sec", "Timeout duration per episode (seconds)"),

    # ---- curriculum.* (4 params) — dataclass defaults in aeroplanax_full_domain_maneuver.py ----
    "curriculum.advance_threshold":       _d(5,  1, 20,  "curriculum_advance_threshold",  "Base success count to advance level"),
    "curriculum.advance_per_level":       _d(3,  0, 10,  "curriculum_advance_per_level",  "Extra successes needed per level"),
    "curriculum.sustained_on_target":     _d(10, 1, 50,  "sustained_on_target_steps",     "Base consecutive on-target steps required"),
    "curriculum.sustained_per_level":     _d(2,  0, 10,  "sustained_on_target_per_level", "Extra sustained steps per level"),
}


def get_registry_summary() -> str:
    """Format registry as a compact table for Claude API prompt."""
    lines = ["| Parameter | Current | Min | Max | Description |",
             "|-----------|---------|-----|-----|-------------|"]
    for name, info in PARAM_REGISTRY.items():
        lines.append(
            f"| {name} | {info['default']} | {info['min']} | {info['max']} | {info['description']} |"
        )
    return "\n".join(lines)


def validate_changes(changes: dict) -> tuple[dict, list[str]]:
    """Validate parameter changes against registry.
    Returns (valid_changes, errors)."""
    valid = {}
    errors = []
    for key, value in changes.items():
        if key not in PARAM_REGISTRY:
            errors.append(f"Unknown parameter: {key}")
            continue
        info = PARAM_REGISTRY[key]
        # Type coercion
        if isinstance(info["default"], bool):
            if not isinstance(value, bool):
                errors.append(f"{key}: expected bool, got {type(value).__name__}")
                continue
        elif isinstance(info["default"], int) and not isinstance(info["default"], bool):
            try:
                value = int(value)
            except (ValueError, TypeError):
                errors.append(f"{key}: cannot convert {value!r} to int")
                continue
        elif isinstance(info["default"], float):
            try:
                value = float(value)
            except (ValueError, TypeError):
                errors.append(f"{key}: cannot convert {value!r} to float")
                continue
        # Range check (skip for bool)
        if not isinstance(value, bool):
            if value < info["min"] or value > info["max"]:
                errors.append(f"{key}: value {value} out of range [{info['min']}, {info['max']}]")
                continue
        valid[key] = value
    return valid, errors
