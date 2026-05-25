import json
from typing import Any, Dict, Optional

import numpy as np


def scalar(x: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def bool_scalar(x: Any, default: bool = False) -> bool:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return bool(arr.reshape(-1)[0])
    except Exception:
        return default


def int_scalar(x: Any, default: int = 0) -> int:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


def finite_all(values) -> bool:
    try:
        arr = np.asarray(values, dtype=np.float64)
        return bool(np.all(np.isfinite(arr)))
    except Exception:
        return False


def done_flag_from_info(info: Optional[Dict[str, Any]], agent_name: str = "agent_0") -> bool:
    if not info:
        return False
    dones = info.get("terminal_dones_before_reset", {})
    if isinstance(dones, dict):
        if agent_name in dones:
            return bool_scalar(dones[agent_name])
        if "__all__" in dones:
            return bool_scalar(dones["__all__"])
    return bool_scalar(info.get("terminal_env_done_before_reset", False))


def terminal_state_from_info(info: Optional[Dict[str, Any]], fallback_state: Any = None) -> Any:
    if info and "terminal_state_before_reset" in info:
        return info["terminal_state_before_reset"]
    return fallback_state


def terminal_flags(state: Any, params: Any = None, agent_id: int = 0) -> Dict[str, Any]:
    ps = state.plane_state
    vt = scalar(ps.vt[agent_id])
    altitude = scalar(ps.altitude[agent_id])
    alpha_deg = float(np.degrees(scalar(ps.alpha[agent_id])))
    beta_deg = float(np.degrees(scalar(ps.beta[agent_id])))
    ax = scalar(ps.ax[agent_id])
    ay = scalar(ps.ay[agent_id])
    az = scalar(ps.az[agent_id])
    p = scalar(ps.P[agent_id])
    q = scalar(ps.Q[agent_id])
    r = scalar(ps.R[agent_id])
    load_max = max(abs(ax), abs(ay), abs(az))
    time_step = scalar(getattr(state, "time", 0.0))
    if params is not None:
        timeout_steps = 400.0 * float(params.sim_freq) / float(params.agent_interaction_steps)
    else:
        timeout_steps = 2000.0
    numbers = [
        vt,
        altitude,
        alpha_deg,
        beta_deg,
        ax,
        ay,
        az,
        p,
        q,
        r,
        scalar(ps.roll[agent_id]),
        scalar(ps.pitch[agent_id]),
        scalar(ps.yaw[agent_id]),
    ]
    return {
        "state_time": time_step,
        "timeout_steps": timeout_steps,
        "plane_status": int_scalar(ps.status[agent_id]),
        "plane_is_crashed": bool_scalar(ps.is_crashed[agent_id]),
        "plane_is_alive": bool_scalar(ps.is_alive[agent_id]),
        "plane_is_success": bool_scalar(ps.is_success[agent_id]),
        "nan_or_invalid": not finite_all(numbers),
        "overload": load_max >= 10.0,
        "load_max_component": load_max,
        "low_speed": vt / 340.0 < 0.01,
        "high_speed": vt / 340.0 > 3.0,
        "altitude_limit": altitude < 2500.0 or altitude > 1.0e9,
        "low_altitude": altitude < 2500.0,
        "high_altitude": altitude > 1.0e9,
        "extreme_rotation": float(np.sqrt(p * p + q * q + r * r)) > 1000.0,
        "timeout": time_step >= timeout_steps,
        "alpha_violation": alpha_deg < -20.0 or alpha_deg > 45.0,
        "beta_violation": abs(beta_deg) > 45.0,
        "vt": vt,
        "altitude": altitude,
        "alpha_deg": alpha_deg,
        "beta_deg": beta_deg,
        "ax": ax,
        "ay": ay,
        "az": az,
        "P": p,
        "Q": q,
        "R": r,
    }


def classify_terminal_reason(
    state: Any,
    params: Any = None,
    done_flag: bool = False,
    planner_completed: bool = False,
    agent_id: int = 0,
) -> Dict[str, Any]:
    flags = terminal_flags(state, params=params, agent_id=agent_id)
    reason = "unknown_done" if done_flag else "running"
    if planner_completed and not done_flag:
        reason = "success"
    elif flags["nan_or_invalid"]:
        reason = "nan_or_invalid"
    elif flags["overload"]:
        reason = "overload"
    elif flags["low_speed"]:
        reason = "low_speed"
    elif flags["high_speed"]:
        reason = "crash"
    elif flags["altitude_limit"]:
        reason = "altitude_limit"
    elif flags["plane_is_crashed"] or flags["extreme_rotation"]:
        reason = "crash"
    elif flags["timeout"]:
        reason = "timeout"
    elif flags["alpha_violation"] and done_flag:
        reason = "alpha_violation"
    elif flags["beta_violation"] and done_flag:
        reason = "beta_violation"
    elif done_flag:
        reason = "unknown_done"
    return {
        "terminal_reason_classified": reason,
        "terminal_reason_raw": json.dumps(flags, sort_keys=True),
        **flags,
    }
