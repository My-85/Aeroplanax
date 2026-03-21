# envs/termination_conditions/full_domain_crashed.py
# -*- coding: utf-8 -*-
"""
Relaxed crash detection for full-domain maneuver training.
Bypasses the hardcoded 2500m floor in core/utils.py.
"""
from typing import Tuple
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState, atmos


def full_domain_crashed_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
) -> Tuple[bool, bool]:
    """
    Crash conditions (relaxed for full-domain maneuvers):
      - low altitude:   500m  (from 2500m)
      - overload:       12G   (from 10G)
      - low speed:      Mach 0.005 (~1.7 m/s, from 0.01)
      - qbar stall:     10%   (from 30%)
    Retained:
      - collision (R < 20m)
      - extreme angular rate (>1000 rad/s)
      - high speed (>Mach 3)
      - high altitude (>30km)
    """
    ps: FighterPlaneState = state.plane_state

    # ---- collision ----
    cur_pos = jnp.array([ps.north[agent_id], ps.east[agent_id], ps.altitude[agent_id]]).reshape(-1, 1)
    all_pos = jnp.vstack((ps.north, ps.east, ps.altitude))
    dist = jnp.linalg.norm(cur_pos - all_pos, axis=0)
    dist = dist.at[agent_id].set(jnp.finfo(jnp.float32).max)
    alive = ps.is_alive | ps.is_locked
    dist = jnp.where(alive, dist, jnp.finfo(jnp.float32).max)
    mask_collision = jnp.any(dist < 20.0)

    # ---- extreme angular rate ----
    P, Q, R = ps.P[agent_id], ps.Q[agent_id], ps.R[agent_id]
    mask_extreme = jnp.sqrt(P ** 2 + Q ** 2 + R ** 2) > 1000.0

    # ---- high speed (>Mach 3) ----
    mask_high_speed = (ps.vt[agent_id] / 340.0) > 3.0

    # ---- low speed (Mach 0.005 ~ 1.7 m/s) ----
    mask_low_speed = (ps.vt[agent_id] / 340.0) < 0.005

    # ---- low altitude (500m) ----
    min_alt = getattr(params, "crash_altitude_limit", 500.0)
    mask_low_alt = ps.altitude[agent_id] < min_alt

    # ---- high altitude (30km) ----
    mask_high_alt = ps.altitude[agent_id] > 30000.0

    # ---- overload (12G) ----
    max_g = getattr(params, "nz_hard_limit", 12.0)
    mask_overload = (
        (jnp.abs(ps.ax[agent_id]) >= max_g)
        | (jnp.abs(ps.ay[agent_id]) >= max_g)
        | (jnp.abs(ps.az[agent_id]) >= max_g)
    )

    # ---- qbar stall (10%) ----
    alt_ft = ps.altitude[agent_id] / 0.3048
    vt_ft  = jnp.maximum(ps.vt[agent_id] / 0.3048, 0.1)
    _, qbar, _ = atmos(alt_ft, vt_ft)

    alt_mid_ft = ((getattr(params, "min_altitude", 500.0)
                   + getattr(params, "max_altitude", 20000.0)) * 0.5) / 0.3048
    vt_ref_ft  = getattr(params, "max_vt", 400.0) / 0.3048
    _, qbar_ref, _ = atmos(alt_mid_ft, vt_ref_ft)

    qn = jnp.clip(qbar / (qbar_ref + 1e-6), 0.0, 10.0)
    thresh = getattr(params, "qbar_crash_frac", 0.10)
    mask_qbar = qn < thresh

    # ---- combine ----
    crashed = (
        mask_collision | mask_extreme | mask_high_speed | mask_low_speed
        | mask_low_alt | mask_high_alt | mask_overload | mask_qbar
    )
    success = False
    return crashed, success
