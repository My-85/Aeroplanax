#!/usr/bin/env python3
"""Debug Planax dynamics in isolation"""
import sys
sys.path.insert(0, '/home/dqy/aeroplanax/new/20251215最新代码库/Planax')

import jax
import jax.numpy as jnp
import numpy as np

from envs.core.simulators.fighterplane.dynamics import (
    FighterPlaneState,
    FighterPlaneControlState,
    update as planax_update,
    quaternion_to_rpy
)

# Initialize state — same as validation script
initial_state = FighterPlaneState(
    north=0.0, east=0.0, altitude=4572.0,
    roll=0.0, pitch=0.0, yaw=0.0,
    vel_x=0.0, vel_y=0.0, vel_z=0.0,
    vt=152.4,
    q0=1.0, q1=0.0, q2=0.0, q3=0.0,
    alpha=0.0, beta=0.0,
    P=0.0, Q=0.0, R=0.0,
    T=0.5 * 0.225 * 76300 / 0.3048,  # ~28160 lbf
    el=0.0, ail=0.0, rud=0.0,
    ax=0.0, ay=0.0, az=0.0
)
print(f"Initial T = {initial_state.T:.1f} lbf")
print(f"Initial vt = {initial_state.vt:.2f} m/s = {initial_state.vt/0.3048:.2f} ft/s")
print(f"Initial alt = {initial_state.altitude:.1f} m = {initial_state.altitude/0.3048:.0f} ft")
print(f"Initial status = {initial_state.status} (ALIVE = 0)")
print(f"is_alive = {initial_state.is_alive}, is_locked = {initial_state.is_locked}")

# Constant zero controls (trim test)
control = FighterPlaneControlState(
    throttle=0.5,
    elevator=0.0,
    aileron=0.0,
    rudder=0.0,
    leading_edge_flap=0.0
)

dt = 0.02
state = initial_state
print("\nRun 100 steps with constant control (throttle=0.5, all surfaces=0):")
print("-" * 100)
print(f"{'step':<5}{'time':<8}{'vt':<10}{'alt(m)':<10}{'roll':<10}{'pitch':<10}{'yaw':<10}{'P':<10}{'Q':<10}{'R':<10}")
for i in range(101):
    if i % 10 == 0:
        roll, pitch, yaw = quaternion_to_rpy(state.q0, state.q1, state.q2, state.q3)
        print(f"{i:<5}{i*dt:<8.2f}{float(state.vt):<10.3f}{float(state.altitude):<10.2f}{float(roll):<10.5f}{float(pitch):<10.5f}{float(yaw):<10.5f}{float(state.P):<10.5f}{float(state.Q):<10.5f}{float(state.R):<10.5f}")
    state = planax_update(state, control, dt)
