#!/usr/bin/env python3
"""
Quick test to verify JSBSim API works correctly
"""
import sys
sys.path.insert(0, '/home/dqy/aeroplanax/new/20251215最新代码库/Planax')

import jsbsim
import numpy as np
from pathlib import Path

print("Testing JSBSim API...")

# Suppress JSBSim verbose output
import os
old_stdout_fd = os.dup(1)
old_stderr_fd = os.dup(2)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1)
os.dup2(devnull, 2)

try:
    # Initialize
    fdm = jsbsim.FGFDMExec(None)
    jsbsim_root = '/home/dqy/aeroplanax/new/20251215最新代码库/jsbsim/jsbsim'
    fdm.set_root_dir(jsbsim_root)
    fdm.load_model('f16')
    fdm.set_dt(0.02)

    # Set initial conditions
    fdm.set_property_value('ic/h-sl-ft', 15000.0)
    fdm.set_property_value('ic/u-fps', 500.0)
    fdm.set_property_value('ic/v-fps', 0.0)
    fdm.set_property_value('ic/w-fps', 0.0)
    fdm.set_property_value('ic/phi-deg', 0.0)
    fdm.set_property_value('ic/theta-deg', 0.0)
    fdm.set_property_value('ic/psi-deg', 0.0)

    # Reset
    fdm.reset_to_initial_conditions(0)

finally:
    # Restore stdout/stderr
    os.dup2(old_stdout_fd, 1)
    os.dup2(old_stderr_fd, 2)
    os.close(devnull)
    os.close(old_stdout_fd)
    os.close(old_stderr_fd)

print("✓ JSBSim initialized successfully")

# Run a few steps
print("Running 10 simulation steps...")
for i in range(10):
    fdm.set_property_value('fcs/throttle-cmd-norm', 0.5)
    fdm.set_property_value('fcs/elevator-cmd-norm', 0.0)
    fdm.set_property_value('fcs/aileron-cmd-norm', 0.0)
    fdm.set_property_value('fcs/rudder-cmd-norm', 0.0)
    fdm.run()

    if i % 3 == 0:
        alt = fdm.get_property_value('position/h-sl-ft')
        vt = fdm.get_property_value('velocities/vt-fps')
        print(f"  Step {i}: alt={alt:.1f} ft, vt={vt:.1f} ft/s")

print("\n✓ JSBSim API test PASSED!")
print("\nAll JSBSim methods are working correctly:")
print("  - set_root_dir()")
print("  - load_model()")
print("  - set_dt()")
print("  - set_property_value()")
print("  - reset_to_initial_conditions()")
print("  - run()")
print("  - get_property_value()")
