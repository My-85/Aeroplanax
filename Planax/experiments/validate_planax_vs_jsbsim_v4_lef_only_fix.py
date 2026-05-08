#!/usr/bin/env python3
"""
Trajectory-level Fidelity Validation: Planax vs JSBSim (v2)

Improvements over v1:
1. Use JSBSim's trim algorithm to compute the actual trim state
2. Initialize BOTH simulators with the same trim state (alpha, theta, throttle, elevator)
3. Shorter simulation duration (10s) to minimize integrator drift accumulation
4. Test scenarios still: trim, elevator doublet, coordinated turn, sinusoidal

Author: Generated for IEEE RA-L manuscript
Date: 2026-05-07
"""

import os
import sys
import numpy as np
from pathlib import Path
import csv
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Planax dynamics
from envs.core.simulators.fighterplane.dynamics import (
    FighterPlaneState,
    FighterPlaneControlState,
    update as planax_env_update,
    quaternion_to_rpy
)

# Try to import JSBSim
try:
    import jsbsim
    JSBSIM_AVAILABLE = True
except ImportError:
    JSBSIM_AVAILABLE = False
    print("ERROR: JSBSim not available. This script requires JSBSim.")
    sys.exit(1)


# Helper to suppress JSBSim verbose output
class SuppressOutput:
    def __enter__(self):
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, 1)
        os.dup2(self.devnull, 2)
        return self

    def __exit__(self, *args):
        os.dup2(self.old_stdout, 1)
        os.dup2(self.old_stderr, 2)
        os.close(self.devnull)
        os.close(self.old_stdout)
        os.close(self.old_stderr)


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (rad) to the quaternion that Planax dynamics expects
    to be *stored* in state (q_{Body}^{NED}).

    Planax dynamics stores q_Body_to_NED, and retrieves Euler angles via
        quaternion_to_rpy(q0, -q1, -q2, -q3)
    which expects q_NED_to_Body. So the stored quaternion is the conjugate
    of q_NED_to_Body, i.e. (w, -x, -y, -z).
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    # q_NED_to_Body (standard Z-Y-X Tait-Bryan)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    # Store q_Body_to_NED = conjugate of q_NED_to_Body
    return qw, -qx, -qy, -qz


def compute_jsbsim_trim(altitude_ft=15000.0, vt_fps=500.0, dt=0.02):
    """Use JSBSim's built-in trim algorithm to compute trim state.
    Returns dict with trim variables (in SI units)."""
    with SuppressOutput():
        fdm = jsbsim.FGFDMExec(None)
        fdm.set_root_dir(str(Path(__file__).parent.parent.parent / "jsbsim" / "jsbsim"))
        fdm.load_model('f16')
        fdm.set_dt(dt)
        fdm.set_property_value('ic/h-sl-ft', altitude_ft)
        fdm.set_property_value('ic/u-fps', vt_fps)
        fdm.set_property_value('ic/v-fps', 0.0)
        fdm.set_property_value('ic/w-fps', 0.0)
        fdm.set_property_value('ic/phi-deg', 0.0)
        fdm.set_property_value('ic/theta-deg', 0.0)
        fdm.set_property_value('ic/psi-deg', 0.0)
        fdm.reset_to_initial_conditions(0)
        fdm.set_property_value('simulation/do_simple_trim', 1)

    trim = {
        'altitude_m': fdm.get_property_value('position/h-sl-meters'),
        'altitude_ft': fdm.get_property_value('position/h-sl-ft'),
        'vt_ms': fdm.get_property_value('velocities/vt-fps') * 0.3048,
        'vt_fps': fdm.get_property_value('velocities/vt-fps'),
        'alpha_rad': fdm.get_property_value('aero/alpha-rad'),
        'beta_rad': fdm.get_property_value('aero/beta-rad'),
        'theta_rad': fdm.get_property_value('attitude/theta-rad'),
        'phi_rad': fdm.get_property_value('attitude/phi-rad'),
        'psi_rad': 0.0,
        'throttle_norm': fdm.get_property_value('fcs/throttle-cmd-norm'),
        'elevator_norm': fdm.get_property_value('fcs/elevator-cmd-norm'),
        'aileron_norm': fdm.get_property_value('fcs/aileron-cmd-norm'),
        'rudder_norm': fdm.get_property_value('fcs/rudder-cmd-norm'),
        'elevator_pos_deg': fdm.get_property_value('fcs/elevator-pos-deg'),
    }
    return trim


def compute_planax_trim(trim_jsb: Dict, dt=0.02):
    """Compute Planax's own trim point by numerically solving for
    (alpha, elevator_deg, throttle_norm) such that body-axis accelerations
    are all ≈ 0.

    Because Planax's aero model has slight differences from JSBSim's,
    the two simulators have slightly different trim solutions.
    """
    from scipy.optimize import least_squares
    import jax.numpy as jnp
    from envs.core.simulators.fighterplane.dynamics import nlplant

    alt_ft = trim_jsb['altitude_ft']
    vt_fps = trim_jsb['vt_fps']

    def residual(params):
        alpha_rad, elevator_deg, throttle_norm = params
        theta_rad = alpha_rad  # level flight
        q0 = np.cos(theta_rad / 2.0)
        q2 = -np.sin(theta_rad / 2.0)
        T_lbf = throttle_norm * 0.225 * 76300 / 0.3048

        x = jnp.array([
            0.0, 0.0, alt_ft,
            0.0, theta_rad, 0.0,
            vt_fps, alpha_rad, 0.0,
            0.0, 0.0, 0.0,
            q0, 0.0, q2, 0.0,
        ])
        u = jnp.array([T_lbf, elevator_deg, 0.0, 0.0, 0.0])
        xdot = nlplant(jnp.hstack((x, u)))

        # Balanced residuals (equal weights); we have 3 knobs and want 3 eqs:
        # dvt/dt ≈ 0  (thrust = drag)
        # dalpha/dt ≈ 0  (lift = weight, level flight)
        # dQ/dt ≈ 0  (pitch moment balance)
        return np.array([
            float(xdot[6]),       # dvt/dt (ft/s²) — scale ~1
            float(xdot[7]) * 500, # dalpha/dt (rad/s) → scale ×500 so ~ ft/s² magnitude
            float(xdot[10]) * 500 # dQ/dt → same scaling
        ])

    # Initial guess: try JSBSim value first, then a better starting point
    x0_candidates = [
        np.array([trim_jsb['alpha_rad'], trim_jsb['elevator_pos_deg'], trim_jsb['throttle_norm']]),
        np.array([np.radians(4.0), 2.0, 0.01]),   # typical Planax trim guess
        np.array([np.radians(4.0), 0.0, 0.3]),
        np.array([np.radians(2.0), 1.0, 0.2]),
    ]

    # Bounds
    lb = np.array([np.radians(-5.0), -25.0, 0.0])
    ub = np.array([np.radians(20.0), 25.0, 1.0])

    best_res = None
    best_norm = np.inf
    for x0 in x0_candidates:
        try:
            res = least_squares(residual, x0, bounds=(lb, ub),
                                method='trf', xtol=1e-12, ftol=1e-12,
                                max_nfev=5000)
            rn = np.linalg.norm(res.fun)
            if rn < best_norm:
                best_norm = rn
                best_res = res
        except Exception:
            continue

    res = best_res
    alpha_rad, el_deg, th_norm = res.x
    final_residual = np.linalg.norm(res.fun)
    print(f"  Planax trim search: success={res.success}, residual_norm={final_residual:.4e}")
    print(f"    alpha={np.degrees(alpha_rad):.3f} deg (JSBSim {np.degrees(trim_jsb['alpha_rad']):.3f})")
    print(f"    elevator={el_deg:.3f} deg (JSBSim {trim_jsb['elevator_pos_deg']:.3f})")
    print(f"    throttle={th_norm:.4f} (JSBSim {trim_jsb['throttle_norm']:.4f})")
    print(f"    residuals: dvt={res.fun[0]:.4f}, dalpha×500={res.fun[1]:.4f}, dQ×500={res.fun[2]:.4f}")

    trim_planax = dict(trim_jsb)
    trim_planax['alpha_rad'] = float(alpha_rad)
    trim_planax['theta_rad'] = float(alpha_rad)
    trim_planax['elevator_pos_deg'] = float(el_deg)
    trim_planax['throttle_norm'] = float(th_norm)
    trim_planax['elevator_norm'] = float(el_deg) / 45.0
    return trim_planax



    """Use JSBSim's built-in trim algorithm to compute trim state.

    Returns dict with trim variables (in SI units)."""
    with SuppressOutput():
        fdm = jsbsim.FGFDMExec(None)
        fdm.set_root_dir(str(Path(__file__).parent.parent.parent / "jsbsim" / "jsbsim"))
        fdm.load_model('f16')
        fdm.set_dt(dt)
        fdm.set_property_value('ic/h-sl-ft', altitude_ft)
        fdm.set_property_value('ic/u-fps', vt_fps)
        fdm.set_property_value('ic/v-fps', 0.0)
        fdm.set_property_value('ic/w-fps', 0.0)
        fdm.set_property_value('ic/phi-deg', 0.0)
        fdm.set_property_value('ic/theta-deg', 0.0)
        fdm.set_property_value('ic/psi-deg', 0.0)
        fdm.reset_to_initial_conditions(0)
        # Run JSBSim simple trim (mode 1 = wings-level longitudinal trim)
        fdm.set_property_value('simulation/do_simple_trim', 1)

    trim = {
        'altitude_m': fdm.get_property_value('position/h-sl-meters'),
        'altitude_ft': fdm.get_property_value('position/h-sl-ft'),
        'vt_ms': fdm.get_property_value('velocities/vt-fps') * 0.3048,
        'vt_fps': fdm.get_property_value('velocities/vt-fps'),
        'alpha_rad': fdm.get_property_value('aero/alpha-rad'),
        'beta_rad': fdm.get_property_value('aero/beta-rad'),
        'theta_rad': fdm.get_property_value('attitude/theta-rad'),
        'phi_rad': fdm.get_property_value('attitude/phi-rad'),
        'psi_rad': 0.0,
        'throttle_norm': fdm.get_property_value('fcs/throttle-cmd-norm'),
        'elevator_norm': fdm.get_property_value('fcs/elevator-cmd-norm'),
        'aileron_norm': fdm.get_property_value('fcs/aileron-cmd-norm'),
        'rudder_norm': fdm.get_property_value('fcs/rudder-cmd-norm'),
        'elevator_pos_deg': fdm.get_property_value('fcs/elevator-pos-deg'),
        'thrust_lbs': fdm.get_property_value('propulsion/total-fuel-lbs') * 0 +
                      fdm.get_property_value('forces/fbx-prop-lbs'),
    }
    return trim


# ============================================================
# Planax Simulator Wrapper
# ============================================================
class PlanaxSimulator:
    def __init__(self, dt=0.02):
        self.dt = dt

    def reset_from_trim(self, trim: Dict) -> FighterPlaneState:
        """Initialize Planax state from JSBSim trim solution."""
        # Trim quaternion: roll=0, pitch=theta_trim, yaw=0
        q0, q1, q2, q3 = euler_to_quaternion(0.0, trim['theta_rad'], 0.0)

        # Initial T (lbf): use JSBSim's actual prop force as starting thrust
        # Use the throttle*max calculation that Planax dynamics expects internally
        T_init = trim['throttle_norm'] * 0.225 * 76300 / 0.3048

        # Initial elevator deflection (degrees) — from JSBSim trim
        el_init = trim['elevator_pos_deg']

        state = FighterPlaneState(
            north=0.0, east=0.0,
            altitude=trim['altitude_m'],
            roll=0.0, pitch=trim['theta_rad'], yaw=0.0,
            vel_x=trim['vt_ms'] * np.cos(trim['theta_rad']) * np.cos(trim['alpha_rad']),
            vel_y=0.0,
            vel_z=trim['vt_ms'] * np.sin(trim['theta_rad'] - trim['alpha_rad']),
            vt=trim['vt_ms'],
            q0=q0, q1=q1, q2=q2, q3=q3,
            alpha=trim['alpha_rad'],
            beta=trim['beta_rad'],
            P=0.0, Q=0.0, R=0.0,
            T=T_init,
            el=el_init, ail=0.0, rud=0.0,
            ax=0.0, ay=0.0, az=0.0
        )
        return state

    def step(self, state: FighterPlaneState, control: Dict) -> FighterPlaneState:
        action = FighterPlaneControlState(
            throttle=control['throttle'],
            elevator=control['elevator'],
            aileron=control['aileron'],
            rudder=control['rudder'],
            leading_edge_flap=0.0
        )
        return planax_env_update(state, action, self.dt)

    def get_state_dict(self, state: FighterPlaneState) -> Dict:
        # Match Planax dynamics: call quaternion_to_rpy(q0, -q1, -q2, -q3)
        roll, pitch, yaw = quaternion_to_rpy(state.q0, -state.q1, -state.q2, -state.q3)
        return {
            'time': 0.0,
            'north': float(state.north),
            'east': float(state.east),
            'altitude': float(state.altitude),
            'roll': float(roll),
            'pitch': float(pitch),
            'yaw': float(yaw),
            'vt': float(state.vt),
            'alpha': float(state.alpha),
            'beta': float(state.beta),
            'P': float(state.P),
            'Q': float(state.Q),
            'R': float(state.R),
            'q0': float(state.q0), 'q1': float(state.q1),
            'q2': float(state.q2), 'q3': float(state.q3),
        }


# ============================================================
# JSBSim Simulator Wrapper
# ============================================================
class JSBSimSimulator:
    def __init__(self, dt=0.02):
        self.dt = dt
        with SuppressOutput():
            self.fdm = jsbsim.FGFDMExec(None)
            jsbsim_root = Path(__file__).parent.parent.parent / "jsbsim" / "jsbsim"
            self.fdm.set_root_dir(str(jsbsim_root))
            self.fdm.load_model('f16')
            self.fdm.set_dt(dt)

    def reset_from_trim(self, trim: Dict):
        """Re-trim JSBSim using the same conditions used to compute the trim."""
        with SuppressOutput():
            self.fdm.set_property_value('ic/h-sl-ft', trim['altitude_ft'])
            self.fdm.set_property_value('ic/u-fps', trim['vt_fps'])
            self.fdm.set_property_value('ic/v-fps', 0.0)
            self.fdm.set_property_value('ic/w-fps', 0.0)
            self.fdm.set_property_value('ic/phi-deg', 0.0)
            self.fdm.set_property_value('ic/theta-deg', 0.0)
            self.fdm.set_property_value('ic/psi-deg', 0.0)
            self.fdm.reset_to_initial_conditions(0)
            self.fdm.set_property_value('simulation/do_simple_trim', 1)

    def step(self, control: Dict):
        # control['*'] are *delta* commands relative to trim, so add trim baseline
        self.fdm.set_property_value('fcs/throttle-cmd-norm', control['throttle'])
        self.fdm.set_property_value('fcs/elevator-cmd-norm', control['elevator'])
        self.fdm.set_property_value('fcs/aileron-cmd-norm', control['aileron'])
        self.fdm.set_property_value('fcs/rudder-cmd-norm', control['rudder'])
        self.fdm.run()

    def get_state_dict(self) -> Dict:
        return {
            'time': self.fdm.get_property_value('simulation/sim-time-sec'),
            'north': self.fdm.get_property_value('position/distance-from-start-lat-mt'),
            'east': self.fdm.get_property_value('position/distance-from-start-lon-mt'),
            'altitude': self.fdm.get_property_value('position/h-sl-meters'),
            'roll': self.fdm.get_property_value('attitude/phi-rad'),
            'pitch': self.fdm.get_property_value('attitude/theta-rad'),
            'yaw': self.fdm.get_property_value('attitude/psi-rad'),
            'vt': self.fdm.get_property_value('velocities/vt-fps') * 0.3048,
            'alpha': self.fdm.get_property_value('aero/alpha-rad'),
            'beta': self.fdm.get_property_value('aero/beta-rad'),
            'P': self.fdm.get_property_value('velocities/p-rad_sec'),
            'Q': self.fdm.get_property_value('velocities/q-rad_sec'),
            'R': self.fdm.get_property_value('velocities/r-rad_sec'),
        }


# ============================================================
# Control sequence generation
# ============================================================
def generate_control_sequence(scenario: str, duration: float, dt: float, trim: Dict) -> List[Dict]:
    """Generate control sequences. Throttle/elevator are absolute normalized commands.
    Trim baseline is included so trim scenario uses trim throttle exactly."""
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)
    controls = []

    # Trim baseline values
    th_trim = trim['throttle_norm']
    el_trim = trim['elevator_norm']

    if scenario == 'trim':
        for _ in range(n_steps):
            controls.append({
                'throttle': th_trim,
                'elevator': el_trim,
                'aileron': 0.0,
                'rudder': 0.0
            })

    elif scenario == 'elevator_doublet':
        # Hold trim for 2s, then doublet ±0.1, return to trim
        for ti in t:
            if ti < 2.0:
                el = el_trim
            elif ti < 3.0:
                el = el_trim + 0.1
            elif ti < 4.0:
                el = el_trim - 0.1
            else:
                el = el_trim
            controls.append({
                'throttle': th_trim,
                'elevator': el,
                'aileron': 0.0,
                'rudder': 0.0
            })

    elif scenario == 'coordinated_turn':
        # Hold trim for 2s, then small aileron + rudder for 5s
        for ti in t:
            if ti < 2.0:
                ail, rud = 0.0, 0.0
            elif ti < 7.0:
                ail, rud = 0.1, 0.05
            else:
                ail, rud = 0.0, 0.0
            controls.append({
                'throttle': th_trim,
                'elevator': el_trim,
                'aileron': ail,
                'rudder': rud
            })

    elif scenario == 'sinusoidal':
        # Small-amplitude sinusoidal inputs around trim
        freq = 0.3  # Hz
        for ti in t:
            controls.append({
                'throttle': th_trim,
                'elevator': el_trim + 0.1 * np.sin(2 * np.pi * freq * ti),
                'aileron': 0.05 * np.sin(2 * np.pi * freq * ti * 1.5),
                'rudder': 0.03 * np.sin(2 * np.pi * freq * ti * 0.8)
            })

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return controls


# ============================================================
# Run a single comparison
# ============================================================
def run_comparison(scenario: str, trim_planax: Dict, trim_jsbsim: Dict,
                   duration: float, dt: float):
    """Run Planax and JSBSim from their respective trim points.

    Each simulator uses its own native trim so neither is off-nominal at t=0.
    Control sequences are defined as *deltas* from the trim throttle/elevator,
    expressed in normalized command units.
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario}, duration: {duration}s, dt: {dt}s")
    print(f"{'='*60}")

    # Build control sequences for each simulator (uses its own trim baseline)
    controls_planax = generate_control_sequence(scenario, duration, dt, trim_planax)
    controls_jsbsim = generate_control_sequence(scenario, duration, dt, trim_jsbsim)

    # --- Planax ---
    print("Running Planax...")
    planax_sim = PlanaxSimulator(dt=dt)
    planax_state = planax_sim.reset_from_trim(trim_planax)
    planax_traj = []
    for i, ctrl in enumerate(controls_planax):
        sd = planax_sim.get_state_dict(planax_state)
        sd['time'] = i * dt
        planax_traj.append(sd)
        planax_state = planax_sim.step(planax_state, ctrl)
    print(f"  Planax: {len(planax_traj)} steps")

    # --- JSBSim ---
    print("Running JSBSim...")
    jsbsim_sim = JSBSimSimulator(dt=dt)
    jsbsim_sim.reset_from_trim(trim_jsbsim)
    jsbsim_traj = []
    for i, ctrl in enumerate(controls_jsbsim):
        sd = jsbsim_sim.get_state_dict()
        jsbsim_traj.append(sd)
        jsbsim_sim.step(ctrl)
    print(f"  JSBSim: {len(jsbsim_traj)} steps")

    return planax_traj, jsbsim_traj


# ============================================================
# I/O and metrics
# ============================================================
def save_csv(traj: List[Dict], filename: str):
    if not traj:
        return
    fieldnames = list(traj[0].keys())
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(traj)
    print(f"  Saved {filename}")


def compute_metrics(planax_traj: List[Dict], jsbsim_traj: List[Dict]) -> Dict:
    variables = ['vt', 'alpha', 'beta', 'P', 'Q', 'R', 'altitude', 'north', 'east', 'roll', 'pitch', 'yaw']
    metrics = {}
    n = min(len(planax_traj), len(jsbsim_traj))
    for var in variables:
        p = np.array([s[var] for s in planax_traj[:n]])
        j = np.array([s[var] for s in jsbsim_traj[:n]])
        e = p - j
        metrics[var] = {
            'rmse': float(np.sqrt(np.mean(e**2))),
            'max_abs_error': float(np.max(np.abs(e))),
            'final_error': float(e[-1]),
            'mean_planax': float(np.mean(p)),
            'mean_jsbsim': float(np.mean(j)),
        }
    return metrics


def plot_comparison(planax_traj, jsbsim_traj, scenario, output_dir):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Planax vs JSBSim: {scenario} (trim-initialized, 10s)', fontsize=14)

    variables = [
        ('vt', 'Airspeed (m/s)'), ('alpha', 'Alpha (rad)'), ('beta', 'Beta (rad)'),
        ('P', 'Roll Rate p (rad/s)'), ('Q', 'Pitch Rate q (rad/s)'), ('R', 'Yaw Rate r (rad/s)'),
        ('altitude', 'Altitude (m)'), ('roll', 'Roll (rad)'), ('pitch', 'Pitch (rad)')
    ]

    t_p = np.array([s['time'] for s in planax_traj])
    t_j = np.array([s['time'] for s in jsbsim_traj])

    for idx, (var, label) in enumerate(variables):
        ax = axes[idx // 3, idx % 3]
        ax.plot(t_p, [s[var] for s in planax_traj], 'r-', label='Planax', linewidth=1.6)
        ax.plot(t_j, [s[var] for s in jsbsim_traj], 'b--', label='JSBSim', linewidth=1.4)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, f'{scenario}_comparison_v4.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def generate_latex_table(all_metrics, output_file):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Trajectory-level consistency with JSBSim under matched open-loop controls (10s, trim-initialized).}",
        r"\label{tab:fidelity_validation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variable & Trim & Elevator Doublet & Coordinated Turn & Sinusoidal \\",
        r"\midrule",
    ]
    var_labels = {
        'vt': r'$V_t$ RMSE (m/s)',
        'alpha': r'$\alpha$ RMSE (deg)',
        'beta': r'$\beta$ RMSE (deg)',
        'P': r'$p$ RMSE (deg/s)',
        'Q': r'$q$ RMSE (deg/s)',
        'R': r'$r$ RMSE (deg/s)',
        'altitude': r'$h$ RMSE (m)',
    }
    deg_vars = {'alpha', 'beta', 'P', 'Q', 'R'}
    scenarios = ['trim', 'elevator_doublet', 'coordinated_turn', 'sinusoidal']

    for var, label in var_labels.items():
        row = [label]
        for sc in scenarios:
            v = all_metrics.get(sc, {}).get(var, {}).get('rmse', None)
            if v is None or np.isnan(v):
                row.append('--')
            else:
                if var in deg_vars:
                    v = np.degrees(v)
                row.append(f"{v:.3f}")
        lines.append(" & ".join(row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    txt = "\n".join(lines)
    with open(output_file, 'w') as f:
        f.write(txt)
    print(f"\nLaTeX table saved to {output_file}")
    return txt


# ============================================================
# Main
# ============================================================
def main():
    output_dir = Path(__file__).parent.parent / "results" / "fidelity_validation_lef_only_fix"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Planax vs JSBSim Validation v2 (each simulator uses native trim)")
    print("=" * 60)

    # --- Step 1a: Compute JSBSim trim ---
    print("\nStep 1a: Computing JSBSim trim...")
    trim_jsb = compute_jsbsim_trim(altitude_ft=15000.0, vt_fps=500.0, dt=0.02)
    print(f"  alpha={np.degrees(trim_jsb['alpha_rad']):.3f} deg, "
          f"theta={np.degrees(trim_jsb['theta_rad']):.3f} deg, "
          f"throttle={trim_jsb['throttle_norm']:.4f}, "
          f"elevator_pos={trim_jsb['elevator_pos_deg']:.3f} deg")

    # --- Step 1b: Compute Planax native trim ---
    print("\nStep 1b: Computing Planax's native trim...")
    trim_planax = compute_planax_trim(trim_jsb, dt=0.02)

    # Save trim solutions
    with open(output_dir / 'trim_solution_jsbsim.json', 'w') as f:
        json.dump(trim_jsb, f, indent=2)
    with open(output_dir / 'trim_solution_planax.json', 'w') as f:
        json.dump(trim_planax, f, indent=2)

    # --- Step 2: Run scenarios (10 seconds each) ---
    duration = 10.0
    dt = 0.02
    scenarios = ['trim', 'elevator_doublet', 'coordinated_turn', 'sinusoidal']

    all_metrics = {}
    for sc in scenarios:
        ptraj, jtraj = run_comparison(sc, trim_planax, trim_jsb, duration, dt)
        save_csv(ptraj, output_dir / f'{sc}_planax_v4.csv')
        save_csv(jtraj, output_dir / f'{sc}_jsbsim_v4.csv')
        plot_comparison(ptraj, jtraj, sc, str(output_dir))
        all_metrics[sc] = compute_metrics(ptraj, jtraj)

        # Print quick summary
        print(f"  Quick summary for {sc}:")
        for v in ['vt', 'alpha', 'P', 'Q', 'R', 'altitude', 'pitch']:
            m = all_metrics[sc][v]
            scale = ' rad' if v in ('alpha', 'P', 'Q', 'R', 'pitch') else (' m/s' if v == 'vt' else ' m')
            print(f"    {v:10s} RMSE={m['rmse']:.4f}{scale}, max|err|={m['max_abs_error']:.4f}")

    # --- Step 3: Save metrics & LaTeX table ---
    with open(output_dir / 'metrics_summary_v4.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'metrics_summary_v4.json'}")

    latex_txt = generate_latex_table(all_metrics, str(output_dir / 'validation_table_v4.tex'))

    # --- Step 4: Markdown summary ---
    summary = output_dir / 'validation_summary_v4.md'
    with open(summary, 'w') as f:
        f.write("# Trajectory-Level Fidelity Validation v2 (Native-Trim)\n\n")
        f.write("## Protocol\n\n")
        f.write("Both simulators are initialized from their own **native trim** point — ")
        f.write("a steady-state solution computed by each simulator's own aerodynamic model ")
        f.write("at the same altitude (15,000 ft) and airspeed (500 ft/s). ")
        f.write("This accounts for small intrinsic differences in the two aerodynamic ")
        f.write("implementations (interpolation, CG bookkeeping, leading-edge flap, etc.) ")
        f.write("and lets us compare the **response** to open-loop control inputs rather than ")
        f.write("trim-offset artifacts.\n\n")
        f.write("- F-16A model, NASA TP-1538 aerodynamic data\n")
        f.write(f"- Duration: {duration}s per scenario, dt={dt}s (50 Hz)\n")
        f.write("- Coordinate frame: NED, units: SI for comparison\n")
        f.write("- Control sequences defined as **deltas from each simulator's own trim**\n\n")
        f.write("## Trim Solutions\n\n")
        f.write("| Variable | JSBSim | Planax |\n|---|---|---|\n")
        for key in ['alpha_rad', 'theta_rad', 'throttle_norm', 'elevator_norm',
                    'elevator_pos_deg']:
            f.write(f"| {key} | {trim_jsb[key]:.6f} | {trim_planax[key]:.6f} |\n")
        f.write("\n## Results\n\n")
        for sc in scenarios:
            f.write(f"### {sc.replace('_', ' ').title()}\n\n")
            f.write("| Variable | RMSE | Max |Err| | Final Err | Mean Planax | Mean JSBSim |\n")
            f.write("|---|---|---|---|---|---|\n")
            for v, m in all_metrics[sc].items():
                f.write(f"| {v} | {m['rmse']:.4f} | {m['max_abs_error']:.4f} | {m['final_error']:.4f} | {m['mean_planax']:.4f} | {m['mean_jsbsim']:.4f} |\n")
            f.write("\n")
        f.write("## LaTeX table\n\n```latex\n" + latex_txt + "\n```\n")
    print(f"Summary saved to {summary}")

    print("\n" + "=" * 60)
    print("Validation v2 complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
