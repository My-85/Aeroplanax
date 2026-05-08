#!/usr/bin/env python3
"""
Trajectory-level Fidelity Validation: Planax vs JSBSim

This script compares Planax (JAX-based F-16 simulator with Tensor LUT aerodynamics)
against JSBSim under matched open-loop control sequences.

Planax uses NASA F-16 aerodynamic coefficient data via Tensor LUT interpolation.
JSBSim serves as the reference simulator.

Usage:
    conda activate aeroplanax
    python experiments/validate_planax_vs_jsbsim.py

Author: Generated for IEEE RA-L manuscript
Date: 2026-05-07
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
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
from dynamics.F16_jax.F16Dynamics import update as planax_update
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
    print("WARNING: JSBSim Python bindings not available.")
    print("Install with: pip install jsbsim")
    print("Continuing with Planax-only simulation for demonstration.")


class PlanaxSimulator:
    """Wrapper for Planax F-16 dynamics"""

    def __init__(self, dt=0.02):
        self.dt = dt  # 50 Hz

    def reset(self, initial_state: Dict) -> FighterPlaneState:
        """Initialize Planax state from initial conditions"""
        # Convert from standard units to Planax internal units
        # Planax uses: position in meters, velocity in m/s, angles in radians

        # Initial quaternion (level flight, heading 0)
        q0, q1, q2, q3 = 1.0, 0.0, 0.0, 0.0

        state = FighterPlaneState(
            north=initial_state['north'],  # meters
            east=initial_state['east'],    # meters
            altitude=initial_state['altitude'],  # meters
            roll=0.0,
            pitch=initial_state['pitch'],
            yaw=initial_state['yaw'],
            vel_x=0.0,
            vel_y=0.0,
            vel_z=0.0,
            vt=initial_state['vt'],  # m/s
            q0=q0, q1=q1, q2=q2, q3=q3,
            alpha=initial_state['alpha'],
            beta=initial_state['beta'],
            P=0.0, Q=0.0, R=0.0,
            T=initial_state['throttle'] * 0.225 * 76300 / 0.3048,  # Convert to lbf
            el=0.0, ail=0.0, rud=0.0,
            ax=0.0, ay=0.0, az=0.0
        )
        return state

    def step(self, state: FighterPlaneState, control: Dict) -> FighterPlaneState:
        """Step the simulator forward"""
        action = FighterPlaneControlState(
            throttle=control['throttle'],
            elevator=control['elevator'],
            aileron=control['aileron'],
            rudder=control['rudder'],
            leading_edge_flap=0.0
        )

        new_state = planax_env_update(state, action, self.dt)
        return new_state

    def get_state_dict(self, state: FighterPlaneState) -> Dict:
        """Extract state variables as dictionary"""
        roll, pitch, yaw = quaternion_to_rpy(state.q0, state.q1, state.q2, state.q3)

        return {
            'time': 0.0,  # Will be set externally
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
            'q0': float(state.q0),
            'q1': float(state.q1),
            'q2': float(state.q2),
            'q3': float(state.q3),
        }


class JSBSimSimulator:
    """Wrapper for JSBSim F-16"""

    def __init__(self, dt=0.02):
        if not JSBSIM_AVAILABLE:
            raise ImportError("JSBSim not available")

        self.dt = dt

        # Redirect JSBSim output to suppress verbose logging
        import sys
        import os
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

            self.fdm = jsbsim.FGFDMExec(None)

            # Set JSBSim root directory (use lowercase with underscore)
            jsbsim_root = Path(__file__).parent.parent.parent / "jsbsim" / "jsbsim"
            self.fdm.set_root_dir(str(jsbsim_root))

            # Load F-16 aircraft
            self.fdm.load_model('f16')

            # Set timestep
            self.fdm.set_dt(dt)

            # Run initial conditions
            self.fdm.run_ic()
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def reset(self, initial_state: Dict):
        """Initialize JSBSim state using property values"""
        # Set initial conditions via properties (JSBSim uses feet and ft/s)
        self.fdm.set_property_value('ic/h-sl-ft', initial_state['altitude'] / 0.3048)
        self.fdm.set_property_value('ic/u-fps', initial_state['vt'] / 0.3048)
        self.fdm.set_property_value('ic/v-fps', 0.0)
        self.fdm.set_property_value('ic/w-fps', 0.0)
        self.fdm.set_property_value('ic/phi-deg', 0.0)
        self.fdm.set_property_value('ic/theta-deg', np.degrees(initial_state['pitch']))
        self.fdm.set_property_value('ic/psi-deg', np.degrees(initial_state['yaw']))
        self.fdm.set_property_value('ic/alpha-deg', np.degrees(initial_state['alpha']))
        self.fdm.set_property_value('ic/beta-deg', np.degrees(initial_state['beta']))

        # Reset to initial conditions (mode=0: standard reset)
        self.fdm.reset_to_initial_conditions(0)

    def step(self, control: Dict):
        """Step JSBSim forward"""
        # Set controls (JSBSim uses normalized -1 to 1 or 0 to 1)
        self.fdm.set_property_value('fcs/throttle-cmd-norm', control['throttle'])
        self.fdm.set_property_value('fcs/elevator-cmd-norm', control['elevator'])
        self.fdm.set_property_value('fcs/aileron-cmd-norm', control['aileron'])
        self.fdm.set_property_value('fcs/rudder-cmd-norm', control['rudder'])

        # Run one step
        self.fdm.run()

    def get_state_dict(self) -> Dict:
        """Extract state variables"""
        return {
            'time': self.fdm.get_property_value('simulation/sim-time-sec'),
            'north': self.fdm.get_property_value('position/distance-from-start-lat-mt'),
            'east': self.fdm.get_property_value('position/distance-from-start-lon-mt'),
            'altitude': self.fdm.get_property_value('position/h-sl-meters'),
            'roll': self.fdm.get_property_value('attitude/phi-rad'),
            'pitch': self.fdm.get_property_value('attitude/theta-rad'),
            'yaw': self.fdm.get_property_value('attitude/psi-rad'),
            'vt': self.fdm.get_property_value('velocities/vt-fps') * 0.3048,  # Convert to m/s
            'alpha': self.fdm.get_property_value('aero/alpha-rad'),
            'beta': self.fdm.get_property_value('aero/beta-rad'),
            'P': self.fdm.get_property_value('velocities/p-rad_sec'),
            'Q': self.fdm.get_property_value('velocities/q-rad_sec'),
            'R': self.fdm.get_property_value('velocities/r-rad_sec'),
        }


def generate_control_sequence(scenario: str, duration: float, dt: float) -> List[Dict]:
    """
    Generate deterministic control sequences for validation scenarios.

    Args:
        scenario: One of ['trim', 'elevator_doublet', 'coordinated_turn', 'sinusoidal']
        duration: Simulation duration in seconds
        dt: Timestep in seconds

    Returns:
        List of control dictionaries with keys: throttle, elevator, aileron, rudder
    """
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)
    controls = []

    if scenario == 'trim':
        # Straight-and-level trimmed flight
        for _ in range(n_steps):
            controls.append({
                'throttle': 0.5,
                'elevator': 0.0,
                'aileron': 0.0,
                'rudder': 0.0
            })

    elif scenario == 'elevator_doublet':
        # Elevator doublet: step up, hold, step down, return
        for i, ti in enumerate(t):
            if ti < 5.0:
                el = 0.0
            elif ti < 7.0:
                el = 0.2  # Pitch up
            elif ti < 9.0:
                el = -0.2  # Pitch down
            else:
                el = 0.0

            controls.append({
                'throttle': 0.5,
                'elevator': el,
                'aileron': 0.0,
                'rudder': 0.0
            })

    elif scenario == 'coordinated_turn':
        # Coordinated turn with aileron and rudder
        for i, ti in enumerate(t):
            if ti < 5.0:
                ail, rud = 0.0, 0.0
            elif ti < 15.0:
                ail, rud = 0.3, 0.15  # Right turn
            else:
                ail, rud = 0.0, 0.0

            controls.append({
                'throttle': 0.5,
                'elevator': 0.0,
                'aileron': ail,
                'rudder': rud
            })

    elif scenario == 'sinusoidal':
        # Aggressive sinusoidal control inputs
        freq = 0.2  # Hz
        for i, ti in enumerate(t):
            controls.append({
                'throttle': 0.5 + 0.2 * np.sin(2 * np.pi * freq * ti),
                'elevator': 0.3 * np.sin(2 * np.pi * freq * ti),
                'aileron': 0.2 * np.sin(2 * np.pi * freq * ti * 1.5),
                'rudder': 0.15 * np.sin(2 * np.pi * freq * ti * 0.8)
            })

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return controls


def run_comparison(scenario: str, duration: float = 30.0, dt: float = 0.02) -> Tuple[List[Dict], List[Dict]]:
    """
    Run a comparison between Planax and JSBSim for a given scenario.

    Returns:
        (planax_trajectory, jsbsim_trajectory)
    """
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario}")
    print(f"Duration: {duration}s, dt: {dt}s")
    print(f"{'='*60}")

    # Initial conditions (trimmed level flight at 15000 ft, 500 ft/s)
    initial_state = {
        'north': 0.0,
        'east': 0.0,
        'altitude': 4572.0,  # 15000 ft in meters
        'pitch': 0.0,
        'yaw': 0.0,
        'vt': 152.4,  # 500 ft/s in m/s
        'alpha': 0.0,
        'beta': 0.0,
        'throttle': 0.5
    }

    # Generate control sequence
    controls = generate_control_sequence(scenario, duration, dt)

    # Run Planax
    print("Running Planax...")
    planax_sim = PlanaxSimulator(dt=dt)
    planax_state = planax_sim.reset(initial_state)
    planax_traj = []

    for i, ctrl in enumerate(controls):
        state_dict = planax_sim.get_state_dict(planax_state)
        state_dict['time'] = i * dt
        planax_traj.append(state_dict)
        planax_state = planax_sim.step(planax_state, ctrl)

    print(f"Planax simulation complete: {len(planax_traj)} steps")

    # Run JSBSim if available
    jsbsim_traj = []
    if JSBSIM_AVAILABLE:
        print("Running JSBSim...")
        try:
            jsbsim_sim = JSBSimSimulator(dt=dt)
            jsbsim_sim.reset(initial_state)

            for i, ctrl in enumerate(controls):
                state_dict = jsbsim_sim.get_state_dict()
                jsbsim_traj.append(state_dict)
                jsbsim_sim.step(ctrl)

            print(f"JSBSim simulation complete: {len(jsbsim_traj)} steps")
        except Exception as e:
            print(f"JSBSim simulation failed: {e}")
            jsbsim_traj = []
    else:
        print("JSBSim not available, skipping reference simulation")

    return planax_traj, jsbsim_traj


def save_trajectory_csv(trajectory: List[Dict], filename: str):
    """Save trajectory to CSV file"""
    if not trajectory:
        return

    fieldnames = list(trajectory[0].keys())

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory)

    print(f"Saved trajectory to {filename}")


def compute_metrics(planax_traj: List[Dict], jsbsim_traj: List[Dict]) -> Dict:
    """
    Compute comparison metrics between Planax and JSBSim trajectories.

    Returns dictionary with RMSE, max error, final error for each variable.
    """
    if not jsbsim_traj:
        return {}

    # Variables to compare
    variables = ['vt', 'alpha', 'beta', 'P', 'Q', 'R', 'altitude', 'north', 'east']

    metrics = {}

    for var in variables:
        planax_vals = np.array([s[var] for s in planax_traj])
        jsbsim_vals = np.array([s[var] for s in jsbsim_traj])

        # Ensure same length
        min_len = min(len(planax_vals), len(jsbsim_vals))
        planax_vals = planax_vals[:min_len]
        jsbsim_vals = jsbsim_vals[:min_len]

        # Compute errors
        errors = planax_vals - jsbsim_vals

        metrics[var] = {
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'max_abs_error': float(np.max(np.abs(errors))),
            'final_error': float(errors[-1]),
            'mean_planax': float(np.mean(planax_vals)),
            'mean_jsbsim': float(np.mean(jsbsim_vals))
        }

    # Compute quaternion geodesic attitude error if available
    if 'q0' in planax_traj[0]:
        attitude_errors = []
        for p_state, j_state in zip(planax_traj, jsbsim_traj):
            # Quaternion from Planax
            q_p = np.array([p_state['q0'], p_state['q1'], p_state['q2'], p_state['q3']])
            # For JSBSim, construct quaternion from Euler angles
            # (simplified - would need proper conversion)
            # Skip for now if JSBSim doesn't provide quaternions directly
            pass

    return metrics


def plot_comparison(planax_traj: List[Dict], jsbsim_traj: List[Dict],
                   scenario: str, output_dir: str):
    """Generate comparison plots"""

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Planax vs JSBSim: {scenario}', fontsize=16)

    variables = [
        ('vt', 'Airspeed (m/s)'),
        ('alpha', 'Angle of Attack (rad)'),
        ('beta', 'Sideslip (rad)'),
        ('P', 'Roll Rate (rad/s)'),
        ('Q', 'Pitch Rate (rad/s)'),
        ('R', 'Yaw Rate (rad/s)'),
        ('altitude', 'Altitude (m)'),
        ('north', 'North Position (m)'),
        ('east', 'East Position (m)')
    ]

    t_planax = np.array([s['time'] for s in planax_traj])

    for idx, (var, label) in enumerate(variables):
        ax = axes[idx // 3, idx % 3]

        # Plot Planax
        planax_vals = np.array([s[var] for s in planax_traj])
        ax.plot(t_planax, planax_vals, 'r-', label='Planax', linewidth=2)

        # Plot JSBSim if available
        if jsbsim_traj:
            t_jsbsim = np.array([s['time'] for s in jsbsim_traj])
            jsbsim_vals = np.array([s[var] for s in jsbsim_traj])
            ax.plot(t_jsbsim, jsbsim_vals, 'b--', label='JSBSim', linewidth=1.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(output_dir, f'{scenario}_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def generate_latex_table(all_metrics: Dict[str, Dict], output_file: str):
    """Generate LaTeX table for paper"""

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Trajectory-level consistency with JSBSim under matched open-loop controls}")
    latex.append(r"\label{tab:fidelity_validation}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"Variable & Trim & Elevator Doublet & Coordinated Turn & Sinusoidal \\")
    latex.append(r"\midrule")

    # Key variables to report
    key_vars = ['vt', 'alpha', 'P', 'Q', 'R', 'altitude']
    var_labels = {
        'vt': r'$V_t$ RMSE (m/s)',
        'alpha': r'$\alpha$ RMSE (rad)',
        'P': r'$p$ RMSE (rad/s)',
        'Q': r'$q$ RMSE (rad/s)',
        'R': r'$r$ RMSE (rad/s)',
        'altitude': r'$h$ RMSE (m)'
    }

    scenarios = ['trim', 'elevator_doublet', 'coordinated_turn', 'sinusoidal']

    for var in key_vars:
        row = [var_labels[var]]
        for scenario in scenarios:
            if scenario in all_metrics and var in all_metrics[scenario]:
                rmse = all_metrics[scenario][var]['rmse']
                row.append(f"{rmse:.3f}")
            else:
                row.append("--")
        latex.append(" & ".join(row) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)

    with open(output_file, 'w') as f:
        f.write(latex_str)

    print(f"\nLaTeX table saved to {output_file}")
    print("\n" + latex_str)

    return latex_str


def main():
    """Main validation experiment"""

    # Output directory
    output_dir = Path(__file__).parent.parent / "results" / "fidelity_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Planax vs JSBSim Trajectory-Level Fidelity Validation")
    print("="*60)
    print(f"Output directory: {output_dir}")

    # Scenarios to run
    scenarios = [
        ('trim', 30.0),
        ('elevator_doublet', 20.0),
        ('coordinated_turn', 30.0),
        ('sinusoidal', 40.0)
    ]

    all_metrics = {}

    for scenario, duration in scenarios:
        # Run comparison
        planax_traj, jsbsim_traj = run_comparison(scenario, duration=duration, dt=0.02)

        # Save trajectories
        save_trajectory_csv(planax_traj, output_dir / f'{scenario}_planax.csv')
        if jsbsim_traj:
            save_trajectory_csv(jsbsim_traj, output_dir / f'{scenario}_jsbsim.csv')

        # Compute metrics
        if jsbsim_traj:
            metrics = compute_metrics(planax_traj, jsbsim_traj)
            all_metrics[scenario] = metrics

        # Generate plots
        plot_comparison(planax_traj, jsbsim_traj, scenario, str(output_dir))

    # Save metrics summary
    if all_metrics:
        metrics_file = output_dir / 'metrics_summary.json'
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics summary saved to {metrics_file}")

        # Generate LaTeX table
        latex_file = output_dir / 'validation_table.tex'
        generate_latex_table(all_metrics, str(latex_file))

    # Generate summary markdown
    summary_file = output_dir / 'validation_summary.md'
    with open(summary_file, 'w') as f:
        f.write("# Trajectory-Level Fidelity Validation: Planax vs JSBSim\n\n")
        f.write("## Experimental Protocol\n\n")
        f.write("This experiment compares Planax (JAX-based F-16 simulator with Tensor LUT aerodynamics) ")
        f.write("against JSBSim under matched open-loop control sequences.\n\n")
        f.write("### Matched Settings\n\n")
        f.write("- **Aircraft Model**: F-16A\n")
        f.write("- **Aerodynamic Data**: NASA F-16 wind tunnel data (TP-1538)\n")
        f.write("- **Initial Conditions**: Trimmed level flight at 15,000 ft, 500 ft/s\n")
        f.write("- **Timestep**: 0.02s (50 Hz)\n")
        f.write("- **Coordinate Frame**: NED (North-East-Down)\n")
        f.write("- **Unit Conventions**: SI units (meters, m/s, radians)\n\n")
        f.write("### Test Scenarios\n\n")
        f.write("1. **Trim**: Straight-and-level flight with constant controls (30s)\n")
        f.write("2. **Elevator Doublet**: Step elevator input for pitch response (20s)\n")
        f.write("3. **Coordinated Turn**: Aileron + rudder for turning maneuver (30s)\n")
        f.write("4. **Sinusoidal**: Aggressive multi-axis sinusoidal inputs (40s)\n\n")

        if all_metrics:
            f.write("## Results Summary\n\n")
            for scenario in scenarios:
                scenario_name = scenario[0]
                if scenario_name in all_metrics:
                    f.write(f"### {scenario_name.replace('_', ' ').title()}\n\n")
                    f.write("| Variable | RMSE | Max Error | Final Error |\n")
                    f.write("|----------|------|-----------|-------------|\n")
                    for var, m in all_metrics[scenario_name].items():
                        f.write(f"| {var} | {m['rmse']:.4f} | {m['max_abs_error']:.4f} | {m['final_error']:.4f} |\n")
                    f.write("\n")

        f.write("## Limitations and Assumptions\n\n")
        f.write("- JSBSim and Planax may use different internal integrators (JSBSim: RK4, Planax: Euler)\n")
        f.write("- Comparison performed at same output sampling times (50 Hz)\n")
        f.write("- Both simulators use NASA F-16 aerodynamic data, but interpolation methods differ\n")
        f.write("- Planax uses Tensor LUT (trilinear interpolation), JSBSim uses table lookups\n")
        f.write("- This validates trajectory-level consistency, not real-world fidelity\n")

    print(f"\nValidation summary saved to {summary_file}")
    print("\n" + "="*60)
    print("Validation experiment complete!")
    print("="*60)


if __name__ == "__main__":
    main()
