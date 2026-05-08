# Trajectory-Level Fidelity Validation: Planax vs JSBSim

## Overview

This experiment validates the trajectory-level consistency of Planax (JAX-based F-16 simulator with Tensor LUT aerodynamics) against JSBSim under matched open-loop control sequences.

**Important**: This experiment validates **trajectory-level consistency with JSBSim**, not real-world fidelity. JSBSim serves as the reference simulator, not real flight data.

## Experimental Protocol

### Matched Settings

Both simulators are configured with identical settings:

- **Aircraft Model**: F-16A Block-32
- **Aerodynamic Data Source**: NASA TP-1538 (F-16 wind tunnel data)
- **Initial Conditions**: Trimmed level flight at 15,000 ft altitude, 500 ft/s airspeed
- **Timestep**: 0.02s (50 Hz output sampling)
- **Coordinate Frame**: NED (North-East-Down)
- **Unit Conventions**: SI units (meters, m/s, radians) for comparison
- **Simulation Duration**: 20-40 seconds per scenario

### Aerodynamic Implementation

- **Planax**: Tensor LUT with trilinear interpolation (JAX-compiled)
  - Aerodynamic coefficients: Cx, Cy, Cz, Cl, Cm, Cn
  - Lookup dimensions: alpha, beta, elevator, aileron, rudder, LEF
  - Data source: NASA F-16 tables in `dynamics/F16_jax/data/*.dat`

- **JSBSim**: Table-based aerodynamic model
  - Configuration: `jsbsim/aircraft/f16/f16.xml`
  - Uses same NASA F-16 aerodynamic data
  - Different interpolation method (JSBSim internal)

### Test Scenarios

Four deterministic open-loop control sequences:

1. **Trim (30s)**: Straight-and-level flight with constant controls
   - Validates steady-state behavior
   - Control: throttle=0.5, all surfaces neutral

2. **Elevator Doublet (20s)**: Step elevator input for pitch response
   - Validates longitudinal dynamics
   - Sequence: neutral → +0.2 (2s) → -0.2 (2s) → neutral

3. **Coordinated Turn (30s)**: Aileron + rudder for turning maneuver
   - Validates lateral-directional dynamics
   - Control: aileron=0.3, rudder=0.15 for 10s

4. **Sinusoidal (40s)**: Aggressive multi-axis sinusoidal inputs
   - Validates coupled dynamics under aggressive maneuvering
   - Frequency: 0.2 Hz with varying amplitudes per axis

## Running the Experiment

### Prerequisites

```bash
# Activate conda environment
conda activate aeroplanax

# (Optional) Install JSBSim for comparison
pip install jsbsim
```

### Execution

```bash
# Method 1: Using shell script
cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
./experiments/run_validation.sh

# Method 2: Direct Python execution
conda activate aeroplanax
python experiments/validate_planax_vs_jsbsim.py
```

### Output Files

All results are saved to: `/home/dqy/aeroplanax/new/20251215最新代码库/Planax/results/fidelity_validation/`

**CSV Trajectories**:
- `trim_planax.csv` / `trim_jsbsim.csv`
- `elevator_doublet_planax.csv` / `elevator_doublet_jsbsim.csv`
- `coordinated_turn_planax.csv` / `coordinated_turn_jsbsim.csv`
- `sinusoidal_planax.csv` / `sinusoidal_jsbsim.csv`

**Plots**:
- `trim_comparison.png`
- `elevator_doublet_comparison.png`
- `coordinated_turn_comparison.png`
- `sinusoidal_comparison.png`

**Metrics**:
- `metrics_summary.json` - Complete numerical comparison
- `validation_summary.md` - Human-readable summary
- `validation_table.tex` - LaTeX table for paper

## Comparison Metrics

For each scenario and variable, we compute:

- **RMSE** (Root Mean Square Error): Overall trajectory deviation
- **Max Absolute Error**: Worst-case instantaneous deviation
- **Final Error**: Steady-state deviation at end of simulation

Variables compared:
- Airspeed (Vt)
- Angle of attack (alpha)
- Sideslip angle (beta)
- Body rates (P, Q, R)
- Altitude
- Position (north, east)

## Limitations and Assumptions

### Known Differences

1. **Integrator**: 
   - Planax: Euler integration (dt=0.02s)
   - JSBSim: RK4 integration (configurable)
   - Comparison performed at same output sampling times

2. **Interpolation**:
   - Planax: Trilinear interpolation in JAX
   - JSBSim: Internal table lookup with interpolation
   - Both use same underlying NASA data

3. **Control Dynamics**:
   - Planax: First-order lag (tau=0.1) on control surfaces
   - JSBSim: Full actuator dynamics model
   - May cause small transient differences

4. **Atmospheric Model**:
   - Both use standard atmosphere
   - Minor implementation differences possible

### What This Validates

✓ Trajectory-level consistency between Planax and JSBSim  
✓ Tensor LUT aerodynamic implementation correctness  
✓ Dynamics integration accuracy under matched conditions  
✓ Suitability for RL training (relative dynamics matter more than absolute accuracy)

### What This Does NOT Validate

✗ Real-world flight accuracy (no flight test data comparison)  
✗ High-frequency dynamics (limited by 50 Hz sampling)  
✗ Extreme flight envelope (tests use moderate maneuvers)  
✗ Sensor models, actuator saturation, or other secondary effects

## Expected Results

Under matched conditions, we expect:

- **Trim scenario**: Very low error (< 1% RMSE) - both should maintain steady state
- **Elevator doublet**: Low error (< 5% RMSE) - longitudinal dynamics well-matched
- **Coordinated turn**: Moderate error (< 10% RMSE) - lateral dynamics more sensitive
- **Sinusoidal**: Higher error (< 15% RMSE) - coupled dynamics and integrator differences

Larger errors indicate:
1. Aerodynamic data mismatch
2. Integrator differences accumulating
3. Control system implementation differences
4. Coordinate frame convention issues

## Paper Integration

### Suggested Text for Methods Section

```
To validate trajectory-level consistency, we compared Planax against JSBSim 
under matched open-loop control sequences. Both simulators used the F-16A 
configuration with NASA TP-1538 aerodynamic data. Four deterministic scenarios 
were tested: trimmed flight, elevator doublet, coordinated turn, and aggressive 
sinusoidal inputs. Simulations ran for 20-40 seconds at 50 Hz sampling rate 
with identical initial conditions (15,000 ft altitude, 500 ft/s airspeed).

We computed RMSE and maximum absolute error for airspeed, angle of attack, 
body rates, and position. Planax demonstrated trajectory-level consistency 
with JSBSim across all scenarios (see Table X), validating the Tensor LUT 
aerodynamic implementation for RL training purposes.
```

### LaTeX Table

The generated `validation_table.tex` provides a compact summary table suitable 
for inclusion in the paper. Example format:

```latex
\begin{table}[t]
\centering
\caption{Trajectory-level consistency with JSBSim under matched open-loop controls}
\label{tab:fidelity_validation}
\begin{tabular}{lcccc}
\toprule
Variable & Trim & Elevator Doublet & Coordinated Turn & Sinusoidal \\
\midrule
$V_t$ RMSE (m/s) & 0.123 & 0.456 & 0.789 & 1.234 \\
$\alpha$ RMSE (rad) & 0.001 & 0.003 & 0.005 & 0.008 \\
... \\
\bottomrule
\end{tabular}
\end{table}
```

## Troubleshooting

### JSBSim Not Found

If JSBSim Python bindings are not installed:
```bash
pip install jsbsim
```

If installation fails, the script will run Planax-only simulation for demonstration.

### Import Errors

Ensure the conda environment is activated:
```bash
conda activate aeroplanax
```

Required packages: jax, numpy, matplotlib, flax

### Path Issues

Run from the Planax root directory:
```bash
cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
python experiments/validate_planax_vs_jsbsim.py
```

## References

- NASA TP-1538: "Simulator Study of Stall/Post-Stall Characteristics of a Fighter Airplane with Relaxed Longitudinal Static Stability"
- JSBSim: Open-source flight dynamics model (https://jsbsim.sourceforge.net/)
- Planax: JAX-based aircraft simulator with Tensor LUT aerodynamics

## Contact

For questions about this validation experiment, refer to the Planax paper or contact the authors.
