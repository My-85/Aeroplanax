# Quick Start Guide: Trajectory-Level Fidelity Validation

## What Has Been Implemented

A complete trajectory-level fidelity validation experiment comparing Planax (JAX-based F-16 with Tensor LUT aerodynamics) against JSBSim under matched open-loop control sequences.

## Files Created

### Experiment Files
1. **Main script**: `experiments/validate_planax_vs_jsbsim.py` (20 KB)
2. **Run script**: `experiments/run_validation.sh` (1.3 KB, executable)
3. **Documentation**: `experiments/README_VALIDATION.md` (7.7 KB)
4. **Summary**: `experiments/IMPLEMENTATION_SUMMARY.md` (9.4 KB)

### Paper Integration
5. **Paper guide**: `results/fidelity_validation/PAPER_INTEGRATION.md` (12 KB)

## How to Run

### Option 1: Quick Run (Recommended)

```bash
cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
./experiments/run_validation.sh
```

### Option 2: Manual Run

```bash
cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
conda activate aeroplanax
python experiments/validate_planax_vs_jsbsim.py
```

### Optional: Install JSBSim for Full Comparison

```bash
conda activate aeroplanax
pip install jsbsim
```

**Note**: If JSBSim is not installed, the script will run Planax-only simulation for demonstration purposes.

## What the Experiment Does

### Four Test Scenarios

1. **Trim (30s)**: Straight-and-level flight with constant controls
2. **Elevator Doublet (20s)**: Step elevator input for pitch response
3. **Coordinated Turn (30s)**: Aileron + rudder for turning maneuver
4. **Sinusoidal (40s)**: Aggressive multi-axis sinusoidal inputs

### Outputs Generated

**Location**: `results/fidelity_validation/`

- **8 CSV files**: Trajectories from both simulators (Planax and JSBSim)
- **4 PNG plots**: Comparison plots (9 subplots each showing airspeed, alpha, beta, P, Q, R, altitude, north, east)
- **1 JSON file**: Complete numerical metrics (RMSE, max error, final error)
- **1 Markdown file**: Human-readable summary report
- **1 LaTeX file**: Ready-to-use table for paper

### Runtime

- **Total**: ~5-10 minutes
- **Per scenario**: ~1-2 minutes

## For Your Paper

### Step 1: Run the Experiment

```bash
./experiments/run_validation.sh
```

### Step 2: Get the LaTeX Table

The file `results/fidelity_validation/validation_table.tex` contains a ready-to-use LaTeX table.

Copy it directly into your paper's tables section.

### Step 3: Add Methodology Text

Open `results/fidelity_validation/PAPER_INTEGRATION.md` and copy the "Methodology Description" section to your paper's Methods section.

### Step 4: Include a Comparison Plot

Choose one plot from `results/fidelity_validation/` (e.g., `elevator_doublet_comparison.png`) for the main paper.

Use the figure caption provided in `PAPER_INTEGRATION.md`.

### Step 5: Add Discussion

Copy the discussion points from `PAPER_INTEGRATION.md` to your Discussion section.

### Step 6: Add References

Add the BibTeX entries from `PAPER_INTEGRATION.md`:
- NASA TP-1538 (F-16 aerodynamic data)
- JSBSim (reference simulator)

## Important Notes

### Conservative Language

✓ Use: "trajectory-level consistency with JSBSim"  
✓ Use: "validates the Tensor LUT aerodynamic implementation"  
✓ Use: "suitable for RL training purposes"  

✗ Avoid: "real-world validation" (no flight test data)  
✗ Avoid: "high-fidelity simulation" (ambiguous)  
✗ Avoid: "proves accuracy" (too strong)  

### What This Validates

- ✓ Planax correctly implements NASA F-16 aerodynamic data via Tensor LUT
- ✓ Trajectory-level consistency with JSBSim reference simulator
- ✓ Suitable dynamics for RL training (relative dynamics matter)

### What This Does NOT Validate

- ✗ Real-world flight accuracy (no flight test data comparison)
- ✗ Extreme flight conditions (tests use moderate maneuvers)
- ✗ High-frequency dynamics (50 Hz sampling)

## Troubleshooting

### "JSBSim not found"
```bash
pip install jsbsim
```
Or run anyway - the script will work with Planax-only simulation.

### "Import errors"
```bash
conda activate aeroplanax
```

### "No such file or directory"
```bash
cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax
```

## File Paths Reference

All paths relative to: `/home/dqy/aeroplanax/new/20251215最新代码库/Planax/`

**Experiment files**:
- `experiments/validate_planax_vs_jsbsim.py`
- `experiments/run_validation.sh`
- `experiments/README_VALIDATION.md`
- `experiments/IMPLEMENTATION_SUMMARY.md`

**Output files** (after running):
- `results/fidelity_validation/*.csv` (8 trajectory files)
- `results/fidelity_validation/*.png` (4 comparison plots)
- `results/fidelity_validation/metrics_summary.json`
- `results/fidelity_validation/validation_summary.md`
- `results/fidelity_validation/validation_table.tex`

**Paper integration**:
- `results/fidelity_validation/PAPER_INTEGRATION.md`

## Next Steps

1. **Run the experiment**: `./experiments/run_validation.sh`
2. **Check outputs**: Look in `results/fidelity_validation/`
3. **Review metrics**: Open `validation_summary.md`
4. **Integrate into paper**: Follow steps in `PAPER_INTEGRATION.md`

## Documentation

For detailed information, see:
- **Experiment details**: `experiments/README_VALIDATION.md`
- **Paper integration**: `results/fidelity_validation/PAPER_INTEGRATION.md`
- **Implementation summary**: `experiments/IMPLEMENTATION_SUMMARY.md`

---

**Ready to run!** Execute `./experiments/run_validation.sh` to start the validation experiment.
