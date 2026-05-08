#!/bin/bash
# Run trajectory-level fidelity validation experiment
# Compares Planax vs JSBSim under matched open-loop controls

set -e

echo "============================================================"
echo "Planax vs JSBSim Trajectory-Level Fidelity Validation"
echo "============================================================"

# Check if conda environment exists
if ! conda env list | grep -q "aeroplanax"; then
    echo "ERROR: conda environment 'aeroplanax' not found"
    echo "Please create the environment first"
    exit 1
fi

# Activate environment
echo "Activating conda environment: aeroplanax"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aeroplanax

# Check if JSBSim is installed
echo "Checking for JSBSim Python bindings..."
if python -c "import jsbsim" 2>/dev/null; then
    echo "✓ JSBSim available"
else
    echo "⚠ JSBSim not available - will run Planax-only simulation"
    echo "To install JSBSim: pip install jsbsim"
fi

# Run validation
echo ""
echo "Running validation experiment..."
python experiments/validate_planax_vs_jsbsim.py

echo ""
echo "============================================================"
echo "Validation complete!"
echo "Results saved to: results/fidelity_validation/"
echo "============================================================"
