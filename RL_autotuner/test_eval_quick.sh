#!/bin/bash
# Quick smoke test: only first 2 waypoints

cd /home/dqy/aeroplanax/new/20251215最新代码库/RL_autotuner

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aeroplanax

python evaluate_and_render_baseline.py \
    --checkpoint /home/dqy/aeroplanax/new/20251215最新代码库/Planax/results/heading_pitch_V_discrete_rnn_2026-03-20-19-38/checkpoints/checkpoint_epoch_1350 \
    --output-dir eval_test_quick \
    --label test_quick

echo ""
echo "Checking first 30 frames of ACMI..."
head -100 eval_test_quick/test_quick_result.acmi | grep "^100," | head -5
