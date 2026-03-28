#!/bin/bash
# 测试 evaluator.py 是否能正常评估 autotuned_1350

source ~/miniconda3/etc/profile.d/conda.sh
conda activate aeroplanax

python evaluator.py \
  --checkpoint /home/dqy/aeroplanax/new/20251215最新代码库/Planax/results/heading_pitch_V_discrete_rnn_2026-03-20-19-38/checkpoints/checkpoint_epoch_1350 \
  --waypoint \
  --output test_evaluator_result.json
