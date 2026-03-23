#!/usr/bin/env python3
"""Re-evaluate champion checkpoint with current eval conditions (1 seed, 1000 steps) at all levels."""
import json
from evaluator import evaluate_checkpoint_per_level

CHAMPION_CKPT = "/home/dqy/aeroplanax/new/20251215最新代码库/Planax/results/heading_pitch_V_discrete_rnn_2026-03-20-19-38/checkpoints/checkpoint_epoch_1350"

result = evaluate_checkpoint_per_level(
    CHAMPION_CKPT,
    levels=[0, 1, 2, 3, 5],
    champion_per_level=None,  # No early-exit, run all levels
)

print("\n\n=== RESULTS FOR champion_meta.json ===")
print(json.dumps(result, indent=2))
