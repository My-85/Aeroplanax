# Baseline: Quaternion Full-Envelope Flight Control

**Checkpoint**: `results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600`

## Quick Test

```bash
conda activate aeroplanax
cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax

# S-maneuver waypoint tracking (100 gentle waypoints)
python render_waypoint_s_quat.py

# L3 full-envelope maneuver test (9 aggressive waypoints)
python render_waypoint_L3_test.py
```

Make sure CKPT_PATH in the render scripts points to the checkpoint above.

## Performance

- S-maneuver: 100/100 waypoints, 186s
- L3 test: 9/9 waypoints, 218s
- G-load p95: 6.9G
- Full details: see [BASELINE_REPORT.md](BASELINE_REPORT.md)
