# Vertical Energy Balanced Fine-Tune V2

Status: prepared only. Claude planner-level regression rejected `checkpoint_epoch_658` as the main baseline, so V2 starts from `checkpoint_epoch_619`.

## Main Branch

- Branch name: `vertical_energy_balanced_finetune_v2`
- Source checkpoint: `results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619`
- Do not continue from `checkpoint_epoch_658`
- Do not train full loop
- Primary config: `paper/second_paper/vertical_energy_balanced_finetune_v2_config.json`
- Compatibility config alias: `paper/second_paper/next_vertical_arc_training_config.json`

The replay mix is intentionally conservative:

- original task replay: 35%
- circle / S-curve / figure-eight proxy: 20%
- level flight altitude retention: 15%
- vertical energy / pull-up / arc curriculum: 30%

## Reward / Safety

Retained:

- vertical energy management reward
- low-speed penalty
- alpha/beta penalty
- G penalty

Added or strengthened:

- level flight altitude retention
- circle / S-curve / figure-eight altitude retention
- horizontal altitude drift penalty

## Periodic Proxy Eval

Use the runner so GPU memory is released between training and evaluation:

```bash
JAX_PLATFORMS=cuda MPLCONFIGDIR=/tmp WANDB_MODE=offline python run_vertical_energy_balanced_v2.py --config paper/second_paper/vertical_energy_balanced_finetune_v2_config.json
```

The runner trains one 2.5M-timestep chunk, saves a checkpoint, exits the training process, then runs:

```bash
python eval_vertical_energy_checkpoints.py --suite planner_proxy
```

This repeats every cycle. The proxy eval includes pull-up, 60/90 vertical arc, level flight, heading/pitch retention, level circle, S-curve, figure-eight, and mild climb/descent.

## Repair Path

If horizontal altitude drift is still too large, use:

```bash
JAX_PLATFORMS=cuda MPLCONFIGDIR=/tmp WANDB_MODE=offline python run_vertical_energy_balanced_v2.py --config paper/second_paper/altitude_retention_repair_config.json
```

That config raises original replay to 40%, strengthens altitude retention, and disables 60/90 progression. It is for restoring balanced behavior before any harder vertical arc work.
