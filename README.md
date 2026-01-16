# Planax: A JAX‑Accelerated Fixed‑Wing Multi‑Agent RL Platform

*A lightweight, GPU‑friendly benchmark for high‑fidelity fixed‑wing dynamics, large‑scale MARL, and hierarchical self‑play.*

---

## Features

* **True six‑DOF aircraft & missile dynamics** implemented in pure JAX for just‑in‑time compilation and vectorisation.
* **Gymnax‑style environments** with built‑in support for thousands of parallel roll‑outs on a single GPU.
* **Ready‑to‑use PPO / MAPPO baselines** (single‑agent, multi‑agent, self‑play, hierarchical).
* **Tacview‑compatible replay exporter** for 3‑D debriefing and qualitative analysis.
* **One‑click reproducibility** via the locked `env_min.yml` Conda environment.

---

![Code structure](assets/code_structure.png)

## Installation

```shell
# 1. Clone repository
git clone https://github.com/xuecy22/AeroPlanax.git
cd AeroPlanax

# 2. Create research environment (CUDA 12 example)
conda env create -f env_min.yml
conda activate NeuralPlanex
```

> `env_min.yml` lists every runtime dependency used in the paper, including `jax‑cuda12‑pjrt`, `flax`, `optax`, `gymnax`, `tacview‑logger`, etc. Swap CUDA versions freely as long as **JAX ≥ 0.4.35** is available.

---

## Directory Layout

| Path           | Purpose                                                       |
| -------------- | ------------------------------------------------------------- |
| `dynamics/`    | Six‑DOF aircraft models                             |
| `interpolate/` | Trilinear & spline lookup for aero tables                     |
| `envs/`        | Gymnax‑compatible tasks (`heading`, `formation`, `combat`, …) |
| `train_*.py`   | PPO / MAPPO baselines                                         |
| `render_*.py`  | Offline & real‑time Tacview exporters                         |
| `assets/`      | Figures & GIFs for the README                                 |

---

## Environments

### 1. Heading (single‑agent)

* **Scenario** – one aircraft receives a desired course and must stabilise its attitude while aligning with that heading as quickly as possible.
* **Observations** – the full 16‑dimensional flight state (body‑rates, Euler angles, quaternion, velocity components, angle‑of‑attack, sideslip, altitude e.t.).
* **Actions** – a 4‑way discrete set `[δ_a, δ_e, δ_t, δ_r]` surface commands.
* **Reward** – negative absolute heading error with small penalties for attitude deviation and control effort.
* **Termination** – episode ends after 300 simulation steps(30 s), or instantly if the aircraft stalls, exceeds load limits, or hits the ground.

![Heading task demo](assets/heading.gif)

### 2. Formation

* **Scenario** – form and maintain wedge, line, or diamond spacing while avoiding mid‑air collisions.
* **Observations** – own flight state as above plus relative position/velocity to the virtual slot and the nearest neighbours.
* **Actions** – identical interface to Heading.
* **Reward** – quadratic distance to slot, collision penalty, shape‑keeping bonus, and control cost.
* **Termination** – collision, ground impact, or maximum episode length.

![Formation task demo](assets/formation.gif)

### 3. End‑to‑End Combat (self‑play / vs‑baseline)

* **Scenario** – symmetric dog‑fight ranging from 1 v 1 to 50 v 50. Each agent runs a single end‑to‑end policy that outputs manoeuvres plus missile‑launch commands.
* **Observations** – ego flight state, bearing/range/closure rate of visible opponents, missile inventory, line‑of‑sight angles, and basic fuel information.
* **Actions** – four continuous control surfaces plus a `fire_msl` Boolean.
* **Reward** – +1 for a kill, −1 for being killed, shaping for nose‑on position, energy management, and weapon economy.
* **Termination** – all aircraft on one side destroyed, self‑crash, or a 20 k‑step timeout.

### 4. Hierarchical Combat (self‑play / vs‑baseline)

* **Scenario** – identical arena to End‑to‑End, but each agent is governed by a two‑level policy: a high‑level planner outputs target heading / altitude / speed, while a shared low‑level controller (pre‑trained on Heading) tracks those commands.
* **Observations (high‑level)** – coarse situational awareness vectors (bandit angles, missile cues, remaining fuel, etc.).
* **Actions (high‑level)** – continuous `[Δψ_cmd, h_cmd, v_cmd]` guidance commands.
* **Reward** – same combat‑outcome terms, plus an imitation bonus favouring smooth, feasible guidance.
* **Advantages** – faster learning, clearer long‑horizon credit assignment, and the ability to swap different guidance laws with minimal retraining.

![Hierarchical Combat (self‑play) task demo](assets/5v5_hierarchy.gif)

---

## Quick Start


### Parameter Overview

#### 1 . Environments (aeroplanax_* TaskParams)

##### 1.1 Heading (`envs/aeroplanax_heading.py`) – 20 fields
| # | Field | Default | Description |
|---|-------|---------|-------------|
| 1 | num_allies | 1 | Number of friendly aircraft |
| 2 | num_enemies | 0 | Number of enemy aircraft |
| 3 | num_missiles | 0 | Missile inventory per agent |
| 4 | agent_type | 0 | Aircraft type index |
| 5 | action_type | 1 | Action space type (0 continuous, 1 discrete) |
| 6 | formation_type | 0 | Start formation (0 wedge, 1 line, 2 diamond) |
| 7 | sim_freq | 50 | Main simulation frequency (Hz) |
| 8 | agent_interaction_steps | 10 | Physics sub-steps per decision |
| 9 | max_altitude | 9000 | Maximum target altitude (m) |
| 10 | min_altitude | 4200 | Minimum target altitude (m) |
| 11 | max_vt | 360 | Maximum true airspeed (knots) |
| 12 | min_vt | 120 | Minimum true airspeed (knots) |
| 13 | max_heading_increment | π | Max random heading change (rad) |
| 14 | max_altitude_increment | 2100 | Max random altitude change (m) |
| 15 | max_velocities_u_increment | 100 | Max random speed change (knots) |
| 16 | safe_altitude | 4.0 | Safe-altitude threshold (km) |
| 17 | danger_altitude | 3.5 | Danger-altitude threshold (km) |
| 18 | noise_scale | 0 | State-noise scale |
| 19 | team_spacing | 15000 | Longitudinal spacing inside team (m) |
| 20 | safe_distance | 3000 | Minimum aircraft-to-aircraft distance (m) |

##### 1.2 Formation (`envs/aeroplanax_formation.py`) – 12 fields
| # | Field | Default | Description |
|---|-------|---------|-------------|
| 1 | num_allies | 2 | Friendly aircraft in formation |
| 2 | num_enemies | 0 | Enemy aircraft |
| 3 | agent_type | 0 | Aircraft type |
| 4 | action_type | 0 | Action space (continuous) |
| 5 | formation_type | 0 | Desired formation |
| 6 | max_altitude | 6000 | Initial altitude upper bound (m) |
| 7 | min_altitude | 5800 | Initial altitude lower bound (m) |
| 8 | max_vt | 360 | Max speed (knots) |
| 9 | min_vt | 300 | Min speed (knots) |
| 10 | noise_scale | 0 | State-noise scale |
| 11 | team_spacing | 15000 | Longitudinal spacing (m) |
| 12 | safe_distance | 3000 | Minimum separation (m) |

##### 1.3 Re-Formation (`envs/aeroplanax_reformation.py`) – 19 fields
| # | Field | Default | Description |
|---|-------|---------|-------------|
| 1 | num_allies | 5 | Friendly aircraft |
| 2 | num_enemies | 0 | Enemy aircraft |
| 3 | agent_type | 0 | Aircraft type |
| 4 | action_type | 1 | Action space (discrete) |
| 5 | sim_freq | 50 | Simulation frequency |
| 6 | agent_interaction_steps | 10 | Physics sub-steps |
| 7 | noise_scale | 0 | State-noise scale |
| 8 | global_topK | 1 | Nearest neighbours in global obs |
| 9 | ego_topK | 1 | Nearest neighbours in ego obs |
| 10 | formation_type | 0 | Target formation |
| 11 | max_altitude | 6000 | Altitude upper bound (m) |
| 12 | min_altitude | 5800 | Altitude lower bound (m) |
| 13 | max_vt | 360 | Speed upper bound (knots) |
| 14 | min_vt | 300 | Speed lower bound (knots) |
| 15 | team_spacing | 15000 | Longitudinal spacing (m) |
| 16 | safe_distance | 2000 | Minimum separation (m) |
| 17 | max_xy_increment | 555 | Max random XY offset (m) |
| 18 | max_z_increment | 555 | Max random Z offset (m) |
| 19 | max_communicate_distance | 20000 | Max comms range (m) |

##### 1.4 Combat (`envs/aeroplanax_combat.py`) – 26 fields
| # | Field | Default | Description |
|---|-------|---------|-------------|
| 1 | num_allies | 1 | Friendly aircraft |
| 2 | num_enemies | 1 | Enemy aircraft |
| 3 | num_missiles | 0 | Missile inventory |
| 4 | agent_type | 0 | Aircraft type |
| 5 | action_type | 1 | Action space type |
| 6 | observation_type | 0 | Observation mode |
| 7 | unit_features | 6 | Features per other unit |
| 8 | own_features | 9 | Own-state feature length |
| 9 | formation_type | 0 | Start formation |
| 10 | max_steps | 100 | Episode length |
| 11 | sim_freq | 50 | Simulation frequency |
| 12 | agent_interaction_steps | 10 | Physics sub-steps |
| 13 | use_artillery | False | Enable cannon model |
| 14 | max_altitude | 6000 | Altitude ceiling (m) |
| 15 | min_altitude | 6000 | Altitude floor (m) |
| 16 | max_vt | 240 | Max speed (knots) |
| 17 | min_vt | 240 | Min speed (knots) |
| 18 | safe_altitude | 4.0 | Safe-altitude threshold (km) |
| 19 | danger_altitude | 3.5 | Danger-altitude threshold (km) |
| 20 | max_distance | 5600 | Max weapon range (m) |
| 21 | min_distance | 5600 | Min weapon range (m) |
| 22 | team_spacing | 600 | Initial spacing (m) |
| 23 | safe_distance | 100 | Minimum separation (m) |
| 24 | posture_reward_scale | 100.0 | Posture-reward scale |
| 25 | use_baseline | False | Use baseline AI |
| 26 | use_hierarchy | False | Use hierarchical control |

##### 1.5 Combat-with-Missile (`envs/aeroplanax_combat_with_missile.py`) – 19 fields
| # | Field | Default | Description |
|---|-------|---------|-------------|
| 1 | num_allies | 1 | Friendly aircraft |
| 2 | num_enemies | 0 | Enemy aircraft |
| 3 | num_missiles | 1 | Missile inventory |
| 4 | agent_type | 0 | Aircraft type |
| 5 | action_type | 0 | Continuous actions |
| 6 | formation_type | 0 | Start formation |
| 7 | max_steps | 100 | Episode length |
| 8 | sim_freq | 50 | Simulation frequency |
| 9 | agent_interaction_steps | 50 | Physics sub-steps |
| 10 | max_altitude | 6000 | Altitude ceiling (m) |
| 11 | min_altitude | 5800 | Altitude floor (m) |
| 12 | max_vt | 360 | Max speed (knots) |
| 13 | min_vt | 300 | Min speed (knots) |
| 14 | max_heading_increment | π | Max heading delta (rad) |
| 15 | max_altitude_increment | 0 | Max altitude delta (m) |
| 16 | max_velocities_u_increment | 0 | Max speed delta (knots) |
| 17 | noise_scale | 0 | State-noise scale |
| 18 | team_spacing | 15000 | Longitudinal spacing (m) |
| 19 | safe_distance | 3000 | Minimum separation (m) |

#### 2 . Training Scripts (train_*) — example `train_heading_discrete.py` (24 fields)
| # | Field | Default | Description |
|---|-------|---------|-------------|
| 1 | GROUP | "heading" | wandb group/name |
| 2 | SEED | 42 | Random seed |
| 3 | LR | 3e-4 | Learning rate |
| 4 | NUM_ENVS | 300 | Parallel environments |
| 5 | NUM_ACTORS | 1 | Agents per environment |
| 6 | NUM_STEPS | 3000 | Steps per update |
| 7 | TOTAL_TIMESTEPS | 1e9 | Total training steps |
| 8 | FC_DIM_SIZE | 128 | FC layer width |
| 9 | GRU_HIDDEN_DIM | 128 | GRU hidden size |
| 10 | UPDATE_EPOCHS | 16 | PPO epochs per update |
| 11 | NUM_MINIBATCHES | 5 | Mini-batches per epoch |
| 12 | GAMMA | 0.99 | Discount factor |
| 13 | GAE_LAMBDA | 0.95 | GAE λ |
| 14 | CLIP_EPS | 0.2 | PPO clip ε |
| 15 | ENT_COEF | 1e-3 | Entropy coefficient |
| 16 | VF_COEF | 1 | Value-loss coefficient |
| 17 | MAX_GRAD_NORM | 2 | Gradient-clipping norm |
| 18 | ACTIVATION | "relu" | Activation function |
| 19 | ANNEAL_LR | False | Linear LR decay |
| 20 | DEBUG | True | Enable TensorBoard |
| 21 | OUTPUTDIR | results/{dt} | Output directory |
| 22 | LOGDIR | {OUTPUTDIR}/logs | Log directory |
| 23 | SAVEDIR | {OUTPUTDIR}/checkpoints | Checkpoint directory |
| 24 | LOADDIR | None | Pre-trained weights (opt.) |

#### 3 . Rendering Scripts (render_*) — 18 fields
Same schema as training scripts; typical values:
| Field | Default | Description |
|-------|---------|-------------|
| SEED | 42 | Random seed |
| NUM_ENVS | 1 | Single environment instance |
| NUM_ACTORS | 1 – 2 | Depends on scenario |
| FC_DIM_SIZE | 128 | Network width |
| GRU_HIDDEN_DIM | 128 | GRU size |
| LOADDIR | ./envs/models/baseline | Path to saved policy |
| … | … | All other optimiser & PPO fields identical to training |

> The overview covers 120 distinct parameters across all environment, training, and rendering modules.


### Training

```shell
# single‑agent heading task (≈ 3 hours on one GPU)
python train_heading_discrete.py

# wedge / line / diamond formation (≈ 3 hours on one GPU)
python train_reformation.py

# num‑vs‑num self‑play combat task with hierarchical control (≈ 3 hours on one GPU)
python train_combat_selfplay_hierarchy.py

# num‑vs‑num self‑play combat task with end-to-end control (≈ 3 hours on one GPU)
python train_combat_selfplay.py

# num‑vs‑num vs-baseline combat task with hierarchical control (≈ 3 hours on one GPU)
python train_combat_vsbaseline_hierarchy.py

# num‑vs‑num vs-baseline combat task with end-to-end control (≈ 3 hours on one GPU)
python train_combat_vsbaseline.py

# 
```
The meanings of some common modifiable parameters are as follows.
- `NUM_ENVS` The number of parallel environments.
- `NUM_ACTORS` The number of agents in each environment.
- `NUM_STEPS` The number of trajectory steps collected by each environment before each update.
- `TOTAL_TIMESTEPS` The total number of steps in the entire training process.
- `OUTPUTDIR` Output directory, used to save various output files during the training process.
- `LOGDIR` Log directory, specifically designed to store training logs.
- `SAVEDIR` Model save directory, used to save model checkpoints during the training process.
- `LOADDIR` Directory path for loading pre trained models.

### Evaluation & Rendering

```shell
# single‑agent heading task
python render_heading_discrete.py

# wedge / line / diamond formation
python render_reformation.py

# num‑vs‑num self‑play combat task with hierarchical control
python render_combat_selfplay_hierarchy.py

# num‑vs‑num self‑play combat task with end-to-end control
python render_combat_selfplay.py

# num‑vs‑num vs-baseline combat task with hierarchical control
python render_combat_vsbaseline_hierarchy.py

# num‑vs‑num vs-baseline combat task with end-to-end control
python render_combat_vsbaseline.py
```

This will generate a `*.acmi` file. We can use [**TacView**](https://www.tacview.net/), a universal flight analysis tool, to open the file and watch the render videos.


## Citation

```bibtex
@inproceedings{Planax2025,
  title     = {Planax: A JAX-Based Platform for Efficient and Scalable Multi-Agent Reinforcement Learning in Fixed-Wing Aircraft Systems},
  author    = {Qihan Liu and Chuanyi Xue and Qinyu Dong},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

---

## License

Planax is released under the MIT License. See `LICENSE` for details.
