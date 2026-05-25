# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Planax 是一个基于 JAX 的 F-16 飞行动力学仿真与强化学习训练框架，用于特技机动学习。它通过 JAX 原生的三线性插值实现 NASA TP-1538 气动系数查表，并使用 gymnax 环境训练 PPO 策略（Actor-Critic + GRU/LSTM）。代码库还支持空战、编队飞行和航点跟踪等任务。

## 环境与常用命令

运行任何脚本前，先激活 conda 环境：

```bash
conda activate aeroplanax
```

### 训练

训练脚本遵循 `train_<任务名>.py` 命名规范。当前主要训练入口：

```bash
python train_quat_baseline_iter.py
```

其他活跃的训练脚本：`train_heading_pitch_V_discrete_rnn_new_critic_no_fc2_quaternion_version_add_roll_target.py`、`train_heading_pitch_V_discrete_lstm_new.py`。

所有训练默认使用 GPU（`CUDA_VISIBLE_DEVICES='0'`、`XLA_PYTHON_MEM_FRACTION='0.95'`），通过 wandb 记录日志，通过 orbax 保存 checkpoint。

### 渲染 / 评估

渲染脚本遵循 `render_<任务名>.py` 命名规范。它们加载训练好的 checkpoint 并生成 Tacview ACMI 文件用于可视化：

```bash
python render_heading_pitch_V_discrete.py
```

大多数渲染脚本硬编码了 checkpoint 路径——需要修改 `restore_path` 指向目标 orbax checkpoint 目录。

### 保真度验证（Planax vs JSBSim）

```bash
./experiments/run_validation.sh
```

在 4 个场景（平飞配平、升降舵双脉冲、协调转弯、正弦激励）下，使用相同开环控制序列对比 Planax 和 JSBSim 的轨迹。输出到 `results/fidelity_validation_lef_only_fix/`。

### 运行单个测试 / 调试脚本

```bash
python _debug_training_issue.py
python test.py
```

## 架构

### 两套动力学后端

1. **`dynamics/F16_jax/`** — 独立 JAX F-16 动力学（历史遗留或离线仿真用途）。包含 `F16Dynamics.py`（nlplant 六自由度方程）和 `hifi_F16_AeroData.py`（从 `.dat` 文件加载的气动系数查表）。

2. **`envs/core/simulators/fighterplane/`** — **当前活跃**的动力学后端，集成在 gymnax 环境中。`dynamics.py` 包含环境调用的 step 函数（`update()`）；`aero_data.py` 加载同样的 NASA TP-1538 气动数据，提供 JIT 编译的 `_Cx()`、`_Cz()`、`_Cm()` 等三线性插值函数。

两套后端内部均使用英制单位（ft, slug, lbf, rad），在与环境接口处转换为公制（m, m/s 等）作为观测。

### 另外两套动力学模块（PyTorch）

- **`interpolate/`** — 基于 PyTorch 的神经网络代理模型（`.pth` 文件），训练目标是拟合 NASA 气动系数查表。用于加速推理或作为可微动力学代理。
- **`dynamics/F16_torch/`** — F-16 动力学的另一个 PyTorch 实现，气动力以 PyTorch 神经网络建模。

### 环境架构

```
envs/
├── aeroplanax.py              # 基类 AeroPlanaxEnv（抽象类，兼容 JIT）
├── aeroplanax_<任务名>.py     # 各任务环境（heading_pitch_V, combat 等）
├── core/
│   ├── base_dataclass.py      # BasePlaneState, BaseControlState, BaseMissileState（Flax struct）
│   ├── utils.py               # check_crashed, check_collision, check_extreme_state 等
│   └── simulators/
│       └── fighterplane/      # F-16 动力学 step、LEF 自动调度、减速板
├── reward_functions/          # 模块化奖励函数库（每个文件导出一个函数）
├── termination_conditions/    # 模块化终止条件（坠毁、超时、极端状态等）
├── wrappers.py                # JaxMARLWrapper、LogWrapper，用于对接训练循环
└── utils/utils.py             # ENU/大地坐标转换、配置文件解析
```

**任务专属环境**（如 `aeroplanax_heading_pitch_V.py`）继承 `AeroPlanaxEnv` 并覆写：
- `_init_state()` — 初始飞行状态
- `_reset_task()` — 任务专属重置（目标航向、俯仰、速度）
- `_get_obs()` — 观测向量构建
- 在 `__init__` 中注册对应的奖励函数和终止条件

### 控制接口

- **连续动作**（action_type=0）：4 维 Box `[throttle, elevator, aileron, rudder]`，范围 [-1, 1]
- **离散动作**（action_type=1）：Dict 类型，`throttle: Discrete(31)`、`elevator: Discrete(41)`、`aileron: Discrete(41)`、`rudder: Discrete(41)`、`speed_brake: Discrete(5)` — 通过 `_decode_discrete_actions()` 解码为连续值

减速板（speed brake）是独立的控制通道。JSBSim 风格的 LEF 自动调度（commit `8db4ace`）会根据攻角和马赫数自动设置前缘襟翼角度。

### 训练循环模式

所有训练脚本遵循相同的 PPO 模式：
1. `ActorCriticRNN`（或 LSTM 变体）— Flax 模块，scanned GRU/LSTM + 每个动作维度独立 categorical head
2. `make_train()` — JIT 编译的训练 step，包含 GAE 优势估计、PPO clipped loss、value loss、entropy bonus
3. Rollout 收集 → 优势计算 → PPO 更新，循环 `num_updates` 次
4. 通过 orbax 保存 checkpoint，通过 wandb 记录指标

### 气动数据

`envs/core/simulators/fighterplane/data/` 和 `interpolate/data/` 中的 `.dat` 文件是 NASA TP-1538 系数表。它们遵循 Fortran 列优先（column-major）约定——当前的 reshape 顺序 `(DH, BETA, ALPHA)` **是正确的**（已与 NASA `f16_deq.f` 核验）。函数调用约定：3D 表为 `_Cx((el, beta, alpha))`，2D 表为 `_Cx_lef((beta, alpha))`。

### LEF Bug 修复（关键背景）

`envs/core/simulators/fighterplane/dynamics.py` 第 272-301 行曾存在一个严重 bug：LEF/a20/r30 修正项的参数顺序错误（alpha/beta 颠倒，alpha 被错误地传入了 elevator 槽位）。修复将 `_Cx_lef((alpha, beta))` 改为 `_Cx_lef((beta, alpha))`，将 `_Cx((alpha, beta, 0))` 改为 `_Cx((0.0, beta, alpha))`。该修复已通过 NASA TP-1538 Fortran 原版代码验证，角速率与 JSBSim 的一致性提升了 3-30 倍。详见 `PLANAX_FINAL_FIX_REPORT.md`。

## Git 分支

- `main` — 远程主分支
- `jsbsim-lef-speedbrake-refactor` — **当前分支**，JSBSim 风格 LEF 自动调度和减速板控制重构
- `phase2b_add_obs` — 活跃的训练分支，扩展了观测空间
- `dynamics-correct-fix` — 经 NASA 验证的 LEF bug 修复（推荐合并）
- `dead-code-cleanup` — 清理了 12000+ 行死代码
