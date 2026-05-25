# Codex 工作 Prompt：底层 Quaternion PPO 的垂直能量技能补强

## 0. 你的角色

你是新加入该项目的代码助手。你没有此前上下文，因此请认真阅读本文档。

你的任务不是做上层轨迹抽象，而是负责 **底层 quaternion PPO baseline 的能力补强**，重点解决：

> 当前底层 baseline 能完成水平机动，但在小半径 pull-up / 垂直能量机动中速度掉落严重，无法支撑筋斗圆和全域特技机动。

最终目标：

```text
全域机动
包括筋斗圆
五种代表性特技机动
```

你要做的是：

```text
现有 quaternion baseline
→ 诊断垂直 pull-up 失败原因
→ targeted fine-tuning
→ 补强 pitch / climb / energy management skill
→ 产出可接入上层 planner 的新 checkpoint
```

---

## 1. 项目背景

Planax 是一个高保真固定翼 RL 仿真平台，包含：

- F-16 非线性 6-DOF 动力学；
- NASA 气动数据；
- JAX/XLA 并行仿真；
- 多种 RL baseline；
- 当前任务关注单机固定翼特技机动控制。

---

## 2. 当前 Quaternion Baseline 总结

### 2.1 核心设计

当前 baseline 是一个基于四元数姿态误差的 PPO 策略。

输入目标：

```text
target_heading
target_pitch
target_roll
target_vt
```

环境将它们编码为观测中的：

```text
qv: 当前姿态与目标姿态之间的四元数误差向量部
v_b: 机体系目标方向
dvt: 速度误差
```

策略输出离散舵面指令：

```text
throttle
elevator
aileron
rudder
speedbrake
```

### 2.2 关键设计决策

当前 baseline 已经做过一些重要改进：

1. 配平初始化：
   - roll=0；
   - pitch=0；
   - vt=250；
   - alt=5000；
   - 显著降低初期坠毁。

2. 混合采样课程：
   - heading / pitch / roll / vt 四轴独立采样；
   - L0-L3 难度档位；
   - 概率随 heading_turn_counts 渐进偏移；
   - L0 永远 ≥5%。

3. earned-only 课程推进：
   - 只在 `theta < 5°` 且 `ΔV < 15 m/s` 时递增计数器；
   - 防止 safety timeout 误驱动课程升级。

4. 自适应检查间隔：
   - L0=55步；
   - L1=120步；
   - L2=210步；
   - L3=250步。

5. 观测维度：
   - 21 维；
   - 16 维姿态与飞参；
   - 5 维上一步动作。

6. reward：
   - 姿态跟踪 reward，scale=2.0；
   - 高度惩罚；
   - 坠毁惩罚 -200；
   - G 软惩罚，coef=0.05，clip=5.0。

---

## 3. 重要文件

当前相关文件：

```text
envs/aeroplanax_heading_pitch_V_quaternion_version_add_full_roll.py
    当前 quaternion baseline 训练环境

envs/termination_conditions/unreach_heading_pitch_V_quat.py
    自适应切换 / 任务终止条件

envs/reward_functions/heading_pitch_V_reward_add_roll_target.py
    姿态跟踪 reward

envs/reward_functions/reward_nz_soft_penalty.py
    G 软惩罚 reward

reward不止这两个，还有别的，请你根据env去查看

train_heading_pitch_V_discrete_rnn_new_critic_no_fc2_quaternion_version_add_roll_target.py
    当前训练脚本
```

当前 checkpoint：

```text
results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600
```

代码备份：

```text
envs/models/baseline_quat_20260514/
```

请不要破坏上述文件。  
请复制新文件做实验。

---

## 4. 当前能力边界

### 4.1 已经成功的任务

当前 frozen baseline + 上层 lookahead planner 已经可以完成：

- 直线多航点；
- 水平圆，R=8000 / 5000 / 3000 / 2000 m；
- S-curve；
- Figure-eight；
- 15° 大半径 pull-up，R=8000 m。

### 4.2 当前失败的任务

小半径 pull-up 失败：

| 半径 | 结果 | 现象 |
|---|---|---|
| R=8000 m | 成功 | 15° pull-up，CTE≈72 m |
| R=5000 m | 边界可行 | 质量一般 |
| R=3000 m | 失败 | 速度下降明显 |
| R=2000 m | 失败 | vt_min ≈147 m/s，能量不足 |

Energy compensation 测试：

| target_vt | 结果 |
|---|---|
| 250 | 失败 |
| 280 | 失败 |
| 300 | crash 更快 |

结论：

> 失败主因不是 target 生成错误，也不是 pitch 完全跟踪不上，而是 pull-up 过程中能量/速度保持不足。简单提高 target_vt 无法解决。

---

## 5. 你的总体目标

请在不破坏当前 baseline 的前提下，开发一个 **vertical energy fine-tuning 版本**。

目标能力：

```text
positive pitch ramp tracking
straight climb
large-radius pull-up
medium-radius pull-up
vertical arc
half loop
full loop
```

但不要直接从 full loop 开始。

---

## 6. 总体实施路线

请按以下顺序：

```text
Step 1: 复制当前 env 和 train 脚本，建立 vertical_energy 版本
Step 2: 复现当前 checkpoint 加载和短 rollout
Step 3: 实现诊断任务环境：pitch ramp / climb / pull-up arc
Step 4: 修改 reward，加入 energy-aware 项
Step 5: 设计 vertical curriculum
Step 6: 从现有 checkpoint fine-tune
Step 7: 评估新 checkpoint 是否改善 pull-up boundary
```

---

## 7. Task 1：复制并建立新文件

不要直接改原文件。请复制为新版本：

```text
envs/aeroplanax_heading_pitch_V_quaternion_version_vertical_energy.py

envs/reward_functions/heading_pitch_V_reward_vertical_energy.py

envs/termination_conditions/unreach_heading_pitch_V_quat_vertical_energy.py

train_heading_pitch_V_discrete_rnn_quaternion_vertical_energy_finetune.py
```

保留原始文件不变。

---

## 8. Task 2：复现当前 baseline

先做最小 sanity check：

1. 加载当前 checkpoint epoch_600；
2. 在新 env 中跑：
   - level flight；
   - heading step ±20°；
   - pitch step ±10°；
3. 确保新 env 没有破坏原 baseline 行为。

输出：

```text
reward
termination
vt
altitude
pitch tracking
action distribution
```

---

## 9. Task 3：实现诊断任务类型

请在新 env 或 task params 中支持 task mode：

```text
level_attitude
pitch_ramp
straight_climb
pullup_arc
vertical_arc
```

### 9.1 pitch_ramp

测试：

```text
target_pitch: 0° → +5°
target_pitch: 0° → +10°
target_pitch: 0° → +15°
target_pitch: 0° → +20°
target_pitch: 0° → -5°
target_pitch: 0° → -10°
target_pitch: 0° → -15°
target_pitch: 0° → -20°
```

保持：

```text
target_heading = current heading
target_roll = 0
target_vt = 250
```

### 9.2 straight_climb

测试：

```text
climb angle = +5°, +10°, +15°
descent angle = -5°, -10°
```

### 9.3 pullup_arc

测试：

```text
15° pull-up
30° pull-up
R = 10000, 8000, 5000, 3000, 2000
```

先大半径，后小半径。

---

## 10. Task 4：新增 energy-aware reward

当前 reward 主要是姿态跟踪，需要加入能量管理项。

推荐 reward：

```text
r_total =
    w_att      * r_attitude_tracking
  + w_vt       * r_speed_tracking
  + w_energy   * r_energy_preservation
  + w_climb    * r_climb_progress
  + w_alpha    * r_alpha_safety
  + w_beta     * r_beta_safety
  + w_g        * r_g_safety
  + w_smooth   * r_action_smoothness
  + w_alive    * r_alive
  + w_crash    * r_crash
```

### 10.1 Speed tracking

重点惩罚低速：

```text
vt_target = target_vt
low_speed_threshold = 180 m/s
strong penalty when vt < 170 m/s
```

不要只用对称速度误差。pull-up 中低速更危险。

### 10.2 Energy preservation

可以使用简化机械能 proxy：

```text
E = 0.5 * vt^2 + g * altitude
```

奖励 pull-up 中能量不过快下降。

### 10.3 Climb progress

对于 pull-up / climb 任务，奖励高度或 path progress 增加，但不能牺牲速度。

### 10.4 Alpha / G safety

惩罚：

```text
alpha > 15° 或 18°
G > 9 或 10
```

注意：不要让 G 惩罚过强导致 policy 不敢拉起。

### 10.5 Action smoothness

惩罚动作变化过大，尤其 elevator / throttle。

---

## 11. Task 5：设计 vertical curriculum

请设计课程，不要直接 full loop。

推荐：

```text
Stage 0: level / small attitude tracking 回放
Stage 1: pitch ramp ±5°, ±10°
Stage 2: pitch ramp +15°, +20°
Stage 3: straight climb 5°, 10°
Stage 4: 15° pull-up R=10000, 8000
Stage 5: 15° pull-up R=5000, 3000
Stage 6: 30° pull-up R=10000, 8000
Stage 7: 30° pull-up R=5000, 3000
Stage 8: 60° vertical arc
Stage 9: 90° quarter loop
Stage 10: half loop / full loop
```

必须保留一部分原始 attitude tracking 任务，避免遗忘：

```text
20% original mixed heading/pitch/roll/vt tasks
80% vertical energy tasks
```

---

## 12. Task 6：fine-tuning 策略

请从当前 checkpoint fine-tune：

```text
results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600
```

不要从零训练。

建议：

- learning rate 降低到原来的 1/3 或 1/5；
- 保持 PPO 稳定配置；
- 开启 grad clipping；
- 定期保存 checkpoint；
- 每隔固定 epoch 跑 evaluation suite。

第一轮 debug training 不要太大：

```text
total timesteps = 5e6 到 2e7
num_envs 根据机器资源设置
```

确认 reward 上升、crash 下降后再扩大。

---

## 13. Task 7：evaluation suite

每个 checkpoint 都评估：

### 13.1 保留原技能

```text
heading step ±20°, ±45°
pitch step ±10°
roll target small
level circle R=5000
S-curve A=3000
```

### 13.2 新增垂直技能

```text
pitch ramp +10°, +15°, +20°
straight climb +5°, +10°
15° pull-up R=8000, 5000, 3000, 2000
30° pull-up R=10000, 8000, 5000
```

### 13.3 关键指标

输出：

```text
success
settling time
pitch tracking error
vt_min
energy loss
altitude gain
alpha_max
Gmax
crash_rate
action_saturation
```

重点看：

```text
R=3000 / R=2000 的 15° pull-up 是否改善
30° pull-up 是否可行
原有水平任务是否退化
```

---

## 14. Task 8：输出报告

每次训练后输出：

```text
results/vertical_energy_finetune/YYYYMMDD_HHMM/
├── config.yaml 或 config.json
├── train_log.csv
├── eval_summary.csv
├── checkpoint/
├── plots/
└── report.md
```

`report.md` 必须回答：

1. 是否改善 15° pull-up R=3000 / R=2000？
2. 是否能完成 30° pull-up？
3. 速度最低值是否提高？
4. 能量损失是否下降？
5. alpha/G 是否可控？
6. 原始水平任务是否退化？
7. 是否可以进入 60° / 90° arc？
8. 下一轮训练建议。

---

## 15. 禁止事项

不要做：

- 直接修改原 baseline 文件；
- 从 full loop 开始训练；
- 只奖励 pitch 而不管速度；
- 盲目提高 target_vt；
- 忽略原任务遗忘；
- 不评估原始水平任务就宣布成功；
- 不保存 checkpoint 和 config。

---

## 16. 当前阶段最终目标

本阶段目标不是立即完成全域特技，而是补齐底层能力缺口：

```text
当前 baseline：
  水平机动强
  小半径 pull-up 弱

目标 fine-tuned baseline：
  保持水平机动能力
  提升 pull-up / climb / energy management
  支撑后续 vertical arc / loop / full-envelope maneuver
```

当新 checkpoint 能稳定完成：

```text
15° pull-up R=3000 或 R=2000
30° pull-up R=8000 / 5000
```

再交给上层 planner 接入五种代表性机动任务。
