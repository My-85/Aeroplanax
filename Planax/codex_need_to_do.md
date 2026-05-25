Claude 已完成 loop geometry diagnosis。结论如下：

1. 新版 loop-plane target 是正确的：
   - target body_x 完全对齐 reference tangent；
   - target body_y 完全对齐 loop plane normal；
   - target frame 不是简单 roll flip。

2. 150° new 相比 old 有明显改进：
   - wing_plane_error 从 78.0° 降到 15.0°；
   - 说明飞机更接近真实 loop 平面；
   - 但综合 loop-quality 只是 B-grade，不是 A-grade。

3. 180° new 仍失败：
   - velocity_tangent_error=63.9°；
   - nose_tangent_error=62.6°；
   - wing_plane_error=77.5°；
   - env_alpha 范围约 [-42.1°, 24.2°]；
   - 说明底层 policy 在 80°-180° inverted/top-transition 区间无法稳定跟踪完整三轴姿态与速度切线。

因此，下一轮训练目标不是修 target，也不是单纯 overspeed control，而是：

half_loop_inverted_transition_finetune

# 起点

继续从当前主 baseline 开新分支：

results/vertical_energy_finetune/20260515_1615/checkpoint/checkpoint_epoch_619

不要从 epoch658 / epoch628 / epoch632 继续。

# 必须使用新版 loop-plane target

所有 >90° vertical arc / half-loop 任务必须使用：

experiments/hierarchical_trajectory_tracking/loop_attitude_target.py

不要再用旧 roll=0 target。

# 训练目标

重点训练 80°-180° 区间：

- inverted attitude tracking
- roll/pitch/heading coordinated tracking
- wing_plane_error reduction
- nose_tangent_error reduction
- velocity_tangent_error reduction
- nose_velocity_error / alpha suppression
- q_error_norm reduction

# 训练任务

## Stage 1: 80°-150° retention

保留 ep619 已有能力：

- 60° vertical arc
- 90° quarter loop
- 120° vertical arc
- 150° vertical arc

## Stage 2: inverted top-transition

新增短片段任务：

- 90° → 120°
- 120° → 150°
- 135° → 165°
- 150° → 175°
- 160° → 180°

## Stage 3: partial half-loop

逐步训练：

- 160° arc
- 165° arc
- 170° arc
- 175° arc
- 180° half-loop

半径优先：

- R=15000
- R=12000
- R=10000

# 保留任务

每轮必须保留：

- circle R3000 left/right
- circle R5000 left/right
- S-curve
- figure-eight
- climb/descent
- 15°/30° pull-up
- 60°/90°/150° loop-plane target arc

防止再次出现 epoch658 / epoch628 那种水平能力退化。

# Reward 重点

新增或加强：

- q_error reward
- roll tracking reward for >90° phases
- wing_plane_error penalty
- nose_tangent_error penalty
- velocity_tangent_error penalty
- nose_velocity_error / |alpha| penalty
- high-speed alpha coupling penalty
- action smoothness / saturation penalty
- crash penalty

不要只奖励 path progress。  
如果只奖励 path progress，policy 可能继续用“漂移过顶”的方式刷进度。

# 每轮训练预算

最多自动 5 轮，每轮短训：

- 0.75M - 1.0M timesteps
- LR = 1e-5 到 2e-5

每轮必须 eval，不要一次长训。

# 每轮评估

## Horizontal retention

- circle R3000 left/right
- circle R5000 left/right
- S-curve
- figure-eight

## Vertical retention

- 60°
- 90°
- 120°
- 150°

## New half-loop tasks

- 160°
- 165°
- 170°
- 175°
- 180°

每个任务必须输出：

- CTE
- velocity_tangent_error
- nose_tangent_error
- nose_velocity_error
- wing_plane_error
- env_alpha range
- q_error_norm
- target_roll vs actual_roll
- vt_min / vt_max
- Gmax
- crash / termination

# Promotion gate

只有满足以下条件才 promotion：

- 180° half-loop 至少 B-grade，或 175° 明显优于 ep619；
- 150° 不低于当前 B-grade；
- 60°/90° 不退化；
- circle / S-curve / figure-eight 不退化；
- wing_plane_error 明显下降；
- nose_tangent_error 明显下降；
- velocity_tangent_error 明显下降；
- alpha 峰值下降；
- no overload increase；
- no altitude drift regression。

如果只改善 180°，但水平任务退化，不能 promotion。

# 输出

输出：

results/half_loop_inverted_transition_search/YYYYMMDD_HHMM/
├── search_summary.csv
├── round_reports/
├── best_checkpoint_manifest.json
├── final_report.md
├── configs/
├── checkpoints/
└── plots/

final_report 必须回答：

1. 是否找到 half-loop-capable checkpoint？
2. 180° 是否达到 B/A？
3. 175° 是否明显优于 ep619？
4. wing_plane_error 是否下降？
5. nose_tangent_error 是否下降？
6. velocity_tangent_error 是否下降？
7. alpha 是否下降？
8. 60°/90°/150° 是否保持？
9. 水平轨迹是否保持？
10. 是否建议交给 Claude 做完整 ACMI 回归？