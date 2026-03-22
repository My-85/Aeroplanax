# Phase 2 完整任务：环境课程化升级 + 自动化训练框架改造

本任务分两个部分：
- **Part A**：代码改造（给 quat_baseline_iter 环境添加课程学习）
- **Part B**：自动化训练框架改造（继承 Phase 1 框架，更新 program.md、experiment_runner、evaluator，定义 Phase 2 的 agent 迭代权限和历史追踪机制）

---

## Part A：代码改造 — 给 quat_baseline_iter 添加课程学习

### 背景

我们有一个已训好的 baseline（当前 champion），在 `aeroplanax_quat_baseline_iter.py` 环境中通过 9 轮自动化 reward 参数调优，将 mean_theta_deg 从 24.88° 降到了 20.65°。

但这个 baseline 的训练域被 TaskParams 限制死了：
- max_heading_increment = π/2（±90°）
- max_pitch_increment = π/6（±30°）
- max_roll_increment = π/2（±90°）

实际测试显示：Level 0-2 表现很好（theta 2-7°），但大角度机动（H-90_P-30, full combo）失败（theta > 119°，delta_vt > 180 m/s）。

解决方案：在现有 env 上添加课程学习，逐步扩大训练域。这样可以直接 load 当前 champion checkpoint 继续训练。

### 万向死锁注意

pitch 在接近 ±90° 时欧拉角会出现万向死锁。但我们的 obs 使用的是四元数误差向量（`q_err_vec`）而非欧拉角观测，reward 也基于四元数测地角计算，所以理论上不受 gimbal lock 影响。但你在生成目标时需要注意：
1. 目标 pitch 不要设为精确的 ±90°（留个裕度，比如最大 ±89°）
2. `_quat_from_euler_bn()` 在 pitch=±90° 时数值上仍然是稳定的（cos(45°) 和 sin(45°) 不退化），但 `wrap_PI()` 和 heading 的意义在 pitch=±90° 时会混乱
3. 确保课程表的 pitch 上限不超过 ±89°

### 任务 A1：修改 `Planax/envs/aeroplanax_quat_baseline_iter.py`

**A1a. 修改 `Heading_Pitch_V_TaskState`，添加课程状态字段：**

在现有字段后添加：
```python
curriculum_level: ArrayLike           # int32, 当前课程级别 (0-5)
on_target_steps: ArrayLike            # int32, 连续达标步数
curriculum_success_counts: ArrayLike  # int32, 当前级别累计真实成功次数（超时不算）

create() 方法也要更新，新字段初始化为 0。

A1b. 修改 Heading_Pitch_V_TaskParams，添加课程参数：
curriculum_advance_threshold: int = 3
curriculum_advance_per_level: int = 1
sustained_on_target_steps: int = 3
sustained_on_target_per_level: int = 2

A1c. 修改 _step_task() 方法，实现课程逻辑：

参考 aeroplanax_full_domain_maneuver.py 的 _step_task()（行 470-631）。核心逻辑：

1.计算当前 theta_deg 和 delta_vt（用 env 内的四元数辅助函数）
2.on-target 判定：theta < 10° AND delta_vt < 25
3.更新 on_target_steps（连续达标+1，断了归零）
4.sustained success = on_target_steps >= threshold
5.区分 real_success（sustained tracking 达标）和 timeout（时间到但没达标）
6.只有 real_success 才累加 curriculum_success_counts 和推进 curriculum
7.timeout 仍然切换目标（让训练继续），但不推进课程
8.按 curriculum_level 查表选取目标范围（替代固定的 max_*_increment）
A1d. 课程表：
heading_limits = [π/2, 2π/3, π, π, π, π]                    # ±90° → ±120° → ±180°
pitch_limits   = [π/6, π/4, π/3, 5π/12, 89π/180, 89π/180]      # ±30° → ±45° → ±60° → ±89°
roll_limits    = [π/2, 2π/3, 5π/6, π, π, π]                  # ±90° → ±120° → ±150° → ±180°
speed_min      = [120, 100, 90, 80, 60, 60]
speed_max      = [360, 380, 400, 400, 400, 400]

Level 0 == 当前训练域，champion 一上来就在舒适区。

A1e. 绝对不能改的：

_get_obs() 方法（保持 16D obs）
_get_obs_size() （保持 return 16）
_init_state() 的初始化逻辑
reward_functions 列表
网络架构

任务 A2：修改 Planax/envs/termination_conditions/unreach_quat_baseline.py
当前 success = mask1（只检查时间，不检查达标），这让 agent 可以什么都不做等超时就"成功"。

修改为：区分 real success（sustained on-target）和 timeout。两者都触发目标切换（返回 success=True），但 _step_task 中会根据 on_target_steps 判断是否是 real success。所以 unreach 函数可以保持返回 timeout-based success 作为目标切换信号，但需要确保 _step_task 的 real_success 逻辑正确区分。

或者更好的做法：让 unreach 函数返回两种 success——参考 full_domain 的做法，在 _step_task 中自己判断 success 信号的类型。

任务 B3：重写 RL_autotuner/program.md — Phase 2 Agent SOP
这是最关键的部分。program.md 是 Claude Code CLI agent 的完整操作手册。

必须包含以下所有内容：
# RL Autotuner — Phase 2: Curriculum-Enhanced Reward Tuning

你是一个 RL reward 调优 agent。你的唯一目标是最小化 mean_theta_deg。

## 重大变更：Phase 2 vs Phase 1

Phase 2 的环境已添加课程学习机制：
- Level 0: heading ±90°, pitch ±30°, roll ±90°（= Phase 1 训练域）
- Level 1: heading ±120°, pitch ±45°, roll ±120°
- Level 2: heading ±180°, pitch ±60°, roll ±150°
- Level 3-5: heading ±180°, pitch ±89°, roll ±180°（全域）

agent 在训练中会自动从 Level 0 开始，当 on-target 达标次数满足阈值后自动升级。

你的 Phase 2 目标：**让 agent 尽快通过所有 curriculum level，并在全域范围内保持低 theta**。

## 实验历史的重要性

你必须在每次提出新实验前，认真阅读和分析 results.jsonl 中的全部历史记录。

### 历史记录格式
results.jsonl 中每行是一个 JSON，包含：
- experiment_id: 实验编号
- config_snapshot: 当次使用的 reward 参数
- metrics: 训练日志指标 + eval 指标
- status: "keep"（成为新 champion）/ "discard"（不如 champion）/ "crash"（训练失败）
- description: 你对这次修改的假设说明
- timestamp: 时间戳

### 分析要求
每次提出新实验时，你必须：
1. 读取完整的 results.jsonl（不只是 tail -20，如果有很多行就 cat 全部）
2. 整理出参数变化趋势表：哪个参数往哪个方向调、结果如何
3. 识别出：哪些方向有效（keep 了），哪些方向无效（discard 了），是否有参数振荡
4. 基于以上分析，形成明确的假设，再提出修改
5. 在 git commit message 和 --description 中写清你的分析推理过程，不能只写"试试看"（初次提交记得提交到新分支上）

### 禁止的行为
- 禁止不看历史就盲目调参
- 禁止重复已经被 discard 的相同修改
- 禁止同时改 3 个以上参数
- 禁止不写分析理由

## Phase 2 Agent 可修改范围

### Phase 2a（当前阶段）：仍然只改 reward_config.json

你只能修改 reward_config.json 中的数值参数：
- theta_scale_deg
- speed_error_scale
- w_att / w_speed

这和 Phase 1 完全一样。区别在于环境有了课程，agent 会在训练中遇到更大角度的目标。

### Phase 2b（当 2a 连续 5 次 discard 后自动进入）：开放 reward 逻辑修改

当 results.jsonl 中最近 5 次实验全部是 discard 时，你可以判定"参数空间搜索已饱和"。
此时你获得额外权限：修改 `Planax/envs/reward_functions/quat_baseline_reward.py` 的 `quat_baseline_reward_fn()` 函数逻辑。

Phase 2b 允许修改的内容：
- reward 计算公式（如添加 progress reward、settled bonus、速度变化率惩罚等）
- REWARD_CONFIG 字典中添加新的参数
- 但不允许修改函数签名（state, params, agent_id, reward_scale 四个参数不变）

Phase 2b 的保护：
- 每次修改 reward 逻辑前，必须 git commit 当前状态
- 修改后必须通过 dry-run 验证（CUDA_VISIBLE_DEVICES=1 python experiment_runner.py --mode dry-run）
- 必须通过 champion 比较：新 reward 训出的 theta 必须比 champion 更低
- 如果 discard，git reset --hard HEAD~1 回滚

### Phase 2b  reward 逻辑修改指导
推荐的改造方向（按优先级排序）：
1. **多尺度 Gaussian**：添加 coarse(60-90°) + fine(5-10°) 两个尺度的 Gaussian 加权和，
让大角度有梯度、小角度有精度
2. **Progress reward**：奖励 θ 减小的方向（需要在 REWARD_CONFIG 里存 prev_theta 或
从 state 中计算 delta）
3. **Settled bonus**：θ < 5° 时额外奖励，激励精确保持
4. **速度变化率惩罚**：抑制极端机动中的速度发散
### 约束
- 函数签名不变：(state, params, agent_id, reward_scale) → float
- reward 返回值范围 [0, 1]（或合理范围，不要爆到几百）
- REWARD_CONFIG 字典中可以添加新参数，config_patcher.py 会自动处理
- 每次修改必须 git commit + champion 比较


## 实验循环

LOOP FOREVER:

1. **读取当前状态**：
cat champion/champion_meta.json cat reward_config.json cat results.jsonl

2. **深度分析历史**：整理参数变化趋势，识别有效/无效方向，检查是否需要进入 Phase 2b

3. **提出修改假设**：基于分析，写出你的推理过程和预期效果

4. **编辑 reward_config.json**（Phase 2a）或 `quat_baseline_reward.py`（Phase 2b）

5.**Git commit**：
git add reward_config.json # 或 git add -A（如果改了 .py） git commit -m "experiment: <简短描述你的假设和推理>"

6.**运行实验**：
CUDA_VISIBLE_DEVICES=1 python experiment_runner.py --mode manual-auto --budget 1e8 --description "<同样描述>" > run.log 2>&1（这里device的使用可以在代码里面先写好，统一都用GPU 1）

注意：Phase 2 的训练可能需要更长时间来通过课程。如果 curriculum_level 还停在 0-1，可以考虑增加 budget 到 2e8。

7.**检查结果**：
tail -5 results.jsonl tail -50 run.log grep "curriculum_level" run.log | tail -10
新增关注：curriculum_level 是否在推进？如果一直停在 0，可能需要调低 sustained_on_target_steps 的阈值。

8. **判断结果**：
- `status="keep"` → 新 champion！继续。
- `status="discard"` → `git reset --hard HEAD~1`
- `status="crash"` → `tail -50 run.log` 排查，修复后重试。

9. **回到步骤 1。**

## 关键参数说明

reward = (att_r ^ w_att) * (speed_r ^ w_speed)
att_r = exp(-(theta / theta_scale)^2)
speed_r = exp(-(delta_vt / speed_error_scale)^2)

### 参数（Phase 2a 可调）
- theta_scale_deg (当前 30.0): 姿态 Gaussian 宽度
- speed_error_scale (当前 40.0): 速度 Gaussian 宽度
- w_att (当前 0.7): 姿态权重
- w_speed (当前 0.3): 速度权重

### Phase 2 新增考虑
因为有课程，agent 在高 level 会遇到 theta > 90° 的目标。此时 att_r 的梯度信号是否足够？
- 如果 theta_scale_deg=30°，当 theta=120° 时 att_r ≈ 0，几乎无梯度
- 可能需要将 theta_scale_deg 调大到 60-90°，或在 Phase 2b 中添加多尺度 reward

### 策略建议
1. 先用当前参数跑一轮，观察 curriculum_level 能推进到几
2. 如果 Level 0 就通过很快，说明 Phase 1 的 champion 已经掌握了这个范围
3. 如果在 Level 1-2 卡住，考虑加大 theta_scale_deg 让大角度下有梯度
4. 如果 curriculum 推进正常但 theta 不降，考虑调 w_att/w_speed 平衡
5. 如果连续 5 次 discard，进入 Phase 2b 修改 reward 逻辑

### 已知失败模式
1. theta_scale_deg 太小 → 大角度下 reward ≈ 0，课程推不上去
2. theta_scale_deg 太大 → 近目标处梯度太弱，精度上不去
3. speed 权重太低 → 高 level 速度发散
4. 训练 budget 不够 → curriculum 还没来得及推进就结束了

## 评估指标

**Primary: mean_theta_deg**（lower is better，唯一决定 keep/discard 的指标）
**Tiebreaker: mean_delta_vt**（lower is better）
**Safety: mean_crash_rate**（不应显著增加）
**监控: curriculum_level**（agent 推进到了哪个 level，越高说明在学更难的目标）

## 约束（SAFETY RULES）

- w_att + w_speed ≈ 1.0
- theta_scale_deg > 0
- speed_error_scale > 0
- Phase 2a: 只改 reward_config.json 数值
- Phase 2b: 可改 quat_baseline_reward.py 逻辑，但函数签名不变
- 永远不改 env 文件、train 脚本、evaluator.py、termination conditions
- 永远不改 obs 维度（必须保持 16D）
- 永远不改网络架构（必须保持 GRU=128, FC=128）

## NEVER STOP

实验循环开始后，不要暂停询问人类。人类可能不在电脑前，期望你持续工作直到被手动中断。

如果你用尽了想法：
- 重新阅读完整的 results.jsonl，寻找被忽略的规律
- 尝试组合两个接近成功的修改
- 尝试更激进的参数变化（2x-4x 而不是 20%）
- 检查 run.log 里的 curriculum_level 趋势，针对性调整
- 如果连续 5 次 discard，进入 Phase 2b

每个实验大约 30-90 分钟（课程训练可能更长）。保持运行。

任务 B4：在 results.jsonl 中增强历史记录
修改 experiment_runner.py 的 log_result() 函数，确保每条记录包含：
record = {
    "experiment_id": experiment_id,
    "config_snapshot": config,          # 当次的 reward 参数
    "metrics": {
        # 训练日志指标
        "final_theta_deg": ...,
        "final_episodic_return": ...,
        "final_curriculum_level": ...,  # 新增：训练结束时到达的 level
        "total_env_steps": ...,
        # 正式评估指标（如果有）
        "eval": {
            "mean_theta_deg": ...,
            "mean_delta_vt": ...,
            "mean_crash_rate": ...,
            "mean_on_target_rate": ...,
        }
    },
    "status": status,
    "description": description,         # agent 的分析推理过程
    "timestamp": ...,
}

这确保了 agent 在后续轮次能看到完整的决策历史。
文件路径：
/home/dqy/aeroplanax/new/20251215最新代码库/
├── Planax/
│   ├── envs/
│   │   ├── aeroplanax_quat_baseline_iter.py      ← Part A 主要修改
│   │   ├── aeroplanax_full_domain_maneuver.py     ← 参考（课程逻辑模板，只读）
│   │   ├── reward_functions/
│   │   │   └── quat_baseline_reward.py            ← Phase 2a 不改，Phase 2b 可改
│   │   └── termination_conditions/
│   │       └── unreach_quat_baseline.py           ← Part A 修改
│   └── train_quat_baseline_iter.py                ← Part A 小改（日志增强）
├── RL_autotuner/
│   ├── evaluator.py                               ← Part B 适配
│   ├── experiment_runner.py                       ← Part B 适配
│   ├── config_patcher.py                          ← 不改
│   ├── program.md                                 ← Part B 重写
│   ├── reward_config.json                         ← 不改（agent 后续在迭代中改）
│   ├── champion/champion_meta.json                ← 不改（保留 Phase 1 champion）
│   ├── results.jsonl                              ← 不改（Phase 2 继续追加）
│   └── .backups/                                  ← 不改（保留所有历史）

Champion 信息
Checkpoint: /home/dqy/aeroplanax/new/20251215最新代码库/Planax/results/heading_pitch_V_discrete_rnn_2026-03-20-19-38/checkpoints/checkpoint_epoch_1350
网络: obs=16D, GRU=128, FC=128, action=[31,41,41,41]
reward config: theta_scale_deg=30.0, speed_error_scale=40.0, w_att=0.7, w_speed=0.3
评估: mean_theta_deg=20.65°, mean_delta_vt=16.27

安全检查清单
完成所有修改后验证：
1.cd /home/dqy/aeroplanax/new/20251215最新代码库/Planax && python -c "from envs.aeroplanax_quat_baseline_iter import AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams; print('OK')"
2.python -c "from envs.termination_conditions import unreach_quat_baseline_fn; print('OK')"
3.确认 Level 0 课程范围 == heading ±90°, pitch ±30°, roll ±90°（即当前 champion 的训练域）
4.确认 obs 维度仍然是 16D
5.cd /home/dqy/aeroplanax/new/20251215最新代码库/RL_autotuner && CUDA_VISIBLE_DEVICES=1 python evaluator.py --random-baseline
6.CUDA_VISIBLE_DEVICES=1 python experiment_runner.py --mode dry-run
7.Git commit 所有修改：git add -A && git commit -m "phase2: add curriculum to quat_baseline_iter, update autotuner framework"（新阶段的初次提交记得新开一个分支）
工作优先级：
1.先改 aeroplanax_quat_baseline_iter.py（课程机制，核心）
2.再改 unreach_quat_baseline.py（终止条件配套）
3.更新 train_quat_baseline_iter.py（日志增强）
4.适配 evaluator.py（新 TaskState）
5.适配 experiment_runner.py（日志正则、历史记录增强）
6.重写 program.md（Phase 2 完整 SOP）
7.运行安全检查
8.Git commit
注意：完成代码改造后，不要启动自动化训练循环。只做代码修改和验证，让人类确认后再启动。