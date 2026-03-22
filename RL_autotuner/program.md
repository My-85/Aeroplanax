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

### Phase 2b reward 逻辑修改指导
推荐的改造方向（按优先级排序）：
1. **多尺度 Gaussian**：添加 coarse(60-90°) + fine(5-10°) 两个尺度的 Gaussian 加权和，让大角度有梯度、小角度有精度
2. **Progress reward**：奖励 θ 减小的方向（需要在 REWARD_CONFIG 里存 prev_theta 或从 state 中计算 delta）
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
   ```
   cat champion/champion_meta.json
   cat reward_config.json
   cat results.jsonl
   ```

2. **深度分析历史**：整理参数变化趋势，识别有效/无效方向，检查是否需要进入 Phase 2b

3. **提出修改假设**：基于分析，写出你的推理过程和预期效果

4. **编辑 reward_config.json**（Phase 2a）或 `quat_baseline_reward.py`（Phase 2b）

5. **Git commit**：
   ```
   git add reward_config.json  # 或 git add -A（如果改了 .py）
   git commit -m "experiment: <简短描述你的假设和推理>"
   ```

6. **运行实验**：
   ```
   CUDA_VISIBLE_DEVICES=1 python experiment_runner.py --mode manual-auto --budget 1e8 --description "<同样描述>" > run.log 2>&1
   ```
   注意：Phase 2 的训练可能需要更长时间来通过课程。如果 curriculum_level 还停在 0-1，可以考虑增加 budget 到 2e8。

7. **检查结果**：
   ```
   tail -5 results.jsonl
   tail -50 run.log
   grep "curriculum_level" run.log | tail -10
   ```
   新增关注：curriculum_level 是否在推进？如果一直停在 0，可能需要调低 sustained_on_target_steps 的阈值。

8. **判断结果**：
   - `status="keep"` → 新 champion！继续。
   - `status="discard"` → `git reset --hard HEAD~1`
   - `status="crash"` → `tail -50 run.log` 排查，修复后重试。

9. **回到步骤 1。**

## 关键参数说明

```
reward = (att_r ^ w_att) * (speed_r ^ w_speed)
att_r = exp(-(theta / theta_scale)^2)
speed_r = exp(-(delta_vt / speed_error_scale)^2)
```

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
