# 深度分析报告：为什么训不出更好的 Champion

## 一、Champion #59 现状诊断

| Level | theta_deg | on_target | att_r | speed_r | 诊断 |
|-------|-----------|-----------|-------|---------|------|
| L0 | 29.57° | 19.1% | 0.592 | 0.466 | att_r 尚可，speed_r 偏低 |
| L1 | 45.49° | 12.5% | 0.488 | 0.373 | att_r 下降明显 |
| L2 | 63.48° | 11.5% | 0.384 | 0.367 | Gaussian 梯度开始饱和 |
| L3 | 59.28° | 5.4% | 0.331 | 0.318 | **att_r=0.33，梯度几乎消失** |
| L5 | 73.83° | 5.5% | 0.243 | 0.318 | **att_r=0.24，完全饱和** |

**核心问题**：L3-L5 的 att_r 只有 0.24-0.33，表明 Gaussian reward 在 theta>60° 时梯度趋近于零，agent 收不到有效学习信号。

## 二、为什么后续 8 轮迭代全部失败

1. **L0 精度极其脆弱**（29.57° 是刀尖上的平衡）：任何参数调整（coarse_w、w_att、新 scale）都会破坏 L0 精度，而 per-level early-exit 要求 ALL levels 都优于 champion
2. **训练步数不足**：Champion 训了 1.35B 步（epoch_1575），但后续每轮只有 500M 步（37%）。即使 reward 更好，也来不及收敛
3. **K=4 浪费算力**：4 个候选顺序训练，每个只分到 125M 步，更不够
4. **Reward 设计空间已接近极限**：dual-scale Gaussian + geometric product 已经是很好的设计，单靠微调参数很难突破

## 三、论文对比分析

| 论文 | 核心方法 | 我们缺什么 |
|------|----------|-----------|
| **Eureka** | K=16 采样 × 5 轮迭代，reward reflection（各分量训练曲线） | 我们的 reward reflection 已实现，但 K=4→K=1 后丧失多样性 |
| **DrEureka** | Reward + Domain Randomization 协同设计，RAPP 物理先验 | 我们没有 physics grounding——不知道各参数的有效范围 |
| **Agent²** | 双 agent 架构：Generator 分析 + Target 执行，覆盖 MDP+算法+超参全链路 | 我们只调 reward，没调网络架构/超参/观测空间 |
| **LearningFlow** | Analysis→Generation 两阶段 prompt，ε-greedy curriculum 防过拟合 | 我们的 prompt 没有分析-生成分离，curriculum 是固定的 |
| **Text2Touch** | 短验证 run（150M）+ 完整训练分开，代码简洁性指标 | 我们每轮 500M 全训太慢，没有快速验证阶段 |
| **RAPO** | 检索增强探索，entropy 作为不确定性信号 | 我们没有利用 action entropy 诊断 agent 在哪些 level 困惑 |

## 四、关键洞察：瓶颈在哪里

**不是 reward 的问题，是整体框架权限太窄。** 当前 Phase 2 只允许修改 reward 函数/参数，但真正的瓶颈在：

1. **网络容量**：GRU-128 只有 188K 参数，对 L5（theta>60°, 需要大角度机动）可能容量不足
2. **训练超参**：PPO 的 UPDATE_EPOCHS、LR schedule、GAE_LAMBDA 等直接影响收敛速度
3. **Curriculum 策略**：固定的 6 级 curriculum 切换条件（sustained_on_target_steps）可能不是最优的
4. **观测空间**：22D obs 是否足够？是否需要加入 angular velocity error、theta history 等

## 五、是否开放 Phase 3？

**建议：是的，但要分阶段。**

### Phase 3a（中等权限）——先开放
- 训练超参：`NUM_STEPS`、`UPDATE_EPOCHS`、`LR`、`ANNEAL_LR`、`GAE_LAMBDA`
- Curriculum 参数：`sustained_on_target_steps`、各 level 的角度限制
- 训练 budget：允许 agent 自行决定 100M-1B 步

### Phase 3b（高权限）——如果 3a 仍无突破再开放
- 网络架构：GRU hidden dim、FC layers、是否加 attention
- 观测空间：增减 obs 字段
- 动作空间：离散化粒度

**不建议一次性全开**：搜索空间爆炸，Claude 无法有效探索。

## 六、立即可做的改进（不需要 Phase 3）

1. **增大训练 budget 到 1B+**：接近 champion 的训练量
2. **放宽 keep 条件**：允许"整体 theta 更低 + 最多 1 个 level 略差"也算 keep
3. **加入 RAPP 物理先验**：先扫描 tracking_scale [0.5, 5.0] × crash_penalty [-10, -0.1] 的有效范围，告诉 Claude 边界在哪
4. **Prompt 分两阶段**：先让 Claude 分析（"att_r=0.24 在 L5 说明什么？"），再生成方案
