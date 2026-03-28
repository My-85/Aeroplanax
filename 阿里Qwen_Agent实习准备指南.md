# 阿里 Qwen Agent 实习准备指南

## 一、简历包装建议

### 1. 研究方向修改

原：`深度强化学习、飞行器控制与多机协同博弈`

改为：**深度强化学习、LLM-driven Agentic RL、飞行器自主控制与多机协同博弈**

### 2. 新增项目条目（建议放在 Planax 论文之后）

> **2025.03-至今　　AutoPlanax: Agentic Framework for Automated RL Research（基于 LLM Agent 的自动化强化学习研究框架）　　负责人**
>
> **研发背景**：强化学习研究中，奖励设计、超参调优与训练诊断高度依赖人工试错。参考 Eureka/AI-Scientist 等 auto-research 范式，在 Planax 平台基础上构建 LLM Agent 驱动的自动化 RL 实验框架 AutoPlanax。
>
> **负责内容**：
> ① 设计渐进式权限释放机制：Agent 从奖励参数调优（Phase 1）→ 奖励函数代码重写（Phase 2）→ 环境/训练/观测空间全链路修改（Phase 3）逐阶段解锁，依据训练瓶颈自动判断升级时机；
> ② 构建 reward reflection 与 in-context 进化搜索机制：将历史实验的逐级指标、奖励分量诊断与失败归因结构化注入 LLM 上下文，引导 Agent 基于证据推理而非盲目搜索；
> ③ 设计课程感知的分层评估与 early-exit、Git 自动回滚、早停监控等 scaffolding，支撑 Agent 无人值守连续迭代。
>
> **核心成果**：已完成 68 轮自动实验迭代，建立"假设生成→RL 训练→分层评估→reward reflection→进化搜索"的 auto-research 闭环，框架设计可扩展至任意 RL 任务的自动化研究。

### 3. Planax 条目补充

在「负责内容」②中建议加一句：

> 设计多级课程学习（curriculum learning）机制与自适应奖励函数，支持从简单机动到全域 180° 姿态跟踪的渐进式策略训练。

### 4. 其他信息微调

- 计算机水平补充：`Claude Code/Anthropic API`（用于 agentic workflow 开发）、`VeRL`（了解）
- 编程语言顺序调整：掌握 Python（**PyTorch**、JAX、TensorFlow）、MATLAB、C++、C

### 5. 投递邮件 hook

> "我在清华自动化系研发了基于 LLM Agent 的自动化 RL 研究框架 AutoPlanax（auto-research 范式），已完成 68 轮无人值守自动实验迭代，这与贵团队 scaffolding/harness RL 优化及 auto research 方向高度契合。"

### 6. 你的经历与 JD 六项工作内容的对应关系（面试话术）

| JD 工作内容 | 你的对应经历 |
|---|---|
| 1. memory 机制与 context learning | AutoPlanax 的 results.jsonl 实验记忆 + 结构化上下文注入 |
| 2. self-evolving 与 long-horizon tasks | Phase 1→2→3 自适应升级；8 级课程学习就是 long-horizon |
| 3. scaffolding 和 harness 的 RL 优化 | **最强匹配**：整个 AutoPlanax 就是 scaffolding + harness |
| 4. multi-agent 与 workflow 编排 | Planax 50v50 MAPPO 自博弈；AutoPlanax 的 human↔LLM↔evaluator 三角协作 |
| 5. self-play 机制 | Planax 集群空战自博弈训练 |
| 6. auto research / terminal 任务 | **最强匹配**：AutoPlanax 就是 auto-research 范式的实例 |

---

## 二、Agentic RL 作为研究方向，还需要什么

### 你已经有的

- RL 环境构建 + 训练全栈能力（Planax）
- LLM Agent 驱动 RL 实验的工程系统（AutoPlanax）
- Reward shaping 的实战经验（68 轮迭代，踩过的坑比大多数人都多）

### 要称为"研究方向"，还需要三层东西

**第一层：理论视角（能讲清楚"为什么"）**

- LLM 作为优化器 vs 传统搜索（evolutionary search、Bayesian optimization）的本质区别是什么？你的 in-context 进化搜索和 CMA-ES 比，优势在哪？
- Reward reflection 的信息论解释：LLM 从失败实验中提取了什么信号？和 reward model (RLHF) 的区别？
- 权限释放机制的理论依据：为什么渐进式比一步到位好？和课程学习的关系？

**第二层：Baseline 对比（能证明"比别人好"）**

- 至少一个 ablation study：去掉 reward reflection / 去掉渐进权限 / 随机搜索，性能差多少
- 在 2-3 个不同任务上验证泛化性（不只是无人机姿态跟踪）
- 这是从"工程项目"升级为"研究贡献"的关键门槛

**第三层：发论文（能被同行认可）**

- 目标会议：NeurIPS/ICML/ICLR 的 Agent 或 RL track，或 CoRL/RSS（机器人方向）
- 一篇 AutoPlanax 论文，定位为 "LLM-agent-driven automated RL research framework"，对标 Eureka + AI-Scientist

---

## 三、必读文献清单

### 核心（必须精读）

| 论文 | 为什么读 |
|---|---|
| **Eureka** (Ma et al., ICLR 2024) | 直接上游工作，LLM 写 reward code |
| **DrEureka** (Wang et al., RSS 2025) | Eureka 扩展到 sim-to-real，reward + domain randomization |
| **AI-Scientist** (Lu et al., ICLR 2025 Oral) | Auto-research 范式的标杆，和 AutoPlanax 理念最接近 |
| **Agent²** (Wei et al., NeurIPS 2024) | Agent-generates-agent，和你的权限释放思路类似 |
| **Text2Reward** (Xie et al., ICML 2024) | LLM 生成 dense reward code，有 iterative refinement |
| **Language to Rewards** (Yu et al., NeurIPS 2023) | Google 的 LLM→reward→MPC pipeline |

### 扩展（了解前沿）

| 方向 | 论文 |
|---|---|
| LLM 作为 RL 的 reward model | **MOTIF** (Klissarov et al., NeurIPS 2024) |
| LLM 指导 RL exploration | **ELLM** (Du et al., ICML 2023)、**ReMA** (2025) |
| Self-play + LLM | **Voyager** (Wang et al., NeurIPS 2023 Oral) — Minecraft 里 LLM 驱动的 self-evolving agent |
| RL infra / scaffolding | **VeRL** (Sheng et al., 2024)、**OpenRL** |
| Agent workflow | **CAMEL** (Li et al., NeurIPS 2023)、**AutoGen** (Wu et al., 2023) |
| LLM + evolutionary search | **EvoPrompt** (Guo et al., ACL 2024)、**OpenELM** (Lehman et al., 2024) — 和 in-context 进化搜索直接相关 |
| Automated ML/research | **FunSearch** (Romera-Paredes et al., Nature 2024) — DeepMind 用 LLM 做数学发现 |

### 建议阅读顺序

Eureka → Text2Reward → DrEureka → Agent² → AI-Scientist → Voyager → FunSearch

---

## 四、Claude Code 使用潜力分析

### 已用到的功能（约 40-50%）

- 代码阅读/编辑/调试
- 多文件搜索与理解
- Git 操作
- 长对话中的复杂推理
- 项目记忆（MEMORY.md）
- 子 Agent 并行探索

### 未充分使用的功能

**1. Claude Code 作为 AutoPlanax 的 Agent Runtime**

当前通过 API 调用 Claude，但可以直接用 Claude Code 本身作为 Agent 运行器：

```bash
# Claude Code 非交互模式，直接跑实验循环
claude -p "读取 champion_meta.json 和 results.jsonl，分析最近5轮失败模式，修改 quat_baseline_reward.py，然后运行 python experiment_runner.py --mode manual"
```

比自己写 API wrapper + 解析响应更强大，因为 Claude Code 本身就能读文件、改代码、跑命令、看输出。

**2. Hooks（自动化触发器）**

配置 hooks 让 Claude Code 在特定事件时自动执行操作，如每次编辑 reward 文件后自动跑 lint 检查 JAX 兼容性。

**3. MCP Server 集成**

接入外部工具（数据库、Slack、自定义 API）扩展 Claude Code 能力边界。如接入 wandb API 直接查看训练曲线。

**4. Skill 快捷命令**

`/review-pr`、`/commit` 等内置 skill 可自动化 git 工作流。

**5. 定时任务（CronCreate）**

```
"每30分钟检查一次当前训练的 stdout 输出，如果 theta_deg 停滞超过 20M steps，提醒我"
```

**6. 多 Agent 并行**

同时启动多个 Agent 做独立任务：一个分析实验日志、一个搜索文献、一个写代码，而非串行等待。

**7. Plan Mode**

对复杂实现任务，先进入 plan mode 制定方案审批再执行，避免做到一半方向不对。

**8. Worktree 隔离开发**

在不影响主分支的情况下让 Agent 实验性地修改代码。

### 建议学习的 Claude Code 技能

1. **`claude -p` 批量自动化**（非交互模式）— 对 AutoPlanax 的 Agent 循环非常有价值
2. **配置 CLAUDE.md**（项目级指令文件）— 每次新对话自动加载项目规范
3. **Worktree 隔离开发** — 不影响主分支的实验性代码修改
4. **Context window 管理** — 学会拆分任务避免长对话压缩导致质量下降

---

## 五、建议近期行动项

1. **简历更新**：加入 AutoPlanax 项目条目，调整研究方向和技能关键词
2. **开源 AutoPlanax**：GitHub 上发一个 clean 版本 + README，满足 JD "有影响力的开源项目"
3. **精读 Eureka + AI-Scientist**：面试必问，至少能讲清楚和你工作的异同
4. **准备一个 5 分钟 AutoPlanax demo/presentation**：面试 show case 用
5. **投递邮件**：gxd493162@alibaba-inc.com
