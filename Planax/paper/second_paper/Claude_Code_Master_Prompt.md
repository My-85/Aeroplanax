gemini给的版本：
# 飞行器特技控制：理论落地与实验执行 Prompt (For Claude Code)

**Role:** 你是一位顶尖的航空航天与自动控制领域 AI 研究助手，精通高保真飞行器动力学（基于 JAX 的仿真器）、强化学习算法以及轨迹优化数学建模。

**Context:** 我正在撰写一篇冲击航空航天领域顶刊（AST / IEEE TAES / CJA）的论文。目前我已经拥有一个极其强大的、基于欧拉角和基于四元数（还在训练中，先不用这个baseline）和 PPO 训出的纯 RL 飞机底层特技机动 Baseline。
请协助我完成这篇论文的顶层算法代码化、实验数据采集与论文图表绘制。

---

## 第一部分：论文核心叙事与逻辑基调 (Context for Claude)

在协助我编写代码和分析数据时，请严格遵循本论文的创新逻辑链：

1. **行业痛点:** 传统纯 RL 难以完成长时序、全包线特技飞行（多依赖 PID 兜底），且存在欧拉角万向节死锁和泛化性差的问题。
2. **底层基石 (已完成):** 我们采用了**无奇点四元数通用跟踪 Baseline**。它消除了死锁，将智能体降维成一个纯粹的、跨任务泛化的“姿态/速度跟踪器”。（还在训练中，先不用这个baseline，就先用欧拉角版本的baseline，只要不飞接近90°垂直机动应该就还好）
3. **顶层大脑 (待你实现):** 引入航点追踪来拆解长时序任务。但传统“均匀切分”存在致命缺陷：太稀疏会导致切内道（几何失效）；太密集会超越飞机的作动器物理带宽 $\tau_{act}$（我们平台里面的飞机的这个物理带宽需要你写脚本测试一下，并向我汇报），导致舵面饱和与控制发散。
（注意这里是假设：我们研究了航点密度与闭环跟踪稳定性之间的相互作用。我们假设，过稀的航点会导致几何捷径误差，而过密或突变的航点可能会超过闭环跟踪带宽并增加执行器饱和。）
4. **核心创新:** 提出**带执行器物理带宽约束的自适应航点优化机制 (Adaptive Trajectory Segmentation)**。通过动态规划 (DP) 算法，自动寻找最优航点排布。
5. **完成五个特技机动动作的轨迹跟踪** 包括已经实现的S机动（详见/home/dqy/aeroplanax/new/20251215最新代码库/Planax/render_waypoint_s_euler.py），需要你再完成四个最基本最常见的特技机动动作
---

## 第二部分：代码开发与实验执行清单 (Tasks for Claude)

请基于我现有的 JAX/RL Baseline 环境，按顺序协助我完成以下代码编写与实验验证：

### 👨‍💻 Task 1: 开发自适应航点离散化 DP 求解器 (核心算法落地)
请用纯 Python (NumPy) 编写一个一维动态规划 (DP) 求解脚本，将连续的三维特技轨迹（如垂直筋斗或 S机动）转化为最优离散航点序列 $\mathbf{s} = \{s_1, \dots, s_{N^*}\}$。
* **计算 $E_{geo}$ (弦截误差):** 使用向量叉乘计算航点连线与真实轨迹间的最大垂直偏离。
* **计算 $E_{dyn}$ (动态姿态代价):** 使用相邻线段转角累加法近似曲率积分。
* **引入 $\gamma$ (切换代价):** 作为一个常数惩罚项，使算法能在不人为指定 $N$ （将轨迹拆解成航点的航点个数）的情况下，全自动求出最优航点总数 $N^*$。
* **施加物理带宽约束:** 在 DP 状态转移时，强制要求相邻两点间的距离必须大于 $\Delta s_{min} = (V_{ref} \cdot \tau_{act}) / L_{total}$。
（注意这里ChatGPT在gemini之上给了补充：

）

* *要求：脚本需能接收轨迹数组，输出最优航点的索引与坐标，并具备良好的模块化设计。*

### 🧪 Task 2: 航点密度敏感度实验 (验证 U 型曲线与作动器饱和（注意这里是假设：我们研究了航点密度与闭环跟踪稳定性之间的相互作用。我们假设，过稀的航点会导致几何捷径误差，而过密或突变的航点可能会超过闭环跟踪带宽并增加执行器饱和。）)
编写测试脚本，利用现有的仿真环境和 Baseline 进行对照实验：
* 设定均匀航点切分对比组，强制 $N \in \{5, 10, 20, 40, 80\}$。
* 收集并保存（.npy 或 .csv）每次测试的两个核心指标：**最大跨向误差 (Cross-Track Error)** 和 **升降舵饱和率 (Actuator Saturation Rate)**。
* *预期结果：复现 U 型曲线，证明 $N$ 过小时误差大，$N$ 过大时舵面高频饱和导致失控。*

### 🧪 Task 3: 自适应 DP 算法最优性对比测试
* 运行 Task 1 中的 DP 算法，生成最优自适应航点序列。
* 将该序列输入 Baseline 进行测试，记录飞行轨迹、误差时序数据和控制指令时序数据。
* 与 Task 2 中的盲试最优结果进行对比，证明 DP 算法在更少/合理的航点数下，实现了更低的误差和更平滑的作动器响应。

### 🧪 Task 4: 零样本泛化能力测试 (Zero-Shot Generalization)
* 给定一条 Baseline 在训练时**从未见过**的复杂轨迹（例如 S机动 或 桶滚 或其他基本常见的特技机动轨迹）。
* 调用 Task 1 的 DP 脚本生成该新轨迹的航点。
* 在**坚决不进行网络微调 (No Fine-tuning)** 的前提下，直接使用现有 Baseline 跟踪。
* 记录闭合成功率和轨迹拟合度，证明框架的“任务解耦”能力。

### 📊 Task 5: 期刊级数据可视化渲染
编写 Matplotlib 脚本，读取上述实验保存的数据，渲染符合学术期刊标准（AST/TAES 风格）的图表：
1. **轨迹空间对比图 (Fig 1):** 在 3D 或 2D 投影下，对比展示理想曲线、均匀切分航点、自适应 DP 航点的空间分布（体现转弯处自动加密）。
2. **误差对比时序图 (Fig 2):** 绘制跨向误差随时间的变化，凸显均匀切分在机动段的误差激增，以及自适应策略的平稳。
3. **作动器指令对比图 (Fig 3):** 对比“过密均匀切分”产生的高频震荡/触碰 $\pm 1$ 物理极限的方波，与“DP 物理带宽约束”下的平滑控制曲线。
4. **性能指标表格数据:** 整理输出 Markdown 格式的量化对比表格（包含 N 数量、最大误差、平均误差、饱和率、成功率）。

---
**Next Step:** 请确认你已理解上述研究逻辑和任务目标。如果确认无误，请从 **Task 1** 开始，为我编写 DP 求解器的完整 Python 代码。

###################################################################
###################################################################

ChatGPT给的版本：
# Prompt for Claude Code: Bandwidth-Aware Hierarchical Trajectory Abstraction on Planax

## Role

你是一位精通强化学习、固定翼飞行动力学、高保真仿真、JAX/NumPy/Python 工程实现、轨迹优化与学术论文实验设计的 AI 研究助手。

你需要帮助我在现有 Planax 平台基础上，完成一篇新的方法型论文的实验包装与代码实现。

注意：Planax 作为高保真、高并行固定翼 RL benchmark 已经单独投稿 RA-L。因此，当前这篇论文不能再把 Planax benchmark 本身作为主贡献。Planax 在本文中只作为实验平台和高保真验证环境。本文主贡献应聚焦于：

> 在已训练好的底层 RL 飞行控制 skill 基础上，提出一种面向长时序固定翼机动轨迹跟踪的带宽感知分层轨迹抽象方法。

请严格避免把本文写成“又一篇 Planax 平台论文”。本文核心是：**如何把连续长时序特技轨迹自动转化为底层 RL policy 可稳定执行的子目标序列，并系统分析航点密度、闭环带宽、轨迹曲率和作动器饱和之间的关系。**

---

# 1. Research Positioning

## 1.1 Existing Foundation

我已经拥有：

1. 一个名为 Planax 的高保真固定翼仿真平台；
2. F-16 非线性 6-DOF 动力学；
3. NASA 气动数据表；
4. JAX/XLA 高并行仿真能力；
5. 一个已经训练好的底层 RL baseline，用于单机机动控制；
6. 当前可用的 baseline 主要是欧拉角版本；
7. 四元数版本还在训练中，本文暂时不依赖四元数 baseline。

因此本文不要过度宣称“全包线无奇点控制”或“完整垂直特技飞行能力”。如果使用欧拉角 baseline，应避免接近 pitch = ±90° 的强奇异机动，或者将其作为失败边界分析，而不是主结果。

## 1.2 Target Paper Narrative

本文不再以 Planax benchmark 为核心贡献，而是提出：

> Bandwidth-Aware Hierarchical Trajectory Abstraction for RL-Based Fixed-Wing Maneuver Tracking

核心思想是：

1. 底层 RL policy 负责局部姿态/速度/航向/航点跟踪；
2. 顶层模块负责把连续长时序轨迹自动离散为可执行的子目标序列；
3. 离散化不能只考虑几何误差，还必须考虑闭环控制带宽和作动器饱和；
4. 过稀航点会导致几何切内道和轨迹拟合误差；
5. 过密航点或突变目标会超过底层 RL controller 的闭环响应能力，引发舵面饱和、高频震荡或跟踪失败；
6. 因此本文提出一种 bandwidth-aware adaptive subgoal segmentation 方法，在几何精度和动态可执行性之间自动折中。

不要把 DP 算法包装成“全局最优飞行轨迹优化器”。它只是对 surrogate objective 最优。正式表述应为：

> The proposed dynamic programming solver optimizes a surrogate segmentation objective and is evaluated through closed-loop flight simulations.

---

# 2. Method to Implement

请实现一个模块化的 Python/NumPy 动态规划求解器，用于将连续三维轨迹离散为自适应航点或子目标序列。

不要只实现简单的几何 DP。需要把原来的 Gemini 方案升级为：

> Bandwidth-aware adaptive subgoal segmentation.

---

# 3. Task 1: Implement Bandwidth-Aware Adaptive Segmentation DP Solver

## 3.1 Input

函数输入：

- `trajectory`: shape = `[M, 3]`，连续参考轨迹点，单位为米；
- 可选 `attitude_ref`: shape = `[M, 3]`，如果已有参考欧拉角 `[roll, pitch, yaw]`；
- 可选 `speed_ref`: shape = `[M]` 或常数；
- `v_ref`: 参考速度；
- `dt_ref`: 参考轨迹采样时间间隔；
- `weights`: objective 权重；
- `constraints`: 闭环动态约束参数。

## 3.2 Output

输出：

- 最优航点索引 `waypoint_indices`;
- 最优航点坐标 `waypoints`;
- 每段代价 `segment_costs`;
- 每段几何误差、曲率代价、rate 代价；
- 总代价；
- 可视化需要的 debug 信息。

## 3.3 Objective

请将每个候选 segment `(i, j)` 的代价设计为：

\[
J(i,j) =
\lambda_g E_{geo}(i,j)
+ \lambda_c E_{curv}(i,j)
+ \lambda_r E_{rate}(i,j)
+ \lambda_s E_{switch}
\]

其中：

### 3.3.1 Geometry Error

`E_geo(i,j)` 表示从轨迹点 `i` 到 `j` 用一条直线 chord 近似时，真实轨迹点到 chord 的最大垂直距离或 RMS 垂直距离。

要求实现：

- max cross-track error；
- mean/RMS cross-track error；
- 可通过参数选择。

计算方式：

对于段内每个点 `p_k`，计算其到 chord `p_i -> p_j` 的垂直距离。

如果 chord 长度过短，需要做数值保护。

### 3.3.2 Curvature / Direction-Change Cost

`E_curv(i,j)` 用于惩罚段内方向变化剧烈的轨迹。

可以使用相邻轨迹切向量夹角累加：

\[
E_{curv}(i,j) = \sum_{k=i+1}^{j-1} \arccos
\left(
\frac{t_{k-1}^\top t_k}{\|t_{k-1}\|\|t_k\|}
\right)
\]

也可以同时输出近似曲率统计量，例如：

- accumulated turning angle；
- max local turning angle；
- mean curvature proxy。

### 3.3.3 Rate / Bandwidth Cost

这是本文核心，不要省略。

对于候选 segment `(i,j)`，估计该段飞行时间：

\[
\Delta t_{ij} = \frac{L_{ij}}{V_{ref}}
\]

其中 `L_ij` 可以使用 chord length，也可以使用 arc length。建议默认使用 arc length。

然后估计该 segment 首尾方向或姿态变化率。

如果没有 `attitude_ref`，则至少根据轨迹切向量估计 yaw / flight path angle 的变化。

需要计算：

\[
\dot{\psi}_{req} = \frac{|\Delta \psi|}{\Delta t_{ij}}
\]

\[
\dot{\theta}_{req} = \frac{|\Delta \theta|}{\Delta t_{ij}}
\]

如果有 roll reference，也计算：

\[
\dot{\phi}_{req} = \frac{|\Delta \phi|}{\Delta t_{ij}}
\]

然后定义：

\[
E_{rate}(i,j) =
\max
\left(
\frac{\dot{\psi}_{req}}{\dot{\psi}_{max}},
\frac{\dot{\theta}_{req}}{\dot{\theta}_{max}},
\frac{\dot{\phi}_{req}}{\dot{\phi}_{max}}
\right)
\]

如果某个 reference 不存在，则跳过对应项。

注意角度差需要使用 wrap 到 `[-pi, pi]` 的函数，不能直接相减。

### 3.3.4 Switching Cost

`E_switch` 是常数切换惩罚，用于避免航点过多。

可以设为 1，然后由 `lambda_s` 控制。

---

# 4. Hard Constraints

DP 状态转移时，不仅要看代价，还要判断候选 segment 是否满足物理/闭环可执行性约束。

请实现以下 hard constraints：

## 4.1 Minimum Segment Time

\[
\Delta t_{ij} \geq \tau_{cmd}
\]

其中 `tau_cmd` 表示底层 RL controller 或目标切换机制能够稳定响应的最小时间尺度。

注意：这里不要简单称为 actuator bandwidth。更准确地叫：

> closed-loop command bandwidth / target-switching bandwidth.

## 4.2 Minimum Segment Length

\[
L_{ij} \geq L_{min}
\]

其中：

\[
L_{min} = V_{ref} \tau_{cmd}
\]

如果轨迹使用归一化弧长参数，则可以对应转换为：

\[
\Delta s_{min} = \frac{V_{ref} \tau_{cmd}}{L_{total}}
\]

## 4.3 Maximum Required Heading / Pitch / Roll Rate

如果估计得到姿态或轨迹方向变化率，则要求：

\[
\dot{\psi}_{req} \leq \dot{\psi}_{max}
\]

\[
\dot{\theta}_{req} \leq \dot{\theta}_{max}
\]

\[
\dot{\phi}_{req} \leq \dot{\phi}_{max}
\]

## 4.4 Maximum Local Curvature Proxy

如果段内轨迹曲率或方向变化过大，候选 segment 应被拒绝或强烈惩罚。

实现方式可以是：

- hard reject：`max_turn_angle > max_turn_angle_allowed`;
- 或 soft penalty：加入 `E_curv`。

请把这两种模式都支持。

---

# 5. DP Solver Requirements

请实现标准动态规划：

\[
dp[j] = \min_i dp[i] + J(i,j)
\]

并记录 backpointer。

要求：

1. 支持 `O(M^2)` 初始实现；
2. 对每个候选 segment 预计算代价，便于 debug；
3. 不可行 segment 的 cost 设为 `np.inf`；
4. 支持强制首尾点为航点；
5. 支持最小和最大航点数的可选约束；
6. 支持输出完整 debug dictionary；
7. 支持保存结果为 `.npz` 或 `.json`；
8. 代码结构清晰，便于后续被实验脚本调用。

---

# 6. Task 2: Waypoint Density Sensitivity Experiment

请基于现有 Planax 环境和已训练底层 RL baseline，写实验脚本对比不同航点密度对闭环跟踪性能的影响。

注意：不要预设一定会出现 U 型曲线。实验目标是系统分析：

> waypoint density 与 tracking error、actuator saturation、closed-loop stability 的关系。

## 6.1 Baselines

至少包含：

1. Uniform arc-length waypoints with fixed N:
   - `N = {5, 10, 20, 40, 80}`；
2. Curvature-based sampling；
3. RDP-style geometric simplification；
4. DP without bandwidth constraint；
5. DP with bandwidth constraint；
6. Oracle best uniform N，即在 uniform N 组里后验选择表现最好的 N。

如果时间不足，至少完成：

- Uniform fixed N；
- DP without bandwidth；
- DP with bandwidth；
- Oracle best uniform N。

## 6.2 Metrics

每次 closed-loop rollout 至少保存：

### Tracking Metrics

- max cross-track error；
- mean cross-track error；
- RMS cross-track error；
- final position error；
- success/failure；
- trajectory completion ratio。

### Control Metrics

- elevator saturation rate；
- aileron saturation rate；
- rudder saturation rate；
- throttle saturation rate；
- total actuator saturation rate；
- actuator command RMS；
- actuator command variation；
- actuator command smoothness，例如：

\[
\sum_t \|u_t - u_{t-1}\|^2
\]

### Stability / Safety Metrics

如果环境里能拿到这些变量，也请保存：

- airspeed min/max；
- altitude min/max；
- angle of attack max；
- sideslip max；
- roll/pitch/yaw max；
- episode termination reason。

## 6.3 Data Saving

每次实验保存：

- config；
- method name；
- number of waypoints；
- waypoint coordinates；
- trajectory reference；
- actual trajectory；
- action time series；
- state time series；
- metrics；
- success flag。

建议保存为：

- `.npz` 用于大数组；
- `.csv` 用于 summary table；
- `.json` 用于 config 和 metrics。

---

# 7. Task 3: Closed-Loop Performance Evaluation of Adaptive Segmentation

不要命名为 “DP optimality test”。因为 DP 只对 surrogate objective 最优，不代表真实闭环飞行最优。

请命名为：

> Closed-loop performance evaluation of bandwidth-aware adaptive segmentation.

实验目标：

1. 运行 DP with bandwidth constraint；
2. 运行 DP without bandwidth constraint；
3. 运行 uniform fixed N；
4. 运行 oracle best uniform N；
5. 比较 closed-loop tracking 性能和作动器平滑性；
6. 判断 bandwidth-aware 约束是否减少舵面饱和与高频抖振；
7. 判断 adaptive segmentation 是否以更少或更合理的航点数达到相近或更优的 tracking performance。

## 7.1 Required Figures

生成以下数据，供后续画图：

1. reference trajectory vs actual trajectory；
2. waypoint distribution；
3. cross-track error over time；
4. actuator commands over time；
5. saturation indicator over time；
6. accumulated tracking error；
7. phase-wise error，例如直线段、转弯段、爬升段分别统计。

---

# 8. Task 4: Zero-Shot Generalization Evaluation

本文要强调底层 RL skill 与顶层 trajectory abstraction 的任务解耦能力。

请设计 zero-shot 测试，但要定义清楚 zero-shot 类型。

## 8.1 Zero-Shot Categories

至少支持以下几类：

### 8.1.1 Geometry Zero-Shot

训练时未见过的新轨迹形状，例如：

- S-turn；
- figure-eight；
- level circle；
- climbing turn；
- descending turn；
- smooth slalom。

注意：如果当前 baseline 是欧拉角版本，暂时不要把接近 pitch = ±90° 的 vertical loop 作为主实验。可以放进 failure/boundary case。

### 8.1.2 Scale Zero-Shot

同一类轨迹改变：

- 半径；
- 振幅；
- 总长度；
- 速度；
- 高度变化；
- 曲率强度。

### 8.1.3 Initial-State Zero-Shot

改变初始条件：

- 初始速度；
- 初始高度；
- 初始 heading；
- 初始 roll/pitch/yaw 小扰动。

### 8.1.4 Disturbance / Dynamics Zero-Shot

如果环境支持，加入：

- wind disturbance；
- mass variation；
- aerodynamic coefficient perturbation；
- sensor noise；
- action delay。

## 8.2 Required Output

输出 generalization matrix：

| Train Skill | Test Trajectory | Method | Success Rate | Mean CTE | Max CTE | Saturation Rate | Completion Ratio |
|---|---|---|---|---|---|---|---|

其中本文当前底层 skill 可能是同一个 RL baseline，但测试轨迹和初始条件不同。

---

# 9. Task 5: Visualization and Paper-Ready Figures

请写 Matplotlib 脚本，从实验保存的数据中生成论文图表。

风格要求：

1. 学术论文风格；
2. 字体清晰；
3. 图例简洁；
4. 支持保存为 `.pdf` 和 `.png`；
5. 所有图的横纵轴和单位明确；
6. 不要只画漂亮图，要画能支撑论点的图。

## 9.1 Required Figures

### Fig. 1: Method Overview

可以先输出数据或草图结构，不一定直接画复杂系统图。

内容应表达：

continuous reference trajectory  
→ adaptive segmentation  
→ subgoal sequence  
→ RL tracking policy  
→ high-fidelity Planax closed-loop rollout  
→ metrics and failure analysis.

### Fig. 2: Waypoint Density Sensitivity

横轴：

- number of waypoints；
- 或 effective average segment duration。

纵轴至少包含：

- mean / RMS cross-track error；
- actuator saturation rate；
- command smoothness。

这张图用于说明航点密度与闭环性能之间的非单调关系或 trade-off。

### Fig. 3: Adaptive Segmentation Visualization

展示：

- continuous reference trajectory；
- uniform waypoints；
- DP without bandwidth waypoints；
- DP with bandwidth waypoints。

重点体现：

- 高曲率区域自动加密；
- 但不会无限加密；
- 带宽约束会避免过短 segment。

### Fig. 4: Closed-Loop Tracking Comparison

展示不同方法下：

- reference trajectory；
- actual trajectory；
- tracking error over time。

### Fig. 5: Actuator Command Comparison

展示：

- elevator；
- aileron；
- rudder；
- throttle。

重点比较：

- uniform over-dense waypoints 是否产生高频震荡；
- DP with bandwidth 是否降低饱和率；
- DP without bandwidth 是否可能产生过短 segment。

### Fig. 6: Zero-Shot Generalization Matrix

画成 heatmap 或 table。

显示不同轨迹、不同尺度、不同初始条件下的 success rate / RMS CTE。

## 9.2 Summary Table

输出 Markdown 和 LaTeX 两种格式的表格。

表格包含：

| Method | #Waypoints | Mean CTE | Max CTE | RMS CTE | Saturation Rate | Smoothness Cost | Success Rate | Completion Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

---

# 10. Important Scientific Cautions

请在代码注释、实验命名和结果说明中避免以下过度表述：

## 10.1 不要说

- DP gives globally optimal flight trajectory；
- full-envelope aerobatic control；
- singularity-free control，如果本文没用四元数 baseline；
- solved aerobatic flight control；
- actuator bandwidth is fully captured by waypoint distance；
- Planax benchmark is the main contribution of this paper。

## 10.2 可以说

- DP optimizes a surrogate segmentation objective；
- bandwidth-aware subgoal abstraction；
- closed-loop command bandwidth；
- target-switching frequency constraint；
- high-fidelity closed-loop evaluation on Planax；
- zero-shot trajectory composition using a pretrained low-level RL skill；
- systematic analysis of waypoint density, tracking error, and actuator saturation；
- Planax serves as the high-fidelity simulation backend.

---

# 11. Implementation Order

请按以下顺序执行：

1. 先阅读现有 S-maneuver 脚本：
   `/home/dqy/aeroplanax/new/20251215最新代码库/Planax/render_waypoint_s_euler.py`

2. 找到底层 RL baseline 的输入输出接口：
   - observation；
   - action；
   - target waypoint 或 target attitude 的设置方式；
   - rollout loop；
   - state logging；
   - action logging；
   - termination condition。

3. 实现独立模块：
   `adaptive_segmentation_dp.py`

4. 实现轨迹生成模块：
   `maneuver_trajectories.py`

   至少包含：
   - S-turn；
   - figure-eight；
   - level circle；
   - climbing turn；
   - descending turn；
   - smooth slalom。

   如果使用欧拉角 baseline，暂时不要把 vertical loop / barrel roll 作为主结果。可以作为 optional stress test。

5. 实现实验脚本：
   `run_waypoint_density_experiment.py`

6. 实现 adaptive segmentation 闭环实验：
   `run_adaptive_segmentation_experiment.py`

7. 实现 zero-shot generalization 实验：
   `run_zeroshot_generalization.py`

8. 实现画图脚本：
   `plot_segmentation_results.py`

9. 所有实验输出统一保存到：
   `outputs/adaptive_segmentation/YYYYMMDD_HHMMSS/`

---

# 12. Expected Deliverables

请最终给出：

1. 完整 Python 代码；
2. 每个脚本的运行命令；
3. 每个输出文件的说明；
4. summary csv；
5. paper-ready figures；
6. Markdown summary table；
7. 一段可以放进论文 Experimental Setup 的英文描述；
8. 一段可以放进论文 Method 的英文描述；
9. 一段可以放进论文 Results 的英文分析模板。

---

# 13. Final Reminder

本文不是 Planax benchmark 论文。本文是在 Planax 平台基础上提出和验证一种：

> bandwidth-aware hierarchical trajectory abstraction method for RL-based fixed-wing maneuver tracking.

请始终围绕这个主题组织代码、实验和结果。