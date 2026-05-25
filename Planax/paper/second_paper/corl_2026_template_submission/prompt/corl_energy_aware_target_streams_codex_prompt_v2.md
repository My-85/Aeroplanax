# Codex Prompt: Reorganize CoRL Paper Around Energy-Aware Target Streams with New Experimental Tables

你现在需要修改我的 CoRL 2026 Overleaf 论文。请直接编辑主 `.tex` 文件，不要另起一个全新文件。目标不是重写所有内容，而是在保留已有技术内容、图表、算法和新增实验表格的基础上，重新组织论文标题、结构和叙事逻辑，使其更符合 CoRL 中稿文章的风格。

当前主稿中已经补充了三类关键表格数据：

1. Euler-angle vs quaternion target encoding comparison；
2. Energy-aware vertical extension before/after comparison；
3. Geometry-aware loop-quality frontier table。

这版 prompt 的重点是：**把这些新表格纳入论文主线，而不是让它们像零散实验堆在 Results 里。**

---

## 0. 总体论文定位

请把论文从当前偏 “RH-TSO target-stream optimization paper” 的写法，重组为：

> A paper about learned fixed-wing maneuvering where:
> 1. we first justify why a direct RL flight skill is needed instead of PID/autopilot-style nested control;
> 2. we show why the first Euler-angle RL baseline is limited for complex three-dimensional maneuvering;
> 3. we motivate the quaternion-conditioned direct-actuator RL skill as a more suitable reusable base skill for maneuver composition;
> 4. geometry-aware metrics reveal pseudo-tracking in large-attitude maneuvers;
> 5. energy-aware PPO fine-tuning expands the base skill from horizontal / mild 3D maneuvers toward larger vertical arcs;
> 6. executable target streams and RH-TSO then compose the frozen energy-aware skill into long-horizon maneuver execution.

最终论文定位：

> This paper studies pseudo-tracking and skill composition in learned fixed-wing maneuvering. We build a quaternion-conditioned direct-actuator flight skill, diagnose its large-attitude limitations using geometry-aware maneuver metrics, expand its vertical maneuvering capability through energy-aware PPO fine-tuning, and compose the resulting frozen skill through executable target streams selected by receding-horizon closed-loop rollout.

不要把论文写成“我训练了一个 baseline + 做了一个 RH-TSO 调参器”。要写成：

**learning-control interface + representation evolution + energy-aware capability expansion + executable target-stream composition + honest geometry-aware capability boundary**。

---

## 1. 标题修改

把当前标题：

```latex
\title{Receding-Horizon Target Streams for Learned Fixed-Wing Maneuvering}
```

改为：

```latex
\title{Energy-Aware Target Streams for Learned Fixed-Wing Maneuvering}
```

这个标题先作为最终标题使用。不要使用问句式标题。CoRL 风格的小节标题也尽量使用精炼短语，不要使用疑问句。

---

## 2. 摘要重写方向

请重写 abstract，使其包含以下逻辑，但不要写得过长：

1. Fixed-wing maneuvering is not waypoint tracking.
2. Direct actuator-level RL is attractive because fixed-wing maneuvering couples attitude, energy, velocity direction, lift direction, and actuator limits; a hand-tuned PID/autopilot hierarchy can hide the actuator-level feasibility issue.
3. The first Euler-angle RL baseline exposes limitations on complex three-dimensional maneuvers.
4. A quaternion-conditioned direct-actuator skill provides a reusable attitude-speed tracking interface.
5. Geometry-aware metrics reveal pseudo-tracking: low CTE can coexist with wrong nose direction, velocity tangent, wing plane, AoA, or energy state.
6. Energy-aware PPO fine-tuning expands the base skill toward vertical arcs by shaping energy retention, low-speed safety, vertical progress, and replay.
7. Long-horizon maneuvers are represented as executable target streams, and RH-TSO selects stream parameters by short closed-loop rollouts through the frozen policy.
8. Keep the existing quantitative RH-TSO result:
   - CTE-P90 improvements of 39%, 2%, 24%, and 47% on S-curve, figure-eight, helix, and 90-degree vertical pull-up tasks.
9. Keep the honest capability boundary:
   - 60°–150° vertical arcs are B-grade under geometry-aware evaluation;
   - 180° half-loop remains a failure case.

Important wording:
- Do not claim “solved aerobatics”.
- Do not claim real-world deployment.
- Do not claim quaternion is universally better than Euler. The table shows quaternion is much better on complex circles / S-curve / figure-eight, but Euler is still better on some simple pitch pull-up targets.
- Do not hide the 180° failure.

---

## 3. Introduction 重组

请重写 Introduction，按照下面顺序组织。

### P1: Fixed-Wing Maneuvering Is Not Waypoint Tracking

保留当前稿件中类似下面这句话：

```text
A fixed-wing aircraft cannot stop at a waypoint, rotate in place, and continue.
```

强调 CTE-only 不足，因为 fixed-wing 飞行受 attitude、velocity、lift、energy、actuator authority、安全边界共同约束。

### P2: Direct RL Flight Skill Instead of Only PID/Autopilot Hierarchy

当前稿件已经强调 PID-free，但还不够。请补充：

- classical fixed-wing stacks often use nested PID/autopilot loops;
- these stacks are effective in a nominal envelope;
- however, for learned aggressive maneuver composition, the learned module should directly experience actuator limits, aerodynamic coupling, and energy loss;
- therefore, a direct-actuator RL baseline is a reasonable interface for this paper.

注意：不要贬低 PID/autopilot；写成 “effective in their design regime, but less suitable as the only interface for studying learned maneuver composition.”

### P3: Euler-Angle RL Baseline and Representation Limitation

新增或强化这一段：

- The first direct RL baseline used Euler-angle attitude targets.
- It can support simple and moderate tasks.
- The new table now allows a quantitative but careful statement:
  - Euler is competitive or better on some simple targets such as level flight, pitch +10°, pitch -10°, and pull-up tasks;
  - quaternion encoding substantially improves complex turning and 3D curve tasks such as Circle R5000, S-curve A3000, and Figure-eight R5000.
- Therefore, quaternion is not claimed as universally superior, but it is a better representation for the reusable maneuver-composition skill targeted in this paper.

Use the actual table data carefully:

```latex
Circle $R5000$ right: Euler att. err. 42.2°, heading err. 30.4°; quaternion att. err. 11.0°, heading err. 5.7°.
Circle $R5000$ left: Euler att. err. 58.6°, heading err. 45.9°; quaternion att. err. 13.3°, heading err. 8.7°.
S-curve $A3000$: Euler att. err. 12.0°, heading err. 10.8°; quaternion att. err. 7.0°, heading err. 2.0°.
Figure-eight $R5000$: Euler att. err. 20.7°, heading err. 16.5°; quaternion att. err. 11.2°, heading err. 5.9°.
```

But also acknowledge:

```latex
Pitch $-10^\circ$: Euler att. err. 9.5° vs quaternion 36.9°.
$15^\circ$ pull-up $R3000$: Euler att. err. 4.3° vs quaternion 29.3°.
$30^\circ$ pull-up $R5000$: Euler att. err. 5.5° vs quaternion 21.3°.
```

This makes the paper more honest and credible.

### P4: Quaternion-Conditioned Direct-Actuator Flight Skill

说明：

- policy receives local heading/pitch/roll/airspeed target expressed through quaternion error and auxiliary state features;
- outputs throttle, elevator, aileron, rudder, speed brake directly;
- it is a reusable flight skill, not a trajectory optimizer;
- it becomes the low-level executable module for later target-stream composition.

### P5: Geometry-Aware Diagnosis and Energy-Aware Vertical Extension

这是本文核心之一。请写清楚：

- quaternion base skill still has boundary;
- geometry-aware maneuver metrics reveal pseudo-tracking in large-attitude maneuvers;
- pseudo-tracking means the trajectory may stay close in position but violates actual fixed-wing flight geometry or energy consistency, e.g. large AoA, poor nose-velocity alignment, wrong tangent direction, energy loss, stall/done risk;
- therefore we add energy-aware PPO fine-tuning, not as a minor ablation, but as the capability expansion stage that allows the skill to go beyond horizontal/mild 3D maneuvers toward large vertical arcs.

Use the new vertical-extension table as evidence:
- fine-tuned vertical skill preserves level flight and circle/S-curve performance;
- it fixes catastrophic failures on negative heading changes:
  - Heading -45°: survival 0.00 → 1.00, attitude error 56.9° → 11.9°;
  - Heading -20°: survival 0.00 → 1.00, attitude error 62.2° → 7.8°, stall rate 0.375 → 0.000;
- it reduces stall rate on several tasks:
  - level flight 0.010 → 0.002;
  - circle R5000 right 0.008 → 0.000;
  - S-curve A3000 0.005 → 0.000;
  - 30° pull-up R8000 0.024 → 0.006.

Do not claim it improves every scalar metric. For example:
- S-curve attitude error: 7.0° → 7.8°;
- 30° pull-up attitude error: 17.7° → 18.7°.
Frame this as capability expansion and safety/robustness improvement, not uniform metric dominance.

### P6: Why Target Streams and RH-TSO

请解释为什么 Reference maneuver 不直接变成 actuator command：

- A long-horizon reference maneuver is not directly executable by the aircraft.
- Directly optimizing or prescribing actuator commands over the full horizon is high-dimensional, brittle, and not reusable across maneuvers.
- A static waypoint or static attitude target is also insufficient because it does not encode how attitude, speed, and energy should evolve along the maneuver.
- Therefore, we convert the reference maneuver into an executable stream of local attitude-speed targets for the learned skill.
- RH-TSO optimizes stream parameters, not actuator commands.
- Candidate streams are evaluated by closed-loop rollout through the frozen skill and the same dynamics used at execution.

If the codebase contains relevant ablations/logs, summarize only supported facts. Search keywords:
`target_stream`, `rhtso`, `rollout`, `lattice`, `lookahead`, `airspeed`, `direct_action`, `actuator`, `waypoint`, `static_target`, `ablation`, `vertical`, `energy`, `pseudo`.

### P7: Contributions

把当前五条贡献改成四条。建议如下：

```latex
The paper makes four contributions.
\begin{enumerate}
  \item We develop a direct-actuator fixed-wing RL flight skill and motivate its evolution from Euler-angle targets to quaternion-conditioned attitude-speed targets for maneuver composition.
  \item We introduce geometry-aware maneuver metrics that expose pseudo-tracking cases where cross-track error is small but the aircraft violates the intended flight geometry, energy consistency, or safety envelope.
  \item We propose an energy-aware PPO fine-tuning stage that expands the base quaternion skill toward large-attitude vertical arcs while retaining original-task capability through replay.
  \item We compose the learned skill through executable target streams and Receding-Horizon Target-Stream Optimization, selecting task-dependent stream parameters by closed-loop rollout through the frozen policy.
\end{enumerate}
```

---

## 4. Related Work 调整

保留当前 Related Work 的大方向，但请让每段都服务于本文 gap。建议短语式 paragraph 标题：

```latex
\paragraph{Learning-Control Interfaces for Agile Flight.}
\paragraph{Direct Learned Flight Control.}
\paragraph{Feasibility-Aware Trajectory Tracking.}
\paragraph{Evaluation Beyond Position Error.}
```

重点改法：

- Learning-control interface 部分：说明 CoRL 里常见做法是 learning 不一定直接替代整个 stack，而是放在一个 inspectable interface 上。然后转到本文：我们的 inspectable interface 不是 waypoint/speed，而是 attitude-speed target stream。
- Direct learned control 部分：强调本文不是普通 end-to-end policy，也不是 PID/autopilot wrapper，而是 direct actuator-level skill conditioned on explicit targets。
- Feasibility-aware trajectory tracking 部分：说明别人优化 trajectory / time allocation / MPC，而本文优化的是 learned fixed-wing skill 可执行的 target stream。
- Evaluation 部分：强调 fixed-wing maneuvering 不能只看 CTE，要看 velocity-tangent、nose-velocity、wing-plane、energy、AoA、安全边界。

不要引入太多无关 citation，不要扩展成 survey。

---

## 5. Method 结构重组

请把 Method 重组为下面结构。小节标题不要用疑问句。

```latex
\section{Method}

\subsection{System Overview}
\subsection{Direct-Actuator Flight Skill}
\subsection{Euler-to-Quaternion Target Representation}
\subsection{Geometry-Aware Maneuver Metrics}
\subsection{Energy-Aware Vertical Skill Extension}
\subsection{Executable Target Streams}
\subsection{Receding-Horizon Target-Stream Optimization}
```

### 5.1 System Overview

保留当前 Figure 1，但 caption 要改成两层逻辑：

Training / skill expansion:
- Euler direct RL baseline;
- quaternion-conditioned direct-actuator skill;
- geometry-aware pseudo-tracking diagnosis;
- energy-aware PPO fine-tuning.

Execution / composition:
- reference maneuver;
- executable target stream;
- RH-TSO;
- frozen energy-aware skill;
- fixed-wing dynamics;
- geometry-aware evaluation.

如果当前 Figure 1 图片不包含训练扩展层，请先只修改 caption，不要强行修改图片文件。可以在 caption 中写 “The paper has two stages...” 但不要让 caption 与图完全矛盾。

### 5.2 Direct-Actuator Flight Skill

这里不要只写 “PID-free quaternion-conditioned”。要先写：

- why direct RL skill;
- action output is throttle, elevator, aileron, rudder, speed brake;
- policy learns actuator-level tracking under aerodynamic and energy coupling;
- this is different from an autopilot inner loop.

保留当前 Table `Flight-skill interface`，但可以把标题改成：

```latex
\caption{Direct-actuator flight-skill interface. The target is an attitude-speed command, not a waypoint.}
```

### 5.3 Euler-to-Quaternion Target Representation

新增或改写这一节。现在可以用定量表格，但必须谨慎表述：

- Euler baseline is not useless and should not be described as simply failed.
- Euler can perform well on simple level, heading, and pitch commands.
- Quaternion is advantageous for complex turning/3D composition tasks in the current table.
- Survival rates are not directly comparable because the two environments have different termination logic; keep the note in the table.
- The conclusion should be:
  - quaternion target encoding is selected because it provides a consistent SO(3)-based interface and improves complex curve maneuvers;
  - not because it wins every task.

保留并 polish 当前 table:

```latex
\begin{table*}[t]
  \caption{Comparison between Euler-angle and quaternion target encodings on representative flight maneuvers.
  Both policies use the same GRU Actor-Critic architecture, and only the target-attitude observation encoding is changed.}
  \label{tab:euler-vs-quat}
...
\end{table*}
```

建议 caption 改成更准确的：

```latex
\caption{
Euler-angle and quaternion target encodings on representative maneuvers.
Both policies use the same GRU Actor-Critic architecture; only the target-attitude observation differs.
Quaternion encoding improves complex turning and three-dimensional curve tasks, while simple pitch targets remain favorable to the Euler baseline.
}
```

正文不要说 “quaternion outperforms Euler on all tasks”。

### 5.4 Geometry-Aware Maneuver Metrics

把 metrics 提前放到 energy-aware extension 前面，因为它是发现 pseudo-tracking 的诊断工具，而不是单纯最后评价。

保留已有公式：
- velocity-tangent error;
- nose-tangent error;
- nose-velocity error;
- wing-plane error;
- quaternion geodesic error;
- plus energy, speed, alpha/beta, G-load, altitude safety.

强调 CTE is necessary but not sufficient.

### 5.5 Energy-Aware Vertical Skill Extension

这是核心方法，不要降级为 ablation。请写成 capability expansion stage。

必须写清楚：

- base quaternion skill can still pseudo-track large-attitude maneuvers;
- energy-aware fine-tuning starts from the base quaternion skill;
- PPO fine-tuning includes vertical-arc curriculum;
- reward includes energy retention, low-speed penalty, vertical progress, alpha/beta/G safety, action smoothness, and replay terms;
- replay prevents catastrophic forgetting of horizontal/mild 3D behavior;
- final skill is frozen before target-stream / RH-TSO experiments.

Use the new table in Results to support this method section, but do not duplicate all numbers in Method.

不要声称 180° half-loop solved。写清楚：
- 60°–150° vertical arcs are B-grade;
- 180° half-loop remains failure boundary.

### 5.6 Executable Target Streams

解释为什么 reference maneuver 不直接变成 actuator command：

- Full actuator sequence optimization is high-dimensional and brittle.
- Static waypoints omit attitude, lift, and energy preparation.
- Static attitude-speed targets omit how the maneuver evolves.
- Target streams provide a moving local contract between reference geometry and the learned skill.
- Stream variables: heading, pitch, roll, target airspeed.
- Stream parameters include lookahead, target speed, pitch offset / vertical shaping parameters if present.

如果代码库里能找到不同实验路线，请总结成 1–2 句：
“Preliminary variants such as static waypoint targets / static attitude targets / direct actuator search were less suitable because ...”
但是只写代码和日志中能支撑的内容。

### 5.7 RH-TSO

保留当前 Algorithm 1 和核心公式。

定位写清楚：

- It does not optimize actuators.
- It searches target-stream parameters.
- Each candidate is evaluated by closed-loop rollout through the frozen energy-aware skill.
- Deterministic lattice search is an implementation choice for low-dimensional stream parameters.
- Do not claim global optimality beyond the candidate set.
- Mention CEM/MPPI only as possible extension, not as completed result.

---

## 6. Experiments 结构重组

请把当前 `Experimental Setup` 和 `Results` 重组为一个 CoRL 风格的 `Experiments` section。小节标题用短语，不用疑问句。

推荐结构：

```latex
\section{Experiments}

\subsection{Experimental Setup}
\subsection{Maneuver Suite}
\subsection{Euler-to-Quaternion Skill Evolution}
\subsection{Pseudo-Tracking Diagnosis}
\subsection{Energy-Aware Skill Extension}
\subsection{Target-Stream Optimization}
\subsection{Vertical Maneuver Frontier}
\subsection{Ablations and Runtime}
```

如果篇幅超出，可以合并：
- `Pseudo-Tracking Diagnosis` + `Energy-Aware Skill Extension`;
- `Vertical Maneuver Frontier` 放入 `Energy-Aware Skill Extension`;
- `Ablations and Runtime` 放入 Discussion 或 appendix。

### 6.1 Experimental Setup

保留 simulator 和 policy 描述。强调：
- all results are simulation;
- high-fidelity fixed-wing simulator is experimental substrate, not the contribution;
- target-stream experiments use the frozen energy-aware skill unless otherwise stated.

### 6.2 Maneuver Suite

不要再单独设计一个 “Base Skill Capability” 表格来证明 base quaternion skill 可以完成水平和温和 3D。当前 `figure2` 已经把所有机动组合展示了，所以把它作为 Maneuver Suite / Representative Maneuvers 图。

修改 figure2 caption，使其承担两个作用：
1. 展示本文使用的任务集合；
2. 说明这些任务覆盖 horizontal curves, mild 3D curves, vertical pull-up / vertical arcs。

建议 caption：

```latex
\caption{
\textbf{Representative maneuver suite.}
The evaluation covers horizontal curves, mild three-dimensional curves, vertical pull-up tasks, and loop-like vertical arcs.
These tasks test both reusable attitude-speed skill tracking and long-horizon target-stream composition.
}
```

### 6.3 Euler-to-Quaternion Skill Evolution

把当前 `Euler vs. Quaternion Target Encoding` 小节改为这一小节。

使用 `Table~\ref{tab:euler-vs-quat}`，但正文结论必须精确：

建议写法：

```latex
Table~\ref{tab:euler-vs-quat} shows that the representation choice is task dependent. The Euler-angle baseline remains competitive on simple pitch and pull-up commands, but degrades on complex turning and curve-following maneuvers. For example, on the right $R5000$ circle, quaternion encoding reduces attitude error from $42.2^\circ$ to $11.0^\circ$ and heading error from $30.4^\circ$ to $5.7^\circ$. Similar reductions appear on the left circle, S-curve, and figure-eight tasks. We therefore use the quaternion-conditioned skill as the reusable executor for target-stream composition, while avoiding the stronger claim that it dominates Euler encoding on every individual target.
```

保留 table note：
- survival rates are not directly comparable;
- Euler environment remains permissive even when attitude error exceeds 40°.

### 6.4 Pseudo-Tracking Diagnosis

这里使用当前 figure3 或修改后的 pseudo-tracking figure。

目标：
- 展示 CTE 看起来近，但 AoA / nose-velocity / tangent / energy 状态错误；
- 说明这就是 geometry-aware metrics 的必要性。

### 6.5 Energy-Aware Skill Extension

这个实验结果可以和 6.4 使用同一张 figure，不需要重复画两张。但现在你还有 `Table~\ref{tab:vertical-extension}`，需要保留。

请使用这张表支撑两件事：

1. The fine-tuned vertical skill preserves many original capabilities:
   - level flight survival remains 1.00;
   - heading +45° survival remains 1.00 and attitude error improves 12.8° → 11.1°;
   - circle R5000 right survival remains 1.00 and stall rate improves 0.008 → 0.000;
   - S-curve survival remains 1.00 and stall rate improves 0.005 → 0.000.
2. It corrects catastrophic failures on negative heading changes:
   - heading -45°: survival 0.00 → 1.00, attitude error 56.9° → 11.9°;
   - heading -20°: survival 0.00 → 1.00, attitude error 62.2° → 7.8°, stall rate 0.375 → 0.000.

不要写成 fine-tuned skill improves every metric。因为：
- S-curve attitude error: 7.0° → 7.8°;
- 30° pull-up R8000 attitude error: 17.7° → 18.7°.

更准确表述为：
**energy-aware fine-tuning expands the safe/reliable envelope and reduces stall or catastrophic failure, while mostly preserving original maneuvering ability.**

建议修改 table caption：

```latex
\caption{
Energy-aware vertical extension.
The fine-tuned vertical skill preserves most original-task survival while correcting catastrophic failures on negative heading changes and reducing stall rates.
Each task is evaluated with 10 seeds.
}
```

### 6.6 Target-Stream Optimization

保留当前 RH-TSO result table：

- S-curve: best `(600,220)`, default `(1000,250)`, improvement 39%;
- Figure-eight: improvement 2%;
- Helix / mild-3D: improvement 24%;
- 90° vertical pull-up: best `(1500,280)`, improvement 47%.

说明：
- horizontal/mild 3D prefer shorter lookahead and lower speed;
- vertical pull-up prefers longer lookahead and higher speed;
- this supports task-dependent stream optimization.

不要说 `(600,220)` universally good。

建议正文结论：

```latex
The result is not that a single stream parameter is universally optimal. The best setting for horizontal and mild-3D curves is different from the vertical pull-up setting, which motivates receding-horizon selection rather than a fixed hand-tuned stream.
```

### 6.7 Vertical Maneuver Frontier

现在你已经有 numeric table，不要再写 “verify placeholder”。

请保留并 polish `Table~\ref{tab:loop-frontier}`。

使用数据：

```latex
60°:  CTE 180.0,   vel-tan 12.8,  nose-tan 18.1,  wing-plane 103.6, B
90°:  CTE 152.5,   vel-tan 10.1,  nose-tan 13.1,  wing-plane 66.2,  B
105°: CTE 138.3,   vel-tan 9.5,   nose-tan 12.3,  wing-plane 80.0,  B
120°: CTE 121.3,   vel-tan 8.8,   nose-tan 10.2,  wing-plane 69.5,  B
135°: CTE 120.1,   vel-tan 6.9,   nose-tan 7.5,   wing-plane 60.4,  B
150°: CTE 181.2,   vel-tan 6.6,   nose-tan 8.8,   wing-plane 49.2,  B
180°: CTE 19515.6, vel-tan 140.6, nose-tan 131.7, wing-plane 162.8, Fail
```

正文重点：
- 60°–150° arcs are not perfect but complete the reference geometry with B-grade quality;
- 180° half-loop fails catastrophically;
- this supports an honest capability boundary;
- the failure occurs around inverted-transition / loop-plane departure, not just ordinary tracking error.

Do not say “all large-angle maneuvers solved”.
Do not hide the large wing-plane errors for B-grade arcs. Acknowledge:
- B-grade arcs complete the geometry but still have non-negligible wing-plane errors.

### 6.8 Ablations and Runtime

如果已有数据，保留。没有数据就不要虚构。可以简短写：
- Euler-vs-quaternion ablation is reported in Table 1;
- target stream sweep is reported in Table 3;
- detailed runtime can be added if scripts/logs exist.

---

## 7. 删除或隐藏所有草稿痕迹

请扫描全文，删除正文中所有类似下面的句子：

- “final manuscript should include...”
- “supporting evaluation log is not present...”
- “not restated here because...”
- “metric values are intentionally left as verification placeholders...”
- “verify”
- “to be verified”
- “missing figures are added”
- “The next step is to replace...”
- “The manuscript currently treats...”

如果确实需要保留提醒，请改成 LaTeX 注释：

```latex
% TODO: insert final runtime table if available.
```

不要让这些句子出现在编译后的 PDF 正文中。

特别注意当前 Conclusion 里有：

```text
The next step is to replace the remaining verification placeholders with final loop-quality tables and representative trajectory figures.
```

这句话必须删除。现在已经有 loop-quality numeric table，不能再说 placeholder。

---

## 8. Figure / Table 使用原则

### Figure 1

保留为 System Overview。如果图片暂时没改，先修改 caption，让它强调：
- skill expansion stage;
- execution / composition stage;
- RH-TSO optimizes target streams, not actuators.

### Figure 2

保留当前所有机动组合展示图。把它作为 “Maneuver Suite” 或 “Representative Maneuvers”，不要再额外写一个单独的 base skill capability 表格。

### Figure 3

保留并升级为：
“Pseudo-tracking diagnosis and energy-aware correction”。

这张图同时服务于：
- Pseudo-Tracking Diagnosis;
- Energy-Aware Skill Extension。

建议 caption：

```latex
\caption{
\textbf{Pseudo-tracking diagnosis and energy-aware correction.}
A CTE-only solution can remain close to the spatial reference while flying with incorrect energy state, excessive angle of attack, or poor alignment between reference tangent, velocity, and nose direction.
The energy-aware fine-tuned skill produces a physically more consistent maneuver, improving tangent alignment and safety margins.
}
```

### Algorithm 1

保留 RH-TSO algorithm。

### Table: Policy Interface

保留并更新 caption。

### Table: Euler vs Quaternion

保留并作为 “Euler-to-Quaternion Skill Evolution” 的核心证据。不要过度解读。

### Table: Energy-Aware Vertical Extension

保留并作为 energy-aware PPO fine-tuning 的核心证据。强调:
- capability expansion;
- reduced catastrophic failures;
- reduced stall rates;
- retention of many original tasks.

### Table: RH-TSO Result

保留当前数值，但确认正文表述不夸大。

### Table: Loop Frontier

现在已有数值，不要再出现 `verify`。保留并说明 B-grade vs Fail 的能力边界。

---

## 9. 需要 Codex 在代码库里主动搜索的信息

请在项目代码库里搜索以下关键词，寻找能支撑正文的脚本、配置、日志或实验结果：

```text
euler
quat
quaternion
target_stream
rhtso
lookahead
airspeed
vertical
energy
pseudo
cte
tangent
nose
wing
alpha
beta
stall
done
rollout
lattice
waypoint
static_target
direct_action
actuator
ablation
```

搜索目标：

1. Euler-angle RL baseline 的实现或配置；
2. quaternion RL baseline 的实现或配置；
3. energy-aware vertical fine-tune 的 reward terms / config / curriculum；
4. replay terms 是否存在；
5. RH-TSO / lattice / rollout 代码；
6. target stream 参数 sweep 结果；
7. 是否有 static waypoint / static attitude / direct actuator search 等实验痕迹；
8. figure2 / figure3 对应的数据生成脚本；
9. loop frontier table 的数据生成脚本或日志；
10. energy-aware vertical extension table 的数据来源。

把找到的内容只用于支撑正文已有 claim。不要把代码中不成熟、失败、未验证的实验写成主贡献。

---

## 10. 写作风格要求

1. 所有 section/subsection 标题尽量短语式，不用问句。
2. 不要使用 “we tried many things” 这种口语表达。
3. 不要把 Planax / simulator 本身写成主贡献。
4. 不要夸大为 full aerobatics 或 solved half-loop。
5. 不要 claim real-world deployment。
6. 不要 claim quaternion universally outperforms Euler。
7. 保留 CoRL 风格：问题清楚、接口清楚、图表支撑、限制诚实。
8. 用 “capability expansion” 描述 energy-aware PPO fine-tuning。
9. 用 “pseudo-tracking” 连接 geometry-aware metrics 和 energy-aware extension。
10. 用 “executable target stream” 连接 frozen skill 和 RH-TSO。
11. 对负面结果保持诚实：
    - quaternion not best on all tasks;
    - B-grade vertical arcs still have non-negligible errors;
    - 180° half-loop fails catastrophically.

---

## 11. 推荐最终论文结构

请尽量调整成以下结构：

```latex
\section{Introduction}

\section{Related Work}
\paragraph{Learning-Control Interfaces for Agile Flight.}
\paragraph{Direct Learned Flight Control.}
\paragraph{Feasibility-Aware Trajectory Tracking.}
\paragraph{Evaluation Beyond Position Error.}

\section{Method}
\subsection{System Overview}
\subsection{Direct-Actuator Flight Skill}
\subsection{Euler-to-Quaternion Target Representation}
\subsection{Geometry-Aware Maneuver Metrics}
\subsection{Energy-Aware Vertical Skill Extension}
\subsection{Executable Target Streams}
\subsection{Receding-Horizon Target-Stream Optimization}

\section{Experiments}
\subsection{Experimental Setup}
\subsection{Maneuver Suite}
\subsection{Euler-to-Quaternion Skill Evolution}
\subsection{Pseudo-Tracking Diagnosis}
\subsection{Energy-Aware Skill Extension}
\subsection{Target-Stream Optimization}
\subsection{Vertical Maneuver Frontier}
\subsection{Ablations and Runtime}

\section{Discussion and Limitations}

\section{Conclusion}
```

如果篇幅超出 CoRL 限制，可以合并：
- `Pseudo-Tracking Diagnosis` 和 `Energy-Aware Skill Extension`;
- `Vertical Maneuver Frontier` 放入 `Energy-Aware Skill Extension`;
- `Ablations and Runtime` 放入 appendix 或 Discussion。

---

## 12. 重点段落可直接参考的表述

可以把 Introduction 或 Method 中相关句子改写成类似下面的形式。

### Direct RL and representation evolution

```latex
Our development follows this failure-driven progression. We first remove the hand-tuned inner-loop controller and train a direct-actuator RL skill, so that the policy learns under the same aerodynamic coupling, actuator limits, and energy loss that determine maneuver feasibility. An initial Euler-angle target encoding is sufficient for simple and moderate targets, but it degrades on complex three-dimensional curve-following tasks. The quaternion-conditioned interface substantially reduces attitude and heading errors on circle, S-curve, and figure-eight maneuvers, so we use it as the reusable executor for target-stream composition.
```

### Careful interpretation of Euler vs quaternion table

```latex
The comparison should not be read as a universal dominance claim. The Euler-angle baseline remains favorable on some simple pitch commands. However, for the complex turning maneuvers that matter most for target-stream composition, the quaternion-conditioned skill provides a more consistent interface and substantially lower attitude and heading errors.
```

### Pseudo-tracking

```latex
Geometry-aware evaluation reveals a second limitation: small cross-track error does not imply a valid fixed-wing maneuver. In large-attitude vertical arcs, the aircraft can remain close to the spatial reference while flying with the wrong nose direction, velocity tangent, wing plane, or energy state. We refer to this behavior as pseudo-tracking.
```

### Energy-aware extension

```latex
The energy-aware vertical extension is introduced to address this diagnosed failure mode. Starting from the quaternion-conditioned base skill, we fine-tune with vertical-arc curricula, energy-retention and low-speed safety terms, geometry-alignment rewards, and replay of original tasks. This stage expands the safe and reliable envelope of the skill, correcting catastrophic failures on several held-out heading changes and reducing stall rates, but it does not solve the full half-loop.
```

### Target streams

```latex
A reference maneuver is not converted directly into actuator commands. Full-horizon actuator search is high-dimensional and brittle, while a static waypoint or static attitude target does not specify how attitude, speed, and energy should evolve through the maneuver. We instead represent the maneuver as an executable target stream: a sequence of local heading, pitch, roll, and airspeed commands that the frozen skill can attempt to track.
```

### RH-TSO

```latex
RH-TSO optimizes this interface rather than the actuators. At each replanning step, candidate target-stream parameters are evaluated by closed-loop rollout through the frozen energy-aware skill and the fixed-wing dynamics. The selected stream is therefore not merely geometrically close to the reference, but executable under the learned skill's closed-loop behavior.
```

### Vertical frontier

```latex
The loop-quality frontier is deliberately reported as a boundary rather than a success claim. The $60^\circ$--$150^\circ$ arcs complete the reference geometry with B-grade quality, but their wing-plane errors remain non-negligible. The $180^\circ$ half-loop fails catastrophically, with kilometer-scale CTE divergence and large tangent-alignment errors. This identifies the inverted-transition phase as the main remaining boundary of the current single-skill system.
```

---

## 13. 最终检查清单

修改完成后，请检查：

- [ ] Title 已改成 `Energy-Aware Target Streams for Learned Fixed-Wing Maneuvering`
- [ ] Abstract 包含 direct RL skill、Euler-to-quaternion evolution、pseudo-tracking、energy-aware fine-tune、target streams、RH-TSO、capability boundary
- [ ] Introduction 不是只从 quaternion skill 开始，而是解释 RL baseline → Euler limitation → quaternion skill
- [ ] Euler-vs-quaternion 表格被正确解释：复杂曲线任务 quaternion 更好，但不是所有任务都更好
- [ ] Energy-aware PPO fine-tuning 是主贡献之一，而不是普通 ablation
- [ ] Energy-aware vertical extension 表格被正确解释：扩展安全/可靠边界、修复负 heading catastrophic failures、降低 stall rate，但不是每个标量都提升
- [ ] Reference maneuver 为什么不直接变 actuator command 已解释清楚
- [ ] Figure 2 不再被重复成一个额外的 base skill capability 表格
- [ ] Figure 3 同时支持 pseudo-tracking diagnosis 和 energy-aware correction
- [ ] Loop frontier 表格已经使用真实数值，不再出现 verify placeholders
- [ ] 所有小节标题都是短语式，不用问句
- [ ] 正文无 `verify`、`missing logs`、`final manuscript should include` 等草稿痕迹
- [ ] 不声称 180° half-loop solved
- [ ] 不声称真实飞行验证
- [ ] 不把 simulator 写成主贡献
- [ ] RH-TSO 被描述为 target-stream parameter optimization，不是 actuator-space MPC
