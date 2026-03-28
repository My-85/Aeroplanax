# RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization 

Siwei Zhang, Yun Xiong*, Xi Chen, Zi'an Jia, Renhong Huang, Jiarong Xu, Jiawei Zhang<br>swzhang24@m.fudan.edu.cn<br>Fudan University, Zhejiang University, UC Davis


#### Abstract

Agentic Reinforcement Learning (Agentic RL) has shown remarkable potential in large language model-based (LLM) agents. These works can empower LLM agents to tackle complex tasks via multistep, tool-integrated reasoning. However, an inherent limitation of existing Agentic RL methods is their reliance on a pure on-policy paradigm for exploration, restricting exploration to the agent's selfgenerated outputs and preventing the discovery of new reasoning perspectives for further improvement. While recent efforts incorporate auxiliary off-policy signals to enhance exploration, they typically utilize full off-policy trajectories for trajectory-level policy estimation, overlooking the necessity for the fine-grained, step-level exploratory dynamics within agentic rollout. In this paper, we revisit exploration in Agentic RL and propose $\underline{\mathbf{R}}$ etrieval- $\underline{\mathbf{A}}$ ugmented $\underline{\mathbf{P}}$ olicy Optimization (RAPO), a novel RL framework that introduces retrieval to explicitly expand exploration during training. To achieve this, we decompose the Agentic RL training process into two phases: (i) Hybrid-policy Agentic Rollout, and (ii) Retrieval-aware Policy Optimization. Specifically, we propose a Hybrid-policy Agentic Rollout strategy, which allows the agent to continuously reason over the retrieved off-policy step-level traces. It dynamically extends the agent's reasoning receptive field, enabling broader exploration conditioned on external behaviors. Subsequently, we introduce the Retrieval-aware Policy Optimization mechanism, which calibrates the policy gradient estimation with retrieval reward and importance shaping, stabilizing training and prioritizing retrieval-illuminating exploration. Extensive experiments show that RAPO achieves an $+5.0 \%$ average gain on fourteen datasets across three agentic reasoning tasks, while delivering $\mathbf{1 . 2}$ x faster training efficiency.


## Keywords

Agentic RL; Tool-integrated Reasoning; Large Language Models

## ACM Reference Format:

Siwei Zhang, Yun Xiong*, Xi Chen, Zi'an Jia, Renhong Huang, Jiarong Xu, Jiawei Zhang. 2026. RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'26), August 9-13, 2026, Jeju, Korea. ACM, New York, NY, USA, 19 pages. https: //doi.org/XXXXXXX.XXXXXXX

[^0]
## 1 Introduction

Large Language Model-based (LLM) agents [38, 47, 59] have demonstrated expressive tool-integrated reasoning capabilities across a broad range of real-world tasks. Unlike traditional single-step reasoning [22, 53], LLM agents invoke multi-step reasoning loops through iterative tool calls, enabling dynamic, interactive engagement with environments [10]. Particularly noteworthy is how such capabilities have been realized through Agentic Reinforcement Learning (Agentic RL) [13]. These methods [9, 15, 25] typically adopt group-based algorithms, e.g., GRPO [45], to optimize the step-level reasoning of agents, thus improving their task-solving performance.

Nonetheless, effective policy exploration [9, 42] remains an open challenge in Agentic RL, as it requires the policy to discover sufficiently diverse reasoning trajectories during rollout. Toward this goal, recent Agentic RL methods propose rollout reconstruction, such as branching [11] or tree-search [25], to enrich the candidate behaviors available for step-level exploration. Although effective to some extent, a fundamental limitation inherent in these methods is their rigid reliance on an on-policy exploration, where the model explores and learns exclusively from its self-generated reasoning trajectories across repeated trials. As shown in Fig. 1(a), such a pure on-policy paradigm unconsciously constrains the global exploration space to the intrinsic behaviors of the base agent, leading to insufficient exploration for further performance improvement. This hypothesis is also supported by recent empirical findings [16,67,69], which show that on-policy RL predominantly amplifies pre-existing behaviors within the base LLM, rather than uncovering new reasoning strategies beyond its native exploratory horizon.

To overcome the aforementioned exploration boundary, recent works [51, 64] integrate auxiliary off-policy signals into RL training and exploit external reasoning behaviors to advance policy optimization. However, as illustrated in Fig. 1(b), these approaches leverage the entire off-policy trajectories solely for trajectory-level group computation, overlooking the fine-grained, step-level dynamics within agentic reasoning. As a result, they merely strengthen the policy estimation over the already-observed rollouts in a static manner, failing to dynamically, explicitly enlarge exploration throughout the rollout process. This observation inspires us to ask: Is it possible to explicitly inject off-policy signals into the step-level rollout process in Agentic RL, extending the agent's reasoning receptive field and thereby unlocking broader exploration?

In this paper, we improve Agentic RL at the algorithmic level and propose $\underline{\text { Retrieval- }} \underline{\text { Augmented }} \underline{\text { Policy }} \underline{\text { Optimization (RAPO), a novel }}$ RL framework that introduces retrieval to explicitly broaden exploration during training. Beyond existing Agentic RL methods that only generate pure on-policy trajectories, as depicted in Fig. 1(c), RAPO incorporates a Hybrid-policy Agentic Rollout strategy that

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-02.jpg?height=607&width=1744&top_left_y=292&top_left_x=186)
Figure 1: Comparison between existing methods and our framework. (a) Existing Agentic RL methods are inherently on-policy, resulting in a limited exploration space bounded by the native agent. (b) Off-policy-enhanced RL methods statically integrate full off-policy trajectories for trajectory-level policy estimation, failing to capture the dynamic, step-level exploration within agentic rollout. (c) Our RAPO introduces retrieval and allows the on-policy agent to continuously reason over the retrieved offpolicy step-level traces, explicitly expanding its reasoning receptive field for exploration and thus increasing rollout diversity.

produces hybrid-policy trajectories via off-policy-conditioned reasoning. This design supports the agent to continuously reason over the retrieved off-policy step-level traces, substantially extending its reasoning receptive field and exposing the model to adaptively absorb these external behaviors for subsequent richer exploration.

While incorporating external traces can facilitate exploration, it may inevitably introduce instability during policy optimization. To tackle this issue, we further introduce a Retrieval-aware Policy Optimization mechanism to calibrate the policy gradient estimation within RAPO. Specifically, we design a Retrieval Reward that automatically encapsulates and quantifies the impact of retrieval within each rollout. This reward enables the agent to understand how retrieved information contributes to its reasoning, providing a principled signal for retrieval-aware exploration. Meanwhile, we propose a Retrieval Importance Shaping that selectively prioritizes retrieval-informative tokens during optimization. It rebalances policy estimation using the retrieved-token proportion, encouraging the model to allocate greater attention to those externally-expanded behaviors. Notably, RAPO applies retrieved token masking for stable optimization. As such, our framework can reduce the on-policy generation during agentic rollout while minimizing the number of gradient-bearing tokens within policy updates, ultimately leading to improved training efficiency.

In summary, our key contributions are as follows:

- We address the challenge of exploration in Agentic RL and present RAPO . To the best of our knowledge, RAPO is the first Agentic RL framework that explicitly harnesses retrieval to augment the agent's step-level exploration capabilities during training.
- We develop a Hybrid-policy Agentic Rollout to allow the agent to continuously reason over the retrieved off-policy traces, promoting expanded exploration and thus facilitating rollout diversity.
- We propose a Retrieval-aware Policy Optimization, which calibrates policy gradient estimation through a carefully-designed retrieval reward and importance shaping, effectively improving training stability and overall effectiveness of RAPO.
- Extensive experiments on fourteen benchmarks across three agentic reasoning tasks demonstrate that RAPO consistently outperforms baselines with $5.0 \%$ average gains, while exhibiting markedly 1.2 x faster training efficiency.


## 2 Related Work

### 2.1 Agentic Reinforcement Learning

Agentic Reinforcement Learning (Agentic RL) [3, 4, 14, 32, 37] plays an important role for LLM agents, enabling them to interact with tool environments (e.g., python or search engines) [63] and perform multi-step reasoning guided by feedback. To effectively facilitate such tool-integrated reasoning, existing Agentic RL methods typically employ GRPO-based algorithms [26, 41,50] to train agents to invoke tools for improved downstream problem-solving. Despite these advances, effective policy exploration remains a fundamental challenge in Agentic RL [30, 42, 49]. To this end, recent works attempt to address this limitation via rollout reconstruction, such as adaptive branching [11] or tree-search [25], which significantly improve the exploration capabilities of agents during training.

While effective to some degree, these Agentic RL methods are inherently on-policy, restricting exploration to the native behaviors of the base agent. Instead, our RAPO allows the agent to reason over the retrieved off-policy step-level traces during rollout, explicitly expanding its reasoning receptive field for better exploration.

### 2.2 On- and Off-Policy RL for LLM Training

Based on how experience is utilized during policy optimization, RL for LLM training can be divided into two categories: on-policy and off-policy methods [29, 36, 68]. On-policy methods strictly update the policy through reasoning trajectories generated by the current policy LLM [8, 37, 62], ensuring training stability but potentially constraining the exploration space [35, 67, 69]. To mitigate this issue, recent works augment on-policy RL with off-policy signals to improve performance. They typically introduce an auxiliary

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-03.jpg?height=656&width=1743&top_left_y=283&top_left_x=191)
Figure 2: Overview of the RAPO. RAPO introduces a Hybrid-policy Agentic Rollout strategy that supports off-policy-conditioned reasoning, which enables the agent to receive the retrieved off-policy traces to broaden exploration beyond its intrinsic reasoning behaviors. Meanwhile, it incorporates a Retrieval-aware Policy Optimization mechanism with retrieval reward and importance shaping, ensuring effective and stable policy gradient estimation during training.

LLM [64] or a replay buffer [51] to integrate off-policy trajectories for optimization, enabling effective exploration during training.

The above methods focus on single-step reasoning and incorporate off-policy signals solely for trajectory-level policy estimation. In contrast, we integrate off-policy traces into the multi-step agentic rollout process, preserving step-level exploration dynamics while providing new insights during RL training.

### 2.3 Entropy-Related RL for LLM Training

Entropy [5,55] serves as a well-established measure of model uncertainty, which has been widely used in LLM post-training [13, 71]. Recent works exploit entropy to monitor the reasoning states of LLMs during rollout [57] and incorporate it into policy optimization to improve performance [70]. In the context of Agentic RL, entropy has also been leveraged to enable adaptive branching during rollout [11] and construct balanced supervisory signals during optimization [9], thus enhancing the tool-use capabilities of agents.

Unlike these entropy-related works, we leverage entropy to quantify the contribution of retrieval to the agent's reasoning, allowing us to assess the retrieval-aware exploration for policy estimation.

## 3 Preliminaries

### 3.1 Problem Definition

Definition 3.1. Multi-step Agentic Reasoning. Multi-step agentic reasoning performs iterative Thought-Action-Observation circles between an LLM agent and an external tool environment for task solving. Given a query $q$, at each step $t=\{0,1, \ldots, T-1\}$, the agent $\pi_{\theta}$ generates a thought $\tau_{t}$ and an action $\alpha_{t}$ based on the current reasoning history. The action $\alpha_{t}$ is a parsable instruction that can be executed by tools such as python or search engines, which then returns an observation $o_{t}$ as feedback. We define the resulting reasoning trace at step $t$ as $s_{t}=\left(\tau_{t}, \alpha_{t}, o_{t}\right)$, and denote a complete
$T$-step reasoning trajectory as $\mathcal{S}=\left(s_{0}, s_{1}, \cdots, s_{T-1}\right)$. We also use $\mathcal{S}_{<t}=\left(s_{0}, \ldots, s_{t-1}\right)$ to represent the reasoning history before step $t$. Definition 3.2. Agentic RL. Given a query $q$ sampled from the dataset $Q$, an agent $\pi_{\theta}$, Agentic RL aims to maximize the expected reward of $T$-step agentic reasoning trajectory $\mathcal{S} \sim \pi_{\theta}(\cdot \mid q ; T)$ by:

$$
\begin{equation*}
\max _{\pi_{\theta}} \mathbb{E}_{q \sim Q, \mathcal{S} \sim \pi_{\theta}(\cdot \mid q ; T)}[R(q, \mathcal{S})], \tag{1}
\end{equation*}
$$

where $R(\cdot)$ denotes a rule-based or model-based reward function [45].

### 3.2 Entropy Computation for Step-level Traces

Entropy provides a quantitative measure of uncertainty in LLMgenerated token distributions, which has been widely adopted to monitor the real-time reasoning states of LLMs during training and inference [5,55,57,70]. For the $i$-th token within the $t$-th step-level reasoning trace $s_{t}$, let $\mathbf{p}_{i} \in \mathbb{R}^{V}$ denote its probability vector over the vocabulary size $V$, the token-level entropy is computed as:

$$
\begin{equation*}
H_{s_{t}, i}=-\sum_{j=1}^{V} p_{i, j} \log p_{i, j}, \text { where } \mathbf{p}_{i}=\operatorname{Softmax}\left(\mathbf{z}_{i}\right) \tag{2}
\end{equation*}
$$

Here, $\mathbf{z}_{i} \in \mathbb{R}^{V}$ represents the pre-softmax logits of the $i$-th token within $s_{t}$. The entropy of the step-level trace $s_{t}$ is then obtained by averaging all token-level entropies as $H_{s_{t}}=\operatorname{Mean}\left(\left\{H_{s_{t}, i}\right\}_{i=1}^{\left|s_{t}\right|}\right) \in \mathbb{R}$.

## 4 Methodology

As discussed earlier, the exploration capacity of existing Agentic RL methods is inherently bounded by the native agent. To tackle this issue, we propose RAPO, a novel Agentic RL framework that introduces retrieval to explicitly enlarge exploration during training. To achieve this, our RAPO consists of two key components: (i) a Hybrid-policy Agentic Rollout strategy that conducts dynamic off-policy-conditioned reasoning for step-level exploration expansion; and (ii) a Retrieval-aware Policy Optimization mechanism to
ensure training stability and effectiveness. We will introduce these components in the following subsections.

### 4.1 Hybrid-policy Agentic Rollout

In this section, we introduce Hybrid-policy Agentic Rollout, a strategy that retrieves the off-policy traces from a Step-Trace Buffer and seamlessly inserts them into the ongoing multi-step agentic rollout.
4.1.1 Step-Trace Buffer. Unlike existing works [51] that rely on trajectory-grained buffers for experience replay, we instead construct a Step-Trace Buffer to record step-grained reasoning traces collected from an off-policy agent. Concretely, for each query $q$ within the training dataset, we first employ an off-policy agent $\pi_{\theta_{\text {off }}}$ to generate $N$ independent multi-step reasoning trajectories $\left\{\hat{\mathcal{S}}^{n}\right\}_{n=1}^{N}$, each accompanied by its corresponding outcome reward. To ensure buffer quality, we apply reward-aware filtering and retain only the top-K reward-ranked trajectories. Each preserved trajectory $\hat{\mathcal{S}}=\left(\hat{s}_{0}, \ldots, \hat{s}_{T-1}\right)$ is then decomposed into individual step-level traces, which are stored separately in the Buffer.

As illustrated in Fig. 2, the stored items can be organized as a series of step-level key-value pairs, i.e., $\left\{\right.$ Key: $\hat{\mathcal{S}}_{<t}$, Value: $\left.\hat{s}_{t}\right\}$, where $\hat{\mathcal{S}}_{<t}$ denotes the off-policy reasoning history before each step $t$, and $\hat{s}_{t}$ is the output off-policy trace at that step. Such a step-grained design enables the Buffer to capture localized, context-conditioned reasoning behaviors, rather than treating full trajectories as units. Consequently, our Buffer maintains a large corpus of diverse and high-quality off-policy traces, which serve as reusable reasoning primitives for context-similarity retrieval in hybrid-policy rollout.
4.1.2 Retrieval from Step-Trace Buffer. To explicitly broaden exploration during rollout, we enable the model to retrieve relevant off-policy traces from the Step-Trace Buffer and harness these external behaviors to dynamically extend the agent's reasoning receptive field. Specifically, when retrieval is triggered at the $t$-th step, we construct a retrieval query using the current on-policy reasoning history $\mathcal{S}_{<t}$. A standard RAG-based retrieval procedure [27] is then performed over the Buffer, which returns the Value (i.e., $\hat{s}_{t}$ ) whose Key (i.e., $\hat{\mathcal{S}}_{<t}$ ) is most aligned with the retrieval query (i.e., $\mathcal{S}_{<t}$ ):

$$
\begin{equation*}
\hat{s}_{t} \sim \operatorname{Retrieve}\left(\mathcal{S}_{<\mathrm{t}}\right) . \tag{3}
\end{equation*}
$$

Here, $\hat{s}_{t}$ is the retrieved off-policy step-level trace, where its associated off-policy input history is most similar to the current on-policy contexts. We then concatenate $\hat{s}_{t}$ into the agent's working contexts, allowing the agent to continuously reason over this external trace in subsequent reasoning steps. Notably, both retrieval and concatenation are performed at the step level, which enables off-policy signals to modulate the exploration dynamics within agentic rollout.
4.1.3 Rollout Procedure. Unlike existing works [64] that integrate off-policy signals for static, trajectory-level policy estimation, RAPO utilizes the retrieved off-policy traces to facilitate dynamic, step-level rollout exploration. Concretely, for each query $q$, RAPO produces two sets of rollouts: $N_{\text {on }}$ pure on-policy trajectories, and $N_{\text {hybrid }}$ hybrid-policy trajectories, in which the off-policy traces are interleaved with on-policy reasoning. These hybrid-policy trajectories are produced via a Retrieval-then-Reasoning process as follows:
(1) Initialization: For each query $q$, the on-policy agent first generates $N_{\text {hybrid }}$ first-step reasoning traces $s_{0} \sim \pi_{\theta}(\cdot \mid q ; t=0)$ as the initialization of reasoning contexts.
(2) Retrieval Sampling: At each subsequent step ( $t>0$ ), we introduce retrieval sampling to probabilistically decide whether the next reasoning trace is generated by the on-policy agent, i.e., $s_{t} \sim \pi_{\theta}\left(\cdot \mid \mathcal{S}_{<t} ; t\right)$, or retrieved from the Step-Trace Buffer, i.e., $s_{t} \sim$ Retrieve ( $\mathcal{S}_{<t}$ ), through the following distribution:

$$
\Phi_{\text {hybrid }}\left(s_{t}=a \mid \mathcal{S}_{<t}\right)= \begin{cases}0.5, & \text { if } t>0, a \sim \pi_{\theta}\left(\cdot \mid \mathcal{S}_{<t} ; t\right)  \tag{4}\\ 0.5, & \text { if } t>0, a \sim \operatorname{Retrieve}\left(\mathcal{S}_{<t}\right)\end{cases}
$$

(3) Off-policy-conditioned Reasoning: Once retrieval is triggered, the agent proceeds to reason conditioned on this retrieved trace, allowing external behaviors to impact its decisions.
Such a hybrid-policy rollout enables the on-policy agent to dynamically absorb off-policy traces beyond its native reasoning perspectives. As a result, the agent's reasoning receptive field is substantially broadened by these external behaviors, promoting richer exploration during rollout. Notably, while retrieval sampling introduces stochasticity for diversity, it may also yield trajectories that never trigger retrieval. Therefore, the number of hybrid trajectories per query is variable with the upper-bound of $N_{\text {hybrid }}$.

### 4.2 Retrieval-aware Policy Optimization

Incorporating external traces inevitably introduces noise into policy optimization, potentially leading to training instability and suboptimal performance. To address this issue, we propose Retrieval-aware Policy Optimization, which leverages a retrieval reward and an importance shaping mechanism to regularize policy updates.
4.2.1 Retrieval Reward. Retrievals within RAPO are not guaranteed to always benefit the reasoning process of agent, as they inherently lack any explicit supervision or heuristics. This makes it essential to assess the contribution of each retrieval to the agent's reasoning, so that the model can reliably discriminate between constructive guidance and misleading interference. However, evaluating retrieval contribution is non-trivial and presents two key challenges: (i) how to quantify retrieval quality, i.e., whether retrieval truly provides useful external behaviors; and (ii) how to ensure timely retrieval, i.e., whether retrieval is triggered at stages when external information is actually needed. To tackle these challenges, we draw inspiration from recent entropy-related RL methods [5, 55, 57], which utilize entropy as a proxy for reasoning uncertainty in LLMs, enabling effective monitoring of rollout dynamics [11] and providing invaluable supervision signals for optimization [9]. Motivated by this, we propose an entropy-based retrieval reward that jointly captures retrieval quality and timing as follows:

- Retrieval Quality. Intuitively, if a retrieval successfully provides helpful external behaviors, it will reduce the model's uncertainty and make subsequent reasoning more confident. For a retrieved off-policy trace $\hat{s}_{t}$ at step $t$, we measure its impact on entropy reduction by comparing the entropy before and after retrieval:

$$
\begin{equation*}
H_{\Delta \hat{s}_{t}}=-\frac{H_{\hat{s}_{t+1}}-H_{\hat{s}_{t-1}}}{H_{\hat{s}_{t-1}}} \tag{5}
\end{equation*}
$$

where $H_{\hat{s}_{t-1}}$ and $H_{\hat{s}_{t+1}}$ denote the step-level entropy before and after retrieval, respectively. To enhance discrimination and smooth
the score, we then apply a scaling factor with an activation as:

$$
\begin{equation*}
g_{\hat{s}_{t}}=\tanh \left(2 \cdot H_{\Delta \hat{s}_{t}}\right) \in(-1,1) \tag{6}
\end{equation*}
$$

Here, a positive $g_{\hat{s}_{t}}$ indicates that the corresponding retrieval has reduced uncertainty and provided benefits for the ongoing reasoning, whereas a negative value suggests misleading.

- Retrieval Timing. In RAPO, retrieval-conditioned reasoning aims to encourage the agent to generate more diverse rollouts. Hence, retrieval should ideally occur when the agent is exhibiting strong exploratory behavior, where its reasoning is more susceptible to impact from external traces. High pre-retrieval entropy, $H_{\hat{s}_{t-1}}$, naturally provides such exploratory states. Consequently, we utilize $H_{\hat{s}_{t-1}}$ as a timing signal within retrieval reward.
Overall, for a retrieval triggered at step $t$, we define its reward as:

$$
\begin{equation*}
Z_{\mathrm{ret}}\left(\hat{s}_{t}\right)=g_{\hat{s}_{t}} \cdot H_{\hat{s}_{t-1}}, \tag{7}
\end{equation*}
$$

where $g_{\hat{s}_{t}} \in(-1,1)$ evaluates the quality of the retrieved trace and $H_{\hat{s}_{t-1}}>0$ reflects the retrieval timing. In this way, retrievals that (i) reduce uncertainty and (ii) occur during high-uncertainty states tend to receive higher rewards, while misleading or poorly timed retrievals are more likely to obtain lower or negative rewards. We then average the retrieval rewards across all retrievals within each hybrid-policy trajectory and denote it as $Z_{\text {ret }}=\operatorname{Mean}\left(Z_{\text {ret }}\left(\hat{s}_{t}\right)\right)$. We also assign $Z_{\text {ret }}=0$ for each pure on-policy trajectory.
4.2.2 Retrieval Importance Shaping. Integrating retrievals into RL training introduces an additional challenge, as these retrieved offpolicy tokens do not support gradient backpropagation throughout the training process. This results in sparse gradient signals within hybrid-policy trajectories, leading to insufficient optimization for effectively capturing retrieval-aware exploration. To mitigate this issue, we introduce a Retrieval Importance Shaping mechanism, which rebalances gradients by upweighting on-policy tokens generated under off-policy-conditioned reasoning. The key idea is to compensate the GRPO token-level importance sampling ratio [45] with the retrieved-token proportion, ensuring that the model allocates greater optimization focus to the sparse-gradient contexts within hybrid-policy trajectories.

Formally, given a query $q$ and a policy agent $\pi_{\theta}$, for its generated $j$-th token at step $t, s_{t, j}$, we reshape the GRPO importance sampling ratio $r_{t, j}(\theta)$ [45] using retrieved-token proportion $\mathcal{F}_{\text {ret }}$ :

$$
\begin{equation*}
\hat{r}_{t, j}(\theta)=\left(1+m \cdot \mathcal{F}_{\mathrm{ret}}\right) \cdot r_{t, j}(\theta), \tag{8}
\end{equation*}
$$

where $m>0$ is a predefined hyper-parameter and $\mathcal{F}_{\text {ret }} \in(0,1)$ denotes the proportion of the retrieved tokens to the full trajectory length; $r_{t, j}(\theta)=\pi_{\theta}\left(s_{t, j} \mid q, \mathcal{S}_{<t}\right) / \pi_{\theta_{\text {old }}}\left(s_{t, j} \mid q, \mathcal{S}_{<t}\right)$ is the importance sampling ratio to calibrate the gradient based on policy gradient theory [52], as the solutions are generated by the old policy $\pi_{\theta_{\text {old }}}$ before the update. Note that $\mathcal{F}_{\text {ret }}=0$ for on-policy trajectories, and thus this mechanism introduces no bias in their policy updates. In doing so, the retrieval-driven exploration is effectively consolidated.
4.2.3 Training Objective. Now, we introduce the training objective of our RAPO. Similar to GRPO [45], given a query $q$ and a group of rollouts $\left\{\mathcal{S}^{i}\right\}_{i=1}^{G}$, we compute the advantage of retrieval rewards as:

$$
\begin{equation*}
A_{\mathrm{ret}}^{i}=\frac{Z_{\mathrm{ret}}^{i}-\operatorname{Mean}\left(\left\{Z_{\mathrm{ret}}^{i}\right\}_{i=1}^{G}\right)}{\operatorname{Std}\left(\left\{Z_{\mathrm{ret}}^{i}\right\}_{i=1}^{G}\right)} . \tag{9}
\end{equation*}
$$

Then, we follow AEPO [9] and combine the retrieval advantage with the outcome advantage $A_{\text {acc }}^{i}$ for each rollout by:

$$
\begin{equation*}
A_{\mathrm{RAPO}}^{i}=\left(1+a \cdot A_{\mathrm{ret}}^{i}\right) \cdot A_{\mathrm{acc}}^{i} \tag{10}
\end{equation*}
$$

where $a>0$ is a predefined hyper-parameter. Finally, the training objective of RAPO is defined as follows:

$$
\begin{align*}
J_{\mathrm{RAPO}}(\theta)= & \mathbb{E}_{q \sim Q, \mathcal{S} \sim \pi_{\mathrm{RAPO}}(\cdot \mid q)}\left[\frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \mathcal { S } ^ { i } | } \sum _ { j = 1 } ^ { | \mathcal { S } ^ { i } | } \operatorname { m i n } \left(\hat{r}_{i, j}(\theta) \hat{A}_{\mathrm{RAPO}}^{i}\right.\right. \\
& \left.\left.\operatorname{clip}\left(\hat{r}_{i, j}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{\mathrm{RAPO}}^{i}\right)-\beta \mathbb{D}_{\mathrm{KL}}\right] \tag{11}
\end{align*}
$$

where $\operatorname{clip}(\cdot)$ clamps the importance ratio into $[1-\epsilon, 1+\epsilon]$ to ensure that the current policy is within the trust region [44], and $\mathbb{D}_{\mathrm{KL}}$ is the KL divergence between the current and reference policies [45].

## 5 Experiments

### 5.1 Experimental Settings

5.1.1 Datasets. We conduct experiments with $\mathbf{1 4}$ datasets across three multi-step agentic reasoning tasks: (i) Computational Reasoning, including GSM8K [6], MATH [23], MATH500 [33], AIME2024, and AIME2025 ${ }^{1}$; (ii) Knowledge-Intensive Reasoning, including WebWalkerQA [61], HotpotQA [65], 2WikiMultihopQA [24], Musique [54], and Bamboogle [40]; and (iii) Web-Agentic Reasoning, including SimpleQA [40], GAIA [39], WebWalkerQA [61], and BrowseComp [58]. Due to page limitations, details of these datasets are provided in Sec. C. 1 of the Appendix. All dataset splits follow the standard settings adopted in existing Agentic RL methods [11, 25, 27].
5.1.2 Baselines. For comparison, we select $\mathbf{1 3}$ baselines from three families: (i) Tool-Integrated Reasoning Methods, including Searcho1 [31], Search-R1 [27], and ToolStar [10]; (ii) Off-policy Learning Methods, including SFT, RolloutReplay [51], and LUFFY [64]; and (iii) Agentic RL Methods, including Single-step RL (GRPO [45], DAPO [66], and GPPO [48]) and Multi-step RL (GiGPO [15], Tree-GRPO [25], ARPO [11], and AEPO [9]). Descriptions of all baselines are put in Sec. C.2. For Agentic RL Methods, we evaluate using three representative LLM backbones, including Qwen2.5-3B-instruct [43], Llama3-8B-instruct [21], and Qwen2.5-7B-instruct [43]. For our RAPO, we use AEPO-Qwen3-14B [9] as the default off-policy agent. The results on other off-policy models are also provided in Sec. 5.5.
5.1.3 Training and Evaluation. For Computational Reasoning and Knowledge-Intensive Reasoning, we choose the widely-used RL training dataset from Tool-Star [10]. In these two settings, we employ Python and Search tools, with search results provided by a local search server [27] built on a Wikipedia dump [28]. For WebAgentic Reasoning, we follow the configuration of Tree-GRPO [25] and employ real search APIs during both training and evaluation.
5.1.4 Evaluation Metrics. We follow ARPO [11] and adopt F1 scores as the evaluation metric in both Knowledge-Intensive and WebAgentic Reasoning. For Computational Reasoning, we also follow ARPO [11] to employ LLM-as-Judge to evaluate answers. We use Pass@1 as the metric. Implementation details are put in Sec. C.3.

[^1]Table 1: Results (\%) for Computational Reasoning and Knowledge-Intensive Reasoning tasks. Unless specified, the baselines use Qwen2.5-7B-Instruct as the backbone. The best results are highlighted in bold, and the second-best results are underlined.
| Methods |  | Computational Reasoning |  |  |  |  | Knowledge-Intensive Reasoning |  |  |  |  | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | AIME24 | AIME25 | MATH500 | GSM8K | MATH | WebWalker | HQA | 2Wiki. | MuSiQ. | Bamb. |  |
| Tool-Integrated Reasoning Methods |  |  |  |  |  |  |  |  |  |  |  |  |
|  | Search-o1 | 6.7 | 10.0 | 61.8 | 80.2 | 73.6 | 10.4 | 22.1 | 21.8 | 5.4 | 32.0 | 32.4 |
|  | Search-R1 | 16.7 | 6.7 | 63.8 | 82.4 | 81.2 | 12.0 | 26.9 | 25.9 | 16.2 | 40.4 | 37.2 |
|  | Tool-Star | 30.0 | 26.7 | 77.2 | 89.4 | 85.6 | 18.5 | 38.1 | 40.8 | 14.9 | 41.5 | 46.3 |
| Off-policy Learning Methods |  |  |  |  |  |  |  |  |  |  |  |  |
|  | SFT | 12.2 | 17.3 | 53.8 | 77.9 | 77.3 | 10.9 | 46.0 | 36.9 | 17.2 | 39.7 | 38.9 |
|  | RolloutReplay | 10.8 | 18.0 | 55.0 | 76.9 | 75.0 | 11.1 | 44.8 | 30.7 | 19.8 | 38.9 | 38.1 |
|  | LUFFY | 29.4 | 23.1 | 75.2 | 83.2 | 80.5 | 14.7 | 46.7 | 37.0 | 18.0 | 40.6 | 44.8 |
| RL Methods |  |  |  |  |  |  |  |  |  |  |  |  |
| Single-step RL |  |  |  |  |  |  |  |  |  |  |  |  |
| Qwen2.5-3B-Instruct | GRPO | 20.0 | 13.3 | 72.0 | 86.0 | 81.0 | 7.3 | 39.0 | 36.3 | 15.2 | 36.8 | $40.7\left(\Delta_{\text {base }}\right)$ |
|  | DAPO | 20.0 | 16.7 | 71.2 | 85.0 | 81.2 | 6.8 | 37.2 | 36.1 | 16.2 | 35.9 | $40.6(\downarrow 0.1)$ |
|  | GPPO | 17.2 | 19.5 | 69.7 | 86.3 | 80.5 | 13.9 | 40.0 | 36.9 | 16.5 | 36.9 | 41.7 ( ↑ 1.0) |
| Multi-step Agentic RL |  |  |  |  |  |  |  |  |  |  |  |  |
|  | GiGPO | 21.8 | 19.8 | 70.1 | $\underline{86.4}$ | 80.7 | 12.4 | 25.0 | 40.0 | 18.2 | 40.5 | 41.5 ( ↑ 0.8) |
|  | Tree-GRPO | 19.8 | 19.6 | 70.7 | 85.7 | 80.6 | 13.8 | $\underline{42.4}$ | $\underline{43.7}$ | 17.8 | $\underline{43.2}$ | $\underline{43.7(\uparrow 3.0)}$ |
|  | ARPO | $\underline{23.3}$ | $\underline{20.0}$ | $\underline{71.4}$ | 85.0 | $\underline{82.5}$ | 12.9 | 37.9 | 41.1 | 17.1 | 38.9 | 43.0 ( ↑ 2.3) |
|  | AEPO | 21.8 | 20.0 | 70.8 | 84.6 | 80.8 | $\underline{15.8}$ | 36.1 | 43.2 | $\underline{18.7}$ | 40.0 | 43.2 ( ↑ 2.5) |
|  | Ours | 24.5 | 24.8 | 72.0 | 87.2 | 82.8 | 18.0 | 45.8 | 48.9 | 20.5 | 45.9 | 47.0 ( ↑ 6.3) |
| Single-step RL |  |  |  |  |  |  |  |  |  |  |  |  |
| ![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-06.jpg?height=284&width=38&top_left_y=1356&top_left_x=197) | GRPO | 13.3 | 13.3 | 62.4 | 87.4 | 79.2 | 9.9 | 40.5 | 35.9 | 20.9 | 42.9 | $40.6\left(\Delta_{\text {base }}\right)$ |
|  | DAPO | 16.7 | 13.3 | 61.2 | 87.4 | 76.4 | 18.5 | 36.9 | 37.9 | 28.8 | 41.2 | 41.8 ( ↑ 1.2) |
|  | GPPO | 16.7 | 6.7 | 61.8 | 86.6 | 79.4 | 18.0 | 44.4 | 37.6 | 19.2 | 48.6 | 41.9 ( ↑ 1.3) |
|  | $\bar{M} \bar{u} \bar{t} \bar{i}-\bar{s} t \bar{e} \bar{A} \bar{g} \overline{e n} \overline{t i c} \bar{R} \bar{L}$ |  |  |  |  |  |  |  |  |  |  |  |
|  | GiGPO | 20.0 | 13.3 | 62.4 | 87.4 | 77.2 | 22.1 | 42.8 | 38.9 | 20.1 | 47.9 | 43.2 ( ↑ 2.6) |
|  | Tree-GRPO | 22.1 | 14.2 | 64.2 | 86.1 | 78.1 | $\underline{25.7}$ | 47.9 | $\underline{41.1}$ | 26.1 | 47.8 | 45.3 ( ↑ 4.7) |
|  | ARPO | 23.3 | $\underline{16.7}$ | 64.6 | 88.0 | 80.2 | 23.8 | 48.9 | 40.2 | 25.9 | 48.1 | $\underline{46.0}$ ( ↑ 5.4) |
|  | AEPO | $\underline{26.7}$ | 16.7 | 65.8 | 87.6 | 80.6 | 25.2 | 43.2 | 39.2 | 20.1 | 46.3 | 45.1 ( ↑ 4.5) |
|  | Ours | 27.1 | 17.4 | 66.6 | $\underline{87.7}$ | 80.8 | 28.0 | $\underline{48.7}$ | 41.4 | $\underline{28.1}$ | 49.8 | 47.6 ( ↑ 7.0) |
| Qwen2.5-7B-Instruct | Single-step RL |  |  |  |  |  |  |  |  |  |  |  |
|  | GRPO | 23.3 | 26.7 | 78.0 | 92.8 | 87.8 | 13.0 | 42.5 | 40.7 | 19.1 | 43.2 | 46.7 ( $\left.\Delta_{\text {base }}\right)$ |
|  | DAPO | 20.0 | 23.3 | $\underline{80.4}$ | 91.0 | 88.8 | 16.1 | 38.9 | 25.8 | 27.0 | 40.1 | $45.1(\downarrow 1.6)$ |
|  | GPPO | 26.7 | 23.3 | 76.2 | 91.6 | 87.6 | 22.9 | 42.9 | 40.1 | 21.8 | 46.2 | 47.9 ( ↑ 1.2) |
|  | Mūltii-step Agentic $\overline{\text { RL }}$ |  |  |  |  |  |  |  |  |  |  |  |
|  | GiGPO | 30.0 | 20.0 | 78.4 | 91.6 | 87.6 | 21.7 | 39.7 | 38.4 | 21.1 | 45.1 | 47.4 (↑ 0.7) |
|  | Tree-GRPO | 31.0 | 30.0 | 79.2 | 90.1 | 88.5 | 21.0 | 44.6 | 42.3 | 20.2 | 44.0 | 49.1 ( ↑ 2.4) |
|  | ARPO | 30.0 | 30.0 | 78.8 | 92.2 | 88.8 | 20.1 | 42.1 | $\underline{42.8}$ | 21.8 | $\underline{46.8}$ | 49.3 ( ↑ 2.6) |
|  | AEPO | $\underline{33.3}$ | 30.0 | 80.4 | 92.2 | $\underline{90.0}$ | $\underline{23.3}$ | $\underline{45.8}$ | 37.7 | 18.8 | 44.2 | $\underline{49.6}(\uparrow 2.9)$ |
|  | Ours | 34.2 | $\underline{29.4}$ | 81.1 | $\underline{92.7}$ | 91.5 | 24.9 | 46.8 | 42.9 | $\underline{22.1}$ | 47.6 | 51.3 ( ↑ 4.6) |


### 5.2 Main Results

Computational and Knowledge-Intensive Reasoning. The results for Computational and Knowledge-Intensive Reasoning tasks are presented in Tab. 1. Clearly, RAPO consistently improves the performance of all three LLM backbones across all datasets and tasks. Meanwhile, it achieves new state-of-the-art performance over the strongest baseline. These results validate the effectiveness of

RAPO, which introduces retrieval to facilitate effective policy exploration in Agentic RL. Although existing off-policy learning methods are suboptimal, RAPO still performs well. This highlights the importance of our hybrid-policy rollout strategy, which actively promotes fine-grained, step-level exploration dynamics within agentic rollout. Moreover, Agentic RL tends to exhibit better stability than singlestep RL, emphasizing the necessity of multi-step considerations in tool-integrated reasoning.

Table 2：Results（\％）for Web－Agentic Reasoning tasks．The best results are highlighted in bold，and the second－best are underlined．
| Methods |  | SimpleQA | General AI Assistant |  |  |  | WebWalkerQA |  |  |  | Browse． | Avg． |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | Lv． 1 | Lv． 2 | Lv． 3 | Overall | Easy | Med． | Hard | Overall |  |  |
| ![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-07.jpg?height=92&width=36&top_left_y=455&top_left_x=201) | Qwen2．5－32B |  | 7.7 | 8.8 | 7.7 | 3.0 | 7.6 | 6.2 | 9.4 | 5.8 | 7.4 | 2.2 | 6.6 |
|  | DeepSeek－R1－32B | 12.6 | 19.2 | 7.8 | 4.1 | 11.7 | $\underline{9.4}$ | 13.3 | 9.4 | 11.0 | 2.4 | 10.1 |
| 莫茧露 | GRPO | 61.5 | 17.7 | 14.9 | 4.5 | 14.7 | 8.9 | 11.4 | 11.6 | 10.9 | 2.3 | 15.1 （ $\left.\Delta_{\text {base }}\right)$ |
|  | GiGPO | 61.8 | 18.3 | 17.3 | 3.2 | 13.3 | 8.0 | 10.8 | 9.9 | 11.8 | 2.7 | $14.2(\downarrow 0.9)$ |
|  | Tree－GRPO | 62.4 | 19.3 | 17.5 | 5.7 | 16.8 | 9.3 | 11.8 | $\underline{11.9}$ | 11.2 | 2.7 | $15.3(\uparrow 0.2)$ |
|  | ARPO | $\underline{63.8}$ | 19.7 | 17.7 | 5.2 | 15.8 | 9.8 | 10.6 | 10.8 | $\underline{12.1}$ | 2.6 | $15.3(\uparrow 0.2)$ |
|  | AEPO | 62.1 | 20.8 | $\underline{18.9}$ | 5.4 | $\underline{16.9}$ | 9.0 | 11.7 | 10.4 | 11.6 | $\underline{2.9}$ | $\underline{15.9}(\uparrow 0.8)$ |
|  | Ours | 64.7 | $\underline{20.6}$ | 19.6 | 6.0 | 17.8 | 9.8 | 14.7 | 12.9 | 13.8 | 4.4 | $17.0(\uparrow 1.9)$ |


![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-07.jpg?height=665&width=933&top_left_y=807&top_left_x=131)
Figure 3：Efficiency study．RAPO exhibits clear training ef－ ficiency in rollout time，policy update time，the number of rollout tokens，and the number of tool calls．

Web－Agentic Reasoning．We further evaluate RAPO using real－ world web search APIs in Tab．2．We employ Qwen2．5－7B－instruct as the LLM backbone．From these results，we can see that our pro－ posed RAPO still achieves the best performance across all baselines， highlighting its practicality and generalization for Web－Agentic QA．Additionally，the overall performance gains observed on Web－ Agentic Reasoning tasks are relatively modest compared to other tasks．We infer that this is due to the inherent difficulty of the evaluation data and the unavoidable failures of API calls．

## 5．3 Efficiency Study

In this section，we conduct an efficiency study to empirically vali－ date the additional efficiency benefits from RAPO training．Specifi－ cally，we record several efficiency metrics at each RL training step， including rollout time，policy update time，the number of produced rollout tokens，and the number of tool calls．These experiments are performed on 4 NVIDIA RTX A100（ 80 GB ）GPUs．

The results are reported in Fig．3．RAPO exhibits clear advantages in training efficiency over GRPO．（i）Rollout time is substantially reduced，as retrieval alleviates the need for exhaustive on－policy

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-07.jpg?height=701&width=910&top_left_y=818&top_left_x=1065)
Figure 4：Comparison analysis between pure on－policy roll－ outs and hybrid－policy rollouts．

generation at the rollout stage．（ii）Policy updates are also acceler－ ated via retrieved token masking，which explicitly reduces gradient－ bearing tokens during optimization．（iii）RAPO produces fewer roll－ out tokens，whose distributions are more consistent with off－policy trajectories．This observation proves that the agent has successfully internalized and learned from off－policy reasoning patterns．（iv） RAPO makes fewer tool calls，suggesting that the retrieved traces can supplement external knowledge for the agent and thus reduce its reliance on tool calls．

## 5．4 Benefits of Retrieval

To understand how retrieval influences policy behaviors during rollout，we separately analyze pure on－policy rollouts and hybrid－ policy rollouts，where we visualize their diversity，outcome rewards， and the number of rollout tokens，respectively．To quantify diversity， we randomly sample approximately 7k rollouts from each group and extract their semantic embeddings using BGEM3［2］．We then apply PCA and perform DB－SCAN［12］clustering for visualization．

We present the results in Fig．4．First，hybrid－policy rollouts form more clusters within semantic space，reflecting significantly higher

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-08.jpg?height=345&width=847&top_left_y=285&top_left_x=180)
Figure 5: Robustness study for different off-policy models.

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-08.jpg?height=336&width=828&top_left_y=689&top_left_x=193)
Figure 6: Robustness study for noisy retrieval.

rollout diversity. This can be attributed to the retrieval from external behaviors, which pushes exploration beyond the confined distribution of pure on-policy rollouts. Second, hybrid-policy rollouts achieve higher outcome rewards. This suggests that retrieval can improve reasoning quality, which provides clearer group advantages and facilitates better credit assignment during training. Third, hybrid-policy rollouts exhibit shorter trajectory lengths. This implies that retrieval can prevent redundant reasoning by injecting informative signals, thus allowing the agent to arrive at solutions with fewer rollout tokens.

### 5.5 Robustness Study for Retrieval

5.5.1 Robustness for Different Off-policy Models. To evaluate the impact of buffer quality, we conduct a robustness study using various off-policy models for buffer construction, including AEPO-Qwen3-14B, AEPO-Qwen2.5-7B, AEPO-Qwen2.5-3B [9], Qwen314B, Qwen2.5-7B, and Qwen2.5-3B [43]. As shown in Fig. 5, RAPO consistently outperforms GRPO, demonstrating its strong robustness under various off-policy models. This can be due to our retrievalaware optimization mechanism, which successfully estimates the retrieval from different buffer quality and thus maintains stability. It should also be noted that RAPO benefits from stronger off-policy models, producing better performance under a higher-quality buffer.
5.5.2 Robustness for Noisy Retrieval. We further conduct the robustness study using noisy retrieval, where we introduce retrieval noise by replacing the retrieved trace with a randomly sampled trace at varying perturbation rates $p \in\{0.0,0.2,0.4,0.6,0.8,1.0\}$. As depicted in Fig. 6, RAPO still exhibits strong robustness and consistently outperforms GRPO across all perturbation rates. Remarkably, RAPO surpasses GRPO even when the retrieval is entirely random ( $p=1.0$ ). This indicates that the model can self-regulate by estimating the contribution of retrieved information. The results under $p=1.0$ also suggest that RAPO remains effective even when relevant queries are absent during buffer construction.

Table 3: Ablation study for retrieval reward.
|  | $H^{\text {Low }}$ |  |  |  |  | $H^{\text {High }}$ |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | w/o RR ( $\Delta_{\text {base }}$ ) |  | ↑ $\Delta H$ |  | $\downarrow \Delta H$ | ↑ $\Delta H$ | $\downarrow \Delta H$ (Ours) |
| Web. | 13.8 |  | 16.4 ( ↑ 2.6) |  | $11.4(\downarrow 2.4)$ | $12.4(\downarrow 1.4)$ | 18.0 |
| HQA | 42.5 | 44.2 | ( ↑ 1.7) |  | 40.5 ( ↓ 2.0) | 42.0 ( ↓ 0.5) | 45.8 |
| 2Wiki. | 40.6 | 46.8 | 6.2) |  | $38.6(\downarrow 2.0)$ | $39.4(\downarrow 1.2)$ | 48.9 |
| MuSiQ. | 18.2 | 19.1 | 0.9) |  | $16.4(\downarrow 1.8)$ | $17.8(\downarrow 0.4)$ | 20.5 |
| Bamb. | 40.5 | 41.5 | 1.0) |  | $39.8(\downarrow 0.7)$ | 38.1 ( ↓ 2.4) | 45.9 |


![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-08.jpg?height=362&width=832&top_left_y=743&top_left_x=1104)
Figure 7: Ablation study for model components.

### 5.6 Ablation Study

5.6.1 Ablation for Model Components. We begin our ablation study by evaluating the contributions of the key components within RAPO, including the Retrieval-aware Policy Optimization (RPO) in Sec. 4.2, the Retrieval Reward (RR) in Sec. 4.2.1, the Retrieval Importance Shaping (RIS) in Sec. 4.2.2, the Retrieval Quality (RQ) term in Eq. 7, and the Retrieval Timing (RT) term in Eq. 7. This results in five variants: w/o RPO, w/o RR, w/o RIS, w/o RQ, and $\boldsymbol{w} / \boldsymbol{o} \boldsymbol{R} \boldsymbol{T}$. We also include a Trajectory Off-policy (TO) variant (w/ $\boldsymbol{T O}$ ), where retrievals are conducted at the trajectory level. The results are shown in Fig. 7. We can see that incorporating all components results in the best performance, while the removal of any single component leads to a performance drop. This highlights the effectiveness of each component in RAPO. Notably, the $w / T O$ variant performs worst, underscoring the importance of step-level exploration dynamics during agentic rollout.
5.6.2 Ablation for Retrieval Reward. We further conduct an ablation study to evaluate the designs of retrieval reward, which is formulated to encourage entropy reduction ( $\downarrow \Delta H$ ) at high-entropy states ( $H^{\text {High }}$ ). Specifically, we separately invert this principle via: (i) encouraging entropy increase $(\uparrow \Delta H)$ by $H_{\Delta \hat{s}_{t}} \leftarrow-H_{\Delta \hat{s}_{t}}$ in Eq. 5, and (ii) encouraging retrieval at low-entropy states ( $H^{\text {Low }}$ ) by $H_{\hat{s}_{t-1}} \leftarrow 1 / H_{\hat{s}_{t-1}}$ in Eq. 7. In Tab. 3, we find that encouraging entropy reduction at high-entropy states performs best. This may be owing to the effective estimation of retrieval-aware exploration, providing informative signals during policy optimization. Unfortunately, the variants ( $H^{\text {Low }}, \downarrow \Delta H$ ) and ( $H^{\text {High }}, \uparrow \Delta H$ ) perform poorly (but they still outperform GRPO). We hypothesize that these two settings tend to reinforce the agent's raw reasoning behaviors, preventing it from escaping its native exploration boundary.

## 6 Conclusion and Future Work

In this paper, we investigate Agentic RL and present RAPO, which introduces retrieval to explicitly expand the agent's step-level exploration capability during training. By introducing a Hybrid-policy Agentic Rollout strategy, we can enhance rollout diversity using the retrieved off-policy step-level reasoning traces. Our Retrievalaware Policy Optimization mechanism then calibrates the policy estimation with retrieval reward and importance shaping, facilitating effective and stable RL training. As for future work, we will consider more powerful strategies to construct higher-quality Step-Trace Buffers, such as multi-policy frameworks.

## References

[1] Robert B Ash. 2012. Information theory. Courier Corporation.
[2] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216 4, 5 (2024).
[3] Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z Pan, Wen Zhang, Huajun Chen, et al. 2025. Learning to reason with search for llms via reinforcement learning. arXiv preprint arXiv:2503.19470 (2025).
[4] Yifei Chen, Guanting Dong, and Zhicheng Dou. 2025. Toward Effective ToolIntegrated Reasoning via Self-Evolved Preference Learning. arXiv preprint arXiv:2509.23285 (2025).
[5] Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. 2025. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758 (2025).
[6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 (2021).
[7] Wojciech M Czarnecki, Razvan Pascanu, Simon Osindero, Siddhant Jayakumar, Grzegorz Swirszcz, and Max Jaderberg. 2019. Distilling policy distillation. In The 22nd international conference on artificial intelligence and statistics. PMLR, 1331-1340.
[8] Jia Deng, Jie Chen, Zhipeng Chen, Daixuan Cheng, Fei Bai, Beichen Zhang, Yinqian Min, Yanzipeng Gao, Wayne Xin Zhao, and Ji-Rong Wen. 2025. From trial-and-error to improvement: A systematic analysis of llm exploration mechanisms in rlvr. arXiv preprint arXiv:2508.07534 (2025).
[9] Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, et al. 2026. Agentic entropy-balanced policy optimization. The International World Wide Web Conference (2026).
[10] Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, and Ji-Rong Wen. 2025. Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning. arXiv preprint arXiv:2505.16410 (2025).
[11] Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, et al. 2026. Agentic reinforced policy optimization. The International Conference on Learning Representations (2026).
[12] Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu, et al. 1996. A densitybased algorithm for discovering clusters in large spatial databases with noise. In kdd, Vol. 96. 226-231.
[13] Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, et al. 2025. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems. arXiv preprint arXiv:2508.07407 (2025).
[14] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. 2025. Retool: Reinforcement learning for strategic tool use in llms. arXiv preprint arXiv:2504.11536 (2025).
[15] Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for llm agent training. Proceedings of Neural Information Processing Systems (2025).
[16] Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. 2025. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307 (2025).
[17] Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. 2025. Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL. arXiv:2508.07976 [cs.CL] https: //arxiv.org/abs/2508.07976
[18] Anirudh Goyal, Yoshua Bengio, Matthew Botvinick, and Sergey Levine. 2020. The variational bandwidth bottleneck: Stochastic evaluation on an information budget. arXiv preprint arXiv:2004.11935 (2020).
[19] Anirudh Goyal, Abram Friesen, Andrea Banino, Theophane Weber, Nan Rosemary Ke, Adria Puigdomenech Badia, Arthur Guez, Mehdi Mirza, Peter C Humphreys, Ksenia Konyushova, et al. 2022. Retrieval-augmented reinforcement learning. In International Conference on Machine Learning. PMLR, 7740-7765.
[20] Anirudh Goyal, Riashat Islam, Daniel Strouse, Zafarali Ahmed, Matthew Botvinick, Hugo Larochelle, Yoshua Bengio, and Sergey Levine. 2019. Infobot: Transfer and exploration via the information bottleneck. arXiv preprint arXiv:1901.10902 (2019).
[21] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).
[22] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. 2025. Skywork open reasoner

1 technical report. arXiv preprint arXiv:2505.22312 (2025).
[23] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874 (2021).
[24] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060 (2020).
[25] Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, and Liaoni Wu. 2026. Tree search for llm agent reinforcement learning. The International Conference on Learning Representations (2026).
[26] Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, et al. 2025. Verltool: Towards holistic agentic reinforcement learning with tool use. arXiv preprint arXiv:2509.01055 (2025).
[27] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516 (2025).
[28] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering.. In EMNLP (1). 6769-6781.
[29] Jack Lanchantin, Angelica Chen, Janice Lan, Xian Li, Swarnadeep Saha, Tianlu Wang, Jing Xu, Ping Yu, Weizhe Yuan, Jason E Weston, et al. 2025. Bridging Offline and Online Reinforcement Learning for LLMs. arXiv preprint arXiv:2506.21495 (2025).
[30] Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, et al. 2025. WebSailor: Navigating Super-human Reasoning for Web Agent. arXiv preprint arXiv:2507.02592 (2025).
[31] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic search-enhanced large reasoning models. arXiv preprint arXiv:2501.05366 (2025).
[32] Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yongkang Wu, Ji-Rong Wen, Yutao Zhu, and Zhicheng Dou. 2025. Webthinker: Empowering large reasoning models with deep research capability. arXiv preprint arXiv:2504.21776 (2025).
[33] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let's verify step by step. In The Twelfth International Conference on Learning Representations.
[34] Siqi Liu, Guy Lever, Zhe Wang, Josh Merel, SM Ali Eslami, Daniel Hennes, Wojciech M Czarnecki, Yuval Tassa, Shayegan Omidshafiei, Abbas Abdolmaleki, et al. 2022. From motor control to team play in simulated humanoid football. Science Robotics 7, 69 (2022), eabo0235.
[35] Fanbin Lu, Zhisheng Zhong, Shu Liu, Chi-Wing Fu, and Jiaya Jia. 2025. ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay. arXiv preprint arXiv:2505.16282 (2025).
[36] Yao Lu, Dengdong Fan, Jianzheng Nie, Fan Xu, Jie Chen, Bin Zhou, and Yonghong Tian. 2026. PCL-Reasoner-V1. 5: Advancing Math Reasoning with Offline Reinforcement Learning. arXiv preprint arXiv:2601.14716 (2026).
[37] Jinchang Luo, Mingquan Cheng, Fan Wan, Ni Li, Xiaoling Xia, Shuangshuang Tian, Tingcheng Bian, Haiwei Wang, Haohuan Fu, and Yan Tao. 2025. GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning. arXiv preprint arXiv:2510.20548 (2025).
[38] Xinji Mai, Haotian Xu, Zhong-Zhi Li, Weinong Wang, Jian Hu, Yingying Zhang, Wenqiang Zhang, et al. 2025. Agent rl scaling law: Agent rl with spontaneous code execution for mathematical problem solving. arXiv preprint arXiv:2505.07773 (2025).
[39] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations.
[40] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. 2023. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023. 5687-5711.
[41] Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek HakkaniTür, Gokhan Tur, and Heng Ji. 2025. Toolrl: Reward is all tool learning needs. arXiv preprint arXiv:2504.13958 (2025).
[42] Yulei Qin, Xiaoyu Tan, Zhengbao He, Gang Li, Haojia Lin, Zongyi Li, Zihan Xu, Yuchen Shi, Siqi Cai, Renting Rui, et al. 2025. Learn the ropes, then trust the wins: self-imitation with progressive exploration for agentic reinforcement learning. arXiv preprint arXiv:2509.22601 (2025).
[43] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
[44] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. 2015. Trust Region Policy Optimization. In Proceedings of the 32nd International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 37), Francis Bach and David Blei (Eds.). PMLR, Lille, France, 1889-1897.
[45] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024).
[46] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. HybridFlow: A Flexible and Efficient RLHF Framework. arXiv preprint arXiv: 2409.19256 (2024).
[47] Joykirat Singh, Raghav Magazine, Yash Pandya, and Akshay Nambi. 2025. Agentic reasoning and tool integration for llms via reinforcement learning. arXiv preprint arXiv:2505.01441 (2025).
[48] Zhenpeng Su, Leiyu Pan, Xue Bai, Dening Liu, Guanting Dong, Jiaming Huang, Wenping Hu, Fuzheng Zhang, Kun Gai, and Guorui Zhou. 2025. Klear-reasoner: Advancing reasoning capability via gradient-preserving clipping policy optimization. arXiv preprint arXiv:2508.07629 (2025).
[49] Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, and Jingren Zhou. 2025. Zerosearch: Incentivize the search capability of $l l \mathrm{~ms}$ without searching. arXiv preprint arXiv:2505.04588 (2025).
[50] Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, and Yiming Yang. 2025. Training proactive and personalized llm agents. arXiv preprint arXiv:2511.02208 (2025).
[51] Yifan Sun, Jingyan Shen, Yibin Wang, Tianyu Chen, Zhendong Wang, Mingyuan Zhou, and Huan Zhang. 2025. Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay. Proceedings of Neural Information Processing Systems (2025).
[52] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. 1999. Policy Gradient Methods for Reinforcement Learning with Function Approximation. In Advances in Neural Information Processing Systems, S. Solla, T. Leen, and K. Müller (Eds.), Vol. 12. MIT Press.
[53] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599 (2025).
[54] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics 10 (2022), 539-554.
[55] Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. 2025. Beyond the $80 / 20$ rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939 (2025).
[56] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of PreTrained Transformers. arXiv:2002.10957 [cs.CL] https://arxiv.org/abs/2002.10957
[57] Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. 2025. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571(2025).
[58] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. 2025. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516(2025).
[59] Muning Wen, Ziyu Wan, Jun Wang, Weinan Zhang, and Ying Wen. 2024. Reinforcing $l l m$ agents via policy optimization with action decomposition. Advances in Neural Information Processing Systems 37 (2024), 103774-103805.
[60] Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025. WebDancer: Towards Autonomous Information Seeking Agency. arXiv:2505.22648 [cs.CL] https://arxiv.org/abs/2505.22648
[61] Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, et al. 2025. Webwalker: Benchmarking llms in web traversal. arXiv preprint arXiv:2501.07572 (2025).
[62] Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang, Xinmiao Yu, Dingchu Zhang, Yong Jiang, et al. 2025. Resum: Unlocking long-horizon search intelligence via context summarization. arXiv preprint arXiv:2509.13313 (2025).
[63] Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, and Bo An. 2025. Simpletir: End-to-end reinforcement learning for multi-turn tool-integrated reasoning. arXiv preprint arXiv:2509.02479 (2025).
[64] Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. 2025. Learning to reason under off-policy guidance. Proceedings of Neural Information Processing Systems (2025).
[65] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing. 2369-2380.
[66] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. 2025. Dapo: An opensource llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476 (2025).
[67] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. 2025. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837 (2025).
[68] Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, and Jingren Zhou. 2025. On-policy rl meets off-policy experts: Harmonizing supervised fine-tuning and reinforcement learning via dynamic weighting. arXiv preprint arXiv:2508.11408 (2025).
[69] Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. 2025. Echo chamber: Rl post-training amplifies behaviors learned in pretraining. arXiv preprint arXiv:2504.07912 (2025).
[70] Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, et al. 2025. First return, entropy-eliciting explore. arXiv preprint arXiv:2507.07017 (2025).
[71] Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen. 2025. Large language models for information retrieval: A survey. ACM Transactions on Information Systems 44, 1 (2025), 1-54.

```
Algorithm 1: Training LLM agents with RAPO.
    Input: Dataset $\mathcal{D}$; LLM agent $\pi_{\theta}$; External tools $\mathcal{T}$;
            Step-Trace Buffer $\mathcal{B}$; The number of pure on-policy
            rollouts $N_{\text {on }}$ and hybrid-policy rollouts $N_{\text {hybrid }}$.
    for step = 1 to $S_{\text {total }}$ do
        Initialize old model $\pi_{\theta_{\text {old }}} \leftarrow \pi_{\theta}$;
        Sample a batch of training data $q \subseteq \mathcal{D}$;
        Initialize the rollout pool $\mathcal{P} \leftarrow \emptyset$;
        for $i=1,2, \cdots, N_{\text {on }}+N_{\text {hybrid }}$ do
            Generate first-step trace $s_{0} \sim \pi_{\theta}(\cdot \mid q, t=0)$;
            Update rollout pool $\mathcal{P}$ with $s_{0}$;
        end
        Mark $N_{\text {on }}$ rollouts among $\mathcal{P}$ as $\mathcal{P}_{\text {on }}$ and others as $\mathcal{P}_{\text {hybrid }}$;
        // Hybrid-policy Agentic Rollout
        while $\exists \mathcal{S}_{<t} \in \mathcal{P}$ not terminated do
            if $\mathcal{S}_{<t} \in \mathcal{P}_{\text {on }}$ then
                Generate next trace $s_{t} \sim \pi_{\theta}\left(\cdot \mid q, \mathcal{S}_{<t}\right) ;$
                Update rollout $\operatorname{pool} \mathcal{P}$ with $s_{t}$;
            end
            if $\mathcal{S}_{<t} \in \mathcal{P}_{\text {hybrid }}$ then
                $p \sim \operatorname{Random}() ;$
                if $p<0.5$ then
                    Retrieve off-policy trace $\hat{s}_{t} \sim \operatorname{Retrieve}\left(\mathcal{S}_{<t}\right) ;$
                    Update rollout pool $\mathcal{P}$ with $\hat{s}_{t}$;
                end
                Generate next trace $s_{t / t+1} \sim \pi_{\theta}\left(\cdot \mid q, \mathcal{S}_{<t / t+1}\right)$;
                Update rollout pool $\mathcal{P}$ with $s_{t / t+1}$;
            end
        end
        // Retrieval-aware Policy Optimization
        for Updating iter $=1, \cdots$ do
            Compute retrieval reward $Z_{\text {ret }}$ by Eq. 7;
            Compute retrieval advantage $A_{\text {ret }}$ by Eq. 9;
            Compute final advantage $A_{\text {RAPO }}$ by Eq. 10;
            Conduct retrieval importance shaping by Eq. 8;
            Update the policy model $\pi_{\theta}$ by Eq. 11;
        end
    end
```


## A Notations and Algorithms

We provide the important notations used in this paper and their corresponding descriptions as shown in Tab. 4. Additionally, for clarity, we present the pseudo-codes of RAPO in Algorithm 1.

## B Theoretical Analysis

In this section, we conduct a theoretical analysis to validate the effectiveness of RAPO, showing that the Retrieval-aware Policy Optimization mechanism introduces an implicit information bottleneck, which provably enhances generalization in RL training.

Policy with Retrieval Information. We consider an agent whose policy is conditioned on both the environment state and retrieval information. Let us first denote the off-policy-conditioned reasoning process at step $t$ within RAPO as $\pi_{\theta}\left(\cdot \mid \hat{\mathcal{S}}_{<t}, t\right)$, where $\hat{\mathcal{S}}_{<t}=$

Table 4: Important notations and descriptions.
| Notations | Descriptions |
| :--- | :--- |
| $s_{t}$ | On-policy reasoning trace at step $t$ |
| $\hat{s}_{t}$ | Off-policy reasoning trace at step $t$ |
| $\mathcal{S}_{<t}$ | On-policy reasoning history before step $t$ |
| $\hat{\mathcal{S}}_{<t}$ | Off-policy reasoning history before step $t$ |
|  | Step-level entropy for trace $s_{t}$ |
| $H_{s_{t}} Z_{\text {ret }}^{i}$ | Retrieval reward for the $i$-th rollout within group |
| $\hat{r}_{i, j}(\theta)$ | Retrieval importance shaping for the $j$-th token in $i$-th rollout |
| $A_{\text {RAPO }}^{i}$ | Combined advantage for the $i$-th rollout |
| $J_{\text {RAPO }}(\theta)$ | Training objective of RAPO |


$\left\{s_{0}, \cdots, s_{t-2}, \hat{s}_{t-1}\right\}$ is the reasoning history and $\hat{s}_{t-1} \sim \operatorname{Retrieve}\left(\mathcal{S}_{<t-1}\right)$ is the information from the retrieval process conditioned on the reasoning history at the ( $t-1$ )-th step. The policy agent is parameterized by $\theta$.

The goal of our Retrieval-aware Policy Optimization mechanism is to train agents that, in addition to maximizing outcome rewards, minimize the impacts and contributions of useless information from the retrieval process. We quantify this using the conditional mutual information $I\left(\cdot ; \hat{\mathcal{S}}_{<t} \mid \mathcal{S}_{<t-1}\right)$.

Information-Regularized Objective. This approach of minimizing the impacts and contributions of useless information from the retrieval process can be interpreted as encouraging agents to learn useful reasoning behaviors and to absorb those reasoning behaviors closely, except where diverting from doing so (as a result of using information from the retrieval process) leads to higher reward [7]. To see this, the conditional mutual information can be written as:

$$
\begin{equation*}
I\left(\cdot ; \hat{\mathcal{S}}_{<t} \mid \mathcal{S}_{<t-1}\right)=\mathbb{E}_{\pi_{\theta}}\left[\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}\left(\cdot \mid \mathcal{S}_{<t-1}, \hat{\mathcal{S}}_{<t}\right) \| \bar{\pi}_{\theta}\left(\cdot \mid \mathcal{S}_{<t-1}\right)\right)\right], \tag{12}
\end{equation*}
$$

where the expectation is taken over trajectories induced by $\pi_{\theta}$, and

$$
\bar{\pi}_{\theta}\left(\cdot \mid \mathcal{S}_{<t-1}\right)=\sum_{g} p(g) \pi_{\theta}\left(\cdot \mid \mathcal{S}_{<t-1}, g\right)
$$

is the default policy obtained by marginalizing out retrieval information. We therefore optimize the following objective:

$$
\begin{align*}
J(\theta) & =\mathbb{E}_{\pi_{\theta}}\left[r-\beta I\left(\cdot ; \hat{\mathcal{S}}_{<t} \mid \mathcal{S}_{<t-1}\right)\right] \\
& =\mathbb{E}_{\pi_{\theta}}\left[r-\beta \mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}\left(\cdot \mid \mathcal{S}_{<t-1}, \hat{\mathcal{S}}_{<t}\right) \| \bar{\pi}_{\theta}\left(\cdot \mid \mathcal{S}_{<t-1}\right)\right)\right] \tag{13}
\end{align*}
$$

where $\beta>0$ regulates the trade-off between reward maximization and information dependence, and $\mathbb{D}_{\mathrm{KL}}(\cdot)$ represents the KullbackLeibler (KL) divergence.

Latent Variable Formulation. Following prior works [19, 20], we introduce a latent variable $X$ to parameterize the policy. Specifically, we define an encoder $p_{\text {enc }}\left(X \mid \mathcal{S}_{<t-1}, \hat{\mathcal{S}}_{<t}\right)$, a decoder $p_{\text {dec }}(\cdot \mid \mathcal{S}_{<t-1}, X$ ), and a learned prior $q\left(X \mid \mathcal{S}_{<t-1}\right)$ such that:

$$
\begin{equation*}
\pi_{\theta}\left(\cdot \mid \mathcal{S}_{<t-1}, \hat{\mathcal{S}}_{<t}\right)=\sum_{X} p_{\mathrm{enc}}\left(X \mid \mathcal{S}_{<t-1}, \hat{\mathcal{S}}_{<t}\right) p_{\mathrm{dec}}\left(\cdot \mid \mathcal{S}_{<t-1}, X\right) . \tag{14}
\end{equation*}
$$

The latent variable $X$ captures the subset of retrieved information that the agent deems relevant for decision-making at state $\mathcal{S}_{<t-1}$.

Under this formulation, the objective in Eq. 13 admits the following lower bound:

$$
\begin{equation*}
J(\theta) \geq \tilde{J}(\theta)=\mathbb{E}_{\pi_{\theta}}\left[r-\beta \mathbb{D}_{\mathrm{KL}}\left(p_{\mathrm{enc}}\left(X \mid \mathcal{S}_{<t-1}, \hat{\mathcal{S}}_{<t}\right) \| q\left(X \mid \mathcal{S}_{<t-1}\right)\right)\right] \tag{15}
\end{equation*}
$$

Information Bottleneck. According to the data processing inequality (DPI) theory [1], we have:

$$
I\left(X ; \hat{\mathcal{S}}_{<t} \mid \mathcal{S}_{<t-1}\right) \geq I\left(\cdot ; \hat{\mathcal{S}}_{<t} \mid \mathcal{S}_{<t-1}\right)
$$

which allows us to upper bound the original information regularizer by bounding $I\left(X ; \hat{\mathcal{S}}_{<t} \mid \mathcal{S}_{<t-1}\right)$. Conditioning on a fixed state $\mathcal{S}_{<t-1}=s$ and averaging over $p(s)$, we obtain:

$$
\begin{equation*}
I\left(X ; \hat{\mathcal{S}}_{<t} \mid \mathcal{S}_{<t-1}\right) \leq \sum_{s} p(s) \sum_{g} p(g \mid s) \mathbb{D}_{\mathrm{KL}}(p(X \mid s, g) \| r(X)) \tag{16}
\end{equation*}
$$

where $r(X)$ is a reference distribution.
This formulation corresponds to an information bottleneck objective, which has been shown to improve generalization and robustness in reinforcement learning [18, 34].

## C Experimental Details

## C. 1 Details of Datasets

C.1.1 Datasets of Computational Reasoning. We first introduce the details of the datasets in Computational Reasoning tasks as follows:

- AIME24 ${ }^{2}$ is designed to assess models' capabilities in advanced computational reasoning. It contains 30 carefully selected problems drawn from the American Invitational Mathematics Examination. The problems span diverse mathematical domains, including algebraic manipulation and geometric reasoning.
- AIME25 ${ }^{3}$ is composed of 30 challenging problems sourced directly from the official AIME I and AIME II examinations released in February 2025. The dataset exhibits broad coverage across fundamental mathematical areas such as algebra, geometry, number theory, and combinatorics.
- MATH500 [33] is a curated subset of the larger MATH dataset, selected by OpenAI to emphasize more difficult instances. It includes 500 problems that span topics such as algebra, geometry, calculus, and number theory, with difficulty levels approaching or surpassing undergraduate coursework.
- MATH [23] is a widely used benchmark for studying computational reasoning in machine learning models. The dataset covers a broad range of mathematical disciplines, including abstract algebra, calculus, and discrete mathematics.
- GSM8K [6] is a dataset of grade-school-level mathematics problems released by OpenAI. Each problem typically requires between two and eight reasoning steps and involves fundamental arithmetic and logical operations.
C.1.2 Datasets of Knowledge-Intensive Reasoning. We then describe the datasets in Knowledge-Intensive Reasoning tasks.
- HotPotQA [65] is a benchmark dataset for multi-hop question answering, where all supporting documents are drawn from

[^2]Wikipedia. Benefiting from Wikipedia's broad coverage and wellorganized content, the dataset provides a rich knowledge source for complex reasoning tasks.

- 2WikiMultihopQA [24] is specifically constructed to assess multi-hop question answering. The dataset focuses on evaluating whether models can perform step-by-step reasoning while aggregating evidence from multiple documents, placing strong emphasis on cross-document information integration.
- MuSiQue [54] is a dataset for multi-hop question answering, which is designed to move beyond shallow fact lookup and instead measure a model's capacity for deeper semantic understanding and logical reasoning across multiple pieces of evidence.
- Bamboogle [40] is a multi-hop question-answering dataset. Its test split is relatively small, containing only 125 question-answer pairs, but it is often used to closely examine model behavior in controlled multi-hop reasoning scenarios.
C.1.3 Datasets of Web-Agentic Reasoning. Finally, we introduce the datasets in Web-Agentic Reasoning tasks.
- SimpleQA [40] is a factual question-answering benchmark consisting of 500 short, knowledge-seeking QA pairs. The dataset is adversarially constructed with respect to GPT-4, meaning that the questions are intentionally selected to expose cases where strong language models are prone to making confident but incorrect factual claims.
- GAIA [39] is a comprehensive benchmark designed to evaluate the capabilities of general-purpose AI assistants on real-world tasks. The questions in GAIA require a diverse set of skills, including multi-step reasoning, tool usage, web interaction, and multi-modal understanding.
- WebWalkerQA [61] is a dataset aimed at assessing an agent's ability to perform structured web navigation in order to answer questions. It contains 680 question-answer tasks that require sequential web traversal and information gathering.
- BrowseComp [58] is a challenging benchmark for evaluating agents' proficiency in web browsing and complex information discovery. The dataset consists of 1,266 questions that involve obscure, fragmented, or difficult-to-locate information on the web.


## C.1.4 Datasets for RL Training.

- Computational and Knowledge-Intensive Reasoning. We adopt the training dataset from Tool-Star [10] due to its higher quality and widespread adoption. It includes about 10k questions from both computational reasoning and multi-hop QA reasoning.
- Web-Agentic Reasoning. We follow the training setup of TreeGRPO [10], which is lightweight and cost-efficient for real-world web API usage. The dataset includes 2k samples from ASearcher35 K [17] and 200 hard web QA pairs from WebDancer [60].


## C. 2 Details of Baselines

C.2.1 Tool-Integrated Reasoning Methods. We include three recent tool-integrated reasoning methods for comparison.

- Search-o1 [31] is a RAG-style approach centered on a reasoning model. When search is triggered, the model formulates search queries, retrieves evidence, and then consolidates the returned
content into an intermediate context that is woven into the ongoing reasoning trace. The model subsequently continues generation conditioned on this augmented reasoning context until producing the final answer.
- Search-R1 [27] is an RL-based method that learns a searchcalling policy from scratch. Conceptually, its inference workflow resembles Search-o1: the model emits queries as needed during multi-step reasoning, fetches relevant documents from a search engine, and integrates the retrieved evidence back into its reasoning trajectory before continuing.
- Tool-Star [10] is a tool-augmented reasoning framework that supports six different tool types and emphasizes systematic design in both data construction and model training. It introduces an automated data synthesis pipeline that generates tool-invocation trajectories by combining tool-aware prompting with hint-guided sampling, which are further refined through quality normalization and difficulty-based filtering.
C.2.2 Off-policy Learning Methods. We also include the existing off-policy learning methods as follows:
- RolloutReplay [51] aims to improve training efficiency of RL for single-step reasoning. It incorporates a rollout replay mechanism inspired by experience replay in classical reinforcement learning, allowing recent rollouts to be reused to reduce computational overhead while preserving training stability.
- LUFFY [64] is designed to overcome the on-policy limitation for single-step reasoning. It jointly leverages off-policy trajectory and on-policy rollouts, which builds on a mixed-policy GRPO formulation with theoretical convergence guarantees and further applies policy shaping through regularized importance sampling.
C.2.3 RL Methods. Now, we introduce the details of the RL methods used in this paper.
- GRPO [45] is a single-step RL approach for LLM fine-tuning that operates through group-level policy optimization. It assesses multiple sampled outputs jointly and derives relative reward signals from within-group comparisons, leading to more stable updates and improved sample efficiency during training.
- DAPO [66] improves optimization stability by separating the clipping mechanism from the policy update process. It employs an adaptive data selection strategy that dynamically chooses training samples to sustain informative gradients into learning.
- GPPO [48] modifies the standard PPO objective by disentangling the clipping operation across forward and backward computations. Specifically, clipping is applied during the forward pass to constrain policy updates, while the backward pass retains the unclipped policy ratio to preserve richer gradient information.
- GiGPO [15] is a multi-step RL that computes advantages at multiple granularities. At the episode level, entire trajectories are grouped to estimate coarse-grained advantage signals; At the step level, actions associated with shared anchor states are regrouped across trajectories to derive fine-grained credit assignments.
- Tree-GRPO [25] is a multi-step RL that leverages tree-structured search. It represents each agent interaction step as a node in a search tree, allowing multiple rollouts to share common prefixes.

Furthermore, the tree-based trajectories enable the extraction of step-level supervision signals.

- ARPO [11] is designed for multi-step LLM agents operating in interactive environments. It leverages an entropy-driven adaptive rollout strategy that increases exploration in high-uncertainty steps and incorporates a structured advantage attribution mechanism to distribute credit across branching paths.
- AEPO [9] is a multi-step RL designed to address entropy-related challenges during both rollout generation and policy optimization. It introduces an entropy-aware rollout strategy and an entropy-balanced policy optimization objective to mitigate excessive branching and ensure stable gradient scaling.


## C. 3 Details of Experimental Settings

C.3.1 Buffer Construction. During the Step-Trace Buffer construction, we use AEPO-Qwen3-14B [9] as the default off-policy agent. We set $N=16$ during the generation of off-policy trajectories and set $K=5$ in reward-aware filtering for each query within the training dataset. The scale of our Step-Trace Buffer used in each RL training is provided in the following subsections. The resulting reasoning corpus can be consistently reused, and we will make it publicly available if this paper can be accepted.
C.3.2 RL Training. We build upon the Search-R1 [27] pipeline implemented in the VeRL [46] framework.

For Computational Reasoning and Knowledge-Intensive Reasoning, we use a total training batch size of 32, a PPO mini-batch size of 16, and a global rollout size of 16 across all experiments. For RAPO, we set the number of hybrid-policy trajectories $N_{\text {hybrid }}$ to 8 , the importance shaping weight $m$ to 0.05 , and the retrieval advantage weight $a$ to 0.2 . The KL divergence coefficient and clipping parameter are set to 0 and 0.28 , respectively. The constructed buffer in this setting contains 50,000 off-policy trajectories, comprising 169,489 step-level reasoning traces with a total of 15,648,438 tokens.

For Web-Agentic Reasoning, we adopt the configuration of TreeGRPO [25] and set the training batch size of 128, a PPO mini-batch size of 64 , and a global rollout size of 4 across this group of experiments. During each search call, the top-10 snippets derived from the web SERP API are provided to the agent as external evidence. For RAPO, we set $N_{\text {hybrid }}$ to 2 while keeping other settings consistent with those in Computational and Knowledge-Intensive Reasoning tasks. The resulting buffer contains 11,000 off-policy trajectories, comprising 38,473 step-level reasoning traces and $4,518,194$ tokens in total. Unless specified, experiments are performed on 8 NVIDIA A100 GPUs. Off-policy retrieval is not used during evaluation.
C.3.3 Details of Search Tool. For Computational and KnowledgeIntensive Reasoning, we employ a widely used Wikipedia dump [28] as the corpus and adopt MiniLM [56] as the semantic encoder for document embeddings. During training, we follow the SearchR1 [27] protocol and apply the RAG pipeline to derive the top- 3 most semantically relevant documents for each search step. We also use this pipeline to perform the retrieval within the hybrid-policy rollout of RAPO, which returns the most relevant off-policy trace.

For Web-Agentic Reasoning, we use the Bing Search API (USEN region) as the search engine. At each search step, the top-10 passages are provided to the agent as search feedback.

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-15.jpg?height=372&width=1591&top_left_y=277&top_left_x=244)
Figure 8: Parameter study on the HQA dataset.

## D Prompt Template

In this section, we describe the prompt template used during RAPO training. Following ARPO [11], any content enclosed within the <search> </search> or <python> </python> tags is parsed as a tool invocation, corresponding to the agent's action $\alpha$. The tool outputs are subsequently wrapped within <result></result> tags and returned to the agent as observations $o$, thereby forming a complete one-step reasoning trace $s_{t}=\left(\tau_{t}, \alpha_{t}, o_{t}\right)$.

Different from existing Agentic RL methods, RAPO explicitly exposes retrieved off-policy step traces to the agent. Specifically, the retrieved off-policy traces are inserted into the prompt and delimited by <retrieve> </retrieve> tags, allowing the agent to condition its subsequent reasoning on externally retrieved behaviors in a structured and transparent manner. The overall prompt template used in RAPO is summarized as follows:

## Prompt Template for RAPO Training

## system

You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result></result> tags respectively. Additional context may be enclosed in <retrieve> </retrieve> tags. Use it if helpful. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is [ \boxed\{answer here\}] </answer>. In the last part of the answer, the final exact answer is enclosed within \boxed\{\} with latex format.
user
Euxoamorpha eschata is a moth found in a city situated on the southern shores of what?
assistant

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-15.jpg?height=361&width=918&top_left_y=720&top_left_x=1064)
Figure 9: Training dynamics between RAPO and GRPO.

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-15.jpg?height=361&width=920&top_left_y=1191&top_left_x=1064)
Figure 10: Visualization for the $\boldsymbol{w} /$ Low-Entropy Retrieval variant. Encouraging retrieval at low-entropy states tends to force the agent into excessive overthinking during rollout.

## E Training Dynamics

In this section, we analyze the training dynamics of RAPO, as illustrated in Fig. 9. We report the averaged outcome rewards and the entropy during training. Overall, RAPO exhibits consistently higher outcome rewards than GRPO. Meanwhile, its entropy decreases more steadily, suggesting that the agent learns to produce more confident and coherent reasoning behaviors. These findings highlight the role of retrieval-augmented exploration in improving training effectiveness.

## F Parameter Study

Now, we study how the hyper-parameters impact the performance of RAPO. Specifically, we vary: (i) the importance sampling weight $m$ in Eq. 8; (ii) the retrieval advantage weight $a$ in Eq. 10; and (iii) hybrid-policy rollout size $N_{\text {hybrid }}$ in Sec. 4.1. Note that we fix the global rollout size to 16 across all experiments.

We plot the results in Fig. 8. Over-emphasizing retrieval-related signals (large $m, a$, or $N_{\text {hybrid }}$ ) introduces excessive off-policy noise

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-16.jpg?height=712&width=936&top_left_y=281&top_left_x=137)
Figure 11: Efficiency study over ARPO [11]. RAPO exhibits substantial efficiency gains in rollout time, policy update time, and the number of rollout tokens. RAPO incurs slightly more tool calls than ARPO. This is expected as ARPO reduces up to $50 \%$ of tool calls via branching, whereas RAPO mitigates tool usage via retrieval, avoiding about $25 \%$ of tool calls.

that impairs the agent's on-policy self-reasoning, thus leading to degraded performance. Conversely, under-utilizing retrieval signals (small $m, a$, or $N_{\text {hybrid }}$ ) prevents the agent from effectively absorbing and exploiting external reasoning behaviors, limiting the benefits of retrieval-aware exploration. Overall, setting $m=0.05, a=0.2$, and $N_{\text {hybrid }}=8$ seems to be a generally sweet choice.

## G What Happens When Encouraging Retrieval at Low-Entropy States?

To further validate the design rationale of our retrieval reward, we conduct a focused analysis on the variant $w /$ Low-Entropy Retrieval. In this setting, retrieval is explicitly encouraged at low-entropy states by modifying the timing term in Eq. 7 as $H_{\hat{s}_{t-1}} \leftarrow 1 / H_{\hat{s}_{t-1}}$, consistent with the ablation configuration in Sec. 5.6. We track the number of rollout tokens and entropy respectively and summarize the results in Fig. 10. Compared to RAPO, w/Low-Entropy Retrieval exhibits a clear increasing trend in both rollout token length and entropy as training progresses. This indicates that encouraging retrieval at low-uncertainty states forces the agent to excessively overthink during rollout, which ultimately degrades reasoning quality and leads to inferior performance (See Tab. 3).

## H Limitation

One potential limitation of RAPO is that, while RAPO significantly expands the on-policy exploration space, the exploration is still implicitly bounded by the joint capacity of the on-policy agent and the off-policy models. Future research could investigate more generalized buffer construction paradigms, such as incorporating multiple heterogeneous off-policy agents or dynamically evolving off-policy models, to further broaden exploration in Agentic RL.

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-16.jpg?height=397&width=853&top_left_y=285&top_left_x=1100)
Figure 12: Robustness study for different off-policy models.

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-16.jpg?height=401&width=849&top_left_y=775&top_left_x=1103)
Figure 13: Robustness study for noisy retrieval.

![](https://cdn.mathpix.com/cropped/9c5e4bf3-45bd-4f89-9a29-07f2f52b6e6f-16.jpg?height=359&width=794&top_left_y=1276&top_left_x=1125)
Figure 14: Ablation study for model components.

Another limitation is that RAPO needs to maintain a Step-Trace Buffer that stores off-policy reasoning traces for retrieval. This introduces additional memory and storage overhead during training, particularly in large-scale post-training. While our experiments demonstrate that retrieval-aware exploration yields substantial performance gains, future work could explore more efficient and scalable strategies for buffer construction.

## I Case Study

In this section, we present a qualitative Case Study for the models trained with RAPO. As shown in Tabs. 5-8, the model successfully executes multi-turn agentic tasks through iterative tool invocations and autonomous information aggregation. Interestingly, the model exhibits reflective behaviors during reasoning, suggesting that it has effectively internalized beneficial external behaviors and developed more structured and effective reasoning patterns. These observations further demonstrate the effectiveness of RAPO.

Table 5: Case Study (i).

## Example from HQA dataset

Question: What British singer-songwriter hosted the 16th Annual Young Hollywood Awards?
Ground Truth: Kelly Lee Osbourne
Model Output by RAPO-Qwen2.5-7B
<think> To solve this question, I first need to determine who hosted the 16th Annual Young Hollywood Awards. After identifying the host, I must verify whether the person is a British singer-songwriter. </think>
<search> host of the 16th Annual Young Hollywood Awards </search>
<result> Doc 1 (Title: "Young Hollywood Awards") The Young Hollywood Awards is an annual awards ceremony honoring emerging talent in film, music, and television. The 16th Annual Young Hollywood Awards were held in Los Angeles in 2014 and featured appearances from a variety of young actors and musicians. The ceremony was hosted by Kelly ... Doc 2 (Title: "2014 Young Hollywood Awards") The 2014 ceremony, recognized as the 16th installment of the awards, took place at The Wiltern theater. Television personality and singer Kelly Osbourne served as host for the event. The show highlighted rising stars and ... Doc 3 (Title: "Young Hollywood Awards") Previous hosts of the ceremony have included a mixture of actors and entertainers. For the 2014 edition, Kelly Osbourne was selected due to her strong presence in both television and music, making her ... </result>
<think> The retrieved documents consistently identify Kelly Osbourne as the host. Next, I need to confirm whether she is a British singer-songwriter. </think>
<search> Kelly Osbourne nationality singer songwriter </search>
<result> Doc 1 (Title: "Kelly Osbourne") Kelly Lee Osbourne is an English singer, songwriter, television personality, actress, and fashion designer. Born in London, she is the daughter of Ozzy and Sharon Osbourne. She gained prominence through the reality television ... Doc 2 (Title: "Kelly Osbourne discography") Osbourne released her debut album *Shut Up* in 2002, which featured pop-rock influences. She later recorded several singles and collaborations, establishing ... Doc 3 (Title: "English musicians") England has produced numerous singer-songwriters across genres such as rock, pop, and folk. Kelly Osbourne is frequently categorized among English music artists who ... </result>
<think> The evidence confirms that Kelly Lee Osbourne is English (therefore British) and has an established career as a singer-songwriter. Hence, she satisfies all constraints in the question. </think>
<answer> Kelly Lee Osbourne hosted the 16th Annual Young Hollywood Awards. Kelly Lee Osbourne </answer>

Table 6: Case Study (ii).

## Example from the MuSiQ. dataset

Question: When was the region immediately north of the region prevailing with the disgrace of the Near East and the site of the most growth in desalination for agricultural use established?
Ground Truth: 1932

## Model Output by RAPO-Qwen2.5-7B

<think> The question appears to involve multiple geographic references. First, I need to identify the region described as having the "disgrace of the Near East" and being a major site of desalination growth for agriculture. After locating that region, I should determine which region lies immediately north of it, and finally find the establishment date of that northern region. </think>
<search> region disgrace of the Near East desalination agricultural growth </search>
<result> Doc 1 (Title: "Negev") The Negev is a desert region in southern Israel. Historically, parts of the desert were once described as emblematic of the "disgrace of the Near East" due to arid conditions and limited agricultural productivity. In recent decades, however, the Negev has experienced ... Doc 2 (Title: "Desalination in Israel") Israel has pioneered desalination technologies to support agriculture in arid regions. Significant growth has occurred in the southern desert areas, particularly the Negev, where desalinated water has ... Doc 3 (Title: "Arava Valley") The Arava Valley is another desert area known for agricultural innovation using advanced irrigation techniques, though it is less commonly referenced as the primary site of desalination growth compared ... </result><
<think> The documents consistently indicate that the Negev matches both clues: it was historically characterized negatively and later became a major center for desalination-supported agriculture. Next, I need to identify the region immediately north of the Negev. </think> <search> region north of the Negev </search>
<result> Doc 1 (Title: "Southern District (Israel)") The Southern District is one of Israel's administrative districts and encompasses most of the Negev desert. Directly north of this district lies the ... Doc 2 (Title: "Central District (Israel)") The Central District is an administrative region of Israel located immediately north of the Southern District. It includes major urban centers and ... Doc 3 (Title: "Districts of Israel") Israel is divided into several administrative districts, including Northern, Haifa, Central, Tel Aviv, Jerusalem, and Southern. The Central District borders the ... </result>
<think> Since the Southern District contains the Negev and the Central District lies immediately north of it, the target region should be the Central District. Now I need to determine when the Central District was established. </think>
<search> Central District Israel established </search>
<result> Doc 1 (Title: "Central District (Israel)") The Central District is one of Israel's administrative districts. It was formally established in 1932 during the period of British administration, when regional governance structures ... Doc 2 (Title: "Administrative divisions of Mandatory Palestine") During the early 20th century, several administrative regions were created under British rule. Among them was the Central District, established in 1932 as part of a ... Doc 3 (Title: "Tel Aviv District") The Tel Aviv District was formed later as administrative needs changed, and therefore does not correspond to the earlier establishment ... </result>
<think> The evidence confirms that the Central District-located immediately north of the Southern District containing the Negev-was established in 1932. Therefore, the answer is 1932. </think>
<answer> The final answer is 1932 . </answer>

Table 7: Case Study (iii).

```
Example from GSM8K dataset
Question: Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many
total pets do the three have?
Ground Truth: 28
Model Output by RAPO-Qwen2.5-7B
<think> This problem forms a dependency chain: Cindy → Marcia → Jan. I will compute the intermediate variables first, then aggregate
them. </think>
<python> cindy = 4
marcia $=$ cindy +2
jan = 3 * marcia
print(f"(\{marcia\}, \{jan\})") </python>
<result> $(6,18)$ </result>
<think> Now that Cindy = 4, Marcia = 6, and Jan = 18, I compute the total. </think>
<python> total $=4+6+18$
print(total) </python>
<result> 28 </result>
<answer> 28 </answer>
```

Table 8: Case Study (iv).

## Example from MATH dataset

Question: One computer backs up data every five minutes, while another backs up every two minutes. Find the number of times in twenty-four hours that both computers back up simultaneously. Assume they do not back up at the start of the 24-hour period.

Ground Truth: 144

Model Output by RAPO-Qwen2.5-7B
<think> The computers will back up together at intervals equal to the least common multiple (LCM) of 5 and 2 minutes. I will first compute the LCM to determine the synchronization interval, then calculate how many such intervals occur within 24 hours. </think> <python> import math
interval = math. $\operatorname{lcm}(5,2)$
print(interval) </python>
<result> 10 </result>
<think> The computers synchronize every 10 minutes. Since 24 hours contain $24 \times 60=1440$ minutes and the initial moment is excluded,
I compute how many 10 -minute intervals fit into 1440 minutes. </think>
<python> total_minutes $=24 * 60$
total_minutes // 10
print(total_minutes) </python>
<result> 144 </result>
<answer> The final answer is 144 . </answer>


[^0]:    Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
    KDD'26, Jeju, Korea
    © 2026 ACM.
    ACM ISBN 978-1-4503-XXXX-X/18/06
    https://doi.org/XXXXXXX.XXXXXXX

[^1]:    ${ }^{1}$ https://huggingface.co/datasets/AI-MO/aimo-validation-aime

[^2]:    ${ }^{2}$ https://huggingface.co/datasets/HuggingFaceH4/aime_2024
    ${ }^{3}$ 7https://huggingface.co/datasets/math-ai/aime25

