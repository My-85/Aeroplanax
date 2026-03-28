# LearningFlow: Automated Policy Learning Workflow for Urban Driving with Large Language Models 

Zengqi Peng, Yubin Wang, Xu Han, Lei Zheng, and Jun Ma


#### Abstract

Recent advancements in reinforcement learning (RL) demonstrate the significant potential in autonomous driving. Despite this promise, challenges such as the manual design of reward functions and low sample efficiency in complex environments continue to impede the development of safe and effective driving policies. To tackle these issues, we introduce LearningFlow, an innovative automated policy learning workflow tailored to urban driving. This framework leverages the collaboration of multiple large language model (LLM) agents throughout the RL training process. LearningFlow includes a curriculum sequence generation process and a reward generation process, which work in tandem to guide the RL policy by generating tailored training curricula and reward functions. Particularly, each process is supported by an analysis agent that evaluates training progress and provides critical insights to the generation agent. Through the collaborative efforts of these LLM agents, LearningFlow automates policy learning across a series of complex driving tasks, and it significantly reduces the reliance on manual reward function design while enhancing sample efficiency. Comprehensive experiments are conducted in the high-fidelity CARLA simulator, along with comparisons with other existing methods, to demonstrate the efficacy of our proposed approach. The results demonstrate that LearningFlow excels in generating rewards and curricula. It also achieves superior performance and robust generalization across various driving tasks, as well as commendable adaptation to different RL algorithms.


Index Terms-Autonomous driving, large language model, automated policy learning, curriculum reinforcement learning.

## I. Introduction

With the advancement of artificial intelligence (AI) technologies, significant breakthroughs have been made in generative models, with large language models (LLMs) being one of the most prominent applications [1], [2]. LLMs excel in understanding and generating text, while their integration with additional specialized modules enables multimodal capabilities, such as processing and generating images and videos. This inherent versatility demonstrates the immense potential for

[^0]application across various fields [3]. Meanwhile, autonomous driving technology has made remarkable progress, becoming a focal point of research in AI and transportation [3], [4], [5]. In general, urban driving scenarios are characterized by diverse road structures and task types, such as multi-lane overtaking, on-ramp merging, and intersection crossing. These driving scenarios demand frequent interactions with surrounding vehicles (SVs) exhibiting varying driving styles. The diversity of driving environments and the uncertainty of SV behaviors collectively present significant challenges to achieving a safe and efficient closed-loop urban driving system [6], [7]. Consequently, autonomous driving systems are required to enhance the ability to prevent potential collision risks and optimize task efficiency across different scenarios. From this perspective, urban autonomous driving requires robust interaction-aware decision-making and planning capabilities to safely interact with SVs while efficiently accomplishing various driving tasks.

Reinforcement learning (RL) has demonstrated significant potential for autonomous driving solutions. It enables policies to optimize decision-making through interactions with the environment and the feedback received from these interactions [4], [8], [9], [10]. Despite the significant progress of RL in the autonomous driving community, it still encounters two major challenges. The first challenge lies in the design of reward functions, which RL relies on to guide agents in exploring the environment and improving policies. However, in real-world tasks like autonomous driving, reward signals are often sparse, and this significantly hinders efficient policy learning [11]. Reward shaping is a common approach to provide incremental learning signals to mitigate this issue [12]. Nevertheless, designing an appropriate reward function for complex autonomous driving tasks remains highly challenging. Traditional manual design methods are not only constrained by the subjective experience of the designer but also timeconsuming and tedious [13], [14], [15]. Due to the distinct characteristics of different tasks, the reward functions that are capable of effectively guiding RL policies to learn satisfactory behaviors could vary significantly. Furthermore, the reward function is typically fixed at the beginning of the policy training and cannot be adapted in real-time during the training process. Therefore, the manual design of reward functions often fails to encapsulate the nuanced behaviors needed and provide effective guidance for RL agents in dynamic urban driving scenarios, leading to poor policy convergence. The second challenge is the sample efficiency during online exploration. The complexities of urban driving tasks involve diverse

![](https://cdn.mathpix.com/cropped/363a3fc3-3499-4600-8838-03007fe07260-02.jpg?height=461&width=1441&top_left_y=178&top_left_x=343)
Fig. 1. The proposed LLM-in-the-training-loop CRL training paradigm. The Multi-LLM-agent system generates reward functions and training curriculum sequences for the downstream RL policy through the collaboration of multiple LLM-based agents. The historical training data generated through interactions between the RL policy and the environment is stored in a memory module and then fed back to the Multi-LLM-agent system as reference information for subsequent generation steps.

road structures, varying task objectives, traffic densities, and the driving styles of SVs. Relying on RL policies to learn from samples collected through random exploration in the target task is highly inefficient, potentially requiring an enormous number of training episodes and resulting in unstable performance [16]. Curriculum learning (CL) is a structured approach inspired by human learning, which improves training efficiency by decomposing tasks in a progressive order from simple to complex [17], [18]. This approach ultimately enables the agent to acquire effective policies for complex tasks. While CL provides a promising framework for addressing training inefficiencies in RL for complex tasks, challenges remain in evaluating and dynamically switching between training curricula. Several automated CL algorithms have been proposed to assess training progress and quality, dynamically adjusting the training curriculum at appropriate stages [19], [10]. The training task is modeled as a multi-armed bandit problem, with a curriculum set of distinct difficulty levels to enable curriculum transitions. However, updating the multi-armed bandit algorithm requires incorporating new supervision signals, and parameter tuning is task-specific. This problem could lead to inefficiency when dealing with different scenarios.

To tackle the aforementioned challenges, this paper investigates the integration of curriculum RL (CRL) and LLMs to develop a policy training framework, LearningFlow, which aims to automate curriculum transition and reward design for urban driving tasks. As an emerging technology, LLMs demonstrate remarkable capabilities in problem reasoning and code generation, offering new routines to address the challenges of curriculum selection and reward design in CRL solutions of autonomous driving. By leveraging the vast knowledge base of LLMs, we aim to address the limitations of traditional RL approaches in autonomous driving, enhancing both the scalability and performance of RL-based driving policies. Therefore, this paper presents a novel automated policy training workflow, which consists of multiple LLM agents, for CRL in urban driving. The overview of the proposed training paradigm is shown in Fig. 1 Here, the RL policy is trained with the support of a multi-LLM-agent system, which iteratively generates and refines the training curricula and reward functions throughout the training process. To the best of our knowledge, this work is
the first iterative and online framework that leverages LLMs for automated reward function optimization and curriculum transition in urban driving. The main contributions are listed as follows:

- A novel automated policy learning framework is proposed for training interaction-aware driving policies in complex autonomous driving tasks, which significantly improves driving safety and generalization ability under different traffic densities.
- A collaborative role-playing module involving multiple LLM agents is developed to iteratively generate training curricula and reward functions. This module effectively enhances the exploration of RL agents, thereby significantly improving the training efficiency of driving policies.
- We demonstrate the effectiveness of the proposed approach in the high-fidelity simulator CARLA. The proposed method exhibits effective curriculum generation and reward generation capabilities. It achieves superior performance and commendable generalization ability across various driving tasks and RL algorithms.
The rest of the paper is structured as follows. Section II presents the related work. Section III introduces the problem formulation. Section IV illustrates the proposed framework for automated policy learning. Section V demonstrates the experimental results followed by pertinent analysis. Finally, Section VI summarizes the conclusion and discusses future works.


## II. Related Work

## A. Reward Design for Deep Reinforcement Learning

The reward function plays a critical role in guiding RL agents as they interact with the environment. A high-quality reward function can significantly enhance the performance of RL policies. Therefore, the design of reward functions, often referred to as reward engineering, plays a vital role in RL. Designing suitable reward functions for real-world tasks, such as autonomous driving and robotics, poses challenges due to the sparsity of rewards over long time horizons and the balance between exploration and exploitation. In existing studies, manual design is the most commonly used routine for
constructing reward functions, followed by minor adjustments through trial and error [10], [20], [21]. This approach not only makes the effectiveness of the reward function dependent on expert experience but also renders the entire training process time-consuming and labor-intensive. Furthermore, when dealing with complex multi-task problems, relying solely on expert experience often fails to yield a satisfactory reward function. Multi-task RL techniques are introduced by sharing policies or designing task-specific policies [22], [23], [24], which still do not consider the reward design problems.

To address the challenges arising from reward design, inverse RL (IRL) is utilized to extract reward functions from collected data by observing the agent behaviors of interaction with the environment [25], [26]. Specifically, the deep IRL methods are adopted to infer the driving behaviors by distilling the learned reward model from collected expert demonstrations to score and evaluate trajectories in autonomous driving tasks [27], [28]. However, due to the non-uniqueness of reward functions, the inferred reward function could not fully capture the true motivations of the agent. On the other hand, IRL relies on a large amount of expert data, and collecting high-quality data in complex and dynamic environments is challenging, if not impractical. This reward inference method typically involves complex optimization processes, resulting in heavy computational burdens, long training time, and limitations due to environmental modeling. Additionally, evolutionary algorithms have been introduced for reward shaping to evolve reward functions [12], [29]. With the advancement of foundation model technologies, the design of reward functions for RL tasks can be accomplished by providing relevant prompts to LLMs [30], [31], which offers a promising solution to solve the challenges of reward design in RL.

## B. Training Efficiency for Deep Reinforcement Learning

RL policies improve performance by interacting with the environment to collect experience. However, low sample efficiency remains a significant challenge in applying RL to complex tasks. Directly employing random policies to gather sample information in intricate environments is highly inefficient, which can significantly prolong the time required for policy convergence or even lead to failure in achieving convergence. To tackle this issue, an environment model is introduced to generate virtual samples to improve the sample efficiency, thereby accelerating the training process [32]. However, the effectiveness of the trained policy highly depends on the accuracy of the model.

CL offers a promising solution to mitigate the above challenges [17], [33]. A stage-decaying CL approach is utilized to guide the policy learning of the RL agents [34]. Nevertheless, the predetermined manual scheduling of curriculum transitions heavily relies on expert knowledge, which limits the robustness and effectiveness of the training outcomes. To tackle the aforementioned challenges, various automated CL methods have been proposed [35], [36]. However, these approaches are designed under the assumption that SVs do not respond to the behaviors of the ego vehicle (EV). Besides, the future trajectories of SVs are accessible to the EV.

These simplifications could compromise driving safety and reduce the generalization capability of RL policies. In realworld scenarios, human drivers dynamically adapt their driving maneuvers based on the behaviors of SVs, which is critical for ensuring safe and efficient interactions on the road. Essentially, as most current studies do not comprehensively consider the interactive behaviors of SVs, these simplifications could limit the deployment of autonomous driving techniques in realworld scenarios.

## C. Large Language Model Applications

With the rapid advancement of LLMs, their potential in various tasks has attracted significant attention [3]. Currently, LLMs are primarily utilized in two ways. The first one is the LLMs-in-the-task-loop solution, which utilizes LLMs for embodied inference. The second one is the LLMs-in-the-training-loop solution, which incorporates LLMs for policy learning.

1) LLMs for Embodied Inference: One of the most direct ways to apply LLMs technology to autonomous driving tasks is by embedding the LLMs as a decision-making or planning module within the autonomous driving system. Depending on the embedding method, there are primarily two approaches, serial LLMs solutions and parallel LLMs solutions [37, [38], [39]. For example, DiLu achieves decision-making based on common-sense knowledge by integrating an LLMs-based inference module into the autonomous driving system [37]. However, this sequential structure limits the response speed of the autonomous driving system due to the inference latency of LLMs, posing challenges to meeting real-time requirements. To address this issue, DriveVLM-Dual connects a traditional end-to-end pipeline with an LLMs-based inference system in a parallel manner, forming a fast-slow system that alleviates the real-time performance challenges of LLMs-based autonomous driving solutions [40].
2) LLMs for Policy Learning: In addition to directly deploying LLMs within autonomous driving systems, LLMs are also applied during the training phase [31], [41], [42], [43]. Recent research has explored the use of LLMs for designing reward functions in reinforcement learning and assisting in CL for task decomposition. A general RL training framework has been proposed to generate proxy reward functions by leveraging the in-context learning capabilities and prior knowledge of LLMs [30]. Auto MC-Reward enhances learning efficiency by automatically designing dense reward functions through the introduction of three automated reward modules [31]. Eureka is proposed as an LLM-driven human-level reward generation method for sequential decision-making tasks [14], [44]. CurricuLLM is proposed to decompose complex robotic skills into a sequence of subtasks, facilitating the learning of intricate robotic control tasks [45]. However, the entire training curriculum sequence is predetermined before training and the timing of curriculum transitions is neglected, which potentially hinders the sample efficiency. AutoReward leverages LLMs and CoT-based modules to achieve closed-loop reward generation for autonomous driving [46]. However, this approach requires access to the internal code of the simulation environment, which could result in the leakage of environmental
dynamics. It does not account for traffic density, requiring retraining for driving tasks with varying traffic conditions in the same driving environment. Additionally, AutoReward adjusts the reward function after the whole training cycle, which increases training time and could result in unnecessary computational costs.

## III. Problem Formulation

## A. Problem Statement

This study aims to tackle the challenge of developing safe, effective, and interaction-aware driving policies for various urban driving scenarios, such as multi-lane overtaking and onramp merging. The initial and target positions of both the EV and SVs are randomly generated, while ensuring adherence to traffic regulations. The number of SVs is also random, with interactive behaviors driven by different driving styles. In this context, the EV is required to complete various driving tasks within an environment that contains interactive SVs. Since different driving tasks have distinct characteristics and requirements, this presents a significant challenge in designing appropriate reward functions and suitable training curricula. The goal is to automatically generate reward functions and training curricula to train RL policies that infer decision sequences, guiding the EV to safely and efficiently complete driving tasks across different scenarios. Here, we assume that the EV can access the exact position and velocity information of SVs accurately, yet their goal tasks and driving intentions are unknown. These configurations inject a significant level of randomness into the driving scenarios, rendering the tasks challenging but close to real-world situations.

## B. Learning Environment

In this work, the target tasks are formulated as a Markov Decision Process (MDP). Here, we represent the MDP as a tuple $\mathcal{E}=\langle\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma\rangle$, with each element defined as follows:

State space $\mathcal{S}$ : In this work, $\mathcal{S}$ includes kinematic features of driving vehicles within the observation range of the EV. The state matrix at time step $k$ is defined as shown below:

$$
\mathbf{S}_{k}=\left[\begin{array}{llll}
\mathbf{s}_{k}^{0} & \mathbf{s}_{k}^{1} & \ldots & \mathbf{s}_{k}^{N_{\mathrm{sv}}^{\max }} \tag{1}
\end{array}\right]^{T},
$$

where $N_{\mathrm{sv}}^{\text {max }}$ indicates the maximum number of SVs observed by the EV; $\mathbf{s}_{k}^{0}$ and $\mathbf{s}_{k}^{i}\left(i=1,2, \ldots, N_{\mathrm{sv}}^{\max }\right)$ denote the state of the EV and the state of the $i$-th SV, respectively. In particular, $\mathbf{s}_{k}^{i}$ is defined as follows:

$$
\mathbf{s}_{k}^{i}=\left[\begin{array}{llll}
x_{k}^{i} & y_{k}^{i} & v_{k}^{i} & \psi_{k}^{i} \tag{2}
\end{array}\right]^{T},
$$

where $x_{k}^{i}, y_{k}^{i}, v_{k}^{i}, \psi_{k}^{i}$ are the X -axis and Y -axis coordinates, the speed, and the heading angle of the $i$-th vehicle, respectively.

Action space $\mathcal{A}$ : In this work, a multi-discrete action space consisting of three discrete sub-action spaces is utilized for the RL agent:

$$
\begin{equation*}
\mathcal{A}=\left\{A_{1}, A_{2}, A_{3}\right\}, \tag{3}
\end{equation*}
$$

where $A_{1}, A_{2}$, and $A_{3}$ denote the waypoint, reference velocity, and lane change sub-action spaces, respectively. Further details will be provided in Section IV-E.

State transition dynamics $\mathcal{P}\left(\mathbf{S}_{k+1} \mid \mathbf{S}_{k}, a_{k}\right): \mathcal{P}$ describes the transitions of the environmental state while satisfying the Markov property. It is implicitly determined by the external environment and remains inaccessible to the RL agent.

Reward function $\mathcal{R}$ : The reward function plays a crucial role in RL. It reinforces the correct actions of the agent by providing rewards and penalizes incorrect actions, guiding the exploration of the agent within the environment. A well-designed reward function can significantly enhance the efficiency and performance of the training process. However, designing rewards manually for complex tasks remains challenging. In this work, we leverage the extensive knowledge base of LLMs to design and iteratively refine efficient reward functions for RL agents.

Discount factor $\gamma: \gamma \in(0,1)$ is utilized to discount future accumulated rewards.

## C. Curriculum Sequence Generation Problem

Under the problem defined in Section III.A, we establish the following two-layer curriculum set. The first layer considers the traffic densities, while the second layer takes into account the motion modes of SVs. Specifically, the designed curriculum set consists of $N_{t d}$ subsets, each comprising $N_{m m}$ elements. This two-layer curriculum set can be expressed as:

$$
\begin{equation*}
\boldsymbol{\Omega}=\left\{\Omega_{i j} \mid i=0,1, \ldots, N_{\mathrm{td}}^{\max }, j=0,1, \ldots, N_{\operatorname{mm}}^{\max }\right\}, \tag{4}
\end{equation*}
$$

where $N_{\mathrm{td}}^{\max }$ and $N_{\mathrm{mm}}^{\max }$ denote the number of traffic density types and motion modes of SVs, respectively. In CRL, a sequence of training curricula is required to set up the environment for optimizing the RL policy, which can be expressed as follows:

$$
\begin{equation*}
\left\{\Omega_{l m}^{*}\right\}=\arg \max \sum \mathcal{R}, \tag{5}
\end{equation*}
$$

where $\left\{\Omega_{l m}^{*}\right\}$ is the optimal curriculum sequence, and $\mathcal{R}$ is the reward of the RL agent. In this study, the curriculum sequence is formulated through the use of LLMs, which are employed for their robust generative abilities to guide the learning process.

## IV. Methodology

## A. Overview of the LearningFlow

The detailed architecture of the proposed LearningFlow is illustrated in Fig. 2 First, general knowledge prompts related to autonomous driving tasks are generated separately to the curriculum analysis agent and reward analysis agent to analyze training tasks. Then the generated key points on curriculum generation and reward generation, along with general knowledge prompts for CL, reward generation, and code generation, are input into the curriculum generation agent and reward generation agent to facilitate training course selection and reward generation. Subsequently, the relevant training curricula and reward function codes are extracted from responses of the LLM to initialize the RL agent and interactive environment.

![](https://cdn.mathpix.com/cropped/363a3fc3-3499-4600-8838-03007fe07260-05.jpg?height=1204&width=1621&top_left_y=184&top_left_x=251)
Fig. 2. Overview of the LearningFlow framework for automated driving policy learning with interactive SVs. In the reasoning module, the analysis agents process prompts containing context descriptors, historical training information, and task objectives to perform inference, providing task analysis as a reference for the generation agent. The generation agent, based on the analysis results and other relevant prompts, selects training curricula and generates reward functions to initialize the downstream training environment and RL agent. After initialization, the RL agent interacts with the environment and records training data. Upon completing a certain number of episodes, the training data, along with the decision contents from the LLM agents, are stored in the memory module. These records are then summarized by the reflection module and fed back to the agents in the reasoning module to support the next round of inference.

Finally, the downstream RL executor explores and learns within the environment and reward functions designed by the LLM agents. The responses from LLM agents, along with training history, are recorded in a memory module; and after a specified number of episodes, they are fed back to the LLM agents to update the training curricula and reward functions.

In this work, the proposed LearningFlow framework consists of a reasoning module, a reflection module, and a memory module for curriculum generation and reward generation. By providing appropriate prompts to the proposed framework, the system is able to perform automated policy learning for various autonomous driving tasks. Details will be discussed in the following sections.

## B. Memory Module for Closed-Loop Policy Training Workflow

LLMs inherently lack persistent memory capabilities, meaning that they cannot retain interaction information from previous sessions when processing new queries. Consequently, LLMs are unable to recall past exchanges with the user during subsequent reasoning tasks. Therefore, a memory module is introduced to store historical information from the training
process to enable closed-loop online tuning workflow throughout the training process. This includes inference results from various LLM agents, generated training curricula, designed reward functions, recorded total reward and the individual reward components during training, and training metrics. The extracted information is stored in textual or vectorized form via an extraction module and integrated into prompts for the next session through a reflection module. By incorporating the memory module, LLMs can retain and leverage cross-session historical training data, enabling closed-loop reasoning and automated training adjustments, thereby significantly enhancing the efficiency and adaptability of policy learning.

## C. Iterative Curriculum Sequence Generation

1) Context Descriptor: Given the generality of LLMs, it is necessary to provide relevant contextual information to help them understand task objectives. In this study, we use contextual descriptors to describe the current training task in natural language comprehensively. Considering the complexity of the task, the description includes characteristics of the driving scenario and objectives of policy learning. For the
curriculum generation LLM agent, a carefully designed twolayer curriculum set and curriculum descriptions are included as part of the context. The generated description is then integrated into the system prompts of different agents to facilitate task-specific reasoning of curriculum analysis and generation.
2) Curriculum Analysis Agent: Our target scenarios are complex and challenging, involving not only SVs with diverse driving styles but also diverse traffic densities. To enhance sample efficiency, the designed two-layer curriculum set incorporates not only these factors but also subsets of curricula with different motion modalities of SVs. The unquantifiable nature of driving difficulty across different scenarios poses challenges to traditional curriculum selection algorithms and heightens the requirements for the curriculum generation agent.

Directly generating training curricula could result in LLMs producing low-quality outputs from the LLMs, such as inappropriate course transitions or the omission of critical curricula. To address this issue, we introduce a curriculum analysis agent. In the prompt for the curriculum analysis agent, we provide the foundational knowledge for CL, along with the structure of the two-layer curriculum set (4) and the design objectives for each layer. Specifically, the curriculum set is designed to guide the policy in progressively exploring the environment. It starts with simpler tasks and gradually transitions to more complex ones, thereby enhancing sample efficiency. Before generating training curricula, this agent incrementally analyzes the two-layer curriculum set by considering the current training context. It leverages contextual information from context descriptors, curriculum set information, historical curriculum sequence, current training progress, and historical training data of the RL policy. Based on this analysis, the agent identifies critical aspects to consider in the setup of the next round of CRL training and provides a rationale for the curriculum generation agent to make informed inferences.
3) Curriculum Generation Agent: After the curriculum analysis agent completes its reasoning, the analyzed results are passed to the curriculum generation agent as a reference. Specifically, the prompt generator utilizes the definition and analysis of the curriculum set, principles of CL, and output format prompts as textual context for training curriculum generation. It then directs the LLMs to perform reasoning and return the inferred results, which include the selected training curriculum. Then the selected curriculum is decoded from the response of the LLMs. Through the designed prompts and the collaboration with the curriculum analysis agent, the curriculum generation agent can effectively select appropriate curriculum tasks during the initialization phase and throughout the training process.

Given the complexity of environmental rules and state transitions, as well as the potential suboptimality in the selections generated by LLMs, we incorporate an $\epsilon$-greedy strategy, commonly used to address the exploration-exploitation dilemma in RL [13], to configure the ultimate training environment. This strategy helps the RL agent better understand environmental features and adapt to environmental changes. The specific $\epsilon$ -
curriculum selection strategy is defined as follows:

$$
C_{k}=\left\{\begin{array}{l}
C_{L L M}, \text { with probability } 1-\epsilon,  \tag{6}\\
C_{\text {random }}, \text { with probability } \epsilon,
\end{array}\right.
$$

where $C_{t}$ is the employed curriculum setting at $k$-th episode, $C_{L L M}, C_{\text {random }}$ represent the curriculum selected by the agent and randomly, respectively; and $\epsilon$ is a parameter that decays over the training process.
4) Curriculum Evaluation and Reflector: In the initial phase of training, open-loop automatic curriculum generation is achieved through the collaboration of the analysis agent and the selection agent, which can provide RL agents in different training stages with suitable training environments. Meanwhile, interaction data generated during policy training is recorded in the memory module. To facilitate timely curriculum switching based on historical information, it is necessary to express in language whether the current curriculum is appropriate for subsequent training. For this purpose, a curriculum reflection module is introduced to generate feedback for the curriculum analysis and selection modules, enabling a closedloop automatic curriculum agent workflow. Specifically, the curriculum reflection agent summarizes the characteristics of the historical curriculum sequence, the trajectory of policy rewards, and task performance metrics. It then generates reflection prompts to guide the analysis and selection of the next curriculum phase.

The entire workflow of curriculum sequence generation can be expressed as follows:

$$
\left\{\begin{array}{l}
A_{0}^{C}=L\left(P_{a}^{C}\right)  \tag{7}\\
C_{0}=D_{C}\left(L\left(P_{s}^{C}, A_{0}^{C}\right)\right) \\
A_{n+1}^{C}=L\left(C_{\text {hist }}, A_{n}^{C}, P_{a}^{C}, P_{f}^{C}\right) \\
C_{n+1}=D_{C}\left(L\left(C_{\text {hist }}, P_{s}^{C}, P_{f}^{C}, A_{n+1}^{C}\right)\right) \\
\quad n=0,1, \ldots, N_{\max }
\end{array}\right.
$$

where $L$ refers to the LLM; $A_{n}^{C}, C_{n}$ represent the analysis and selected curriculum for the $n$-th training interval; $C_{\text {hist }}$ is the historical curriculum sequence; $P_{a}^{C}, P_{s}^{C}$, and $P_{f}^{C}$ denote the analysis prompt, selection prompt, and feedback prompt for the curriculum sequence generation, respectively; $D_{C}(\cdot)$ is the decode function to extract the curriculum.

## D. Iterative Reward Generation

1) Context Descriptor: Similar to curriculum agents, it is essential to provide appropriate contextual information to guide LLMs in understanding the objectives of reward generation. Here, we also employ contextual descriptors to express the requirements of reward generation tasks in natural language comprehensively. Considering the intricacy of the reward design process, the descriptors include both general principles of reward functions and task-specific goals, such as the characteristics of driving interactions and the expected outcomes of RL agent training. These descriptions are incorporated into the system prompts of different agents to enable task-specific reasoning for reward analysis and generation.
2) Reward Analysis Agent: In our study, the complexity of autonomous driving tasks arises not only from varying driving scenarios but also from differing traffic densities. This increases the demand for the design of reward functions. Generating the reward function directly could lead to ineffective outputs from the LLMs.

To address this issue, we introduce a reward function analysis agent that analyzes the task before the reward function is designed. For standard RL tasks, manually crafted reward functions typically incorporate accessible environment state variables, action variables, and constants. Therefore, it is reasonable to provide code segments with annotated notes containing accessible environmental variables as context for the reward analysis agent. This approach prevents exposing the internal mechanisms of the environment and eliminates the need for explicit state transition dynamics. The LLM agent leverages contextual information from context descriptors, the code segment of accessible variables, the next training environment determined by the curriculum agents, the current training progress, and the feedback data. This agent offers the key considerations during the reward generation process and provides a reference value range and analytical basis for the reward components, thus supporting the inference process of the reward generation agent.
3) Reward Generation Agent: After the reward analysis agent completes its reasoning, the analyzed results and reference value range are passed to the reward generation agent. Besides, accessible variables, reward function signature, principles of reward function construction, and output format specifications are also provided as textual context for designing reward functions. Specifically, the generation of the reward function is required to output both the total reward and the individual components of the designed sub-rewards. Given the above comprehensive information as context, a general reward generation prompt is crafted for input into the LLMs. Then the reward generation agent calls LLMs to perform reasoning to generate a response containing the designed reward function by utilizing its vast knowledge base and emergence ability. Subsequently, the executable code of the reward function is extracted from the response content and added to the environment program. Through carefully designed prompts and collaboration with the reward analysis agent, the reward generation agent can effectively construct appropriate reward functions throughout the training process.
4) Reward Evaluation and Reflector: Through the collaboration of multiple agents, an open-loop reward generation process is implemented to provide RL agents with an initial reward function. Then the RL agent can explore the environments with the guidance of the generated reward function and related data is collected for policy update and memory storage.

To facilitate the timely refinement of reward functions based on historical data, it is essential to evaluate and articulate whether the current reward function aligns with the training objectives. For this purpose, a reward reflection module is introduced to generate feedback prompts for the reward analysis and generation agents, forming a closed-loop reward generation process and online tuning workflow together. This reflection workflow generates reflection prompts to reshape
and mutate the reward function based on feedback information and the previous reward function for subsequent training phases.

To enable effective closed-loop iteration, a comprehensive evaluation of the designed reward functions is required as a basis for subsequent improvements. It includes information from the interactions of the RL agent, such as historical data about the reward functions and their components, success rates, collision rates, and timeout rates. According to these metrics, the reward agents can integrate information at different levels of granularity. This includes coarse-grained data on the entire reward function and success rates, as well as fine-grained details on individual reward components. These capabilities allow for more effective optimization of the reward function design and the proposal of more targeted reward components. A prompt regarding context-based reward improvements is then provided to analysis and generation agents to propose a new, enhanced generation of the reward function based on previously designed ones. This prompt includes descriptions of the closed-loop feedback and suggestions for improvements, such as adding or removing reward components or modifying the reward coefficients.

The entire workflow of reward generation and online tuning can be expressed as follows:

$$
\left\{\begin{array}{l}
A_{0}^{R}=L\left(P_{a}^{R}\right)  \tag{8}\\
R_{0}=D_{R}\left(L\left(P_{s}^{R}, A_{0}^{R}\right)\right) \\
A_{n+1}^{R}=L\left(R_{\text {hist }}, A_{n}^{R}, P_{a}^{R}, P_{f}^{R}\right) \\
R_{n+1}=D_{R}\left(L\left(R_{\text {hist }}, P_{s}^{R}, P_{f}^{R}, A_{n+1}^{R}\right)\right) \\
\quad n=0,1, \ldots, N_{\max }
\end{array}\right.
$$

where $A_{n}^{R}, R_{n}$ represent the analysis and design reward for the $n$-th training interval; $R_{\text {hist }}$ is the historical reward function; $P_{a}^{R}, P_{s}^{R}$, and $P_{f}^{R}$ denote the analysis prompt, reward generation prompt, and feedback prompt for the reward function, respectively; $D_{R}(\cdot)$ is the decode function to extract the reward function code.

## E. Downstream RL Executor

This work adopts an integrated decision-planning-control strategy using an RL policy as the downstream executor. Specifically, the RL policy generates decision variables based on observations, which are then used as the reference of the model predictive controller to calculate the control signal. The specific details are presented as follows. We use the notations of subscript $k$ to represent $k$-th time step within an episode. The observations of the RL agent are expressed as follows:

$$
\begin{align*}
\mathbf{O}_{k} & =\left[\begin{array}{cccc}
\mathbf{o}_{k}^{0} & \mathbf{o}_{k}^{1} & \ldots & \mathbf{o}_{k}^{N_{\text {obs }}^{\max }}
\end{array}\right]^{T} \\
& =\left[\begin{array}{cccc}
\delta x_{k}^{0} & \delta y_{k}^{0} & v_{k}^{0} & \delta \psi_{k}^{0} \\
\delta x_{k}^{1} & \delta y_{k}^{1} & \delta v_{k}^{1} & \delta \psi_{k}^{1} \\
\vdots & \vdots & \vdots & \vdots \\
\delta x_{k}^{N_{\text {obs }}^{\max }} & \delta y_{k}^{N_{\text {obs }}^{\max }} & \delta v_{k}^{N_{\text {obs }}^{\max }} & \delta \psi_{k}^{N_{\text {obs }}^{\max }}
\end{array}\right], \tag{9}
\end{align*}
$$

where $N_{\text {obs }}^{\text {max }}$ is the maximum number of the observed SVs for the EV; $v_{k}^{0}$ denotes the current speed of the EV; $\delta x_{k}^{i}, \delta y_{k}^{i}, \delta v_{k}^{i}$, and $\delta \psi_{k}^{i}$ represent the differences in the X -axis coordinate, Y-axis coordinate, speed, and heading angle, respectively,
between the EV and the destination position ( $i=0$ ) as well as the $i$-th $\mathrm{SV}\left(i=1,2, \ldots, N_{\text {obs }}^{\text {max }}\right)$.

Here, the RL policy is represented by a neural network $\pi$ parameterized by $\boldsymbol{\theta}$. Given the RL observation $\mathbf{O}_{k}$ at time step $k$, the action of RL agent is generated by:

$$
\begin{equation*}
a_{k}^{R L}=\pi_{\boldsymbol{\theta}}\left(\mathbf{O}_{k}\right) \tag{10}
\end{equation*}
$$

The specific definition of the sub-action spaces in (3) are introduced as follows. The waypoint sub-action space is defined as:

$$
\begin{equation*}
A_{1}=\left\{\mathrm{WP}_{0}, \mathrm{WP}_{1}, \ldots, \mathrm{WP}_{4}\right\}, \tag{11}
\end{equation*}
$$

where $\mathrm{WP}_{i}=\left[x_{i}^{\mathrm{WP}} y_{i}^{\mathrm{WP}} \psi_{i}^{\mathrm{WP}}\right]^{T}$ is the $i$-th waypoint, which includes the reference information about the X -axis, Y -axis coordinates, and heading angle of the waypoint, respectively. Waypoints are provided by a predefined road map and several path-searching methods, such as $A^{*}$ search algorithm. A reference waypoint set is generated at the beginning of the task. The 5 waypoints closest to the $\mathrm{EV}\left(\mathrm{WP}_{i}^{\prime}, i=0,1, \ldots, 4\right)$ are added to the $A_{1}$. The reference velocity sub-action space is defined as:

$$
\begin{equation*}
A_{2}=\left\{0, \frac{v_{\text {limit }}}{4}, \frac{v_{\text {limit }}}{2}, \frac{3 v_{\text {limit }}}{4}, v_{\text {limit }}\right\}, \tag{12}
\end{equation*}
$$

where $v_{\text {limit }}$ is speed limitation of the road. The lane change sub-action space is defined as:

$$
\begin{equation*}
A_{3}=\{-1,0,1\}, \tag{13}
\end{equation*}
$$

where $-1,0$, and 1 represent left lane change, lane keeping, and right lane change maneuvers, respectively.

The coordination of the above three sub-action spaces can enable flexible motion patterns for the EV to interact with SVs exhibiting diverse behaviors. When rapid movement toward the target is needed, the RL agent can select a distant waypoint and high speed; while it can choose the nearest waypoint and low speed for emergency braking. The lanechanging sub-action adds further flexibility. Selecting a closer waypoint on an adjacent lane indicates an urgent lane change, while a distant waypoint enables a smoother lane change for collision avoidance or overtaking. The reward function and the training environment are provided by the LLM agents. After the RL policy generates action outputs based on observations, these actions are decoded and passed to the model predictive controller, which converts them into execution trajectories and control commands to be applied to the EV.

Once the training curriculum and reward function are decoded from the answers of LLM agents, the RL agent explores the environment set by the configuration of the selected curriculum. Relevant historical training information is recorded in the replay buffer. Once a certain number of episodes have been gathered, the RL policy undergoes training to optimize the cumulative objective function that is associated with the sequence of generated training curricula $\left(C_{0}, C_{1}, \ldots, C_{N_{\max }+1}\right)$ as follows:

$$
\begin{equation*}
\boldsymbol{\theta}^{*}=\arg \max _{\boldsymbol{\theta},\left(C_{0}, C_{1}, \ldots, C_{N_{\max }+1}\right)} J(\boldsymbol{\theta}), \tag{14}
\end{equation*}
$$

where $J(\boldsymbol{\theta})$ denotes the objective function for the RL policy with parameter $\boldsymbol{\theta}$.

## User

You are a curriculum analyst tasked with analyzing suitable environment settings for the RL policy in the upcoming training process. Your task is to provide recommendations to the curriculum designer.
\{Curriculum set explanation\}
\{Principles of curriculum learning\}
\{Guidance for curriculum switching analysis\}, ...

## Curriculum Analysis Agent

Given the current training progress at episode number 0 , we are at the very beginning of the training process for the RL policy. Here is a structured guideline for designing the curriculum and transitioning through it:

1. **Initial Curriculum Setup**

- Surrounding Vehicle Density: Begin with No Vehicles. This allows the RL policy to learn the basic mechanics of lane changing and overtaking without the added complexity of other vehicles.
- Surrounding Vehicle Speed Mode: Start with Stationary Mode. This simplifies the environment further, allowing the RL policy to focus on the basic task without dynamic interactions..

2. **Transition Strategy**

- Traffic Density Transition: Monitor the policy's success rate in the current curriculum. Once a high success rate is achieved (e.g., 8090\%), transition to the next level of complexity.
- Gradually increase vehicle density from no vehicles ( $\mathrm{x}=0$ ) to low $(x=1)$, medium ( $x=2$ ), and high ( $x=3$ ) while maintaining the stationary speed mode ( $z=0$ ).
$\cdots$
(a) Curriculum analysis agent


## User

You are a curriculum selector responsible for choosing the best training environment for the RL agent. Your task is to select the optimal curriculum for the upcoming training round, which will initialize the dynamic training environment.
\{Curriculum set contents\}
\{Principles for curriculum transitions\}
\{Curriculum analysis results\}, ...

## Curriculum Generation Agent

Given that the RL agent is at the very beginning of the training process, we should start with the simplest curriculum to allow it to learn the fundamental mechanics of driving tasks. Based on the curriculum analyst's guidelines, we should begin with the multilane overtaking scenario with no surrounding vehicles and stationary speed mode. This will provide a straightforward environment for the agent to grasp basic actions and strategies without the complexity of interacting with other vehicles.

Therefore, the initial curriculum choice is:

$$
0,0
$$

(b) Curriculum generation agent

Fig. 3. Representative segments of the curriculum generation process demonstration during the initial training phase.

## V. Experiments

## A. Experimental Setup

In this section, we implement the automated policy learning workflow in two urban driving scenarios. The experiments are carried out on the Ubuntu 18.04 system with $\operatorname{Intel}(\mathrm{R})$ Core(TM) i9-14900K CPU and NVIDIA GeForce RTX 3090 GPU. All self-driving scenarios involved in experiments are constructed on the CARLA simulator [47]. Here, we select the Tesla Model 3 as the EV and SVs. To validate the capability

![](https://cdn.mathpix.com/cropped/363a3fc3-3499-4600-8838-03007fe07260-09.jpg?height=2272&width=826&top_left_y=210&top_left_x=195)
Fig. 4. Representative segments of the reward generation process demonstration during the initial training phase.

of LearningFlow in automating policy learning across different driving tasks, an on-ramp merging scenario and a multilane overtaking scenario from the Town06 map are chosen for validation. These two scenarios are essential and classic components of urban driving, each with distinct operational characteristics. The overtaking task requires the EV to engage in multiple rounds of interactive decision making with SVs moving in parallel, ensuring a safe overtaking maneuver without disrupting other vehicles. In contrast, the merging task necessitates an accurate assessment of the inter-vehicle gap sizes and driving intentions of SVs on the main road, allowing the EV to enter from a narrow ramp within a limited time and distance. Failure to merge within this time window could force the vehicle to decelerate suddenly or even stop at the end of the lane, which complicates the decision-making process.

Since the task involves selecting curricula and designing reward functions based on contextual information from multiple information sources, robust reasoning capabilities of LLMs are necessary. Additionally, unlike tasks in the LLMs-in-the-task-loop approaches, our LearningFlow framework only calls the LLMs during the curriculum generation and reward design processes, meaning that the frequency of LLM calls is relatively low. Taking this into consideration, we choose the GPT-4o model for our experiments, which strikes a balance between reasoning capability and performance requirements.

Specifically, proximal policy optimization (PPO) [48] is chosen as the RL policy training algorithm considering its ability to balance training stability and efficiency in data utilization. This feature makes it particularly suitable for the stated driving tasks that are investigated. Additionally, PPO is well-suited for multi-dimensional discrete action spaces, which is essential in this work. The clipped objective function of the PPO algorithm is utilized to update the RL policy, which is shown as follows:

$$
\begin{equation*}
J_{k}(\boldsymbol{\theta})=\mathbb{E}_{k}\left[\min \left(\rho_{k}(\boldsymbol{\theta}) \hat{A}_{k}, \operatorname{clip}\left(\rho_{k}(\boldsymbol{\theta}), 1-\epsilon, 1+\epsilon\right) \hat{A}_{k}\right)\right], \tag{15}
\end{equation*}
$$

where $\rho_{k}(\boldsymbol{\theta})$ represents the probability ratio of the new policy to the old policy; $\hat{A}_{k}$ denotes the estimator of the advantage function at time step $k ; \epsilon$ is the clip parameter.

In this work, the actor-critic architecture is adopted to implement the PPO policy of the RL executor. The action network and critic network are set as fully connected networks with 2 hidden layers of 256 units and 128 units by PyTorch and trained with the Adam optimizer. The number of epochs is set to 50 . The learning rate of the action network and critic network are set to $5 \times 10^{-4}$ and $1 \times 10^{-3}$, respectively. $\gamma$ is set to $0.99 . \eta$ is set to 0.2 . The update frequency parameters $N_{P}, N_{C}$, and $N_{R}$ are set to 50,100 , and 1000 , respectively. The low-level model predictive control optimization problem is solved by CasADi [49], with the IPOPT option and single-shooting approach. We train the RL policy using the proposed LearningFlow method within the experimental scenarios, where modifications are made only to the textual descriptions of task characteristics across different scenarios.

In this work, the proposed framework is compared with the following approaches as follows:

![](https://cdn.mathpix.com/cropped/363a3fc3-3499-4600-8838-03007fe07260-10.jpg?height=742&width=802&top_left_y=163&top_left_x=206)
Fig. 5. A failure case of reward generation process by LearningFlow without analysis agents, where the design flaws are highlighted in bold red.

- Vanilla PPO: the vanilla PPO policy is directly trained by PPO algorithm [48] in the task scenario.
- AutoReward: a state-of-the-art approach which iteratively refines the reward function generated by LLMs after the whole training [46]. The policy is directly trained in the task scenario. Only the code of the environmental observation part is provided, which is the same as LearningFlow.
- LearningFlow without analysis process (w/o AP): the proposed method without the analysis agents.
For the sake of fairness, the parameters of the downstream RL executors for all methods are set to be identical. The clip parameter is configured to 0.2 . The SVs operate in an autopilot mode provided by the CARLA simulator, with their driving styles being randomly assigned.

The traffic density settings in the scenarios are introduced as follows. Empty indicates that no SVs are present. In the on-ramp merging scenario, low, medium, and high densities correspond to 2,4 , and 8 SVs , respectively, in the two nearest lanes of the main road near the merging lane. In the multi-lane overtaking scenario, these densities represent the presence of 1,2, and 3 SVs in front of the EV. These settings introduce a variety of traffic conditions and potentially require the EV to interact with different numbers of SVs, thereby increasing the complexity of the driving tasks. We begin by training the RL policy in multi-lane overtaking and on-ramp merging scenarios, respectively. Subsequently, we evaluate the performance of the trained RL policies in these scenarios, testing them with diverse traffic densities, and across different driving tasks.

## B. Demonstration of Collaboration Among Multiple LLM

## Agents

To illustrate the collaboration among multiple LLM agents during the policy training process, we selected an example of
analysis and content generation from the cooperative interactions of four LLM agents, as shown in Figs. 3.4. As a comparison, a reward generation demonstration of LearningFlow without the analysis process is shown in Fig. 5.

As shown in Fig. 3, at the beginning of training, the curriculum analysis agent evaluates the upcoming training curriculum based on the training progress and the core principles of CL, considering factors such as initial curriculum settings, switching strategies, and switching criteria. Subsequently, the curriculum generation agent selects a suitable training curriculum from the curriculum set by leveraging the analyzed results and adhering to CL principles, generating the selection in a standard format. During the subsequent training process, the curriculum analysis and generation agents iteratively analyze and evaluate training curricula by incorporating the historical curriculum sequence and RL policy training data, enabling automatic curriculum switching and thereby improving the sample efficiency of policy learning.

In Fig. 4, during the early stages of training, the reward analysis agent begins by examining the current performance of the policy, considering various factors relevant to the autonomous driving task, and analyzing each reward component, its significance, and the reference value range that should be considered in the reward generation process. Next, the reward generation agent, based on this analysis, generates a reward function in a standardized format, adhering to the function's signature and coding conventions. The generated reward function is then extracted and embedded into the environment code for training. After a certain number of episodes, the reward analysis and generation agents adjust the reward components and coefficients based on the policy's performance, providing reasons for the adjustments along with the reference value range. Through the collaboration of the reward function agents, we achieve the automatic design and online adjustment of the reward function, thereby enhancing the performance of the policy.

In the failure case shown in Fig. 5, the reward generation agent fails to generate an appropriate reward function. Specifically, the reward generation agent misinterprets the expected impact of traffic density $N \_s v$ on the reward function, reversing the operator, which hinders the RL policy when exploring tasks with higher traffic density. Furthermore, without the support of analysis agents, the reward generation agent fails to accurately understand the contribution of the lane_change_times variable in the lane change penalty, which should be calculated in an event-triggered manner. Additionally, it also leads to difficulties in designing reasonable coefficient ranges. For instance, the coefficient and calculation method of the speed reward term could cause its accumulated value to exceed the reward for task completion, leading the agent to prioritize speed rewards at the expense of driving safety and encouraging reckless behavior. This demonstration highlights the crucial role of the analysis agent in the LearningFlow framework.

## C. Comparative Results and Analysis

To quantitatively compare the performance of LearningFlow with the baseline methods, we conduct statistical testing on

TABLE I
Performance comparison among different methods and different driving tasks.
| Methods |  | Empty |  |  | Low Density |  |  | Medium Density |  |  | High Density |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | S(\%) | C(\%) | TO(\%) | S(\%) | C(\%) | TO(\%) | S(\%) | C(\%) | TO(\%) | S(\%) | C(\%) | TO(\%) |
| Vanilla PPO | Overtaking | 99 | 1 | 0 | 80 | 20 | 0 | 69 | 31 | 0 | 62 | 38 | 0 |
|  | Merging | 100 | 0 | 0 | 89 | 11 | 0 | 75 | 25 | 0 | 68 | 32 | 0 |
| AutoReward (iter $=0$ ) | Overtaking | 99 | 0 | 1 | 73 | 27 | 0 | 60 | 40 | 0 | 48 | 52 | 0 |
|  | Merging | 0 | 100 | 0 | 82 | 18 | 0 | 71 | 29 | 0 | 51 | 49 | 0 |
| AutoReward (iter=5) | Overtaking | 100 | 0 | 0 | 85 | 15 | 0 | 76 | 24 | 0 | 70 | 30 | 0 |
|  | Merging | 100 | 0 | 0 | 94 | 6 | 0 | 85 | 15 | 0 | 71 | 29 | 0 |
| LearningFlow (w/o AP) | Overtaking | 100 | 0 | 0 | 89 | 11 | 0 | 75 | 25 | 0 | 69 | 31 | 0 |
|  | Merging | 100 | 0 | 0 | 91 | 9 | 0 | 83 | 17 | 0 | 72 | 28 | 0 |
| LearningFlow | Overtaking | 100 | 0 | 0 | 97 | 3 | 0 | 90 | 10 | 0 | 85 | 15 | 0 |
|  | Merging | 100 | 0 | 0 | 99 | 1 | 0 | 94 | 6 | 0 | 87 | 13 | 0 |


Note: S, C, and TO represent success rate, collision rate, and timeout rate, respectively.

![](https://cdn.mathpix.com/cropped/363a3fc3-3499-4600-8838-03007fe07260-11.jpg?height=1125&width=1724&top_left_y=734&top_left_x=199)
Fig. 6. Key frames of two demonstrations using our method in multi-lane overtaking and on-ramp merging scenarios within CARLA. The upper and lower parts of the sub-figures show third-person views of the EV and bird-eye views, respectively. In the third-person sub-figures, the green rectangles represent the intermediate points determined by the RL policy. In the bird-eye view sub-figures, the red rectangle corresponds to the EV, and the green rectangles denote the SVs.

trained policies trained by all approaches in the on-ramp merging scenario and the multi-lane-overtaking scenario with the different traffic densities. Each learned policy undergoes 100 repeated tests in each task setting. The results are summarized in Table 1.

The table shows that the proposed approach achieves the highest success rate across all testing driving tasks and traffic configurations. Although the success rate tends to decrease as the number of SVs increases, this outcome demonstrates the effectiveness of the LearningFlow framework in the automated policy learning process. It can help to obtain a more effective policy within the same number of training episodes,
thus improving the sample efficiency. Overall, this result demonstrates the effectiveness and generalization capability of LearningFlow across various driving tasks.

Additionally, it is observed that methods utilizing LLMs to assist the training of the RL policy generally outperform the vanilla PPO in most of the test tasks. It is noted that the RL policy trained by the AutoReward method with zero iterations fails to complete the merging task without SVs, likely due to the suboptimal design of the reward function, which arises from the lack of access to the internal environment code. However, this issue is resolved after the reward function undergoes iteration, as demonstrated by AutoReward (iter=5).

TABLE II
Performance comparison at the multi-lane overtaking scenario among different RL algorithms.
| Methods | Empty |  |  | Low Density |  |  | Medium Density |  |  | High Density |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | S (\%) | C (\%) | TO(\%) | S(\%) | C(\%) | TO(\%) | S (\%) | C (\%) | TO(\%) | S (\%) | C(\%) | TO(\%) |
| Vanilla DQN | 100 | 0 | 0 | 78 | 22 | 0 | 70 | 30 | 0 | 59 | 41 | 0 |
| LearningFlow DQN | 100 | 0 | 0 | 92 | 8 | 0 | 86 | 14 | 0 | 81 | 19 | 0 |
| Vanilla SAC | 100 | 0 | 0 | 82 | 18 | 0 | 72 | 28 | 0 | 63 | 37 | 0 |
| LearningFlow SAC | 100 | 0 | 0 | 93 | 7 | 0 | 83 | 17 | 0 | 79 | 21 | 0 |


Note: S, C, and TO represent success rate, collision rate, and timeout rate, respectively.

Moreover, a comparison between the results of LearningFlow and LearningFlow (w/o AP) reveals that the absence of the analysis agent leads to a decline in policy performance. This could be attributed to the tendency of the generation agent to produce erroneous results when dealing with complex tasks, resulting in training instability. This highlights the crucial role of the analysis agent in enhancing the effectiveness of policy training.

## D. Demonstration of Maneuvering Abilities in Driving Tasks

To illustrate the actual overtaking and merging capabilities, we select one outcome from the testing results of all driving tasks using the proposed approach. The snapshots of these two examples are shown in Fig. 6

- Multi-lane overtaking task: In this example, the EV is initialized in the middle lane, while two SVs are initialized at different positions in the right lane. At 0.2 s, the RL policy outputs a reference point ahead, guiding the EV to accelerate for overtaking. Then, at 1.5 s , the RL policy detects a nearby SV performing a left lane change action. In response, the policy outputs a reference point to the left ahead, guiding the EV to perform the overtaking lane change, which is completed in 3.5 s , followed by acceleration. Finally, between 4.6 s and 6.6 s , the EV successfully completes the overtaking maneuver and reaches the target area under the guidance of the RL policy, completing the multi-lane overtaking task safely and efficiently.
- On-ramp merging task: At the beginning of the demonstration, the EV is initialized at a random position on the merging lane. At this point, the RL policy outputs a reference point ahead, guiding the EV towards the main road. Then, at 5.5 s , the EV enters the merging zone. The RL policy detects an SV approaching from the rearright and outputs a reference point ahead to accelerate and increase the distance. At 6.0 s , the rear-right SV slows down and yields, maintaining a safe distance from the EV. The RL policy then outputs a reference point in the right lane ahead, guiding the EV to merge into the target lane. Finally, between 6.9 s and 9 s , the RL policy selects a distant reference point and sets a high reference speed, guiding the EV to smoothly complete the merging maneuver and reach the target area. Ultimately, under the guidance of the RL policy, the EV successfully and efficiently completes the on-ramp merging task.


## E. Adaptation to Different RL Algorithms

To evaluate the generalization of the proposed LearningFlow framework across different RL algorithms, we replace the downstream PPO algorithm with DQN and SAC algorithms and conducted training and testing on the multi-lane overtaking task. The test results are shown in Table II As seen from the statistics in the table, LearningFlow effectively facilitates policy learning across different RL algorithms, improving the success rate of the trained policies in task scenarios. Compared to the RL policies trained with the PPO algorithm in Table 1. the success rates for DQN and SAC are slightly lower. This can be attributed to the fact that the performance of the RL policies also depends on the configuration of the action space and state space. Specifically, DQN could not perform as well as PPO in handling continuous state spaces and multi-dimensional discrete action spaces. Meanwhile, SAC, which is designed for continuous action spaces, is inherently limited in tasks involving multi-dimensional discrete action spaces. Nevertheless, in general, the RL policies trained with the LearningFlow framework show enhanced performance compared to Vanilla RL algorithms. This demonstrates that the proposed approach for reward and curriculum generation is adaptable to various RL algorithms, and this showcases the generalization capability of the proposed framework.

## VI. Conclusion

In this paper, we propose LearningFlow, a closed-loop automated policy learning workflow for autonomous driving, which utilizes the collaboration of LLM agents to generate and dynamically adjust the training curriculum and reward functions of the RL policy. By incorporating analysis and generation agents, our approach enhances the understanding of complex driving tasks, thereby improving the efficiency of automatic curriculum and reward generation, as well as the performance of RL policies. Experimental results demonstrate the effectiveness of the proposed framework. Compared to baseline and SOTA methods, the RL policy trained by LearningFlow achieves the highest success rate. Furthermore, ablation studies validate the effectiveness of our policy learning workflow. Demonstrations in various driving scenarios, involving interactions with SVs, showcase the superior interactionawareness capability of our method. Finally, experimental results indicate that the proposed method is highly compatible with various RL algorithms. Future work includes incorporating advanced diffusion models to enhance the multimodal decision-making capability of the downstream RL executor, thereby further improving the generalization ability of RL solutions.

## References

[1] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong et al., "A survey of large language models," arXiv preprint arXiv:2303.18223, 2023.
[2] Y. Chang, X. Wang, J. Wang, Y. Wu, L. Yang, K. Zhu, H. Chen, X. Yi, C. Wang, Y. Wang et al., "A survey on evaluation of large language models," ACM Transactions on Intelligent Systems and Technology, vol. 15, no. 3, pp. 1-45, 2024.
[3] C. Cui, Y. Ma, X. Cao, W. Ye, Y. Zhou, K. Liang, J. Chen, J. Lu, Z. Yang, K.-D. Liao et al., "A survey on multimodal large language models for autonomous driving," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 958979.
[4] B. R. Kiran, I. Sobh, V. Talpaert, P. Mannion, A. A. Al Sallab, S. Yogamani, and P. Pérez, "Deep reinforcement learning for autonomous driving: A survey," IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 6, pp. 4909-4926, 2021.
[5] K. Muhammad, A. Ullah, J. Lloret, J. Del Ser, and V. H. C. de Albuquerque, "Deep learning for safe autonomous driving: Current challenges and future directions," IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 7, pp. 4316-4336, 2020.
[6] S. Mozaffari, O. Y. Al-Jarrah, M. Dianati, P. Jennings, and A. Mouzakitis, "Deep learning-based vehicle behavior prediction for autonomous driving applications: A review," IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 1, pp. 33-47, 2020.
[7] S. Ettinger, S. Cheng, B. Caine, C. Liu, H. Zhao, S. Pradhan, Y. Chai, B. Sapp, C. R. Qi, Y. Zhou et al., "Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 9710-9719.
[8] Z. Qiao, J. Schneider, and J. M. Dolan, "Behavior planning at urban intersections through hierarchical reinforcement learning," in Proceedings of the International Conference on Robotics and Automation, 2021, pp. 2667-2673.
[9] Y. Wang, Y. Li, Z. Peng, H. Ghazzai, and J. Ma, "Chance-aware lane change with high-level model predictive control through curriculum reinforcement learning," in Proceedings of the International Conference on Robotics and Automation, 2024.
[10] Z. Peng, X. Zhou, L. Zheng, Y. Wang, and J. Ma, "Reward-driven automated curriculum learning for interaction-aware self-driving at unsignalized intersections," in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2024.
[11] S. Booth, W. B. Knox, J. Shah, S. Niekum, P. Stone, and A. Allievi, "The perils of trial-and-error reward design: misdesign through overfitting and invalid task specifications," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 5, 2023, pp. 5920-5929.
[12] Y. Hu, W. Wang, H. Jia, Y. Wang, Y. Chen, J. Hao, F. Wu, and C. Fan, "Learning to utilize shaping rewards: A new approach of reward shaping," Advances in Neural Information Processing Systems, vol. 33, pp. 15931-15 941, 2020.
[13] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction. Cambridge, MA, US: MIT Press, 2018.
[14] Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar, "Eureka: Humanlevel reward design via coding large language models," arXiv preprint arXiv:2310.12931, 2023.
[15] A. Abouelazm, J. Michel, and J. M. Zoellner, "A review of reward functions for reinforcement learning in the context of autonomous driving," arXiv preprint arXiv:2405.01440, 2024.
[16] S. Narvekar, B. Peng, M. Leonetti, J. Sinapov, M. E. Taylor, and P. Stone, "Curriculum learning for reinforcement learning domains: A framework and survey," Journal of Machine Learning Research, vol. 21, no. 181, pp. 1-50, 2020.
[17] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum learning," in Proceedings of the International Conference on Machine Learning. ACM, 2009, pp. 41-48.
[18] P. Soviany, R. T. Ionescu, P. Rota, and N. Sebe, "Curriculum learning: A survey," International Journal of Computer Vision, vol. 130, no. 6, pp. 1526-1565, 2022.
[19] A. Graves, M. G. Bellemare, J. Menick, R. Munos, and K. Kavukcuoglu, "Automated curriculum learning for neural networks," in Proceedings of the International Conference on Machine Learning. PMLR, 2017, pp. 1311-1320.
[20] Z. Zhuang, Z. Fu, J. Wang, C. Atkeson, S. Schwertfeger, C. Finn, and H. Zhao, "Robot parkour learning," in Proceedings of the Conference on Robot Learning. PMLR, 2023, pp. 1-20.
[21] M. Al-Sharman, R. Dempster, M. A. Daoud, M. Nasr, D. Rayside, and W. Melek, "Self-learned autonomous driving at unsignalized intersections: A hierarchical reinforced learning approach for feasible decisionmaking," IEEE Transactions on Intelligent Transportation Systems, 2023.
[22] S. Kai, B. Wang, D. Chen, J. Hao, H. Zhang, and W. Liu, "A multi-task reinforcement learning approach for navigating unsignalized intersections," in Proceedings of the Intelligent Vehicles Symposium, 2020, pp. 1583-1588.
[23] S. Sodhani, A. Zhang, and J. Pineau, "Multi-task reinforcement learning with context-based representations," in Proceedings of the International Conference on Machine Learning. PMLR, 2021, pp. 9767-9779.
[24] Y. Liu, Y. Gao, Q. Zhang, D. Ding, and D. Zhao, "Multi-task safe reinforcement learning for navigating intersections in dense traffic," Journal of the Franklin Institute, vol. 360, no. 17, pp. 13737-13760, 2023.
[25] J. Fu, K. Luo, and S. Levine, "Learning robust rewards with adversarial inverse reinforcement learning," in Proceedings of International Conference on Learning Representations. PMLR, 2018, pp. 1-15.
[26] S. Arora and P. Doshi, "A survey of inverse reinforcement learning: Challenges, methods and progress," Artificial Intelligence, vol. 297, p. 103500, 2021.
[27] T. Phan-Minh, F. Howington, T.-S. Chu, S. U. Lee, M. S. Tomov, N. Li, C. Dicle, S. Findler, F. Suarez-Ruiz, R. Beaudoin et al., "Driving in real life with inverse reinforcement learning," arXiv preprint arXiv:2206.03004, 2022.
[28] J. Nan, W. Deng, R. Zhang, Y. Wang, R. Zhao, and J. Ding, "Interactionaware planning with deep inverse reinforcement learning for humanlike autonomous driving in merge scenarios," IEEE Transactions on Intelligent Vehicles, 2023.
[29] R. Devidze, P. Kamalaruban, and A. Singla, "Exploration-guided reward shaping for reinforcement learning under sparse rewards," Advances in Neural Information Processing Systems, vol. 35, pp. 5829-5842, 2022.
[30] M. Kwon, S. M. Xie, K. Bullard, and D. Sadigh, "Reward design with language models," in Proceedings of International Conference on Learning Representations. PMLR, 2023, pp. 1-18.
[31] H. Li, X. Yang, Z. Wang, X. Zhu, J. Zhou, Y. Qiao, X. Wang, H. Li, L. Lu, and J. Dai, "Auto mc-reward: Automated dense reward design with large language models for minecraft," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16426-16435.
[32] Y. Guan, Y. Ren, S. E. Li, Q. Sun, L. Luo, and K. Li, "Centralized cooperation for connected and automated vehicles at intersections by proximal policy optimization," IEEE Transactions on Vehicular Technology, vol. 69, no. 11, pp. 12597-12608, 2020.
[33] Y. Song, H. Lin, E. Kaufmann, P. Dürr, and D. Scaramuzza, "Autonomous overtaking in Gran Turismo sport using curriculum reinforcement learning," in Proceedings of the International Conference on Robotics and Automation, 2021, pp. 9403-9409.
[34] Z. Peng, X. Zhou, Y. Wang, L. Zheng, M. Liu, and J. Ma, "Curriculum proximal policy optimization with stage-decaying clipping for selfdriving at unsignalized intersections," in Proceedings of the International Intelligent Transportation Systems Conference, 2023, pp. 5027-5033.
[35] Z. Qiao, K. Muelling, J. M. Dolan, P. Palanisamy, and P. Mudalige, "Automatically generated curriculum based reinforcement learning for autonomous vehicles in urban environment," in Proceedings of the Intelligent Vehicles Symposium, 2018, pp. 1233-1238.
[36] S. Khaitan and J. M. Dolan, "State dropout-based curriculum reinforcement learning for self-driving at unsignalized intersections," in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2022, pp. 12219-12 224.
[37] L. Wen, D. Fu, X. Li, X. Cai, T. Ma, P. Cai, M. Dou, B. Shi, L. He, and Y. Qiao, "Dilu: A knowledge-driven approach to autonomous driving with large language models," in Proceedings of International Conference on Learning Representations. PMLR, 2024, pp. 1-20.
[38] Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K.-Y. K. Wong, Z. Li, and H. Zhao, "Drivegpt4: Interpretable end-to-end autonomous driving via large language model," IEEE Robotics and Automation Letters, 2024.
[39] W. Wang, J. Xie, C. Hu, H. Zou, J. Fan, W. Tong, Y. Wen, S. Wu, H. Deng, Z. Li et al., "Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving," arXiv preprint arXiv:2312.09245, 2023.
[40] X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao, "Drivevlm: The convergence of autonomous driving and large vision-language models," arXiv preprint arXiv:2402.12289, 2024.
[41] W. Yu, N. Gileadi, C. Fu, S. Kirmani, K.-H. Lee, M. G. Arenas, H.T. L. Chiang, T. Erez, L. Hasenclever, J. Humplik et al., "Language to
rewards for robotic skill synthesis," in Proceedings of the Conference on Robot Learning. PMLR, 2023, pp. 1-31.
[42] R. Hazra, A. Sygkounas, A. Persson, A. Loutfi, and P. Z. D. Martires, "Revolve: Reward evolution with large language models for autonomous driving," arXiv preprint arXiv:2406.01309, 2024.
[43] Z. Zhou, J. Zhang, J. Zhang, B. Wang, T. Shi, and A. Khamis, "In-context learning for automated driving scenarios," arXiv preprint arXiv:2405.04135, 2024.
[44] Y. J. Ma, W. Liang, H.-J. Wang, S. Wang, Y. Zhu, L. Fan, O. Bastani, and D. Jayaraman, "Dreureka: Language model guided sim-to-real transfer," in Robotics: Science and Systems, 2024.
[45] K. Ryu, Q. Liao, Z. Li, K. Sreenath, and N. Mehr, "Curricullm: Automatic task curricula design for learning complex robot skills using large language models," arXiv preprint arXiv:2409.18382, 2024.
[46] X. Han, Q. Yang, X. Chen, Z. Cai, X. Chu, and M. Zhu, "Autoreward: Closed-loop reward design with large language models for autonomous driving," IEEE Transactions on Intelligent Vehicles, 2024.
[47] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, "CARLA: An open urban driving simulator," in Proceedings of the Conference on Robot Learning. PMLR, 2017, pp. 1-16.
[48] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.
[49] J. A. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, "Casadi: a software framework for nonlinear optimization and optimal control," Mathematical Programming Computation, vol. 11, pp. 1-36, 2019.


[^0]:    Zengqi Peng, Yubin Wang, and Lei Zheng are with the Robotics and Autonomous Systems Thrust, The Hong Kong University of Science and Technology (Guangzhou), China (email: zpeng940@connect.hkust-gz.edu.cn; ywang575@connect.hkust-gz.edu.cn; lzheng135@connect.hkust-gz.edu.cn).

    Xu Han is with the Data Science and Analytics Thrust, The Hong Kong University of Science and Technology (Guangzhou), China (email: xhanab@connect.ust.hk).

    Jun Ma is with the Robotics and Autonomous Systems Thrust, The Hong Kong University of Science and Technology (Guangzhou), China, and also with the Department of Electronic and Computer Engineering, The Hong Kong University of Science and Technology, Hong Kong SAR, China (e-mail: jun.ma@ust.hk).

