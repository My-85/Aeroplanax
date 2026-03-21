你需要修改 /home/dqy/aeroplanax/new/20251215最新代码库/RL_autotuner 下的 autotuner 框架。这个框架仿照 autoresearch（位于 /home/dqy/aeroplanax/new/20251215最新代码库/autoresearch/autoresearch/）搭建，用于自动调优 RL reward 配置。当前框架有几个关键缺陷需要修复。
重要约束（必读）
Planax 源码 FROZEN：Planax/ 目录下的所有 .py 文件（envs/、reward_functions/、termination_conditions/、train_.py、render_.py）都是冻结的，禁止修改它们的逻辑代码。
唯一例外：config_patcher.py 被允许通过正则替换 Planax/envs/reward_functions/full_domain_reward.py 中 REWARD_CONFIG 字典的数值（这是设计好的机制）。
所有新建/修改的文件都放在 RL_autotuner/ 目录下。
训练脚本是 Planax/train_full_domain_maneuver_v3.py，它已支持 DRYRUN_TIMESTEPS 环境变量（第 744-745 行）控制训练步数。
环境是 Planax/envs/aeroplanax_full_domain_maneuver.py，22D obs，单智能体 full_domain 姿态控制。
关键背景信息
训练脚本的日志输出格式（用于正则匹配）
训练脚本 train_full_domain_maneuver_v3.py 的 stdout 格式如下：
env_step=1000000   return=-0.23  episode_length=100    success_times=0.00 curriculum_level=0.00 on_target_steps=0.0 timeout_count=0.0
  reward: r_main=0.5432  theta_deg=85.3  delta_vt=25.4  r_nz=-0.001234  r_qbar=-0.00000012  alt_km=5.2
  loss:   actor_loss=0.0012  value_loss=0.1234  entropy=14.500  approx_kl=0.00123  grad_norm=2.345
  注意：

没有 update= 关键字，只有 env_step=
theta_deg 为 theta_deg=XX.X 格式
delta_vt 为 delta_vt=XX.X 格式
环境 state 结构
AeroPlanaxFullDomainEnv 的 state 包含：

state.plane_state.q0[agent_id], .q1, .q2, .q3 — 当前四元数 q_NB
state.plane_state.vt[agent_id] — 当前真空速
state.target_heading[agent_id], .target_pitch[agent_id], .target_roll[agent_id] — 目标欧拉角
state.target_vt[agent_id] — 目标速度
state.curriculum_level — 当前课程级别
state.on_target_steps — 连续 on-target 步数
计算 theta_deg 的方法（参考 env ）：
q_curr = jnp.array([ps.q0[0], ps.q1[0], ps.q2[0], ps.q3[0]])
q_curr = _quat_normalize(q_curr)
q_tgt = _quat_conj(_quat_from_euler_bn(target_roll, target_pitch, target_heading))
theta = _quat_geodesic_angle(q_curr, q_tgt)
theta_deg = theta * 180.0 / jnp.pi
delta_vt = jnp.abs(vt - target_vt)

但 evaluator 不应该 import env 内部的私有函数。evaluator.py 已经定义了自己的四元数辅助函数（_quat_normalize, _quat_conj, _quat_from_euler_bn, _quat_geodesic_angle），应该用这些。

网络架构
full_domain 用 ActorCriticRNN，action_dim=[31, 41, 41, 41]，GRU_HIDDEN_DIM=256, FC_DIM_SIZE=256。 evaluator.py 中已有正确定义。

已有 baseline（仅作参考，不能直接加载）
用户有一个训好的四元数 baseline checkpoint： /home/dqy/aeroplanax/new/20251215最新代码库/results/baseline（四元数版本）/checkpoints/checkpoint_epoch_1000


但该 checkpoint 是在 aeroplanax_heading_pitch_V_quaternion_version_add_full_roll.py 环境下训的：

obs=16D，GRU_HIDDEN_DIM=128，FC_DIM_SIZE=128
与 full_domain 环境（obs=22D，GRU=256，FC=256）完全不兼容
不能加载到 full_domain evaluator 中

同时还有一个训好的欧拉角baseline checkpoint：  /home/dqy/aeroplanax/new/20251215最新代码库/results/baseline（欧拉角版本）/checkpoints/checkpoint_epoch_600


该两个 baseline 仅作为参考，你可以探索一下这两个baseline说明theta_deg多少度是可达的。

请阅读位于 /home/dqy/aeroplanax/new/20251215最新代码库/autoresearch/autoresearch/program.md，理解 autoresearch 的 LOOP FOREVER 模式、keep/discard via git commit/reset、NEVER STOP 原则。 你需要将这些理念适配到 RL_autotuner 的 program.md 中。

任务 1：初始化 Git 仓库
项目根目录是/home/dqy/aeroplanax/new/20251215最新代码库/RL_autotuner/。

在此目录下 git init（如果还没有的话）
创建 .gitignore，排除以下内容：
__pycache__/
*.pyc
results/
wandb/
.backups/
*.log
run.log
results.jsonl
.DS_Store
*.egg-info/

注意：results.jsonl 不提交到 git（和 autoresearch 的 results.tsv 一样，实验记录独立于 git）。

git add -A && git commit -m "initial: freeze codebase for RL autotuner Phase 1"
创建实验分支：git checkout -b autotuner/phase1
设置 remote：git remote add origin https://github.com/My-85/Aeroplanax.git 注意：只设置 remote，不要 push。

任务 2：修复 evaluator.py — 补全物理量提取
当前 evaluator.py 的 run_eval_episode() 第 246-251 行有一个 pass 占位符：
# Extract theta_deg and delta_vt from state if available
if hasattr(state, "env_state"):
    es = state.env_state
    if hasattr(es, "plane_state"):
        # theta_deg and delta_vt are in info dict
        pass

这里必须补全。需要从 env state 中计算 theta_deg 和 delta_vt。
具体修改：
2a. 从 state 计算 theta_deg 和 delta_vt
在 run_eval_episode() 的每个 step 后，使用 evaluator.py 自带的四元数辅助函数，从 state 中直接计算物理量。

注意 LogWrapper 的 state 嵌套结构：真正的 env state 在 state.env_state 中。

对每个 env（vmap 过后是 batch），需要：

取出当前四元数 q_curr = [q0, q1, q2, q3]（shape: (num_envs, 4)）
取出目标欧拉角 target_heading, target_pitch, target_roll
计算 q_tgt = _quat_conj(_quat_from_euler_bn(target_roll, target_pitch, target_heading))
计算 theta = _quat_geodesic_angle(q_curr, q_tgt)，theta_deg = theta * 180/pi
计算 delta_vt = |vt - target_vt|
由于这些操作需要对每个 env 做，可能需要 vmap 或 batch 运算。

将每步的 mean theta_deg 和 mean delta_vt 追加到 all_theta_deg 和 all_delta_vt 列表。

2b. 补全 on_target 判定
on_target = (theta_deg_batch < 10.0) & (delta_vt_batch < 25.0)
all_on_target.append(float(jnp.mean(on_target.astype(jnp.float32))))

2c. 修改返回值
run_eval_episode() 应返回：
{
    "seed": seed,
    "mean_theta_deg": float(np.mean(all_theta_deg)) if all_theta_deg else None,
    "mean_delta_vt": float(np.mean(all_delta_vt)) if all_delta_vt else None,
    "mean_per_step_reward": mean_reward,
    "crash_rate": crash_rate,
    "on_target_rate": float(np.mean(all_on_target)) if all_on_target else None,
    "total_steps": num_steps,
}

2d. 修改 evaluate_checkpoint() 的 aggregate
aggregate 应包含：
{
    "mean_theta_deg": float(np.mean([r["mean_theta_deg"] for r in all_results if r["mean_theta_deg"] is not None])),
    "std_theta_deg": ...,
    "mean_delta_vt": ...,
    "std_delta_vt": ...,
    "mean_per_step_reward": ...,
    "std_per_step_reward": ...,
    "mean_crash_rate": ...,
    "std_crash_rate": ...,
    "mean_on_target_rate": ...,
    "std_on_target_rate": ...,
}

任务 3：在 experiment_runner.py 中接入正式评估
当前 run_experiment() 只用 extract_training_metrics()（从 stdout 正则提取）。

修改 run_experiment() 函数：

1.训练完成后，如果 status != "crashed" 且 checkpoint_path 存在：
from evaluator import evaluate_checkpoint, EVAL_CONFIG
try:
    eval_result = evaluate_checkpoint(checkpoint_path, dict(EVAL_CONFIG))
    eval_metrics = eval_result["aggregate"]
except Exception as e:
    print(f"  Formal evaluation failed: {e}, falling back to training metrics")
    eval_metrics = None
2.如果正式评估成功，用 eval_metrics 做 champion 比较；否则 fallback 到 extract_training_metrics()。

3.修改 is_better_than_champion() 的 key 名，使其兼容两种 metrics 来源：

    正式评估用 mean_theta_deg
    训练日志 fallback 用 final_theta_deg
    函数里应该先查 mean_theta_deg，没有再查 final_theta_deg
4.在 log_result 中同时记录 training_metrics 和 eval_metrics。

任务 4：建立初始 Champion Baseline
当前 champion/champion_meta.json 是一个空 placeholder（所有 metrics 为 null）。 需要建立一个真实的初始 champion。

4a. 在 evaluator.py 中添加 evaluate_random_policy() 函数
def evaluate_random_policy(config: dict = None) -> dict:
    """Run evaluation with randomly initialized network (no checkpoint).
    This establishes the "random policy" baseline that any trained agent must beat.
    """
    if config is None:
        config = dict(EVAL_CONFIG)

    env_params = FullDomain_TaskParams()
    env = AeroPlanaxFullDomainEnv(env_params)
    env = LogWrapper(env)
    num_actors = env.num_agents

    network = ActorCriticRNN([31, 41, 41, 41], config=config)
    rng = jax.random.PRNGKey(0)
    obs_shape = env.observation_space(env.agents[0], env_params).shape

    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * num_actors, *obs_shape)),
        jnp.zeros((1, config["NUM_ENVS"] * num_actors)),
    )
    init_hstate = ScannedRNN.initialize_carry(
        config["NUM_ENVS"] * num_actors, config["GRU_HIDDEN_DIM"]
    )
    network_params = network.init(rng, init_hstate, init_x)

    loaded = {
        "params": network_params,
        "epoch": 0,
        "network": network,
        "env": env,
        "env_params": env_params,
        "num_actors": num_actors,
    }

    all_results = []
    for seed in config.get("EVAL_SEEDS", EVAL_CONFIG["EVAL_SEEDS"]):
        print(f"  Running random policy eval seed={seed}...")
        result = run_eval_episode(loaded, config, seed)
        all_results.append(result)
        print(f"    theta={result.get('mean_theta_deg', '?')}deg, crash_rate={result['crash_rate']:.4f}")

    # Aggregate across seeds (same logic as evaluate_checkpoint)
    aggregate = { ... }  # same aggregation as evaluate_checkpoint

    return {
        "checkpoint_path": None,
        "epoch": 0,
        "eval_config": {k: v for k, v in config.items() if k in ["NUM_ENVS", "NUM_STEPS", "NUM_EPISODES"]},
        "per_seed_results": all_results,
        "aggregate": aggregate,
        "timestamp": datetime.now().isoformat(),
    }

4b. 在 experiment_runner.py 中添加 --init-baseline 模式
elif args.mode == "init-baseline":
    from evaluator import evaluate_random_policy, EVAL_CONFIG
    print("Evaluating random policy as initial baseline...")
    result = evaluate_random_policy(dict(EVAL_CONFIG))
    metrics = result["aggregate"]
    save_champion({
        "experiment_id": 0,
        "description": "Random policy baseline (untrained network). Reference: quaternion baseline trained in heading_pitch_V env achieved ~20deg theta. This full_domain task is harder.",
        "config_snapshot": load_config(),
        "checkpoint_path": None,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "status": "baseline",
    })
    print(f"Initial baseline established:")
    print(f"  theta_deg = {metrics.get('mean_theta_deg', '?')}deg")
    print(f"  delta_vt = {metrics.get('mean_delta_vt', '?')}")
    print(f"  crash_rate = {metrics.get('mean_crash_rate', '?')}")

4c. 运行 init-baseline
完成代码修改后，运行：

cd /home/dqy/aeroplanax/new/20251215最新代码库/RL_autotuner
CUDA_VISIBLE_DEVICES=1 python experiment_runner.py --mode init-baseline
用 GPU 1（GPU 0 可能在跑别的实验）。把实际结果写入 champion_meta.json。

任务 5：修复早停逻辑
当前 run_training() 的早停（第 196-213 行）搜索 update[=:\s]+(\d+)， 但训练脚本实际输出的是 env_step=XXXXXXXX，没有 update= 字段。

修改：
    1.将 m_update = re.search(r"update[=:\s]+(\d+)", line) 改为：
        m_update = re.search(r"env_step[=:\s]+(\d+)", line)

    2.早停判断需要适配 env_step 而不是 update 数。当前 EARLY_STOP_MIN_UPDATES=80 对应 update 次 数，而 env_step 每次增加 NUM_ENVS * NUM_STEPS = 1000 * 1000 = 1_000_000。所以 80 个 update = 80M env_steps。修改为：
    EARLY_STOP_MIN_STEPS = 80_000_000   # 80M steps before checking early stop
    EARLY_STOP_WINDOW = 50              # check over last 50 theta records
    EARLY_STOP_THETA_TOL = 3.0          # minimum improvement in degrees
    
    并修改判断为 if env_step >= EARLY_STOP_MIN_STEPS and len(theta_history) >= EARLY_STOP_WINDOW:

    3.确认 theta_deg 的正则 theta_deg[=:\s]+([\d.]+) 能匹配实际日志格式 theta_deg=85.3。看起来可以匹配，无需改动。

任务 6：新增 manual-auto 模式
在 experiment_runner.py 中新增 --mode manual-auto：
def run_manual_auto_mode(budget: int, description: str = ""):
    """Non-interactive single experiment: read current reward_config.json, train, evaluate, keep/discard.
    Designed to be called by Claude Code CLI agent.
    """
    config = load_config()
    if not description:
        description = "manual-auto experiment"
    result = run_experiment(config, budget, description)
    print(f"\nResult: {result['status']} (experiment #{result['experiment_id']})")
    return result

在 argparse 中添加：
parser.add_argument("--mode", choices=["manual", "auto", "manual-auto", "dry-run", "init-baseline"], default="manual")
parser.add_argument("--description", type=str, default="", help="Experiment description (for manual-auto mode)")

在 main 分支中添加：
elif args.mode == "manual-auto":
    run_manual_auto_mode(int(args.budget), args.description)


任务 7：改造 program.md 为 Claude Code CLI Agent SOP
用 autoresearch 的 program.md 风格重写 RL_autotuner/program.md。请先阅读 /home/dqy/aeroplanax/new/20251215最新代码库/autoresearch/autoresearch/program.md 理解其风格。

必须包含的核心内容：

7a. 身份
你是一个 RL reward 调优 agent。你的唯一目标是最小化 mean_theta_deg（平均姿态跟踪误差角度）。

7b. 实验循环（LOOP FOREVER）
LOOP FOREVER:
1. 读取当前状态：
   - cat champion/champion_meta.json  （当前最佳）
   - cat reward_config.json           （当前配置）
   - tail -20 results.jsonl           （最近实验历史）

2. 分析历史，提出一个修改假设（只改 reward_config.json 中的 1-2 个数值参数）

3. 编辑 reward_config.json

4. git commit -m "experiment: <简短描述>"

5. 运行实验：
   CUDA_VISIBLE_DEVICES=1 python experiment_runner.py --mode manual-auto --budget 1e8 --description "<同样描述>" > run.log 2>&1

6. 检查结果：
   tail -5 results.jsonl
   grep "theta_deg" run.log | tail -5

7. 判断结果：
   - 如果最后一行 status="keep" -> 新 champion！继续下一轮。
   - 如果 status="discard" -> git reset --hard HEAD~1 回到上一个 commit。
   - 如果 status="crash" -> tail -50 run.log 查看错误，修复后重试。

8. 回到步骤 1。


7c. 约束
Phase 1 约束：

你只能修改 reward_config.json 中的数值
每次最多改 2 个参数
你不能修改任何 .py 文件
你不能修改评估协议
你必须在 git commit message 和 --description 中说明你的假设
7d. 可用信息
champion/champion_meta.json：当前最佳的配置和评估结果
results.jsonl：所有实验历史（不提交到 git）
run.log：最近一次训练的完整日志
program.md：本文件（你的操作规范）
7e. NEVER STOP 原则
NEVER STOP：实验循环开始后，不要暂停询问人类是否继续。人类可能不在电脑前，期望你持续工作直到被手动中断。如果你用尽了想法，重新阅读 program.md 中的策略建议、已知失败模式，尝试组合之前接近成功的修改，或尝试更激进的参数变化。循环一直运行，直到人类中断你。每个实验大约需要 30-60 分钟。如果你运行 8 小时，可以完成 8-16 个实验。

7f. 策略建议和已知失败模式
保留现有 program.md 中的 "Key Parameters to Tune"、"Known Failure Modes"、"Strategy Suggestions" 部分，它们很有价值。

7g. 参考信息
在 program.md 中添加一段参考：用户有一个在 heading_pitch_V_quaternion_version_add_full_roll 环境下训好的四元数 baseline（obs=16D, GRU=128），theta_deg 约 20 度。该环境的目标范围是 heading ±90 度, pitch ±30 度, roll ±90 度, speed 120-360。这说明在类似任务下 theta_deg 20 度是可达的，可以借鉴这个四元数版本的env、reward、terminal condition设计，但 full_domain 环境有 8 级课程和更大的目标范围，所以更难。agent 的长期目标应该是让 full_domain 训练达到类似或更好的跟踪精度。

任务 8：最终验证
完成所有代码修改后：
    1. 运行 dry-run 验证配置正确：
    cd /home/dqy/aeroplanax/new/20251215最新代码库/RL_autotuner
    python experiment_runner.py --mode dry-run

    2.运行 init-baseline 建立初始 champion（任务 4c）：
    CUDA_VISIBLE_DEVICES=1 python experiment_runner.py --mode init-baseline

    3.git commit 所有改动：
    cd /home/dqy/aeroplanax/new/20251215最新代码库
    git add RL_autotuner/
    git commit -m "autotuner: fix evaluator, add git workflow, manual-auto mode, init-baseline"

    4.确认 champion_meta.json 已被实际 metrics 填充（不再是 null）

执行顺序
严格按以下顺序执行：

任务 1（git init）— 这样后续所有改动都有 git 保护
任务 2（evaluator.py 修复）
任务 3（experiment_runner.py 接入正式评估）
任务 5（早停修复）
任务 6（manual-auto 模式）
任务 7（program.md 改造）
任务 8 的第 1 步（dry-run 验证）
任务 4（init-baseline，需要实际跑 GPU）
任务 8 的第 3-4 步（git commit 并确认）


参考文件清单
在开始之前，请先阅读以下文件理解上下文：

autoresearch/autoresearch/program.md — autoresearch 的 agent SOP，重点学习 LOOP FOREVER 和 keep/discard via git
RL_autotuner/program.md — 当前版本，需要改造
RL_autotuner/experiment_runner.py — 主编排器
RL_autotuner/evaluator.py — 评估器（需修复）
RL_autotuner/config_patcher.py — config 写入机制
Planax/envs/aeroplanax_full_domain_maneuver.py — 环境（FROZEN，只读，理解 state 结构用）
Planax/train_full_domain_maneuver_v3.py — 训练脚本（FROZEN，只读，理解日志格式和配置用）
RL_autotuner/RL自动化调参框架搭建指导.md — 之前的审计报告，有详细的问题分析