深度审计报告：Planax 自动化 RL 实验系统                                                                   
                                                                                                            
  ---                                                                                                       
  A. autoresearch 的本质设计                                                                                
                                                                                                            
  最小自动实验闭环                                                                                          
                                                                                                            
  LOOP:                                                                                                     
    1. git commit (一个想法 = 一个 commit)                                                                  
    2. uv run train.py > run.log  (固定 5 分钟 wall-clock)                                                  
    3. grep val_bpb run.log       (提取唯一指标)                                                            
    4. if improved → keep commit (前进)                                                                     
       else       → git reset   (丢弃)                                                                      
    5. 追加到 results.tsv (不提交到 git)                                                                    
                                                                                                            
  固定死的 vs 允许改的                                                                                      
                                                                                                            
  ┌──────────────────────────────────────────┬──────────────────────────────────────────┐                   
  │                  固定死                  │                  允许改                  │                   
  ├──────────────────────────────────────────┼──────────────────────────────────────────┤                   
  │ prepare.py（数据、tokenizer、eval 函数） │ train.py（模型、优化器、超参、训练循环） │
  ├──────────────────────────────────────────┼──────────────────────────────────────────┤
  │ 时间预算 = 300 秒 wall-clock             │ 架构选择（层数、head 数、MLP 类型）      │
  ├──────────────────────────────────────────┼──────────────────────────────────────────┤
  │ 评估指标 = val_bpb（bits per byte）      │ batch size、LR、schedule                 │
  ├──────────────────────────────────────────┼──────────────────────────────────────────┤
  │ 数据集分片 & 验证集                      │ 所有训练细节                             │
  ├──────────────────────────────────────────┼──────────────────────────────────────────┤
  │ 评估 token 数 = 40×524288                │ ——                                       │
  └──────────────────────────────────────────┴──────────────────────────────────────────┘

  为什么比"让 LLM 自动改代码"更稳                     

  1. 单文件修改：只改 train.py，不可能产生跨文件不一致                                                      
  2. 不可变评估：eval 在 prepare.py 里，agent 改不了→消除 Goodhart 效应                                     
  3. 固定预算：5 分钟 wall-clock，所有实验直接可比                                                          
  4. 原子回滚：git reset 一步撤销，无残留                                                                   
  5. 快速迭代：每小时跑 ~12 个实验，统计信号强                                                              

  真正值得迁移到 RL 的思想

    1. Champion/Challenger 机制：keep/discard 二元判定，基于单一不可变指标                                    
  2. results.tsv 独立于 git：实验记录不被 rollback 破坏                                                     
  3. program.md 研究规范：agent 的行为规则是文本协议，不是硬编码                                            
  4. 固定 eval 预算：所有实验在相同条件下评估                                                               

  不能直接照搬到 RL 的部分                            

  1. 5 分钟 wall-clock → RL 训练需要小时级别，不能用 wall-clock 做预算（编译开销占比高）
  2. val_bpb 单一指标 → RL 需要多维指标（theta_deg、crash_count、on_target_steps），不能压到一个数          
  3. 从头训练 → RL 太贵，需要 checkpoint 续训，但 autoresearch 每次从零开始
  4. 单文件 → RL 的 reward/env/termination/training 天然分离，不可能压到一个文件

  ---                                                 
  B. aircraft-rl-tuner 审计（按严重程度排序）                                                               

  CRITICAL-1: 没有 Champion 基线——无法判断改动是否真的改善了                                                

  位置: auto_train_loop.py:77-115 (check_pass)                                                              

  check_pass() 对比的是固定绝对阈值（success_rate>0.8、crash_count<1.0），而不是"比上一轮好"。当 Claude 修改
 reward 后，episodic_return
  的量纲完全变了（tracking_scale 从 1→3，crash_penalty 从 -1→-5），但阈值不变。系统无法区分"agent 真的变好了
"和"reward 数值变了"。     

  CRITICAL-2: 编辑范围无约束——一次改动可能同时改 4 个文件的 20 处                                           

  位置: auto_train_loop.py:214-447                    

  Claude 一次迭代可以输出 20 个 edit，分散在 reward、env、termination、training 四个文件中。当改动失败时，根
本无法判断是哪个改动导致的。autoresearch              
  只允许改一个文件，这是它稳定的核心原因。                                                                  

  CRITICAL-3: Dry-run 只检测语法错误，不检测 reward 逻辑错误                                                

  位置: auto_train_loop.py:642-705                    

  2M 步 dry-run 只跑 2 个 update。无法检测：                                                                
  - reward 在特定 theta 值下 NaN                      
  - 策略发散（需要 100M+ 步才显现）
  - 课程逻辑卡死（2M 步不可能触发 curriculum advancement）

CRITICAL-4: 没有 keep/discard 机制——所有改动都直接推进                                                    

  位置: auto_train_loop.py:1007-1062                  

  autoresearch 的核心：改动必须证明比 champion 更好才保留。但 aircraft-rl-tuner 的逻辑是：Claude            
  改了就用，用了就训，训完就评。即使评估结果更差，也不会回滚到之前最好的代码版本——只回滚到本轮修改前（可能 $
身就不好）。               

  CRITICAL-5: Reward 变化导致跨轮指标不可比                                                                 

  位置: full_domain_reward.py REWARD_CONFIG 多次修改                                                        

  v14: crash_penalty=-5.0, alive_bonus=0.005, tracking_scale=1.0                                            
  v15: crash_penalty=-1.0, alive_bonus=0.15, tracking_scale=1.0                                             
  v16: crash_penalty=-5.0, alive_bonus=0.005, tracking_scale=3.0                                            

  三个版本的 episodic_return 量纲完全不同，但 iteration_log 里的 mean_total_reward 直接混在一起比较。       

  HIGH-1: param_registry.py 存在但完全没被使用                                                              

  位置: aircraft-rl-tuner/scripts/param_registry.py                                                         

  定义了 51 个参数的合法范围，有 validate_changes() 函数。但 auto_train_loop.py 从未 import 它。Claude 不知 
道参数范围，可以随意改。   

  HIGH-2: 结构化编辑记录缺失                          

  位置: auto_train_loop.py:1052-1060                  

  iteration_log 只记录 code_changes_summary（一句话），不记录具体的 old→new。无法检测同一参数在多轮间的振荡 
（如 crash_penalty: -5→-1→-5→-1）。                   

  HIGH-3: 早停可能在正常探索阶段误触发                                                                      

  位置: auto_train_loop.py:726-755                    

  EARLY_STOP_MIN_UPDATES=100 意味着 100M 步后开始检查。但 RL 训练前 100-200M 步可能仍在"policy 随机探索"阶段
，theta 自然波动大。2°     
  的阈值在这个阶段可能产生误报。                      

  HIGH-4: Checkpoint 恢复不验证兼容性                 

  位置: auto_train_loop.py:1018-1029


  如果 Claude 修改了 observation space（比如添加一个维度），旧 checkpoint 的网络输入维度不匹配。当前 fallbac
k 是从头训练，丢失所有之前的训练进度。                

  ---                                                 
  C. 当前系统"方向对不对"                             

  明确回答：aircraft-rl-tuner 的方向是错的                                                                  

  不是"代码有 bug"的问题，是实验制度设计有根本性缺陷。具体来说：                                            

  最阻碍成功的 3 个根因                               

  根因 1：没有固定评估协议                            

  每次 Claude 改 reward，episodic_return 的量纲就变了。系统用这个变量来判断"好不好"，但这个变量跨轮不可比。a
utoresearch 用不可变的     
  val_bpb，我们需要一个与 reward 配置无关的评估指标（比如 theta_deg、on_target_steps、crash_count，这些是物 
理量，不随 reward 变化）。 

  根因 2：没有 champion 基线 + keep/discard 机制                                                            

  系统永远在"向前走"，没有"回头看"。应该维护一个 champion checkpoint+代码快照，所有新实验必须在同一评估协议 
下证明比 champion 好，否则丢弃。                      

  根因 3：agent 的编辑范围太大                        

  一次可以改 4 个文件的 20 个地方。失败后无法归因。应该像 autoresearch 一样，第一阶段只允许改 REWARD_CONFIG 
字典里的数值。             

  可以救吗？                                          

  部分可以救。应该保留的部分：                        
  - evaluate_maneuver.py 的评估框架（waypoint 测试逻辑）                                                    
  - auto_train_loop.py 的 SDK 调用和流式输出基础设施                                                        
  - 早停机制（但需要调参）                            
  - backup/rollback 机制（需要加 champion 概念）                                                            

  不应该直接在上面硬修的原因：                        
  核心问题不是代码 bug，而是实验制度缺失。在没有固定评估指标和 champion
  机制的框架上堆更多代码，只会让问题更复杂。需要先建立实验制度，再复用基础设施。                  [210/2313]

  ---                                                 
  D. Planax 版 autoresearch 最小可落地蓝图                                                                  

  1. 文件/模块编辑权限                                

  ┌───────────────────────────────────────────────────────────────────┬────────────────────────┬───────────$
──────────────────┐        
  │                               文件                                │          权限          │            
 原因             │        
  ├───────────────────────────────────────────────────────────────────┼────────────────────────┼────────────
──────────────────┤        
  │ envs/aeroplanax_full_domain_maneuver.py                           │ FROZEN                 │ 环境逻辑、 
观测空间、课程逻辑 │       
  ├───────────────────────────────────────────────────────────────────┼────────────────────────┼────────────
──────────────────┤        
  │ envs/termination_conditions/*.py                                  │ FROZEN                 │ 成功/失败判
定                │        
  ├───────────────────────────────────────────────────────────────────┼────────────────────────┼────────────
──────────────────┤        
  │ envs/reward_functions/full_domain_reward.py 的 REWARD_CONFIG 字典 │ Phase 1 唯一可改       │ 数值参数调 
优                 │       
  ├───────────────────────────────────────────────────────────────────┼────────────────────────┼────────────
──────────────────┤        
  │ envs/reward_functions/full_domain_reward.py 的计算逻辑            │ Phase 2 可改           │ 需要 champi
on 机制保护       │        
  ├───────────────────────────────────────────────────────────────────┼────────────────────────┼────────────
──────────────────┤        
  │ train_full_domain_maneuver_v3.py                                  │ FROZEN（训练超参除外） │ 训练循环不 
动                 │       
  └───────────────────────────────────────────────────────────────────┴────────────────────────┴────────────
──────────────────┘        

  2. 第一阶段：只允许改 REWARD_CONFIG 数值                                                                  

  将 REWARD_CONFIG 提取为独立 JSON：                  
  // reward_config.json  ← 唯一可编辑文件                                                                   
  {                                                   
    "tracking_scale": 3.0,                            
    "gaussian_scale_coarse_deg": 80.0,                                                                      
    "progress_coeff": 3.0,                            
    "alive_bonus": 0.005,                             
    "crash_penalty": -5.0,                            
    ...
  }                                                                                               [165/2313]

  agent 只能提交一个新的 reward_config.json，训练脚本读取它。                                               

  3. 固定实验预算                                     

  FIXED_BUDGET = {                                    
      "timesteps": 2e8,        # 200M steps per experiment (固定，不可改)                                   
      "eval_episodes": 20,     # 评估回合数（固定）                                                         
      "eval_max_steps": 2000,  # 每回合最大步数（固定）                                                     
  }                                                   

  所有实验使用完全相同的训练和评估预算。                                                                    

  4. 跨轮唯一主比较指标                               

  PRIMARY_METRICS = {                                 
      "theta_deg_mean": "越低越好",      # 主指标：平均姿态误差                                             
      "crash_count_mean": "越低越好",     # 辅指标：坠毁次数                                                
      "on_target_ratio": "越高越好",      # 辅指标：on-target 比例                                          
  }                                                   

  关键：这些是物理量，不随 reward 配置变化。theta_deg=30° 在 v14、v15、v16 下都是同一个意思。

  不使用 episodic_return 作为比较指标（它随 reward 变化）。                                                 

  5. results.jsonl 记录字段                           

  {                                                   
    "experiment_id": "exp_007",                       
    "timestamp": "2026-03-18T15:30:00",                                                                     
    "reward_config_hash": "a3b2c1d0",                 
    "reward_config": {"tracking_scale": 3.0, ...},                                                          
    "train_budget_steps": 2e8,                        
    "eval_episodes": 20,                              
    "metrics": {                                      
      "theta_deg_mean": 45.2,                         
      "theta_deg_std": 12.3,                          
      "crash_count_mean": 23.5,                       
      "on_target_ratio": 0.02,                        
      "curriculum_level_mean": 0.1,                   
      "delta_vt_mean": 15.3                           
    },                                                
    "training_curve": {                               
      "theta_at_25pct": 78.0,
      "theta_at_50pct": 55.0,
      "theta_at_75pct": 48.0,                         
      "theta_at_100pct": 45.2                         
    },                                                
    "status": "keep",                                 
    "vs_champion": "+12.5° improvement",                                                                    
    "description": "increase tracking_scale from 2.0 to 3.0",                                               
    "parent_experiment": "exp_006"                    
  }                                                   

  6. Champion / Challenger / Discard 机制                                                                   

  CHAMPION: 当前最佳 experiment（theta_deg_mean 最低）                                                      
            包含：reward_config.json + checkpoint + eval_report                                             

  每轮实验：                                          
    1. 从 champion 的 reward_config.json 出发，修改一处参数                                                 
    2. 从头训练 200M 步（不续训，消除 checkpoint 偏见）                                                     
    3. 用固定评估协议跑 20 episode                    
    4. IF theta_deg_mean < champion.theta_deg_mean:                                                         
         status = "keep", 新 champion                 
       ELSE:                                          
         status = "discard", 回滚到 champion                                                                
    5. 记录到 results.jsonl                           

  7. 何时从"参数搜索"升级到"代码级 redesign"                                                                

  IF 连续 10 次参数修改都被 discard:                  
    → 判定为"参数空间搜索已饱和"                      
    → 升级到 Phase 2：允许修改 reward 计算逻辑                                                              
    → 但仍然使用 champion 机制保护                    

  IF 连续 5 次代码修改都被 discard:                   
    → 判定为"当前 reward 架构已饱和"                  
    → 升级到 Phase 3：允许修改 env 参数                                                                     
    → 或者 escalate 给人类                            

  8. Evaluator 与训练解耦                             

  # evaluator.py — 完全独立，不 import reward 或 training                                                   
  def evaluate(checkpoint_path: str, config: EvalConfig) -> EvalReport:                                     
      """                                             
      纯推理评估。不依赖 reward 函数。                                                                      
      直接测量物理量：theta_deg, delta_vt, crash_count, on_target_steps                                     
      """
      # 加载 network params（partial_restore, 跳过 opt_state）                                     
      # 用固定 waypoints 跑 N episodes                                                                      
      # 返回 EvalReport（只含物理量，不含 reward 值）                                                       

  9. 如何避免 reward 变化导致跨轮不可比                                                                     

  核心原则：比较指标必须是与 reward 配置无关的物理量。                                                      

  - theta_deg：geodesic angle，纯几何量                                                                     
  - crash_count：坠毁次数，纯环境事件                 
  - on_target_steps：theta<10° 的步数，纯几何判定                                                           
  - delta_vt：速度误差，纯物理量                      

  绝不用 episodic_return 做跨轮比较。它可以用来做轮内训练曲线分析，但不能用来比较不同 reward 配置。         

  10. program.md 研究规范                             

  # Planax RL Research Protocol                       

  ## 你是谁                                           
  你是一个 RL reward 调优 agent。你的目标是最小化 theta_deg_mean。                                          

  ## 实验循环                                         
  1. 读取 champion/reward_config.json                 
  2. 提出一个修改假设（只改一个参数，或最多两个相关参数）                                                   
  3. 写出新的 reward_config.json                      
  4. 系统自动训练 200M 步 + 评估 20 episodes                                                                
  5. 比较 theta_deg_mean：如果改善 → keep，否则 → discard                                                   

  ## 约束                                             
  - 你只能修改 reward_config.json 中的数值                                                                  
  - 每次最多改 2 个参数                               
  - 你必须在 description 中说明你的假设                                                                     
  - 你不能修改任何 .py 文件（Phase 1）                                                                      
  - 你不能修改评估协议                                

  ## 可用信息                                         
  - champion/eval_report.json：当前最佳的评估结果                                                           
  - results.jsonl：所有实验历史                       
  - training_curves/：每次训练的 theta_deg 曲线                                                             

  ## 策略建议                                         
  - 先做大步长搜索（2x/0.5x），找到有效方向                                                                 
  - 再做精细搜索（±10%），锁定最优值                  
  - 不要同时改多个参数
  - 如果连续 3 次 discard，换一个参数                                                              

  ---                                                 
  E. 最小迁移路径（3 阶段）                           

  Phase 1：最小可运行版本（~2 天工作量）                                                                    

  目标：固定预算 + 固定评估 + champion 机制                                                                 

  1. 将 REWARD_CONFIG 提取到 reward_config.json，训练脚本读取它                                             
  2. 写一个 experiment_runner.py（替代 auto_train_loop.py 的核心循环）：                                    
    - 固定训练 200M 步                                
    - 固定评估 20 episodes                            
    - 只记录物理量指标（theta_deg, crash_count, on_target_steps）                                           
  3. 实现 champion 目录：champion/reward_config.json + champion/checkpoint/ + champion/eval_report.json     
  4. 实现 keep/discard：新实验的 theta_deg < champion 的 theta_deg → keep，否则 discard
  5. 写 results.jsonl 记录每次实验                    
  6. 写 program.md 研究规范                           

  此阶段 Claude 只能改 reward_config.json 中的数值。                                                        

  Phase 2：稳定自动实验版本（~3 天工作量）                                                                  

  目标：可靠自动循环 + 智能搜索                       

  1. 集成 Claude SDK 调用（复用 auto_train_loop.py 的 streaming 基础设施）                                  
  2. Claude 读取 results.jsonl + program.md + champion/eval_report.json，输出新的 reward_config.json        
  3. 添加"禁止重复失败修改"逻辑：从 results.jsonl 提取 discard 记录，传入 prompt
  4. 添加累计预算追踪                                 
  5. 添加参数范围校验（复用 param_registry.py）                                                             
  6. 添加"参数空间饱和"检测（连续 N 次 discard → 升级请求）                                                 
  7. 解耦 evaluator：评估脚本不 import reward 函数，只用 network 做推理                                     

  Phase 3：允许代码级 redesign（~5 天工作量）                                                               

  目标：突破参数空间限制                              

  1. 当 Phase 2 检测到参数空间饱和（连续 10 次 discard），进入 Phase 3                                      
  2. 允许 Claude 修改 full_domain_reward.py 的计算逻辑（但仍然 FROZEN env 和 termination）
  3. 代码修改必须通过 dry-run + synthetic trajectory test（在 theta=0°, 30°, 60°, 90° 处验证 reward 正确性）
  4. Champion 机制仍然生效：代码改动必须产生比 champion 更低的 theta_deg                                    
  5. 代码版本通过 git commit 管理，每次改动是一个 commit                                                    

  ---                                                 
  核心结论
  当前最值得先改的确实是实验制度，而不是 reward 本身。

  过去一周，我们在 reward 上做了 v10→v11→v12→v13→v14→v15→v16 共 7 个版本的修改，没有一个版本让 theta 降到
  30° 以下。原因不是"还没找到正确的 reward 参数"，而是：

  1. 每次改完 reward 后没有可靠的方式判断"这次改动是否真的帮助了"
  2. 改动范围太大（一次改 5-6 个参数），无法归因
  3. 没有 champion 基线，每轮都从混乱状态出发

  建议立即冻结 Planax 的所有 .py 文件，先建立 Phase 1 的实验制度（固定预算 + 固定评估 + champion +
  keep/discard），然后在这个制度下系统地搜索 REWARD_CONFIG 的参数空间。