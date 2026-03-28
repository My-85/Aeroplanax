  #!/bin/bash                                                                                         
  export CUDA_VISIBLE_DEVICES=0                                                                       
  export XLA_PYTHON_MEM_FRACTION=0.90                                                                 
                                                                                                      
  SCRIPT_DIR="/home/dqy/aeroplanax/new/20251215最新代码库/RL_autotuner"                               
  cd $SCRIPT_DIR                                                                                      
                                                                                                      
  echo "=========================================="                                                   
  echo "Batch Evaluation: 3 Checkpoints"                                                              
  echo "=========================================="                                                   
                                                                                                      
  echo ""                                                                                             
  echo "[1/3] Evaluating Quat Baseline (epoch_1000)..."                                               
  python evaluator.py --checkpoint "/home/dqy/aeroplanax/new/20251215最新代码库/results/baseline（四元数版本）/checkpoints/checkpoint_epoch_1000" --waypoint --output eval_results/quat_baseline_epoch1000.json                                                                                                                              
                                                                                                                                                                         
  echo ""                                                                                                                                                                
  echo "[2/3] Evaluating Autotuned_1350..."                                                                                                                              
  python evaluator.py --checkpoint "/home/dqy/aeroplanax/new/20251215最新代码库/Planax/results/heading_pitch_V_discrete_rnn_2026-03-20-19-38/checkpoints/checkpoint_epoch_1350" --waypoint --output eval_results/autotuned_1350.json                                                                                                                                       
                                                                                                                                                                         
  echo ""                                                                                                                                                                
  echo "[3/3] Evaluating Champion #59..."                                                                                                                                
  python evaluator.py --checkpoint "/home/dqy/aeroplanax/new/20251215最新代码库/Planax/results/heading_pitch_V_discrete_rnn_2026-03-24-08-55/checkpoints/checkpoint_epoch_1575" --waypoint --output eval_results/champion_59.json

  echo ""                                                    
  echo "==========================================" 
  echo "All evaluations complete!"                 
  echo "Results saved to: $SCRIPT_DIR/eval_results/"
  echo "=========================================="
