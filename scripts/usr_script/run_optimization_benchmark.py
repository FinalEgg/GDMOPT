
import sys
import os
import torch
import numpy as np

# 将项目根目录添加到 python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入配置和训练脚本
from config.run_config import RunConfig
from config.env_config import EnvConfig
from config.mapping_config import MappingConfig
from scripts.train_model import train_model

def run_optimization_benchmark():
    """
    Continuous run of TD3, DDPG, SAC on Optimization Task
    """
    
    # 设置基础配置
    RunConfig.EXPERIMENT_ID = 'exp_optimization_benchmark'
    
    # 设置环境参数
    EnvConfig.STEPS_PER_EPISODE = 200 # 优化任务通常步数较少
    EnvConfig.OPT_DIM = 2 # 2维优化问题
    EnvConfig.OPT_BOUNDS = 5.0 # [-5, 5]
    
    # 共同训练参数
    # RunConfig.EPOCH = 10 # Commented out to use value from run_config.py or set higher here
    # RunConfig.STEP_PER_EPOCH = 1000
    
    # Override for benchmark (Optional: set specific benchmark length)
    RunConfig.EPOCH = 50 
    RunConfig.STEP_PER_EPOCH = 2000

    RunConfig.BATCH_SIZE = 64
    RunConfig.BUFFER_SIZE = 20000
    RunConfig.BACKBONE = 'mlp'
    RunConfig.ACTION_MODE = 'pure_power' # 虽然叫 pure_power，但在 OptimizationEnv 中只用了值
    RunConfig.NUM_TRAIN_ENVS = 4 # 并行环境
    RunConfig.NUM_TEST_ENVS = 4
    RunConfig.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    algos = ['td3', 'ddpg', 'sac']
    
    for algo in algos:
        print(f"\n{'='*40}")
        print(f"Running Benchmark for Algorithm: {algo.upper()}")
        print(f"{'='*40}\n")
        
        # 修改算法配置
        RunConfig.ALGO = algo
        
        # Optional: Add a tag to identify this benchmark run
        # 可选: 添加标签以识别此次基准测试运行
        RunConfig.RUN_ID = f"benchmark_dim{EnvConfig.OPT_DIM}"
        
        # 运行训练
        try:
            # call train_model without arguments -> New Run with Auto Path Generation
            # 调用 train_model 不带参数 -> 使用自动路径生成的新运行
            train_model(resume_from_log=None)
            
        except Exception as e:
            print(f"Error running {algo}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_optimization_benchmark()
