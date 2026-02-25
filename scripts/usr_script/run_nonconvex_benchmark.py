
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

def run_nonconvex_benchmark():
    """
    Continuous run of TD3, DDPG, SAC on Non-Convex Optimization Task (Rastrigin)
    """
    
    # 设置基础配置: 指向 Non-Convex 实验配置 (对应 MappingConfig 中的 exp_optimization_nonconvex -> env_id: rastrigin)
    RunConfig.EXPERIMENT_ID = 'exp_optimization_nonconvex'
    
    # 设置环境参数
    # Rastrigin 是多峰函数，容易陷入局部最优，具有挑战性
    EnvConfig.STEPS_PER_EPISODE = 200 
    EnvConfig.OPT_DIM = 2 # 2维 Rastrigin 函数
    EnvConfig.OPT_BOUNDS = 5.12 # Rastrigin 函数的标准定义域通常是 [-5.12, 5.12]
    
    # 共同训练参数
    # 非凸优化可能需要更多步数来收敛
    RunConfig.EPOCH = 50
    RunConfig.STEP_PER_EPOCH = 2000
    RunConfig.BATCH_SIZE = 64
    RunConfig.BUFFER_SIZE = 50000 # 稍微增大 Buffer
    RunConfig.BACKBONE = 'mlp'
    RunConfig.ACTION_MODE = 'pure_power'
    RunConfig.NUM_TRAIN_ENVS = 8 # 增加并行环境以增强探索
    RunConfig.NUM_TEST_ENVS = 4
    RunConfig.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 增加探索噪声，帮助跳出局部最优
    RunConfig.EXPLORATION_NOISE = 0.2

    algos = ['td3', 'ddpg', 'sac']
    
    for algo in algos:
        print(f"\n{'='*40}")
        print(f"Running Non-Convex Benchmark (Rastrigin) for Algorithm: {algo.upper()}")
        print(f"{'='*40}\n")
        
        # 修改算法配置
        RunConfig.ALGO = algo
        
        # 使用特定的标签
        RunConfig.RUN_ID = f"rastrigin_{algo}_dim{EnvConfig.OPT_DIM}"
        
        # 运行训练
        try:
            # resume_from_log=None 表示新起一个实验，会自动生成带时间戳的目录
            train_model(resume_from_log=None)
        except Exception as e:
            print(f"Error running {algo}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_nonconvex_benchmark()
