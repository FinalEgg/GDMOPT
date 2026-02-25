
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

def run_pendulum_benchmark():
    """
    Benchmark TD3, DDPG, SAC on Classic Control Task (Inverted Pendulum)
    """
    
    # 设置基础配置
    RunConfig.EXPERIMENT_ID = 'exp_classic_pendulum'
    
    # 设置环境参数
    # Pendulum 标准步长为 200，任务是尽快将摆锤甩起并保持直立
    EnvConfig.STEPS_PER_EPISODE = 200 
    
    # 共同训练参数
    RunConfig.EPOCH = 20
    RunConfig.STEP_PER_EPOCH = 5000 # 增加每轮交互步数，因为控制任务需要较多样本
    RunConfig.BATCH_SIZE = 128
    RunConfig.BUFFER_SIZE = 50000 
    RunConfig.BACKBONE = 'mlp'
    RunConfig.ACTION_MODE = 'pure_power' # 虽然名字叫 pure_power，但在 PendulumEnv 中被解释为 raw continuous action
    RunConfig.NUM_TRAIN_ENVS = 8
    RunConfig.NUM_TEST_ENVS = 10
    RunConfig.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 调整探索噪声 (Pendulum 动作范围大，给予适度噪声)
    RunConfig.EXPLORATION_NOISE = 0.1

    # 重要：设置 Gamma > 0 来启用多步奖励累积
    # Pendulum 是典型的连续控制问题，当前动作严重影响未来状态
    RunConfig.GAMMA = 0.99 

    algos = ['td3', 'ddpg', 'sac']
    
    for algo in algos:
        print(f"\n{'='*40}")
        print(f"Running Pendulum Benchmark for Algorithm: {algo.upper()}")
        print(f"{'='*40}\n")
        
        # 修改算法配置
        RunConfig.ALGO = algo
        RunConfig.RUN_ID = f"pendulum_{algo}_gamma0.99"
        
        # 运行训练
        try:
            train_model(resume_from_log=None)
        except Exception as e:
            print(f"Error running {algo}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_pendulum_benchmark()
