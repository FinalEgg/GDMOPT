"""
Model Evaluation Script.
模型评估脚本。

This script loads a trained model checkpoint and evaluates its performance
in the environment. It reports mean reward, standard deviation, and episode lengths.
该脚本加载已训练的模型检查点并在环境中评估其性能。它报告平均奖励、标准差和回合长度。
"""

import sys
import os
import torch
import numpy as np
from tianshou.data import Collector, Batch
from tianshou.env import DummyVectorEnv

# Add project root to path
# 将项目根目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.setup import make_env_instance, make_policy
from config.env_config import EnvConfig
from config.run_config import RunConfig

def evaluate_model(checkpoint_path: str = None):
    """
    Run model evaluation.
    运行模型评估。

    Args:
    -----
    checkpoint_path : str, optional
        Path to the checkpoint file (.pth) to load.
        要加载的检查点文件 (.pth) 的路径。

    Processes:
    1. Sets up the environment (Test Mode).
    2. Initializes the policy.
    3. Loads model weights from checkpoint (if provided).
    4. Runs evaluation using Tianshou Collector.
    5. Prints performance metrics.

    处理流程:
    1. 设置环境 (测试模式)。
    2. 初始化策略。
    3. 从检查点加载模型权重 (如果提供)。
    4. 使用 Tianshou Collector 运行评估。
    5. 打印性能指标。
    """
    print("=== Model Inference/Evaluation Script ===")
    print("=== 模型推理/评估脚本 ===")
    
    # 1. Setup Env (Test Mode)
    # We use a distinct seed for testing
    # 1. 设置环境 (测试模式)
    # 我们使用不同的种子进行测试
    env = make_env_instance(RunConfig, EnvConfig, seed=100)
    # Tianshou needs VectorEnv
    # Tianshou 需要 VectorEnv
    test_env = DummyVectorEnv([lambda: env])
    
    # 2. Setup Policy
    # 2. 设置策略
    policy = make_policy(RunConfig, env)
    
    # 3. Load Checkpoint
    # 3. 加载检查点
    if checkpoint_path:
        print(f"Loading checkpoint from (加载检查点): {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=RunConfig.DEVICE)
            policy.load_state_dict(checkpoint['policy'])
            print("Checkpoint loaded successfully. (检查点加载成功)")
        except Exception as e:
            print(f"Error loading checkpoint (加载检查点错误): {e}")
            return
    else:
        print("[WARNING] No checkpoint provided. Running with random weights.")
        print("[警告] 未提供检查点。使用随机权重运行。")
        
    policy.eval()

    # 4. Inference / Evaluation
    # 4. 推理 / 评估
    print("\n--- Starting Evaluation (开始评估) ---")
    # Deterministic evaluation (no random exploration noise)
    # 确定性评估 (无随机探索噪声)
    collector = Collector(policy, test_env, exploration_noise=False)
    
    # Collect n episodes
    # 收集 n 个回合
    n_episodes = 10
    result = collector.collect(n_episode=n_episodes)
    
    print("\n--- Evaluation Results (评估结果) ---")
    print(f"Episodes (回合数): {result['n/ep']}")
    print(f"Mean Reward (平均奖励): {result['rew']:.4f}")
    print(f"Mean Length (平均长度): {result['len']:.2f}")
    print(f"Std Reward (奖励标准差): {result['rew_std']:.4f}")
    
    # Optional: Detailed 1-step inference check
    # 可选: 详细的单步推理检查
    print("\n--- Single Step Inference Check (单步推理检查) ---")
    obs, _ = env.reset()
    batch = Batch(obs=[obs], info={}) # Tianshou Batch
    with torch.no_grad():
        start_act = policy(batch)
    print(f"Action Output Shape (动作输出形状): {start_act.act.shape}")
    print(f"Action Output Sample (动作输出样本): {start_act.act[0][:5]} ...")


if __name__ == "__main__":
    import argparse
    from tianshou.data import Batch
    
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint)
