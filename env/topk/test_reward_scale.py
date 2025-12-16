import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from env.topk.env import TopKEnv
from env.topk.config import M, N, K_MAX

def test_reward_scale(num_episodes=100):
    print(f"开始测试 Top-K 环境奖励值范围 (Episodes: {num_episodes})...")
    
    env = TopKEnv()
    
    # 1. 测试随机动作 (Random Actions)
    print("\n--- 测试随机动作 (Random Actions) ---")
    rewards_random = []
    capacities_random = []
    baselines_random = []
    
    for i in range(num_episodes):
        env.reset(seed=i)
        # 随机动作: (N * K_MAX)
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        
        rewards_random.append(reward)
        _, cap = env._calculate_capacity()
        capacities_random.append(cap)
        baselines_random.append(env.baseline_capacity)
        
    rewards_random = np.array(rewards_random)
    capacities_random = np.array(capacities_random)
    baselines_random = np.array(baselines_random)
    
    print(f"Reward - Mean: {np.mean(rewards_random):.4f}, Std: {np.std(rewards_random):.4f}")
    print(f"Reward - Min:  {np.min(rewards_random):.4f}, Max: {np.max(rewards_random):.4f}")
    print(f"Capacity - Mean: {np.mean(capacities_random):.4f}")
    print(f"Baseline - Mean: {np.mean(baselines_random):.4f}")
    
    # 2. 测试启发式动作 (Heuristic Actions: Max Power to Strongest Link)
    print("\n--- 测试启发式动作 (Heuristic: Max Power to Best Link) ---")
    rewards_heuristic = []
    capacities_heuristic = []
    
    for i in range(num_episodes):
        env.reset(seed=i)
        
        # 构造动作: 每个 UAV 只给第 0 个连接 (最强连接) 满功率
        # action shape: (N * K_MAX)
        # Reshape to (N, K_MAX)
        action_matrix = np.zeros((N, K_MAX))
        # 第一个连接通常是最强的 (基于 env 逻辑: sorted_betas)
        action_matrix[:, 0] = 1.0 
        
        action = action_matrix.flatten()
        _, reward, _, _, _ = env.step(action)
        
        rewards_heuristic.append(reward)
        _, cap = env._calculate_capacity()
        capacities_heuristic.append(cap)
        
    rewards_heuristic = np.array(rewards_heuristic)
    capacities_heuristic = np.array(capacities_heuristic)
    
    print(f"Reward - Mean: {np.mean(rewards_heuristic):.4f}, Std: {np.std(rewards_heuristic):.4f}")
    print(f"Reward - Min:  {np.min(rewards_heuristic):.4f}, Max: {np.max(rewards_heuristic):.4f}")
    print(f"Capacity - Mean: {np.mean(capacities_heuristic):.4f}")

if __name__ == "__main__":
    test_reward_scale()
