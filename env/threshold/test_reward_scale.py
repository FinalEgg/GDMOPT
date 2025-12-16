
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from env.threshold.env import ThresholdEnv
from env.threshold.config import M, N

def test_reward_scale(num_episodes=100):
    print(f"开始测试奖励值范围 (Episodes: {num_episodes})...")
    
    # 1. 测试 Geometric Reward (预训练阶段)
    print("\n--- 测试 Geometric Reward (Pre-training) ---")
    env_geo = ThresholdEnv(reward_mode="geometric")
    rewards_geo = []
    
    for i in range(num_episodes):
        env_geo.reset(seed=i)
        # 随机动作
        action = env_geo.action_space.sample()
        _, reward, _, _, _ = env_geo.step(action)
        rewards_geo.append(reward)
        
    rewards_geo = np.array(rewards_geo)
    print(f"Mean: {np.mean(rewards_geo):.4f}")
    print(f"Std:  {np.std(rewards_geo):.4f}")
    print(f"Min:  {np.min(rewards_geo):.4f}")
    print(f"Max:  {np.max(rewards_geo):.4f}")
    
    # 2. 测试 Threshold Reward (微调阶段)
    print("\n--- 测试 Threshold Reward (Fine-tuning) ---")
    env_thr = ThresholdEnv(reward_mode="threshold")
    rewards_thr = []
    
    # 为了测试 Threshold Reward，我们需要一些能触发阈值的动作
    # 使用启发式动作来增加触发概率
    
    for i in range(num_episodes):
        env_thr.reset(seed=i)
        
        # 启发式动作
        heuristic_power = np.zeros((M, N))
        best_bs_indices = np.argmax(env_thr.beta_matrix, axis=0)
        for uav_idx, bs_idx in enumerate(best_bs_indices):
            heuristic_power[bs_idx, uav_idx] = 1.0
        total_power_h = np.sum(heuristic_power, axis=1, keepdims=True)
        total_power_h[total_power_h == 0] = 1.0 
        mask_h = total_power_h > 1.0
        heuristic_power = np.where(mask_h, heuristic_power / total_power_h, heuristic_power)
        
        action = heuristic_power.flatten()
        _, reward, _, _, _ = env_thr.step(action)
        rewards_thr.append(reward)
        
    rewards_thr = np.array(rewards_thr)
    print(f"Mean: {np.mean(rewards_thr):.4f}")
    print(f"Std:  {np.std(rewards_thr):.4f}")
    print(f"Min:  {np.min(rewards_thr):.4f}")
    print(f"Max:  {np.max(rewards_thr):.4f}")

if __name__ == "__main__":
    test_reward_scale()
