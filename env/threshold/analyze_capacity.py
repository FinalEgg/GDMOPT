#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 Threshold 环境的容量分布，以确定合理的奖励阈值。
使用 env/threshold/config.py 中的新参数。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from env.threshold.env import ThresholdEnv
from env.threshold.config import M, N

def analyze_capacity_distribution(num_episodes=1000):
    print(f"开始分析容量分布 (Episodes: {num_episodes})...")
    print(f"参数: M={M}, N={N}")
    
    env = ThresholdEnv(reward_mode="threshold")
    
    random_capacities = []
    heuristic_capacities = [] # 连接到信道最好的基站
    full_capacities = [] # 全连接
    
    for i in range(num_episodes):
        env.reset(seed=i)
        
        # 1. 随机动作 (Random Action)
        # 随机功率分配，归一化
        random_power = np.random.rand(M, N)
        # 归一化
        total_power = np.sum(random_power, axis=1, keepdims=True)
        mask = total_power > 1.0
        if np.any(mask):
            random_power = np.where(mask, random_power / total_power, random_power)
            
        env.power_matrix = random_power
        _, cap_random = env._calculate_capacity()
        random_capacities.append(cap_random)
        
        # 2. 启发式动作 (Heuristic: Max Channel Gain)
        # 每个 UAV 连接到信道增益 (Beta) 最大的基站，并分配最大功率
        # 注意：基站功率有限，如果多个 UAV 抢一个 BS，需要均分
        heuristic_power = np.zeros((M, N))
        
        # 找到每个 UAV 最好的 BS
        best_bs_indices = np.argmax(env.beta_matrix, axis=0) # (N,)
        
        # 简单的分配策略：每个 UAV 尝试向最好的 BS 请求 1.0 功率
        # 然后 BS 进行归一化
        for uav_idx, bs_idx in enumerate(best_bs_indices):
            heuristic_power[bs_idx, uav_idx] = 1.0
            
        # 归一化
        total_power_h = np.sum(heuristic_power, axis=1, keepdims=True)
        # 避免除以0
        total_power_h[total_power_h == 0] = 1.0 
        
        # 如果 BS 功率超标，则均分
        mask_h = total_power_h > 1.0
        heuristic_power = np.where(mask_h, heuristic_power / total_power_h, heuristic_power)
        
        env.power_matrix = heuristic_power
        _, cap_heuristic = env._calculate_capacity()
        heuristic_capacities.append(cap_heuristic)

        # 3. 全连接均分 (Full Connection Uniform)
        # 所有连接都建立，功率均分 (1/N)
        full_power = np.ones((M, N)) / N
        env.power_matrix = full_power
        _, cap_full = env._calculate_capacity()
        full_capacities.append(cap_full)
        
    # 统计分析
    def print_stats(name, data):
        data = np.array(data)
        print(f"\n--- {name} ---")
        print(f"  Mean: {np.mean(data):.4f}")
        print(f"  Std:  {np.std(data):.4f}")
        print(f"  Min:  {np.min(data):.4f}")
        print(f"  Max:  {np.max(data):.4f}")
        print(f"  25%:  {np.percentile(data, 25):.4f}")
        print(f"  50%:  {np.percentile(data, 50):.4f} (Median)")
        print(f"  75%:  {np.percentile(data, 75):.4f}")
        print(f"  90%:  {np.percentile(data, 90):.4f}")
        print(f"  95%:  {np.percentile(data, 95):.4f}")
        return data

    d_random = print_stats("随机策略 (Random)", random_capacities)
    d_heuristic = print_stats("启发式策略 (Best Channel)", heuristic_capacities)
    d_full = print_stats("全连接策略 (Full Uniform)", full_capacities)
    
    return d_random, d_heuristic, d_full

if __name__ == "__main__":
    analyze_capacity_distribution()
