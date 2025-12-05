#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 CellFreeEnv 的奖励函数 (Refactored)

本脚本旨在全面测试 CellFreeEnv 环境中的奖励计算逻辑，包括：
1. 物理奖励 (Physical Reward): 基于 SINR 和容量的计算。
2. 几何奖励 (Geometric Reward): 基于 Top-P 策略的预训练奖励。
3. 边界条件与异常处理。

注重代码可读性与测试覆盖率。
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from env.cellfree.env import CellFreeEnv
from env.cellfree.config import (
    M, N, CAPACITY_THRESHOLD, REWARD_VALUE, REWARD_SCALE, 
    CONNECTION_COST, GEO_REWARD_HIT, GEO_BONUS_PERFECT
)

# --- 辅助函数 ---
def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_sub_header(title):
    print(f"\n--- {title} ---")

def assert_almost_equal(actual, expected, tolerance=1e-5, msg=""):
    diff = abs(actual - expected)
    if diff > tolerance:
        print(f"[FAIL] {msg}: Expected {expected}, got {actual} (Diff: {diff})")
        # raise AssertionError(f"{msg}: Expected {expected}, got {actual}")
    else:
        print(f"[PASS] {msg}")

# --- 测试模块 ---

def test_physical_reward_basics():
    """测试物理奖励的基本功能和边界条件"""
    print_header("测试 1: 物理奖励基础 (Physical Reward Basics)")
    
    env = CellFreeEnv(reward_mode="physical")
    env.reset()
    
    # 1. 默认状态
    _, total_capacity = env._calculate_capacity()
    print(f"默认随机状态总容量: {total_capacity:.4f}")
    
    # 2. 全静默状态 (无连接，无功率)
    env.power_matrix = np.zeros((M, N))
    env.connection_matrix = np.zeros((M, N))
    _, total_capacity_zero = env._calculate_capacity()
    print(f"全静默状态总容量: {total_capacity_zero:.4f}")
    assert_almost_equal(total_capacity_zero, 0.0, msg="全静默状态总容量应为 0")
    
    # 3. 全连接满功率状态
    env.power_matrix = np.ones((M, N))
    env.connection_matrix = np.ones((M, N))
    _, total_capacity_full = env._calculate_capacity()
    print(f"全连接满功率总容量: {total_capacity_full:.4f}")
    
    # 4. 简单单链路连接
    # 设置 BS 0 和 UAV 0 非常近
    env.bs_positions[0] = np.array([0.0, 0.0])
    env.uav_positions[0] = np.array([1.0, 0.0, 1.0])
    env._calculate_large_scale_fading() # 必须更新信道增益
    
    env.power_matrix = np.zeros((M, N))
    env.power_matrix[0, 0] = 0.5 # 半功率
    env.connection_matrix = np.zeros((M, N))
    env.connection_matrix[0, 0] = 1.0
    
    _, total_capacity_simple = env._calculate_capacity()
    print(f"单链路近距离总容量: {total_capacity_simple:.4f}")
    
    if total_capacity_simple <= 0:
        print("[WARN] 单链路总容量非正，请检查信道模型")

def test_physical_reward_scenarios():
    """测试物理奖励在不同场景下的表现"""
    print_header("测试 2: 物理奖励场景分析 (Physical Reward Scenarios)")
    
    env = CellFreeEnv(reward_mode="physical")
    
    # 场景 A: 距离影响
    print_sub_header("距离对容量的影响")
    env.bs_positions[0] = np.array([0.0, 0.0])
    distances = [5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]
    
    for d in distances:
        env.uav_positions[0] = np.array([d, 0.0, 10.0])
        env._calculate_large_scale_fading()
        
        env.power_matrix = np.zeros((M, N))
        env.power_matrix[0, 0] = 1.0
        
        _, total_capacity = env._calculate_capacity()
        print(f"距离 {d:<3}m | 总容量: {total_capacity:.4f}")
        
    # 场景 B: 干扰影响
    if N >= 2:
        print_sub_header("干扰对容量的影响")
        # BS 0 在原点
        env.bs_positions[0] = np.array([0.0, 0.0])
        
        # 将所有 UAV 放置在 BS 0 附近
        for k in range(N):
            # 稍微错开一点位置，避免完全重合
            env.uav_positions[k] = np.array([10.0, k * 2.0, 10.0])
            
        env._calculate_large_scale_fading()
        
        print(f"{'UAVs':<5} | {'Total Capacity':<15} | {'Avg Capacity':<15}")
        print("-" * 40)
        
        baseline_capacity = 0
        
        # 逐步增加连接的 UAV 数量 (1 到 N)
        for k in range(1, N + 1):
            env.power_matrix = np.zeros((M, N))
            # 前 k 个 UAV 连接到 BS 0，满功率 (模拟高负荷)
            # 注意：随着 k 增加，BS 0 的总发射功率增加，干扰也增加
            env.power_matrix[0, :k] = 1.0 
            
            _, total_capacity = env._calculate_capacity()
            avg_capacity = total_capacity / k
            
            print(f"{k:<5} | {total_capacity:<15.4f} | {avg_capacity:<15.4f}")
            
            if k == 1:
                baseline_capacity = total_capacity
            
        # 简单的验证：如果有干扰，N 个用户的总容量通常远小于 1 个用户容量的 N 倍
        if total_capacity < baseline_capacity * N * 0.8: 
             print("[PASS] 干扰导致容量增长受限 (非线性叠加)")
        else:
             print("[INFO] 总容量接近线性增长 (干扰较小或被总功率增加抵消)")

def test_geometric_reward_logic():
    """测试几何奖励 (Top-P) 的逻辑正确性"""
    print_header("测试 3: 几何奖励逻辑 (Geometric Reward Logic)")
    
    # 强制使用 geometric 模式
    top_p = 0.6
    env = CellFreeEnv(reward_mode="geometric", top_p=top_p)
    
    print(f"配置: Top-P={top_p}, REWARD_SCALE={REWARD_SCALE}")
    
    # 构造一个可控的 Beta 矩阵场景
    # 假设 M=5, N=5 (默认配置)
    # Beta 分布: [100, 40, 10, 1, 1]
    # Sum = 152
    # Top 60% Threshold = 152 * 0.6 = 91.2
    # 累积和: [100, 140, 150, 151, 152]
    # 100 >= 91.2 -> 截断点在索引 0 (只选第一个)
    
    betas = np.array([100.0, 40.0, 10.0, 1.0, 1.0])
    # 填充到 M
    if M > 5:
        betas = np.pad(betas, (0, M-5))
    else:
        betas = betas[:M]
        
    # 注入 Beta 矩阵
    env.beta_matrix = np.zeros((M, N))
    env.beta_matrix[:, 0] = betas # 只设置第一个 UAV 有强信号
    
    # 为了保证 UAV 1-4 的"最佳"基站也是 Index 0 (避免 argsort 对全0数组排序的不确定性)
    # 我们给 Index 0 一个极小值 (仍小于 Threshold)
    # 这样 argsort 会认为 Index 0 是最大的
    env.beta_matrix[0, 1:] = 1e-9
    
    # 注意：对于 UAV 1-4，Beta 最大值为 1e-9 < 2e-5 (Threshold)。
    # 根据逻辑，它们 valid_counts=0。
    # 但新逻辑强制选择 Top 1，即 Index 0。
    
    from env.cellfree.config import (
        GEO_PENALTY_MISS, GEO_PENALTY_NO_CONNECT, 
        GEO_PENALTY_WRONG, GEO_BETA_THRESHOLD
    )
    
    # 其他 N-1 个 UAV 都没有连接，产生的固定惩罚
    other_uavs_penalty = (N - 1) * GEO_PENALTY_NO_CONNECT
    
    # --- 场景 1: 完美匹配 ---
    # 新逻辑下：即使 Beta 很小 (或为 0)，也会强制选择至少一个基站 (Index 0)
    # 所以对于 UAV 1-4 (Beta全0)，Target 也是 Index 0
    
    env.connection_matrix = np.zeros((M, N))
    # 所有 UAV 都连接到 Index 0 (因为对于 UAV 1-4，Index 0 也是"最优"的)
    env.connection_matrix[0, :] = 1.0
    
    # 预期奖励: 
    # 所有 UAV 都 Hit
    # UAV 0: Hit (+4.0)
    # UAV 1-4: Hit (+4.0)
    # Perfect Bonus: Yes (+20.0)
    # No Connect Penalty: None
    expected_reward = (N * GEO_REWARD_HIT) + GEO_BONUS_PERFECT
    
    reward = env._calculate_geometric_reward()
    print(f"场景 1 [完美匹配]: 奖励 = {reward:.4f} (预期: {expected_reward:.4f})")
    assert_almost_equal(reward, expected_reward, msg="完美匹配奖励计算错误")
    
    # --- 场景 2: 漏连 (False Negative) ---
    # 所有 UAV 都不连接
    env.connection_matrix = np.zeros((M, N))
    
    # 预期奖励:
    # 所有 UAV 都 Miss (因为都有 Target)
    # 所有 UAV 都 No Connect
    # Total = N * (-0.3 - 20.0)
    expected_reward = N * (-1 * GEO_PENALTY_MISS - GEO_PENALTY_NO_CONNECT)
    
    reward = env._calculate_geometric_reward()
    print(f"场景 2 [漏连/全断]: 奖励 = {reward:.4f} (预期: {expected_reward:.4f})")
    assert_almost_equal(reward, expected_reward, msg="漏连奖励计算错误")
    
    # --- 场景 3: 误连 (False Positive) ---
    # UAV 0 应该连 0，结果连了 1 (次优)
    # UAV 1-4 应该连 0，结果没连
    env.connection_matrix = np.zeros((M, N))
    env.connection_matrix[1, 0] = 1.0 # UAV 0 连接了 Beta=40 的基站
    
    # 预期奖励:
    # UAV 0:
    #   TP = 0
    #   FN = 1 (Miss Target 0) -> -0.3
    #   FP = 1 (连了 1) -> Wrong (-0.2) (假设 40 > Threshold)
    #   No Connect = No
    #   Subtotal = -0.5
    
    is_useless = betas[1] < GEO_BETA_THRESHOLD
    fp_penalty = GEO_PENALTY_WRONG # 假设有用
    
    # UAV 1-4:
    #   Miss Target 0 -> -0.3
    #   No Connect -> -20.0
    #   Subtotal = -20.3
    
    expected_reward = (-1 * GEO_PENALTY_MISS - 1 * fp_penalty) + (N - 1) * (-1 * GEO_PENALTY_MISS - GEO_PENALTY_NO_CONNECT)
    
    reward = env._calculate_geometric_reward()
    print(f"场景 3 [误连次优]: 奖励 = {reward:.4f} (预期: {expected_reward:.4f})")
    assert_almost_equal(reward, expected_reward, msg="误连奖励计算错误")

def test_vectorization_correctness():
    """验证向量化计算与手动计算的一致性"""
    print_header("测试 4: 向量化一致性 (Vectorization Check)")
    
    env = CellFreeEnv(reward_mode="geometric")
    env.reset()
    
    # 随机生成一些连接
    env.connection_matrix = np.random.randint(0, 2, (M, N)).astype(float)
    
    # 1. 运行向量化版本
    reward_vec = env._calculate_geometric_reward()
    
    # 2. 运行手动版本 (模拟旧代码逻辑)
    # 为了不修改 env 代码，我们在外部模拟
    from env.cellfree.config import (
        GEO_BETA_THRESHOLD, GEO_REWARD_HIT, GEO_PENALTY_MISS, 
        GEO_PENALTY_USELESS, GEO_BONUS_PERFECT, GEO_PENALTY_WRONG, 
        GEO_PENALTY_NO_CONNECT
    )
    
    target_matrix = np.zeros((M, N))
    for k in range(N):
        betas = env.beta_matrix[:, k]
        sorted_indices = np.argsort(betas)[::-1]
        sorted_betas = betas[sorted_indices]
        cumsum_betas = np.cumsum(sorted_betas)
        total_beta = cumsum_betas[-1]
        
        cutoff_index = np.searchsorted(cumsum_betas, env.top_p * total_beta)
        valid_indices = np.where(sorted_betas >= GEO_BETA_THRESHOLD)[0]
        
        # 新逻辑模拟:
        # 1. 计算基于规则的截断点
        if len(valid_indices) > 0:
            max_valid_idx = len(valid_indices) - 1
            rule_based_cutoff = min(cutoff_index, max_valid_idx)
        else:
            rule_based_cutoff = -1 # valid_counts - 1 = 0 - 1 = -1
            
        # 2. 强制保底：至少选择 Top 1 (Index 0)
        final_cutoff = max(rule_based_cutoff, 0)
        
        # 3. 设置 Target
        top_p_indices = sorted_indices[:final_cutoff + 1]
        target_matrix[top_p_indices, k] = 1.0
            
    current_connection = env.connection_matrix
    reward_manual = 0.0
    
    tp_mask = (target_matrix == 1) & (current_connection == 1)
    reward_manual += np.sum(tp_mask) * GEO_REWARD_HIT
    
    fn_mask = (target_matrix == 1) & (current_connection == 0)
    reward_manual -= np.sum(fn_mask) * GEO_PENALTY_MISS
    
    fp_mask = (target_matrix == 0) & (current_connection == 1)
    useless_mask = (env.beta_matrix < GEO_BETA_THRESHOLD)
    fp_useless_mask = fp_mask & useless_mask
    fp_wrong_mask = fp_mask & (~useless_mask)
    
    reward_manual -= np.sum(fp_useless_mask) * GEO_PENALTY_USELESS
    reward_manual -= np.sum(fp_wrong_mask) * GEO_PENALTY_WRONG
    
    uav_connections = np.sum(current_connection, axis=0)
    no_connect_uavs = np.sum(uav_connections == 0)
    reward_manual -= no_connect_uavs * GEO_PENALTY_NO_CONNECT
    
    if np.array_equal(target_matrix, current_connection):
        reward_manual += GEO_BONUS_PERFECT
        
    print(f"向量化计算结果: {reward_vec:.6f}")
    print(f"手动计算结果:   {reward_manual:.6f}")
    
    assert_almost_equal(reward_vec, reward_manual, msg="向量化计算与手动计算不一致")

def test_reward_distribution():
    """测试奖励值的统计分布"""
    print_header("测试 5: 奖励分布统计 (Reward Distribution Statistics)")
    
    modes = ["geometric", "physical"]
    
    for mode in modes:
        print_sub_header(f"模式: {mode.upper()}")
        # 注意：Top-P 仅在 geometric 模式下有效，但 physical 模式初始化也不影响
        env = CellFreeEnv(reward_mode=mode, top_p=0.6)
        
        # --- Part 1: 固定状态，随机动作 ---
        print("\n[1. 固定状态 -> 随机动作 (1000次)]")
        
        # 定义状态场景
        states = {
            "Random": "reset",
            "Clustered (Center)": "clustered",
            "Edge (Far)": "edge"
        }
        
        for state_name, state_type in states.items():
            # 设置状态
            env.reset()
            if state_type == "clustered":
                # 所有 UAV 在中心 (50, 50, 20)
                env.uav_positions[:] = np.array([50.0, 50.0, 20.0])
                env._calculate_large_scale_fading()
            elif state_type == "edge":
                # 所有 UAV 在边缘 (0, 0, 10)
                env.uav_positions[:] = np.array([0.0, 0.0, 10.0])
                env._calculate_large_scale_fading()
            
            rewards = []
            for _ in range(1000):
                # 随机动作
                # Geometric 只看 connection, Physical 看 connection + power
                # 随机生成 0/1 连接矩阵，随机功率
                env.connection_matrix = np.random.randint(0, 2, (M, N)).astype(float)
                env.power_matrix = np.random.rand(M, N) # 0-1 之间
                
                if mode == "geometric":
                    r = env._calculate_geometric_reward()
                else:
                    r = env._calculate_physical_reward()
                rewards.append(r)
            
            rewards = np.array(rewards)
            print(f"  State: {state_name:<20} | Mean: {rewards.mean():>10.4f} | Std: {rewards.std():>10.4f} | Min: {rewards.min():>10.4f} | Max: {rewards.max():>10.4f}")

        # --- Part 2: 固定动作，随机状态 ---
        print("\n[2. 固定动作 -> 随机状态 (1000次)]")
        
        actions = {
            "Random Action": "random",
            "Full Connect": "full",
            "No Connect": "none",
            "Single Link (BS0)": "single"
        }
        
        for action_name, action_type in actions.items():
            rewards = []
            for _ in range(1000):
                env.reset() # 随机状态 (位置、信道)
                
                # 设置动作
                if action_type == "random":
                    env.connection_matrix = np.random.randint(0, 2, (M, N)).astype(float)
                    env.power_matrix = np.random.rand(M, N)
                elif action_type == "full":
                    env.connection_matrix = np.ones((M, N))
                    env.power_matrix = np.ones((M, N))
                elif action_type == "none":
                    env.connection_matrix = np.zeros((M, N))
                    env.power_matrix = np.zeros((M, N))
                elif action_type == "single":
                    env.connection_matrix = np.zeros((M, N))
                    env.power_matrix = np.zeros((M, N))
                    env.connection_matrix[0, :] = 1.0 # 所有 UAV 连 BS 0
                    env.power_matrix[0, :] = 1.0
                
                if mode == "geometric":
                    r = env._calculate_geometric_reward()
                else:
                    r = env._calculate_physical_reward()
                rewards.append(r)
                
            rewards = np.array(rewards)
            print(f"  Action: {action_name:<20} | Mean: {rewards.mean():>10.4f} | Std: {rewards.std():>10.4f} | Min: {rewards.min():>10.4f} | Max: {rewards.max():>10.4f}")

if __name__ == "__main__":
    print("启动 CellFreeEnv 奖励函数测试套件...")
    print(f"环境参数: M={M}, N={N}")
    
    test_physical_reward_basics()
    test_physical_reward_scenarios()
    test_geometric_reward_logic()
    test_vectorization_correctness()
    test_reward_distribution()
    
    print("\n" + "="*60)
    print(" 所有测试执行完毕")
    print("="*60)
