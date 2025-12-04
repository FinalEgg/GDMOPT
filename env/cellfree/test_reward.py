#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 CellFreeEnv 的奖励函数
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from env.cellfree.env import CellFreeEnv
from env.cellfree.config import M, N, CAPACITY_THRESHOLD, REWARD_VALUE

def test_reward_function():
    """测试奖励函数的输出"""
    print("开始测试 CellFreeEnv 奖励函数...")

    # 创建环境
    env = CellFreeEnv()
    print(f"环境创建成功: M={M}, N={N}")

    # 测试1: 默认状态（随机连接和功率）
    env.reset()
    reward = env._calculate_reward()
    print(f"测试1 - 默认状态奖励: {reward}")
    assert isinstance(reward, (int, float)), "奖励应为数值"

    # 测试2: 设置全连接状态
    env.connection_matrix = np.ones((M, N))  # 所有基站连接所有 UAV
    env.power_matrix = np.ones((M, N)) * 0.5  # 中等功率
    reward_full = env._calculate_reward()
    print(f"测试2 - 全连接状态奖励: {reward_full}")

    # 测试3: 设置无连接状态
    env.connection_matrix = np.zeros((M, N))
    env.power_matrix = np.zeros((M, N)) # 确保功率也为0
    reward_none = env._calculate_reward()
    print(f"测试3 - 无连接状态奖励: {reward_none}")
    assert reward_none == 0.0, "无连接时奖励应为0"

    # 测试4: 设置高功率状态
    env.connection_matrix = np.ones((M, N))
    env.power_matrix = np.ones((M, N))  # 最大功率
    reward_high = env._calculate_reward()
    print(f"测试4 - 高功率状态奖励: {reward_high}")

    # 测试5: 简单连接 - 仅基站0连接UAV0，功率0.5，其他无连接
    env.bs_positions[0] = np.array([0.0, 0.0])  # 设置基站位置
    env.uav_positions[0] = np.array([1.0, 0.0, 1.0])  # 设置UAV位置，近距离
    env._calculate_large_scale_fading() # 更新缓存
    env.connection_matrix = np.zeros((M, N))
    env.connection_matrix[0, 0] = 1
    env.power_matrix = np.zeros((M, N))
    env.power_matrix[0, 0] = 0.5
    reward_simple = env._calculate_reward()
    print(f"测试5 - 简单连接奖励: {reward_simple}")
    assert reward_simple > 0, "有连接时奖励应大于0"

    # 测试6: 两个连接 - 基站0连接UAV0和UAV1，功率0.5（如果N>=2）
    if N >= 2:
        env.connection_matrix = np.zeros((M, N))
        env.connection_matrix[0, 0] = 1
        env.connection_matrix[0, 1] = 1
        env.power_matrix = np.zeros((M, N))
        env.power_matrix[0, 0] = 0.5
        env.power_matrix[0, 1] = 0.5
        reward_two = env._calculate_reward()
        print(f"测试6 - 两个连接奖励: {reward_two}")
        # 注意：由于随机位置和干扰，奖励可能不严格增加，但应>0

    # 测试7: 检查奖励范围
    assert reward >= 0, "奖励不应为负"
    assert reward_simple >= 0, "简单奖励不应为负"

    print("所有测试通过！")
    print(f"配置: CAPACITY_THRESHOLD={CAPACITY_THRESHOLD}, REWARD_VALUE={REWARD_VALUE}")
    print(f"简单连接奖励: {reward_simple} (预期 >0，由于随机位置可能很小)")
    if N >= 2:
        print(f"两个连接奖励: {reward_two} (预期 >0)")

def test_manual_scenarios():
    """手动设定状态的测试"""
    print("\n开始手动测试场景...")

    env = CellFreeEnv()

    # 第一组测试：固定水平位置，改变高度
    print("\n第一组测试：无人机在基站正上方，改变高度")
    bs_pos = np.array([50.0, 50.0])  # 基站位置
    env.bs_positions[0] = bs_pos
    heights = [1, 5, 10, 20, 30, 50]  # 高度列表
    power = 0.5  # 固定功率

    for h in heights:
        env.uav_positions[0] = np.array([50.0, 50.0, h])
        env._calculate_large_scale_fading() # 更新缓存
        env.connection_matrix = np.zeros((M, N))
        env.connection_matrix[0, 0] = 1
        env.power_matrix = np.zeros((M, N))
        env.power_matrix[0, 0] = power
        reward = env._calculate_reward()
        print(f"BS pos: {bs_pos}, UAV pos: [50.0, 50.0, {h}], Power: {power}, Reward: {reward}")

    # 第二组测试：固定高度，改变水平位置
    print("\n第二组测试：无人机高度固定，改变水平位置")
    fixed_h = 10.0
    x_positions = np.arange(0, 101, 10)  # 从0到100，每10单位

    for x in x_positions:
        env.uav_positions[0] = np.array([x, 50.0, fixed_h])
        env._calculate_large_scale_fading() # 更新缓存
        env.connection_matrix = np.zeros((M, N))
        env.connection_matrix[0, 0] = 1
        env.power_matrix = np.zeros((M, N))
        env.power_matrix[0, 0] = power
        reward = env._calculate_reward()
        print(f"BS pos: {bs_pos}, UAV pos: [{x}, 50.0, {fixed_h}], Power: {power}, Reward: {reward}")

def test_power_allocation_effect():
    """测试功率分配对奖励的影响：一个基站，两个无人机，一个近一个远"""
    print("\n开始测试功率分配效果...")

    env = CellFreeEnv()

    # 设置基站位置（使用第一个基站）
    env.bs_positions[0] = np.array([50.0, 50.0])

    # 设置无人机位置：UAV0 近，UAV1 远
    env.uav_positions[0] = np.array([50.0, 50.0, 10.0])  # 近：正上方，高度10
    env.uav_positions[1] = np.array([100.0, 100.0, 50.0])  # 远：角落，高度50
    env._calculate_large_scale_fading() # 更新缓存

    # 连接矩阵：基站0 连接 UAV0 和 UAV1
    env.connection_matrix = np.zeros((M, N))
    env.connection_matrix[0, 0] = 1
    env.connection_matrix[0, 1] = 1

    # 功率矩阵：UAV0 功率固定为0.5，UAV1 功率从0.1到1.0逐步增加
    fixed_power_near = 0.5
    power_levels_far = np.arange(0.1, 1.1, 0.1)  # 0.1, 0.2, ..., 1.0

    print("功率分配测试：近无人机功率固定为0.5，远无人机功率逐渐增大")

    for power_far in power_levels_far:
        env.power_matrix = np.zeros((M, N))
        env.power_matrix[0, 0] = fixed_power_near
        env.power_matrix[0, 1] = power_far
        reward = env._calculate_reward()
        print(f"远无人机功率: {power_far:.1f}, 总奖励: {reward:.4f}")

    print("测试完成：期望看到奖励值逐渐减小（由于干扰增加）")

def test_comprehensive_analysis():
    """全面分析奖励函数数值规模，辅助参数调优"""
    print("\n" + "="*50)
    print("开始全面奖励数值分析 (Comprehensive Reward Analysis)")
    print("="*50)
    
    env = CellFreeEnv()
    
    # 1. 软门控与有效功率测试
    print("\n[分析 1] 软门控 (Soft Gate) 行为分析")
    print(f"{'Action':<10} | {'Effective Power':<15} | {'Connection Status':<18}")
    print("-" * 50)
    actions = [0.0, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0]
    
    # 临时模拟 step 中的门控逻辑
    threshold = 0.1
    k = 50.0
    
    for a in actions:
        gate = 1.0 / (1.0 + np.exp(-k * (a - threshold)))
        eff_p = a * gate
        is_connected = eff_p > 0.01
        status = "CONNECTED" if is_connected else "DISCONNECTED"
        print(f"{a:<10.4f} | {eff_p:<15.6f} | {status:<18}")

    # 2. 单链路奖励规模测试 (Capacity vs Cost)
    print("\n[分析 2] 单链路奖励规模 (Capacity vs Cost)")
    print(f"当前配置: CONNECTION_COST = {getattr(env, 'CONNECTION_COST', 0.02)} (假设值, 请核对config)")
    
    env.reset()
    # 设置一个标准近距离场景
    env.bs_positions[0] = np.array([0.0, 0.0])
    env.uav_positions[0] = np.array([10.0, 0.0, 10.0]) # 距离约 14m
    env._calculate_large_scale_fading()
    
    print(f"{'Power':<10} | {'Raw Capacity':<15} | {'Cost Penalty':<15} | {'Final Reward':<15}")
    print("-" * 60)
    
    test_powers = [0.05, 0.07, 0.1, 0.2, 0.5, 0.8, 1.0]
    
    for p in test_powers:
        env.power_matrix = np.zeros((M, N))
        env.power_matrix[0, 0] = p
        
        # 手动应用门控以获得真实物理功率
        gate = 1.0 / (1.0 + np.exp(-k * (p - threshold)))
        eff_p = p * gate
        env.power_matrix[0, 0] = eff_p
        
        # 计算各部分
        # 注意：这里我们需要访问 env 内部逻辑或近似计算
        # 为了准确，我们直接调用 _calculate_physical_reward (如果存在) 或 _calculate_reward
        # 并手动计算 cost
        
        # 临时移除 cost 计算纯 Capacity (假设 env._calculate_reward 包含 cost)
        # 我们需要一种方法分离它们。
        # 由于无法直接分离，我们通过计算两次来推断：
        # 1. 正常计算
        total_reward = env._calculate_physical_reward() if hasattr(env, '_calculate_physical_reward') else env._calculate_reward()
        
        # 2. 手动计算 Cost
        # 假设 env.py 中 cost = CONNECTION_COST * sum(power)
        # 注意：env.py 中使用的是 effective power
        from env.cellfree.config import CONNECTION_COST
        cost = CONNECTION_COST * eff_p
        
        # 推断 Capacity 部分
        # Reward = Capacity/10 - Cost
        # Capacity/10 = Reward + Cost
        capacity_term = total_reward + cost
        raw_capacity = capacity_term * 10.0
        
        print(f"{p:<10.2f} | {raw_capacity:<15.4f} | {cost:<15.4f} | {total_reward:<15.4f}")

    # 3. 多用户干扰场景下的奖励极值
    print("\n[分析 3] 多用户场景奖励极值 (N=5)")
    env.reset()
    
    # 场景 A: 理想情况 (所有用户都在不同基站附近，互不干扰)
    print("场景 A: 理想空间复用 (Spatial Multiplexing)")
    env.power_matrix = np.zeros((M, N))
    for i in range(min(M, N)):
        # 假装每个用户 i 离基站 i 很近
        # 我们直接修改 beta_matrix 来模拟"完美信道"，而不去改坐标
        env.beta_matrix = np.zeros((M, N))
        env.beta_matrix[i, i] = 1e-5 # 假设一个较大的增益 (Path Loss通常很小，如1e-6)
        # 实际上 Path Loss = D^-alpha. 距离10m -> 10^-2 = 0.01. 
        # 让我们用真实计算
        env.bs_positions[i] = np.array([i*100.0, 0.0])
        env.uav_positions[i] = np.array([i*100.0 + 5.0, 0.0, 10.0]) # 非常近
        env.power_matrix[i, i] = 1.0 # 满功率
    
    env._calculate_large_scale_fading()
    # 重新应用功率 (因为 reset 会清空)
    env.power_matrix = np.zeros((M, N))
    for i in range(min(M, N)):
        env.power_matrix[i, i] = 1.0
        
    reward_ideal = env._calculate_physical_reward() if hasattr(env, '_calculate_physical_reward') else env._calculate_reward()
    print(f"理想最大奖励 (Max Reward): {reward_ideal:.4f}")
    
    # 场景 B: 恶劣情况 (所有用户挤在一起，全功率发射)
    print("场景 B: 严重干扰 (Heavy Interference)")
    for i in range(N):
        env.uav_positions[i] = np.array([50.0, 50.0, 10.0]) # 都在中心
    env._calculate_large_scale_fading()
    env.power_matrix = np.ones((M, N)) # 全连接，全功率
    
    reward_bad = env._calculate_physical_reward() if hasattr(env, '_calculate_physical_reward') else env._calculate_reward()
    print(f"全干扰奖励 (Min Reward): {reward_bad:.4f}")
    
    # 场景 C: 稀疏惩罚测试 (全0功率)
    print("场景 C: 全静默 (All Silence)")
    env.power_matrix = np.zeros((M, N))
    reward_zero = env._calculate_physical_reward() if hasattr(env, '_calculate_physical_reward') else env._calculate_reward()
    print(f"全静默奖励 (Zero Power): {reward_zero:.4f}")

if __name__ == "__main__":
    test_reward_function()
    test_manual_scenarios()
    test_power_allocation_effect()
    test_comprehensive_analysis()