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
    reward_none = env._calculate_reward()
    print(f"测试3 - 无连接状态奖励: {reward_none}")
    assert reward_none == 0.0, "无连接时奖励应为0"

    # 测试4: 设置高功率状态
    env.connection_matrix = np.ones((M, N))
    env.power_matrix = np.ones((M, N))  # 最大功率
    reward_high = env._calculate_reward()
    print(f"测试4 - 高功率状态奖励: {reward_high}")

    # 测试5: 检查奖励范围
    # 奖励应为总容量，>=0
    assert reward >= 0, "奖励不应为负"
    # 理论最大容量取决于参数，但这里不检查上限

    print("所有测试通过！")
    print(f"配置: CAPACITY_THRESHOLD={CAPACITY_THRESHOLD}, REWARD_VALUE={REWARD_VALUE}")

if __name__ == "__main__":
    test_reward_function()