import sys
import os
import numpy as np
import torch
import pprint

# Add project root to path
# 将项目根目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.setup import make_env_instance
from config.env_config import EnvConfig
from config.run_config import RunConfig

def check_env():
    """
    Run environment check.
    运行环境检查。

    Performs the following checks:
    1. Instantiates the environment with default configurations.
    2. Inspects Observation and Action spaces.
    3. Runs a random interaction loop to test reset and step functions.
    4. Calculates and prints basic statistics on rewards.
    5. Checks for NaNs or constant zero rewards.

    执行以下检查：
    1. 使用默认配置实例化环境。
    2. 检查观测空间和动作空间。
    3. 运行随机交互循环以测试重置和步进功能。
    4. 计算并打印奖励的基本统计数据。
    5. 检查是否存在 NaN 或持续为零的奖励。
    """
    print("=== Environment Validation Script ===")
    print("=== 环境验证脚本 ===")
    
    # Use default config
    # 使用默认配置
    env = make_env_instance(RunConfig, EnvConfig, seed=42)
    
    print(f"Env Class (环境类): {type(env)}")
    print(f"Obs Space (观测空间): {env.observation_space}")
    print(f"Act Space (动作空间): {env.action_space}")
    
    # 1. Reset
    # 1. 重置
    obs, info = env.reset(seed=42)
    print("\n--- Initial State (初始状态) ---")
    print(f"Obs Shape (观测维度): {obs.shape}")
    
    # 2. Random Step Loop
    # 2. 随机步进循环
    num_steps = 100
    rewards = []
    
    print(f"\n--- Running {num_steps} random steps (运行 {num_steps} 个随机步) ---")
    for i in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        if done or truncated:
            obs, info = env.reset()
            
    # 3. Stats
    # 3. 统计信息
    rewards = np.array(rewards)
    stats = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards)
    }
    
    print("\n--- Reward Stats (奖励统计) ---")
    pprint.pprint(stats)
    
    if stats['mean_reward'] == 0.0 and stats['std_reward'] == 0.0:
        print("\n[WARNING] All rewards are zero. Check Reward Function.")
        print("[警告] 所有奖励均为零。请检查奖励函数。")
    elif np.isnan(stats['mean_reward']):
        print("\n[ERROR] NaNs detected in rewards.")
        print("[错误] 在奖励中检测到 NaN。")
    else:
        print("\n[PASS] Reward distribution looks valid (sanity check only).")
        print("[通过] 奖励分布看起来有效 (仅作完备性检查)。")

if __name__ == "__main__":
    check_env()
