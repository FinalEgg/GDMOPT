import sys
import os
import numpy as np
import time
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from config.default_config import DefaultConfig
from envs.fix_topp_env import FixTopPEnv
from envs.wrappers import ThresholdActionWrapper

def test_env_statistics():
    """
    测试 FixTopPEnv 环境的统计特性。
    随机生成大量无人机位置，统计连接数和基准奖励值。
    """
    print("\n================ 环境统计测试 (FixTopPEnv) ================")
    
    # 1. 初始化配置和环境
    config = DefaultConfig()
    # 确保使用 FixTopPEnv
    env = FixTopPEnv(config)
    
    # 测试参数
    num_episodes = 1000
    print(f"测试轮数: {num_episodes}")
    print(f"配置参数: M={config.M}, N={config.N}, Top-P={config.TOP_P_THRESHOLD}, K_max={env.k_max}")
    print(f"衰落阈值: {config.GEO_BETA_THRESHOLD}")
    
    # 统计数据容器
    all_connection_counts = [] # 记录每个无人机的连接基站数量
    all_baseline_rewards = []  # 记录每轮的基准奖励值（通过全0动作获得）
    
    start_time = time.time()
    
    for i in tqdm(range(num_episodes), desc="正在进行随机测试"):
        # 重置环境，随机生成无人机位置
        obs, _ = env.reset()
        
        # 执行全0动作
        # 在 FixTopPEnv 中，全0动作意味着模型分配功率为0。
        # 但是环境内部会计算 Baseline（均分功率），奖励 = Actual(0) - Baseline
        # 所以返回的 reward 即为 -BaselineScore
        action = np.zeros(env.action_space.shape)
        
        _, reward, _, _, _ = env.step(action)
        
        # 1. 统计连接数
        # env.top_p_mask 是 (M, N) 矩阵，1表示连接，0表示断开
        # 对 axis=0 求和，得到每个 UAV 连接的基站数 (N,)
        if hasattr(env, 'top_p_mask'):
            conns_per_uav = np.sum(env.top_p_mask, axis=0)
            all_connection_counts.extend(conns_per_uav)
        else:
            print("警告: 环境中未找到 top_p_mask 属性，无法统计连接数。")
            
        # 2. 统计基准奖励
        # reward = -Baseline, 所以 Baseline = -reward
        all_baseline_rewards.append(-reward)
        
    end_time = time.time()
    
    # 计算统计量
    avg_conn = np.mean(all_connection_counts)
    std_conn = np.std(all_connection_counts)
    
    avg_reward = np.mean(all_baseline_rewards)
    std_reward = np.std(all_baseline_rewards)
    
    print("\n---------------- 测试结果 ----------------")
    print(f"耗时: {end_time - start_time:.2f} 秒")
    print(f"总样本数 (UAV): {len(all_connection_counts)}")
    print(f"总样本数 (Episode): {len(all_baseline_rewards)}")
    
    print("\n1. 无人机连接基站数量统计:")
    print(f"   平均连接数: {avg_conn:.4f}")
    print(f"   标准差 (Std): {std_conn:.4f}")
    print(f"   最大连接数: {np.max(all_connection_counts)}")
    print(f"   最小连接数: {np.min(all_connection_counts)}")
    
    print("\n2. 基准奖励值统计 (均分功率):")
    print(f"   平均奖励: {avg_reward:.4f}")
    print(f"   标准差 (Std): {std_reward:.4f}")
    print(f"   说明: 此值为基站将功率平均分配给连接用户时环境能获得的得分。")
    
    print("\n================ 测试结束 ================")

if __name__ == "__main__":
    test_env_statistics()
