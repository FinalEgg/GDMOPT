
import sys
import os
import numpy as np
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from env.topp.env import TopPEnv
from env.topp.config import M, N, P

def test_random_agent(episodes=10, steps=100):
    print(f"Testing TopPEnv with Random Agent for {episodes} episodes, {steps} steps each.")
    print(f"Configuration: M={M}, N={N}, Top-P={0.6} (Default)")
    
    env = TopPEnv(top_p=0.6)
    
    total_rewards = []
    total_capacities = []
    avg_powers = []
    avg_connections = []
    
    start_time = time.time()
    
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_capacity = 0
        ep_power = 0
        ep_connections = 0
        
        # 检查初始状态掩码
        # 状态结构: [Link Features (N * M * 2), Pos Features (N * 3)]
        # Link Features: [Beta, Angle] per link
        # Beta 应该是 masked 的
        
        # 简单的检查：获取 connection_matrix
        conn_matrix = env.connection_matrix
        # 获取 beta from state (需要反向解析，或者直接检查 env.beta_matrix * conn_matrix)
        # 这里我们直接检查 env 内部状态一致性
        
        # 验证 masked_beta 是否生效
        # 重新计算 masked_beta
        masked_beta_ref = env.beta_matrix * env.connection_matrix
        log_beta_ref = np.log10(masked_beta_ref + 1e-20)
        norm_log_beta_ref = (log_beta_ref + 10.0) / 5.0
        
        # 从 obs 中提取 beta features
        # obs shape: N * (M*2 + 3)
        # Link features: N * M * 2
        uav_feat_dim = M * 2 + 3
        obs_reshaped = obs.reshape(N, uav_feat_dim)
        link_feats = obs_reshaped[:, :M*2].reshape(N, M, 2)
        beta_feats_obs = link_feats[:, :, 0].T # (M, N)
        
        # 检查一致性
        diff = np.abs(beta_feats_obs - norm_log_beta_ref)
        if np.max(diff) > 1e-5:
            print(f"[Error] Episode {ep}: State observation does not match masked beta matrix!")
            print(f"Max Diff: {np.max(diff)}")
        
        # 检查未连接的 beta 是否为 "0" (即 log_beta = -20, norm = -2)
        unconnected_mask = (conn_matrix == 0)
        if np.any(unconnected_mask):
            unconnected_betas = beta_feats_obs[unconnected_mask]
            # -2.0 is the expected value for 0 input to log10(x+1e-20) -> (-20+10)/5 = -2
            if not np.allclose(unconnected_betas, -2.0, atol=1e-5):
                 print(f"[Error] Episode {ep}: Unconnected links do not have zeroed beta features!")
                 print(f"Values found: {unconnected_betas[:5]}...")

        for st in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录数据
            capacity, total_cap = env._calculate_capacity()
            
            # Debug: 检查为什么容量为 0
            if total_cap < 0.01 and st == 0: # 只在每回合第一步检查
                print(f"  [Debug Ep{ep}] Low Capacity: {total_cap:.6f}")
                # 检查连接数
                n_conns = np.sum(env.connection_matrix)
                print(f"  [Debug Ep{ep}] Connections: {n_conns}")
                # 检查 Beta 统计
                connected_betas = env.beta_matrix[env.connection_matrix == 1]
                if len(connected_betas) > 0:
                    print(f"  [Debug Ep{ep}] Connected Betas: Min={np.min(connected_betas):.2e}, Max={np.max(connected_betas):.2e}, Mean={np.mean(connected_betas):.2e}")
                else:
                    print(f"  [Debug Ep{ep}] No connections!")
                
                # 检查 Power
                print(f"  [Debug Ep{ep}] Total Power Assigned: {np.sum(env.power_matrix):.4f}")
            
            ep_reward += reward
            ep_capacity += total_cap
            ep_power += np.sum(env.power_matrix)
            ep_connections += np.sum(env.connection_matrix)
            
            if terminated or truncated:
                break
        
        total_rewards.append(ep_reward / steps)
        total_capacities.append(ep_capacity / steps)
        avg_powers.append(ep_power / steps)
        avg_connections.append(ep_connections / steps)
        
        print(f"Episode {ep+1}: Avg Reward={ep_reward/steps:.4f}, Avg Cap={ep_capacity/steps:.4f}, Avg Power={ep_power/steps:.4f}, Conns={ep_connections/steps:.1f}")

    end_time = time.time()
    
    print("\n" + "="*30)
    print("Summary Statistics (Random Agent)")
    print("="*30)
    print(f"Mean Reward: {np.mean(total_rewards):.4f} +/- {np.std(total_rewards):.4f}")
    print(f"Mean Capacity: {np.mean(total_capacities):.4f} +/- {np.std(total_capacities):.4f}")
    print(f"Mean Power: {np.mean(avg_powers):.4f} +/- {np.std(avg_powers):.4f}")
    print(f"Mean Connections: {np.mean(avg_connections):.1f} (Target Top-P ratio: {0.6})")
    print(f"Total Time: {end_time - start_time:.2f}s")
    print("="*30)

if __name__ == "__main__":
    test_random_agent()
