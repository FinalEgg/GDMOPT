
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from env.threshold.env import ThresholdEnv
from env.threshold.config import (
    M, N, GEO_BETA_THRESHOLD, GEO_REWARD_HIT, GEO_PENALTY_MISS,
    GEO_PENALTY_USELESS, GEO_BONUS_PERFECT, GEO_PENALTY_WRONG,
    GEO_PENALTY_NO_CONNECT
)

def analyze_geometric_reward(num_episodes=100):
    print(f"开始分析几何奖励 (Episodes: {num_episodes})...")
    env = ThresholdEnv(reward_mode="geometric")
    
    stats = {
        "random": {"reward": [], "hit": [], "miss": [], "fp_useless": [], "fp_wrong": [], "no_connect": [], "perfect": []},
        "heuristic": {"reward": [], "hit": [], "miss": [], "fp_useless": [], "fp_wrong": [], "no_connect": [], "perfect": []}
    }
    
    # 新增统计数据容器
    target_conn_counts = []
    all_log_betas = []
    
    for strategy in ["random", "heuristic"]:
        print(f"\nTesting Strategy: {strategy}")
        for i in range(num_episodes):
            env.reset(seed=i)
            
            if strategy == "random":
                action = env.action_space.sample()
            else:
                # Heuristic: Connect to best BS
                heuristic_power = np.zeros((M, N))
                best_bs_indices = np.argmax(env.beta_matrix, axis=0)
                for uav_idx, bs_idx in enumerate(best_bs_indices):
                    heuristic_power[bs_idx, uav_idx] = 1.0
                # Normalize
                total_power_h = np.sum(heuristic_power, axis=1, keepdims=True)
                total_power_h[total_power_h == 0] = 1.0 
                mask_h = total_power_h > 1.0
                heuristic_power = np.where(mask_h, heuristic_power / total_power_h, heuristic_power)
                action = heuristic_power.flatten()
            
            # Step
            _, reward, _, _, _ = env.step(action)
            
            # Manually calculate components for analysis
            # Re-implement logic from env._calculate_geometric_reward to get components
            # Note: env.step() already updated connection_matrix based on action
            
            # 1. Calculate Target Matrix
            sorted_indices = np.argsort(env.beta_matrix, axis=0)[::-1]
            sorted_betas = np.take_along_axis(env.beta_matrix, sorted_indices, axis=0)
            cumsum_betas = np.cumsum(sorted_betas, axis=0)
            total_beta = cumsum_betas[-1, :]
            threshold_values = total_beta * env.top_p
            mask_cumsum = cumsum_betas >= threshold_values[None, :]
            cutoff_indices = np.argmax(mask_cumsum, axis=0)
            valid_counts = np.sum(sorted_betas >= GEO_BETA_THRESHOLD, axis=0)
            rule_based_cutoffs = np.minimum(cutoff_indices, valid_counts - 1)
            final_cutoffs = np.maximum(rule_based_cutoffs, 0)
            
            target_matrix = np.zeros((M, N))
            row_indices = np.arange(M)[:, None]
            selection_mask = (row_indices <= final_cutoffs[None, :])
            target_bs_indices = sorted_indices[selection_mask]
            col_indices = np.tile(np.arange(N), (M, 1))
            target_uav_indices = col_indices[selection_mask]
            target_matrix[target_bs_indices, target_uav_indices] = 1.0
            
            # 收集环境统计数据 (仅在 random 策略循环中收集一次，避免重复)
            if strategy == "random":
                # 统计每个 UAV 在 Target Matrix 中的连接数
                conns = np.sum(target_matrix, axis=0)
                target_conn_counts.extend(conns)
                
                # 统计 Log Beta
                lb = np.log10(env.beta_matrix + 1e-30)
                all_log_betas.extend(lb.flatten())

            current_connection = env.connection_matrix
            
            tp_count = np.sum((target_matrix == 1) & (current_connection == 1))
            fn_count = np.sum((target_matrix == 1) & (current_connection == 0))
            
            fp_mask = (target_matrix == 0) & (current_connection == 1)
            useless_mask = (env.beta_matrix < GEO_BETA_THRESHOLD)
            fp_useless_count = np.sum(fp_mask & useless_mask)
            fp_wrong_count = np.sum(fp_mask & (~useless_mask))
            
            uav_connections = np.sum(current_connection, axis=0)
            no_connect_uavs = np.sum(uav_connections == 0)
            
            is_perfect = np.array_equal(target_matrix, current_connection)
            
            stats[strategy]["reward"].append(reward)
            stats[strategy]["hit"].append(tp_count)
            stats[strategy]["miss"].append(fn_count)
            stats[strategy]["fp_useless"].append(fp_useless_count)
            stats[strategy]["fp_wrong"].append(fp_wrong_count)
            stats[strategy]["no_connect"].append(no_connect_uavs)
            stats[strategy]["perfect"].append(1 if is_perfect else 0)

    # Print Stats
    for strategy in ["random", "heuristic"]:
        print(f"\n--- Strategy: {strategy} ---")
        s = stats[strategy]
        print(f"Reward: Mean={np.mean(s['reward']):.4f}, Std={np.std(s['reward']):.4f}")
        print(f"Hit (TP): Mean={np.mean(s['hit']):.4f}")
        print(f"Miss (FN): Mean={np.mean(s['miss']):.4f}")
        print(f"FP Useless: Mean={np.mean(s['fp_useless']):.4f}")
        print(f"FP Wrong: Mean={np.mean(s['fp_wrong']):.4f}")
        print(f"No Connect UAVs: Mean={np.mean(s['no_connect']):.4f}")
        print(f"Perfect Ratio: {np.mean(s['perfect']):.4f}")

    # Print Environment Analysis
    print("\n" + "="*40)
    print("--- Environment & Target Analysis ---")
    print("="*40)
    
    tcc = np.array(target_conn_counts)
    print(f"Target Connections per UAV (top_p={env.top_p}):")
    print(f"  Mean: {np.mean(tcc):.4f}")
    print(f"  Std:  {np.std(tcc):.4f}")
    print(f"  Min:  {np.min(tcc):.4f}")
    print(f"  Max:  {np.max(tcc):.4f}")
    print(f"  Median: {np.median(tcc):.4f}")
    print(f"  Distribution (Counts): {np.bincount(tcc.astype(int))}") 

    lbs = np.array(all_log_betas)
    print(f"\nLog10(Beta) Statistics (Channel Strength):")
    print(f"  Mean: {np.mean(lbs):.4f}")
    print(f"  Std:  {np.std(lbs):.4f}")
    print(f"  Min:  {np.min(lbs):.4f}")
    print(f"  Max:  {np.max(lbs):.4f}")
    print(f"  25%:  {np.percentile(lbs, 25):.4f}")
    print(f"  50%:  {np.percentile(lbs, 50):.4f} (Median)")
    print(f"  75%:  {np.percentile(lbs, 75):.4f}")
    print(f"  90%:  {np.percentile(lbs, 90):.4f}")

if __name__ == "__main__":
    analyze_geometric_reward()
