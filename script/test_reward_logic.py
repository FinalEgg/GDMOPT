
import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from env.cellfree.env import CellFreeEnv
from env.cellfree.config import M, N, REWARD_SCALE

def test_reward_logic():
    print("================ REWARD LOGIC TEST ================")
    
    # 1. Initialize Environment
    # We use a specific seed to ensure reproducibility if needed, 
    # but here we will manually inject beta_matrix to control the test case.
    env = CellFreeEnv(reward_mode="geometric", top_p=0.95)
    env.reset()
    
    print(f"Config: M={M}, N={N}, REWARD_SCALE={REWARD_SCALE}")
    
    # 2. Mock the Beta Matrix (Channel Gains)
    # Scenario: 
    # User 0 is close to BS 0 and BS 1.
    # User 1 is close to BS 0 and BS 1.
    # Other BSs are far.
    # This creates contention for BS 0 and BS 1.
    
    # Reset beta matrix
    env.beta_matrix = np.zeros((M, N))
    
    # User 0: Strong channel to BS 0, BS 1
    env.beta_matrix[0, 0] = 1.0
    env.beta_matrix[1, 0] = 0.9
    
    # User 1: Strong channel to BS 0, BS 1
    env.beta_matrix[0, 1] = 1.0
    env.beta_matrix[1, 1] = 0.9
    
    # Others are 0 for simplicity
    
    print("\n[Scenario Setup]")
    print("User 0 Betas: BS0=1.0, BS1=0.9 (Others 0)")
    print("User 1 Betas: BS0=1.0, BS1=0.9 (Others 0)")
    print("Top-P should select BS0 and BS1 for both users.")
    
    # 3. Calculate Expected Target (Internal Logic Check)
    # We replicate the logic inside _calculate_geometric_reward to see what it expects
    target_matrix = np.zeros((M, N))
    for k in range(N):
        betas = env.beta_matrix[:, k]
        sorted_indices = np.argsort(betas)[::-1]
        sorted_betas = betas[sorted_indices]
        cumsum_betas = np.cumsum(sorted_betas)
        total_beta = cumsum_betas[-1]
        cutoff_index = np.searchsorted(cumsum_betas, env.top_p * total_beta)
        top_p_indices = sorted_indices[:cutoff_index + 1]
        target_matrix[top_p_indices, k] = 1.0
        
    print("\n[Raw Target Matrix (Before Normalization)]")
    print(target_matrix[:2, :2]) # Show relevant part
    
    bs_load = np.sum(target_matrix, axis=1, keepdims=True)
    print(f"BS 0 Load: {bs_load[0,0]}")
    print(f"BS 1 Load: {bs_load[1,0]}")
    
    mask = bs_load > 1.0
    normalized_target = np.where(mask, target_matrix / bs_load, target_matrix)
    
    print("\n[Normalized Target Matrix (Expected by Current Reward)]")
    print(normalized_target[:2, :2])
    # Expecting 0.5 for all active links if load is 2.0
    
    # 4. Test Different Actions
    
    def evaluate_action(name, action_matrix):
        # Inject action
        env.power_matrix = action_matrix.copy()
        
        # Apply Env's Normalization (Step logic)
        total_power = np.sum(env.power_matrix, axis=1, keepdims=True)
        mask = total_power > 1.0
        if np.any(mask):
             env.power_matrix = np.where(mask, env.power_matrix / total_power, env.power_matrix)
        
        # Calculate Reward
        reward = env._calculate_geometric_reward()
        
        print(f"\n--- Action: {name} ---")
        print(f"Input Action (Subset):\n{action_matrix[:2, :2]}")
        print(f"Effective Action (After Env Norm):\n{env.power_matrix[:2, :2]}")
        print(f"Reward: {reward:.4f}")
        return reward

    # Case A: Perfect Uniform Match (0.5, 0.5)
    # Since target is normalized to 0.5, inputting 0.5 should be perfect.
    action_a = np.zeros((M, N))
    action_a[0, 0] = 0.5; action_a[1, 0] = 0.5
    action_a[0, 1] = 0.5; action_a[1, 1] = 0.5
    r_a = evaluate_action("A: Perfect Uniform (0.5)", action_a)
    
    # Case B: High Power Input (1.0, 1.0) -> Normalized by Env to (0.5, 0.5)
    # This should also be perfect because Env normalizes it to match the normalized target.
    action_b = np.zeros((M, N))
    action_b[0, 0] = 1.0; action_b[1, 0] = 1.0
    action_b[0, 1] = 1.0; action_b[1, 1] = 1.0
    r_b = evaluate_action("B: High Power (1.0) -> Normalized", action_b)
    
    # Case C: Skewed Power (0.9, 0.1) -> Sum=1.0 (No Env Norm needed)
    # But Target is 0.5, 0.5.
    # MSE will be non-zero.
    # User Question: Should this be penalized?
    action_c = np.zeros((M, N))
    action_c[0, 0] = 0.9; action_c[1, 0] = 0.1
    action_c[0, 1] = 0.9; action_c[1, 1] = 0.1
    r_c = evaluate_action("C: Skewed Power (0.9, 0.1)", action_c)
    
    # Case D: Wrong Topology (Connect to BS 2 instead of BS 1)
    action_d = np.zeros((M, N))
    action_d[0, 0] = 0.5; action_d[2, 0] = 0.5 # Wrong BS
    action_d[0, 1] = 0.5; action_d[1, 1] = 0.5 # Correct
    r_d = evaluate_action("D: Wrong Topology (User 0 on BS2)", action_d)

    print("\n[Analysis]")
    print(f"Reward A (Uniform): {r_a:.4f}")
    print(f"Reward C (Skewed):  {r_c:.4f}")
    print(f"Reward D (Wrong):   {r_d:.4f}")
    
    if r_c < r_a * 0.8: # Significant drop
        print("\n[CONCLUSION] Current Reward Function PENALIZES skewed power allocation.")
        print("If you want 'power allocation to not matter', this needs to be changed.")
    else:
        print("\n[CONCLUSION] Current Reward Function is tolerant to skewed power.")

if __name__ == "__main__":
    test_reward_logic()
