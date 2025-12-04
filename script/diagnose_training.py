
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from env.cellfree.env import CellFreeEnv
from model.sac.sac import Actor, DuelingCritic
from env.cellfree.config import M, N, REWARD_SCALE

def diagnose():
    print("================ DIAGNOSTIC START ================")
    
    # 1. Initialize Environment
    print("\n[1] Initializing Environment...")
    try:
        env = CellFreeEnv(reward_mode="geometric", top_p=0.95)
        s, _ = env.reset()
        print("    Environment initialized successfully.")
        print(f"    State Dim: {env.observation_space.shape[0]}")
        print(f"    Action Dim: {env.action_space.shape[0]}")
    except Exception as e:
        print(f"    ERROR initializing environment: {e}")
        return

    # 2. Initialize Model
    print("\n[2] Initializing Model...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actor = Actor(state_dim, action_dim, hidden_dim=256).to(device)
    critic = DuelingCritic(state_dim, action_dim, hidden_dim=256).to(device)
    print("    Model initialized.")

    # 3. Check Reward Logic & Conflicts
    print("\n[3] Checking Reward Logic & Conflicts...")
    # Force a reset to get fresh matrices
    env.reset()
    
    # Calculate what the target power WOULD be
    # We access internal variables for diagnosis
    conflicts = 0
    total_target_sum = 0
    
    print("    Analyzing Top-P Targets for current user positions...")
    for k in range(N):
        betas = env.beta_matrix[:, k]
        sorted_indices = np.argsort(betas)[::-1]
        sorted_betas = betas[sorted_indices]
        cumsum_betas = np.cumsum(sorted_betas)
        total_beta = cumsum_betas[-1]
        cutoff_index = np.searchsorted(cumsum_betas, env.top_p * total_beta)
        top_p_indices = sorted_indices[:cutoff_index + 1]
        
        # Just for logging
        # print(f"    User {k}: Needs {len(top_p_indices)} BSs for Top-{env.top_p}")
    
    # Check BS contention
    # Construct the ideal target matrix
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
        
    # Check sum per BS
    bs_sums = np.sum(target_matrix, axis=1)
    overloaded_bs = np.sum(bs_sums > 1.0)
    max_load = np.max(bs_sums)
    
    print(f"    BS Overload Check (Raw Target):")
    print(f"    - Max requested load on a single BS: {max_load:.2f} (Limit is 1.0)")
    print(f"    - Number of overloaded BSs: {overloaded_bs} / {M}")
    
    if overloaded_bs > 0:
        print("    [INFO] Raw target is overloaded. Verifying normalization logic...")
        # Simulate normalization
        bs_load = np.sum(target_matrix, axis=1, keepdims=True)
        mask = bs_load > 1.0
        normalized_target = np.where(mask, target_matrix / bs_load, target_matrix)
        new_max_load = np.max(np.sum(normalized_target, axis=1))
        print(f"    - Max load after normalization: {new_max_load:.2f}")
        if new_max_load <= 1.0001:
            print("    [SUCCESS] Normalization logic works. Reward target is feasible.")
        else:
            print("    [ERROR] Normalization failed!")

    # 4. Check Model Outputs & Loss Components
    print("\n[4] Checking Model Outputs & Loss Components (Dummy Batch)...")
    batch_size = 32
    states = []
    for _ in range(batch_size):
        s, _ = env.reset()
        states.append(s)
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    
    # Forward pass
    action, log_prob, mean, log_std, gate_probs = actor.sample(states)
    
    print(f"    Mean stats: min={mean.min().item():.4f}, max={mean.max().item():.4f}, avg={mean.mean().item():.4f}")
    print(f"    LogStd stats: min={log_std.min().item():.4f}, max={log_std.max().item():.4f}, avg={log_std.mean().item():.4f}")
    print(f"    Std stats: min={log_std.exp().min().item():.4f}, max={log_std.exp().max().item():.4f}")
    print(f"    Gate Probs: min={gate_probs.min().item():.4f}, max={gate_probs.max().item():.4f}, avg={gate_probs.mean().item():.4f}")
    
    # Check for saturation
    if log_std.max().item() >= 1.9:
        print("    [WARNING] LogStd is hitting max clamp (2.0). Model is maximizing entropy aggressively.")
        
    # Check Q-values
    q1, q2 = critic(states, action)
    print(f"    Q-values: min={q1.min().item():.4f}, max={q1.max().item():.4f}, avg={q1.mean().item():.4f}")
    
    # Check Loss Terms magnitudes
    alpha = 0.2 # Dummy alpha
    min_q = torch.min(q1, q2)
    entropy_term = -alpha * log_prob
    
    print(f"    LogProb: min={log_prob.min().item():.4f}, max={log_prob.max().item():.4f}, avg={log_prob.mean().item():.4f}")
    print(f"    Entropy Term (-alpha * log_prob): avg={entropy_term.mean().item():.4f}")
    print(f"    Q Term (min_q): avg={min_q.mean().item():.4f}")
    
    sparsity_coef = 0.05
    sparsity_loss = sparsity_coef * gate_probs.mean()
    print(f"    Sparsity Loss ({sparsity_coef} * mean_gate): {sparsity_loss.item():.4f}")
    
    total_loss = (alpha * log_prob - min_q).mean() + sparsity_loss
    print(f"    Total Actor Loss: {total_loss.item():.4f}")
    
    # 5. Check Gradient Flow
    print("\n[5] Checking Gradient Flow...")
    total_loss.backward()
    
    has_grad = True
    for name, param in actor.named_parameters():
        if param.grad is None:
            print(f"    [WARNING] No gradient for {name}")
            has_grad = False
        else:
            if torch.isnan(param.grad).any():
                print(f"    [ERROR] NaN gradient for {name}")
            if param.grad.abs().max() == 0:
                # print(f"    [INFO] Zero gradient for {name}") # Common for ReLU dead zones
                pass
                
    if has_grad:
        print("    Gradients appear to be flowing.")

    print("\n================ DIAGNOSTIC END ================")

if __name__ == "__main__":
    diagnose()
