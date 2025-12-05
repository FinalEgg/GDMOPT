
import sys
import os
import torch
import numpy as np
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.cellfree.env import CellFreeEnv
from env.cellfree.config import REWARD_SCALE, M, N
from model.sac.sac import Actor

def diagnose(model_path, device='cuda'):
    print(f"Loading model from {model_path}...")
    
    # 1. Setup Environment
    env = CellFreeEnv(reward_mode="geometric", top_p=0.6)
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 2. Load Model
    actor = Actor(state_dim, action_dim, hidden_dim=256).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if it's a full policy state dict (keys start with _actor)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_actor.'):
                # Remove '_actor.' prefix
                new_key = k[7:]
                new_state_dict[new_key] = v
        
        if new_state_dict:
            print("Detected policy state_dict, extracting actor weights...")
            actor.load_state_dict(new_state_dict)
        else:
            # Try loading directly if it was just the actor saved (unlikely based on error)
            actor.load_state_dict(state_dict)
            
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Print keys to help debug if it fails again
        # print("Keys in state_dict:", list(state_dict.keys())[:5])
        return

    actor.eval()
    
    # 3. Run Inference
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get raw outputs
        mean, log_std, gate_logits = actor(state_tensor)
        
        # Get deterministic action (Test mode)
        # In sample(), for test: y_soft = sigmoid(gate_logits), y_hard = (y_soft > 0.5)
        y_soft = torch.sigmoid(gate_logits)
        y_hard = (y_soft > 0.5).float()
        
        # Power action (tanh -> [0,1])
        # In deterministic mode, we usually take the mean. 
        # But SAC sample() usually samples. For deterministic evaluation (test), 
        # standard implementations often use the mean for the continuous part.
        # Let's look at how `sample` is implemented in the user's code or Tianshou.
        # The user's `sample` method:
        # normal = Normal(mean, std); x_t = normal.rsample()
        # If we want deterministic, we should probably use `mean` directly for power?
        # But let's stick to what the code likely does. 
        # Actually, Tianshou's SAC policy `forward` usually returns `(batch, logits)` or similar.
        # But here we are using the custom Actor directly.
        
        # Let's assume deterministic power = tanh(mean) transformed.
        power_action_raw = torch.tanh(mean)
        power_action = (power_action_raw + 1) / 2
        
        final_action = power_action * y_hard
        
    # 4. Analyze Outputs
    print("\n" + "="*30)
    print("Diagnostic Results")
    print("="*30)
    
    print(f"\n[Gate Logits Statistics]")
    logits_np = gate_logits.cpu().numpy().flatten()
    print(f"Min: {logits_np.min():.4f}")
    print(f"Max: {logits_np.max():.4f}")
    print(f"Mean: {logits_np.mean():.4f}")
    print(f"Std: {logits_np.std():.4f}")
    print(f"Values (First 10): {logits_np[:10]}")
    
    print(f"\n[Gate Probabilities (Sigmoid)]")
    probs_np = y_soft.cpu().numpy().flatten()
    print(f"Min: {probs_np.min():.4f}")
    print(f"Max: {probs_np.max():.4f}")
    print(f"Mean: {probs_np.mean():.4f}")
    
    print(f"\n[Hard Gates (Threshold 0.5)]")
    gates_np = y_hard.cpu().numpy().flatten()
    active_gates = np.sum(gates_np)
    print(f"Active Gates: {active_gates} / {len(gates_np)}")
    print(f"All Closed? {active_gates == 0}")
    
    print(f"\n[Power Actions (Continuous)]")
    power_np = power_action.cpu().numpy().flatten()
    print(f"Mean Power: {power_np.mean():.4f}")
    
    print(f"\n[Final Actions (Power * Gate)]")
    action_np = final_action.cpu().numpy().flatten()
    print(f"Non-zero Actions: {np.count_nonzero(action_np)}")
    
    # 5. Check Reward for "All Zeros" Baseline
    print("\n" + "="*30)
    print("Baseline Check: All-Zero Action")
    print("="*30)
    
    # Reset env to be sure
    env.reset()
    # Force all zero action
    zero_action = np.zeros(action_dim)
    _, reward_zero, _, _, _ = env.step(zero_action)
    print(f"Reward for All-Zero Action: {reward_zero:.4f}")
    
    # Check Reward for Model Action
    # We need to reset again because step() advances state/time usually, 
    # but here it's a static snapshot per step mostly. 
    # However, to compare fairly against the SAME topology, we should have used the same env instance without reset if possible,
    # or seeded it.
    
    # Let's re-seed and check both
    env.seed(42)
    env.reset()
    _, r_zero, _, _, _ = env.step(np.zeros(action_dim))
    
    env.seed(42)
    env.reset()
    _, r_model, _, _, _ = env.step(action_np)
    
    print(f"Scenario (Seed 42):")
    print(f"  - All-Zero Reward: {r_zero:.4f}")
    print(f"  - Model Action Reward: {r_model:.4f}")
    
    if active_gates == 0:
        print("\n[Conclusion]")
        print("The model is outputting ALL ZEROS.")
        print("This explains why the Physical Reward (Fine-tuning) is 0.0.")
        print("It also explains the Pre-training reward: it matches the 'All-Zero' baseline.")
        print("Reason: The 'gate_linear' bias is initialized to -3.0, and training hasn't been long enough to overcome this bias.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the .pth model file')
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        diagnose(args.model)
    else:
        print(f"File not found: {args.model}")
