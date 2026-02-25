
import os
import sys
import torch
import numpy as np
from tianshou.data import Batch
import gymnasium as gym

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from envs.optimization_env import OptimizationEnv
from networks.factory import ModelFactory
from policies import CustomSACPolicy, CustomDDPGPolicy, CustomTD3Policy, CustomDiffusionPolicy
import torch.nn as nn

class TwinCriticWrapper(nn.Module):
    def __init__(self, critic1, critic2):
        super().__init__()
        self.critic1 = critic1
        self.critic2 = critic2
    
    def forward(self, obs, act=None):
        return self.critic1(obs, act), self.critic2(obs, act)
        
    def q_min(self, obs, act=None):
        q1, q2 = self(obs, act)
        return torch.min(q1, q2)

def test_algorithm(algo_name, env, state_dim, action_dim, device):
    print(f"\n{'='*20} Testing {algo_name} ... {'='*20}")
    
    # 1. Create Models (Actor & Critic)
    # Using 'mlp_baseline' from config/model_config.py
    model_id = "mlp_baseline"
    
    # Note: For Diffusion, we generally don't need a separate Critic for sampling in the same way, 
    # but Tianshou policies might require it. DiffusionOPT usually has its own Critic.
    
    try:
        if algo_name == 'diffusion':
            actor = ModelFactory.create_actor(model_id, state_dim, action_dim, algo='diffusion', device=device)
            # DiffusionOPT often uses a specific Critic structure (Q-function for guidance or just dummy if only BC).
            # But in our CustomDiffusionPolicy (DiffusionOPT), it takes a critic.
            # Usually Diffusion Critic takes (s, a) -> Q.
            c1 = ModelFactory.create_critic(model_id, state_dim, action_dim, device=device)
            c2 = ModelFactory.create_critic(model_id, state_dim, action_dim, device=device)
            critic = TwinCriticWrapper(c1, c2).to(device)
        elif algo_name == 'sac':
            actor = ModelFactory.create_actor(model_id, state_dim, action_dim, algo='sac', device=device)
            critic1 = ModelFactory.create_critic(model_id, state_dim, action_dim, device=device)
            critic2 = ModelFactory.create_critic(model_id, state_dim, action_dim, device=device)
        elif algo_name in ['ddpg', 'td3']:
            actor = ModelFactory.create_actor(model_id, state_dim, action_dim, algo=algo_name, device=device)
            critic1 = ModelFactory.create_critic(model_id, state_dim, action_dim, device=device)
            critic2 = ModelFactory.create_critic(model_id, state_dim, action_dim, device=device)
    except Exception as e:
        print(f"Error creating models for {algo_name}: {e}")
        return

    # 2. Create Optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    if algo_name == 'sac':
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)
    elif algo_name == 'td3':
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)
    elif algo_name == 'ddpg':
        critic_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    elif algo_name == 'diffusion':
        critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # 3. Create Policy
    try:
        if algo_name == 'sac':
            policy = CustomSACPolicy(
                actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
                action_space=env.action_space, sparsity_coef=0.01
            )
        elif algo_name == 'ddpg':
            policy = CustomDDPGPolicy(
                actor, actor_optim, critic1, critic_optim,
                action_space=env.action_space, sparsity_coef=0.01
            )
        elif algo_name == 'td3':
            policy = CustomTD3Policy(
                actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
                action_space=env.action_space, sparsity_coef=0.01
            )
        elif algo_name == 'diffusion':
            policy = CustomDiffusionPolicy(
                state_dim=state_dim,
                actor=actor,
                actor_optim=actor_optim,
                action_dim=action_dim,
                critic=critic,
                critic_optim=critic_optim,
                device=device,
                action_space=env.action_space # Some implementations might need this or not
            )
    except Exception as e:
        print(f"Error creating policy for {algo_name}: {e}")
        # print stack trace
        import traceback
        traceback.print_exc()
        return

    # 4. Generate Fake Data Batch
    # We simulate a batch of data to test the 'learn' step.
    batch_size = 32
    obs = np.random.randn(batch_size, state_dim).astype(np.float32)
    act = np.random.uniform(-1, 1, size=(batch_size, action_dim)).astype(np.float32)
    rew = np.random.randn(batch_size).astype(np.float32)
    obs_next = np.random.randn(batch_size, state_dim).astype(np.float32)
    done = np.zeros(batch_size).astype(np.bool_)
    
    batch = Batch(
        obs=obs,
        act=act,
        rew=rew,
        obs_next=obs_next,
        done=done,
        # to_torch removed to allow Tianshou native method
    )
    
    # 5. Run Learn Loop and Monitor Loss
    print(f"Running training loop for {algo_name}...")
    losses = []
    
    policy.train()
    
    try:
        for i in range(10): # Run for 10 steps
            
            # Special handling for Diffusion: it expects batch.returns
            if algo_name == 'diffusion':
                # For testing purposes, we validly assume returns ~ reward (ignoring next state value or using simplified assumption)
                # Or we can just set it to reward for checking backprop flow.
                batch.returns = torch.from_numpy(batch.rew).to(device).float().unsqueeze(1)
            
            result = policy.learn(batch)
            
            if algo_name == 'sac':
                loss = result.get('loss/actor', 0) + result.get('loss/critic', 0)
            elif algo_name == 'td3':
                # TD3 actor update is delayed, so 'loss/actor' might not update every step
                loss = result.get('loss/critic', 0) + result.get('loss/actor', 0)
            elif algo_name == 'ddpg':
                loss = result.get('loss/actor', 0) + result.get('loss/critic', 0)
            elif algo_name == 'diffusion':
                loss = result.get('overall_loss', 0)
            
            losses.append(loss)
            print(f"Step {i}: Loss = {loss:.4f} " + str({k: f"{v:.4f}" for k, v in result.items() if 'loss' in k}))
            
        print(f"Finished {algo_name}. Final Loss: {losses[-1]:.4f}")
        
    except Exception as e:
        print(f"Error during training {algo_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup Environment
    # Using small dimension for testing
    state_dim = 10 
    action_dim = 10
    env = OptimizationEnv(dim=state_dim)
    
    # Algorithims to test
    algos = ['sac', 'ddpg', 'td3', 'diffusion']
    
    for algo in algos:
        test_algorithm(algo, env, state_dim, action_dim, device)

if __name__ == "__main__":
    main()
