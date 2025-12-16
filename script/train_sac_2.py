
# Import necessary libraries
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer
from env import make_cellfree_env
from policy import SAC
from model.sac import Actor, DuelingCritic
from env.cellfree.config import GEO_REWARD_HIT, GEO_BONUS_PERFECT, N
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    # Common args
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=1e6)
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('--wd', type=float, default=1e-5) # Reduced weight decay
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='combined')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr-decay', action='store_true', default=False)
    
    # Pre-training args
    parser.add_argument('--pretrain-epoch', type=int, default=400)
    parser.add_argument('--pretrain-step-per-epoch', type=int, default=1)
    parser.add_argument('--pretrain-step-per-collect', type=int, default=1)
    
    # Fine-tuning args
    parser.add_argument('--finetune-epoch', type=int, default=1200)
    parser.add_argument('--finetune-step-per-epoch', type=int, default=100)
    parser.add_argument('--finetune-step-per-collect', type=int, default=1000)

    # SAC args
    parser.add_argument('--actor-lr', type=float, default=1e-4) # Increased LR
    parser.add_argument('--critic-lr', type=float, default=3e-4) # Increased LR
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2) # Lower initial alpha to reduce noise dominance
    parser.add_argument('--auto-alpha', action='store_true', default=True) # Enable Auto-Alpha
    parser.add_argument('--alpha-lr', type=float, default=3e-4) # Increased Alpha LR

    # PER args
    parser.add_argument('--prioritized-replay', action='store_true', default=True)
    parser.add_argument('--prior-alpha', type=float, default=0.4)
    parser.add_argument('--prior-beta', type=float, default=0.4)
    
    # New args
    parser.add_argument('--sparsity-coef', type=float, default=0.01)
    parser.add_argument('--top-p', type=float, default=0.6)

    args = parser.parse_known_args()[0]
    return args

def setup_policy(args, env, actor_lr, critic_lr):
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.max_action = 1.
    args.exploration_noise = args.exploration_noise * args.max_action

    # Create actor
    # Ensure hidden_dim matches visualize.py (256)
    actor_net = Actor(state_dim=args.state_shape, action_dim=args.action_shape, hidden_dim=256)
    actor = actor_net.to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=actor_lr, weight_decay=args.wd)

    # Create critic
    critic = DuelingCritic(state_dim=args.state_shape, action_dim=args.action_shape, hidden_dim=256).to(args.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=critic_lr, weight_decay=args.wd)

    # Configure Alpha
    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        alpha = (target_entropy, None, args.alpha_lr, args.alpha) # (target_entropy, optim, lr, initial_alpha)
    else:
        alpha = args.alpha

    # Define policy
    policy = SAC(
        args.state_shape,
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        reward_normalization=bool(args.rew_norm),
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.finetune_epoch, # Use max epoch for decay
        sparsity_coef=args.sparsity_coef # Pass sparsity coef
    )
    return policy

def run_training_phase(args, phase_name, env, train_envs, test_envs, policy, epochs, step_per_epoch, step_per_collect, log_path, stop_fn=None):
    print(f"\n{'='*20} Starting {phase_name} Phase {'='*20}")
    
    # Setup buffer (New buffer for each phase to avoid reward distribution shift)
    if args.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
    else:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )

    # Setup collector
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    # Logger
    writer = SummaryWriter(os.path.join(log_path, phase_name))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, f'{phase_name}_policy.pth'))

    def debug_hook(epoch, env_step):
        """Print model output statistics at the start of each test phase."""
        print(f"\n[Epoch {epoch}] Debugging Model Outputs:")
        
        # Generate a batch of sample states
        batch_size = 32
        # Use test_envs to generate observations. 
        # This ensures we get normalized observations if normalization is enabled.
        # test_envs.reset() returns a batch of observations (size = test_num).
        # We can collect multiple batches if needed, but test_num (default 10) is usually enough for a quick check.
        # If we want exactly batch_size (32), we can loop.
        
        obs_list = []
        rew_list = []
        current_count = 0
        while current_count < batch_size:
            # Reset test_envs to get new observations
            batch_obs, _ = test_envs.reset()
            
            # Calculate action for these observations to get reward
            with torch.no_grad():
                batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float32, device=args.device)
                mean, _, gate_logits = policy._actor(batch_obs_tensor)
                power_val = (torch.tanh(mean) + 1) / 2
                gate_prob = torch.sigmoid(gate_logits)
                gate_action = (gate_prob > 0.5).float()
                action = power_val * gate_action
                action_np = action.cpu().numpy()
            
            # Step environment to get reward
            _, rews, _, _, _ = test_envs.step(action_np)
            
            obs_list.append(batch_obs)
            rew_list.append(rews)
            current_count += len(batch_obs)
            
        # Concatenate and slice to get exactly batch_size
        obs_array = np.concatenate(obs_list, axis=0)[:batch_size]
        rew_array = np.concatenate(rew_list, axis=0)[:batch_size]
            
        with torch.no_grad():
            # Prepare input tensor
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=args.device)
            
            # Forward pass through Actor
            # policy._actor is the underlying network (Note the underscore)
            mean, log_std, gate_logits = policy._actor(obs_tensor)
            
            # Calculate derived values
            power_raw = torch.tanh(mean) # Range [-1, 1]
            power_val = (power_raw + 1) / 2 # Range [0, 1]
            gate_prob = torch.sigmoid(gate_logits) # Range [0, 1]
            
            # Generate deterministic action for Critic evaluation
            gate_action = (gate_prob > 0.5).float()
            action = power_val * gate_action
            
            # Evaluate Critic (V and Q)
            # policy._critic is the underlying network
            v1, a1, v2, a2 = policy._critic.get_value_details(obs_tensor, action)
            q1 = v1 + a1
            q2 = v2 + a2
            
            # Print Statistics
            print(f"  > Mean (Power Param):  Min={mean.min():.4f}, Max={mean.max():.4f}, Avg={mean.mean():.4f}")
            print(f"  > Power Output (0-1):  Min={power_val.min():.4f}, Max={power_val.max():.4f}, Avg={power_val.mean():.4f}")
            print(f"  > LogStd:              Min={log_std.min():.4f}, Max={log_std.max():.4f}, Avg={log_std.mean():.4f}")
            print(f"  > Gate Logits:         Min={gate_logits.min():.4f}, Max={gate_logits.max():.4f}, Avg={gate_logits.mean():.4f}")
            print(f"  > Gate Prob:           Min={gate_prob.min():.4f}, Max={gate_prob.max():.4f}, Avg={gate_prob.mean():.4f}")
            print(f"  > Active Gates (>0.5): Ratio={(gate_prob > 0.5).float().mean():.4f}")
            print(f"  > Critic V (State Val): Min={v1.min():.4f}, Max={v1.max():.4f}, Avg={v1.mean():.4f}")
            print(f"  > Critic Q (Action Val): Min={q1.min():.4f}, Max={q1.max():.4f}, Avg={q1.mean():.4f}")
            print(f"  > Env Reward (Norm):   Min={rew_array.min():.4f}, Max={rew_array.max():.4f}, Avg={rew_array.mean():.4f}")
            print("-" * 50)
        
        # CRITICAL: Reset test_envs and test_collector after manual stepping
        # This prevents "Episode has terminated" errors in the subsequent test_episode call
        # because we manually stepped the environment to termination (len=1) above.
        test_envs.reset()
        test_collector.reset_env()
        test_collector.reset_buffer() # Clear buffer to avoid stale data issues

    # Trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        epochs,
        step_per_epoch,
        step_per_collect,
        args.test_num,
        args.batch_size,
        save_best_fn=save_best_fn,
        stop_fn=stop_fn,
        test_fn=debug_hook, # Add the debug hook here
        logger=logger,
        test_in_train=False
    )
    pprint.pprint(result)
    return result

def main(args=get_args()):
    # Setup logging
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    base_log_path = os.path.join(args.logdir, args.log_prefix, 'sac', 'cellfree', time_now)
    os.makedirs(base_log_path, exist_ok=True)
    
    # --- Phase 1: Pre-training (Geometric Reward) ---
    print("Initializing Environment for Pre-training (Geometric)...")
    # Use Top-P logic for pre-training target
    # Enable reward normalization to help Critic convergence
    env_geo, train_envs_geo, test_envs_geo = make_cellfree_env(
        args.training_num, args.test_num, reward_mode="geometric", top_p=args.top_p, norm_reward=True
    )
    
    # Calculate Baseline Reward (All Zeros)
    print("Calculating Baseline Reward for All-Zero Action...")
    dummy_env = env_geo
    dummy_env.reset()
    _, zero_reward, _, _, _ = dummy_env.step(np.zeros(dummy_env.action_space.shape))
    print(f"Baseline Reward (All Zeros): {zero_reward:.4f}")
    
    # Initialize Policy
    policy = setup_policy(args, env_geo, args.actor_lr, args.critic_lr)
    
    # --- Phase 0: Critic Warm-up (Random Actions) ---
    print("\n" + "="*50)
    print(" Starting Phase 0: Critic Warm-up")
    print("="*50)
    
    # Setup buffer for warm-up
    warmup_buffer = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs_geo))
    
    # Collect random data
    print("Collecting random data for warm-up...")
    # Use a random policy to collect data
    # We can just use the policy with high exploration noise, or manually sample random actions
    # Here we use the policy but since it's initialized randomly, it acts somewhat randomly.
    # But to be sure, we can force random actions in collection if we wanted.
    # For simplicity, let's just collect using the current policy (which is random initialized).
    
    # Actually, to ensure "completely random actions", we can use a dummy collector loop
    # But Tianshou's Collector is convenient. Let's just use the policy.
    # Since the policy is untrained, it outputs random actions (with high entropy).
    
    warmup_collector = Collector(policy, train_envs_geo, warmup_buffer)
    warmup_collector.collect(n_step=10000, random=True) # random=True forces random actions from action space
    
    print(f"Collected {len(warmup_buffer)} samples.")
    
    # Warm-up Loop
    warmup_epochs = 20
    warmup_steps = 1000
    print(f"Warming up Critic for {warmup_epochs} epochs ({warmup_steps} steps/epoch)...")
    
    for epoch in range(warmup_epochs):
        losses = []
        for _ in range(warmup_steps):
            batch, indices = warmup_buffer.sample(args.batch_size)
            # Update ONLY Critic
            res = policy.learn(batch, update_actor=False)
            losses.append(res['loss/critic'])
            
        avg_loss = np.mean(losses)
        print(f"  Warm-up Epoch {epoch+1}/{warmup_epochs} | Critic Loss: {avg_loss:.4f}")
        
    print("Critic Warm-up Completed.\n")
    
    # Define stop function for pre-training
    # Geometric reward max is dynamic now.
    # We calculate the threshold based on the configuration parameters.
    # Minimum Perfect Reward = (Min Connections * Reward per Hit) + Perfect Bonus
    # Min Connections = N (Since we force at least 1 connection per UAV)
    stop_threshold = (N * GEO_REWARD_HIT) + GEO_BONUS_PERFECT
    
    print(f"Pre-training Stop Threshold (Dynamic): {stop_threshold:.4f}")
    print(f"  - N (UAVs): {N}")
    print(f"  - Hit Reward: {GEO_REWARD_HIT}")
    print(f"  - Perfect Bonus: {GEO_BONUS_PERFECT}")
    
    def stop_fn_geo(mean_rewards):
        return mean_rewards >= stop_threshold
    
    # Run Pre-training
    run_training_phase(
        args, "pretrain", 
        env_geo, train_envs_geo, test_envs_geo, 
        policy, 
        args.pretrain_epoch, 
        args.pretrain_step_per_epoch, 
        args.pretrain_step_per_collect,
        base_log_path,
        stop_fn=stop_fn_geo
    )
    
    # Save pre-trained weights explicitly
    pretrain_weight_path = os.path.join(base_log_path, 'pretrain_final.pth')
    torch.save(policy.state_dict(), pretrain_weight_path)
    print(f"Pre-training finished. Weights saved to {pretrain_weight_path}")
    
    # Cleanup Phase 1 envs
    train_envs_geo.close()
    test_envs_geo.close()
    
    # --- Phase 2: Fine-tuning (Physical Reward) ---
    print("\nInitializing Environment for Fine-tuning (Physical)...")
    env_phy, train_envs_phy, test_envs_phy = make_cellfree_env(
        args.training_num, args.test_num, reward_mode="physical", top_p=args.top_p
    )
    
    # Re-initialize Policy structure (to be safe and clean)
    # Reduce Learning Rate for Fine-tuning
    finetune_actor_lr = args.actor_lr * 0.1
    finetune_critic_lr = args.critic_lr * 0.1
    print(f"Reducing Learning Rate: Actor {args.actor_lr}->{finetune_actor_lr}, Critic {args.critic_lr}->{finetune_critic_lr}")
    
    policy_phy = setup_policy(args, env_phy, finetune_actor_lr, finetune_critic_lr)
    
    # Load Pre-trained Weights
    print("Loading pre-trained weights...")
    policy_phy.load_state_dict(torch.load(pretrain_weight_path))
    
    # Run Fine-tuning
    run_training_phase(
        args, "finetune", 
        env_phy, train_envs_phy, test_envs_phy, 
        policy_phy, 
        args.finetune_epoch, 
        args.finetune_step_per_epoch, 
        args.finetune_step_per_collect,
        base_log_path
    )
    
    print("All training phases completed.")

if __name__ == '__main__':
    main()
