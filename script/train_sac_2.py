
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
    parser.add_argument('--pretrain-epoch', type=int, default=200)
    parser.add_argument('--pretrain-step-per-epoch', type=int, default=100)
    parser.add_argument('--pretrain-step-per-collect', type=int, default=1000)
    
    # Fine-tuning args
    parser.add_argument('--finetune-epoch', type=int, default=1300)
    parser.add_argument('--finetune-step-per-epoch', type=int, default=100)
    parser.add_argument('--finetune-step-per-collect', type=int, default=1000)

    # SAC args
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2) # Initial alpha
    parser.add_argument('--auto-alpha', action='store_true', default=True) # Enable Auto-Alpha
    parser.add_argument('--alpha-lr', type=float, default=1e-5) # Alpha learning rate

    # PER args
    parser.add_argument('--prioritized-replay', action='store_true', default=True)
    parser.add_argument('--prior-alpha', type=float, default=0.4)
    parser.add_argument('--prior-beta', type=float, default=0.4)

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
        alpha = (target_entropy, None, args.alpha_lr) # (target_entropy, optim, lr) - optim is created inside SAC
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
    env_geo, train_envs_geo, test_envs_geo = make_cellfree_env(
        args.training_num, args.test_num, reward_mode="geometric", k_nearest=3
    )
    
    # Initialize Policy
    policy = setup_policy(args, env_geo, args.actor_lr, args.critic_lr)
    
    # Define stop function for pre-training
    # Geometric reward max is REWARD_SCALE (10.0). If we reach 9.5, it's converged.
    from env.cellfree.config import REWARD_SCALE
    def stop_fn_geo(mean_rewards):
        return mean_rewards >= REWARD_SCALE * 0.95
    
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
        args.training_num, args.test_num, reward_mode="physical"
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
