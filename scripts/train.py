import argparse
import os
import sys
import torch
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
from tianshou.env import DummyVectorEnv
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default_config import DefaultConfig
from config.train_config import TrainConfig
from envs import CellFreeEnv, TopPEnv, TopKEnv, FixTopPEnv, QuadraticEnv, RastriginEnv
from envs.wrappers import FlattenObservationWrapper, ThresholdActionWrapper, HybridActionWrapper
from networks.actors import DeepSetsActor, GNNActor, DeepSetsActorProb
from networks.critics import DeepSetsCritic, GNNCritic
from networks.diffusion import Diffusion
from networks.diffusion_actors import DeepSetsDiffusionModel
from networks.diffusion_critics import DeepSetsDoubleCritic
from policies import CustomDDPGPolicy, CustomSACPolicy, CustomTD3Policy
from policies.custom_diffusion import DiffusionOPT

def get_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--env', type=str, default=TrainConfig.ENV, choices=['cellfree', 'topp', 'topk', 'fix_topp', 'quadratic', 'rastrigin'])
    parser.add_argument('--algo', type=str, default=TrainConfig.ALGO, choices=['ddpg', 'sac', 'td3', 'diffusion'])
    parser.add_argument('--backbone', type=str, default=TrainConfig.BACKBONE, choices=['deepsets', 'gnn', 'mlp'])
    parser.add_argument('--action-mode', type=str, default=TrainConfig.ACTION_MODE, choices=['threshold', 'hybrid', 'pure_power'])
    parser.add_argument('--seed', type=int, default=TrainConfig.SEED)
    parser.add_argument('--device', type=str, default=TrainConfig.DEVICE)
    parser.add_argument('--logdir', type=str, default=TrainConfig.LOGDIR)
    
    # Network
    parser.add_argument('--hidden-dim', type=int, default=TrainConfig.HIDDEN_DIM, help='Hidden dimension for networks')

    # Training
    parser.add_argument('--epoch', type=int, default=TrainConfig.EPOCH)
    parser.add_argument('--step-per-epoch', type=int, default=TrainConfig.STEP_PER_EPOCH)
    parser.add_argument('--collect-per-step', type=int, default=TrainConfig.COLLECT_PER_STEP) # 增加收集步数，提高吞吐量
    parser.add_argument('--batch-size', type=int, default=TrainConfig.BATCH_SIZE) # 增大 Batch Size 稳定训练
    parser.add_argument('--buffer-size', type=int, default=TrainConfig.BUFFER_SIZE)
    parser.add_argument('--lr', type=float, default=TrainConfig.LR)
    parser.add_argument('--gamma', type=float, default=TrainConfig.GAMMA, help='Discount factor. Set to 0 for stateless/contextual bandit environments.')
    parser.add_argument('--tau', type=float, default=TrainConfig.TAU)
    
    # Algo Specific
    parser.add_argument('--exploration-noise', type=float, default=TrainConfig.EXPLORATION_NOISE) # 增大探索噪声
    parser.add_argument('--policy-noise', type=float, default=TrainConfig.POLICY_NOISE)
    parser.add_argument('--noise-clip', type=float, default=TrainConfig.NOISE_CLIP)
    parser.add_argument('--update-actor-freq', type=int, default=TrainConfig.UPDATE_ACTOR_FREQ)
    parser.add_argument('--alpha', type=float, default=TrainConfig.ALPHA) # SAC entropy
    parser.add_argument('--auto-alpha', default=TrainConfig.AUTO_ALPHA, action='store_true')

    # Diffusion Specific
    parser.add_argument('--diffusion-steps', type=int, default=TrainConfig.DIFFUSION_STEPS)
    parser.add_argument('--diffusion-beta-schedule', type=str, default=TrainConfig.DIFFUSION_BETA_SCHEDULE)
    parser.add_argument('--lr-decay', default=TrainConfig.LR_DECAY, action='store_true')
    parser.add_argument('--lr-maxt', type=int, default=TrainConfig.LR_MAXT)
    parser.add_argument('--bc-coef', default=TrainConfig.BC_COEF, action='store_true')

    return parser.parse_args()

def make_env(env_name, config, action_mode='threshold'):
    if env_name == 'cellfree':
        env = CellFreeEnv(config)
    elif env_name == 'topp':
        env = TopPEnv(config)
    elif env_name == 'topk':
        env = TopKEnv(config)
    elif env_name == 'fix_topp':
        env = FixTopPEnv(config)
    elif env_name == 'quadratic':
        return QuadraticEnv(dim=10)
    elif env_name == 'rastrigin':
        return RastriginEnv(dim=10)
    else:
        raise ValueError(f"Unknown env: {env_name}")
        
    # Apply Wrappers (Only for CellFree family)
    # 1. Action Wrapper (Decouple Model Output from Env Input)
    if action_mode == 'threshold':
        env = ThresholdActionWrapper(env, threshold=0.2)
    elif action_mode == 'pure_power':
        from envs.wrappers import PurePowerActionWrapper
        env = PurePowerActionWrapper(env)
    elif action_mode == 'hybrid':
        env = HybridActionWrapper(env)
        
    # 2. Observation Wrapper (Decouple Env State from Model Input)
    # FixTopPEnv already returns a flat vector, so we don't need FlattenObservationWrapper for it?
    # Let's check FixTopPEnv._get_obs. It returns self.cached_state which is np.concatenate(obs_list).
    # So it is already flat.
    # However, FlattenObservationWrapper handles Dict spaces.
    # FixTopPEnv uses Box space, so FlattenObservationWrapper might be redundant or harmless if it checks space type.
    # But wait, CellFreeEnv uses Dict. TopPEnv uses Box.
    # FlattenObservationWrapper implementation:
    # class FlattenObservationWrapper(gym.ObservationWrapper):
    #     def __init__(self, env):
    #         super().__init__(env)
    #         if isinstance(env.observation_space, gym.spaces.Dict):
    #             ...
    #         else:
    #             self.observation_space = env.observation_space
    #     def observation(self, observation):
    #         if isinstance(observation, dict):
    #             return np.concatenate(list(observation.values()))
    #         return observation
    
    # So it is safe to apply.
    env = FlattenObservationWrapper(env)
    
    return env

def main():
    args = get_args()
    config = DefaultConfig()
    
    # 1. Environment
    # Use DummyVectorEnv for simple parallelization/compatibility
    train_envs = DummyVectorEnv([lambda: make_env(args.env, config, args.action_mode) for _ in range(1)])
    
    # For test_envs, we wrap it to scale reward by 1/STEPS so that the "sum" displayed by Tianshou is actually the "mean"
    def make_test_env():
        env = make_env(args.env, config, args.action_mode)
        from gymnasium.wrappers import TransformReward
        # Scale reward to display mean reward per step
        # Note: This only affects the evaluation metric, not the training updates
        # We also need to undo the REWARD_SCALE to show the "real" physical reward
        env = TransformReward(env, lambda r: r / config.STEPS_PER_EPISODE / config.REWARD_SCALE)
        return env

    test_envs = DummyVectorEnv([make_test_env for _ in range(1)])
    
    # Get shape from a single instance
    env_instance = make_env(args.env, config, args.action_mode)
    state_shape = env_instance.observation_space.shape or env_instance.observation_space.n
    action_shape = env_instance.action_space.shape or env_instance.action_space.n
    max_action = env_instance.action_space.high[0]
    
    print(f"Env: {args.env}, Algo: {args.algo}, Backbone: {args.backbone}, Action Mode: {args.action_mode}")
    print(f"State Shape: {state_shape}, Action Shape: {action_shape}")
    
    # 2. Network
    if args.backbone == 'mlp':
        # Standard MLP
        net_a = Net(state_shape, hidden_sizes=[args.hidden_dim, args.hidden_dim], device=args.device)
        if args.algo == 'sac':
            actor = ActorProb(net_a, action_shape, max_action=max_action, device=args.device, unbounded=True).to(args.device)
        else:
            actor = Actor(net_a, action_shape, max_action=max_action, device=args.device).to(args.device)
        
        # Critic(s)
        # DDPG: 1 Critic
        # SAC/TD3: 2 Critics
        if args.algo == 'ddpg':
            net_c = Net(state_shape, action_shape, hidden_sizes=[args.hidden_dim, args.hidden_dim], concat=True, device=args.device)
            critic = Critic(net_c, device=args.device).to(args.device)
            critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)
        else: # sac, td3
            net_c1 = Net(state_shape, action_shape, hidden_sizes=[args.hidden_dim, args.hidden_dim], concat=True, device=args.device)
            critic1 = Critic(net_c1, device=args.device).to(args.device)
            net_c2 = Net(state_shape, action_shape, hidden_sizes=[args.hidden_dim, args.hidden_dim], concat=True, device=args.device)
            critic2 = Critic(net_c2, device=args.device).to(args.device)
            critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
            critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)
            
    elif args.backbone == 'deepsets':
        # DeepSets
        # Note: DeepSetsActor/Critic need to be adapted if they don't match Tianshou interface perfectly
        # Assuming they do based on previous context
        
        # Fix for tuple shapes
        s_dim = state_shape[0] if isinstance(state_shape, tuple) else state_shape
        a_dim = action_shape[0] if isinstance(action_shape, tuple) else action_shape

        if args.algo == 'diffusion':
            net = DeepSetsDiffusionModel(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
            critic = DeepSetsDoubleCritic(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
            
            actor = Diffusion(
                state_dim=s_dim,
                action_dim=a_dim,
                model=net,
                max_action=max_action,
                beta_schedule=args.diffusion_beta_schedule, 
                n_timesteps=args.diffusion_steps, 
            ).to(args.device)
            
            critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)
        else:
            if args.algo == 'sac':
                actor = DeepSetsActorProb(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
            else:
                actor = DeepSetsActor(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
            
            if args.algo == 'ddpg':
                critic = DeepSetsCritic(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
                critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)
            else:
                critic1 = DeepSetsCritic(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
                critic2 = DeepSetsCritic(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
                critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
                critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)
            
    elif args.backbone == 'gnn':
        # GNN
        # Fix for tuple shapes
        s_dim = state_shape[0] if isinstance(state_shape, tuple) else state_shape
        a_dim = action_shape[0] if isinstance(action_shape, tuple) else action_shape

        actor = GNNActor(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
        
        if args.algo == 'ddpg':
            critic = GNNCritic(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
            critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)
        else:
            critic1 = GNNCritic(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
            critic2 = GNNCritic(s_dim, a_dim, hidden_dim=args.hidden_dim).to(args.device)
            critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
            critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)

    # 3. Policy
    if args.algo == 'ddpg':
        policy = CustomDDPGPolicy(
            actor,
            actor_optim,
            critic,
            critic_optim,
            tau=args.tau,
            gamma=args.gamma,
            exploration_noise=None, # Handled in collector
            action_space=env_instance.action_space
        )
    elif args.algo == 'diffusion':
        policy = DiffusionOPT(
            s_dim,
            actor,
            actor_optim,
            a_dim,
            critic,
            critic_optim,
            device=args.device,
            tau=args.tau,
            gamma=args.gamma,
            exploration_noise=args.exploration_noise,
            action_space=env_instance.action_space,
            lr_decay=args.lr_decay,
            lr_maxt=args.lr_maxt,
            bc_coef=args.bc_coef
        )
    elif args.algo == 'td3':
        policy = CustomTD3Policy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=args.tau,
            gamma=args.gamma,
            exploration_noise=None,
            policy_noise=args.policy_noise,
            update_actor_freq=args.update_actor_freq,
            noise_clip=args.noise_clip,
            action_space=env_instance.action_space
        )
    elif args.algo == 'sac':
        if args.auto_alpha:
            target_entropy = -np.prod(env_instance.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.lr)
            alpha = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha = args.alpha

        policy = CustomSACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=args.tau,
            gamma=args.gamma,
            alpha=alpha,
            action_space=env_instance.action_space
        )
        # if args.auto_alpha:
        #     policy.set_alpha(args.alpha) # Tianshou handles auto alpha internally if configured

    # 4. Collector
    if args.algo in ['ddpg', 'td3']:
        from tianshou.exploration import GaussianNoise
        policy.set_exp_noise(GaussianNoise(sigma=args.exploration_noise))
    elif args.algo == 'diffusion':
        # Diffusion policy handles exploration internally or via noise injection in forward
        pass

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs)))
    test_collector = Collector(policy, test_envs)
    
    # 5. Logger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.logdir, args.env, args.algo, args.backbone, timestamp)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, train_interval=100, update_interval=100)

    # --- New Stages ---
    from scripts.pretrain import pretrain_critic, pretrain_actor_supervised
    from scripts.heuristic_search import collect_demonstration_data, flatten_obs_with_env
    
    # 0. Data Collection (if needed)
    dataset_path = os.path.join(args.logdir, 'demonstration_data.npz')
    if not os.path.exists(dataset_path):
        print("Generating demonstration data...")
        # Create a temporary env for collection
        collect_env = make_env(args.env, config, args.action_mode)
        # We need to wrap the flatten logic into the collection process
        # But collect_demonstration_data handles it internally if we modify it to use flatten_obs_with_env
        # Let's just call it. Note: collect_demonstration_data needs to be updated to use flatten_obs_with_env
        # We will update heuristic_search.py to use the helper we wrote.
        
        # Actually, let's just run the collection script separately or call it here.
        # For simplicity, we assume the user will run it or we call it now.
        # But collect_demonstration_data in heuristic_search.py needs to be robust.
        # Let's assume we updated heuristic_search.py to use flatten_obs_with_env.
        
        # Re-import to ensure we have the function
        from scripts.heuristic_search import collect_demonstration_data
        collect_demonstration_data(collect_env, num_episodes=100, save_path=dataset_path)

    # 1. Critic Warmup
    # Collect random data first
    # print("Warming up buffer with random actions...")
    # train_collector.collect(n_step=10000, random=True)
    
    warmup_data_path = os.path.join(args.logdir, 'random_data.npz')
    pretrain_critic(policy, train_collector, logger, steps=TrainConfig.PRETRAIN_CRITIC_STEPS, save_path=warmup_data_path)
    
    # 2. Actor Pretraining
    pretrain_actor_supervised(policy, dataset_path, epochs=TrainConfig.PRETRAIN_ACTOR_EPOCHS, logger=logger)
    
    # --- End New Stages ---
    
    # 6. Trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.collect_per_step,
        episode_per_test=10,
        batch_size=args.batch_size,
        logger=logger,
        save_best_fn=lambda policy: torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    )
    
    print(f"Training finished! Result: {result}")
    # print("Pretraining finished. Skipping formal RL training.")

if __name__ == '__main__':
    main()
