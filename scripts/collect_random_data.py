import argparse
import os
import numpy as np
import gymnasium as gym
import torch
from tqdm import tqdm
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.train_config import TrainConfig
from config.default_config import DefaultConfig
from envs import CellFreeEnv, TopPEnv, TopKEnv, FixTopPEnv, QuadraticEnv, RastriginEnv
from envs.wrappers import FlattenObservationWrapper, ThresholdActionWrapper, PurePowerActionWrapper

def make_env(env_name, config, action_mode='threshold'):
    # Merge configs: TrainConfig overrides DefaultConfig if conflict (though they usually have different keys)
    # But here we need DefaultConfig for Env parameters (M, N, etc.)
    # We can create a merged config object or just pass DefaultConfig if Env only needs that.
    # The Env classes usually take 'config' and access config.M, config.N etc.
    # TrainConfig has training params. DefaultConfig has Env params.
    # Let's create a simple merged class or object.
    
    class MergedConfig(DefaultConfig, TrainConfig):
        pass
        
    full_config = MergedConfig()
    
    if env_name == 'cellfree':
        env = CellFreeEnv(full_config)
    elif env_name == 'topp':
        env = TopPEnv(full_config)
    elif env_name == 'topk':
        env = TopKEnv(full_config)
    elif env_name == 'fix_topp':
        env = FixTopPEnv(full_config)
    elif env_name == 'quadratic':
        return QuadraticEnv(dim=10)
    elif env_name == 'rastrigin':
        return RastriginEnv(dim=10)
    else:
        raise ValueError(f"Unknown env: {env_name}")
        
    # Apply Wrappers (Only for CellFree family)
    if action_mode == 'threshold':
        env = ThresholdActionWrapper(env, threshold=0.2)
    elif action_mode == 'pure_power':
        env = PurePowerActionWrapper(env)
    # elif action_mode == 'hybrid':
    #     env = HybridActionWrapper(env)
        
    # Observation Wrapper
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservationWrapper(env)
        
    return env

class RandomPolicy:
    """Dummy policy for Tianshou Collector"""
    def map_action(self, act):
        return act
    def map_action_inverse(self, act):
        return act
    def process_fn(self, batch, buffer, indices):
        return batch
    def __call__(self, batch, state=None, **kwargs):
        return Batch(act=np.zeros(len(batch))), state

def collect_random_data(args):
    print(f"Initializing {args.num_envs} environments: {args.env}...")
    
    # Define env factory
    def get_env():
        return make_env(args.env, None, args.action_mode)
        
    # Create Vector Env
    envs = DummyVectorEnv([get_env for _ in range(args.num_envs)])
    
    # Create Buffer
    # We add some buffer to avoid "buffer full" errors during batch collection
    buffer = VectorReplayBuffer(args.steps + args.num_envs * 100, buffer_num=len(envs))
    
    # Create Collector
    policy = RandomPolicy()
    # Tianshou's Collector will use policy(batch) to get actions if random=False.
    # If random=True, it samples from env.action_space.
    # However, we need to ensure the sampled actions are in [-1, 1] if using PurePowerActionWrapper.
    # PurePowerActionWrapper sets action_space to Box(-1, 1).
    # So env.action_space.sample() will return values in [-1, 1].
    # This is correct for RL training.
    
    collector = Collector(policy, envs, buffer)
    
    print(f"Collecting {args.steps} random steps...")
    
    # Collect with progress bar
    pbar = tqdm(total=args.steps, desc="Collecting")
    current_steps = 0
    batch_size = 1000 # Update progress every 1000 steps
    
    while current_steps < args.steps:
        # Collect random data
        needed = min(batch_size, args.steps - current_steps)
        result = collector.collect(n_step=needed, random=True)
        
        # Tianshou collector returns a dict with 'n/st' or 'n_step' depending on version
        # Or it might return just the number of steps if using older version?
        # Let's check keys
        # print(result.keys())
        
        # Common keys: 'n/ep', 'n/st', 'rews', 'lens', 'idxs'
        collected = result.get('n/st', result.get('n_step', 0))
        
        pbar.update(collected)
        current_steps += collected
        
    pbar.close()
    
    # Save data
    save_path = os.path.join(args.logdir, 'random_data.npz')
    print(f"Saving data to {save_path}...")
    
    # Extract from buffer
    data = buffer[:]
    
    # Truncate to requested steps
    if len(data) > args.steps:
        data = data[:args.steps]
        
    np.savez_compressed(
        save_path,
        obs=data.obs,
        act=data.act,
        rew=data.rew,
        done=data.done,
        obs_next=data.obs_next
    )
    print("Done.")
    
    envs.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=TrainConfig.ENV)
    parser.add_argument('--action-mode', type=str, default=TrainConfig.ACTION_MODE)
    parser.add_argument('--steps', type=int, default=int(TrainConfig.PRETRAIN_CRITIC_STEPS))
    parser.add_argument('--logdir', type=str, default=TrainConfig.LOGDIR)
    parser.add_argument('--num-envs', type=int, default=10, help="Number of parallel environments")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
        
    collect_random_data(args)
