"""
setup.py: Utility module for initializing Environments and Policies.
Acts as a bridge between Configuration files (config/) and Envs/Networks.
"""

from typing import Tuple, Dict, Any, Type
import gymnasium as gym
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.exploration import GaussianNoise
import torch

# Import Configs
from config.env_config import EnvConfig
from config.run_config import RunConfig
from config.mapping_config import MappingConfig
from config.model_config import ModelConfig

# Import Factory
from networks.factory import ModelFactory, get_model
from policies import CustomDDPGPolicy, CustomTD3Policy, CustomSACPolicy, CustomDiffusionPolicy


def make_env_instance(run_config: RunConfig, env_config: EnvConfig, seed: int = 0):
    """
    Creates a single Gym environment instance based on configs.
    Uses MappingConfig to resolve config IDs to classes.
    """
    exp_id = run_config.EXPERIMENT_ID
    # Get config dict from MappingConfig
    # Use classmethod if available or access dict directly
    try:
        exp_def_full = MappingConfig.get_experiment_config(exp_id)
        # exp_def_full contains {env_id, wrapper_id, model_arch_id}
        env_id = exp_def_full['env_id']
        wrapper_id = exp_def_full['wrapper_id']
    except Exception:
        # Fallback if get_experiment_config is tricky or fails
        exp_def = MappingConfig.EXPERIMENT_MAP.get(exp_id)
        if not exp_def:
            raise ValueError(f"Experiment ID {exp_id} not found in MappingConfig.")
        env_id = exp_def['env_id']
        wrapper_id = exp_def['wrapper_id']

    env_class = MappingConfig.ENV_REGISTRY.get(env_id)
    if not env_class:
        raise ValueError(f"Env ID {env_id} not found in Registry.")
        
    wrapper_configs = MappingConfig.WRAPPER_REGISTRY.get(wrapper_id, [])
    
    # Instantiate Base Env
    # envs expect a config object (Class or Instance)
    env = env_class(env_config)
    # env.seed(seed) # Removed: Gymnasium Envs use reset(seed=...)
    
    # Apply Wrappers
    # wrapper_configs is list of (WrapperClass, kwargs)
    for wrapper_cls, kwargs in wrapper_configs:
        env = wrapper_cls(env, **kwargs)
        
    return env

def make_policy(run_config: RunConfig, env) -> BasePolicy:
    """
    Creates a Policy instance using ModelFactory and Configs.
    """
    exp_id = run_config.EXPERIMENT_ID
    # Resolve Model ID
    try:
        exp_def_full = MappingConfig.get_experiment_config(exp_id)
        model_id = exp_def_full['model_arch_id']
    except:
        exp_def = MappingConfig.EXPERIMENT_MAP.get(exp_id)
        model_id = exp_def['model_id']
    
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    
    device = run_config.DEVICE
    algo = run_config.ALGO
    
    # Get Actor
    actor = get_model(
        model_id=model_id,
        role='actor',
        state_shape=state_shape,
        action_shape=action_shape,
        algo=algo,
        device=device
    )
    
    # Get Critic (Double Critic for TD3/SAC)
    critic1 = get_model(
        model_id=model_id,
        role='critic',
        state_shape=state_shape,
        action_shape=action_shape,
        device=device
    )
    critic2 = get_model(
        model_id=model_id,
        role='critic',
        state_shape=state_shape,
        action_shape=action_shape,
        device=device
    )

    # 2. Initialize Policy
    policy_name = algo # 'ddpg', 'sac', 'td3'
    
    if policy_name == 'ddpg':
        # DDPG only uses 1 critic usually
        policy = CustomDDPGPolicy(
            actor=actor,
            actor_optim=torch.optim.Adam(actor.parameters(), lr=run_config.LR_ACTOR),
            critic=critic1,
            critic_optim=torch.optim.Adam(critic1.parameters(), lr=run_config.LR_CRITIC),
            sparsity_coef=run_config.SPARSITY_COEF,
            gamma=run_config.GAMMA,
            estimation_step=run_config.N_STEP,
            action_space=env.action_space
        )
    elif policy_name == 'td3':
        policy = CustomTD3Policy(
            actor=actor,
            actor_optim=torch.optim.Adam(actor.parameters(), lr=run_config.LR_ACTOR),
            critic1=critic1,
            critic1_optim=torch.optim.Adam(critic1.parameters(), lr=run_config.LR_CRITIC),
            critic2=critic2,
            critic2_optim=torch.optim.Adam(critic2.parameters(), lr=run_config.LR_CRITIC),
            sparsity_coef=run_config.SPARSITY_COEF,
            tau=run_config.TAU,
            gamma=run_config.GAMMA,
            exploration_noise=GaussianNoise(sigma=run_config.EXPLORATION_NOISE),
            policy_noise=run_config.POLICY_NOISE,
            update_actor_freq=run_config.UPDATE_ACTOR_FREQ,
            noise_clip=run_config.NOISE_CLIP,
            action_space=env.action_space
        )
    elif policy_name == 'sac':
        policy = CustomSACPolicy(
            actor=actor,
            actor_optim=torch.optim.Adam(actor.parameters(), lr=run_config.LR_ACTOR),
            critic1=critic1,
            critic1_optim=torch.optim.Adam(critic1.parameters(), lr=run_config.LR_CRITIC),
            critic2=critic2,
            critic2_optim=torch.optim.Adam(critic2.parameters(), lr=run_config.LR_CRITIC),
            sparsity_coef=run_config.SPARSITY_COEF,
            tau=run_config.TAU,
            gamma=run_config.GAMMA,
            alpha=run_config.ALPHA,
            estimation_step=run_config.N_STEP,
            action_space=env.action_space
        )
    elif policy_name == 'diffusion':
        policy = CustomDiffusionPolicy(
            state_dim=state_shape,
            actor=actor,
            actor_optim=torch.optim.Adam(actor.parameters(), lr=run_config.LR_ACTOR),
            action_dim=action_shape,
            critic=critic1,
            critic_optim=torch.optim.Adam(critic1.parameters(), lr=run_config.LR_CRITIC),
            device=device,
            tau=run_config.TAU,
            gamma=run_config.GAMMA,
            reward_normalization=run_config.REWARD_NORMALIZATION,
            estimation_step=run_config.N_STEP,
            lr_decay=getattr(run_config, 'LR_DECAY', False),
            lr_maxt=getattr(run_config, 'EPOCH', 1000),
            bc_coef=getattr(run_config, 'BC_COEF', False),
            exploration_noise=run_config.EXPLORATION_NOISE,
            action_space=env.action_space
        )
    else:
        raise ValueError(f"Unknown Policy: {policy_name}")

    return policy

