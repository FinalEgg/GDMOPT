#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 CellFree UAV 环境的状态和模型推测结果
通用版本，支持多种模型类型，并根据环境名称动态加载配置
"""

import sys
import os
import argparse
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tianshou.data import Batch

def get_args():
    parser = argparse.ArgumentParser(description='Visualize CellFree UAV model inference')
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['ddpg', 'sac', 'diffusion_opt', 'combined'],
                        help='Model algorithm type')
    parser.add_argument('--run-name', type=str, required=True,
                        help='Run name (e.g., Nov13-145717)')
    parser.add_argument('--env', type=str, default=None,
                        help='Environment name (cellfree, threshold, etc.). If None, tries to detect from run path.')
    parser.add_argument('--save-weights', action='store_true',
                        help='Save model weights to txt file')
    return parser.parse_args()

def find_run_path(run_name):
    """Search for the run directory in log folder"""
    log_root = 'log'
    for root, dirs, files in os.walk(log_root):
        if run_name in dirs:
            return os.path.join(root, run_name)
    return None

def detect_env_from_path(run_path):
    """Extract environment name from run path"""
    # Path structure: log/prefix/algorithm/env_name/run_name
    # or log/combined/algorithm/env_name/run_name
    parts = os.path.normpath(run_path).split(os.sep)
    # Assuming run_name is the last part
    if len(parts) >= 2:
        # env_name should be the parent of run_name
        # But wait, structure is .../env_name/run_name
        return parts[-2]
    return 'cellfree' # Default fallback

def get_env_and_config(env_name, run_name=None):
    """根据环境名称动态加载环境类和配置模块"""
    
    # Auto-detect if env_name is not provided
    if env_name is None and run_name is not None:
        run_path = find_run_path(run_name)
        if run_path:
            detected_env = detect_env_from_path(run_path)
            print(f"Auto-detected environment: {detected_env} from {run_path}")
            env_name = detected_env
        else:
            print(f"Could not find run {run_name}, defaulting to 'cellfree'")
            env_name = 'cellfree'
    elif env_name is None:
        env_name = 'cellfree'

    try:
        if env_name == 'cellfree':
            from env.cellfree.env import CellFreeEnv
            import env.cellfree.config as config
            env = CellFreeEnv()
        elif env_name == 'threshold':
            from env.threshold.env import ThresholdEnv
            import env.threshold.config as config
            env = ThresholdEnv()
        elif env_name == 'optimization':
            from env.optimization.env import OptimizationEnv
            import env.optimization.config as config
            env = OptimizationEnv()
        elif env_name == 'pendulum':
            from env.pendulum.env import PendulumEnv
            import env.pendulum.config as config # 假设存在
            env = PendulumEnv()
        else:
            # 尝试通用导入模式
            env_module = importlib.import_module(f'env.{env_name}.env')
            config_module = importlib.import_module(f'env.{env_name}.config')
            # 假设环境类名是 EnvNameEnv (例如 ThresholdEnv)
            class_name = f"{env_name.capitalize()}Env"
            if hasattr(env_module, class_name):
                env_class = getattr(env_module, class_name)
                env = env_class()
            else:
                # 尝试找任何继承自 gym.Env 的类，或者直接实例化第一个类
                # 这里简化处理，如果找不到特定类名，可能需要手动添加支持
                raise ValueError(f"Could not find environment class for {env_name}")
            config = config_module
            
        return env, config
    except ImportError as e:
        print(f"Error loading environment '{env_name}': {e}")
        sys.exit(1)

def load_model(algorithm, run_name, env_name, state_dim, action_dim, config):
    """根据算法类型加载模型"""
    # 优先尝试自动搜索路径
    run_path = find_run_path(run_name)
    
    possible_paths = []
    if run_path:
        possible_paths.append(os.path.join(run_path, 'finetune_policy.pth'))
        possible_paths.append(os.path.join(run_path, 'pretrain_policy.pth'))
        possible_paths.append(os.path.join(run_path, 'policy.pth'))
    
    # 备用路径 (保留旧逻辑)
    possible_paths.extend([
        f'log/combined/{algorithm}/{env_name}/{run_name}/finetune_policy.pth',
        f'log/combined/{algorithm}/{env_name}/{run_name}/pretrain_policy.pth',
        f'log/default/{algorithm}/{env_name}/{run_name}/finetune_policy.pth',
        f'log/default/{algorithm}/{env_name}/{run_name}/pretrain_policy.pth',
        f'log/default/{algorithm}/{env_name}/{run_name}/policy.pth',
        f'log/{run_name}/policy.pth'
    ])
    
    log_path = None
    for path in possible_paths:
        if os.path.exists(path):
            log_path = path
            break
            
    if log_path is None:
        print(f"Error: Could not find model weights in any of these locations:")
        for p in possible_paths[:5]: # 只打印前几个
            print(f" - {p}")
        if run_path:
             print(f" - (and other paths in {run_path})")
        log_path = possible_paths[0] # Fallback

    print(f"Loading model from: {log_path}")

    if algorithm == 'combined':
        from model.combined.combined_model import CombinedModel
        from policy.combined.combined import CombinedOPT
        connection_dim = config.M * config.N
        power_dim = config.M * config.N
        model = CombinedModel(
            state_dim=state_dim,
            connection_dim=connection_dim,
            power_dim=power_dim
        )
        policy = CombinedOPT(
            state_dim=state_dim,
            actor=model,
            actor_optim=None,
            connection_dim=connection_dim,
            power_dim=power_dim,
            critic=None,
            critic_optim=None,
            device='cpu'
        )
    elif algorithm == 'ddpg':
        from policy.ddpg.ddpg import DDPG
        from model.ddpg.ddpg import Actor, DuelingCritic
        
        # 初始化网络 (注意：训练脚本 train_ddpg.py 中使用的是 hidden_dim=256)
        actor = Actor(state_dim, action_dim, hidden_dim=256)
        critic = DuelingCritic(state_dim, action_dim, hidden_dim=256)
        
        actor_optim = torch.optim.AdamW(actor.parameters())
        critic_optim = torch.optim.AdamW(critic.parameters())
        
        policy = DDPG(
            state_dim=state_dim,
            actor=actor,
            actor_optim=actor_optim,
            action_dim=action_dim,
            critic=critic,
            critic_optim=critic_optim,
            device='cpu'
        )
    elif algorithm == 'sac':
        from policy.sac.sac import SAC
        from model.sac.sac import Actor
        model = Actor(state_dim, action_dim, hidden_dim=256)
        policy = SAC(
            state_dim=state_dim,
            actor=model,
            actor_optim=None,
            action_dim=action_dim,
            critic=None,
            critic_optim=None,
            device='cpu'
        )
    elif algorithm == 'diffusion_opt':
        from policy.diffusion_opt.diffusion_opt import DiffusionOPT
        from model.diffusion.diffusion import Diffusion
        from model.diffusion.model import MLP
        from model.diffusion.model import DoubleCritic
        actor_net = MLP(state_dim=state_dim, action_dim=action_dim)
        actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=actor_net,
            max_action=1.0,
            n_timesteps=6,
            beta_schedule='vp'
        )
        critic = DoubleCritic(state_dim=state_dim, action_dim=action_dim)
        dummy_optim = torch.optim.SGD(actor.parameters(), lr=0.01)
        policy = DiffusionOPT(
            state_dim=state_dim,
            actor=actor,
            actor_optim=dummy_optim,
            action_dim=action_dim,
            critic=critic,
            critic_optim=dummy_optim,
            device='cpu',
            tau=0.005,
            gamma=1.0
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if os.path.exists(log_path):
        ckpt = torch.load(log_path, map_location='cpu')
        policy.load_state_dict(ckpt, strict=False)
        print(f"Loaded model weights successfully.")
    else:
        print(f"Warning: Model path {log_path} not found, using random initialized model.")

    policy.eval()
    return policy

def infer_actions(policy, algorithm, state, config):
    """使用模型推测动作"""
    state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    M, N = config.M, config.N
    
    with torch.no_grad():
        if algorithm == 'combined':
            batch = Batch(obs=state_tensor)
            result = policy.forward(batch)
            actions = result.act.squeeze(0).numpy()
            # 分离连接和功率
            connection_actions = actions[:M*N].reshape(M, N)
            power_actions = actions[M*N:].reshape(M, N)
        else:
            # 对于其他算法，动作是功率矩阵
            batch = Batch(obs=state_tensor)
            result = policy.forward(batch)
            
            if hasattr(result, 'act'):
                actions = result.act
            else:
                actions = result[0]
            
            # 处理 Actor 返回的元组 (action, gate_prob)
            if isinstance(actions, tuple):
                actions = actions[0]

            if isinstance(actions, torch.Tensor):
                actions = actions.squeeze(0).numpy()
            
            # 动作是功率矩阵 (M*N)
            power_actions = actions.reshape(M, N)
            
            # 应用软门控逻辑来决定连接状态
            # 尝试从 config 获取 GATE_TH，如果没有则默认为 0 (即所有非零功率都算连接)
            gate_th = getattr(config, 'GATE_TH', 0.0)
            
            # 新逻辑：Subtract & Rescale (ReLU-like)
            effective_power = np.maximum(0, power_actions - gate_th)
            if gate_th < 1.0 and gate_th > 0.0:
                effective_power = effective_power / (1.0 - gate_th)
            
            # 更新 power_actions 为有效功率
            power_actions = effective_power
            
            connection_actions = (effective_power > 0.001).astype(float)
            
    return connection_actions, power_actions

def calculate_reward_from_actions(env, connection_actions, power_actions):
    """根据动作计算奖励"""
    # 注意：这里假设 env 实例有 connection_matrix 和 power_matrix 属性
    # 不同的 env 实现可能不同，这里主要针对 CellFree 和 Threshold
    if hasattr(env, 'connection_matrix'):
        env.connection_matrix = connection_actions
    if hasattr(env, 'power_matrix'):
        env.power_matrix = power_actions
        
    # 归一化功率分配：每个基站的总功率分配之和为1 (如果环境需要)
    # 这里简单处理，如果 env 有 power_matrix，尝试归一化
    if hasattr(env, 'power_matrix'):
        M = env.power_matrix.shape[0]
        for m in range(M):
            total_power = np.sum(env.power_matrix[m, :])
            if total_power > 0:
                env.power_matrix[m, :] /= total_power
            
    # 兼容新的环境接口 (reward_mode)
    if hasattr(env, '_calculate_physical_reward'):
        return env._calculate_physical_reward()
    elif hasattr(env, '_calculate_reward'):
        return env._calculate_reward()
    else:
        print("Warning: Environment has no known reward calculation method.")
        return 0.0

def save_weights(policy, algorithm, run_name, env_name):
    """保存模型权重到txt文件"""
    log_dir = f'log/default/{algorithm}/{env_name}/{run_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    output_path = os.path.join(log_dir, 'model_weights.txt')
    
    with open(output_path, 'w') as f:
        f.write(f"Model Weights for {algorithm} (Run: {run_name})\n")
        f.write("="*80 + "\n\n")
        
        if algorithm == 'sac':
            model = policy._actor
        elif algorithm == 'ddpg':
            model = policy._actor
        elif algorithm == 'combined':
            model = policy.actor
        elif algorithm == 'diffusion_opt':
            model = policy.actor
        else:
            model = policy
            
        for name, param in model.named_parameters():
            f.write(f"Parameter: {name}\n")
            f.write(f"Shape: {list(param.shape)}\n")
            f.write("Values:\n")
            np.set_printoptions(threshold=np.inf, linewidth=200)
            f.write(str(param.data.cpu().numpy()))
            f.write("\n" + "-"*80 + "\n")
            
    print(f"Model weights saved to {output_path}")

def visualize(bs_positions, uav_positions, connection_actions, power_actions, config):
    """可视化"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 设置坐标轴
    X, Y = config.X, config.Y
    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Network Visualization (Env: {config.__name__})')
    ax.grid(True, alpha=0.3)

    # 绘制基站
    ax.scatter(bs_positions[:, 0], bs_positions[:, 1], color='red', s=50, label='Base Station', alpha=0.8)
    for i, (x, y) in enumerate(bs_positions):
        ax.text(x, y+3, f'BS{i}', ha='center', va='bottom', fontsize=8, color='red')

    # 绘制无人机
    ax.scatter(uav_positions[:, 0], uav_positions[:, 1], color='blue', s=50, label='UAV', alpha=0.8)
    for i, (x, y, z) in enumerate(uav_positions):
        ax.text(x, y+3, f'UAV{i}', ha='center', va='bottom', fontsize=8, color='blue')

    # 绘制连接
    M, N = config.M, config.N
    max_power = np.max(power_actions)
    for m in range(M):
        for n in range(N):
            if connection_actions[m, n] > 0.5:  # 连接阈值
                power = power_actions[m, n]
                # 颜色强度基于功率
                alpha = min(1.0, power / max_power) if max_power > 0 else 0.5
                color = (0, 1, 0, alpha)  # 绿色，透明度表示功率
                bs_x, bs_y = bs_positions[m]
                uav_x, uav_y, _ = uav_positions[n]
                ax.plot([bs_x, uav_x], [bs_y, uav_y], color=color, linewidth=2, alpha=alpha)

    # 图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.show()

def main():
    args = get_args()

    # 1. 动态加载环境和配置
    env, config = get_env_and_config(args.env, args.run_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. 生成随机状态
    state, _ = env.reset()
    
    # 尝试获取位置信息，不同环境可能属性名不同
    if hasattr(env, 'bs_positions'):
        bs_positions = env.bs_positions
    else:
        # Fallback: 尝试从 config 获取或者随机生成用于可视化
        print("Warning: env.bs_positions not found. Using random positions for visualization.")
        bs_positions = np.random.rand(config.M, 2) * np.array([config.X, config.Y])

    if hasattr(env, 'uav_positions'):
        uav_positions = env.uav_positions
    else:
        print("Warning: env.uav_positions not found. Using random positions for visualization.")
        uav_positions = np.random.rand(config.N, 3) * np.array([config.X, config.Y, config.H])

    print(f"Environment: {args.env if args.env else 'Auto-detected'}")
    print(f"Config: M={config.M}, N={config.N}, X={config.X}, Y={config.Y}")
    print(f"Generated state shape: {state.shape}")

    # 3. 加载模型
    policy = load_model(args.algorithm, args.run_name, args.env, state_dim, action_dim, config)

    if args.save_weights:
        save_weights(policy, args.algorithm, args.run_name, args.env)

    # 4. 推测动作
    connection_actions, power_actions = infer_actions(policy, args.algorithm, state, config)
    print(f"Connection actions shape: {connection_actions.shape}")
    print(f"Power actions shape: {power_actions.shape}")

    # 打印详细动作矩阵
    print("\nConnection Actions (BS-UAV matrix):")
    for m in range(config.M):
        row = [f"{connection_actions[m, n]:.2f}" for n in range(config.N)]
        print(f"BS{m}: {' '.join(row)}")

    print("\nPower Actions (BS-UAV matrix):")
    for m in range(config.M):
        row = [f"{power_actions[m, n]:.2f}" for n in range(config.N)]
        print(f"BS{m}: {' '.join(row)}")

    # 5. 计算并打印奖励
    reward = calculate_reward_from_actions(env, connection_actions, power_actions)
    print(f"\nPredicted Reward: {reward}")

    # 6. 可视化
    visualize(bs_positions, uav_positions, connection_actions, power_actions, config)

if __name__ == "__main__":
    main()
