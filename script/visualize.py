#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 CellFree UAV 环境的状态和模型推测结果
通用版本，支持多种模型类型
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
from env.cellfree.env import CellFreeEnv
from env.cellfree.config import M, N, X, Y, H

def get_args():
    parser = argparse.ArgumentParser(description='Visualize CellFree UAV model inference')
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['ddpg', 'sac', 'diffusion_opt', 'combined'],
                        help='Model algorithm type')
    parser.add_argument('--run-name', type=str, required=True,
                        help='Run name (e.g., Nov13-145717)')
    parser.add_argument('--env', type=str, default='cellfree',
                        help='Environment name')
    parser.add_argument('--save-weights', action='store_true',
                        help='Save model weights to txt file')
    return parser.parse_args()

def load_model(algorithm, run_name, env_name, state_dim, action_dim):
    """根据算法类型加载模型"""
    # 支持新的日志路径结构 (combined/sac/cellfree/...)
    # 尝试多种可能的路径
    possible_paths = [
        f'log/combined/{algorithm}/{env_name}/{run_name}/finetune_policy.pth', # 新结构 (微调后)
        f'log/combined/{algorithm}/{env_name}/{run_name}/pretrain_policy.pth', # 新结构 (预训练)
        f'log/default/{algorithm}/{env_name}/{run_name}/policy.pth',           # 旧结构
        f'log/{run_name}/policy.pth'                                            # 简单结构
    ]
    
    log_path = None
    for path in possible_paths:
        if os.path.exists(path):
            log_path = path
            break
            
    if log_path is None:
        print(f"Error: Could not find model weights in any of these locations:")
        for p in possible_paths:
            print(f" - {p}")
        # Fallback to default for error message consistency
        log_path = possible_paths[0]

    debug_file = os.path.join(os.path.dirname(log_path), 'debug_output.txt')

    if algorithm == 'combined':
        from model.combined.combined_model import CombinedModel
        from policy.combined.combined import CombinedOPT
        connection_dim = M * N
        power_dim = M * N
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
        from policy.ddpg.ddpg_show import DDPG
        from model.actor.actor import Actor
        from model.diffusion.model import DoubleCritic
        actor = Actor(state_dim, action_dim)
        critic = DoubleCritic(state_dim, action_dim)
        policy = DDPG(
            state_dim=state_dim,
            actor=actor,
            actor_optim=None,
            action_dim=action_dim,
            critic=critic,
            critic_optim=None,
            tau=0.005,
            gamma=1.0,
            device='cpu',
            debug_file=debug_file
        )
    elif algorithm == 'sac':
        from policy.sac.sac import SAC
        from model.sac.sac import Actor
        # 注意：Actor 的 hidden_dim 必须与训练时一致 (256)
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
        # Provide dummy optimizer for inference
        dummy_optim = torch.optim.SGD(actor.parameters(), lr=0.01)
        policy = DiffusionOPT(
            state_dim=state_dim,
            actor=actor,
            actor_optim=dummy_optim,  # Dummy optimizer
            action_dim=action_dim,
            critic=critic,
            critic_optim=dummy_optim,  # Dummy optimizer
            device='cpu',
            tau=0.005,
            gamma=1.0
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if os.path.exists(log_path):
        ckpt = torch.load(log_path, map_location='cpu')
        policy.load_state_dict(ckpt, strict=False)
        print(f"Loaded model from {log_path}")
    else:
        print(f"Model path {log_path} not found, using random model")

    policy.eval()
    return policy

def infer_actions(policy, algorithm, state):
    """使用模型推测动作"""
    from tianshou.data import Batch
    state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
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
            # SAC 输出可能是 (action, log_prob) 元组，或者 Batch 对象
            # Tianshou 的 forward 返回 Batch，其中 act 是动作
            if hasattr(result, 'act'):
                actions = result.act
            else:
                actions = result[0] # 假设是元组 (action, state)
                
            if isinstance(actions, torch.Tensor):
                actions = actions.squeeze(0).numpy()
            
            # 动作是功率矩阵 (M*N)
            power_actions = actions.reshape(M, N)
            
            # 应用软门控逻辑来决定连接状态 (与 env.py 保持一致)
            from env.cellfree.config import GATE_TH
            
            # 新逻辑：Subtract & Rescale (ReLU-like)
            effective_power = np.maximum(0, power_actions - GATE_TH)
            if GATE_TH < 1.0:
                effective_power = effective_power / (1.0 - GATE_TH)
            
            # 更新 power_actions 为有效功率，以便可视化真实效果
            power_actions = effective_power
            
            connection_actions = (effective_power > 0.001).astype(float)
            
    return connection_actions, power_actions

def generate_random_state():
    """生成随机状态"""
    env = CellFreeEnv()
    state, _ = env.reset()
    return state, env.bs_positions, env.uav_positions

def calculate_reward_from_actions(env, connection_actions, power_actions):
    """根据动作计算奖励"""
    env.connection_matrix = connection_actions
    env.power_matrix = power_actions
    # 归一化功率分配：每个基站的总功率分配之和为1
    for m in range(M):
        total_power = np.sum(env.power_matrix[m, :])
        if total_power > 0:
            env.power_matrix[m, :] /= total_power
            
    # 兼容新的环境接口 (reward_mode)
    # 默认使用物理奖励进行评估
    if hasattr(env, '_calculate_physical_reward'):
        return env._calculate_physical_reward()
    elif hasattr(env, '_calculate_reward'):
        return env._calculate_reward()
    else:
        raise AttributeError("Environment has no reward calculation method")

def save_weights(policy, algorithm, run_name, env_name):
    """保存模型权重到txt文件"""
    log_dir = f'log/default/{algorithm}/{env_name}/{run_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    output_path = os.path.join(log_dir, 'model_weights.txt')
    
    with open(output_path, 'w') as f:
        f.write(f"Model Weights for {algorithm} (Run: {run_name})\n")
        f.write("="*80 + "\n\n")
        
        # 获取核心网络
        if algorithm == 'sac':
            # SAC主要关注Actor网络
            # 注意：policy._actor 是在 SAC.__init__ 中定义的
            model = policy._actor
        elif algorithm == 'ddpg':
            model = policy._actor
        elif algorithm == 'combined':
            model = policy.actor
        elif algorithm == 'diffusion_opt':
            model = policy.actor
        else:
            model = policy
            
        # 遍历参数
        for name, param in model.named_parameters():
            f.write(f"Parameter: {name}\n")
            f.write(f"Shape: {list(param.shape)}\n")
            f.write("Values:\n")
            # 设置打印选项以显示更多内容
            np.set_printoptions(threshold=np.inf, linewidth=200)
            f.write(str(param.data.cpu().numpy()))
            f.write("\n" + "-"*80 + "\n")
            
    print(f"Model weights saved to {output_path}")

def visualize(bs_positions, uav_positions, connection_actions, power_actions):
    """可视化"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 设置坐标轴
    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('CellFree UAV Network Visualization')
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

    # 获取环境参数
    env = CellFreeEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 生成随机状态
    state, _ = env.reset()
    bs_positions = env.bs_positions
    uav_positions = env.uav_positions
    
    print(f"Generated state shape: {state.shape}")
    print(f"BS positions: {bs_positions}")
    print(f"UAV positions: {uav_positions}")

    # 加载模型
    policy = load_model(args.algorithm, args.run_name, args.env, state_dim, action_dim)

    if args.save_weights:
        save_weights(policy, args.algorithm, args.run_name, args.env)

    # 推测动作
    connection_actions, power_actions = infer_actions(policy, args.algorithm, state)
    print(f"Connection actions shape: {connection_actions.shape}")
    print(f"Power actions shape: {power_actions.shape}")

    # 打印详细动作矩阵
    print("\nConnection Actions (BS-UAV matrix):")
    for m in range(M):
        row = [f"{connection_actions[m, n]:.2f}" for n in range(N)]
        print(f"BS{m}: {' '.join(row)}")

    print("\nPower Actions (BS-UAV matrix):")
    for m in range(M):
        row = [f"{power_actions[m, n]:.2f}" for n in range(N)]
        print(f"BS{m}: {' '.join(row)}")

    # 计算并打印奖励
    reward = calculate_reward_from_actions(env, connection_actions, power_actions)
    print(f"\nPredicted Reward: {reward}")

    # 可视化
    visualize(bs_positions, uav_positions, connection_actions, power_actions)

if __name__ == "__main__":
    main()