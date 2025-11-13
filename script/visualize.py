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
    return parser.parse_args()

def load_model(algorithm, run_name, env_name, state_dim, action_dim):
    """根据算法类型加载模型"""
    log_path = f'log/default/{algorithm}/{env_name}/{run_name}/policy.pth'

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
        from policy.ddpg.ddpg import DDPG
        from model.actor.actor import Actor
        actor = Actor(state_dim, action_dim)
        policy = DDPG(
            state_dim=state_dim,
            actor=actor,
            actor_optim=None,
            action_dim=action_dim,
            critic=None,
            critic_optim=None,
            tau=0.005,
            gamma=1.0,
            device='cpu'
        )
    elif algorithm == 'sac':
        from policy.sac.sac import SAC
        from model.sac.sac import SACModel
        model = SACModel(state_dim, action_dim)
        policy = SAC(
            state_dim=state_dim,
            actor=model,
            actor_optim=None,
            action_dim=action_dim,
            critic=None,
            critic_optim=None,
            value=None,
            value_optim=None,
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
        policy = DiffusionOPT(
            state_dim=state_dim,
            actor=actor,
            actor_optim=None,
            action_dim=action_dim,
            critic=critic,
            critic_optim=None,
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

def generate_random_state():
    """生成随机状态"""
    env = CellFreeEnv()
    state, _ = env.reset()
    return state, env.bs_positions, env.uav_positions

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
            # 对于其他算法，动作是连续的
            batch = Batch(obs=state_tensor)
            result = policy.forward(batch)
            actions = result.act.squeeze(0).numpy()
            # 假设动作是功率，连接基于阈值
            connection_actions = (actions[:M*N] > 0.5).reshape(M, N).astype(float)
            power_actions = actions[M*N:].reshape(M, N)
    return connection_actions, power_actions

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
    state, bs_positions, uav_positions = generate_random_state()
    print(f"Generated state shape: {state.shape}")
    print(f"BS positions: {bs_positions}")
    print(f"UAV positions: {uav_positions}")

    # 加载模型
    policy = load_model(args.algorithm, args.run_name, args.env, state_dim, action_dim)

    # 推测动作
    connection_actions, power_actions = infer_actions(policy, args.algorithm, state)
    print(f"Connection actions shape: {connection_actions.shape}")
    print(f"Power actions shape: {power_actions.shape}")

    # 可视化
    visualize(bs_positions, uav_positions, connection_actions, power_actions)

if __name__ == "__main__":
    main()