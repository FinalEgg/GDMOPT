#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 Top-K DDPG 模型推测结果
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
    parser = argparse.ArgumentParser(description='Visualize Top-K DDPG model inference')
    parser.add_argument('--run-name', type=str, default='Dec11-000306',
                        help='Run name (e.g., Dec10-123456)')
    parser.add_argument('--env', type=str, default='topk',
                        help='Environment name (default: topk)')
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

def get_env_and_config(env_name):
    """加载环境和配置"""
    try:
        if env_name == 'topk':
            from env.topk.env import TopKEnv
            import env.topk.config as config
            env = TopKEnv()
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
        return env, config
    except ImportError as e:
        print(f"Error loading environment '{env_name}': {e}")
        sys.exit(1)

def load_model(run_name, state_dim, action_dim, config):
    """加载 DDPG Top-K 模型"""
    run_path = find_run_path(run_name)
    
    possible_paths = []
    if run_path:
        possible_paths.append(os.path.join(run_path, 'policy.pth'))
        possible_paths.append(os.path.join(run_path, 'best_policy.pth'))
    
    # 备用路径
    possible_paths.extend([
        f'log/default/ddpg/topk/{run_name}/policy.pth',
        f'log/{run_name}/policy.pth'
    ])
    
    log_path = None
    for path in possible_paths:
        if os.path.exists(path):
            log_path = path
            break
            
    if log_path is None:
        print(f"Error: Could not find model weights in any of these locations:")
        for p in possible_paths[:5]:
            print(f" - {p}")
        log_path = possible_paths[0] # Fallback

    print(f"Loading model from: {log_path}")

    from policy.ddpg.ddpg import DDPG
    from model.ddpg_topk.ddpg import Actor, Critic
    
    # 构造 args 对象以传递给 Actor/Critic
    class Args:
        pass
    args = Args()
    # 这些参数在 Actor/Critic 内部其实不直接用，而是用 config 中的 M, N, K_MAX
    # 但为了保持接口一致性，我们还是传进去
    
    actor = Actor(args)
    critic = Critic(args)
    
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

    if os.path.exists(log_path):
        ckpt = torch.load(log_path, map_location='cpu')
        policy.load_state_dict(ckpt, strict=False)
        print(f"Loaded model weights successfully.")
    else:
        print(f"Warning: Model path {log_path} not found, using random initialized model.")

    policy.eval()
    return policy

def infer_actions(policy, state, config):
    """使用模型推测动作"""
    state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    
    with torch.no_grad():
        batch = Batch(obs=state_tensor)
        # Actor 返回 (action, state)
        result = policy.forward(batch)
        
        if hasattr(result, 'act'):
            actions = result.act
        elif isinstance(result, tuple):
            actions = result[0]
        else:
            actions = result
        
        if isinstance(actions, torch.Tensor):
            actions = actions.squeeze(0).numpy()
            
    return actions

def save_weights(policy, run_name):
    """保存模型权重到txt文件"""
    log_dir = f'log/default/ddpg/topk/{run_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    output_path = os.path.join(log_dir, 'model_weights.txt')
    
    with open(output_path, 'w') as f:
        f.write(f"Model Weights for DDPG Top-K (Run: {run_name})\n")
        f.write("="*80 + "\n\n")
        
        model = policy._actor
            
        for name, param in model.named_parameters():
            f.write(f"Parameter: {name}\n")
            f.write(f"Shape: {list(param.shape)}\n")
            f.write("Values:\n")
            np.set_printoptions(threshold=np.inf, linewidth=200)
            f.write(str(param.data.cpu().numpy()))
            f.write("\n" + "-"*80 + "\n")
            
    print(f"Model weights saved to {output_path}")

def visualize(env, bs_positions, uav_positions, power_matrix, config):
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
    # 使用环境真实的 connection_matrix 和 power_matrix
    connection_matrix = env.connection_matrix
    M, N = config.M, config.N
    max_power = np.max(power_matrix)
    
    for m in range(M):
        for n in range(N):
            if connection_matrix[m, n] > 0.5:  # 连接存在
                power = power_matrix[m, n]
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

    # 1. 加载环境和配置
    env, config = get_env_and_config(args.env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. 生成随机状态
    state, _ = env.reset()
    
    bs_positions = env.bs_positions
    uav_positions = env.uav_positions

    print(f"Environment: {args.env}")
    print(f"Config: M={config.M}, N={config.N}, X={config.X}, Y={config.Y}, K_MAX={config.K_MAX}")
    print(f"Generated state shape: {state.shape}")

    # 3. 加载模型
    policy = load_model(args.run_name, state_dim, action_dim, config)

    if args.save_weights:
        save_weights(policy, args.run_name)

    # 4. 推测动作
    raw_actions = infer_actions(policy, state, config)
    
    # 5. 应用动作到环境以计算奖励和更新状态
    # TopKEnv.step 接受 (N * K_MAX) 的动作
    _, reward, _, _, _ = env.step(raw_actions)
    
    # 获取经过环境处理后的真实功率矩阵（已映射回 M*N 并归一化）
    final_power_matrix = env.power_matrix

    print(f"Raw actions shape: {raw_actions.shape}")
    print(f"Final power matrix shape: {final_power_matrix.shape}")

    # 打印详细动作矩阵
    print("\nPower Actions (BS-UAV matrix) [After Mapping & Norm]:")
    for m in range(config.M):
        row = [f"{final_power_matrix[m, n]:.2f}" for n in range(config.N)]
        print(f"BS{m}: {' '.join(row)}")
        
    print("\nConnection Matrix (Top-K Rule):")
    for m in range(config.M):
        row = [f"{int(env.connection_matrix[m, n])}" for n in range(config.N)]
        print(f"BS{m}: {' '.join(row)}")

    # 5. 打印奖励
    print(f"\nCalculated Reward: {reward}")
    _, capacity = env._calculate_capacity()
    print(f"Total Capacity: {capacity:.4f}")
    print(f"Baseline Capacity: {env.baseline_capacity:.4f}")

    # 6. 可视化
    visualize(env, bs_positions, uav_positions, final_power_matrix, config)

if __name__ == "__main__":
    main()
