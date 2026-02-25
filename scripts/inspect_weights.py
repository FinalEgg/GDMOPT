"""
Weight & Activation Inspection Script.
权重与激活值检查脚本。

The script inspects the model's weights and the activation distributions during inference.
This is useful for debugging exploding/vanishing gradients or dead neurons.
该脚本用于检查模型权重以及推理过程中的激活值分布。这对于调试梯度爆炸/消失或神经元死亡非常有用。
"""

import sys
import os
import torch
import numpy as np
import collections
from tianshou.data import Batch

# Add project root to path
# 将项目根目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.setup import make_env_instance, make_policy
from config.env_config import EnvConfig
from config.run_config import RunConfig

def inspect_weights(checkpoint_path: str = None):
    """
    Inspect model weights and activations.
    检查模型权重和激活值。

    Analysis steps:
    1. Instantiates environment and policy.
    2. Hooks into Actor network layers (Linear/Conv).
    3. Runs a single forward pass with random observation.
    4. Reports mean/std/min/max of activations (checking for saturation/dead relus).
    5. Reports mean/std of weights.

    分析步骤:
    1. 实例化环境和策略。
    2. Hook 到 Actor 网络的层 (Linear/Conv)。
    3. 使用随机观测运行单次前向传播。
    4. 报告激活值的均值/标准差/最值 (检查饱和度/ReLU死亡)。
    5. 报告权重的均值/标准差。
    """
    print("=== Weight & Activation Inspection Script ===")
    print("=== 权重与激活值检查脚本 ===")

    # 1. Setup
    # 1. 设置
    env = make_env_instance(RunConfig, EnvConfig, seed=123)
    policy = make_policy(RunConfig, env)

    # 2. Load
    # 2. 加载
    if checkpoint_path:
        print(f"Loading (加载): {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=RunConfig.DEVICE)
            policy.load_state_dict(checkpoint['policy'])
        except Exception as e:
            print(f"Failed to load (加载失败): {e}")
            return
    else:
        print("[INFO] No checkpoint. Inspecting initialized weights.")
        print("[信息] 无检查点。检查初始化权重。")

    policy.eval()
    
    # 3. Register Hooks for Activation Stats
    # 3. 注册 Hook 以获取激活统计信息
    activations = collections.defaultdict(list)

    def get_activation_hook(name):
        """
        Create a hook to record output statistics.
        创建一个 Hook 来记录输出统计信息。
        """
        def hook(model, input, output):
            # output can be tensor or tuple
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            
            # Detach and move to CPU
            data = out_tensor.detach().cpu().numpy()
            activations[name] = data
        return hook

    # Hook into Actor's modules
    # Hook 到 Actor 的模块中
    # We iterate named_modules and hook 'Linear' or 'ReLU' layers usually
    # 我们通常遍历 named_modules 并 Hook 'Linear' 或 'ReLU' 层
    
    print("\n--- Registering Hooks on Actor (在 Actor 上注册 Hooks) ---")
    # Tianshou policy wraps model in `policy.actor`
    actor_model = policy.actor
    
    for name, module in actor_model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            print(f"Hooking: {name} ({type(module).__name__})")
            module.register_forward_hook(get_activation_hook(name))
            
    # 4. Forward Pass
    # 4. 前向传播
    obs, _ = env.reset()
    batch = Batch(obs=[obs], info={})
    
    print("\n--- Running Forward Pass (运行前向传播) ---")
    with torch.no_grad():
        policy(batch)
        
    # 5. Analyze
    # 5. 分析结果
    print("\n--- Activation Statistics (激活值统计) ---")
    print(f"{'Layer':<40} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'% Zoes/Sat'}")
    print("-" * 100)
    
    for name, data in activations.items():
        if data.size == 0: continue
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Saturation checks (e.g. Tanh +/- 1, ReLU 0)
        # 饱和度检查 (例如 Tanh +/- 1, ReLU 0)
        # Roughly check near 0
        zeros_pct = np.sum(np.abs(data) < 1e-6) / data.size * 100
        
        print(f"{name:<40} | {mean_val:>10.4f} | {std_val:>10.4f} | {min_val:>10.4f} | {max_val:>10.4f} | {zeros_pct:>6.1f}%")

    print("\n--- Weight Statistics (权重统计) ---")
    print(f"{'Layer':<40} | {'Mean':<10} | {'Std':<10} | {'AbsMean > 0'}")
    print("-" * 100)
    for name, param in actor_model.named_parameters():
         if param.requires_grad:
            data = param.detach().cpu().numpy()
            mean_val = np.mean(data)
            std_val = np.std(data)
            abs_mean = np.mean(np.abs(data))
            
            print(f"{name:<40} | {mean_val:>10.4f} | {std_val:>10.4f} | {abs_mean:>10.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect model weights.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()
    
    inspect_weights(args.checkpoint)
