"""
Model Training Launcher / 模型训练启动器
==================================================
Function / 功能:
    Launches training for RL models using Tianshou OffpolicyTrainer.
    使用 Tianshou OffpolicyTrainer 启动 RL 若模型训练。

Description / 描述:
    A streamlined training script leveraging configuration classes (RunConfig, EnvConfig)
    and utility factories (make_env_instance, make_policy).
    利用配置类 (RunConfig, EnvConfig) 和工厂工具 (make_env_instance, make_policy) 的简化训练脚本。
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Tianshou
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger

# Add project root to path / 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project modules / 项目模块
from scripts.utils.setup import make_env_instance, make_policy
from scripts.utils.config_manager import ConfigManager # Import Config Manager
from config.env_config import EnvConfig
from config.run_config import RunConfig
from config.mapping_config import MappingConfig

def train_model(resume_from_log=None):
    """
    Main training function / 主训练函数
    
    Args:
        resume_from_log (str, optional): 
            Path to an existing log directory to resume training. 
            If None, starts a new run with auto-generated path.
            如果为 None, 则自动生成路径开始新运行。如果提供了路径，则尝试从该目录恢复训练。
    """
    print("=== Model Training Script / 模型训练脚本 ===")
    
    # 1. Directories / 目录设置
    if resume_from_log:
        # Resume Mode: Use provided path
        log_path = resume_from_log
        print(f"Resume Mode: Using existing log path: {log_path}")
        # Verify config existence (Optional strict check)
        # if not os.path.exists(os.path.join(log_path, 'config.json')): ...
    else:
        # New Run Mode: Generate path
        # Retrieve logical names for directory structure
        try:
             exp_config = MappingConfig.get_experiment_config(RunConfig.EXPERIMENT_ID)
             env_name_logical = exp_config['env_id']
        except:
             env_name_logical = RunConfig.EXPERIMENT_ID # Fallback
             
        log_path = ConfigManager.generate_log_path(
            base_log_dir=RunConfig.LOGDIR,
            env_name=env_name_logical,
            algo_name=RunConfig.ALGO,
            backbone_name=RunConfig.BACKBONE,
            run_tag=RunConfig.RUN_ID # Use RUN_ID as descriptive tag if provided
        )
        print(f"New Run Mode: Generated log path: {log_path}")
        
    os.makedirs(log_path, exist_ok=True)
    
    # Save Configuration for this run (Overwrite if resuming? Maybe backup first? For now overwrite/update)
    # 保存本次运行的配置
    ConfigManager.save_config(log_path, RunConfig, EnvConfig)
    
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    
    # 2. Environments / 环境设置
    # Define environment factory functions / 定义环境工厂函数
    def make_train_env():
        # Random seed for training envs / 训练环境使用随机种子
        return make_env_instance(RunConfig, EnvConfig, seed=np.random.randint(0, 10000))
        
    def make_test_env():
        # Fixed seed for testing envs to ensure fair comparison / 测试环境使用固定种子以确保公平比较
        return make_env_instance(RunConfig, EnvConfig, seed=42)

    # Initialize Vector Environments (Parallel or Serial) / 初始化向量化环境 (并行或串行)
    # DummyVectorEnv runs sequentially; use SubprocVectorEnv for multiprocessing if needed.
    # DummyVectorEnv 串行运行；若需多进程可使用 SubprocVectorEnv。
    train_envs = DummyVectorEnv([make_train_env for _ in range(RunConfig.NUM_TRAIN_ENVS)])
    test_envs = DummyVectorEnv([make_test_env for _ in range(RunConfig.NUM_TEST_ENVS)])
    
    # Validation Env for Policy setup (Use single instance to infer shapes)
    # 用于策略设置的验证环境 (使用单个实例推断形状)
    dummy_env = make_train_env()
    
    # 3. Policy / 策略
    policy = make_policy(RunConfig, dummy_env)

    # Check for existing checkpoint to resume / 检查是否存在检查点以恢复训练
    ckpt_path = os.path.join(log_path, 'checkpoint.pth')
    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=RunConfig.DEVICE)
        policy.load_state_dict(checkpoint['policy'])
        print("Loaded policy successfully.")
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    # 4. Collector / 收集器
    # Initialize Replay Buffer / 初始化经验回放池
    buffer = VectorReplayBuffer(RunConfig.BUFFER_SIZE, len(train_envs))
    
    # Train Collector (with exploration) / 训练收集器 (带探索)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    # Test Collector (deterministic) / 测试收集器 (确定性)
    test_collector = Collector(policy, test_envs, exploration_noise=False) 
    
    # 5. Trainer / 训练器
    print("\n--- Starting Training / 开始训练 ---")
    
    # Helper to save the best policy / 保存最佳策略的辅助函数
    def save_best_fn(policy):
        torch.save({'policy': policy.state_dict()}, os.path.join(log_path, 'policy_best.pth'))
        
    # Helper to save checkpoints / 保存检查点的辅助函数
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # Save every epoch or based on frequency / 每轮保存
        torch.save({'policy': policy.state_dict()}, os.path.join(log_path, 'checkpoint.pth'))
        
    # Pre-collect to fill buffer slightly before training starts / 训练开始前预收集数据填充 Buffer
    print("Pre-collecting data... / 正在预收集数据...")
    train_collector.collect(n_step=RunConfig.BATCH_SIZE * 5)
    
    # Run Trainer / 运行训练器
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=RunConfig.EPOCH,
        step_per_epoch=RunConfig.STEP_PER_EPOCH,
        step_per_collect=RunConfig.STEP_PER_COLLECT,
        episode_per_test=RunConfig.TEST_NUM,
        batch_size=RunConfig.BATCH_SIZE,
        update_per_step=RunConfig.UPDATE_PER_STEP,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        verbose=True
    ).run()
    
    print("\n--- Training Finished / 训练完成 ---")
    print(result)
    print(f"Config used / 使用配置: {RunConfig.EXPERIMENT_ID}")

if __name__ == "__main__":
    train_model()
