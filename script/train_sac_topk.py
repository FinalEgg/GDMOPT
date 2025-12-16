
# 导入必要的库
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
from tianshou.policy import SACPolicy
from env.topk import make_topk_env
from model.sac_topk.sac import Actor, Critic, SingleCritic
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="SAC Top-K 训练脚本")
    
    # ==================== 基础参数 (Common Args) ====================
    parser.add_argument('--algorithm', type=str, default='sac', help='算法名称')
    parser.add_argument('--env', type=str, default='topk', help='环境名称')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='训练设备 (cpu/cuda)')
    parser.add_argument('--logdir', type=str, default='log', help='日志保存目录')
    parser.add_argument('--log-prefix', type=str, default='default', help='日志前缀')
    parser.add_argument('--resume-path', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--note', type=str, default='', help='实验备注')
    parser.add_argument('--watch', action='store_true', default=False, help='是否仅观察模型表现而不训练')
    parser.add_argument('--render', type=float, default=0.1, help='渲染间隔')
    
    # ==================== 训练超参数 (Training Hyperparameters) ====================
    parser.add_argument('--buffer-size', type=int, default=100000, help='经验回放池大小')
    parser.add_argument('-b', '--batch-size', type=int, default=512, help='批次大小')
    parser.add_argument('--wd', type=float, default=1e-3, help='权重衰减 (Weight Decay)')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--n-step', type=int, default=1, help='N步回报')
    parser.add_argument('--training-num', type=int, default=10, help='训练环境数量')
    parser.add_argument('--test-num', type=int, default=10, help='测试环境数量')
    parser.add_argument('--rew-norm', type=int, default=0, help='是否对奖励进行归一化')
    parser.add_argument('--lr-decay', action='store_true', default=True, help='是否启用学习率衰减')
    
    # ==================== SAC 特定参数 (SAC Specific Args) ====================
    parser.add_argument('--actor-lr', type=float, default=1e-4, help='Actor 学习率')
    parser.add_argument('--critic-lr', type=float, default=1e-4, help='Critic 学习率')
    parser.add_argument('--alpha-lr', type=float, default=3e-4, help='Alpha 学习率')
    parser.add_argument('--tau', type=float, default=0.005, help='软更新系数')
    parser.add_argument('--alpha', type=float, default=0.2, help='熵正则化系数')
    parser.add_argument('--auto-alpha', action='store_true', default=True, help='是否自动调整 Alpha')

    # ==================== 优先经验回放参数 (PER Args) ====================
    parser.add_argument('--prioritized-replay', action='store_true', default=True, help='是否使用优先经验回放')
    parser.add_argument('--prior-alpha', type=float, default=0.4, help='PER Alpha 参数')
    parser.add_argument('--prior-beta', type=float, default=0.4, help='PER Beta 参数')
    
    # ==================== 环境特定参数 (Env Specific Args) ====================
    parser.add_argument('--top-p', type=float, default=0.6, help='Top-P 策略参数 (用于第一阶段筛选)')

    # ==================== 训练阶段参数 (Training Phase Args) ====================
    parser.add_argument('--warmup-samples', type=int, default=1e5, help='预热阶段采集的样本数量')
    parser.add_argument('--warmup-epochs', type=int, default=100, help='预热阶段训练轮数')
    parser.add_argument('--warmup-steps-per-epoch', type=int, default=1000, help='预热阶段每轮步数')
    
    parser.add_argument('--epoch', type=int, default=2000, help='训练总轮数')
    parser.add_argument('--step-per-epoch', type=int, default=1000, help='每轮步数')
    parser.add_argument('--step-per-collect', type=int, default=1000, help='每次收集步数')

    args = parser.parse_known_args()[0]
    return args

def setup_policy(args, env):
    """初始化策略网络 (Actor, Critic) 和 SAC 算法"""
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.max_action = 1.

    # 创建 Actor 网络 (SAC Model)
    actor_net = Actor(args)
    actor = actor_net.to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=args.wd)

    # 创建 Critic 网络 (SAC Model - Double Critic)
    critic1 = SingleCritic(args).to(args.device)
    critic1_optim = torch.optim.AdamW(critic1.parameters(), lr=args.critic_lr, weight_decay=args.wd)
    critic2 = SingleCritic(args).to(args.device)
    critic2_optim = torch.optim.AdamW(critic2.parameters(), lr=args.critic_lr, weight_decay=args.wd)

    # Alpha 设置: 当前 Tianshou 版本不支持在构造函数中传入 target_entropy/alpha_optim
    # 使用固定 alpha（可通过参数调整）
    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        reward_normalization=bool(args.rew_norm),
        estimation_step=args.n_step,
        action_space=env.action_space,
        deterministic_eval=True
    )

    # Ensure policy moves data to the correct device during forward
    policy = policy.to(args.device)
    return policy

def run_training(args, env, train_envs, test_envs, policy, log_path):
    """执行训练循环"""
    print(f"\n{'='*20} 开始训练 (SAC Top-K Compressed) {'='*20}")
    
    # 设置经验回放池
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

    # 设置采集器
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    # 设置日志记录器
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return False

    def debug_hook(epoch, env_step):
        """调试钩子：记录详细的 Critic 和 Actor 状态"""
        if env_step % 1000 == 0: # 每 1000 步记录一次
            # 1. 记录 Critic 值分布
            # 从 buffer 中采样一批数据
            if len(buffer) > 100:
                batch, _ = buffer.sample(128)
                obs = torch.tensor(batch.obs, device=args.device, dtype=torch.float32)
                act = torch.tensor(batch.act, device=args.device, dtype=torch.float32)
                
                with torch.no_grad():
                    q1 = policy.critic1(obs, act)
                    q2 = policy.critic2(obs, act)
                    
                    # 记录 Q 值统计
                    writer.add_scalar('debug/critic_q1_mean', q1.mean().item(), env_step)
                    writer.add_scalar('debug/critic_q2_mean', q2.mean().item(), env_step)
                    
                    # 记录 Actor 输出分布 (Mean)
                    logits, _ = policy.actor(obs)
                    mu, std = logits
                    writer.add_scalar('debug/actor_mu_mean', mu.mean().item(), env_step)
                    writer.add_scalar('debug/actor_std_mean', std.mean().item(), env_step)
                    
                    print(f"\n[Epoch {epoch}] Debug Info:")
                    print(f"  > Actor Mu:     Avg={mu.mean().item():.4f}")
                    print(f"  > Actor Std:    Avg={std.mean().item():.4f}")
                    print(f"  > Critic Q1:    Avg={q1.mean().item():.4f}, Min={q1.min().item():.4f}, Max={q1.max().item():.4f}")
                    print(f"  > Critic Q2:    Avg={q2.mean().item():.4f}, Min={q2.min().item():.4f}, Max={q2.max().item():.4f}")

    # ==================== Phase 0: Critic Warm-up ====================
    print(f"\n{'='*20} Phase 0: Critic Warm-up {'='*20}")
    # 1. 随机采集数据
    print(f"Collecting {args.warmup_samples} random samples...")
    policy.train()
    # SAC 默认就是随机策略 (基于 Alpha)，但为了 Warmup 我们可以强制随机
    train_collector.collect(n_step=args.warmup_samples, random=True)
    
    print(f"Buffer size: {len(buffer)}")
    
    # 2. 仅训练 Critic
    print("Training Critic only...")
    # 冻结 Actor
    for param in policy.actor.parameters():
        param.requires_grad = False
        
    # 手动循环更新 Critic
    policy.train()
    for epoch in range(args.warmup_epochs):
        losses = []
        for step in range(args.warmup_steps_per_epoch):
            batch, indices = buffer.sample(args.batch_size)
            # 计算 Target Q (returns)
            batch = policy.process_fn(batch, buffer, indices)
            # 更新 Critic，不更新 Actor
            res = policy.learn(batch, update_actor=False)
            critic_loss = res['loss/critic1'] + res['loss/critic2']
            losses.append(critic_loss)
            
            # Debug Print every 500 steps
            if step % 500 == 0 and epoch % 5 == 0:
                obs = torch.tensor(batch.obs, device=args.device, dtype=torch.float32)
                act = torch.tensor(batch.act, device=args.device, dtype=torch.float32)
                rew = torch.tensor(batch.rew, device=args.device, dtype=torch.float32)
                with torch.no_grad():
                    q1 = policy.critic1(obs, act)
                    q2 = policy.critic2(obs, act)
                
                print(f"\n[Warmup Debug] Epoch {epoch} Step {step}")
                print(f"  > Reward: Mean={rew.mean().item():.4f}, Std={rew.std().item():.4f}, Min={rew.min().item():.4f}, Max={rew.max().item():.4f}")
                print(f"  > Q1 Val: Mean={q1.mean().item():.4f}, Std={q1.std().item():.4f}")
                print(f"  > Loss:   {critic_loss:.6f}")
            
            # 更新优先级 (如果使用 PER)
            if args.prioritized_replay:
                pass
                
        if (epoch + 1) % 5 == 0:
            print(f"Warmup Epoch {epoch+1}/{args.warmup_epochs}: Critic Loss = {np.mean(losses):.4f}")
            
    # 解冻 Actor
    for param in policy.actor.parameters():
        param.requires_grad = True
    print("Critic Warm-up Finished.")

    # ==================== Phase 1: Main Training ====================
    print(f"\n{'='*20} Phase 1: Main Training (SAC Top-K Compressed) {'='*20}")

    # 开始训练
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=1,
        test_in_train=False,
        train_fn=debug_hook 
    )
    
    pprint.pprint(result)

def main(args=get_args()):
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建环境
    env, train_envs, test_envs = make_topk_env(
        args.training_num, 
        args.test_num, 
        top_p=args.top_p,
        norm_reward=bool(args.rew_norm)
    )
    
    # 设置日志路径
    timestamp = datetime.now().strftime("%b%d-%H%M%S")
    log_path = os.path.join(args.logdir, args.log_prefix, args.algorithm, args.env, timestamp)
    
    # 初始化策略
    policy = setup_policy(args, env)
    
    # 运行训练
    run_training(args, env, train_envs, test_envs, policy, log_path)

if __name__ == '__main__':
    main()
