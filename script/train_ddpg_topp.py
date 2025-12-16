
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
from env.topp import make_topp_env
from policy import DDPG
from model.ddpg_topp import Actor, DuelingCritic
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="DDPG Top-P 训练脚本")
    
    # ==================== 基础参数 (Common Args) ====================
    parser.add_argument('--algorithm', type=str, default='ddpg', help='算法名称')
    parser.add_argument('--env', type=str, default='topp', help='环境名称')
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
    
    # ==================== DDPG 特定参数 (DDPG Specific Args) ====================
    parser.add_argument('--actor-lr', type=float, default=1e-5, help='Actor 学习率')
    parser.add_argument('--critic-lr', type=float, default=1e-5, help='Critic 学习率')
    parser.add_argument('--tau', type=float, default=0.005, help='软更新系数')
    parser.add_argument("--exploration-noise", type=float, default=0.1, help='探索噪声标准差')
    parser.add_argument('--policy-noise', type=float, default=0.2, help='目标策略平滑噪声 (TD3)')
    parser.add_argument('--noise-clip', type=float, default=0.5, help='噪声截断范围 (TD3)')
    
    # ==================== 优先经验回放参数 (PER Args) ====================
    parser.add_argument('--prioritized-replay', action='store_true', default=True, help='是否使用优先经验回放')
    parser.add_argument('--prior-alpha', type=float, default=0.4, help='PER Alpha 参数')
    parser.add_argument('--prior-beta', type=float, default=0.4, help='PER Beta 参数')
    
    # ==================== 环境特定参数 (Env Specific Args) ====================
    parser.add_argument('--top-p', type=float, default=0.6, help='Top-P 策略参数')

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
    """初始化策略网络 (Actor, Critic) 和 DDPG 算法"""
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.max_action = 1.
    args.exploration_noise = args.exploration_noise * args.max_action

    # 创建 Actor 网络 (无 Gate)
    actor_net = Actor(state_dim=args.state_shape, action_dim=args.action_shape, hidden_dim=256)
    actor = actor_net.to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=args.wd)

    # 创建 Critic 网络 (Dueling 架构)
    critic = DuelingCritic(state_dim=args.state_shape, action_dim=args.action_shape, hidden_dim=256).to(args.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.wd)

    # 定义 DDPG 策略
    policy = DDPG(
        args.state_shape,
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        reward_normalization=bool(args.rew_norm),
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        exploration_noise=args.exploration_noise,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip
    )
    return policy

def run_training(args, env, train_envs, test_envs, policy, log_path):
    """执行训练循环"""
    print(f"\n{'='*20} 开始训练 (Top-P Rule Based) {'='*20}")
    
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
                    q1, q2 = policy._critic(obs, act)
                    
                    # 记录 Q 值统计
                    writer.add_scalar('debug/critic_q1_mean', q1.mean().item(), env_step)
                    writer.add_scalar('debug/critic_q2_mean', q2.mean().item(), env_step)
                    
                    # 记录 Actor 输出分布
                    actor_out = policy._actor(obs)
                    writer.add_scalar('debug/actor_power_mean', actor_out.mean().item(), env_step)
                    
                    print(f"\n[Epoch {epoch}] Debug Info:")
                    print(f"  > Power Output: Avg={actor_out.mean().item():.4f}")
                    print(f"  > Critic Q1:    Avg={q1.mean().item():.4f}, Min={q1.min().item():.4f}, Max={q1.max().item():.4f}")
                    print(f"  > Critic Q2:    Avg={q2.mean().item():.4f}, Min={q2.min().item():.4f}, Max={q2.max().item():.4f}")

    # ==================== Phase 0: Critic Warm-up ====================
    print(f"\n{'='*20} Phase 0: Critic Warm-up {'='*20}")
    # 1. 随机采集数据
    print(f"Collecting {args.warmup_samples} random samples...")
    policy.train()
    # 使用随机策略采集数据
    original_noise = policy._exploration_noise
    policy.set_exp_noise(1.0) # 强噪声
    train_collector.collect(n_step=args.warmup_samples, random=True)
    policy.set_exp_noise(original_noise) # 恢复噪声
    
    print(f"Buffer size: {len(buffer)}")
    
    # 2. 仅训练 Critic
    print("Training Critic only...")
    # 冻结 Actor
    for param in policy._actor.parameters():
        param.requires_grad = False
        
    # 手动循环更新 Critic
    policy.train()
    for epoch in range(args.warmup_epochs):
        losses = []
        for _ in range(args.warmup_steps_per_epoch):
            batch, indices = buffer.sample(args.batch_size)
            # 计算 Target Q (returns)
            batch = policy.process_fn(batch, buffer, indices)
            # 更新 Critic，不更新 Actor
            res = policy.learn(batch, update_actor=False)
            losses.append(res['loss/critic'])
            
            # 更新优先级 (如果使用 PER)
            if args.prioritized_replay:
                # 简单处理：不更新优先级，或者假设 learn 内部处理了
                pass
                
        if (epoch + 1) % 5 == 0:
            print(f"Warmup Epoch {epoch+1}/{args.warmup_epochs}: Critic Loss = {np.mean(losses):.4f}")
            
    # 解冻 Actor
    for param in policy._actor.parameters():
        param.requires_grad = True
    print("Critic Warm-up Finished.")

    # ==================== Phase 1: Main Training ====================
    print(f"\n{'='*20} Phase 1: Main Training (Top-P Rule Based) {'='*20}")

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
    env, train_envs, test_envs = make_topp_env(
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
