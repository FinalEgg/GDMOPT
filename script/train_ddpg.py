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
from env import make_pendulum_env, make_optimization_env, make_cellfree_env
from env.threshold import make_threshold_env
from policy import DDPG
from model.ddpg import Actor, DuelingCritic
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="DDPG 训练脚本")
    
    # ==================== 基础参数 (Common Args) ====================
    parser.add_argument('--algorithm', type=str, default='ddpg', help='算法名称')
    parser.add_argument('--env', type=str, default='threshold', choices=['pendulum', 'optimization', 'cellfree', 'threshold'], help='环境名称')
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
    parser.add_argument('--sparsity-coef', type=float, default=0.01, help='稀疏性惩罚系数 (用于 Gate)')
    
    # ==================== 优先经验回放参数 (PER Args) ====================
    parser.add_argument('--prioritized-replay', action='store_true', default=True, help='是否使用优先经验回放')
    parser.add_argument('--prior-alpha', type=float, default=0.4, help='PER Alpha 参数')
    parser.add_argument('--prior-beta', type=float, default=0.4, help='PER Beta 参数')
    
    # ==================== 环境特定参数 (Env Specific Args) ====================
    parser.add_argument('--dim', type=int, default=2, help='优化环境维度 (仅用于 optimization 环境)')
    parser.add_argument('--top-p', type=float, default=0.6, help='Top-P 策略参数 (用于预训练)')

    # ==================== 阶段 0: Critic 预热 (Phase 0: Critic Warm-up) ====================
    parser.add_argument('--warmup-samples', type=int, default=500000, help='预热阶段采集的样本数量 (1e5)')
    parser.add_argument('--warmup-epochs', type=int, default=5000, help='预热阶段训练轮数')
    parser.add_argument('--warmup-steps-per-epoch', type=int, default=1000, help='预热阶段每轮步数')

    # ==================== 阶段 1: 预训练 (Phase 1: Pre-training) ====================
    parser.add_argument('--pretrain-epoch', type=int, default=100, help='预训练总轮数')
    parser.add_argument('--pretrain-step-per-epoch', type=int, default=1000, help='预训练每轮步数')
    parser.add_argument('--pretrain-step-per-collect', type=int, default=1000, help='预训练每次收集步数')
    
    # ==================== 阶段 1.5: 中间预热 (Phase 1.5: Intermediate Warm-up) ====================
    parser.add_argument('--inter-warmup-samples', type=int, default=1000000, help='中间预热阶段采集的样本数量')
    parser.add_argument('--inter-warmup-epochs', type=int, default=5000, help='中间预热阶段训练轮数')
    parser.add_argument('--inter-warmup-steps-per-epoch', type=int, default=1000, help='中间预热阶段每轮步数')

    # ==================== 阶段 2: 微调 (Phase 2: Fine-tuning) ====================
    parser.add_argument('--finetune-epoch', type=int, default=1000, help='微调总轮数')
    parser.add_argument('--finetune-step-per-epoch', type=int, default=1000, help='微调每轮步数')
    parser.add_argument('--finetune-step-per-collect', type=int, default=1000, help='微调每次收集步数')

    # 兼容旧参数名 (Legacy)
    parser.add_argument('-e', '--epoch', type=int, default=1000, help='总轮数 (非 cellfree 环境使用)')
    parser.add_argument('--step-per-epoch', type=int, default=1000, help='每轮步数 (非 cellfree 环境使用)')
    parser.add_argument('--step-per-collect', type=int, default=1000, help='每次收集步数 (非 cellfree 环境使用)')

    args = parser.parse_known_args()[0]
    return args

def setup_policy(args, env):
    """初始化策略网络 (Actor, Critic) 和 DDPG 算法"""
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.max_action = 1.
    args.exploration_noise = args.exploration_noise * args.max_action

    # 创建 Actor 网络
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
        lr_maxt=args.finetune_epoch,
        exploration_noise=args.exploration_noise,
        sparsity_coef=args.sparsity_coef
    )
    return policy

def run_training_phase(args, phase_name, env, train_envs, test_envs, policy, epochs, step_per_epoch, step_per_collect, log_path, stop_fn=None):
    """执行单个训练阶段的通用函数"""
    print(f"\n{'='*20} 开始阶段: {phase_name} {'='*20}")
    
    # 设置经验回放池 (每个阶段使用新的 Buffer 以避免分布偏移)
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
    writer = SummaryWriter(os.path.join(log_path, phase_name))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        """保存最佳模型回调"""
        torch.save(policy.state_dict(), os.path.join(log_path, f'{phase_name}_policy.pth'))

    def debug_hook(epoch, env_step):
        """调试钩子：在每个测试周期开始时打印模型输出统计信息"""
        print(f"\n[Epoch {epoch}] 模型输出调试信息:")
        
        # 生成一批样本状态
        batch_size = 32
        
        obs_list = []
        rew_list = []
        current_count = 0
        while current_count < batch_size:
            # 重置测试环境以获取新观测
            batch_obs, _ = test_envs.reset()
            
            # 计算动作以获取奖励
            with torch.no_grad():
                batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float32, device=args.device)
                
                # 获取 Actor 输出 (Action, Gate Prob)
                action_tuple = policy._actor(batch_obs_tensor)
                if isinstance(action_tuple, tuple):
                    action, gate_prob = action_tuple
                else:
                    action = action_tuple
                    gate_prob = torch.ones_like(action) # Dummy
                
                action_np = action.cpu().numpy()
            
            # 执行动作获取奖励
            _, rews, _, _, _ = test_envs.step(action_np)
            
            obs_list.append(batch_obs)
            rew_list.append(rews)
            current_count += len(batch_obs)
            
        # 拼接并截取到 batch_size
        obs_array = np.concatenate(obs_list, axis=0)[:batch_size]
        rew_array = np.concatenate(rew_list, axis=0)[:batch_size]
            
        with torch.no_grad():
            # 准备输入张量
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=args.device)
            
            # 前向传播 Actor 以获取详细信息 (适配 DeepSets 架构)
            actor = policy._actor
            batch_size = obs_tensor.shape[0]
            
            # 手动执行 Actor 前向过程以获取中间变量
            state_reshaped = obs_tensor.view(batch_size, actor.num_uavs, actor.state_per_uav)
            local_feat = actor.local_encoder(state_reshaped)
            global_feat = torch.max(local_feat, dim=1)[0]
            global_feat_expanded = global_feat.unsqueeze(1).expand(-1, actor.num_uavs, -1)
            fusion_feat = torch.cat([local_feat, global_feat_expanded], dim=2)
            decoded = actor.decoder(fusion_feat)
            
            power = actor.power_head(decoded).view(batch_size, -1)
            gate_logits = actor.gate_head(decoded).view(batch_size, -1)
            gate_prob = torch.sigmoid(gate_logits)
            
            gate_action = (gate_prob > 0.5).float()
            action = power * gate_action
            
            # 评估 Critic (Q值)
            q1, q2 = policy._critic(obs_tensor, action)
            
            # 计算 V 值 (手动调用 Critic 内部组件)
            critic = policy._critic
            batch_size_v = obs_tensor.shape[0]
            s_v = obs_tensor.view(batch_size_v, critic.num_uavs, critic.state_per_uav)
            
            v1 = critic._process_stream(critic.v_encoder, critic.v_head, s_v)
            v2 = critic._process_stream(critic.v_encoder2, critic.v_head2, s_v)
            
            # 打印统计信息
            print(f"  > 功率输出 (0-1):      Min={power.min():.4f}, Max={power.max():.4f}, Avg={power.mean():.4f}")
            print(f"  > 门控 Logits:         Min={gate_logits.min():.4f}, Max={gate_logits.max():.4f}, Avg={gate_logits.mean():.4f}")
            print(f"  > 门控概率:            Min={gate_prob.min():.4f}, Max={gate_prob.max():.4f}, Avg={gate_prob.mean():.4f}")
            print(f"  > 激活门控比例 (>0.5): Ratio={(gate_prob > 0.5).float().mean():.4f}")
            print(f"  > Critic V1:           Min={v1.min():.4f}, Max={v1.max():.4f}, Avg={v1.mean():.4f}")
            print(f"  > Critic V2:           Min={v2.min():.4f}, Max={v2.max():.4f}, Avg={v2.mean():.4f}")
            print(f"  > Critic Q1:           Min={q1.min():.4f}, Max={q1.max():.4f}, Avg={q1.mean():.4f}")
            print(f"  > Critic Q2:           Min={q2.min():.4f}, Max={q2.max():.4f}, Avg={q2.mean():.4f}")
            print(f"  > 环境奖励 (归一化):   Min={rew_array.min():.4f}, Max={rew_array.max():.4f}, Avg={rew_array.mean():.4f}")
            print("-" * 50)
        
        # 关键：手动步进后重置测试环境和采集器，防止状态错乱
        test_envs.reset()
        test_collector.reset_env()
        test_collector.reset_buffer()

    # 启动训练器
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        epochs,
        step_per_epoch,
        step_per_collect,
        args.test_num,
        args.batch_size,
        save_best_fn=save_best_fn,
        stop_fn=stop_fn,
        test_fn=debug_hook, # 添加调试钩子
        logger=logger,
        test_in_train=False
    )
    pprint.pprint(result)
    return result

def main(args=get_args()):
    # 设置日志路径
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    base_log_path = os.path.join(args.logdir, args.log_prefix, args.algorithm, args.env, time_now)
    os.makedirs(base_log_path, exist_ok=True)

    if args.env in ['cellfree', 'threshold']:
        make_env_fn = make_cellfree_env if args.env == 'cellfree' else make_threshold_env

        # ==================== 阶段 1: 预训练 (Geometric Reward) ====================
        print("初始化预训练环境 (Geometric)...")
        env_geo, train_envs_geo, test_envs_geo = make_env_fn(
            args.training_num, args.test_num, reward_mode="geometric", top_p=args.top_p, norm_reward=True
        )
        
        policy = setup_policy(args, env_geo)
        
        if args.resume_path:
            ckpt = torch.load(args.resume_path, map_location=args.device)
            policy.load_state_dict(ckpt)
            print("已加载模型: ", args.resume_path)

        # ==================== 阶段 0: Critic 预热 (随机动作) ====================
        print("\n" + "="*50)
        print(" 开始阶段 0: Critic 预热 (Critic Warm-up)")
        print("="*50)
        
        warmup_buffer = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs_geo))
        warmup_collector = Collector(policy, train_envs_geo, warmup_buffer)
        
        print(f"正在采集 {args.warmup_samples} 个随机样本用于预热...")
        warmup_collector.collect(n_step=args.warmup_samples, random=True)
        print(f"已采集 {len(warmup_buffer)} 个样本。")
        
        print(f"正在预热 Critic: {args.warmup_epochs} 轮 (每轮 {args.warmup_steps_per_epoch} 步)...")
        
        for epoch in range(args.warmup_epochs):
            losses = []
            for _ in range(args.warmup_steps_per_epoch // args.batch_size):
                batch, indices = warmup_buffer.sample(args.batch_size)
                # 学习前计算回报 (Target Q)
                batch = policy.process_fn(batch, warmup_buffer, indices)
                # 仅更新 Critic
                res = policy.learn(batch, update_actor=False)
                losses.append(res['loss/critic'])
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Critic Loss = {np.mean(losses):.4f}")
        
        # ==================== 执行阶段 1: 预训练 ====================
        run_training_phase(
            args, "pretrain", env_geo, train_envs_geo, test_envs_geo, policy,
            args.pretrain_epoch, args.pretrain_step_per_epoch, args.pretrain_step_per_collect, base_log_path
        )
        
        # ==================== 阶段 1.5: 中间 Critic 预热 (Intermediate Warm-up) ====================
        print("\n" + "="*50)
        print(" 开始阶段 1.5: 中间 Critic 预热 (Intermediate Critic Warm-up)")
        print("="*50)
        
        # 创建新的 Buffer 用于中间预热 (丢弃旧数据)
        inter_buffer = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs_geo))
        inter_collector = Collector(policy, train_envs_geo, inter_buffer)
        
        print(f"正在采集 {args.inter_warmup_samples} 个样本用于中间预热...")
        inter_collector.collect(n_step=args.inter_warmup_samples) # 使用当前策略采集
        print(f"已采集 {len(inter_buffer)} 个样本。")
        
        print(f"正在预热 Critic: {args.inter_warmup_epochs} 轮 (每轮 {args.inter_warmup_steps_per_epoch} 步)...")
        
        for epoch in range(args.inter_warmup_epochs):
            inter_losses = []
            for _ in range(args.inter_warmup_steps_per_epoch // args.batch_size):
                batch, indices = inter_buffer.sample(args.batch_size)
                batch = policy.process_fn(batch, inter_buffer, indices)
                res = policy.learn(batch, update_actor=False)
                inter_losses.append(res['loss/critic'])
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Critic Loss = {np.mean(inter_losses):.4f}")

        # ==================== 阶段 2: 微调 (Normal Reward) ====================
        print("\n初始化微调环境 (Normal)...")
        env, train_envs, test_envs = make_env_fn(
            args.training_num, args.test_num, reward_mode="normal", norm_reward=True
        )
        
        # 执行阶段 2: 微调
        run_training_phase(
            args, "finetune", env, train_envs, test_envs, policy,
            args.finetune_epoch, args.finetune_step_per_epoch, args.finetune_step_per_collect, base_log_path
        )

    else:
        # 兼容其他环境 (Legacy support)
        if args.env == 'pendulum':
            env, train_envs, test_envs = make_pendulum_env(args.training_num, args.test_num)
        elif args.env == 'optimization':
            env, train_envs, test_envs = make_optimization_env(args.training_num, args.test_num, dim=args.dim)
            
        policy = setup_policy(args, env)
        
        run_training_phase(
            args, "train", env, train_envs, test_envs, policy,
            args.epoch, args.step_per_epoch, args.step_per_collect, base_log_path
        )

if __name__ == '__main__':
    main(get_args())
