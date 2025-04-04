import argparse
import sys
import os
import pprint
import torch
import torch.nn as nn
import numpy as np
from os import path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.policy import TD3Policy
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.trainer.offpolicy import OffpolicyTrainer
from actors.neural.TD3Actor import TD3Actor
import time

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from env import make_aigc_env


def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--task', type=str, default='AaaS')
    parser.add_argument('--seed', type=int, default=int(time.time() * 1000) % (2**32 - 1))
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])  # 增大隐藏层
    parser.add_argument('--wd', type=float, default=1e-5)  # 减小权重衰减，防止过度正则化
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=100)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action="store_true", default=False)

    # TD3特有参数
    parser.add_argument('--actor-lr', type=float, default=3e-5)  # 稍微提高Actor学习率
    parser.add_argument('--critic-lr', type=float, default=1e-4)  # 减小Critic学习率，增加稳定性
    parser.add_argument('--tau', type=float, default=0.005)  # 保持原有的软更新系数
    parser.add_argument('--policy-noise', type=float, default=0.2)  # 添加策略噪声参数
    parser.add_argument('--noise-clip', type=float, default=0.5)  # 添加噪声裁剪参数
    parser.add_argument('--update-actor-freq', type=int, default=2)  # actor更新频率
    parser.add_argument('--max-grad-norm', type=float, default=0.5)  # 梯度裁剪范数
    args = parser.parse_known_args()[0]
    return args

def main(args=get_args()):
    # create environments
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]

    # seed
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)

    # 日志设置
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args.logdir, args.log_prefix, "TD3", time_now)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, train_interval=10)  # 更频繁记录训练日志

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
        # 同时保存最新检查点，用于意外中断后继续训练
        torch.save(policy.state_dict(), os.path.join(log_path, 'latest.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # 每100个epoch保存一次检查点
        if epoch % 100 == 0:
            ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
            torch.save({"model": policy.state_dict(), 
                       "epoch": epoch,
                       "env_step": env_step,
                       "gradient_step": gradient_step}, ckpt_path)
            return ckpt_path

    def stop_fn(mean_rewards):
        if args.reward_threshold:
            return mean_rewards >= args.reward_threshold
        return False
    
    # 创建Actor
    hidden_dim = args.hidden_sizes[0] if args.hidden_sizes else 256
    actor = TD3Actor(args.state_shape, hidden_dim, args.action_shape, device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )
    
    # 创建Critic
    def create_critic(device=args.device):
        net = Net(
            args.state_shape + args.action_shape,
            hidden_sizes=args.hidden_sizes,
            activation=nn.Mish,
            device=device
        )
        return Critic(net, device=device).to(device)
    
    critic1 = create_critic()
    critic1_optim = torch.optim.Adam(
        critic1.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )
    
    critic2 = create_critic()
    critic2_optim = torch.optim.Adam(
        critic2.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    # policy
    policy = TD3Policy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        args.tau,
        args.gamma,
        exploration_noise=0.1,  # 添加探索噪声
        policy_noise=args.policy_noise,  # TD3特有的目标策略平滑
        noise_clip=args.noise_clip,  # 噪声裁剪
        update_actor_freq=args.update_actor_freq,  # actor更新频率
        estimation_step=args.n_step,
        reward_normalization=args.rew_norm
    )

    # 加载之前的策略
    if args.resume_path:
        if os.path.isfile(args.resume_path):
            checkpoint = torch.load(args.resume_path, map_location=args.device)
            # 支持加载检查点字典或直接的模型状态
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                policy.load_state_dict(checkpoint["model"])
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}.")
            else:
                policy.load_state_dict(checkpoint)
            print("Loaded agent from: ", args.resume_path)
        else:
            print(f"Warning: Resume path {args.resume_path} does not exist. Starting fresh training.")

    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # trainer
    if not args.watch:
        # 创建trainer实例
        trainer = OffpolicyTrainer(
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
            save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=True,  # 启用训练中测试，更好地监控性能
            resume_from_log=args.resume_path is not None,  # 支持从检查点恢复训练
        )
        
        # 保存原有的 policy_update_fn
        _old_update_fn = trainer.policy_update_fn
        
        # 定义新的 policy_update_fn，添加梯度裁剪和权重裁剪
        def new_policy_update_fn(data, result):
            # 前向传播和损失计算由原始更新函数处理
            _old_update_fn(data, result)
            
            # 权重裁剪
            policy.actor.clip_weights(clip_value=0.05)
            
            # 梯度裁剪 - 在下一次更新前应用
            policy.actor.clip_gradients(max_norm=args.max_grad_norm)
            
            # 定期输出网络权重统计信息(每100次更新)
            if trainer.gradient_step % 100 == 0:
                # 收集权重统计数据
                actor_weights = []
                for name, param in policy.actor.named_parameters():
                    if 'weight' in name:
                        actor_weights.append(param.data.abs().mean().item())
                
                # 记录到tensorboard
                if len(actor_weights) > 0:
                    avg_weight = sum(actor_weights) / len(actor_weights)
                    # 修复: 使用writer直接写入，而不是通过logger.write
                    writer.add_scalar("train/avg_actor_weight", avg_weight, trainer.gradient_step)        
                # 替换trainer的更新函数
                trainer.policy_update_fn = new_policy_update_fn
        
        # 运行训练
        result = trainer.run()
        pprint.pprint(result)

    # 评估模式
    if __name__ == '__main__':
        np.random.seed(args.seed+3)
        env, _, _ = make_aigc_env(args.training_num, args.test_num)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())