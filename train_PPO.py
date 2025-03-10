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
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.trainer import onpolicy_trainer

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from env import make_aigc_env   # 修改这里，从旧的make_env改为make_aigc_env

class PPOActor(Actor):
    def __init__(self, net, action_shape, hidden_sizes=(), max_action=1.0, device='cpu'):
        super().__init__(net, action_shape, hidden_sizes, max_action, device=device)
        # 定义一个可学习的 log_std 参数
        self.log_std = nn.Parameter(torch.zeros(action_shape))

    def forward(self, obs, state=None, info={}):
        logits, hidden = super().forward(obs, state, info)
        mean = self._max * torch.tanh(logits)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        return (mean, std), hidden
    
def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--step-per-collect', type=int, default=100)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=99999)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action="store_true", default=False)

    # for ppo
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    parser.add_argument('--reward-threshold', type=float, default=None)
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 修改日志保存路径，参考 train_GDM-DRL 的写法
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args.logdir, args.log_prefix, "PPO", time_now)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        if args.reward_threshold:
            return mean_rewards >= args.reward_threshold
        return False

    # model
    net = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Mish,
        device=args.device
    )
    actor = PPOActor(net, args.action_shape, device=args.device,hidden_sizes=[], max_action=1.0).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(
        actor_critic.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # policy
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        torch.distributions.Normal,  # 使用Normal以处理连续动作
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)

    # trainer
    if not args.watch:
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=args.step_per_collect,
            update_per_step=args.update_per_step,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger
        )
        pprint.pprint(result)

    # Watch the performance
    if __name__ == '__main__':
        np.random.seed(args.seed)
        # 为查看效果，仅构建单环境
        env, _, _ = make_aigc_env(1, 1)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1)  # 移除 render 参数，避免调用未实现的 render 方法
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())