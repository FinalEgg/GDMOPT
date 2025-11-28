# Import necessary libraries
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
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from env import make_pendulum_env, make_optimization_env, make_cellfree_env
from policy import SAC
from model.sac import Actor, DuelingCritic
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Define a function to get command line arguments
def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument('--algorithm', type=str, default='sac')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=1e6)#1e6
    parser.add_argument('-e', '--epoch', type=int, default=1000)# 1000
    parser.add_argument('--step-per-epoch', type=int, default=100)# 100
    parser.add_argument('--step-per-collect', type=int, default=1000)#1000
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument(
        '--device', type=str, default='cuda:0')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action='store_true', default=False)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--note', type=str, default='')

    # for sac
    parser.add_argument('--actor-lr', type=float, default=1e-5)
    parser.add_argument('--critic-lr', type=float, default=1e-5)
    parser.add_argument('--value-lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    parser.add_argument('--alpha', type=float, default=0.2)  # temperature parameter

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.4)
    parser.add_argument('--prior-beta', type=float, default=0.4)
    parser.add_argument('--env', type=str, default='cellfree', choices=['pendulum', 'optimization', 'cellfree'])
    parser.add_argument('--dim', type=int, default=2)  # For optimization env

    # Parse arguments and return them
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # create environments
    if args.env == 'pendulum':
        env, train_envs, test_envs = make_pendulum_env(args.training_num, args.test_num)
    elif args.env == 'optimization':
        env, train_envs, test_envs = make_optimization_env(args.training_num, args.test_num, dim=args.dim)
    elif args.env == 'cellfree':
        env, train_envs, test_envs = make_cellfree_env(args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]  # Now it's shape for Box
    args.max_action = 1.

    args.exploration_noise = args.exploration_noise * args.max_action
    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # create actor
    actor_net = Actor(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    )
    actor = actor_net.to(args.device)
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )

    # Create critic
    critic = DuelingCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    ## Setup logging
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args.logdir, args.log_prefix, args.algorithm, args.env, time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # Define policy
    policy = SAC(
        args.state_shape,
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        reward_normalization=bool(args.rew_norm),
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
    )

    # Load a previous policy if a path is provided
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # Setup buffer
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

    # Setup collector
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # Trainer
    if not args.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False
        )
        pprint.pprint(result)

    # Watch the performance
    if args.watch:
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1) #, render=args.render
        print(result)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())