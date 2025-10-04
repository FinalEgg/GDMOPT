import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from env import make_aigc_env
from policy import DiffusionOPT
from model.diffusion import Diffusion, MLP, DoubleCritic
from tianshou.data import Batch, to_torch

def test_inference():
    # 创建环境
    env = make_aigc_env(test_num=1)[0]

    # 定义模型参数 (根据训练时的参数)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # Now continuous

    # 创建模型
    model = MLP(state_dim=state_dim, action_dim=action_dim)
    diffusion = Diffusion(state_dim=state_dim, action_dim=action_dim, model=model, max_action=1.0, beta_schedule='vp', n_timesteps=6, loss_type='l2', clip_denoised=True, bc_coef=False)
    critic = DoubleCritic(state_dim=state_dim, action_dim=action_dim)

    # 创建优化器
    actor_optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)

    # 创建策略
    policy = DiffusionOPT(
        state_dim=state_dim,
        actor=diffusion,
        actor_optim=actor_optim,
        action_dim=action_dim,
        critic=critic,
        critic_optim=critic_optim,
        device='cpu'
    )

    # 加载训练好的模型
    policy_path = 'log/default/diffusion/Apr04-132705/policy.pth'  # 根据实际路径调整
    try:
        policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
        policy.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 重置环境
    obs_list, info_list = env.reset()
    obs = obs_list[0]
    done = False
    total_reward = 0

    while not done:
        # 获取动作
        obs_tensor = to_torch(obs, device='cpu').unsqueeze(0)
        batch = Batch(obs=obs_tensor)
        with torch.no_grad():
            result = policy.forward(batch)
            action = result.act.squeeze().numpy()  # Continuous action

        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        obs = next_obs

        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    test_inference()