import torch
import numpy as np
import matplotlib.pyplot as plt
from env import make_aigc_env, config as cnf
from actors.diffusion import Diffusion
from actors.diffusion.model import MLP
from visualize.predict_utils import launch_visualization, load_actor
from test.test_case import fixed_grid_environment

def state(mode):
    import time
    if mode == 1:
        current_time = int(time.time() * 1000) & 0xFFFFFFFF
        np.random.seed(current_time)
        x_a = np.random.uniform(0, 1, cnf.NUM_A_AP)
        y_a = np.random.uniform(0, 1, cnf.NUM_A_AP)
        h_a = np.random.uniform(0, 1, cnf.NUM_A_AP)
        x_g = np.random.uniform(0, 1, cnf.NUM_G_AP)
        y_g = np.random.uniform(0, 1, cnf.NUM_G_AP)
        x_u = np.random.uniform(0, 1, cnf.NUM_USERS)
        y_u = np.random.uniform(0, 1, cnf.NUM_USERS)
        num_links_a = cnf.NUM_A_AP * cnf.NUM_USERS
        num_links_g = cnf.NUM_G_AP * (cnf.NUM_USERS + cnf.NUM_A_AP)
        num_links = num_links_a + num_links_g
        p_alloc = np.random.uniform(0, 1, num_links)
        reward_in = [0]
        states = np.concatenate([x_a, y_a, h_a, x_g, y_g, x_u, y_u, p_alloc, reward_in])
        return states
    elif mode == 2:
        states = np.array([0.66745387, 0.37393011, 0.07839815, 0.39788222, 0.68525645,
                           0.1135878, 0.54680112, 0.76521719, 0.06463682, 0.46406943,
                           0.7922931, 0.95806476, 0.51846068, 0.06952451, 0.07222627,
                           0.19793278, 0.34558805, 0.08172587, 0.08332238, 0.6322325,
                           0.46489023, 0.66037147, 0.06351993, 0.86848459, 0.94208818,
                           0.5180761, 0.46945286, 0.83407335, 0.43001843, 0.95114709,
                           0.79497774, 0.27064889, 0.22297726, 0.35596628, 0.27442176,
                           0.])
        return states
    elif mode == 3:
        # 调用test_case.py中的fixed_grid_environment函数
        return fixed_grid_environment()

def simulate_gdm(model_path, interval=1.5, enable_layer_dist=True):
    env, _, _ = make_aigc_env(1, 1)
    custom_state = state(3)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建GDM模型
    actor_net = MLP(state_dim=state_dim, action_dim=action_dim)
    actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor_net,
        max_action=max_action,
        beta_schedule='vp',
        n_timesteps=6
    )
    
    # 加载模型权重
    actor = load_actor(actor, model_path, device)
    
    # 调用可视化函数并显示窗口
    launch_visualization(actor, custom_state, device, interval=interval, enable_layer_dist=enable_layer_dist)
    plt.show()  # 阻塞显示图形窗口

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GDM模型UAV网络仿真')
    parser.add_argument('--model', type=str, default="log/default/diffusion/model-42/policy.pth",
                        help='模型路径')
    parser.add_argument('--interval', type=float, default=1.5,
                        help='更新间隔(秒)')
    parser.add_argument('--disable-layer-dist', action='store_true',
                        help='禁用层分布可视化')
    
    args = parser.parse_args()
    
    simulate_gdm(
        model_path=args.model, 
        interval=args.interval, 
        enable_layer_dist=not args.disable_layer_dist
    )