# Python
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import time
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor
from tianshou.policy import PPOPolicy
from env import make_aigc_env
from env import config as cnf
from env.utility import arr2mat, arr_prep, calc_cluster

# 全局变量，用于记录 UAV 历史轨迹，每个元素为 (旧的 x_a, y_a)
history = []

# 自定义PPO Actor，与train_PPO.py中保持一致
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

# 辅助函数：在指定坐标轴 ax 中绘制网络状态和预测动作，
# 如果 history 不为空，则用灰色连线显示 UAV 历史轨迹
def plot_network(ax, state, action, history=None):
    # 从 state 中剔除 reward 部分
    state_clean = state[:-1]
    # 调用预处理函数 map，将 state 和 action 拆分为各部分
    position, _, move, power_alloc_action = arr_prep(state_clean, action)
    
    n_a = cnf.NUM_A_AP
    n_g = cnf.NUM_G_AP
    n_u = cnf.NUM_USERS

    # 拆分位置信息
    aerial_end = 3 * n_a
    ground_end = aerial_end + 2 * n_g
    user_end = ground_end + 2 * n_u
    aerial_state = position[:aerial_end].reshape(3, n_a)  # [x_a, y_a, h_a]
    x_a, y_a, h_a = aerial_state
    ground_state = position[aerial_end:ground_end].reshape(2, n_g)  # [x_g, y_g]
    x_g, y_g = ground_state
    user_state = position[ground_end:user_end].reshape(2, n_u)  # [x_u, y_u]
    x_u, y_u = user_state

    ax.clear()
    ax.set_title('Network State and Predicted Actions')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True)

    # 绘制地面AP
    ax.scatter(x_g, y_g, c='red', marker='^', s=100, label='Ground AP')
    for idx, (xi, yi) in enumerate(zip(x_g, y_g)):
        ax.text(xi, yi-20, f'G{idx}', fontsize=10, ha='center', color='red')
        
    # 绘制用户
    ax.scatter(x_u, y_u, c='green', marker='o', s=100, label='User')
    for idx, (xi, yi) in enumerate(zip(x_u, y_u)):
        ax.text(xi, yi-20, f'U{idx}', fontsize=10, ha='center', color='green')
        
    # 绘制空中AP（UAV），并显示高度信息
    ax.scatter(x_a, y_a, c='blue', marker='s', s=100, label='UAV')
    for idx, (xi, yi, hi) in enumerate(zip(x_a, y_a, h_a)):
        ax.text(xi, yi-30, f'A{idx}\nH:{hi:.1f}', fontsize=10, ha='center', color='blue')
    
    # 绘制 UAV 运动方向箭头及新位置标记
    new_x_a = x_a + move[:n_a]
    new_y_a = y_a + move[n_a:2*n_a]
    for i in range(n_a):
        ax.arrow(x_a[i], y_a[i], move[i], move[n_a+i],
                 head_width=2, head_length=2, fc='blue', ec='blue', alpha=0.3)
        ax.scatter(new_x_a[i], new_y_a[i], c='blue', marker='x', s=100)
        
    # 绘制地面AP到用户的连线
    for g in range(n_g):
        for u in range(n_u):
            if G_eta := arr2mat(power_alloc_action, position)[1][g, u] > 0:
                ax.plot([x_g[g], x_u[u]], [y_g[g], y_u[u]], 'r--', alpha=0.3)
                mid_x = (x_g[g] + x_u[u]) / 2
                mid_y = (y_g[g] + y_u[u]) / 2
                ax.text(mid_x, mid_y, f'{arr2mat(power_alloc_action, position)[1][g, u]:.2f}', fontsize=8)

    # 绘制空中AP到用户的连线
    for a in range(n_a):
        for u in range(n_u):
            if A_eta := arr2mat(power_alloc_action, position)[3][a, u] > 0:
                ax.plot([x_a[a], x_u[u]], [y_a[a], y_u[u]], 'b--', alpha=0.3)
                mid_x = (x_a[a] + x_u[u]) / 2
                mid_y = (y_a[a] + y_u[u]) / 2
                ax.text(mid_x, mid_y, f'{arr2mat(power_alloc_action, position)[3][a, u]:.2f}', fontsize=8)
    
    # 绘制地面AP到空中AP的连线
    for g in range(n_g):
        for a in range(n_a):
            idx = n_u + a
            if arr2mat(power_alloc_action, position)[1][g, idx] > 0:
                ax.plot([x_g[g], x_a[a]], [y_g[g], y_a[a]], 'r--', alpha=0.3)
                mid_x = (x_g[g] + x_a[a]) / 2
                mid_y = (y_g[g] + y_a[a]) / 2
                ax.text(mid_x, mid_y, f'{arr2mat(power_alloc_action, position)[1][g, idx]:.2f}', fontsize=8)
    
    # 计算AP聚簇情况（仅考虑地面AP和UAV）
    _, cluster_labels = calc_cluster(arr2mat(power_alloc_action, position)[3], arr2mat(power_alloc_action, position)[1])
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if idx < n_g + n_a:
            clusters.setdefault(label, []).append(f'{"G" if idx<n_g else "A"+str(idx-n_g)}')
    cluster_annotation = ', '.join([f'Cluster{label+1}: {" ".join(clusters[label])}' for label in sorted(clusters.keys())])
    ax.text(0.5, -0.12, cluster_annotation, fontsize=12, color='black',
            transform=ax.transAxes, ha='center')
    
    # 绘制 UAV 历史轨迹（仅显示 UAV 历史位置）
    if history:
        for (hx, hy) in history:
            ax.scatter(hx, hy, color='gray', marker='o', s=50, alpha=0.7)
    
    ax.legend()

def simulate_ppo(model_path, interval=1.5):
    # 创建环境、生成初始状态，并加载模型
    env, _, _ = make_aigc_env(1, 1)
    n_a = cnf.NUM_A_AP
    custom_state = state(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0
    
    # 创建PPO模型的actor网络
    net = Net(
        state_dim,
        hidden_sizes=[256, 256],
        activation=nn.Mish,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    actor = PPOActor(
        net,
        action_dim,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_action=max_action
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载模型参数
    state_dict = torch.load(model_path, map_location=device)
    actor_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('actor_critic.actor.') or k.startswith('_actor.') or k.startswith('actor.'):
            # 提取actor相关参数
            key = k.replace('actor_critic.actor.', '').replace('_actor.', '').replace('actor.', '')
            actor_state_dict[key] = v
    
    actor.load_state_dict(actor_state_dict, strict=False)
    actor.to(device)
    actor.eval()

    # 初始化图形窗口
    fig, ax = plt.subplots(figsize=(12,8))
    plt.subplots_adjust(bottom=0.25)

    # 初始预测并绘图 
    with torch.no_grad():
        tensor_state = torch.FloatTensor(custom_state).unsqueeze(0).to(device)
        # PPO actor返回(均值,标准差)元组，只需要均值
        predicted_action, _ = actor(tensor_state)
        # 获取均值作为动作
        mean_action = predicted_action[0]
    
    # 确保动作是一维数组
    action = mean_action.detach().cpu().numpy().flatten()
    plot_network(ax, custom_state, action, history)

    # 定义自动更新函数
    def update(event=None):
        nonlocal custom_state
        with torch.no_grad():
            tensor_state = torch.FloatTensor(custom_state).unsqueeze(0).to(device)
            # PPO actor返回(均值,标准差)元组，只需要均值
            predicted_action, _ = actor(tensor_state)
            # 获取均值作为动作
            mean_action = predicted_action[0]
        
        # 确保动作是一维数组
        action = mean_action.detach().cpu().numpy().flatten()
        
        # 利用 map 函数获得当前 UAV 运动量（move），进而计算新 UAV 坐标
        state_clean = custom_state[:-1]
        _, _, move, _ = arr_prep(state_clean, action)
        # UAV 初始坐标：custom_state[0:n_a] 为 x 坐标，custom_state[n_a:2*n_a] 为 y 坐标
        x_a_old = custom_state[0:n_a].copy()
        y_a_old = custom_state[n_a:2*n_a].copy()
        x_a_new = x_a_old + move[:n_a]
        y_a_new = y_a_old + move[n_a:2*n_a]
        # 限制 UAV 坐标在 [0, MAX_X] 和 [0, MAX_Y] 内
        x_a_new = np.clip(x_a_new, 0, cnf.MAX_X)
        y_a_new = np.clip(y_a_new, 0, cnf.MAX_Y)
        # 记录历史轨迹
        history.append((x_a_old, y_a_old))
        # 更新状态中的 UAV 坐标（高度保持不变）
        custom_state[0:n_a] = x_a_new
        custom_state[n_a:2*n_a] = y_a_new
        # 重新绘制图像
        plot_network(ax, custom_state, action, history)
        fig.canvas.draw_idle()

    # 构造 Timer 用于自动更新: interval 单位为毫秒
    timer = fig.canvas.new_timer(interval=int(interval * 1000))
    timer.add_callback(update)

    # 定义 Start 按钮回调，启动定时器
    def start_auto(event):
        timer.start()

    # 定义 Stop 按钮回调，停止定时器
    def stop_auto(event):
        timer.stop()

    # 创建 Start 和 Stop 按钮
    ax_start = plt.axes([0.3, 0.1, 0.2, 0.075])
    btn_start = Button(ax_start, 'Start')
    btn_start.on_clicked(start_auto)

    ax_stop = plt.axes([0.6, 0.1, 0.2, 0.075])
    btn_stop = Button(ax_stop, 'Stop')
    btn_stop.on_clicked(stop_auto)

    plt.show()

def state(mode):
    if mode == 1:
        current_time = int(time.time() * 1000) & 0xFFFFFFFF
        np.random.seed(current_time)
        x_a = np.random.uniform(0, cnf.MAX_X, cnf.NUM_A_AP)
        y_a = np.random.uniform(0, cnf.MAX_Y, cnf.NUM_A_AP)
        h_a = np.random.uniform(0, cnf.MAX_H, cnf.NUM_A_AP)
        x_g = np.random.uniform(0, cnf.MAX_X, cnf.NUM_G_AP)
        y_g = np.random.uniform(0, cnf.MAX_Y, cnf.NUM_G_AP)
        x_u = np.random.uniform(0, cnf.MAX_X, cnf.NUM_USERS)
        y_u = np.random.uniform(0, cnf.MAX_Y, cnf.NUM_USERS)
        num_links_a = cnf.NUM_A_AP * cnf.NUM_USERS
        num_links_g = cnf.NUM_G_AP * (cnf.NUM_USERS + cnf.NUM_A_AP)
        num_links = num_links_a + num_links_g
        p_alloc = np.random.uniform(0, 1, num_links)
        reward_in = [0]
        states = np.concatenate([x_a, y_a, h_a, x_g, y_g, x_u, y_u, p_alloc, reward_in])
        return states

if __name__ == "__main__":
    # 使用PPO模型路径
    model_path = "log/default/PPO/model-1/policy.pth"  # 替换为您的PPO模型路径
    simulate_ppo(model_path)