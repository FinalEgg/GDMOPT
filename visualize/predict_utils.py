import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from env import config as cnf
from env.utility import arr2mat, arr_prep, calc_cluster
import os

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(profile="full")

def load_actor(actor, model_path, device):
    """
    Load the specified actor model weights and transfer the model to the specified device.
    Also strip any prefixes such as "actor." or "_actor." from the state dict keys.
    同时将加载的权重数据保存到本代码所在目录的 weights.txt 文件中，并注明层序信息。
    """
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    # Extract actor-related parameters and remove possible prefixes.
    for k, v in state_dict.items():
        new_key = k.replace('actor.', '').replace('_actor.', '')
        new_state_dict[new_key] = v
    try:
        actor.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print("Error loading actor state_dict: ", str(e))
        print("Some parameters may not match. Proceeding with partial model parameters.")
    actor.to(device)
    actor.eval()

    # # 保存权重数据到本代码所在目录的 weights.txt
    # current_dir = os.path.dirname(__file__)
    # weights_file = os.path.join(current_dir, 'weights.txt')
    # try:
    #     with open(weights_file, 'w') as f:
    #         f.write("Weights:\n")
    #         for idx, (layer_name, weight) in enumerate(new_state_dict.items(), 1):
    #             f.write(f"layer:{idx} - {layer_name}:\n")
    #             f.write(str(weight) + "\n\n")
    #     print(f"模型权重已成功保存至: {weights_file}")
    # except Exception as e:
    #     print("保存权重文件失败:", str(e))
    return actor

def predict_action(actor, state, device, predict_method="sample"):
    """
    针对传入的状态 state（通常为一维 NumPy 数组），
    构造 tensor 后调用 actor 进行预测，并返回 NumPy 数组格式的预测结果。
    参数 predict_method 指定调用 actor 的方式，当 actor 实现 sample 方法时可使用该方法，
    否则直接调用 actor 的 forward 方法。
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        if predict_method == "sample" and hasattr(actor, "sample"):
            predicted = actor.sample(state_tensor)
        else:
            predicted = actor(state_tensor)
    # 如果返回的是元组（例如 PPO 返回均值与标准差），取第一个结果
    if isinstance(predicted, tuple):
        predicted = predicted[0]
    return predicted.cpu().numpy().squeeze(0)

def load_and_predict(actor, model_path, state, device, predict_method="sample"):
    """
    一个便捷的封装函数，先加载 actor 模型，再根据给定状态 state 执行预测。
    返回预测的动作（NumPy 数组）。
    """
    actor = load_actor(actor, model_path, device)
    return predict_action(actor, state, device, predict_method)

def plot_network(ax, state, action, history=None):
    """
    在指定的坐标轴 ax 中绘制网络状态、预测动作、各节点连线，并在图像下方标注 reward、簇划分以及每个用户的信道容量。
    """
    # 根据 state 和 action 提取实际位置、移动量等信息
    pos_len = 3 * cnf.NUM_A_AP + 2 * cnf.NUM_G_AP + 2 * cnf.NUM_USERS
    actual_position, _, move, power_alloc_action, norm_position, norm_move = arr_prep(state[:pos_len], action)

    n_a = cnf.NUM_A_AP
    n_g = cnf.NUM_G_AP
    n_u = cnf.NUM_USERS

    # 拆分位置信息
    aerial_end = 3 * n_a
    ground_end = aerial_end + 2 * n_g
    user_end = ground_end + 2 * n_u

    aerial_state = actual_position[:aerial_end].reshape(3, n_a)  # [x_a, y_a, h_a]
    x_a, y_a, h_a = aerial_state
    ground_state = actual_position[aerial_end:ground_end].reshape(2, n_g)  # [x_g, y_g]
    x_g, y_g = ground_state
    user_state = actual_position[ground_end:user_end].reshape(2, n_u)      # [x_u, y_u]
    x_u, y_u = user_state

    # 设置坐标轴属性
    ax.clear()
    ax.set_title('Network State and Predicted Actions')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True)
    ax.set_xlim(0, cnf.MAX_X)
    ax.set_ylim(0, cnf.MAX_Y)

    # 绘制地面AP（红色三角形）
    ax.scatter(x_g, y_g, c='red', marker='^', s=100, label='Ground AP')
    for idx, (xi, yi) in enumerate(zip(x_g, y_g)):
        ax.text(xi, yi - 20, f'G{idx}', fontsize=10, ha='center', color='red')

    # 绘制用户（绿色圆圈）并标注各用户总信道容量（后续计算）
    ax.scatter(x_u, y_u, c='green', marker='o', s=100, label='User')
    # 先计算通过 arr2mat 得到的矩阵及用户信道容量
    G_beta, G_eta, A_beta, A_eta = arr2mat(power_alloc_action, actual_position)
    # Ground capacity：G_c,计算每个用户来自地面AP的容量
    G_c = np.zeros((n_g, n_a + n_u))
    for i in range(n_u + n_a):
        noise_coef = np.sum(np.delete(G_eta, i, axis=1), axis=1)
        tot_noise = np.sum(noise_coef * cnf.P_G * G_beta[:, i])
        signal = cnf.P_G * G_beta[:, i] * G_eta[:, i]
        G_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))
    ground_cap_user = np.sum(G_c[:, :n_u], axis=0)
    # Aerial capacity：A_c，计算每个用户来自 UAV 的容量
    A_c = np.zeros((n_a, n_u))
    for i in range(n_u):
        noise_coef = np.sum(np.delete(A_eta, i, axis=1), axis=1)
        tot_noise = np.sum(noise_coef * cnf.P_A * A_beta[:, i])
        signal = cnf.P_A * A_beta[:, i] * A_eta[:, i]
        A_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))
    aerial_cap_user = np.sum(A_c, axis=0)
    user_total_cap = ground_cap_user + aerial_cap_user

    for idx, (xi, yi) in enumerate(zip(x_u, y_u)):
        ax.text(xi, yi - 20, f'U{idx}\nCap:{user_total_cap[idx]:.2f}', fontsize=10, ha='center', color='green')

    # 绘制UAV（蓝色方形）以及运动方向
    ax.scatter(x_a, y_a, c='blue', marker='s', s=100, label='UAV')
    new_x_a = x_a + move[:n_a]
    new_y_a = y_a + move[n_a:2*n_a]
    for idx, (xi, yi, hi) in enumerate(zip(x_a, y_a, h_a)):
        ax.text(xi, yi - 30, f'A{idx}\nH:{hi:.1f}', fontsize=10, ha='center', color='blue')
        ax.arrow(xi, yi, move[idx], move[n_a+idx],
                 head_width=2, head_length=2, fc='blue', ec='blue', alpha=0.3)
        ax.scatter(new_x_a[idx], new_y_a[idx], c='blue', marker='x', s=100)

    # 绘制连线：Ground AP 到 User（红色虚线）
    mat_G = arr2mat(power_alloc_action, actual_position)
    for g in range(n_g):
        for u in range(n_u):
            if mat_G[1][g, u] > 0:
                ax.plot([x_g[g], x_u[u]], [y_g[g], y_u[u]], 'r--', alpha=0.3)
                mid_x = (x_g[g] + x_u[u]) / 2
                mid_y = (y_g[g] + y_u[u]) / 2
                ax.text(mid_x, mid_y, f'{mat_G[1][g, u]:.2f}', fontsize=8)

    # 绘制连线：UAV 到 User（蓝色虚线），并标注系数
    mat_AU = arr2mat(power_alloc_action, actual_position)[3]
    for a in range(n_a):
        for u in range(n_u):
            if mat_AU[a, u] > 0:
                ax.plot([x_a[a], x_u[u]], [y_a[a], y_u[u]], 'b--', alpha=0.3)
                mid_x = (x_a[a] + x_u[u]) / 2
                mid_y = (y_a[a] + y_u[u]) / 2
                ax.text(mid_x, mid_y, f'{mat_AU[a, u]:.2f}', fontsize=8)

    # 绘制连线：Ground AP 到 UAV（绿色虚线），并标注系数
    for g in range(n_g):
        for a in range(n_a):
            uav_col = n_u + a  # 假设矩阵中前 n_u 列为用户，后 n_a 列为 UAV
            if mat_G[1][g, uav_col] > 0:
                ax.plot([x_g[g], x_a[a]], [y_g[g], y_a[a]], 'g--', alpha=0.3)
                mid_x = (x_g[g] + x_a[a]) / 2
                mid_y = (y_g[g] + y_a[a]) / 2
                ax.text(mid_x, mid_y, f'{mat_G[1][g, uav_col]:.2f}', fontsize=8)

    # 计算 reward 及簇划分信息（调用 calc_util 和 calc_cluster）
    from env.utility import calc_util
    reward_val, _, _, _, _, _, _ = calc_util(state, action)
    _, cluster_labels = calc_cluster(A_eta, G_eta)
    total_nodes = n_g + n_a + n_u
    clusters = {}
    for idx in range(total_nodes):
        # 对于 ground APs、UAVs、用户，根据索引转成对应标识
        if idx < n_g:
            label = f'G{idx}'
        elif idx < n_g + n_a:
            label = f'A{idx - n_g}'
        else:
            label = f'U{idx - n_g - n_a}'
        clusters.setdefault(cluster_labels[idx] + 1, []).append(label)
    cluster_text = "\n".join([f"Cluster {cid}: " + " ".join(clusters[cid]) for cid in sorted(clusters.keys())])

    # 在图像下方添加 reward 与簇划分信息
    final_text = f"Reward: {reward_val:.2f}\n{cluster_text}"
    ax.text(0.5, -0.12, final_text, transform=ax.transAxes, fontsize=12, ha='center', va='top')
    
    # 绘制 UAV 历史轨迹
    if history:
        for uav_idx in history.keys():
            traj = history[uav_idx]
            if traj:
                xs, ys = zip(*traj)
                ax.plot(xs, ys, marker='o', linestyle='--', color='gray',
                        alpha=0.7, label=f'UAV {uav_idx} History')
    ax.legend()

def plot_layer_distributions(actor, input_tensor):
    """
    计算并绘制网络内部各层输出值的分布图，
    每一层结果以直方图展示，反映该层输出的分布范围。
    """
    import torch.nn.functional as F
    with torch.no_grad():
        nodes = {}
        nodes["Input"] = input_tensor
        x = actor.preprocess.fc1(input_tensor)
        nodes["fc1"] = x
        x = F.mish(x)
        nodes["mish1"] = x
        x = actor.preprocess.fc2(x)
        nodes["fc2"] = x
        x = F.mish(x)
        nodes["mish2"] = x
        x_last = actor.last(x)
        nodes["last_linear"] = x_last
        x_out = actor._max * torch.tanh(x_last / 5)
        nodes["Output"] = x_out

    num_layers = len(nodes)
    fig_dist = plt.figure("Layer Distributions", figsize=(12, 3 * num_layers))
    for idx, (name, tensor_val) in enumerate(nodes.items()):
        ax_sub = fig_dist.add_subplot(num_layers, 1, idx + 1)
        data = tensor_val.cpu().numpy().reshape(-1)
        ax_sub.hist(data, bins=30, color='skyblue', edgecolor='gray')
        ax_sub.set_title(f"{name}: shape={list(tensor_val.shape)}")
        ax_sub.set_xlabel("Value")
        ax_sub.set_ylabel("Count")
    fig_dist.tight_layout()
    fig_dist.show()

def launch_visualization(actor, custom_state, device, interval=1.5, enable_layer_dist=True):
    """
    创建一个可视化窗口，用于展示网络状态、预测动作、UAV历史轨迹，
    以及显示当前 reward、簇划分信息。顶部左侧一横排放置三个按钮。
    
    返回: (fig, ax, history)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(top=0.85, bottom=0.15)  # 为顶部按钮和底部信息预留空间
    
    # 初始化 history 为字典，每个 UAV 对应一个空列表
    history = {i: [] for i in range(cnf.NUM_A_AP)}
    
    with torch.no_grad():
        tensor_state = torch.FloatTensor(custom_state).unsqueeze(0).to(device)
        predicted_action = actor(tensor_state)[0]
    action = predicted_action.detach().cpu().numpy().flatten()
    plot_network(ax, custom_state, action, history)
    
    # 在左上角横排放置三个按钮：Start, Stop, Show Dist
    ax_start = plt.axes([0.01, 0.88, 0.1, 0.05])
    btn_start = Button(ax_start, 'Start')
    ax_stop = plt.axes([0.12, 0.88, 0.1, 0.05])
    btn_stop = Button(ax_stop, 'Stop')
    ax_dist = plt.axes([0.23, 0.88, 0.1, 0.05])
    btn_dist = Button(ax_dist, 'Show Dist')
    
    if enable_layer_dist:
        def click_show_dist(event):
            with torch.no_grad():
                tensor_state = torch.FloatTensor(custom_state).unsqueeze(0).to(device)
            plot_layer_distributions(actor, tensor_state)
        btn_dist.on_clicked(click_show_dist)
    
    def update(event=None):
        nonlocal custom_state, history
        # 预测动作
        with torch.no_grad():
            tensor_state = torch.FloatTensor(custom_state).unsqueeze(0).to(device)
            predicted_action = actor(tensor_state)[0]
        action = predicted_action.detach().cpu().numpy().flatten()
        
        # 提取归一化位置部分（前 pos_len 个元素）
        pos_len = 3 * cnf.NUM_A_AP + 2 * cnf.NUM_G_AP + 2 * cnf.NUM_USERS
        norm_pos = custom_state[:pos_len]
        # 利用归一化状态和动作计算实际位置及归一化移动量
        actual_position, _, _, _, _, norm_move = arr_prep(norm_pos, action)
        
        # 记录 UAV 历史轨迹（转换为非归一化后便于观察）
        aerial_actual = actual_position[:3*cnf.NUM_A_AP].reshape(3, cnf.NUM_A_AP)
        for i in range(cnf.NUM_A_AP):
            history[i].append((aerial_actual[0, i].copy(), aerial_actual[1, i].copy()))
    
        # 更新归一化位置部分：仅更新 x, y, h（前 3*NUM_A_AP 元素）
        new_norm_x = norm_pos[0:cnf.NUM_A_AP] + norm_move[0:cnf.NUM_A_AP]
        new_norm_y = norm_pos[cnf.NUM_A_AP:2*cnf.NUM_A_AP] + norm_move[cnf.NUM_A_AP:2*cnf.NUM_A_AP]
        new_norm_h = norm_pos[2*cnf.NUM_A_AP:3*cnf.NUM_A_AP] + norm_move[2*cnf.NUM_A_AP:3*cnf.NUM_A_AP]
    
        new_norm_x = np.clip(new_norm_x, 0, 1)
        new_norm_y = np.clip(new_norm_y, 0, 1)
        new_norm_h = np.clip(new_norm_h, 0, 1)
        
        updated_norm_pos = np.concatenate([
            new_norm_x,
            new_norm_y,
            new_norm_h,
            norm_pos[3*cnf.NUM_A_AP:]
        ])
        custom_state[:pos_len] = updated_norm_pos
    
        plot_network(ax, custom_state, action, history)
        fig.canvas.draw_idle()
    
    timer = fig.canvas.new_timer(interval=int(interval * 1000))
    timer.add_callback(update)
    
    def start_auto(event):
        timer.start()
    
    def stop_auto(event):
        timer.stop()
    
    btn_start.on_clicked(start_auto)
    btn_stop.on_clicked(stop_auto)
    
    plt.show()
    return fig, ax, history