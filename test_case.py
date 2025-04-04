import numpy as np
import time
import os
import sys

# 添加环境目录到系统路径，以便导入env模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'env'))
import config as cnf

def fixed_grid_environment():
    """
    生成一个观测空间，其中地面基站等距均匀分布在矩形空间中，用户也均匀分布（固定值）
    
    返回:
        state: 归一化的观测空间，包含位置和功率分配信息
    """
    n_a = cnf.NUM_A_AP
    n_g = cnf.NUM_G_AP
    n_u = cnf.NUM_USERS
    
    # 1. 为无人机生成随机位置 (x, y, h) - 用随机位置，因为无人机通常不是固定的
    x_a = np.random.uniform(0, 1, n_a)
    y_a = np.random.uniform(0, 1, n_a)
    h_a = np.random.uniform(0.2, 1, n_a)  # 高度归一化，避免太低
    
    # 2. 地面基站等距均匀分布
    # 计算网格尺寸
    grid_size_x = int(np.ceil(np.sqrt(n_g)))
    grid_size_y = int(np.ceil(n_g / grid_size_x))
    
    x_grid = np.linspace(0.1, 0.9, grid_size_x)
    y_grid = np.linspace(0.1, 0.9, grid_size_y)
    
    x_g = np.zeros(n_g)
    y_g = np.zeros(n_g)
    
    idx = 0
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            if idx < n_g:
                x_g[idx] = x_grid[j]
                y_g[idx] = y_grid[i]
                idx += 1
    
    # 3. 用户均匀分布
    # 使用更多格点确保用户分布更均匀
    user_grid_size_x = int(np.ceil(np.sqrt(n_u)))
    user_grid_size_y = int(np.ceil(n_u / user_grid_size_x))
    
    x_user_grid = np.linspace(0.05, 0.95, user_grid_size_x)
    y_user_grid = np.linspace(0.05, 0.95, user_grid_size_y)
    
    x_u = np.zeros(n_u)
    y_u = np.zeros(n_u)
    
    idx = 0
    for i in range(user_grid_size_y):
        for j in range(user_grid_size_x):
            if idx < n_u:
                x_u[idx] = x_user_grid[j]
                y_u[idx] = y_user_grid[i]
                idx += 1
    
    # 4. 随机生成功率分配
    num_links_a = n_a * n_u
    num_links_g = n_g * (n_u + n_a)
    num_links = num_links_a + num_links_g
    p_alloc = np.random.uniform(0, 1, num_links)
    
    # 5. 将所有部分组合成状态向量
    position = np.concatenate([x_a, y_a, h_a, x_g, y_g, x_u, y_u])
    state = np.concatenate([position, p_alloc, [0]])  # 添加初始奖励值0
    
    return state

def random_users_environment():
    """
    生成一个观测空间，其中地面基站等距均匀分布在矩形空间中，用户均匀随机分布
    
    返回:
        state: 归一化的观测空间，包含位置和功率分配信息
    """
    # 使用当前时间作为随机种子
    np.random.seed(int(time.time() * 1000) % (2**32 - 1))
    
    n_a = cnf.NUM_A_AP
    n_g = cnf.NUM_G_AP
    n_u = cnf.NUM_USERS
    
    # 1. 为无人机生成随机位置 (x, y, h)
    x_a = np.random.uniform(0, 1, n_a)
    y_a = np.random.uniform(0, 1, n_a)
    h_a = np.random.uniform(0.2, 1, n_a)  # 高度归一化，避免太低
    
    # 2. 地面基站等距均匀分布 (与第一个函数相同)
    grid_size_x = int(np.ceil(np.sqrt(n_g)))
    grid_size_y = int(np.ceil(n_g / grid_size_x))
    
    x_grid = np.linspace(0.1, 0.9, grid_size_x)
    y_grid = np.linspace(0.1, 0.9, grid_size_y)
    
    x_g = np.zeros(n_g)
    y_g = np.zeros(n_g)
    
    idx = 0
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            if idx < n_g:
                x_g[idx] = x_grid[j]
                y_g[idx] = y_grid[i]
                idx += 1
    
    # 3. 用户均匀随机分布
    x_u = np.random.uniform(0, 1, n_u)
    y_u = np.random.uniform(0, 1, n_u)
    
    # 4. 随机生成功率分配
    num_links_a = n_a * n_u
    num_links_g = n_g * (n_u + n_a)
    num_links = num_links_a + num_links_g
    p_alloc = np.random.uniform(0, 1, num_links)
    
    # 5. 将所有部分组合成状态向量
    position = np.concatenate([x_a, y_a, h_a, x_g, y_g, x_u, y_u])
    state = np.concatenate([position, p_alloc, [0]])  # 添加初始奖励值0
    
    return state

def corner_base_stations_environment():
    """
    生成一个观测空间，其中地面基站分布在矩形左下侧1/4处，用户均匀随机分布
    
    返回:
        state: 归一化的观测空间，包含位置和功率分配信息
    """
    # 使用当前时间作为随机种子
    np.random.seed(int(time.time() * 1000) % (2**32 - 1))
    
    n_a = cnf.NUM_A_AP
    n_g = cnf.NUM_G_AP
    n_u = cnf.NUM_USERS
    
    # 1. 为无人机生成随机位置 (x, y, h)
    x_a = np.random.uniform(0, 1, n_a)
    y_a = np.random.uniform(0, 1, n_a)
    h_a = np.random.uniform(0.2, 1, n_a)  # 高度归一化，避免太低
    
    # 2. 地面基站分布在矩形左下侧1/4处
    # 计算左下1/4区域的边界
    x_max = 0.5
    y_max = 0.5
    
    # 在左下1/4区域内等距分布基站
    grid_size_x = int(np.ceil(np.sqrt(n_g)))
    grid_size_y = int(np.ceil(n_g / grid_size_x))
    
    x_grid = np.linspace(0.05, x_max - 0.05, grid_size_x)
    y_grid = np.linspace(0.05, y_max - 0.05, grid_size_y)
    
    x_g = np.zeros(n_g)
    y_g = np.zeros(n_g)
    
    idx = 0
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            if idx < n_g:
                x_g[idx] = x_grid[j]
                y_g[idx] = y_grid[i]
                idx += 1
    
    # 3. 用户均匀随机分布在整个区域
    x_u = np.random.uniform(0, 1, n_u)
    y_u = np.random.uniform(0, 1, n_u)
    
    # 4. 随机生成功率分配
    num_links_a = n_a * n_u
    num_links_g = n_g * (n_u + n_a)
    num_links = num_links_a + num_links_g
    p_alloc = np.random.uniform(0, 1, num_links)
    
    # 5. 将所有部分组合成状态向量
    position = np.concatenate([x_a, y_a, h_a, x_g, y_g, x_u, y_u])
    state = np.concatenate([position, p_alloc, [0]])  # 添加初始奖励值0
    
    return state

def main():
    """
    测试三个环境生成函数并可视化结果
    """
    import matplotlib.pyplot as plt
    
    # 测试三个函数
    state1 = fixed_grid_environment()
    state2 = random_users_environment()
    state3 = corner_base_stations_environment()
    
    # 提取位置信息
    n_a = cnf.NUM_A_AP
    n_g = cnf.NUM_G_AP
    n_u = cnf.NUM_USERS
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [
        "Fixed Grid Environment",
        "Random Users Environment",
        "Corner Base Stations Environment"
    ]
    
    states = [state1, state2, state3]
    
    for i, (ax, state, title) in enumerate(zip(axes, states, titles)):
        # 提取位置信息
        aerial_end = 3 * n_a
        ground_end = aerial_end + 2 * n_g
        user_end = ground_end + 2 * n_u
        
        # 提取无人机坐标
        x_a = state[:n_a]
        y_a = state[n_a:2*n_a]
        h_a = state[2*n_a:3*n_a]
        
        # 提取地面基站坐标
        x_g = state[3*n_a:3*n_a+n_g]
        y_g = state[3*n_a+n_g:3*n_a+2*n_g]
        
        # 提取用户坐标
        x_u = state[3*n_a+2*n_g:3*n_a+2*n_g+n_u]
        y_u = state[3*n_a+2*n_g+n_u:3*n_a+2*n_g+2*n_u]
        
        # 绘制节点
        ax.scatter(x_g, y_g, c='red', marker='^', s=100, label='Ground BS')
        ax.scatter(x_u, y_u, c='green', marker='o', s=80, label='User')
        ax.scatter(x_a, y_a, c='blue', marker='s', s=80, label='UAV')
        
        # 为无人机添加高度标签
        for j, (x, y, h) in enumerate(zip(x_a, y_a, h_a)):
            ax.annotate(f'h={h:.2f}', (x, y), xytext=(x, y-0.05), 
                         fontsize=8, ha='center', va='top')
        
        # 设置轴标签和标题
        ax.set_xlabel('X-coordinate (normalized)')
        ax.set_ylabel('Y-coordinate (normalized)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('test_environments.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()