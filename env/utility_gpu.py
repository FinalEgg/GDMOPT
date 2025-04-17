import cupy as cp
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
from env import config as cnf
import networkx as nx
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'

n_a = cnf.NUM_A_AP  # number of aerial APs
n_g = cnf.NUM_G_AP  # number of ground APs
n_u = cnf.NUM_USERS # number of users

def arr_prep(state, aution):
    """
    使用 CuPy 在 GPU 上预处理状态和动作。
    
    Parameters:
      state: 表示状态的 1D 数组
      aution: 表示动作的 1D 数组
    
    Returns:
      actual_position: 通过缩放归一化位置得到的实际位置
      power_alloc_state: 状态中的功率分配部分
      actual_move: 通过缩放归一化移动得到的实际移动值
      power_alloc_action: 映射到[0,1]范围的功率分配动作
      norm_position: 原始归一化位置(范围[0,1])
      norm_move: 原始归一化移动值(范围[0,1])
    """
    # 转换为 CuPy 数组
    state = cp.asarray(state)
    aution = cp.asarray(aution)
    
    # 确定归一化位置部分的长度
    pos_len = 3 * cnf.NUM_A_AP + 2 * cnf.NUM_G_AP + 2 * cnf.NUM_USERS
    
    # 提取归一化位置和功率分配状态
    norm_position = state[:pos_len]
    power_alloc_state = state[pos_len:]
    
    # 构建缩放向量 - 一次性创建所有缩放因子
    # 空中 AP 缩放因子: [MAX_X, MAX_X, ..., MAX_Y, MAX_Y, ..., MAX_H, MAX_H, ...]
    aerial_scales = cp.concatenate([
        cp.full(cnf.NUM_A_AP, cnf.MAX_X),
        cp.full(cnf.NUM_A_AP, cnf.MAX_Y),
        cp.full(cnf.NUM_A_AP, cnf.MAX_H)
    ])
    
    # 地面 AP 缩放因子: [MAX_X, MAX_X, ..., MAX_Y, MAX_Y, ...]
    ground_scales = cp.concatenate([
        cp.full(cnf.NUM_G_AP, cnf.MAX_X),
        cp.full(cnf.NUM_G_AP, cnf.MAX_Y)
    ])
    
    # 用户缩放因子: [MAX_X, MAX_X, ..., MAX_Y, MAX_Y, ...]
    user_scales = cp.concatenate([
        cp.full(cnf.NUM_USERS, cnf.MAX_X),
        cp.full(cnf.NUM_USERS, cnf.MAX_Y)
    ])
    
    # 将所有缩放因子合并为一个数组
    all_scales = cp.concatenate([aerial_scales, ground_scales, user_scales])
    
    # 使用向量化操作进行缩放 - 一次性计算所有位置
    actual_position = norm_position * all_scales
    
    # 处理 aution (动作) 部分
    move_length = cnf.NUM_A_AP * 3
    move = aution[:move_length]
    power_alloc_action = aution[move_length:]
    
    # 创建速度缩放因子数组
    velocity_scales = cp.concatenate([
        cp.full(cnf.NUM_A_AP, cnf.MAX_V_X),
        cp.full(cnf.NUM_A_AP, cnf.MAX_V_Y),
        cp.full(cnf.NUM_A_AP, cnf.MAX_V_H)
    ])
    
    # 映射到实际速度值
    actual_move = move * velocity_scales
    
    # 计算归一化移动
    norm_move = actual_move / aerial_scales[:move_length]
    
    # 将功率分配动作映射到 [0,1] 范围
    power_alloc_action = cp.clip(cp.tanh(power_alloc_action), 0, 1)
    
    return actual_position, power_alloc_state, actual_move, power_alloc_action, norm_position, norm_move

def arr2mat(power_alloc, position):
    """
    在 GPU 上处理功率分配和位置数据，转换为信道评估所需的结构化矩阵。
    
    Parameters:
      power_alloc: 表示功率分配系数的 1D 数组（范围 [0,1]）
      position: 包含 AP 和用户实际位置的 1D 数组
                预期顺序:
                  - 前 3*n_a 个元素：空中 AP 位置 (x, y, h)
                  - 接下来 2*n_g 个元素：地面 AP 位置 (x, y)
                  - 最后 2*n_u 个元素：用户位置 (x, y)
                  
    Returns:
      G_beta: 从地面 AP 到用户和空中 AP 的路径损耗因子矩阵
      G_eta: 地面 AP 的归一化功率分配矩阵（每行和上限为 1）
      A_beta: 从空中 AP 到用户的路径损耗因子矩阵
      A_eta: 空中 AP 的归一化功率分配矩阵（每行和上限为 1）
    """
    # 确保输入在 GPU 上
    power_alloc = cp.asarray(power_alloc)
    position = cp.asarray(position)
    
    # 重塑位置数据为结构化数组
    # 空中 AP：前 3*n_a 个元素 (x, y, h)
    aerial_positions = position[:3 * n_a].reshape(3, n_a)
    x_a, y_a, h_a = aerial_positions

    # 地面 AP：接下来 2*n_g 个元素 (x, y)
    ground_positions = position[3 * n_a:3 * n_a + 2 * n_g].reshape(2, n_g)
    x_g, y_g = ground_positions

    # 用户：最后 2*n_u 个元素 (x, y)
    user_positions = position[3 * n_a + 2 * n_g:3 * n_a + 2 * n_g + 2 * n_u].reshape(2, n_u)
    x_u, y_u = user_positions

    # 初始化矩阵
    G_beta = cp.zeros((n_g, n_a + n_u))
    A_beta = cp.zeros((n_a, n_u))
    
    # 向量化 calc_beta 函数:
    # 将循环替换为向量化操作，以提高 GPU 计算效率
    
    # 1. 计算地面 AP 到用户的 beta
    # 创建网格以同时执行所有计算
    g_indices, u_indices = cp.meshgrid(cp.arange(n_g), cp.arange(n_u), indexing='ij')
    
    # 计算水平距离和总距离
    hor_dist_gu = cp.sqrt((x_g[g_indices] - x_u[u_indices])**2 + 
                         (y_g[g_indices] - y_u[u_indices])**2)
    total_dist_gu = hor_dist_gu  # 高度为0，总距离等于水平距离
    
    # 用户为地面单元，没有LOS/NLOS区分，直接使用距离的NLOS次方
    G_beta[:, :n_u] = total_dist_gu ** cnf.NLOS
    
    # 2. 计算地面 AP 到空中 AP 的 beta
    g_indices, a_indices = cp.meshgrid(cp.arange(n_g), cp.arange(n_a), indexing='ij')
    
    # 计算水平距离和总距离
    hor_dist_ga = cp.sqrt((x_g[g_indices] - x_a[a_indices])**2 + 
                         (y_g[g_indices] - y_a[a_indices])**2)
    total_dist_ga = cp.sqrt(hor_dist_ga**2 + h_a[a_indices]**2)
    
    # 计算角度（度）
    theta = cp.degrees(cp.arctan(h_a[a_indices] / (hor_dist_ga + 1e-9)))
    
    # 计算视距概率
    los_prob = 1 / (1 + cnf.LOS_COEF1 * cp.exp(-cnf.LOS_COEF2 * (theta - cnf.LOS_COEF1)))
    
    # 计算 beta 值
    G_beta[:, n_u:] = los_prob * total_dist_ga**cnf.LOS + (1-los_prob) * total_dist_ga**cnf.NLOS
    
    # 3. 计算空中 AP 到用户的 beta
    a_indices, u_indices = cp.meshgrid(cp.arange(n_a), cp.arange(n_u), indexing='ij')
    
    # 计算水平距离和总距离
    hor_dist_au = cp.sqrt((x_a[a_indices] - x_u[u_indices])**2 + 
                         (y_a[a_indices] - y_u[u_indices])**2)
    total_dist_au = cp.sqrt(hor_dist_au**2 + h_a[a_indices]**2)
    
    # 计算角度（度）
    theta = cp.degrees(cp.arctan(h_a[a_indices] / (hor_dist_au + 1e-9)))
    
    # 计算视距概率
    los_prob = 1 / (1 + cnf.LOS_COEF1 * cp.exp(-cnf.LOS_COEF2 * (theta - cnf.LOS_COEF1)))
    
    # 计算 beta 值
    A_beta = los_prob * total_dist_au**cnf.LOS + (1-los_prob) * total_dist_au**cnf.NLOS
    
    # 拆分 power_alloc 数组为地面和空中 AP 部分
    # 地面 AP 功率分配占据前 n_g*(n_u+n_a) 个元素
    idx_split = n_g * (n_u + n_a)
    G_eta = power_alloc[:idx_split].reshape(n_g, -1)
    A_eta = power_alloc[idx_split:].reshape(n_a, -1)
    
    # 归一化每一行，使总和不超过 1
    # 使用向量化操作替代循环
    G_row_sums = cp.sum(G_eta, axis=1, keepdims=True)
    G_row_sums_gt_1 = G_row_sums > 1
    G_eta = cp.where(G_row_sums_gt_1, G_eta / G_row_sums, G_eta)
    
    A_row_sums = cp.sum(A_eta, axis=1, keepdims=True)
    A_row_sums_gt_1 = A_row_sums > 1
    A_eta = cp.where(A_row_sums_gt_1, A_eta / A_row_sums, A_eta)
    
    return G_beta, G_eta, A_beta, A_eta

def calc_penalty(A_c, G_c, uav_positions=None, uav_move=None):
    """
    使用 CuPy 在 GPU 上计算基于信道容量约束和 UAV 碰撞条件的惩罚。

    在以下情况下应用惩罚:
      - 总地面到 UAV 的容量低于预定义的最小值。
      - UAV 到用户的容量超过其地面容量。
      - 用户的综合容量（来自地面和 UAV）低于预定义的最小值。
      - UAV 预测将超出边界或其实际高度（归一化前）小于 1，
        考虑当前动作的移动（如果提供）。

    Parameters:
      A_c (cp.ndarray): 从空中 AP 到用户的信道容量，形状 (n_a, n_u)。
      G_c (cp.ndarray): 从地面 AP 的信道容量，形状 (n_g, n_a+n_u)。
                      前 n_u 列对应用户，剩余 n_a 列对应 UAV。
      uav_positions (cp.ndarray, optional): UAV 当前实际位置（归一化前），形状 (3, n_a)，
                                      每一列表示 [x, y, height]。 
      uav_move (cp.ndarray, optional): 每个 UAV 的预测移动，形状 (3, n_a)，表示 [x, y, height] 的变化。

    Returns:
      float: 总惩罚。
    """
    # 确保输入在 GPU 上
    A_c = cp.asarray(A_c)
    G_c = cp.asarray(G_c)
    if uav_positions is not None:
        uav_positions = cp.asarray(uav_positions)
    if uav_move is not None:
        uav_move = cp.asarray(uav_move)
    
    # 低信道条件的惩罚系数（假定为负值）
    low_channel_penalty_coef = -cnf.LOW_CHANNEL
    sqrt5_minus_1_div_2 = cp.float32(0.618034)  # 预计算常量

    def inverse_penalty(current_capacity, required_capacity, penalty_coef):
        """
        计算基于与所需容量的偏差的缩放惩罚。
        添加一个小的 delta 以避免除以零。
        """
        delta = required_capacity * sqrt5_minus_1_div_2
        penalty = penalty_coef * (3 * required_capacity / (current_capacity + delta) -
                                 3 * required_capacity / (required_capacity + delta))
        return penalty

    total_penalty = cp.float32(0.0)

    # 提取与 UAV 相关的来自地面 AP 的信道容量。
    # 在 G_c 中，列 [n_u:] 对应于 UAV。
    uav_ground_capacity = G_c[:, n_u:]
    
    # 计算所有 UAV 的地面总容量（按 UAV 索引）
    total_ground_uav_capacities = cp.sum(uav_ground_capacity, axis=0)  # 形状 (n_a,)
    
    # 计算每个 UAV 到所有用户的总容量
    user_capacities = cp.sum(A_c, axis=1)  # 形状 (n_a,)
    
    # 条件 1：如果 UAV 的地面容量低于最小要求
    low_capacity_mask = total_ground_uav_capacities < cnf.min_capacity_uav
    if cp.any(low_capacity_mask):
        # 仅对低容量的 UAV 应用惩罚
        low_cap_uavs = cp.where(low_capacity_mask)[0]
        for i in low_cap_uavs:
            total_penalty += inverse_penalty(total_ground_uav_capacities[i],
                                           cnf.min_capacity_uav,
                                           low_channel_penalty_coef)
    
    # 条件 2：如果 UAV 的地面容量小于其到用户的容量
    insufficient_capacity_mask = total_ground_uav_capacities < user_capacities
    if cp.any(insufficient_capacity_mask):
        # 仅对容量不足的 UAV 应用惩罚
        insuf_cap_uavs = cp.where(insufficient_capacity_mask)[0]
        for i in insuf_cap_uavs:
            total_penalty += inverse_penalty(total_ground_uav_capacities[i],
                                           user_capacities[i],
                                           low_channel_penalty_coef)

    # 为每个用户计算总接收容量（来自地面和空中 AP）
    user_ground_capacities = cp.sum(G_c[:, :n_u], axis=0)  # 形状 (n_u,)
    user_aerial_capacities = cp.sum(A_c, axis=0)  # 形状 (n_u,)
    total_user_capacities = user_ground_capacities + user_aerial_capacities  # 形状 (n_u,)
    
    # 条件 3：如果用户的总容量低于最小要求
    user_low_capacity_mask = total_user_capacities < cnf.min_capacity_user
    if cp.any(user_low_capacity_mask):
        # 仅对容量不足的用户应用惩罚
        low_cap_users = cp.where(user_low_capacity_mask)[0]
        for u in low_cap_users:
            total_penalty += inverse_penalty(total_user_capacities[u],
                                           cnf.min_capacity_user,
                                           low_channel_penalty_coef)
    
    # 新碰撞惩罚：
    # 检查应用当前动作的移动后的预测 UAV 位置。
    # 如果提供了 uav_move，计算未来位置 = 当前位置 + 移动，
    # 否则，直接使用当前 uav_positions。
    if uav_positions is not None:
        if uav_move is not None:
            future_positions = uav_positions + uav_move
        else:
            future_positions = uav_positions
            
        # 检查所有 UAV 位置是否超出有效边界或高度是否小于 1
        x_out = (future_positions[0] < 0) | (future_positions[0] > cnf.MAX_X)
        y_out = (future_positions[1] < 0) | (future_positions[1] > cnf.MAX_Y)
        h_out = (future_positions[2] < 1)
        out_of_bounds = x_out | y_out | h_out
        
        # 对超出边界的 UAV 应用惩罚
        num_out_of_bounds = cp.sum(out_of_bounds)
        if num_out_of_bounds > 0:
            total_penalty -= num_out_of_bounds * cnf.CRASH

    # 将结果保留在 GPU 上
    return total_penalty

def calc_cluster(A_eta, G_eta):
    """
    简化版的集群计算函数，总是返回单一连通图。
    
    该版本假设网络中的所有节点都属于同一个集群，不再执行实际的连通分量计算。
    函数输入和输出假定都在 GPU 上。
    
    参数:
      A_eta (cp.ndarray): 空中 AP 的功率分配矩阵
      G_eta (cp.ndarray): 地面 AP 的功率分配矩阵
    
    返回:
      adj_matrix (cp.ndarray): 表示完全连通图的布尔邻接矩阵
      cluster_labels (cp.ndarray): 所有节点都归为同一集群的标签数组
    """
    import cupy as cp
    
    # 总节点数: 地面 AP + 空中 AP + 用户
    total_nodes = n_g + n_a + n_u
    
    # 创建完全连通图的邻接矩阵 (所有节点间都有连接)
    adj_matrix = cp.ones((total_nodes, total_nodes), dtype=bool)
    
    # 对角线元素设为 False (节点不与自身连接)
    for i in range(total_nodes):
        adj_matrix[i, i] = False
    
    # 所有节点都归为集群 0
    cluster_labels = cp.zeros(total_nodes, dtype=int)
    
    return adj_matrix, cluster_labels

def calc_linkcost(adj_matrix, cluster_labels):
    """
    使用 CuPy 在 GPU 上计算基于 UAV 连接和集群合作的总链路成本。

    链路成本计算方式：
      1. 对于每个 UAV，添加与连接用户数量成比例的成本。
      2. 对于每个集群，添加与集群中地面 AP 数量和用户数量成比例的合作成本。

    参数:
      adj_matrix (cp.ndarray): 表示节点间连接的布尔邻接矩阵
      cluster_labels (cp.ndarray): 包含每个节点集群标签的数组
                                 节点按以下顺序排列：
                                   - 地面 AP: 索引 0 到 n_g-1
                                   - UAV: 索引 n_g 到 n_g+n_a-1
                                   - 用户: 索引 n_g+n_a 到 n_g+n_a+n_u-1

    返回:
      float: 计算的总链路成本（留在 GPU 上）
    """
    # 确保输入在 GPU 上
    adj_matrix = cp.asarray(adj_matrix)
    cluster_labels = cp.asarray(cluster_labels)

    # 初始化总成本
    total_cost = cp.float32(0.0)

    # 1. 计算 UAV 连接成本
    # 创建一个用户索引范围的掩码
    user_indices = cp.arange(n_g + n_a, n_g + n_a + n_u, dtype=int)
    
    # 对每个 UAV（索引从 n_g 到 n_g+n_a-1）计算连接用户数量
    for a in range(n_a):
        uav_idx = n_g + a  # UAV 在节点数组中的索引
        
        # 提取当前 UAV 到所有用户的连接情况
        user_connections = adj_matrix[uav_idx, user_indices]
        
        # 计算连接的用户数量并添加相应成本
        num_connected_users = cp.sum(user_connections)
        total_cost += num_connected_users * cnf.LINK_COST

    # 2. 计算集群合作成本
    # 获取唯一的集群 ID（在 GPU 上执行）
    unique_clusters = cp.unique(cluster_labels)
    
    # 创建地面 AP 和用户的索引掩码
    ground_ap_indices = cp.arange(n_g)
    user_indices = cp.arange(n_g + n_a, n_g + n_a + n_u)
    
    # 对每个集群计算成本
    for cluster_id in unique_clusters:
        # 创建当前集群中节点的布尔掩码
        cluster_mask = (cluster_labels == cluster_id)
        
        # 计算集群中的地面 AP 数量
        ground_ap_mask = cluster_mask[ground_ap_indices]
        num_ground_aps = cp.sum(ground_ap_mask)
        
        # 计算集群中的用户数量
        user_mask = cluster_mask[user_indices]
        num_users = cp.sum(user_mask)
        
        # 计算集群的合作成本
        cluster_cost = num_ground_aps * num_users * cnf.LINK_COST
        total_cost += cluster_cost

    # 返回总成本（保留在 GPU 上）
    return total_cost  

def calc_util(state, aution):
    """
    在 GPU 上计算给定状态和动作的效用（奖励）。

    过程:
      1. 使用 arr_prep 预处理输入状态和动作，获取位置、移动（归一化和非归一化）和功率分配动作。
      2. 使用 arr2mat 将功率分配动作和位置转换为结构化矩阵（G_beta, G_eta, A_beta, A_eta）用于信道评估。
      3. 使用 Shannon 容量公式计算地面 AP 和空中 AP 的信道容量。
      4. 根据信道容量约束计算惩罚。
      5. 确定网络集群并计算链路成本。
      6. 组合信道容量并计算最终奖励。

    返回:
      reward: 计算的奖励值。
      tot_capacity: 总网络信道容量。
      punishment: 基于容量约束的计算惩罚。
      link_cost: 基于集群合作和 UAV 连接的计算链路成本。
      expert_action: 占位专家动作（当前为 0）。
      subopt_expert_action: 占位次优专家动作（当前为 0）。
      real_action: 去归一化移动和功率分配动作的拼接。
    """
    # 1. 预处理状态和动作以提取位置、移动和功率分配动作
    #    state 包含归一化位置 [0,1] 和功率分配状态。
    #    aution 包含归一化移动和功率分配动作。
    actual_position, _, actual_move, power_alloc_action, norm_position, norm_move = arr_prep(state, aution)

    # 2. 将功率分配动作和位置转换为结构化矩阵用于信道评估。
    G_beta, G_eta, A_beta, A_eta = arr2mat(power_alloc_action, actual_position)

    # 3. 计算地面 AP 的信道容量。
    # 初始化地面信道容量矩阵，形状: (n_g, n_a + n_u)
    G_c = cp.zeros((n_g, n_a + n_u), dtype=cp.float32)
    
    # 为所有信道并行计算噪声系数
    # 预计算所有地面 AP 的功率分配和
    G_eta_sums = cp.sum(G_eta, axis=1, keepdims=True)
    
    # 对每个接收方（用户和 UAV）
    for i in range(n_u + n_a):
        # 对索引 i 创建掩码，用于从总和中减去当前信道的贡献
        mask = cp.ones(G_eta.shape[1], dtype=bool)
        mask[i] = False
        
        # 计算除第 i 个信道外的所有其他信道的干扰(噪声)总和
        noise_coef1 = cp.sum(G_eta[:, mask], axis=1)  # shape: (n_g,)
        
        # 计算其他信号贡献的总噪声
        if i < n_u:
            mask_a = cp.ones(A_eta.shape[1], dtype=bool)
            mask_a[i] = False
            noise_coef2 = cp.sum(A_eta[:, mask_a], axis=1)
            tot_noise = cp.sum(noise_coef1 * cnf.P_G * G_beta[:, i]) + cp.sum(noise_coef2 * cnf.P_A * A_beta[:, i])
        else:
            tot_noise = cp.sum(noise_coef1 * cnf.P_G * G_beta[:, i])
        
        # 计算第 i 个信道的接收信号
        signal = cnf.P_G * G_beta[:, i] * G_eta[:, i]
        
        # Shannon 容量公式 (log2(1 + SINR))
        # 添加稳定性因子避免除零
        G_c[:, i] = cp.log2(1 + signal / (tot_noise + cnf.white_noise))

    # 4. 计算空中 (UAV) AP 的信道容量。
    # 初始化空中信道容量矩阵，形状: (n_a, n_u)
    A_c = cp.zeros((n_a, n_u), dtype=cp.float32)
    
    # 预计算所有空中 AP 的功率分配和
    A_eta_sums = cp.sum(A_eta, axis=1, keepdims=True)
    
    # 对每个用户
    for i in range(n_u):
        # 创建用于掩码操作的数组
        mask_g = cp.ones(G_eta.shape[1], dtype=bool)
        mask_g[i] = False
        
        mask_a = cp.ones(A_eta.shape[1], dtype=bool)
        mask_a[i] = False
        
        # 高效计算噪声系数
        noise_coef1 = cp.sum(G_eta[:, mask_g], axis=1)
        noise_coef2 = cp.sum(A_eta[:, mask_a], axis=1)
        
        # 计算总噪声
        tot_noise = cp.sum(noise_coef1 * cnf.P_G * G_beta[:, i]) + cp.sum(noise_coef2 * cnf.P_A * A_beta[:, i])
        
        # 计算信号
        signal = cnf.P_A * A_beta[:, i] * A_eta[:, i]
        
        # Shannon 容量公式
        A_c[:, i] = cp.log2(1 + signal / (tot_noise + cnf.white_noise))

    # 5. 根据信道容量约束计算惩罚。
    uav_positions = actual_position[:3 * cnf.NUM_A_AP].reshape(3, cnf.NUM_A_AP)
    uav_move = actual_move.reshape(3, cnf.NUM_A_AP)
    # 调用 calc_penalty 并带上可选的 UAV 位置
    punishment = calc_penalty(A_c, G_c, uav_positions, uav_move)

    # 6. 确定网络集群并计算链路成本。
    adj_matrix, cluster_labels = calc_cluster(A_eta, G_eta)
    link_cost = calc_linkcost(adj_matrix, cluster_labels)

    # 7. 组合地面和空中 AP 信道容量。
    # 假设 cnf.NUM_AP 表示 AP 的总数 (地面 + 空中)
    capacity = cp.zeros((cnf.NUM_AP, n_u), dtype=cp.float32)
    # 地面 AP 容量: 前 n_g 行，仅用于用户信道
    capacity[:n_g, :] = G_c[:, :n_u]
    # 空中 AP 容量: 剩余行
    capacity[n_g:, :] = A_c[:, :n_u]
    tot_capacity = cp.sum(capacity)

    # 8. 组合容量、惩罚和链路成本形成最终奖励。
    p_c = cnf.punishment_coef
    c_c = cnf.cost_coef
    reward = tot_capacity * (1 - p_c - c_c) + punishment * p_c - link_cost * c_c

    # 占位专家动作 (对于模仿模式，如果需要)
    expert_action = cp.float32(0)
    subopt_expert_action = cp.float32(0)

    # 实际动作是归一化移动和功率分配动作的拼接
    real_action = cp.concatenate((norm_move, power_alloc_action))
    
    # 将结果转换为 NumPy 数组以与环境兼容
    # 注意: env.py 期望的是 NumPy 数组而非 CuPy 数组
    return float(reward.get()), float(tot_capacity.get()), float(punishment.get()), float(link_cost.get()), \
          float(expert_action), float(subopt_expert_action), real_action.get()