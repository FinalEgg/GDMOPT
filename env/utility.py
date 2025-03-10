import numpy as np
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

def map(state, aution):
    """
    预处理State和Aution
    - state前面 [3*cnf.NUM_A_AP + 2*cnf.NUM_G_AP + 2*cnf.NUM_USERS] 位为位置信息，后面为 power_alloc_state
    - aution前面 NUM_A_AP*3 位为速度信息，需要映射到实际速度；后面为 power_alloc_action，需要映射到 [0,1]
    """
    import numpy as np
    # 确保输入为 numpy 数组
    state = np.array(state)
    aution = np.array(aution)
    
    # 拆分 state
    pos_len = 3 * cnf.NUM_A_AP + 2 * cnf.NUM_G_AP + 2 * cnf.NUM_USERS
    position = state[:pos_len]
    power_alloc_state = state[pos_len:]
    
    # 拆分 aution
    move_num = cnf.NUM_A_AP * 3
    move = aution[:move_num]
    power_alloc_action = aution[move_num:]
    
    # 映射 move 到实际速度，参考 CompUtility 开头部分
    scales = np.array([cnf.MAX_V_X] * cnf.NUM_A_AP +
                      [cnf.MAX_V_Y] * cnf.NUM_A_AP +
                      [cnf.MAX_V_H] * cnf.NUM_A_AP)
    move_mapped = move * scales
    
    # 将 power_alloc_action 映射到 [0,1]，参考 split 函数开头处理方法
    power_alloc_action_mapped = np.clip(power_alloc_action, 0, 1)
    
    return position, power_alloc_state, move_mapped, power_alloc_action_mapped

def split(power_alloc,position):
    # 函数传入的power_alloc是功率分配，[-1,1]
    # 函数传入的position只有位置信息，已经经过放大映射
    power_alloc = power_alloc.detach().numpy() if torch.is_tensor(power_alloc) else np.array(power_alloc)
    position = position.detach().numpy() if torch.is_tensor(position) else np.array(position)

    # reshape position into structured array
    aerial_position = position[:3*n_a].reshape(3, n_a)  # [x_a, y_a, h_a]
    x_a, y_a, h_a = aerial_position
    
    ground_position = position[3*n_a:3*n_a + 2*n_g].reshape(2, n_g)  # [x_g, y_g]
    x_g, y_g = ground_position
    
    user_position = position[3*n_a + 2*n_g: 3*n_a + 2*n_g + 2* n_u].reshape(2, n_u)  # [x_u, y_u]
    x_u, y_u = user_position

    def calc_beta(x1, y1, x2, y2, h2):
        hor_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        dist =  np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + h2**2)
        theta = np.arctan(h2/hor_dist)
        prob = 1/(1+cnf.LOS_COEF1*np.exp(-cnf.LOS_COEF2*(theta-cnf.LOS_COEF1)))
        return prob*dist**cnf.LOS+(1-prob)*dist**cnf.NLOS


    # power_alloc: 
    # n_g rows: n_u digits for users, n_a digits for aerial APs
    # n_a rows: n_u digits for users
    
    # Ground AP 
    G_beta = np.zeros((n_g, n_a + n_u))
    G_eta = np.zeros((n_g, n_a + n_u))
    
    for i in range(n_g):
        for j in range(n_u):
            G_beta[i, j] = calc_beta(x_g[i], y_g[i], x_u[j], y_u[j], 0)
    for i in range(n_g):
        for j in range(n_a):
            G_beta[i, n_u + j] = calc_beta(x_g[i], y_g[i], x_a[j], y_a[j], h_a[j])

    # Aerial AP
    A_beta = np.zeros((n_a, n_u))
    A_eta = np.zeros((n_a, n_u))
    
    for i in range(n_a):
        for j in range(n_u):
            A_beta[i, j] = calc_beta(x_a[i], y_a[i], x_u[j], y_u[j], h_a[i])
    
    start = n_g*(n_u+n_a)

    # G_eta
    G_eta = power_alloc[:start].reshape(n_g, -1)  # 重塑为(n_g, n_u+n_a) 
    # A_eta
    A_eta = power_alloc[start:].reshape(n_a, -1)  # 重塑为(n_a, n_u)  

    # 对 G_eta 每一行进行归一化(总和大于1时归一化，总和小于1时保持不变，0始终是0)
    for i in range(G_eta.shape[0]):
        row_sum = np.sum(G_eta[i])
        if row_sum > 1:
            G_eta[i] = G_eta[i] / row_sum

    # 对 A_eta 每一行进行归一化(总和大于1时归一化，总和小于1时保持不变，0始终是0)
    for i in range(A_eta.shape[0]):
        row_sum = np.sum(A_eta[i])
        if row_sum > 1:
            A_eta[i] = A_eta[i] / row_sum

    return G_beta, G_eta, A_beta, A_eta

def CompPunishment(A_c,G_c,A_eta,G_eta):
    # punished when uav has too little chanel capacity
    # punished when the uav's capacity of user is lesser than the capacity of ground ap
    punishment = 0
    overload_punish_coef = -cnf.OVER_LOAD_PUNISHMENT
    low_channel_punishment = -cnf.LOW_CHANNEL

    def inverse_punishment(x, min_capacity, punishment_coef):
        delta = min_capacity * (math.sqrt(5) - 1) / 2
        punishment = punishment_coef * (3* min_capacity / (x + delta) - 3* min_capacity / (min_capacity + delta))
        return punishment
    
    # 获取UAV部分的信道容量 (G_c[:, n_u:])
    uav_ground_capacity = G_c[:, n_u:]  # 地面AP到UAV的信道容量
    
    # 检查每个UAV
    for i in range(n_a):
        # 计算当前UAV与所有地面AP的信道容量总和
        total_ground_uav_capacity = np.sum(uav_ground_capacity[:, i])
        user_capacity = np.sum(A_c[i, :])
        
        # 条件2: 检查UAV的总信道容量是否低于最小要求
        if total_ground_uav_capacity < cnf.min_capacity_uav:
            punishment += inverse_punishment(total_ground_uav_capacity, cnf.min_capacity_uav, low_channel_punishment)
            
        # 条件3: 检查UAV与用户的信道容量是否大于其与地面AP的信道容量
        if total_ground_uav_capacity < user_capacity:
            punishment += inverse_punishment(total_ground_uav_capacity, user_capacity, low_channel_punishment)

    # 条件4：检查每个用户的总信道容量
    for u in range(n_u):
        # 计算用户从地面AP获得的容量
        ground_capacity = np.sum(G_c[:, u])
        # 计算用户从UAV获得的容量
        aerial_capacity = np.sum(A_c[:, u])
        # 计算总容量
        total_user_capacity = ground_capacity + aerial_capacity
        
        # 如果用户的总信道容量为0，添加一个很大的惩罚
        if total_user_capacity < cnf.min_capacity_user:
            punishment += inverse_punishment(total_user_capacity, cnf.min_capacity_user, low_channel_punishment) 

    return punishment

def CompCluster(A_eta,G_eta):

    # 创建无向图
    G = nx.Graph()
    
    # 添加所有节点
    total_nodes = n_g + n_a + n_u
    G.add_nodes_from(range(total_nodes))
    
    # 添加地面基站与用户之间的边
    for g in range(n_g):
        for u in range(n_u):
            if G_eta[g, u] > 0:
                G.add_edge(g, n_g + n_a + u)
                
    # 添加地面基站与UAV之间的边
    for g in range(n_g):
        for a in range(n_a):
            if G_eta[g, n_u + a] > 0:
                G.add_edge(g, n_g + a)
                
    # 添加UAV与用户之间的边
    for a in range(n_a):
        for u in range(n_u):
            if A_eta[a, u] > 0:
                G.add_edge(n_g + a, n_g + n_a + u)
    
    # 获取连通分量
    clusters = list(nx.connected_components(G))
    
    # 创建标签数组
    cluster_labels = np.zeros(total_nodes, dtype=int)
    for i, cluster in enumerate(clusters):
        for node in cluster:
            cluster_labels[node] = i
    
    # 创建邻接矩阵
    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=bool)
    for edge in G.edges():
        adj_matrix[edge[0], edge[1]] = True
        adj_matrix[edge[1], edge[0]] = True  # 因为是无向图，所以要对称
            
    return adj_matrix, cluster_labels

def CompLinkCost(adj_matrix,cluster_labels):

    total_cost=0
    # 1. 计算UAV的连接成本
    for a in range(n_a):
        uav_idx = n_g + a
        # 计算该UAV连接的用户数量
        user_connections = np.sum(adj_matrix[uav_idx, n_g+n_a:])
        total_cost += user_connections * cnf.LINK_COST
    
    # 2. 计算每个簇的协作成本
    n_clusters = len(np.unique(cluster_labels))
    for cluster_id in range(n_clusters):
        # 获取簇内的节点
        cluster_mask = (cluster_labels == cluster_id)
        
        # 计算簇内的地面基站数量
        n_ground_aps = np.sum(cluster_mask[:n_g])
        
        # 计算簇内的用户数量
        n_users = np.sum(cluster_mask[n_g+n_a:])
        
        # 计算该簇的协作成本
        cluster_cost = n_ground_aps * n_users * cnf.LINK_COST
        total_cost += cluster_cost
    
    return total_cost    

# Function to compute utility (reward) for the given state and action
def CompUtility(State, Aution):

    position, _, move, power_alloc_action = map(State, Aution)

    G_beta, G_eta, A_beta, A_eta = split(power_alloc_action, position)
     # 计算地面AP的信道容量
    G_c = np.zeros((n_g, n_a + n_u))
    for i in range(n_u + n_a):  
        noise_coef = np.sum(G_eta[:, [j for j in range(n_u+n_a) if j != i]], axis=1)  # shape: (n_g,)
        tot_noise = np.sum(noise_coef * cnf.P_G * G_beta[:, i])
        signal = cnf.P_G * G_beta[:, i] * G_eta[:, i]
        G_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))

    # 计算空中AP的信道容量
    A_c = np.zeros((n_a, n_u))
    for i in range(n_u):  
        noise_coef = np.sum(A_eta[:, [j for j in range(n_u) if j != i]], axis=1)  # shape: (n_a,)
        tot_noise = np.sum(noise_coef * cnf.P_A * A_beta[:, i])
        signal = cnf.P_A * A_beta[:, i] * A_eta[:, i]
        A_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))

    # 计算惩罚
    punishment = CompPunishment(A_c, G_c, A_eta, G_eta)
    # 计算簇
    adj_matrix, cluster_labels = CompCluster(A_eta, G_eta)
    # 计算连接成本
    link_cost = CompLinkCost(adj_matrix, cluster_labels)

    # 合并地面AP和空中AP的信道容量
    capacity = np.zeros((cnf.NUM_AP, n_u))
    capacity[:n_g, :] = G_c[:, :n_u]  # 地面AP的容量
    capacity[n_g:, :] = A_c[:, :n_u]  # 空中AP的容量
    tot_capacity = np.sum(capacity)

    p_c = cnf.punishment_coef
    c_c = cnf.cost_coef
    reward = tot_capacity*(1-p_c-c_c) + punishment*p_c - link_cost*c_c

    #当启用行为模仿模式时，需要提出算法，在返回值中传递专家动作
    expert_action = 0
    subopt_expert_action = 0

    return reward,tot_capacity, punishment,link_cost, expert_action, subopt_expert_action, np.concatenate((move, power_alloc_action))
