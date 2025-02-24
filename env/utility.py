import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
from env import config as cnf
import networkx as nx
os.environ['KMP_DUPLICATE_LIB_OK']='True'

n_a = cnf.NUM_A_AP  # number of aerial APs
n_g = cnf.NUM_G_AP  # number of ground APs
n_u = cnf.NUM_USERS # number of users

def split(action,state):
    action = action.detach().numpy() if torch.is_tensor(action) else np.array(action)
    state = state.detach().numpy() if torch.is_tensor(state) else np.array(state)

    action = np.where(action < 0.5, 0, 2*(action - 0.5))

    # reshape state into structured array
    aerial_state = state[:3*n_a].reshape(3, n_a)  # [x_a, y_a, h_a]
    x_a, y_a, h_a = aerial_state
    
    ground_state = state[3*n_a:3*n_a + 2*n_g].reshape(2, n_g)  # [x_g, y_g]
    x_g, y_g = ground_state
    
    user_state = state[3*n_a + 2*n_g:].reshape(2, n_u)  # [x_u, y_u]
    x_u, y_u = user_state

    def calc_beta(x1, y1, x2, y2, h2):
        hor_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        dist =  np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + h2**2)
        theta = np.arctan(h2/hor_dist)
        prob = 1/(1+cnf.LOS_COEF1*np.exp(-cnf.LOS_COEF2*(theta-cnf.LOS_COEF1)))
        return prob*dist**cnf.LOS+(1-prob)*dist**cnf.NLOS


    # action: 
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
    action_g = action[:start].reshape(n_g, -1)  # 重塑为(n_g, n_u+n_a)
    G_eta = action_g  
    
    # A_eta
    action_a = action[start:].reshape(n_a, -1)  # 重塑为(n_a, n_u)
    A_eta = action_a  

    # normalization using numpy operations
    # G_eta
    row_sums_g = np.sum(G_eta, axis=1, keepdims=True)  
    mask_g = row_sums_g != 0  
    G_eta = np.where(mask_g, G_eta / row_sums_g, G_eta) 

    # A_eta
    row_sums_a = np.sum(A_eta, axis=1, keepdims=True) 
    mask_a = row_sums_a != 0  
    A_eta = np.where(mask_a, A_eta / row_sums_a, A_eta)  

    return G_beta, G_eta, A_beta, A_eta

def CompPunishment(A_c,G_c):
    # punished when uav has too little chanel capacity
    # punished when the uav's capacity of user is lesser than the capacity of ground ap
    punishment = 0
    major_punish_coef = -cnf.MAJOR_PUNISHMENT
    minor_punish_coef = -cnf.MINOR_PUNISHMENT

    
    # 获取UAV部分的信道容量 (G_c[:, n_u:])
    uav_ground_capacity = G_c[:, n_u:]  # 地面AP到UAV的信道容量
    
    # 检查每个UAV
    for i in range(n_a):
        # 计算当前UAV与所有地面AP的信道容量总和
        total_ground_uav_capacity = np.sum(uav_ground_capacity[:, i])
        user_capacity = np.sum(A_c[i, :])
        
        # 条件1: 检查UAV的总信道容量是否低于最小要求
        if total_ground_uav_capacity < cnf.min_capacity:
            punishment += major_punish_coef*(cnf.min_capacity - total_ground_uav_capacity)
            
        # 条件2: 检查UAV与用户的信道容量是否大于其与地面AP的信道容量
        if total_ground_uav_capacity < user_capacity:
            punishment += minor_punish_coef*(user_capacity - total_ground_uav_capacity)
    
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

    move_num = cnf.NUM_A_AP*3
    move = Aution[:move_num]
    scales = np.array([cnf.MAX_V_X] * cnf.NUM_A_AP +
                    [cnf.MAX_V_Y] * cnf.NUM_A_AP +
                    [cnf.MAX_V_H] * cnf.NUM_A_AP)
    move = move * scales

    raw_power_alloc = Aution[move_num:]

    position = State[:3*cnf.NUM_A_AP+2*cnf.NUM_G_AP+2*cnf.NUM_USERS]

    G_beta, G_eta, A_beta, A_eta = split(raw_power_alloc, position)
     # 计算地面AP的信道容量
    G_c = np.zeros((n_g, n_a + n_u))
    for i in range(n_u + n_a):  
        noise_coef = np.sum(G_eta[:, [j for j in range(n_u+n_a) if j != i]], axis=1)  # shape: (n_g,)
        tot_noise = np.sum(noise_coef * cnf.P_G * G_beta[:, i])
        signal = np.sum(cnf.P_G * G_beta[:, i] * G_eta[:, i])
        G_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))

    # 计算空中AP的信道容量
    A_c = np.zeros((n_a, n_u))
    for i in range(n_u):  
        noise_coef = np.sum(A_eta[:, [j for j in range(n_u) if j != i]], axis=1)  # shape: (n_a,)
        tot_noise = np.sum(noise_coef * cnf.P_A * A_beta[:, i])
        signal = np.sum(cnf.P_A * A_beta[:, i] * A_eta[:, i])
        A_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))

    # 计算惩罚
    punishment = CompPunishment(A_c, G_c)
    # 计算簇
    adj_matrix, cluster_labels = CompCluster(A_eta, G_eta)
    # 计算连接成本
    link_cost = CompLinkCost(adj_matrix, cluster_labels)

    # 合并地面AP和空中AP的信道容量
    capacity = np.zeros((cnf.NUM_AP, n_u))
    capacity[:n_g, :] = G_c[:, :n_u]  # 地面AP的容量
    capacity[n_g:, :] = A_c[:, :n_u]  # 空中AP的容量

    reward = np.sum(capacity) + punishment - link_cost

    expert_action = 0
    subopt_expert_action = 0

    return reward, expert_action, subopt_expert_action, Aution

def load_test_data(filename='test_data.txt'):
    test_data = {}
    try:
        test_data_path = os.path.join(os.path.dirname(__file__), filename)
        test_data['state'] = np.loadtxt(test_data_path, delimiter=',', max_rows=1)
        test_data['action'] = np.loadtxt(test_data_path, delimiter=',', skiprows=1, max_rows=1)
        test_data['A_c'] = np.loadtxt(test_data_path, delimiter=',', skiprows=2, max_rows=n_a)
        test_data['G_c'] = np.loadtxt(test_data_path, delimiter=',', skiprows=2+n_a, max_rows=n_g)
        return test_data
    except Exception as e:
        print(f"读取测试数据失败: {str(e)}")
        return None

def test_utility_functions():
    print("===开始测试功能===")
    
    # 1. 创建测试数据
    print("\n1. 加载测试数据...")
    test_data = load_test_data()
    if test_data is None:
        print("无法加载测试数据，测试终止")
        return
        
    test_state = test_data['state']
    test_action = test_data['action']
    print(f"测试状态数据:\n{test_state}")
    print(f"测试动作数据:\n{test_action}")
    
    move_num = n_a * 3
    power_alloc_size = n_g*(n_u+n_a) + n_a*n_u
    test_action = np.random.uniform(0, 1, move_num + power_alloc_size)
    print(f"测试动作数据:\n{test_action}")
    
    # 2. 测试split函数
    print("\n2. 测试split函数...")
    try:
        G_beta, G_eta, A_beta, A_eta = split(test_action[move_num:], test_state)
        print("✓ split函数测试通过")
        print(f"- G_beta shape: {G_beta.shape}")
        print(f"- G_beta 值:\n{G_beta}")
        print(f"- G_eta shape: {G_eta.shape}")
        print(f"- G_eta 值:\n{G_eta}")
        print(f"- A_beta shape: {A_beta.shape}")
        print(f"- A_beta 值:\n{A_beta}")
        print(f"- A_eta shape: {A_eta.shape}")
        print(f"- A_eta 值:\n{A_eta}")
    except Exception as e:
        print(f"× split函数测试失败: {str(e)}")
    
    # 3. 测试CompPunishment函数
    print("\n3. 测试CompPunishment函数...")
    try:
        A_c = np.random.uniform(0, 10, (n_a, n_u))
        G_c = np.random.uniform(0, 10, (n_g, n_a + n_u))
        print(f"测试用A_c矩阵:\n{A_c}")
        print(f"测试用G_c矩阵:\n{G_c}")
        punishment = CompPunishment(A_c, G_c)
        print(f"✓ CompPunishment函数测试通过")
        print(f"- 惩罚值: {punishment}")
    except Exception as e:
        print(f"× CompPunishment函数测试失败: {str(e)}")
    
    # 4. 测试CompCluster函数
    print("\n4. 测试CompCluster函数...")
    try:
        adj_matrix, cluster_labels = CompCluster(A_eta, G_eta)
        print("✓ CompCluster函数测试通过")
        print(f"- 邻接矩阵:\n{adj_matrix}")
        print(f"- 簇标签: {cluster_labels}")
        print(f"- 簇的数量: {len(np.unique(cluster_labels))}")
        # 输出每个簇的节点
        for i in range(len(np.unique(cluster_labels))):
            nodes = np.where(cluster_labels == i)[0]
            print(f"  簇 {i} 包含的节点: {nodes}")
    except Exception as e:
        print(f"× CompCluster函数测试失败: {str(e)}")
    
    # 5. 测试CompLinkCost函数
    print("\n5. 测试CompLinkCost函数...")
    try:
        link_cost = CompLinkCost(adj_matrix, cluster_labels)
        print("✓ CompLinkCost函数测试通过")
        print(f"- 连接成本: {link_cost}")
        # 分解成本计算
        uav_cost = sum([np.sum(adj_matrix[n_g + a, n_g+n_a:]) * cnf.LINK_COST for a in range(n_a)])
        print(f"- UAV连接成本: {uav_cost}")
        cluster_cost = link_cost - uav_cost
        print(f"- 簇协作成本: {cluster_cost}")
    except Exception as e:
        print(f"× CompLinkCost函数测试失败: {str(e)}")
    
    # 6. 测试完整的CompUtility函数
    print("\n6. 测试CompUtility函数...")
    try:
        reward, expert_action, subopt_expert_action, action = CompUtility(test_state, test_action)
        print("✓ CompUtility函数测试通过")
        print(f"- 总奖励值: {reward}")
        print(f"- 专家动作: {expert_action}")
        print(f"- 次优专家动作: {subopt_expert_action}")
        print(f"- 实际动作: {action}")
    except Exception as e:
        print(f"× CompUtility函数测试失败: {str(e)}")


if __name__ == "__main__":
    test_utility_functions()