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

def arr_prep(state, aution):
    """
    Preprocess state and action.
    
    Parameters:
      state: A 1D array where the first (3*NUM_A_AP + 2*NUM_G_AP + 2*NUM_USERS) elements 
             represent normalized positions (in the range [0,1]) of aerial APs, ground APs, and users.
             The remaining elements correspond to power allocation state (in the range [0,1]).
      aution: A 1D array where the first (NUM_A_AP*3) elements are normalized move values 
              (in [0,1]) to be mapped to actual velocities. The remaining elements represent 
              power allocation actions that will be mapped to [0,1].
    
    Returns:
      actual_position: The de-normalized positions obtained by scaling normalized positions 
                       with their corresponding maximum values.
      power_alloc_state: The power allocation part from state.
      move_mapped: The actual move values obtained by scaling normalized moves with maximum velocities.
      power_alloc_action: The mapped power allocation actions in the range [0,1].
      norm_position: The original normalized positions (range [0,1]).
      norm_move: The original normalized move values (range [0,1]).
    """
    state = np.array(state)
    aution = np.array(aution)
    
    # Determine the length of the normalized position part
    pos_len = 3 * cnf.NUM_A_AP + 2 * cnf.NUM_G_AP + 2 * cnf.NUM_USERS
    
    # Extract normalized positions and power allocation state from state
    norm_position = state[:pos_len]
    power_alloc_state = state[pos_len:]
    
    # De-normalize positions to obtain actual positions
    actual_position = np.empty_like(norm_position, dtype=float)
    
    # For aerial APs: first 3*NUM_A_AP elements correspond to x, y, h.
    aerial_scale = np.array([cnf.MAX_X] * cnf.NUM_A_AP +
                            [cnf.MAX_Y] * cnf.NUM_A_AP +
                            [cnf.MAX_H] * cnf.NUM_A_AP)
    actual_position[:3*cnf.NUM_A_AP] = norm_position[:3*cnf.NUM_A_AP] * aerial_scale
    
    # For ground APs: next 2*NUM_G_AP elements correspond to x, y.
    ground_scale = np.array([cnf.MAX_X] * cnf.NUM_G_AP +
                            [cnf.MAX_Y] * cnf.NUM_G_AP)
    start_ground = 3*cnf.NUM_A_AP
    end_ground = start_ground + 2*cnf.NUM_G_AP
    actual_position[start_ground:end_ground] = norm_position[start_ground:end_ground] * ground_scale
    
    # For users: last 2*NUM_USERS elements correspond to x, y.
    user_scale = np.array([cnf.MAX_X] * cnf.NUM_USERS +
                          [cnf.MAX_Y] * cnf.NUM_USERS)
    start_user = end_ground
    end_user = start_user + 2*cnf.NUM_USERS
    actual_position[start_user:end_user] = norm_position[start_user:end_user] * user_scale
    
    # Process aution (action) part
    move_length = cnf.NUM_A_AP * 3
    move = aution[:move_length]
    power_alloc_action = aution[move_length:]
    
    # Map moves to actual velocity values and normalized values
    velocity_scales = np.array([cnf.MAX_V_X] * cnf.NUM_A_AP +
                               [cnf.MAX_V_Y] * cnf.NUM_A_AP +
                               [cnf.MAX_V_H] * cnf.NUM_A_AP)
    actual_move = move * velocity_scales
    norm_move = actual_move/aerial_scale

    # Map power allocation actions into [0,1] using tanh and clipping
    power_alloc_action = np.clip(np.tanh(power_alloc_action), 0, 1)
    
    return actual_position, power_alloc_state, actual_move, power_alloc_action, norm_position, norm_move

def arr2mat(power_alloc, position):
    """
    Process the power allocation and position data into structured matrices for channel evaluation.
    
    Parameters:
      power_alloc: 1D array representing power allocation coefficients (range [0,1]).
      position: 1D array containing de-normalized positions for APs and users.
                Expected order:
                  - First 3*n_a elements: aerial APs positions (x, y, h)
                  - Next 2*n_g elements: ground APs positions (x, y)
                  - Last 2*n_u elements: user positions (x, y)
                  
    Returns:
      G_beta: Matrix of path loss factors from ground APs to users and aerial APs.
      G_eta: Normalized power allocation matrix for ground APs (each row sum capped at 1).
      A_beta: Matrix of path loss factors from aerial APs to users.
      A_eta: Normalized power allocation matrix for aerial APs (each row sum capped at 1).
    """
    import numpy as np
    import torch

    # Convert tensors to numpy arrays if required
    if torch.is_tensor(power_alloc):
        power_alloc = power_alloc.detach().numpy()
    else:
        power_alloc = np.array(power_alloc)
        
    if torch.is_tensor(position):
        position = position.detach().numpy()
    else:
        position = np.array(position)
    
    # Reshape position data into structured arrays
    # Aerial APs: first 3*n_a elements (x, y, h)
    aerial_positions = position[:3 * n_a].reshape(3, n_a)
    x_a, y_a, h_a = aerial_positions

    # Ground APs: next 2*n_g elements (x, y)
    ground_positions = position[3 * n_a:3 * n_a + 2 * n_g].reshape(2, n_g)
    x_g, y_g = ground_positions

    # Users: last 2*n_u elements (x, y)
    user_positions = position[3 * n_a + 2 * n_g:3 * n_a + 2 * n_g + 2 * n_u].reshape(2, n_u)
    x_u, y_u = user_positions

    def calc_beta(x1, y1, x2, y2, h2):
        """
        Calculate the effective path loss factor between two nodes.
        
        Parameters:
          x1, y1: Coordinates of the transmitter.
          x2, y2: Coordinates of the receiver.
          h2:     Height (used for aerial APs).
          
        Returns:
          Effective path loss factor combining LOS and NLOS components.
        """
        # Compute horizontal distance and overall distance
        hor_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        total_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + h2**2)
        # Prevent division by zero
        theta = np.degrees(np.arctan(h2 / (hor_dist + 1e-9)))
        los_prob = 1 / (1 + cnf.LOS_COEF1 * np.exp(-cnf.LOS_COEF2 * (theta - cnf.LOS_COEF1)))
        return los_prob * total_dist ** cnf.LOS + (1 - los_prob) * total_dist ** cnf.NLOS

    # Initialize matrices for ground APs:
    # - G_beta: path loss factors
    # - G_eta: corresponding power allocation coefficients 
    G_beta = np.zeros((n_g, n_a + n_u))
    G_eta = np.zeros((n_g, n_a + n_u))
    
    # Calculate G_beta for ground APs
    for i in range(n_g):
        # For users (columns 0 ~ n_u-1)
        for j in range(n_u):
            G_beta[i, j] = calc_beta(x_g[i], y_g[i], x_u[j], y_u[j], 0)
        # For aerial APs (columns n_u ~ n_u+n_a-1)
        for j in range(n_a):
            G_beta[i, n_u + j] = calc_beta(x_g[i], y_g[i], x_a[j], y_a[j], h_a[j])

    # Initialize matrices for aerial APs:
    # - A_beta: path loss factors from aerial APs to users
    # - A_eta: corresponding power allocation coefficients
    A_beta = np.zeros((n_a, n_u))
    A_eta = np.zeros((n_a, n_u))
    
    for i in range(n_a):
        for j in range(n_u):
            A_beta[i, j] = calc_beta(x_a[i], y_a[i], x_u[j], y_u[j], h_a[i])
            
    # Split power_alloc array into ground and aerial AP parts.
    # Ground APs power allocation occupies the first: n_g*(n_u+n_a) elements.
    idx_split = n_g * (n_u + n_a)
    G_eta = power_alloc[:idx_split].reshape(n_g, -1)
    A_eta = power_alloc[idx_split:].reshape(n_a, -1)
    
    # Normalize each row so that the sum does not exceed 1.
    for i in range(G_eta.shape[0]):
        row_sum = np.sum(G_eta[i])
        if row_sum > 1:
            G_eta[i] /= row_sum
            
    for i in range(A_eta.shape[0]):
        row_sum = np.sum(A_eta[i])
        if row_sum > 1:
            A_eta[i] /= row_sum
            
    return G_beta, G_eta, A_beta, A_eta

def calc_penalty(A_c, G_c, uav_positions=None, uav_move=None):
    """
    Calculate the penalty based on channel capacity constraints and UAV crash conditions.

    Penalties are applied if:
      - The total ground-to-UAV capacity is below a predefined minimum.
      - A UAV's capacity to users exceeds its ground capacity.
      - A user's combined capacity (from ground and UAV) is below a predefined minimum.
      - (New) A UAV is predicted to be out of boundary or its actual height (before normalization) is less than 1,
        taking into account the movement of the current action (if provided).

    Parameters:
      A_c (np.array): Channel capacities from aerial APs to users, shape (n_a, n_u).
      G_c (np.array): Channel capacities from ground APs, shape (n_g, n_a+n_u).
                      The first n_u columns correspond to users, and the remaining n_a columns to UAVs.
      A_eta (np.array): Power allocation matrix for aerial APs.
      G_eta (np.array): Power allocation matrix for ground APs.
      uav_positions (np.array, optional): Current actual positions of UAVs (before normalization) with shape (3, n_a),
                                      where each column represents [x, y, height]. 
      uav_move (np.array, optional): Predicted move for each UAV with shape (3, n_a), representing change in [x, y, height].

    Returns:
      float: Total penalty.
    """
    import math
    import numpy as np

    # Penalty coefficient for low channel conditions (assumed to be negative)
    low_channel_penalty_coef = -cnf.LOW_CHANNEL

    def inverse_penalty(current_capacity, required_capacity, penalty_coef):
        """
        Compute a scaled penalty based on the deviation from the required capacity.
        A small delta is added to avoid division by zero.
        """
        delta = required_capacity * (math.sqrt(5) - 1) / 2
        penalty = penalty_coef * (3 * required_capacity / (current_capacity + delta) -
                                  3 * required_capacity / (required_capacity + delta))
        return penalty

    total_penalty = 0.0

    # Extract UAV-related channel capacity from ground APs.
    # In G_c, columns [n_u:] correspond to UAVs.
    uav_ground_capacity = G_c[:, n_u:]
    
    # For each UAV (aerial AP)
    for i in range(n_a):
        # Sum the capacity from all ground APs to the i-th UAV
        total_ground_uav_capacity = np.sum(uav_ground_capacity[:, i])
        # Sum the capacity from the i-th UAV to all users
        user_capacity = np.sum(A_c[i, :])
        
        # Condition 1: If UAV's ground capacity is below the minimum requirement
        if total_ground_uav_capacity < cnf.min_capacity_uav:
            total_penalty += inverse_penalty(total_ground_uav_capacity,
                                             cnf.min_capacity_uav,
                                             low_channel_penalty_coef)
            
        # Condition 2: If UAV's ground capacity is less than its capacity to users
        if total_ground_uav_capacity < user_capacity:
            total_penalty += inverse_penalty(total_ground_uav_capacity,
                                             user_capacity,
                                             low_channel_penalty_coef)

    # For each user, check total received capacity (from both ground and aerial APs)
    for u in range(n_u):
        # Capacity from ground APs (for the u-th user)
        user_ground_capacity = np.sum(G_c[:, u])
        # Capacity from aerial APs (for the u-th user)
        user_aerial_capacity = np.sum(A_c[:, u])
        total_user_capacity = user_ground_capacity + user_aerial_capacity
        
        # Condition 3: If user's total capacity is below the minimum requirement
        if total_user_capacity < cnf.min_capacity_user:
            total_penalty += inverse_penalty(total_user_capacity,
                                             cnf.min_capacity_user,
                                             low_channel_penalty_coef)
    
    # New Crash Penalty:
    # Check predicted UAV positions after applying the current action's movement.
    # If uav_move is provided, compute future position = current position + move,
    # otherwise, use current uav_positions directly.
    if uav_positions is not None:
        if uav_move is not None:
            future_positions = uav_positions + uav_move
        else:
            future_positions = uav_positions
        # uav_positions (or future_positions) is expected to be a (3, n_a) numpy array: [x, y, height]
        for i in range(future_positions.shape[1]):
            x, y, h = future_positions[:, i]
            # Check if UAV is out of valid boundaries or if its height is less than 1.
            if (x < 0 or x > cnf.MAX_X or y < 0 or y > cnf.MAX_Y or h < 1):
                total_penalty -= cnf.CRASH

    return total_penalty

def calc_cluster(A_eta, G_eta):
    """
    Compute the clustering (i.e., connected components) of the network based on power allocation matrices.
    
    The graph is constructed as follows:
      - Nodes represent ground APs, aerial APs (UAVs), and users.
      - Ground APs are indexed from 0 to n_g-1.
      - Aerial APs (UAVs) are indexed from n_g to n_g+n_a-1.
      - Users are indexed from n_g+n_a to n_g+n_a+n_u-1.
      - An edge is added between a ground AP and a user if the corresponding G_eta value > 0.
      - An edge is added between a ground AP and a UAV if the corresponding G_eta value > 0.
      - An edge is added between a UAV and a user if the corresponding A_eta value > 0.
    
    Returns:
      adj_matrix (np.array): Boolean adjacency matrix of the constructed graph.
      cluster_labels (np.array): An array where each element is the cluster label of the corresponding node.
    """
    import networkx as nx
    import numpy as np

    # Total number of nodes: ground APs + aerial APs + users
    total_nodes = n_g + n_a + n_u

    # Create an undirected graph and add all nodes
    graph = nx.Graph()
    graph.add_nodes_from(range(total_nodes))

    # Add edges between ground APs and users based on G_eta.
    # In G_eta: columns 0 to n_u-1 correspond to users.
    for g in range(n_g):
        for u in range(n_u):
            if G_eta[g, u] > 0:
                user_idx = n_g + n_a + u  # user node index in the graph
                graph.add_edge(g, user_idx)

    # Add edges between ground APs and aerial APs (UAVs) based on G_eta.
    # In G_eta: columns n_u to n_u+n_a-1 correspond to UAVs.
    for g in range(n_g):
        for a in range(n_a):
            if G_eta[g, n_u + a] > 0:
                uav_idx = n_g + a  # UAV node index in the graph
                graph.add_edge(g, uav_idx)

    # Add edges between aerial APs (UAVs) and users based on A_eta.
    # In A_eta: rows correspond to UAVs, columns to users.
    for a in range(n_a):
        for u in range(n_u):
            if A_eta[a, u] > 0:
                uav_idx = n_g + a  # UAV node index
                user_idx = n_g + n_a + u  # user node index
                graph.add_edge(uav_idx, user_idx)

    # Identify connected components (clusters)
    clusters = list(nx.connected_components(graph))

    # Assign each node a cluster label
    cluster_labels = np.zeros(total_nodes, dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            cluster_labels[node] = cluster_id

    # Create boolean adjacency matrix from the graph
    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=bool)
    for u, v in graph.edges():
        adj_matrix[u, v] = True
        adj_matrix[v, u] = True  # ensure symmetry for undirected graph

    return adj_matrix, cluster_labels

def calc_linkcost(adj_matrix, cluster_labels):
    """
    Calculate the total link cost based on UAV connections and cluster cooperation costs.

    The total link cost is computed as follows:
      1. For each UAV, add a cost proportional to the number of connected users.
      2. For each cluster, add a cooperation cost proportional to the number of ground APs
         and number of users in the cluster.

    Parameters:
      adj_matrix (np.array): A boolean adjacency matrix representing the connectivity among nodes.
      cluster_labels (np.array): An array containing the cluster label for each node.
                                  Nodes are ordered as follows:
                                    - Ground APs: indices 0 to n_g-1
                                    - UAVs: indices n_g to n_g+n_a-1
                                    - Users: indices n_g+n_a to n_g+n_a+n_u-1

    Returns:
      float: The total calculated link cost.
    """
    import numpy as np

    total_cost = 0.0

    # 1. Compute UAV connection cost.
    # For every UAV (indexed from n_g to n_g+n_a-1), count the number of connected users.
    for a in range(n_a):
        uav_idx = n_g + a  # UAV index in the node array.
        # Users are located from index n_g+n_a to n_g+n_a+n_u-1.
        user_connections = np.sum(adj_matrix[uav_idx, n_g + n_a:])
        total_cost += user_connections * cnf.LINK_COST

    # 2. Compute cluster cooperation cost.
    # Each cluster's cost is proportional to the number of ground APs and users within that cluster.
    for cluster_id in np.unique(cluster_labels):
        # Create a boolean mask for nodes in the current cluster.
        cluster_mask = (cluster_labels == cluster_id)
        
        # Count ground APs (indices 0 to n_g-1) in the cluster.
        num_ground_aps = np.sum(cluster_mask[:n_g])
        
        # Count users (indices n_g+n_a to n_g+n_a+n_u-1) in the cluster.
        num_users = np.sum(cluster_mask[n_g + n_a:])
        
        # Calculate the cluster's cooperation cost.
        cluster_cost = num_ground_aps * num_users * cnf.LINK_COST
        total_cost += cluster_cost

    return total_cost   

def calc_util(state, aution):
    """
    Compute the utility (reward) for the given state and action.

    Process:
      1. Preprocess the input state and action using arr_prep to obtain positions,
         moves (both normalized and de-normalized) and power allocation actions.
      2. Convert the power allocation actions and positions into structured matrices
         (G_beta, G_eta, A_beta, A_eta) for channel evaluation using arr2mat.
      3. Compute the channel capacities for ground APs and aerial APs using the Shannon
         capacity formula.
      4. Calculate penalties based on channel capacity constraints.
      5. Determine network clusters and compute link cost.
      6. Combine the channel capacities and compute the final reward.

    Returns:
      reward: Computed reward value.
      tot_capacity: Total network channel capacity.
      punishment: Computed penalty based on capacity constraints.
      link_cost: Computed link cost based on cluster cooperation and UAV connections.
      expert_action: Placeholder expert action (currently 0).
      subopt_expert_action: Placeholder sub-optimal expert action (currently 0).
      real_action: Concatenation of de-normalized move and power allocation action.
    """
    import numpy as np
    import math

    # 1. Preprocess state and action to extract positions, moves, and power allocation actions
    #    state contains normalized positions [0,1] and power allocation state.
    #    aution contains normalized moves and power allocation actions.
    actual_position, _, actual_move, power_alloc_action, norm_position, norm_move = arr_prep(state, aution)

    # 2. Convert power allocation actions and positions into structured matrices for channel evaluation.
    G_beta, G_eta, A_beta, A_eta = arr2mat(power_alloc_action, actual_position)

    # 3. Compute channel capacities for ground APs.
    # Initialize ground channel capacity matrix, shape: (n_g, n_a + n_u)
    G_c = np.zeros((n_g, n_a + n_u))
    for i in range(n_u + n_a):
        # Sum the interference (noise) from all other channels except the i-th one.
        # Using np.delete to remove column i for clarity.
        noise_coef1 = np.sum(np.delete(G_eta, i, axis=1), axis=1)  # shape: (n_g,)

        # Calculate total noise contributed by other signals.
        if i<5:
            noise_coef2 = np.sum(np.delete(A_eta, i, axis=1), axis=1)
            tot_noise = np.sum(noise_coef1 * cnf.P_G * G_beta[:, i]) + np.sum(noise_coef2 * cnf.P_A * A_beta[:, i])
        else:
            tot_noise = np.sum(noise_coef1 * cnf.P_G * G_beta[:, i])
        
        # Compute the received signal for the i-th channel.
        signal = cnf.P_G * G_beta[:, i] * G_eta[:, i]
        # Shannon capacity formula (log2(1 + SINR)).
        G_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))

    # 4. Compute channel capacities for aerial (UAV) APs.
    # Initialize aerial channel capacity matrix, shape: (n_a, n_u)
    A_c = np.zeros((n_a, n_u))
    for i in range(n_u):
        noise_coef1 = np.sum(np.delete(G_eta, i, axis=1), axis=1)
        noise_coef2 = np.sum(np.delete(A_eta, i, axis=1), axis=1)
        tot_noise = np.sum(noise_coef1 * cnf.P_G * G_beta[:, i]) + np.sum(noise_coef2 * cnf.P_A * A_beta[:, i])
        signal = cnf.P_A * A_beta[:, i] * A_eta[:, i]
        A_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))

    # 5. Calculate penalty based on channel capacity constraints.
    uav_positions = actual_position[:3 * cnf.NUM_A_AP].reshape(3, cnf.NUM_A_AP)
    uav_move = actual_move.reshape(3, cnf.NUM_A_AP)
    # Call calc_penalty with optional UAV positions (uav_move can be passed if available, here set to None)
    punishment = calc_penalty(A_c, G_c,uav_positions, uav_move)

    # 6. Determine network clusters and compute link cost.
    adj_matrix, cluster_labels = calc_cluster(A_eta, G_eta)
    link_cost = calc_linkcost(adj_matrix, cluster_labels)

    # 7. Combine ground and aerial AP channel capacities.
    # Assume cnf.NUM_AP represents total number of APs (ground + aerial).
    capacity = np.zeros((cnf.NUM_AP, n_u))
    # Ground APs capacity: first n_g rows, only for user channels.
    capacity[:n_g, :] = G_c[:, :n_u]
    # Aerial APs capacity: remaining rows.
    capacity[n_g:, :] = A_c[:, :n_u]
    tot_capacity = np.sum(capacity)

    # 8. Combine capacities, penalties, and link cost to form the final reward.
    p_c = cnf.punishment_coef
    c_c = cnf.cost_coef
    reward = tot_capacity * (1 - p_c - c_c) + punishment * p_c - link_cost * c_c

    # Placeholder expert actions (for imitation mode, if needed)
    expert_action = 0
    subopt_expert_action = 0

    # The real action is the concatenation of the normalized move and power allocation action.
    real_action = np.concatenate((norm_move, power_alloc_action))
    
    return reward, tot_capacity, punishment, link_cost, expert_action, subopt_expert_action, real_action

def calc_util_obs(power_alloc_action, actual_position):
    """
    计算给定功率分配和实际位置的效用（奖励）。
    
    与calc_util的区别是：该函数直接接受功率分配动作和实际位置，而不需要对状态和动作进行预处理。
    
    过程:
      1. 使用arr2mat将功率分配动作和位置转换为结构化矩阵(G_beta, G_eta, A_beta, A_eta)用于信道评估。
      2. 使用Shannon容量公式计算地面AP和空中AP的信道容量。
      3. 根据信道容量约束计算惩罚。
      4. 确定网络集群并计算链路成本。
      5. 组合信道容量并计算最终奖励。
    
    参数:
      power_alloc_action: 功率分配动作数组，范围[0,1]
      actual_position: 所有AP和用户的实际位置(非归一化)
      
    返回:
      reward: 计算的奖励值
      tot_capacity: 总网络信道容量
      punishment: 基于容量约束的计算惩罚
      link_cost: 基于集群合作和UAV连接的计算链路成本
    """
    import numpy as np
    import math
    
    # 1. 将功率分配动作和位置转换为结构化矩阵用于信道评估
    G_beta, G_eta, A_beta, A_eta = arr2mat(power_alloc_action, actual_position)
    
    # 2. 计算地面AP的信道容量
    # 初始化地面信道容量矩阵，形状: (n_g, n_a + n_u)
    G_c = np.zeros((n_g, n_a + n_u))
    for i in range(n_u + n_a):
        # 计算除第i个信道外的所有其他信道的干扰(噪声)总和
        # 使用np.delete移除列i以提高清晰度
        noise_coef1 = np.sum(np.delete(G_eta, i, axis=1), axis=1)  # shape: (n_g,)
        
        # 计算其他信号贡献的总噪声
        if i < n_u:
            noise_coef2 = np.sum(np.delete(A_eta, i, axis=1), axis=1)
            tot_noise = np.sum(noise_coef1 * cnf.P_G * G_beta[:, i]) + np.sum(noise_coef2 * cnf.P_A * A_beta[:, i])
        else:
            tot_noise = np.sum(noise_coef1 * cnf.P_G * G_beta[:, i])
        
        # 计算第i个信道的接收信号
        signal = cnf.P_G * G_beta[:, i] * G_eta[:, i]
        # Shannon容量公式 (log2(1 + SINR))
        G_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))
    
    # 3. 计算空中(UAV) AP的信道容量
    # 初始化空中信道容量矩阵，形状: (n_a, n_u)
    A_c = np.zeros((n_a, n_u))
    for i in range(n_u):
        noise_coef1 = np.sum(np.delete(G_eta, i, axis=1), axis=1)
        noise_coef2 = np.sum(np.delete(A_eta, i, axis=1), axis=1)
        tot_noise = np.sum(noise_coef1 * cnf.P_G * G_beta[:, i]) + np.sum(noise_coef2 * cnf.P_A * A_beta[:, i])
        signal = cnf.P_A * A_beta[:, i] * A_eta[:, i]
        A_c[:, i] = np.log2(1 + signal / (tot_noise + cnf.white_noise))
    
    # 4. 根据信道容量约束计算惩罚
    uav_positions = actual_position[:3 * cnf.NUM_A_AP].reshape(3, cnf.NUM_A_AP)
    # 这里不传入uav_move参数，因为在观察函数中我们只关心当前位置
    punishment = calc_penalty(A_c, G_c, uav_positions)
    
    # 5. 确定网络集群并计算链路成本
    adj_matrix, cluster_labels = calc_cluster(A_eta, G_eta)
    link_cost = calc_linkcost(adj_matrix, cluster_labels)
    
    # 6. 组合地面和空中AP信道容量
    # 假设cnf.NUM_AP表示AP的总数(地面 + 空中)
    capacity = np.zeros((cnf.NUM_AP, n_u))
    # 地面AP容量: 前n_g行，仅用于用户信道
    capacity[:n_g, :] = G_c[:, :n_u]
    # 空中AP容量: 剩余行
    capacity[n_g:, :] = A_c[:, :n_u]
    tot_capacity = np.sum(capacity)
    
    # 7. 组合容量、惩罚和链路成本形成最终奖励
    p_c = cnf.punishment_coef
    c_c = cnf.cost_coef
    reward = tot_capacity * (1 - p_c - c_c) + punishment * p_c - link_cost * c_c
    
    return reward, tot_capacity, punishment, link_cost