import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
from env import config as cnf
os.environ['KMP_DUPLICATE_LIB_OK']='True'

n_a = cnf.NUM_A_AP  # number of aerial APs
n_g = cnf.NUM_G_AP  # number of ground APs
n_u = cnf.NUM_USERS # number of users

def split(action,state):
    action = action.detach().numpy() if torch.is_tensor(action) else np.array(action)
    state = state.detach().numpy() if torch.is_tensor(state) else np.array(state)

    # reshape state into structured array
    aerial_state = state[:3*n_a].reshape(3, n_a)  # [x_a, y_a, h_a]
    x_a, y_a, h_a = aerial_state
    
    ground_state = state[3*n_a:3*n_a + 2*n_g].reshape(2, n_g)  # [x_g, y_g]
    x_g, y_g = ground_state
    
    user_state = state[3*n_a + 2*n_g:].reshape(2, n_u)  # [x_u, y_u]
    x_u, y_u = user_state

    # action: 
    # n_g rows: 1 digit total power allocation, n_u digits for users, n_a digits for aerial APs
    # n_a rows: 1 digit total power allocation, n_u digits for users
    
    p_tot=np.zeros(cnf.NUM_AP)  # total power allocation

    # Ground AP 
    G_beta = np.zeros((n_g, n_a + n_u))
    G_eta = np.zeros((n_g, n_a + n_u))
    # G_beta: shape (n_g, n_a + n_u)
    x_g_expanded = x_g[:, np.newaxis]  # 形状: (n_g, 1)
    y_g_expanded = y_g[:, np.newaxis]  # 形状: (n_g, 1)
    dist_to_users = np.sqrt((x_g_expanded - x_u)**2 + (y_g_expanded - y_u)**2)
    G_beta[:, :n_u] = dist_to_users ** -2
    dist_to_aerial = np.sqrt(
        (x_g_expanded - x_a)**2 + 
        (y_g_expanded - y_a)**2 + 
        h_a**2
    )
    G_beta[:, n_u:] = dist_to_aerial ** -2

    # Aerial AP
    A_beta = np.zeros((n_a, n_u))
    A_eta = np.zeros((n_a, n_u))
    # A_beta: shape (n_a, n_u)
    x_a_expanded = x_a[:, np.newaxis]  # 形状: (n_a, 1)
    y_a_expanded = y_a[:, np.newaxis]  # 形状: (n_a, 1)
    h_a_expanded = h_a[:, np.newaxis]  # 形状: (n_a, 1)
    distances = np.sqrt(
        (x_a_expanded - x_u)**2 + 
        (y_a_expanded - y_u)**2 + 
        h_a_expanded**2
    )
    A_beta = distances ** -2
    
    start = n_g*(n_u+n_a)
    # G_eta
    action_g = action[:start].reshape(n_g, -1)  # 重塑为(n_g, n_u+n_a)
    G_eta = action_g  
    
    # A_eta
    action_a = action[start:].reshape(n_a, -1)  # 重塑为(n_a, n_u+1)
    A_eta = action_a  # 去掉每行的第一个元素(原来的总功率分配)

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

# Function to compute utility (reward) for the given state and action
def CompUtility(State, Aution):
    G_beta, G_eta, A_beta, A_eta = split(Aution, State)
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


    # 合并地面AP和空中AP的信道容量
    capacity = np.zeros((cnf.NUM_AP, n_u))
    capacity[:n_g, :] = G_c[:, :n_u]  # 地面AP的容量
    capacity[n_g:, :] = A_c[:, :n_u]  # 空中AP的容量
    reward = np.sum(capacity)

    expert_action = 0
    subopt_expert_action = 0

    return reward, expert_action, subopt_expert_action, Aution