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
    #split state
    idx = 0
    x_a = state[idx:idx + n_a]  #x of aerial APs
    idx += n_a
    y_a = state[idx:idx + n_a]  #y of aerial APs
    idx += n_a
    h_a = state[idx:idx + n_a]  #h of aerial APs
    idx += n_a
    x_g = state[idx:idx + n_g]  #x of ground APs
    idx += n_g
    y_g = state[idx:idx + n_g]  #y of ground APs
    idx += n_g
    x_u = state[idx:idx + n_u]  #x of users
    idx += n_u
    y_u = state[idx:idx + n_u]  #y of users

    # action: 
    # n_g rows: 1 digit total power allocation, n_u digits for users, n_a digits for aerial APs
    # n_a rows: 1 digit total power allocation, n_u digits for users
    
    p_tot=np.zeros(cnf.NUM_AP)  # total power allocation

    # Ground AP
    G_beta = np.zeros((n_g, n_a + n_u))
    G_eta = np.zeros((n_g, n_a + n_u))
    for i in range(n_g):    # ith G_AP
        p_tot[i] = action[i*(1+n_u+n_a)]    # total power allocation
        for j in range(n_u):
            G_beta[i, j] = compute_beta(x_g[i], y_g[i], 0, x_u[j], y_u[j])
            G_eta[i, j] = action[i*(n_u+n_a+1)+1+j]
        for j in range(n_a):
            G_beta[i, n_u+j] = compute_beta(x_g[i], y_g[i], h_a[j], x_a[j], y_a[j])
            G_eta[i,n_u+j] = action[i*(n_u+n_a+1)+1+n_u+j]

    # UAV 
    A_beta = np.zeros((n_a, n_u))
    A_eta = np.zeros((n_a, n_u))
    for i in range(n_a):
        p_tot[n_g+i] = action[n_g*(n_a+n_u+1)+i*(1+n_u)]
        for j in range(n_u):
            A_beta[i, j] = compute_beta(x_a[i], y_a[i], h_a[i], x_u[j], y_u[j])
            A_eta[i,j]=action[n_g*(n_a+n_u+1)+i*(1+n_u)+1+j]
    
    # UAV speed
    speed = action[-n_a*3:]

    # normalization
    for i in range(G_eta.shape[0]):
        row_sum = 0
        for j in range(G_eta.shape[1]):
            if G_eta[i, j] < cnf.threshold:
                G_eta[i, j] = 0  
            row_sum += G_eta[i, j]
        if row_sum != 0:
            for j in range(G_eta.shape[1]):
                G_eta[i, j] = G_eta[i, j] / row_sum

    for i in range(A_eta.shape[0]):
            row_sum = 0
            for j in range(A_eta.shape[1]):
                if A_eta[i, j] < cnf.threshold:
                    A_eta[i, j] = 0
                row_sum += A_eta[i, j]
            if row_sum != 0:
                for j in range(A_eta.shape[1]):
                    A_eta[i, j] = A_eta[i, j] / row_sum

    return G_beta, G_eta, A_beta, A_eta, p_tot, speed

def compute_beta(x1, y1, h1, x2, y2):
    xi1 = cnf.xi1
    xi2 = cnf.xi2
    alpha1 = cnf.LOS
    alpha2 = cnf.NLOS

    R = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    D = math.sqrt(R**2 + h1**2)

    if R == 0:
        theta = math.pi / 2
    else:
        theta = math.atan(h1 / R)
    
    P_L = 1.0 / (1.0 + xi1 * math.exp(-xi2 * (theta - xi1)))
    P_NL = 1.0 - P_L

    beta = P_L * (D ** (-alpha1)) + P_NL * (D ** (-alpha2))
    return beta

# Function to compute utility (reward) for the given state and action
def CompUtility(State, Aution):
    weight = torch.from_numpy(np.array(Aution)).float()
    position = torch.from_numpy(np.array(State)).float()
    weight = torch.abs(weight)
    G_beta, G_eta, A_beta, A_eta, p_tot, speed = split(weight, position)

    G_c = np.zeros((n_g, n_a + n_u))
    noise=0
    signal=0
    for i in range(G_beta.shape[1]): #ith user
        for j in range(G_beta.shape[0]): #jth AP
            signal = signal + cnf.P_G*p_tot[j]*G_beta[j,i]*G_eta[j,i]
            noise = noise+ cnf.P_G*p_tot[j]*G_beta[j,i]*G_eta[j,i]
        G_c[j,i] = np.log2(1+signal/(noise+cnf.white_noise))

    A_c = np.zeros((n_a, n_u))
    noise=0
    signal=0
    for i in range(A_beta.shape[1]): #ith user
        for j in range(A_beta.shape[0]):
            signal = signal + cnf.P_A*p_tot[j]*A_beta[j,i]*A_eta[j,i]
            noise = noise+ cnf.P_A*p_tot[j]*A_beta[j,i]*A_eta[j,i]
    A_c[j,i] = np.log2(1+signal/(noise+cnf.white_noise))

    # 无人机信道容量超出限制问题应当使用惩罚项解决
    # for i in range(n_u,n_u+n_a): #ap in G_c
    #     front_haul=0
    #     for j in range(n_g): # jth G in G_c
    #         front_haul = front_haul + cnf.P_G*p_tot[j]*G_beta[j,i]
    #     capacity[i-n_u+n_g,j] = min(front_haul,A_c[i-n_u,j])

    capacity =np.zeros((cnf.NUM_AP, n_u))
    for j in range(n_u):
        for i in range(n_g):
            capacity[i,j] = G_c[i,j]
        for i in range(n_a):
            capacity[i+n_g,j] = A_c[i,j]
    
    # 计算总信道容量
    total_channel_capacity = np.sum(capacity)
    reward = total_channel_capacity 

    expert_action = 0
    subopt_expert_action = 0

    return reward, expert_action, subopt_expert_action, speed