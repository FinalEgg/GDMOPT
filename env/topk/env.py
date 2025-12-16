
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeObservation
from tianshou.env import DummyVectorEnv
import numpy as np

from .config import (
    X, Y, H, M, N, P, pd, pu, TAU_P, ALPHA1, ALPHA2, XI1, XI2, 
    CAPACITY_THRESHOLD, FIXED_REWARD, STEPS_PER_EPISODE, NOISE_POWER, 
    GEO_BETA_THRESHOLD, K_MAX
)

class TopKEnv(gym.Env):
    def __init__(self, top_p=0.6):
        self.top_p = top_p
        self._num_steps = 0
        self._terminated = False

        # 初始化位置
        self._init_bs_positions()
        self.uav_positions = np.random.uniform(0, [X, Y, H], (N, 3))

        # 状态空间设计 (Compressed)
        # 每个 UAV 输入: 
        # 1. 自身位置 (3维: x, y, z)
        # 2. Top-K 个连接基站的信息 (K * 3维: BS_ID, Beta, Angle)
        # 总维度: 3 + K_MAX * 3
        self.state_per_uav = 3 + K_MAX * 3
        obs_dim = N * self.state_per_uav
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

        # 动作空间设计 (Compressed)
        # 每个 UAV 输出: Top-K 个连接基站的功率
        action_dim = N * K_MAX
        self._action_space = Box(low=0, high=1, shape=(action_dim,))

        self._steps_per_episode = STEPS_PER_EPISODE

        # 物理矩阵
        self.connection_matrix = np.zeros((M, N))
        self.power_matrix = np.zeros((M, N))
        self.beta_matrix = np.zeros((M, N))
        self.gamma_matrix = np.zeros((M, N))
        
        # 辅助索引矩阵: 记录每个 UAV 连接的 BS ID，用于动作映射
        # Shape: (N, K_MAX). 值 -1 表示无连接 (Padding)
        self.connection_indices = np.full((N, K_MAX), -1, dtype=int)
        
        self.cached_state = np.zeros(obs_dim)
        self.baseline_capacity = 0.0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state(self):
        return self.cached_state

    def _init_bs_positions(self):
        """Initialize BS positions in a 4x4 grid"""
        if M == 16:
            # Create a 4x4 grid
            x_coords = np.linspace(X/8, 7*X/8, 4)
            y_coords = np.linspace(Y/8, 7*Y/8, 4)
            xv, yv = np.meshgrid(x_coords, y_coords)
            self.bs_positions = np.column_stack((xv.flatten(), yv.flatten()))
        else:
            # Fallback to random if M is not 16
            self.bs_positions = np.random.uniform(0, [X, Y], (M, 2))

    def _calculate_large_scale_fading(self):
        """计算大尺度衰落"""
        bs_pos_exp = self.bs_positions[:, np.newaxis, :]
        uav_pos_exp = self.uav_positions[np.newaxis, :, :]
        
        diff_xy = bs_pos_exp - uav_pos_exp[:, :, :2]
        Rmk = np.linalg.norm(diff_xy, axis=2)
        
        diff_z = -uav_pos_exp[:, :, 2]
        Dmk = np.sqrt(Rmk**2 + diff_z**2)
        
        Hmk = uav_pos_exp[:, :, 2]
        theta_mk = np.degrees(np.arctan2(Hmk, Rmk + 1e-8))
        self.theta_matrix = theta_mk # 保存角度用于状态
        
        PmkL = 1 / (1 + XI1 * np.exp(-XI2 * (theta_mk - XI1)))
        self.beta_matrix = PmkL * Dmk**(-ALPHA1) + (1 - PmkL) * Dmk**(-ALPHA2)
        
        tau_pu_beta = TAU_P * pu * self.beta_matrix
        self.gamma_matrix = (tau_pu_beta * self.beta_matrix) / (tau_pu_beta + 1)

    def _update_connection_logic(self):
        """
        混合 Top-P 和 Top-K 连接策略
        规则:
        1. 计算 Top-P 集合。
        2. 过滤掉 Beta < GEO_BETA_THRESHOLD 的链路。
        3. 如果连接数 < 3，保持 Top-P 结果。
        4. 如果连接数 > 3，只取前 3 强。
        5. 如果连接数 == 0，强制连接最强的一个 (忽略阈值)。
        """
        self.connection_matrix = np.zeros((M, N))
        self.connection_indices = np.full((N, K_MAX), -1, dtype=int)
        
        for n in range(N):
            # 获取该 UAV 到所有 BS 的信道增益
            betas = self.beta_matrix[:, n]
            
            # 排序 (从大到小)
            sorted_idx = np.argsort(betas)[::-1]
            sorted_betas = betas[sorted_idx]
            
            # 1. Top-P 截断
            cumsum = np.cumsum(sorted_betas)
            total = cumsum[-1]
            # 找到累积功率达到 P% 的位置
            cutoff_idx = np.searchsorted(cumsum, total * self.top_p)
            # 包含 cutoff_idx 本身
            candidates_idx = sorted_idx[:cutoff_idx+1]
            
            # 2. 阈值过滤
            valid_candidates = []
            for idx in candidates_idx:
                if betas[idx] >= GEO_BETA_THRESHOLD:
                    valid_candidates.append(idx)
            
            # 3 & 4. Top-K 限制
            final_connections = valid_candidates[:K_MAX]
            
            # 5. 保底连接
            if len(final_connections) == 0:
                # 强制连接最强的一个
                final_connections = [sorted_idx[0]]
                
            # 更新矩阵
            for i, bs_idx in enumerate(final_connections):
                self.connection_matrix[bs_idx, n] = 1.0
                self.connection_indices[n, i] = bs_idx

    def _update_state(self):
        """构建压缩状态"""
        # 1. UAV 位置归一化
        pos_feat = self.uav_positions / np.array([X, Y, H]) # (N, 3)
        
        # 2. 连接信息
        conn_feats = []
        for n in range(N):
            uav_conn_feat = []
            for k in range(K_MAX):
                bs_idx = self.connection_indices[n, k]
                
                if bs_idx != -1:
                    # 有连接
                    beta = self.beta_matrix[bs_idx, n]
                    angle = self.theta_matrix[bs_idx, n]
                    
                    # 归一化
                    log_beta = np.log10(beta + 1e-20)
                    norm_beta = (log_beta + 10.0) / 5.0
                    norm_angle = angle / 180.0
                    
                    # BS ID 归一化 (0-1) 或者保持整数供 Embedding 使用
                    # 这里我们存为浮点数，但在 Model 中需要还原为 Int
                    # 为了方便，我们这里存归一化的 ID，Model 里再处理?
                    # 不，Model 需要 Int。我们直接存 ID。
                    # 但是 Box space 是 float32。
                    # 我们存 ID + 0.1 (避免 0 和 padding 混淆? 不用，-1 是 padding)
                    # 直接存 ID。
                    
                    uav_conn_feat.extend([bs_idx, norm_beta, norm_angle])
                else:
                    # Padding (无连接)
                    # ID = -1 (或者 M, 如果 Embedding size = M+1)
                    # Beta = -2.0 (代表 0)
                    # Angle = 0
                    uav_conn_feat.extend([-1.0, -2.0, 0.0])
            
            conn_feats.append(uav_conn_feat)
            
        conn_feats = np.array(conn_feats) # (N, K*3)
        
        # 拼接
        self.cached_state = np.hstack([pos_feat, conn_feats]).flatten()

    def _calculate_capacity(self):
        """计算容量 (物理模型)"""
        signal_components = np.sqrt(self.power_matrix) * self.gamma_matrix
        signals = np.sum(signal_components, axis=0)
        numerator = pd * (signals ** 2)
        
        weighted_power = self.power_matrix * self.gamma_matrix
        T = np.sum(weighted_power, axis=1)
        interference_source = T[:, np.newaxis] - weighted_power
        interference_matrix = self.beta_matrix * interference_source
        interferences = np.sum(interference_matrix, axis=0)
        
        denominator = pd * interferences + NOISE_POWER
        SINR = numerator / denominator
        Capacity = np.log2(1 + SINR)
        Capacity = np.clip(Capacity, 0, 10)
        total_capacity = np.sum(Capacity)
        
        return Capacity, total_capacity

    def _calculate_baseline_capacity(self):
        """计算基准容量 (等功率分配)"""
        original_power = self.power_matrix.copy()
        
        # 等功率分配: 每个基站将功率平均分给连接的 UAV
        num_connections = np.sum(self.connection_matrix, axis=1, keepdims=True)
        # 避免除零
        avg_power = np.divide(1.0, num_connections, where=num_connections > 0)
        self.power_matrix = self.connection_matrix * avg_power
        
        _, baseline_cap = self._calculate_capacity()
        self.power_matrix = original_power
        return baseline_cap

    def step(self, action):
        # action shape: (N * K_MAX,)
        # 1. 还原动作矩阵
        raw_actions = action.reshape(N, K_MAX)
        
        # 2. 映射到物理功率矩阵 (M, N)
        self.power_matrix = np.zeros((M, N))
        for n in range(N):
            for k in range(K_MAX):
                bs_idx = self.connection_indices[n, k]
                if bs_idx != -1:
                    power_val = raw_actions[n, k]
                    # 确保非负
                    power_val = max(0.0, power_val)
                    self.power_matrix[bs_idx, n] = power_val
                    
        # 3. 基站功率归一化 (Per-BS Constraint)
        # 方案 1: 后处理归一化
        bs_total_power = np.sum(self.power_matrix, axis=1) # (M,)
        
        for m in range(M):
            if bs_total_power[m] > 1.0:
                scale = 1.0 / bs_total_power[m]
                self.power_matrix[m, :] *= scale
                
        # 4. 计算奖励
        _, total_capacity = self._calculate_capacity()
        
        # 相对奖励
        raw_reward = (total_capacity - self.baseline_capacity)
        scaled_reward = raw_reward * 0.1
        reward = np.clip(scaled_reward, -5.0, 5.0)
        
        self._num_steps += 1
        terminated = self._num_steps >= self._steps_per_episode
        
        return self.state, reward, False, terminated, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._num_steps = 0
        
        self._init_bs_positions()
        self.uav_positions = np.random.uniform(0, [X, Y, H/2], (N, 3))
        
        self._calculate_large_scale_fading()
        self._update_connection_logic()
        self._update_state()
        
        self.power_matrix = np.zeros((M, N))
        self.baseline_capacity = self._calculate_baseline_capacity()
        
        return self.state, {}

    def seed(self, seed=None):
        np.random.seed(seed)

def make_topk_env(training_num=0, test_num=0, top_p=0.6, norm_reward=False):
    def _make_env():
        e = TopKEnv(top_p=top_p)
        if norm_reward:
            e = NormalizeObservation(e)
        return e

    env = TopKEnv(top_p=top_p)
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv([_make_env for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv([_make_env for _ in range(test_num)])
        test_envs.seed(0)
        
    return env, train_envs, test_envs
