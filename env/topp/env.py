
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeObservation
from tianshou.env import DummyVectorEnv
import numpy as np

# 导入独立的配置
from .config import (
    X, Y, H, M, N, P, pd, pu, TAU_P, ALPHA1, ALPHA2, XI1, XI2, 
    CAPACITY_THRESHOLD, FIXED_REWARD, STEPS_PER_EPISODE, NOISE_POWER, 
    GEO_BETA_THRESHOLD
)

class TopPEnv(gym.Env):
    def __init__(self, top_p=0.6):
        self.top_p = top_p
        self._num_steps = 0
        self._terminated = False

        # 初始化基站位置
        self._init_bs_positions()

        # 初始化无人机位置
        self.uav_positions = np.random.uniform(0, [X, Y, H], (N, 3))

        # 观测空间
        self.uav_feature_dim = M * 2 + 3
        obs_dim = N * self.uav_feature_dim
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

        # 动作空间：仅功率分配矩阵 (M*N)
        # 注意：连接矩阵由 top_p 规则决定，不再是动作的一部分
        action_dim = M * N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))

        self._steps_per_episode = STEPS_PER_EPISODE

        # 初始化矩阵
        self.connection_matrix = np.zeros((M, N))
        self.power_matrix = np.zeros((M, N))
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
        
        PmkL = 1 / (1 + XI1 * np.exp(-XI2 * (theta_mk - XI1)))
        self.beta_matrix = PmkL * Dmk**(-ALPHA1) + (1 - PmkL) * Dmk**(-ALPHA2)
        
        tau_pu_beta = TAU_P * pu * self.beta_matrix
        self.gamma_matrix = (tau_pu_beta * self.beta_matrix) / (tau_pu_beta + 1)
        
    def _update_state(self):
        """更新状态观测值"""
        # 仅在观测中对未连接的链路进行掩码处理
        # 注意：self.beta_matrix 本身保持不变，用于计算连接
        masked_beta = self.beta_matrix * self.connection_matrix
        
        # 构造状态
        log_beta = np.log10(masked_beta + 1e-20)
        norm_log_beta = (log_beta + 10.0) / 5.0
        
        # 角度信息也应该被掩码吗？通常如果链路断开，角度信息也不重要了
        # 但为了保持一致性，我们可以保留角度，或者也掩码。
        # 用户只提到了“将没有连接的大尺度衰落系数全部定为0”
        # 所以我们只处理 beta
        
        Hmk = self.uav_positions[:, 2] # (N,)
        # 需要重新计算 theta_mk 吗？它在 _calculate_large_scale_fading 中计算过但没存下来
        # 为了避免重复计算，我们可以将其存为成员变量，或者重新计算
        
        bs_pos_exp = self.bs_positions[:, np.newaxis, :]
        uav_pos_exp = self.uav_positions[np.newaxis, :, :]
        diff_xy = bs_pos_exp - uav_pos_exp[:, :, :2]
        Rmk = np.linalg.norm(diff_xy, axis=2)
        theta_mk = np.degrees(np.arctan2(Hmk[None, :], Rmk + 1e-8))
        
        norm_angle = theta_mk / 180.0
        
        beta_feat = norm_log_beta.T[:, :, np.newaxis]
        angle_feat = norm_angle.T[:, :, np.newaxis]
        link_features = np.concatenate([beta_feat, angle_feat], axis=2).reshape(N, -1)
        pos_features = self.uav_positions / np.array([X, Y, H])
        
        self.cached_state = np.hstack([link_features, pos_features]).flatten()

    def _calculate_capacity(self):
        """计算容量"""
        # 仅考虑连接矩阵为 1 的链路
        # power_matrix 已经在 step 中被 connection_matrix 掩码过了
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
        """计算当前拓扑下的基准容量 (等功率分配)"""
        # 保存当前的功率矩阵
        original_power = self.power_matrix.copy()
        
        # 设置为等功率分配 (每个连接的链路分配 1.0/连接数，或者简单地设为 0.5)
        # 这里我们假设每个基站平均分配功率给连接的 UAV
        # 由于 connection_matrix 已经确定，我们计算每个基站连接的 UAV 数量
        num_connections = np.sum(self.connection_matrix, axis=1, keepdims=True)
        # 避免除零
        avg_power = np.divide(1.0, num_connections, where=num_connections > 0)
        
        self.power_matrix = self.connection_matrix * avg_power
        
        _, baseline_cap = self._calculate_capacity()
        
        # 恢复原始功率矩阵
        self.power_matrix = original_power
        
        return baseline_cap

    def _calculate_threshold_reward(self):
        """基于相对提升的奖励计算"""
        _, total_capacity = self._calculate_capacity()
        
        # 如果基准容量太小，说明场景极难，避免除零
        if self.baseline_capacity < 1e-3:
            # 这种情况下，任何正容量都是巨大的提升
            reward = total_capacity * 10.0
        else:
            # 计算相对于基准的提升比例
            # 例如：基准 10，当前 12 -> 提升 20% -> 奖励 0.2 * Scale
            # 或者直接用差值：12 - 10 = 2
            
            # 方案 A: 差值奖励 (更稳定)
            raw_reward = (total_capacity - self.baseline_capacity)
            
            # 奖励缩放与截断 (Reward Scaling & Clipping)
            # 1. 缩放: 将差值缩小，使其落在一个对神经网络友好的区间 (例如 -1 到 1 附近)
            # 假设容量差值通常在 -50 到 +50 之间
            scaled_reward = raw_reward * 0.1 
            
            # 2. 截断: 防止极端的离群值破坏 Critic 的梯度
            reward = np.clip(scaled_reward, -5.0, 5.0)
            
        return reward

    def _update_connection_matrix_topp(self):
        """根据 Top-P 规则更新连接矩阵"""
        sorted_indices = np.argsort(self.beta_matrix, axis=0)[::-1]
        sorted_betas = np.take_along_axis(self.beta_matrix, sorted_indices, axis=0)
        cumsum_betas = np.cumsum(sorted_betas, axis=0)
        total_beta = cumsum_betas[-1, :]
        threshold_values = total_beta * self.top_p
        mask_cumsum = cumsum_betas >= threshold_values[None, :]
        cutoff_indices = np.argmax(mask_cumsum, axis=0)
        valid_counts = np.sum(sorted_betas >= GEO_BETA_THRESHOLD, axis=0)
        rule_based_cutoffs = np.minimum(cutoff_indices, valid_counts - 1)
        final_cutoffs = np.maximum(rule_based_cutoffs, 0)
        
        target_matrix = np.zeros((M, N))
        row_indices = np.arange(M)[:, None]
        selection_mask = (row_indices <= final_cutoffs[None, :])
        target_bs_indices = sorted_indices[selection_mask]
        col_indices = np.tile(np.arange(N), (M, 1))
        target_uav_indices = col_indices[selection_mask]
        target_matrix[target_bs_indices, target_uav_indices] = 1.0
        
        self.connection_matrix = target_matrix

    def step(self, action):
        assert not self._terminated, "Episode has terminated"

        # 1. 更新信道 (假设静态信道，reset时已计算，如果是动态信道需在此更新)
        # 目前是静态信道，所以 beta_matrix 不变，connection_matrix 也不变
        # 如果引入移动性，这里需要更新位置和信道，并重新计算 connection_matrix
        
        # 2. 确定连接 (Top-P)
        # 对于静态场景，reset 时计算一次即可。但为了通用性，这里调用一次
        self._update_connection_matrix_topp()
        
        # 更新状态观测 (应用连接掩码)
        self._update_state()

        # 3. 应用动作 (功率分配)
        raw_power = action.reshape(M, N)
        
        # 强制应用连接掩码：未连接的链路功率强制为 0
        self.power_matrix = raw_power * self.connection_matrix

        # 功率归一化 (每个基站总功率 <= 1)
        total_power = np.sum(self.power_matrix, axis=1, keepdims=True)
        mask = total_power > 1.0
        
        # 使用 np.divide 的 where 参数避免除零警告
        # 当 mask 为 False 时，不进行除法，直接返回原值 (通过 np.where 再次选择)
        # 这里为了安全，分母加一个极小值，或者只对需要归一化的部分计算
        
        if np.any(mask):
            normalized_power = np.divide(self.power_matrix, total_power, where=mask) # 仅在 mask 为 True 处计算，其他地方可能未定义
            # 但 np.divide 的 where 行为是：where=False 的地方保留原数组的值（如果提供了 out）或者未初始化
            # 更简单的方法是分母加 epsilon
            self.power_matrix = np.where(mask, self.power_matrix / (total_power + 1e-20), self.power_matrix)

        # 4. 计算奖励
        reward = self._calculate_threshold_reward()

        self._num_steps += 1
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True

        return self.state, reward, False, self._terminated, {'num_steps': self._num_steps}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        self._num_steps = 0
        self._terminated = False

        self._init_bs_positions()
        self.uav_positions = np.random.uniform(0, [X, Y, H/2], (N, 3))
        
        self._calculate_large_scale_fading()
        self._update_connection_matrix_topp() # 初始化连接矩阵
        self._update_state() # 初始化状态
        
        self.power_matrix = np.zeros((M, N))
        
        # 计算基准容量
        self.baseline_capacity = self._calculate_baseline_capacity()

        return self.state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        np.random.seed(seed)

def make_topp_env(training_num=0, test_num=0, top_p=0.6, norm_reward=False):
    """创建 TopPEnv 的辅助函数"""
    def _make_env():
        e = TopPEnv(top_p=top_p)
        if norm_reward:
            e = NormalizeObservation(e)
        return e

    env = TopPEnv(top_p=top_p)
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv([_make_env for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv([_make_env for _ in range(test_num)])
        test_envs.seed(0)
        
    return env, train_envs, test_envs
