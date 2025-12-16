import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from tianshou.env import DummyVectorEnv
import numpy as np

# 导入独立的配置
from .config import (
    X, Y, H, M, N, P, pd, pu, TAU_P, ALPHA1, ALPHA2, XI1, XI2, 
    CAPACITY_THRESHOLD, FIXED_REWARD, STEPS_PER_EPISODE, NOISE_POWER, 
    GEO_BETA_THRESHOLD, GEO_REWARD_HIT, GEO_PENALTY_MISS, 
    GEO_PENALTY_USELESS, GEO_BONUS_PERFECT, GEO_PENALTY_WRONG, 
    GEO_PENALTY_NO_CONNECT
)

# 复用 CellFreeEnv 的大部分逻辑，但重写关键部分以使用本地配置
# 为了完全解耦，这里选择直接复制并修改 CellFreeEnv 的核心逻辑，
# 而不是继承，这样修改 config 不会影响原环境。

class ThresholdEnv(gym.Env):
    def __init__(self, reward_mode="threshold", top_p=0.6):
        self.reward_mode = reward_mode
        self.top_p = top_p
        self._num_steps = 0
        self._terminated = False

        # 初始化基站位置（均匀分布）
        self.bs_positions = np.random.uniform(0, [X, Y], (M, 2))

        # 初始化无人机位置
        self.uav_positions = np.random.uniform(0, [X, Y, H], (N, 3))

        # 观测空间
        self.uav_feature_dim = M * 2 + 3
        obs_dim = N * self.uav_feature_dim
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

        # 动作空间：功率分配矩阵 (M*N)
        action_dim = M * N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))

        self._steps_per_episode = STEPS_PER_EPISODE

        # 初始化矩阵
        self.connection_matrix = np.zeros((M, N))
        self.power_matrix = np.zeros((M, N))
        self.beta_matrix = np.zeros((M, N))
        self.gamma_matrix = np.zeros((M, N))
        self.cached_state = np.zeros(obs_dim)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state(self):
        return self.cached_state

    def _calculate_large_scale_fading(self):
        """计算大尺度衰落 (复用逻辑，使用本地参数)"""
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
        
        # 构造状态
        log_beta = np.log10(self.beta_matrix + 1e-20)
        norm_log_beta = (log_beta + 10.0) / 5.0
        norm_angle = theta_mk / 180.0
        
        beta_feat = norm_log_beta.T[:, :, np.newaxis]
        angle_feat = norm_angle.T[:, :, np.newaxis]
        link_features = np.concatenate([beta_feat, angle_feat], axis=2).reshape(N, -1)
        pos_features = self.uav_positions / np.array([X, Y, H])
        
        self.cached_state = np.hstack([link_features, pos_features]).flatten()

    def _calculate_capacity(self):
        """计算容量 (复用逻辑，使用本地参数)"""
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

    def _calculate_threshold_reward(self):
        """
        基于 Sigmoid 的平滑奖励计算。
        使用 Sigmoid 函数将容量映射到 [0, FIXED_REWARD] 区间。
        Sigmoid 中心设在 CAPACITY_THRESHOLD，斜率由 scale 控制。
        """
        _, total_capacity = self._calculate_capacity()
        
        # Sigmoid 参数
        # scale: 控制 Sigmoid 的陡峭程度。值越大越陡峭，越接近阶跃函数。
        # 经验值: scale=2.0 时，在 threshold +/- 2.0 范围内从 0.02 升至 0.98
        scale = 2.0 
        
        sigmoid_val = 1 / (1 + np.exp(-scale * (total_capacity - CAPACITY_THRESHOLD)))
        
        return FIXED_REWARD * sigmoid_val

    def _calculate_geometric_reward(self):
        """几何奖励 (用于预训练，使用本地参数)"""
        # ... (逻辑与原版相同，但使用本地 config)
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
            
        current_connection = self.connection_matrix
        reward = 0.0
        
        tp_count = np.sum((target_matrix == 1) & (current_connection == 1))
        reward += tp_count * GEO_REWARD_HIT
        
        fn_count = np.sum((target_matrix == 1) & (current_connection == 0))
        reward -= fn_count * GEO_PENALTY_MISS
        
        fp_mask = (target_matrix == 0) & (current_connection == 1)
        useless_mask = (self.beta_matrix < GEO_BETA_THRESHOLD)
        fp_useless_count = np.sum(fp_mask & useless_mask)
        fp_wrong_count = np.sum(fp_mask & (~useless_mask))
        
        reward -= fp_useless_count * GEO_PENALTY_USELESS
        reward -= fp_wrong_count * GEO_PENALTY_WRONG
        
        uav_connections = np.sum(current_connection, axis=0)
        no_connect_uavs = np.sum(uav_connections == 0)
        reward -= no_connect_uavs * GEO_PENALTY_NO_CONNECT
        
        if np.array_equal(target_matrix, current_connection):
            reward += GEO_BONUS_PERFECT
            
        return reward

    def step(self, action):
        assert not self._terminated, "Episode has terminated"

        power_actions = action.reshape(M, N)
        self.power_matrix = power_actions
        self.connection_matrix = (self.power_matrix > 0.001).astype(float)

        # 功率归一化
        total_power = np.sum(self.power_matrix, axis=1, keepdims=True)
        mask = total_power > 1.0
        if np.any(mask):
             self.power_matrix = np.where(mask, self.power_matrix / total_power, self.power_matrix)

        # 计算奖励
        if self.reward_mode == "geometric":
            reward = self._calculate_geometric_reward()
        else:
            # 默认为阈值奖励
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

        self.bs_positions = np.random.uniform(0, [X, Y], (M, 2))
        self.uav_positions = np.random.uniform(0, [X, Y, H/2], (N, 3))
        self.connection_matrix = np.zeros((M, N))
        self.power_matrix = np.zeros((M, N))
        
        self._calculate_large_scale_fading()

        return self.state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        np.random.seed(seed)

def make_threshold_env(training_num=0, test_num=0, reward_mode="threshold", top_p=0.6, norm_reward=False):
    """创建 ThresholdEnv 的辅助函数"""
    def _make_env():
        e = ThresholdEnv(reward_mode=reward_mode, top_p=top_p)
        if norm_reward:
            e = NormalizeObservation(e)
            # e = NormalizeReward(e) # Tianshou DDPG 不支持或不稳定，手动缩放奖励
        return e

    env = ThresholdEnv(reward_mode=reward_mode, top_p=top_p)
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv([_make_env for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv([_make_env for _ in range(test_num)])
        test_envs.seed(0)
        
    return env, train_envs, test_envs
