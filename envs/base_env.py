import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from .core.channel import ChannelModel

class BaseCellFreeEnv(gym.Env):
    """
    Cell-Free Environment Base Class.
    去蜂窝网络环境基类。
    
    Responsibilities (职责):
    1. Manage Physical State: BS/UAV positions, Channel Matrices (Beta, Gamma).
    2. Manage Episode Lifecycle: reset(), step(), termination.
    3. Core Physics: Apply Action (Power), Calculate Reward.
    
    Attributes (属性):
    - M (int): Number of Base Stations (基站数量).
    - N (int): Number of UAVs (用户数量).
    - channel_model (ChannelModel): Physics engine for channel fading (信道模型).
    - bs_positions (np.ndarray): (M, 2) coords.
    - uav_positions (np.ndarray): (N, 3) coords.
    - beta_matrix (np.ndarray): (M, N) Large-scale fading coefficients.
    - connection_matrix (np.ndarray): (M, N) Binary connectivity status.
    - power_matrix (np.ndarray): (M, N) Transmit power in Watts.
    """
    def __init__(self, config):
        """
        Initialize the environment.
        初始化环境。
        
        Args:
            config: EnvConfig object containing global parameters.
        """
        self.config = config
        self.M = config.M
        self.N = config.N
        self.X = config.X
        self.Y = config.Y
        self.H = config.H
        
        self.channel_model = ChannelModel(config)
        
        self._num_steps = 0
        self._terminated = False
        self._steps_per_episode = config.STEPS_PER_EPISODE
        
        # Initialize matrices (状态矩阵初始化)
        self.bs_positions = None
        self.uav_positions = None
        self.beta_matrix = np.zeros((self.M, self.N))
        self.gamma_matrix = np.zeros((self.M, self.N))
        self.theta_matrix = np.zeros((self.M, self.N))
        self.connection_matrix = np.zeros((self.M, self.N))
        self.power_matrix = np.zeros((self.M, self.N))
        
        # Define spaces (子类需实现或在此定义通用空间)
        self._observation_space = None
        self._action_space = None
        
        # Default Reward Mode (默认奖励计算模式)
        self.reward_mode = 'geometric'

    def set_reward_mode(self, mode):
        """
        Set the reward calculation mode.
        切换奖励计算模式。
        
        Args:
            mode: 'geometric' (基于几何匹配) or 'capacity' (基于信道容量).
        """
        if mode not in ['geometric', 'capacity']:
            raise ValueError(f"Unknown reward mode: {mode}")
        self.reward_mode = mode
        print(f"[Env] Reward Mode switched to: {self.reward_mode}")

    @property
    def observation_space(self):
        """Get the observation space."""
        return self._observation_space

    @property
    def action_space(self):
        """Get the action space."""
        return self._action_space

    def _init_bs_positions(self):
        """
        Initialize BS positions. Can be overridden.
        初始化基站位置。默认逻辑：若M=16则生成4x4网格，否则随机分布。
        """
        if self.M == 16:
            # Create a 4x4 grid (均匀分布网格)
            x_coords = np.linspace(self.X/8, 7*self.X/8, 4)
            y_coords = np.linspace(self.Y/8, 7*self.Y/8, 4)
            xv, yv = np.meshgrid(x_coords, y_coords)
            self.bs_positions = np.column_stack((xv.flatten(), yv.flatten()))
        else:
            # Random distribution (随机分布)
            self.bs_positions = np.random.uniform(0, [self.X, self.Y], (self.M, 2))

    def reset(self, seed=None, options=None):
        """
        Reset the environment state.
        重置环境状态：生成新位置，重新计算信道增益。
        """
        super().reset(seed=seed)
        self._num_steps = 0
        self._terminated = False
        
        self._init_bs_positions()
        self.uav_positions = np.random.uniform(0, [self.X, self.Y, self.H], (self.N, 3))
        
        # Initial channel calculation (计算初始信道参数)
        self.beta_matrix, self.gamma_matrix, self.theta_matrix = \
            self.channel_model.calculate_large_scale_fading(self.bs_positions, self.uav_positions)
            
        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one physics step.
        执行一步物理仿真。
        
        Flow:
        1. Apply Action -> Update Power/Connection.
        2. Calculate Reward -> Based on mode.
        3. Get Observation -> Next state.
        
        Args:
            action: Action from Policy/Wrapper (normalized).
        """
        self._num_steps += 1
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True
            
        # 1. Apply action (应用动作)
        self._apply_action(action)
        
        # 2. Calculate Reward (计算奖励)
        if self.reward_mode == 'geometric':
            reward = self._calculate_geometric_reward()
        elif self.reward_mode == 'capacity':
            reward = self._calculate_capacity_reward()
        else:
            reward = 0.0
        
        # 3. Get Observation (获取观测)
        obs = self._get_obs()
        
        return obs, reward, self._terminated, False, {}

    def _get_obs(self):
        """
        Abstract method to get observation.
        抽象方法：子类需实现具体的观测向量构造。
        """
        raise NotImplementedError

    def _apply_action(self, action):
        """
        Apply the physical action (Power Matrix).
        将归一化的动作转换为物理功率并更新连接状态。
        
        Args:
            action: (M, N) Normalized Power Matrix [0, 1].
        """
        # Scale to physical power (恢复物理单位: Watts)
        self.power_matrix = action * self.config.P
        
        # Update connection matrix based on power > 0 (or small epsilon)
        # 更新连接矩阵：功率大于0即视为连接
        self.connection_matrix = (self.power_matrix > 1e-9).astype(float)

    def _calculate_capacity_reward(self):
        """
        Calculate reward based on Channel Capacity (Sum Rate).
        计算基于信道容量的奖励。
        
        Formula: R = Sum(Capacity) or Threshold Count based on config.
        """
        capacity, _ = self.channel_model.calculate_capacity(
            self.power_matrix, 
            self.beta_matrix, 
            self.gamma_matrix, 
            self.config.pd, 
            self.config.NOISE_POWER
        )
        
        if self.config.CAPACITY_REWARD_TYPE == 'threshold_fixed':
            # Count satisfied users (满足速率阈值的用户数)
            satisfied_uavs = np.sum(capacity >= self.config.CAPACITY_THRESHOLD)
            reward = satisfied_uavs * self.config.FIXED_REWARD
        elif self.config.CAPACITY_REWARD_TYPE == 'sum_capacity':
            # Sum Rate (总和速率)
            reward = np.sum(capacity)
        else:
            reward = 0.0
        
        return reward

    def _calculate_geometric_reward(self):
        """
        Geometric reward function based on Top-P matching.
        基于 Top-P 几何匹配的奖励函数。
        
        Logic (逻辑):
        1. Calculate Ideal Top-P Set: Sort Beta -> CumSum -> Cutoff.
        2. Compare: Compare current connection_matrix with Ideal Set.
        3. Penalty/Bonus: Apply rewards for TP, FN, FP (Useless/Wrong).
        """
        # Sort Beta matrix (M, N) along axis 0 (BS dimension) descending
        # 对每个用户，按信道质量(Beta)对基站排序
        sorted_indices = np.argsort(self.beta_matrix, axis=0)[::-1]
        sorted_betas = np.take_along_axis(self.beta_matrix, sorted_indices, axis=0)
        
        # Calculate Cumulative Sum (计算累积能量分布)
        cumsum_betas = np.cumsum(sorted_betas, axis=0)
        total_beta = cumsum_betas[-1, :] # (N,) Total Beta per UAV
        
        # Calculate Top-P Threshold (计算能量截断阈值)
        threshold_values = total_beta * self.config.TOP_P_THRESHOLD
        
        # Find cutoff indices (找到截断位置)
        mask_cumsum = cumsum_betas >= threshold_values[None, :]
        cutoff_indices = np.argmax(mask_cumsum, axis=0) # (N,) First index exceeding threshold
        
        # Absolute threshold filtering (过滤掉极其微弱的信道)
        valid_counts = np.sum(sorted_betas >= self.config.GEO_BETA_THRESHOLD, axis=0) # (N,)
        
        # Final cutoff: min(Top-P cutoff, valid count - 1)
        # 截断点取 Top-P 与 有效信道数 的交集
        rule_based_cutoffs = np.minimum(cutoff_indices, valid_counts - 1)
        
        # Force at least Top 1 (保证至少连接一个基站)
        final_cutoffs = np.maximum(rule_based_cutoffs, 0)
        
        # Build Target Matrix (构建理想连接矩阵)
        target_matrix = np.zeros((self.M, self.N))
        row_indices = np.arange(self.M)[:, None]
        selection_mask = (row_indices <= final_cutoffs[None, :])
        
        target_bs_indices = sorted_indices[selection_mask]
        col_indices = np.tile(np.arange(self.N), (self.M, 1))
        target_uav_indices = col_indices[selection_mask]
        
        target_matrix[target_bs_indices, target_uav_indices] = 1.0
            
        # Calculate Reward (计算各种匹配奖励)
        current_connection = self.connection_matrix
        reward = 0.0
        
        # True Positive (正确连接): 奖励
        tp_count = np.sum((target_matrix == 1) & (current_connection == 1))
        reward += tp_count * self.config.GEO_REWARD_HIT
        
        # False Negative (漏连接): 惩罚
        fn_count = np.sum((target_matrix == 1) & (current_connection == 0))
        reward -= fn_count * self.config.GEO_PENALTY_MISS
        
        # False Positive (错误连接)
        fp_mask = (target_matrix == 0) & (current_connection == 1)
        
        # Useless (Beta < Threshold) (连接了完全无用的弱信道): 重罚
        useless_mask = (self.beta_matrix < self.config.GEO_BETA_THRESHOLD)
        fp_useless_count = np.sum(fp_mask & useless_mask)
        
        # Wrong (Beta >= Threshold but not in Top-P) (连接了次优信道): 轻罚
        fp_wrong_count = np.sum(fp_mask & (~useless_mask))
        
        reward -= fp_useless_count * self.config.GEO_PENALTY_USELESS
        reward -= fp_wrong_count * self.config.GEO_PENALTY_WRONG
        
        # No Connect Penalty (断连惩罚)
        uav_connections = np.sum(current_connection, axis=0)
        no_connect_uavs = np.sum(uav_connections == 0)
        reward -= no_connect_uavs * self.config.GEO_PENALTY_NO_CONNECT
        
        # Perfect Bonus (完美匹配奖励)
        if np.array_equal(target_matrix, current_connection):
            reward += self.config.GEO_BONUS_PERFECT
            
        return reward
