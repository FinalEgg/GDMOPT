import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from core.channel import ChannelModel

class BaseCellFreeEnv(gym.Env):
    def __init__(self, config):
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
        
        # Initialize matrices
        self.bs_positions = None
        self.uav_positions = None
        self.beta_matrix = np.zeros((self.M, self.N))
        self.gamma_matrix = np.zeros((self.M, self.N))
        self.theta_matrix = np.zeros((self.M, self.N))
        self.connection_matrix = np.zeros((self.M, self.N))
        self.power_matrix = np.zeros((self.M, self.N))
        
        # Define spaces (to be implemented by subclasses or set here if common)
        self._observation_space = None
        self._action_space = None
        
        # Default Reward Mode
        self.reward_mode = 'geometric'

    def set_reward_mode(self, mode):
        """
        Set the reward calculation mode.
        Args:
            mode: 'geometric' or 'capacity'
        """
        if mode not in ['geometric', 'capacity']:
            raise ValueError(f"Unknown reward mode: {mode}")
        self.reward_mode = mode
        print(f"[Env] Reward Mode switched to: {self.reward_mode}")

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _init_bs_positions(self):
        """Initialize BS positions. Can be overridden."""
        if self.M == 16:
            # Create a 4x4 grid
            x_coords = np.linspace(self.X/8, 7*self.X/8, 4)
            y_coords = np.linspace(self.Y/8, 7*self.Y/8, 4)
            xv, yv = np.meshgrid(x_coords, y_coords)
            self.bs_positions = np.column_stack((xv.flatten(), yv.flatten()))
        else:
            self.bs_positions = np.random.uniform(0, [self.X, self.Y], (self.M, 2))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._num_steps = 0
        self._terminated = False
        
        self._init_bs_positions()
        self.uav_positions = np.random.uniform(0, [self.X, self.Y, self.H], (self.N, 3))
        
        # Initial channel calculation
        self.beta_matrix, self.gamma_matrix, self.theta_matrix = \
            self.channel_model.calculate_large_scale_fading(self.bs_positions, self.uav_positions)
            
        return self._get_obs(), {}

    def step(self, action):
        self._num_steps += 1
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True
            
        # 1. Apply action (update power_matrix and connection_matrix)
        # Action is expected to be Normalized Power Matrix (M, N) from Wrapper
        self._apply_action(action)
        
        # 2. Calculate Reward
        if self.reward_mode == 'geometric':
            reward = self._calculate_geometric_reward()
        elif self.reward_mode == 'capacity':
            reward = self._calculate_capacity_reward()
        else:
            reward = 0.0
        
        # 3. Get Observation
        obs = self._get_obs()
        
        return obs, reward, self._terminated, False, {}

    def _get_obs(self):
        raise NotImplementedError

    def _apply_action(self, action):
        """
        Apply the physical action (Power Matrix).
        Args:
            action: (M, N) Normalized Power Matrix [0, 1]
        """
        # Scale to physical power
        self.power_matrix = action * self.config.P
        
        # Update connection matrix based on power > 0 (or small epsilon)
        # The Wrapper handles the logic of "Thresholding" or "Hybrid" decision
        # Here we just reflect the physics: Power > 0 means connected.
        self.connection_matrix = (self.power_matrix > 1e-9).astype(float)

    def _calculate_capacity_reward(self):
        """
        Calculate reward based on Channel Capacity.
        """
        capacity, _ = self.channel_model.calculate_capacity(
            self.power_matrix, 
            self.beta_matrix, 
            self.gamma_matrix, 
            self.config.pd, 
            self.config.NOISE_POWER
        )
        
        if self.config.CAPACITY_REWARD_TYPE == 'threshold_fixed':
            # capacity is (N,)
            satisfied_uavs = np.sum(capacity >= self.config.CAPACITY_THRESHOLD)
            reward = satisfied_uavs * self.config.FIXED_REWARD
        elif self.config.CAPACITY_REWARD_TYPE == 'sum_capacity':
            reward = np.sum(capacity)
        else:
            reward = 0.0
        
        return reward

    def _calculate_geometric_reward(self):
        """
        Geometric reward function based on Top-P matching.
        """
        # Sort Beta matrix (M, N) along axis 0 (BS dimension) descending
        sorted_indices = np.argsort(self.beta_matrix, axis=0)[::-1]
        sorted_betas = np.take_along_axis(self.beta_matrix, sorted_indices, axis=0)
        
        # Calculate Cumulative Sum
        cumsum_betas = np.cumsum(sorted_betas, axis=0)
        total_beta = cumsum_betas[-1, :] # (N,) Total Beta per UAV
        
        # Calculate Top-P Threshold
        threshold_values = total_beta * self.config.TOP_P_THRESHOLD
        
        # Find cutoff indices
        mask_cumsum = cumsum_betas >= threshold_values[None, :]
        cutoff_indices = np.argmax(mask_cumsum, axis=0) # (N,)
        
        # Absolute threshold filtering
        valid_counts = np.sum(sorted_betas >= self.config.GEO_BETA_THRESHOLD, axis=0) # (N,)
        
        # Final cutoff: min(Top-P cutoff, valid count - 1)
        rule_based_cutoffs = np.minimum(cutoff_indices, valid_counts - 1)
        
        # Force at least Top 1
        final_cutoffs = np.maximum(rule_based_cutoffs, 0)
        
        # Build Target Matrix
        target_matrix = np.zeros((self.M, self.N))
        row_indices = np.arange(self.M)[:, None]
        selection_mask = (row_indices <= final_cutoffs[None, :])
        
        target_bs_indices = sorted_indices[selection_mask]
        col_indices = np.tile(np.arange(self.N), (self.M, 1))
        target_uav_indices = col_indices[selection_mask]
        
        target_matrix[target_bs_indices, target_uav_indices] = 1.0
            
        # Calculate Reward
        current_connection = self.connection_matrix
        reward = 0.0
        
        # True Positive
        tp_count = np.sum((target_matrix == 1) & (current_connection == 1))
        reward += tp_count * self.config.GEO_REWARD_HIT
        
        # False Negative
        fn_count = np.sum((target_matrix == 1) & (current_connection == 0))
        reward -= fn_count * self.config.GEO_PENALTY_MISS
        
        # False Positive
        fp_mask = (target_matrix == 0) & (current_connection == 1)
        
        # Useless (Beta < Threshold)
        useless_mask = (self.beta_matrix < self.config.GEO_BETA_THRESHOLD)
        fp_useless_count = np.sum(fp_mask & useless_mask)
        
        # Wrong (Beta >= Threshold but not in Top-P)
        fp_wrong_count = np.sum(fp_mask & (~useless_mask))
        
        reward -= fp_useless_count * self.config.GEO_PENALTY_USELESS
        reward -= fp_wrong_count * self.config.GEO_PENALTY_WRONG
        
        # No Connect Penalty
        uav_connections = np.sum(current_connection, axis=0)
        no_connect_uavs = np.sum(uav_connections == 0)
        reward -= no_connect_uavs * self.config.GEO_PENALTY_NO_CONNECT
        
        # Perfect Bonus
        if np.array_equal(target_matrix, current_connection):
            reward += self.config.GEO_BONUS_PERFECT
            
        return reward
