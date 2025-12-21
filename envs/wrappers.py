import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the Dict observation into a single vector.
    Compatible with MLP/DeepSets models.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Compute flattened dimension
        # We assume the dict contains 'log_beta', 'angle', 'uav_pos'
        # Structure: For each UAV, concat [LogBeta (M), Angle (M), Pos (3)]
        
        self.M = env.unwrapped.M
        self.N = env.unwrapped.N
        
        # Structure: For each UAV, concat [LogBeta (M), Sin (M), Cos (M), BS_X (M), BS_Y (M), Pos (3)]
        # Total features per BS = 5
        self.uav_feature_dim = self.M * 5 + 3
        self.obs_dim = self.N * self.uav_feature_dim
        
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))

    def observation(self, obs):
        # obs is a Dict
        log_beta = obs['log_beta'].T # (N, M)
        angle = obs['angle'].T       # (N, M)
        uav_pos = obs['uav_pos']     # (N, 3)
        
        # Normalize Angle (Assuming obs['angle'] is normalized 0-1 or raw degrees?)
        # In FixTopPEnv._get_obs: norm_angle = self.theta_matrix / 180.0
        # So it is normalized [0, 1].
        # We need Sin/Cos for better representation.
        theta = angle * 180.0 * np.pi / 180.0
        sin_angle = np.sin(theta)
        cos_angle = np.cos(theta)
        
        # BS Positions (Need access to Env)
        # Assuming Env has bs_positions and X, Y
        env = self.env.unwrapped
        bs_pos = env.bs_positions / [env.X, env.Y] # (M, 2)
        
        # Structure: Per UAV -> [BS_1_Feats, ..., BS_M_Feats, UAV_Self_Feats]
        # BS_Feats: [log_beta, sin, cos, bs_x, bs_y] (5 dims)
        # UAV_Feats: [x, y, z] (3 dims)
        
        obs_list = []
        for n in range(self.N):
            # Per BS features
            for m in range(self.M):
                obs_list.extend([
                    log_beta[n, m],
                    sin_angle[n, m],
                    cos_angle[n, m],
                    bs_pos[m, 0],
                    bs_pos[m, 1]
                ])
            # Self features
            obs_list.extend(uav_pos[n])
            
        return np.array(obs_list, dtype=np.float32)

class ThresholdActionWrapper(gym.ActionWrapper):
    """
    Interprets action as Power Matrix.
    Applies thresholding: if power < 0.2 (scaled), connection is OFF.
    """
    def __init__(self, env, threshold=0.2):
        super().__init__(env)
        self.threshold = threshold
        # Action space is M*N (0-1)
        self.M = env.unwrapped.M
        self.N = env.unwrapped.N
        self.action_space = Box(low=0, high=1, shape=(self.M * self.N,))

    def action(self, action):
        # action: (M*N,) -> (M, N)
        power_matrix = action.reshape(self.M, self.N)
        
        mask = (power_matrix > self.threshold).astype(float)
        masked_power = power_matrix * mask
        
        return masked_power

class PurePowerActionWrapper(gym.ActionWrapper):
    """
    Interprets action strictly as Power Matrix without thresholding.
    Output is continuous [0, 1].
    """
    def __init__(self, env):
        super().__init__(env)
        self.M = env.unwrapped.M
        self.N = env.unwrapped.N
        # Change to [-1, 1] for RL compatibility
        self.action_space = Box(low=-1, high=1, shape=(self.M * self.N,))

    def action(self, action):
        # action: (M*N,) -> (M, N)
        # Map [-1, 1] to [0, 1]
        action = (action + 1.0) / 2.0
        # Clip to ensure [0, 1]
        action = np.clip(action, 0.0, 1.0)
        
        # Directly return the reshaped power matrix
        # The environment will handle physical scaling ( * P )
        return action.reshape(self.M, self.N)

class HybridActionWrapper(gym.ActionWrapper):
    """
    Action: [Power (M*N), Connection (M*N)]
    """
    def __init__(self, env):
        super().__init__(env)
        self.M = env.unwrapped.M
        self.N = env.unwrapped.N
        self.action_space = Box(low=0, high=1, shape=(2 * self.M * self.N,))

    def action(self, action):
        split_idx = self.M * self.N
        power_raw = action[:split_idx].reshape(self.M, self.N)
        conn_raw = action[split_idx:].reshape(self.M, self.N)
        
        mask = (conn_raw > 0.5).astype(float)
        masked_power = power_raw * mask
        
        return masked_power
