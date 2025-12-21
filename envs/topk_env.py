from gymnasium.spaces import Box
import numpy as np
from .base_env import BaseCellFreeEnv

class TopKEnv(BaseCellFreeEnv):
    def __init__(self, config):
        super().__init__(config)
        self.K = config.K_MAX
        
        # Observation Space: N * (3 + K * 3)
        # Per UAV: [Pos (3), Top-K BS Info (K * 3: ID, Beta, Angle)]
        self.state_per_uav = 3 + self.K * 3
        obs_dim = self.N * self.state_per_uav
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

        # Action Space: N * K (Power for Top-K BSs)
        action_dim = self.N * self.K
        self._action_space = Box(low=0, high=1, shape=(action_dim,))
        
        self.connection_indices = np.full((self.N, self.K), -1, dtype=int)
        self.cached_state = np.zeros(obs_dim)

    def _get_obs(self):
        # 1. Identify Top-K BSs for each UAV based on Beta
        # Sort indices descending
        sorted_indices = np.argsort(self.beta_matrix, axis=0)[::-1] # (M, N)
        
        obs_list = []
        for n in range(self.N):
            # UAV Position
            uav_pos = self.uav_positions[n] / [self.X, self.Y, self.H]
            obs_list.append(uav_pos)
            
            # Top-K BSs
            top_k_indices = sorted_indices[:self.K, n]
            self.connection_indices[n] = top_k_indices # Store for action mapping
            
            for k in range(self.K):
                bs_idx = top_k_indices[k]
                beta = self.beta_matrix[bs_idx, n]
                angle = self.theta_matrix[bs_idx, n]
                
                # Normalize
                norm_bs_id = bs_idx / self.M
                norm_beta = (np.log10(beta + 1e-20) + 10.0) / 5.0
                norm_angle = angle / 180.0
                
                obs_list.append([norm_bs_id, norm_beta, norm_angle])
                
        self.cached_state = np.concatenate(obs_list)
        return self.cached_state

    def _apply_action(self, action):
        # Action: (N * K)
        reshaped_action = action.reshape(self.N, self.K)
        
        self.power_matrix = np.zeros((self.M, self.N))
        self.connection_matrix = np.zeros((self.M, self.N))
        
        for n in range(self.N):
            for k in range(self.K):
                bs_idx = self.connection_indices[n, k]
                if bs_idx != -1:
                    power_val = reshaped_action[n, k]
                    self.power_matrix[bs_idx, n] = power_val * self.config.P
                    if power_val > 1e-3:
                        self.connection_matrix[bs_idx, n] = 1.0
