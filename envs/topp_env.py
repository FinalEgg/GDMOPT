from gymnasium.spaces import Box
import numpy as np
from .base_env import BaseCellFreeEnv

class TopPEnv(BaseCellFreeEnv):
    def __init__(self, config):
        super().__init__(config)
        
        # Observation Space: N * (M * 2 + 3)
        self.uav_feature_dim = self.M * 2 + 3
        obs_dim = self.N * self.uav_feature_dim
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

        # Action Space: Power Matrix (M * N)
        action_dim = self.M * self.N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))
        
        self.cached_state = np.zeros(obs_dim)

    def _get_obs(self):
        # Same as CellFreeEnv
        log_beta = np.log10(self.beta_matrix + 1e-20)
        norm_log_beta = (log_beta + 10.0) / 5.0
        norm_angle = self.theta_matrix / 180.0
        norm_uav_pos = self.uav_positions / [self.X, self.Y, self.H]
        
        obs_list = []
        for n in range(self.N):
            obs_list.append(norm_log_beta[:, n])
            obs_list.append(norm_angle[:, n])
            obs_list.append(norm_uav_pos[n])
            
        self.cached_state = np.concatenate(obs_list)
        return self.cached_state

    def _apply_action(self, action):
        # 1. Calculate Top-P Mask
        sorted_indices = np.argsort(self.beta_matrix, axis=0)[::-1]
        sorted_betas = np.take_along_axis(self.beta_matrix, sorted_indices, axis=0)
        cumsum_betas = np.cumsum(sorted_betas, axis=0)
        total_beta = cumsum_betas[-1, :]
        threshold_values = total_beta * self.config.TOP_P_THRESHOLD
        
        mask_cumsum = cumsum_betas >= threshold_values[None, :]
        cutoff_indices = np.argmax(mask_cumsum, axis=0)
        
        # Build Mask
        top_p_mask = np.zeros((self.M, self.N))
        for n in range(self.N):
            # Get indices of BSs in Top-P
            # The cutoff index is the first one that exceeds threshold, so we include it.
            # We need the original indices corresponding to 0..cutoff
            cutoff = cutoff_indices[n]
            valid_bs_indices = sorted_indices[:cutoff+1, n]
            top_p_mask[valid_bs_indices, n] = 1.0
            
        # 2. Apply Action (Power) masked by Top-P
        raw_power = action.reshape(self.M, self.N) * self.config.P
        self.power_matrix = raw_power * top_p_mask
        self.connection_matrix = (self.power_matrix > 1e-3).astype(float)
