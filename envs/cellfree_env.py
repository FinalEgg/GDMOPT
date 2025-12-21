from gymnasium.spaces import Box
import numpy as np
from .base_env import BaseCellFreeEnv

from gymnasium.spaces import Box, Dict
import numpy as np
from .base_env import BaseCellFreeEnv

class CellFreeEnv(BaseCellFreeEnv):
    def __init__(self, config):
        super().__init__(config)
        
        # Observation Space: Dict
        # Decoupled from Model Input
        self._observation_space = Dict({
            'log_beta': Box(low=-np.inf, high=np.inf, shape=(self.M, self.N)),
            'angle': Box(low=-np.inf, high=np.inf, shape=(self.M, self.N)),
            'uav_pos': Box(low=-np.inf, high=np.inf, shape=(self.N, 3))
        })

        # Action Space: Physical Power Matrix (M * N)
        # Decoupled from Model Output (Wrappers handle conversion)
        action_dim = self.M * self.N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))
        
    def _get_obs(self):
        # Return Raw/Semi-processed Dict
        
        # LogBeta
        log_beta = np.log10(self.beta_matrix + 1e-20)
        norm_log_beta = (log_beta + 10.0) / 5.0
        
        # Angle (Normalized)
        norm_angle = self.theta_matrix / 180.0
        
        # UAV Pos (Normalized)
        norm_uav_pos = self.uav_positions / [self.X, self.Y, self.H]
        
        return {
            'log_beta': norm_log_beta,
            'angle': norm_angle,
            'uav_pos': norm_uav_pos
        }

    # _apply_action is now handled in BaseCellFreeEnv (Physical Layer)
    # The Wrappers (Threshold/Hybrid) will convert Model Action -> Power Matrix

