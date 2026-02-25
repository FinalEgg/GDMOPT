from gymnasium.spaces import Box, Dict
import numpy as np
from .base_env import BaseCellFreeEnv

class CellFreeEnv(BaseCellFreeEnv):
    """
    Standard Cell-Free Massive MIMO Environment.
    标准去蜂窝大规模 MIMO 环境。
    
    Features (特性):
    1. Dynamic Topology: Random BS and UAV positions per episode.
    2. Continuous Action: Agent directly controls power (processed by wrapper).
    3. Dict Observation: Structured state for modular processing.
    """
    def __init__(self, config):
        """
        Initialize standard environment.
        初始化标准环境。
        """
        super().__init__(config)
        
        # Observation Space: Dict (字典形式观测空间)
        # Decoupled from Model Input (与模型输入解耦)
        self._observation_space = Dict({
            'log_beta': Box(low=-np.inf, high=np.inf, shape=(self.M, self.N)),
            'angle': Box(low=-np.inf, high=np.inf, shape=(self.M, self.N)),
            'uav_pos': Box(low=-np.inf, high=np.inf, shape=(self.N, 3))
        })

        # Action Space: Physical Power Matrix (M * N) (物理功率矩阵)
        # Decoupled from Model Output (Wrappers handle conversion)
        # Flattened for Gym compatibility
        action_dim = self.M * self.N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))
        
    def _get_obs(self):
        """
        Get structured observation.
        获取结构化观测数据。
        
        Returns:
            dict: {log_beta, angle, uav_pos} (normalized)
        """
        # LogBeta (对数大尺度衰落)
        log_beta = np.log10(self.beta_matrix + 1e-20)
        norm_log_beta = (log_beta + 10.0) / 5.0 # Approx range [-1, 1]
        
        # Angle (角度归一化)
        norm_angle = self.theta_matrix / 180.0
        
        # UAV Pos (位置归一化)
        norm_uav_pos = self.uav_positions / [self.X, self.Y, self.H]
        
        return {
            'log_beta': norm_log_beta,
            'angle': norm_angle,
            'uav_pos': norm_uav_pos
        }

    # Note: _apply_action is inherited from BaseCellFreeEnv (Default physical mapping)
    # The Wrappers (Threshold/Hybrid) will convert Model Action -> Power Matrix

