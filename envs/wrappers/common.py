import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Observation Flattener Wrapper.
    观测展平包装器。
    
    Function (功能):
    - Flattens the Dict observation into a single vector (将字典观测展平为向量).
    - Augments data with Sin/Cos of angles and BS positions (增强特征: 角度的正余弦, 基站位置).
    
    Target Structure (目标结构):
    - For each UAV (每个用户):
      [BS_1_Features (5), ..., BS_M_Features (5), UAV_Pos (3)]
    - BS_Features: [LogBeta, Sin(Theta), Cos(Theta), BS_X, BS_Y]
    
    Designed for MLP/DeepSets models (适用于 MLP 或 DeepSets 模型).
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.M = env.unwrapped.M
        self.N = env.unwrapped.N
        
        # Feature Dimension Definition (特征维度定义)
        # BS Features: LogBeta(1) + Sin(1) + Cos(1) + Pos(2) = 5
        # UAV Features: Pos(3)
        self.uav_feature_dim = self.M * 5 + 3
        self.obs_dim = self.N * self.uav_feature_dim
        
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def observation(self, obs):
        """
        Process observation.
        处理观测数据。
        
        Args:
            obs: Dict observation from Env.
        
        Returns:
            np.ndarray: Flattened vector.
        """
        # obs is a Dict
        log_beta = obs['log_beta'].T # Transpose to (N, M) for UAV-centric view
        angle = obs['angle'].T       # (N, M)
        uav_pos = obs['uav_pos']     # (N, 3)
        
        # Convert Angle to Sin/Cos (角度特征增强)
        # Assuming obs['angle'] is normalized [0, 1] mapped from [0, 180] degrees
        theta_rad = angle * np.pi # [0, 1] -> [0, pi]
        sin_angle = np.sin(theta_rad)
        cos_angle = np.cos(theta_rad)
        
        # BS Positions (Get from Env) (获取基站位置)
        env = self.env.unwrapped
        # Normalize BS positions
        bs_pos_norm = env.bs_positions / [env.X, env.Y] # (M, 2)
        
        # Construct Feature Vector (构建特征向量)
        obs_list = []
        for n in range(self.N):
            # Per BS features (Relating to this UAV)
            for m in range(self.M):
                obs_list.extend([
                    log_beta[n, m],
                    sin_angle[n, m],
                    cos_angle[n, m],
                    bs_pos_norm[m, 0],
                    bs_pos_norm[m, 1]
                ])
            # Self features
            obs_list.extend(uav_pos[n])
            
        return np.array(obs_list, dtype=np.float32)

class ThresholdActionWrapper(gym.ActionWrapper):
    """
    Threshold-based Discrete/Continuous Action Wrapper.
    基于阈值的离散/连续动作包装器。
    
    Logic (逻辑):
    - Input: Continuous vector (M*N).
    - Processing: Apply threshold. If value < threshold, set to 0.
    - Output: Sparse Power Matrix (M, N).
    """
    def __init__(self, env, threshold=0.2):
        super().__init__(env)
        self.threshold = threshold
        # Action space is M*N [0, 1]
        self.M = env.unwrapped.M
        self.N = env.unwrapped.N
        self.action_space = Box(low=0, high=1, shape=(self.M * self.N,), dtype=np.float32)

    def action(self, action):
        """
        Apply thresholding to action.
        应用阈值过滤。
        """
        power_matrix = action.reshape(self.M, self.N)
        
        # Hard Thresholding (硬阈值)
        mask = (power_matrix > self.threshold).astype(float)
        masked_power = power_matrix * mask
        
        return masked_power

class PurePowerActionWrapper(gym.ActionWrapper):
    """
    Continuous Power Control Wrapper.
    连续功率控制包装器。
    
    Logic (逻辑):
    - Input: Continuous vector [-1, 1] (Standard RL API).
    - Processing: Map to [0, 1] linearly.
    - Output: Normalized Power Matrix (M, N).
    """
    def __init__(self, env):
        super().__init__(env)
        self.M = env.unwrapped.M
        self.N = env.unwrapped.N
        # Change to [-1, 1] for RL compatibility (RL 算法通常输出 [-1, 1])
        self.action_space = Box(low=-1, high=1, shape=(self.M * self.N,), dtype=np.float32)

    def action(self, action):
        """
        Process action.
        处理动作。
        """
        # Map [-1, 1] to [0, 1]
        action = (action + 1.0) / 2.0
        # Clip to ensure numerical stability
        action = np.clip(action, 0.0, 1.0)
        
        return action.reshape(self.M, self.N)

class HybridActionWrapper(gym.ActionWrapper):
    """
    Hybrid Action Wrapper (Power + Connectivity).
    混合动作包装器 (功率 + 连接开关).
    
    Logic (逻辑):
    - Input: Vector of size 2 * M * N.
      - Part 1: Power levels.
      - Part 2: Connection probabilities/logits.
    - Processing: If Connection > 0.5, apply Power. Else 0.
    """
    def __init__(self, env):
        super().__init__(env)
        self.M = env.unwrapped.M
        self.N = env.unwrapped.N
        # Action dim = Power + Connectivity
        self.action_space = Box(low=0, high=1, shape=(2 * self.M * self.N,), dtype=np.float32)

    def action(self, action):
        """
        Parse hybrid action.
        解析混合动作。
        """
        split_idx = self.M * self.N
        power_raw = action[:split_idx].reshape(self.M, self.N)
        conn_raw = action[split_idx:].reshape(self.M, self.N)
        
        # Connection Decision (连接决策)
        mask = (conn_raw > 0.5).astype(float)
        
        # Apply Mask
        masked_power = power_raw * mask
        
        return masked_power
