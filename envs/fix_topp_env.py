
from gymnasium.spaces import Box, Dict
import numpy as np
from .base_env import BaseCellFreeEnv

class FixTopPEnv(BaseCellFreeEnv):
    """
    Fixed Position Top-P Environment.
    固定基站位置 + Top-P 约束环境。
    
    Logic (逻辑):
    1. Fixed Topology: BS positions are hardcoded (optimized scenario).
    2. 3-Layer UAVs: UAVs are generated in 3 altitude layers (100m, 200m, 300m).
    3. Action Constraint: 
       - Phase 1: Determine candidate connections using Top-P logic (Geometrically filtered).
       - Phase 2: Agent allocates power only to candidates.
       - Phase 3: Enforce Total Power Constraint per BS.
    
    State (状态):
    - Dict Observation: Log-Beta, Arrival Angle, UAV Positions.
    
    Reward (奖励):
    - Delta Capacity: Improvement over Equal Power Allocation baseline.
    """
    def __init__(self, config):
        """
        Initialize the environment.
        初始化环境配置。
        
        Args:
            config: EnvConfig object.
        """
        # Override config parameters for this specific environment if needed
        # But usually config is passed in. We assume config is already updated or we force values here.
        # For safety, we enforce the fixed BS positions here.
        super().__init__(config)
        
        # Fixed BS Positions (X, Y) (固定基站坐标)
        self.fixed_bs_positions = np.array([
            [7054.3, 6976.0],
            [8574.8, 4873.3],
            [4719.4, 8318.9],
            [5978.0, 9178.2],
            [6922.8, 3596.6],
            [991.98, 7083.8],
            [2021.3, 9422.1],
            [1887.7, 2918.1],
            [943.49, 2155.0],
            [3040.0, 4974.7]
        ])
        
        # Ensure M matches (确保配置与硬编码一致)
        if self.M != 10:
            print(f"[Warning] Config M={self.M} does not match Fixed BS count 10. Forcing M=10.")
            self.M = 10
            # Re-init matrices dependent on M
            self.beta_matrix = np.zeros((self.M, self.N))
            self.gamma_matrix = np.zeros((self.M, self.N))
            self.theta_matrix = np.zeros((self.M, self.N))
            self.connection_matrix = np.zeros((self.M, self.N))
            self.power_matrix = np.zeros((self.M, self.N))

        # Action Space: Power Matrix (M * N)
        # Flattened for compatibility with standard RL algorithms
        action_dim = self.M * self.N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))
        
        # Observation Space: Dict
        # Decoupled from Model Input (Separate features for separate processing)
        self._observation_space = Dict({
            'log_beta': Box(low=-np.inf, high=np.inf, shape=(self.M, self.N)),
            'angle': Box(low=-np.inf, high=np.inf, shape=(self.M, self.N)),
            'uav_pos': Box(low=-np.inf, high=np.inf, shape=(self.N, 3))
        })
        
        self.cached_state = None # Not used in Dict mode
        
        # Set Reward Mode to Capacity (Default for optimization)
        self.set_reward_mode('capacity')
        
        # K_max for connection limit (每个用户最大连接基站数限制)
        self.k_max = 5

    def _init_bs_positions(self):
        """
        Override to use fixed positions.
        重写：加载固定基站坐标。
        """
        self.bs_positions = self.fixed_bs_positions.copy()

    def reset(self, seed=None, options=None):
        """
        Reset environment with 3-Layer UAV distribution.
        重置环境，生成分层分布的无人机。
        
        Logic:
        1. Layers: 100m, 200m, 300m.
        2. Assign each UAV to a random layer.
        3. Random X, Y coordinates.
        """
        # Standard reset for other things
        super().reset(seed=seed)
        
        # Re-generate UAV positions based on 3-layer structure
        # Layer 1: 100m, Layer 2: 200m, Layer 3: 300m
        heights = [100.0, 200.0, 300.0]
        
        # Randomly assign each UAV to a layer (随机分配高度层)
        uav_heights = np.random.choice(heights, size=self.N)
        
        # X, Y are random in the area (0 to 10000)
        # Random X, Y for generalization.
        uav_xy = np.random.uniform(0, [self.X, self.Y], (self.N, 2))
        
        self.uav_positions = np.column_stack((uav_xy, uav_heights))
        
        # Recalculate channels with new positions (重新计算信道)
        self.beta_matrix, self.gamma_matrix, self.theta_matrix = \
            self.channel_model.calculate_large_scale_fading(self.bs_positions, self.uav_positions)
            
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Construct Dictionary Observation.
        构造字典形式的观测状态。
        
        Returns:
            dict: {
                'log_beta': Normalized Log-Scale Large Scale Fading,
                'angle': Normalized Angle of Arrival,
                'uav_pos': Normalized UAV coordinates
            }
        """
        # 1. Log Beta (Normalized)
        # Log10 scale to handle large dynamic range [1e-13, 1e-9] -> [-130, -90] dB
        log_beta = np.log10(self.beta_matrix + 1e-20)
        norm_log_beta = (log_beta + 10.0) / 5.0 # Approx range normalization
        
        # 2. Angle (Normalized)
        # Degrees [0, 180] -> [0, 1]
        norm_angle = self.theta_matrix / 180.0
        
        # 3. UAV Positions (Normalized)
        # (N, 3)
        norm_uav_pos = self.uav_positions / [self.X, self.Y, 300.0] # Max height 300
        
        return {
            'log_beta': norm_log_beta,
            'angle': norm_angle,
            'uav_pos': norm_uav_pos
        }

    def _apply_action(self, action):
        """
        Apply Hybrid Action Logic (Top-P Mask + Power Allocation).
        应用混合动作逻辑：Top-P 掩码 + 功率分配。
        
        Steps:
        1. Top-P Filter: Determine physically viable connections based on Beta.
        2. K-max Constraint: Limit max connections per UAV to self.k_max.
        3. Power Masking: Apply mask to Agent's raw power output.
        4. BS Power Normalization: Ensure sum(Power) per BS <= P_max.
        """
        # 1. Calculate Top-P Mask (计算几何 Top-P 掩码)
        # Sort descending
        sorted_indices = np.argsort(self.beta_matrix, axis=0)[::-1]
        sorted_betas = np.take_along_axis(self.beta_matrix, sorted_indices, axis=0)
        
        # Cumulative Sum
        cumsum_betas = np.cumsum(sorted_betas, axis=0)
        total_beta = cumsum_betas[-1, :]
        
        # Threshold
        threshold_values = total_beta * self.config.TOP_P_THRESHOLD
        
        # Find cutoff (Top-P)
        mask_cumsum = cumsum_betas >= threshold_values[None, :]
        cutoff_indices = np.argmax(mask_cumsum, axis=0) # First index where sum >= threshold
        
        # 2. Apply K_max constraint (应用最大连接数限制)
        # The number of connected BSs cannot exceed K_max
        final_cutoffs = np.minimum(cutoff_indices, self.k_max - 1)
        
        # 3. Build Mask (构建掩码矩阵)
        top_p_mask = np.zeros((self.M, self.N))
        for n in range(self.N):
            cutoff = final_cutoffs[n]
            
            # Rule 2: Absolute Beta Threshold (Ignore very weak signals)
            # Unless it's the ONLY connection (Rule 1: At least one)
            
            # Get candidate indices up to cutoff
            candidate_indices = sorted_indices[:cutoff+1, n]
            candidate_betas = sorted_betas[:cutoff+1, n]
            
            # Filter by Beta Threshold
            valid_mask = candidate_betas >= self.config.GEO_BETA_THRESHOLD
            
            # Rule 1: Force at least one connection (the strongest one)
            if not np.any(valid_mask):
                # If no BS meets threshold, keep the strongest one (index 0)
                valid_mask[0] = True
            elif not valid_mask[0]:
                 # If strongest is filtered out (sanity check)
                 valid_mask[0] = True
                 
            # Apply valid mask to candidates
            final_indices = candidate_indices[valid_mask]
            
            top_p_mask[final_indices, n] = 1.0
            
        # Save mask for reward calculation (baseline comparison needs this)
        self.top_p_mask = top_p_mask
            
        # 4. Apply Action (Power) masked by Top-P (应用掩码到动作)
        # raw_power is normalized [0, 1] * P_max
        raw_power = action.reshape(self.M, self.N)
        
        # Apply mask first (zero out invalid connections)
        masked_power = raw_power * top_p_mask
        
        # 5. Normalize Power per BS (基站总功率归一化)
        # Constraint: Sum of power allocated by one BS to all UAVs <= 1.0 (normalized)
        bs_total_power = np.sum(masked_power, axis=1) # (M,)
        scaling_factors = np.ones(self.M)
        overloaded_mask = bs_total_power > 1.0
        scaling_factors[overloaded_mask] = 1.0 / (bs_total_power[overloaded_mask] + 1e-9)
        
        # Apply scaling: (M, N) * (M, 1)
        final_normalized_power = masked_power * scaling_factors[:, None]
        
        # Convert to physical power (Watts)
        self.power_matrix = final_normalized_power * self.config.P
        
        # Connection status (Active if Power > Threshold)
        self.connection_matrix = (self.power_matrix > 1e-3).astype(float)

    def _calculate_capacity_reward(self):
        """
        Calculate Delta Capacity Reward.
        计算增量容量奖励 (相比平均分配基线的提升)。
        
        Formula: Reward = Scale * (SumRate_Actual - SumRate_Baseline)
        Baseline: Equal power allocation among Top-P candidates.
        """
        # 1. Calculate Actual Capacity (当前实际容量)
        actual_capacity, _ = self.channel_model.calculate_capacity(
            self.power_matrix, 
            self.beta_matrix, 
            self.gamma_matrix, 
            self.config.pd, 
            self.config.NOISE_POWER
        )
        
        # 2. Calculate Baseline Power Matrix (构建基线功率矩阵: 平均分配)
        # Equal allocation: P_max / Num_Connected_UAVs
        baseline_power = np.zeros_like(self.power_matrix)
        
        # Count connections per BS based on Top-P mask
        bs_connection_counts = np.sum(self.top_p_mask, axis=1) # (M,)
        
        # Avoid division by zero
        active_bs_mask = bs_connection_counts > 0
        
        # Calculate power per UAV for each active BS
        power_per_uav = np.zeros(self.M)
        power_per_uav[active_bs_mask] = self.config.P / bs_connection_counts[active_bs_mask]
        
        # Assign power
        baseline_power = power_per_uav[:, None] * self.top_p_mask
        
        # 3. Calculate Baseline Capacity (基线容量)
        baseline_capacity, _ = self.channel_model.calculate_capacity(
            baseline_power, 
            self.beta_matrix, 
            self.gamma_matrix, 
            self.config.pd, 
            self.config.NOISE_POWER
        )
        
        # 4. Calculate Reward Difference (计算差值)
        if self.config.CAPACITY_REWARD_TYPE == 'threshold_fixed':
            actual_score = np.sum(actual_capacity >= self.config.CAPACITY_THRESHOLD) * self.config.FIXED_REWARD
            baseline_score = np.sum(baseline_capacity >= self.config.CAPACITY_THRESHOLD) * self.config.FIXED_REWARD
            reward = actual_score - baseline_score
        elif self.config.CAPACITY_REWARD_TYPE == 'sum_capacity':
            reward = np.sum(actual_capacity) - np.sum(baseline_capacity)
        else:
            reward = 0.0
            
        return reward * self.config.REWARD_SCALE
