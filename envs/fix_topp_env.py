
from gymnasium.spaces import Box, Dict
import numpy as np
from .base_env import BaseCellFreeEnv

class FixTopPEnv(BaseCellFreeEnv):
    def __init__(self, config):
        # Override config parameters for this specific environment if needed
        # But usually config is passed in. We assume config is already updated or we force values here.
        # For safety, we enforce the fixed BS positions here.
        super().__init__(config)
        
        # Fixed BS Positions (X, Y)
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
        
        # Ensure M matches
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
        action_dim = self.M * self.N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))
        
        # Observation Space: Dict
        # Decoupled from Model Input
        self._observation_space = Dict({
            'log_beta': Box(low=-np.inf, high=np.inf, shape=(self.M, self.N)),
            'angle': Box(low=-np.inf, high=np.inf, shape=(self.M, self.N)),
            'uav_pos': Box(low=-np.inf, high=np.inf, shape=(self.N, 3))
        })
        
        self.cached_state = None # Not used in Dict mode
        
        # Set Reward Mode to Capacity
        self.set_reward_mode('capacity')
        
        # K_max for connection limit
        self.k_max = 5

    def _init_bs_positions(self):
        """Override to use fixed positions."""
        self.bs_positions = self.fixed_bs_positions.copy()

    def reset(self, seed=None, options=None):
        # Override reset to handle 3-layer UAV distribution
        # We call super().reset() but we need to intercept the UAV position generation
        
        # Standard reset for other things
        super().reset(seed=seed)
        
        # Re-generate UAV positions based on 3-layer structure
        # Layer 1: 100m, Layer 2: 200m, Layer 3: 300m
        heights = [100.0, 200.0, 300.0]
        
        # Randomly assign each UAV to a layer
        uav_heights = np.random.choice(heights, size=self.N)
        
        # X, Y are random in the area (0 to 10000)
        # Or should they follow the "lanes"? 
        # "6 lanes per layer... spacing 1500m"
        # For RL training, random position is usually better for generalization unless we strictly simulate the lanes.
        # Let's assume random X, Y for now, but fixed Z levels.
        uav_xy = np.random.uniform(0, [self.X, self.Y], (self.N, 2))
        
        self.uav_positions = np.column_stack((uav_xy, uav_heights))
        
        # Recalculate channels with new positions
        self.beta_matrix, self.gamma_matrix, self.theta_matrix = \
            self.channel_model.calculate_large_scale_fading(self.bs_positions, self.uav_positions)
            
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Log Beta (Normalized)
        log_beta = np.log10(self.beta_matrix + 1e-20)
        norm_log_beta = (log_beta + 10.0) / 5.0 # Approx range [-1, 1]
        
        # 2. Angle (Normalized)
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
        # 1. Calculate Top-P Mask
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
        
        # 2. Apply K_max constraint
        # The number of connected BSs cannot exceed K_max
        # So the cutoff index cannot be larger than K_max - 1
        final_cutoffs = np.minimum(cutoff_indices, self.k_max - 1)
        
        # 3. Build Mask
        top_p_mask = np.zeros((self.M, self.N))
        for n in range(self.N):
            cutoff = final_cutoffs[n]
            
            # Rule 2: Absolute Beta Threshold (Ignore very weak signals)
            # Unless it's the ONLY connection (Rule 1: At least one)
            # We check sorted_betas for this UAV
            
            # Get candidate indices up to cutoff
            candidate_indices = sorted_indices[:cutoff+1, n]
            candidate_betas = sorted_betas[:cutoff+1, n]
            
            # Filter by Beta Threshold
            # config.GEO_BETA_THRESHOLD is used as the absolute threshold
            valid_mask = candidate_betas >= self.config.GEO_BETA_THRESHOLD
            
            # Rule 1: Force at least one connection (the strongest one)
            if not np.any(valid_mask):
                # If no BS meets threshold, keep the strongest one (index 0)
                valid_mask[0] = True
            elif not valid_mask[0]:
                 # If strongest is filtered out (shouldn't happen if logic is consistent, but for safety)
                 valid_mask[0] = True
                 
            # Apply valid mask to candidates
            final_indices = candidate_indices[valid_mask]
            
            top_p_mask[final_indices, n] = 1.0
            
        # Save mask for reward calculation
        self.top_p_mask = top_p_mask
            
        # 4. Apply Action (Power) masked by Top-P
        # Action is (M*N) -> (M, N)
        # raw_power is normalized [0, 1] * P_max
        raw_power = action.reshape(self.M, self.N)
        
        # Apply mask first
        masked_power = raw_power * top_p_mask
        
        # 5. Normalize Power per BS
        # Constraint: Sum of power allocated by one BS to all UAVs <= 1.0 (normalized)
        # If sum > 1.0, scale down proportionally
        bs_total_power = np.sum(masked_power, axis=1) # (M,)
        scaling_factors = np.ones(self.M)
        overloaded_mask = bs_total_power > 1.0
        scaling_factors[overloaded_mask] = 1.0 / (bs_total_power[overloaded_mask] + 1e-9)
        
        # Apply scaling: (M, N) * (M, 1)
        final_normalized_power = masked_power * scaling_factors[:, None]
        
        # Convert to physical power
        self.power_matrix = final_normalized_power * self.config.P
        
        # Connection status
        self.connection_matrix = (self.power_matrix > 1e-3).astype(float)

    def _calculate_capacity_reward(self):
        """
        Calculate reward based on Channel Capacity, subtracting the baseline.
        Baseline: Equal power allocation for connected UAVs (defined by top_p_mask).
        """
        # 1. Calculate Actual Capacity
        actual_capacity, _ = self.channel_model.calculate_capacity(
            self.power_matrix, 
            self.beta_matrix, 
            self.gamma_matrix, 
            self.config.pd, 
            self.config.NOISE_POWER
        )
        
        # 2. Calculate Baseline Power Matrix
        # Equal allocation: P_max / Num_Connected_UAVs
        baseline_power = np.zeros_like(self.power_matrix)
        
        # Count connections per BS
        bs_connection_counts = np.sum(self.top_p_mask, axis=1) # (M,)
        
        # Avoid division by zero
        active_bs_mask = bs_connection_counts > 0
        
        # Calculate power per UAV for each active BS
        # (M,)
        power_per_uav = np.zeros(self.M)
        power_per_uav[active_bs_mask] = self.config.P / bs_connection_counts[active_bs_mask]
        
        # Assign power
        # (M, N) = (M, 1) * (M, N)
        baseline_power = power_per_uav[:, None] * self.top_p_mask
        
        # 3. Calculate Baseline Capacity
        baseline_capacity, _ = self.channel_model.calculate_capacity(
            baseline_power, 
            self.beta_matrix, 
            self.gamma_matrix, 
            self.config.pd, 
            self.config.NOISE_POWER
        )
        
        # 4. Calculate Reward Difference
        if self.config.CAPACITY_REWARD_TYPE == 'threshold_fixed':
            actual_score = np.sum(actual_capacity >= self.config.CAPACITY_THRESHOLD) * self.config.FIXED_REWARD
            baseline_score = np.sum(baseline_capacity >= self.config.CAPACITY_THRESHOLD) * self.config.FIXED_REWARD
            reward = actual_score - baseline_score
        elif self.config.CAPACITY_REWARD_TYPE == 'sum_capacity':
            reward = np.sum(actual_capacity) - np.sum(baseline_capacity)
        else:
            reward = 0.0
            
        return reward * self.config.REWARD_SCALE
