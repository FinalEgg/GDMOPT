import numpy as np

class ChannelModel:
    def __init__(self, config):
        self.config = config
        self.XI1 = config.XI1
        self.XI2 = config.XI2
        self.ALPHA1 = config.ALPHA1
        self.ALPHA2 = config.ALPHA2
        self.TAU_P = config.TAU_P
        self.pu = config.pu

    def calculate_large_scale_fading(self, bs_positions, uav_positions):
        """
        Calculate Large Scale Fading coefficients (Beta) and Gamma.
        
        Args:
            bs_positions: (M, 2) array of BS coordinates [x, y]
            uav_positions: (N, 3) array of UAV coordinates [x, y, z]
            
        Returns:
            beta_matrix: (M, N) Large scale fading coefficients
            gamma_matrix: (M, N) SINR related coefficients
            theta_matrix: (M, N) Angle information (degrees)
        """
        # bs_positions: (M, 2) -> (M, 1, 2)
        # uav_positions: (N, 3) -> (1, N, 3)
        
        bs_pos_exp = bs_positions[:, np.newaxis, :] # (M, 1, 2)
        uav_pos_exp = uav_positions[np.newaxis, :, :] # (1, N, 3)
        
        # 1. Calculate horizontal distance Rmk (M, N)
        diff_xy = bs_pos_exp - uav_pos_exp[:, :, :2]
        Rmk = np.linalg.norm(diff_xy, axis=2)
        
        # 2. Calculate 3D distance Dmk (M, N)
        diff_z = -uav_pos_exp[:, :, 2] # (1, N) -> (M, N) via broadcast
        Dmk = np.sqrt(Rmk**2 + diff_z**2)
        
        # 3. Calculate Angle Theta (M, N)
        Hmk = uav_pos_exp[:, :, 2] # (1, N) -> (M, N) via broadcast
        theta_mk = np.degrees(np.arctan2(Hmk, Rmk + 1e-8))
        
        # 4. Calculate Beta (M, N)
        PmkL = 1 / (1 + self.XI1 * np.exp(-self.XI2 * (theta_mk - self.XI1)))
        beta_matrix = PmkL * Dmk**(-self.ALPHA1) + (1 - PmkL) * Dmk**(-self.ALPHA2)
        
        # 5. Calculate Gamma (M, N)
        tau_pu_beta = self.TAU_P * self.pu * beta_matrix
        gamma_matrix = (tau_pu_beta * beta_matrix) / (tau_pu_beta + 1)
        
        return beta_matrix, gamma_matrix, theta_mk

    def calculate_capacity(self, power_matrix, beta_matrix, gamma_matrix, pd, noise_power):
        """
        Calculate Downlink Capacity for each UAV.
        
        Args:
            power_matrix: (M, N) Transmit power from BS m to UAV k
            beta_matrix: (M, N) Large scale fading
            gamma_matrix: (M, N) SINR coefficient
            pd: Downlink power scaling factor
            noise_power: Noise power
            
        Returns:
            capacity: (N,) Capacity per UAV
            sinr: (N,) SINR per UAV
        """
        # 1. Signal Component
        # signal_components: (M, N)
        signal_components = np.sqrt(power_matrix) * gamma_matrix
        
        # signals: (N,) Sum over BSs
        signals = np.sum(signal_components, axis=0)
        
        # Numerator: (N,)
        numerator = pd * (signals ** 2)
        
        # 2. Interference Component
        # Weighted power per BS: (M, N)
        weighted_power = power_matrix * gamma_matrix
        
        # Total effective power per BS: (M,)
        T = np.sum(weighted_power, axis=1)
        
        # Interference source for each link (m, k): T_m - P_mk * gamma_mk
        # (M, N)
        interference_source = T[:, np.newaxis] - weighted_power
        
        # Received interference: (M, N)
        interference_matrix = beta_matrix * interference_source
        
        # Total interference per UAV: (N,)
        interferences = np.sum(interference_matrix, axis=0)
        
        # Denominator
        denominator = pd * interferences + noise_power
        
        # 3. SINR & Capacity
        sinr = numerator / denominator
        capacity = np.log2(1 + sinr)
        
        return capacity, sinr
