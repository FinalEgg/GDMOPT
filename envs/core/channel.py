import numpy as np

class ChannelModel:
    """
    去蜂窝大规模 MIMO 物理信道模型。
    
    物理原理:
    - 概率视距/非视距模型 (Probabilistic LoS/NLoS Model).
    - 大尺度衰落 (路径损耗 + 阴影衰落).
    - 信道估计误差 (MMSE).
    - 下行链路谱效率 (容量) 计算.
    
    参考:
    - T. Ngo et al., "Cell-Free Massive MIMO versus Small Cells", 2017.
    """
    def __init__(self, config):
        """
        初始化信道模型参数。
        
        参数:
            config: 包含物理常数的配置对象。
                XI1, XI2: 视距概率参数.
                ALPHA1: 视距路径损耗指数.
                ALPHA2: 非视距路径损耗指数.
                TAU_P: 导频序列长度.
                pu: 导频功率.
        """
        self.config = config
        self.XI1 = config.XI1
        self.XI2 = config.XI2
        self.ALPHA1 = config.ALPHA1
        self.ALPHA2 = config.ALPHA2
        self.TAU_P = config.TAU_P
        self.pu = config.pu

    def calculate_large_scale_fading(self, bs_positions, uav_positions):
        """
        计算大尺度衰落系数 (Beta) 和估计参数 Gamma。
        
        逻辑:
        1. 计算三维距离和仰角。
        2. 计算视距概率 (PmkL)。
        3. 基于 PmkL 和路径损耗指数计算 Beta。
        4. 计算 Gamma (与 MMSE 信道估计相关)。
        
        参数:
            bs_positions: (M, 2) 基站坐标数组 [x, y]。
            uav_positions: (N, 3) 无人机坐标数组 [x, y, z]。
            
        返回:
            beta_matrix: (M, N) 大尺度衰落系数。
            gamma_matrix: (M, N) MMSE 估计系数。
            theta_matrix: (M, N) 到达角 (度)。
        """
        # 广播维度以进行矩阵运算
        # bs_positions: (M, 2) -> (M, 1, 2)
        # uav_positions: (N, 3) -> (1, N, 3)
        
        bs_pos_exp = bs_positions[:, np.newaxis, :] # (M, 1, 2)
        uav_pos_exp = uav_positions[np.newaxis, :, :] # (1, N, 3)
        
        # 1. 计算水平距离 Rmk (M, N)
        diff_xy = bs_pos_exp - uav_pos_exp[:, :, :2]
        Rmk = np.linalg.norm(diff_xy, axis=2)
        
        # 2. 计算三维距离 Dmk (M, N)
        diff_z = -uav_pos_exp[:, :, 2] # (1, N) -> (M, N) 通过广播 (BS 位于 z=0)
        Dmk = np.sqrt(Rmk**2 + diff_z**2)
        
        # 3. 计算角度 Theta (M, N) (仰角 degrees)
        Hmk = uav_pos_exp[:, :, 2] # (1, N) -> (M, N) 通过广播
        theta_mk = np.degrees(np.arctan2(Hmk, Rmk + 1e-8))
        
        # 4. 计算 Beta (M, N) (大尺度衰落)
        # 视距概率函数: P_LoS = 1 / (1 + a * exp(-b * (theta - a)))
        PmkL = 1 / (1 + self.XI1 * np.exp(-self.XI2 * (theta_mk - self.XI1)))
        
        # 有效路径损耗: P_LoS * D^-alpha1 + (1-P_LoS) * D^-alpha2
        beta_matrix = PmkL * Dmk**(-self.ALPHA1) + (1 - PmkL) * Dmk**(-self.ALPHA2)
        
        # 5. 计算 Gamma (M, N) (信道估计系数)
        # MMSE 估计: gamma = (tau * pu * beta^2) / (tau * pu * beta + 1)
        tau_pu_beta = self.TAU_P * self.pu * beta_matrix
        gamma_matrix = (tau_pu_beta * beta_matrix) / (tau_pu_beta + 1)
        
        return beta_matrix, gamma_matrix, theta_mk

    def calculate_capacity(self, power_matrix, beta_matrix, gamma_matrix, pd, noise_power):
        """
        Calculate Downlink Capacity (Spectral Efficiency) for each UAV.
        计算每个用户的下行容量 (频谱效率)。
        
        公式 (标准去蜂窝下行链路共轭波束成形):
        SINR_k = |Sum(sqrt(P_mk) * gamma_mk)|^2 / (Sum(Beta_mk * (Sum(P_mj * gamma_mj) - P_mk * gamma_mk)) + Noise)
        
        参数:
            power_matrix: (M, N) 发射功率系数 (归一化或物理值)。
            beta_matrix: (M, N) 大尺度衰落。
            gamma_matrix: (M, N) SINR 系数。
            pd: 下行最大功率 (缩放因子)。
            noise_power: 噪声功率 (sigma^2)。
            
        返回:
            capacity: (N,) 每个无人机的容量 (bits/s/Hz)。
            sinr: (N,) 每个无人机的 SINR。
        """
        # 1. 信号分量 (有用信号)
        # 来自所有基站的信号相干合并
        # signal_components: (M, N) -> 每个基站对每个无人机的贡献
        signal_components = np.sqrt(power_matrix) * gamma_matrix
        
        # signals: (N,) 对基站求和
        signals = np.sum(signal_components, axis=0)
        
        # 缩放后的信号功率: pd * |Sum|^2
        numerator = pd * (signals ** 2)
        
        # 2. 干扰分量 (用户间干扰)
        # 基站 m 辐射的总功率 (由信道估计加权): Sum_k(P_mk * gamma_mk)
        # weighted_power: (M, N)
        weighted_power = power_matrix * gamma_matrix
        
        # 每个基站的总有效功率: (M,)
        T = np.sum(weighted_power, axis=1)
        
        # 链路 (m, k) 的干扰源排除发给 k 的信号
        # Source_mk = T_m - P_mk * gamma_mk
        # (M, N)
        interference_source = T[:, np.newaxis] - weighted_power
        
        # 无人机 k 从基站 m 接收到的干扰功率: Beta_mk * Source_mk
        # (M, N)
        interference_matrix = beta_matrix * interference_source
        
        # 每个无人机的总干扰: Sum_m(Interference_mk) -> (N,)
        interferences = np.sum(interference_matrix, axis=0)
        
        # 分母 (干扰 + 噪声)
        denominator = pd * interferences + noise_power
        
        # 3. SINR & 容量r
        capacity = np.log2(1 + sinr)
        
        return capacity, sinr
