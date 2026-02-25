import numpy as np

class ChannelModelBF:
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

    def calculate_beamforming_gain(self, bs_positions, uav_positions, beam_matrix):
        """
        计算波束赋形带来的增益系数。
        采用余弦指数模型计算方向性增益。
        
        原理:
        对于基站 m，如果它发射波束对准用户 k (beam_matrix[m,k]=True)，
        则该波束对用户 j 造成的信号强度取决于向量 mk 与 mj 的夹角余弦。
        Gain(m, j) = Sum_{k \in Targets} [ max(0, cos(theta_{mk, mj})) ^ exponent ]
        
        参数:
            bs_positions: (M, 2) 基站坐标 [x, y]
            uav_positions: (N, 3) 无人机坐标 [x, y, z]
            beam_matrix: (M, N) 布尔矩阵或 0/1 矩阵，指示波束对准关系。
            
        返回:
            total_gain: (M, N) 结合了波束赋形影响后的增益系数矩阵 (通常在 0 到 1 之间，多波束可能大于 1)。
        """
        if beam_matrix is None:
            return np.ones((bs_positions.shape[0], uav_positions.shape[0]))
            
        # 波束宽度指数 (Beam Width Index)
        # n = 1: 宽波束 (余弦)
        # n > 1: 窄波束
        # 暂时硬编码或从 config 读取，这里设为 8 (较窄波束)
        BEAM_EXPONENT = getattr(self.config, 'BEAM_EXPONENT', 8.0)
        
        # 1. 计算通过所有 M 个基站到所有 N 个用户的单位方向向量
        # bs: (M, 1, 2), uav: (1, N, 3)
        bs_pos_exp = bs_positions[:, np.newaxis, :]
        uav_pos_exp = uav_positions[np.newaxis, :, :]
        
        # 相对位置向量 (M, N, 3)
        vec_mj = np.zeros((bs_positions.shape[0], uav_positions.shape[0], 3))
        vec_mj[:, :, :2] = uav_pos_exp[:, :, :2] - bs_pos_exp
        vec_mj[:, :, 2] = uav_pos_exp[:, :, 2] # 基站 z=0
        
        # 归一化得到方向向量 (M, N, 3)
        norms = np.linalg.norm(vec_mj, axis=2, keepdims=True) + 1e-9
        dirs_mj = vec_mj / norms
        
        # 2. 计算余弦相似度矩阵 (M, N_target, N_victim)
        # 这是一个大张量计算: 对于每个 BS m，计算所有目标 k 和受害者 j 之间的夹角余弦
        # cos_theta[m, k, j] = dot(dir_mk, dir_mj)
        # 使用 einsum: 'mki, mji -> mkj' (i 是 xyz 维度)
        cos_theta = np.einsum('mki,mji->mkj', dirs_mj, dirs_mj)
        
        # 3. 应用波束形状 (ReLU + Power)
        # 只保留正余弦 (前方)，背瓣为 0
        gain_pattern = np.maximum(0, cos_theta) ** BEAM_EXPONENT
        
        # 4. 根据 beam_matrix 聚合增益
        # beam_matrix: (M, N) -> (M, N, 1) 广播到 j 维
        # active_gains[m, k, j] = gain_pattern[m, k, j] * beam_matrix[m, k]
        # total_gain[m, j] = sum_k(active_gains)
        
        # 将 beam_matrix 转换为浮点掩码
        mask = beam_matrix.astype(np.float32) # (M, N)
        
        # 加权求和: 对于每个 BS m 到用户 j，累加所有激活波束的贡献
        # formula: G_mj = sum_k ( mask_mk * gain_mkj )
        bf_gain_matrix = np.einsum('mkj,mk->mj', gain_pattern, mask)
        
        # 这里的 bf_gain_matrix 可能包含 0 (如果没有波束指向该用户附近)
        # 或者大于 1 (如果多个波束指向该用户附近)
        # 如果一个基站没有任何激活波束，它通常不会发射功率，所以增益为0是合理的。
        # 如果 beam_matrix 全为 False，则返回全 0 矩阵 (如果不发射) 或全 1 (如果是全向)。
        # 根据上下文，如果 beam_matrix 为空，通常意味着没有波束赋形（全向），
        # 但如果在调用此函数时传入了矩阵，说明启用了 BF。
        # 这里处理一种特殊情况：如果某行全为 False (基站不工作)，这由功率分配决定，这里只计算“如果发射”的增益。
        # 但如果是全向天线模式，应该在外部不传 beam_matrix。
        # 为了防止全 0 导致的问题 (如果基站开启但没对准)，我们这里假设这纯粹是BF增益。
        
        # 限制最大值 (可选)
        # bf_gain_matrix = np.clip(bf_gain_matrix, 0, 1.0) 
        
        return bf_gain_matrix

    def calculate_large_scale_fading(self, bs_positions, uav_positions, beam_matrix=None):
        """
        计算大尺度衰落系数 (Beta) 和估计参数 Gamma。
        
        逻辑:
        1. 计算三维距离和仰角。
        2. 计算视距概率 (PmkL)。
        3. 基于 PmkL 和路径损耗指数计算 Beta。
        4. 计算 Gamma (与 MMSE 信道估计相关)。
        5. (新增) 如果提供了 beam_matrix，计算波束赋形增益并乘到 Beta 上。
        
        参数:
            bs_positions: (M, 2) 基站坐标数组 [x, y]。
            uav_positions: (N, 3) 无人机坐标数组 [x, y, z]。
            beam_matrix: (M, N, bool) 可选。波束对准矩阵。
            
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
        
        # 5. 波束赋形增益修正
        if beam_matrix is not None:
            bf_gain = self.calculate_beamforming_gain(bs_positions, uav_positions, beam_matrix)
            beta_matrix = beta_matrix * bf_gain

        # 6. 计算 Gamma (M, N) (信道估计系数)
        # MMSE 估计: gamma = (tau * pu * beta^2) / (tau * pu * beta + 1)
        # 注意: 这里是否应该在计算 Gamma 之前应用 BF 增益?
        # Beta 代表大尺度接收功率系数。波束赋形实际上增加了接收功率。
        # 因此，Beta 应该包含 BF 增益。
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
