import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from tianshou.env import DummyVectorEnv
import numpy as np
from .config import X, Y, H, M, N, P, pd, pu, TAU_P, ALPHA1, ALPHA2, XI1, XI2, CAPACITY_THRESHOLD, REWARD_VALUE, STEPS_PER_EPISODE, NOISE_POWER, CARRIER_FREQUENCY, PATH_LOSS_EXPONENT, CONNECTION_COST, REWARD_SCALE, GATE_K, GATE_TH

class CellFreeEnv(gym.Env):

    def __init__(self, reward_mode="physical", k_nearest=3, action_mode="raw", top_p=0.6):
        self.reward_mode = reward_mode
        self.k_nearest = k_nearest
        self.action_mode = action_mode
        self.top_p = top_p
        self._num_steps = 0
        self._terminated = False

        # 初始化基站位置（均匀分布）
        self.bs_positions = np.random.uniform(0, [X, Y], (M, 2))  # 每个基站的 [x, y]

        # 初始化无人机位置
        self.uav_positions = np.random.uniform(0, [X, Y, H], (N, 3))  # 每个无人机的 [x, y, z]

        # 观测空间：
        # 每个 UAV 的特征 = [与所有 BS 的 LogBeta/角度 (M*2)] + [自身位置 (x,y,z) (3)]
        # 总维度 = N * (M * 2 + 3)
        self.uav_feature_dim = M * 2 + 3
        obs_dim = N * self.uav_feature_dim
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

        # 动作空间：功率分配矩阵 (M*N)，所有在 [0,1] 范围内
        action_dim = M * N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))

        self._steps_per_episode = STEPS_PER_EPISODE

        # 初始化连接和功率矩阵
        self.connection_matrix = np.zeros((M, N))
        self.power_matrix = np.zeros((M, N))
        
        # 缓存变量
        self.beta_matrix = np.zeros((M, N))
        self.cached_state = np.zeros(obs_dim)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state(self):
        return self.cached_state

    def _calculate_large_scale_fading(self):
        """
        预计算所有 BS-UAV 对的几何关系和大尺度衰落系数。
        在一个 Episode 内，如果位置不变，只需在 reset 时调用一次。
        """
        # 向量化计算
        # bs_positions: (M, 2) -> (M, 1, 2)
        # uav_positions: (N, 3) -> (1, N, 3)
        
        bs_pos_exp = self.bs_positions[:, np.newaxis, :] # (M, 1, 2)
        uav_pos_exp = self.uav_positions[np.newaxis, :, :] # (1, N, 3)
        
        # 1. 计算水平距离 Rmk (M, N)
        # uav_pos_exp[:, :, :2] 取前两维 (x, y)
        diff_xy = bs_pos_exp - uav_pos_exp[:, :, :2]
        Rmk = np.linalg.norm(diff_xy, axis=2)
        
        # 2. 计算 3D 距离 Dmk (M, N)
        # bs z=0, uav z=uav_pos_exp[:, :, 2]
        # diff_z = 0 - uav_z
        diff_z = -uav_pos_exp[:, :, 2] # (1, N) -> (M, N) via broadcast
        # Dmk = sqrt(Rmk^2 + diff_z^2)
        Dmk = np.sqrt(Rmk**2 + diff_z**2)
        
        # 3. 计算角度 Theta (M, N)
        # Hmk = uav_z
        Hmk = uav_pos_exp[:, :, 2] # (1, N) -> (M, N) via broadcast
        theta_mk = np.degrees(np.arctan2(Hmk, Rmk + 1e-8))
        
        # 4. 计算 Beta (M, N)
        PmkL = 1 / (1 + XI1 * np.exp(-XI2 * (theta_mk - XI1)))
        self.beta_matrix = PmkL * Dmk**(-ALPHA1) + (1 - PmkL) * Dmk**(-ALPHA2)
        
        # 5. 预计算 Gamma 矩阵 (M, N)
        # gamma_mk = (tau_p * pu * beta_mk^2) / (tau_p * pu * beta_mk + 1)
        tau_pu_beta = TAU_P * pu * self.beta_matrix
        self.gamma_matrix = (tau_pu_beta * self.beta_matrix) / (tau_pu_beta + 1)
        
        # 6. 构造状态向量
        # LogBeta
        log_beta = np.log10(self.beta_matrix + 1e-20)
        norm_log_beta = (log_beta + 10.0) / 5.0
        
        # Angle (Normalized)
        norm_angle = theta_mk / 180.0
        
        # 组合特征: (N, M, 2) -> [LogBeta, Angle]
        # Transpose to (N, M)
        beta_feat = norm_log_beta.T[:, :, np.newaxis]
        angle_feat = norm_angle.T[:, :, np.newaxis]
        
        link_features_2d = np.concatenate([beta_feat, angle_feat], axis=2)
        link_features = link_features_2d.reshape(N, -1)
        
        # Pos features
        pos_features = self.uav_positions / np.array([X, Y, H])
        
        state_matrix = np.hstack([link_features, pos_features])
        self.cached_state = state_matrix.flatten()

    def step(self, action):
        assert not self._terminated, "Episode has terminated"

        # 解析动作：功率分配 (M*N)
        power_actions = action.reshape(M, N)

        # 更新功率矩阵
        self.power_matrix = power_actions
        pass

        # 更新连接矩阵：用于统计
        self.connection_matrix = (self.power_matrix > 0.001).astype(float)

        # 功率约束：每个基站的总功率分配之和不能超过1
        # 向量化归一化
        total_power = np.sum(self.power_matrix, axis=1, keepdims=True) # (M, 1)
        # 找到需要归一化的基站索引
        mask = total_power > 1.0
        # 避免除以0 (虽然 mask 保证了 > 1.0)
        # 利用广播直接更新
        # 注意：mask 是 (M, 1)，power_matrix 是 (M, N)
        # 我们只更新那些 mask 为 True 的行
        if np.any(mask):
             self.power_matrix = np.where(mask, self.power_matrix / total_power, self.power_matrix)

        # 计算奖励
        if self.reward_mode == "geometric":
            reward = self._calculate_geometric_reward()
        else:
            reward = self._calculate_physical_reward()

        self._num_steps += 1
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True

        terminated = False
        truncated = self._terminated
        info = {'num_steps': self._num_steps}

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Gymnasium API requires seed and options
        if seed is not None:
            self.seed(seed)
            
        self._num_steps = 0
        self._terminated = False

        # 重新初始化位置
        self.bs_positions = np.random.uniform(0, [X, Y], (M, 2))
        self.uav_positions = np.random.uniform(0, [X, Y, H/2], (N, 3))
        self.connection_matrix = np.zeros((M, N))
        self.power_matrix = np.zeros((M, N))
        
        # 预计算信道增益和状态
        self._calculate_large_scale_fading()

        return self.state, {'num_steps': self._num_steps}

    def _calculate_geometric_reward(self):
        """
        基于几何拓扑的预训练奖励函数 (Redesigned)。
        目标：引导智能体连接到信道增益最好的 Top-P 集合。
        
        优化：
        - 向量化计算，移除 Python 循环，提高计算效率。
        - 规范化注释，清晰解释每一步的物理含义。
        
        奖励组成：
        1. 基础奖励 (TP)：正确连接给予 GEO_REWARD_HIT。
        2. 完美匹配 (Perfect)：完全匹配 Top-P 结果给予 GEO_BONUS_PERFECT。
        3. 漏连惩罚 (FN)：应连未连给予 GEO_PENALTY_MISS。
        4. 误连惩罚 (FP)：
           - 无用连接 (Beta < Threshold)：给予 GEO_PENALTY_USELESS。
           - 次优连接 (Beta >= Threshold 但不在 Top-P)：给予 GEO_PENALTY_WRONG。
        5. 无连接惩罚 (No Connect)：UAV 未连接任何基站给予 GEO_PENALTY_NO_CONNECT。
        """
        from .config import GEO_BETA_THRESHOLD, GEO_REWARD_HIT, GEO_PENALTY_MISS, GEO_PENALTY_USELESS, GEO_BONUS_PERFECT, GEO_PENALTY_WRONG, GEO_PENALTY_NO_CONNECT
        
        # --- 1. 构建全局目标矩阵 (Target Matrix) ---
        # 目标：找出每个 UAV 的 Top-P 基站集合，且 Beta >= GEO_BETA_THRESHOLD
        
        # 对 Beta 矩阵进行排序 (M, N)，axis=0 (基站维度)
        # sort_indices: 排序后的索引，从大到小
        # sorted_betas: 排序后的 Beta 值
        sorted_indices = np.argsort(self.beta_matrix, axis=0)[::-1]
        sorted_betas = np.take_along_axis(self.beta_matrix, sorted_indices, axis=0)
        
        # 计算累积和 (CumSum)
        cumsum_betas = np.cumsum(sorted_betas, axis=0)
        total_beta = cumsum_betas[-1, :] # (N,) 每个 UAV 的总 Beta
        
        # 计算 Top-P 阈值
        # threshold_values: (N,)
        threshold_values = total_beta * self.top_p
        
        # 找到满足 Top-P 的截断位置
        # mask_cumsum: (M, N) 哪些位置的累积和已经超过阈值
        # argmax 会返回第一个 True 的索引，即截断位置
        mask_cumsum = cumsum_betas >= threshold_values[None, :]
        cutoff_indices = np.argmax(mask_cumsum, axis=0) # (N,)
        
        # --- 绝对阈值过滤 ---
        # 只有 Beta >= GEO_BETA_THRESHOLD 的基站才有效
        # 统计每个 UAV 有多少个有效基站
        valid_counts = np.sum(sorted_betas >= GEO_BETA_THRESHOLD, axis=0) # (N,)
        
        # 最终截断索引：取 Top-P 截断和有效基站数量的较小值
        # 注意：索引是 0-based，所以如果 valid_counts=3，最大索引是 2
        # 修改：强制至少选择一个基站 (即使 valid_counts=0)
        # 逻辑：final_cutoffs 至少为 0 (即选择 sorted_indices[0])
        
        # 1. 计算基于规则的截断点 (可能为 -1)
        rule_based_cutoffs = np.minimum(cutoff_indices, valid_counts - 1)
        
        # 2. 强制保底：至少选择 Top 1
        final_cutoffs = np.maximum(rule_based_cutoffs, 0)
        
        # 构建目标矩阵 (Target Matrix)
        target_matrix = np.zeros((M, N))
        
        # 生成网格索引以进行向量化赋值
        # 我们需要将 sorted_indices 中前 final_cutoffs + 1 个位置设为 1
        
        # row_indices: (M, 1) -> (0, 1, 2, ..., M-1)
        row_indices = np.arange(M)[:, None]
        
        # selection_mask: (M, N)
        # 只要行索引 <= 最终截断索引即可 (final_cutoffs 保证 >= 0)
        selection_mask = (row_indices <= final_cutoffs[None, :])
        
        # 获取目标基站的原始索引
        target_bs_indices = sorted_indices[selection_mask]
        
        # 获取对应的 UAV 索引
        # col_indices: (M, N) -> (0, 0...; 1, 1...; ...)
        col_indices = np.tile(np.arange(N), (M, 1))
        target_uav_indices = col_indices[selection_mask]
        
        # 赋值
        target_matrix[target_bs_indices, target_uav_indices] = 1.0
            
        # --- 2. 计算奖励 (Vectorized) ---
        current_connection = self.connection_matrix
        
        reward = 0.0
        
        # True Positive (正确连接): Target=1 & Current=1
        tp_count = np.sum((target_matrix == 1) & (current_connection == 1))
        reward += tp_count * GEO_REWARD_HIT
        
        # False Negative (漏连): Target=1 & Current=0
        fn_count = np.sum((target_matrix == 1) & (current_connection == 0))
        reward -= fn_count * GEO_PENALTY_MISS
        
        # False Positive (误连): Target=0 & Current=1
        fp_mask = (target_matrix == 0) & (current_connection == 1)
        
        # 细分误连：
        # 1. 无用连接 (Useless): Beta < Threshold
        useless_mask = (self.beta_matrix < GEO_BETA_THRESHOLD)
        fp_useless_count = np.sum(fp_mask & useless_mask)
        
        # 2. 次优连接 (Wrong): Beta >= Threshold (但不在 Top-P)
        fp_wrong_count = np.sum(fp_mask & (~useless_mask))
        
        reward -= fp_useless_count * GEO_PENALTY_USELESS
        reward -= fp_wrong_count * GEO_PENALTY_WRONG
        
        # --- 3. 全局约束惩罚 ---
        
        # 无连接惩罚 (No Connect): UAV 未连接任何基站
        uav_connections = np.sum(current_connection, axis=0) # (N,)
        no_connect_uavs = np.sum(uav_connections == 0)
        reward -= no_connect_uavs * GEO_PENALTY_NO_CONNECT
        
        # 完美匹配奖励 (Perfect Bonus)
        # 只有当所有连接都完全匹配时才给予
        if np.array_equal(target_matrix, current_connection):
            reward += GEO_BONUS_PERFECT
            
        return reward

    def _calculate_capacity(self):
        """
        计算每个 UAV 的下行链路容量。
        
        Returns:
            Capacity (np.ndarray): 每个 UAV 的容量 (N,)
            total_capacity (float): 总容量
        """
        # 1. 准备数据
        # self.gamma_matrix: (M, N) 预计算的大尺度衰落因子
        # self.power_matrix: (M, N) 当前功率分配动作 (eta_mk)
        
        # 2. 计算信号分量 (Signal Component)
        # 假设相干传输，信号幅度假加
        # signal_components: (M, N)
        signal_components = np.sqrt(self.power_matrix) * self.gamma_matrix
        
        # 对每个 UAV 求和得到总接收信号幅度
        # signals: (N,)
        signals = np.sum(signal_components, axis=0)
        
        # 接收信号功率 (Numerator)
        # numerator: (N,)
        numerator = pd * (signals ** 2)
        
        # 3. 计算干扰功率 (Interference Power)
        # 计算每个基站的总加权发射功率
        # weighted_power: (M, N) = eta_mk * gamma_mk
        weighted_power = self.power_matrix * self.gamma_matrix
        
        # T: (M,) 每个基站的总有效发射功率
        T = np.sum(weighted_power, axis=1)
        
        # 计算每个 UAV 接收到的干扰
        # 干扰源 = 基站总功率 - 发给该 UAV 的有用功率
        # interference_source: (M, N)
        interference_source = T[:, np.newaxis] - weighted_power
        
        # 考虑路径损耗 beta_mk
        # interference_matrix: (M, N)
        interference_matrix = self.beta_matrix * interference_source
        
        # 对每个 UAV 求和得到总干扰
        # interferences: (N,)
        interferences = np.sum(interference_matrix, axis=0)
        
        # 总干扰 + 噪声
        denominator = pd * interferences + NOISE_POWER
        
        # 4. 计算 SINR 和 Capacity
        SINR = numerator / denominator
        Capacity = np.log2(1 + SINR)
        
        # 截断容量以防止极端值 (Clip)
        Capacity = np.clip(Capacity, 0, 10)
        total_capacity = np.sum(Capacity)
        
        return Capacity, total_capacity

    def _calculate_physical_reward(self):
        """
        计算物理层的下行链路容量奖励 (Downlink Capacity Reward)。
        
        Returns:
            reward (float): 归一化后的总容量减去功率成本。
        """
        # 1. 计算容量
        _, total_capacity = self._calculate_capacity()
        
        # 2. 计算惩罚项 (Penalty)
        # 功率消耗惩罚：鼓励在不显著降低容量的情况下减少功率使用
        power_penalty = CONNECTION_COST * np.sum(self.power_matrix)
        
        # 3. 最终奖励
        # 缩放奖励以便于训练
        reward = (total_capacity - power_penalty) * REWARD_SCALE
        
        return reward

    def _update_uav_positions(self):
        # 无人机的简单随机移动
        movement = np.random.normal(0, 10, (N, 3))  # 小随机移动
        self.uav_positions += movement
        # 保持在边界内
        self.uav_positions = np.clip(self.uav_positions, 0, [X, Y, H/2])

    def seed(self, seed=None):
        np.random.seed(seed)


def make_cellfree_env(training_num=0, test_num=0, reward_mode="physical", k_nearest=3, action_mode="raw", top_p=0.6, norm_reward=False):
    """Cell-free UAV 环境的包装函数。
    :return: 一个元组 (单个环境, 训练环境, 测试环境)。
    """
    def _make_env():
        e = CellFreeEnv(reward_mode=reward_mode, k_nearest=k_nearest, action_mode=action_mode, top_p=top_p)
        if norm_reward:
            e = NormalizeObservation(e)
            e = NormalizeReward(e)
        return e

    env = CellFreeEnv(reward_mode=reward_mode, k_nearest=k_nearest, action_mode=action_mode, top_p=top_p)
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv([_make_env for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        # For testing, we usually don't update the normalization stats, but we need to apply the normalization
        # Ideally, we should share the stats from training envs, but for simplicity here we just apply it.
        # Note: NormalizeReward in test envs might make evaluation metrics confusing (normalized reward).
        # Usually we DO NOT normalize reward in test envs if we want to see real performance.
        # But if the policy expects normalized state, we MUST normalize observation.
        
        def _make_test_env():
            e = CellFreeEnv(reward_mode=reward_mode, k_nearest=k_nearest, action_mode=action_mode, top_p=top_p)
            if norm_reward:
                e = NormalizeObservation(e)
                # We typically do NOT normalize reward for test envs to track real performance
                # e = NormalizeReward(e) 
            return e
            
        test_envs = DummyVectorEnv([_make_test_env for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs