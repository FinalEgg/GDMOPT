import gymnasium as gym
from gymnasium.spaces import Box
from tianshou.env import DummyVectorEnv
import numpy as np
from .config import X, Y, H, M, N, P, pd, pu, TAU_P, ALPHA1, ALPHA2, XI1, XI2, CAPACITY_THRESHOLD, REWARD_VALUE, STEPS_PER_EPISODE, NOISE_POWER, CARRIER_FREQUENCY, PATH_LOSS_EXPONENT, CONNECTION_COST, REWARD_SCALE, GATE_K, GATE_TH

class CellFreeEnv(gym.Env):

    def __init__(self, reward_mode="physical", k_nearest=3, action_mode="raw"):
        self.reward_mode = reward_mode
        self.k_nearest = k_nearest
        self.action_mode = action_mode # "raw" or "top_k"
        self._num_steps = 0
        self._terminated = False

        # 初始化基站位置（均匀分布）
        self.bs_positions = np.random.uniform(0, [X, Y], (M, 2))  # 每个基站的 [x, y]

        # 初始化无人机位置（在3D空间中均匀分布，更靠近地面以减少距离）
        self.uav_positions = np.random.uniform(0, [X, Y, H/2], (N, 3))  # 每个无人机的 [x, y, z]，z限制在0-H/2

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

        # --- 软阈值处理 (Soft Thresholding) ---
        # 目的：模拟物理连接的稀疏性，同时保持梯度可导以供RL训练
        # 新逻辑：Subtract & Rescale (ReLU-like)
        # 如果 power < GATE_TH，则 effective_power = 0
        # 如果 power >= GATE_TH，则 effective_power = (power - GATE_TH) / (1 - GATE_TH)
        # 为了保持梯度，我们使用一个平滑的近似，或者直接使用 ReLU
        # 这里使用 ReLU，因为 SAC 的探索噪声可以帮助跳过死区
        
        # 1. 减去阈值并截断
        effective_power = np.maximum(0, self.power_matrix - GATE_TH)
        
        # 2. 重新缩放到 [0, 1]
        if GATE_TH < 1.0:
            effective_power = effective_power / (1.0 - GATE_TH)
        
        self.power_matrix = effective_power

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

    def reset(self):
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
        基于几何拓扑的预训练奖励函数。
        目标：引导智能体连接到信道增益最好的 K 个基站，并断开其他连接。
        """
        reward = 0.0
        
        # 遍历每个用户
        for k in range(N):
            # 获取该用户与所有基站的信道增益 (beta)
            betas = self.beta_matrix[:, k] # (M,)
            
            # 找到最大的 K 个值的索引
            k_eff = min(self.k_nearest, M)
            top_k_indices = np.argsort(betas)[-k_eff:]
            
            # 构建目标功率向量：邻居为1，非邻居为0
            target_power = np.zeros(M)
            target_power[top_k_indices] = 1.0
            
            # 获取当前分配的功率
            current_power = self.power_matrix[:, k] # (M,)
            
            # 计算 MSE 损失作为负奖励
            # 我们希望 current_power 接近 target_power
            mse = np.mean((current_power - target_power)**2)
            
            # 奖励 = 1 - MSE (最大化)
            # 这样奖励总是 <= 1，且当完全匹配时为 1
            reward += (1.0 - mse)
            
        # 归一化并缩放
        return (reward / N) * REWARD_SCALE

    def _calculate_physical_reward(self):
        # 1. 使用预计算的 Gamma 矩阵 (M, N)
        # self.gamma_matrix 已经在 _calculate_large_scale_fading 中计算并缓存
        
        # 2. 计算信号功率 (Signal Power)
        # Signal_k = sum_m (sqrt(eta_mk) * gamma_mk)
        # 只有连接的基站才贡献信号，但这里我们假设所有分配了功率的基站都贡献信号
        # connection_matrix 实际上是由 power_matrix > 0.01 决定的，
        # 但物理上只要 power > 0 就有信号。我们直接用 power_matrix 计算。
        
        # 预计算加权功率项，避免重复计算
        weighted_power = self.power_matrix * self.gamma_matrix
        
        # sqrt(eta) * gamma = sqrt(eta) * gamma
        # 注意：这里公式可能是 sqrt(eta) * gamma，也可能是 eta * gamma
        # 根据之前的代码：signal_components = np.sqrt(self.power_matrix) * gamma_matrix
        # 这是一个相干叠加的假设 (Coherent Joint Transmission)
        signal_components = np.sqrt(self.power_matrix) * self.gamma_matrix
        
        # 对每个用户 k 求和 (axis=0 是基站维)
        signals = np.sum(signal_components, axis=0) # (N,)
        numerator = pd * (signals ** 2)
        
        # 3. 计算干扰功率 (Interference Power)
        # I_k = sum_m [ beta_mk * ( (sum_j eta_mj * gamma_mj) - eta_mk * gamma_mk ) ]
        # 令 T_m = sum_j (eta_mj * gamma_mj) 为基站 m 发送的总加权功率
        
        # T_m: 每个基站的总发射效应 (M,)
        # 使用预计算的 weighted_power
        T = np.sum(weighted_power, axis=1)
        
        # 构造干扰矩阵 (M, N)
        # 对于每个 (m, k)，干扰源是 T_m - eta_mk * gamma_mk
        # 利用广播: T[:, None] 是 (M, 1)，减去 (M, N)
        interference_source = T[:, None] - weighted_power
        
        # 乘以路径损耗 beta_mk
        interference_matrix = self.beta_matrix * interference_source
        
        # 对每个用户 k 求和得到总干扰
        interferences = np.sum(interference_matrix, axis=0) # (N,)
        
        denominator = pd * interferences + NOISE_POWER
        
        # 4. 计算 SINR 和 Capacity
        SINR = numerator / denominator
        Capacity = np.log2(1 + SINR)
        
        # 截断和求和
        Capacity = np.clip(Capacity, 0, 10)
        total_capacity = np.sum(Capacity)
        
        # --- 稀疏性惩罚 (Sparsity Penalty) ---
        # 惩罚非零连接，鼓励智能体断开无用连接
        # 使用平滑的 L1 正则化或直接惩罚连接数
        # 这里我们惩罚 "有效连接数" (power > 0.01)
        # 为了保持梯度，我们使用 power 的加权和作为惩罚项的一部分
        
        # 1. 硬连接惩罚 (用于最终评估，但梯度不连续)
        # num_connections = np.sum(self.power_matrix > 0.01)
        
        # 2. 软连接惩罚 (Soft L1 Penalty)
        # 惩罚所有功率的总和，或者使用 log barrier
        # 这里简单地惩罚总功率消耗，系数设为 CONNECTION_COST
        # 这样智能体在收益(Capacity)小于成本(Cost)时会倾向于关闭连接
        power_penalty = CONNECTION_COST * np.sum(self.power_matrix)
        
        # 缩放奖励以稳定训练 (可选，根据之前经验)
        # 最终奖励 = (容量收益 - 功率成本) * 缩放因子
        return (total_capacity / 10.0 - power_penalty) * REWARD_SCALE

    def _update_uav_positions(self):
        # 无人机的简单随机移动
        movement = np.random.normal(0, 10, (N, 3))  # 小随机移动
        self.uav_positions += movement
        # 保持在边界内
        self.uav_positions = np.clip(self.uav_positions, 0, [X, Y, H/2])

    def seed(self, seed=None):
        np.random.seed(seed)


def make_cellfree_env(training_num=0, test_num=0, reward_mode="physical", k_nearest=3):
    """Cell-free UAV 环境的包装函数。
    :return: 一个元组 (单个环境, 训练环境, 测试环境)。
    """
    env = CellFreeEnv(reward_mode=reward_mode, k_nearest=k_nearest)
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv(
            [lambda: CellFreeEnv(reward_mode=reward_mode, k_nearest=k_nearest) for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv(
            [lambda: CellFreeEnv(reward_mode=reward_mode, k_nearest=k_nearest) for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs