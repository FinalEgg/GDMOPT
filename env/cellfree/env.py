import gymnasium as gym
from gymnasium.spaces import Box
from tianshou.env import DummyVectorEnv
import numpy as np
from .config import X, Y, H, M, N, P, pd, pu, TAU_P, ALPHA1, ALPHA2, XI1, XI2, CAPACITY_THRESHOLD, REWARD_VALUE, STEPS_PER_EPISODE, NOISE_POWER, CARRIER_FREQUENCY, PATH_LOSS_EXPONENT

class CellFreeEnv(gym.Env):

    def __init__(self):
        self._num_steps = 0
        self._terminated = False

        # 初始化基站位置（均匀分布）
        self.bs_positions = np.random.uniform(0, [X, Y], (M, 2))  # 每个基站的 [x, y]

        # 初始化无人机位置（在3D空间中均匀分布，更靠近地面以减少距离）
        self.uav_positions = np.random.uniform(0, [X, Y, H/2], (N, 3))  # 每个无人机的 [x, y, z]，z限制在0-H/2

        # 观测空间：基站和无人机的位置
        # 基站位置 (M*2), 无人机位置 (N*3)
        obs_dim = M*2 + N*3
        self._observation_space = Box(low=0, high=max(X, Y, H), shape=(obs_dim,))

        # 动作空间：连接决策 (M*N) + 功率分配 (M*N), 所有在 [0,1] 范围内
        action_dim = 2 * M * N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))

        self._steps_per_episode = STEPS_PER_EPISODE

        # 初始化连接和功率矩阵
        self.connection_matrix = np.zeros((M, N))
        self.power_matrix = np.zeros((M, N))

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state(self):
        # 状态包括：基站位置，无人机位置
        bs_flat = self.bs_positions.flatten()
        uav_flat = self.uav_positions.flatten()
        return np.concatenate([bs_flat, uav_flat])

    def step(self, action):
        assert not self._terminated, "Episode has terminated"

        # 解析动作：前 M*N 为连接，后 M*N 为功率分配
        connection_actions = action[:M*N].reshape(M, N)
        power_actions = action[M*N:].reshape(M, N)

        # 更新连接矩阵：>0.8 表示连接
        self.connection_matrix = (connection_actions > 0.8).astype(float)

        # 更新功率矩阵：功率系数 = power_actions
        self.power_matrix = power_actions

        # 计算奖励
        reward = self._calculate_reward()

        # 更新无人机位置（简单随机移动用于模拟）
        self._update_uav_positions()

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

        return self.state, {'num_steps': self._num_steps}

    def _calculate_reward(self):
        total_capacity = 0
        for k in range(N):  # 对于每个无人机 k
            # 找到 Ak：服务无人机 k 的基站（已连接）
            Ak = np.where(self.connection_matrix[:, k] == 1)[0]
            if len(Ak) == 0:
                continue
            # 计算信号功率（无人机 k 的期望信号）
            signal = 0
            for m in Ak:
                bs_pos = self.bs_positions[m]
                uav_pos = self.uav_positions[k]
                # 基站 m 和无人机 k 之间的水平距离
                Rmk = np.linalg.norm(bs_pos - uav_pos[:2])
                # 基站 m 和无人机 k 之间的高度差
                Hmk = uav_pos[2]  # 基站假设在 z=0
                # 基站 m 和无人机 k 之间的 3D 欧氏距离
                Dmk = np.linalg.norm(np.concatenate([bs_pos, [0]]) - uav_pos)
                # 仰角（度）
                theta_mk = np.degrees(np.arctan2(Hmk, Rmk + 1e-8))
                # 视线（LoS）链路的概率
                PmkL = 1 / (1 + XI1 * np.exp(-XI2 * (theta_mk - XI1)))
                # 大尺度衰落均值（线性尺度）
                beta_mk = PmkL * Dmk**(-ALPHA1) + (1 - PmkL) * Dmk**(-ALPHA2)
                # 估计信道功率（MMSE 估计）
                gamma_mk = TAU_P * pu * beta_mk**2 / (TAU_P * pu * beta_mk + 1)
                # 功率控制系数
                eta_mk = self.power_matrix[m, k]
                # 累加信号贡献
                signal += np.sqrt(eta_mk) * gamma_mk
            # SINR 的分子：期望信号功率
            numerator = pd * signal**2
            # 计算干扰功率
            interference = 0
            for m in Ak:  # 遍历服务用户 k 的基站 m
                bs_pos = self.bs_positions[m]  # 获取基站 m 的位置
                uav_pos = self.uav_positions[k]  # 获取用户 k 的位置
                # 计算基站 m 和用户 k 之间的水平距离（用于仰角计算）
                Rmk_intf = np.linalg.norm(bs_pos - uav_pos[:2])
                # 计算基站 m 和用户 k 之间的 3D 欧氏距离
                Dmk_intf = np.linalg.norm(np.concatenate([bs_pos, [0]]) - uav_pos)
                # 计算仰角（度），用于 LoS 概率
                theta_mk_intf = 180 / np.pi * np.arctan(uav_pos[2] / Rmk_intf)
                # 计算 LoS 链路概率
                PmkL_intf = 1 / (1 + XI1 * np.exp(-XI2 * (theta_mk_intf - XI1)))
                # 计算大尺度衰落均值（线性尺度），针对 m 和 k
                beta_mk = PmkL_intf * Dmk_intf**(-ALPHA1) + (1 - PmkL_intf) * Dmk_intf**(-ALPHA2)
                for kp in range(N):  # 遍历所有用户 kp
                    if self.connection_matrix[m, kp] == 1:  # 如果基站 m 连接到用户 kp
                        uav_pos_p = self.uav_positions[kp]  # 获取用户 kp 的位置
                        # 计算基站 m 和用户 kp 之间的水平距离
                        Rmp = np.linalg.norm(bs_pos - uav_pos_p[:2])
                        # 计算基站 m 和用户 kp 之间的 3D 距离
                        Dmp = np.linalg.norm(np.concatenate([bs_pos, [0]]) - uav_pos_p)
                        # 计算仰角（度）
                        theta_mp = 180 / np.pi * np.arctan(uav_pos_p[2] / Rmp)
                        # 计算 LoS 概率
                        PmpL = 1 / (1 + XI1 * np.exp(-XI2 * (theta_mp - XI1)))
                        # 计算大尺度衰落（针对 m 和 kp）
                        beta_mp_temp = PmpL * Dmp**(-ALPHA1) + (1 - PmpL) * Dmp**(-ALPHA2)
                        # 计算估计信道功率（MMSE 估计），针对 m 和 kp
                        gamma_mp = TAU_P * pu * beta_mp_temp**2 / (TAU_P * pu * beta_mp_temp + 1)
                        # 获取功率系数（针对 m 和 kp）
                        eta_mp = self.power_matrix[m, kp]
                        # 累加干扰功率：使用 beta_mk（针对接收者 k）和 gamma_mp（针对发送者 kp）
                        interference += eta_mp * gamma_mp * beta_mk
            # SINR 的分母：干扰 + 噪声
            denominator = pd * interference + 1
            # 信干噪比
            SINR_k = numerator / denominator
            # 可达下行速率（信道容量，单位：bits/symbol）
            C_k = np.log2(1 + SINR_k)
            # 累加总容量
            total_capacity += C_k
        return total_capacity

    def _update_uav_positions(self):
        # 无人机的简单随机移动
        movement = np.random.normal(0, 10, (N, 3))  # 小随机移动
        self.uav_positions += movement
        # 保持在边界内
        self.uav_positions = np.clip(self.uav_positions, 0, [X, Y, H/2])

    def seed(self, seed=None):
        np.random.seed(seed)


def make_cellfree_env(training_num=0, test_num=0):
    """Cell-free UAV 环境的包装函数。
    :return: 一个元组 (单个环境, 训练环境, 测试环境)。
    """
    env = CellFreeEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv(
            [lambda: CellFreeEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv(
            [lambda: CellFreeEnv() for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs