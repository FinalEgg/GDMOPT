import numpy as np
import copy

class GeneticAlgorithmSearch:
    """
    基于遗传算法 (Genetic Algorithm, GA) 的功率分配优化器。
    该类用于在给定的环境状态下，搜索近似最优的功率分配策略，从而生成高质量的 (State, Action) 数据集。
    
    核心特性：
    1.  **向量化评估 (Vectorized Evaluation)**: 
        利用 NumPy 的广播机制，一次性计算整个种群（例如 100 个个体）的适应度（Sum Rate）。
        相比于逐个调用环境的 step 函数，这种方法避免了 Python 循环的巨大开销，大幅提升了搜索效率。
        
    2.  **混合初始化 (Hybrid Initialization)**: 
        种群初始化不再是纯随机，而是结合了启发式知识。
        - 20% 的个体基于信道增益 (Beta) 进行初始化（功率与信道增益成正比），为 GA 提供良好的起点。
        - 80% 的个体保持随机，确保种群的多样性和探索能力。
        
    3.  **物理模型一致性 (Physical Consistency)**: 
        内置了与 `FixTopPEnv` 环境完全一致的物理计算逻辑，包括：
        - Top-P 连接掩码计算 (Top-P Masking)
        - 功率归一化约束 (Power Normalization)
        - SINR 和 Capacity 计算公式
        这确保了 GA 优化的目标与强化学习环境的 Reward 是一致的。
    """
    
    def __init__(self, env, pop_size=100, num_generations=50, elite_ratio=0.1, mutation_rate=0.2, mutation_scale=0.1):
        """
        初始化遗传算法优化器。

        Args:
            env: FixTopPEnv 环境实例，用于获取物理参数（如 BS 位置、功率限制等）。
            pop_size (int): 种群规模，即每一代包含的个体数量。默认 100。
            num_generations (int): 进化的代数。默认 50。
            elite_ratio (float): 精英保留比例。每一代中适应度最高的个体将直接复制到下一代。默认 0.1。
            mutation_rate (float): 变异概率。每个基因发生变异的概率。默认 0.2。
            mutation_scale (float): 变异噪声的标准差。变异时添加的高斯噪声强度。默认 0.1。
        """
        # 确保访问的是原始环境，以获取物理参数
        self.env = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.M = self.env.M # 基站数量
        self.N = self.env.N # 无人机数量
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.elite_size = int(pop_size * elite_ratio)
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        
        # 缓存环境的关键物理参数，避免重复访问
        self.pd = self.env.config.pd # 下行链路功率缩放因子
        self.noise_power = self.env.config.NOISE_POWER # 噪声功率
        self.top_p_threshold = self.env.config.TOP_P_THRESHOLD # Top-P 阈值系数
        self.geo_beta_threshold = self.env.config.GEO_BETA_THRESHOLD # 几何 Beta 阈值
        self.k_max = self.env.k_max # 最大连接数限制

    def search(self, obs):
        """
        执行遗传算法搜索，寻找当前状态下的最优动作。

        Args:
            obs: 当前环境的观测值（注意：此参数在当前实现中未直接使用，而是直接从 self.env 读取内部状态）。
                 在实际使用中，确保 self.env 处于与 obs 对应的状态。

        Returns:
            best_action (np.ndarray): 搜索到的最优动作，扁平化的一维向量，长度为 M * N。
                                      范围在 [0, 1] 之间。
        """
        # 1. 获取当前环境的信道状态信息 (CSI)
        # beta_matrix: 大尺度衰落系数 (M, N)
        # gamma_matrix: SINR 相关系数 (M, N)
        beta_matrix = self.env.beta_matrix
        gamma_matrix = self.env.gamma_matrix
        
        # 2. 预计算 Top-P Mask
        # 在当前时隙内，信道状态是不变的，因此 Top-P 连接掩码也是固定的。
        # 我们只需计算一次，后续所有个体的评估都复用这个掩码。
        top_p_mask = self._calculate_top_p_mask(beta_matrix)
        
        # 3. 初始化种群
        # 生成初始的功率分配方案矩阵 (Pop_Size, M, N)
        population = self._init_population(beta_matrix)
        
        best_individual = None
        best_fitness = -np.inf
        
        # 4. 进化循环
        for gen in range(self.num_generations):
            # 4.1 评估适应度 (Batch Calculation)
            # 计算整个种群的适应度（即 Sum Capacity）。
            # fitness: (Pop_Size,)
            fitness = self._evaluate_population(population, beta_matrix, gamma_matrix, top_p_mask)
            
            # 4.2 记录本代最优个体
            current_best_idx = np.argmax(fitness)
            current_best_fitness = fitness[current_best_idx]
            
            # 更新全局最优
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
            
            # 4.3 选择 (Selection)
            # 使用锦标赛选择法选出父代
            parents = self._tournament_selection(population, fitness)
            
            # 4.4 交叉 (Crossover)
            # 对父代进行算术交叉，生成后代
            offspring = self._crossover(parents)
            
            # 4.5 变异 (Mutation)
            # 对后代进行高斯变异，增加种群多样性
            offspring = self._mutate(offspring)
            
            # 4.6 精英保留与种群更新 (Elitism)
            # 将上一代的精英个体直接保留到下一代，防止最优解丢失
            sorted_indices = np.argsort(fitness)[::-1]
            elites = population[sorted_indices[:self.elite_size]]
            
            # 填充剩余位置
            num_offspring_needed = self.pop_size - self.elite_size
            population = np.vstack((elites, offspring[:num_offspring_needed]))
            
        # --- Post-Processing for Data Collection ---
        # 1. Get the best raw individual
        best_raw = best_individual # (M, N) in [0, 1]
        
        # 2. Apply Mask (Set disconnected links to 0)
        # Note: top_p_mask is (M, N)
        masked_action = best_raw * top_p_mask
        
        # 3. Apply Power Normalization (Sum <= 1)
        bs_total_power = np.sum(masked_action, axis=1) # (M,)
        scaling_factors = np.ones(self.M)
        overloaded_mask = bs_total_power > 1.0
        scaling_factors[overloaded_mask] = 1.0 / (bs_total_power[overloaded_mask] + 1e-9)
        final_action = masked_action * scaling_factors[:, None]
        
        # 4. Inverse Map to [-1, 1] for RL Agent
        # RL Agent (Tanh) -> [-1, 1] -> Wrapper -> [0, 1]
        # So we need to save data in [-1, 1] domain.
        # Formula: y = (x + 1) / 2  =>  x = y * 2 - 1
        rl_action = final_action * 2.0 - 1.0
        
        # 返回处理后的动作，符合 RL Agent 的输出空间 [-1, 1]
        return rl_action.flatten()

    def get_baseline_reward(self):
        """
        计算基准方法的奖励值。
        基准方法：Top-P 连接 + 均分功率 (Equal Power Allocation)。
        """
        beta_matrix = self.env.beta_matrix
        gamma_matrix = self.env.gamma_matrix
        top_p_mask = self._calculate_top_p_mask(beta_matrix)
        
        # 构造全 1 的种群 (1, M, N)
        # 在 _evaluate_population 中，全 1 会被 Top-P Mask 过滤，
        # 然后被归一化，从而实现均分功率。
        population = np.ones((1, self.M, self.N))
        
        fitness = self._evaluate_population(population, beta_matrix, gamma_matrix, top_p_mask)
        return fitness[0]

    def _calculate_top_p_mask(self, beta_matrix):
        """
        计算 Top-P 连接掩码。
        该逻辑完全复现了 FixTopPEnv 中的 _apply_action 方法的前半部分。
        
        规则：
        1. 对每个 UAV，按 Beta 值降序排列 BS。
        2. 计算累积 Beta 和，直到达到总和的 P% (TOP_P_THRESHOLD)。
        3. 截断连接数不超过 K_max。
        4. 过滤掉 Beta 值低于绝对阈值 (GEO_BETA_THRESHOLD) 的弱连接。
        5. 保证至少有一个连接。
        
        Returns:
            top_p_mask (np.ndarray): (M, N) 的二进制掩码，1 表示连接，0 表示断开。
        """
        # 按 Beta 降序排列索引
        sorted_indices = np.argsort(beta_matrix, axis=0)[::-1]
        sorted_betas = np.take_along_axis(beta_matrix, sorted_indices, axis=0)
        
        # 计算累积和
        cumsum_betas = np.cumsum(sorted_betas, axis=0)
        total_beta = cumsum_betas[-1, :]
        
        # 计算截断阈值
        threshold_values = total_beta * self.top_p_threshold
        
        # 找到截断位置 (第一个累积和 >= 阈值的索引)
        mask_cumsum = cumsum_betas >= threshold_values[None, :]
        cutoff_indices = np.argmax(mask_cumsum, axis=0)
        
        # 应用最大连接数限制 K_max
        final_cutoffs = np.minimum(cutoff_indices, self.k_max - 1)
        
        # 构建掩码
        top_p_mask = np.zeros((self.M, self.N))
        for n in range(self.N):
            cutoff = final_cutoffs[n]
            candidate_indices = sorted_indices[:cutoff+1, n]
            candidate_betas = sorted_betas[:cutoff+1, n]
            
            # 过滤弱连接
            valid_mask = candidate_betas >= self.geo_beta_threshold
            
            # 兜底策略：如果所有连接都被过滤，保留最强的一个
            if not np.any(valid_mask):
                valid_mask[0] = True
            elif not valid_mask[0]:
                 valid_mask[0] = True
                 
            final_indices = candidate_indices[valid_mask]
            top_p_mask[final_indices, n] = 1.0
            
        return top_p_mask

    def _init_population(self, beta_matrix):
        """
        初始化种群。采用混合策略：
        1. 启发式部分 (20%)：功率分配与信道增益 Beta 成正比。这模拟了注水算法的一种简化形式。
        2. 随机部分 (80%)：在 [0, 1] 范围内均匀采样。
        
        Returns:
            population (np.ndarray): (Pop_Size, M, N) 的初始种群。
        """
        population = np.random.uniform(0, 1, (self.pop_size, self.M, self.N))
        
        # 计算启发式个体数量
        heuristic_count = int(self.pop_size * 0.2)
        
        # 构造基础启发式解：Power ~ Beta
        sum_beta = np.sum(beta_matrix, axis=1, keepdims=True) + 1e-9
        base_heuristic = beta_matrix / sum_beta
        
        # 添加噪声生成多个变体
        for i in range(heuristic_count):
            noise = np.random.normal(0, 0.1, (self.M, self.N))
            population[i] = np.clip(base_heuristic + noise, 0, 1)
            
        return population

    def _evaluate_population(self, population, beta_matrix, gamma_matrix, top_p_mask):
        """
        向量化计算整个种群的适应度 (Sum Capacity)。
        这是算法的核心性能瓶颈，因此使用了全向量化操作。
        
        Args:
            population: (Pop, M, N) 原始功率矩阵。
            beta_matrix: (M, N) 大尺度衰落。
            gamma_matrix: (M, N) SINR 系数。
            top_p_mask: (M, N) 连接掩码。
            
        Returns:
            fitness: (Pop,) 每个个体的总容量。
        """
        pop_size = population.shape[0]
        
        # 1. 应用 Top-P 掩码
        # 利用广播机制：(Pop, M, N) * (1, M, N) -> (Pop, M, N)
        masked_power = population * top_p_mask[None, :, :]
        
        # 2. 功率归一化 (Power Normalization)
        # 约束：每个 BS 的总发射功率不能超过 P_max (归一化后为 1.0)
        # 计算每个 BS 的总功率: (Pop, M)
        bs_total_power = np.sum(masked_power, axis=2)
        
        # 计算缩放因子
        scaling_factors = np.ones((pop_size, self.M))
        overloaded_mask = bs_total_power > 1.0
        # 如果总功率 > 1，则缩放系数 = 1 / 总功率
        scaling_factors[overloaded_mask] = 1.0 / (bs_total_power[overloaded_mask] + 1e-9)
        
        # 应用缩放: (Pop, M, N) * (Pop, M, 1)
        final_power = masked_power * scaling_factors[:, :, None]
        
        # 转换为物理功率值 (Watts)
        P_matrix = final_power * self.env.config.P
        
        # 3. 计算容量 (Capacity Calculation)
        # 扩展维度以支持广播: (1, M, N)
        beta_exp = beta_matrix[None, :, :]
        gamma_exp = gamma_matrix[None, :, :]
        
        # --- 信号部分 (Signal) ---
        # sqrt(P) * gamma
        signal_components = np.sqrt(P_matrix) * gamma_exp # (Pop, M, N)
        signals = np.sum(signal_components, axis=1) # 对 BS 求和 -> (Pop, N)
        numerator = self.pd * (signals ** 2)
        
        # --- 干扰部分 (Interference) ---
        # 加权功率: P * gamma
        weighted_power = P_matrix * gamma_exp # (Pop, M, N)
        T = np.sum(weighted_power, axis=2) # 每个 BS 的总加权功率 -> (Pop, M)
        
        # 干扰源项: T_m - P_mk * gamma_mk (减去有用信号部分)
        # (Pop, M, 1) - (Pop, M, N) -> (Pop, M, N)
        interference_source = T[:, :, None] - weighted_power
        
        # 接收到的干扰: beta * interference_source
        interference_matrix = beta_exp * interference_source
        interferences = np.sum(interference_matrix, axis=1) # 对 BS 求和 -> (Pop, N)
        
        denominator = self.pd * interferences + self.noise_power
        
        # --- SINR & Capacity ---
        sinr = numerator / denominator
        capacity = np.log2(1 + sinr)
        
        # 返回总容量 (Sum Rate): (Pop,)
        return np.sum(capacity, axis=1)

    def _tournament_selection(self, population, fitness, k=3):
        """
        锦标赛选择 (Tournament Selection)。
        每次随机选取 k 个个体，选择其中适应度最高的一个作为父代。
        重复此过程直到选出足够数量的父代。
        """
        parents = []
        for _ in range(self.pop_size):
            # 随机选择 k 个候选者的索引
            indices = np.random.randint(0, self.pop_size, k)
            # 找到其中适应度最高的索引
            best_idx = indices[np.argmax(fitness[indices])]
            parents.append(population[best_idx])
        return np.array(parents)

    def _crossover(self, parents, alpha=0.5):
        """
        算术交叉 (Arithmetic Crossover)。
        父代两两配对，生成两个线性组合的后代。
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        """
        offspring = []
        # 步长为 2 遍历父代
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)] # 循环取模防止越界
            
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = (1 - alpha) * p1 + alpha * p2
            
            offspring.append(c1)
            offspring.append(c2)
            
        return np.array(offspring)

    def _mutate(self, offspring):
        """
        高斯变异 (Gaussian Mutation)。
        以一定的概率 (mutation_rate) 向基因添加高斯噪声。
        """
        # 生成变异掩码 (True 表示该位置发生变异)
        mask = np.random.random(offspring.shape) < self.mutation_rate
        # 生成噪声
        noise = np.random.normal(0, self.mutation_scale, offspring.shape)
        
        # 应用变异
        offspring[mask] += noise[mask]
        # 截断到 [0, 1] 范围
        return np.clip(offspring, 0, 1)


def collect_demonstration_data(env, num_episodes=100, save_path='demonstration_data.npz'):
    """
    采集演示数据集的主函数。
    
    流程：
    1. 重置环境。
    2. 使用遗传算法搜索当前状态下的最优动作。
    3. 执行动作，获取真实奖励。
    4. 将 (State, Action, Reward) 保存到列表。
    5. 循环结束后保存为 .npz 文件。
    """
    # 实例化 GA 搜索器
    # 使用较小的种群和代数以平衡速度和质量
    # 100 Pop * 50 Gen = 5000 次评估/步
    searcher = GeneticAlgorithmSearch(env, pop_size=100, num_generations=50)
    
    obs_list = []
    act_list = []
    rew_list = []
    obs_next_list = []
    done_list = []
    baseline_rew_list = []
    
    print(f"Starting GA-based data collection for {num_episodes} episodes...")
    print("Press Ctrl+C to stop collection and save current data.")
    
    try:
        for i in range(num_episodes):
            obs, _ = env.reset()
            
            # 1. GA Search
            best_action = searcher.search(obs)
            
            # 2. Calculate Baseline Reward (Top-P + Equal Power)
            baseline_rew = searcher.get_baseline_reward()
            
            # 3. Execute Action
            # 注意：step 会改变环境内部状态（如 _num_steps），但对于单步优化问题影响不大
            # 只要我们每次都 reset 即可。
            next_obs, reward, terminated, truncated, _ = env.step(best_action)
            done = terminated or truncated
            
            # 4. Process Observation
            flat_obs = flatten_obs_with_env(obs, env)
            flat_next_obs = flatten_obs_with_env(next_obs, env)
            
            obs_list.append(flat_obs)
            act_list.append(best_action)
            rew_list.append(reward)
            obs_next_list.append(flat_next_obs)
            done_list.append(done)
            baseline_rew_list.append(baseline_rew)
            
            if (i + 1) % 10 == 0:
                print(f"Collected {i + 1}/{num_episodes} episodes. GA: {reward:.4f} | Base: {baseline_rew:.4f}")
                
    except KeyboardInterrupt:
        print("\nCollection interrupted by user. Saving collected data...")
        
    finally:
        if len(obs_list) > 0:
            print(f"Saving {len(obs_list)} samples to {save_path}...")
            np.savez(save_path, 
                     obs=np.array(obs_list), 
                     act=np.array(act_list), 
                     rew=np.array(rew_list),
                     obs_next=np.array(obs_next_list),
                     done=np.array(done_list),
                     baseline_rew=np.array(baseline_rew_list))
            print("Done.")
        else:
            print("No data collected.")

def flatten_obs_with_env(obs_dict, env):
    """
    将 FixTopPEnv 的 Dict 观测扁平化为 (N * 53) 的向量。
    用于适配 DeepSets 网络的输入格式。
    
    如果输入已经是扁平化的 numpy 数组，则直接返回。
    """
    if isinstance(obs_dict, np.ndarray):
        return obs_dict

    # 确保访问的是原始环境
    env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    N = env.N
    M = env.M
    
    log_beta = obs_dict['log_beta'].T # (N, M)
    angle = obs_dict['angle'].T # (N, M)
    
    # 还原角度 (obs 中是归一化的)
    theta = angle * 180.0 * np.pi / 180.0 # Radian
    sin_angle = np.sin(theta)
    cos_angle = np.cos(theta)
    
    uav_pos = obs_dict['uav_pos'] # (N, 3)
    
    # BS Pos (归一化)
    bs_pos = env.bs_positions / [env.X, env.Y] # (M, 2)
    
    # 构建每个 UAV 的特征
    uav_features = []
    for n in range(N):
        feats = []
        # Per BS features: [log_beta, sin, cos, bs_x, bs_y]
        for m in range(M):
            feats.extend([
                log_beta[n, m],
                sin_angle[n, m],
                cos_angle[n, m],
                bs_pos[m, 0],
                bs_pos[m, 1]
            ])
        # Self features: [uav_x, uav_y, uav_z]
        feats.extend(uav_pos[n])
        uav_features.append(feats)
        
    return np.array(uav_features).flatten()

if __name__ == "__main__":
    import sys
    import os
    import argparse
    
    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config.default_config import DefaultConfig
    from envs.fix_topp_env import FixTopPEnv
    from envs.wrappers import PurePowerActionWrapper
    from config.train_config import TrainConfig
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', type=int, default=TrainConfig.PRETRAIN_EPISODES, help='Number of episodes to collect')
    args = parser.parse_args()
    
    # Create log directory if not exists
    if not os.path.exists('log'):
        os.makedirs('log')
        
    config = DefaultConfig()
    env = FixTopPEnv(config)
    # 必须使用 Wrapper，因为 GA 返回的是 [-1, 1] 的动作，而原始环境期望 [0, 1]
    env = PurePowerActionWrapper(env)
    
    save_path = os.path.join('log', 'demonstration_data.npz')
    collect_demonstration_data(env, num_episodes=args.num_episodes, save_path=save_path)
