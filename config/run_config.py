import torch

class RunConfig:
    """
    运行配置 (Running Configuration).
    定义单次运行的超参数、实验ID和计算资源。
    用于控制训练流程 (Train Loop) 和算法选择 (Algorithm Selection)。
    """
    
    # =========================================================================
    # 实验选择 (Experiment Selection)
    # =========================================================================
    
    # 填写 mapping_config.py 中的 EXPERIMENT_MAP key
    # 例如: 'exp_baseline_mlp', 'exp_deepsets'
    EXPERIMENT_ID = 'exp_cartpole_td3'
    
    # 运行标识 (Run ID)
    # 用于生成日志文件夹名称，如 logs/{EXPERIMENT_ID}/{RUN_ID}
    # 建议包含日期或特性描述
    RUN_ID = 'td3_classic_test'
    
    # 并行环境数量
    NUM_TRAIN_ENVS = 8
    NUM_TEST_ENVS = 4

    # =========================================================================
    # 训练超参数 (Training Hyperparameters)
    # =========================================================================
    
    # 强化学习算法
    # 可选: 'td3' , 'ddpg', 'sac', 'diffusion'
    # 注意：某些算法可能需要特定的 Model Output 结构 (如 SAC 需要 mean+std)
    ALGO = 'td3'


    
    # 神经网络骨干架构
    # 可选: 'deepsets' (推荐, 适合置换不变性), 'gnn' (图神经网络), 'mlp' (普通全连接)
    BACKBONE = 'deepsets'
    
    # 动作模式
    # 'pure_power': 直接输出功率值 (推荐)
    # 'threshold': 输出阈值用于连接控制
    # 'hybrid': 混合输出
    ACTION_MODE = 'pure_power'
    
    # 随机种子，用于复现实验结果
    SEED = 1
    
    # 运行设备 ('cuda' 或 'cpu')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 日志保存目录
    LOGDIR = 'log'
    
    # =========================================================================
    # 网络结构 (Network Architecture)
    # =========================================================================
    
    # 隐藏层维度
    # 较大的维度能拟合更复杂的函数，但计算量更大
    HIDDEN_DIM = 256

    # =========================================================================
    # 训练参数 (Training Parameters)
    # =========================================================================
    
    # 总训练轮数 (Epochs)
    EPOCH = 1000
    
    # 每轮包含的更新步数 (Steps per Epoch)
    # 每轮训练中，策略网络会更新这么多次
    STEP_PER_EPOCH = 100
    
    # 每次更新前收集的环境步数 (Collect Steps per Update)
    # 增加此值可以提高吞吐量，但可能降低样本效率
    COLLECT_PER_STEP = 10
    STEP_PER_COLLECT = 10 # Alias for compatibility
    
    # 每次更新的梯度步数 (Updates per Step)
    UPDATE_PER_STEP = 0.1
    
    # 测试时的 Episode 数量
    TEST_NUM = 10
    
    # 批次大小 (Batch Size)
    # 每次梯度下降使用的样本数量
    BATCH_SIZE = 64

    
    # 经验回放缓冲区大小 (Replay Buffer Size)
    # 存储历史经验的最大数量
    BUFFER_SIZE = int(3e6)
    
    # 学习率 (Learning Rate)
    # 控制参数更新的步长
    LR = 3e-4
    LR_ACTOR = 3e-4      # Actor specific
    LR_CRITIC = 3e-4     # Critic specific
    
    # 折扣因子 (Gamma)
    # 0.0 表示只关注即时奖励 (Contextual Bandit 设置)
    # 接近 1.0 表示关注长期累积奖励
    GAMMA = 0.0
    
    # N-Step Returns
    N_STEP = 1
    
    # Sparsity Regularization (For specific custom policies)
    SPARSITY_COEF = 0.0

    # 软更新系数 (Tau)
    # 用于目标网络 (Target Network) 的平滑更新
    TAU = 0.005
    
    # 奖励归一化 (Reward Normalization)
    # 是否开启 Tianshou 自带的奖励归一化 (基于移动平均)
    # 这有助于稳定训练，特别是当奖励值规模较大时
    REWARD_NORMALIZATION = True
    
    # =========================================================================
    # 算法特定参数 (Algorithm Specific)
    # =========================================================================
    
    # --- TD3 / DDPG ---
    # 探索噪声 (Exploration Noise)
    # 在动作中添加的高斯噪声标准差，用于促进探索
    # 对于复杂的优化问题，较小的噪声 (0.1) 可能更稳定
    EXPLORATION_NOISE = 0.1
    
    # --- TD3 Only ---
    # 策略噪声 (Policy Noise)
    # 在计算目标动作时添加的噪声，用于平滑价值估计
    POLICY_NOISE = 0.2
    
    # 噪声截断 (Noise Clip)
    # 限制策略噪声的范围 [-c, c]
    NOISE_CLIP = 0.5
    
    # Actor 更新频率
    # Critic 更新多少次后，才更新一次 Actor (延迟更新策略)
    UPDATE_ACTOR_FREQ = 2
    
    # --- SAC Only ---
    # 熵正则化系数 (Alpha)
    # 控制探索与利用的平衡
    ALPHA = 0.2
    
    # 自动调整 Alpha
    # 是否自动学习最佳的熵系数
    AUTO_ALPHA = True

    # --- Diffusion Only ---
    # 扩散步数 (Timesteps)
    # 5步通常足够用于优化问题，且推理速度快
    DIFFUSION_STEPS = 5
    
    # Beta 调度策略 ('linear', 'cosine', 'vp')
    DIFFUSION_BETA_SCHEDULE = 'vp'
    
    # 学习率衰减
    # 有助于模型收敛到更优解
    LR_DECAY = True
    
    # 学习率衰减最大步数
    LR_MAXT = 200000
    
    # 行为克隆系数 (Behavior Cloning Coefficient)
    # 如果为 True，则使用 BC Loss
    BC_COEF = False
    
    # =========================================================================
    # 预训练与预热配置 (Pretrain & Warmup)
    # =========================================================================
    
    # --- 1. Critic Warmup (随机数据阶段) ---
    # 采集多少步随机数据用于 Warmup
    WARMUP_STEPS = 100000
    
    # 使用随机数据训练 Critic 多少轮 (Epochs)
    # 1 Epoch = 遍历一次随机数据集
    WARMUP_EPOCHS = 100
    
    # --- 2. Actor Pretrain (演示数据阶段) ---
    # 采集多少条演示数据 (Episodes)
    # 使用遗传算法 (GA) 生成高质量数据
    PRETRAIN_EPISODES = 100000
    
    # 使用演示数据监督学习训练 Actor 多少轮 (Epochs)
    PRETRAIN_EPOCHS = 100
