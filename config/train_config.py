import torch

class TrainConfig:
    """
    训练配置类 (Training Configuration)。
    这些值作为 scripts/train.py 中命令行参数的默认值。
    您可以在这里修改默认设置，也可以通过命令行参数覆盖它们。
    """
    # =========================================================================
    # 通用设置 (General Settings)
    # =========================================================================
    
    # 环境名称
    # 可选: 'fix_topp' (固定BS位置+TopP连接), 'cellfree', 'topp', 'topk', 'quadratic', 'rastrigin'
    ENV = 'fix_topp'
    
    # 强化学习算法
    # 可选: 'td3' (推荐), 'ddpg', 'sac'
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
    EPOCH = 2000
    
    # 每轮包含的更新步数 (Steps per Epoch)
    # 每轮训练中，策略网络会更新这么多次
    STEP_PER_EPOCH = 2000
    
    # 每次更新前收集的环境步数 (Collect Steps per Update)
    # 增加此值可以提高吞吐量，但可能降低样本效率
    COLLECT_PER_STEP = 16
    
    # 批次大小 (Batch Size)
    # 每次梯度下降使用的样本数量
    BATCH_SIZE = 256
    
    # 经验回放缓冲区大小 (Replay Buffer Size)
    # 存储历史经验的最大数量
    BUFFER_SIZE = int(3e6)
    
    # 学习率 (Learning Rate)
    # 控制参数更新的步长
    LR = 3e-4
    
    # 折扣因子 (Gamma)
    # 0.0 表示只关注即时奖励 (Contextual Bandit 设置)
    # 接近 1.0 表示关注长期累积奖励
    GAMMA = 0.0
    
    # 软更新系数 (Tau)
    # 用于目标网络 (Target Network) 的平滑更新
    TAU = 0.005
    
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
    # 预训练阶段 (Pretraining Stages)
    # =========================================================================
    
    # Critic 预热步数
    # 在正式训练前，使用随机数据预训练 Critic 的步数
    PRETRAIN_CRITIC_STEPS = 200000
    
    # Actor 监督学习轮数
    # 注意：这里的 Epoch 是监督学习的概念，指遍历一次完整的预训练数据集。
    PRETRAIN_ACTOR_EPOCHS = 10000
