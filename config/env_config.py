
class EnvConfig:
    """
    全局环境配置参数 (Global Environment Configuration).
    定义适用于大多环境的共享物理参数。
    注意:
    - 功率单位均为瓦特 (Watts)，而非 dBm。
    - 距离单位均为米 (Meters)。
    """
    # Environment dimensions (Area Size)
    X = 10000.0  # Length of the area (meters) [10km]
    Y = 10000.0  # Width of the area (meters) [10km]
    H = 300.0    # UAV flight height (meters)

    # Network components
    M = 10      # Number of ground base stations
    N = 5       # Number of UAVs

    # Power parameters
    P = 50.0    # Maximum transmit power per base station (Watts)
    pd = P      # Downlink transmit power per BS (Watts)
    pu = 5.0    # Pilot transmit power per UAV (Watts)
    NOISE_POWER = 1e-18 # Noise power (Watts)

    # Channel parameters
    TAU_P = 20  # Pilot sequence length (symbols)
    ALPHA1 = 1.2 # LoS path loss exponent
    ALPHA2 = 2.1 # NLoS path loss exponent
    XI1 = 25.0   # LoS probability parameter
    XI2 = 0.1   # LoS probability parameter

    # Reward parameters
    CAPACITY_THRESHOLD = 15.0 
    FIXED_REWARD = 1.0
    
    # Geometric Reward Parameters
    GEO_BETA_THRESHOLD = 1e-18
    GEO_REWARD_HIT = 0.15
    GEO_PENALTY_MISS = 0.02
    GEO_PENALTY_USELESS = 0.05
    GEO_BONUS_PERFECT = 0.5
    GEO_PENALTY_WRONG = 0.01
    GEO_PENALTY_NO_CONNECT = 0.1

    # Episode parameters
    STEPS_PER_EPISODE = 1000
    
    # Optimization Benchmark Parameters
    OPT_DIM = 2
    OPT_BOUNDS = 10.0
    
    # Top-K specific
    K_MAX = 5
    
    # Top-P specific
    TOP_P_THRESHOLD = 0.75
    
    # Capacity Reward Type
    # 'threshold_fixed': Reward = FIXED_REWARD if Capacity > THRESHOLD
    # 'sum_capacity': Reward = Sum(Capacity)
    # 调参建议: 使用 sum_capacity 以获得更平滑的梯度，避免 threshold 模式下的稀疏奖励问题
    CAPACITY_REWARD_TYPE = 'sum_capacity' 
    
    # Reward Scaling
    # 将奖励放大以便于神经网络训练 (建议目标范围在 1.0 ~ 10.0 之间)
    REWARD_SCALE = 10

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(v)}
