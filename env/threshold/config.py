
import numpy as np

# 场景参数 (Realistic Scale)
X = 1000.0  # 区域长度 (m)
Y = 1000.0  # 区域宽度 (m)
H = 100.0   # UAV飞行高度 (m)

# 设备数量
M = 16      # 基站数量 (BS) - 增加密度
N = 5      # 无人机数量 (UAV)

# 通信参数
P = 10.0    # 基站发射功率 (Watts) - Small Cell
NOISE_POWER = 1e-13 # 噪声功率 (Watts) ~ -100dBm
B = 20e6    # 带宽 (Hz) - 20MHz

# 归一化导频功率 (Pilot SNR)
# 假设导频功率与发射功率相同，归一化噪声
pu = P / NOISE_POWER 
pd = 1.0
TAU_P = N # 导频长度通常 >= 用户数

# 信道模型参数 (Urban Micro/Macro)
ALPHA1 = 2.3  # LoS 路径损耗指数
ALPHA2 = 3.7  # NLoS 路径损耗指数
XI1 = 3.0     # LoS 阴影衰落标准差 (dB)
XI2 = 8.0     # NLoS 阴影衰落标准差 (dB)
C = 3e8       # 光速
FC = 2e9      # 载波频率 2GHz

# 奖励阈值 (根据分析结果调整: Random~15.8, Heuristic~19.6)
CAPACITY_THRESHOLD = 17.0 
FIXED_REWARD = 1.0  # Scaled down from 10.0

# 几何奖励参数
STEPS_PER_EPISODE = 1000
GEO_BETA_THRESHOLD = 1e-9
GEO_REWARD_HIT = 0.15      # Increased from 0.1 to encourage hitting best BS
GEO_PENALTY_MISS = 0.02    # Decreased from 0.05 to be less harsh on partial matches
GEO_PENALTY_USELESS = 0.05 
GEO_BONUS_PERFECT = 0.5    
GEO_PENALTY_WRONG = 0.01   
GEO_PENALTY_NO_CONNECT = 0.1
