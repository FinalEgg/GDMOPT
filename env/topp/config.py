
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
pd = P      # 下行发射功率 (Watts)
pu = 0.1    # 上行导频功率 (Watts)
NOISE_POWER = 1e-13 # 噪声功率 (Watts) ~ -100dBm
FC = 2e9      # 载波频率 2GHz

# 信道参数
TAU_P = 10  # 导频序列长度
ALPHA1 = 2.0 # LoS 路径损耗指数
ALPHA2 = 4.0 # NLoS 路径损耗指数
XI1 = 9.61   # LoS 概率参数
XI2 = 0.16   # LoS 概率参数

# 奖励阈值 (根据分析结果调整: Random~2.0-9.0)
# 降低阈值以避免梯度消失，让 Agent 在初期能获得正向反馈
CAPACITY_THRESHOLD = 5.0 
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
