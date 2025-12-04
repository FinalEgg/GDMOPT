# Configuration parameters for Cell-free UAV environment

# Environment dimensions
X = 100  # Length of the area (meters)
Y = 100  # Width of the area (meters)
H = 50   # Height of the area (meters)

# Network components
M = 5    # Number of ground base stations
N = 5    # Number of UAVs

# Power parameters
P = 50   # Maximum transmit power per base station (Watts)
pd = P   # Downlink transmit power per BS (Watts)
pu = 0.1 # Pilot transmit power per UAV (Watts)

# Channel parameters
TAU_P = 10  # Pilot sequence length (symbols)
ALPHA1 = 2.0 # LoS path loss exponent
ALPHA2 = 4.0 # NLoS path loss exponent
XI1 = 9.6 # LoS probability parameter
XI2 = 0.28 # LoS probability parameter

# Reward parameters
CAPACITY_THRESHOLD = 0.1  # Threshold for UAV channel capacity (bits/symbol)
REWARD_VALUE = 1.0        # Reward value when capacity exceeds threshold
CONNECTION_COST = 0.05    # Penalty for each active connection (power > 0.01)
REWARD_SCALE = 20.0       # Scale factor for the final reward

# Soft Gate parameters
GATE_K = 20.0             # Steepness of the sigmoid gate (Not used in new logic)
GATE_TH = 0.2             # Threshold for the power activation (ReLU-like)

# Other parameters
NOISE_POWER = 1e-9  # Noise power (Watts)
CARRIER_FREQUENCY = 2.4e9  # Carrier frequency (Hz)
PATH_LOSS_EXPONENT = 2.0   # Path loss exponent

# Episode parameters
STEPS_PER_EPISODE = 1