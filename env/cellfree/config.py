# Configuration parameters for Cell-free UAV environment

# Environment dimensions
X = 1000  # Length of the area (meters)
Y = 1000  # Width of the area (meters)
H = 100   # Height of the area (meters)

# Network components
M = 4    # Number of ground base stations
N = 3    # Number of UAVs

# Power parameters
P = 10   # Maximum transmit power per base station (Watts)

# Other parameters
NOISE_POWER = 1e-9  # Noise power (Watts)
CARRIER_FREQUENCY = 2.4e9  # Carrier frequency (Hz)
PATH_LOSS_EXPONENT = 2.0   # Path loss exponent

# Episode parameters
STEPS_PER_EPISODE = 100