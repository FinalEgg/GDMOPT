
import numpy as np
import matplotlib.pyplot as plt

# Constants from config.py
ALPHA1 = 2.0 # LoS path loss exponent
ALPHA2 = 4.0 # NLoS path loss exponent
XI1 = 9.6 # LoS probability parameter
XI2 = 0.28 # LoS probability parameter
GEO_BETA_THRESHOLD = 2e-5

def calculate_beta(dist, height):
    # dist is the 3D distance
    # height is the relative height (z difference)
    
    # Avoid division by zero or invalid arcsin
    if dist < height:
        dist = height
        
    # Calculate elevation angle (degrees)
    # sin(theta) = height / dist
    theta_rad = np.arcsin(height / dist)
    theta_deg = np.degrees(theta_rad)
    
    # LoS Probability
    P_LoS = 1 / (1 + XI1 * np.exp(-XI2 * (theta_deg - XI1)))
    
    # Path Loss (Beta)
    # Note: The formula in env.py is beta = P * D^-a1 + (1-P) * D^-a2
    # This is actually Gain, not Loss in dB. It's the linear channel gain.
    beta = P_LoS * (dist ** -ALPHA1) + (1 - P_LoS) * (dist ** -ALPHA2)
    
    return beta, P_LoS

def find_distance_for_threshold(threshold, height=25.0):
    # Search for distance where beta crosses threshold
    # Beta decreases as distance increases.
    
    d_min = height
    d_max = 1000.0
    
    for _ in range(100):
        d_mid = (d_min + d_max) / 2
        beta, _ = calculate_beta(d_mid, height)
        
        if beta > threshold:
            d_min = d_mid
        else:
            d_max = d_mid
            
    return d_max

if __name__ == "__main__":
    # Assume typical UAV height
    h_uav = 25.0 
    
    dist_threshold = find_distance_for_threshold(GEO_BETA_THRESHOLD, h_uav)
    
    print(f"Threshold Beta: {GEO_BETA_THRESHOLD}")
    print(f"Assumed UAV Height: {h_uav} m")
    print(f"Calculated 3D Distance: {dist_threshold:.2f} m")
    
    # Also check horizontal distance
    r_horizontal = np.sqrt(dist_threshold**2 - h_uav**2)
    print(f"Corresponding Horizontal Distance: {r_horizontal:.2f} m")
    
    # Check Beta at this distance to verify
    beta_val, p_los = calculate_beta(dist_threshold, h_uav)
    print(f"Beta at {dist_threshold:.2f}m: {beta_val:.2e}")
    print(f"LoS Probability at this point: {p_los:.4f}")
    
    # Check pure LoS and NLoS cases for reference
    d_los = (GEO_BETA_THRESHOLD)**(-1/ALPHA1)
    d_nlos = (GEO_BETA_THRESHOLD)**(-1/ALPHA2)
    print(f"\nReference (Theoretical limits):")
    print(f"Pure LoS Distance (D^-{ALPHA1}): {d_los:.2f} m")
    print(f"Pure NLoS Distance (D^-{ALPHA2}): {d_nlos:.2f} m")
