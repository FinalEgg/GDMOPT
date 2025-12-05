
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.cellfree.env import CellFreeEnv

def check_beta():
    env = CellFreeEnv()
    env.reset()
    
    betas = env.beta_matrix.flatten()
    print(f"Beta Stats:")
    print(f"Min: {betas.min():.2e}")
    print(f"Max: {betas.max():.2e}")
    print(f"Mean: {betas.mean():.2e}")
    print(f"Median: {np.median(betas):.2e}")
    
    # Check percentiles
    print(f"10th percentile: {np.percentile(betas, 10):.2e}")
    print(f"90th percentile: {np.percentile(betas, 90):.2e}")
    
    # Check what 1% of max looks like
    max_beta = betas.max()
    print(f"1% of Max ({max_beta:.2e}): {max_beta * 0.01:.2e}")

if __name__ == "__main__":
    check_beta()
