import numpy as np
import os

path = 'log/random_data.npz'
if os.path.exists(path):
    data = np.load(path)
    rew = data['rew']
    print(f"Reward Stats: Mean={np.mean(rew):.4f}, Std={np.std(rew):.4f}, Min={np.min(rew):.4f}, Max={np.max(rew):.4f}")
else:
    print("File not found.")
