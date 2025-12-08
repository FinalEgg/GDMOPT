
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from env.cellfree.config import M, N
from model.sac.sac import Actor

print(f"Current M={M}, N={N}")

model_path = r'd:\Code\GDMOPT\log\combined\sac\cellfree\Dec08-011103\pretrain_policy.pth'
print(f"Loading {model_path}")

try:
    ckpt = torch.load(model_path, map_location='cpu')
    print("Checkpoint loaded.")
    
    actor = Actor(100, 100) # dims are ignored in current Actor
    
    state_dict = None
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'actor' in ckpt:
            state_dict = ckpt['actor']
        else:
            state_dict = ckpt
            
    if state_dict:
        print("State dict found. Keys:", list(state_dict.keys())[:5])
        try:
            actor.load_state_dict(state_dict, strict=True)
            print("Loaded with strict=True")
        except Exception as e:
            print(f"Strict load failed: {e}")
            actor.load_state_dict(state_dict, strict=False)
            print("Loaded with strict=False")
    else:
        print("No state dict found")

except Exception as e:
    print(f"Failed: {e}")
