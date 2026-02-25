"""
Diffusion Model Benchmark Script
================================
Runs the Diffusion Policy on three benchmark scenarios:
1. Quadratic Optimization (Convex)
2. Rastrigin Function (Non-Convex)
3. Pendulum (Classic Control)

This script overrides RunConfig settings to use the 'diffusion' algorithm.
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.run_config import RunConfig
from scripts.train_model import train_model

def run_diffusion_benchmarks():
    # ==========================================
    # Global Diffusion Settings
    # ==========================================
    print(">>> Configuring Global Settings for Diffusion Benchmark")
    RunConfig.ALGO = 'diffusion'
    RunConfig.BACKBONE = 'mlp'  # Standard backbone for diffusion noise model
    
    # Training Hyperparameters for Diffusion
    RunConfig.LR_ACTOR = 1e-4   # Diffusion often benefits from lower LR
    RunConfig.LR_CRITIC = 3e-4
    RunConfig.BATCH_SIZE = 256  # Larger batch size often helps diffusion
    RunConfig.HIDDEN_DIM = 256
    
    # Training Duration
    RunConfig.EPOCH = 50        # Number of epochs for benchmark
    RunConfig.STEP_PER_EPOCH = 500
    RunConfig.COLLECT_PER_STEP = 1
    
    # Environment Interaction
    RunConfig.NUM_TRAIN_ENVS = 4
    RunConfig.NUM_TEST_ENVS = 4
    
    # Diffusion Specific Configs
    # These attributes might not be in the original RunConfig class definition,
    # but we inject them here so setup.py can read them via getattr.
    RunConfig.LR_DECAY = True
    RunConfig.BC_COEF = False   # Pure RL (maximize Q), set True for Behavior Cloning if expert data exists
    RunConfig.EXPLORATION_NOISE = 0.1 # Not strictly used by diffusion but good to have
    
    # ==========================================
    # Scenarios
    # ==========================================
    scenarios = [
        {
            "name": "Quadratic Optimization (Convex)",
            "exp_id": "exp_optimization_benchmark",
            "run_id": "diffusion_quadratic_v1",
            "gamma": 0.95 # Optimization tasks usually have lower gamma or 0
        },
        {
            "name": "Rastrigin Function (Non-Convex)",
            "exp_id": "exp_optimization_nonconvex",
            "run_id": "diffusion_rastrigin_v1",
            "gamma": 0.95
        },
        {
            "name": "Pendulum (Control)",
            "exp_id": "exp_classic_pendulum",
            "run_id": "diffusion_pendulum_v1",
            "gamma": 0.99 # Control tasks need higher gamma
        }
    ]

    # ==========================================
    # Execution Loop
    # ==========================================
    for sc in scenarios:
        print(f"\n\n{'='*60}")
        print(f"STARTING BENCHMARK: {sc['name']}")
        print(f"Experiment ID: {sc['exp_id']}")
        print(f"{'='*60}\n")
        
        # Configure Run
        RunConfig.EXPERIMENT_ID = sc['exp_id']
        RunConfig.RUN_ID = sc['run_id']
        RunConfig.GAMMA = sc['gamma']
        
        # Reset specific configs if needed between runs
        # (e.g. if one env needs specific settings)
        
        try:
            train_model()
        except Exception as e:
            print(f"!!! Error running {sc['name']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_diffusion_benchmarks()
