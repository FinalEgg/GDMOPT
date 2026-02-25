"""
Policy Module.
策略模块。

Contains custom RL policies integrated with Tianshou.
包含集成到 Tianshou 的自定义强化学习策略。

Exports:
- CustomDDPGPolicy: DDPG with sparsity support.
- CustomTD3Policy: TD3 with sparsity support.
- CustomSACPolicy: SAC with sparsity support.
- DiffusionOPT (CustomDiffusionPolicy): Diffusion-based policy.
"""

from .custom_ddpg import CustomDDPGPolicy
from .custom_sac import CustomSACPolicy
from .custom_td3 import CustomTD3Policy
from .custom_diffusion import DiffusionOPT as CustomDiffusionPolicy

__all__ = [
    "CustomDDPGPolicy",
    "CustomSACPolicy",
    "CustomTD3Policy",
    "CustomDiffusionPolicy"
]
