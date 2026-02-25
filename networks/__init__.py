"""
Neural Network Module.
神经网络模块。

Structure (结构):
- blocks/: Base architectures (MLP, DeepSets) (基础架构模块).
- wrappers/: Functional wrappers (Heads, Diffusion) (功能包装器).
- factory.py: Central interface for model creation (模型创建工厂).

Exports (导出):
- get_model: Unified entry point (统一入口).
- Blocks and Heads for custom assembly.
"""

from .factory import ModelFactory, get_model
from .blocks import MLPBlock, DeepSetsBlock
from .wrappers import DeterministicHead, StochasticHead, CriticHead, DiffusionWrapper

__all__ = [
    "ModelFactory",
    "get_model",
    "MLPBlock",
    "DeepSetsBlock",
    "DeterministicHead", 
    "StochasticHead", 
    "CriticHead",
    "DiffusionWrapper"
]
