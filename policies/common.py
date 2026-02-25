import torch
from typing import Dict, Any

class SparsityMixin:
    """
    Sparsity Regularization Mixin.
    稀疏度正则化混入类。

    Provides functionality to add L1 regularization to actions during training, 
    encouraging sparsity in the policy's output.
    提供在训练期间向动作添加 L1 正则化的功能，鼓励策略输出的稀疏性。
    """
    
    def __init__(self, sparsity_coef: float = 0.0):
        """
        Initialize the SparsityMixin.
        初始化 SparsityMixin。

        Args:
        -----
        sparsity_coef : float
            Coefficient for L1 regularization (default: 0.0).
            L1 正则化系数 (默认: 0.0)。
        """
        self.sparsity_coef = sparsity_coef

    def _get_sparsity_loss(self, act: torch.Tensor) -> torch.Tensor:
        """
        Calculate L1 sparsity loss on actions.
        计算动作的 L1 稀疏损失。

        Args:
        -----
        act : torch.Tensor
            Action tensor of shape (Batch, ActionDim).
            形状为 (Batch, ActionDim) 的动作张量。

        Returns:
        --------
        torch.Tensor
            The calculated sparsity loss.
            计算得到的稀疏损失。
        """
        if self.sparsity_coef <= 1e-9:
            return torch.tensor(0.0, device=act.device)
        
        return self.sparsity_coef * torch.mean(torch.abs(act))

    def _log_sparsity_metrics(self, result: Dict[str, Any], act: torch.Tensor):
        """
        Log sparsity related metrics to the result dictionary.
        将稀疏性相关指标记录到结果字典中。

        Args:
        -----
        result : Dict[str, Any]
            Dictionary to store metrics.
            存储指标的字典。
        act : torch.Tensor
            Action tensor.
            动作张量。
        """
        if self.sparsity_coef > 0:
            result["loss/sparsity"] = self._get_sparsity_loss(act).item()
            result["act/l1_norm"] = torch.mean(torch.abs(act)).item()

