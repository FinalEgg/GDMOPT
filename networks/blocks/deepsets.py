import torch
import torch.nn as nn
from .mlp import MLPBlock

class DeepSetsBlock(nn.Module):
    """
    DeepSets Permutation Invariant Block.
    DeepSets 置换不变性模块。
    
    Theory (原理):
    - f(X) = rho( aggregation( phi(x) for x in X ) )
    - Handles input sets of variable size (though typically fixed N in RL).
    - Guarantees that the output is invariant to the order of elements in input.
    - 保证输出对输入集合元素的顺序不敏感（置换不变性）。
    
    Structure (结构):
    1. Phi (Local Encoder): Encodes each element independently. (局部编码器)
    2. Aggregation: Pools element features (Mean/Sum/Max). (聚合层)
    3. Rho (Global Decoder): Processes global feature. (全局解码器)
    
    Expected Input (输入):
    - Tensor shape: (Batch, SetSize, FeatDim).
    """
    def __init__(self, input_dim: int, output_dim: int, config: dict):
        """
        Initialize DeepSets.
        
        Args:
            input_dim (int): Feature dimension per element (每个元素的特征数).
            output_dim (int): Final output embedding size.
            config (dict): Configuration dictionary.
                - phi_hidden_sizes: List[int] for Phi network.
                - rho_hidden_sizes: List[int] for Rho network.
                - aggregation: 'mean', 'sum', or 'max'.
        """
        super().__init__()
        
        # DeepSets Specific Configs (DeepSets 特有配置)
        phi_hidden = config.get('phi_hidden_sizes', [128, 128])
        rho_hidden = config.get('rho_hidden_sizes', [128])
        activation = config.get('activation', 'relu')
        self.aggregation_type = config.get('aggregation', 'mean')
        
        self.feat_dim = input_dim 
        
        # 1. Phi Network (Applied to each element) (Phi 网络: 逐元素处理)
        # Input: (Batch * SetSize, FeatDim) -> Output: (Batch * SetSize, Latent)
        phi_config = {
            'hidden_sizes': phi_hidden,
            'activation': activation,
            'use_layer_norm': config.get('use_layer_norm', False),
            'dropout': config.get('dropout', 0.0)
        }
        self.phi = MLPBlock(input_dim, phi_hidden[-1], phi_config)
        self.latent_dim = phi_hidden[-1]
        
        # 2. Rho Network (Applied to aggregated latent) (Rho 网络: 全局处理)
        # Input: (Batch, Latent) -> Output: (Batch, OutputDim)
        rho_config = {
            'hidden_sizes': rho_hidden,
            'activation': activation,
            'use_layer_norm': config.get('use_layer_norm', False),
            'dropout': config.get('dropout', 0.0)
        }
        self.rho = MLPBlock(self.latent_dim, output_dim, rho_config)
        self.output_dim = self.rho.output_dim

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (Batch, SetSize, FeatDim).
            
        Returns:
            out: (Batch, OutputDim).
        """
        if x.dim() < 3:
            # Safety check: Caller must reshape flat vectors to 3D sets before calling this.
            raise ValueError(f"DeepSetsBlock expects 3D input (Batch, SetSize, FeatDim), got {x.shape}")
            
        b, n, f = x.shape
        
        # 1. Apply Phi to each element (应用局部编码)
        # Flatten batch and set dims to apply MLP
        x_flat = x.reshape(b * n, f)
        phi_out = self.phi(x_flat)
        # Reshape back to set structure
        phi_out = phi_out.reshape(b, n, -1)
        
        # 2. Aggregation (聚合)
        # Reduces along dimension 1 (Set Dimension)
        if self.aggregation_type == 'mean':
            agg_out = phi_out.mean(dim=1)
        elif self.aggregation_type == 'sum':
            agg_out = phi_out.sum(dim=1)
        elif self.aggregation_type == 'max':
            agg_out = phi_out.max(dim=1)[0]
        else:
             # Default mean
             agg_out = phi_out.mean(dim=1)
             
        # 3. Apply Rho (应用全局解码)
        out = self.rho(agg_out)
        
        return out
