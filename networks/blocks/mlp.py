import torch
import torch.nn as nn

def get_activation(act_name: str) -> nn.Module:
    """
    Get activation module by name.
    根据名称获取激活函数模块。
    
    Args:
        act_name (str): 'relu', 'tanh', 'gelu', 'elu', 'leaky_relu', 'sigmoid'.
        
    Returns:
        nn.Module: Activation layer.
    """
    name = act_name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        # Default to ReLU if unknown
        return nn.ReLU()

class ResidualBlock(nn.Module):
    """
    Residual Linear Block.
    残差线性模块。
    
    Structure: Output = Input + Dropout(Act(LayerNorm(Linear(Input))))
    结构：x + f(x)，用于深层网络训练稳定性。
    """
    def __init__(self, size, activation, dropout=0.0, use_layer_norm=True):
        """
        Initialize Residual Block.
        
        Args:
            size (int): Input and Output dimension (输入输出维度相同).
            activation (nn.Module): Activation instance (激活函数).
            dropout (float): Dropout probability.
            use_layer_norm (bool): Whether to use LayerNorm.
        """
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.act = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(size) if use_layer_norm else nn.Identity()
        
    def forward(self, x):
        """Forward pass with skip connection."""
        return x + self.dropout(self.act(self.ln(self.fc(x))))

class MLPBlock(nn.Module):
    """
    Multilayer Perceptron (MLP) Block.
    多层感知机与骨干网络模块。
    
    Function (功能):
    - Generic feed-forward network construction (通用前馈网络构建).
    - Supports Residual connections, LayerNorm, Dropout (支持残差、层归一化、Dropout).
    
    Configuration (配置 - config dict):
    - hidden_sizes (list[int]): e.g., [256, 256].
    - activation (str): Activation function name.
    - use_layer_norm (bool): Enable LayerNorm.
    - use_residual (bool): Enable Residual connections (dimension must match).
    - dropout (float): Dropout rate.
    
    Usage (用法):
    - As a Backbone: output_dim=None, returns features.
    - As a Head: output_dim=action_dim, returns logic/action.
    """
    def __init__(self, input_dim: int, output_dim: int, config: dict):
        """
        Initialize MLP.
        
        Args:
            input_dim (int): Input vector size.
            output_dim (int/None): Output vector size. If None, uses last hidden size.
            config (dict): Model configuration dictionary.
        """
        super().__init__()
        
        # Config Extraction (配置解析)
        # Default to [256, 256] if not specified
        hidden_sizes = config.get('hidden_sizes', [256, 256])
        activation_name = config.get('activation', 'relu')
        use_ln = config.get('use_layer_norm', False)
        use_res = config.get('use_residual', False)
        dropout = config.get('dropout', 0.0)
        
        layers = []
        in_size = input_dim
        
        # Build Hidden Layers (构建隐藏层)
        for h_size in hidden_sizes:
            if use_res and in_size == h_size:
                # Use Residual Block if dimensions match (仅当维度一致时才能使用残差)
                layers.append(ResidualBlock(
                    h_size, 
                    get_activation(activation_name), 
                    dropout, 
                    use_ln
                ))
            else:
                # Standard Linear Block (标准线性层组合)
                layers.append(nn.Linear(in_size, h_size))
                if use_ln:
                    layers.append(nn.LayerNorm(h_size))
                layers.append(get_activation(activation_name))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            in_size = h_size
            
        # Build Output Layer (构建输出层)
        if output_dim is not None and output_dim > 0:
            layers.append(nn.Linear(in_size, output_dim))
            self.output_dim = output_dim
        else:
            # If no specific output dim, acts as feature extractor
            # 如果未指定输出维度，则视作特征提取器
            self.output_dim = in_size
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass."""
        return self.net(x)
