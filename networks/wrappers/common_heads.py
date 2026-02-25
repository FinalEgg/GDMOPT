import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

class DeterministicHead(nn.Module):
    """
    Deterministic Policy Head (e.g. for TD3, DDPG).
    确定性策略头，用于 DDPG/TD3 等输出确定动作的算法。
    
    Structure (结构):
    - Input: Extracted Features (from Backbone). (骨干网络提取的特征)
    - Output: Action values in [-1, 1] (via Tanh). (Tanh 激活输出动作)
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize Head.
        
        Args:
            backbone (nn.Module): Pre-trained or initialized feature extractor.
            feature_dim (int): Input dimension from backbone.
            action_dim (int): Output action dimension.
            hidden_dim (int): Hidden layer size in the head.
        """
        super().__init__()
        self.backbone = backbone
        
        # Simple head (简单的多层感知机头)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Map to [-1, 1] standard RL action range
        )
        
    def forward(self, obs, state=None, info={}):
        """
        Forward pass.
        
        Args:
            obs: Observation tensor or numpy array.
            state: Hidden state (RNN only, ignored here).
            
        Returns:
            act: Action tensor (Batch, ActionDim).
            state: Hidden state (None).
        """
        # Handle numpy input from Tianshou (处理 Tianshou 传入的 numpy 数据)
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
            
        features = self.backbone(obs)
        act = self.head(features)
        
        # Tianshou Actor expects returns: logits, state
        return act, state

class StochasticHead(nn.Module):
    """
    Stochastic Gaussian Policy Head (e.g. for SAC).
    随机高斯策略头，用于 SAC 等算法。
    
    Structure (结构):
    - Independent/Shared layers for Mu (Mean) and LogStd (Std Dev).
    - Output: (Mu, Sigma) tuple.
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, action_dim: int, hidden_dim: int = 128, 
                 log_std_min=-20, log_std_max=2):
        """
        Initialize Stochastic Head.
        
        Args:
            backbone (nn.Module): Feature extractor.
            feature_dim (int): Input size.
            action_dim (int): Action size.
            hidden_dim (int): Hidden size.
            log_std_min (float): Min log standard deviation (for numeric stability).
            log_std_max (float): Max log standard deviation.
        """
        super().__init__()
        self.backbone = backbone
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared intermediate layer (共享中间层)
        self.common = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for Mean and LogStd (分离输出头)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, state=None, info={}):
        """
        Forward pass.
        
        Returns:
            (mu, sigma): Tuple of tensors.
            state: None.
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
            
        features = self.backbone(obs)
        x = self.common(features)
        
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        
        # Clamp log_std to prevent numerical explosion (裁剪 LogStd)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        # Tianshou ActorProb expects: (mu, sigma), state
        return (mu, log_std.exp()), state

class CriticHead(nn.Module):
    """
    Standard Critic Head (Q-Network).
    标准评论家头 (Q 网络)。
    
    Function (功能):
    - Estimates Q(s, a).
    - Handles concatenation of State and Action input.
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, hidden_dim: int = 128):
        """
        Initialize Critic Head.
        
        Args:
            backbone (nn.Module): Feature extractor (initialized with state_dim + action_dim).
            feature_dim (int): Output dimension of backbone.
            hidden_dim (int): Hidden layer size.
        """
        super().__init__()
        self.backbone = backbone
        
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Single scalar Q-value (输出单一 Q 值)
        )
        
    def forward(self, obs, act=None, info={}):
        """
        Forward pass.
        
        Args:
            obs: Observation.
            act: Action (optional, but required for Q(s, a)).
        """
        # Tensor conversion
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        if act is not None and isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=next(self.parameters()).device)
            
        # Concat Obs and Act (拼接状态和动作)
        if act is not None:
            # Flatten if necessary (e.g. if obs is image or deep set, handling might be different)
            # Here we assume standard vector concatenation for MLP backbone
            if obs.dim() > 2: obs = obs.reshape(obs.size(0), -1)
            if act.dim() > 2: act = act.reshape(act.size(0), -1)
            x = torch.cat([obs, act], dim=1)
        else:
            x = obs
            
        # Pass through backbone and head
        features = self.backbone(x)
        q_val = self.head(features)
        
        return q_val
