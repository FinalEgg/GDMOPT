import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from .backbones.deepsets import DeepSetsEncoder
from .backbones.gnn import GNNEncoder

class DeepSetsActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.encoder = DeepSetsEncoder(state_dim, action_dim, hidden_dim)
        
        # Calculate action per UAV
        if self.encoder.num_uavs > 0:
            self.action_per_uav = int(action_dim / self.encoder.num_uavs)
        else:
            self.action_per_uav = action_dim # Fallback
            
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_per_uav),
            nn.Sigmoid() # Output in [0, 1]
        )
        
    def forward(self, obs, state=None, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
            
        feat = self.encoder(obs) # (B, N, Hidden)
        action = self.head(feat) # (B, N, A_i)
        return action.view(obs.shape[0], -1), state

class GNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.encoder = GNNEncoder(state_dim, action_dim, hidden_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_linear = nn.Linear(hidden_dim, 1)
        self.log_std_linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        feat = self.encoder(state) # (B, N, M, 3*E)
        dec_out = self.decoder(feat)
        
        mean = self.mean_linear(dec_out)
        log_std = self.log_std_linear(dec_out)
        
        batch_size = state.shape[0]
        mean = mean.view(batch_size, -1)
        log_std = log_std.view(batch_size, -1)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = (y_t + 1) / 2
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean, std, None # Tianshou expects 5 values

class DeepSetsActorProb(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.encoder = DeepSetsEncoder(state_dim, action_dim, hidden_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Calculate action per UAV
        if self.encoder.num_uavs > 0:
            self.action_per_uav = int(action_dim / self.encoder.num_uavs)
        else:
            self.action_per_uav = action_dim # Fallback
            
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_linear = nn.Linear(hidden_dim, self.action_per_uav)
        self.log_std_linear = nn.Linear(hidden_dim, self.action_per_uav)
        
    def forward(self, obs, state=None, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
            
        feat = self.encoder(obs) # (B, N, Hidden)
        dec_out = self.decoder(feat)
        
        mean = self.mean_linear(dec_out)
        log_std = self.log_std_linear(dec_out)
        
        batch_size = obs.shape[0]
        mean = mean.view(batch_size, -1)
        log_std = log_std.view(batch_size, -1)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        return (mean, std), state
