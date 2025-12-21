import torch
import torch.nn as nn
from .backbones.deepsets import DeepSetsEncoder

class DeepSetsDoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Infer dims (Same as DeepSetsDiffusionModel)
        num_uavs = int((state_dim - 5 * action_dim) / 3)
        if num_uavs > 0 and (state_dim - 5 * action_dim) % 3 == 0:
             num_bs = int(action_dim / num_uavs)
             state_per_uav = 5 * num_bs + 3
             action_per_uav = num_bs
        else:
            num_uavs = int((state_dim - 2 * action_dim) / 3)
            if num_uavs > 0 and (state_dim - 2 * action_dim) % 3 == 0:
                num_bs = int(action_dim / num_uavs)
                state_per_uav = 2 * num_bs + 3
                action_per_uav = num_bs
            else:
                 num_uavs = 1
                 state_per_uav = state_dim
                 action_per_uav = action_dim
        
        self.num_uavs = num_uavs
        self.state_per_uav = state_per_uav
        self.action_per_uav = action_per_uav
        self.combined_per_uav = state_per_uav + action_per_uav
        
        # Encoder
        self.local_encoder = nn.Sequential(
            nn.Linear(self.combined_per_uav, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Heads
        self.q1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        batch_size = state.shape[0]
        
        state_reshaped = state.view(batch_size, self.num_uavs, self.state_per_uav)
        action_reshaped = action.view(batch_size, self.num_uavs, self.action_per_uav)
        
        combined = torch.cat([state_reshaped, action_reshaped], dim=2)
        
        local_feat = self.local_encoder(combined)
        global_feat = torch.max(local_feat, dim=1)[0] # (B, Hidden)
        
        q1 = self.q1_head(global_feat)
        q2 = self.q2_head(global_feat)
        
        return q1, q2

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
