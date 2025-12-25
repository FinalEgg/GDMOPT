import torch
import torch.nn as nn
import numpy as np
from .backbones.deepsets import DeepSetsEncoder
from .backbones.gnn import GNNEncoder, HGNNLayer

class DeepSetsCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Adjust state_dim to include action
        # Original state_dim = N * (2M + 3)
        # Action dim = N * M
        # We want to feed (State + Action) into the encoder?
        # Or just concat at the end?
        # DeepSets structure is good for state processing.
        # Let's use a simple MLP for Critic if we don't want to complicate things, 
        # but to maintain permutation equivariance, we should use DeepSets.
        
        # Strategy: Concatenate action to state before reshaping?
        # State: (N, 2M+3). Action: (N, M).
        # Combined: (N, 3M+3).
        
        # Robust inference logic
        # Try New Logic First (5 features per BS)
        self.num_uavs = int((state_dim - 5 * action_dim) / 3)
        
        if self.num_uavs > 0 and (state_dim - 5 * action_dim) % 3 == 0:
             self.num_bs = int(action_dim / self.num_uavs)
             self.state_per_uav = 5 * self.num_bs + 3
        else:
            # Fallback to Old Logic (2 features per BS)
            self.num_uavs = int((state_dim - 2 * action_dim) / 3)
            if self.num_uavs > 0 and (state_dim - 2 * action_dim) % 3 == 0:
                self.num_bs = int(action_dim / self.num_uavs)
                self.state_per_uav = 2 * self.num_bs + 3
            else:
                 # Fallback
                 self.num_uavs = 1
                 self.state_per_uav = state_dim
        
        # New "state" dim for the encoder
        combined_dim = state_dim + action_dim
        
        # We need to modify DeepSetsEncoder to accept this combined dim?
        # Or just instantiate it with the larger dim.
        # But DeepSetsEncoder infers N and M from dims.
        # Let's manually calculate the per-uav dim.
        
        self.action_per_uav = self.num_bs
        self.combined_per_uav = self.state_per_uav + self.action_per_uav
        
        self.local_encoder = nn.Sequential(
            nn.Linear(self.combined_per_uav, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act=None, info={}):
        # obs: (B, N * S_i)
        # act: (B, N * A_i)
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        if act is not None and isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=next(self.parameters()).device)
            
        state = obs
        action = act
        batch_size = state.shape[0]
        
        state_reshaped = state.view(batch_size, self.num_uavs, self.state_per_uav)
        action_reshaped = action.view(batch_size, self.num_uavs, self.action_per_uav)
        
        combined = torch.cat([state_reshaped, action_reshaped], dim=2)
        
        local_feat = self.local_encoder(combined)
        
        # Global Pooling (Max + Sum)
        global_max = torch.max(local_feat, dim=1)[0]
        global_sum = torch.sum(local_feat, dim=1)
        global_feat = torch.cat([global_max, global_sum], dim=1) # (B, 2*Hidden)
        
        # Fusion -> (B, N, 3*Hidden)
        global_expanded = global_feat.unsqueeze(1).expand(-1, self.num_uavs, -1)
        fused = torch.cat([local_feat, global_expanded], dim=2)
        
        q_values = self.head(fused) # (B, N, 1)
        return q_values.sum(dim=1) # Sum over UAVs to get total Q

class GNNCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.num_uavs = int((state_dim - 2 * action_dim) / 3)
        self.num_bs = int(action_dim / self.num_uavs)
        self.M = self.num_bs
        self.N = self.num_uavs
        self.embed_dim = hidden_dim
        
        # Encoders
        self.uav_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.bs_embedding = nn.Embedding(self.M, self.embed_dim)
        
        # Edge Encoder takes (LogBeta, Angle, Power) -> 3 dims
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        self.layers = nn.ModuleList([HGNNLayer(self.embed_dim) for _ in range(2)])
        
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        batch_size = state.shape[0]
        
        # Parse State
        x = state.view(batch_size, self.N, self.M * 2 + 3)
        link_part = x[:, :, :self.M * 2].view(batch_size, self.N, self.M, 2)
        uav_pos = x[:, :, self.M * 2:]
        
        # Parse Action
        # Action: (B, N*M) -> (B, N, M, 1)
        act = action.view(batch_size, self.N, self.M, 1)
        
        # Combine Link Part and Action
        link_input = torch.cat([link_part, act], dim=-1) # (B, N, M, 3)
        
        # Initialize Features
        uav_feats = self.uav_encoder(uav_pos)
        bs_ids = torch.arange(self.M, device=state.device).expand(batch_size, -1)
        bs_feats = self.bs_embedding(bs_ids)
        edge_feats = self.edge_encoder(link_input)
        
        p1 = uav_pos.unsqueeze(2)
        p2 = uav_pos.unsqueeze(1)
        uav_dists = torch.norm(p1 - p2, dim=-1, keepdim=True)
        
        # Message Passing
        for layer in self.layers:
            uav_feats, bs_feats = layer(uav_feats, bs_feats, edge_feats, uav_dists)
            
        # Decoding
        uav_expanded = uav_feats.unsqueeze(2).expand(-1, -1, self.M, -1)
        bs_expanded = bs_feats.unsqueeze(1).expand(-1, self.N, -1, -1)
        
        decoder_input = torch.cat([uav_expanded, bs_expanded, edge_feats], dim=-1)
        
        q_values = self.decoder(decoder_input) # (B, N, M, 1)
        
        # Sum over all links to get total Q
        return q_values.sum(dim=(1, 2))
