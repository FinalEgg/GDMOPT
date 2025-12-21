import torch
import torch.nn as nn

class DeepSetsEncoder(nn.Module):
    """
    Permutation Equivariant Encoder based on DeepSets.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, features_per_entity=None):
        super(DeepSetsEncoder, self).__init__()
        
        # Infer N and M
        # If features_per_entity is provided, we use it to calculate N
        # Otherwise we assume standard CellFree structure: state_dim = N * (2M + 3), action_dim = N * M
        
        if features_per_entity is not None:
            # Generic mode (e.g. for Optimization Env or TopK)
            # We assume state_dim = N * features_per_entity
            self.num_uavs = int(state_dim / features_per_entity)
            self.state_per_uav = features_per_entity
        else:
            # Legacy/Standard CellFree mode inference
            # Old Logic: state_dim = N * (2M + 3) -> This was for [LogBeta, Angle, Pos]
            # New Logic (Wrapper): state_dim = N * (5M + 3) -> [LogBeta, Sin, Cos, BS_X, BS_Y, Pos]
            # action_dim = N * M
            # M = action_dim / N
            # state_dim = N * (5 * (action_dim/N) + 3) = 5 * action_dim + 3 * N
            # 3 * N = state_dim - 5 * action_dim
            
            # Try New Logic First
            self.num_uavs = int((state_dim - 5 * action_dim) / 3)
            
            # Check if valid
            if self.num_uavs > 0 and (state_dim - 5 * action_dim) % 3 == 0:
                 self.num_bs = int(action_dim / self.num_uavs)
                 self.state_per_uav = 5 * self.num_bs + 3
            else:
                # Fallback to Old Logic (2M + 3)
                self.num_uavs = int((state_dim - 2 * action_dim) / 3)
                if self.num_uavs > 0 and (state_dim - 2 * action_dim) % 3 == 0:
                    self.num_bs = int(action_dim / self.num_uavs)
                    self.state_per_uav = 2 * self.num_bs + 3
                else:
                     # Fallback for cases where inference fails (e.g. Optimization Env)
                     # Assume N=1 if dimensions are small
                     self.num_uavs = 1
                     self.state_per_uav = state_dim
        
        self.local_encoder = nn.Sequential(
            nn.Linear(self.state_per_uav, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.output_dim = hidden_dim * 2 # Local + Global

    def forward(self, state):
        # state: (Batch, N * State_Per_UAV)
        batch_size = state.shape[0]
        
        # 1. Reshape
        state_reshaped = state.view(batch_size, self.num_uavs, self.state_per_uav)
        
        # 2. Local Encoding -> (Batch, N, Hidden)
        local_feat = self.local_encoder(state_reshaped)
        
        # 3. Global Pooling -> (Batch, Hidden)
        global_feat = torch.max(local_feat, dim=1)[0]
        
        # 4. Fusion -> (Batch, N, 2*Hidden)
        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, self.num_uavs, -1)
        fused_feat = torch.cat([local_feat, global_feat_expanded], dim=2)
        
        return fused_feat
