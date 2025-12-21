import torch
import torch.nn as nn
import numpy as np
from .backbones.deepsets import DeepSetsEncoder
from .diffusion_helpers import SinusoidalPosEmb

class DeepSetsDiffusionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, t_dim=16):
        super().__init__()
        
        # We need to infer num_uavs and per-entity dims to configure DeepSetsEncoder correctly.
        # Standard logic: state_dim = N * S_i, action_dim = N * A_i
        # We want to encode (State + Action) -> Hidden
        # So input dim per entity is S_i + A_i
        
        # Let's reuse the logic from DeepSetsEncoder to infer N and S_i
        # But we need to do it manually here because we are combining state and action.
        
        # Infer N from state_dim (assuming standard structure)
        # Try New Logic First (5 features per BS)
        num_uavs = int((state_dim - 5 * action_dim) / 3)
        if num_uavs > 0 and (state_dim - 5 * action_dim) % 3 == 0:
             num_bs = int(action_dim / num_uavs)
             state_per_uav = 5 * num_bs + 3
             action_per_uav = num_bs
        else:
            # Fallback to Old Logic (2 features per BS)
            num_uavs = int((state_dim - 2 * action_dim) / 3)
            if num_uavs > 0 and (state_dim - 2 * action_dim) % 3 == 0:
                num_bs = int(action_dim / num_uavs)
                state_per_uav = 2 * num_bs + 3
                action_per_uav = num_bs
            else:
                 # Fallback
                 num_uavs = 1
                 state_per_uav = state_dim
                 action_per_uav = action_dim
        
        self.num_uavs = num_uavs
        self.state_per_uav = state_per_uav
        self.action_per_uav = action_per_uav
        self.combined_per_uav = state_per_uav + action_per_uav
        
        # Encoder for (State + Action)
        # We pass features_per_entity to force DeepSetsEncoder to use our calculated dim
        self.encoder = DeepSetsEncoder(
            state_dim + action_dim, 
            action_dim, # This arg is ignored if features_per_entity is set, but needed for init
            hidden_dim, 
            features_per_entity=self.combined_per_uav
        )
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        # Decoder
        # Input: Hidden (from encoder) + Time Embedding
        # We concatenate Time Embedding to Hidden?
        # Hidden is (B, N, H). Time is (B, T).
        # Expand Time to (B, N, T).
        
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + t_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.action_per_uav)
        )

    def forward(self, x, time, state):
        # x: (B, N*A_i) - Noisy Action
        # time: (B,)
        # state: (B, N*S_i)
        
        batch_size = x.shape[0]
        
        # Reshape and Concatenate
        x_reshaped = x.view(batch_size, self.num_uavs, self.action_per_uav)
        state_reshaped = state.view(batch_size, self.num_uavs, self.state_per_uav)
        
        combined = torch.cat([state_reshaped, x_reshaped], dim=2) # (B, N, S_i+A_i)
        combined_flat = combined.view(batch_size, -1) # DeepSetsEncoder expects flat input if features_per_entity is set?
        # Wait, DeepSetsEncoder implementation:
        # x = x.view(batch_size, self.num_uavs, self.state_per_uav)
        # So we should pass flat input.
        
        # Encode
        # Note: DeepSetsEncoder usually does Global Pooling.
        # But for Diffusion, we want to predict noise PER ENTITY (per UAV).
        # So we should NOT use the global pooling output of DeepSetsEncoder.
        # We need the LOCAL features.
        
        # Let's check DeepSetsEncoder again.
        # It returns `global_feat` (B, Hidden).
        # It has `self.local_encoder`.
        
        # I should access `self.encoder.local_encoder` directly?
        # Or modify DeepSetsEncoder to return local features?
        # Or just use `self.encoder.local_encoder` here.
        
        local_feat = self.encoder.local_encoder(combined) # (B, N, Hidden)
        
        # Time Embedding
        t_emb = self.time_mlp(time) # (B, T_dim)
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, self.num_uavs, -1) # (B, N, T_dim)
        
        # Concatenate
        feat = torch.cat([local_feat, t_emb_expanded], dim=2) # (B, N, Hidden+T_dim)
        
        # Decode
        out = self.mid_layer(feat) # (B, N, A_i)
        
        return out.view(batch_size, -1) # (B, N*A_i)
