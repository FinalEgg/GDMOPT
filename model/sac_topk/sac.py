import torch
import torch.nn as nn
import numpy as np
from env.topk.config import M, N, K_MAX

class Actor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.M = M
        self.N = N
        self.K = K_MAX
        
        # Embeddings
        # M 个基站 ID (0 ~ M-1) + 1 个 Padding ID (M)
        self.bs_embedding = nn.Embedding(M + 1, 8) 
        
        # UAV Feature Extractor
        self.uav_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Connection Feature Extractor
        # Input: Emb(8) + Beta(1) + Angle(1) = 10
        self.conn_mlp = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Combined Processor
        # Input: UAV_Feat(32) + Conn_Feat(32) = 64
        # SAC Actor outputs Mean and LogStd
        self.backbone = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(64, 1)
        self.log_std_layer = nn.Linear(64, 1)
        
    def forward(self, obs, state=None, info={}):
        # obs shape: (Batch, N * (3 + K*3))
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        # 强制将 obs 移动到与网络参数相同的设备，避免 CPU/GPU 混用
        obs = obs.to(self.mu_layer.weight.device)
            
        batch_size = obs.shape[0]
        
        # Reshape
        obs = obs.reshape(batch_size, self.N, 3 + self.K * 3)
        
        # Split
        uav_pos = obs[:, :, :3] # (B, N, 3)
        conn_data = obs[:, :, 3:] # (B, N, K*3)
        conn_data = conn_data.reshape(batch_size, self.N, self.K, 3)
        
        # Process UAV
        uav_feat = self.uav_mlp(uav_pos) # (B, N, 32)
        uav_feat_expanded = uav_feat.unsqueeze(2).expand(-1, -1, self.K, -1) # (B, N, K, 32)
        
        # Process Connections
        bs_ids = conn_data[:, :, :, 0].long() # (B, N, K)
        # Handle padding: -1 -> M
        bs_ids = torch.where(bs_ids < 0, torch.tensor(self.M, device=obs.device), bs_ids)
        bs_ids = torch.clamp(bs_ids, 0, self.M)
        
        bs_emb = self.bs_embedding(bs_ids) # (B, N, K, 8)
        
        other_feats = conn_data[:, :, :, 1:] # (B, N, K, 2)
        
        conn_input = torch.cat([bs_emb, other_feats], dim=-1) # (B, N, K, 10)
        conn_feat = self.conn_mlp(conn_input) # (B, N, K, 32)
        
        # Combine
        combined = torch.cat([uav_feat_expanded, conn_feat], dim=-1) # (B, N, K, 64)
        
        # Backbone
        features = self.backbone(combined) # (B, N, K, 64)
        
        # Output Heads
        mu = self.mu_layer(features).squeeze(-1) # (B, N, K)
        log_std = self.log_std_layer(features).squeeze(-1) # (B, N, K)
        
        # Flatten for Tianshou
        mu = mu.reshape(batch_size, -1)
        log_std = log_std.reshape(batch_size, -1)
        
        # Clip log_std for stability
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        return (mu, std), state

class SingleCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.M = M
        self.N = N
        self.K = K_MAX
        
        self.bs_embedding = nn.Embedding(M + 1, 8)
        
        self.uav_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        self.conn_mlp = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Action is (B, N, K)
        # We concat action to the combined features
        # Input: 64 + 1 (Action) = 65
        self.agg_mlp = nn.Sequential(
            nn.Linear(65, 64),
            nn.ReLU()
        )
        
        # Global Aggregation (Pool over N and K)
        self.global_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32)

        # 强制将 obs/act 移动到与网络参数相同的设备，避免 CPU/GPU 混用
        device = self.agg_mlp[0].weight.device
        obs = obs.to(device)
        act = act.to(device)
            
        batch_size = obs.shape[0]
        
        # Reshape Obs
        obs = obs.reshape(batch_size, self.N, 3 + self.K * 3)
        uav_pos = obs[:, :, :3]
        conn_data = obs[:, :, 3:].reshape(batch_size, self.N, self.K, 3)
        
        # Reshape Act
        act = act.reshape(batch_size, self.N, self.K, 1)
        
        # Features
        uav_feat = self.uav_mlp(uav_pos).unsqueeze(2).expand(-1, -1, self.K, -1)
        
        bs_ids = conn_data[:, :, :, 0].long()
        bs_ids = torch.where(bs_ids < 0, torch.tensor(self.M, device=obs.device), bs_ids)
        bs_ids = torch.clamp(bs_ids, 0, self.M)
        
        bs_emb = self.bs_embedding(bs_ids)
        other_feats = conn_data[:, :, :, 1:]
        conn_feat = self.conn_mlp(torch.cat([bs_emb, other_feats], dim=-1))
        
        # Combine with Action
        combined = torch.cat([uav_feat, conn_feat, act], dim=-1) # (B, N, K, 65)
        
        # Process per-link
        link_hidden = self.agg_mlp(combined) # (B, N, K, 64)
        
        # Aggregate (Mean pool over N and K)
        flat_hidden = link_hidden.reshape(batch_size, -1, 64)
        pooled = torch.mean(flat_hidden, dim=1) # (B, 64)
        
        value = self.global_mlp(pooled) # (B, 1)
        return value

class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Q1 = SingleCritic(args)
        self.Q2 = SingleCritic(args)
        
    def forward(self, obs, act, info={}):
        q1 = self.Q1(obs, act, info)
        q2 = self.Q2(obs, act, info)
        return q1, q2
