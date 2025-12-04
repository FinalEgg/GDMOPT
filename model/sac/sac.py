import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from env.cellfree.config import M, N  # Import M and N from config

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Attention components
        self.M = M
        self.N = N
        # Feature dim: M links * 2 features + 3 pos coords
        self.feature_dim = M * 2 + 3
        self.embed_dim = 128       # Increased embedding dimension
        
        # Embedding layer
        self.embedding = nn.Linear(self.feature_dim, self.embed_dim)
        
        # Self-Attention layers (Deeper)
        self.attention1 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        
        # Output MLP (Wider and Deeper)
        self.net = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Added layer
            nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        batch_size = state.shape[0]
        
        # 1. Reshape: (Batch, N * Feature_Dim) -> (Batch, N, Feature_Dim)
        x = state.view(batch_size, self.N, self.feature_dim)
        
        # 2. Embedding
        x = self.embedding(x)
        x = F.relu(x)
        
        # 3. Self-Attention Block 1
        attn_out1, _ = self.attention1(x, x, x)
        x = x + attn_out1 # Residual
        x = F.layer_norm(x, x.shape[1:]) # Optional: LayerNorm after residual
        
        # 4. Self-Attention Block 2
        attn_out2, _ = self.attention2(x, x, x)
        x = x + attn_out2 # Residual
        
        # 5. Flatten
        x = x.reshape(batch_size, -1)
        
        # 6. MLP Output
        x = self.net(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick
        
        # Tanh Transform (Standard SAC)
        # Action range: [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Scale to [0, 1]
        action = (y_t + 1) / 2
        
        # Log Prob Correction
        # log_prob(y) = log_prob(x) - log(dy/dx)
        # dy/dx = 1 - tanh^2(x)
        # We also need to account for the scaling (x0.5)
        # But wait, if we treat y_t as the action variable for the Tanh distribution,
        # then we just transform y_t -> action.
        # Let's stick to the standard TanhNormal implementation logic.
        
        log_prob = normal.log_prob(x_t)
        # Correction for Tanh squashing
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        # Correction for scaling / 2
        log_prob -= torch.log(torch.tensor(2.0))
        
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std


class DuelingCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuelingCritic, self).__init__()
        
        # Attention components (Shared config with Actor)
        self.M = M
        self.N = N
        self.feature_dim = M * 2 + 3
        self.embed_dim = 128
        
        # Embedding layers
        self.v_embedding = nn.Linear(self.feature_dim, self.embed_dim)
        self.v_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        
        # For A branch
        self.action_feature_dim = M 
        self.a_embedding = nn.Linear(self.feature_dim + self.action_feature_dim, self.embed_dim)
        self.a_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)

        # Q1 architecture (Wider and Deeper)
        self.v1_net = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Added layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.a1_net = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Added layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture
        self.v2_net = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Added layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.a2_net = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Added layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        batch_size = state.shape[0]
        
        # --- Process State for V(s) ---
        # 1. Reshape State: (Batch, N * Feature_Dim) -> (Batch, N, Feature_Dim)
        s_seq = state.view(batch_size, self.N, self.feature_dim)
        
        # 2. Embedding & Attention for V
        v_emb = F.relu(self.v_embedding(s_seq))
        v_attn, _ = self.v_attention(v_emb, v_emb, v_emb)
        v_feat = (v_emb + v_attn).reshape(batch_size, -1) # Residual + Flatten
        
        # --- Process State + Action for A(s, a) ---
        # 1. Reshape Action: (Batch, M*N) -> (Batch, N, M)
        a_seq = action.view(batch_size, self.M, self.N)
        a_seq = a_seq.permute(0, 2, 1) # (Batch, N, M)
        
        # 2. Concatenate State and Action per UAV: (Batch, N, Feature_Dim + M)
        sa_seq = torch.cat([s_seq, a_seq], dim=2)
        
        # 3. Embedding & Attention for A
        a_emb = F.relu(self.a_embedding(sa_seq))
        a_attn, _ = self.a_attention(a_emb, a_emb, a_emb)
        a_feat = (a_emb + a_attn).reshape(batch_size, -1)
        
        # --- Compute Q values ---
        v1 = self.v1_net(v_feat)
        a1 = self.a1_net(a_feat)
        q1 = v1 + a1
        
        v2 = self.v2_net(v_feat)
        a2 = self.a2_net(a_feat)
        q2 = v2 + a2
        
        return q1, q2