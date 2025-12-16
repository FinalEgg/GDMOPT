import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEncoder(nn.Module):
    """
    使用注意力机制处理单个 UAV 的状态，专注于重要的基站连接。
    
    输入: state_per_uav (2*M + 3)
    输出: feature_vector (hidden_dim)
    """
    def __init__(self, state_per_uav, num_bs, hidden_dim, embed_dim=64):
        super().__init__()
        self.num_bs = num_bs
        self.state_per_uav = state_per_uav
        self.uav_feature_dim = 3
        self.bs_feature_dim = 2 # (beta, angle)

        # 1. BS 特征嵌入层
        self.bs_embed = nn.Sequential(
            nn.Linear(self.bs_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        # 2. 注意力分数计算网络
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        # 3. UAV 自身特征处理层
        self.uav_embed = nn.Sequential(
            nn.Linear(self.uav_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        # 4. 最终融合层
        # 输入: BS聚合特征 (embed_dim) + UAV自身特征 (embed_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (Batch, 2*M + 3) or (Batch, N, 2*M + 3)
        
        # 处理多维输入 (Batch, N, Dim) -> (Batch*N, Dim)
        is_3d = x.dim() == 3
        if is_3d:
            batch_size, num_uavs, _ = x.shape
            x = x.view(batch_size * num_uavs, -1)
        
        current_batch_size = x.shape[0]

        # 1. 分离 BS 特征和 UAV 特征
        # x shape: (Current_Batch, 35)
        # bs_features 应该取前 32 个元素 (16 * 2)
        bs_features = x[:, :self.num_bs * self.bs_feature_dim].contiguous().view(current_batch_size, self.num_bs, self.bs_feature_dim)
        uav_features = x[:, self.num_bs * self.bs_feature_dim:]

        # 2. 计算 BS 特征嵌入
        bs_embedded = self.bs_embed(bs_features) # (Current_Batch, M, embed_dim)

        # 3. 计算注意力分数
        attn_scores = self.attention_net(bs_embedded) # (Current_Batch, M, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (Current_Batch, M, 1)

        # 4. 应用注意力权重，聚合 BS 特征
        aggregated_bs_features = (bs_embedded * attn_weights).sum(dim=1)

        # 5. 处理 UAV 自身特征
        uav_embedded = self.uav_embed(uav_features) # (Current_Batch, embed_dim)

        # 6. 融合
        combined_features = torch.cat([aggregated_bs_features, uav_embedded], dim=1)
        output = self.fusion_layer(combined_features) # (Current_Batch, hidden_dim)

        # 如果输入是 3D，还原输出形状
        if is_3d:
            output = output.view(batch_size, num_uavs, -1)

        return output

class Actor(nn.Module):
    """
    基于 DeepSets 架构的置换等变 Actor 网络 (Permutation Equivariant Actor)。
    仅输出功率分配，无门控网络。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        
        # 自动推断 N (UAV数量) 和 M (基站数量)
        # state_dim = N * (2M + 3)
        # action_dim = N * M (Power only)
        self.num_uavs = int((state_dim - 2 * action_dim) / 3)
        self.num_bs = int(action_dim / self.num_uavs)
        self.state_per_uav = 2 * self.num_bs + 3
        self.action_per_uav = self.num_bs
        
        print(f"[Actor] Detected Architecture: {self.num_uavs} UAVs, {self.num_bs} BSs.")
        print(f"[Actor] Per UAV State Dim: {self.state_per_uav}, Action Dim: {self.action_per_uav}")

        # 1. 局部特征提取器 (Local Encoder)
        self.local_encoder = AttentionEncoder(
            state_per_uav=self.state_per_uav,
            num_bs=self.num_bs,
            hidden_dim=hidden_dim
        )
        
        # 2. 解码器 (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 3. 输出头 (Heads)
        # Power Head: 输出功率分配 [0, 1]
        self.power_head = nn.Sequential(
            nn.Linear(hidden_dim, self.action_per_uav),
            nn.Sigmoid() 
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        # state: (Batch, N * State_Per_UAV)
        batch_size = state.shape[0]
        
        # 1. Reshape to (Batch, N, State_Per_UAV)
        state_reshaped = state.view(batch_size, self.num_uavs, self.state_per_uav)
        
        # 2. Local Encoding -> (Batch, N, Hidden)
        local_feat = self.local_encoder(state_reshaped)
        
        # 3. Global Pooling -> (Batch, Hidden)
        global_feat = torch.max(local_feat, dim=1)[0]
        
        # 4. Broadcast & Fusion -> (Batch, N, 2*Hidden)
        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, self.num_uavs, -1)
        fusion_feat = torch.cat([local_feat, global_feat_expanded], dim=2)
        
        # 5. Decoding -> (Batch, N, Hidden)
        decoded = self.decoder(fusion_feat)
        
        # 6. Heads -> (Batch, N, Action_Per_UAV)
        power = self.power_head(decoded)
        
        # 7. Flatten Output -> (Batch, N * Action_Per_UAV)
        power_flat = power.view(batch_size, -1)
        
        return power_flat

class DuelingCritic(nn.Module):
    """
    基于 DeepSets 架构的置换不变 Critic 网络 (Permutation Invariant Critic)。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingCritic, self).__init__()
        
        self.num_uavs = int((state_dim - 2 * action_dim) / 3)
        self.num_bs = int(action_dim / self.num_uavs)
        self.state_per_uav = 2 * self.num_bs + 3
        self.action_per_uav = self.num_bs
        
        # Q1 V-Stream (State Value)
        self.v_encoder = nn.Sequential(
            nn.Linear(self.state_per_uav, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q1 A-Stream (Advantage)
        self.a_encoder = nn.Sequential(
            nn.Linear(self.state_per_uav + self.action_per_uav, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.a_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 (Double Q-Learning)
        self.v_encoder2 = nn.Sequential(
            nn.Linear(self.state_per_uav, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.v_head2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.a_encoder2 = nn.Sequential(
            nn.Linear(self.state_per_uav + self.action_per_uav, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.a_head2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _process_stream(self, encoder, head, x, pooling='max'):
        local_feat = encoder(x) # (Batch, N, Hidden)
        if pooling == 'max':
            global_feat = torch.max(local_feat, dim=1)[0] # (Batch, Hidden)
        else:
            global_feat = torch.mean(local_feat, dim=1)
        out = head(global_feat) # (Batch, 1)
        return out

    def forward(self, state, action):
        batch_size = state.shape[0]
        
        s = state.view(batch_size, self.num_uavs, self.state_per_uav)
        a = action.view(batch_size, self.num_uavs, self.action_per_uav)
        sa = torch.cat([s, a], dim=2) 
        
        # Q1
        v1 = self._process_stream(self.v_encoder, self.v_head, s)
        a1 = self._process_stream(self.a_encoder, self.a_head, sa)
        
        # Q2
        v2 = self._process_stream(self.v_encoder2, self.v_head2, s)
        a2 = self._process_stream(self.a_encoder2, self.a_head2, sa)
        
        # Standard Double Q-learning: Independent Critics
        q1 = v1 + a1
        q2 = v2 + a2
        
        return q1, q2
