import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from env.cellfree.config import M, N  # Import M and N from config

class HGNNLayer(nn.Module):
    """
    异构图神经网络层 (Heterogeneous GNN Layer)
    包含三个消息传递步骤:
    1. UAV <-> UAV (干扰感知)
    2. BS -> UAV (下行信号聚合)
    3. UAV -> BS (上行/反馈聚合)
    """
    def __init__(self, embed_dim):
        super(HGNNLayer, self).__init__()
        self.embed_dim = embed_dim
        
        # --- 1. Interference Message Passing (UAV -> UAV) ---
        # Input: [UAV_i, UAV_j, Dist_ij]
        self.int_msg_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Attention for aggregation
        self.int_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.uav_update_1 = nn.LayerNorm(embed_dim)

        # --- 2. Downlink Message Passing (BS -> UAV) ---
        # Input: [BS, UAV, Edge_Feat]
        self.dl_msg_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + embed_dim, embed_dim), # Edge feat is embedded
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.dl_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.uav_update_2 = nn.LayerNorm(embed_dim)
        
        # --- 3. Uplink Message Passing (UAV -> BS) ---
        # Input: [UAV, BS, Edge_Feat]
        self.ul_msg_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.ul_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.bs_update = nn.LayerNorm(embed_dim)

    def forward(self, uav_feats, bs_feats, edge_feats, uav_dists):
        """
        Args:
            uav_feats: (B, N, Embed)
            bs_feats: (B, M, Embed)
            edge_feats: (B, N, M, Embed) - Pre-embedded edge features
            uav_dists: (B, N, N, 1) - Pairwise distances
        """
        B, N, _ = uav_feats.shape
        M = bs_feats.shape[1]
        
        # --- 1. UAV Interference Aggregation ---
        # Expand for pairwise: (B, N, N, Embed)
        uav_i = uav_feats.unsqueeze(2).expand(-1, -1, N, -1) # (B, N, N, E) - Destination
        uav_j = uav_feats.unsqueeze(1).expand(-1, N, -1, -1) # (B, N, N, E) - Source
        
        # Message calculation
        # Cat: [Source, Dest, Dist] -> (B, N, N, 2E+1)
        int_input = torch.cat([uav_j, uav_i, uav_dists], dim=-1)
        int_msg = self.int_msg_mlp(int_input) # (B, N, N, E)
        
        # Aggregation via Attention
        # Query: UAV_i (B*N, 1, E)
        # Key/Value: Neighbors (B*N, N, E)
        q_int = uav_feats.view(B * N, 1, self.embed_dim)
        kv_int = int_msg.view(B * N, N, self.embed_dim)
        
        int_agg, _ = self.int_attn(q_int, kv_int, kv_int) # (B*N, 1, E)
        int_agg = int_agg.view(B, N, self.embed_dim)
        
        # Update UAV features (Residual)
        uav_feats = self.uav_update_1(uav_feats + int_agg)
        
        # --- 2. Downlink Aggregation (BS -> UAV) ---
        # Expand BS to (B, N, M, E)
        bs_expanded = bs_feats.unsqueeze(1).expand(-1, N, -1, -1)
        uav_expanded = uav_feats.unsqueeze(2).expand(-1, -1, M, -1)
        
        # Message: [BS, UAV, Edge]
        dl_input = torch.cat([bs_expanded, uav_expanded, edge_feats], dim=-1)
        dl_msg = self.dl_msg_mlp(dl_input) # (B, N, M, E)
        
        # Aggregation
        # Query: UAV (B*N, 1, E)
        # Key/Value: BS Messages (B*N, M, E)
        q_dl = uav_feats.view(B * N, 1, self.embed_dim)
        kv_dl = dl_msg.view(B * N, M, self.embed_dim)
        
        dl_agg, _ = self.dl_attn(q_dl, kv_dl, kv_dl)
        dl_agg = dl_agg.view(B, N, self.embed_dim)
        
        # Update UAV features
        uav_feats = self.uav_update_2(uav_feats + dl_agg)
        
        # --- 3. Uplink Aggregation (UAV -> BS) ---
        # Message: [UAV, BS, Edge] (Reusing dl_input structure but different MLP)
        # Note: dl_input is (B, N, M, ...), we need to view it from BS perspective
        ul_input = torch.cat([uav_expanded, bs_expanded, edge_feats], dim=-1)
        ul_msg = self.ul_msg_mlp(ul_input) # (B, N, M, E)
        
        # Aggregation
        # Query: BS (B*M, 1, E)
        # Key/Value: UAV Messages (B*M, N, E) -> Need to transpose N and M
        ul_msg_trans = ul_msg.transpose(1, 2) # (B, M, N, E)
        
        q_ul = bs_feats.view(B * M, 1, self.embed_dim)
        kv_ul = ul_msg_trans.reshape(B * M, N, self.embed_dim)
        
        ul_agg, _ = self.ul_attn(q_ul, kv_ul, kv_ul)
        ul_agg = ul_agg.view(B, M, self.embed_dim)
        
        # Update BS features
        bs_feats = self.bs_update(bs_feats + ul_agg)
        
        return uav_feats, bs_feats


class Actor(nn.Module):
    """
    SAC Actor 网络 (基于异构图神经网络 HGNN)
    
    架构设计:
    1. 节点: UAV (N个), BS (M个)
    2. 边: 
       - 通信边 (B->U): 携带信道信息 (Beta, Angle)
       - 干扰边 (U<->U): 携带距离信息
    3. 流程:
       - 编码器: 将物理特征映射为节点/边嵌入
       - 消息传递 (GNN Layers): 迭代更新节点状态，模拟信号传播和干扰协调
       - 边解码器: 基于更新后的节点和边特征，生成每条链路的动作 (Power, Gate)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.M = M
        self.N = N
        self.embed_dim = 128
        
        # --- 1. Encoders ---
        # UAV Node Encoder (Pos: x, y, z)
        self.uav_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # BS Node Embedding (Static ID based, since BS pos is fixed/latent)
        self.bs_embedding = nn.Embedding(M, self.embed_dim)
        
        # Communication Edge Encoder (LogBeta, Angle)
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # --- 2. GNN Layers ---
        # Stack 2 layers of message passing
        self.layers = nn.ModuleList([HGNNLayer(self.embed_dim) for _ in range(2)])
        
        # --- 3. Edge Decoder (Policy Head) ---
        # Input: [UAV_Feat, BS_Feat, Edge_Feat]
        self.decoder_input_dim = self.embed_dim * 3
        
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_linear = nn.Linear(hidden_dim, 1)
        self.log_std_linear = nn.Linear(hidden_dim, 1)
        self.gate_linear = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用 Kaiming 初始化 (针对 ReLU 优化)
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # 特殊初始化: 输出层
        # 我们希望初始输出比较小，避免一开始就饱和 (tanh -> -1/1, sigmoid -> 0/1)
        if hasattr(self, 'mean_linear') and m == self.mean_linear:
            nn.init.uniform_(m.weight, -0.003, 0.003)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        if hasattr(self, 'log_std_linear') and m == self.log_std_linear:
            nn.init.uniform_(m.weight, -0.003, 0.003)
            if m.bias is not None:
                # 初始 log_std 设为 0 (std=1) 或者更小，避免初始探索过大或过小
                nn.init.constant_(m.bias, 0)

        if hasattr(self, 'gate_linear') and m == self.gate_linear:
             nn.init.uniform_(m.weight, -0.003, 0.003)
             if m.bias is not None:
                # 初始 bias 设为 0.5 -> sigmoid(0.5) ≈ 0.62 (略微偏向开启)
                # 或者设为 0 -> sigmoid(0) = 0.5 (完全中立)
                nn.init.constant_(m.bias, 0.5)

    def forward(self, state):
        batch_size = state.shape[0]
        
        # --- 1. Parse State ---
        x = state.view(batch_size, self.N, self.M * 2 + 3)
        link_part = x[:, :, :self.M * 2].view(batch_size, self.N, self.M, 2) # (B, N, M, 2)
        uav_pos = x[:, :, self.M * 2:] # (B, N, 3)
        
        # --- 2. Initialize Features ---
        # UAV Nodes
        uav_feats = self.uav_encoder(uav_pos) # (B, N, E)
        
        # BS Nodes (Learnable Embeddings)
        bs_ids = torch.arange(self.M, device=state.device).expand(batch_size, -1) # (B, M)
        bs_feats = self.bs_embedding(bs_ids) # (B, M, E)
        
        # Comm Edges
        edge_feats = self.edge_encoder(link_part) # (B, N, M, E)
        
        # Interference Edges (Pairwise Distances)
        # uav_pos: (B, N, 3)
        p1 = uav_pos.unsqueeze(2) # (B, N, 1, 3)
        p2 = uav_pos.unsqueeze(1) # (B, 1, N, 3)
        uav_dists = torch.norm(p1 - p2, dim=-1, keepdim=True) # (B, N, N, 1)
        
        # --- 3. Message Passing ---
        for layer in self.layers:
            uav_feats, bs_feats = layer(uav_feats, bs_feats, edge_feats, uav_dists)
            
        # --- 4. Decoding ---
        # Prepare inputs for edge decoder
        # Need to combine: UAV_i, BS_j, Edge_ij
        uav_expanded = uav_feats.unsqueeze(2).expand(-1, -1, self.M, -1) # (B, N, M, E)
        bs_expanded = bs_feats.unsqueeze(1).expand(-1, self.N, -1, -1)   # (B, N, M, E)
        
        # Cat: (B, N, M, 3*E)
        decoder_input = torch.cat([uav_expanded, bs_expanded, edge_feats], dim=-1)
        
        dec_out = self.decoder(decoder_input)
        
        mean = self.mean_linear(dec_out)
        log_std = self.log_std_linear(dec_out)
        gate_logits = self.gate_linear(dec_out)
        
        mean = mean.view(batch_size, -1)
        log_std = log_std.view(batch_size, -1)
        gate_logits = gate_logits.view(batch_size, -1)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, gate_logits

    def sample(self, state):
        mean, log_std, gate_logits = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        
        y_t = torch.tanh(x_t)
        power_action = (y_t + 1) / 2
        
        if self.training:
            u = torch.rand_like(gate_logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            temp = 1.0
            y_soft = torch.sigmoid((gate_logits + gumbel_noise) / temp)
        else:
            y_soft = torch.sigmoid(gate_logits)
            
        y_hard = (y_soft > 0.5).float()
        mask = (y_hard - y_soft).detach() + y_soft
        
        action = power_action * mask
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob -= torch.log(torch.tensor(2.0))
        
        epsilon = 1e-6
        gate_log_prob = mask * torch.log(y_soft + epsilon) + (1 - mask) * torch.log(1 - y_soft + epsilon)
        
        total_log_prob = log_prob + gate_log_prob
        total_log_prob = total_log_prob.sum(1, keepdim=True)
        
        return action, total_log_prob, mean, log_std, y_soft


class DuelingCritic(nn.Module):
    """
    SAC Critic 网络 (Dueling 架构 + HGNN)
    
    升级版: 采用与 Actor 相同的异构图神经网络 (HGNN) 结构来提取特征。
    保留 Dueling 架构: Q(s, a) = V(s) + A(s, a)
    
    结构:
    - V-Stream: 输入 State (Graph)，经过 HGNN 提取特征，输出 V(s)。
    - A-Stream: 输入 State + Action (Graph with Edge Attributes)，经过 HGNN 提取特征，输出 A(s, a)。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuelingCritic, self).__init__()
        
        self.M = M
        self.N = N
        self.embed_dim = 128
        
        # ==================== Q1 Network ====================
        # --- V-Stream (State Value) ---
        # Encoders
        self.q1_v_uav_enc = nn.Sequential(nn.Linear(3, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.q1_v_bs_emb = nn.Embedding(M, self.embed_dim)
        self.q1_v_edge_enc = nn.Sequential(nn.Linear(2, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        # GNN Layers
        self.q1_v_gnn = nn.ModuleList([HGNNLayer(self.embed_dim) for _ in range(2)])
        # Head
        self.q1_v_head = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # --- A-Stream (Advantage) ---
        # Encoders
        self.q1_a_uav_enc = nn.Sequential(nn.Linear(3, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.q1_a_bs_emb = nn.Embedding(M, self.embed_dim)
        # Edge Encoder: State Feat (2) + Action (1) = 3
        self.q1_a_edge_enc = nn.Sequential(nn.Linear(3, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        # GNN Layers
        self.q1_a_gnn = nn.ModuleList([HGNNLayer(self.embed_dim) for _ in range(2)])
        # Head
        self.q1_a_head = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # ==================== Q2 Network ====================
        # --- V-Stream ---
        self.q2_v_uav_enc = nn.Sequential(nn.Linear(3, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.q2_v_bs_emb = nn.Embedding(M, self.embed_dim)
        self.q2_v_edge_enc = nn.Sequential(nn.Linear(2, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.q2_v_gnn = nn.ModuleList([HGNNLayer(self.embed_dim) for _ in range(2)])
        self.q2_v_head = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # --- A-Stream ---
        self.q2_a_uav_enc = nn.Sequential(nn.Linear(3, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.q2_a_bs_emb = nn.Embedding(M, self.embed_dim)
        self.q2_a_edge_enc = nn.Sequential(nn.Linear(3, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.q2_a_gnn = nn.ModuleList([HGNNLayer(self.embed_dim) for _ in range(2)])
        self.q2_a_head = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _forward_branch(self, state, action, uav_enc, bs_emb, edge_enc, gnn_layers, head, is_advantage=False):
        batch_size = state.shape[0]
        
        # --- 1. Parse State ---
        x = state.view(batch_size, self.N, self.M * 2 + 3)
        link_part = x[:, :, :self.M * 2].view(batch_size, self.N, self.M, 2) # (B, N, M, 2)
        uav_pos = x[:, :, self.M * 2:] # (B, N, 3)
        
        # --- 2. Initialize Features ---
        uav_feats = uav_enc(uav_pos) # (B, N, E)
        
        bs_ids = torch.arange(self.M, device=state.device).expand(batch_size, -1)
        bs_feats = bs_emb(bs_ids) # (B, M, E)
        
        # Edge Features
        if is_advantage:
            # Concatenate action to edge features
            # action: (B, N*M) -> (B, N, M, 1)
            act = action.view(batch_size, self.N, self.M, 1)
            edge_input = torch.cat([link_part, act], dim=-1) # (B, N, M, 3)
            edge_feats = edge_enc(edge_input)
        else:
            edge_feats = edge_enc(link_part) # (B, N, M, 2)
            
        # Distances (for Interference)
        p1 = uav_pos.unsqueeze(2) # (B, N, 1, 3)
        p2 = uav_pos.unsqueeze(1) # (B, 1, N, 3)
        uav_dists = torch.norm(p1 - p2, dim=-1, keepdim=True) # (B, N, N, 1)
        
        # --- 3. Message Passing ---
        for layer in gnn_layers:
            uav_feats, bs_feats = layer(uav_feats, bs_feats, edge_feats, uav_dists)
            
        # --- 4. Readout ---
        # Flatten UAV features (B, N*E)
        flat_feats = uav_feats.view(batch_size, -1)
        val = head(flat_feats)
        
        return val

    def forward(self, state, action):
        # Q1
        v1 = self._forward_branch(state, action, self.q1_v_uav_enc, self.q1_v_bs_emb, self.q1_v_edge_enc, self.q1_v_gnn, self.q1_v_head, is_advantage=False)
        a1 = self._forward_branch(state, action, self.q1_a_uav_enc, self.q1_a_bs_emb, self.q1_a_edge_enc, self.q1_a_gnn, self.q1_a_head, is_advantage=True)
        q1 = v1 + a1
        
        # Q2
        v2 = self._forward_branch(state, action, self.q2_v_uav_enc, self.q2_v_bs_emb, self.q2_v_edge_enc, self.q2_v_gnn, self.q2_v_head, is_advantage=False)
        a2 = self._forward_branch(state, action, self.q2_a_uav_enc, self.q2_a_bs_emb, self.q2_a_edge_enc, self.q2_a_gnn, self.q2_a_head, is_advantage=True)
        q2 = v2 + a2
        
        return q1, q2

    def get_value_details(self, state, action):
        """Helper for debugging: returns separated V and A values"""
        v1 = self._forward_branch(state, action, self.q1_v_uav_enc, self.q1_v_bs_emb, self.q1_v_edge_enc, self.q1_v_gnn, self.q1_v_head, is_advantage=False)
        a1 = self._forward_branch(state, action, self.q1_a_uav_enc, self.q1_a_bs_emb, self.q1_a_edge_enc, self.q1_a_gnn, self.q1_a_head, is_advantage=True)
        
        v2 = self._forward_branch(state, action, self.q2_v_uav_enc, self.q2_v_bs_emb, self.q2_v_edge_enc, self.q2_v_gnn, self.q2_v_head, is_advantage=False)
        a2 = self._forward_branch(state, action, self.q2_a_uav_enc, self.q2_a_bs_emb, self.q2_a_edge_enc, self.q2_a_gnn, self.q2_a_head, is_advantage=True)
        
        return v1, a1, v2, a2