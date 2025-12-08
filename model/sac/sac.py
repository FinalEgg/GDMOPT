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
    SAC Critic 网络 (Dueling 架构)
    
    采用 Dueling Network 结构: Q(s, a) = V(s) + A(s, a)
    
    设计动机:
    在 Cell-Free 场景中，环境状态 (UAV 位置、信道质量) 对奖励的影响非常大。
    - V(s) (状态价值): 捕捉环境本身的"好坏" (Baseline)。例如，如果所有 UAV 都在边缘且信道极差，V(s) 会很低。
    - A(s, a) (优势函数): 捕捉动作相对于当前状态平均水平的优劣。
    
    这种分离有助于降低方差，特别是在环境随机性很强的情况下 (如用户提到的"运气好坏")，
    让 V 网络吸收环境偏差，A 网络专注于学习策略。
    
    关于 "减去均值" (Subtract Mean):
    在离散动作 Dueling DQN 中，常使用 Q = V + (A - mean(A))。
    在连续动作空间，计算 A 的均值需要对动作空间积分，计算成本过高。
    因此这里采用直接相加 Q = V + A，依靠优化器自然分离 V 和 A 的功能。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuelingCritic, self).__init__()
        
        self.M = M
        self.N = N
        
        # Dimensions
        self.bs_feat_dim = 2 # LogBeta, Angle
        self.uav_pos_dim = 3 # x, y, z
        self.action_dim_per_link = 1 # Power per link (Gate 已经融合在 Action 数值中了)
        self.embed_dim = 128
        
        # ==================== Q1 Network ====================
        # --- V-Stream (State Value) ---
        self.q1_v_uav_encoder = nn.Sequential(nn.Linear(self.uav_pos_dim, self.embed_dim), nn.ReLU())
        self.q1_v_bs_encoder = nn.Sequential(nn.Linear(self.bs_feat_dim, self.embed_dim), nn.ReLU())
        self.q1_v_cross = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.q1_v_self = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.q1_v_head = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # --- A-Stream (Advantage) ---
        self.q1_a_uav_encoder = nn.Sequential(nn.Linear(self.uav_pos_dim, self.embed_dim), nn.ReLU())
        self.q1_a_bs_action_encoder = nn.Sequential(nn.Linear(self.bs_feat_dim + self.action_dim_per_link, self.embed_dim), nn.ReLU())
        self.q1_a_cross = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.q1_a_self = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.q1_a_head = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # ==================== Q2 Network ====================
        # --- V-Stream ---
        self.q2_v_uav_encoder = nn.Sequential(nn.Linear(self.uav_pos_dim, self.embed_dim), nn.ReLU())
        self.q2_v_bs_encoder = nn.Sequential(nn.Linear(self.bs_feat_dim, self.embed_dim), nn.ReLU())
        self.q2_v_cross = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.q2_v_self = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.q2_v_head = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # --- A-Stream ---
        self.q2_a_uav_encoder = nn.Sequential(nn.Linear(self.uav_pos_dim, self.embed_dim), nn.ReLU())
        self.q2_a_bs_action_encoder = nn.Sequential(nn.Linear(self.bs_feat_dim + self.action_dim_per_link, self.embed_dim), nn.ReLU())
        self.q2_a_cross = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.q2_a_self = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.q2_a_head = nn.Sequential(
            nn.Linear(self.N * self.embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _forward_dueling_branch(self, state, action, 
                                v_uav_enc, v_bs_enc, v_cross, v_self, v_head,
                                a_uav_enc, a_bs_act_enc, a_cross, a_self, a_head,
                                return_components=False):
        batch_size = state.shape[0]
        
        # --- Parse State ---
        x = state.view(batch_size, self.N, self.M * 2 + 3)
        link_part = x[:, :, :self.M * 2].view(batch_size, self.N, self.M, 2)
        uav_pos = x[:, :, self.M * 2:]
        
        # --- V-Stream (State Value) ---
        # 1. Embeddings
        v_uav_emb = v_uav_enc(uav_pos).unsqueeze(2) # (Batch, N, 1, Embed)
        v_bs_emb = v_bs_enc(link_part) # (Batch, N, M, Embed)
        
        # 2. Attention
        v_uav_flat = v_uav_emb.view(batch_size * self.N, 1, self.embed_dim)
        v_bs_flat = v_bs_emb.view(batch_size * self.N, self.M, self.embed_dim)
        
        v_ctx_flat, _ = v_cross(v_uav_flat, v_bs_flat, v_bs_flat)
        v_ctx = v_ctx_flat.view(batch_size, self.N, self.embed_dim) + v_uav_emb.squeeze(2)
        
        v_global, _ = v_self(v_ctx, v_ctx, v_ctx)
        v_global = v_global + v_ctx
        
        # 3. Head -> V(s)
        v_val = v_head(v_global.reshape(batch_size, -1))
        
        # --- A-Stream (Advantage) ---
        # 1. Embeddings (Include Action)
        a_uav_emb = a_uav_enc(uav_pos).unsqueeze(2)
        
        act = action.view(batch_size, self.N, self.M, 1)
        bs_act_input = torch.cat([link_part, act], dim=3)
        a_bs_act_emb = a_bs_act_enc(bs_act_input)
        
        # 2. Attention
        a_uav_flat = a_uav_emb.view(batch_size * self.N, 1, self.embed_dim)
        a_bs_act_flat = a_bs_act_emb.view(batch_size * self.N, self.M, self.embed_dim)
        
        a_ctx_flat, _ = a_cross(a_uav_flat, a_bs_act_flat, a_bs_act_flat)
        a_ctx = a_ctx_flat.view(batch_size, self.N, self.embed_dim) + a_uav_emb.squeeze(2)
        
        a_global, _ = a_self(a_ctx, a_ctx, a_ctx)
        a_global = a_global + a_ctx
        
        # 3. Head -> A(s, a)
        a_val = a_head(a_global.reshape(batch_size, -1))
        
        if return_components:
            return v_val, a_val
            
        # --- Combine ---
        # Q(s, a) = V(s) + A(s, a)
        return v_val + a_val

    def forward(self, state, action):
        q1 = self._forward_dueling_branch(state, action,
                                          self.q1_v_uav_encoder, self.q1_v_bs_encoder, self.q1_v_cross, self.q1_v_self, self.q1_v_head,
                                          self.q1_a_uav_encoder, self.q1_a_bs_action_encoder, self.q1_a_cross, self.q1_a_self, self.q1_a_head)
                                          
        q2 = self._forward_dueling_branch(state, action,
                                          self.q2_v_uav_encoder, self.q2_v_bs_encoder, self.q2_v_cross, self.q2_v_self, self.q2_v_head,
                                          self.q2_a_uav_encoder, self.q2_a_bs_action_encoder, self.q2_a_cross, self.q2_a_self, self.q2_a_head)
        return q1, q2

    def get_value_details(self, state, action):
        """Helper for debugging: returns separated V and A values"""
        # Branch 1
        v1, a1 = self._forward_dueling_branch(state, action,
                                          self.q1_v_uav_encoder, self.q1_v_bs_encoder, self.q1_v_cross, self.q1_v_self, self.q1_v_head,
                                          self.q1_a_uav_encoder, self.q1_a_bs_action_encoder, self.q1_a_cross, self.q1_a_self, self.q1_a_head,
                                          return_components=True)
        # Branch 2
        v2, a2 = self._forward_dueling_branch(state, action,
                                          self.q2_v_uav_encoder, self.q2_v_bs_encoder, self.q2_v_cross, self.q2_v_self, self.q2_v_head,
                                          self.q2_a_uav_encoder, self.q2_a_bs_action_encoder, self.q2_a_cross, self.q2_a_self, self.q2_a_head,
                                          return_components=True)
        return v1, a1, v2, a2