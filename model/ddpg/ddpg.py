import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    基于 DeepSets 架构的置换等变 Actor 网络 (Permutation Equivariant Actor)。
    
    设计理念:
    1. 对称性 (Symmetry): 多无人机场景下，无人机的顺序不应影响策略的逻辑。
       如果输入中交换了 UAV_i 和 UAV_j 的位置，输出动作也应相应交换。
    2. 模块化 (Modularity): 采用 "Local Encoder + Global Pooling + Local Decoder" 的结构。
    3. 简单性 (Simplicity): 针对 Top-P 拟合任务，使用轻量级的 MLP 和 Max Pooling。
    
    结构:
    - Input: (Batch, N * State_Per_UAV)
    - Reshape -> (Batch, N, State_Per_UAV)
    - Local Encoder: 提取每个 UAV 的局部特征 -> (Batch, N, Hidden)
    - Global Pooling: 聚合全局上下文信息 (Max Pooling) -> (Batch, Hidden)
    - Fusion: 拼接局部特征和全局特征 -> (Batch, N, 2*Hidden)
    - Decoder: 生成每个 UAV 的动作 -> (Batch, N, Action_Per_UAV)
    - Output: Flatten -> (Batch, N * Action_Per_UAV)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        
        # 自动推断 N (UAV数量) 和 M (基站数量)
        # 公式推导:
        # state_dim = N * (2M + 3)
        # action_dim = N * M
        # => M = action_dim / N
        # => state_dim = N * (2 * (action_dim/N) + 3) = 2 * action_dim + 3 * N
        # => N = (state_dim - 2 * action_dim) / 3
        self.num_uavs = int((state_dim - 2 * action_dim) / 3)
        self.num_bs = int(action_dim / self.num_uavs)
        self.state_per_uav = 2 * self.num_bs + 3
        self.action_per_uav = self.num_bs
        
        print(f"[Actor] Detected Architecture: {self.num_uavs} UAVs, {self.num_bs} BSs.")
        print(f"[Actor] Per UAV State Dim: {self.state_per_uav}, Action Dim: {self.action_per_uav}")

        # 1. 局部特征提取器 (Local Encoder)
        # 独立处理每个 UAV 的状态，保证置换等变性
        self.local_encoder = nn.Sequential(
            nn.Linear(self.state_per_uav, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 2. 解码器 (Decoder)
        # 输入: 局部特征 (Hidden) + 全局特征 (Hidden) = 2 * Hidden
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
        
        # Gate Head: 输出门控 Logits
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, self.action_per_uav),
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        
        # 初始化 Gate Head 使其初始状态倾向于稍微开放或中性
        if hasattr(self, 'gate_head'):
            for m in self.gate_head.modules():
                if isinstance(m, nn.Linear):
                     nn.init.uniform_(m.weight, -0.003, 0.003)
                     if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0) # Bias=0 -> Sigmoid(0)=0.5

    def forward(self, state):
        # state: (Batch, N * State_Per_UAV)
        batch_size = state.shape[0]
        
        # 1. Reshape to (Batch, N, State_Per_UAV)
        state_reshaped = state.view(batch_size, self.num_uavs, self.state_per_uav)
        
        # 2. Local Encoding -> (Batch, N, Hidden)
        # 这一步是并行的，每个 UAV 共享相同的权重
        local_feat = self.local_encoder(state_reshaped)
        
        # 3. Global Pooling -> (Batch, Hidden)
        # 使用 Max Pooling 聚合全局信息，这是置换不变的 (Permutation Invariant)
        # 意味着无论 UAV 顺序如何，提取出的全局特征是一样的
        global_feat = torch.max(local_feat, dim=1)[0]
        
        # 4. Broadcast & Fusion -> (Batch, N, 2*Hidden)
        # 将全局特征复制 N 份，拼接到每个 UAV 的局部特征上
        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, self.num_uavs, -1)
        fusion_feat = torch.cat([local_feat, global_feat_expanded], dim=2)
        
        # 5. Decoding -> (Batch, N, Hidden)
        decoded = self.decoder(fusion_feat)
        
        # 6. Heads -> (Batch, N, Action_Per_UAV)
        power = self.power_head(decoded)
        gate_logits = self.gate_head(decoded)
        
        # 7. Flatten Output -> (Batch, N * Action_Per_UAV)
        power_flat = power.view(batch_size, -1)
        gate_logits_flat = gate_logits.view(batch_size, -1)
        
        # Gate logic with STE (Straight-Through Estimator)
        gate_prob = torch.sigmoid(gate_logits_flat)
        
        if self.training:
            gate_hard = (gate_prob > 0.5).float()
            gate_mask = (gate_hard - gate_prob).detach() + gate_prob
        else:
            gate_mask = (gate_prob > 0.5).float()
            
        action = power_flat * gate_mask
        
        return action, gate_prob

class DuelingCritic(nn.Module):
    """
    基于 DeepSets 架构的置换不变 Critic 网络 (Permutation Invariant Critic)。
    
    设计理念:
    Critic 需要评估整个状态(和动作)的价值。无论 UAV 的输入顺序如何，
    只要集合内容一样，输出的价值 V 或 Q 应该是一样的。
    
    结构:
    - Input: State, Action
    - Process: 对每个 UAV 的 (State_i, Action_i) 进行编码。
    - Pooling: 对所有 UAV 的特征进行 Sum/Max Pooling，得到全局特征。
    - Output: 基于全局特征输出价值。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingCritic, self).__init__()
        
        # 自动推断维度 (同 Actor)
        self.num_uavs = int((state_dim - 2 * action_dim) / 3)
        self.num_bs = int(action_dim / self.num_uavs)
        self.state_per_uav = 2 * self.num_bs + 3
        self.action_per_uav = self.num_bs
        
        # Q1 V-Stream (State Value)
        # 输入: State_i -> Local -> Global -> V
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
        # 输入: (State_i, Action_i) -> Local -> Global -> A
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
        
        # Q2 (Double Q-Learning) - 独立的第二套网络
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
        # x: (Batch, N, Dim)
        # 1. Local Encoding
        local_feat = encoder(x) # (Batch, N, Hidden)
        
        # 2. Global Pooling (Invariant)
        if pooling == 'max':
            global_feat = torch.max(local_feat, dim=1)[0] # (Batch, Hidden)
        else:
            global_feat = torch.mean(local_feat, dim=1)
            
        # 3. Head
        out = head(global_feat) # (Batch, 1)
        return out

    def forward(self, state, action):
        batch_size = state.shape[0]
        
        # Reshape inputs
        s = state.view(batch_size, self.num_uavs, self.state_per_uav)
        a = action.view(batch_size, self.num_uavs, self.action_per_uav)
        sa = torch.cat([s, a], dim=2) # (Batch, N, State+Action)
        
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
