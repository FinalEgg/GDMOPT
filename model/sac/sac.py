import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from env.cellfree.config import M, N  # Import M and N from config

class Actor(nn.Module):
    """
    SAC Actor 网络 (策略网络)
    
    该网络负责根据当前环境状态生成动作。
    针对 Cell-Free UAV 场景，采用了基于 Attention 的架构来捕捉 UAV 与基站 (BS) 之间以及 UAV 相互之间的拓扑关系。
    
    输出包含两部分：
    1. 连续动作 (Power): 每个链路的发射功率控制。
    2. 离散动作 (Gate): 每个链路的连接开关 (0 或 1)。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.M = M  # 基站数量
        self.N = N  # UAV 数量
        
        # --- 维度定义 ---
        self.bs_feat_dim = 2   # 基站特征维度: [LogBeta (大尺度衰落), Angle (角度)]
        self.uav_pos_dim = 3   # UAV 位置维度: [x, y, z]
        self.embed_dim = 128   # 嵌入层维度 (Hidden Size)
        
        # --- 1. 特征编码器 (Feature Encoders) ---
        # 将原始的物理特征映射到高维嵌入空间
        
        # 基站特征编码器
        self.bs_encoder = nn.Sequential(
            nn.Linear(self.bs_feat_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # UAV 位置编码器
        self.uav_encoder = nn.Sequential(
            nn.Linear(self.uav_pos_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # --- 2. 交叉注意力 (Cross-Attention): UAV -> BS ---
        # 作用: 让每个 UAV "观察" 所有的基站，并根据自身的嵌入 (Query) 和基站的嵌入 (Key/Value) 
        # 聚合局部环境信息。这有助于 UAV 识别哪些基站信号强，值得连接。
        # Query: UAV Embed, Key/Value: BS Embed
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        
        # --- 3. 自注意力 (Self-Attention): UAV -> UAV ---
        # 作用: 让 UAV 之间进行信息交互。这是多智能体协作的关键。
        # 通过关注其他 UAV 的状态，当前 UAV 可以感知潜在的干扰源，从而调整功率或避开拥堵的基站。
        # Query/Key/Value: UAV Context (来自上一步的输出)
        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        
        # --- 4. 解码器与输出头 (Decoder / Output Heads) ---
        # 我们需要为每个 UAV-BS 链路生成动作。
        # 输入: 拼接 (基站嵌入, 全局 UAV 上下文)
        # 这样每个链路的决策既考虑了该基站的特性，也考虑了 UAV 的全局协作状态。
        self.decoder_input_dim = self.embed_dim * 2
        
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 输出头 (针对每个链路)
        self.mean_linear = nn.Linear(hidden_dim, 1)    # 连续动作均值 (Mean)
        self.log_std_linear = nn.Linear(hidden_dim, 1) # 连续动作标准差的对数 (Log Std)
        self.gate_linear = nn.Linear(hidden_dim, 1)    # 离散动作 Logits (Gate)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # 特殊初始化: Gate 输出层
        if hasattr(self, 'gate_linear') and m == self.gate_linear:
             if m.bias is not None:
                # 初始化为 0.5 (Sigmoid(0.5) ~= 0.62) 
                # 鼓励在训练初期进行更多的连接探索，避免一开始就陷入全断开的局部最优
                nn.init.constant_(m.bias, 0.5)

    def forward(self, state):
        """
        前向传播
        Args:
            state: 状态张量 (Batch, N * (M*2 + 3))
        Returns:
            mean: 功率分布均值 (Batch, N*M)
            log_std: 功率分布对数标准差 (Batch, N*M)
            gate_logits: 连接概率 Logits (Batch, N*M)
        """
        batch_size = state.shape[0]
        
        # --- 1. 状态解析 (Parse State) ---
        # 将扁平化的状态向量重塑为结构化数据
        # x: (Batch, N, M*2 + 3)
        x = state.view(batch_size, self.N, self.M * 2 + 3)
        
        # 分离链路特征和位置特征
        # link_part: (Batch, N, M*2) -> (Batch, N, M, 2) [LogBeta, Angle]
        link_part = x[:, :, :self.M * 2]
        # pos_part: (Batch, N, 3) [x, y, z]
        pos_part = x[:, :, self.M * 2:]
        
        bs_features = link_part.view(batch_size, self.N, self.M, 2)
        uav_pos = pos_part 
        
        # --- 2. 特征嵌入 (Embeddings) ---
        # BS Embed: (Batch, N, M, Embed)
        bs_embed = self.bs_encoder(bs_features)
        
        # UAV Embed: (Batch, N, Embed) -> (Batch, N, 1, Embed)
        # 增加一个维度以便于 Attention 处理
        uav_embed = self.uav_encoder(uav_pos).unsqueeze(2)
        
        # --- 3. 交叉注意力 (Intra-UAV: UAV selects BS) ---
        # 这里我们将 Batch 和 N 维度合并，因为每个 UAV 在这一步是独立处理其周围基站的
        bs_embed_flat = bs_embed.view(batch_size * self.N, self.M, self.embed_dim)
        uav_embed_flat = uav_embed.view(batch_size * self.N, 1, self.embed_dim)
        
        # Query: UAV, Key/Value: BS
        # uav_context_flat: (Batch * N, 1, Embed)
        uav_context_flat, _ = self.cross_attention(uav_embed_flat, bs_embed_flat, bs_embed_flat)
        
        # 残差连接 (Residual Connection)
        uav_context_flat = uav_context_flat + uav_embed_flat
        
        # 恢复形状: (Batch, N, Embed)
        uav_context = uav_context_flat.view(batch_size, self.N, self.embed_dim)
        
        # --- 4. 自注意力 (Inter-UAV: Coordination) ---
        # 现在 UAV 之间进行交互
        # global_context: (Batch, N, Embed)
        global_context, _ = self.self_attention(uav_context, uav_context, uav_context)
        
        # 残差连接
        global_context = global_context + uav_context
        
        # --- 5. 解码与动作生成 (Decoding) ---
        # 将全局上下文扩展到每个基站维度，以便与基站嵌入拼接
        # (Batch, N, 1, Embed) -> (Batch, N, M, Embed)
        global_context_expanded = global_context.unsqueeze(2).expand(-1, -1, self.M, -1)
        
        # 拼接: (Batch, N, M, Embed * 2)
        # 结合了 "这个基站怎么样" (bs_embed) 和 "我现在整体情况如何" (global_context)
        decoder_input = torch.cat([bs_embed, global_context_expanded], dim=3)
        
        # 通过解码器 MLP
        # (Batch, N, M, Hidden)
        dec_out = self.decoder(decoder_input)
        
        # 输出头: (Batch, N, M, 1)
        mean = self.mean_linear(dec_out)
        log_std = self.log_std_linear(dec_out)
        gate_logits = self.gate_linear(dec_out)
        
        # 展平为 (Batch, N * M) 以符合 SAC 接口
        mean = mean.view(batch_size, -1)
        log_std = log_std.view(batch_size, -1)
        gate_logits = gate_logits.view(batch_size, -1)
        
        # 限制 log_std 范围，防止数值不稳定
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, gate_logits

    def sample(self, state):
        """
        采样动作
        Args:
            state: 状态张量
        Returns:
            action: 最终组合动作 (Power * Gate)
            log_prob: 动作的对数概率
            mean: 功率均值
            log_std: 功率对数标准差
            y_soft: 门控概率 (用于稀疏性损失计算)
        """
        mean, log_std, gate_logits = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # --- 连续动作 (Power) ---
        # 重参数化采样 (Reparameterization Trick)
        x_t = normal.rsample() 
        # Tanh 变换: 将高斯分布映射到 (-1, 1)
        y_t = torch.tanh(x_t)
        # 缩放到 [0, 1] 区间作为功率比例
        power_action = (y_t + 1) / 2
        
        # --- 离散动作 (Gate) ---
        # 使用 Gumbel-Sigmoid 进行可微的离散采样
        if self.training:
            # 训练阶段: 加入 Gumbel 噪声以进行探索
            # gumbel = -log(-log(u))
            u = torch.rand_like(gate_logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            temp = 1.0 # 温度系数，控制近似程度
            y_soft = torch.sigmoid((gate_logits + gumbel_noise) / temp)
        else:
            # 测试阶段: 确定性输出或直接使用 Sigmoid 概率
            y_soft = torch.sigmoid(gate_logits)
            
        # 直通估计器 (Straight-Through Estimator, STE)
        # 前向传播使用硬阈值 (0 或 1)，反向传播使用软梯度 (Sigmoid)
        y_hard = (y_soft > 0.5).float()
        mask = (y_hard - y_soft).detach() + y_soft
        
        # --- 组合动作 ---
        # 最终动作 = 功率 * 连接掩码
        # 如果 mask=0，则功率强制为 0；如果 mask=1，则功率为 power_action
        action = power_action * mask
        
        # --- 计算 Log Probability ---
        # 1. 连续动作 (Power) 的 Log Prob
        log_prob = normal.log_prob(x_t)
        # Tanh 变换的雅可比行列式修正
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        # 缩放修正 (因为我们做了 (y+1)/2)
        log_prob -= torch.log(torch.tensor(2.0))
        
        # 2. 离散动作 (Gate) 的 Log Prob
        # 我们需要计算所选动作 (mask) 的对数概率
        # mask 是 0 或 1 (经过 STE)
        # y_soft 是选 1 的概率 P(gate=1)
        # log P(gate) = mask * log(y_soft) + (1-mask) * log(1-y_soft)
        epsilon = 1e-6
        gate_log_prob = mask * torch.log(y_soft + epsilon) + (1 - mask) * torch.log(1 - y_soft + epsilon)
        
        # 3. 合并 Log Prob
        # 假设连续动作和离散动作是独立的，联合概率是乘积，Log 概率是和
        total_log_prob = log_prob + gate_log_prob
        
        # 对所有维度 (N*M) 求和得到样本的 Log Prob
        total_log_prob = total_log_prob.sum(1, keepdim=True)
        
        # 返回 y_soft 是为了在外部计算稀疏性正则化损失 (L1 Loss 等)
        return action, total_log_prob, mean, log_std, y_soft


class Critic(nn.Module):
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
        super(Critic, self).__init__()
        
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
                                a_uav_enc, a_bs_act_enc, a_cross, a_self, a_head):
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