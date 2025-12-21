import torch
import torch.nn as nn

class HGNNLayer(nn.Module):
    def __init__(self, embed_dim):
        super(HGNNLayer, self).__init__()
        self.embed_dim = embed_dim
        
        # 1. Interference Message Passing (UAV -> UAV)
        self.int_msg_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.int_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.uav_update_1 = nn.LayerNorm(embed_dim)

        # 2. Downlink Message Passing (BS -> UAV)
        self.dl_msg_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.dl_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.uav_update_2 = nn.LayerNorm(embed_dim)
        
        # 3. Uplink Message Passing (UAV -> BS)
        self.ul_msg_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.ul_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.bs_update = nn.LayerNorm(embed_dim)

    def forward(self, uav_feats, bs_feats, edge_feats, uav_dists):
        B, N, _ = uav_feats.shape
        M = bs_feats.shape[1]
        
        # 1. UAV Interference
        uav_i = uav_feats.unsqueeze(2).expand(-1, -1, N, -1)
        uav_j = uav_feats.unsqueeze(1).expand(-1, N, -1, -1)
        int_input = torch.cat([uav_j, uav_i, uav_dists], dim=-1)
        int_msg = self.int_msg_mlp(int_input)
        
        q_int = uav_feats.view(B * N, 1, self.embed_dim)
        kv_int = int_msg.view(B * N, N, self.embed_dim)
        int_agg, _ = self.int_attn(q_int, kv_int, kv_int)
        int_agg = int_agg.view(B, N, self.embed_dim)
        uav_feats = self.uav_update_1(uav_feats + int_agg)
        
        # 2. Downlink
        bs_expanded = bs_feats.unsqueeze(1).expand(-1, N, -1, -1)
        uav_expanded = uav_feats.unsqueeze(2).expand(-1, -1, M, -1)
        dl_input = torch.cat([bs_expanded, uav_expanded, edge_feats], dim=-1)
        dl_msg = self.dl_msg_mlp(dl_input)
        
        q_dl = uav_feats.view(B * N, 1, self.embed_dim)
        kv_dl = dl_msg.view(B * N, M, self.embed_dim)
        dl_agg, _ = self.dl_attn(q_dl, kv_dl, kv_dl)
        dl_agg = dl_agg.view(B, N, self.embed_dim)
        uav_feats = self.uav_update_2(uav_feats + dl_agg)
        
        # 3. Uplink
        ul_input = torch.cat([uav_expanded, bs_expanded, edge_feats], dim=-1)
        ul_msg = self.ul_msg_mlp(ul_input)
        ul_msg_trans = ul_msg.transpose(1, 2)
        
        q_ul = bs_feats.view(B * M, 1, self.embed_dim)
        kv_ul = ul_msg_trans.reshape(B * M, N, self.embed_dim)
        ul_agg, _ = self.ul_attn(q_ul, kv_ul, kv_ul)
        ul_agg = ul_agg.view(B, M, self.embed_dim)
        bs_feats = self.bs_update(bs_feats + ul_agg)
        
        return uav_feats, bs_feats

class GNNEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GNNEncoder, self).__init__()
        
        # Infer N and M
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
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # GNN Layers
        self.layers = nn.ModuleList([HGNNLayer(self.embed_dim) for _ in range(2)])
        
        self.output_dim = self.embed_dim * 3

    def forward(self, state):
        batch_size = state.shape[0]
        
        # Parse State
        x = state.view(batch_size, self.N, self.M * 2 + 3)
        link_part = x[:, :, :self.M * 2].view(batch_size, self.N, self.M, 2)
        uav_pos = x[:, :, self.M * 2:]
        
        # Initialize Features
        uav_feats = self.uav_encoder(uav_pos)
        bs_ids = torch.arange(self.M, device=state.device).expand(batch_size, -1)
        bs_feats = self.bs_embedding(bs_ids)
        edge_feats = self.edge_encoder(link_part)
        
        p1 = uav_pos.unsqueeze(2)
        p2 = uav_pos.unsqueeze(1)
        uav_dists = torch.norm(p1 - p2, dim=-1, keepdim=True)
        
        # Message Passing
        for layer in self.layers:
            uav_feats, bs_feats = layer(uav_feats, bs_feats, edge_feats, uav_dists)
            
        # Prepare Output
        uav_expanded = uav_feats.unsqueeze(2).expand(-1, -1, self.M, -1)
        bs_expanded = bs_feats.unsqueeze(1).expand(-1, self.N, -1, -1)
        
        # (B, N, M, 3*E)
        output = torch.cat([uav_expanded, bs_expanded, edge_feats], dim=-1)
        
        return output
