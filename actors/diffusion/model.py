import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb
from tianshou.data import Batch, ReplayBuffer, to_torch

import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb
from tianshou.data import Batch, ReplayBuffer, to_torch

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, act_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_fn(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

class MLP(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation='mish'
    ):
        super(MLP, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU
        
        # 改进后的状态处理网络
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 时间处理网络保持不变，但增加Layer Normalization
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.LayerNorm(t_dim * 2),
            nn.Linear(t_dim * 2, t_dim),
            nn.LayerNorm(t_dim)
        )
        
        # 输入处理层
        self.input_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, _act) 
            for _ in range(2)  # 使用2个残差块增加深度
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, time, state):
        processed_state = self.state_mlp(state)
        t = self.time_mlp(time)
        x = torch.cat([x, t, processed_state], dim=1)
        
        # 通过新的网络结构
        x = self.input_layer(x)
        
        # 应用残差块
        for block in self.res_blocks:
            x = block(x)
            
        # 输出层
        x = self.output_layer(x)
        
        return x


class DoubleCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            activation='mish'
    ):
        super(DoubleCritic, self).__init__()
        # _act = nn.Mish if activation == 'mish' else nn.ReLU
        _act = nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.q1_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))
        self.q2_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))
    # def forward(self, obs):
    #     return self.q1_net(obs), self.q2_net(obs)
    #
    # def q_min(self, obs):
    #     return torch.min(*self.forward(obs))
    def forward(self, state, action):
        processed_state = self.state_mlp(state)
        x = torch.cat([processed_state, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def q_min(self, obs, action):
        # obs = to_torch(obs, device='cuda:0', dtype=torch.float32)
        # action = to_torch(action, device='cuda:0', dtype=torch.float32)
        return torch.min(*self.forward(obs, action))
