import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.continuous import Actor as TsActor
from tianshou.utils.net.common import MLP

class TD3Net(nn.Module):
    """
    改进的特征提取网络，添加了残差连接和层归一化。
    架构: fc1 -> LN -> Mish -> fc2 -> LN -> Mish -> fc3 -> LN -> Mish.
    
    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int): 隐藏层维度
    """
    def __init__(self, input_dim, hidden_dim):
        super(TD3Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.output_dim = hidden_dim
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 使用更适合Mish的初始化策略
        # 根据研究，Mish表现类似于GELU，使用1.0作为gain更合适
        gain = 1.0
        
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
        nn.init.constant_(self.fc2.bias, 0)
        
        # 使用正交初始化确保良好的特征提取
        nn.init.orthogonal_(self.fc3.weight, gain=gain)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, obs, state=None, info={}):
        """
        具有残差连接的前向传播
        """
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, device=self.fc1.weight.device, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        # 第一层 (没有残差连接)
        x = self.fc1(obs)
        x = self.ln1(x)
        x = F.mish(x)
        
        # 第二层 (添加残差连接)
        identity = x
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.mish(x)
        x = x + identity  # 残差连接
        
        # 第三层 (添加残差连接)
        identity = x
        x = self.fc3(x)
        x = self.ln3(x)
        x = F.mish(x)
        x = x + identity  # 残差连接
        
        return x, state

class TD3Actor(TsActor):
    """
    优化后的TD3 Actor，使用改进的特征提取网络和权重初始化
    """
    def __init__(self, input_dim, hidden_dim, output_dim, max_action=1.0, device="cuda"):
        # 构建特征提取网络
        preprocess_net = TD3Net(input_dim, hidden_dim)
        # 使用空hidden_sizes列表创建单一线性层映射
        super(TD3Actor, self).__init__(
            preprocess_net=preprocess_net,
            action_shape=(output_dim,),
            hidden_sizes=[],
            max_action=max_action,
            device=device,
            preprocess_net_output_dim=hidden_dim,
        )
        # 重新初始化最后一层，使用适当的增益值平衡初始输出范围和梯度流动
        last_layer = self.last.model[-1] if hasattr(self.last, "model") else self.last
        nn.init.orthogonal_(last_layer.weight, gain=0.1)  # 从0.01增加到0.1，避免梯度消失
        nn.init.constant_(last_layer.bias, 0)
    
    def forward(self, obs, state=None, info={}):
        """
        Actor网络的前向传播
        """
        # 通过预处理网络提取特征
        features, state = self.preprocess(obs, state, info)
        # 将特征映射到初步动作逻辑
        action_logits = self.last(features)
        # 使用tanh将最终动作限定在范围内并按max_action缩放
        # 将除数从5调整为2.5，在保持输出范围的同时增强梯度流动
        action = self._max * torch.tanh(action_logits / 2.5)  
        return action, state

    def clip_weights(self, clip_value=0.02):
        """
        裁剪最后一层的权重以防止过高的输出
        """
        if hasattr(self.last, "model"):
            for param in self.last.model[-1].parameters():
                param.data.clamp_(-clip_value, clip_value)
        else:
            for param in self.last.parameters():
                param.data.clamp_(-clip_value, clip_value)
    
    def clip_gradients(self, max_norm=1.0):
        """
        裁剪梯度范数以防止梯度爆炸
        
        Args:
            max_norm (float): 梯度裁剪的最大范数
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)