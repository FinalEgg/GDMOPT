import torch
import torch.nn as nn
from model.actor.actor import Actor
from model.diffusion.diffusion import Diffusion
from model.diffusion.model import MLP
from .config import CombinedConfig

class CombinedModel(nn.Module):
    def __init__(self, state_dim, connection_dim, power_dim, config=None,
                 threshold=None, actor_hidden_dim=None, diffusion_hidden_dim=None,
                 diffusion_timesteps=None, max_power=None,
                 diffusion_beta_schedule=None,
                 use_soft_threshold=None, soft_threshold_temperature=None):
        # Use provided config or create a new one
        self.config = config if config is not None else CombinedConfig()
        super(CombinedModel, self).__init__()
        
        # 保存维度信息
        self.state_dim = state_dim
        self.connection_dim = connection_dim
        self.power_dim = power_dim
        
        # 从config获取参数并支持用户覆盖
        self.threshold = threshold if threshold is not None else self.config.threshold
        self.max_power = max_power if max_power is not None else self.config.max_power
        self.actor_hidden_dim = actor_hidden_dim if actor_hidden_dim is not None else self.config.actor_hidden_dim
        self.diffusion_hidden_dim = diffusion_hidden_dim if diffusion_hidden_dim is not None else self.config.diffusion_hidden_dim
        self.diffusion_timesteps = diffusion_timesteps if diffusion_timesteps is not None else self.config.diffusion_timesteps
        self.diffusion_beta_schedule = diffusion_beta_schedule if diffusion_beta_schedule is not None else self.config.diffusion_beta_schedule
        self.use_soft_threshold = use_soft_threshold if use_soft_threshold is not None else self.config.use_soft_threshold
        self.soft_threshold_temperature = soft_threshold_temperature if soft_threshold_temperature is not None else self.config.soft_threshold_temperature
        
        # 离散决策网络 - 用于确定无人机与地面基站之间的连接情况
        self.connection_actor = Actor(state_dim, connection_dim, hidden_dim=self.actor_hidden_dim)
        
        # 为Diffusion模型创建MLP网络
        diffusion_model = MLP(
            state_dim=state_dim + connection_dim,  # 状态空间 + 连接情况作为输入
            action_dim=power_dim,
            hidden_dim=self.diffusion_hidden_dim
        )
        
        # Diffusion模型 - 用于功率分配
        self.diffusion = Diffusion(
            state_dim=state_dim + connection_dim,  # 状态空间 + 连接情况作为输入
            action_dim=power_dim,
            model=diffusion_model,
            max_action=self.max_power,
            n_timesteps=self.diffusion_timesteps,
            beta_schedule=self.diffusion_beta_schedule
        )
    
    def apply_threshold(self, connection_prob, threshold=None):
        """应用阈值将概率转换为0/1连接矩阵
        
        Args:
            connection_prob: 连接概率矩阵
            threshold: 可选的自定义阈值，如果不提供则使用默认阈值
            
        Returns:
            二进制连接矩阵
        """
        # 如果没有提供阈值，使用默认阈值
        if threshold is None:
            threshold = self.threshold
        
        # 根据是否使用软阈值选择不同的转换方法
        if self.use_soft_threshold and self.training:
            # 软阈值 - 使用sigmoid进行平滑过渡
            # (connection_prob - threshold) 使阈值点在0处，然后通过温度参数控制平滑度
            soft_output = torch.sigmoid((connection_prob - threshold) / self.soft_threshold_temperature)
            return soft_output
        else:
            # 硬阈值 - 直接二值化
            return (connection_prob > threshold).float()
    
    def forward(self, state, threshold=None, *args, **kwargs):
        # 第一部分：使用Actor网络预测连接概率
        connection_prob = self.connection_actor(state)
        
        # 应用阈值将概率转换为0/1连接矩阵
        connection_matrix = self.apply_threshold(connection_prob, threshold)
        
        # 准备Diffusion模型的输入：拼接状态和连接矩阵
        diffusion_input = torch.cat([state, connection_matrix], dim=1)
        
        # 第二部分：使用Diffusion模型进行功率分配
        power_allocation = self.diffusion(diffusion_input, *args, **kwargs)
        
        # 确保功率分配在有效范围内
        power_allocation = torch.clamp(power_allocation, 0, self.max_power)
        
        # 返回连接情况和功率分配作为整体输出
        return connection_matrix, power_allocation
    
    def get_connection_prob(self, state):
        """获取原始连接概率输出"""
        return self.connection_actor(state)
    
    def set_threshold(self, threshold):
        """动态设置阈值参数"""
        self.threshold = threshold
    
    def set_soft_threshold(self, use_soft, temperature=1.0):
        """设置是否使用软阈值及温度参数
        
        Args:
            use_soft: 是否使用软阈值
            temperature: 软阈值的温度参数，值越小过渡越陡峭
        """
        self.use_soft_threshold = use_soft
        self.soft_threshold_temperature = temperature
    
    def get_binary_connection(self, state, threshold=None):
        """获取二值化的连接矩阵，无论是否启用了软阈值
        
        Args:
            state: 输入状态
            threshold: 可选的自定义阈值
            
        Returns:
            二进制连接矩阵
        """
        connection_prob = self.connection_actor(state)
        # 强制使用硬阈值
        return (connection_prob > (threshold if threshold is not None else self.threshold)).float()
    
    def loss(self, state, target_connection, target_power, weights=1.0):
        """计算组合模型的损失
        
        Args:
            state: 输入状态
            target_connection: 目标连接矩阵
            target_power: 目标功率分配
            weights: 可选的损失权重
            
        Returns:
            组合损失值
        """
        # 计算连接预测损失
        connection_prob = self.connection_actor(state)
        connection_loss = nn.BCELoss()(connection_prob, target_connection)
        
        # 使用目标连接矩阵计算功率分配损失
        diffusion_input = torch.cat([state, target_connection], dim=1)
        power_loss = self.diffusion.loss(target_power, diffusion_input, weights)
        
        # 返回组合损失
        return connection_loss + power_loss
    
    def sample(self, state, verbose=False, return_diffusion=False, threshold=None):
        """生成样本，使用与diffusion模型相同的接口
        
        Args:
            state: 输入状态
            verbose: 是否显示进度
            return_diffusion: 是否返回完整的扩散过程
            threshold: 可选的自定义阈值
            
        Returns:
            连接矩阵和功率分配（以及可选的扩散过程）
        """
        # 获取连接矩阵
        connection_matrix = self.get_binary_connection(state, threshold)
        
        # 准备Diffusion模型的输入
        diffusion_input = torch.cat([state, connection_matrix], dim=1)
        
        # 使用diffusion模型的sample方法
        if return_diffusion:
            power_allocation, diffusion = self.diffusion.sample(
                diffusion_input, verbose=verbose, return_diffusion=return_diffusion
            )
            return connection_matrix, power_allocation, diffusion
        else:
            power_allocation = self.diffusion.sample(
                diffusion_input, verbose=verbose, return_diffusion=return_diffusion
            )
            return connection_matrix, power_allocation
    
    def forward(self, state, threshold=None, output_mode='all', *args, **kwargs):
        """前向传播函数
        
        Args:
            state: 输入状态
            threshold: 可选的自定义阈值
            output_mode: 输出模式，可选值：
                - 'all': 返回连接矩阵和功率分配
                - 'connection': 只返回连接矩阵
                - 'power': 只返回功率分配
                - 'prob_and_connection': 返回连接概率和连接矩阵
            *args, **kwargs: 传递给diffusion模型的额外参数
            
        Returns:
            根据output_mode返回不同的输出组合
        """
        # 第一部分：使用Actor网络预测连接概率
        connection_prob = self.connection_actor(state)
        
        # 应用阈值将概率转换为0/1连接矩阵
        connection_matrix = self.apply_threshold(connection_prob, threshold)
        
        # 根据输出模式返回不同的结果
        if output_mode == 'connection':
            return connection_matrix
        elif output_mode == 'prob_and_connection':
            return connection_prob, connection_matrix
        
        # 对于需要功率分配的模式，继续计算
        # 准备Diffusion模型的输入：拼接状态和连接矩阵
        diffusion_input = torch.cat([state, connection_matrix], dim=1)
        
        # 第二部分：使用Diffusion模型进行功率分配
        power_allocation = self.diffusion(diffusion_input, *args, **kwargs)
        
        # 确保功率分配在有效范围内
        power_allocation = torch.clamp(power_allocation, 0, self.max_power)
        
        # 根据输出模式返回结果
        if output_mode == 'power':
            return power_allocation
        else:  # 'all' or any other value
            return connection_matrix, power_allocation
    
    def batch_forward(self, states, thresholds=None, output_mode='all'):
        """批量处理多个状态
        
        Args:
            states: 状态批次，形状为(batch_size, state_dim)
            thresholds: 可选的阈值批次或单个阈值
            output_mode: 输出模式，与forward相同
            
        Returns:
            批量处理的结果
        """
        # 确保输入是批处理格式
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        # 如果提供了阈值批次，确保其形状与状态批次匹配
        if thresholds is not None and torch.is_tensor(thresholds) and thresholds.dim() > 0:
            if len(thresholds) != len(states):
                raise ValueError(f"Thresholds length ({len(thresholds)}) must match states batch size ({len(states)})")
            
            # 对每个状态单独应用不同的阈值
            results = []
            for i in range(len(states)):
                result = self.forward(states[i:i+1], threshold=thresholds[i], output_mode=output_mode)
                results.append(result)
            
            # 根据输出模式合并结果
            if output_mode == 'all':
                # 合并连接矩阵和功率分配
                connection_matrices = torch.cat([r[0] for r in results], dim=0)
                power_allocations = torch.cat([r[1] for r in results], dim=0)
                return connection_matrices, power_allocations
            elif output_mode == 'prob_and_connection':
                # 合并连接概率和连接矩阵
                connection_probs = torch.cat([r[0] for r in results], dim=0)
                connection_matrices = torch.cat([r[1] for r in results], dim=0)
                return connection_probs, connection_matrices
            else:
                # 对于单一输出类型，直接合并
                return torch.cat(results, dim=0)
        else:
            # 使用相同的阈值处理整个批次
            return self.forward(states, threshold=thresholds, output_mode=output_mode)