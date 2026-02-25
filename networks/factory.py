import torch
import torch.nn as nn
from config.model_config import ModelConfig
from networks.blocks.mlp import MLPBlock
from networks.blocks.deepsets import DeepSetsBlock
from networks.wrappers.common_heads import DeterministicHead, StochasticHead, CriticHead
from networks.wrappers.diffusion import DiffusionWrapper

class ReshapeModule(nn.Module):
    """
    Reshape Module.
    重塑模块。
    
    Function (功能):
    - Transforms flattened input back to set structure for DeepSets.
    - 将展平的输入 (Flat Input) 还原为集合结构 (Set Structure)。
    
    Dims (维度):
    - Input: (Batch, N * F)
    - Output: (Batch, N, F)
    """
    def __init__(self, num_entities, features_per_entity):
        super().__init__()
        self.n = int(num_entities) # 实体数量 (N)
        self.f = int(features_per_entity) # 每个实体的特征维度 (F)
    
    def forward(self, x):
        """
        Forward pass.
        前向传播。
        """
        if x.dim() == 2:
            return x.reshape(x.size(0), self.n, self.f)
        return x

class ModelFactory:
    """
    Neural Network Factory.
    神经网络工厂类。
    
    Responsibilities (职责):
    1. Assemble diverse neural network architectures based on config.
       (根据配置组装各种神经网络架构)
    2. Manage components: Backbones (Blocks) + Heads + Wrappers.
       (管理组件：骨干网络 + 输出头 + 包装器)
    3. Support various algorithms: DDPG, TD3, SAC, Diffusion.
       (支持多种算法)
    
    Architecture (架构):
    - Actor (策略网络): Backbone -> Head (Deterministic/Stochastic) OR Diffusion Wrapper.
    - Critic (价值网络): Backbone (Input: State+Action) -> Head (Value).
    """
    
    @staticmethod
    def _infer_set_dimensions(state_dim, action_dim):
        """
        Infer functional dimensions N and F.
        推断集合维度 N 和特征维度 F。
        
        Currently acts as a placeholder or helper.
        当前作为占位符或辅助函数。
        """
        return None, None

    @staticmethod
    def create_actor(model_id: str, state_dim: int, action_dim: int, algo: str, max_action: float = 1.0, device: str = 'cpu') -> nn.Module:
        """
        Create Actor Network.
        创建策略网络 (Actor)。
        
        Args:
            model_id (str): ID to look up in ModelConfig (模型配置ID).
            state_dim (int): Dimension of state input (状态维度).
            action_dim (int): Dimension of action output (动作维度).
            algo (str): Algorithm name ('sac', 'td3', 'ddpg', 'diffusion').
            max_action (float): Action scaling factor (动作缩放).
            device (str): Computation device (设备).
            
        Returns:
            nn.Module: Constructed Actor model.
        """
        config = ModelConfig.get_config(model_id)
        model_type = config.get('type', 'mlp')
        
        # 1. Build Backbone / Core Model (构建骨干网络)
        backbone = None
        feature_dim = 0
        
        if model_type == 'mlp':
            if algo == 'diffusion':
                # Diffusion Backbone Input: State + Action + Time (1)
                # 扩散模型骨干输入：状态 + 动作(带噪) + 时间步 embedding(此处简化为1维输入或由Wrapper处理)
                # 注意：具体时间嵌入通常在 DiffusionWrapper 内部或 MLPBlock 的变体中处理。
                # 此处简化实现：我们将 State+Action+t 作为输入拼接到 MLP。
                input_dim = state_dim + action_dim + 1
                
                # For Diffusion, the "Backbone" essentially predicts the noise directly.
                # 对于扩散模型，骨干网络直接预测噪声（输出维度 = 动作维度）。
                core_model = MLPBlock(input_dim, output_dim=action_dim, config=config)
                
                # Wrap with Diffusion Logic (Sampling, Loss)
                # 使用扩散逻辑包装 (包含采样过程、损失计算)
                model = DiffusionWrapper(core_model, state_dim, action_dim, max_action, 
                                         beta_schedule=config.get('beta_schedule', 'linear'),
                                         n_timesteps=config.get('diffusion_steps', 100))
                return model.to(device)
            else:
                # Standard Actor Backbone: Input = State
                # 标准 Actor 骨干：输入为状态
                backbone = MLPBlock(state_dim, output_dim=None, config=config)
                feature_dim = backbone.output_dim
        
        elif model_type == 'deepsets':
            # DeepSets handling (DeepSets 处理)
            # We need to reshape flat input to set input.
            # 需要将展平的 Gym 观测 (Flat) 重塑为集合 (Set) 结构。
            
            # Strategy: Look for explicit 'features_per_entity' in config
            # 策略：从配置中读取每个实体的特征维度 'features_per_entity'
            feat_per_entity = config.get('features_per_entity', None)
            
            if feat_per_entity:
                num_entities = state_dim // feat_per_entity
                reshape_layer = ReshapeModule(num_entities, feat_per_entity)
                
                # DeepSets Block (DeepSets 模块)
                ds_block = DeepSetsBlock(input_dim=feat_per_entity, output_dim=None, config=config)
                feature_dim = ds_block.output_dim
                
                # Chain: Reshape -> DeepSets
                # 串联：重塑层 -> DeepSets 骨干
                backbone = nn.Sequential(reshape_layer, ds_block)
            else:
                # Fallback or Error
                # 安全起见，如果配置缺失则报错
                raise ValueError("DeepSets config requires 'features_per_entity' to be set when using flat Env observations.")

        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # 2. Attach Head (For non-diffusion actors) (附加输出头)
        model = None
        if algo == 'sac':
            # Soft Actor-Critic: Gaussian Policy Head (输出均值和标准差)
            model = StochasticHead(backbone, feature_dim, action_dim)
        elif algo in ['td3', 'ddpg']:
            # Deterministic Policy Head (输出确定性动作)
            model = DeterministicHead(backbone, feature_dim, action_dim)
        else:
            raise ValueError(f"Unknown algo for Actor creation: {algo}")
            
        return model.to(device)

    @staticmethod
    def create_critic(model_id: str, state_dim: int, action_dim: int, device: str = 'cpu') -> nn.Module:
        """
        Create Critic Network.
        创建价值网络 (Critic)。
        
        Args:
            model_id (str): Model Config ID.
            state_dim (int): State dimension.
            action_dim (int): Action dimension.
            device (str): Device.
            
        Returns:
            nn.Module: Critic model (Q-function).
        """
        config = ModelConfig.get_config(model_id)
        model_type = config.get('type', 'mlp')
        
        # Critic Input is usually (State, Action) concatenated
        # Critic 输入通常是 (State, Action) 的拼接
        
        if model_type == 'mlp':
            input_dim = state_dim + action_dim
            backbone = MLPBlock(input_dim, output_dim=None, config=config)
            feature_dim = backbone.output_dim
            
            # Critic Head outputs scalar Q-value
            # Critic 头输出标量 Q 值
            model = CriticHead(backbone, feature_dim)
            
        elif model_type == 'deepsets':
            # DeepSets Critic Logic (DeepSets Critic 逻辑)
            # Typically requires careful design on where to inject Action.
            # 标准做法：(State, Action) -> Concat per element -> Phi -> Rho -> Q
            # 或者是：State -> Phi -> Rho -> Global -> Concat Action -> MLP -> Q
            
            # Currently not implemented for this baseline.
            # 当前未实现 DeepSets 形式的 Critic (通常 Actor 用 DeepSets，Critic 可用 MLP 处理 Global State + Action)。
            raise NotImplementedError("DeepSets Critic factory not yet implemented.")
            
        else:
             raise ValueError(f"Unknown model type: {model_type}")
             
        return model.to(device)


def get_model(model_id: str, role: str, state_shape: int, action_shape: int, algo: str = None, device: str = 'cpu'):
    """
    Facade function to retrieve models.
    获取模型的统一入口函数。
    
    Args:
        model_id (str): Config ID.
        role (str): 'actor' or 'critic'.
        state_shape (int/tuple): Shape of state.
        action_shape (int/tuple): Shape of action.
        algo (str): Algorithm name (required for actor).
        device (str): Device.
    """
    state_dim = state_shape
    action_dim = action_shape
    
    # Handle Tuple Shapes from Gym (处理 Gym 的 Tuple 形状)
    if hasattr(state_dim, "__len__"): 
        import numpy as np
        state_dim = int(np.prod(state_dim))
    if hasattr(action_dim, "__len__"):
        import numpy as np
        action_dim = int(np.prod(action_dim))
        
    if role == 'actor':
        if algo is None:
            raise ValueError("Algo required for Actor creation (e.g. 'ddpg', 'sac').")
        return ModelFactory.create_actor(model_id, state_dim, action_dim, algo, device=device)
    elif role == 'critic':
        return ModelFactory.create_critic(model_id, state_dim, action_dim, device=device)
    else:
        raise ValueError(f"Unknown role: {role}")
