from typing import List, Optional, Dict, Union

class ModelConfig:
    """
    模型架构配置 (Model Architecture Configuration).
    定义神经网络的具体结构参数 (层数, 维度, 激活函数等)。
    ModelFactory 使用此注册表来构建具体的 PyTorch 模型。
    """
    
    REGISTRY = {
        # ---------------------------------------------------------------------
        # MLP Variants
        # ---------------------------------------------------------------------
        "mlp_baseline": {
            "type": "mlp",
            "hidden_sizes": [256, 256],
            "activation": "relu",
            "use_layer_norm": False,
            "use_residual": False,
            "dropout": 0.0
        },
        
        "mlp_deep_res": {
            "type": "mlp",
            "hidden_sizes": [256, 256, 256, 256],
            "activation": "relu",
            "use_layer_norm": True,
            "use_residual": True, # Specific handling in Factory needed
            "dropout": 0.1
        },

        # ---------------------------------------------------------------------
        # DeepSets Variants (Permutation Invariant)
        # ---------------------------------------------------------------------
        "deepset_standard": {
            "type": "deepsets",
            "phi_hidden_sizes": [128, 128], # Encoder per element
            "rho_hidden_sizes": [128, 64],  # Decoder after aggregation
            "activation": "relu",
            "aggregation": "mean" # 'mean', 'sum', 'max'
        },
        
        # ---------------------------------------------------------------------
        # GNN Variants
        # ---------------------------------------------------------------------
        "gnn_gat": {
            "type": "gnn",
            "gnn_type": "gat", # Graph Attention Network
            "hidden_dim": 128,
            "num_layers": 2,
            "heads": 4,
            "activation": "elu"
        }
    }

    @classmethod
    def get_config(cls, model_id: str) -> Dict:
        if model_id not in cls.REGISTRY:
            raise ValueError(f"Model ID '{model_id}' not found in registry. Available: {list(cls.REGISTRY.keys())}")
        return cls.REGISTRY[model_id]
