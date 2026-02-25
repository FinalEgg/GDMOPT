from typing import Dict
from envs.cellfree_env import CellFreeEnv
from envs.fix_topp_env import FixTopPEnv
from envs.optimization_env import QuadraticEnv, RastriginEnv
from envs.classic_env import CartPoleEnv, PendulumEnv
from envs.wrappers import FlattenObservationWrapper, PurePowerActionWrapper, ThresholdActionWrapper, HybridActionWrapper


class MappingConfig:
    """
    映射配置 (Mapping Configuration).
    定义组件间的组合关系：Experiment -> (Environment, Wrapper, Model).
    作用：
    1. 注册表 (Registry): 集中管理所有可用组件类。
    2. 实验定义 (Experiment Definitions): 将组件组合为具体的实验方案。
    使用方法: 通过 Experiment ID 从 EXPERIMENT_MAP 查找对应的 Env/Wrapper/Model 配置。
    """
    
    # 1. 环境注册表
    # 将字符串 ID 映射到环境类
    ENV_REGISTRY = {
        "fix_topp": FixTopPEnv,
        "cell_free": CellFreeEnv,
        "quadratic": QuadraticEnv,
        "rastrigin": RastriginEnv,
        "cartpole": CartPoleEnv,
        "pendulum": PendulumEnv
    }


    
    # 2. 包装器配置
    # 定义给定设置应用哪些包装器
    # 键: config_name 
    # 值: 包装器类列表 或 (包装器类, 参数字典)
    WRAPPER_REGISTRY = {
        "default_mlp": [
            (FlattenObservationWrapper, {}),
            (PurePowerActionWrapper, {})
        ],
        "threshold_mlp": [
            (FlattenObservationWrapper, {}),
            (ThresholdActionWrapper, {"threshold": 0.2})
        ],
        "hybrid_mlp": [
            (FlattenObservationWrapper, {}),
            (HybridActionWrapper, {})
        ],
        "raw_gnn": [
             # GNN 可能接受原始字典观测，因此不需要扁平化包装器
             (PurePowerActionWrapper, {})
        ],
        "none": []
    }
    

    # 3. 模型注册表 (来自 ModelConfig 的配置 ID)
    # 我们将 "场景模型 ID" 映射到底层 "架构 ID"
    # 这允许实验指向相同的架构
    MODEL_ID_MAP = {
        "baseline": "mlp_baseline",
        "deep_residual": "mlp_deep_res",
        "deepset_v1": "deepset_standard",
        "gnn_v1": "gnn_gat"
    }
    
    # 4. 全局映射注册表 (实验定义)
    # 将环境、包装器和模型组合成唯一的实验 ID
    EXPERIMENT_MAP = {
        # ID: (EnvID, WrapperID, ModelID)
        
        "exp_baseline_mlp": {
            "env_id": "fix_topp",
            "wrapper_id": "default_mlp",
            "model_id": "baseline"
        },
        
        "exp_threshold_deep": {
            "env_id": "cell_free",
            "wrapper_id": "threshold_mlp",
            "model_id": "deep_residual"
        },
        
        "exp_deepsets": {
            "env_id": "fix_topp",
            "wrapper_id": "raw_gnn", # DeepSets 通常接受原始或轻微处理的数据
            "model_id": "deepset_v1"
        },
        
        "exp_cartpole_td3": {
            "env_id": "cartpole",
            "wrapper_id": "none",
            "model_id": "baseline"
        },
        
        "exp_optimization_benchmark": {
            "env_id": "quadratic",
            "wrapper_id": "none",
            "model_id": "baseline"
        },
        
        "exp_optimization_nonconvex": {
            "env_id": "rastrigin",
            "wrapper_id": "none",
            "model_id": "baseline"
        },
        
        "exp_classic_pendulum": {
            "env_id": "pendulum",
            "wrapper_id": "none",
            "model_id": "baseline"
        }
    }

    
    @classmethod
    def get_experiment_config(cls, experiment_id: str) -> Dict:
        if experiment_id not in cls.EXPERIMENT_MAP:
             raise ValueError(f"Experiment ID '{experiment_id}' not found.")
        
        exp = cls.EXPERIMENT_MAP[experiment_id]
        
        # 解析模型架构 ID
        model_scenario_id = exp["model_id"]
        arch_id = cls.MODEL_ID_MAP.get(model_scenario_id, model_scenario_id) # 如果未映射则回退到自身
        
        return {
            "env_id": exp["env_id"],
            "wrapper_id": exp["wrapper_id"],
            "model_arch_id": arch_id
        }

    
    @classmethod
    def get_env_class(cls, env_id):
        if env_id not in cls.ENV_REGISTRY:
            raise ValueError(f"Unknown environment ID: {env_id}. Available: {list(cls.ENV_REGISTRY.keys())}")
        return cls.ENV_REGISTRY[env_id]
        
    @classmethod
    def get_wrappers(cls, wrapper_config_name):
        return cls.WRAPPER_REGISTRY.get(wrapper_config_name, [])
