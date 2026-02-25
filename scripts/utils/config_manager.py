
import json
import os
import datetime
from typing import Dict, Any, Type

class ConfigManager:
    """
    配置管理工具 (Configuration Management Utilities).
    
    Responsibilities:
    1. Serialize Config Classes (RunConfig, EnvConfig) to JSON/Dict.
       (将配置类序列化为 JSON/字典)
    2. Save and Load configurations to/from disk.
       (保存和加载配置到磁盘)
    3. Generate standardized log paths based on config.
       (根据配置生成标准化的日志路径)
    """

    @staticmethod
    def config_to_dict(config_cls: Type) -> Dict[str, Any]:
        """
        Convert a configuration class to a dictionary.
        Ignores dunder methods and private attributes.
        """
        config_dict = {}
        for key, value in config_cls.__dict__.items():
            # Filter criteria:
            # 1. Not starting with underscore (private/dunder)
            # 2. Not callable (methods, classmethods)
            # 3. Not a classmethod object (checking isinstance(value, classmethod) might fail if it's already bound)
            # Safe way: check if it's a basic type or ignore functions
            
            if key.startswith('__'):
                continue
                
            if callable(value) or isinstance(value, (classmethod, staticmethod)):
                continue
                
            config_dict[key] = value
        return config_dict

    @staticmethod
    def save_config(log_dir: str, run_config_cls: Type, env_config_cls: Type) -> None:
        """
        Save current configuration state to a JSON file in the log directory.
        """
        os.makedirs(log_dir, exist_ok=True)
        
        full_config = {
            "RunConfig": ConfigManager.config_to_dict(run_config_cls),
            "EnvConfig": ConfigManager.config_to_dict(env_config_cls),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=4, ensure_ascii=False)
        print(f"Configuration saved to {config_path}")

    @staticmethod
    def load_config(log_dir: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        """
        config_path = os.path.join(log_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json found at {log_dir}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def generate_log_path(
        base_log_dir: str, 
        env_name: str, 
        algo_name: str, 
        backbone_name: str, 
        run_tag: str = None
    ) -> str:
        """
        Generate standardized log path structure:
        log_dir / env_name / algo_backbone / run_tag_timestamp
        
        Args:
            base_log_dir: Root log directory (e.g., 'log')
            env_name: Name of the environment (e.g., 'cellfree', 'quadratic')
            algo_name: Algorithm name (e.g., 'td3', 'sac')
            backbone_name: Backbone name (e.g., 'mlp', 'deepsets')
            run_tag: Optional custom tag for the run (e.g., 'test_lr3e4')
        """
        # Level 1: Environment
        l1 = env_name
        
        # Level 2: Algorithm + Backbone
        l2 = f"{algo_name}_{backbone_name}"
        
        # Level 3: Run Instance (Time + Tag)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_tag:
            l3 = f"{timestamp}_{run_tag}"
        else:
            l3 = f"{timestamp}"
            
        full_path = os.path.join(base_log_dir, l1, l2, l3)
        return full_path
