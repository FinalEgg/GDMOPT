# Combined Model

这个目录包含了组合模型（Combined Model）的实现，该模型整合了离散决策（基于Actor网络）和功率分配（基于Diffusion模型）功能。

## 目录结构

- `combined_model.py`: 组合模型的主要实现
- `config.py`: 模型配置参数
- `__init__.py`: 模块导出文件

## 模型说明

CombinedModel类实现了以下功能：
1. 基于Actor网络的离散决策：预测无人机与地面基站的连接概率，并通过阈值处理转换为0/1连接矩阵
2. 基于Diffusion模型的功率分配：以状态空间和连接情况为输入，输出功率分配

## 配置参数

配置参数可以通过`CombinedConfig`类进行设置和管理，主要参数包括：

### Actor网络参数
- `actor_hidden_dim`: Actor网络隐藏层维度
- `actor_learning_rate`: Actor网络学习率
- `actor_weight_decay`: Actor网络权重衰减

### Diffusion模型参数
- `diffusion_hidden_dim`: Diffusion模型隐藏层维度
- `diffusion_timesteps`: 扩散步数
- `diffusion_learning_rate`: Diffusion模型学习率
- `diffusion_weight_decay`: Diffusion模型权重衰减
- `diffusion_beta_schedule`: Beta调度策略
- `max_power`: 最大功率限制

### 阈值处理参数
- `threshold`: 连接概率阈值
- `use_soft_threshold`: 是否使用软阈值
- `soft_threshold_temperature`: 软阈值温度参数

## 使用方法

### 导入模型

```python
from model.combined import CombinedModel, CombinedConfig
```

### 创建模型实例

```python
# 创建配置
config = CombinedConfig()
config.update(
    actor_hidden_dim=256,
    diffusion_hidden_dim=256,
    threshold=0.5
)

# 创建模型
model = CombinedModel(
    state_dim=10,
    connection_dim=5,
    power_dim=5,
    config=config
)
```

### 模型前向传播

```python
# 输入状态
state = torch.randn(batch_size, state_dim)

# 获取连接矩阵和功率分配
connection_matrix, power_allocation = model(state)

# 获取特定输出
connection_prob = model(state, output_mode='prob')
```

### 训练模型

使用`script/train_combined.py`脚本进行训练：

```bash
python script/train_combined.py --state_dim 10 --connection_dim 5 --power_dim 5 --epochs 100 --batch_size 64
```

训练后的模型和日志将保存在`log/combined/`目录下。