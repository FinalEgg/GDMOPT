# GDMOPT: 网络优化中的深度强化学习与生成模型框架

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Tianshou](https://img.shields.io/badge/Framework-Tianshou-green)](https://github.com/thu-ml/tianshou)

**GDMOPT** 是一个模块化、高可扩展的科研实验框架，专为解决无线通信网络优化（如 Cell-Free 网络）及经典运筹优化问题而设计。该框架基于 **Tianshou** 深度强化学习库与 **PyTorch** 构建，集成了 TD3、DDPG、SAC 等主流 DRL 算法，并支持生成扩散模型（Diffusion Model）进行策略优化。

核心特性：
- **配置驱动（Config-Driven）**：一键切换算法、环境与网络结构。
- **模块化设计**：环境（Env）、策略（Policy）、网络（Network）完全解耦。
- **基准测试完备**：内置经典控制（Pendulum）、凸/非凸优化（Quadratic/Rastrigin）及 Cell-Free 通信场景。
- **完善的日志管理**：自动化的日志路径生成、配置序列化保存及断点续训支持。

---

## 🏗️ 目录结构说明

```
GDMOPT/
├── config/                  # [配置中心] 全局控制的核心
│   ├── env_config.py        # 环境物理参数（如基站数量、功率限制、优化边界）
│   ├── run_config.py        # 训练超参数（如 Epochs, LR, BatchSize, 算法选择）
│   ├── mapping_config.py    # 实验注册表（将 Env, Wrapper, Model 绑定为实验 ID）
│   └── model_config.py      # 模型细节配置（如层数、激活函数）
│
├── envs/                    # [环境层] 物理世界模拟
│   ├── core/                # 核心物理引擎（如 channel.py 信道建模）
│   ├── wrappers/            # Gymnasium 包装器（观测归一化、动作调整）
│   ├── base_env.py          # Cell-Free 环境基类
│   ├── cellfree_env.py      # 具体通信环境实现
│   ├── optimization_env.py  # 数值优化基准环境 (Quadratic, Rastrigin)
│   └── classic_env.py       # 经典控制环境包装 (CartPole, Pendulum)
│
├── networks/                # [网络层] 神经网络工厂
│   ├── blocks/              # 基础模块 (MLP, DeepSets, GAT)
│   ├── wrappers/            # 此处的 Wrapper 指网络输出头 (Deterministic/Stochastic Head)
│   └── factory.py           # ModelFactory，根据配置组装 Actor/Critic
│
├── policies/                # [策略层] RL 算法定制
│   ├── custom_ddpg.py       # 定制 DDPG (支持稀疏性约束等)
│   ├── custom_td3.py        # 定制 TD3
│   └── custom_sac.py        # 定制 SAC
│
├── scripts/                 # [执行层] 脚本与入口
│   ├── train_model.py       # 通用训练入口函数
│   ├── usr_script/          # 用户实验脚本 (推荐在此运行实验)
│   │   ├── run_optimization_benchmark.py  # 凸优化基准测试
│   │   ├── run_nonconvex_benchmark.py     # 非凸优化(Rastrigin)测试
│   │   └── run_pendulum_benchmark.py      # 倒立摆(多步决策)测试
│   └── utils/               # 工具库
│       ├── config_manager.py # 配置序列化与路径生成
│       └── setup.py          # 环境与策略的工厂组装逻辑
│
└── log/                     # [日志层] 自动生成的实验记录
    └── {EnvName}/           # 按环境分类
        └── {Algo_Backbone}/ # 按算法和网络分类
            └── {Time_Tag}/  # 具体运行实例 (含 Checkpoints, TensorBoard, config.json)
```

---

## 🚀 快速上手

### 1. 环境准备
确保已安装 Python 3.9+ 及 PyTorch。
```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 运行基准测试
框架内置了多种场景的自动化运行脚本。

**场景 A：非凸函数优化 (Rastrigin)**
测试模型寻找全局最优解的能力，考察探索性。
```bash
python scripts/usr_script/run_nonconvex_benchmark.py
```
*此脚本将自动测试 TD3, DDPG, SAC 三种算法在 Rastrigin 函数上的表现。*

**场景 B：经典多步控制 (Pendulum)**
测试模型在时序关联任务中的长期规划能力（Gamma = 0.99）。
```bash
python scripts/usr_script/run_pendulum_benchmark.py
```

**场景 C：简单凸优化 (Quadratic)**
验证算法收敛性的基础测试。
```bash
python scripts/usr_script/run_optimization_benchmark.py
```

### 3. 自定义训练
如果不使用 benchmark 脚本，可以直接 modify `config/run_config.py`，然后运行：
```bash
python scripts/train_model.py
```

---

## ⚙️ 配置系统详解

本框架的核心在于 `config/` 目录下的三个文件。

### 1. 定义物理世界 (`env_config.py`)
修改环境参数，例如优化问题的维度或通信网络的规模。
```python
# config/env_config.py
OPT_DIM = 2        # 优化变量维度
M = 10             # 基站数量
STEPS_PER_EPISODE = 200
```

### 2. 注册实验组合 (`mapping_config.py`)
将 `Environment` + `Wrapper` + `Model` 组合成一个 `EXPERIMENT_ID`。
```python
# config/mapping_config.py
"exp_optimization_nonconvex": {
    "env_id": "rastrigin",   # 指向 RastriginEnv
    "wrapper_id": "none",    # 不使用观测包装器
    "model_id": "baseline"   # 使用基础 MLP 网络
}
```

### 3. 控制训练流程 (`run_config.py`)
选择算法、设置训练时长和超参数。
```python
# config/run_config.py
EXPERIMENT_ID = 'exp_optimization_nonconvex' # 选择上面定义的实验
ALGO = 'td3'                                 # 选择算法
EPOCH = 50                                   # 训练轮数
```

---

## 📊 日志与结果

训练启动后，系统会在 `log/` 目录下自动生成结构化的文件夹：
```
log/rastrigin/td3_mlp/20260225_143000_benchmark_dim2/
├── config.json              # 记录本次运行的完整配置副本
├── events.out.tfevents...   # TensorBoard 训练曲线
├── checkpoint.pth           # 最新模型权重 (每 Epoch 保存)
└── policy_best.pth          # 最佳模型权重 (测试集得分最高时保存)
```

**查看训练曲线：**
```bash
tensorboard --logdir log
```

**断点续训：**
`train_model.py` 支持 `resume_from_log` 参数。若指定目录，系统将自动加载 `checkpoint.pth` 恢复训练状态。在 User Script 中，默认行为是创建新实验（`None`），如需续训请修改脚本入参。

---

## 🛠️ 高级功能

### 网络工厂 (Factory)
在 `networks/factory.py` 中，我们使用工厂模式动态构建网络。
支持以下骨干网络 (Backbone)：
- `mlp`: 标准多层感知机。
- `deepsets`: 具有置换不变性的集合网络，适合处理无序的基站/用户集合。
- `gnn`: 图神经网络 (待完善)。

### 算法扩展
在 `policies/` 目录下，我们继承 Tianshou 的策略类实现了 `CustomDDPG`, `CustomTD3` 等，允许注入自定义逻辑（如稀疏性正则化 Loss）。
