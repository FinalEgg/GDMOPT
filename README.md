# GDMOPT：网络优化中的生成扩散模型教程（中文 README）

本项目是“Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization”的教学代码与演示页面，聚焦于将生成扩散模型（GDM）应用于网络优化任务，并与深度强化学习（如 DDPG）进行对比与结合。项目包含命令行脚本、可视化 GUI、网页教程与训练日志/模型。

- 项目主页与教学页面：详见 [index.html](index.html)
- 代表性脚本：DDPG 训练脚本 [script/train_ddpg.py](script/train_ddpg.py)，GDM 训练脚本 [script/train_diffusion.py](script/train_diffusion.py)
- 参数 GUI：见 [Software/parameter_gui.py](Software/parameter_gui.py) 中的类 [`Software.parameter_gui.GUI`](Software/parameter_gui.py)
- 实用工具：[`model.diffusion.utils.Progress`](model/diffusion/utils.py)、[`model.diffusion.utils.EarlyStopping`](model/diffusion/utils.py)，以及对应教学版本 [`diffusion.utils.Progress`](diffusion/utils.py)、[`diffusion.utils.EarlyStopping`](diffusion/utils.py)
- 网页样式与资源：如 [static/css/index.css](static/css/index.css)

在网络优化场景中，我们通常以约束优化形式描述问题：

- 目标：最大化/最小化目标函数 $f(x)$
- 约束：$g_i(x) \le 0, \; h_j(x) = 0$

示例形式：$\max_x f(x)\ \text{s.t.}\ g(x)\le 0$。GDM 通过前向扩散与反向去噪学习可行决策分布，DDPG 则通过策略梯度在环境交互中学习。

---

## 环境准备

建议使用 Conda 创建独立环境（Python 3.8）：

```sh

conda create --name gdmopt python==3.8

conda activate gdmopt

pip install tianshou==0.4.11

pip install matplotlib==3.7.3

pip install scipy==1.10.1

```

如需可视化 GUI，请保证已安装 tkinter（多数系统自带；若缺失请根据操作系统安装）。

---

## 快速开始

1) 运行网页教程（本地打开）

- 直接用浏览器打开 [index.html](index.html)，内含项目介绍、运行指引与图示。

2) 命令行训练

- 训练 DDPG：

```sh

pythonscript/train_ddpg.py

```

- 训练生成扩散模型（GDM）优化器：

```sh

pythonscript/train_diffusion.py

```

3) GUI 启动

```sh

pythonSoftware/parameter_gui.py

```

- GUI 参数面板定义见 [Software/parameter_gui.py](Software/parameter_gui.py) 的 [`Software.parameter_gui.GUI`](Software/parameter_gui.py)。
- 常用参数键（在 GUI 中）：log_prefix、render、rew_norm、resume_path、watch、prioritized_replay、lr_decay、note（参见源码控件初始化）。

---

## 日志与模型

- 训练日志与模型权重默认保存于 log/ 路径下（可通过脚本/GUI 参数控制前缀等）。
- 示例：

  - DDPG 权重文件： [log/default/ddpg/optimization/Oct05-000901/policy.pth](log/default/ddpg/optimization/Oct05-000901/policy.pth)
  - GDM 事件日志（TensorBoard 事件文件示例）：[log/default/diffusion_opt/optimization/Oct05-000933/events.out.tfevents.1759594173.LAPTOP-6C6RISE9.19764.0](log/default/diffusion_opt/optimization/Oct05-000933/events.out.tfevents.1759594173.LAPTOP-6C6RISE9.19764.0)

说明：.pth 为二进制模型快照。根据任务不同，可能包含策略网络、价值网络或扩散模型参数。

---

## 代码结构（摘录）

- 核心训练脚本

  - DDPG：[script/train_ddpg.py](script/train_ddpg.py)
  - GDM：[script/train_diffusion.py](script/train_diffusion.py)
- 工具与通用组件

  - 扩散训练进度与早停（教学版）：[`diffusion.utils`](diffusion/utils.py) 中的 [`diffusion.utils.Progress`](diffusion/utils.py)、[`diffusion.utils.EarlyStopping`](diffusion/utils.py)
  - 扩散训练进度与早停（模型版）：[`model.diffusion.utils`](model/diffusion/utils.py) 中的 [`model.diffusion.utils.Progress`](model/diffusion/utils.py)、[`model.diffusion.utils.EarlyStopping`](model/diffusion/utils.py)
- 策略模块入口

  - [policy/__init__.py](policy/__init__.py)
  - 具体算法与变体位于 policy 子目录（如 ddpg、diffusion_opt）
- 可视化与网页

  - 教学页面与资源： [index.html](index.html), [static/css/index.css](static/css/index.css)

---

## 常见问题

- 安装冲突/依赖版本不匹配

  - 固定 Python 3.8 与给定依赖版本；如需升级，请逐项验证兼容性。
- 无法打开 GUI

  - 请检查 tkinter 是否可用；Linux 可能需额外安装（如 `sudo apt-get install python3-tk`）。
- 模型文件过大/不可读

  - .pth 为二进制文件，仅能由相应框架加载；请通过训练脚本或评估脚本进行读取。

---

## 实验说明与对比

- 本项目提供 DDPG 与 GDM 两种优化范式对比。对于简化问题，DRL 未必劣于 GDM；对于更复杂/含状态转移的优化问题，考虑将 GDM 与 DRL 结合具备潜力（可作为扩展方向见下文“扩展空间”）。

---

## 扩展空间（Roadmap 占位）

- 新环境接入与定制

  - 接入更复杂的网络优化任务（带时序状态、随机性、约束投影等）。
- GDM 与 DRL 融合

  - 例如用 GDM 生成高质量候选解，再由 DRL 策略筛选/微调，或将 GDM 作为策略先验。
- 约束处理机制

  - 探索软约束正则、可行性投影、拉格朗日乘子等方法，刻画 $g_i(x)\le0$ 的满足性。
- 评估指标与可视化

  - 增加更丰富的指标（可行率、稳定性、收敛速度等）与可视化面板。
- 多任务/迁移学习

  - 针对不同网络条件的快速自适应与跨场景泛化。

（以上为占位纲要，后续将补充实现与文档）

---

## 引用

如本项目或教程对您的研究有帮助，请引用以下论文：

```bibtex

@article{du2023beyond,

  title={Beyond deep reinforcement learning: A tutorial on generative diffusion models in network optimization},

  author={Du, Hongyang and Zhang, Ruichen and Liu, Yinqiu and Wang, Jiacheng and Lin, Yijing and Li, Zonghang and Niyato, Dusit and Kang, Jiawen and Xiong, Zehui and Cui, Shuguang and Ai, Bo and Zhou, Haibo and Kim, Dong In},

  journal={arXiv preprint arXiv:2308.05384},

  year={2023}

}

```

---

## 许可证

- 网页模板与页面版权：参见页面底部说明，使用 Creative Commons Attribution-ShareAlike 4.0（CC BY-SA 4.0）协议元素（详见 [index.html](index.html) 页脚）。
- 代码协议：待补充（占位）。

---
