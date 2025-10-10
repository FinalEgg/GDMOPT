# GDMOPT：网络优化中的生成扩散模型

包含：

- 训练脚本：DDPG / SAC / 生成扩散优化器（Diffusion）
- Cell-free UAV 网络仿真可视化界面（Tkinter + Matplotlib）
- 日志与权重管理

主页/介绍页：见根目录 [index.html](index.html)

---

## 1. 安装与环境（Windows / PowerShell）

推荐使用 Python 3.8–3.10，并创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 基础依赖
pip install torch
pip install tianshou==0.4.11
pip install gymnasium matplotlib scipy

# 如需 TensorBoard（可选）
pip install tensorboard
```

注意：

- 项目已迁移至 Gymnasium，请避免安装旧版 gym；若已安装可卸载：`pip uninstall -y gym`。
- Tkinter 多数环境自带；若缺失请根据系统安装。

---

## 2. 训练（含 cell-free 环境）

三个训练脚本均支持 `--env` 选择：`pendulum`、`optimization`、`cellfree`。为了与仿真界面联动，请使用 `cellfree` 环境训练，权重将保存在：

```
log/<log_prefix>/<algorithm>/cellfree/<timestamp>/policy.pth
```

示例（在仓库根目录运行）：

```powershell
# DDPG（Cell-free）
python script\train_ddpg.py --env cellfree --epoch 10 --training-num 1 --test-num 1

# SAC（Cell-free）
python script\train_sac.py --env cellfree --epoch 10 --training-num 1 --test-num 1

# Diffusion 优化器（Cell-free）
python script\train_diffusion.py --env cellfree --epoch 10 --training-num 1 --test-num 1
```

说明：

- 三个脚本默认 `--log-prefix default`，因此日志根路径为 `log/default/...`。
- Diffusion 的脚本算法名为 `diffusion_opt`，其日志目录为 `log/default/diffusion_opt/...`。

---

## 3. 仿真可视化（Simulation GUI）

运行：

```powershell
python script\simulation\simulation.py
```

界面功能：

- 模型下拉：`ddpg` / `sac` / `diffusion`
- 权重下拉：在选择模型后自动扫描 `log/default/<model>/cellfree/**/policy.pth`
- 开始/暂停按钮：启动/暂停仿真

注意事项：

- 为避免参数不匹配，仿真只读取 cell-free 环境下训练得到的权重。
- 目前 Diffusion 的训练目录名为 `diffusion_opt`，如在仿真中选择 `diffusion` 请确保对应目录存在或调整代码以匹配目录名。
- 界面关闭时程序会自动退出。

可视化细节：

- 每架无人机自动分配不同颜色；起飞同时以相同颜色虚线标注其规划路线。
- 基站以蓝色三角标记，充电站以黄色方块标记，起点为绿色，终点为红色。
- 若加载了模型，将绘制无人机与基站的连接强度（阈值显示）。

安全加载：

- 仿真中的模型加载使用 `torch.load(..., weights_only=True)`，建议使用较新的 PyTorch 版本。

---

## 4. 日志与可视化

日志/权重保存路径示例：

- DDPG（cellfree）：`log/default/ddpg/cellfree/<timestamp>/policy.pth`
- SAC（cellfree）：`log/default/sac/cellfree/<timestamp>/policy.pth`
- Diffusion（cellfree）：`log/default/diffusion_opt/cellfree/<timestamp>/...`

如需 TensorBoard：

```powershell
tensorboard --logdir log
```

---

## 5. 代码结构（摘录）

- 训练脚本：
  - `script/train_ddpg.py`
  - `script/train_sac.py`
  - `script/train_diffusion.py`
- 仿真：
  - `script/simulation/simulation.py`（GUI 主程序）
  - `script/simulation/model_prediction.py`（模型加载与推理）
  - `script/simulation/drone_path.py`（路径规划示例）
  - `script/simulation/sim_config.py`（仿真参数）
- 环境：
  - `env/cellfree/env.py`（Gymnasium 环境实现，`make_cellfree_env`）
  - 其他环境：`env/pendulum`、`env/optimization`、`env/aigc`
- 策略与模型：
  - 策略：`policy/ddpg`、`policy/sac`、`policy/diffusion_opt`
  - 模型：`model/actor.py`、`model/sac/`、`model/diffusion/`

---

## 6. 常见问题（FAQ）

1) Gym 警告/不兼容：

   - 使用 Gymnasium：`pip install gymnasium`；卸载旧 gym：`pip uninstall -y gym`。
2) 权重下拉为空或加载失败：

   - 确认已使用 `--env cellfree` 训练，并产生 `policy.pth`。
   - 确认目录为：`log/default/<model>/cellfree/**/policy.pth`。
3) 关闭窗口程序未退出：

   - 已在仿真中绑定关闭事件并退出主循环，可直接关闭窗口结束程序。
4) NumPy 类型错误（dtype cast）：

   - 仿真中已统一为浮点坐标；如修改代码，请确保位置与增量运算使用浮点类型。
5) PyTorch 安全加载提示：

   - 我们使用 `weights_only=True`，建议使用新版本 PyTorch；如遇兼容问题可降级为常规 `torch.load(path, map_location=...)`。

---

## 7. 引用

如本项目或教程对您的研究有帮助，请引用：

```bibtex
@article{du2023beyond,
  title={Beyond deep reinforcement learning: A tutorial on generative diffusion models in network optimization},
  author={Du, Hongyang and Zhang, Ruichen and Liu, Yinqiu and Wang, Jiacheng and Lin, Yijing and Li, Zonghang and Niyato, Dusit and Kang, Jiawen and Xiong, Zehui and Cui, Shuguang and Ai, Bo and Zhou, Haibo and Kim, Dong In},
  journal={arXiv preprint arXiv:2308.05384},
  year={2023}
}
```

---

## 8. 许可证

- 网页模板与页面版权：见 [index.html](index.html) 页脚（CC BY-SA 4.0 元素）。
- 代码许可：后续补充。

---
