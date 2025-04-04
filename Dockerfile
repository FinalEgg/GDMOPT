# 尝试使用不同的 PyTorch 镜像版本
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件到容器中
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirement.txt

# 设置环境变量
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 默认命令运行TD3训练
CMD ["python", "train_TD3.py"]