# 使用轻量级 Python 镜像，解决大文件下载困难的问题
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 配置 pip 使用清华源加速下载 (解决网络问题关键步骤)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装系统依赖 (Gym/OpenCV 等可能需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements_docker.txt .

# 安装 Python 依赖
# 注意：PyTorch 单独指定官方源下载太慢，这里利用 pip 的 extra-index-url 机制
# 如果这一步还是慢，可以考虑使用 cpu 版本或者提前下载好 wheel
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements_docker.txt

# 复制项目代码
COPY . .

# 如果是 Linux/Mac 下构建，可能需要给脚本赋予执行权限
RUN chmod +x scripts/run_all_benchmarks.sh

# 创建日志目录
RUN mkdir -p log

# 默认启动命令：直接运行全套基准测试
CMD ["bash", "scripts/run_all_benchmarks.sh"]
