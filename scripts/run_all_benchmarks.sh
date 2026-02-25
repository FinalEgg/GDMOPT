#!/bin/bash
set -e  # 如果任何一个脚本报错，立即停止

echo "=================================================="
echo "Starting GDMOPT Full Benchmark Suite"
echo "=================================================="

echo "[1/4] Running Optimization Benchmark (Convex)..."
python scripts/usr_script/run_optimization_benchmark.py

echo "[2/4] Running Non-Convex Benchmark (Rastrigin)..."
python scripts/usr_script/run_nonconvex_benchmark.py

echo "[3/4] Running Pendulum Benchmark (Classic Control)..."
python scripts/usr_script/run_pendulum_benchmark.py

echo "[4/4] Running Diffusion Benchmark..."
python scripts/usr_script/run_diffusion_benchmark.py

echo "=================================================="
echo "All Benchmarks Completed Successfully!"
echo "=================================================="
