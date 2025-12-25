import os
import subprocess
import sys
import time

def run_command(command, cwd, description):
    print(f"\n{'='*60}")
    print(f"Starting Step: {description}")
    print(f"Command: {command}")
    print(f"Working Directory: {cwd}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        # 使用 shell=True 以确保能正确解析 python 命令
        subprocess.check_call(command, cwd=cwd, shell=True)
        duration = time.time() - start_time
        print(f"\n[SUCCESS] {description} completed in {duration:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

def main():
    # 获取项目根目录 (假设此脚本位于 scripts/ 文件夹下)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # D:\Code\GDMOPT
    
    print(f"Project Root detected as: {project_root}")
    
    # 1. 收集随机数据 (Critic Warmup)
    # 默认步数由 config 决定，也可以通过 --steps 覆盖
    run_command(
        "python scripts/collect_random_data.py", 
        cwd=project_root,
        description="1. Collect Random Data (Critic Warmup)"
    )
    
    # 2. 收集演示数据 (Actor Pretrain)
    # 默认 100000 episodes 可能太久，这里为了演示流程设置为 1000 (您可以根据需要修改)
    # 如果您想跑全量，请去掉 --num-episodes 参数或设为 100000
    run_command(
        "python scripts/heuristic_search.py --num-episodes 1000", 
        cwd=project_root,
        description="2. Collect Demonstration Data (GA Search)"
    )
    
    # 3. 正式训练 (SAC + DeepSets + FixTopP)
    run_command(
        "python scripts/train.py --env fix_topp --algo sac --backbone deepsets", 
        cwd=project_root,
        description="3. Train SAC Agent"
    )
    
    print(f"\n{'='*60}")
    print("All steps completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
