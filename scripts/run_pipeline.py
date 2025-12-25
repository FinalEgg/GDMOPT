import os
import subprocess
import sys
import time
import argparse

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
    parser = argparse.ArgumentParser(description="Run the full training pipeline.")
    parser.add_argument('--force-random', action='store_true', help='Force regenerate random data (Critic Warmup)')
    parser.add_argument('--force-demo', action='store_true', help='Force regenerate demonstration data (Actor Pretrain)')
    parser.add_argument('--load-pretrained', type=str, default=None, help='Path to pretrained policy to resume training from (skips warmup/pretrain)')
    args = parser.parse_args()

    # =========================================================================
    # 全局参数配置 (Global Configuration)
    # 您可以在这里统一控制整个流程的参数
    # =========================================================================
    
    # 引入 TrainConfig 以保持一致性
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.train_config import TrainConfig

    # 1. 随机数据采集 (Critic Warmup)
    # 作用: 填充 Replay Buffer，预热 Critic
    RANDOM_COLLECT_STEPS = TrainConfig.WARMUP_STEPS
    
    # 2. 演示数据采集 (Actor Pretrain)
    # 作用: 生成高质量 (State, Action) 对，用于 Actor 行为克隆
    # 注意: 10万条数据采集可能需要几十分钟
    DEMO_COLLECT_EPISODES = TrainConfig.PRETRAIN_EPISODES
    
    # 3. 正式训练 (RL Training)
    # 作用: 启动 SAC/TD3/DDPG 训练
    # 具体的 Epoch 和 Step 参数在 config/train_config.py 中定义 (默认 2000 Epochs)
    ENV_NAME = TrainConfig.ENV
    ALGO_NAME = TrainConfig.ALGO
    BACKBONE_NAME = TrainConfig.BACKBONE
    
    # =========================================================================
    
    # 获取项目根目录 (假设此脚本位于 scripts/ 文件夹下)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # D:\Code\GDMOPT
    
    print(f"Project Root detected as: {project_root}")
    print(f"Configuration:")
    print(f"  - Random Steps: {RANDOM_COLLECT_STEPS}")
    print(f"  - Demo Episodes: {DEMO_COLLECT_EPISODES}")
    print(f"  - Algorithm: {ALGO_NAME} ({BACKBONE_NAME}) on {ENV_NAME}")
    
    # Define data paths
    random_data_path = os.path.join(project_root, 'log', 'random_data.npz')
    demo_data_path = os.path.join(project_root, 'log', 'demonstration_data.npz')

    # 1. 收集随机数据 (Critic Warmup)
    if args.load_pretrained:
        print(f"\n[SKIP] Skipping Random Data Collection (Resuming from pretrained: {args.load_pretrained})")
    elif os.path.exists(random_data_path) and not args.force_random:
        print(f"\n[SKIP] Random data found at {random_data_path}. Skipping generation.")
    else:
        run_command(
            f"python scripts/collect_random_data.py --steps {RANDOM_COLLECT_STEPS}", 
            cwd=project_root,
            description="1. Collect Random Data (Critic Warmup)"
        )
    
    # 2. 收集演示数据 (Actor Pretrain)
    if args.load_pretrained:
        print(f"\n[SKIP] Skipping Demonstration Data Collection (Resuming from pretrained: {args.load_pretrained})")
    elif os.path.exists(demo_data_path) and not args.force_demo:
        print(f"\n[SKIP] Demonstration data found at {demo_data_path}. Skipping generation.")
    else:
        run_command(
            f"python scripts/heuristic_search.py --num-episodes {DEMO_COLLECT_EPISODES}", 
            cwd=project_root,
            description="2. Collect Demonstration Data (GA Search)"
        )
    
    # 3. 正式训练 (SAC + DeepSets + FixTopP)
    # 注意：我们需要显式开启 --do-warmup, --do-pretrain, --do-rl
    # 因为数据已经准备好了（无论是刚收集的还是跳过的），train.py 需要执行加载和训练逻辑。
    
    train_cmd = f"python scripts/train.py --env {ENV_NAME} --algo {ALGO_NAME} --backbone {BACKBONE_NAME}"
    
    if args.load_pretrained:
        # 如果加载预训练模型，只执行 RL 阶段，并传入路径
        train_cmd += f" --do-rl --load-pretrained {args.load_pretrained}"
    else:
        # 否则执行完整流程
        train_cmd += " --do-warmup --do-pretrain --do-rl"
        
    run_command(
        train_cmd, 
        cwd=project_root,
        description=f"3. Train {ALGO_NAME.upper()} Agent"
    )
    
    print(f"\n{'='*60}")
    print("All steps completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
