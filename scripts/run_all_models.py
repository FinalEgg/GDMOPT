import subprocess
import sys
import os
import argparse

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
        print("Success!")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run training for multiple algorithms sequentially.")
    parser.add_argument('--algos', nargs='+', default=['sac'], 
                        choices=['ddpg', 'td3', 'sac', 'diffusion'],
                        help="List of algorithms to run (space separated). Default: all")
    parser.add_argument('--env', type=str, default='fix_topp', help="Environment name")
    parser.add_argument('--backbone', type=str, default='deepsets', help="Backbone network")
    
    args = parser.parse_args()
    
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print(f"Starting sequential run for algorithms: {args.algos}")
    print(f"Environment: {args.env}, Backbone: {args.backbone}")
    
    for algo in args.algos:
        print(f"\n{'='*50}")
        print(f"Running Algorithm: {algo.upper()}")
        print(f"{'='*50}")
        
        cmd = f"python scripts/train.py --env {args.env} --algo {algo} --backbone {args.backbone}"
        run_command(cmd)
        
    print("\nAll selected algorithms finished successfully!")

if __name__ == "__main__":
    main()
