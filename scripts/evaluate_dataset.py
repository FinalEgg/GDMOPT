import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def evaluate_dataset(dataset_path):
    """
    评估数据集质量，统计奖励值的分布情况。
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    
    if 'rew' not in data:
        print("Error: Dataset does not contain 'rew' (reward) key. Please re-collect data.")
        return
        
    rewards = data['rew']
    num_samples = len(rewards)
    
    # Check for baseline rewards
    has_baseline = 'baseline_rew' in data
    if has_baseline:
        baseline_rewards = data['baseline_rew']
    
    print(f"\n=== Dataset Statistics (GA Optimized) ===")
    print(f"Number of Samples: {num_samples}")
    print(f"Mean Reward:       {np.mean(rewards):.4f}")
    print(f"Std Dev:           {np.std(rewards):.4f}")
    print(f"Min Reward:        {np.min(rewards):.4f}")
    print(f"Max Reward:        {np.max(rewards):.4f}")
    print(f"Median Reward:     {np.median(rewards):.4f}")
    
    if has_baseline:
        print(f"\n=== Baseline Statistics (Top-P + Equal Power) ===")
        print(f"Mean Reward:       {np.mean(baseline_rewards):.4f}")
        print(f"Std Dev:           {np.std(baseline_rewards):.4f}")
        print(f"Min Reward:        {np.min(baseline_rewards):.4f}")
        print(f"Max Reward:        {np.max(baseline_rewards):.4f}")
        
        # Comparison
        improvement = rewards - baseline_rewards
        better_count = np.sum(improvement > 0)
        better_ratio = better_count / num_samples * 100
        mean_improvement = np.mean(improvement)
        
        print(f"\n=== Comparison (GA vs Baseline) ===")
        print(f"GA Better Than Baseline: {better_count}/{num_samples} ({better_ratio:.2f}%)")
        print(f"Mean Improvement:        {mean_improvement:.4f}")
        print(f"Max Improvement:         {np.max(improvement):.4f}")
    
    # Percentiles
    print(f"\n=== Percentiles (GA Rewards) ===")
    print(f"25th Percentile:   {np.percentile(rewards, 25):.4f}")
    print(f"50th Percentile:   {np.percentile(rewards, 50):.4f}")
    print(f"75th Percentile:   {np.percentile(rewards, 75):.4f}")
    print(f"90th Percentile:   {np.percentile(rewards, 90):.4f}")
    print(f"95th Percentile:   {np.percentile(rewards, 95):.4f}")
    print(f"99th Percentile:   {np.percentile(rewards, 99):.4f}")
    
    # Histogram
    # We can't show plot in terminal, but we can print a simple text histogram
    print(f"\n=== Distribution (Text Histogram) ===")
    counts, bins = np.histogram(rewards, bins=10)
    max_count = np.max(counts)
    scale = 40.0 / max_count # Scale to 40 chars width
    
    for i in range(len(counts)):
        bar_len = int(counts[i] * scale)
        bar = '#' * bar_len
        range_str = f"{bins[i]:6.2f} - {bins[i+1]:6.2f}"
        print(f"{range_str} | {bar} ({counts[i]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate collected dataset quality.")
    parser.add_argument('--path', type=str, default='log/demonstration_data.npz', help='Path to the dataset file')
    args = parser.parse_args()
    
    evaluate_dataset(args.path)
