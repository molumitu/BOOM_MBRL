#!/usr/bin/env python3
"""
Test script to compare MPPI performance with different PI/Flow configurations.

This script runs two comprehensive ablation studies:

1. **PI/Flow Ablation Study** (4 configurations):
   - Normal (PI+Flow): num_pi_trajs=24, num_flow_trajs=48
   - No PI (Flow Only): num_pi_trajs=0, num_flow_trajs=48
   - No Flow (PI Only): num_pi_trajs=24, num_flow_trajs=0
   - Pure MPPI (No PI/Flow): num_pi_trajs=0, num_flow_trajs=0

2. **Flow Trajectories Ablation Study** (9 configurations):
   - Tests num_flow_trajs from 0 to 512 (with num_pi_trajs=0)
   - Values: 0, 64, 128, 192, 256, 320, 384, 448, 512
   - Generates performance curves showing impact of flow trajectory count

All results are saved to mppi_test_results/{task_name}/ with:
- CSV data files
- Comparison plots
- Performance curves
- Episode trend charts
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Set environment variables for rendering
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"  # Use EGL for offscreen rendering
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]

os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from omegaconf import OmegaConf
from boom.envs import make_env
from boom.boom_alg import BOOM


def run_flow_pi_ablation(model_path, config_path, task_name, num_episodes, results_dir):
    """
    Run PI/Flow ablation study with 4 configurations:
    1. Normal (PI+Flow)
    2. No PI (Flow Only)
    3. No Flow (PI Only)
    4. Pure MPPI (No PI/Flow)

    Args:
        model_path: Path to model checkpoint
        config_path: Path to config file
        task_name: Name of the task
        num_episodes: Number of episodes per run
        results_dir: Directory to save results

    Returns:
        dict: Results from all 4 configurations
    """
    print("\n" + "="*60)
    print("PI/Flow Ablation Study (4 Configurations)")
    print("="*60)

    results = {}

    # Run 1: Normal configuration
    print("\n")
    normal_stats = evaluate_with_config(
        model_path=model_path,
        config_path=config_path,
        task_name=task_name,
        num_episodes=num_episodes,
        run_label="Normal (PI+Flow)"
    )
    results['normal'] = normal_stats

    # Run 2: No PI (Flow Only)
    print("\n")
    no_pi_stats = evaluate_with_config(
        model_path=model_path,
        config_path=config_path,
        task_name=task_name,
        num_episodes=num_episodes,
        num_pi_trajs=0,
        run_label="No PI (Flow Only)"
    )
    results['no_pi'] = no_pi_stats

    # Run 3: No Flow (PI Only)
    print("\n")
    no_flow_only_stats = evaluate_with_config(
        model_path=model_path,
        config_path=config_path,
        task_name=task_name,
        num_episodes=num_episodes,
        num_flow_trajs=0,
        run_label="No Flow (PI Only)"
    )
    results['no_flow_only'] = no_flow_only_stats

    # Run 4: Pure MPPI (No PI/Flow)
    print("\n")
    pure_mppi_stats = evaluate_with_config(
        model_path=model_path,
        config_path=config_path,
        task_name=task_name,
        num_episodes=num_episodes,
        num_pi_trajs=0,
        num_flow_trajs=0,
        run_label="Pure MPPI (No PI/Flow)"
    )
    results['pure_mppi'] = pure_mppi_stats

    # Display results summary
    print("\n" + "="*60)
    print("PI/Flow Ablation Summary")
    print("="*60)
    print(f"Normal (PI+Flow):")
    print(f"  Mean reward: {normal_stats['mean_reward']:.2f} +/- {normal_stats['std_reward']:.2f}")
    print(f"  Mean length:  {normal_stats['mean_length']:.1f} +/- {normal_stats['std_length']:.1f}")
    print(f"\nNo PI (Flow Only):")
    print(f"  Mean reward: {no_pi_stats['mean_reward']:.2f} +/- {no_pi_stats['std_reward']:.2f}")
    print(f"  Mean length:  {no_pi_stats['mean_length']:.1f} +/- {no_pi_stats['std_length']:.1f}")
    print(f"\nNo Flow (PI Only):")
    print(f"  Mean reward: {no_flow_only_stats['mean_reward']:.2f} +/- {no_flow_only_stats['std_reward']:.2f}")
    print(f"  Mean length:  {no_flow_only_stats['mean_length']:.1f} +/- {no_flow_only_stats['std_length']:.1f}")
    print(f"\nPure MPPI (No PI/Flow):")
    print(f"  Mean reward: {pure_mppi_stats['mean_reward']:.2f} +/- {pure_mppi_stats['std_reward']:.2f}")
    print(f"  Mean length:  {pure_mppi_stats['mean_length']:.1f} +/- {pure_mppi_stats['std_length']:.1f}")

    # Compute differences
    no_pi_reward_diff = no_pi_stats['mean_reward'] - normal_stats['mean_reward']
    no_pi_length_diff = no_pi_stats['mean_length'] - normal_stats['mean_length']
    no_flow_only_reward_diff = no_flow_only_stats['mean_reward'] - normal_stats['mean_reward']
    no_flow_only_length_diff = no_flow_only_stats['mean_length'] - normal_stats['mean_length']
    pure_mppi_reward_diff = pure_mppi_stats['mean_reward'] - normal_stats['mean_reward']
    pure_mppi_length_diff = pure_mppi_stats['mean_length'] - normal_stats['mean_length']

    print(f"\n--- Performance Differences (vs Normal) ---")
    print(f"No PI:          Reward: {no_pi_reward_diff:+.2f}, Length: {no_pi_length_diff:+.1f}")
    print(f"No Flow:       Reward: {no_flow_only_reward_diff:+.2f}, Length: {no_flow_only_length_diff:+.1f}")
    print(f"Pure MPPI:     Reward: {pure_mppi_reward_diff:+.2f}, Length: {pure_mppi_length_diff:+.1f}")
    print("="*60)

    # Save results and create plots
    print("\n" + "="*60)
    print("Saving PI/Flow Ablation Results")
    print("="*60)

    save_results_to_csv(results, f"{task_name}_pi_flow_ablation", results_dir)
    plot_comparison(results, f"{task_name}_pi_flow", results_dir)
    plot_episode_trends(results, f"{task_name}_pi_flow", results_dir)

    print(f"\n✓ PI/Flow ablation results saved to: {results_dir}")

    return results


def run_flow_trajs_ablation(model_path, config_path, task_name, num_episodes, results_dir, num_points=9):
    """
    Run flow trajectories ablation study.
    Tests num_flow_trajs from 0 to 512 with num_pi_trajs=0.

    Args:
        model_path: Path to model checkpoint
        config_path: Path to config file
        task_name: Name of the task
        num_episodes: Number of episodes per run
        results_dir: Directory to save results
        num_points: Number of test points (default: 9)

    Returns:
        dict: Results with num_flow_trajs as keys
    """
    print("\n" + "="*60)
    print(f"Flow Trajectories Ablation Study ({num_points} Configurations)")
    print("="*60)

    # Generate test values from 0 to 512
    flow_trajs_values = np.linspace(0, 512, num_points, dtype=int)

    results = {}

    for i, num_flow_trajs in enumerate(flow_trajs_values):
        print(f"\nProgress: {i+1}/{num_points}")

        stats = evaluate_with_config(
            model_path=model_path,
            config_path=config_path,
            task_name=task_name,
            num_episodes=num_episodes,
            num_pi_trajs=0,  # Always disable PI
            num_flow_trajs=int(num_flow_trajs),
            run_label=f"Flow Trajs = {num_flow_trajs}"
        )
        results[num_flow_trajs] = stats

    # Display results summary
    print("\n" + "="*60)
    print("Flow Trajectories Ablation Summary")
    print("="*60)
    print("Key findings from flow trajectories sweep:")

    # Find best configuration
    best_flow_trajs = max(results.keys(),
                         key=lambda k: results[k]['mean_reward'])
    best_reward = results[best_flow_trajs]['mean_reward']

    # Compare with key points
    reward_0 = results[0]['mean_reward']
    reward_48 = results[48]['mean_reward'] if 48 in results else None
    reward_512 = results[512]['mean_reward']

    print(f"Best performance: {best_reward:.2f} at num_flow_trajs={best_flow_trajs}")
    print(f"Pure MPPI (0):   {reward_0:.2f}")
    if reward_48:
        print(f"Default Flow (48): {reward_48:.2f}")
    print(f"All Flow (512):  {reward_512:.2f}")

    # Show trend
    print(f"\n--- Performance Trend ---")
    for num_flow_trajs in sorted(results.keys()):
        stats = results[num_flow_trajs]
        print(f"num_flow_trajs={num_flow_trajs:3d}: Reward={stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")

    print("="*60)

    # Save results and create plots
    print("\n" + "="*60)
    print("Saving Flow Trajectories Ablation Results")
    print("="*60)

    save_flow_trajs_csv(results, task_name, results_dir)
    plot_flow_trajs_curve(results, task_name, results_dir)

    print(f"\n✓ Flow trajectories ablation results saved to: {results_dir}")

    return results


def plot_flow_trajs_curve(results, task_name, save_dir):
    """
    Plot flow trajectories ablation curve.

    Args:
        results: Dictionary with num_flow_trajs as keys and stats as values
        task_name: Name of the task
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract data
    flow_trajs = sorted(results.keys())
    mean_rewards = [results[k]['mean_reward'] for k in flow_trajs]
    std_rewards = [results[k]['std_reward'] for k in flow_trajs]
    mean_lengths = [results[k]['mean_length'] for k in flow_trajs]
    std_lengths = [results[k]['std_length'] for k in flow_trajs]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean Reward vs num_flow_trajs
    ax1.errorbar(flow_trajs, mean_rewards, yerr=std_rewards,
                 marker='o', linewidth=2.5, markersize=8,
                 capsize=5, capthick=2, color='#2E86AB',
                 ecolor='#A23B72', elinewidth=2)
    ax1.set_xlabel('Number of Flow Trajectories', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax1.set_title(f'Reward vs Flow Trajectories - {task_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add value labels
    for i, (x, y, std) in enumerate(zip(flow_trajs, mean_rewards, std_rewards)):
        ax1.text(x, y + std + 2, f'{y:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#2E86AB')

    # Highlight important points
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Pure MPPI')
    ax1.axvline(x=48, color='green', linestyle='--', alpha=0.5, label='Default Flow')
    ax1.axvline(x=512, color='red', linestyle='--', alpha=0.5, label='All Flow')
    ax1.legend(fontsize=10)

    # Plot 2: Mean Episode Length vs num_flow_trajs
    ax2.errorbar(flow_trajs, mean_lengths, yerr=std_lengths,
                 marker='s', linewidth=2.5, markersize=8,
                 capsize=5, capthick=2, color='#C73E1D',
                 ecolor='#A23B72', elinewidth=2)
    ax2.set_xlabel('Number of Flow Trajectories', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Episode Length', fontsize=12, fontweight='bold')
    ax2.set_title(f'Episode Length vs Flow Trajectories - {task_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add value labels
    for i, (x, y, std) in enumerate(zip(flow_trajs, mean_lengths, std_lengths)):
        ax2.text(x, y + std + 1, f'{y:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#C73E1D')

    # Highlight important points
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Pure MPPI')
    ax2.axvline(x=48, color='green', linestyle='--', alpha=0.5, label='Default Flow')
    ax2.axvline(x=512, color='red', linestyle='--', alpha=0.5, label='All Flow')
    ax2.legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(save_dir, f"{task_name}_flow_trajs_curve_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Flow trajectories curve saved to: {plot_path}")
    plt.close()


def save_flow_trajs_csv(results, task_name, save_dir):
    """
    Save flow trajectories ablation results to CSV.

    Args:
        results: Dictionary with num_flow_trajs as keys and stats as values
        task_name: Name of the task
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"{task_name}_flow_trajs_{timestamp}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['num_flow_trajs', 'Mean Reward', 'Std Reward', 'Mean Length', 'Std Length'])

        # Write data for each configuration
        for num_flow_trajs in sorted(results.keys()):
            stats = results[num_flow_trajs]
            writer.writerow([
                num_flow_trajs,
                f"{stats['mean_reward']:.4f}",
                f"{stats['std_reward']:.4f}",
                f"{stats['mean_length']:.4f}",
                f"{stats['std_length']:.4f}"
            ])

    print(f"Flow trajectories CSV saved to: {csv_path}")
    return csv_path


def save_results_to_csv(results, task_name, save_dir):
    """
    Save evaluation results to CSV file.

    Args:
        results: Dictionary containing all run results
        task_name: Name of the task
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"{task_name}_{timestamp}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Run', 'Mean Reward', 'Std Reward', 'Mean Length', 'Std Length'] +
                       [f'Episode {i+1} Reward' for i in range(len(results['normal']['all_rewards']))])

        # Write data for each run
        for run_name, stats in results.items():
            row = [
                run_name,
                f"{stats['mean_reward']:.4f}",
                f"{stats['std_reward']:.4f}",
                f"{stats['mean_length']:.4f}",
                f"{stats['std_length']:.4f}"
            ]
            # Add individual episode rewards
            row.extend([f"{r:.4f}" for r in stats['all_rewards']])
            writer.writerow(row)

    print(f"\nCSV results saved to: {csv_path}")
    return csv_path


def plot_comparison(results, task_name, save_dir):
    """
    Create comparison plots for evaluation results.

    Args:
        results: Dictionary containing all run results
        task_name: Name of the task
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare data
    run_names = list(results.keys())
    display_names = {
        'normal': 'Normal (PI+Flow)',
        'no_pi': 'No PI\n(Flow Only)',
        'no_flow_only': 'No Flow\n(PI Only)',
        'pure_mppi': 'Pure MPPI\n(No PI/Flow)'
    }

    labels = [display_names.get(name, name) for name in run_names]
    mean_rewards = [results[name]['mean_reward'] for name in run_names]
    std_rewards = [results[name]['std_reward'] for name in run_names]
    mean_lengths = [results[name]['mean_length'] for name in run_names]
    std_lengths = [results[name]['std_length'] for name in run_names]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean Reward
    colors1 = ['#2E86AB', '#F18F01', '#C73E1D', '#A23B72']
    bars1 = ax1.bar(range(len(run_names)), mean_rewards, yerr=std_rewards,
                    capsize=5, alpha=0.8, color=colors1, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax1.set_title(f'Reward Comparison - {task_name}', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(run_names)))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=mean_rewards[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars1, mean_rewards, std_rewards)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Mean Episode Length
    colors2 = ['#2E86AB', '#F18F01', '#C73E1D', '#A23B72']
    bars2 = ax2.bar(range(len(run_names)), mean_lengths, yerr=std_lengths,
                    capsize=5, alpha=0.8, color=colors2, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Episode Length', fontsize=12, fontweight='bold')
    ax2.set_title(f'Episode Length Comparison - {task_name}', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(run_names)))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=mean_lengths[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars2, mean_lengths, std_lengths)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(save_dir, f"{task_name}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()


def plot_episode_trends(results, task_name, save_dir):
    """
    Plot episode-by-episode reward trends.

    Args:
        results: Dictionary containing all run results
        task_name: Name of the task
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, ax = plt.subplots(figsize=(12, 6))

    display_names = {
        'normal': 'Normal (PI+Flow)',
        'no_pi': 'No PI (Flow Only)',
        'no_flow_only': 'No Flow (PI Only)',
        'pure_mppi': 'Pure MPPI (No PI/Flow)'
    }

    colors = {
        'normal': '#2E86AB',
        'no_pi': '#F18F01',
        'no_flow_only': '#C73E1D',
        'pure_mppi': '#A23B72'
    }

    markers = {
        'normal': 'o',
        'no_pi': '^',
        'no_flow_only': 'D',
        'pure_mppi': 's'
    }

    # Plot each configuration
    for run_name, stats in results.items():
        episodes = range(1, len(stats['all_rewards']) + 1)
        rewards = stats['all_rewards']
        label = display_names.get(run_name, run_name)
        ax.plot(episodes, rewards, marker=markers.get(run_name, 'o'),
               color=colors.get(run_name, 'gray'), linewidth=2, markersize=8,
               label=label, alpha=0.8)

    ax.set_xlabel('Episode Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title(f'Episode-wise Reward Trends - {task_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(save_dir, f"{task_name}_trends_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Trends plot saved to: {plot_path}")
    plt.close()


@torch.no_grad()
def evaluate_with_config(model_path, config_path, task_name, num_episodes=5,
                         num_pi_trajs=None, num_flow_trajs=None, run_label=""):
    """
    Evaluate a model with specific configuration.

    Args:
        model_path: Path to the trained model checkpoint
        config_path: Path to the config file
        task_name: Name of the task
        num_episodes: Number of episodes to evaluate
        num_pi_trajs: Override for num_pi_trajs (None uses config default)
        num_flow_trajs: Override for num_flow_trajs (None uses config default)
        run_label: Label for this run (e.g., "Normal", "No PI/Flow")

    Returns:
        dict: Statistics from the run
    """
    print("="*60)
    print(f"{run_label} Run: {task_name}")
    print("="*60)

    # Load config
    cfg = OmegaConf.load(config_path)

    # Override settings for evaluation
    cfg.task = task_name
    cfg.env_type = "dm_control"
    cfg.eval_episodes = num_episodes
    cfg.task_dim = 0
    cfg.compile = False  # Disable compile to match saved model

    # Disable struct mode to allow setting missing required values
    OmegaConf.set_struct(cfg, False)

    # Override planning parameters if specified
    if num_pi_trajs is not None:
        cfg.num_pi_trajs = num_pi_trajs
        print(f"Overriding num_pi_trajs to {num_pi_trajs}")
    if num_flow_trajs is not None:
        cfg.num_flow_trajs = num_flow_trajs
        print(f"Overriding num_flow_trajs to {num_flow_trajs}")

    print(f"Task: {cfg.task}")
    print(f"Model path: {model_path}")
    print(f"Number of episodes: {num_episodes}")
    print(f"num_pi_trajs: {cfg.num_pi_trajs}")
    print(f"num_flow_trajs: {cfg.num_flow_trajs}")

    # Create environment
    # print("\nCreating environment...")
    env = make_env(cfg)
    # print(f"Environment created: {cfg.task}")

    # Create agent and load model
    # print("\nCreating agent...")
    agent = BOOM(cfg)
    print(f"Loading checkpoint from: {model_path}")
    agent.load(model_path)
    print("Model loaded successfully!")

    # Evaluate
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60)

    all_rewards = []
    all_episode_lengths = []

    for episode_idx in range(num_episodes):
        obs, done, ep_reward, t = env.reset()[0], False, 0, 0

        print(f"\nEpisode {episode_idx + 1}/{num_episodes}")

        # Create progress bar for this episode
        max_steps = 1000
        pbar = tqdm(total=max_steps, desc=f"  Steps", unit="step", ncols=100)

        while not done:
            # Get action from agent
            action, _, _ = agent.act(obs, t0=(t == 0), eval_mode=True)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            ep_reward += reward
            t += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'reward': f'{ep_reward:.1f}'})

            # Adjust max steps if we exceed it
            if t >= pbar.total:
                pbar.total = t + 100

        pbar.close()

        all_rewards.append(ep_reward)
        all_episode_lengths.append(t)

        # Try to get success rate if available
        success = info.get("success", None)
        if success is not None:
            print(f"  ✓ Reward: {ep_reward:.2f}, Success: {success}, Steps: {t}")
        else:
            print(f"  ✓ Reward: {ep_reward:.2f}, Steps: {t}")

    # env.close()

    # Compute statistics
    stats = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_length': np.mean(all_episode_lengths),
        'std_length': np.std(all_episode_lengths),
        'all_rewards': all_rewards,
        'all_lengths': all_episode_lengths
    }

    # Print summary
    print("\n" + "="*60)
    print(f"{run_label} Run Summary")
    print("="*60)
    print(f"Mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"Mean episode length: {stats['mean_length']:.1f} +/- {stats['std_length']:.1f}")
    print(f"All rewards: {[f'{r:.2f}' for r in all_rewards]}")
    print(f"All lengths: {all_episode_lengths}")
    print("="*60)

    return stats


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare MPPI with and without flow/PI trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default test (5 episodes)
  python %(prog)s

  # Run with 10 episodes
  python %(prog)s --episodes 10

  # Use specific model
  python %(prog)s --model logs/humanoid-run/11/0415-202822-flow-rho/models/final.pt
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logs/humanoid-run/11/0415-202822-flow-rho/models/final.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="boom/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="humanoid-run",
        help="Name of the task"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes per run"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Create results directory
    results_dir = os.path.join("mppi_test_results", args.task)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*60)
    print("MPPI Comparison Test")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Episodes per run: {args.episodes}")
    print(f"Results directory: {results_dir}")
    print("="*60)

    # =====================================================
    # Part 1: PI/Flow Ablation Study (4 configurations)
    # =====================================================
    # Comment out the next line to skip this part
    # run_flow_pi_ablation(
    #     model_path=args.model,
    #     config_path=args.config,
    #     task_name=args.task,
    #     num_episodes=args.episodes,
    #     results_dir=results_dir
    # )

    # =====================================================
    # Part 2: Flow Trajectories Ablation Study (9 configurations)
    # =====================================================
    # Comment out the next line to skip this part
    run_flow_trajs_ablation(
        model_path=args.model,
        config_path=args.config,
        task_name=args.task,
        num_episodes=args.episodes,
        results_dir=results_dir,
        num_points=9
    )

    print("\n" + "="*60)
    print("All experiments completed successfully!")
    print(f"Results directory: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
