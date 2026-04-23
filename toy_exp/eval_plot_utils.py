"""
Utility functions for standardized evaluation plotting.

This module provides functions to create standardized plots during evaluation:

================================================================================
DEBUG_INFO FORMAT (from flow_alg.py)
================================================================================

debug_info = {
    # 候选轨迹信息（第一次 MPPI 迭代）
    'pi_candidates': {
        'actions': Tensor[H, num_pi, A] or None,  # 从 pi 策略采样的轨迹动作
        'values': Tensor[num_pi] or None,          # 对应的 value 值
    },
    'flow_candidates': {
        'actions': Tensor[H, num_flow, A] or None,  # 从 flow 策略采样的轨迹动作
        'values': Tensor[num_flow] or None,          # 对应的 value 值
    },
    'random_candidates': {
        'actions': Tensor[H, num_random, A] or None,  # 随机采样的轨迹动作
        'values': Tensor[num_random] or None,          # 对应的 value 值
    },

    # MPPI 迭代优化信息（每次迭代的统计）
    'refinements': [
        {
            'iteration': int,                              # 迭代索引
            'mean': Tensor[H, A],                         # 当前迭代的高斯均值
            'std': Tensor[H, A],                          # 当前迭代的高斯标准差
            'elite_values': Tensor[num_elites],           # 精英轨迹的 value 值
            'init_actions': Tensor[H, num_samples, A],     # 第0次迭代的动作样本
            'init_values': Tensor[num_samples],            # 第0次迭代的 value 值
            'final_actions': Tensor[H, num_samples, A],    # 最后一次迭代的动作样本
            'final_values': Tensor[num_samples],           # 最后一次迭代的 value 值
        },
        # ... 更多迭代（每个迭代一个字典）
    ],
}

维度说明:
- H: horizon（轨迹长度）
- A: action_dim（动作维度）
- num_pi/num_flow/num_random: 各策略采样的轨迹数量
- num_elites: 精英轨迹数量
- num_samples: MPPI 每次迭代的采样数（num_samples = num_pi + num_flow + num_random）

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
from typing import List, Dict, Any, Optional
import torch
from boom.common.debug import plot_mppi_trajectories_on_axis, plot_real_trajectories, plot_trajs


def plot_action_distribution(debug_info: Dict[str, Any], save_dir: str):
    """Plot action distribution histogram with value scatter on secondary y-axis."""
    # Collect all candidates with types and values
    all_actions = []
    all_types = []  # 'pi', 'flow', 'random'
    all_values = []  # Store trajectory values

    if 'pi_candidates' in debug_info and debug_info['pi_candidates']['actions'] is not None:
        pi_actions = debug_info['pi_candidates']['actions']
        pi_values = debug_info['pi_candidates']['values']
        if torch.is_tensor(pi_actions):
            pi_actions = pi_actions.cpu().numpy()
        if torch.is_tensor(pi_values):
            pi_values = pi_values.cpu().numpy()
        all_actions.append(pi_actions)
        all_values.append(pi_values)
        all_types.extend(['pi'] * pi_actions.shape[1])

    if 'flow_candidates' in debug_info and debug_info['flow_candidates']['actions'] is not None:
        flow_actions = debug_info['flow_candidates']['actions']
        flow_values = debug_info['flow_candidates']['values']
        if torch.is_tensor(flow_actions):
            flow_actions = flow_actions.cpu().numpy()
        if torch.is_tensor(flow_values):
            flow_values = flow_values.cpu().numpy()
        all_actions.append(flow_actions)
        all_values.append(flow_values)
        all_types.extend(['flow'] * flow_actions.shape[1])

    if 'random_candidates' in debug_info and debug_info['random_candidates']['actions'] is not None:
        random_actions = debug_info['random_candidates']['actions']
        random_values = debug_info['random_candidates']['values']
        if torch.is_tensor(random_actions):
            random_actions = random_actions.cpu().numpy()
        if torch.is_tensor(random_values):
            random_values = random_values.cpu().numpy()
        all_actions.append(random_actions)
        all_values.append(random_values)
        all_types.extend(['random'] * random_actions.shape[1])

    if not all_actions:
        print("No candidate data available for action distribution")
        return

    # Concatenate all candidates
    all_actions = np.concatenate(all_actions, axis=1)  # [H, total, A]
    all_values = np.concatenate(all_values, axis=0)  # [total]

    # Get first timestep actions
    first_actions = all_actions[0, :, :]  # [num_samples, action_dim]

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()  # Create secondary y-axis

    # Count each type
    pi_count = sum(1 for t in all_types if t == 'pi')
    flow_count = sum(1 for t in all_types if t == 'flow')
    random_count = sum(1 for t in all_types if t == 'random')

    # Separate actions by type for histogram
    pi_actions = []
    flow_actions = []
    random_actions = []
    pi_values_list = []
    flow_values_list = []
    random_values_list = []

    for i, action_type in enumerate(all_types):
        if action_type == 'pi':
            pi_actions.append(first_actions[i, 0])
            pi_values_list.append(all_values[i])
        elif action_type == 'flow':
            flow_actions.append(first_actions[i, 0])
            flow_values_list.append(all_values[i])
        elif action_type == 'random':
            random_actions.append(first_actions[i, 0])
            random_values_list.append(all_values[i])

    # Plot histograms on primary y-axis (left)
    if pi_actions:
        ax1.hist(pi_actions, bins=50, range=(-1, 1), alpha=0.3,
                color='#9467bd', label=f'PI Hist (n={len(pi_actions)})',
                edgecolor='black')
    if flow_actions:
        ax1.hist(flow_actions, bins=50, range=(-1, 1), alpha=0.3,
                color='#1f77b4', label=f'Flow Hist (n={len(flow_actions)})',
                edgecolor='black')
    if random_actions:
        ax1.hist(random_actions, bins=50, range=(-1, 1), alpha=0.3,
                color='gray', label=f'Random Hist (n={len(random_actions)})',
                edgecolor='black')

    # Plot scatter points on secondary y-axis (right)
    # Add small jitter to x-axis to avoid overlapping points
    jitter_strength = 0.01

    if pi_actions and pi_values_list:
        jitter = np.random.randn(len(pi_actions)) * jitter_strength
        ax2.scatter(np.array(pi_actions) + jitter, pi_values_list,
                   c='#9467bd', alpha=0.6, s=20, label=f'PI Values (n={len(pi_actions)})',
                   edgecolors='black', linewidths=0.5, zorder=10)

    if flow_actions and flow_values_list:
        jitter = np.random.randn(len(flow_actions)) * jitter_strength
        ax2.scatter(np.array(flow_actions) + jitter, flow_values_list,
                   c='#1f77b4', alpha=0.6, s=20, label=f'Flow Values (n={len(flow_actions)})',
                   edgecolors='black', linewidths=0.5, zorder=10)

    if random_actions and random_values_list:
        jitter = np.random.randn(len(random_actions)) * jitter_strength
        ax2.scatter(np.array(random_actions) + jitter, random_values_list,
                   c='gray', alpha=0.4, s=10, label=f'Random Values (n={len(random_actions)})',
                   edgecolors='black', linewidths=0.5, zorder=10)

    # Set labels and titles
    ax1.set_xlabel('Action Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12, color='#333333')
    ax2.set_ylabel('Trajectory Value', fontsize=12, color='#333333')
    ax1.set_xlim(-1, 1)

    # Set title
    ax1.set_title('Initial Action Distribution (Step 0, Action Dim 0)\nHistogram + Value Scatter Plot',
                fontsize=14, fontweight='bold')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    # Grid settings
    ax1.grid(True, alpha=0.3, axis='y')
    ax2.grid(False)  # Don't show grid for secondary axis

    # Add statistics box
    all_first_actions = first_actions[:, 0]
    mean_action = all_first_actions.mean()
    std_action = all_first_actions.std()
    mean_value = all_values.mean()
    std_value = all_values.std()
    stats_text = (f"Action - Mean: {mean_action:.3f}, Std: {std_action:.3f}\n"
                 f"Value - Mean: {mean_value:.3f}, Std: {std_value:.3f}\n"
                 f"Total: {len(all_types)} (PI:{pi_count}, Flow:{flow_count}, Random:{random_count})")
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save figure
    save_path = os.path.join(save_dir, 'mppi_1_action_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_buffer_stats(replay_obs, replay_action, replay_reward, step, save_dir, prefix='act'):
    """
    Plot action-value distribution as arrows on a 2D grid.

    Args:
        replay_obs: Observations (positions), shape [T, B, 2]
        replay_action: Actions (directions), shape [T, B, D]
        replay_reward: Values, shape [B, 1]
        step: Current training step
        save_dir: Directory to save the plot
        prefix: Prefix for filename ('act' or 'mu')
    """
    # Convert to numpy
    replay_obs = replay_obs.detach().cpu().numpy()
    replay_action = replay_action.detach().cpu().numpy()
    replay_reward = replay_reward.detach().cpu().numpy()

    T, B, D = replay_action.shape

    # Flatten replay_reward from [B, 1] to [B]
    values = replay_reward[:, 0]  # [B]

    # Only use t=0
    t = 0
    positions = replay_obs[t]  # [B, 2]
    actions = replay_action[t]  # [B, D] - D=1, action in [-1, 1]

    # Convert 1D action [-1, 1] to angle [-pi, pi], then to 2D direction vector
    # action values are linearly mapped: -1 -> -pi, 1 -> pi
    angles = actions[:, 0] * np.pi  # [B], now in [-pi, pi]
    action_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # [B, 2]

    # Create figure with 2 subplots: full view and zoomed view
    fig = plt.figure(figsize=(8, 4))

    # Left subplot: full view [-1.1, 1.1]
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.quiver(
        positions[:, 0], positions[:, 1],  # Arrow origins
        action_dirs[:, 0], action_dirs[:, 1],  # Arrow directions (already normalized)
        values,  # Color
        cmap='viridis',
        alpha=0.9,
        scale=15,  # Controls arrow size (larger = shorter arrows)
        width=0.008,  # Arrow shaft width
        headwidth=4,  # Arrow head width
        headlength=5,  # Arrow head length
        headaxislength=4,  # Arrow head axis length
        pivot='mid'
    )

    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_xlabel('Position X', fontsize=10)
    ax1.set_ylabel('Position Y', fontsize=10)
    ax1.set_title('Full View', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Value', fontsize=9)

    # Right subplot: zoomed view [-0.1, 0.1]
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.quiver(
        positions[:, 0], positions[:, 1],  # Arrow origins
        action_dirs[:, 0], action_dirs[:, 1],  # Arrow directions (already normalized)
        values,  # Color
        cmap='viridis',
        alpha=0.9,
        scale=15,  # Controls arrow size (larger = shorter arrows)
        width=0.008,  # Arrow shaft width
        headwidth=4,  # Arrow head width
        headlength=5,  # Arrow head length
        headaxislength=4,  # Arrow head axis length
        pivot='mid'
    )

    ax2.set_xlim(-0.1, 0.1)
    ax2.set_ylim(-0.1, 0.1)
    ax2.set_xlabel('Position X', fontsize=10)
    ax2.set_ylabel('Position Y', fontsize=10)
    ax2.set_title('Zoomed View', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Value', fontsize=9)

    plt.tight_layout()

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'buffer_{prefix}_step_{step}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Buffer statistics plot saved to {save_path}")


def plot_mppi_refinements(debug_info: Dict[str, Any], save_dir: str):
    """
    Plot MPPI refinement process: mean and std evolution over iterations.

    Args:
        debug_info: Debug information from flow_alg.plan()
        save_dir: Directory to save the plot
    """

    if 'refinements' not in debug_info or not debug_info['refinements']:
        print("No refinements data available")
        return

    refinements = debug_info['refinements']
    num_iterations = len(refinements)
    horizon = refinements[0]['mean'].shape[0]
    action_dim = refinements[0]['mean'].shape[1]

    # Extract data over iterations
    means = [r['mean'].cpu().numpy() if torch.is_tensor(r['mean']) else r['mean']
             for r in refinements]  # List of [H, A]
    stds = [r['std'].cpu().numpy() if torch.is_tensor(r['std']) else r['std']
            for r in refinements]  # List of [H, A]
    elite_values = [r['elite_values'].cpu().numpy() if torch.is_tensor(r['elite_values']) else r['elite_values']
                    for r in refinements]  # List of [num_elites]

    # Stack arrays
    means = np.stack(means, axis=0)  # [num_iters, H, A]
    stds = np.stack(stds, axis=0)    # [num_iters, H, A]

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Mean evolution for first timestep, first action dim
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(means[:, 0, 0], 'b-', linewidth=2, marker='o', label='Mean')
    ax1.fill_between(range(num_iterations),
                     means[:, 0, 0] - stds[:, 0, 0],
                     means[:, 0, 0] + stds[:, 0, 0],
                     alpha=0.3, color='blue', label='±1 Std')
    ax1.set_xlabel('MPPI Iteration', fontsize=11)
    ax1.set_ylabel('Action Value', fontsize=11)
    ax1.set_title('Step 0, Action Dim 0: Mean Evolution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Std evolution for first timestep
    ax2 = plt.subplot(2, 3, 2)
    for a in range(min(action_dim, 4)):  # Plot up to 4 action dims
        ax2.plot(stds[:, 0, a], linewidth=2, marker='o', label=f'Action Dim {a}')
    ax2.set_xlabel('MPPI Iteration', fontsize=11)
    ax2.set_ylabel('Std', fontsize=11)
    ax2.set_title('Step 0: Std Evolution by Action Dim', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Elite values evolution
    ax3 = plt.subplot(2, 3, 3)
    elite_values_array = np.stack(elite_values, axis=0)  # [num_iters, num_elites]
    for i in range(min(elite_values_array.shape[1], 5)):  # Plot up to 5 elites
        ax3.plot(elite_values_array[:, i], linewidth=1.5, alpha=0.7, label=f'Elite {i}')
    ax3.set_xlabel('MPPI Iteration', fontsize=11)
    ax3.set_ylabel('Value', fontsize=11)
    ax3.set_title('Elite Values Evolution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mean heatmap (timestep vs iteration, for action dim 0)
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(means[:, :, 0].T, aspect='auto', cmap='RdBu_r',
                     extent=[0, num_iterations, horizon-1, 0])
    ax4.set_xlabel('MPPI Iteration', fontsize=11)
    ax4.set_ylabel('Timestep', fontsize=11)
    ax4.set_title('Action Dim 0: Mean Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Action Value')

    # Plot 5: Std heatmap (timestep vs iteration, for action dim 0)
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(stds[:, :, 0].T, aspect='auto', cmap='YlOrRd',
                     extent=[0, num_iterations, horizon-1, 0])
    ax5.set_xlabel('MPPI Iteration', fontsize=11)
    ax5.set_ylabel('Timestep', fontsize=11)
    ax5.set_title('Action Dim 0: Std Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='Std')

    # Plot 6: Elite values statistics
    ax6 = plt.subplot(2, 3, 6)
    elite_mean = elite_values_array.mean(axis=1)
    elite_std = elite_values_array.std(axis=1)
    elite_max = elite_values_array.max(axis=1)
    elite_min = elite_values_array.min(axis=1)

    ax6.plot(elite_mean, 'g-', linewidth=2, marker='o', label='Mean')
    ax6.fill_between(range(num_iterations),
                     elite_mean - elite_std,
                     elite_mean + elite_std,
                     alpha=0.2, color='green')
    ax6.plot(elite_max, 'r--', linewidth=1.5, alpha=0.7, label='Max')
    ax6.plot(elite_min, 'b--', linewidth=1.5, alpha=0.7, label='Min')
    ax6.set_xlabel('MPPI Iteration', fontsize=11)
    ax6.set_ylabel('Value', fontsize=11)
    ax6.set_title('Elite Values Statistics', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, 'mppi_refinements.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_mppi_trajs(debug_info, save_dir):
    fig = plt.figure(figsize=(12, 6))
    # Initial MPPI trajectories
    ax1 = plt.subplot(1, 2, 1)
    plot_mppi_trajectories_on_axis(ax1, debug_info, step_size=0.4, goal_radius=0.1)
    ax1.set_title('Initial MPPI Trajectories', fontsize=12, fontweight='bold')

    # Plot 8: Final MPPI trajectories
    ax2 = plt.subplot(1, 2, 2)
    last_iter = debug_info['refinements'][-1]
    plot_mppi_trajectories_on_axis(
        ax2, 
        {'random_candidates': {
            'actions': last_iter['final_actions'], 
            'values': last_iter['final_values'],
        }}, 
        step_size=0.4, 
        goal_radius=0.1
    )
    ax2.set_title('Final MPPI Trajectories', fontsize=12, fontweight='bold')
    # Save figure
    save_path = os.path.join(save_dir, 'mppi_trajs.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")



def plot_mppi_init_vs_final(debug_info: Dict[str, Any], save_dir: str):
    """
    Plot comparison of initial vs final MPPI iterations.

    Args:
        debug_info: Debug information from flow_alg.plan()
        save_dir: Directory to save the plot
    """
    if 'refinements' not in debug_info or not debug_info['refinements']:
        print("No refinements data available")
        return

    refinements = debug_info['refinements']

    # Get first and last iteration data
    first_iter = refinements[0]
    last_iter = refinements[-1]

    # Check if we have init and final data
    has_init = 'init_actions' in first_iter and first_iter['init_actions'] is not None
    has_final = 'final_actions' in last_iter and last_iter['final_actions'] is not None

    if not has_init or not has_final:
        print("No init/final iteration data available")
        return

    init_actions = first_iter['init_actions']  # [H, num_samples, A]
    init_values = first_iter['init_values']    # [num_samples]
    final_actions = last_iter['final_actions'] # [H, num_samples, A]
    final_values = last_iter['final_values']   # [num_samples]

    # Convert to numpy if needed
    if torch.is_tensor(init_actions):
        init_actions = init_actions.cpu().numpy()
    if torch.is_tensor(init_values):
        init_values = init_values.cpu().numpy()
    if torch.is_tensor(final_actions):
        final_actions = final_actions.cpu().numpy()
    if torch.is_tensor(final_values):
        final_values = final_values.cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(16, 8))

    # Plot 1: Initial action distribution (step 0, action dim 0)
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(init_actions[0, :, 0], bins=50, range=(-1, 1), alpha=0.6,
             color='blue', edgecolor='black', label='Initial')
    ax1.set_xlabel('Action Value', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Initial: Action Distribution (Step 0, Dim 0)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add statistics
    init_mean = init_actions[0, :, 0].mean()
    init_std = init_actions[0, :, 0].std()
    ax1.axvline(init_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {init_mean:.3f}')
    ax1.legend(fontsize=9)

    # Plot 2: Final action distribution (step 0, action dim 0)
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(final_actions[0, :, 0], bins=50, range=(-1, 1), alpha=0.6,
             color='green', edgecolor='black', label='Final')
    ax2.set_xlabel('Action Value', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Final: Action Distribution (Step 0, Dim 0)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics
    final_mean = final_actions[0, :, 0].mean()
    final_std = final_actions[0, :, 0].std()
    ax2.axvline(final_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {final_mean:.3f}')
    ax2.legend(fontsize=9)

    # Plot 3: Side-by-side comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(init_actions[0, :, 0], bins=50, range=(-1, 1), alpha=0.5,
             color='blue', label='Initial', edgecolor='black')
    ax3.hist(final_actions[0, :, 0], bins=50, range=(-1, 1), alpha=0.5,
             color='green', label='Final', edgecolor='black')
    ax3.set_xlabel('Action Value', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Initial value distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(init_values, bins=50, alpha=0.6, color='blue', edgecolor='black')
    ax4.set_xlabel('Value', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Initial: Value Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Final value distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(final_values, bins=50, alpha=0.6, color='green', edgecolor='black')
    ax5.set_xlabel('Value', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Final: Value Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Value comparison scatter
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(init_values, final_values, alpha=0.5, s=20)
    ax6.plot([init_values.min(), init_values.max()],
             [init_values.min(), init_values.max()],
             'r--', linewidth=2, label='No Improvement')
    ax6.set_xlabel('Initial Value', fontsize=11)
    ax6.set_ylabel('Final Value', fontsize=11)
    ax6.set_title('Value Improvement', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Add improvement statistics
    improvement = final_values - init_values
    mean_improvement = improvement.mean()
    pct_improved = (improvement > 0).mean() * 100

    stats_text = f"Mean Improvement: {mean_improvement:.3f}\n% Improved: {pct_improved:.1f}%"
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, 'mppi_3_init_vs_final.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_all_debug_info(real_trajectories, debug_info, save_dir):
    """
    Plot all debug_info visualizations in one function call.
    Args:
        real_trajectories: Real evaluation trajectories
        debug_info: Debug information from flow_alg.plan()
        save_dir: Directory to save the plots
    """

    print("Generating debug_info visualizations...")

    # Plot action distribution
    plot_action_distribution(debug_info, save_dir)

    # Plot MPPI refinements (includes init and final trajectories)
    plot_mppi_refinements(debug_info, save_dir)
    plot_mppi_trajs(debug_info, save_dir)

    # Plot init vs final comparison
    plot_mppi_init_vs_final(debug_info, save_dir)

    # Plot real trajectories
    plot_real_trajectories(real_trajectories, save_dir, filename='final_trajectories.png',
                          step_size=0.4, goal_radius=0.1)

    print("All debug_info visualizations completed!")
