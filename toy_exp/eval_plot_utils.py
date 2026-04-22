"""
Utility functions for standardized evaluation plotting.

This module provides functions to create standardized plots during evaluation:
- mppi_0_combined.png: 1x3 grid showing MPPI trajectory evolution
  [0] All proposals
  [1] Elite 1 first MPPI iteration
  [2] Elite 1 last MPPI iteration
- mppi_1_action_distribution.png: Action distribution histogram with value scatter
- buffer_stat.png: Action-reward distribution from replay buffer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
from typing import List, Dict, Any, Optional
import torch



def plot_combined_mppi_figure(debug_info: Dict[str, Any], env, save_dir: str, step_size: float, goal_radius: float):
    """Plot 0: Combined figure with 3 subplots in 1x3 grid.
    - [0]: All proposals
    - [1]: Elite 1 first MPPI iteration
    - [2]: Elite 1 last MPPI iteration
    """
    from boom.common.debug import _simulate_from_actions, _setup_goals_on_axis, _compute_colormap_norm

    # Create figure with 1x3 grid
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ==================== Subplot 0: All proposals ====================
    ax0 = fig.add_subplot(gs[0, 0])
    _plot_all_proposals_on_axis(debug_info, ax0, step_size, goal_radius)

    # ==================== Subplot 1: Elite 1 first MPPI iteration ====================
    ax1 = fig.add_subplot(gs[0, 1])
    _plot_elite_mppi_iteration_on_axis(debug_info, ax1, step_size, goal_radius,
                                        elite_idx=0, iteration='first')

    # ==================== Subplot 2: Elite 1 last MPPI iteration ====================
    ax2 = fig.add_subplot(gs[0, 2])
    _plot_elite_mppi_iteration_on_axis(debug_info, ax2, step_size, goal_radius,
                                        elite_idx=0, iteration='last')

    # Save combined figure
    save_path = os.path.join(save_dir, 'mppi_0_combined.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def _plot_all_proposals_on_axis(debug_info: Dict[str, Any], ax, step_size: float, goal_radius: float):
    """Plot all proposals on given axis with different styles for each type."""
    from boom.common.debug import _simulate_from_actions, _setup_goals_on_axis, _compute_colormap_norm

    # Collect all candidates
    all_actions = []
    all_values = []
    all_types = []  # 'pi', 'flow', or 'random'

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
        ax.text(0.5, 0.5, 'No candidate data',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, style='italic', color='gray')
        return

    # Concatenate all candidates
    all_actions = np.concatenate(all_actions, axis=1)  # [H, total, A]
    all_values = np.concatenate(all_values, axis=0)  # [total]

    H, num_samples, action_dim = all_actions.shape

    # Compute colormap and get top-5
    norm, cmap = _compute_colormap_norm(all_values)
    top_5_indices = np.argsort(all_values)[-5:]

    _setup_goals_on_axis(ax, goal_radius, show_gradient=False)

    # Plot trajectories with different styles based on type
    for i in range(num_samples):
        actions_rad = all_actions[:, i, :] * np.pi
        positions = _simulate_from_actions(actions_rad, step_size)
        value = all_values[i]
        traj_type = all_types[i] if i < len(all_types) else 'unknown'

        # Highlight top-5 with red color
        is_top_5 = i in top_5_indices

        if is_top_5:
            color = 'red'
        else:
            # Use value-based colormap for non-top-5
            if norm is not None and cmap is not None:
                color = cmap(norm(value))
            else:
                color = '#9467bd'

        # Different styles for different types (similar to debug.py)
        if traj_type == 'random':
            linestyle = ':'
            linewidth = 2.5 if is_top_5 else 1.0
            alpha = 0.9 if is_top_5 else 0.4
            zorder = 4
        elif traj_type == 'pi':
            linestyle = '--'
            linewidth = 3.0 if is_top_5 else 1.5
            alpha = 1.0 if is_top_5 else 0.6
            zorder = 5
        else:  # flow
            linestyle = '-'
            linewidth = 3.0 if is_top_5 else 1.5
            alpha = 1.0 if is_top_5 else 0.6
            zorder = 6

        ax.plot(positions[:, 0], positions[:, 1],
              color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth, zorder=zorder)

    # Add colorbar
    if norm is not None and cmap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_ylabel('Trajectory Value', fontsize=10)

    # Count types
    pi_count = sum(1 for t in all_types if t == 'pi')
    flow_count = sum(1 for t in all_types if t == 'flow')
    random_count = sum(1 for t in all_types if t == 'random')

    # Add legend for trajectory types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle=':', linewidth=1.0, alpha=0.6, label=f'Random (n={random_count})'),
        Line2D([0], [0], color='#9467bd', linestyle='--', linewidth=1.5, alpha=0.6, label=f'PI (n={pi_count})'),
        Line2D([0], [0], color='#1f77b4', linestyle='-', linewidth=1.5, alpha=0.6, label=f'Flow (n={flow_count})'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

    ax.set_title(f'All Proposals (Top-5 Bold, N={num_samples})',
                fontsize=14, fontweight='bold')


def _plot_elite_mppi_iteration_on_axis(debug_info: Dict[str, Any], ax, step_size: float, goal_radius: float,
                                       elite_idx: int, iteration: str = 'first'):
    """Plot elite MPPI iteration (first or last) on given axis.

    Args:
        debug_info: Debug information from MPPI planning
        ax: Matplotlib axis to plot on
        step_size: Step size for trajectory simulation
        goal_radius: Goal radius for environment visualization
        elite_idx: Index of the elite trajectory to plot
        iteration: 'first' or 'last' iteration to plot
        iteration: 'first' or 'last' - which MPPI iteration to plot
    """
    from boom.common.debug import _simulate_from_actions, _setup_goals_on_axis, _compute_colormap_norm

    refinements = debug_info.get('refinements', [])

    if elite_idx >= len(refinements) or not refinements[elite_idx]:
        ax.text(0.5, 0.5, f'No Elite {elite_idx+1} Data',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, style='italic', color='gray')
        return

    elite_refinement = refinements[elite_idx]

    # Get the requested iteration
    if iteration == 'first':
        iter_data = elite_refinement[0] if elite_refinement else None
        title_suffix = 'First Iter'
        actions_key = 'init_actions'
        values_key = 'init_values'
    elif iteration == 'last':
        iter_data = elite_refinement[-1] if elite_refinement else None
        title_suffix = 'Last Iter'
        actions_key = 'final_actions'
        values_key = 'final_values'
    else:
        ax.text(0.5, 0.5, f'Invalid iteration: {iteration}',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, style='italic', color='gray')
        return

    if iter_data is None or actions_key not in iter_data:
        ax.text(0.5, 0.5, f'No Elite {elite_idx+1} {title_suffix} Data',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, style='italic', color='gray')
        return

    init_actions = iter_data[actions_key]  # [H, num_samples, A]
    init_values = iter_data[values_key]    # [num_samples]

    if torch.is_tensor(init_actions):
        init_actions = init_actions.cpu().numpy()
    if torch.is_tensor(init_values):
        init_values = init_values.cpu().numpy()

    H, num_samples, action_dim = init_actions.shape

    # Compute colormap
    norm, cmap = _compute_colormap_norm(init_values)

    _setup_goals_on_axis(ax, goal_radius, show_gradient=False)

    # All sampled trajectories use thin uniform style (no type distinction needed)
    sample_linewidth = 0.5  # Very thin lines for the many random samples
    sample_alpha = 0.3  # Low alpha for better visibility

    for i in range(num_samples):
        actions_rad = init_actions[:, i, :] * np.pi
        positions = _simulate_from_actions(actions_rad, step_size)
        value = init_values[i]

        # Use value-based colormap
        if norm is not None and cmap is not None:
            color = cmap(norm(value))
        else:
            color = '#1f77b4'

        ax.plot(positions[:, 0], positions[:, 1],
              color=color, alpha=sample_alpha, linewidth=sample_linewidth, zorder=6)

    # Plot mean trajectory (the unique elite trajectory)
    mean = iter_data.get('mean')
    if mean is not None:
        if torch.is_tensor(mean):
            mean = mean.cpu().numpy()
        mean_actions_rad = mean * np.pi
        mean_positions = _simulate_from_actions(mean_actions_rad, step_size)

        # Mean trajectory uses thick red line to stand out
        ax.plot(mean_positions[:, 0], mean_positions[:, 1],
              color='red', linewidth=3.0, zorder=7, label='Mean Elite Trajectory')

    # Add colorbar
    if norm is not None and cmap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_ylabel('Value', fontsize=10)

    # Update title
    ax.set_title(f'Elite {elite_idx+1}: {title_suffix} (N={num_samples})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)


def plot_action_distribution(debug_info: Dict[str, Any], env, save_dir: str):
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



def plot_final_trajectories(real_trajectories: List[Dict], env, save_path: str):
    """
    Plot all final execution trajectories from evaluation episodes.

    Args:
        real_trajectories: List of trajectory dictionaries
        env: Environment instance
        save_path: Path to save the figure
    """
    # Use boom.common.debug.plot_trajs for consistency
    from boom.common.debug import plot_trajs

    # Create save directory
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # Plot using boom's function
    plot_trajs(
        trajectories=real_trajectories,
        step=0,  # Placeholder, will be renamed by caller
        save_dir=save_dir,
        traj_type='real',
        layout='single',
        use_value_cmap=False,
        step_size=env.step_size,
        goal_radius=env.goal_radius,
        show_stats=True
    )

    # Rename the file to match expected name
    old_path = os.path.join(save_dir, 'trajectories_step_0.png')
    if os.path.exists(old_path):
        os.rename(old_path, save_path)
        print(f"Saved final trajectories plot to {save_path}")


def plot_buffer_stats(replay_action, replay_reward, step, save_dir):
    """
    Plot action-reward distribution from replay buffer.

    Args:
        replay_action: Tensor of shape [T, B, D] or numpy array
        replay_reward: Tensor of shape [T, B, 1] or numpy array
        step: Current training step
        save_dir: Directory to save the plot
    """
    # Convert to numpy if needed
    if torch.is_tensor(replay_action):
        replay_action = replay_action.detach().cpu().numpy()
    if torch.is_tensor(replay_reward):
        replay_reward = replay_reward.detach().cpu().numpy()

    T, B, D = replay_action.shape

    # Create figure with subplots
    fig = plt.figure(figsize=(4 * T, 3 * min(D, 2)))

    # Plot scatter plots for each timestep and action dimension
    for t in range(T):
        for d in range(min(D, 2)):  # Limit to 2 action dimensions
            ax = plt.subplot(min(D, 2), T, d * T + t + 1)

            action_values = replay_action[t, :, d]
            reward_values = replay_reward[:, 0]

            im = ax.scatter(action_values, reward_values, c=reward_values,
                          cmap='RdYlGn', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

            ax.set_xlabel('Action', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f'Timestep {t}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Value', fontsize=9)

    plt.tight_layout()

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'buffer_stat_step_{step}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Buffer statistics plot saved to {save_path}")
