"""
Utility functions for standardized evaluation plotting.

This module provides functions to create standardized plots during evaluation:
- pi_mppi.png: 2x2 grid showing PI-MPPI trajectory evolution
- flow_mppi.png: 2x2 grid showing Flow-MPPI trajectory evolution
- buffer_stat.png: Action-reward distribution from replay buffer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
from typing import List, Dict, Any, Optional
import torch


def prepare_mppi_plot_data(debug_info: Dict[str, Any], policy_type: str = 'pi') -> Dict[str, Any]:
    """
    Prepare MPPI data for plotting using boom.common.debug.plot_trajs format.

    Args:
        debug_info: Debug information from MPPI planning
        policy_type: 'pi' or 'flow'

    Returns:
        Dictionary with plot data in plot_trajs format
    """
    mppi_key = f'{policy_type}_mppi'
    if mppi_key not in debug_info:
        return None

    mppi_data = debug_info[mppi_key]
    iterations = mppi_data.get('iterations', [])

    if not iterations:
        return None

    # Get initial actions and values
    init_actions = mppi_data.get('init_actions')  # [H, num_samples, A]
    if init_actions is None:
        return None

    # Convert to numpy if tensor
    if torch.is_tensor(init_actions):
        init_actions = init_actions.cpu().numpy()

    # Get values from first iteration
    first_iter = iterations[0]
    values = first_iter.get('values')  # [num_samples]
    if torch.is_tensor(values):
        values = values.cpu().numpy()

    # Determine number of guided vs random trajectories
    # Based on the MPPI implementation, samples are organized as:
    # [policy_guided, random_samples]
    num_samples = init_actions.shape[1]

    # For PI-MPPI: first num_pi_trajs are pi-guided, rest are random
    # For Flow-MPPI: first num_flow_trajs are flow-guided, rest are random
    # We need to determine this from the configuration
    # For now, assume first half are guided, second half are random
    num_guided = num_samples // 2
    num_random = num_samples - num_guided

    # Split actions and values
    guided_actions = init_actions[:, :num_guided, :]  # [H, num_guided, A]
    random_actions = init_actions[:, num_guided:, :]  # [H, num_random, A]

    guided_values = values[:num_guided] if values is not None else None
    random_values = values[num_guided:] if values is not None else None

    # Create plot data in plot_trajs format
    plot_data = {
        'num_pi': 0,
        'num_flow': 0,
        'num_random': num_random,
    }

    if policy_type == 'pi':
        plot_data['pi_actions'] = guided_actions
        plot_data['pi_values'] = guided_values
    else:  # flow
        plot_data['flow_actions'] = guided_actions
        plot_data['flow_values'] = guided_values

    plot_data['random_actions'] = random_actions
    plot_data['random_values'] = random_values

    return plot_data


def plot_mppi_2x2(debug_info: Dict[str, Any], env, save_path: str, policy_type: str = 'pi', real_trajectories=None):
    """
    Create 2 separate plots for multimodal MPPI visualization.

    Layout:
    - Plot 0: Combined figure with all proposals (top-left) + 5 elite MPPI iterations (2x3 grid)
    - Plot 1: Action distribution histogram (pi/flow)

    Args:
        debug_info: Debug information from MPPI planning (multimodal format)
        env: Environment instance for plotting context
        save_path: Base path for saving figures (will append _0, _1, etc.)
        policy_type: Unused (kept for compatibility)
        real_trajectories: List of real execution trajectories from evaluation (not used here)
    """
    if not debug_info:
        print(f"Warning: No debug data available")
        return

    # Import from boom.common.debug
    from boom.common.debug import _simulate_from_actions, _setup_goals_on_axis, _compute_colormap_norm

    # Get environment parameters
    step_size = getattr(env, 'step_size', 0.1)
    goal_radius = getattr(env, 'goal_radius', 0.1)

    # Create save directory
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # Generate plots
    _plot_combined_mppi_figure(debug_info, env, save_dir, step_size, goal_radius)  # Plot 0
    _plot_action_distribution(debug_info, env, save_dir)  # Plot 1

    print(f"Saved 2 MPPI plots to {save_dir}")


def _plot_combined_mppi_figure(debug_info: Dict[str, Any], env, save_dir: str, step_size: float, goal_radius: float):
    """Plot 0: Combined figure with all proposals (0,0) + 5 elite MPPI iterations in 2x3 grid."""
    from boom.common.debug import _simulate_from_actions, _setup_goals_on_axis, _compute_colormap_norm

    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ==================== Top-left: All proposals ====================
    ax0 = fig.add_subplot(gs[0, 0])
    _plot_all_proposals_on_axis(debug_info, ax0, step_size, goal_radius)

    # ==================== Plot elite MPPI iterations in remaining subplots ====================
    refinements = debug_info.get('refinements', [])

    # Grid positions for elite plots: [0,1], [0,2], [1,0], [1,1], [1,2]
    elite_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for i, pos in enumerate(elite_positions):
        if i < len(refinements):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            _plot_elite_mppi_on_axis(debug_info, ax, step_size, goal_radius, i)
        else:
            # Create empty subplot if no refinement data
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            ax.text(0.5, 0.5, f'No Elite {i+1} Data',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, style='italic', color='gray')

    # Save combined figure
    save_path = os.path.join(save_dir, 'mppi_0_combined.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def _plot_all_proposals_on_axis(debug_info: Dict[str, Any], ax, step_size: float, goal_radius: float):
    """Plot all proposals on given axis."""
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

    for i in range(num_samples):
        actions_rad = all_actions[:, i, :] * np.pi
        positions = _simulate_from_actions(actions_rad, step_size)
        value = all_values[i]
        traj_type = all_types[i] if i < len(all_types) else 'unknown'

        # Use value-based colormap
        if norm is not None and cmap is not None:
            color = cmap(norm(value))
        else:
            color = '#9467bd'

        # Highlight top-5 with thicker lines
        is_top_5 = i in top_5_indices
        linewidth = 3.5 if is_top_5 else 1.5
        alpha = 1.0 if is_top_5 else 0.6

        # Different line styles for different types
        if traj_type == 'random':
            linestyle = ':'
        elif traj_type == 'pi':
            linestyle = '--'
        else:  # flow
            linestyle = '-'

        ax.plot(positions[:, 0], positions[:, 1],
              color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth, zorder=6)

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

    ax.set_title(f'All Proposals (Top-5 Bold, N={num_samples})\nPI:{pi_count} Flow:{flow_count} Random:{random_count}',
                fontsize=14, fontweight='bold')


def _plot_elite_mppi_on_axis(debug_info: Dict[str, Any], ax, step_size: float, goal_radius: float, elite_idx: int):
    """Plot elite MPPI initial iteration on given axis."""
    from boom.common.debug import _simulate_from_actions, _setup_goals_on_axis, _compute_colormap_norm

    refinements = debug_info.get('refinements', [])

    if elite_idx >= len(refinements) or not refinements[elite_idx]:
        ax.text(0.5, 0.5, f'No Elite {elite_idx+1} Data',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, style='italic', color='gray')
        return

    elite_refinement = refinements[elite_idx]
    first_iter = elite_refinement[0] if elite_refinement else None

    if first_iter is None or 'init_actions' not in first_iter:
        ax.text(0.5, 0.5, f'No Elite {elite_idx+1} Data',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, style='italic', color='gray')
        return

    init_actions = first_iter['init_actions']  # [H, num_samples, A]
    init_values = first_iter['init_values']    # [num_samples]

    if torch.is_tensor(init_actions):
        init_actions = init_actions.cpu().numpy()
    if torch.is_tensor(init_values):
        init_values = init_values.cpu().numpy()

    H, num_samples, action_dim = init_actions.shape

    # Compute colormap
    norm, cmap = _compute_colormap_norm(init_values)

    _setup_goals_on_axis(ax, goal_radius, show_gradient=False)

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
              color=color, alpha=0.5, linewidth=1.2, zorder=6)

    # Plot mean trajectory
    mean = first_iter.get('mean')
    if mean is not None:
        if torch.is_tensor(mean):
            mean = mean.cpu().numpy()
        mean_actions_rad = mean * np.pi
        mean_positions = _simulate_from_actions(mean_actions_rad, step_size)
        ax.plot(mean_positions[:, 0], mean_positions[:, 1],
              color='red', linewidth=3.0, zorder=7, label='Mean')

    # Add colorbar
    if norm is not None and cmap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_ylabel('Value', fontsize=10)

    ax.set_title(f'Elite {elite_idx+1}: MPPI Init (N={num_samples})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)


def _plot_action_distribution(debug_info: Dict[str, Any], env, save_dir: str):
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
