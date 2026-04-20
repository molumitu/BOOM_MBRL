import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_mppi_debug(actions, values, mean, std, iter_idx, mppi_debug_dir,
                    num_pi=0, num_pi_derived=0, num_flow=0, num_flow_derived=0, num_random=None):
    """
    Plot MPPI action distributions with value overlay.

    Args:
        actions: [horizon, num_samples, action_dim]
        values: [num_samples] - value for each trajectory
        mean: [horizon, action_dim] - mean action at each timestep
        std: [horizon, action_dim] - std at each timestep
        iter_idx: current iteration index
        mppi_debug_dir: directory to save debug plots
        num_pi: number of pi-guided trajectories
        num_pi_derived: number of pi-derived trajectories
        num_flow: number of flow-guided trajectories
        num_flow_derived: number of flow-derived trajectories
        num_random: number of random trajectories (None to auto-detect)
    """
    if mppi_debug_dir is None:
        return

    # Convert to numpy
    actions_np = actions.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()
    mean_np = mean.detach().cpu().numpy()
    std_np = std.detach().cpu().numpy()

    horizon, num_samples, action_dim = actions_np.shape

    # Auto-detect num_random if not provided
    if num_random is None:
        num_random = num_samples - num_pi - num_pi_derived - num_flow - num_flow_derived

    # Create indices for different trajectory types
    # Layout: [pi_base | pi_derived | flow_base | flow_derived | random]
    pi_indices = slice(0, num_pi) if num_pi > 0 else None
    pi_derived_indices = slice(num_pi, num_pi + num_pi_derived) if num_pi_derived > 0 else None
    flow_indices = slice(num_pi + num_pi_derived, num_pi + num_pi_derived + num_flow) if num_flow > 0 else None
    flow_derived_indices = slice(num_pi + num_pi_derived + num_flow,
                                num_pi + num_pi_derived + num_flow + num_flow_derived) if num_flow_derived > 0 else None
    random_indices = slice(num_pi + num_pi_derived + num_flow + num_flow_derived, num_samples) if num_random > 0 else None

    # Create directory for action distribution plots
    act_dir = os.path.join(mppi_debug_dir, 'act')
    os.makedirs(act_dir, exist_ok=True)


    # ========== Plot: Action Distributions for All Timesteps ==========
    # Create subplots: action_dim rows, horizon columns
    fig, axes = plt.subplots(action_dim, horizon, figsize=(4*horizon, 4*action_dim))
    # fig, axes = plt.subplots(horizon, action_dim, figsize=(4*action_dim, 4*horizon))
    if action_dim == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    if horizon == 1:
        axes = axes.reshape(-1, 1)  # Ensure 2D array

    for dim in range(action_dim):
        for t in range(horizon):
            ax = axes[dim, t]

            # Get actions for this dimension and timestep
            actions_at_t = actions_np[t, :, dim]

            # Plot stacked histograms for different trajectory types
            if pi_indices is not None:
                pi_actions = actions_np[t, pi_indices, dim]
                ax.hist(pi_actions, bins=50, alpha=0.5, color='#9467bd', edgecolor='black',
                       label=f'Pi (n={num_pi})' if t == 0 else '')

            if pi_derived_indices is not None:
                pi_derived_actions = actions_np[t, pi_derived_indices, dim]
                ax.hist(pi_derived_actions, bins=50, alpha=0.5, color='#d62728', edgecolor='black',
                       label=f'Pi-derived (n={num_pi_derived})' if t == 0 else '')

            if flow_indices is not None:
                flow_actions = actions_np[t, flow_indices, dim]
                ax.hist(flow_actions, bins=50, alpha=0.5, color='#1f77b4', edgecolor='black',
                       label=f'Flow (n={num_flow})' if t == 0 else '')

            if flow_derived_indices is not None:
                flow_derived_actions = actions_np[t, flow_derived_indices, dim]
                ax.hist(flow_derived_actions, bins=50, alpha=0.5, color='#ff7f0e', edgecolor='black',
                       label=f'Flow-derived (n={num_flow_derived})' if t == 0 else '')

            if random_indices is not None:
                random_actions = actions_np[t, random_indices, dim]
                ax.hist(random_actions, bins=50, alpha=0.4, color='#7f7f7f', edgecolor='black',
                       label=f'Random (n={num_random})' if t == 0 else '')

            # Plot mean and std as vertical lines
            ax.axvline(mean_np[t, dim], color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_np[t, dim]:.3f}')
            ax.axvline(mean_np[t, dim] + std_np[t, dim], color='orange', linestyle=':', linewidth=1.5, label=f'+{std_np[t, dim]:.2f} Std')
            ax.axvline(mean_np[t, dim] - std_np[t, dim], color='orange', linestyle=':', linewidth=1.5, label=f'-{std_np[t, dim]:.2f} Std')

            # Add value overlay only for first timestep
            if t == 0:
                # Create twin axis for values
                ax2 = ax.twinx()

                # Plot values for different trajectory types
                if pi_indices is not None:
                    pi_actions = actions_np[t, pi_indices, dim]
                    pi_values = values_np[pi_indices]
                    sort_idx = np.argsort(pi_actions)
                    ax2.plot(pi_actions[sort_idx], pi_values[sort_idx], 'o-',
                            color='#9467bd', alpha=0.8, linewidth=1, markersize=3,
                            label='Pi Value' if num_pi > 0 else '')

                if pi_derived_indices is not None:
                    pi_derived_actions = actions_np[t, pi_derived_indices, dim]
                    pi_derived_values = values_np[pi_derived_indices]
                    sort_idx = np.argsort(pi_derived_actions)
                    ax2.plot(pi_derived_actions[sort_idx], pi_derived_values[sort_idx], 'o-',
                            color='#d62728', alpha=0.8, linewidth=1, markersize=3,
                            label='Pi-derived Value' if num_pi_derived > 0 else '')

                if flow_indices is not None:
                    flow_actions = actions_np[t, flow_indices, dim]
                    flow_values = values_np[flow_indices]
                    sort_idx = np.argsort(flow_actions)
                    ax2.plot(flow_actions[sort_idx], flow_values[sort_idx], 'o-',
                            color='#1f77b4', alpha=0.8, linewidth=1, markersize=3,
                            label='Flow Value' if num_flow > 0 else '')

                if flow_derived_indices is not None:
                    flow_derived_actions = actions_np[t, flow_derived_indices, dim]
                    flow_derived_values = values_np[flow_derived_indices]
                    sort_idx = np.argsort(flow_derived_actions)
                    ax2.plot(flow_derived_actions[sort_idx], flow_derived_values[sort_idx], 'o-',
                            color='#ff7f0e', alpha=0.8, linewidth=1, markersize=3,
                            label='Flow-derived Value' if num_flow_derived > 0 else '')

                if random_indices is not None:
                    random_actions = actions_np[t, random_indices, dim]
                    random_values = values_np[random_indices]
                    sort_idx = np.argsort(random_actions)
                    ax2.plot(random_actions[sort_idx], random_values[sort_idx], 'o-',
                            color='#7f7f7f', alpha=0.6, linewidth=1, markersize=3,
                            label='Random Value' if num_random > 0 else '')

                ax2.set_ylabel('Trajectory Value', fontsize=9, color='darkgreen')
                ax2.tick_params(axis='y', labelcolor='darkgreen', labelsize=8)

            # Labels and styling
            ax.set_xlabel(f'Action Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)

            # Add title for each subplot
            if dim == 0:
                ax.set_title(f'Timestep {t}', fontsize=10, fontweight='bold')
            if t == 0:
                ax.set_ylabel(f'Dim {dim}\n Count', fontsize=9)

            # Add grid
            ax.grid(True, alpha=0.3, axis='x')

            # Add statistics text box (only show简要stats)
            stats_text = f'μ={mean_np[t, dim]:.2f}\nσ={std_np[t, dim]:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)

            # Legend only for first timestep to avoid clutter
            if t == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                try:
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=6)
                except:
                    ax.legend(lines1, labels1, loc='upper right', fontsize=7)

    fig.suptitle(f'MPPI Action Distributions - Iter {iter_idx}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(act_dir, f'iter{iter_idx:04d}.png'), dpi=150)
    plt.close()


# ============================================================================
# Unified Trajectory Plotting Function
# ============================================================================

# Goal configuration for four-goal navigation environment
GOAL_CONFIG = {
    'top_left': {
        'pos': np.array([-1.0, 1.0]),
        'color': '#9467bd',  # purple
        'label': 'Top Left'
    },
    'top_right': {
        'pos': np.array([1.0, 1.0]),
        'color': '#1f77b4',  # blue
        'label': 'Top Right'
    },
    'bottom_left': {
        'pos': np.array([-1.0, -1.0]),
        'color': '#2ca02c',  # green
        'label': 'Bottom Left'
    },
    'bottom_right': {
        'pos': np.array([1.0, -1.0]),
        'color': '#ff7f0e',  # orange
        'label': 'Bottom Right'
    }
}


def _detect_traj_type(trajectories):
    """Auto-detect trajectory type from data structure.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        str: 'real' or 'mppi'
        
    Raises:
        ValueError: If trajectory format is not recognized
    """
    if len(trajectories) == 0:
        raise ValueError("Empty trajectory list")
    
    first_traj = trajectories[0]
    
    # Real trajectories have 'trajectory' key with position data
    if 'trajectory' in first_traj:
        return 'real'
    
    # MPPI trajectories have action keys
    elif any(key in first_traj for key in ['pi_actions', 'flow_actions', 'random_actions']):
        return 'mppi'
    
    else:
        raise ValueError("Unknown trajectory format. Expected 'trajectory' key for real trajectories "
                        "or action keys ('pi_actions', 'flow_actions', 'random_actions') for MPPI trajectories.")


def _simulate_from_actions(actions, step_size):
    """Simulate position trajectory from action sequence.
    
    Args:
        actions: (H, action_dim) array of actions (angles in radians)
        step_size: Step size for each action
        
    Returns:
        positions: (H+1, 2) array of positions
    """
    from matplotlib.patches import Circle
    
    pos = np.array([0.0, 0.0])
    positions = [pos.copy()]
    
    for t in range(actions.shape[0]):
        theta = actions[t, 0]
        dx = step_size * np.cos(theta)
        dy = step_size * np.sin(theta)
        pos = pos + np.array([dx, dy])
        pos = np.clip(pos, -1.0, 1.0)  # Clip to boundaries
        positions.append(pos.copy())
    
    return np.array(positions)


def _setup_goals_on_axis(ax, goal_radius, show_gradient=True):
    """Draw boundary, goals, and origin marker on axis.
    
    Args:
        ax: Matplotlib axis
        goal_radius: Radius for drawing goal circles
        show_gradient: If True, draw gradient effect (outer glow + inner highlight)
    """
    from matplotlib.patches import Rectangle, Circle
    
    # Remove axes
    ax.set_axis_off()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    
    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # White background
    ax.set_facecolor('white')
    
    # Draw boundary
    boundary = Rectangle((-1, -1), 2, 2, linewidth=2,
                        edgecolor='#414141', facecolor='none', zorder=1)
    ax.add_patch(boundary)
    
    # Draw goals
    for goal_name, goal_info in GOAL_CONFIG.items():
        if show_gradient:
            # Outer glow
            goal_circle_outer = Circle(goal_info['pos'], goal_radius * 1.5,
                                      color=goal_info['color'], alpha=0.15, zorder=2)
            ax.add_patch(goal_circle_outer)
            # Inner highlight
            goal_circle_inner = Circle(goal_info['pos'], goal_radius * 0.5,
                                      color=goal_info['color'], alpha=0.7, zorder=3)
            ax.add_patch(goal_circle_inner)
        
        # Main goal
        goal_circle = Circle(goal_info['pos'], goal_radius,
                           color=goal_info['color'], alpha=0.4, zorder=2)
        ax.add_patch(goal_circle)
    
    # Draw origin
    origin_circle = Circle((0, 0), 0.05, color='red', alpha=0.8, zorder=7)
    ax.add_patch(origin_circle)


def _compute_colormap_norm(all_values):
    """Compute global normalization for value-based colormap.
    
    Args:
        all_values: List/array of all trajectory values
        
    Returns:
        norm: plt.Normalize object (or None)
        cmap: matplotlib colormap (or None)
    """
    if len(all_values) == 0:
        return None, None
    
    all_values = np.array(all_values)
    vmin, vmax = all_values.min(), all_values.max()
    
    if vmax - vmin > 1e-6:
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.get_cmap('viridis')
        return norm, cmap
    else:
        return None, None


def _process_real_trajectories(trajectories):
    """Extract and normalize real trajectory arrays.

    Args:
        trajectories: List of dicts with 'trajectory', 'reached_goal', 'reward'

    Returns:
        List of tuples: (positions_array, traj_type, value)
    """
    processed = []

    for traj_data in trajectories:
        traj = traj_data['trajectory']
        reached_goal = traj_data.get('reached_goal')
        reward = traj_data.get('reward', 0)

        traj_array = np.array(traj)

        # Handle different trajectory shapes
        if traj_array.ndim == 3:  # (T, 1, 2) or (T, batch, 2)
            traj_array = traj_array.reshape(-1, 2)
        elif traj_array.ndim == 2 and traj_array.shape[1] == 2:  # (T, 2) - already correct
            pass
        elif traj_array.ndim == 2 and traj_array.shape[0] == 2:  # (2, T) - need transpose
            traj_array = traj_array.T

        # Determine trajectory type (use goal name or 'failed')
        if reached_goal is not None:
            traj_type = reached_goal  # 'top_left', 'top_right', etc.
        else:
            traj_type = 'failed'

        # Use reward as the value
        processed.append((traj_array, traj_type, reward))

    return processed


def _process_mppi_trajectories(mppi_trajs, step_size):
    """Simulate MPPI trajectories from action sequences.

    Args:
        mppi_trajs: List of dicts with pi/flow/random actions and values (including derived)
        step_size: Step size for simulation

    Returns:
        List of tuples: (positions_array, traj_type, value)
    """
    processed = []

    for init_info in mppi_trajs:
        # Process pi trajectories
        if init_info.get('pi_actions') is not None and init_info.get('num_pi', 0) > 0:
            pi_actions = init_info['pi_actions'] * np.pi  # [H, num_pi, A]
            num_pi = init_info['num_pi']
            pi_values = init_info.get('pi_values')

            if pi_values is not None:
                pi_values_np = pi_values.cpu().numpy() if hasattr(pi_values, 'cpu') else pi_values
            else:
                pi_values_np = None

            for traj_idx in range(num_pi):
                positions = _simulate_from_actions(pi_actions[:, traj_idx, :], step_size)
                value = pi_values_np[traj_idx] if pi_values_np is not None else None
                processed.append((positions, 'pi', value))

        # Process pi-derived trajectories
        if init_info.get('pi_derived_actions') is not None and init_info.get('num_pi_derived', 0) > 0:
            pi_derived_actions = init_info['pi_derived_actions'] * np.pi  # [H, num_pi_derived, A]
            num_pi_derived = init_info['num_pi_derived']
            pi_derived_values = init_info.get('pi_derived_values')

            if pi_derived_values is not None:
                pi_derived_values_np = pi_derived_values.cpu().numpy() if hasattr(pi_derived_values, 'cpu') else pi_derived_values
            else:
                pi_derived_values_np = None

            for traj_idx in range(num_pi_derived):
                positions = _simulate_from_actions(pi_derived_actions[:, traj_idx, :], step_size)
                value = pi_derived_values_np[traj_idx] if pi_derived_values_np is not None else None
                processed.append((positions, 'pi_derived', value))

        # Process flow trajectories
        if init_info.get('flow_actions') is not None and init_info.get('num_flow', 0) > 0:
            flow_actions = init_info['flow_actions'] * np.pi  # [H, num_flow, A]
            num_flow = init_info['num_flow']
            flow_values = init_info.get('flow_values')

            if flow_values is not None:
                flow_values_np = flow_values.cpu().numpy() if hasattr(flow_values, 'cpu') else flow_values
            else:
                flow_values_np = None

            for traj_idx in range(num_flow):
                positions = _simulate_from_actions(flow_actions[:, traj_idx, :], step_size)
                value = flow_values_np[traj_idx] if flow_values_np is not None else None
                processed.append((positions, 'flow', value))

        # Process flow-derived trajectories
        if init_info.get('flow_derived_actions') is not None and init_info.get('num_flow_derived', 0) > 0:
            flow_derived_actions = init_info['flow_derived_actions'] * np.pi  # [H, num_flow_derived, A]
            num_flow_derived = init_info['num_flow_derived']
            flow_derived_values = init_info.get('flow_derived_values')

            if flow_derived_values is not None:
                flow_derived_values_np = flow_derived_values.cpu().numpy() if hasattr(flow_derived_values, 'cpu') else flow_derived_values
            else:
                flow_derived_values_np = None

            for traj_idx in range(num_flow_derived):
                positions = _simulate_from_actions(flow_derived_actions[:, traj_idx, :], step_size)
                value = flow_derived_values_np[traj_idx] if flow_derived_values_np is not None else None
                processed.append((positions, 'flow_derived', value))

        # Process random trajectories
        if init_info.get('random_actions') is not None and init_info.get('num_random', 0) > 0:
            random_actions = init_info['random_actions'] * np.pi  # [H, num_random, A]
            num_random = min(init_info['num_random'], 100)  # Limit for clarity
            random_values = init_info.get('random_values')

            if random_values is not None:
                random_values_np = random_values.cpu().numpy() if hasattr(random_values, 'cpu') else random_values
            else:
                random_values_np = None

            for traj_idx in range(num_random):
                positions = _simulate_from_actions(random_actions[:, traj_idx, :], step_size)
                value = random_values_np[traj_idx] if random_values_np is not None else None
                processed.append((positions, 'random', value))

    return processed


def plot_trajs(
    trajectories,
    step,
    save_dir,
    traj_type='auto',
    layout='auto',
    use_value_cmap=True,
    step_size=0.1,
    goal_radius=0.1,
    max_episodes=4,
    show_stats=True,
    dpi=200
):
    """Unified trajectory plotting function.
    
    Handles both real evaluation trajectories and MPPI planning trajectories
    with automatic type detection, flexible layouts, and optional value-based
    colormap.
    
    Primary logic follows plot_mppi_init_trajs approach:
    - If trajectory values are provided, uses value-based colormap (viridis)
    - Line styles: flow=solid (-), pi=dashed (--), random=dotted (:)
    - All trajectories: linewidth=1.5, alpha=0.7
    
    Args:
        trajectories: List of trajectory dictionaries. Format depends on traj_type:
            - Real: [{'trajectory': [(x,y),...], 'reached_goal': str, 'reward': float}, ...]
            - MPPI: [{'pi_actions': [H,num_pi,A], 'pi_values': [num_pi],
                      'flow_actions': [H,num_flow,A], 'flow_values': [num_flow],
                      'random_actions': [H,num_random,A], 'random_values': [num_random],
                      'num_pi': int, 'num_flow': int, 'num_random': int}, ...]
        step (int): Current training step (for filename and title)
        save_dir (str): Directory to save the plot
        traj_type (str): 'real', 'mppi', or 'auto' to detect from data structure
        layout (str): 'single', 'multi', or 'auto' to decide based on data
        use_value_cmap (bool): If True, apply value-based colormap when values available
        step_size (float): Step size for simulating from action sequences
        goal_radius (float): Radius for drawing goal circles
        max_episodes (int): Maximum episodes to show in multi-episode layout
        show_stats (bool): For real trajectories, display success rate and avg reward
        dpi (int): Figure DPI
        
    Returns:
        None (saves figure to disk)
    """
    if len(trajectories) == 0:
        return
    
    # Auto-detect trajectory type if needed
    if traj_type == 'auto':
        traj_type = _detect_traj_type(trajectories)
    
    # Determine layout
    if layout == 'auto':
        if traj_type == 'mppi' and len(trajectories) > 1:
            layout = 'multi'
        else:
            layout = 'single'
    
    # Process trajectories
    if traj_type == 'real':
        processed = _process_real_trajectories(trajectories)
    else:  # mppi
        processed = _process_mppi_trajectories(trajectories, step_size)
    
    # Collect all values for colormap (across all episodes)
    all_values = []
    for positions, traj_type_label, value in processed:
        if value is not None:
            all_values.append(value)
    
    # Compute colormap normalization
    if use_value_cmap and len(all_values) > 0:
        global_norm, global_cmap = _compute_colormap_norm(all_values)
    else:
        global_norm, global_cmap = None, None
    
    # Create figure
    if layout == 'multi':
        num_eps = min(len(trajectories), max_episodes)
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        # Plot each episode
        for ep_idx in range(num_eps):
            ax = axes[ep_idx]
            
            # Setup goals
            _setup_goals_on_axis(ax, goal_radius, show_gradient=False)
            
            # Get processed trajectories for this episode
            if traj_type == 'mppi':
                episode_processed = _process_mppi_trajectories([trajectories[ep_idx]], step_size)
            else:
                episode_processed = _process_real_trajectories([trajectories[ep_idx]])
            
            # Plot trajectories
            _plot_trajectories_on_axis(ax, episode_processed, global_norm, global_cmap, 
                                       use_value_cmap, traj_type)
            
            # Add episode title
            ax.set_title(f'Episode {ep_idx + 1}', fontsize=12, fontweight='bold')
        
        # Add colorbar if using colormap
        if use_value_cmap and len(all_values) > 0:
            _add_colorbar(fig, global_norm, global_cmap)
        
        plt.suptitle(f'MPPI Initial Trajectories - Step {step}', fontsize=14, fontweight='bold', y=0.995)
    
    else:  # single layout
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Setup goals
        _setup_goals_on_axis(ax, goal_radius, show_gradient=True)
        
        # Plot all trajectories
        _plot_trajectories_on_axis(ax, processed, global_norm, global_cmap, 
                                   use_value_cmap, traj_type)
        
        # Add statistics for real trajectories
        if traj_type == 'real' and show_stats:
            _add_statistics(ax, trajectories, step)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    if layout == 'multi':
        filename = f'mppi_init_trajs_step_{step}.png'
    else:
        filename = f'trajectories_step_{step}.png'
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Trajectory plot saved to: {save_path}")


def _plot_trajectories_on_axis(ax, processed, norm, cmap, use_value_cmap, traj_type):
    """Plot processed trajectories on a given axis.

    Args:
        ax: Matplotlib axis
        processed: List of (positions, traj_type, value/reward) tuples
        norm: Normalize object for colormap
        cmap: Colormap object
        use_value_cmap: Whether to use value-based colormap
        traj_type: 'real' or 'mppi'
    """
    # Track which trajectory types we've seen to avoid duplicate labels
    seen_types = set()

    for positions, traj_type_label, value in processed:
        # Determine color and line style
        if use_value_cmap and norm is not None and cmap is not None and value is not None:
            # Only use value-based colormap for non-derived trajectories
            if traj_type_label not in ['pi_derived', 'flow_derived']:
                color = cmap(norm(value))
            else:
                # Use fixed colors for derived trajectories (not value-based)
                if traj_type_label == 'pi_derived':
                    color = '#d62728'  # red (distinct from pi purple)
                elif traj_type_label == 'flow_derived':
                    color = '#ff7f0e'  # orange (distinct from flow blue)
                else:
                    color = 'gray'
        elif traj_type == 'real':
            # Use goal-based colors for real trajectories
            if traj_type_label in GOAL_CONFIG:
                color = GOAL_CONFIG[traj_type_label]['color']
            else:  # failed
                color = 'gray'
        else:
            # Default colors for MPPI trajectories without values
            if traj_type_label == 'pi':
                color = '#9467bd'  # purple
            elif traj_type_label == 'pi_derived':
                color = '#d62728'  # red (distinct from pi purple)
            elif traj_type_label == 'flow':
                color = '#1f77b4'  # blue
            elif traj_type_label == 'flow_derived':
                color = '#ff7f0e'  # orange (distinct from flow blue)
            else:  # random
                color = '#7f7f7f'  # gray

        # Determine line style
        if traj_type == 'mppi':
            if traj_type_label == 'flow':
                line_style = '-'
            elif traj_type_label == 'flow_derived':
                line_style = '-'
            elif traj_type_label == 'pi':
                line_style = '--'
            elif traj_type_label == 'pi_derived':
                line_style = '--'
            else:  # random
                line_style = ':'
            alpha = 0.7
            linewidth = 1.5
        elif traj_type == 'real':
            if traj_type_label == 'failed':
                line_style = '--'
                alpha = 0.35
                linewidth = 1.5
            else:  # successful
                line_style = '-'
                alpha = 0.4
                linewidth = 2
        else:
            line_style = '-'
            alpha = 0.7
            linewidth = 1.5

        # Plot trajectory with label (only first time we see each type)
        label = None
        if traj_type_label not in seen_types:
            if traj_type == 'mppi':
                if traj_type_label == 'flow':
                    label = 'Flow trajectories'
                elif traj_type_label == 'flow_derived':
                    label = 'Flow-derived trajectories'
                elif traj_type_label == 'pi':
                    label = 'Pi trajectories'
                elif traj_type_label == 'pi_derived':
                    label = 'Pi-derived trajectories'
                else:
                    label = 'Random trajectories'
            elif traj_type == 'real':
                if traj_type_label == 'failed':
                    label = 'Failed'
                else:
                    label = GOAL_CONFIG[traj_type_label]['label']

            # Mark this type as seen
            if label is not None:
                seen_types.add(traj_type_label)

        ax.plot(positions[:, 0], positions[:, 1],
               color=color, alpha=alpha, linewidth=linewidth,
               linestyle=line_style, label=label, zorder=6)

    # Add legend if we have trajectories
    if len(seen_types) > 0:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)


def _add_colorbar(fig, norm, cmap):
    """Add colorbar to figure.
    
    Args:
        fig: Matplotlib figure
        norm: Normalize object
        cmap: Colormap object
    """
    from matplotlib.cm import ScalarMappable
    
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Trajectory Value (Cumulative Reward)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)


def _add_statistics(ax, trajectories, step):
    """Add statistics text box to axis.
    
    Args:
        ax: Matplotlib axis
        trajectories: List of trajectory dictionaries
        step: Current training step
    """
    num_trajectories = len(trajectories)
    
    # Count successful trajectories
    num_success = 0
    total_reward = 0
    
    for traj_data in trajectories:
        if traj_data.get('reached_goal') is not None:
            num_success += 1
        total_reward += traj_data.get('reward', 0)
    
    success_rate = num_success / num_trajectories if num_trajectories > 0 else 0
    avg_reward = total_reward / num_trajectories if num_trajectories > 0 else 0
    
    # Create stats text
    stats_text = (f"Step: {step}\n"
                 f"N: {num_trajectories}\n"
                 f"Success: {success_rate:.0%}\n"
                 f"Avg Reward: {avg_reward:.2f}")
    
    # Add stats text in corner
    ax.text(-1.15, 1.1, stats_text, fontsize=11, ha='left', va='center',
           color='#555555', fontfamily='monospace', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='#DDDDDD', alpha=0.9, linewidth=1.5), zorder=15)
