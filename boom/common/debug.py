import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
        assert traj_array.shape[1] == 2

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
        mppi_trajs: List of dicts with pi/flow/random actions and values
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
    save_dir,
    traj_type,
    filename = 'trajectories_step.png',
    use_value_cmap=True,
    step_size=0.1,
    goal_radius=0.1,
    max_episodes=4,
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
        save_dir (str): Directory to save the plot
        traj_type (str): 'real', 'mppi', or 'auto' to detect from data structure
        use_value_cmap (bool): If True, apply value-based colormap when values available
        step_size (float): Step size for simulating from action sequences
        goal_radius (float): Radius for drawing goal circles
        max_episodes (int): Maximum episodes to show in multi-episode layout
        dpi (int): Figure DPI
        
    Returns:
        None (saves figure to disk)
    """
    if len(trajectories) == 0:
        return
    
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
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Setup goals
    _setup_goals_on_axis(ax, goal_radius, show_gradient=True)
    
    # Plot all trajectories
    _plot_trajectories_on_axis(ax, processed, global_norm, global_cmap, 
                                use_value_cmap, traj_type)
        
    
    plt.tight_layout()
    
    # Save figure
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
            color = cmap(norm(value))
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
            elif traj_type_label == 'flow':
                color = '#1f77b4'  # blue
            else:  # random
                color = '#7f7f7f'  # gray

        # Determine line style
        if traj_type == 'mppi':
            if traj_type_label == 'flow':
                line_style = '-'
            elif traj_type_label == 'pi':
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
                elif traj_type_label == 'pi':
                    label = 'Pi trajectories'
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


def plot_real_trajectories(trajectories, save_dir, filename='real_trajectories.png',
                           step_size=0.1, goal_radius=0.1, dpi=200):
    """Plot real evaluation trajectories.

    Args:
        trajectories: List of trajectory dicts with 'trajectory', 'reached_goal', 'reward'
        save_dir: Directory to save the plot
        filename: Output filename
        step_size: Step size for visualization
        goal_radius: Goal radius for visualization
        dpi: Figure DPI
    """
    plot_trajs(
        trajectories=trajectories,
        save_dir=save_dir,
        traj_type='real',
        filename=filename,
        use_value_cmap=False,
        step_size=step_size,
        goal_radius=goal_radius,
        dpi=dpi
    )


def plot_mppi_trajectories_on_axis(ax, debug_info, step_size=0.1, goal_radius=0.1):
    """Plot MPPI trajectories on a given axis.

    Args:
        ax: Matplotlib axis
        actions: Action sequences [H, num_samples, A]
        values: Corresponding values [num_samples]
        step_size: Step size for trajectory simulation
        goal_radius: Goal radius for visualization
    """

    # Create MPPI trajectory format
    traj = {
        'flow_actions': None,
        'flow_values': None,
        'num_flow': None,
        'pi_actions': None,
        'pi_values': None,
        'num_pi': None,
        'random_actions': None,
        'random_values': None,
        'num_random': None,
    }
    if 'flow_candidates' in debug_info:
        traj.update({
            'flow_actions': debug_info['flow_candidates']['actions'],
            'flow_values': debug_info['flow_candidates']['values'],
            'num_flow': debug_info['flow_candidates']['values'].shape[0],
        })
    if 'pi_candidates' in debug_info:
        traj.update({
            'pi_actions': debug_info['pi_candidates']['actions'],
            'pi_values': debug_info['pi_candidates']['values'],
            'num_pi': debug_info['pi_candidates']['values'].shape[0],
        })
    if 'random_candidates' in debug_info:
        traj.update({
            'random_actions': debug_info['random_candidates']['actions'],
            'random_values': debug_info['random_candidates']['values'],
            'num_random': debug_info['random_candidates']['values'].shape[0],
        })


    # Process trajectories
    processed = _process_mppi_trajectories([traj], step_size)

    # Compute colormap normalization
    value_tensors = []
    for key in ['flow_values', 'pi_values', 'random_values']:
        if key in traj and traj[key] is not None:
            value_tensors.append(traj[key])

    if len(value_tensors) > 0:
        all_values = torch.cat(value_tensors).cpu().numpy()
    else:
        all_values = np.array([])
    norm, cmap = _compute_colormap_norm(all_values)

    # Setup goals
    _setup_goals_on_axis(ax, goal_radius, show_gradient=True)

    # Plot trajectories
    _plot_trajectories_on_axis(ax, processed, norm, cmap, use_value_cmap=True, traj_type='mppi')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
