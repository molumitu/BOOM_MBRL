"""
Generate Directed Random Trajectories for Four-Goal Navigation Environment

This script generates trajectories that are "directed random" - meaning each step
is biased towards the goal direction with some noise, rather than completely random.

Method:
- For each trajectory, randomly select a target goal
- At each step, compute optimal angle towards goal: θ_optimal = atan2(goal_y - y, goal_x - x)
- Add Gaussian noise: θ = θ_optimal + N(0, σ²)
- Execute action and continue until reaching goal or timeout

Output:
- List of trajectories with positions, actions, target goal, success status
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from typing import List, Dict, Tuple, Optional
import json
import os
from four_goal_nav_env import FourGoalNavigationEnv

os.makedirs("figures", exist_ok=True)


def compute_optimal_angle(current_pos: np.ndarray, goal_pos: np.ndarray) -> float:
    """
    Compute the optimal angle to move from current position towards goal.
    
    Args:
        current_pos: Current position (x, y)
        goal_pos: Goal position (x, y)
        
    Returns:
        Optimal angle in radians [-π, π]
    """
    dx = goal_pos[0] - current_pos[0]
    dy = goal_pos[1] - current_pos[1]
    return np.arctan2(dy, dx)


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-π, π]
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def generate_directed_random_trajectory(
    env: FourGoalNavigationEnv,
    target_goal: str,
    noise_std: float = np.pi / 6,
    max_steps: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None
) -> Dict:
    """
    Generate a single directed random trajectory towards a specific goal.
    
    Args:
        env: The navigation environment
        target_goal: Name of target goal ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        noise_std: Standard deviation of Gaussian noise added to optimal angle
        max_steps: Maximum steps (uses env.max_steps if None)
        rng: Random state for reproducibility
        
    Returns:
        Dictionary containing trajectory data
    """
    if rng is None:
        rng = np.random.RandomState()
    
    if max_steps is None:
        max_steps = env.max_steps
    
    # Reset environment
    obs, _ = env.reset()
    
    # Get goal position
    goal_pos = env.goals[target_goal]
    
    # Initialize storage
    positions = [obs.copy()]
    actions = []
    
    success = False
    steps = 0
    
    for step in range(max_steps):
        # Compute optimal angle towards goal
        optimal_angle = compute_optimal_angle(env.agent_pos, goal_pos)
        
        # Add Gaussian noise
        noise = rng.normal(0, noise_std)
        action_angle = normalize_angle(optimal_angle + noise)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(np.array([action_angle]))
        
        # Record data
        positions.append(obs.copy())
        actions.append(action_angle)
        steps = step + 1
        
        if terminated:
            success = True
            break
        elif truncated:
            break
    
    return {
        'trajectory': positions,
        'actions': actions,
        'target_goal': target_goal,
        'success': success,
        'steps': steps,
        'final_pos': positions[-1].copy()
    }


def generate_multiple_trajectories(
    env_params: Dict,
    num_trajectories: int = 100,
    noise_std: float = np.pi / 6,
    goal_selection: str = 'random',
    base_seed: int = 42,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate multiple directed random trajectories.
    
    Args:
        env_params: Environment parameters
        num_trajectories: Number of trajectories to generate
        noise_std: Standard deviation of Gaussian noise
        goal_selection: How to select goals ('random', 'uniform', or specific goal name)
        base_seed: Base random seed
        verbose: Print progress
        
    Returns:
        List of trajectory data dictionaries
    """
    rng = np.random.RandomState(base_seed)
    
    # Goal selection strategy
    goal_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    trajectories = []
    
    if verbose:
        print(f"Generating {num_trajectories} directed random trajectories...")
        print(f"  Noise std: {noise_std:.4f} rad ({np.degrees(noise_std):.1f}°)")
        print(f"  Goal selection: {goal_selection}")
    
    for i in range(num_trajectories):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{num_trajectories}")
        
        # Create fresh environment
        env = FourGoalNavigationEnv(**env_params)
        
        # Select target goal
        if goal_selection == 'random':
            target_goal = rng.choice(goal_names)
        elif goal_selection == 'uniform':
            # Ensure uniform distribution across goals
            target_goal = goal_names[i % 4]
        else:
            target_goal = goal_selection
        
        # Generate trajectory
        traj_data = generate_directed_random_trajectory(
            env=env,
            target_goal=target_goal,
            noise_std=noise_std,
            rng=rng
        )
        
        trajectories.append(traj_data)
    
    if verbose:
        # Print statistics
        success_count = sum(1 for t in trajectories if t['success'])
        print(f"\nGeneration complete!")
        print(f"  Success rate: {success_count}/{num_trajectories} ({100*success_count/num_trajectories:.1f}%)")
        
        # Count by goal
        goal_counts = {g: {'total': 0, 'success': 0} for g in goal_names}
        for t in trajectories:
            goal_counts[t['target_goal']]['total'] += 1
            if t['success']:
                goal_counts[t['target_goal']]['success'] += 1
        
        print(f"  By goal:")
        for g in goal_names:
            total = goal_counts[g]['total']
            success = goal_counts[g]['success']
            print(f"    {g}: {success}/{total} success")
    
    return trajectories


def plot_directed_random_trajectories(
    trajectories: List[Dict],
    env_params: Dict,
    save_path: str = "figures/four_goal_directed_trajs.png",
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot all directed random trajectories.
    
    Args:
        trajectories: List of trajectory data
        env_params: Environment parameters
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Environment setup
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw boundary
    boundary = Rectangle((-1, -1), 2, 2, linewidth=2,
                        edgecolor='black', facecolor='none')
    ax.add_patch(boundary)
    
    # Goal setup
    goal_radius = env_params.get('goal_radius', 0.1)
    goals = {
        'top_left': {'pos': np.array([-1.0, 1.0]), 'color': 'red'},
        'top_right': {'pos': np.array([1.0, 1.0]), 'color': 'green'},
        'bottom_left': {'pos': np.array([-1.0, -1.0]), 'color': 'blue'},
        'bottom_right': {'pos': np.array([1.0, -1.0]), 'color': 'orange'}
    }
    
    # Draw goals
    for goal_name, goal_info in goals.items():
        goal_circle = Circle(goal_info['pos'], goal_radius,
                            color=goal_info['color'], alpha=0.5,
                            label=goal_name.replace('_', ' ').title())
        ax.add_patch(goal_circle)
    
    # Draw start point
    ax.scatter(0, 0, c='purple', s=200, marker='o',
               edgecolors='black', linewidth=2, label='Start', zorder=10)
    
    # Separate trajectories by target goal
    trajectories_by_goal = {g: [] for g in goals.keys()}
    for t in trajectories:
        trajectories_by_goal[t['target_goal']].append(t)
    
    # Plot trajectories
    for goal_name, goal_info in goals.items():
        for t in trajectories_by_goal[goal_name]:
            traj_array = np.array(t['trajectory'])
            alpha = 0.7 if t['success'] else 0.3
            linestyle = '-' if t['success'] else '--'
            ax.plot(traj_array[:, 0], traj_array[:, 1],
                    color=goal_info['color'], alpha=alpha, linewidth=1.5,
                    linestyle=linestyle)
    
    # Statistics
    total = len(trajectories)
    success = sum(1 for t in trajectories if t['success'])
    success_rate = success / total if total > 0 else 0
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='purple', marker='o', markersize=10,
               label='Start', markeredgecolor='black', linestyle='none'),
    ]
    
    for goal_name, goal_info in goals.items():
        count = len(trajectories_by_goal[goal_name])
        success_count = sum(1 for t in trajectories_by_goal[goal_name] if t['success'])
        legend_elements.append(
            Line2D([0], [0], color=goal_info['color'], linewidth=1.5,
                   label=f'{goal_name.replace("_", " ").title()} ({success_count}/{count})')
        )
    
    ax.legend(handles=legend_elements, loc='center left', fontsize=10)
    
    ax.set_title(f'Directed Random Trajectories (N={total}, Success: {success_rate:.1%})',
                fontsize=12)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()


def plot_action_distribution(
    trajectories: List[Dict],
    save_path: str = "figures/four_goal_directed_action_distribution.png",
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Plot the distribution of first actions for each goal.
    
    Args:
        trajectories: List of trajectory data
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, subplot_kw={'projection': 'polar'})
    
    goal_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    goal_angles = {
        'top_left': 3*np.pi/4,
        'top_right': np.pi/4,
        'bottom_left': -3*np.pi/4,
        'bottom_right': -np.pi/4
    }
    
    for idx, goal_name in enumerate(goal_names):
        ax = axes[idx // 2, idx % 2]
        
        # Get first actions for this goal
        first_actions = [t['actions'][0] for t in trajectories 
                        if t['target_goal'] == goal_name and len(t['actions']) > 0]
        
        if len(first_actions) > 0:
            # Plot histogram
            ax.hist(first_actions, bins=36, density=True, alpha=0.7, color='steelblue')
            
            # Mark optimal angle
            optimal = goal_angles[goal_name]
            ax.axvline(optimal, color='red', linestyle='--', linewidth=2, 
                      label=f'Optimal: {np.degrees(optimal):.0f}°')
            
            # Statistics
            mean_angle = np.mean(first_actions)
            std_angle = np.std(first_actions)
            ax.axvline(mean_angle, color='green', linestyle='-', linewidth=2,
                      label=f'Mean: {np.degrees(mean_angle):.0f}°')
        
        ax.set_title(f'{goal_name.replace("_", " ").title()}\n(N={len(first_actions)})', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Distribution of First Actions by Target Goal', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Action distribution saved to {save_path}")
    plt.show()


def save_trajectories(
    trajectories: List[Dict],
    save_path: str = "results/directed_trajectories.npy"
) -> None:
    """
    Save trajectories to numpy file.
    
    Args:
        trajectories: List of trajectory data
        save_path: Path to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to serializable format
    save_data = []
    for t in trajectories:
        save_data.append({
            'trajectory': np.array(t['trajectory']),
            'actions': np.array(t['actions']),
            'target_goal': t['target_goal'],
            'success': t['success'],
            'steps': t['steps'],
            'final_pos': t['final_pos']
        })
    
    np.save(save_path, save_data, allow_pickle=True)
    print(f"Trajectories saved to {save_path}")


def main():
    """Main function to generate and visualize directed random trajectories."""
    print("=" * 70)
    print("Directed Random Trajectory Generation")
    print("=" * 70)
    
    # Environment parameters (same as four_goal_nav_env.py)
    ENV_PARAMS = {
        'step_size': 0.10,
        'goal_radius': 0.1,
        'max_steps': 40,
        'step_penalty': -0.01
    }
    
    # Generation parameters
    NUM_TRAJECTORIES = 100
    NOISE_STD = np.pi / 3
    BASE_SEED = 42
    
    print(f"\nEnvironment Parameters:")
    print(f"  Step size: {ENV_PARAMS['step_size']}")
    print(f"  Goal radius: {ENV_PARAMS['goal_radius']}")
    print(f"  Max steps: {ENV_PARAMS['max_steps']}")
    
    print(f"\nGeneration Parameters:")
    print(f"  Number of trajectories: {NUM_TRAJECTORIES}")
    print(f"  Noise std: {NOISE_STD:.4f} rad ({np.degrees(NOISE_STD):.1f}°)")
    print(f"  Base seed: {BASE_SEED}")
    
    # Generate trajectories
    trajectories = generate_multiple_trajectories(
        env_params=ENV_PARAMS,
        num_trajectories=NUM_TRAJECTORIES,
        noise_std=NOISE_STD,
        goal_selection='uniform',  # Ensure equal distribution across goals
        base_seed=BASE_SEED,
        verbose=True
    )
    
    # Plot trajectories
    print("\nPlotting trajectories...")
    plot_directed_random_trajectories(
        trajectories=trajectories,
        env_params=ENV_PARAMS,
        save_path="figures/four_goal_directed_trajs.png"
    )
    
    # Plot action distribution
    print("\nPlotting action distribution...")
    plot_action_distribution(
        trajectories=trajectories,
        save_path="figures/four_goal_directed_action_distribution.png"
    )
    
    # Save trajectories
    print("\nSaving trajectories...")
    save_trajectories(
        trajectories=trajectories,
        save_path="results/directed_trajectories.npy"
    )
    
    # Print sample trajectory
    print("\n" + "=" * 70)
    print("Sample Trajectory (first one):")
    print("=" * 70)
    t = trajectories[0]
    print(f"  Target goal: {t['target_goal']}")
    print(f"  Success: {t['success']}")
    print(f"  Steps: {t['steps']}")
    print(f"  First 5 positions: {np.array(t['trajectory'][:5]).round(3)}")
    print(f"  First 5 actions (rad): {np.array(t['actions'][:5]).round(3)}")
    print(f"  First 5 actions (deg): {np.degrees(t['actions'][:5]).round(1)}")


if __name__ == "__main__":
    main()