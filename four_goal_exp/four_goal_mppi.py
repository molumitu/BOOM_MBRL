"""
MPPI Training and Visualization for Four-Goal Navigation Environment

This script implements Model Predictive Path Integral (MPPI) optimization
with warm-start initialization from filtered directed random trajectories
for the four-goal navigation environment.

Key features:
- Two-stage strategy: filtered random trajectory initialization + MPPI optimization
- Proper MPPI implementation with weighted update
- Separate environments for each rollout to avoid state contamination
- Clear visualization of trajectories to four goals
- First action distribution analysis with KDE fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import os
from typing import Tuple, List, Dict, Optional
import sys
from datetime import datetime
import json
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import local modules
from four_goal_nav_env import FourGoalNavigationEnv

# Create output directories
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


# =============================================================================
# Goal Configuration
# =============================================================================

GOAL_CONFIG = {
    'top_left': {
        'pos': np.array([-1.0, 1.0]),
        'color': 'red',
        'optimal_angle': 3 * np.pi / 4,  # 135°
        'label': 'Top Left'
    },
    'top_right': {
        'pos': np.array([1.0, 1.0]),
        'color': 'green',
        'optimal_angle': np.pi / 4,  # 45°
        'label': 'Top Right'
    },
    'bottom_left': {
        'pos': np.array([-1.0, -1.0]),
        'color': 'blue',
        'optimal_angle': -3 * np.pi / 4,  # -135°
        'label': 'Bottom Left'
    },
    'bottom_right': {
        'pos': np.array([1.0, -1.0]),
        'color': 'orange',
        'optimal_angle': -np.pi / 4,  # -45°
        'label': 'Bottom Right'
    }
}


# =============================================================================
# Helper Functions for Directed Random Trajectories
# =============================================================================

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
    noise_std: float = np.pi / 3,
    max_steps: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None
) -> Dict:
    """
    Generate a single directed random trajectory towards a specific goal.

    At each step, computes the optimal angle towards the goal and adds
    Gaussian noise to create exploratory behavior with directional bias.

    Args:
        env: The navigation environment
        target_goal: Name of target goal ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        noise_std: Standard deviation of Gaussian noise added to optimal angle
        max_steps: Maximum steps (uses env.max_steps if None)
        rng: Random state for reproducibility

    Returns:
        Dictionary containing:
            - 'states': List of (x, y) positions
            - 'actions': List of actions taken
            - 'rewards': List of rewards received
            - 'success': Whether the goal was reached
            - 'target_goal': The target goal name
            - 'final_pos': Final (x, y) position
            - 'steps': Number of steps taken
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
    states = [obs.copy()]
    actions = []
    rewards = []

    success = False
    steps = 0
    reached_which = None

    for step in range(max_steps):
        # Compute optimal angle towards goal
        optimal_angle = compute_optimal_angle(env.agent_pos, goal_pos)

        # Add Gaussian noise
        noise = rng.normal(0, noise_std)
        action_angle = normalize_angle(optimal_angle + noise)

        # Clip action to valid range
        action_angle = np.clip(action_angle, -np.pi, np.pi)

        # Execute action
        obs, reward, terminated, truncated, info = env.step(np.array([action_angle]))

        # Record data
        states.append(obs.copy())
        actions.append(action_angle)
        rewards.append(reward)
        steps = step + 1

        if terminated:
            success = True
            reached_which = info.get('reached_which', None)
            break
        elif truncated:
            break

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'success': success,
        'target_goal': target_goal,
        'final_pos': states[-1].copy(),
        'steps': steps,
        'reached_which': reached_which
    }


# =============================================================================
# Trajectory Generation and Filtering
# =============================================================================

def generate_and_filter_trajectories(
    env_params: Dict,
    num_candidates: int = 200,
    distance_threshold: float = 0.3,
    noise_std: float = np.pi / 3,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = True
) -> Tuple[List[Dict], List[str]]:
    """
    Generate directed random trajectories and filter those close to goals.
    
    Args:
        env_params: Environment parameters
        num_candidates: Number of candidate trajectories to generate
        distance_threshold: Distance threshold for filtering
        noise_std: Standard deviation of noise for directed random trajectories
        rng: Random state for reproducibility
        verbose: Whether to print progress
        
    Returns:
        filtered_trajectories: List of filtered trajectory data
        target_goals: List of target goals for each filtered trajectory
    """
    if rng is None:
        rng = np.random.RandomState()
    
    goal_names = list(GOAL_CONFIG.keys())
    
    filtered_trajectories = []
    target_goals = []
    
    if verbose:
        print(f"Generating {num_candidates} candidate trajectories...")
        print(f"  Distance threshold: {distance_threshold}")
        print(f"  Noise std: {noise_std:.4f} rad ({np.degrees(noise_std):.1f}°)")
    
    for i in range(num_candidates):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{num_candidates}")
        
        # Create fresh environment
        env = FourGoalNavigationEnv(**env_params)
        
        # Randomly select target goal
        target_goal = rng.choice(goal_names)
        
        # Generate directed random trajectory
        traj_data = generate_directed_random_trajectory(
            env=env,
            target_goal=target_goal,
            noise_std=noise_std,
            rng=rng
        )
        
        # Check if trajectory passes filter
        final_pos = traj_data['final_pos']
        goal_pos = GOAL_CONFIG[target_goal]['pos']
        dist_to_target = np.linalg.norm(final_pos - goal_pos)
        
        # Also check distance to any goal (not just the target)
        min_dist_to_any_goal = float('inf')
        closest_goal = None
        for goal_name, goal_info in GOAL_CONFIG.items():
            dist = np.linalg.norm(final_pos - goal_info['pos'])
            if dist < min_dist_to_any_goal:
                min_dist_to_any_goal = dist
                closest_goal = goal_name
        
        # Filter criteria: close to any goal or successful
        if traj_data['success'] or min_dist_to_any_goal < distance_threshold:
            filtered_trajectories.append(traj_data)
            # Use the closest goal as the target for MPPI
            target_goals.append(closest_goal if not traj_data['success'] else traj_data['target_goal'])
    
    if verbose:
        print(f"\nFiltering complete!")
        print(f"  Filtered trajectories: {len(filtered_trajectories)}/{num_candidates}")
        
        # Count by goal
        goal_counts = {g: 0 for g in goal_names}
        for g in target_goals:
            goal_counts[g] += 1
        print(f"  By goal: {goal_counts}")
    
    return filtered_trajectories, target_goals


# =============================================================================
# Rollout and MPPI Functions
# =============================================================================

def rollout_trajectory(
    env: FourGoalNavigationEnv,
    action_sequence: np.ndarray,
    return_actions: bool = False
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], float, Dict]:
    """
    Rollout a trajectory given an action sequence.

    Args:
        env: Environment instance (will be reset)
        action_sequence: Array of shape (H, 1) where H is horizon
        return_actions: Whether to return the executed actions

    Returns:
        state_trajectory: List of (x, y) positions
        action_trajectory: List of actions if return_actions=True, else None
        total_reward: Cumulative reward for the trajectory
        info: Dictionary with termination information
    """
    # Reset environment for this rollout
    obs, _ = env.reset()
    state_trajectory = [obs.copy()]
    action_trajectory = [] if return_actions else None

    total_reward = 0.0
    terminated = False
    truncated = False
    reached_goal = None

    H = len(action_sequence)

    for t in range(H):
        action = action_sequence[t].reshape(1)  # Ensure shape (1,)

        obs, reward, terminated, truncated, step_info = env.step(action)

        state_trajectory.append(obs.copy())
        if return_actions:
            action_trajectory.append(action.copy())
        total_reward += reward

        if terminated:
            reached_goal = step_info.get('reached_which', None)
            break
        if truncated:
            break

    info = {
        'terminated': terminated,
        'truncated': truncated,
        'reached_goal': reached_goal,
        'final_pos': state_trajectory[-1],
        'trajectory_length': len(state_trajectory) - 1
    }

    return state_trajectory, action_trajectory, total_reward, info


def _rollout_worker(args: Tuple) -> Tuple[List[np.ndarray], float, Dict]:
    """
    Worker function for parallel rollout execution.

    Args:
        args: Tuple of (env_class, env_params, action_sequence)

    Returns:
        state_trajectory, total_reward, info
    """
    env_class, env_params, action_sequence = args
    env = env_class(**env_params)
    state_traj, _, total_reward, info = rollout_trajectory(env, action_sequence)
    return state_traj, total_reward, info


def parallel_rollout_evaluation(
    env_class,
    env_params: Dict,
    action_sequences: np.ndarray,
    n_workers: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, List, List]:
    """
    Evaluate multiple action sequences in parallel.

    Args:
        env_class: Environment class
        env_params: Environment parameters
        action_sequences: Array of shape (num_samples, horizon, 1)
        n_workers: Number of parallel workers (default: CPU count)
        verbose: Whether to print progress

    Returns:
        reward_samples: Array of rewards for each sample
        state_trajectories: List of state trajectories
        infos: List of info dictionaries
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(action_sequences))

    num_samples = len(action_sequences)

    # Prepare arguments for workers
    args_list = [(env_class, env_params, action_sequences[i])
                 for i in range(num_samples)]

    # Execute rollouts in parallel
    reward_samples = np.zeros(num_samples)
    state_trajectories = [None] * num_samples
    infos = [None] * num_samples

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_rollout_worker, args): i
                   for i, args in enumerate(args_list)}

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                state_traj, total_reward, info = future.result()
                reward_samples[idx] = total_reward
                state_trajectories[idx] = state_traj
                infos[idx] = info

                completed += 1
                if verbose and completed % 10 == 0:
                    print(f"    Completed {completed}/{num_samples} rollouts")
            except Exception as e:
                print(f"    Error in rollout {idx}: {e}")
                reward_samples[idx] = -float('inf')

    return reward_samples, state_trajectories, infos


def mppi_optimize_with_warm_start(
    env_class,
    env_params: dict,
    initial_action_sequence: np.ndarray,
    horizon: int = 40,
    num_samples: int = 128,
    num_iterations: int = 50,
    lambda_: float = 1.0,
    noise_sigma: float = 0.5,
    elite_fraction: float = 0.1,
    seed: Optional[int] = None,
    verbose: bool = False,
    use_parallel: bool = True,
    n_workers: Optional[int] = None
) -> Dict:
    """
    Perform MPPI optimization with warm start from filtered trajectory.

    Uses elite sampling (like TDMPC2/CEM): only top-k% samples participate in update.

    Args:
        env_class: Environment class (FourGoalNavigationEnv)
        env_params: Parameters for environment initialization
        initial_action_sequence: Initial action sequence from filtered trajectory
        horizon: Planning horizon
        num_samples: Number of samples per iteration (K)
        num_iterations: Number of MPPI iterations (M)
        lambda_: Temperature parameter for MPPI
        noise_sigma: Standard deviation for action noise
        elite_fraction: Fraction of elite samples to use (e.g., 0.1 = top 10%)
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        use_parallel: Whether to use parallel rollout evaluation
        n_workers: Number of parallel workers (default: CPU count)

    Returns:
        Dictionary containing best trajectory, actions, and metrics
    """
    # Initialize random generator
    rng = np.random.RandomState(seed)
    
    # Use provided initial action sequence as warm start
    # Pad or truncate to match horizon
    if len(initial_action_sequence) < horizon:
        # Pad with zeros (or repeat last action)
        padding = np.zeros((horizon - len(initial_action_sequence), 1))
        nominal_actions = np.vstack([initial_action_sequence, padding])
    else:
        nominal_actions = initial_action_sequence[:horizon].copy()
    
    # Track best trajectory across all iterations
    best_total_reward = -float('inf')
    best_state_trajectory = None
    best_action_sequence = None
    best_info = None
    
    # Store iteration metrics for analysis
    iteration_rewards = []
    iteration_best_rewards = []
    
    # MPPI iterations
    for iter_idx in range(num_iterations):
        if verbose:
            print(f"  Iteration {iter_idx + 1}/{num_iterations}")
        
        # Create environment instances for parallel rollouts
        reward_samples = np.zeros(num_samples)
        state_trajectories = []
        action_sequences = []
        infos = []
        
        # Generate noisy action sequences with decay
        # Noise decay: 每次迭代衰减5%，从初始噪声逐渐减小
        current_sigma = noise_sigma * (0.95 ** iter_idx)
        noise = rng.normal(0, current_sigma, size=(num_samples, horizon, 1))
        action_samples = nominal_actions + noise

        # Clip actions to valid range [-pi, pi]
        action_samples = np.clip(action_samples, -np.pi, np.pi)

        # Evaluate each sample (parallel or serial)
        if use_parallel and num_samples > 4:
            reward_samples, state_trajectories, infos = parallel_rollout_evaluation(
                env_class, env_params, action_samples,
                n_workers=n_workers, verbose=verbose
            )
        else:
            # Serial evaluation for small batches or debugging
            for sample_idx in range(num_samples):
                # Create a fresh environment for this rollout
                env = env_class(**env_params)

                state_traj, _, total_reward, info = rollout_trajectory(
                    env, action_samples[sample_idx]
                )

                reward_samples[sample_idx] = total_reward
                state_trajectories.append(state_traj)
                infos.append(info)

        # Store action sequences
        action_sequences = action_samples

        # Update best trajectory
        for sample_idx in range(num_samples):
            if reward_samples[sample_idx] > best_total_reward:
                best_total_reward = reward_samples[sample_idx]
                best_state_trajectory = state_trajectories[sample_idx]
                best_action_sequence = action_sequences[sample_idx]
                best_info = infos[sample_idx]
        
        # Elite sampling MPPI update (like TDMPC2/CEM)
        # Select top-k% elite samples
        top_k = max(1, int(elite_fraction * num_samples))
        elite_indices = np.argsort(reward_samples)[-top_k:]
        
        # Get elite samples
        elite_rewards = reward_samples[elite_indices]
        elite_actions = action_samples[elite_indices]
        
        # Compute weights for elite samples only
        max_reward = np.max(elite_rewards)
        exp_rewards = np.exp((1.0 / lambda_) * (elite_rewards - max_reward))
        weights = exp_rewards / (np.sum(exp_rewards) + 1e-8)
        
        # Reshape weights for broadcasting
        weights_reshaped = weights.reshape(top_k, 1, 1)
        
        # Weighted average of elite action sequences
        nominal_actions = np.sum(weights_reshaped * elite_actions, axis=0)
        
        # Clip to valid action range
        nominal_actions = np.clip(nominal_actions, -np.pi, np.pi)
        
        # Store metrics
        iteration_rewards.append(reward_samples.mean())
        iteration_best_rewards.append(best_total_reward)
        
        if verbose:
            print(f"    Mean reward: {reward_samples.mean():.3f}, "
                  f"Best: {best_total_reward:.3f}, "
                  f"Reached: {best_info['reached_goal'] if best_info['reached_goal'] else 'None'}")
    
    return {
        'best_state_trajectory': best_state_trajectory,
        'best_action_sequence': best_action_sequence,
        'best_total_reward': best_total_reward,
        'best_info': best_info,
        'nominal_actions_final': nominal_actions,
        'iteration_rewards': iteration_rewards,
        'iteration_best_rewards': iteration_best_rewards,
        'seed': seed
    }


# =============================================================================
# Experiment Runner
# =============================================================================

def run_four_goal_experiments(
    env_params: dict,
    num_experiments: int = 50,
    num_candidates: int = 200,
    distance_threshold: float = 0.3,
    noise_std: float = np.pi / 3,
    horizon: int = 40,
    num_samples: int = 128,
    num_iterations: int = 50,
    lambda_: float = 1.0,
    noise_sigma: float = 0.5,
    elite_fraction: float = 0.1,
    base_seed: int = 42,
    verbose: bool = True,
    use_parallel: bool = True,
    n_workers: Optional[int] = None
) -> List[Dict]:
    """
    Run multiple independent four-goal MPPI experiments.

    Args:
        env_params: Environment parameters
        num_experiments: Number of experiments (N)
        num_candidates: Number of candidate trajectories for filtering
        distance_threshold: Distance threshold for filtering
        noise_std: Noise for directed random trajectories
        horizon: Planning horizon
        num_samples: Number of samples per iteration
        num_iterations: Number of MPPI iterations
        lambda_: Temperature parameter
        noise_sigma: Action noise standard deviation
        elite_fraction: Fraction of elite samples (e.g., 0.1 = top 10%)
        base_seed: Base random seed
        verbose: Whether to print progress
        use_parallel: Whether to use parallel rollout evaluation
        n_workers: Number of parallel workers

    Returns:
        List of experiment results
    """
    all_results = []
    
    print("=" * 70)
    print("FOUR-GOAL MPPI EXPERIMENTS")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  N_EXPERIMENTS: {num_experiments}")
    print(f"  NUM_CANDIDATES: {num_candidates}")
    print(f"  DISTANCE_THRESHOLD: {distance_threshold}")
    print(f"  NOISE_STD: {noise_std:.4f} rad")
    print(f"  HORIZON: {horizon}")
    print(f"  NUM_SAMPLES: {num_samples}")
    print(f"  NUM_ITERATIONS: {num_iterations}")
    print(f"  LAMBDA: {lambda_}")
    print(f"  NOISE_SIGMA: {noise_sigma}")
    print("-" * 70)
    
    for exp_idx in range(num_experiments):
        if verbose:
            print(f"\nExperiment {exp_idx + 1}/{num_experiments}")
        
        # Use different seed for each experiment
        seed = base_seed + exp_idx
        rng = np.random.RandomState(seed)
        
        # Stage 1: Generate and filter trajectories
        filtered_trajs, target_goals = generate_and_filter_trajectories(
            env_params=env_params,
            num_candidates=num_candidates,
            distance_threshold=distance_threshold,
            noise_std=noise_std,
            rng=rng,
            verbose=False
        )
        
        # If no trajectories pass filter, generate more
        if len(filtered_trajs) == 0:
            print("  Warning: No trajectories passed filter, generating more...")
            filtered_trajs, target_goals = generate_and_filter_trajectories(
                env_params=env_params,
                num_candidates=num_candidates * 2,
                distance_threshold=distance_threshold,
                noise_std=noise_std,
                rng=rng,
                verbose=False
            )
        
        if len(filtered_trajs) == 0:
            print("  Error: Still no trajectories. Using random initialization.")
            # Fallback to random initialization
            initial_actions = rng.uniform(-np.pi, np.pi, size=(horizon, 1))
        else:
            # Select the best filtered trajectory (closest to goal)
            best_idx = 0
            best_dist = float('inf')
            for i, (traj, goal) in enumerate(zip(filtered_trajs, target_goals)):
                final_pos = traj['final_pos']
                goal_pos = GOAL_CONFIG[goal]['pos']
                dist = np.linalg.norm(final_pos - goal_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            # Convert trajectory actions to numpy array
            selected_traj = filtered_trajs[best_idx]
            actions = np.array(selected_traj['actions']).reshape(-1, 1)
            initial_actions = actions
        
        if verbose:
            print(f"  Selected trajectory: {len(filtered_trajs)} filtered, "
                  f"using trajectory with {len(initial_actions)} actions")
        
        # Stage 2: MPPI optimization with warm start
        result = mppi_optimize_with_warm_start(
            env_class=FourGoalNavigationEnv,
            env_params=env_params,
            initial_action_sequence=initial_actions,
            horizon=horizon,
            num_samples=num_samples,
            num_iterations=num_iterations,
            lambda_=lambda_,
            noise_sigma=noise_sigma,
            elite_fraction=elite_fraction,
            seed=seed,
            verbose=False,
            use_parallel=use_parallel,
            n_workers=n_workers
        )
        
        all_results.append(result)
        
        if verbose:
            print(f"  Best reward: {result['best_total_reward']:.3f}, "
                  f"Reached: {result['best_info']['reached_goal']}")
    
    return all_results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_four_goal_trajectories(
    results: List[Dict],
    env_params: dict,
    save_path: str = "figures/four_goal_mppi_best_trajectories.png",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot all best trajectories on a single figure.
    
    Args:
        results: List of experiment results
        env_params: Environment parameters
        save_path: Path to save the figure
        title: Figure title (if None, auto-generated)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Environment bounds
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw boundary
    boundary = Rectangle((-1, -1), 2, 2, linewidth=2,
                        edgecolor='black', facecolor='none')
    ax.add_patch(boundary)
    
    # Goal radius
    goal_radius = env_params.get('goal_radius', 0.1)
    
    # Draw goals
    for goal_name, goal_info in GOAL_CONFIG.items():
        goal_circle = Circle(goal_info['pos'], goal_radius,
                            color=goal_info['color'], alpha=0.5,
                            label=goal_info['label'])
        ax.add_patch(goal_circle)
    
    # Draw start point
    ax.scatter(0, 0, c='purple', s=200, marker='o',
               edgecolors='black', linewidth=2, label='Start', zorder=10)
    
    # Separate trajectories by which goal they reached
    trajectories_by_goal = {g: [] for g in GOAL_CONFIG.keys()}
    trajectories_to_none = []
    
    for result in results:
        traj = result['best_state_trajectory']
        info = result['best_info']
        
        if info['reached_goal'] in trajectories_by_goal:
            trajectories_by_goal[info['reached_goal']].append(traj)
        else:
            trajectories_to_none.append(traj)
    
    # Plot trajectories for each goal
    for goal_name, goal_info in GOAL_CONFIG.items():
        for traj in trajectories_by_goal[goal_name]:
            traj_array = np.array(traj)
            ax.plot(traj_array[:, 0], traj_array[:, 1],
                    color=goal_info['color'], alpha=0.5, linewidth=1.5)
    
    # Plot unsuccessful trajectories (gray)
    for traj in trajectories_to_none:
        traj_array = np.array(traj)
        ax.plot(traj_array[:, 0], traj_array[:, 1],
                'gray', alpha=0.3, linewidth=1.0, linestyle='--')
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='purple', marker='o', markersize=10,
               label='Start', markeredgecolor='black', linestyle='none'),
        Line2D([0], [0], color='gray', alpha=0.5, marker='o', markersize=15,
               label='Goals', linestyle='none'),
    ]
    
    for goal_name, goal_info in GOAL_CONFIG.items():
        count = len(trajectories_by_goal[goal_name])
        legend_elements.append(
            Line2D([0], [0], color=goal_info['color'], linewidth=1.5, alpha=0.5,
                   label=f'To {goal_info["label"]} ({count})')
        )
    legend_elements.append(
        Line2D([0], [0], color='gray', linewidth=1.0, alpha=0.3, linestyle='--',
               label=f'No Goal ({len(trajectories_to_none)})')
    )
    
    ax.legend(handles=legend_elements, loc='center left', fontsize=10,
              bbox_to_anchor=(1.02, 0.5))
    
    # Set title
    if title is None:
        num_experiments = len(results)
        avg_reward = np.mean([r['best_total_reward'] for r in results])
        total_success = sum(len(trajs) for trajs in trajectories_by_goal.values())
        success_rate = total_success / num_experiments
        
        title = (f"Four-Goal MPPI Best Trajectories (N={num_experiments})\n"
                f"Success Rate: {success_rate:.1%}, "
                f"Avg Reward: {avg_reward:.2f}")
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_first_action_distribution(
    results: List[Dict],
    save_path: str = "figures/first_action_distribution.png",
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot distribution of first actions across all experiments.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Extract first actions
    first_actions = []
    reached_goals = []
    
    for result in results:
        if result['best_action_sequence'] is not None and len(result['best_action_sequence']) > 0:
            first_actions.append(result['best_action_sequence'][0, 0])
            reached_goals.append(result['best_info']['reached_goal'])
    
    if len(first_actions) == 0:
        print("No first actions to plot")
        return
    
    first_actions = np.array(first_actions)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    bins = 72  # More bins for smoother visualization
    counts, bin_edges, patches = ax.hist(first_actions, bins=bins, density=True,
                                          alpha=0.6, color='steelblue',
                                          edgecolor='white', linewidth=0.5)
    
    # Compute KDE
    if len(first_actions) > 1:
        kde = stats.gaussian_kde(first_actions)
        x_range = np.linspace(-np.pi, np.pi, 500)
        kde_values = kde(x_range)
        ax.plot(x_range, kde_values, 'navy', linewidth=2, label='KDE Fit')
    
    # Add vertical lines for optimal directions
    for goal_name, goal_info in GOAL_CONFIG.items():
        ax.axvline(x=goal_info['optimal_angle'], color=goal_info['color'],
                  linestyle='--', linewidth=2, alpha=0.8,
                  label=f"{goal_info['label']} ({np.degrees(goal_info['optimal_angle']):.0f}°)")
    
    # Set labels and title
    ax.set_xlabel('First Action Angle (radians)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    # Count successes
    success_count = sum(1 for g in reached_goals if g is not None)
    total_count = len(first_actions)
    
    ax.set_title(f'Distribution of First Actions (N={total_count}, Success: {success_count}/{total_count})',
                fontsize=12)
    
    # Set x-axis limits and ticks
    ax.set_xlim(-np.pi, np.pi)
    xticks = np.array([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, 
                       np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    xtick_labels = [r'$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
                    r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"First action distribution saved to {save_path}")
    plt.show()


def plot_convergence_analysis(
    results: List[Dict],
    env_params: dict,
    save_path: str = "figures/four_goal_mppi_convergence.png"
) -> None:
    """
    Analyze and visualize MPPI convergence patterns.
    
    Args:
        results: List of experiment results
        env_params: Environment parameters
        save_path: Path to save the convergence plot
    """
    if len(results) == 0:
        print("No results to analyze")
        return
    
    # Check first experiment's iteration progress
    first_result = results[0]
    iteration_rewards = first_result['iteration_rewards']
    iteration_best_rewards = first_result['iteration_best_rewards']
    
    print(f"\nExample experiment (Seed: {first_result['seed']}):")
    print("Iteration progress:")
    for i, (avg, best) in enumerate(zip(iteration_rewards, iteration_best_rewards)):
        if i < 5 or i > len(iteration_rewards) - 6:
            print(f"  Iter {i+1:2d}: Avg reward = {avg:.3f}, Best reward = {best:.3f}")
        elif i == 5:
            print("  ...")
    
    # Create convergence plot for first experiment
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = range(1, len(iteration_rewards) + 1)
    
    ax.plot(iterations, iteration_rewards, 'b-o', label='Average Reward', markersize=3)
    ax.plot(iterations, iteration_best_rewards, 'r-s', label='Best Reward', markersize=3)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title(f'Four-Goal MPPI Convergence (Seed: {first_result["seed"]})', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add horizontal line for optimal reward
    optimal_reward = 1.0 + env_params['step_penalty'] * (np.sqrt(2) / env_params['step_size'])
    ax.axhline(y=optimal_reward, color='g', linestyle='--', alpha=0.5,
              label=f'Theoretical Optimal (~{optimal_reward:.2f})')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nConvergence plot saved to {save_path}")
    
    plt.show()


def plot_action_sequence_heatmap(
    results: List[Dict],
    save_path: str = "figures/action_sequence_heatmap.png",
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Plot heatmap of action sequences over time steps, grouped by reached goal.

    Args:
        results: List of experiment results
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Group action sequences by goal
    action_sequences_by_goal = {g: [] for g in GOAL_CONFIG.keys()}

    for result in results:
        if result['best_action_sequence'] is not None:
            goal = result['best_info']['reached_goal']
            if goal in action_sequences_by_goal:
                # Extract first 20 time steps (or less if shorter)
                actions = result['best_action_sequence'][:, 0]
                action_sequences_by_goal[goal].append(actions[:20])

    # Create figure with subplots for each goal
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (goal_name, goal_info) in enumerate(GOAL_CONFIG.items()):
        ax = axes[idx]

        if len(action_sequences_by_goal[goal_name]) == 0:
            ax.text(0.5, 0.5, f'No data for {goal_info["label"]}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{goal_info["label"]} (N=0)')
            continue

        # Create matrix: each row is a trajectory, columns are time steps
        max_len = max(len(seq) for seq in action_sequences_by_goal[goal_name])
        matrix = np.zeros((len(action_sequences_by_goal[goal_name]), max_len))

        for i, seq in enumerate(action_sequences_by_goal[goal_name]):
            matrix[i, :len(seq)] = seq

        # Plot heatmap
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                      vmin=-np.pi, vmax=np.pi, interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Action (rad)', rotation=270, labelpad=15)

        # Set labels
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Trajectory Index', fontsize=10)
        ax.set_title(f'{goal_info["label"]} (N={len(action_sequences_by_goal[goal_name])})',
                    fontsize=11, fontweight='bold')

        # Set y-axis ticks
        if len(action_sequences_by_goal[goal_name]) > 10:
            step = len(action_sequences_by_goal[goal_name]) // 5
            ax.set_yticks(range(0, len(action_sequences_by_goal[goal_name]), step))
        else:
            ax.set_yticks(range(len(action_sequences_by_goal[goal_name])))

    plt.suptitle('Action Sequence Heatmap by Goal',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Action sequence heatmap saved to {save_path}")
    plt.show()


def plot_action_distribution_by_timestep(
    results: List[Dict],
    save_path: str = "figures/action_distribution_by_timestep.png",
    figsize: Tuple[int, int] = (20, 12)
) -> None:
    """
    Plot action distribution at different time steps, grouped by goal.

    Args:
        results: List of experiment results
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Group actions by goal and time step
    actions_by_goal_time = {g: {} for g in GOAL_CONFIG.keys()}
    time_steps_to_show = [0, 2, 5, 10, 15, 20]  # Show specific time steps

    for result in results:
        if result['best_action_sequence'] is not None:
            goal = result['best_info']['reached_goal']
            if goal in actions_by_goal_time:
                actions = result['best_action_sequence'][:, 0]
                for t in time_steps_to_show:
                    if t < len(actions):
                        if t not in actions_by_goal_time[goal]:
                            actions_by_goal_time[goal][t] = []
                        actions_by_goal_time[goal][t].append(actions[t])

    # Create figure - need 4 goals × 6 time steps = 24 subplots
    fig, axes = plt.subplots(4, 6, figsize=figsize)
    fig.suptitle('Action Distribution at Different Time Steps',
                fontsize=14, fontweight='bold')

    for goal_idx, (goal_name, goal_info) in enumerate(GOAL_CONFIG.items()):
        for time_idx, t in enumerate(time_steps_to_show):
            ax = axes[goal_idx, time_idx]

            if t not in actions_by_goal_time[goal] or len(actions_by_goal_time[goal][t]) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=8)
            else:
                actions = actions_by_goal_time[goal][t]

                # Plot histogram
                bins = np.linspace(-np.pi, np.pi, 31)
                counts, _, _ = ax.hist(actions, bins=bins, density=True,
                                      alpha=0.7, color=goal_info['color'],
                                      edgecolor='white', linewidth=0.5)

                # Add KDE if enough data
                if len(actions) > 3:
                    try:
                        kde = stats.gaussian_kde(actions)
                        x_range = np.linspace(-np.pi, np.pi, 100)
                        kde_values = kde(x_range)
                        ax.plot(x_range, kde_values, 'darkblue',
                               linewidth=1.5, alpha=0.8)
                    except:
                        pass

                # Add optimal angle line
                ax.axvline(x=goal_info['optimal_angle'], color='red',
                          linestyle='--', linewidth=2, alpha=0.7)

                # Statistics
                mean_action = np.mean(actions)
                std_action = np.std(actions)
                ax.set_title(f't={t}: μ={np.degrees(mean_action):.1f}°±{np.degrees(std_action):.1f}°',
                           fontsize=8)

            # Formatting
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, linewidth=0.5)

            # Set x-axis labels
            if goal_idx == 3:  # Bottom row (4 rows total: 0,1,2,3)
                ax.set_xlabel('Action (rad)', fontsize=7)
                if time_idx == 0:
                    ax.set_xticks([-np.pi, 0, np.pi])
                    ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'], fontsize=7)
                else:
                    ax.set_xticklabels([], fontsize=7)
            else:
                ax.set_xticklabels([])

            # Set y-axis labels
            if time_idx == 0:
                ax.set_ylabel(goal_info['label'], fontsize=9, fontweight='bold')
            else:
                ax.set_yticklabels([])

            # Column titles
            if goal_idx == 0:
                axes[0, time_idx].set_title(f'Time Step {t}', fontsize=10,
                                          fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Action distribution by timestep saved to {save_path}")
    plt.show()


def plot_iteration_progression(
    results: List[Dict],
    save_path: str = "figures/iteration_progression.png",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot how rewards progress across MPPI iterations for all experiments.

    Args:
        results: List of experiment results
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot average reward progression
    for idx, result in enumerate(results):
        iteration_rewards = result['iteration_rewards']
        iterations = range(1, len(iteration_rewards) + 1)
        ax1.plot(iterations, iteration_rewards, alpha=0.3, linewidth=0.8,
                color='gray', label='Individual experiments' if idx == 0 else '')

    # Plot mean and confidence interval
    min_len = min(len(r['iteration_rewards']) for r in results)
    all_rewards = np.array([r['iteration_rewards'][:min_len] for r in results])

    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    iterations = range(1, min_len + 1)

    ax1.plot(iterations, mean_rewards, 'b-', linewidth=2.5, label='Mean')
    ax1.fill_between(iterations, mean_rewards - std_rewards, mean_rewards + std_rewards,
                    alpha=0.3, color='blue', label='±1 Std Dev')

    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Average Reward', fontsize=11)
    ax1.set_title('MPPI Convergence: Average Reward per Iteration', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)

    # Plot best reward progression
    for idx, result in enumerate(results):
        iteration_best = result['iteration_best_rewards']
        iterations = range(1, len(iteration_best) + 1)
        ax2.plot(iterations, iteration_best, alpha=0.3, linewidth=0.8,
                color='gray')

    all_best = np.array([r['iteration_best_rewards'][:min_len] for r in results])
    mean_best = np.mean(all_best, axis=0)
    std_best = np.std(all_best, axis=0)

    ax2.plot(iterations, mean_best, 'r-', linewidth=2.5, label='Mean Best')
    ax2.fill_between(iterations, mean_best - std_best, mean_best + std_best,
                    alpha=0.3, color='red')

    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Best Reward', fontsize=11)
    ax2.set_title('MPPI Convergence: Best Reward per Iteration', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9)

    plt.suptitle(f'MPPI Iteration Progression (N={len(results)} experiments)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Iteration progression saved to {save_path}")
    plt.show()


def plot_goal_specific_statistics(
    results: List[Dict],
    save_path: str = "figures/goal_specific_statistics.png",
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Plot goal-specific statistics and comparisons.

    Args:
        results: List of experiment results
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Extract data by goal
    data_by_goal = {g: {'rewards': [], 'lengths': [], 'first_actions': []}
                   for g in GOAL_CONFIG.keys()}

    for result in results:
        goal = result['best_info']['reached_goal']
        if goal in data_by_goal:
            data_by_goal[goal]['rewards'].append(result['best_total_reward'])
            data_by_goal[goal]['lengths'].append(result['best_info']['trajectory_length'])
            if result['best_action_sequence'] is not None and len(result['best_action_sequence']) > 0:
                data_by_goal[goal]['first_actions'].append(result['best_action_sequence'][0, 0])

    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Box plot of rewards by goal
    ax1 = fig.add_subplot(gs[0, 0])
    goal_labels = []
    reward_data = []
    colors = []

    for goal_name, goal_info in GOAL_CONFIG.items():
        if len(data_by_goal[goal_name]['rewards']) > 0:
            goal_labels.append(goal_info['label'])
            reward_data.append(data_by_goal[goal_name]['rewards'])
            colors.append(goal_info['color'])

    bp = ax1.boxplot(reward_data, labels=goal_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Total Reward', fontsize=10)
    ax1.set_title('Reward Distribution by Goal', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Box plot of trajectory lengths by goal
    ax2 = fig.add_subplot(gs[0, 1])
    length_data = [data_by_goal[g]['lengths'] for g in GOAL_CONFIG.keys()
                  if len(data_by_goal[g]['lengths']) > 0]

    bp2 = ax2.boxplot(length_data, labels=goal_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Trajectory Length (steps)', fontsize=10)
    ax2.set_title('Trajectory Length by Goal', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Violin plot of first actions by goal
    ax3 = fig.add_subplot(gs[0, 2])
    action_data = [data_by_goal[g]['first_actions'] for g in GOAL_CONFIG.keys()
                  if len(data_by_goal[g]['first_actions']) > 0]

    parts = ax3.violinplot(action_data, positions=range(len(action_data)),
                          showmeans=True, showmedians=True)

    for i, (goal_name, goal_info) in enumerate(GOAL_CONFIG.items()):
        if i < len(parts['bodies']):
            parts['bodies'][i].set_facecolor(goal_info['color'])
            parts['bodies'][i].set_alpha(0.7)

    ax3.set_xticks(range(len(goal_labels)))
    ax3.set_xticklabels(goal_labels)
    ax3.set_ylabel('First Action (rad)', fontsize=10)
    ax3.set_title('First Action Distribution by Goal', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Success rate pie chart
    ax4 = fig.add_subplot(gs[1, 0])
    goal_counts = [len(data_by_goal[g]['rewards']) for g in GOAL_CONFIG.keys()]
    total_count = sum(goal_counts)

    if total_count > 0:
        wedges, texts, autotexts = ax4.pie(goal_counts, labels=goal_labels,
                                            autopct='%1.1f%%',
                                            colors=[GOAL_CONFIG[g]['color']
                                                   for g in GOAL_CONFIG.keys()],
                                            startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    ax4.set_title('Success Rate by Goal', fontsize=11, fontweight='bold')

    # 5. Statistics table
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    table_data = []
    for goal_name, goal_info in GOAL_CONFIG.items():
        if len(data_by_goal[goal_name]['rewards']) > 0:
            rewards = data_by_goal[goal_name]['rewards']
            lengths = data_by_goal[goal_name]['lengths']

            table_data.append([
                goal_info['label'],
                f"{len(rewards)}",
                f"{np.mean(rewards):.3f} ± {np.std(rewards):.3f}",
                f"{np.mean(lengths):.1f} ± {np.std(lengths):.1f}"
            ])

    table = ax5.table(cellText=table_data,
                     colLabels=['Goal', 'N', 'Reward (mean±std)', 'Length (mean±std)'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.3, 0.3])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax5.set_title('Detailed Statistics', fontsize=11, fontweight='bold', pad=20)

    plt.suptitle(f'Goal-Specific Performance Analysis (N={len(results)})',
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Goal-specific statistics saved to {save_path}")
    plt.show()


def print_statistics(results: List[Dict]) -> None:
    """
    Print comprehensive statistics about the experiments.
    
    Args:
        results: List of experiment results
    """
    num_experiments = len(results)
    
    # Count successes
    goal_success = {g: 0 for g in GOAL_CONFIG.keys()}
    no_success = 0
    
    total_rewards = []
    trajectory_lengths = []
    
    for result in results:
        info = result['best_info']
        
        if info['reached_goal'] in goal_success:
            goal_success[info['reached_goal']] += 1
        else:
            no_success += 1
        
        total_rewards.append(result['best_total_reward'])
        trajectory_lengths.append(info['trajectory_length'])
    
    # Calculate statistics
    total_success = sum(goal_success.values())
    success_rate = total_success / num_experiments
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(trajectory_lengths)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT STATISTICS")
    print("=" * 70)
    print(f"Total experiments: {num_experiments}")
    print(f"Success rate: {success_rate:.1%}")
    for goal_name, goal_info in GOAL_CONFIG.items():
        count = goal_success[goal_name]
        print(f"  Reached {goal_info['label']}: {count} ({count/num_experiments:.1%})")
    print(f"  No goal reached: {no_success} ({no_success/num_experiments:.1%})")
    print(f"Average best reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"Average trajectory length: {avg_length:.1f} steps")
    print(f"Reward range: [{min(total_rewards):.3f}, {max(total_rewards):.3f}]")
    print("=" * 70)


def save_results_to_file(results: List[Dict], filepath: str) -> None:
    """
    Save experiment results to a file.
    
    Args:
        results: List of experiment results
        filepath: Path to save the results
    """
    # Convert to a format that can be saved
    save_data = {
        'results': results,
        'timestamp': np.datetime64('now')
    }
    
    np.save(filepath, save_data, allow_pickle=True)
    print(f"Results saved to {filepath}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Main function to run four-goal MPPI experiments and visualize results.
    """
    print("=" * 70)
    print("MPPI FOR FOUR-GOAL NAVIGATION ENVIRONMENT")
    print("=" * 70)
    
    # ============== CONFIGURABLE PARAMETERS ==============
    
    # Environment parameters
    ENV_PARAMS = {
        'step_size': 0.10,
        'goal_radius': 0.1,
        'max_steps': 40,
        'step_penalty': -0.01
    }
    
    # Stage 1: Trajectory generation and filtering
    NUM_CANDIDATES = 200
    DISTANCE_THRESHOLD = 0.3
    NOISE_STD = np.pi / 3
    
    # Stage 2: MPPI optimization
    HORIZON = 40
    NUM_SAMPLES = 128
    NUM_ITERATIONS = 50
    LAMBDA = 0.1  # 减小以放大高奖励样本权重差异
    NOISE_SIGMA = 0.2  # 减小以稳定优化方向
    ELITE_FRACTION = 0.1  # 精英采样比例：只用top 10%的样本更新
    
    # Experiments
    N_EXPERIMENTS = 50
    BASE_SEED = 42

    # Parallelization
    USE_PARALLEL = True  # Enable parallel rollout evaluation
    N_WORKERS = None  # None = auto-detect CPU count
    
    # ============== ENVIRONMENT INITIALIZATION ==============
    print("\nEnvironment Configuration:")
    print(f"  Step size: {ENV_PARAMS['step_size']}")
    print(f"  Goal radius: {ENV_PARAMS['goal_radius']}")
    print(f"  Max steps: {ENV_PARAMS['max_steps']}")
    print(f"  Step penalty: {ENV_PARAMS['step_penalty']}")
    
    # ============== RUN EXPERIMENTS ==============
    print(f"\nParallelization: {USE_PARALLEL}")
    if USE_PARALLEL:
        print(f"  Workers: {N_WORKERS if N_WORKERS else 'Auto-detect'}")

    results = run_four_goal_experiments(
        env_params=ENV_PARAMS,
        num_experiments=N_EXPERIMENTS,
        num_candidates=NUM_CANDIDATES,
        distance_threshold=DISTANCE_THRESHOLD,
        noise_std=NOISE_STD,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        num_iterations=NUM_ITERATIONS,
        lambda_=LAMBDA,
        noise_sigma=NOISE_SIGMA,
        elite_fraction=ELITE_FRACTION,
        base_seed=BASE_SEED,
        verbose=True,
        use_parallel=USE_PARALLEL,
        n_workers=N_WORKERS
    )
    
    # ============== PRINT STATISTICS ==============
    print_statistics(results)
    
    # ============== CREATE OUTPUT DIRECTORY ==============
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    output_dir = os.path.join("figures", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Paths for saved artifacts
    SAVE_FIGURE_PATH = os.path.join(output_dir, "mppi_best_trajectories.png")
    CONVERGENCE_PATH = os.path.join(output_dir, "mppi_convergence_example.png")
    FIRST_ACTION_PATH = os.path.join(output_dir, "first_action_distribution.png")
    ACTION_HEATMAP_PATH = os.path.join(output_dir, "action_sequence_heatmap.png")
    ACTION_BY_TIMESTEP_PATH = os.path.join(output_dir, "action_distribution_by_timestep.png")
    ITERATION_PROGRESSION_PATH = os.path.join(output_dir, "iteration_progression.png")
    GOAL_STATS_PATH = os.path.join(output_dir, "goal_specific_statistics.png")
    RESULTS_PATH = os.path.join(output_dir, "mppi_results.npy")
    
    # ============== VISUALIZE RESULTS ==============
    print("\n" + "=" * 70)
    print("VISUALIZING RESULTS")
    print("=" * 70)
    
    # Plot all best trajectories
    plot_four_goal_trajectories(
        results=results,
        env_params=ENV_PARAMS,
        save_path=SAVE_FIGURE_PATH,
        title=None,
        figsize=(12, 10)
    )
    
    # Plot convergence analysis
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    plot_convergence_analysis(results, ENV_PARAMS, save_path=CONVERGENCE_PATH)
    
    # Plot first action distribution
    print("\n" + "=" * 70)
    print("FIRST ACTION DISTRIBUTION")
    print("=" * 70)
    plot_first_action_distribution(results, save_path=FIRST_ACTION_PATH)

    # Plot action sequence heatmap
    print("\n" + "=" * 70)
    print("ACTION SEQUENCE HEATMAP")
    print("=" * 70)
    plot_action_sequence_heatmap(results, save_path=ACTION_HEATMAP_PATH)

    # Plot action distribution by timestep
    print("\n" + "=" * 70)
    print("ACTION DISTRIBUTION BY TIMESTEP")
    print("=" * 70)
    plot_action_distribution_by_timestep(results, save_path=ACTION_BY_TIMESTEP_PATH)

    # Plot iteration progression
    print("\n" + "=" * 70)
    print("ITERATION PROGRESSION")
    print("=" * 70)
    plot_iteration_progression(results, save_path=ITERATION_PROGRESSION_PATH)

    # Plot goal-specific statistics
    print("\n" + "=" * 70)
    print("GOAL-SPECIFIC STATISTICS")
    print("=" * 70)
    plot_goal_specific_statistics(results, save_path=GOAL_STATS_PATH)
    
    # ============== SAVE RESULTS ==============
    save_results_to_file(results, RESULTS_PATH)
    
    # Save parameters as JSON
    try:
        env_params_path = os.path.join(output_dir, 'env_params.json')
        with open(env_params_path, 'w', encoding='utf-8') as f:
            json.dump(ENV_PARAMS, f, ensure_ascii=False, indent=2)
        
        mppi_params = {
            'num_experiments': N_EXPERIMENTS,
            'num_candidates': NUM_CANDIDATES,
            'distance_threshold': DISTANCE_THRESHOLD,
            'noise_std': NOISE_STD,
            'horizon': HORIZON,
            'num_samples': NUM_SAMPLES,
            'num_iterations': NUM_ITERATIONS,
            'lambda': LAMBDA,
            'noise_sigma': NOISE_SIGMA,
            'elite_fraction': ELITE_FRACTION,
            'base_seed': BASE_SEED,
            'use_parallel': USE_PARALLEL,
            'n_workers': N_WORKERS,
        }
        mppi_params_path = os.path.join(output_dir, 'mppi_params.json')
        with open(mppi_params_path, 'w', encoding='utf-8') as f:
            json.dump(mppi_params, f, ensure_ascii=False, indent=2)
        
        print(f"Saved env params to {env_params_path}")
        print(f"Saved MPPI params to {mppi_params_path}")
    except Exception as e:
        print(f"Failed to save params JSON: {e}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"  - Best trajectories: {SAVE_FIGURE_PATH}")
    print(f"  - Convergence plot: {CONVERGENCE_PATH}")
    print(f"  - First action distribution: {FIRST_ACTION_PATH}")
    print(f"  - Action sequence heatmap: {ACTION_HEATMAP_PATH}")
    print(f"  - Action distribution by timestep: {ACTION_BY_TIMESTEP_PATH}")
    print(f"  - Iteration progression: {ITERATION_PROGRESSION_PATH}")
    print(f"  - Goal-specific statistics: {GOAL_STATS_PATH}")
    print(f"  - Results data: {RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()