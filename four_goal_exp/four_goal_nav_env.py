"""
Four-Goal Navigation Environment for Testing Multi-Modal Planning

This environment creates a symmetric quad-goal scenario where an agent must choose
between four equally optimal actions at the initial state: move to any of the four corners.

Environment Design:
- State space: 2D position (x, y) in [-1, 1] x [-1, 1]
- Action space: Direction angle theta in [-pi, pi] with fixed step size
- Goals: Four goals at corners: (-1, 1), (1, 1), (-1, -1), (1, -1), all with radius 0.1
- Reward: +1 for reaching any goal, -0.01 per step (optional)
- Termination: Reach goal or exceed max_steps

Purpose: Demonstrate that unimodal Gaussian policies struggle with multi-modal
supervision, while flow policies can naturally handle it. This four-goal version
creates a more complex multi-modal distribution (4 modes) compared to the two-goal version.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from typing import Tuple, Optional, List, Dict
import gymnasium as gym
from gymnasium import spaces
import os

os.makedirs("figures", exist_ok=True)


class FourGoalNavigationEnv(gym.Env):
    """
    Symmetric quad-goal navigation environment.
    
    The agent starts at (0, 0) and must navigate to one of four corner goals:
    - top_left: (-1, 1)
    - top_right: (1, 1)
    - bottom_left: (-1, -1)
    - bottom_right: (1, -1)
    
    From the center, the optimal first actions form a clear four-modal distribution:
    - Move to top_left: theta ≈ 3π/4 (135°)
    - Move to top_right: theta ≈ π/4 (45°)
    - Move to bottom_left: theta ≈ -3π/4 or 5π/4 (-135° or 225°)
    - Move to bottom_right: theta ≈ -π/4 or 7π/4 (-45° or 315°)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        step_size: float = 0.1,
        goal_radius: float = 0.1,
        max_steps: int = 30,
        step_penalty: float = -0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize the environment.
        
        Args:
            step_size: Fixed step size per action
            goal_radius: Radius of goal regions
            max_steps: Maximum number of steps per episode
            step_penalty: Penalty per step (encourages efficiency)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Environment parameters
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        
        # Four goals at corners of [-1, 1] x [-1, 1]
        self.goals = {
            'top_left': np.array([-1.0, 1.0]),
            'top_right': np.array([1.0, 1.0]),
            'bottom_left': np.array([-1.0, -1.0]),
            'bottom_right': np.array([1.0, -1.0])
        }
        
        # State space: 2D position in [-1, 1] x [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Action space: angle theta in [-pi, pi]
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
        )
        
        # State variables
        self.agent_pos = None
        self.current_step = 0
        self.trajectory = None
        self.terminated = False
        self.truncated = False
        
        # Rendering
        self.fig = None
        self.ax = None
        self.agent_circle = None
        self.trajectory_line = None
        
        # Set seed
        if seed is not None:
            self.reset_seed(seed)
    
    def reset_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.np_random = np.random.RandomState(seed)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed (optional)
            options: Additional options (not used)
            
        Returns:
            observation: Initial state (0, 0)
            info: Dictionary with additional information
        """
        if seed is not None:
            self.reset_seed(seed)
        
        # Reset to center
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        self.trajectory = [self.agent_pos.copy()]
        self.terminated = False
        self.truncated = False
        
        info = {
            'agent_pos': self.agent_pos.copy(),
            'goals': {k: v.copy() for k, v in self.goals.items()},
            'step_size': self.step_size
        }
        
        return self.agent_pos.copy(), info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Direction angle theta in [-pi, pi]
            
        Returns:
            observation: New position (x, y)
            reward: Reward for this step
            terminated: Episode ended successfully (reached goal)
            truncated: Episode ended due to timeout or boundary
            info: Additional information
        """
        theta = action[0]
        
        # Compute new position
        dx = self.step_size * np.cos(theta)
        dy = self.step_size * np.sin(theta)
        new_pos = self.agent_pos + np.array([dx, dy])
        
        # Clip to boundaries [-1, 1] x [-1, 1]
        new_pos = np.clip(new_pos, -1.0, 1.0)
        
        # Check if moved (clipped means hit boundary)
        hit_boundary = not np.allclose(self.agent_pos + np.array([dx, dy]), new_pos)
        
        # Update agent position
        self.agent_pos = new_pos.astype(np.float32)
        self.trajectory.append(self.agent_pos.copy())
        self.current_step += 1
        
        # Compute reward
        reward = self.step_penalty  # Step penalty by default
        reached_goal = False
        reached_which = None
        
        # Check if reached any goal
        for goal_name, goal_pos in self.goals.items():
            dist = np.linalg.norm(self.agent_pos - goal_pos)
            if dist < self.goal_radius:
                reward = 1.0
                reached_goal = True
                reached_which = goal_name
                self.terminated = True
                break
        
        # Truncate if hit boundary or exceeded max steps
        if hit_boundary:
            self.truncated = True
        elif self.current_step >= self.max_steps:
            self.truncated = True
        
        # Compute distances to all goals for info
        distances = {}
        for goal_name, goal_pos in self.goals.items():
            distances[f'dist_{goal_name}'] = np.linalg.norm(self.agent_pos - goal_pos)
        
        # Info dictionary
        info = {
            'agent_pos': self.agent_pos.copy(),
            **distances,
            'reached_goal': reached_goal,
            'reached_which': reached_which,
            'hit_boundary': hit_boundary,
            'step': self.current_step
        }
        
        return self.agent_pos.copy(), reward, self.terminated, self.truncated, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: 'human' for display, 'rgb_array' for array output
            
        Returns:
            rgb_array if mode='rgb_array', None otherwise
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.ax.clear()
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Four-Goal Navigation Environment')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        
        # Draw boundary
        boundary = Rectangle((-1, -1), 2, 2, linewidth=2, 
                           edgecolor='black', facecolor='none')
        self.ax.add_patch(boundary)
        
        # Draw four goals with different colors
        goal_colors = {
            'top_left': 'red',
            'top_right': 'green',
            'bottom_left': 'blue',
            'bottom_right': 'orange'
        }
        for goal_name, goal_pos in self.goals.items():
            goal_circle = Circle(goal_pos, self.goal_radius,
                                color=goal_colors[goal_name], alpha=0.5, 
                                label=goal_name.replace('_', ' ').title())
            self.ax.add_patch(goal_circle)
        
        # Draw agent
        agent_circle = Circle(self.agent_pos, self.step_size/2,
                            color='purple', alpha=0.7, label='Agent')
        self.ax.add_patch(agent_circle)
        
        # Draw trajectory if exists
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'purple', alpha=0.5, linewidth=1)
            # Draw arrows
            for i in range(len(traj)-1):
                self.ax.arrow(traj[i, 0], traj[i, 1],
                            traj[i+1, 0]-traj[i, 0], traj[i+1, 1]-traj[i, 1],
                            head_width=0.05, head_length=0.05, fc='purple', ec='purple', alpha=0.3)
        
        self.ax.legend(loc='center left')
        
        if mode == 'rgb_array':
            self.fig.canvas.draw()
            # Get ARGB buffer (4 channels)
            buffer = self.fig.canvas.tostring_argb()
            img = np.frombuffer(buffer, dtype=np.uint8)
            
            # Get canvas dimensions
            width, height = self.fig.canvas.get_width_height()
            img = img.reshape((height, width, 4))
            
            # Convert ARGB to RGB (discard alpha channel)
            img = img[:, :, 1:]
            return img
        
        plt.tight_layout()
        plt.pause(0.001)
        return None
    
    def close(self):
        """Clean up rendering resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    

def run_random_rollouts(
    env_class,
    env_params: dict,
    num_episodes: int = 100,
    verbose: bool = False
):
    """
    Run random rollouts from the center starting point.
    
    Args:
        env_class: Environment class
        env_params: Environment parameters
        num_episodes: Number of random episodes to run
        verbose: Whether to print progress
        
    Returns:
        List of rollout results (trajectory, total_reward, info)
    """
    results = []
    
    print(f"Running {num_episodes} random rollouts...")
    
    for ep_idx in range(num_episodes):
        if verbose and ep_idx % 20 == 0:
            print(f"  Episode {ep_idx + 1}/{num_episodes}")
        
        # Create fresh environment for each episode
        env = env_class(**env_params)
        
        # Reset environment
        obs, _ = env.reset()
        trajectory = [obs.copy()]
        total_reward = 0.0
        terminated = False
        truncated = False
        reached_goal = None
        
        # Run until termination
        for step in range(env_params.get('max_steps', 30)):
            # Random action from uniform distribution
            action = np.random.uniform(-np.pi, np.pi, size=(1,))
            
            obs, reward, terminated, truncated, step_info = env.step(action)
            trajectory.append(obs.copy())
            total_reward += reward
            
            if terminated:
                reached_goal = step_info['reached_which']
                break
            if truncated:
                break
        
        results.append({
            'trajectory': trajectory,
            'total_reward': total_reward,
            'reached_goal': reached_goal,
            'final_pos': trajectory[-1],
            'trajectory_length': len(trajectory) - 1
        })
    
    return results


def plot_trajectories(
    results,
    env_params: dict,
    save_path: str = "figures/four_goal_random_trajs.png",
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    """
    Plot all random trajectories on a single figure.
    
    Args:
        results: List of random rollout results
        env_params: Environment parameters
        save_path: Path to save the figure
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
    
    # Goal positions and colors
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
    
    # Separate trajectories by goal reached
    trajectories_by_goal = {goal_name: [] for goal_name in goals.keys()}
    trajectories_to_none = []
    
    for result in results:
        traj = result['trajectory']
        reached_goal = result['reached_goal']
        
        if reached_goal in trajectories_by_goal:
            trajectories_by_goal[reached_goal].append(traj)
        else:
            trajectories_to_none.append(traj)
    
    # Plot trajectories to each goal with matching colors
    for goal_name, goal_info in goals.items():
        for traj in trajectories_by_goal[goal_name]:
            traj_array = np.array(traj)
            ax.plot(traj_array[:, 0], traj_array[:, 1],
                    color=goal_info['color'], alpha=0.5, linewidth=1.5)
    
    # Plot unsuccessful trajectories (gray)
    for traj in trajectories_to_none:
        traj_array = np.array(traj)
        ax.plot(traj_array[:, 0], traj_array[:, 1],
                'gray', alpha=0.5, linewidth=1.0, linestyle='--')
    
    # Calculate statistics
    num_experiments = len(results)
    total_success = sum(len(trajs) for trajs in trajectories_by_goal.values())
    success_rate = total_success / num_experiments
    avg_reward = np.mean([r['total_reward'] for r in results])
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='purple', marker='o', markersize=10,
               label='Start', markeredgecolor='black', linestyle='none'),
        Line2D([0], [0], color='gray', alpha=0.5, marker='o', markersize=15,
               label='Goals', linestyle='none'),
    ]
    
    # Add trajectory counts for each goal
    goal_labels = {
        'top_left': 'Top Left',
        'top_right': 'Top Right', 
        'bottom_left': 'Bottom Left',
        'bottom_right': 'Bottom Right'
    }
    for goal_name, goal_info in goals.items():
        count = len(trajectories_by_goal[goal_name])
        legend_elements.append(
            Line2D([0], [0], color=goal_info['color'], linewidth=1.5, alpha=0.5,
                   label=f'To {goal_labels[goal_name]} ({count})')
        )
    legend_elements.append(
        Line2D([0], [0], color='gray', linewidth=1.0, alpha=0.3, linestyle='--',
               label=f'No Goal ({len(trajectories_to_none)})')
    )
    
    ax.legend(handles=legend_elements, loc='center left', fontsize=10)
    
    # Set title
    title = (f"Four-Goal Random Trajectories (N={num_experiments})\n"
             f"Success Rate: {success_rate:.1%}, "
             f"Avg Reward: {avg_reward:.2f}")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Random trajectories saved to {save_path}")
    
    plt.show()


def print_env_info(env):
    """Print environment information and parameters."""
    print("\nEnvironment Info:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Goals:")
    for goal_name, goal_pos in env.goals.items():
        print(f"    {goal_name}: {goal_pos}")
    print(f"  Goal radius: {env.goal_radius}")
    print(f"  Step size: {env.step_size}")
    print(f"  Max steps: {env.max_steps}")
    print(f"  Step penalty: {env.step_penalty}")



def test_render_during_simulation(env):
    """Test render method during a live simulation."""
    print("\n" + "=" * 70)
    print("Testing render Method During Simulation")
    print("=" * 70)
    
    print("\nRunning a single episode with live rendering:")
    obs, info = env.reset(seed=999)
    
    print("Initial state:")
    env.render()
    plt.pause(1.0)
    
    # Optimal actions to reach top_right goal
    actions = [np.pi/4] * 12  # 45 degrees
    for step, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step([action])
        
        print(f"Step {step}: action={action:.2f} rad, pos=({obs[0]:.2f}, {obs[1]:.2f}), reward={reward:.2f}")
        env.render()
        plt.pause(0.5)
        
        if terminated:
            print(f"✓ Reached {info['reached_which']} goal!")
            break
        elif truncated:
            print(f"✗ Truncated")
            break
    
    plt.pause(2.0)
    plt.close('all')


def test_render_rgb_mode(env):
    """Test render method with rgb_array mode."""
    print("\n" + "=" * 70)
    print("Testing render with rgb_array Mode")
    print("=" * 70)
    
    env.reset(seed=777)
    obs, reward, terminated, truncated, info = env.step([np.pi/4])
    print(f"First step: pos=({obs[0]:.2f}, {obs[1]:.2f})")
    
    rgb_array = env.render(mode='rgb_array')
    print(f"RGB array shape: {rgb_array.shape}")
    print(f"RGB array dtype: {rgb_array.dtype}")
    print(f"RGB array min/max: {rgb_array.min()}/{rgb_array.max()}")
    
    if rgb_array is not None:
        import matplotlib.image as mpimg
        mpimg.imsave('figures/four_goal_render_rgb_array.png', rgb_array)
        print("RGB array saved to 'figures/four_goal_render_rgb_array.png'")


def test_gym_interface_with_render(env):
    """Test gym interface with optimal actions and rendering."""
    print("\n" + "=" * 70)
    print("Testing Gym Interface with Render")
    print("=" * 70)
    
    # Test optimal actions to each goal
    optimal_angles = {
        'top_left': 3*np.pi/4,      # 135 degrees
        'top_right': np.pi/4,       # 45 degrees
        'bottom_left': -3*np.pi/4,  # -135 degrees
        'bottom_right': -np.pi/4    # -45 degrees
    }
    
    for goal_name, theta in optimal_angles.items():
        env.reset()
        print(f"\n--- Direct to {goal_name} goal (theta={np.degrees(theta):.0f}°) ---")
        for step in range(15):
            obs, reward, terminated, truncated, info = env.step([theta])
            env.render()
            plt.pause(0.1)
            if terminated:
                print(f"✓ Reached {info['reached_which']} goal in {step+1} steps!")
                break
            elif truncated:
                print(f"✗ Truncated at step {step+1}")
                break
        plt.pause(0.5)
    
    plt.close('all')


def run_random_rollouts_visualization(env_params, num_episodes):
    """Run random rollouts visualization with statistics."""
    print("\n" + "=" * 70)
    print("RANDOM ROLLOUTS VISUALIZATION")
    print("=" * 70)
    
    random_results = run_random_rollouts(
        env_class=FourGoalNavigationEnv,
        env_params=env_params,
        num_episodes=num_episodes,
        verbose=True
    )
    
    # Count goal reaches
    goal_counts = {goal_name: 0 for goal_name in ['top_left', 'top_right', 'bottom_left', 'bottom_right']}
    none_count = 0
    
    for r in random_results:
        if r['reached_goal'] in goal_counts:
            goal_counts[r['reached_goal']] += 1
        else:
            none_count += 1
    
    total_success = sum(goal_counts.values())
    success_rate = total_success / len(random_results)
    
    print(f"\nRandom Rollout Statistics:")
    print(f"  Total episodes: {len(random_results)}")
    print(f"  Success rate: {success_rate:.1%}")
    for goal_name, count in goal_counts.items():
        print(f"  {goal_name}: {count}")
    print(f"  No goal: {none_count}")
    
    plot_trajectories(
        results=random_results,
        env_params=env_params,
        save_path="figures/four_goal_random_trajs.png",
        figsize=(12, 10)
    )


def main():
    """Test the environment with random actions."""
    print("=" * 70)
    print("Four-Goal Navigation Environment Test")
    print("=" * 70)
    
    ENV_PARAMS = {
        'step_size': 0.40,
        'goal_radius': 0.1,
        'max_steps': 10,
        'step_penalty': -0.01
    }
    num_episodes = 512
    
    env = FourGoalNavigationEnv(
        **ENV_PARAMS,
        # seed=42
    )
    
    print_env_info(env)
    
    # test_render_during_simulation(env)
    
    # test_render_rgb_mode(env)
    
    # test_gym_interface_with_render(env)
    
    run_random_rollouts_visualization(ENV_PARAMS, num_episodes)


if __name__ == "__main__":
    main()