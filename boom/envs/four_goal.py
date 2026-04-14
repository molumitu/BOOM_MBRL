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
        self._max_episode_steps = max_steps

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
            'step_size': self.step_size,
            'success': False  # For compatibility with BOOM
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
            'step': self.current_step,
            'success': reached_goal  # For compatibility with BOOM
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

        self.ax.legend(loc='upper right')

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


# BOOM-compatible wrapper
class FourGoalEnv:
    """Wrapper class for BOOM compatibility."""

    def __init__(self, cfg):
        # Extract parameters from config or use defaults
        step_size = getattr(cfg, 'step_size', 0.2)
        goal_radius = getattr(cfg, 'goal_radius', 0.15)
        max_steps = getattr(cfg, 'max_steps', 25)
        step_penalty = getattr(cfg, 'step_penalty', -0.01)

        self._env = FourGoalNavigationEnv(
            step_size=step_size,
            goal_radius=goal_radius,
            max_steps=max_steps,
            step_penalty=step_penalty
        )
        self.max_episode_steps = max_steps

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render(mode='rgb_array')

    def __getattr__(self, name):
        """Delegate any other attributes to the wrapped environment."""
        return getattr(self._env, name)


def make_env(cfg):
    """Create a four-goal navigation environment for BOOM."""
    return FourGoalEnv(cfg)


if __name__ == "__main__":
    # Test the environment
    env = FourGoalNavigationEnv(
        step_size=0.2,
        goal_radius=0.15,
        max_steps=25,
        step_penalty=-0.01,
        seed=42
    )

    print("Environment Info:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Goals:")
    for goal_name, goal_pos in env.goals.items():
        print(f"    {goal_name}: {goal_pos}")
    print(f"  Goal radius: {env.goal_radius}")
    print(f"  Step size: {env.step_size}")
    print(f"  Max steps: {env.max_steps}")

    # Run a few random episodes
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        print(f"\n=== Episode {ep + 1} ===")

        for step in range(env.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                print(f"✓ Reached {info['reached_which']} goal in {step + 1} steps!")
                break
            elif truncated:
                print(f"✗ Truncated at step {step + 1}")
                break
