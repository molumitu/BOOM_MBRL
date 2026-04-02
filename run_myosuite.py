#!/usr/bin/env python3
"""
Simple script to run MyoSuite environments and save videos.
Uses random actions by default (no trained model needed).
"""

import os
import sys
import numpy as np
from pathlib import Path

# Set environment variables for rendering
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]

os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

# Import myosuite FIRST to register all environments
import myosuite

import gymnasium as gym
from boom.envs.myosuite import MYOSUITE_TASKS, MyoSuiteWrapper


class VideoRecorder:
    """Video recorder that saves frames to video file."""

    def __init__(self, save_path, fps=30):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.fps = fps

    def init(self, env):
        """Initialize recording."""
        self.frames = []

    def record(self, env):
        """Record a frame from the environment."""
        frame = env.render()
        self.frames.append(frame)

    def save(self):
        """Save frames as video file."""
        if len(self.frames) == 0:
            return

        frames_array = np.stack(self.frames, axis=0)

        try:
            import imageio
            imageio.mimsave(
                str(self.save_path),
                frames_array,
                fps=self.fps,
                quality=8,
                codec='libx264'
            )
            print(f"✓ Video saved: {self.save_path}")
            print(f"  Shape: {frames_array.shape}, Duration: {len(self.frames)/self.fps:.2f}s")
        except ImportError:
            print("Warning: imageio not installed. Saving as npz...")
            npz_path = self.save_path.with_suffix('.npz')
            np.savez_compressed(npz_path, frames=frames_array)
            print(f"✓ Saved: {npz_path}")


def run_env(task_name, num_episodes=3, max_steps=200, seed=None, save_dir="./videos"):
    """
    Run a MyoSuite environment with random actions and save videos.

    Args:
        task_name: Name of the task (e.g., 'myo-reach', 'myo-pose', 'myo-pen-twirl-hard')
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        save_dir: Directory to save videos
    """
    print("=" * 60)
    print(f"MyoSuite Runner: {task_name}")
    print("=" * 60)

    # Check if task exists
    if task_name not in MYOSUITE_TASKS:
        print(f"Error: Unknown task '{task_name}'")
        print(f"Available tasks: {list(MYOSUITE_TASKS.keys())}")
        sys.exit(1)

    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed: {seed}")

    # Create environment
    print(f"\nCreating environment: {MYOSUITE_TASKS[task_name]}")
    env = gym.make(MYOSUITE_TASKS[task_name])
    env = MyoSuiteWrapper(env, type('obj', (object,), {'obs': 'state', 'task': task_name}))
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max episode steps: {env._max_episode_steps}")

    # Create save directory
    save_dir = Path(save_dir) / task_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run episodes
    print("\n" + "=" * 60)
    print(f"Running {num_episodes} episodes with random actions...")
    print("=" * 60)

    all_rewards = []
    all_lengths = []

    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=seed if seed is not None else None)
        done, ep_reward, t = False, 0, 0

        # Create video recorder
        video_path = save_dir / f"episode_{episode_idx + 1}.mp4"
        recorder = VideoRecorder(video_path)
        recorder.init(env)

        print(f"\nEpisode {episode_idx + 1}/{num_episodes}")

        while not done and t < max_steps:
            # Random action
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            t += 1

            # Record frame
            recorder.record(env)

            print(f"  Step {t}: reward={reward:.3f}, total={ep_reward:.2f}", end='\r')

        # Save video
        recorder.save()

        all_rewards.append(ep_reward)
        all_lengths.append(t)

        success = info.get("solved", False)
        print(f"\n  ✓ Reward: {ep_reward:.2f}, Steps: {t}, Success: {success}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Mean reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"Mean length: {np.mean(all_lengths):.1f} +/- {np.std(all_lengths):.1f}")
    print(f"Videos saved to: {save_dir}")
    print("=" * 60)

    env.close()


def main():
    import argparse

    # Get available tasks
    available_tasks = sorted(MYOSUITE_TASKS.keys())

    parser = argparse.ArgumentParser(
        description="Run MyoSuite environments with random actions and save videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run myo-reach task
  python %(prog)s --task myo-reach

  # Run myo-pen-twirl-hard for 5 episodes
  python %(prog)s --task myo-pen-twirl-hard --episodes 5

  # Run with custom settings
  python %(prog)s --task myo-obj-hold --episodes 3 --max-steps 300 --seed 42

Available tasks:
  {', '.join(available_tasks)}
        """
    )

    parser.add_argument(
        "--task",
        type=str,
        default="myo-reach",
        help=f"Task name (default: myo-reach). Available: {', '.join(available_tasks[:5])}, ...)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./videos",
        help="Directory to save videos (default: ./videos)"
    )

    args = parser.parse_args()

    # Validate task
    if args.task not in MYOSUITE_TASKS:
        print(f"Error: Unknown task '{args.task}'")
        print(f"\nAvailable tasks:")
        for task in available_tasks:
            print(f"  - {task}")
        sys.exit(1)

    # Run environment
    run_env(
        task_name=args.task,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
