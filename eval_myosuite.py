#!/usr/bin/env python3
"""
Generic evaluation script for MyoSuite environments.
Loads a trained model and saves video of the agent's performance.

Supports any MyoSuite task: myo-reach, myo-pose, myo-obj-hold, myo-pen-twirl, etc.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Set environment variables for rendering
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"  # Use EGL for offscreen rendering
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]

os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from omegaconf import OmegaConf
from boom.envs import make_env
from boom.boom_alg import BOOM


class VideoRecorder:
    """Video recorder that saves frames directly to video file."""

    def __init__(self, save_dir, episode_idx=0, fps=30):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.episode_idx = episode_idx
        self.fps = fps

    def init(self, env):
        """Initialize recording for an episode."""
        self.frames = []

    def record(self, env):
        """Record a frame from the environment."""
        frame = env.render()
        self.frames.append(frame)

    def save(self, step, key='video'):
        """Save frames as video file (mp4)."""
        if len(self.frames) == 0:
            return

        # Convert frames to numpy array
        frames_array = np.stack(self.frames, axis=0)  # (T, H, W, C)
        video_path = self.save_dir / f"{key}_ep{self.episode_idx}_step{step}.mp4"

        try:
            import imageio
            # Save as mp4 video
            imageio.mimsave(
                str(video_path),
                frames_array,
                fps=self.fps,
                quality=8,
                codec='libx264'
            )
            print(f"Saved video to {video_path}")
            print(f"  Shape: {frames_array.shape}, Duration: {len(self.frames)/self.fps:.2f}s")
        except ImportError:
            print("Error: imageio not installed. Install with: pip install imageio[ffmpeg]")
            print("Falling back to saving as npz...")
            npz_path = self.save_dir / f"{key}_ep{self.episode_idx}_step{step}.npz"
            np.savez_compressed(npz_path, frames=frames_array)
            print(f"Saved to {npz_path}")


@torch.no_grad()
def evaluate_model(model_path, config_path, task_name, num_episodes=5, save_dir="./eval_videos"):
    """
    Evaluate a trained model and save videos.

    Args:
        model_path: Path to the trained model checkpoint
        config_path: Path to the config file used for training
        task_name: Name of the MyoSuite task (e.g., 'myo-reach', 'myo-pose', 'myo-pen-twirl-hard')
        num_episodes: Number of episodes to evaluate
        save_dir: Directory to save videos
    """
    print("="*60)
    print(f"MyoSuite Model Evaluation: {task_name}")
    print("="*60)

    # Load config
    print(f"\nLoading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Override settings for evaluation
    cfg.task = task_name
    cfg.env_type = "myosuite"
    cfg.eval_episodes = num_episodes
    cfg.task_dim = 0

    # Disable struct mode to allow setting missing required values
    OmegaConf.set_struct(cfg, False)

    print(f"Task: {cfg.task}")
    print(f"Model path: {model_path}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Save directory: {save_dir}")

    # Create environment
    print("\nCreating environment...")
    env = make_env(cfg)
    print(f"Environment created: {cfg.task}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create agent and load model
    print("\nCreating agent...")
    agent = BOOM(cfg)
    print(f"Loading checkpoint from: {model_path}")
    agent.load(model_path)
    print("Model loaded successfully!")

    # Create video recorder
    video_recorder = VideoRecorder(save_dir)

    # Evaluate
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60)

    all_rewards = []
    all_episode_lengths = []

    for episode_idx in range(num_episodes):
        obs, done, ep_reward, t = env.reset()[0], False, 0, 0

        # Initialize video recording
        video_recorder.episode_idx = episode_idx
        video_recorder.init(env)

        print(f"\nEpisode {episode_idx + 1}/{num_episodes}")

        # Create progress bar for this episode
        max_steps = 1000  # Estimated max steps, will adjust if needed
        pbar = tqdm(total=max_steps, desc=f"  Steps", unit="step", ncols=100)

        while not done:
            # Get action from agent
            action, _, _ = agent.act(obs, t0=(t == 0), eval_mode=True)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            ep_reward += reward
            t += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'reward': f'{ep_reward:.1f}'})

            # Record frame
            video_recorder.record(env)

            # Adjust max steps if we exceed it
            if t >= pbar.total:
                pbar.total = t + 100

        pbar.close()

        # Save video for this episode
        video_recorder.save(episode_idx)

        all_rewards.append(ep_reward)
        all_episode_lengths.append(t)

        # Try to get success rate if available (some tasks have it)
        success = info.get("success", None)
        if success is not None:
            print(f"  ✓ Reward: {ep_reward:.2f}, Success: {success}, Steps: {t}")
        else:
            print(f"  ✓ Reward: {ep_reward:.2f}, Steps: {t}")

    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Mean reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"Mean episode length: {np.mean(all_episode_lengths):.1f} +/- {np.std(all_episode_lengths):.1f}")
    print(f"All rewards: {[f'{r:.2f}' for r in all_rewards]}")
    print(f"All lengths: {all_episode_lengths}")
    print(f"\nVideos saved to: {save_dir}")
    print("="*60)

    env.close()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate trained models on MyoSuite environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate myo-reach task
  python %(prog)s --task myo-reach --model logs/myo-reach/1/models/final.pt

  # Evaluate myo-pen-twirl-hard with 10 episodes
  python %(prog)s --task myo-pen-twirl-hard --model logs/myo-pen-twirl-hard/3/default/models/final.pt --episodes 10

  # Evaluate myo-obj-hold with custom save directory
  python %(prog)s --task myo-obj-hold --model logs/myo-obj-hold/2/models/final.pt --save-dir ./my_videos

Supported tasks:
  Hand tasks:
    - myo-reach, myo-reach-hard (hand reach tasks)
    - myo-pose, myo-pose-hard (hand pose tasks)
    - myo-obj-hold, myo-obj-hold-hard (object holding tasks)
    - myo-key-turn, myo-key-turn-hard (key turning tasks)
    - myo-pen-twirl, myo-pen-twirl-hard (pen twirling tasks)

  Arm tasks:
    - myo-arm-reach, myo-arm-reach-hard (arm reach tasks)

  See myosuite_tasks.md for detailed task information.
        """
    )
    parser.add_argument(
        "--task",
        type=str,
        default="myo-reach",
        help="Name of the MyoSuite task (e.g., 'myo-reach', 'myo-pose', 'myo-pen-twirl-hard')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logs/myo-reach/5/default/models/step_40000.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="boom/config.yaml",
        help="Path to config file (default: boom/config.yaml)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to evaluate (default: 3)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./eval_videos",
        help="Directory to save videos (default: ./eval_videos)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Create custom save directory based on task name
    task_save_dir = os.path.join(args.save_dir, args.task)

    # Run evaluation
    evaluate_model(
        model_path=args.model,
        config_path=args.config,
        task_name=args.task,
        num_episodes=args.episodes,
        save_dir=task_save_dir
    )


if __name__ == "__main__":
    main()
