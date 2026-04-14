#!/usr/bin/env python3
"""
Generate evaluation videos for MyoSuite and dm_control tasks.

Supports multiple algorithms and task groups with automatic model finding
(for MyoSuite) or manual model specification (for dm_control).
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse

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


# MyoSuite tasks to evaluate
MYO_TASKS = [
    "myo-reach",
    "myo-reach-hard",
    "myo-key-turn",
    "myo-key-turn-hard",
    "myo-pen-twirl",
    "myo-pen-twirl-hard",
    "myo-obj-hold",
    "myo-obj-hold-hard",
    "myo-pose",
    "myo-pose-hard",
]


class VideoRecorder:
    """Video recorder that saves frames directly to video file."""

    def __init__(self, save_dir, task_name, episode_idx=0, fps=30):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.task_name = task_name
        self.episode_idx = episode_idx
        self.fps = fps

    def init(self, env):
        """Initialize recording for an episode."""
        self.frames = []

    def record(self, env):
        """Record a frame from the environment."""
        frame = env.render()
        self.frames.append(frame)

    def save(self, reward):
        """Save frames as video file (mp4)."""
        if len(self.frames) == 0:
            return

        # Convert frames to numpy array
        frames_array = np.stack(self.frames, axis=0)  # (T, H, W, C)
        video_path = self.save_dir / f"{self.task_name}_ep{self.episode_idx}_reward_{reward:.0f}.mp4"

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
            npz_path = self.save_dir / f"{self.task_name}_ep{self.episode_idx}_reward_{reward:.0f}.npz"
            np.savez_compressed(npz_path, frames=frames_array)
            print(f"Saved to {npz_path}")


def find_best_model(log_dir: Path) -> Optional[Path]:
    """
    Find the best available model in a directory.
    Prefers 1M steps, then final.pt, then the largest step checkpoint.

    Args:
        log_dir: Directory containing model checkpoints

    Returns:
        Path to the best model file, or None if no models found
    """
    models_dir = log_dir / "models"
    if not models_dir.exists():
        return None

    # Prefer 1M step model
    model_1m = models_dir / "step_1000000.pt"
    if model_1m.exists():
        return model_1m

    # Then try final.pt
    final_model = models_dir / "final.pt"
    if final_model.exists():
        return final_model

    # Otherwise, find the largest step checkpoint
    step_models = list(models_dir.glob("step_*.pt"))
    if step_models:
        # Sort by step number (extract from filename)
        def extract_step(path):
            try:
                return int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                return 0

        step_models.sort(key=extract_step, reverse=True)
        return step_models[0]

    return None


def find_flow_models(task: str, seed: int = 11) -> Optional[Tuple[Path, Path]]:
    """
    Find flow models for a task.

    Args:
        task: Task name (e.g., 'myo-reach')
        seed: Random seed to use (default: 11)

    Returns:
        Tuple of (model_path, config_path) or None if not found
    """
    task_dir = Path("logs") / task / str(seed)
    if not task_dir.exists():
        return None

    # Find flow directories
    flow_dirs = sorted(task_dir.glob("*-flow"), reverse=True)
    if not flow_dirs:
        return None

    # Use the latest flow directory
    latest_flow = flow_dirs[0]

    # Find best model
    model_path = find_best_model(latest_flow)
    if model_path is None:
        return None

    # Use the main config file instead of wandb config
    config_path = Path("boom/config.yaml")
    if not config_path.exists():
        return None

    return model_path, config_path


def find_boom_models(task: str, seed: int = 11) -> Optional[Tuple[Path, Path]]:
    """
    Find boom (non-flow) models for a task.

    Args:
        task: Task name (e.g., 'myo-reach')
        seed: Random seed to use (default: 11)

    Returns:
        Tuple of (model_path, config_path) or None if not found
    """
    task_dir = Path("logs") / task / str(seed)
    if not task_dir.exists():
        return None

    # Find non-flow directories (raw or no suffix)
    boom_dirs = []
    for d in task_dir.iterdir():
        if d.is_dir() and not d.name.endswith("-flow"):
            boom_dirs.append(d)

    if not boom_dirs:
        return None

    # Use the latest directory
    latest_boom = sorted(boom_dirs, reverse=True)[0]

    # Find best model
    model_path = find_best_model(latest_boom)
    if model_path is None:
        return None

    # Use the main config file instead of wandb config
    config_path = Path("boom/config.yaml")
    if not config_path.exists():
        return None

    return model_path, config_path


@torch.no_grad()
def evaluate_model(
    model_path: Path,
    config_path: Path,
    task_name: str,
    env_type: str,
    num_episodes: int = 5,
    save_dir: str = "./eval_videos"
):
    """
    Evaluate a trained model and save videos.

    Args:
        model_path: Path to the trained model checkpoint
        config_path: Path to the config file used for training
        task_name: Name of the task (e.g., 'myo-reach', 'dog-run')
        env_type: Environment type ('myosuite' or 'dm_control')
        num_episodes: Number of episodes to evaluate
        save_dir: Directory to save videos
    """
    from omegaconf import OmegaConf
    from boom.envs import make_env
    from boom.boom_alg import BOOM

    env_name = "MyoSuite" if env_type == "myosuite" else "DM Control"
    print("="*60)
    print(f"{env_name} Model Evaluation: {task_name}")
    print("="*60)

    # Load config
    print(f"\nLoading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Override settings for evaluation
    cfg.task = task_name
    cfg.env_type = env_type
    cfg.eval_episodes = num_episodes
    cfg.task_dim = 0

    # Disable struct mode to allow setting missing required values
    OmegaConf.set_struct(cfg, False)

    print(f"Task: {cfg.task}")
    print(f"Environment type: {cfg.env_type}")
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
    video_recorder = VideoRecorder(save_dir, task_name)

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
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_steps, desc=f"  Steps", unit="step", ncols=100)
        except ImportError:
            pbar = None

        while not done:
            # Get action from agent
            result = agent.act(obs, t0=(t == 0), eval_mode=True)

            # Handle different return types for myosuite vs dm_control
            if env_type == "myosuite":
                action = result[0]  # MyoSuite returns (action, ...)
            else:
                action = result[0] if isinstance(result, tuple) else result

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            ep_reward += reward
            t += 1

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({'reward': f'{ep_reward:.1f}'})
                # Adjust max steps if we exceed it
                if t >= pbar.total:
                    pbar.total = t + 100

            # Record frame
            video_recorder.record(env)

        if pbar is not None:
            pbar.close()

        # Save video for this episode
        video_recorder.save(ep_reward)

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

    # # Close environment if it has a close method
    # if hasattr(env, 'close'):
    #     env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation videos for MyoSuite and dm_control tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate videos for MyoSuite MFP models (default)
  python %(prog)s --tasks myo --algorithm mfp

  # Generate videos for MyoSuite BOOM models
  python %(prog)s --tasks myo --algorithm boom

  # Generate videos for MyoSuite with 5 episodes each
  python %(prog)s --tasks myo --algorithm mfp --episodes 5

  # Generate video for a specific MyoSuite task
  python %(prog)s --tasks myo --algorithm mfp --task myo-reach-hard

  # Generate videos for dm_control tasks
  python %(prog)s --tasks dmc --algorithm boom --task dog-run --model logs/dog-run/1/models/final.pt

  # Generate dm_control videos with 10 episodes
  python %(prog)s --tasks dmc --algorithm boom --task dog-run --model logs/dog-run/1/models/final.pt --episodes 10
        """
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default="myo",
        choices=["myo", "dmc"],
        help="Task group to evaluate (default: myo)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="mfp",
        help="Algorithm name for directory naming (default: mfp)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to evaluate (default: all tasks in group)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (required for dmc tasks)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="boom/config.yaml",
        help="Path to config file (default: boom/config.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed for MyoSuite model finding (default: 11)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes per task (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_videos",
        help="Base output directory (default: eval_videos)",
    )

    args = parser.parse_args()

    # Validate arguments for dmc tasks
    if args.tasks == "dmc" and args.model is None:
        parser.error("--model is required for dmc tasks")

    # Set output directory: eval_videos/{tasks}/{algorithm}/
    output_dir = Path(args.output_dir) / args.tasks / args.algorithm

    print(f"\nGenerating videos for {args.tasks.upper()} task group")
    print(f"Algorithm: {args.algorithm}")
    print(f"Output directory: {output_dir}")
    print(f"Episodes per task: {args.episodes}")

    if args.tasks == "myo":
        print(f"Seed: {args.seed}")

    # Process based on task group
    if args.tasks == "myo":
        # Determine which tasks to process
        if args.task:
            tasks_to_process = [args.task]
        else:
            tasks_to_process = MYO_TASKS

        # Choose model finder based on algorithm
        if args.algorithm == "mfp":
            find_models = find_flow_models
        else:
            find_models = find_boom_models

        print(f"\nProcessing {len(tasks_to_process)} MyoSuite tasks...")

        # Process each task
        results = {}
        for task in tasks_to_process:
            print(f"\n[{tasks_to_process.index(task) + 1}/{len(tasks_to_process)}] Checking {task}...")

            # Find models
            model_info = find_models(task, args.seed)
            if model_info is None:
                print(f"  ⊘ No models found for {task}, skipping")
                results[task] = "skipped"
                continue

            model_path, config_path = model_info
            print(f"  ✓ Found model: {model_path.name}")

            # Generate video (output_dir already includes tasks/algorithm)
            try:
                evaluate_model(
                    model_path=model_path,
                    config_path=config_path,
                    task_name=task,
                    env_type="myosuite",
                    num_episodes=args.episodes,
                    save_dir=str(output_dir)
                )
                results[task] = "success"
            except Exception as e:
                print(f"✗ Failed to generate video for {task}: {e}")
                results[task] = "failed"

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for task, status in results.items():
            symbol = {
                "success": "✓",
                "failed": "✗",
                "skipped": "⊘",
            }[status]
            print(f"{symbol} {task}: {status}")

        success_count = sum(1 for s in results.values() if s == "success")
        failed_count = sum(1 for s in results.values() if s == "failed")
        skipped_count = sum(1 for s in results.values() if s == "skipped")

        print(f"\nTotal: {len(results)} tasks")
        print(f"  Success: {success_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"{'='*60}\n")

    else:  # args.tasks == "dmc"
        # Determine which task to process
        if args.task is None:
            parser.error("--task is required for dmc tasks")

        # Check if model exists
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            sys.exit(1)

        # Run evaluation (output_dir already includes tasks/algorithm)
        evaluate_model(
            model_path=args.model,
            config_path=args.config,
            task_name=args.task,
            env_type="dm_control",
            num_episodes=args.episodes,
            save_dir=str(output_dir)
        )


if __name__ == "__main__":
    main()
