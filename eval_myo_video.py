#!/usr/bin/env python3
"""
Generate evaluation videos for MyoSuite tasks.

Automatically finds the best available models (preferring 1M steps) and
generates videos for all available tasks.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse


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


def generate_video(
    task: str,
    model_path: Path,
    config_path: Path,
    output_dir: Path,
    num_episodes: int = 1,
):
    """
    Generate video for a task using eval_myosuite.py.

    Args:
        task: Task name
        model_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Directory to save videos
        num_episodes: Number of episodes to render
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "eval_myosuite.py",
        "--task", task,
        "--model", str(model_path),
        "--config", str(config_path),
        "--episodes", str(num_episodes),
        "--save-dir", str(output_dir),
    ]

    print(f"\n{'='*60}")
    print(f"Generating video for: {task}")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Successfully generated video for {task}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate video for {task}: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation videos for MyoSuite tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate videos for MFP models (default)
  python %(prog)s --algorithm mfp

  # Generate videos for BOOM models
  python %(prog)s --algorithm boom

  # Generate videos with 5 episodes each
  python %(prog)s --algorithm mfp --episodes 5

  # Generate video for a specific task
  python %(prog)s --algorithm mfp --task myo-reach-hard
        """
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="mfp",
        choices=["mfp", "boom"],
        help="Algorithm to evaluate (default: mfp)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to evaluate (default: all tasks)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed (default: 11)",
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

    # Determine which tasks to process
    if args.task:
        tasks_to_process = [args.task]
    else:
        tasks_to_process = MYO_TASKS

    # Set output directory
    output_dir = Path(args.output_dir) / args.algorithm

    # Choose model finder based on algorithm
    if args.algorithm == "mfp":
        find_models = find_flow_models
    else:  # boom
        find_models = find_boom_models

    print(f"\nGenerating videos for {args.algorithm.upper()} algorithm")
    print(f"Output directory: {output_dir}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Seed: {args.seed}")

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

        # Generate video
        success = generate_video(
            task=task,
            model_path=model_path,
            config_path=config_path,
            output_dir=output_dir,
            num_episodes=args.episodes,
        )

        results[task] = "success" if success else "failed"

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


if __name__ == "__main__":
    main()
