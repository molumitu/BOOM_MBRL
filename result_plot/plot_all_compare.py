import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import os


# Algorithm directory names and their display names
ALGORITHMS = {
    "mfp": "MFP(Ours)",
    "boom": "BOOM",
    "dreamerv3": "DreamerV3",
    "sac": "SAC",
    "tdmpc": "TDMPC",
    "tdmpc2": "TDMPC2",
}

# Colors for each algorithm
ALGORITHM_COLORS = {
    "MFP(Ours)": "#d62728", # red
    "BOOM": "#1f77b4",      # blue
    "DreamerV3": "#2ca02c", # green
    "SAC": "#ff7f0e",       # orange
    "TDMPC": "#9467bd",     # purple
    "TDMPC2": "#8c564b",    # brown
}


def get_task_filename(task_name: str, algorithm: str) -> str:
    """
    Map task name to filename for different algorithms.

    For boom/mfp: use short names (myo-reach)
    For sac/tdmpc/tdmpc2: use names with 'hand' (myo-hand-reach)
    """
    if algorithm in ["boom", "mfp"]:
        return task_name
    else:
        # Insert 'hand' after 'myo-'
        if task_name.startswith("myo-"):
            return task_name.replace("myo-", "myo-hand-")
        return task_name


def load_algorithm_data(
    base_dir: Path,
    algorithm: str,
    task_name: str,
    value_column: str = "success",
) -> Optional[pd.DataFrame]:
    """
    Load data for a specific algorithm and task.

    Args:
        base_dir: Base directory containing algorithm folders
        algorithm: Algorithm directory name
        task_name: Task name (e.g., 'myo-reach')
        value_column: Column name for the value to plot ('success' or 'reward')

    Returns:
        DataFrame with columns [step, value_column, seed] or None if file not found
    """
    filename = get_task_filename(task_name, algorithm)
    filepath = base_dir / algorithm / f"{filename}.csv"

    if not filepath.exists():
        return None

    try:
        df = pd.read_csv(filepath)
        # Ensure required columns exist
        if "step" not in df.columns or value_column not in df.columns:
            return None
        return df
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def compute_mean_std(
    dfs: List[pd.DataFrame],
    value_column: str = "success",
) -> Optional[pd.DataFrame]:
    """
    Compute mean and std for multiple DataFrames with different seeds.

    Args:
        dfs: List of DataFrames with columns [step, value_column, seed]
        value_column: Column name for the value to plot ('success' or 'reward')

    Returns:
        DataFrame with columns [step, mean, std]
    """
    if not dfs:
        return None

    # Combine all data
    all_data = []
    for df in dfs:
        if df is not None and len(df) > 0:
            all_data.append(df)

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)

    # Group by step and compute mean/std
    grouped = combined.groupby("step")[value_column].agg(["mean", "std"]).reset_index()
    grouped.columns = ["step", "mean", "std"]

    return grouped


def plot_single_task(
    task_name: str,
    base_dir: Path,
    algorithms: List[str],
    output_dir: Path,
    max_step: float = 1e6,
    value_column: str = "success",
    ylabel: str = "Success",
):
    """
    Plot a single task comparison.

    Args:
        task_name: Task name (e.g., 'myo-reach')
        base_dir: Base directory containing algorithm folders
        algorithms: List of algorithm directory names to plot
        output_dir: Directory to save the plot
        max_step: Maximum step to consider (default: 1e6)
        value_column: Column name for the value to plot ('success' or 'reward')
        ylabel: Label for y-axis
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    has_data = False

    for algo_dir in algorithms:
        algo_name = ALGORITHMS.get(algo_dir, algo_dir)
        if algo_name not in ALGORITHM_COLORS:
            continue

        # Load data
        df = load_algorithm_data(base_dir, algo_dir, task_name, value_column=value_column)
        if df is None:
            continue

        # Compute mean and std
        stats = compute_mean_std([df], value_column=value_column)
        if stats is None:
            continue

        # Filter by max_step
        stats = stats[stats["step"] <= max_step]
        if len(stats) == 0:
            continue

        has_data = True

        # Plot mean line
        color = ALGORITHM_COLORS[algo_name]
        ax.plot(
            stats["step"],
            stats["mean"],
            color=color,
            linewidth=2,
            label=algo_name,
        )

        # Plot std as shaded area
        ax.fill_between(
            stats["step"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=color,
            alpha=0.2,
        )

    if has_data:
        ax.set_xlabel("Step", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(task_name.replace("-", " ").title(), fontsize=16)
        ax.grid(True, alpha=0.3)
        if value_column == "success":
            ax.set_ylim([0, 1])

        # Save without legend
        output_path = output_dir / f"{task_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {output_path}")

    plt.close()


def plot_legend(output_dir: Path):
    """
    Create a separate legend figure.

    Args:
        output_dir: Directory to save legend
    """
    fig, ax = plt.subplots(figsize=(12, 1.5))
    ncol = 6

    # Create dummy lines for legend
    for algo_name, color in ALGORITHM_COLORS.items():
        ax.plot(
            [],
            [],
            color=color,
            linewidth=2,
            label=algo_name,
        )

    # Add legend
    ax.legend(
        loc="center",
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=ncol,
    )

    # Hide axes
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis("off")

    # Save legend
    output_path = output_dir / "legend.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {output_path}")

    plt.close()


def plot_gathered_tasks(
    task_names: List[str],
    base_dir: Path,
    algorithms: List[str],
    output_dir: Path,
    max_step: float = 1e6,
    value_column: str = "success",
    ylabel: str = "Success",
):
    """
    Plot all tasks in a single gathered figure.

    Args:
        task_names: List of task names
        base_dir: Base directory containing algorithm folders
        algorithms: List of algorithm directory names to plot
        output_dir: Directory to save the plot
        max_step: Maximum step to consider (default: 1e6)
        value_column: Column name for the value to plot ('success' or 'reward')
        ylabel: Label for y-axis
    """
    n_tasks = len(task_names)

    # Determine layout: if <=5 tasks, 1 row; otherwise max 5 columns per row
    if n_tasks <= 5:
        n_cols = n_tasks
        n_rows = 1
    else:
        n_cols = 5
        n_rows = (n_tasks + n_cols - 1) // n_cols  # Round up

    # Create figure with n_rows for tasks + 1 row for legend
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows + 1.5))
    height_ratios = [1] * n_rows + [0.15]
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=height_ratios, hspace=0.3, wspace=0.3)

    # Plot each task
    for idx, task_name in enumerate(task_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        has_data = False

        for algo_dir in algorithms:
            algo_name = ALGORITHMS.get(algo_dir, algo_dir)
            if algo_name not in ALGORITHM_COLORS:
                continue

            # Load data
            df = load_algorithm_data(base_dir, algo_dir, task_name, value_column=value_column)
            if df is None:
                continue

            # Compute mean and std
            stats = compute_mean_std([df], value_column=value_column)
            if stats is None:
                continue

            # Filter by max_step
            stats = stats[stats["step"] <= max_step]
            if len(stats) == 0:
                continue

            has_data = True

            # Plot mean line
            color = ALGORITHM_COLORS[algo_name]
            ax.plot(
                stats["step"],
                stats["mean"],
                color=color,
                linewidth=2,
                label=algo_name,
            )

            # Plot std as shaded area
            ax.fill_between(
                stats["step"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                color=color,
                alpha=0.2,
            )

        if has_data:
            ax.set_xlabel("Step", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(task_name.replace("-", " ").title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            if value_column == "success":
                ax.set_ylim([0, 1])
            ax.tick_params(labelsize=8)

    # Create legend in the bottom row
    legend_ax = fig.add_subplot(gs[n_rows, :])
    legend_ax.axis("off")

    # Create dummy lines for legend
    for algo_name, color in ALGORITHM_COLORS.items():
        legend_ax.plot(
            [],
            [],
            color=color,
            linewidth=2,
            label=algo_name,
        )

    # Add legend
    legend_ax.legend(
        loc="center",
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=6,
    )

    # Save figure
    output_path = output_dir / "gathered_tasks.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {output_path}")

    plt.close()


def plot_all_compare(
    tasks: str = "myo",
    base_dir: str = "result_plot",
    output_dir: str = "result_plot/figure",
    max_step: float = 1e6,
    gather: bool = False,
):
    """
    Plot all task comparisons.

    Args:
        tasks: Task type to plot: 'myo' or 'dmc' (default: "myo")
        base_dir: Base directory containing algorithm folders
        output_dir: Directory to save plots
        max_step: Maximum step to consider (default: 1e6)
        gather: If True, plot all tasks in a single figure (default: False)
    """
    base_path = Path(base_dir)

    # Determine value column and ylabel based on task type
    if tasks == "myo":
        value_column = "success"
        ylabel = "Success"
    elif tasks == "dmc":
        value_column = "reward"
        ylabel = "Reward"
    else:
        raise ValueError(f"Unknown task type: {tasks}. Must be 'myo' or 'dmc'.")

    print(f"\nProcessing task group: {tasks}")

    # Get task names from both boom and mfp directories (union)
    task_names = set()

    for algo_dir in ["boom", "mfp"]:
        task_dir = base_path / tasks / algo_dir
        if task_dir.exists():
            csv_files = list(task_dir.glob("*.csv"))
            task_names.update([f.stem for f in csv_files])

    task_names = sorted(list(task_names))

    if not task_names:
        print(f"Warning: No tasks found in {base_path / tasks / 'boom'} or {base_path / tasks / 'mfp'}")
        return

    print(f"Found {len(task_names)} tasks")

    # Create output directory
    task_output_dir = Path(output_dir) / tasks
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Plot based on gather mode
    algorithms = ["boom", "mfp", "dreamerv3", "sac", "tdmpc", "tdmpc2"]

    if gather:
        # Gather all tasks in one figure
        print("Plotting gathered tasks...")
        plot_gathered_tasks(
            task_names=task_names,
            base_dir=base_path / tasks,
            algorithms=algorithms,
            output_dir=task_output_dir,
            max_step=max_step,
            value_column=value_column,
            ylabel=ylabel,
        )
    else:
        # Plot each task separately
        for task_name in task_names:
            print(f"Plotting {task_name}...")
            plot_single_task(
                task_name=task_name,
                base_dir=base_path / tasks,
                algorithms=algorithms,
                output_dir=task_output_dir,
                max_step=max_step,
                value_column=value_column,
                ylabel=ylabel,
            )

        # Create legend
        print("Creating legend...")
        plot_legend(task_output_dir)

    print(f"Completed! Results saved to {task_output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot task comparison across algorithms")
    parser.add_argument(
        "--tasks",
        default="myo",
        choices=["myo", "dmc"],
        help="Task type to plot: 'myo' or 'dmc' (default: myo)",
    )
    parser.add_argument(
        "--base-dir",
        default="result_plot",
        help="Base directory containing algorithm folders",
    )
    parser.add_argument(
        "--output-dir",
        default="result_plot/figure",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=1e6,
        help="Maximum step to consider (default: 1e6)",
    )
    parser.add_argument(
        "--gather",
        action="store_true",
        help="Plot all tasks in a single gathered figure",
    )

    args = parser.parse_args()

    plot_all_compare(
        tasks=args.tasks,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        max_step=args.max_step,
        gather=args.gather,
    )
