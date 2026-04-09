#!/usr/bin/env python3
"""
Analysis script for num_flow_trajs ablation study.

This script reads evaluation results from all experiments and generates:
1. A CSV file with aggregated results
2. A plot showing TAR (episode_reward) vs iterations for each num_flow_trajs value

Usage:
    python analyze_ablation.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wandb


# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TASK = "dog-run"
ENV_TYPE = "dm_control"
SEED = 1
NUM_FLOW_TRAJS_VALUES = [12, 24, 48, 72]
WANDB_PROJECT = "shallowdream0745-thu/flow"

# Output files
CSV_OUTPUT = SCRIPT_DIR / "ablation_results.csv"
PLOT_OUTPUT = SCRIPT_DIR / "ablation_results.png"


def fetch_wandb_eval_data(num_flow_trajs: int) -> pd.DataFrame:
    """
    Fetch evaluation data from wandb for a given experiment.

    Args:
        num_flow_trajs: The num_flow_trajs value to fetch data for

    Returns:
        DataFrame with columns: step, episode_reward
    """
    exp_name = f"num_flow_trajs_{num_flow_trajs}"

    # Initialize wandb API
    api = wandb.Api()

    # Query runs with filters
    runs = api.runs(
        WANDB_PROJECT,
        filters={
            "config.task": TASK,
            "config.seed": SEED,
            "config.exp_name": exp_name,
            "state": "finished"  # Only get completed runs
        }
    )

    if not runs:
        raise ValueError(
            f"No wandb runs found for num_flow_trajs={num_flow_trajs} "
            f"(exp_name={exp_name}, task={TASK}, seed={SEED})"
        )

    # Use the most recent run if there are multiple
    run = runs[-1]
    print(f"Found run: {run.name} (ID: {run.id})")

    # Fetch eval_episode_reward history
    history = run.history(keys=["eval/episode_reward"])

    if history.empty:
        raise ValueError(
            f"No eval/episode_reward data found in run {run.id}"
        )

    # Rename columns to match expected format
    # wandb uses "_step" for step and "eval/episode_reward" for the metric
    df = pd.DataFrame({
        'step': history['_step'],
        'episode_reward': history['eval/episode_reward']
    })

    # Remove any rows with NaN values
    df = df.dropna()

    return df


def load_eval_data(num_flow_trajs: int) -> pd.DataFrame:
    """Load evaluation data for a single experiment from wandb."""
    try:
        df = fetch_wandb_eval_data(num_flow_trajs)

        # Ensure the columns exist
        if 'step' not in df.columns or 'episode_reward' not in df.columns:
            raise ValueError(f"Data from wandb missing required columns")

        df['num_flow_trajs'] = num_flow_trajs
        return df
    except Exception as e:
        print(f"Error loading data for num_flow_trajs={num_flow_trajs}: {e}")
        return pd.DataFrame()


def aggregate_results() -> pd.DataFrame:
    """
    Aggregate results from all experiments.

    Returns:
        DataFrame with columns: num_flow_trajs, iteration, TAR
    """
    all_data = []

    for num_flow_trajs in NUM_FLOW_TRAJS_VALUES:
        print(f"Loading data for num_flow_trajs={num_flow_trajs}...")
        df = load_eval_data(num_flow_trajs)

        if not df.empty:
            # Rename columns to match expected output format
            df = df.rename(columns={
                'step': 'iteration',
                'episode_reward': 'TAR'
            })

            # Select only the columns we need
            df = df[['num_flow_trajs', 'iteration', 'TAR']]
            all_data.append(df)
        else:
            print(f"Warning: No data found for num_flow_trajs={num_flow_trajs}")

    if not all_data:
        raise ValueError("No data could be loaded from any experiment!")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def save_csv_results(df: pd.DataFrame, output_path: Path):
    """Save aggregated results to CSV."""
    df = df.sort_values(['num_flow_trajs', 'iteration'])
    df.to_csv(output_path, index=False)
    print(f"CSV results saved to: {output_path}")


def plot_results(df: pd.DataFrame, output_path: Path):
    """
    Plot TAR vs iteration for each num_flow_trajs value.

    Args:
        df: DataFrame with columns num_flow_trajs, iteration, TAR
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))

    # Define colors for each line
    colors = plt.cm.viridis(np.linspace(0, 1, len(NUM_FLOW_TRAJS_VALUES)))

    for i, num_flow_trajs in enumerate(NUM_FLOW_TRAJS_VALUES):
        data = df[df['num_flow_trajs'] == num_flow_trajs]

        if data.empty:
            print(f"Warning: No data to plot for num_flow_trajs={num_flow_trajs}")
            continue

        # Sort by iteration to ensure proper line plotting
        data = data.sort_values('iteration')

        plt.plot(
            data['iteration'],
            data['TAR'],
            marker='o',
            markersize=4,
            linewidth=2,
            label=f'num_flow_trajs={num_flow_trajs}',
            color=colors[i],
            alpha=0.8
        )

    plt.xlabel('Iteration (Environment Steps)', fontsize=14, fontweight='bold')
    plt.ylabel('TAR (Total Average Return)', fontsize=14, fontweight='bold')
    plt.title('Impact of num_flow_trajs on Performance', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also display
    plt.show()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for each experiment."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    for num_flow_trajs in NUM_FLOW_TRAJS_VALUES:
        data = df[df['num_flow_trajs'] == num_flow_trajs]

        if data.empty:
            continue

        final_tar = data['TAR'].iloc[-1]
        max_tar = data['TAR'].max()
        mean_tar = data['TAR'].mean()
        std_tar = data['TAR'].std()

        print(f"\nnum_flow_trajs = {num_flow_trajs}:")
        print(f"  Final TAR:   {final_tar:.2f}")
        print(f"  Max TAR:     {max_tar:.2f}")
        print(f"  Mean TAR:    {mean_tar:.2f} ± {std_tar:.2f}")
        print(f"  Data points: {len(data)}")

    print("="*70)


def main():
    """Main analysis workflow."""
    print("="*70)
    print("NUM_FLOW_TRAJS ABLATION STUDY ANALYSIS")
    print("="*70)
    print(f"\nTask: {TASK}")
    print(f"Environment type: {ENV_TYPE}")
    print(f"Seed: {SEED}")
    print(f"Testing num_flow_trajs values: {NUM_FLOW_TRAJS_VALUES}")
    print()

    try:
        # Load and aggregate data
        print("Loading evaluation data...")
        df = aggregate_results()

        if df.empty:
            print("ERROR: No data found! Please ensure all experiments have completed.")
            return

        print(f"Successfully loaded {len(df)} data points from {len(NUM_FLOW_TRAJS_VALUES)} experiments.")

        # Save to CSV
        print("\nSaving CSV results...")
        save_csv_results(df, CSV_OUTPUT)

        # Print summary statistics
        print_summary_statistics(df)

        # Generate plot
        print("\nGenerating plot...")
        plot_results(df, PLOT_OUTPUT)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved:")
        print(f"  - CSV: {CSV_OUTPUT}")
        print(f"  - Plot: {PLOT_OUTPUT}")

    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
