import wandb
import pandas as pd
from pathlib import Path
from typing import List, Dict


def get_column_name(key: str) -> str:
    """
    Convert wandb metric key to CSV column name.
    """
    mapping = {
        "eval/episode_success": "success",
        "eval/episode_reward": "reward",
    }
    if key in mapping:
        return mapping[key]
    # For other keys, use the last part after '/'
    return key.split("/")[-1]


def get_result(
    entity: str = "shallowdream0745-thu",
    project: str = "flow",
    base_dir: str = ".",
    tasks: str = "myo",
):
    """
    Fetch task results from wandb and save to CSV files.

    Args:
        entity: Wandb entity name
        project: Wandb project name
        base_dir: Base directory for saving CSV files
        tasks: Task type to fetch - "myo" or "dmc" (default: "myo")
    """
    # Set keys based on task type
    if tasks == "myo":
        selected_keys = ["eval/episode_success"]
    elif tasks == "dmc":
        selected_keys = ["eval/episode_reward"]
    else:
        raise ValueError(f"Unknown task type: {tasks}. Must be 'myo' or 'dmc'")

    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(f"{entity}/{project}")

    # Process runs with update_flow=false (save to boom/)
    print(f"Processing runs with update_flow=false for {tasks}...")
    process_runs(
        runs,
        selected_keys=selected_keys,
        update_filter=False,
        output_dir=Path(base_dir) / tasks / "boom",
        tasks=tasks,
    )

    # Process runs with update_flow=true (save to mfp/)
    print(f"\nProcessing runs with update_flow=true for {tasks}...")
    process_runs(
        runs,
        selected_keys=selected_keys,
        update_filter=True,
        output_dir=Path(base_dir) / tasks / "mfp",
        tasks=tasks,
    )


def process_runs(
    runs: List,
    selected_keys: List[str],
    update_filter: bool,
    output_dir: Path,
    tasks: str,
):
    """
    Process runs and save to CSV files.

    Args:
        runs: List of wandb runs
        selected_keys: List of metric keys to fetch
        update_filter: Filter by update_flow config (True/False)
        output_dir: Output directory for CSV files
        tasks: Task type to fetch - "myo" or "dmc"
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group runs by task name
    task_runs: Dict[str, List] = {}

    for run in runs:
        # Check if run matches filter criteria
        config = run.config

        # Filter based on task type
        if tasks == "myo":
            # Check if task contains "myo"
            task = config.get("task", "")
            if "myo" not in task:
                continue

            # Check seed (> 10)
            seed = config.get("seed")
            if seed is None or seed <= 10:
                continue

        elif tasks == "dmc":
            # Check env_type is dm_control
            env_type = config.get("env_type")
            if env_type != "dm_control":
                continue
            # Check seed (> 10)
            seed = config.get("seed")
            if seed is None or seed <= 10:
                continue

            # No seed filter for dmc
            task = config.get("task", "")

        else:
            raise ValueError(f"Unknown task type: {tasks}. Must be 'myo' or 'dmc'")

        # Check update_flow
        update_flow = config.get("update_flow")
        if update_flow != update_filter:
            continue

        # Group by task name
        if task not in task_runs:
            task_runs[task] = []
        task_runs[task].append(run)

    # Process each task
    for task_name, runs_list in task_runs.items():
        print(f"  Processing task: {task_name} ({len(runs_list)} runs)")

        # Collect all data from runs
        all_data = []

        for run in runs_list:
            try:
                # Fetch history with selected keys
                history = run.history(keys=selected_keys + ["_step"])

                # Extract data
                seed = run.config.get("seed", 0)
                for _, row in history.iterrows():
                    data_row = {"step": row["_step"], "seed": seed}
                    for key in selected_keys:
                        col_name = get_column_name(key)
                        data_row[col_name] = row.get(key, None)
                    all_data.append(data_row)

            except Exception as e:
                print(f"    Warning: Failed to fetch data for run {run.id}: {e}")
                continue

        if not all_data:
            print(f"    Warning: No data found for {task_name}")
            continue

        # Create DataFrame and save
        df = pd.DataFrame(all_data)

        # Reorder columns: step, selected_keys columns, seed
        columns = ["step"] + [get_column_name(key) for key in selected_keys] + ["seed"]
        df = df[columns]

        # Sort by step
        df = df.sort_values("step")

        # Save to CSV
        output_path = output_dir / f"{task_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"    Saved to {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch task results from wandb")
    parser.add_argument(
        "--entity",
        default="shallowdream0745-thu",
        help="Wandb entity name",
    )
    parser.add_argument(
        "--project",
        default="flow",
        help="Wandb project name",
    )
    parser.add_argument(
        "--base-dir",
        default="result_plot",
        help="Base directory for saving CSV files",
    )
    parser.add_argument(
        "--tasks",
        default="myo",
        choices=["myo", "dmc"],
        help="Task type to fetch - 'myo' or 'dmc' (default: myo)",
    )

    args = parser.parse_args()

    get_result(
        entity=args.entity,
        project=args.project,
        base_dir=args.base_dir,
        tasks=args.tasks,
    )
