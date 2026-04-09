#!/bin/bash

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Raw experiments (update_flow=false)

# MyoSuite tasks
declare -a MYOSUITE_TASKS=("myo-reach" "myo-reach-hard" "myo-obj-hold" "myo-obj-hold-hard")

# Seeds to run
declare -a SEEDS=(11 12 13 14 15)

echo "Starting raw experiments (update_flow=false)..."

# Run MyoSuite tasks
for task in "${MYOSUITE_TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running myosuite task: $task, seed: $seed"
        python boom/train.py task=$task env_type=myosuite update_flow=false seed=$seed exp_name=myo_flow_${task} extra="raw"
    done
done

echo "All raw experiments completed!"
