#!/bin/bash

# Flow experiments (update_flow=true)

# DM Control tasks
# declare -a DM_CONTROL_TASKS=("dog-run" "humanoid-run" "walker-run" "humanoid-walk")
declare -a DM_CONTROL_TASKS=("humanoid-run" "humanoid-walk")

# Seeds to run
declare -a SEEDS=(11 12 13)

echo "Starting flow experiments (update_flow=true)..."

# Run DM Control tasks
for task in "${DM_CONTROL_TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running dm_control task: $task, seed: $seed"
        python boom/train.py task=$task env_type=dm_control update_flow=true seed=$seed exp_name=dmc_flow_${task} extra="flow"
    done
done

echo "All flow experiments completed!"
