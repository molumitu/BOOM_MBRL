#!/bin/bash

# Flow experiments (update_flow=true)

# DM Control tasks
declare -a DM_CONTROL_TASKS=("dog-run" "humanoid-run" "walker-run" "humanoid-walk")

# MyoSuite tasks
declare -a MYOSUITE_TASKS=("myo-reach" "myo-reach-hard" "myo-obj-hold" "myo-obj-hold-hard")

# Seeds to run
declare -a SEEDS=(11 12 13 14 15)

echo "Starting flow experiments (update_flow=true)..."

# Run DM Control tasks
for task in "${DM_CONTROL_TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running dm_control task: $task, seed: $seed"
        python boom/train.py task=$task env_type=dm_control update_flow=true seed=$seed
    done
done

# Run MyoSuite tasks
for task in "${MYOSUITE_TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running myosuite task: $task, seed: $seed"
        python boom/train.py task=$task env_type=myosuite update_flow=true seed=$seed
    done
done

echo "All flow experiments completed!"
