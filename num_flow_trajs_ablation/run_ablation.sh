#!/bin/bash

# Ablation study for num_flow_trajs parameter
# This script runs 4 experiments with num_flow_trajs = 12, 24, 48, 72
# Parallel execution on 2 GPUs: experiments 1-2 on GPU 0, experiments 3-4 on GPU 1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Experiment parameters
TASK="dog-run"
ENV_TYPE="dm_control"
SEED=1
STEPS=1001000

# Array of num_flow_trajs values to test
declare -a NUM_FLOW_TRAJS_VALUES=(12 24 48 72)

# Function to run a single experiment
run_experiment() {
    local num_flow_trajs=$1
    local gpu_id=$2
    local exp_name="num_flow_trajs_${num_flow_trajs}"

    echo "Starting experiment: num_flow_trajs=${num_flow_trajs} on GPU ${gpu_id}"
    echo "Exp name: ${exp_name}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python ${PROJECT_DIR}/boom/train.py \
        task=${TASK} \
        env_type=${ENV_TYPE} \
        update_flow=true \
        seed=${SEED} \
        num_flow_trajs=${num_flow_trajs} \
        steps=${STEPS} \
        exp_name=${exp_name} \
        wandb_silent=true

    echo "Completed experiment: num_flow_trajs=${num_flow_trajs}"
}

# Run experiments in parallel
# Experiments 1-2 on GPU 0, Experiments 3-4 on GPU 1

echo "=========================================="
echo "Starting num_flow_trajs ablation study"
echo "=========================================="
echo "Task: ${TASK}"
echo "Environment: ${ENV_TYPE}"
echo "Seed: ${SEED}"
echo "Testing num_flow_trajs values: ${NUM_FLOW_TRAJS_VALUES[@]}"
echo ""

# Launch experiments on GPU 0
run_experiment ${NUM_FLOW_TRAJS_VALUES[0]} 0 &
EXP1_PID=$!
echo "Launched experiment 1 (num_flow_trajs=${NUM_FLOW_TRAJS_VALUES[0]}) on GPU 0, PID: $EXP1_PID"

run_experiment ${NUM_FLOW_TRAJS_VALUES[1]} 0 &
EXP2_PID=$!
echo "Launched experiment 2 (num_flow_trajs=${NUM_FLOW_TRAJS_VALUES[1]}) on GPU 0, PID: $EXP2_PID"

# Wait a bit before launching on GPU 1 to avoid potential issues
sleep 5

# Launch experiments on GPU 1
run_experiment ${NUM_FLOW_TRAJS_VALUES[2]} 1 &
EXP3_PID=$!
echo "Launched experiment 3 (num_flow_trajs=${NUM_FLOW_TRAJS_VALUES[2]}) on GPU 1, PID: $EXP3_PID"

run_experiment ${NUM_FLOW_TRAJS_VALUES[3]} 1 &
EXP4_PID=$!
echo "Launched experiment 4 (num_flow_trajs=${NUM_FLOW_TRAJS_VALUES[3]}) on GPU 1, PID: $EXP4_PID"

# Wait for all experiments to complete
echo ""
echo "All experiments launched. Waiting for completion..."

wait $EXP1_PID
echo "Experiment 1 (num_flow_trajs=${NUM_FLOW_TRAJS_VALUES[0]}) completed!"

wait $EXP2_PID
echo "Experiment 2 (num_flow_trajs=${NUM_FLOW_TRAJS_VALUES[1]}) completed!"

wait $EXP3_PID
echo "Experiment 3 (num_flow_trajs=${NUM_FLOW_TRAJS_VALUES[2]}) completed!"

wait $EXP4_PID
echo "Experiment 4 (num_flow_trajs=${NUM_FLOW_TRAJS_VALUES[3]}) completed!"

echo ""
echo "=========================================="
echo "All ablation experiments completed!"
echo "=========================================="
echo ""
echo "You can now analyze the results by running:"
echo "  cd ${SCRIPT_DIR}"
echo "  python analyze_ablation.py"
