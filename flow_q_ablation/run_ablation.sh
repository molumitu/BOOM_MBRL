#!/bin/bash
# Flow Q Coefficient Ablation Study
# Tests flow_q_coef = 0 (pure flow matching) vs flow_q_coef = 1 (hybrid)

set -e

# Task configuration
TASK="dog-run"
ENV_TYPE="dm_control"

# Experiment configuration
EXP_GROUP="flow_q_ablation"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="$(dirname "$0")/logs"
mkdir -p ${LOG_DIR}

echo "======================================"
echo "Flow Q Coefficient Ablation Study"
echo "======================================"
echo "Task: ${TASK}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# ========================================
# Experiment 1: Pure Flow Matching (flow_q_coef=0)
# ========================================
echo "🔵 Starting Experiment 1: Pure Flow Matching (flow_q_coef=0) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python boom/train.py \
    task=${TASK} \
    env_type=${ENV_TYPE} \
    flow_q_coef=0.0 \
    extra="ablation_q0" \
    exp_name=${EXP_GROUP}/pure_flow_matching \
    2>&1 | tee ${LOG_DIR}/pure_flow_gpu0_${TIMESTAMP}.log &

PID_GPU0=$!
echo "  PID: ${PID_GPU0} (GPU 0)"
echo "  Log: ${LOG_DIR}/pure_flow_gpu0_${TIMESTAMP}.log"
sleep 5

# ========================================
# Experiment 2: Hybrid Training (flow_q_coef=1)
# ========================================
echo "🟢 Starting Experiment 2: Hybrid Training (flow_q_coef=1) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python boom/train.py \
    task=${TASK} \
    env_type=${ENV_TYPE} \
    flow_q_coef=1.0 \
    extra="ablation_q1" \
    exp_name=${EXP_GROUP}/hybrid_training \
    2>&1 | tee ${LOG_DIR}/hybrid_gpu1_${TIMESTAMP}.log &

PID_GPU1=$!
echo "  PID: ${PID_GPU1} (GPU 1)"
echo "  Log: ${LOG_DIR}/hybrid_gpu1_${TIMESTAMP}.log"
echo ""

# ========================================
# Monitor Experiments
# ========================================
echo "======================================"
echo "Both experiments are running in parallel"
echo "======================================"
echo ""
echo "GPU 0 - Pure Flow Matching (flow_q_coef=0.0)"
echo "  PID: ${PID_GPU0}"
echo "  Command: tail -f ${LOG_DIR}/pure_flow_gpu0_${TIMESTAMP}.log"
echo ""
echo "GPU 1 - Hybrid Training (flow_q_coef=1.0)"
echo "  PID: ${PID_GPU1}"
echo "  Command: tail -f ${LOG_DIR}/hybrid_gpu1_${TIMESTAMP}.log"
echo ""
echo "======================================"
echo "Waiting for experiments to complete..."
echo "======================================"
echo ""

# Wait for both processes
wait ${PID_GPU0}
EXIT_CODE0=$?
echo "✅ GPU 0 (Pure Flow Matching) finished with exit code: ${EXIT_CODE0}"

wait ${PID_GPU1}
EXIT_CODE1=$?
echo "✅ GPU 1 (Hybrid Training) finished with exit code: ${EXIT_CODE1}"

echo ""
echo "======================================"
echo "All experiments completed!"
echo "======================================"
echo "Results summary:"
echo "  - Pure Flow Matching: exit code ${EXIT_CODE0}"
echo "  - Hybrid Training: exit code ${EXIT_CODE1}"
echo ""
echo "Log files:"
echo "  - ${LOG_DIR}/pure_flow_gpu0_${TIMESTAMP}.log"
echo "  - ${LOG_DIR}/hybrid_gpu1_${TIMESTAMP}.log"
echo "======================================"
