# Four Goal Navigation - Complete Training Design

## Overview
Complete training pipeline for four-goal navigation task using BOOM algorithm with MPPI planning.

## Design Philosophy

### Key Differences from Quick Demo
- **Quick Demo**: Single-state MPPI sampling, no training loop
- **Complete Training**: Full RL pipeline with data collection, model updates, and evaluation

### Training Architecture

```
Initialization
    ↓
Training Loop (N steps)
    ├─ Collect Episode
    │   ├─ Reset environment
    │   ├─ For each step:
    │   │   ├─ MPPI planning
    │   │   ├─ Execute action
    │   │   └─ Store transition
    │   └─ Return trajectory
    │
    ├─ Update Models
    │   ├─ Sample batch from replay buffer
    │   ├─ Update Q-function
    │   ├─ Update Value function
    │   ├─ Update Policy (MLP/Flow)
    │   └─ Update encoder
    │
    ├─ Logging
    │   └─ Metrics to wandb/csv
    │
    └─ Periodic Evaluation
        ├─ Run eval episodes
        ├─ Compute success rate
        └─ Save checkpoints
```

## Parameter Configuration Strategy

### 1. Training Parameters (Complete vs Quick)

| Parameter | Quick Demo | Complete Training | Rationale |
|-----------|-----------|-------------------|-----------|
| steps | 1,000 | 100,000 | Sufficient for convergence |
| batch_size | 64 | 256 | Standard for stability |
| buffer_size | 10,000 | 500,000 | Diverse experience replay |
| episode_length | 40 | 40 | Task-specific |

### 2. MPPI Planning Parameters

| Parameter | Quick Demo | Complete Training | Rationale |
|-----------|-----------|-------------------|-----------|
| iterations | 50 | 6 | Fewer needed with trained policy |
| num_samples | 128 | 512 | Better coverage |
| num_elites | 13 (10%) | 64 (12.5%) | Standard elite ratio |
| horizon | 40 | 3 | Shorter horizon sufficient |
| max_std | 0.2 | 2.0 | More exploration |
| temperature | 0.1 | 0.5 | Balanced exploitation |

### 3. Policy-Specific Parameters

**MLP Configuration:**
- `num_pi_trajs`: 24
- `num_flow_trajs`: 0
- `update_flow`: False

**Flow Configuration:**
- `num_pi_trajs`: 24
- `num_flow_trajs`: 48
- `update_flow`: True
- `flow_q_coef`: 1.0
- `flow_mode`: 'action'

## Training Phases

### Phase 1: Initialization (Steps 0-1000)
- Random exploration
- Warm up replay buffer
- Initialize value estimates

### Phase 2: Active Learning (Steps 1000-50000)
- MPPI-guided exploration
- Policy learning
- Flow matching (if enabled)

### Phase 3: Fine-tuning (Steps 50000-100000)
- Exploitation of learned policy
- Refinement of value estimates
- Stability improvement

## Evaluation Protocol

### Frequency
- Every 5000 steps
- 10 evaluation episodes
- Separate from training data

### Metrics
- **Success Rate**: Reached any goal within radius
- **Goal Coverage**: Distribution over 4 goals
- **Episode Return**: Cumulative reward
- **Action Diversity**: Entropy of action distribution
- **Planning Time**: MPPI computation time

### Checkpoint Strategy
- Save every 10000 steps
- Keep best 3 checkpoints
- Save final model

## Multi-Modality Assessment

### Expected Behavior
- **Unimodal**: Policy always goes to same goal (bad)
- **Bimodal**: Policy discovers 2 goals (acceptable)
- **Multi-modal**: Policy covers all 4 goals (ideal)

### Measurement
1. **Goal Visit Distribution**: Count visits to each goal
2. **Action Direction Analysis**: Histogram of initial actions
3. **Trajectory Clustering**: Group similar trajectories

### Success Criteria
- All 4 goals reachable
- Near-uniform goal distribution (±20%)
- Stable across multiple seeds

## Implementation Details

### Replay Buffer
- Uniform sampling
- Prioritized by episode return (optional)
- Frame stacking for temporal context (optional)

### Model Architecture
- Encoder: 2-layer, 256-dim
- Policy: MLP or Normalizing Flow
- Q-function: 5 ensemble Q-networks
- Value function: 5 ensemble V-networks

### MPPI Integration
- Elite selection for action candidates
- Flow-based proposal distribution
- Adaptive temperature scheduling

## Hyperparameter Sensitivity

### Critical Parameters
1. **Horizon**: Too short → myopic, Too long → slow
2. **Temperature**: Too low → exploit, Too high → random
3. **Flow coefficient**: Balances flow vs policy

### Tuning Strategy
- Start with reference values
- Grid search on key params
- Monitor multi-modality metrics

## Expected Results

### Baseline (MLP)
- Success rate: 60-80%
- Goal coverage: 2-3 goals
- Stable but limited diversity

### Flow Matching
- Success rate: 70-90%
- Goal coverage: 3-4 goals
- Better multi-modality

## Troubleshooting

### Common Issues
1. **Single Mode Collapse** → Increase entropy_coef, decrease temperature
2. **Unstable Learning** → Decrease lr, increase batch_size
3. **Slow Convergence** → Increase num_samples, adjust horizon
4. **Memory Issues** → Decrease buffer_size, batch_size

### Debug Tips
- Visualize MPPI action distributions
- Check Q-value estimates
- Monitor replay buffer diversity
- Profile computation time

## Computational Requirements

### Hardware
- GPU: NVIDIA RTX 3090 or better
- RAM: 16GB minimum
- Storage: 10GB for checkpoints

### Runtime Estimates
- 100k steps: 4-8 hours
- 50k steps: 2-4 hours
- Per 1000 steps: 2-5 minutes

## Usage

### Basic Training
```bash
python toy_train.py --policy_type mlp --steps 100000
```

### Flow Training
```bash
python toy_train.py --policy_type flow --steps 100000
```

### Ablation Studies
```bash
python toy_train.py --policy_type flow --flow_q_coef 0.5
python toy_train.py --policy_type flow --horizon 5
```

### Multiple Seeds
```bash
for seed in 1 2 3; do
    python toy_train.py --policy_type flow --seed $seed
done
```

## File Structure

```
toy_exp/
├── toy_train.py           # Main training script
├── TRAINING_DESIGN.md     # This file
├── configs/
│   ├── mlp_full.yaml      # MLP configuration
│   └── flow_full.yaml     # Flow configuration
├── results/
│   ├── mlp_seed1/         # MLP results
│   └── flow_seed1/        # Flow results
└── logs/
    └── wandb/             # Experiment logs
```

## Future Improvements

1. **Curriculum Learning**: Start with 2 goals, add more gradually
2. **Reward Shaping**: Intrinsic rewards for goal diversity
3. **Hierarchical Policy**: High-level goal selection + low-level control
4. **Ensemble Diversity**: Explicitly encourage diverse behaviors
5. **Unsupervised Discovery**: Cluster modes without predefined goals
