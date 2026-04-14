# Experiment Configuration Update

## Date
2026-04-08

## Problem
Previous experiments showed MPPI not forming clear 4 peaks in the action distribution. The issue was that the MPPI parameters didn't match the reference implementation in `two_goal_exp/four_goal_mppi.py`.

## Solution
Updated `train_fixed_plots.py` configuration to match the reference parameters:

### Environment Parameters
| Parameter | Old Value | New Value | Reference |
|-----------|-----------|-----------|-----------|
| step_size | 0.2 | 0.10 | four_goal_mppi.py |
| goal_radius | 0.15 | 0.1 | four_goal_mppi.py |
| max_steps | 25 | 40 | four_goal_mppi.py |

### MPPI Parameters
| Parameter | Old Value | New Value | Reference |
|-----------|-----------|-----------|-----------|
| iterations | 6 | 50 | four_goal_mppi.py |
| num_samples | 512 | 128 | four_goal_mppi.py |
| num_elites | 64 | 13 (10% of 128) | four_goal_mppi.py elite_fraction |
| horizon | 3 | 40 | four_goal_mppi.py |
| max_std | 2.0 | 0.2 | four_goal_mppi.py NOISE_SIGMA |
| temperature | 0.5 | 0.1 | four_goal_mppi.py LAMBDA |

## Key Insights

1. **Longer Horizon (40 vs 3)**: The reference uses a much longer planning horizon to reach distant goals
2. **More MPPI Iterations (50 vs 6)**: More refinement iterations to converge to good trajectories
3. **Smaller Action Noise (0.2 vs 2.0)**: More stable exploration during MPPI optimization
4. **Lower Temperature (0.1 vs 0.5)**: Higher selectivity for elite samples
5. **Elite Sampling**: Using top 10% of samples for weighted average update

## Expected Results

With these updated parameters, MPPI should:
- Discover all 4 goal directions (45°, 135°, -45°, -135°)
- Form clearer peaks in the action distribution
- Show better multi-modal behavior

The Flow policy should maintain multiple modes better than MLP policy.

## Training Status

Running: `python train_fixed_plots.py`

Output will be saved to:
- `results/` directory for CSV data
- `results/fixed_plot.png` for visualization

## Notes

- Training will take longer due to increased horizon (40 vs 3) and iterations (50 vs 6)
- Each MPPI planning step now evaluates 128 trajectories over 40 steps = 5120 rollouts per action
- Previous: 512 samples × 3 steps = 1536 rollouts per action
- Approximately 3.3x more computation per action
