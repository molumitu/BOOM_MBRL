# Plot Explore Usage Guide

This document describes how to use the plotting scripts in the `plot_explore/` directory to generate visualizations for the four-goal navigation experiments.

## Overview

The `plot_explore/` directory contains multiple Python scripts for generating different types of trajectory visualizations:

- **plot_explore.py** - Generate exploratory trajectory visualizations with random smooth perturbations
- **plot_optimal_gaussian.py** - Visualize optimal trajectories using Gaussian policy
- **plot_optimal_flow.py** - Visualize optimal trajectories using flow matching policy
- **plot_plan_gaussian.py** - Visualize planning trajectories with Gaussian policy
- **plot_plan_flow.py** - Visualize planning trajectories with flow matching policy
- **plot_train_gaussian.py** - Visualize training trajectories with Gaussian policy
- **plot_train_flow.py** - Visualize training trajectories with flow matching policy

## Quick Start

### Run All Scripts at Once

The easiest way to generate all plots is to use the provided shell script:

```bash
cd /home/zhangxiangteng/code/boom2/four_goal_exp/plot_explore
./run_all_plots.sh
```

Or from any directory:

```bash
/home/zhangxiangteng/code/boom2/four_goal_exp/plot_explore/run_all_plots.sh
```

This script will:
1. Execute all 7 plotting scripts in order
2. Show progress for each script
3. Display a summary of successful/failed runs
4. Exit with error code if any script fails

### Run Individual Scripts

You can also run individual plotting scripts:

```bash
# From the plot_explore directory
cd /home/zhangxiangteng/code/boom2/four_goal_exp/plot_explore

# Run specific script
python plot_explore.py
python plot_optimal_gaussian.py
python plot_optimal_flow.py
```

## Output

All generated plots are saved in the `figures/explore/` directory (created automatically if it doesn't exist):

```
figures/explore/
├── explore.png              # Exploratory trajectories
├── optimal_gaussian.png     # Optimal Gaussian policy
├── optimal_flow.png         # Optimal flow matching policy
├── plan_gaussian.png        # Planning Gaussian policy
├── plan_flow.png            # Planning flow matching policy
├── train_gaussian.png       # Training Gaussian policy
└── train_flow.png           # Training flow matching policy
```

## Requirements

- Python 3.x
- Required packages: `numpy`, `matplotlib`, `PIL`
- The robot icon file: `robot.png` (already in plot_explore directory)

## Script Details

### Execution Order

Scripts are executed in this order to ensure dependencies are met:

1. **plot_explore.py** - Generates baseline exploratory trajectories
2. **plot_optimal_gaussian.py** - Optimal trajectories with Gaussian baseline
3. **plot_optimal_flow.py** - Optimal trajectories with flow matching
4. **plot_plan_gaussian.py** - Planning results with Gaussian
5. **plot_plan_flow.py** - Planning results with flow matching
6. **plot_train_gaussian.py** - Training results with Gaussian
7. **plot_train_flow.py** - Training results with flow matching

### Common Parameters

Most scripts share similar parameters:
- `NUM_TRAJECTORIES` - Number of trajectories to generate (default: 1024)
- `MAX_STEPS` - Maximum steps per trajectory (default: 300)
- `STEP_SIZE` - Movement step size (default: 0.01)
- `GOAL_RADIUS` - Goal detection radius (default: 0.1)

## Troubleshooting

### Script Fails to Run

If a script fails, check:
1. Python environment is activated (use conda env `boom2`)
2. All required packages are installed
3. `robot.png` exists in the `plot_explore/` directory

### Missing Dependencies

Install required packages:

```bash
conda activate boom2
pip install numpy matplotlib pillow
```

### Permission Denied

If you get a permission error running `run_all_plots.sh`:

```bash
chmod +x /home/zhangxiangteng/code/boom2/four_goal_exp/plot_explore/run_all_plots.sh
```

## Notes

- Each script generates its plot and displays it (blocks execution until closed)
- All scripts save their output automatically
- The `run_all_plots.sh` script will continue running even if you close plot windows
- Check the console output for progress updates and statistics
