"""
Shared configuration for four-goal navigation experiments.

This module contains the goal configuration, color schemes, and default
environment parameters used across all visualization and trajectory scripts.
"""

import numpy as np

# Goal configuration with consistent color scheme
# Colors: blue, green, orange, purple (elegant, publication-quality colors)
GOAL_CONFIG = {
    'top_left': {
        'pos': np.array([-1.0, 1.0]),
        'color': '#9467bd',  # purple
        'optimal_angle': 3 * np.pi / 4,  # 135°
        'label': 'Top Left'
    },
    'top_right': {
        'pos': np.array([1.0, 1.0]),
        'color': '#1f77b4',  # blue
        'optimal_angle': np.pi / 4,  # 45°
        'label': 'Top Right'
    },
    'bottom_left': {
        'pos': np.array([-1.0, -1.0]),
        'color': '#2ca02c',  # green
        'optimal_angle': -3 * np.pi / 4,  # -135°
        'label': 'Bottom Left'
    },
    'bottom_right': {
        'pos': np.array([1.0, -1.0]),
        'color': '#ff7f0e',  # orange
        'optimal_angle': -np.pi / 4,  # -45°
        'label': 'Bottom Right'
    }
}

# Default environment parameters
DEFAULT_ENV_PARAMS = {
    'step_size': 0.01,
    'goal_radius': 0.1,
    'max_steps': 300,
    'step_penalty': -0.01
}

# Trajectory generation presets for different modes
TRAJECTORY_PRESETS = {
    'explore': {
        'angle_std': 0.8,
        'max_angle_change': 0.1,
        'num_trajectories': 1024,
        'description': 'Random exploration with uniform angle distribution'
    },
    'train_gaussian': {
        'angle_std': 0.8,
        'max_angle_change': 0.1,
        'num_trajectories': 1024,
        'description': 'Training data for Gaussian policy (uniform random)'
    },
    'train_flow': {
        'angle_std': 2.0,
        'max_angle_change': 0.05,
        'num_trajectories': 256,
        'description': 'Training data for flow policy (multi-modal)'
    },
    'plan_gaussian': {
        'angle_std': 2.0,
        'max_angle_change': 0.05,
        'num_trajectories': 256,
        'description': 'Gaussian planning (biased toward top-right)'
    },
    'optimal_flow': {
        'angle_std': 2.0,
        'max_angle_change': 0.05,
        'num_trajectories': 256,
        'description': 'Optimal flow planning (evenly distributed goals)'
    }
}
