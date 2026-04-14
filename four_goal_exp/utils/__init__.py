"""
Utilities for four-goal navigation experiments.

This module provides shared functionality for trajectory generation,
visualization, and configuration management.
"""

from .config import GOAL_CONFIG, DEFAULT_ENV_PARAMS
from .trajectory import generate_trajectories, print_trajectory_statistics
from .visualization import plot_four_goal_trajectories_clean

__all__ = [
    'GOAL_CONFIG',
    'DEFAULT_ENV_PARAMS',
    'generate_trajectories',
    'print_trajectory_statistics',
    'plot_four_goal_trajectories_clean',
]
