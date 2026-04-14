"""
Trajectory generation utilities for four-goal navigation experiments.

This module provides functions to generate smooth trajectories from the origin
using angle inertia for smoothness. Supports different generation modes for
various experimental scenarios.
"""

import numpy as np
from typing import List, Dict, Optional
from .config import GOAL_CONFIG, TRAJECTORY_PRESETS


def generate_trajectories(
    mode: str = 'explore',
    num_trajectories: Optional[int] = None,
    max_steps: int = 300,
    step_size: float = 0.01,
    goal_radius: float = 0.1,
    angle_std: Optional[float] = None,
    max_angle_change: Optional[float] = None,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Generate smooth trajectories from origin using angle inertia for smoothness.

    The angle inertia method limits the rate of angle change between steps,
    creating smooth curves instead of jittery random walks.

    Args:
        mode: Trajectory generation mode ('explore', 'train_gaussian', 'train_flow',
              'plan_gaussian', 'optimal_flow'). Uses presets if angle_std and
              max_angle_change are not provided.
        num_trajectories: Number of trajectories to generate. If None, uses preset.
        max_steps: Maximum steps per trajectory
        step_size: Step size for movement (smaller = smoother)
        goal_radius: Radius for goal detection
        angle_std: Standard deviation of noise. If None, uses preset based on mode.
        max_angle_change: Maximum angle change per step (radians). If None, uses preset.
        seed: Random seed for reproducibility

    Returns:
        List of trajectory dictionaries compatible with plot functions.
        Each dict contains:
            - 'best_state_trajectory': List of (x, y) positions
            - 'best_info': Dict with 'reached_goal' key
            - 'best_total_reward': Placeholder value (0.0)
    """
    # Get preset parameters if not explicitly provided
    if mode in TRAJECTORY_PRESETS:
        preset = TRAJECTORY_PRESETS[mode]
        if angle_std is None:
            angle_std = preset['angle_std']
        if max_angle_change is None:
            max_angle_change = preset['max_angle_change']
        if num_trajectories is None:
            num_trajectories = preset['num_trajectories']
    else:
        # Default values
        if angle_std is None:
            angle_std = 0.8
        if max_angle_change is None:
            max_angle_change = 0.1
        if num_trajectories is None:
            num_trajectories = 256

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    trajectories = []
    goal_names_list = list(GOAL_CONFIG.keys())

    for i in range(num_trajectories):
        # Start at origin
        position = np.array([0.0, 0.0])
        trajectory = [position.copy()]

        # Determine primary angle based on mode
        if mode == 'explore' or mode == 'train_gaussian':
            # Uniform random direction (0 to 2pi)
            primary_angle = np.random.uniform(0, 2 * np.pi)
        elif mode == 'train_flow':
            # Multi-modal: angles clustered around four optimal directions
            goal_idx = i % 4
            target_goal = goal_names_list[goal_idx]
            optimal_angle = GOAL_CONFIG[target_goal]['optimal_angle']
            primary_angle = np.random.normal(optimal_angle, 0.4)
        elif mode == 'plan_gaussian':
            # Biased toward top-right (blue) goal
            primary_angle = np.random.normal(np.pi/4, 0.4)
        elif mode == 'optimal_flow':
            # Evenly distribute among all four goals
            goal_idx = i % 4
            target_goal = goal_names_list[goal_idx]
            optimal_angle = GOAL_CONFIG[target_goal]['optimal_angle']
            primary_angle = np.random.normal(optimal_angle, 0.4)
        else:
            # Default: uniform random
            primary_angle = np.random.uniform(0, 2 * np.pi)

        # Initialize current angle to primary direction
        current_angle = primary_angle

        for step in range(max_steps):
            # Generate desired angle with noise around primary direction
            noise = np.random.normal(0, angle_std)
            desired_angle = primary_angle + noise

            # Limit angle change for smoothness (inertia effect)
            angle_change = desired_angle - current_angle
            # Normalize to [-pi, pi] for shortest path
            angle_change = (angle_change + np.pi) % (2 * np.pi) - np.pi
            # Clamp to maximum change rate
            angle_change = np.clip(angle_change, -max_angle_change, max_angle_change)

            # Update current angle
            current_angle = current_angle + angle_change

            # Update position
            action = np.array([np.cos(current_angle), np.sin(current_angle)])
            position = position + step_size * action
            trajectory.append(position.copy())

            # Check if reached any goal
            reached_goal = None
            for goal_name, goal_info in GOAL_CONFIG.items():
                dist = np.linalg.norm(position - goal_info['pos'])
                if dist < goal_radius:
                    reached_goal = goal_name
                    break

            # Check boundary [-1.0, 1.0]
            if abs(position[0]) > 1.0 or abs(position[1]) > 1.0:
                break

            # Stop if reached a goal
            if reached_goal is not None:
                break

        # Determine which goal was reached (if any)
        final_pos = trajectory[-1]
        reached_goal = None
        for goal_name, goal_info in GOAL_CONFIG.items():
            dist = np.linalg.norm(final_pos - goal_info['pos'])
            if dist < goal_radius:
                reached_goal = goal_name
                break

        trajectories.append({
            'best_state_trajectory': trajectory,
            'best_info': {'reached_goal': reached_goal},
            'best_total_reward': 0.0  # placeholder
        })

    return trajectories


def print_trajectory_statistics(trajectories: List[Dict]) -> None:
    """
    Print statistics about generated trajectories.

    Args:
        trajectories: List of trajectory dictionaries
    """
    goal_success = {g: 0 for g in GOAL_CONFIG.keys()}
    no_success = 0

    for traj in trajectories:
        goal = traj['best_info']['reached_goal']
        if goal in goal_success:
            goal_success[goal] += 1
        else:
            no_success += 1

    print(f"  Number of trajectories: {len(trajectories)}")
    print(f"  Success rate: {sum(goal_success.values()) / len(trajectories):.1%}")
    for goal_name, count in goal_success.items():
        label = GOAL_CONFIG[goal_name]['label']
        print(f"    {label}: {count} ({count/len(trajectories):.1%})")
    print(f"    No goal: {no_success} ({no_success/len(trajectories):.1%})")
