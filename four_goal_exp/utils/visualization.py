"""
Visualization utilities for four-goal navigation experiments.

This module provides plotting functions for visualizing trajectories,
goals, and analysis results with clean, publication-ready aesthetics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.lines import Line2D
import os
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from typing import List, Dict, Optional, Tuple
from .config import GOAL_CONFIG


def plot_four_goal_trajectories_clean(
    results: List[Dict],
    env_params: dict,
    save_path: str = "four_goal_trajectories_clean.png",
    figsize: Tuple[int, int] = (10, 10),
    highlight_goal: Optional[str] = None,
    show_arrows: bool = True,
    show_unsuccessful: bool = True,
    unsuccessful_sample_ratio: float = 0.2,
    plot_style: str = 'all',  # 'all', 'optimal', 'single'
    robot_png_path: Optional[str] = None
) -> None:
    """
    Plot trajectories with clean aesthetics - no axes, borders, or legends.

    Args:
        results: List of experiment results with trajectory data
        env_params: Environment parameters (for goal_radius)
        save_path: Path to save the figure
        figsize: Figure size
        highlight_goal: If specified, only color this goal's trajectories
        show_arrows: Whether to show directional arrows from start
        show_unsuccessful: Whether to show unsuccessful trajectories
        unsuccessful_sample_ratio: Ratio of unsuccessful trajectories to show (0-1)
        plot_style: 'all' (show all trajectories), 'optimal' (show shortest per goal),
                   'single' (only show highlight_goal trajectories)
        robot_png_path: Path to robot PNG image (if None, looks in script directory)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Remove axes, ticks, and spines completely
    ax.set_axis_off()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')

    # Hide all spines (no border)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Draw faint boundary
    boundary = Rectangle((-1, -1), 2, 2, linewidth=3,
                        edgecolor='#414141', facecolor='none', zorder=1)
    ax.add_patch(boundary)

    # Goal radius
    goal_radius = env_params.get('goal_radius', 0.1)

    # Draw goals with gradient-like effect (multiple circles)
    for goal_name, goal_info in GOAL_CONFIG.items():
        # Outer glow
        goal_circle_outer = Circle(goal_info['pos'], goal_radius * 1.5,
                                   color=goal_info['color'], alpha=0.15, zorder=2)
        ax.add_patch(goal_circle_outer)

        # Main goal
        goal_circle = Circle(goal_info['pos'], goal_radius,
                            color=goal_info['color'], alpha=0.4, zorder=3)
        ax.add_patch(goal_circle)

        # Inner highlight
        goal_circle_inner = Circle(goal_info['pos'], goal_radius * 0.5,
                                   color=goal_info['color'], alpha=0.7, zorder=4)
        ax.add_patch(goal_circle_inner)

    # Draw directional arrows (semi-transparent)
    if show_arrows:
        arrow_length = 0.8
        arrow_start_dist = 0.2

        for goal_name, goal_info in GOAL_CONFIG.items():
            angle = goal_info['optimal_angle']
            start_x = arrow_start_dist * np.cos(angle)
            start_y = arrow_start_dist * np.sin(angle)
            end_x = (arrow_start_dist + arrow_length) * np.cos(angle)
            end_y = (arrow_start_dist + arrow_length) * np.sin(angle)

            # Determine arrow color and style
            if plot_style == 'single' and highlight_goal:
                # In single mode, only highlight the specified goal
                if goal_name == highlight_goal:
                    arrow_color = goal_info['color']
                    arrow_alpha = 0.8
                else:
                    arrow_color = 'gray'
                    arrow_alpha = 0.4
            else:
                # In other modes, color all arrows
                arrow_color = goal_info['color']
                arrow_alpha = 0.5 if plot_style == 'optimal' else 0.8

            arrow = FancyArrow(start_x, start_y, end_x - start_x, end_y - start_y,
                        width=0.1, head_width=0.3, head_length=0.3,
                        color=arrow_color, alpha=arrow_alpha, zorder=8,
                        length_includes_head=True)
            ax.add_patch(arrow)

    # Separate trajectories by which goal they reached
    trajectories_by_goal = {g: [] for g in GOAL_CONFIG.keys()}
    trajectories_to_none = []

    for result in results:
        traj = result['best_state_trajectory']
        info = result['best_info']

        if info['reached_goal'] in trajectories_by_goal:
            trajectories_by_goal[info['reached_goal']].append(traj)
        else:
            trajectories_to_none.append(traj)

    # Plot trajectories based on style
    if plot_style == 'optimal':
        # Plot only the shortest trajectory for each goal
        for goal_name, goal_info in GOAL_CONFIG.items():
            if len(trajectories_by_goal[goal_name]) > 0:
                # Find trajectory with minimum steps (shortest path)
                optimal_traj = min(trajectories_by_goal[goal_name], key=len)
                traj_array = np.array(optimal_traj)
                ax.plot(traj_array[:, 0], traj_array[:, 1],
                    color=goal_info['color'], alpha=0.8, linewidth=4,
                    solid_capstyle='round', zorder=6)
    elif plot_style == 'single':
        # Plot only trajectories to the highlighted goal
        if highlight_goal and highlight_goal in trajectories_by_goal:
            goal_info = GOAL_CONFIG[highlight_goal]
            for traj in trajectories_by_goal[highlight_goal]:
                traj_array = np.array(traj)
                ax.plot(traj_array[:, 0], traj_array[:, 1],
                        color=goal_info['color'], alpha=0.5, linewidth=1.8,
                        solid_capstyle='round', zorder=6)

        # Still show some unsuccessful trajectories
        if show_unsuccessful and len(trajectories_to_none) > 0:
            sample_size = int(len(trajectories_to_none) * unsuccessful_sample_ratio)
            sampled_indices = np.random.choice(len(trajectories_to_none),
                                              size=sample_size, replace=False)
            for idx in sampled_indices:
                traj = trajectories_to_none[idx]
                traj_array = np.array(traj)
                ax.plot(traj_array[:, 0], traj_array[:, 1],
                        'gray', alpha=0.25, linewidth=1.2, linestyle='--',
                        solid_capstyle='round', zorder=6)
    else:  # plot_style == 'all'
        # Plot trajectories for each goal
        for goal_name, goal_info in GOAL_CONFIG.items():
            for traj in trajectories_by_goal[goal_name]:
                traj_array = np.array(traj)

                # Determine color based on highlight_goal
                if highlight_goal and goal_name != highlight_goal:
                    # Non-highlighted goals are gray
                    color = 'gray'
                    alpha = 0.4
                    linewidth = 2
                else:
                    # Highlighted goal uses its color
                    color = goal_info['color']
                    alpha = 0.4
                    linewidth = 3

                ax.plot(traj_array[:, 0], traj_array[:, 1],
                        color=color, alpha=alpha, linewidth=linewidth,
                        solid_capstyle='round', zorder=6)

        # Plot unsuccessful trajectories (gray, dashed)
        if show_unsuccessful and len(trajectories_to_none) > 0:
            sample_size = int(len(trajectories_to_none) * unsuccessful_sample_ratio)
            sampled_indices = np.random.choice(len(trajectories_to_none),
                                              size=sample_size, replace=False)

            for idx in sampled_indices:
                traj = trajectories_to_none[idx]
                traj_array = np.array(traj)
                ax.plot(traj_array[:, 0], traj_array[:, 1],
                        'gray', alpha=0.35, linewidth=1.5, linestyle='--',
                        solid_capstyle='round', zorder=6)

    # Draw red circle at origin (starting point)
    origin_circle = Circle((0, 0), 0.05, color='red', alpha=0.8, zorder=7)
    ax.add_patch(origin_circle)

    if os.path.exists(robot_png_path):
        robot_img = Image.open(robot_png_path)
        img_size = robot_img.size

        # Apply alpha transparency
        robot_img = robot_img.convert('RGBA')
        r, g, b, a = robot_img.split()
        a = a.point(lambda x: int(x * 0.8))
        robot_img = Image.merge('RGBA', (r, g, b, a))

        # Calculate appropriate zoom
        target_size = 200
        zoom_factor = target_size / max(img_size)

        # Create OffsetImage
        robot_image = OffsetImage(robot_img, zoom=zoom_factor, zorder=1000)
        ab = AnnotationBbox(robot_image, (0, 0),
                            frameon=False, pad=0,
                            box_alignment=(0.5, 0.5),
                            zorder=1000)
        ax.add_artist(ab)
        print(f"  Loaded robot image: {img_size}, zoom={zoom_factor:.4f}")

    plt.tight_layout(pad=0)

    # Save figure
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1,
               facecolor='#FBFBFB', edgecolor='none')
    print(f"Figure saved to {save_path}")

    plt.show()


def generate_failed_trajectories(
    target_goal: str = 'top_right',
    num_failed: int = 8,
    num_steps: int = 40
) -> List[np.ndarray]:
    """
    Generate failed trajectories that start from center and head towards a goal
    but don't reach it (they stop early or veer off course).

    Args:
        target_goal: Name of the target goal
        num_failed: Number of failed trajectories to generate
        num_steps: Maximum number of steps per trajectory

    Returns:
        List of failed trajectories as numpy arrays
    """
    failed_trajs = []
    goal_pos = GOAL_CONFIG[target_goal]['pos']

    for i in range(num_failed):
        # Start from center
        traj = [[0.0, 0.0]]

        # Direction towards goal with some noise
        direction = goal_pos / np.linalg.norm(goal_pos)

        # Add randomness
        np.random.seed(i + 42)
        noise = np.random.normal(0, 0.15, 2)
        direction = direction + noise
        direction = direction / np.linalg.norm(direction)

        # Random step size
        step_size = 0.08 + np.random.uniform(-0.02, 0.03)

        # Number of steps (less than max_steps)
        max_steps_for_this_traj = np.random.randint(15, 35)

        for step in range(max_steps_for_this_traj):
            current_pos = np.array(traj[-1])

            # Add curvature
            if step > 5:
                perp_direction = np.array([-direction[1], direction[0]])
                curvature = perp_direction * np.sin(step * 0.15) * 0.3
                slowdown = max(0.3, 1.0 - step * 0.02)
                new_pos = current_pos + direction * step_size * slowdown + curvature * 0.5
            else:
                new_pos = current_pos + direction * step_size

            traj.append(new_pos.tolist())

            # Check if too close to goal (shouldn't happen)
            dist_to_goal = np.linalg.norm(np.array(new_pos) - goal_pos)
            if dist_to_goal < 0.15:
                break

        failed_trajs.append(np.array(traj))

    return failed_trajs
