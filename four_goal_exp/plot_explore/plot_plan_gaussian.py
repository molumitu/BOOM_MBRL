"""
Plot Example: Generate and visualize trajectories

This script generates simulated trajectories from origin with smooth random
perturbations and creates a clean visualization with robot icons, directional
arrows, and goal markers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.lines import Line2D
import os
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

# Goal configuration (same as in four_goal_mppi.py)
# Colors from plot_gaussian.py: blue, green, orange, purple
GOAL_CONFIG = {
    'top_left': {
        'pos': np.array([-1.0, 1.0]),
        'color': '#9467bd',  # purple
        'optimal_angle': 3 * np.pi / 4,
        'label': 'Top Left'
    },
    'top_right': {
        'pos': np.array([1.0, 1.0]),
        'color': '#1f77b4',  # blue
        'optimal_angle': np.pi / 4,
        'label': 'Top Right'
    },
    'bottom_left': {
        'pos': np.array([-1.0, -1.0]),
        'color': '#2ca02c',  # green
        'optimal_angle': -3 * np.pi / 4,
        'label': 'Bottom Left'
    },
    'bottom_right': {
        'pos': np.array([1.0, -1.0]),
        'color': '#ff7f0e',  # orange
        'optimal_angle': -np.pi / 4,
        'label': 'Bottom Right'
    }
}


def plot_four_goal_trajectories_clean(
    results,
    env_params: dict,
    save_path: str = "mppi_best_trajectories_clean.png",
    figsize: tuple = (10, 10)
) -> None:
    """
    Plot all best trajectories with clean aesthetics - no axes, borders, or legends.

    Args:
        results: List of experiment results
        env_params: Environment parameters
        save_path: Path to save the figure
        figsize: Figure size
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
    arrow_length = 0.8
    arrow_start_dist = 0.2

    for goal_name, goal_info in GOAL_CONFIG.items():
        angle = goal_info['optimal_angle']
        start_x = arrow_start_dist * np.cos(angle)
        start_y = arrow_start_dist * np.sin(angle)
        end_x = (arrow_start_dist + arrow_length) * np.cos(angle)
        end_y = (arrow_start_dist + arrow_length) * np.sin(angle)

        # Only color top_right arrow, others are gray
        if goal_name == 'top_right':
            arrow = FancyArrow(start_x, start_y, end_x - start_x, end_y - start_y,
                        width=0.1, head_width=0.3, head_length=0.3,
                        color=goal_info['color'], alpha=0.8, zorder=8,
                        length_includes_head=True)
        else:
            arrow = FancyArrow(start_x, start_y, end_x - start_x, end_y - start_y,
                        width=0.1, head_width=0.3, head_length=0.3,
                        color='gray', alpha=0.4, zorder=8,
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

    # Plot trajectories for each goal (successful trajectories)
    for goal_name, goal_info in GOAL_CONFIG.items():
        for traj in trajectories_by_goal[goal_name]:
            traj_array = np.array(traj)
            # Only color top_right (blue) trajectories, others are gray
            if goal_name == 'top_right':
                ax.plot(traj_array[:, 0], traj_array[:, 1],
                    color=goal_info['color'], alpha=0.4, linewidth=3,
                    solid_capstyle='round', zorder=6)

    # Plot unsuccessful trajectories (gray, dashed) - only 30% for clarity
    if len(trajectories_to_none) > 0:
        # Randomly sample 30% of unsuccessful trajectories
        sample_ratio = 0.2
        sample_size = int(len(trajectories_to_none) * sample_ratio)
        sampled_indices = np.random.choice(len(trajectories_to_none),
                                          size=sample_size, replace=False)

        for idx in sampled_indices:
            traj = trajectories_to_none[idx]
            traj_array = np.array(traj)
            # Unsuccessful trajectories: dashed line, lower alpha, same thickness
            ax.plot(traj_array[:, 0], traj_array[:, 1],
                    'gray', alpha=0.5, linewidth=1.5, linestyle='--',
                    solid_capstyle='round', zorder=6)

    # Draw red circle at origin (starting point)
    origin_circle = Circle((0, 0), 0.05, color='red', alpha=0.8, zorder=7)
    ax.add_patch(origin_circle)

    # Add statistics as text overlay
    num_experiments = len(results)
    total_success = sum(len(trajs) for trajs in trajectories_by_goal.values())
    success_rate = total_success / num_experiments
    avg_reward = np.mean([r['best_total_reward'] for r in results])

    # Draw robot agent from PNG file - ADD LAST to ensure it's on top
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(script_dir, 'robot.png')

    # Load the PNG image
    robot_img = Image.open(png_path)
    img_size = robot_img.size  # (width, height)

    # Apply alpha transparency to the image
    robot_img = robot_img.convert('RGBA')
    # Modify alpha channel: multiply by 0.9
    r, g, b, a = robot_img.split()
    a = a.point(lambda x: int(x * 0.8))
    robot_img = Image.merge('RGBA', (r, g, b, a))

    # Calculate appropriate zoom to make robot visible
    target_size = 200
    zoom_factor = target_size / max(img_size)
    # zoom_factor = 1.0

    # Create OffsetImage with very high zorder to ensure it's on top of everything
    robot_image = OffsetImage(robot_img, zoom=zoom_factor, zorder=1000)

    # Add the robot image to the plot centered at (0, 0)
    # Using add_artist instead of AnnotationBbox for better zorder control
    ab = AnnotationBbox(robot_image, (0, 0),
                        frameon=False, pad=0,
                        box_alignment=(0.5, 0.5),
                        zorder=1000)
    ax.add_artist(ab)
    print(f"  Loaded robot image: {img_size}, zoom={zoom_factor:.4f}")

    # Add statistics as text overlay
    num_experiments = len(results)
    total_success = sum(len(trajs) for trajs in trajectories_by_goal.values())
    success_rate = total_success / num_experiments
    avg_reward = np.mean([r['best_total_reward'] for r in results])

    # # Create stats text
    # stats_text = f"N={num_experiments}\nSuccess: {success_rate:.0%}\nAvg Reward: {avg_reward:.2f}"

    # # Add stats text in corner
    # ax.text(-1.1, 1.05, stats_text, fontsize=10, ha='left', va='top',
    #        color='#555555', fontfamily='monospace', fontweight='bold',
    #        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
    #                 edgecolor='#DDDDDD', alpha=0.9, linewidth=1.5), zorder=15)

    plt.tight_layout(pad=0)

    # Save figure with the same background color
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1,
               facecolor='#FBFBFB', edgecolor='none')
    print(f"Figure saved to {save_path}")

    plt.show()


def generate_trajectories(
    num_trajectories: int = 400,
    max_steps: int = 300,
    step_size: float = 0.01,
    goal_radius: float = 0.1
) -> list:
    """
    Generate smooth trajectories from origin using angle inertia for smoothness.

    The angle inertia method limits the rate of angle change between steps,
    creating smooth curves instead of jittery random walks.

    Args:
        num_trajectories: Number of trajectories to generate
        max_steps: Maximum steps per trajectory
        step_size: Step size for movement (smaller = smoother)
        goal_radius: Radius for goal detection

    Returns:
        List of trajectory dictionaries compatible with plot function
    """
    trajectories = []
    goal_positions = [g['pos'] for g in GOAL_CONFIG.values()]
    goal_names = list(GOAL_CONFIG.keys())

    # Smoothness parameters
    angle_std = 2.0  # Standard deviation of noise
    max_angle_change = 0.05  # Maximum angle change per step (radians, ~8.6 degrees)

    for i in range(num_trajectories):
        # Start at origin
        position = np.array([0.0, 0.0])
        trajectory = [position.copy()]

        # Primary direction towards top-right (pi/4) with random variation
        # Using larger variation so some trajectories reach goal, some don't
        primary_angle = np.random.normal(np.pi/4, 0.4)  # mean=pi/4, std=0.4
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

            # Check boundary [ -1.0, 1.0]
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


def main():
    """Main function to generate trajectories and plot."""
    print("=" * 70)
    print("GENERATING AND PLOTTING TRAJECTORIES")
    print("=" * 70)

    # Configuration parameters
    NUM_TRAJECTORIES = 256
    MAX_STEPS = 300
    STEP_SIZE = 0.01
    GOAL_RADIUS = 0.1

    os.makedirs("figures/explore", exist_ok=True)

    # Generate simulated trajectories
    print(f"\nGenerating {NUM_TRAJECTORIES} trajectories...")
    trajectories = generate_trajectories(
        num_trajectories=NUM_TRAJECTORIES,
        max_steps=MAX_STEPS,
        step_size=STEP_SIZE,
        goal_radius=GOAL_RADIUS
    )
    print(f"Generated {len(trajectories)} trajectories")

    # Environment parameters
    env_params = {
        'step_size': STEP_SIZE,
        'goal_radius': GOAL_RADIUS,
        'max_steps': MAX_STEPS,
        'step_penalty': -0.01
    }

    # Print basic statistics
    print("\nBasic Statistics:")
    print(f"  Number of trajectories: {len(trajectories)}")

    goal_success = {g: 0 for g in GOAL_CONFIG.keys()}
    no_success = 0
    for traj in trajectories:
        goal = traj['best_info']['reached_goal']
        if goal in goal_success:
            goal_success[goal] += 1
        else:
            no_success += 1

    print(f"  Success rate: {sum(goal_success.values()) / len(trajectories):.1%}")
    for goal_name, count in goal_success.items():
        label = GOAL_CONFIG[goal_name]['label']
        print(f"    {label}: {count} ({count/len(trajectories):.1%})")
    print(f"    No goal: {no_success} ({no_success/len(trajectories):.1%})")

    # Plot trajectories with clean aesthetics
    print("\nPlotting clean visualization...")
    output_path = "figures/explore/plan_gaussian.png"
    plot_four_goal_trajectories_clean(
        results=trajectories,
        env_params=env_params,
        save_path=output_path,
        figsize=(10, 10)
    )

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
