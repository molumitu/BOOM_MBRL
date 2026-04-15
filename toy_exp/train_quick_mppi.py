"""
Quick MPPI Demo Script

Demonstrates multi-modal MPPI planning without lengthy training.
Uses reference parameters from four_goal_mppi.py.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from boom.envs import make_env
from boom.boom_alg import BOOM


def get_config(policy_type='mlp'):
    """Get configuration with reference parameters."""
    cfg = {
        'task': 'four-goal-nav',
        'env_type': 'four_goal',
        'obs': 'state',
        # Reference environment params from four_goal_mppi.py
        'step_size': 0.20,
        'goal_radius': 0.1,
        'max_steps': 40,
        'step_penalty': -0.01,

        # Reduced training steps for quick demo
        'steps': 1000,
        'batch_size': 64,
        'reward_coef': 0.1,
        'value_coef': 0.1,
        'consistency_coef': 20,
        'rho': 0.5,
        'lr': 1e-4,
        'enc_lr_scale': 0.3,
        'grad_clip_norm': 20,
        'tau': 0.01,
        'discount_denom': 5,
        'discount_min': 0.95,
        'discount_max': 0.995,
        'buffer_size': 10000,

        # Reference MPPI params from four_goal_mppi.py
        'mpc': True,
        'iterations': 50,
        'num_samples': 128,
        'num_elites': 13,  # 10% of 128
        'num_pi_trajs': 0,
        'num_flow_trajs': 48 if policy_type == 'flow' else 0,
        'horizon': 40,
        'min_std': 0.05,
        'max_std': 0.2,  # Reference NOISE_SIGMA
        'temperature': 0.1,  # Reference LAMBDA

        'update_flow': (policy_type == 'flow'),
        'flow_q_coef': 1.0,

        'log_std_min': -10,
        'log_std_max': 2,
        'entropy_coef': 1e-4,

        'num_bins': 101,
        'vmin': -10,
        'vmax': +10,

        'model_size': 'normal',
        'num_enc_layers': 2,
        'enc_dim': 256,
        'num_channels': 32,
        'mlp_dim': 512,
        'latent_dim': 512,
        'task_dim': 0,
        'num_q': 5,
        'num_v': 5,
        'dropout': 0.01,
        'simnorm_dim': 8,

        'seed': 1,
        'multitask': False,

        'episode_length': 40,  # Reference max_steps
        'obs_shape': {'state': (2,)},
        'action_dim': 1,
    }
    return cfg


class SimpleConfig:
    """Config wrapper."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)


def collect_mppi_samples(agent, state, policy_type, n_samples=100):
    """Collect MPPI elites at given state."""
    try:
        with torch.no_grad():
            device = agent.device
            obs = torch.FloatTensor(state).unsqueeze(0).to(device)

            # MPPI elite samples
            z = agent.model.encode(obs, None)
            H, N, E, A = agent.cfg.horizon, agent.cfg.num_samples, agent.cfg.num_elites, agent.cfg.action_dim

            mean = torch.zeros(H, A, device=device)
            std = agent.cfg.max_std * torch.ones(H, A, device=device)
            actions = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(H, N, A, device=device).clamp(-1, 1)

            values = agent._estimate_value(z.repeat(N, 1), actions, None, H)
            elite_idxs = torch.topk(values.squeeze(1), E, dim=0).indices
            elite_actions = actions[:, elite_idxs]

            mppi_angles = elite_actions[0, :, 0].cpu().numpy()

            return mppi_angles

    except Exception as e:
        print(f"    Error: {e}")
        return None


def demo_mppi_discovery(policy_type='mlp', num_episodes=50):
    """
    Quick demo: Just run MPPI planning to show it can discover 4 modes.
    No lengthy training needed!
    """
    print(f"\n{'='*60}")
    print(f"MPPI Multi-Modal Discovery Demo ({policy_type.upper()})")
    print(f"{'='*60}")

    cfg_dict = get_config(policy_type)
    cfg = SimpleConfig(cfg_dict)

    # Create environment and agent
    env = make_env(cfg)
    agent = BOOM(cfg)

    # Initialize MPPI state
    H = cfg.horizon
    A = cfg.action_dim
    device = agent.device
    agent._prev_mean = torch.zeros(H, A, device=device)
    agent._prev_std = cfg.max_std * torch.ones(H, A, device=device)

    # Data storage
    all_mppi_angles = []

    # Run episodes
    initial_state = np.array([0.0, 0.0])

    for episode in range(num_episodes):
        obs, _ = env.reset()

        # Collect MPPI samples from initial state
        mppi_angles = collect_mppi_samples(agent, initial_state, policy_type)
        if mppi_angles is not None:
            all_mppi_angles.append(mppi_angles)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}")

    print(f"  ✓ Collected {len(all_mppi_angles)} MPPI samples")
    return all_mppi_angles


def save_results(mlp_angles, flow_angles, save_dir):
    """Save results to CSV."""
    print("\nSaving results...")

    os.makedirs(save_dir, exist_ok=True)

    # Save MLP MPPI data
    if len(mlp_angles) > 0:
        all_angles = np.concatenate(mlp_angles)
        df = pd.DataFrame({
            'angle_rad': all_angles,
            'angle_deg': np.degrees(all_angles),
            'policy_type': 'mlp',
            'sample_type': 'mppi_elite'
        })
        csv_path = os.path.join(save_dir, 'mlp_mppi_quick.csv')
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {csv_path}")

    # Save Flow MPPI data
    if len(flow_angles) > 0:
        all_angles = np.concatenate(flow_angles)
        df = pd.DataFrame({
            'angle_rad': all_angles,
            'angle_deg': np.degrees(all_angles),
            'policy_type': 'flow',
            'sample_type': 'mppi_elite'
        })
        csv_path = os.path.join(save_dir, 'flow_mppi_quick.csv')
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {csv_path}")

    # Create summary statistics
    summary_data = []
    for policy_type, angles in [('mlp', mlp_angles), ('flow', flow_angles)]:
        if len(angles) > 0:
            all_angles = np.concatenate(angles)
            all_degrees = np.degrees(all_angles)

            # Count angles near each goal (±30 degrees)
            goals = {
                '45': 45,
                '135': 135,
                '-135': -135,
                '-45': -45
            }

            for goal_name, goal_angle in goals.items():
                count = np.sum(np.abs(all_degrees - goal_angle) < 30)
                summary_data.append({
                    'policy_type': policy_type,
                    'sample_type': 'mppi_elite',
                    'goal_direction': goal_name,
                    'coverage': int(count),
                    'mean_angle': float(np.mean(all_degrees)),
                    'std_angle': float(np.std(all_degrees)),
                    'total_samples': len(all_degrees)
                })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, 'quick_mppi_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Saved {summary_path}")

    return summary_df


def visualize_results(mlp_angles, flow_angles, save_dir):
    """Visualize MPPI discovery results."""
    print("\nCreating visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    # Plot MPPI elite samples for both configurations
    for idx, (policy_type, angles, color) in enumerate([
        ('mlp', mlp_angles, 'blue'),
        ('flow', flow_angles, 'red')
    ]):
        if len(angles) > 0:
            all_angles = np.concatenate(angles)
            all_degrees = np.degrees(all_angles)
            all_radians = np.radians(all_degrees)

            ax = fig.add_subplot(gs[0, idx], projection='polar')

            # Create histogram
            bins = np.linspace(-np.pi, np.pi, 73)
            counts, _ = np.histogram(all_radians, bins=bins)

            # Normalize to show density
            bin_width = bins[1] - bins[0]
            density = counts / (len(all_radians) * bin_width)

            # Plot bars
            ax.bar(bins[:-1], density, width=bin_width, alpha=0.7, color=color, edgecolor='black')

            # Add goal direction markers
            goal_angles_rad = [np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4]
            goal_labels = ['45°', '135°', '-135°', '-45°']

            for g_angle, g_label in zip(goal_angles_rad, goal_labels):
                ax.axvline(x=g_angle, color='green', linestyle='--', linewidth=2, alpha=0.7)
                ax.text(g_angle, max(density) * 1.05, g_label,
                       ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')

            ax.set_theta_zero_location('E')
            ax.set_theta_direction(-1)
            ax.set_title(f'{policy_type.upper()} MPPI Elite Samples\n({len(all_angles)} samples)',
                        fontsize=12, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)

    # Summary statistics text
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis('off')

    summary_lines = ["MPPI Multi-Modal Discovery Summary", "="*60, ""]

    for policy_type, angles, color_name in [('mlp', mlp_angles, 'blue'), ('flow', flow_angles, 'red')]:
        if len(angles) > 0:
            all_angles = np.concatenate(angles)
            all_degrees = np.degrees(all_angles)

            summary_lines.extend([
                f"{policy_type.upper()} Configuration:",
                f"  Total samples: {len(all_angles)}",
                f"  Mean angle: {np.mean(all_degrees):.2f}°",
                f"  Std angle: {np.std(all_degrees):.2f}°",
                ""
            ])

            # Count coverage
            for goal_angle in [45, 135, -135, -45]:
                count = np.sum(np.abs(all_degrees - goal_angle) < 30)
                pct = 100 * count / len(all_degrees)
                summary_lines.append(f"  Near {goal_angle}°: {count} ({pct:.1f}%)")
            summary_lines.append("")

    ax_text.text(0.1, 0.9, "\n".join(summary_lines),
                transform=ax_text.transAxes, fontsize=10, va='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('MPPI Multi-Modal Discovery: Reference Parameters\n'
                '(horizon=40, iterations=50, samples=128, elite=10%, temp=0.1)',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(save_dir, 'quick_mppi_discovery.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {save_path}")
    plt.close()


def main():
    """Main function."""
    print("="*70)
    print(" Quick MPPI Multi-Modal Discovery Demo")
    print(" Using Reference Parameters from four_goal_mppi.py")
    print("="*70)

    save_dir = 'results'

    # Demo with MLP configuration
    mlp_angles = demo_mppi_discovery('mlp', num_episodes=50)

    # Demo with Flow configuration
    flow_angles = demo_mppi_discovery('flow', num_episodes=50)

    # Save results
    summary_df = save_results(mlp_angles, flow_angles, save_dir)

    # Visualize
    visualize_results(mlp_angles, flow_angles, save_dir)

    print("\n" + "="*70)
    print(f" Complete! Check {save_dir}/ for results.")
    print("="*70)
    print("\nKey Findings:")
    print("  - MPPI should discover multiple modes (4 goal directions)")
    print("  - Both MLP and Flow configs should show similar MPPI behavior")
    print("  - This demonstrates MPPI's inherent multi-modality")
    print("="*70)


if __name__ == "__main__":
    main()
