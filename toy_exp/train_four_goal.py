"""
Simple training script for four-goal navigation using existing BOOM infrastructure.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import warnings
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf
from boom.envs import make_env
from boom.boom_alg import BOOM
from boom.trainer.online_trainer import OnlineTrainer
from boom.common.buffer import Buffer
from boom.common.logger import Logger


def get_config(policy_type='mlp', seed=1, steps=10_000):
    """Get training configuration as OmegaConf."""
    cfg = {
        # Task and environment
        'task': 'four-goal-nav',
        'env_type': 'four_goal',
        'obs': 'state',

        # Environment parameters
        'step_size': 0.40,
        'goal_radius': 0.1,
        'max_steps': 40,
        'step_penalty': -0.01,

        # Training parameters
        'steps': steps,
        'batch_size': 256,
        'reward_coef': 0.1,
        'value_coef': 0.1,
        'consistency_coef': 20,
        'rho': 0.5,
        'lr': 3e-4,
        'enc_lr_scale': 0.3,
        'grad_clip_norm': 20,
        'tau': 0.01,
        'discount_denom': 5,
        'discount_min': 0.95,
        'discount_max': 0.995,
        'buffer_size': 500_000,

        # MPPI planning parameters
        'mpc': True,
        'iterations': 20,
        'num_samples': 512,
        'num_elites': 32,
        'num_pi_trajs': 24,
        'num_flow_trajs': 48 if policy_type == 'flow' else 0,
        'horizon': 3,
        'min_std': 0.05,
        'max_std': 1.0,
        'temperature': 10,

        # Flow-specific parameters
        'update_flow': (policy_type == 'flow'),
        'flow_q_coef': 0.0,

        # Actor parameters
        'log_std_min': -10,
        'log_std_max': 2,
        'entropy_coef': 1e-4,

        # Critic parameters
        'num_bins': 101,
        'vmin': -10,
        'vmax': +10,

        # Architecture
        'model_size': 'normal',
        'num_enc_layers': 1,
        'enc_dim': 32,
        'num_channels': 32,
        'mlp_dim': 64,
        'latent_dim': 64,
        'task_dim': 0,
        'num_q': 5,
        'num_v': 5,
        'dropout': 0.01,
        'simnorm_dim': 8,

        # Experiment setup
        'seed': seed,
        'multitask': False,

        # Evaluation
        'eval_episodes': 10,
        'eval_freq': 1000,
        'eval_pi': False,
        'eval_value': False,
        'eval_flow': (policy_type == 'flow'),
        'eval_mode': False,
        'save_video': False,
        'save_traj': True,  # Save trajectories for visualization

        # Logging
        'exp_name': f'four_goal_{policy_type}_seed{seed}',
        'extra': '',
        'wandb_project': 'four-goal-navigation',
        'wandb_entity': 'shallowdream0745-thu',
        'wandb_silent': False,
        'disable_wandb': False,
        'save_csv': True,
        'save_agent': True,
        'checkpoint': None,
    }

    # Set working directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = project_root / 'toy_exp' / 'results' / cfg['exp_name'] / timestamp
    os.makedirs(work_dir, exist_ok=True)
    cfg['work_dir'] = work_dir  # Keep as Path object
    cfg['data_dir'] = work_dir

    # Convert to OmegaConf (but keep Path objects)
    cfg = OmegaConf.create(cfg)

    # Manually set fields that parse_cfg would normally set
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.task_title = cfg.task.replace("-", " ").title()

    return cfg


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train four-goal navigation with BOOM')
    parser.add_argument('--policy_type', type=str, default='mlp',
                        choices=['mlp', 'flow'],
                        help='Policy type')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Number of training steps')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--eval_freq', type=int, default=500,
                        help='Evaluation frequency')

    args = parser.parse_args()

    # Get config
    cfg = get_config(policy_type=args.policy_type, seed=args.seed, steps=args.steps)
    cfg.eval_freq = args.eval_freq

    # Print config
    print("=" * 70)
    print(f"Training Configuration: {args.policy_type.upper()} Policy")
    print("=" * 70)
    print(f"Steps: {cfg.steps:,}")
    print(f"Seed: {cfg.seed}")
    print(f"Update_flow: {cfg.update_flow}")
    print(f"Work Dir: {cfg.work_dir}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)
    print()

    # Set seed
    torch.manual_seed(cfg.seed)
    import numpy as np
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Create environment, agent, buffer, and logger
    env = make_env(cfg)
    agent = BOOM(cfg)
    buffer = Buffer(cfg)
    logger = Logger(cfg)

    # Create trainer and train
    trainer = OnlineTrainer(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=buffer,
        logger=logger,
    )

    print("Starting training...")
    trainer.train()

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
