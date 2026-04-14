"""
Complete Training Script for Four-Goal Navigation Task

Implements full BOOM training pipeline with MPPI planning.
Supports both MLP and Flow-based policies.
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from boom.envs import make_env
from boom.boom_alg import BOOM


class TrainingConfig:
    """Complete training configuration."""

    def __init__(self, policy_type='mlp'):
        # Task and environment
        self.task = 'four-goal-nav'
        self.env_type = 'four_goal'
        self.obs = 'state'

        # Environment parameters
        self.step_size = 0.40
        self.goal_radius = 0.1
        self.max_steps = 40
        self.step_penalty = -0.01

        # Training parameters (COMPLETE)
        self.steps = 100_000
        self.batch_size = 256
        self.reward_coef = 0.1
        self.value_coef = 0.1
        self.consistency_coef = 20
        self.rho = 0.5
        self.lr = 3e-4
        self.enc_lr_scale = 0.3
        self.grad_clip_norm = 20
        self.tau = 0.01
        self.discount_denom = 5
        self.discount_min = 0.95
        self.discount_max = 0.995
        self.buffer_size = 500_000

        # MPPI planning parameters (COMPLETE)
        self.mpc = True
        self.iterations = 20
        self.num_samples = 512
        self.num_elites = 32
        self.num_pi_trajs = 24
        self.num_flow_trajs = 48 if policy_type == 'flow' else 0
        self.horizon = 3
        self.min_std = 0.05
        self.max_std = 2.0
        self.temperature = 0.5

        # Flow-specific parameters
        self.update_flow = (policy_type == 'flow')
        self.flow_q_coef = 1.0
        self.flow_mode = 'action'

        # Actor parameters
        self.log_std_min = -10
        self.log_std_max = 2
        self.entropy_coef = 1e-4

        # Critic parameters
        self.num_bins = 101
        self.vmin = -10
        self.vmax = +10

        # Architecture
        self.model_size = 'normal'
        self.num_enc_layers = 2
        self.enc_dim = 256
        self.num_channels = 32
        self.mlp_dim = 512
        self.latent_dim = 512
        self.task_dim = 0
        self.num_q = 5
        self.num_v = 5
        self.dropout = 0.01
        self.simnorm_dim = 8

        # Experiment setup
        self.seed = 1
        self.multitask = False
        self.episode_length = 40
        self.obs_shape = {'state': (2,)}
        self.action_dim = 1

        # Logging and evaluation
        self.eval_freq = 5_000
        self.eval_episodes = 20
        self.save_freq = 10_000
        self.log_freq = 1_000

        # Policy type
        self.policy_type = policy_type

        # Output directories
        self.exp_name = f'four_goal_{policy_type}_seed{self.seed}'
        self.save_dir = str(project_root / 'toy_exp' / 'results' / self.exp_name)

    def get(self, key, default=None):
        """Dictionary-like get method for compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key):
        """Dictionary-like subscript access."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Dictionary-like subscript assignment."""
        setattr(self, key, value)


class ReplayBuffer:
    """Simple replay buffer for transitions."""

    def __init__(self, capacity, obs_shape, action_dim):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.size = 0
        self.ptr = 0

        # Pre-allocate buffers
        self.obses = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obses = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        """Add transition to buffer."""
        self.obses[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obses[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample random batch from buffer."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'obs': torch.FloatTensor(self.obses[idxs]),
            'action': torch.FloatTensor(self.actions[idxs]),
            'reward': torch.FloatTensor(self.rewards[idxs]),
            'next_obs': torch.FloatTensor(self.next_obses[idxs]),
            'done': torch.FloatTensor(self.dones[idxs]),
        }

        return batch

    def __len__(self):
        return self.size


class ToyTrainer:
    """Complete trainer for four-goal navigation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup
        self._setup_seed()
        self._setup_directories()
        self._setup_logging()

        # Create environment and agent
        self.env = make_env(cfg)
        self.agent = BOOM(cfg, device=self.device)

        # Replay buffer
        obs_shape = cfg.obs_shape['state']
        self.replay_buffer = ReplayBuffer(
            cfg.buffer_size,
            obs_shape,
            cfg.action_dim
        )

        # Training state
        self.global_step = 0
        self.episode_num = 0
        self.best_success_rate = 0.0

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)

    def _setup_seed(self):
        """Setup random seed."""
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

    def _setup_directories(self):
        """Create output directories."""
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        os.makedirs(self.cfg.save_dir + '/checkpoints', exist_ok=True)
        os.makedirs(self.cfg.save_dir + '/logs', exist_ok=True)

    def _setup_logging(self):
        """Setup logging to wandb."""
        wandb.init(
            project='four-goal-navigation',
            name=self.cfg.exp_name,
            config=vars(self.cfg),
            dir=self.cfg.save_dir
        )

    def collect_episode(self, eval_mode=False):
        """Collect one episode of experience."""
        obs, _ = self.env.reset()
        episode_data = {
            'obses': [],
            'actions': [],
            'rewards': [],
            'next_obses': [],
            'dones': [],
            'episode_return': 0.0,
            'episode_length': 0,
            'success': False,
            'reached_goal': None
        }

        for step in range(self.cfg.max_steps):
            # Get action from agent (MPPI planning)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                action_tuple = self.agent.act(obs_tensor, t0=True, eval_mode=eval_mode)
                action = action_tuple[0]  # Already on CPU from act() method

            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            episode_data['obses'].append(obs.cpu().numpy().copy())
            episode_data['actions'].append(action.cpu().numpy().copy())
            episode_data['rewards'].append(reward)
            episode_data['next_obses'].append(next_obs.cpu().numpy().copy())
            episode_data['dones'].append(float(done))

            episode_data['episode_return'] += reward
            episode_data['episode_length'] += 1

            # Check success
            if info.get('success', False):
                episode_data['success'] = True
                episode_data['reached_goal'] = info.get('goal', None)
                break

            obs = next_obs

            if done:
                break

        return episode_data

    def add_episode_to_buffer(self, episode_data):
        """Add episode transitions to replay buffer."""
        n_steps = len(episode_data['obses'])

        for i in range(n_steps):
            self.replay_buffer.add(
                episode_data['obses'][i],
                episode_data['actions'][i],
                episode_data['rewards'][i],
                episode_data['next_obses'][i],
                episode_data['dones'][i]
            )

    def update_agent(self):
        """Update agent using batch from replay buffer."""
        if len(self.replay_buffer) < self.cfg.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.cfg.batch_size)

        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Update agent
        update_info = self.agent.update(batch)

        return update_info

    def evaluate(self):
        """Run evaluation episodes."""
        print(f"\n{'='*60}")
        print(f"Evaluation at step {self.global_step}")
        print(f"{'='*60}")

        eval_results = {
            'episode_returns': [],
            'episode_lengths': [],
            'successes': [],
            'reached_goals': [],
            'initial_actions': []
        }

        for episode in range(self.cfg.eval_episodes):
            episode_data = self.collect_episode(eval_mode=True)

            eval_results['episode_returns'].append(episode_data['episode_return'])
            eval_results['episode_lengths'].append(episode_data['episode_length'])
            eval_results['successes'].append(episode_data['success'])

            if episode_data['reached_goal'] is not None:
                eval_results['reached_goals'].append(episode_data['reached_goal'])

            # Track initial action for diversity analysis
            if len(episode_data['actions']) > 0:
                eval_results['initial_actions'].append(episode_data['actions'][0])

        # Compute metrics
        metrics = {
            'eval/mean_return': np.mean(eval_results['episode_returns']),
            'eval/std_return': np.std(eval_results['episode_returns']),
            'eval/mean_length': np.mean(eval_results['episode_lengths']),
            'eval/success_rate': np.mean(eval_results['successes']),
            'eval/total_successes': np.sum(eval_results['successes']),
        }

        # Goal diversity
        if len(eval_results['reached_goals']) > 0:
            goals = np.array(eval_results['reached_goals'])
            for goal_id in range(4):
                count = np.sum(goals == goal_id)
                metrics[f'eval/goal_{goal_id}_count'] = int(count)

        # Action diversity
        if len(eval_results['initial_actions']) > 0:
            actions = np.array(eval_results['initial_actions'])
            metrics['eval/mean_initial_action'] = float(np.mean(actions))
            metrics['eval/std_initial_action'] = float(np.std(actions))
            metrics['eval/action_entropy'] = float(-np.sum(
                np.histogram(actions, bins=20, density=True)[0] *
                np.log(np.histogram(actions, bins=20, density=True)[0] + 1e-10)
            ))

        # Print summary
        print(f"  Success Rate: {metrics['eval/success_rate']:.2%}")
        print(f"  Mean Return: {metrics['eval/mean_return']:.3f}")
        print(f"  Mean Length: {metrics['eval/mean_length']:.1f}")

        if len(eval_results['reached_goals']) > 0:
            print(f"  Goal Distribution: {[metrics.get(f'eval/goal_{i}_count', 0) for i in range(4)]}")

        print(f"{'='*60}\n")

        # Save checkpoint if best
        if metrics['eval/success_rate'] > self.best_success_rate:
            self.best_success_rate = metrics['eval/success_rate']
            self.save_checkpoint(suffix='best')

        return metrics

    def save_checkpoint(self, suffix=''):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.cfg.save_dir,
            'checkpoints',
            f'checkpoint_{self.global_step}_{suffix}.pt'
        )

        torch.save({
            'global_step': self.global_step,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'best_success_rate': self.best_success_rate,
            'cfg': vars(self.cfg),
        }, checkpoint_path)

        print(f"  ✓ Saved checkpoint to {checkpoint_path}")

    def train(self):
        """Main training loop."""
        print(f"\n{'='*70}")
        print(f" Complete Training: {self.cfg.policy_type.upper()} Policy")
        print(f" Total Steps: {self.cfg.steps:,}")
        print(f" Device: {self.device}")
        print(f"{'='*70}\n")

        pbar = tqdm(total=self.cfg.steps, desc="Training")

        # Training loop
        while self.global_step < self.cfg.steps:
            # Collect episode
            episode_data = self.collect_episode(eval_mode=False)
            self.episode_num += 1

            # Add to replay buffer
            self.add_episode_to_buffer(episode_data)

            # Update metrics
            self.train_metrics['episode_returns'].append(episode_data['episode_return'])
            self.train_metrics['episode_lengths'].append(episode_data['episode_length'])
            self.train_metrics['successes'].append(episode_data['success'])

            # Update agent
            if len(self.replay_buffer) >= self.cfg.batch_size:
                # Multiple updates per episode for efficiency
                for _ in range(min(10, len(self.replay_buffer) // self.cfg.batch_size)):
                    update_info = self.update_agent()

                    if update_info:
                        for k, v in update_info.items():
                            self.train_metrics[k].append(v)

                    self.global_step += 1
                    pbar.update(1)

                    # Periodic evaluation
                    if self.global_step % self.cfg.eval_freq == 0:
                        eval_metrics = self.evaluate()
                        self.eval_metrics.update(eval_metrics)

                        # Log to wandb
                        wandb.log(eval_metrics, step=self.global_step)

                    # Periodic logging
                    if self.global_step % self.cfg.log_freq == 0:
                        recent_returns = self.train_metrics['episode_returns'][-100:]
                        log_dict = {
                            'train/episode_return': np.mean(recent_returns),
                            'train/episode_length': np.mean(self.train_metrics['episode_lengths'][-100:]),
                            'train/success_rate': np.mean(self.train_metrics['successes'][-100:]),
                            'train/buffer_size': len(self.replay_buffer),
                            'train/episodes': self.episode_num,
                        }
                        wandb.log(log_dict, step=self.global_step)

                    # Periodic saving
                    if self.global_step % self.cfg.save_freq == 0:
                        self.save_checkpoint()

                    if self.global_step >= self.cfg.steps:
                        break

        pbar.close()

        # Final evaluation and save
        print(f"\n{'='*70}")
        print(f" Training Complete!")
        print(f"{'='*70}\n")

        final_metrics = self.evaluate()
        self.save_checkpoint(suffix='final')

        # Save training history
        self.save_training_history()

        wandb.finish()

        return final_metrics

    def save_training_history(self):
        """Save training metrics to CSV."""
        # Prepare training data
        train_df = pd.DataFrame({
            'episode_returns': self.train_metrics['episode_returns'],
            'episode_lengths': self.train_metrics['episode_lengths'],
            'successes': self.train_metrics['successes'],
        })

        train_path = os.path.join(self.cfg.save_dir, 'training_history.csv')
        train_df.to_csv(train_path, index=False)
        print(f"  ✓ Saved training history to {train_path}")

        # Prepare evaluation data
        if self.eval_metrics:
            eval_df = pd.DataFrame(self.eval_metrics)
            eval_path = os.path.join(self.cfg.save_dir, 'evaluation_history.csv')
            eval_df.to_csv(eval_path, index=False)
            print(f"  ✓ Saved evaluation history to {eval_path}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train four-goal navigation')
    parser.add_argument('--policy_type', type=str, default='mlp',
                        choices=['mlp', 'flow'],
                        help='Policy type')
    parser.add_argument('--steps', type=int, default=100_000,
                        help='Number of training steps')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--eval_freq', type=int, default=5_000,
                        help='Evaluation frequency')
    parser.add_argument('--save_freq', type=int, default=10_000,
                        help='Save frequency')

    args = parser.parse_args()

    # Create config
    cfg = TrainingConfig(policy_type=args.policy_type)
    cfg.steps = args.steps
    cfg.seed = args.seed
    cfg.eval_freq = args.eval_freq
    cfg.save_freq = args.save_freq

    # Create trainer and train
    trainer = ToyTrainer(cfg)
    final_metrics = trainer.train()

    print(f"\n{'='*70}")
    print(f" Final Results: {cfg.policy_type.upper()} (Seed {cfg.seed})")
    print(f"{'='*70}")
    print(f"  Best Success Rate: {trainer.best_success_rate:.2%}")
    print(f"  Final Success Rate: {final_metrics['eval/success_rate']:.2%}")
    print(f"  Final Mean Return: {final_metrics['eval/mean_return']:.3f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
