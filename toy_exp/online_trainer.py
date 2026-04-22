from time import time
import math
import numpy as np
import torch
from tqdm import trange
from tensordict.tensordict import TensorDict
from boom.trainer.base import Trainer
import matplotlib.pyplot as plt
import os
from toy_exp.eval_plot_utils import plot_final_trajectories, plot_buffer_stats


class OnlineTrainer(Trainer):
    """Trainer class for single-task online training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._nan_tensor = None
        self.replay_sample_list = []
        self._last_save_step = 0  # Track last save step for checkpointing

    def common_metrics(self):
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def save_checkpoint(self, identifier):
        """Save model checkpoint to disk.
        
        Args:
            identifier: Step number or name to identify this checkpoint
        """
        self.logger.save_agent(self.agent, identifier=str(identifier))
        print(f"Checkpoint saved at step {self._step} with identifier: {identifier}")

    @torch.no_grad()
    def eval(self):
        """
        Evaluation function with standardized plotting.
        Creates eval/step_{}/ folder with:
        - pi_mppi.png: 2x2 grid of PI-MPPI visualization
        - flow_mppi.png: 2x2 grid of Flow-MPPI visualization
        - buffer_stat.png: Action-reward distribution from buffer
        """
        ep_rewards, ep_successes = [], []
        mppi_debug_data = []  # Collect MPPI debug info from first episode
        real_trajectories = []  # Collect real trajectories from all episodes

        # Run evaluation episodes
        for i in trange(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            trajectory = []

            # Get initial position
            if hasattr(self.env, '_env'):
                trajectory.append(self.env._env.agent_pos.copy())
            elif hasattr(self.env, 'agent_pos'):
                trajectory.append(self.env.agent_pos.copy())

            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))

            while not done:
                # Get debug info only for first episode, first timestep
                need_debug = (i == 0 and t == 0)
                result = self.agent.act(obs, t0=(t == 0), eval_mode=True, debug=need_debug)

                if need_debug:
                    action, _, _, debug_info = result
                    if debug_info:
                        mppi_debug_data.append(debug_info)
                else:
                    action, _, _ = result

                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1

                # Record trajectory
                if hasattr(self.env, '_env'):
                    trajectory.append(self.env._env.agent_pos.copy())
                elif hasattr(self.env, 'agent_pos'):
                    trajectory.append(self.env.agent_pos.copy())

                if self.cfg.save_video:
                    self.logger.video.record(self.env)

            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])

            # Save trajectory data
            real_trajectories.append({
                'trajectory': trajectory,
                'reached_goal': info.get('reached_which', None),
                'reward': ep_reward
            })

            if self.cfg.save_video:
                self.logger.video.save(self._step, key='results/video')

        # Create standardized evaluation plots
        eval_save_dir = os.path.join(self.cfg.work_dir, 'eval', f'step_{self._step}')
        os.makedirs(eval_save_dir, exist_ok=True)

        # Plot final trajectories
        final_trajs_path = os.path.join(eval_save_dir, 'final_trajectories.png')
        plot_final_trajectories(real_trajectories, self.env, final_trajs_path)

        # Plot MPPI debug figures
        if mppi_debug_data and len(mppi_debug_data) > 0:
            debug_info = mppi_debug_data[0]  # Use first episode data

            # Get environment parameters
            step_size = getattr(self.env, 'step_size', 0.1)
            goal_radius = getattr(self.env, 'goal_radius', 0.1)

            try:
                # Plot combined figure and action distribution
                from toy_exp.eval_plot_utils import plot_combined_mppi_figure, plot_action_distribution
                plot_combined_mppi_figure(debug_info, self.env, eval_save_dir, step_size, goal_radius)
                plot_action_distribution(debug_info, self.env, eval_save_dir)
            except Exception as e:
                print(f"Error plotting MPPI: {e}")
                import traceback
                traceback.print_exc()

        print(f"Evaluation plots saved to {eval_save_dir}")


        # Evaluate PI policy if enabled
        if self.cfg.eval_pi:
            ep_rewards_pi, ep_successes_pi = [], []
            for i in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
                while not done:
                    action, _, _ = self.agent.act(obs, t0=(t == 0), eval_mode=True, use_pi=True)
                    obs, reward, done, truncated, info = self.env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    t += 1
                ep_rewards_pi.append(ep_reward)
                ep_successes_pi.append(info["success"])
        else:
            ep_rewards_pi, ep_successes_pi = [np.nan], [np.nan]

        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_reward_pi=np.nanmean(ep_rewards_pi),
            episode_success_pi=np.nanmean(ep_successes_pi),
        )

    @torch.no_grad()
    def eval_value(self, n_samples=100):
        mc_ep_rewards, q_values = [], []
        device = self.agent.device

        for _ in range(n_samples):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            while not done:
                action, _, _ = self.agent.act(obs, t0=(t == 0), eval_mode=True, use_pi=True)
                obs, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                ep_reward += reward * (self.agent.discount ** t)
                t += 1
            mc_ep_rewards.append(ep_reward)

        for _ in range(n_samples):
            obs = self.env.reset()[0]
            action, _, _ = self.agent.act(obs, t0=True, eval_mode=True, use_pi=True)
            task = None
            obs_encoded = self.agent.model.encode(obs.to(device), task)
            q_val = self.agent.model.Q(obs_encoded, action.to(device), task, return_type="avg")
            q_values.append(q_val.item())

        return dict(
            mc_value=np.nanmean(mc_ep_rewards),
            q_value=np.nanmean(q_values),
        )

    def plot_action_reward_distribution(self, replay_action, replay_reward, step):
        """Plot action-reward distribution as scatter plots.

        Args:
            replay_action: Tensor of shape [T, B, D] where T is time steps, B is batch size, D is action dim
            replay_reward: Tensor of shape [T, B, 1]
            step: Current training step for filename
        """
        # Convert to numpy if needed
        if torch.is_tensor(replay_action):
            replay_action = replay_action.detach().cpu().numpy()
        if torch.is_tensor(replay_reward):
            replay_reward = replay_reward.detach().cpu().numpy()

        T, B, D = replay_action.shape

        # Create figure with T columns and D rows
        fig, axes = plt.subplots(D, T, figsize=(4 * T, 3 * D), squeeze=False)

        # Plot scatter plots for each timestep and action dimension
        for t in range(T):
            for d in range(D):
                ax = axes[d, t]
                # Extract action values for dimension d across all batches
                action_values = replay_action[t, :, d]  # Shape: [B]
                reward_values = replay_reward[:, 0]  # Shape: [B]

                # Create scatter plot
                ax.scatter(action_values, reward_values, alpha=0.5, s=20)
                ax.set_xlabel(f'Action')
                ax.set_ylabel('Reward')
                ax.set_title(f'Timestep {t}, Action dim {d}')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        save_dir = os.path.join(self.cfg.work_dir, 'action_reward_plots')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'action_reward_dist_step_{step}.png')
        print(f"Buffer plot saved to {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Action-reward distribution plot saved to {save_path}")

    def to_td(self, obs, action=None, mu=None, std=None, reward=None):
        """Creates a TensorDict for a new episode."""
        obs = TensorDict(obs, batch_size=(), device="cpu") if isinstance(obs, dict) else obs.unsqueeze(0).cpu()

        if action is None:
            action = self.env.rand_act()
        if mu is None:
            mu = action.clone()
        if std is None:
            std = torch.full_like(action, math.exp(self.cfg.log_std_max))
        if reward is None:
            reward = torch.tensor(float("nan"))

        td_dict = {
            "obs": obs,
            "action": action.unsqueeze(0),
            "mu": mu.unsqueeze(0),
            "std": std.unsqueeze(0),
            "reward": reward.unsqueeze(0),
        }
        return TensorDict(td_dict, batch_size=(1,))

    def train(self):
        train_metrics, done, eval_next = {}, True, True

        while self._step <= self.cfg.steps:
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True
            
            # Periodic checkpoint saving
            save_freq = getattr(self.cfg, 'save_freq', 100000)  # Default: every 100k steps
            if self._step > 0 and self._step - self._last_save_step >= save_freq:
                self.save_checkpoint(f"step_{self._step}")
                self._last_save_step = self._step
            
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    if self.cfg.eval_value:
                        eval_metrics.update(self.eval_value())
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")

                    # Plot buffer statistics
                    if self._step > self.cfg.seed_steps:
                        # Sample a batch from replay buffer and estimate values
                        replay_obs, replay_action, replay_mu, replay_std, replay_reward, replay_task = self.buffer.sample()

                        # Stack into tensors if they are lists
                        if isinstance(replay_obs, list):
                            replay_obs = torch.stack(replay_obs)
                        if isinstance(replay_action, list):
                            replay_action = torch.stack(replay_action)

                        # Encode observations to latent space
                        device = self.agent.device
                        replay_z = self.agent.model.encode(replay_obs[0].to(device), None)

                        # Estimate values using the model
                        horizon = getattr(self.agent.cfg, 'horizon', 1)
                        replay_reward_estimated = self.agent._estimate_value(
                            replay_z,
                            replay_action.to(device),
                            None,
                            horizon
                        )

                        # Plot buffer statistics
                        eval_save_dir = os.path.join(self.cfg.work_dir, 'eval', f'step_{self._step}')
                        plot_buffer_stats(
                            replay_action,
                            replay_reward_estimated,
                            self._step,
                            eval_save_dir
                        )

                    eval_next = False
      
                if self._step > 0:
                    rewards = torch.tensor([td["reward"] for td in self._tds[1:]])
                    train_metrics.update(
                        episode_reward=rewards.sum(),
                        episode_success=info["success"],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                    self.logger.log({
                        'return': train_metrics['episode_reward'],
                        'episode_length': len(self._tds[1:]),
                        'success': train_metrics['episode_success'],
                        'success_subtasks': info.get('success_subtasks', None),
                        'step': self._step,
                    }, "results")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]

            if self._step > self.cfg.seed_steps:
                action, mu, std = self.agent.act(obs, t0=(len(self._tds) == 1))
            else:
                # action, mu, std = self.agent.act(obs, t0=(len(self._tds) == 1))
                action = self.env.rand_act()
                mu, std = action.clone(), torch.full_like(action, math.exp(self.cfg.log_std_max))

            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            self._tds.append(self.to_td(obs, action, mu, std, reward))
            
            if self._step >= self.cfg.seed_steps:
                if self._step % 100 == 0 \
                  or len(self.replay_sample_list) == 0 \
                  or self.count >= 100:
                    self.replay_sample_list = []
                    # print("Replaying new data from buffer...")
                    for _ in range(100):
                        replay_obs, replay_action, replay_mu, replay_std, replay_reward, replay_task = self.buffer.sample()
                        self.replay_sample_list.append(
                            (replay_obs, replay_action, replay_mu, replay_std, replay_reward, replay_task)
                        )
                    self.count = 0

                replay_sample = self.replay_sample_list[self.count]
                self.count += 1
                
                num_updates = self.cfg.seed_steps if self._step == self.cfg.seed_steps else 1
                if self._step == self.cfg.seed_steps:
                    print(f"Pretraining agent on seed data...({self._step})")
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(replay_sample, self._step)
                    train_metrics.update(_train_metrics)
            self._step += 1

        self.logger.finish(self.agent)
