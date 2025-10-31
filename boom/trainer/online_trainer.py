from time import time
import math
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from boom.trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._nan_tensor = None
        self.replay_sample_list = []
        

    def common_metrics(self):
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    @torch.no_grad()
    def eval(self):
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                action, _, _ = self.agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                self.logger.video.save(self._step, key='results/video')

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

        return TensorDict({
            "obs": obs,
            "action": action.unsqueeze(0),
            "mu": mu.unsqueeze(0),
            "std": std.unsqueeze(0),
            "reward": reward.unsqueeze(0),
        }, batch_size=(1,))

    def train(self):
        train_metrics, done, eval_next = {}, True, True

        while self._step <= self.cfg.steps:
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    if self.cfg.eval_value:
                        eval_metrics.update(self.eval_value())
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
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
                    print("Pretraining agent on seed data...")
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(replay_sample, self._step)
                    train_metrics.update(_train_metrics)
            self._step += 1

        self.logger.finish(self.agent)
