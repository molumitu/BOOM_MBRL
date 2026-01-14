import torch
import torch.nn.functional as F
import sys
import os
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boom.common import math
from boom.common.scale import RunningScale
from boom.common.world_model import WorldModel
from boom.bpo_core import _bpograd

class BOOM:
	"""
	Current implementation supports both state and pixel observations.
	"""

	def __init__(self, cfg, device=None):
		self.cfg = cfg
		if device is not None:
			self.device = device
		else:
			if torch.cuda.is_available():
				self.device = torch.device("cuda")
			else:
				self.device = torch.device("cpu")
		self.model = WorldModel(cfg, self.device).to(self.device)
		self.optim = torch.optim.Adam(
			[
				{
					"params": self.model._encoder.parameters(),
					"lr": self.cfg.lr * self.cfg.enc_lr_scale,
				},
				{"params": self.model._dynamics.parameters()},
				{"params": self.model._reward.parameters()},
				{"params": self.model._Qs.parameters()},
				{
					"params": self.model._task_emb.parameters()
					if self.cfg.multitask
					else []
				},
			],
			lr=self.cfg.lr,
		)
		self.pi_optim = torch.optim.Adam(
			self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
		)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.log_pi_scale = RunningScale(cfg) # policy log-probability scale
		self.cfg.iterations += 2 * int(
			cfg.action_dim >= 20
		)  # Heuristic for large action spaces
		self.discount = (
			torch.tensor(
				[self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
				device="cuda",
			)
			if self.cfg.multitask
			else self._get_discount(cfg.episode_length)
		)

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
				episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
				float: Discount factor for the task.
		"""
		frac = episode_length / self.cfg.discount_denom
		return min(
			max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
		)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
				fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
				fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, use_pi=False, use_diffusion=False):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
				obs (torch.Tensor): Observation from the environment.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (int): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		z = self.model.encode(obs, task)
		if self.cfg.mpc and not use_pi and not use_diffusion:
			a, mu, std = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
		elif use_pi:
			mu, pi, log_pi, log_std = self.model.pi(z, task)
			if eval_mode:
				a = mu[0]
			else:
				a = pi[0]
			mu, std = mu[0], log_std.exp()[0]
		return a.cpu(), mu.cpu(), std.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task, horizon, eval_mode=False):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G += discount * reward
			discount *= (
				self.discount[torch.tensor(task)]
				if self.cfg.multitask
				else self.discount
			)
		return G + discount * self.model.Q(
			z, self.model.pi(z, task)[1], task, return_type="avg"
		)

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
				z (torch.Tensor): Latent state from which to plan.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: Action to take in the environment.
		"""
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(
				self.cfg.horizon,
				self.cfg.num_pi_trajs,
				self.cfg.action_dim,
				device=self.device,
			)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon - 1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std * torch.ones(
			self.cfg.horizon, self.cfg.action_dim, device=self.device
		)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(
			self.cfg.horizon,
			self.cfg.num_samples,
			self.cfg.action_dim,
			device=self.device,
		)
		if self.cfg.num_pi_trajs > 0:
			actions[:, : self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):
			# Sample actions
			actions[:, self.cfg.num_pi_trajs :] = (
				mean.unsqueeze(1)
				+ std.unsqueeze(1)
				* torch.randn(
					self.cfg.horizon,
					self.cfg.num_samples - self.cfg.num_pi_trajs,
					self.cfg.action_dim,
					device=std.device,
				)
			).clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task, self.cfg.horizon).nan_to_num_(0)
			elite_idxs = torch.topk(
				value.squeeze(1), self.cfg.num_elites, dim=0
			).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score /= score.sum(0)
			score = score.squeeze()
			score_sum = score.sum() + 1e-9

			mean = torch.einsum('e,hea->ha', score, elite_actions) / score_sum

			diff = elite_actions - mean.unsqueeze(1)  # (H, E, A)
			variance = torch.einsum('e,hea->ha', score, diff ** 2) / score_sum
			std = torch.sqrt(variance).clamp_(self.cfg.min_std, self.cfg.max_std)

		# Select action
		score_tensor = score.squeeze() # torch.Size([64])
		index = torch.multinomial(score_tensor, 1).item()
		actions = elite_actions[:, index] # torch.Size([3, 64, 38]) -> torch.Size([3, 38])
		self._prev_mean = mean
		mu, std = actions[0], std[0]
		if not eval_mode:
			a = mu + std * torch.randn(self.cfg.action_dim, device=std.device)
		else:
			a = mu
		return a.clamp_(-1, 1), mu, std
	
	def update_pi(self, zs, action, mu, std, task, step):
		"""
		Update policy using a sequence of latent states.

		Args:
				zs (torch.Tensor): Sequence of latent states.
				action (torch.Tensor): Sequence of actions.
				task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		_, pis, log_pis, log_std = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type="min")
		self.scale.update(qs[0])
		qs = self.scale(qs)
			
		############### Compute max Q loss ###############
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()

		############### Compute min KL loss ###############
		action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
		std = log_std.exp().detach()
		std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
		eps = (pis - mu) / std
		forward_kl = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
		forward_kl = self.scale(forward_kl) if self.scale.value > 2.0 else torch.zeros_like(forward_kl)
		forward_kl = torch.softmax(qs.detach().squeeze(),dim=-1) * forward_kl
		fkl_loss = - (forward_kl.sum(dim=-1) * rho).mean()
		
		############### Combine losses and update ###############
		if self.cfg.action_dim >= 24 and self.cfg.action_dim <= 48:
			self.lamda = self.cfg.action_dim * 1/1000
		else:
			self.lamda = self.cfg.action_dim * 1/100
		pi_loss = q_loss + self.lamda * fkl_loss

		params = list(self.model._pi.parameters())
		if fkl_loss.requires_grad:
			grads1 = torch.autograd.grad(2 * q_loss, params, retain_graph=True, allow_unused=True)
			grads2 = torch.autograd.grad(2 * fkl_loss, params, retain_graph=True, allow_unused=True)
			g1_vec = torch.cat([g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1) 
								for g, p in zip(grads1, params)])
			g2_vec = torch.cat([g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1) 
								for g, p in zip(grads2, params)])

			g_com_vec, weight_1, weight_2 = _bpograd(g1_vec, self.lamda * g2_vec)
			self.pi_optim.zero_grad(set_to_none=True)
			pointer = 0
			for p in params:
				num_param = p.numel()
				p.grad = g_com_vec[pointer:pointer + num_param].view_as(p).clone()
				pointer += num_param
		else:
			self.pi_optim.zero_grad(set_to_none=True)
			pi_loss.backward()

		torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item(), q_loss.item(), fkl_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
				next_z (torch.Tensor): Latent state at the following time step.
				reward (torch.Tensor): Reward at the current time step.
				task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = (
			self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		)
		return reward + discount * self.model.Q(
			next_z, pi, task, return_type="min", target=True
		)

	def update(self, replay_sample, step):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
				buffer (common.buffer.Buffer): Replay buffer.

		Returns:
				dict: Dictionary of training statistics.
		"""
		obs, action, mu, std, reward, task = replay_sample # mu and std are from Gaussian policy used for data collection	
		
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)
			
		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(
			self.cfg.horizon + 1,
			self.cfg.batch_size,
			self.cfg.latent_dim,
			device=self.device,
		)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
			zs[t + 1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type="all")
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += (
				math.soft_ce(reward_preds[t], reward[t], self.cfg).mean()
				* self.cfg.rho**t
			)
			for q in range(self.cfg.num_q):
				value_loss += (
					math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean()
					* self.cfg.rho**t
				)
		consistency_loss *= 1 / self.cfg.horizon
		reward_loss *= 1 / self.cfg.horizon
		value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)

		total_loss = (
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.value_coef * value_loss
		)
			
		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(
			self.model.parameters(), self.cfg.grad_clip_norm
		)
		self.optim.step()

		# Update policy
		pi_loss, pi_q_loss, pi_fkl_loss  = self.update_pi(_zs.detach(), action.detach(), mu.detach(), std.detach(), task, step)
		
		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"pi_q_loss": pi_q_loss,
			"pi_fkl_loss": pi_fkl_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value)
		}
