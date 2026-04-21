import torch
import torch.nn.functional as F
import sys
import os
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boom.common import math
from boom.common.scale import RunningScale
from boom.common.world_model import WorldModel
from boom.common.debug import plot_mppi_debug

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

		if self.cfg.update_flow:
			# Flow policy optimizer
			self.flow_optim = torch.optim.Adam(
				self.model._flow_pi.parameters(),
				lr=self.cfg.lr,
				eps=1e-5
			)

		self.model.eval()
		self.scale_pi = RunningScale(cfg)
		if self.cfg.update_flow:
			self.scale_flow = RunningScale(cfg)
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

		# Apply torch.compile if enabled
		if getattr(cfg, 'compile', False):
			self.model._encoder = torch.compile(self.model._encoder)
			self.model._dynamics = torch.compile(self.model._dynamics)
			self.model._reward = torch.compile(self.model._reward)
			self.model._Qs = torch.compile(self.model._Qs)
			self.model._pi = torch.compile(self.model._pi)
			if self.cfg.update_flow:
				self.model._flow_pi = torch.compile(self.model._flow_pi)

		# Create debug directory
		self.mppi_debug_dir = None

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
		model_state_dict = state_dict["model"]

		# Handle checkpoints saved with torch.compile (which adds _orig_mod prefix)
		# by removing the prefix for compatibility with non-compiled models
		cleaned_state_dict = {}
		for key, value in model_state_dict.items():
			new_key = key.replace("._orig_mod.", ".")
			cleaned_state_dict[new_key] = value

		self.model.load_state_dict(cleaned_state_dict)

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, use_pi=False, use_flow=False, debug=False):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
				obs (torch.Tensor): Observation from the environment.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (int): Task index (only used for multi-task experiments).
				debug (bool): If True, return MPPI trajectory debug info.

		Returns:
				action: Action to take in the environment.
				mu: Mean action.
				std: Std of action distribution.
				debug_info (dict, optional): MPPI trajectory debug info (only if debug=True).
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		z = self.model.encode(obs, task)
		if self.cfg.mpc and not use_pi and not use_flow:
			if debug:
				a, mu, std, info = self.plan(z, t0=t0, eval_mode=eval_mode, task=task, debug=debug)
				return a.cpu(), mu.cpu(), std.cpu(), info
			a, mu, std = self.plan(z, t0=t0, eval_mode=eval_mode, task=task, debug=debug)
			return a.cpu(), mu.cpu(), std.cpu()
		elif use_pi:
			mu, pi, log_pi, log_std = self.model.pi(z, task)
			if eval_mode:
				a = mu[0]
			else:
				a = pi[0]
			mu, std = mu[0], log_std.exp()[0]
			return a.cpu(), mu.cpu(), std.cpu()

		else:
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
	def plan(self, z, t0=False, eval_mode=False, task=None, debug=False):
		"""
		Plan a sequence of actions using the learned world model with multimodal MPPI refinement.

		This method:
		1. Generates trajectory candidates from both pi and flow policies
		2. Evaluates all candidates and selects top K* trajectories with lowest cost
		3. Refines each selected trajectory independently using MPPI (parallel execution)
		4. Selects the lowest-cost refined trajectory for execution

		By refining promising modes separately, this preserves multimodal structure
		and avoids collapsing distinct modes into a suboptimal intermediate solution.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (torch.Tensor): Task index for multi-task experiments.
			debug (bool): If True, save and return MPPI trajectory debug info.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Configuration
		num_pi = self.cfg["num_pi_trajs"]
		num_flow = self.cfg["num_flow_trajs"] if self.cfg["update_flow"] else 0
		num_proposals = num_pi + num_flow
		num_elite_trajs = getattr(self.cfg, "num_elite_trajs", 5)  # K*: number of top trajectories to refine
		mppi_iterations = self.cfg.iterations  # MPPI refinement iterations per elite trajectory

		# Initialize debug info storage
		debug_info = {}
		if debug:
			debug_info = {
				'pi_candidates': [],
				'flow_candidates': [],
				'random_candidates': [],
				'elite_indices': [],
				'elite_types': [],
				'refinements': [],
			}
		if num_proposals == 0:
			raise ValueError("num_pi_trajs + num_flow_trajs must be > 0 for multimodal MPPI refinement")

		# ==================================================
		# Step 1: Generate trajectory candidates from pi and flow
		# ==================================================
		all_proposal_actions = []
		proposal_types = []  # Track which policy generated each trajectory

		# 1a) Generate trajectories from pi
		pi_actions = None
		if num_pi > 0:
			pi_actions = torch.empty(
				self.cfg.horizon,
				num_pi,
				self.cfg.action_dim,
				device=self.device,
			)
			_z = z.repeat(num_pi, 1)
			for t in range(self.cfg.horizon - 1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]
			all_proposal_actions.append(pi_actions)
			proposal_types.extend(['pi'] * num_pi)

		# 1b) Generate trajectories from flow
		flow_actions = None
		if num_flow > 0:
			flow_actions = torch.empty(
				self.cfg.horizon,
				num_flow,
				self.cfg.action_dim,
				device=self.device,
			)
			_z = z.repeat(num_flow, 1)
			for t in range(self.cfg.horizon - 1):
				flow_actions[t] = self.model.flow_policy(_z)
				_z = self.model.next(_z, flow_actions[t], task)
			flow_actions[-1] = self.model.flow_policy(_z)
			all_proposal_actions.append(flow_actions)
			proposal_types.extend(['flow'] * num_flow)

		# 1c) Generate random trajectories to fill up to num_samples
		num_random = max(0, self.cfg.num_samples - num_proposals)
		random_actions = None
		if num_random > 0:
			random_actions = torch.rand(
				self.cfg.horizon,
				num_random,
				self.cfg.action_dim,
				device=self.device,
			) * 2 - 1  # Uniform in [-1, 1]
			all_proposal_actions.append(random_actions)
			proposal_types.extend(['random'] * num_random)

		# Concatenate all proposals: [H, num_total, A]
		all_proposal_actions = torch.cat(all_proposal_actions, dim=1)

		num_total_samples = num_pi + num_flow + num_random

		# ==================================================
		# Step 2: Evaluate all candidates and select top K*
		# ==================================================
		# Evaluate all proposal candidates
		z_expanded = z.repeat(num_total_samples, 1)
		proposal_values = self._estimate_value(z_expanded, all_proposal_actions, task, self.cfg.horizon).nan_to_num_(0)

		# Select top K* trajectories with lowest cost (highest value)
		k_star = min(num_elite_trajs, num_total_samples)
		elite_indices = torch.topk(proposal_values.squeeze(1), k_star, dim=0).indices
		elite_proposal_actions = all_proposal_actions[:, elite_indices]  # [H, K*, A]
		elite_proposal_values = proposal_values[elite_indices]

		if debug:
			debug_info['pi_candidates'] = {
				'actions': pi_actions.cpu().clone() if pi_actions is not None else None,
				'values': proposal_values.squeeze(1)[:num_pi].cpu().clone() if num_pi > 0 else None,
			}
			debug_info['flow_candidates'] = {
				'actions': flow_actions.cpu().clone() if flow_actions is not None else None,
				'values': proposal_values.squeeze(1)[num_pi:num_pi+num_flow].cpu().clone() if num_flow > 0 else None,
			}
			debug_info['random_candidates'] = {
				'actions': random_actions.cpu().clone() if random_actions is not None else None,
				'values': proposal_values.squeeze(1)[num_pi+num_flow:num_total_samples].cpu().clone() if num_random > 0 else None,
			}
			debug_info['elite_indices'] = elite_indices.cpu().clone()
			debug_info['elite_types'] = [proposal_types[i.item()] for i in elite_indices]

		# ==================================================
		# Step 3: Independent MPPI refinement for each elite trajectory (parallel)
		# ==================================================
		refined_trajectories = []
		refined_values = []

		for k in range(k_star):
			# Initialize MPPI with this elite trajectory as reference
			ref_traj = elite_proposal_actions[:, k]  # [H, A]

			# Initialize mean and std for refinement
			mean = ref_traj.clone()  # Start from the elite trajectory
			std = self.cfg.max_std * torch.ones(
				self.cfg.horizon, self.cfg.action_dim, device=self.device
			)

			# Prepare sampling around this reference trajectory
			z_samples = z.repeat(self.cfg.num_samples, 1)

			for iter_idx in range(mppi_iterations):
				# Sample perturbations around the reference trajectory
				actions = (
					mean.unsqueeze(1)
					+ std.unsqueeze(1)
					* torch.randn(
						self.cfg.horizon,
						self.cfg.num_samples,
						self.cfg.action_dim,
						device=std.device,
					)
				)
				# Apply periodic wrapping for angular actions
				actions = ((actions + 1) % 2) - 1

				if self.cfg.multitask:
					actions = actions * self.model._action_masks[task]

				# Compute values and select elite
				value = self._estimate_value(z_samples, actions, task, self.cfg.horizon).nan_to_num_(0)
				elite_idxs = torch.topk(
					value.squeeze(1), self.cfg.num_elites, dim=0
				).indices
				elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

				# Update mean and std
				max_value = elite_value.max(0)[0]
				score = torch.exp(self.cfg.temperature * (elite_value - max_value))
				score /= score.sum(0)
				score = score.squeeze()
				score_sum = score.sum() + 1e-9

				mean = torch.einsum('e,hea->ha', score, elite_actions) / score_sum

				diff = elite_actions - mean.unsqueeze(1)
				variance = torch.einsum('e,hea->ha', score, diff ** 2) / score_sum
				std = torch.sqrt(variance).clamp_(self.cfg.min_std, self.cfg.max_std)

				# Save debug info for this refinement
				if debug:
					if len(debug_info['refinements']) <= k:
						debug_info['refinements'].append([])

					iter_debug = {
						'iteration': iter_idx,
						'mean': mean.cpu().clone(),
						'std': std.cpu().clone(),
						'elite_values': elite_value.squeeze(1).cpu().clone(),
					}

					# Save initial iteration samples for visualization
					if iter_idx == 0:
						iter_debug['init_actions'] = actions.cpu().clone()  # [H, num_samples, A]
						iter_debug['init_values'] = value.squeeze(1).cpu().clone()  # [num_samples]

					debug_info['refinements'][k].append(iter_debug)

			# Select best action from final iteration (sample from elite set)
			final_score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			final_score /= final_score.sum(0)
			final_score = final_score.squeeze()

			best_idx = torch.multinomial(final_score, 1).item()
			best_trajectory = elite_actions[:, best_idx]  # [H, A]
			best_value = elite_value[best_idx]

			refined_trajectories.append(best_trajectory)
			refined_values.append(best_value)

		# Stack all refined trajectories
		refined_trajectories = torch.stack(refined_trajectories, dim=0)  # [K*, H, A]
		refined_values = torch.stack(refined_values, dim=0)  # [K*, 1]

		# ==================================================
		# Step 4: Select lowest-cost refined trajectory for execution
		# ==================================================
		best_overall_idx = torch.argmax(refined_values.squeeze(1))
		selected_trajectory = refined_trajectories[best_overall_idx]  # [H, A]

		if debug:
			debug_info['refined_trajectories'] = refined_trajectories.cpu().clone()
			debug_info['refined_values'] = refined_values.squeeze(1).cpu().clone()
			debug_info['selected_idx'] = best_overall_idx.item()
			debug_info['selected_trajectory'] = selected_trajectory.cpu().clone()

		

		# Extract first action
		mu = selected_trajectory[0]

		# Estimate std from elite trajectories for exploration
		if not t0:
			std = self._prev_std if hasattr(self, '_prev_std') else self.cfg.max_std * torch.ones(self.cfg.action_dim, device=self.device)
		else:
			std = self.cfg.max_std * torch.ones(self.cfg.action_dim, device=self.device)

		# Save for next iteration
		self._prev_mean = selected_trajectory[1:]  # Shift for next step
		self._prev_mean = torch.cat([
			self._prev_mean,
			torch.zeros(1, self.cfg.action_dim, device=self.device)
		], dim=0)
		self._prev_std = std

		if not eval_mode:
			a = mu + std * torch.randn(self.cfg.action_dim, device=std.device)
		else:
			a = mu

		# Periodic wrapping: map action from [-1, 1] with wraparound
		# This handles the circular nature of angular actions where -1 and 1 represent the same angle
		wrapped_a = ((a + 1) % 2) - 1
		if debug:
			return wrapped_a, mu, std, debug_info
		return wrapped_a, mu, std

	def update_flow(self, z, action, task):
		"""
		Compute flow matching loss with executed action as supervision target.

		Args:
			z: [N, latent_dim]
			action: [N, action_dim] - Executed action
			task: task info for Q computation

		Returns:
			flow_bc_loss: scalar
			info: dict
		"""
		N = z.shape[0]
		device = z.device
		dtype = z.dtype

		# Use executed action as supervision target
		assert action is not None, "action mode requires action parameter"
		x1 = action

		# Sample x0 from standard normal
		x0 = torch.randn_like(x1)

		# Sample random time
		t = torch.rand(N, device=device, dtype=dtype) * (1.0 - 1e-3)   # [N]
		t_exp = t[:, None]                                               # [N, 1]

		# straight path: x_t = (1 - t)x0 + t x1
		xt = (1.0 - t_exp) * x0 + t_exp * x1

		target_vel = (x1 - x0)

		flow_input = torch.cat([z, xt, t_exp], dim=-1)
		pred_vel = self.model._flow_pi(flow_input)

		# Compute Q-value weights for loss weighting using advantage-based method
		with torch.no_grad():
			# Compute Q-values using target action x1 (executed action)
			Q_values = self.model.Q(z, x1, task, return_type="avg")

			# Sample actions from current FLOW policy for advantage computation
			flow_actions = self.model.flow_policy(z)  # [N, action_dim]

			# Compute minimum Q-value of current flow policy actions
			min_q_flow = self.model.Q(z, flow_actions, task, return_type="min")  # [N, 1]

			# Compute advantage: how much better is executed action than flow policy?
			advantage = Q_values - min_q_flow  # [N, 1]

			# Apply ReLU to zero out negative advantages
			weights = torch.relu(advantage).detach()  # [N, 1]

			# Apply exponential transformation with mean centering
			weights = torch.exp(weights - weights.mean())  # [N, 1]

			# Apply clamp (use tighter bounds like ref-flow)
			weights = torch.clamp(weights, min=1e-3, max=1.0)

		# Compute weighted MSE loss
		flow_bc_loss = ((pred_vel - target_vel) ** 2 * weights.unsqueeze(-1)).mean()

		info = {
			"flow_bc_loss": flow_bc_loss.detach(),
			"flow_pred_abs": pred_vel.abs().mean().detach(),
			"flow_target_abs": target_vel.abs().mean().detach(),
			"flow_xt_abs": xt.abs().mean().detach(),
			"flow_weight_mean": weights.mean().detach(),
			"flow_weight_max": weights.max().detach(),
			"flow_weight_min": weights.min().detach(),
			"flow_weight_zero_pct": (weights < 1e-3).float().mean().detach() * 100,
			"flow_advantage_mean": advantage.mean().detach(),
			"flow_advantage_std": advantage.std().detach(),
			"flow_q_flow_mean": min_q_flow.mean().detach(),
			"flow_q_values_mean": Q_values.mean().detach(),
		}
		return flow_bc_loss, info

	def update_pi(self, zs, action, mu, std, task, step):
		"""
		Update Gaussian policy and mean-flow policy.

		Args:
			zs: [H, B, latent_dim]
			action: [H, B, action_dim]
			mu: [H, B, action_dim]
			std: [H, B, action_dim]
			task: task info
			step: training step

		Returns:
			info: dict
		"""
		H, B, _ = zs.shape

		# =========================
		# 1) Update Gaussian Policy
		# =========================
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		_, pis, log_pis, log_std = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type="min")
		self.scale_pi.update(qs[0])
		qs = self.scale_pi(qs)

		############### Compute max Q loss ###############
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()

		############### Compute min KL loss ###############
		action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
		std = log_std.exp().detach()
		std = torch.clamp(std, min=self.cfg.min_std)

		eps = (pis - mu) / std
		forward_kl = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
		forward_kl = self.scale_pi(forward_kl) if self.scale_pi.value > 2.0 else torch.zeros_like(forward_kl)
		forward_kl = torch.softmax(qs.detach().squeeze(),dim=-1) * forward_kl
		fkl_loss = - (forward_kl.sum(dim=-1) * rho).mean()

		############### Combine losses and update ###############
		pi_loss = q_loss + (self.cfg.action_dim / 1000) * fkl_loss
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.model._pi.parameters(), self.cfg.grad_clip_norm
		)
		self.pi_optim.step()

		if self.cfg.update_flow:
			# =========================
			# 2) Update Flow Policy with Hybrid Loss
			# =========================
			self.flow_optim.zero_grad(set_to_none=True)

			# flatten [H, B, ...] -> [H*B, ...]
			z_flat = zs.reshape(H * B, -1).detach()

			# Prepare data: use executed action
			action_flat = action.reshape(H * B, -1).detach()

			# ========================================
			# Part 1: Flow Matching Loss (监督项)
			# ========================================
			flow_bc_loss, flow_info = self.update_flow(
				z_flat,
				action_flat,
				task
			)

			# ========================================
			# Part 2: Max Q Actor Loss (强化学习项)
			# ========================================
			# Freeze critic, only update flow policy
			self.model.track_q_grad(False)

			# Generate actions from flow policy (fully differentiable)
			a_flow = self.model.flow_policy(z_flat)  # [H*B, action_dim]

			# Compute Q-values using shared critic
			flow_q = self.model.Q(z_flat, a_flow, task, return_type="min")  # [H*B, 1]

			# Actor loss: maximize Q-values
			flow_q_loss = -flow_q.mean()


			# ========================================
			# Part 3: Combine Losses
			# ========================================
			flow_q_coef = getattr(self.cfg, "flow_q_coef", 1.0)
			total_flow_loss = flow_bc_loss + flow_q_coef * flow_q_loss

			# Backward and update
			total_flow_loss.backward()
			torch.nn.utils.clip_grad_norm_(
				self.model._flow_pi.parameters(),
				self.cfg.grad_clip_norm,
			)
			self.flow_optim.step()

		# Restore critic gradient tracking
		self.model.track_q_grad(True)


		info = {
			"pi_loss": float(pi_loss.item()),
			"pi_q_loss": float(q_loss.item()),
			"pi_fkl_loss": float(fkl_loss.item()),
			"pi_scale": float(self.scale_pi.value),
		}

		if self.cfg.update_flow:
			info.update({
				# "flow_scale": float(self.scale_flow.value),
				"flow_bc_loss": float(flow_bc_loss.item()),
				"flow_q_loss": float(flow_q_loss.item()),
				"flow_total_loss": float(total_flow_loss.item()),
				"flow_pred_abs": float(flow_info["flow_pred_abs"].item()),
				"flow_target_abs": float(flow_info["flow_target_abs"].item()),
				"flow_xt_abs": float(flow_info["flow_xt_abs"].item()),
				"flow_weight_mean": float(flow_info["flow_weight_mean"].item()),
				"flow_weight_max": float(flow_info["flow_weight_max"].item()),
				"flow_weight_min": float(flow_info["flow_weight_min"].item()),
				"flow_weight_zero_pct": float(flow_info["flow_weight_zero_pct"].item()),
				"flow_advantage_mean": float(flow_info["flow_advantage_mean"].item()),
				"flow_advantage_std": float(flow_info["flow_advantage_std"].item()),
				"flow_q_flow_mean": float(flow_info["flow_q_flow_mean"].item()),
				"flow_q_values_mean": float(flow_info["flow_q_values_mean"].item()),
			})
		return info

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
		Main update function. One iteration of model learning.
		"""
		obs, action, mu, std, reward, task = replay_sample
		# mu and std are from Gaussian policy used for data collection

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

		# Update policies
		pi_info = self.update_pi(
			_zs.detach(),
			action.detach(),
			mu.detach(),
			std.detach(),
			task,
			step,
		)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		stats = {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
		}
		stats.update(pi_info)
		return stats
