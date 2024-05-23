from time import time
from collections import deque

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self, horizon):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				action = self.agent.act(obs, t0=t==0, eval_mode=True, horizon=horizon)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1,))
		return td

	def get_horizon(self, step):
		"""
		start_horizon: ???
		mid_horizon: ???
		reach_mid_horizon: ???
		end_horizon: ???
		reach_end_horizon: ???
		"""
		# linearly interpolate between start_horizon and mid_horizon
		if step < self.cfg.reach_mid_horizon:
			horizon = self.cfg.start_horizon + (self.cfg.mid_horizon - self.cfg.start_horizon) * step / self.cfg.reach_mid_horizon
			if self.cfg.start_horizon > self.cfg.mid_horizon:
				horizon = np.ceil(horizon)
				
		# linearly interpolate between mid_horizon and end_horizon
		elif step < self.cfg.reach_end_horizon:
			horizon = self.cfg.mid_horizon + (self.cfg.end_horizon - self.cfg.mid_horizon) * (step - self.cfg.reach_mid_horizon) / (self.cfg.reach_end_horizon - self.cfg.reach_mid_horizon)
			if self.cfg.mid_horizon > self.cfg.end_horizon:
				horizon = np.ceil(horizon)
		else:
			horizon = self.cfg.end_horizon
		return int(horizon)

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					for h in [1,2,3,5,7]:
						eval_metrics = self.eval(h)
						# append the horizon to the eval_metrics keys
						eval_metrics = {f"{key}-{h}": value for key, value in eval_metrics.items()}
						eval_metrics.update(self.common_metrics())
						self.logger.log(eval_metrics, 'eval')
						eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			horizon = self.get_horizon(self._step)
			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1, horizon=horizon)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer, horizon)
				train_metrics.update(_train_metrics)

			self._step += 1
	
		self.logger.finish(self.agent)
