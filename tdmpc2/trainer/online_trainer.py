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
		self.last_horizon = self.cfg.min_horizon

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
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

	# def get_horizon(self, steps, consistency_loss_history, increase_stop=150_000):
	# 	"""
	# 	Return the horizon for the current step, starting at self.cfg.min_horizon and ending at self.cfg.max_horizon
	# 	We will just do a linear interpolation between the two values.
	# 	"""
	# 	new_horizon = int(self.cfg.min_horizon + (self.cfg.max_horizon - self.cfg.min_horizon) * steps / increase_stop)
	# 	if steps >= increase_stop:
	# 		if steps % 5000 == 0:
	# 			# assume consistency_loss_history is full and has 10000 elements
	# 			first_half_mean = np.mean(list(consistency_loss_history)[:5000])
	# 			second_half_mean = np.mean(list(consistency_loss_history)[5000:])
	# 			if second_half_mean < first_half_mean:
	# 				new_horizon = self.last_horizon - 2
	# 			else:
	# 				new_horizon = self.last_horizon + 1
	# 		else:
	# 			new_horizon = self.last_horizon
	# 	new_horizon = max(self.cfg.min_horizon, min(self.cfg.max_horizon, new_horizon))
	# 	self.last_horizon = new_horizon
	# 	return new_horizon

	def get_horizon(self, steps, consistency_loss_history, reward_loss_history, value_loss_history, total_loss_history, increase_stop=150_000): #total loss version
		"""
		Return the horizon for the current step, starting at self.cfg.min_horizon and ending at self.cfg.max_horizon
		We will just do a linear interpolation between the two values.
		"""
		new_horizon = self.cfg.last_horizon #int(self.cfg.min_horizon + (self.cfg.max_horizon - self.cfg.min_horizon) * steps / increase_stop)
		#if steps >= increase_stop:
		if steps >= 10000 and steps % 5000 == 0:
			# assume consistency_loss_history is full and has 10000 elements
			if self.cfg.horizon_mode == "inverse-total-loss":
				first_half_mean = np.mean(list(total_loss_history)[:5000])
				second_half_mean = np.mean(list(total_loss_history)[5000:])
				if second_half_mean < first_half_mean:
					new_horizon = self.last_horizon + 1
				else:
					new_horizon = self.last_horizon - 1
			elif self.cfg.horizon_mode == "inverse-reward-loss":
				first_half_mean = np.mean(list(reward_loss_history)[:5000])
				second_half_mean = np.mean(list(reward_loss_history)[5000:])
				if second_half_mean < first_half_mean:
					new_horizon = self.last_horizon + 1
				else:
					new_horizon = self.last_horizon - 1
			else:
				raise Exception("Not a defined mode!")
			# else:
			# 	new_horizon = self.last_horizon
		new_horizon = max(self.cfg.min_horizon, min(self.cfg.max_horizon, new_horizon))
		self.last_horizon = new_horizon
		return new_horizon

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True
		# keep track of 5000 steps moving average of consistency loss
		# to do that need 10000 steps of buffer. use deque
		consistency_loss_history = deque(maxlen=10000)
		reward_loss_history = deque(maxlen=10000)
		value_loss_history = deque(maxlen=10000)
		total_loss_history = deque(maxlen=10000)
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
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

			horizon = self.get_horizon(self._step, consistency_loss_history, reward_loss_history, value_loss_history, total_loss_history)
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
				consistency_loss_history.append(_train_metrics['consistency_loss'])
				reward_loss_history.append(_train_metrics['reward_loss'])
				value_loss_history.append(_train_metrics['value_loss'])
				total_loss_history.append(_train_metrics['total_loss'])

			self._step += 1
	
		self.logger.finish(self.agent)
