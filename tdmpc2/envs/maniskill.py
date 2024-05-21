import gymnasium as gym
import numpy as np
from envs.wrappers.time_limit import TimeLimit

import mani_skill.envs


MANISKILL_TASKS = {
	'push-cube': dict(
		env='PushCube-v1',
		control_mode='pd_ee_delta_pos',
	),
	'pick-cube': dict(
		env='PickCube-v1',
		control_mode='pd_ee_delta_pos',
	),
	'stack-cube': dict(
		env='StackCube-v1',
		control_mode='pd_ee_delta_pos',
	),
	'pick-ycb': dict(
		env='PickSingleYCB-v1',
		control_mode='pd_ee_delta_pose',
	),
	'turn-faucet': dict(
		env='TurnFaucet-v1',
		control_mode='pd_ee_delta_pose',
	),
}


class ManiSkillWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		# currently this is a 1x41 state vector, want to unsqueeze it to 41
		# Box(-inf, inf, (1, 42), float32)
		self.observation_space = gym.spaces.Box(
			self.observation_space.low[0],
			self.observation_space.high[0],
			shape=self.observation_space.shape[1:],
			dtype=self.observation_space.dtype,
		)
		self.action_space = gym.spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=self.env.action_space.dtype,
		)

	def reset(self):
		return self.env.reset()[0][0]
	
	def step(self, action):
		reward = 0
		for _ in range(2):
			obs, r, _, _, info = self.env.step(action)
			reward += r[0]
		for key in info:
			info[key] = info[key][0]
		return obs[0], reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, **kwargs):
		return self.env.render()[0]


def make_env(cfg):
	"""
	Make ManiSkill2 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	task_cfg = MANISKILL_TASKS[cfg.task]
	env = gym.make(
    task_cfg['env'], # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="rgb_array",
	)
	env = ManiSkillWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=100)
	env.max_episode_steps = env._max_episode_steps
	return env
