import gymnasium as gym
import numpy as np
from envs.wrappers.time_limit import TimeLimit

import mani_skill.envs


MANISKILL_TASKS = {
	'lift-cube': dict(
		env='LiftCube-v1',
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
		self.observation_space = self.env.observation_space
		self.action_space = gym.spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=self.env.action_space.dtype,
		)

	def reset(self):
		return self.env.reset()[0]
	
	def step(self, action):
		reward = 0
		for _ in range(2):
			obs, r, _, info = self.env.step(action)
			reward += r
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, args, **kwargs):
		return self.env.render(mode='cameras')


def make_env(cfg):
	"""
	Make ManiSkill2 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	task_cfg = MANISKILL_TASKS[cfg.task]
	# env = gym.make(
	# 	task_cfg['env'],
	# 	obs_mode='state',
	# 	control_mode=task_cfg['control_mode'],
	# 	# render_camera_cfgs=dict(width=384, height=384),
	# )
	env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
		# max_episode_steps=100
	)
	env = ManiSkillWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=100)
	env.max_episode_steps = env._max_episode_steps
	return env
