import os
import sys
import gymnasium as gym
cwd = os.path.dirname(__file__)
sys.path.append(f'{cwd}/controlgym')
import controlgym
import torch
import numpy as np

class ControlGymWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.n_observation = env.unwrapped.n_observation
		self.n_action = env.unwrapped.n_action
		self.n_state = env.unwrapped.n_state
		self.n_steps = env.unwrapped.n_steps
		self.id = env.unwrapped.id


		_ , self.task = self.cfg.task.split('-', maxsplit=1)
		self.default_dt = self.env.unwrapped.sample_time
		self.MAX_EPISODE_TIME = 10 # 10 seconds
		self.max_episode_steps = int(self.MAX_EPISODE_TIME / self.env.unwrapped.sample_time)
		self.max_episode_steps = min(max(self.max_episode_steps, 100), 500) # guarantee at least 100 <= x <= 500 steps/ep for a given dt
		self._step_count = 0
		
	def reset(self, **kwargs):
		obs, _ = self.env.reset(**kwargs)
		obs = obs.astype(np.float32)
		self._step_count = 0
		return obs

	def step(self, action):
		obs, reward, truncated, done, info = self.env.step(action)
		self._step_count += 1

		# NOTE: normalize reward due to its extreme range (e.g: -10^5)
		# NOTE: since reward is defined as - sum(LQ error), it's guaranteed to be <= 0
		if self.task in ['burgers']:
			pass
		elif self.task in ['wave']:
			reward = -1.0 * np.sqrt(-reward - 1e-8)
		else:
			pass
			# raise NotImplementedError
		
		# normalize reward by 1/T
		reward *= 1/self.max_episode_steps
		reward = np.array(reward).astype(np.float32)

		# termination conditions
		done = (done or self._step_count >= self.max_episode_steps)

		if "success" not in info:
			info["success"] = -1

		return obs, reward, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped 
	
	def render(self, *args, **kwargs):
		return self.env.render().copy()
	
	# get env's simulation Δt (for simulating cases where Δt < Δt_{default})
	def get_sim_dt(self):
		# simulation timestep is defined differently between different ControlGym envs
		return self.env.unwrapped.sample_time
	
	# set env's simulation Δt (for simulating cases where Δt < Δt_{default})
	def set_sim_dt(self, dt):
		# simulation timestep is defined differently between different ControlGym envs
		if (dt is not None):
			try:
				self.env.unwrapped.sample_time = dt
				self.max_episode_steps = int(self.MAX_EPISODE_TIME / self.env.unwrapped.sample_time)
				self.max_episode_steps = min(max(self.max_episode_steps, 100), 500) # guarantee at least 100 steps/ep for a given dt
			except:
				raise NotImplementedError(f'Timestep variations not yet implemented for {self.task}')

	""" method for step env with non-default timestep
		for ControlGym, simply: 
			set_sim_dt(dt) -> step(action)
	"""
	def step_adaptive_dt(self, action, dt):
		""" Simulation stepping: 
				(1) set_sim_dt(dt)
				(2) step(action)
		"""
		# 1. Set simulation dt
		self.set_sim_dt(dt)
		# 2. Execute action
		if isinstance(action, torch.Tensor):
			action = action.detach().cpu().numpy()
		obs, reward, done, info = self.step(action)
		# 3. return the last simulation step's [obs, reward, done, info]
		return torch.Tensor(obs), torch.tensor(reward), done, info
	
def make_env(cfg):
	"""
	task format: 'pde-{task}'
	e.g:
		pde-wave
		pde-burgers
	"""
	_ , task = cfg.task.split('-', maxsplit=1)
	env = controlgym.make(task)    
	env = ControlGymWrapper(env, cfg) # adaptive time-stepping

	# set integration ∆t for some envs: avoid overflow
	if task in ['burgers']:
		env.unwrapped.integration_time = 0.0001
	return env