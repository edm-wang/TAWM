import os
import sys
import warnings
warnings.filterwarnings('ignore')
import time
from collections import defaultdict

# Ensure package root on path when running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import hydra
from termcolor import colored

from common.parser import parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2


@hydra.main(config_name='config', config_path='..', version_base='1.3')
def compute_overhead(cfg: dict):
	"""Compute inference compute overhead across Δt for mw-basketball.

	Outputs CSV with per-Δt:
	- Timestep
	- StepsPerEpisode (algorithm steps == inference calls)
	- AvgInferencePerStepSec
	- AvgEpisodeInferenceSec (sum of inference per episode)
	- SimulatedEpisodeTimeSec (cfg.episode_length * dt)
	"""
	# Task focus per user request
	cfg.task = 'mw-basketball'
	cfg.multitask = False
	print(colored(f'Task: {cfg.task}', 'green', attrs=['bold']))

	# Setup env and cfg
	env = make_env(cfg)
	cfg = parse_cfg(cfg)
	if cfg.task[:2] == 'mw':
		cfg.default_dt = env.env.env.sim.model.opt.timestep
	elif cfg.task[:3] == 'pde':
		cfg.default_dt = env.get_sim_dt()
	else:
		raise ValueError(f'{cfg.task} is undefined.')
	cfg.episode_length = env.max_episode_steps
	print(colored(f'Default dt = {cfg.default_dt}', 'green'))

	# Model
	agent = TDMPC2(cfg)
	agent.model = agent.model.to('cuda' if torch.cuda.is_available() else 'cpu')
	agent.model.eval()
	# Expect cfg.checkpoint to be a file path; user should override when running
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found!'
	agent.load(cfg.checkpoint)

	# Δt set for Meta-World
	eval_dts = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]

	rows = []
	for dt in eval_dts:
		dt = round(dt, 4)
		print(colored(f'Δt={dt}', 'cyan'))

		# Adjust environment episode length to accommodate dt
		n_steps = int(dt // cfg.default_dt)
		dt_0 = round(dt - n_steps * cfg.default_dt, 4)
		n = int(dt_0 > 0) + n_steps
		if cfg.task[:2] == 'mw':
			env.env._max_episode_steps = cfg.episode_length * n
			env.max_episode_steps = cfg.episode_length * n
		elif cfg.task[:3] == 'pde':
			pass
		else:
			raise NotImplementedError

		# Measure inference times per step across episodes
		per_episode_inference_time = []  # wall time spent in agent.act per episode
		steps_per_episode = []           # algorithmic steps (== number of agent.act calls)
		per_step_times = []              # aggregate per-step inference times across episodes

		for episode in range(cfg.eval_episodes):
			obs, done, t = env.reset(), False, -1
			ep_infer_sum = 0.0
			while not done:
				t += 1
				start = time.time()
				_ = agent.act(obs, t0=(t == 0), eval_mode=True, timestep=dt)
				end = time.time()
				per_step_times.append(end - start)
				ep_infer_sum += (end - start)
				# Step env with adaptive dt
				obs, reward, done, info = env.step_adaptive_dt(_, dt)
				done = (t >= cfg.episode_length - 1) or done
			# episode finished
			per_episode_inference_time.append(ep_infer_sum)
			steps_per_episode.append(t + 1)

		# Aggregate
		steps_ep = int(np.mean(steps_per_episode))
		avg_infer_per_step = float(np.mean(per_step_times)) if per_step_times else 0.0
		avg_ep_infer = float(np.mean(per_episode_inference_time)) if per_episode_inference_time else 0.0
		simulated_episode_time = cfg.episode_length * dt

		rows.append({
			'Timestep': dt,
			'StepsPerEpisode': steps_ep,
			'AvgInferencePerStepSec': round(avg_infer_per_step, 6),
			'AvgEpisodeInferenceSec': round(avg_ep_infer, 6),
			'SimulatedEpisodeTimeSec': round(simulated_episode_time, 6),
		})

	# Save CSV
	logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', cfg.task)
	logs_dir = os.path.abspath(logs_dir)
	os.makedirs(logs_dir, exist_ok=True)
	out_csv = os.path.join(logs_dir, 'compute_overhead.csv')
	pd.DataFrame(rows).to_csv(out_csv, index=False)
	print(colored(f'Saved compute overhead to {out_csv}', 'green', attrs=['bold']))


if __name__ == '__main__':
	compute_overhead()


