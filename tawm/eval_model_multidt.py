import numpy as np
import mujoco

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian as torch_jacobian
import pandas as pd
import hydra
from tdmpc2 import TDMPC2
from common.parser import parse_cfg
from common.layers import dec
from common import math
from common.seed import set_seed
from common import TASK_SET
from envs import make_env

from termcolor import colored
from datetime import datetime
from tqdm import tqdm
import time
from collections import defaultdict

cwd = os.path.dirname(__file__)
if not os.path.exists(cwd):
    raise Exception(f'{cwd} does not exist!')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
env = None

"""
    Test model performance on various inference-time observation rates
    
    Save eval result in:
    3 cases: SAVE TO
        (1) baseline (non-time-aware): trained on default dt (2.5ms)
            -> single-step   : `logs/{task}/eval_multidt_baseline_{seed}.csv`
            OR
            -> adjusted steps: `logs/{task}/eval_multidt_baseline_adjusted_{seed}.csv`
            
        (2) baseline (non-time-aware): trained on non-default dt
            -> single-step   : `logs/{task}/eval_multidt_baseline_{traindt}_{seed}.csv`
            OR
            -> adjusted steps: `logs/{task}/eval_multidt_baseline_{traindt}_adjusted_{seed}.csv`

        (3) time-aware model : 
            -> single-step   : `logs/{task}/eval_multidt_timeaware_{seed}.csv`
            OR
            -> adjusted steps: `logs/{task}/eval_multidt_baseline_{traindt}_adjusted_{seed}.csv`
"""

@hydra.main(config_name='config', config_path='.', version_base='1.3')
def eval(cfg: dict):
    print(colored('Task:', 'green', attrs=['bold']), cfg.task, 
          colored('Seed:', 'green', attrs=['bold']), cfg.seed)
    if cfg.multi_dt:
        print(colored('Time-aware evaluation', 'green', attrs=['bold']))
    else:
        print(colored('Adjusted Step:', 'green', attrs=['bold']), cfg.eval_steps_adjusted)
    print(colored('Planning Horizon:', 'green', attrs=['bold']), cfg.horizon)
    # env parameters
    global env
    env = make_env(cfg)
    cfg = parse_cfg(cfg)
    if cfg.task[:2] == 'mw':
        cfg.default_dt = env.env.env.sim.model.opt.timestep
    elif cfg.task[:3] == 'pde':
        cfg.default_dt = env.get_sim_dt()
    else:
        raise ValueError(f'{cfg.task} is undefined.')
    print(colored(f'Default dt = {cfg.default_dt}', 'green'))

    """ Set episode lengths (total step(action, dt) steps)"""
    if cfg.task in TASK_SET['highway']: 
        cfg.episode_length = 500 # done=True when reached env.config["duration"] anyway
    elif cfg.task[:7] == 'sustain':
        cfg.episode_length = env.max_timestep
    else: 
        cfg.episode_length = env.max_episode_steps

    """ initialize model architectures"""
    tdmpc2 = TDMPC2(cfg)
    tdmpc2.model = tdmpc2.model.to(device)
    """ load trained weights"""
    model_weights = cfg.checkpoint
    tdmpc2.load(model_weights)
    tdmpc2.model.eval()

    """ test performance on different Δt's """
    ep_rewards = {}
    ep_successes = {}
    # csv file to save evaluation result
    df_eval = pd.DataFrame(columns=['Timestep', 'Reward', 'Success'])
    rows = []
    # measure inference time
    inference_times = defaultdict(list)

    """ define eval Δt values """
    if cfg.task[:3] == 'pde':
        # ControlGym PDE envs
        eval_dts = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
    else: 
        # Meta-World
        eval_dts = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]

    """===========================================
            EVALUATION ON DIFFERENT Δt's
    =========================================="""
    for dt in eval_dts:
        dt = round(dt, 4)
        tdmpc2.cfg.eval_dt = dt
        
        """ 1. initialize simulation with new timestep"""
        print(colored('SIMULATION TIMESTEP:', 'green', attrs=['bold']), dt)

        """ 2. adjust _max_episode_steps to accomodate different timestepping """
        # adjust episode length to accomodate different timestep:
        # a long simulation timestep = multiple default steps
        n_steps = int(dt // cfg.default_dt)
        dt_0 = round(dt - n_steps * cfg.default_dt, 4)
        n = int(dt_0 > 0) + n_steps # num actual simulation steps per algorithmic step
        if cfg.task[:2] == 'mw':
            # specific to Meta-World envs
            env.env._max_episode_steps = cfg.episode_length * n # env -> TimeLimit() wrapper -> _max_episode_steps
            env.max_episode_steps = cfg.episode_length * n      # env -> max_episode_steps
        elif cfg.task[:3] == 'pde':
            # already adjusted terminating conditions in Env Wrapper
            pass
        else:
            raise NotImplementedError

        """ 3. collect episode rewards"""
        ep_rewards[dt] = []
        ep_successes[dt] = []
        for episode in tqdm(range(cfg.eval_episodes), desc='Eval episodes', position=0):
            obs, ep_reward, done, t = env.reset(), 0, False, -1

            while not done:
                t += 1
                # """ 1. take action """
                start = time.time()
                action = tdmpc2.act(obs, t0=t==0, eval_mode=True, timestep=dt)
                end = time.time()
                inference_times[dt].append(end-start)
                # """ 2. step to next physics state in simulation"""
                # obs, reward, done, info = env.step(action)
                obs, reward, done, info = env_step_adaptive_dt(action, dt, cfg)
                ep_reward += reward
                done = (t >= cfg.episode_length-1) or done
                # print(t, reward, done)

            # print(f'\tTotal episode {episode+1} reward:', ep_reward)
            ep_rewards[dt].append(float(ep_reward.detach().cpu().numpy()))
            ep_successes[dt].append(info['success'])
            
            # write to eval csv
            rows.append({'Timestep': dt, 
                        'Reward': float(ep_reward.detach().cpu().numpy()), 
                        'Success': info['success']})

        print(f'\tAvg reward on env dt={dt}:', np.array(ep_rewards[dt]).mean())
        if cfg.task[:2] == 'mw':
            print(f'\tAvg success rate on env dt={dt}:', np.array(ep_successes[dt]).mean())
        print(f'\tAvg inference time on dt={dt}:', np.array(inference_times[dt]).mean())
        print(f'\tQ1/Q3 inference time on dt={dt}:', 
              np.quantile(np.array(inference_times[dt]), 0.25).round(3), 
              np.quantile(np.array(inference_times[dt]), 0.75).round(3))
        
        print('='*40)

    """ Save evaluation results """
    if not cfg.multi_dt:
        # Non-time-aware model
        if cfg.train_dt is None:
            save_file = f'eval_multidt_baseline_adjusted_{cfg.seed}.csv' if (cfg.eval_steps_adjusted) else f'eval_multidt_baseline_{cfg.seed}.csv'
        else:
            save_file = f'eval_multidt_baseline_{cfg.train_dt}_adjusted_{cfg.seed}.csv' if (cfg.eval_steps_adjusted) else f'eval_multidt_baseline_{cfg.train_dt}_{cfg.seed}.csv'
    else:
        # Time-aware model
        if cfg.dt_sampler == 'log-uniform':
            if cfg.integrator == 'rk4': # default
                save_file = f'eval_multidt_timeaware_{cfg.seed}.csv'
            else:
                save_file = f'eval_multidt_timeaware_{cfg.integrator}_{cfg.seed}.csv'
        else:
            if cfg.integrator == 'rk4': # default
                save_file = f'eval_multidt_timeaware_{cfg.dt_sampler}_{cfg.seed}.csv'
            else:
                save_file = f'eval_multidt_timeaware_{cfg.dt_sampler}_{cfg.integrator}_{cfg.seed}.csv'
        
    # Save evals 
    df_eval = pd.DataFrame(rows)
    df_eval.to_csv(f'{cwd}/logs/{cfg.task}/{save_file}', index=False)


def set_env_timestep(dt, cfg):
    if (dt is not None): #  and (self.env.env.env.physics.model.opt.timestep != dt)
        """ (1) set simulation timestep """
        # Meta-World envs
        if cfg.task[:2] == 'mw': 
            env.obs_dt = dt
        # ControlGym PDE envs
        elif cfg.task[:3] == 'pde':
            env.set_sim_dt(dt)
        else:
            raise NotImplementedError

def get_env_timestep(cfg):
    # Meta-World envs
    if cfg.task[:2] == 'mw': 
        return env.obs_dt
    # PDE ControlGym
    elif cfg.task[:3] == 'pde':
        return env.get_sim_dt()
    else:
        raise NotImplementedError

""" method for step env with non-default timestep
    case 1: dt <= default_dt: 
        (1) set env.timestep to dt; 
        (2) then env.step()
    case 2: dt > default_dt : 
        (1) segment dt = dt0 + N*default_dt; 
        (2) env.step() with dt0; 
        (3) env.step() with dt for N times
"""
def env_step_adaptive_dt(action, dt, cfg):
    # Meta-World envs:
    if cfg.task[:2] == 'mw':
        """ Use defined adaptive timestepping wrapper """
        return env.step_adaptive_dt(action, dt)
    # SustainGym & PyGame & PDE ControlGym envs
    elif cfg.task[:3] == 'pde':
        """ Use defined adaptive timestepping
            Mechanism: since it is linear, simply: 
                set_sim_dt(dt) -> normal step(action)
        """
        return env.step_adaptive_dt(action, dt)

if __name__ == '__main__':
    start = datetime.now()
    eval()
    end = datetime.now()

    print('Execution time:', end-start)