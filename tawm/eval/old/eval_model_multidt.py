import numpy as np
import mujoco

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the tawm package root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
from PIL import Image

# Resolve TAWM package root (two levels above this file: tawm/eval/old/ -> tawm)
tawm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if not os.path.exists(tawm_root):
    raise Exception(f'{tawm_root} does not exist!')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
env = None
save_video = True # set this only if you want to save evaluation visualizations

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

@hydra.main(config_name='config', config_path='../..', version_base='1.3')
def eval(cfg: dict):
    global save_video

    print(colored('Task:', 'green', attrs=['bold']), cfg.task, 
          colored('Seed:', 'green', attrs=['bold']), cfg.seed)
    if cfg.multi_dt:
        print(colored('Time-aware evaluation', 'green', attrs=['bold']))
    else:
        print(colored('Adjusted Step:', 'green', attrs=['bold']), cfg.eval_steps_adjusted)
    print(colored('Planning Horizon:', 'green', attrs=['bold']), cfg.horizon)

    if cfg.task[:3] == 'pde':
        save_video = False # PDE Control envs doesnt have rendering
    print(colored('save_video: ' + str(save_video), 'green'))
    
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
    elif cfg.task[:2] == 'mw': 
        # Meta-World
        eval_dts = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
    else:
        raise NotImplementedError

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
        frames = []
        for episode in tqdm(range(cfg.eval_episodes), desc='Eval episodes', position=0):
            obs, ep_reward, done, t = env.reset(), 0, False, -1

            while not done:
                t += 1

                # """ 0. video rendering"""
                if (episode == 0) and save_video:
                    frames.append(env.render())
                # """ 1. take action """
                start = time.time()
                action = tdmpc2.act(obs, t0=t==0, eval_mode=True, timestep=dt)
                end = time.time()
                inference_times[dt].append(end-start)
                # """ 2. step to next physics state in simulation"""
                # obs, reward, done, info = env.step(action)
                obs, reward, done, info = env.step_adaptive_dt(action, dt)
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
        
            # **********************************************************
            #   save video rendering: auto define video name from cfg
            # **********************************************************
            if (episode == 0) and save_video:
                save_video_func(frames, eval_dt=dt, cfg=cfg)

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
            save_file = f'eval_multidt_timeaware_{cfg.dt_sampler}_{cfg.integrator}_{cfg.seed}.csv'
        
    # Save evals 
    df_eval = pd.DataFrame(rows)
    logs_dir = os.path.join(tawm_root, 'logs', f'{cfg.task}')
    os.makedirs(logs_dir, exist_ok=True)
    df_eval.to_csv(os.path.join(logs_dir, f'{save_file}'), index=False)


def save_video_func(frames, eval_dt: float, cfg: dict):
    # Save demo video
    fps = 15 * (cfg.default_dt / eval_dt) # smaller dt => more frame => adjust higher fps
    # fps = int(fps)
    video_path = None
    task = cfg.task

    ########################
    ### save_video_path  ###
    ########################
    logs_task_dt_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'logs', task, f'eval_dt={eval_dt}')
    os.makedirs(logs_task_dt_dir, exist_ok=True)
    if cfg.multi_dt:
        # time-aware model
        video_path = os.path.join(logs_task_dt_dir, f'tawm_{cfg.dt_sampler}_{cfg.integrator}_{cfg.seed}.gif')
    else:
        # base model
        # train_dt: fixed dt base model is trained on (by default, train_dt = default_dt)
        train_dt = cfg.train_dt if (cfg.train_dt is not None) else cfg.default_dt
        cfg.train_dt = cfg.train_dt if (cfg.train_dt is not None) else cfg.default_dt
        video_path = os.path.join(logs_task_dt_dir, f'baseline_traindt={cfg.train_dt}_{cfg.seed}.gif')

    # save frames to mp4 video
    print(colored(f'Saving {len(frames)} frames to {video_path}.', 'green', attrs=['bold']))
    duration = 1000/fps
    frames = [Image.fromarray(img) for img in frames]
    frames[0].save(video_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    # # address issue with duration not correct
    # vid = Image.open(video_path)
    # vid.info['duration'] = duration
    # vid.save(video_path, save_all=True)


if __name__ == '__main__':
    start = datetime.now()
    eval()
    end = datetime.now()

    print('Execution time:', end-start)