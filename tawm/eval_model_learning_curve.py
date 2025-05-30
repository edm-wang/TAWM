"""
    Evaluate intermediate models and save learning curves 
    on each task across different seeds
"""

import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')

import torch # type: ignore
import torch.nn.functional as F # type: ignore
from torch.autograd.functional import jacobian as torch_jacobian # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import hydra # type: ignore
from tdmpc2 import TDMPC2
from common.parser import parse_cfg
from common.layers import dec
from common import math
from common.seed import set_seed
from common import TASK_SET
from envs import make_env

from termcolor import colored # type: ignore
from datetime import datetime
from tqdm import tqdm # type: ignore
import time

""" =========================================================
    Create a struct to store:
        (1) current evaluated task
        (2) current evaluation time step
    
    -> to pass additional arguments to
       eval_learning_curve() due to 
       @hydra.main(config_name='config', config_path='.')
========================================================= """
class Evalution_Settings:
    model_checkpoints = '/fs/nexus-scratch/anhu/world-model-checkpoints'
    task = None
    eval_dt = None
    model_type = None # 'baseline' or 'timeaware'
    
eval_settings = Evalution_Settings()
cwd = os.path.dirname(__file__)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
env = None

""" ==================================================
    GENERATE REWARD CURVE FOR ALL TASKS FOR 
    SOME EVALUATION TIMESTEP SIZE eval_dt

    --------------------------------------------------------------------------
    assumptions:
        All intermediate model weights are in: 
        
        Case 1: `{model_dir}/{task}/singledt-{training_dt}/{seed}/step_{training_step}.pt`
            -> non-time-aware model trained on `training_dt` with `seed`
            
        Case 2: `{model_dir}/{task}/multidt-{sampler_type}/{seed}/step_{training_step}.pt`
            -> time-aware model trained with `seed`
            -> `sampler_type`: "uniform" or "log-uniform"

    --------------------------------------------------------------------------
    save_dir: `{cwd}/logs/{task}/return_curve_{model_type}_eval_dt={eval_dt}.csv` 
      where: 
        - model_type = 'baseline' or 'timeaware'
        - eval_dt    = evaluation time step size (simulation time step size)
    
=================================================== """
@hydra.main(config_name='config', config_path='.', version_base='1.3')
def eval_learning_curve(cfg):
    """ 0(a). Set log dir for current `task` & `eval_dt` 
              & other settings 
    """
    global env
    cfg.task = eval_settings.task        # set evaluation task
    cfg.eval_dt = eval_settings.eval_dt  # set evaluation time step
    cfg.multitask = False
    cfg.multi_dt = True if (eval_settings.model_type[:9] == 'timeaware') else False

    log_dir = f'/logs/{cfg.task}/return_curve_{eval_settings.model_type}_eval_dt={cfg.eval_dt}.csv'
    print('Evaluating reward curves for', colored(f'{cfg.task}', 'green'), 
          'at', colored(f'eval_dt={cfg.eval_dt}', 'green'), 
          'saving at', colored(log_dir, 'green'))
    log_dir = f'{cwd}/{log_dir}'
    
    """ 0(b). Initialize CSV result file """
    df_eval = pd.DataFrame(columns=['step', 'episode_reward', 'episode_success', 'seed'])
    rows = []

    
    """ 1. Initialize the environment & retrieve default dt """
    env = make_env(cfg)
    cfg = parse_cfg(cfg)
    if cfg.task[:2] == 'mw':
        cfg.default_dt = env.env.env.sim.model.opt.timestep
    elif cfg.task[:3] == 'pde':
        cfg.default_dt = env.default_dt
    else:
        cfg.default_dt = env.env.env.physics.model.opt.timestep
    # 4-decimal places rounding of dt
    cfg.default_dt = round(cfg.default_dt, 4)

    
    """ 2. Iterate through different model seeds """
    for seed in [1,2,3]:
        cfg.seed = seed
        # model_name = 'multidt' if cfg.multi_dt else f'singledt-{cfg.default_dt}'
        if eval_settings.model_type == 'timeaware':
            model_name = 'multidt-log-uniform-rk4' 
        elif eval_settings.model_type == 'timeaware-euler':
            model_name = 'multidt-log-uniform-euler' 
        elif eval_settings.model_type == 'baseline': 
            model_name = f'singledt-{cfg.default_dt}'
        else:
            raise Exception(f'no model {eval_settings.model_type}')
        
        model_checkpoints = f'{eval_settings.model_checkpoints}/{cfg.task}/{model_name}/{seed}'
        # print(os.path.exists(model_checkpoints))
        if not os.path.exists(model_checkpoints):
            print(colored(f'{model_type}\'s weights for {cfg.task}, seed {cfg.seed} not available!'))
            continue
        
        """ 2.1. Iterate intermediate-step models """
        step_weights = sorted(os.listdir(model_checkpoints), key=natural_sort_key)
        # print(step_weights)

        for weight in tqdm(step_weights, desc=f'Evaluating intermediate-step models: {cfg.task}, seed {cfg.seed}'):
            """ 2.3. initialize model and load intermediate-step weights for current {seed} """
            tdmpc2 = TDMPC2(cfg) 
            tdmpc2.model = tdmpc2.model.to(device)
            tdmpc2.load(f'{model_checkpoints}/{weight}') 
            tdmpc2.model.eval()

            """ 2.4. evaluate current intermediate-step model at current seed
                -> eval performance of {task}-{seed} model at current {seed} at {eval_dt} simulation step
            """
            step = re.split(r'(\d+)', weight)[1]
            row = eval_intermediate_step_model(tdmpc2, step, cfg)

            """ 2.5. add result to the CSV result file"""
            rows.append(row)

    """ 3. Save return curve for {task} evaluated on simulation time step {eval_dt} """
    df_eval = pd.DataFrame(rows)
    df_eval.to_csv(log_dir, index=False)

""" ==================================================
    Evaluate each intermediate-step model seed

    input:
        - agent: the trained world model
        - env  : the environment 
        - cfg  : config file for experiments

    output:
        - rows : a row of evaluation: (step, episode_reward, episode_success, seed), 
                 averaged over cfg.eval_episodes
================================================== """
def eval_intermediate_step_model(agent, step, cfg):
    """ 1. adjust _max_episode_steps to accomodate different timestepping """
    # adjust episode length to accomodate different timestep:
    # a long simulation timestep = multiple default steps
    dt = cfg.eval_dt
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

    # 2. initialize world models
    # 2a. update obs & action space in cfg to initialize model
    # if cfg.multi_dt:
    #     cfg.obs_shape = {cfg.get('obs', 'state'): [env.d_obs+1]}
    # else:
    #     cfg.obs_shape = {cfg.get('obs', 'state'): [env.d_obs]}

    """ 2. collect episode rewards"""
    ep_rewards = {}
    ep_successes = {}
    ep_rewards[dt] = []
    ep_successes[dt] = []
    inference_time = []
    for episode in range(cfg.eval_episodes):
        obs, ep_reward, done, t = env.reset(), 0, False, -1

        while not done:
            t += 1
            # """ 1. take action """
            start = time.time()
            action = agent.act(obs, t0=t==0, eval_mode=True, timestep=dt)
            end = time.time()
            inference_time.append(end-start)
            # """ 2. step to next physics state in simulation"""
            # obs, reward, done, info = env.step(action)
            # obs, reward, done, info = env.env.step_adaptive_dt(action, dt)
            obs, reward, done, info = env_step_adaptive_dt(action, dt, cfg)
            ep_reward += reward
            done = (t >= cfg.episode_length-1) or done
            # print(t, reward, done)

        # print(f'\tTotal episode {episode+1} reward:', ep_reward)
        ep_rewards[dt].append(float(ep_reward.detach().cpu().numpy()))
        ep_successes[dt].append(info['success'])
        
    # output a row of evaluation: (step, episode_reward, episode_success, seed) averaged over cfg.eval_episodes
    row = { 
            'step': step, 
            'episode_reward': np.array(ep_rewards[dt]).mean(), 
            'episode_success': np.array(ep_successes[dt]).mean(), 
            'avg_model_exec_time': np.array(inference_time).mean(),
            'seed': cfg.seed
        }
    
    # * POST-PROCESSING eval metrics for some envs
    if cfg.task == 'pygame-flappybird':
        # flappy bird: measure in # holes passed => offset terminating reward of -5
        row['episode_reward'] = row['episode_reward'] + 5
        
    return row

def set_env_timestep(dt, cfg):
    if (dt is not None): #  and (self.env.env.env.physics.model.opt.timestep != dt)
        """ (1) set simulation timestep """
        # Meta-World envs
        if cfg.task[:2] == 'mw': 
            env.env.env.sim.model.opt.timestep = dt
        # ControlGym PDE envs
        elif cfg.task[:3] == 'pde':
            env.set_sim_dt(dt)
        else:
            raise NotImplementedError

def get_env_timestep(cfg):
    # Meta-World envs
    if cfg.task[:2] == 'mw': 
        return env.env.env.sim.model.opt.timestep
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
    # Meta-world envs
    if cfg.task[:2] == 'mw':
        """ Use defined adaptive timestepping wrapper """
        return env.env.step_adaptive_dt(action, dt)
    # PDE ControlGym envs
    elif cfg.task[:3] == 'pde':
        """ Use defined adaptive timestepping
            Mechanism: since it is linear, simply: 
                set_sim_dt(dt) -> normal step(action)
        """
        return env.step_adaptive_dt(action, dt)
    else:
        raise NotImplementedError
    
    # #############################################################################
    # """ Simulation stepping: 
    #         (1) break large timestep -> smaller timesteps 
    #         (2) step multiple sub-steps -> multiple env.step()
    # """
    # # 1. segment the current step of dt into different simulation steps,
    # #    where each step has max time-stepping of default dt
    # n_steps = int(dt // cfg.default_dt)
    # # remainder timestep dt_0:  dt = dt_0 + N * default_dt
    # dt_0 = round(dt - n_steps * cfg.default_dt, 5)
    # # print('Timestep:', dt, '\t| n_steps:', n_steps, '\t| default_dt:', self.cfg.default_dt, '\t| dt_0', dt_0)

    # # 2. execute the first simulation with timestep = dt_0
    # if dt_0 > 0:
    #     set_env_timestep(dt_0, cfg)
    #     obs, reward, done, info = env.step(action)
    # # 3. execute the subsequent simulations with default timestep
    # set_env_timestep(cfg.default_dt, cfg)
    # for _ in range(n_steps):
    #     obs, reward, done, info = env.step(action)
    # # 4. return the last simulation step's [obs, reward, done, info]
    # return obs, reward, done, info

""" 
    Sort intermediate weights chronologically:
        step_0.pt -> step_50000.pt -> ... -> step_1000000.pt
"""
def natural_sort_key(s):
    # Extract numeric parts of the string for sorting
    return [int(num) if num.isdigit() else num for num in re.split(r'(\d+)', s)]

if __name__ == '__main__':
    """ List of tasks & evaluation timestep"""
    
    # tasks = ['mw-assembly', 'mw-basketball', 'mw-box-close', 
    #          'mw-faucet-open', 'mw-hammer', 'mw-handle-pull', 
    #          'mw-lever-pull', 'mw-pick-out-of-hole', 'mw-sweep-into']
    # tasks = ['mw-assembly', 'mw-basketball', 'mw-box-close']
    # tasks = ['mw-faucet-open', 'mw-hammer', 'mw-handle-pull']
    # tasks = ['mw-lever-pull', 'mw-pick-out-of-hole', 'mw-sweep-into']
    # tasks = ['mw-box-close']
    # tasks = ['mw-handle-pull']
    tasks = ['mw-sweep-into']
    # eval_dts = [0.001, 0.0025 0.01, 0.02, 0.03, 0.05]
    # eval_dts = [0.001, 0.0025 0.01, 0.02, 0.03, 0.05]
    eval_dts = [0.0025]

    # tasks = ["mw-basketball"]
    # eval_dts = [0.01, 0.02, 0.03, 0.05]

    # tasks = ["mw-box-close"]
    # eval_dts = [0.001, 0.01, 0.02, 0.03, 0.05]
    # tasks = ["mw-hammer"]
    # eval_dts = [0.03, 0.05]

    # tasks = ["mw-handle-pull"]
    # eval_dts = [0.001, 0.01, 0.02, 0.03, 0.05]

    # tasks = ['mw-pick-out-of-hole']
    # eval_dts = [0.05]

    # tasks = ['mw-sweep-into']
    # eval_dts = [0.001, 0.01, 0.02, 0.03, 0.05]

    # tasks = ['pygame-flappybird']
    # eval_dts = [0.01, 0.0333, 0.05, 0.1, 0.2]

    
    # tasks = ['pde-burgers', 'pde-wave']
    # # tasks = ['pde-allen_cahn', 'pde-wave']
    # # tasks = ['pde-wave']
    # eval_dts = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    

    """ Evaluate the learning curve for each task on each eval_dt """
    # tasks = tasks[:3]
    # tasks = tasks[3:5]
    # tasks = tasks[5:7]
    # tasks = tasks[7:9]
    print('Tasks:', tasks)
    print('Eval dts:', eval_dts)
    for task in tasks:
        for eval_dt in eval_dts:
            # for model_type in ['baseline', 'timeaware-rk4']:
            # for model_type in ['timeaware-euler']:
            for model_type in ['timeaware']:
                # set new settings for evaluation 
                eval_settings.task = task
                eval_settings.eval_dt = eval_dt
                eval_settings.model_type = model_type
                # evaluate learning curve on {task} with {eval_dt} evaluation timestep
                eval_learning_curve()
        print('='*50)
