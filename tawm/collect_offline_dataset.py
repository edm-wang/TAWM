import os
import numpy as np
import torch
from tensordict.tensordict import TensorDict

import hydra
from termcolor import colored

from tdmpc2 import TDMPC2
from common import TASK_SET
from common.parser import parse_cfg
from envs import make_env

from tqdm import tqdm
from datetime import datetime

import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

cwd = os.path.dirname(__file__)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

""" ==================================================
    Code to collect data using RANDOM actions
==================================================="""
@hydra.main(config_name='config', config_path='.')
def collect_data(cfg:dict):
    """ Create saving path"""
    os.system(f'mkdir -p {cfg.data_dir}')
    print('Saving data in', colored(cfg.data_dir, 'blue'))

    """ Set Task ID for all tasks in task_set"""
    TASK_IDs = {}
    for _id, task in enumerate(TASK_SET[cfg.task_set]):
        TASK_IDs[task] = _id
    print(colored('Task name & ID:', 'yellow'))
    print('\t', colored(TASK_IDs, 'yellow'))
    
    """ Preprocessing: Get largest obs & action dimension for zero-paddding """
    global obs_dim, action_dim
    obs_dim = 1
    action_dim = 1
    for task in TASK_SET[cfg.task_set]:
        # 1. Initialize env
        _cfg = cfg.copy()
        _cfg.task = task
        _cfg.episode_length = cfg.ep_length
        task_id = TASK_IDs[task]

        # 2. Get obs & action dimension
        env = make_env(_cfg)
        # `_cfg.obs_shape` & `_cfg.action_dim` are set in make_env(_cfg)
        obs_dim = max(_cfg.obs_shape['state'][0], obs_dim)
        action_dim = max(_cfg.action_dim, action_dim)
    print(colored(f'Multitask {cfg.task_set}: \t obs_dim={obs_dim} \t action_dim={action_dim}', 'green', attrs=['bold']))
    print(colored(f'Temporal aware: \t {cfg.multi_dt}', 'green', attrs=['bold']))

    """ Collect data from each individual task"""
    if cfg.specific_task is not None:
        # collect data for a speficied individual task only
        if task not in TASK_SET[cfg.task_set]:
            raise Exception(f'Task {task} is not in {cfg.task_set}')
        else:
            _collect_task_data(cfg, cfg.specific_task, task_ids=TASK_IDs) 
    else:
        # collect data for all tasks in specified task_set
        for task in TASK_SET[cfg.task_set]:
            _collect_task_data(cfg, task, task_ids=TASK_IDs) 

""" ==================================================
        Data collection for a given task
================================================== """
def _collect_task_data(cfg:dict, task:str, task_ids:dict):
    """ 
        Cmd-line Arguments
            task_set    : 'mt9', 'mt10', 'mt30', 'mt80'
            num_eps     : number of episodes collected per task (original TDMPC2: 24,000)
            ep_length   : 100 (DMControl & Meta-World) or 500 (DMControl only)
            data_dir    : saving data path
    """
    num_eps = cfg.num_eps
    ep_length = cfg.ep_length
    data_dir = cfg.data_dir

    """ 
        TensorDict to store data for all tasks
        format: 
            {
                'obs'		: Tensor(N, Ep_length, padded_obs_dim)
                'action'	: Tensor(N, Ep_length, padded_action_dim)
                'reward'	: Tensor(N, Ep_length)
                'task'		: Tensor(N, Ep_length)
                'timestep'	: Tensor(N, Ep_length)
            }
    """
    data_tds = [None] * num_eps

    # 1. Initialize env
    _cfg = cfg.copy()
    _cfg.task = task
    _cfg.episode_length = ep_length
    task_id = task_ids[task]
    env = make_env(_cfg)

    # 2. Load task model for data collection / experience sampling
    _cfg = parse_cfg(_cfg)
    # NOTE: Settings to speed up agent prediction -> faster data collection
    _cfg.horizon = 1 # _cfg.mpc = False
    _cfg.multi_dt = True # initialize multidt world model -> collect data
    # Initialize & load model
    model_name = task.replace('-', '_') + '_multidt_exp11'
    model_weight = f'{cwd}/agents/{model_name}.pt'
    print(colored(f'collect {task}\'s offline data using {model_weight}', 'yellow'))
    agent = TDMPC2(_cfg)
    agent.model = agent.model.to(device)
    agent.load(model_weight)
    agent.model.eval()

    # 3. Set default timestep
    if task[:2] == 'mw':
        default_dt = env.env.env.sim.model.opt.timestep
    else:
        default_dt = env.env.env.physics.model.opt.timestep
    print('Task:', colored(task, 'blue'), '| default timestep =', colored(default_dt, 'blue'))

    # 4. Execute `num_eps` episodes for each env
    start = datetime.now()
    for ep in range(num_eps):
        # 4.1. Set timestep for simulation
        if cfg.multi_dt:
            dt = np.random.uniform(low=0.001, high=round(default_dt*1.9, 4))
            dt = round(dt, 4)
            if task[:2] == 'mw': 
                # Meta-World envs
                env.env.env.sim.model.opt.timestep = round(dt, 4)
            else: 
                # DMControl envs
                env.env.env.physics.model.opt.timestep = round(dt, 4)
        else:
            dt = default_dt
        # print(dt)

        # 4.2. Simulation for `ep_length` steps
        #      action sampled from 3 sources
        #       (1) 10%: random sampling; 
        #       (2) 30%: agent's model-free planner (pi); 
        #       (3) 60%: agent's mpc planner
        obs, done, t = env.reset(), False, 0
        ep_tds = [to_td(_pad_obs(obs))]
        while (not done) and (t < ep_length):
            # sample actions
            if ep < int(num_eps*0.1):
                action = env.rand_act() # random action
            elif ep < int(num_eps*0.4): 
                agent.cfg.mpc = False   # action ~ model-free planner
                action = agent.act(obs, t0=t==0, eval_mode=False, timestep=dt) # eval_mode=False for diverse dataset
            else:
                agent.cfg.mpc = True    # action ~ MPC planner
                action = agent.act(obs, t0=t==0, eval_mode=False, timestep=dt) # eval_mode=False for diverse dataset
            obs, reward, done, info = env.step(action)
            t += 1
            # add experience to TensorDict()
            ep_tds.append(to_td(_pad_obs(obs), _pad_action(action), reward, task_id, dt))
            
        # 4.3. Concatenate all timesteps in current simulation / episode
        ep_tds = torch.cat(ep_tds)
        # 4.4. Add simulation to dataset TensorDict()
        # data_tds.append(ep_tds)
        data_tds[ep] = ep_tds

        if (ep+1) % 500 == 0:
            end = datetime.now()
            print('\t Task', colored(task, 'blue'), f'| \t {ep+1} episodes collected \t | timelapse = {end-start}.')

    # 5. Concatenate all simulations / episodes
    batch_size = [len(data_tds), ep_length+1]
    keys = data_tds[0].keys()
    data_tds = {key: torch.stack([td[key] for td in data_tds]) for key in keys}
    data_tds = TensorDict(data_tds, batch_size=batch_size)
    # Overide NAN task_id at td[:,0], then cast to torch.int32
    data_tds['task'][:,0] = data_tds['task'][:,1]
    data_tds['task'] = data_tds['task'].int()

    # 6. save task dataset
    torch.save(data_tds, f'{data_dir}/offline-data-{task}.pt')

def _pad_obs(obs):
    if obs_dim != obs.shape[0]:
        obs = torch.cat([obs, torch.zeros(obs_dim-obs.shape[0], dtype=obs.dtype, device=obs.device)])
    return obs

def _pad_action(action):
    if action_dim != action.shape[0]:
        action = torch.cat([action, torch.zeros(action_dim-action.shape[0], dtype=action.dtype, device=action.device)])
    return action


def to_td(obs, action=None, reward=None, task=None, timestep=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(torch.rand((action_dim)), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))
        task = torch.tensor(float('nan')) if (task is None) else torch.tensor(task, dtype=torch.int32)
        timestep = torch.tensor(float('nan')) if (timestep is None) else torch.tensor(timestep)
            
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            task=task.unsqueeze(0),
            timestep=timestep.unsqueeze(0),
        ), batch_size=(1,))

        return td

if __name__ == '__main__':
    collect_data()