import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch # type: ignore
import numpy as np # type: ignore

import hydra # type: ignore
from termcolor import colored # type: ignore

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger
from common import TASK_SET

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
    """
    Script for training single-task / multi-task TD-MPC2 agents.

    Most relevant args:
        `task`: task name (or mt30/mt80 for multi-task training)
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `steps`: number of training/environment steps (default: 10M)
        `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ```
        $ python train.py task=mt80 model_size=48
        $ python train.py task=mt30 model_size=317
        $ python train.py task=dog-run steps=7000000
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.steps > 0, 'Must train for at least 1 step.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
    
    """ Specify Trainer to train model given model architecture & environment"""
    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
    env = make_env(cfg)
        
    if not cfg.multitask:
        """ retrieve default time step of the environment"""
        if cfg.default_dt is None:
            if cfg.task[:2] == 'mw':
                print('Using Meta-World Environment')
                cfg.default_dt = env.env.env.sim.model.opt.timestep
            elif cfg.task[:3] == 'pde':
                print('Using ControlGym Environment')
                cfg.default_dt = round(env.get_sim_dt(), 4)
            else:
                raise NotImplementedError(f'{cfg.task} is not implemented!')

        """ variable timestep size """
        if cfg.multi_dt:
            # print(colored('SIMULATION TIMESTEP:', 'green', attrs=['bold']), f'Uniform([0.001, {round(max(0.01, min(0.1, cfg.default_dt*2)), 4)}])') # 0.01 <= Max dt <= 0.1
            if cfg.task[:2] == 'mw':
                print(colored('OBSERVATION TIMESTEP:', 'green', attrs=['bold']), f'{cfg.dt_sampler}([0.001, 0.05] sec)') # Max dt = max(0.05, 2 * default_dt)
            elif cfg.task[:3] == 'pde':
                max_dt = 1.0
                print(colored('OBSERVATION TIMESTEP:', 'green', attrs=['bold']), f'{cfg.dt_sampler}([0.01, {max_dt}] sec)') # Max dt = 1.0 sec
            else:
                raise NotImplementedError

            print(colored('DEFAULT SIM TIMESTEP:', 'green', attrs=['bold']), cfg.default_dt)
        else:
            print(colored('OBSERVATION TIMESTEP:', 'green', attrs=['bold']), cfg.default_dt)
    else:
        print(colored(f'TIMESTEP-AWARE for {cfg.task}', 'green', attrs=['bold']))

    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5*cfg.episode_length)
    print(colored('Pretraining steps :', 'blue'), cfg.seed_steps)
    
    
    
    """ set model checkpoint directory"""
    # model_type = 'multidt' if cfg.multi_dt else 'singledt'
    if cfg.multi_dt:
        print(colored('Integration method:', 'blue'), cfg.integrator)
        print(colored('Sampling method   :', 'blue'), cfg.dt_sampler)
        # time-aware checkpoint
        cfg.checkpoint = f'{cfg.checkpoint}/{cfg.task}/multidt-{cfg.dt_sampler}-{cfg.integrator}/{cfg.seed}'
    else:
        print(colored('Model:', 'blue'), 'baseline')
        # single-dt checkpoint (based on default training dt)
        cfg.checkpoint = f'{cfg.checkpoint}/{cfg.task}/singledt-{cfg.default_dt}/{cfg.seed}'
    print(colored('Checkpoint path:', 'blue'), cfg.checkpoint)
    os.system(f'mkdir -p {cfg.checkpoint}')
    assert os.path.exists(cfg.checkpoint)

    trainer = trainer_cls(
        cfg=cfg,
        env=env,
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print('\nTraining completed successfully')


if __name__ == '__main__':
    train()
