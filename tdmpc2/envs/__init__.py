from copy import deepcopy
import warnings

import gym # type: ignore
import gymnasium # type: ignore

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper
from common import TASK_SET

def missing_dependencies(task):
    raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
    from envs.metaworld import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies
try:
    from envs.control_gym import make_env as make_controlgym_env # type: ignore
    # make_controlgym_env = missing_dependencies
except:
    make_controlgym_env = missing_dependencies

warnings.filterwarnings('ignore', category=DeprecationWarning)
    

def make_env(cfg):
    """
    Make an environment for TD-MPC2 experiments.
    """
    gym.logger.set_level(40)
    if cfg.multitask:
        raise NotImplementedError("")
    else:
        env = None
        for fn in [make_metaworld_env, make_controlgym_env]:
            try:
                env = fn(cfg)
            except:
                pass
        if env is None:
            raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
        env = TensorWrapper(env)
    if cfg.get('obs', 'state') == 'rgb':
        env = PixelWrapper(cfg, env)
    try: # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except: # Box
        try: 
            cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
        except:
            cfg.obs_shape = {'state': env.obs_shape}
    
    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        cfg.action_dim = 1
    else:
        if cfg.task[:2] == 'f1': # f1tenth racing
            cfg.action_dim = env.action_dim
        else:
            cfg.action_dim = env.action_space.shape[0]
    
    # set episode length
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5*cfg.episode_length)
    return env
