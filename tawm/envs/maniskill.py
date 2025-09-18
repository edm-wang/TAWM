import numpy as np # type: ignore
import torch # type: ignore
import gym # type: ignore
import gymnasium # type: ignore
from envs.wrappers.time_limit import TimeLimit

import mani_skill2.envs


MANISKILL_TASKS = {
    'lift-cube': dict(
        env='LiftCube-v0',
        control_mode='pd_ee_delta_pos',
    ),
    'pick-cube': dict(
        env='PickCube-v0',
        control_mode='pd_ee_delta_pos',
    ),
    'stack-cube': dict(
        env='StackCube-v0',
        control_mode='pd_ee_delta_pos',
    ),
    'pick-ycb': dict(
        env='PickSingleYCB-v0',
        control_mode='pd_ee_delta_pose',
    ),
    'turn-faucet': dict(
        env='TurnFaucet-v0',
        control_mode='pd_ee_delta_pose',
    ),
}


class ManiSkillWrapper(gym.Wrapper):
    """Extended ManiSkill2 wrapper with timestep control for TAWM."""
    
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
        self.default_dt = self.get_sim_dt()
        self.obs_dt = self.get_sim_dt() # initial obs_dt; for monitoring purpose
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Convert to numpy array if it's a dict
        if isinstance(obs, dict):
            obs = np.concatenate([v.flatten() for v in obs.values()])
        return obs.astype(np.float32)

    def step(self, action):
        reward = 0
        for _ in range(2):
            obs, r, terminated, truncated, info = self.env.step(action)
            reward += r
            done = terminated or truncated
            if done:
                break
        # Convert to numpy array if it's a dict
        if isinstance(obs, dict):
            obs = np.concatenate([v.flatten() for v in obs.values()])
        obs = obs.astype(np.float32)
        return obs, reward, done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()
    
    # get env's simulation Δt 
    def get_sim_dt(self):
        # ManiSkill2 envs - get from the underlying physics simulation
        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'model'):
            return self.env.sim.model.opt.timestep
        else:
            # Default timestep for ManiSkill2
            return 0.01
    
    # set env's simulation Δt (for simulating cases where Δt < Δt_{default})
    def set_sim_dt(self, dt):
        if (dt is not None):
            assert dt > 0 and dt <= self.default_dt
            if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'model'):
                self.env.sim.model.opt.timestep = dt

    """ method for step env with non-default observation timestep
        case 1: Δt <= default_Δt: 
            (1) set env.timestep to dt; 
            (2) then env.step()
        case 2: Δt > default_Δt : 
            (1) segment dt = Δt_0 + N * default_Δt; 
            (2) env.step() with Δt_0; 
            (3) env.step() with Δt for N times
    """
    def step_adaptive_dt(self, action, dt):
        """ Simulation stepping: 
                (1) break large timestep -> smaller timesteps 
                (2) step multiple sub-steps -> multiple env.step()
        """
        # 1. segment the current step of dt into different simulation steps,
        #    where each step has max time-stepping of default dt
        n_steps = int(dt // self.default_dt)
        # remainder timestep dt_0:  Δt = Δt_0 + N * default_Δt
        dt_0 = round(dt - n_steps * self.default_dt, 5)

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # 2. execute the first simulation with timestep = Δt_0
        if dt_0 > 0:
            self.set_sim_dt(dt_0)
            obs, reward, done, info = self.step(action)
        # 3. execute the subsequent simulations with default timestep default_Δt
        self.set_sim_dt(self.default_dt)
        for _ in range(n_steps):
            obs, reward, done, info = self.step(action)
        # 4. return the last simulation step's [obs, reward, done, info]
        return torch.Tensor(obs).float(), \
                torch.tensor(reward).float(), \
                done, info


def make_env(cfg):
    """
    Make ManiSkill2 environment using TDMPC2 integration with TAWM timestep control.
    """
    if cfg.task not in MANISKILL_TASKS:
        raise ValueError('Unknown task:', cfg.task)
    assert cfg.obs == 'state', 'This task only supports state observations.'
    task_cfg = MANISKILL_TASKS[cfg.task]
    env = gymnasium.make(
        task_cfg['env'],
        obs_mode='state',
        control_mode=task_cfg['control_mode'],
        render_camera_cfgs=dict(width=384, height=384),
        render_mode='cameras',
    )
    env = ManiSkillWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
