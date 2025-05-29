import numpy as np # type: ignore
import torch # type: ignore
import gym # type: ignore
from envs.wrappers.time_limit import TimeLimit

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE # type: ignore


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env._freeze_rand_vec = False
        self.default_dt = self.get_sim_dt()
        self.obs_dt = self.get_sim_dt() # initial obs_dt; for monitoring purpose
        
    def reset(self, **kwargs):
        obs = super().reset(**kwargs).astype(np.float32)
        self.env.step(np.zeros(self.env.action_space.shape))
        return obs

    def step(self, action):
        reward = 0
        for _ in range(2):
            obs, r, _, info = self.env.step(action.copy())
            reward += r
        obs = obs.astype(np.float32)
        return obs, reward, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render(
            offscreen=True, resolution=(384, 384), camera_name=self.camera_name
        ).copy()
    
    # get env's simulation Δt 
    def get_sim_dt(self):
        # Meta-World envs
        return self.env.sim.model.opt.timestep
    # set env's simulation Δt (for simulating cases where Δt < Δt_{default})
    def set_sim_dt(self, dt):
        if (dt is not None):
            assert dt > 0 and dt <= self.default_dt
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
    Make Meta-World environment.
    """
    env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-observable"
    if not cfg.task.startswith('mw-') or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        raise ValueError('Unknown task:', cfg.task)
    assert cfg.obs == 'state', 'This task only supports state observations.'
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed)
    env = MetaWorldWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
