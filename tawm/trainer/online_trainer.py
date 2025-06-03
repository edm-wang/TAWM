import os
from time import time

import numpy as np # type: ignore
import torch # type: ignore
from tensordict.tensordict import TensorDict # type: ignore
import torchrl # type: ignore
from common import TASK_SET

import gymnasium # type: ignore

from trainer.base import Trainer
from termcolor import colored # type: ignore


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._train_success_states = 0
        self._start_time = time()
        # self.best_eval_reward = -99999999
        print(f'Episode length for {self.cfg.task}: {self.cfg.episode_length}')

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    """ get obs for tdmpc2 in Torch.Tensor() type
        UPDATE: no dt concat -> store dt in buffer
    """
    def get_obs_tensor(self, obs : torch.Tensor):
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs
        else:
            obs_tensor = torch.from_numpy(obs)
            if obs_tensor.dtype == torch.float64:
                obs_tensor = obs_tensor.float()
        
        return obs_tensor

    """=======================================
            Training Helper: set env Δt
    ======================================="""
    def set_env_timestep(self, dt):
        if (dt is not None): #  and (self.env.env.env.physics.model.opt.timestep != dt)
            # Meta-World envs
            if self.cfg.task[:2] == 'mw': 
                self.env.obs_dt = dt
            # ControlGym envs
            elif self.cfg.task[:3] == 'pde':
                _ , task = self.cfg.task.split('-', maxsplit=1)
                self.env.set_sim_dt(dt)
            else: 
                raise ValueError(f'{self.cfg.task} is not defined.')

    """=======================================
            Training Helper: get env Δt
    ======================================="""
    def get_env_timestep(self):
        # Meta-World envs
        if self.cfg.task[:2] == 'mw': 
            return self.env.obs_dt
        # ControlGym envs (PDE control)
        elif self.cfg.task[:3] == 'pde':
            return self.env.get_sim_dt()
        else: 
            raise ValueError(f'{self.cfg.task} is not defined.')

    """=======================================
        method for step env with non-default timestep
        case 1: Δt <= Δt_{default}: 
            (1) set env.timestep to Δt; 
            (2) then env.step()
        case 2: Δt > Δt_{default} : 
            (1) segment Δt = Δt_0 + N*Δt_{default}; 
            (2) env.step() with Δt_0; 
            (3) env.step() with Δt for N times
    ======================================="""
    def env_step_adaptive_dt(self, action, dt):
        # Meta-World envs
        if self.cfg.task[:2] == 'mw':
            return self.env.step_adaptive_dt(action, dt)
        # ControlGym envs
        elif self.cfg.task[:3] == 'pde':
            return self.env.step_adaptive_dt(action, dt)
        else:
            raise ValueError(f'{self.cfg.task} is not defined.')

    def eval(self):
        """ evaluate using default timestep if trained on multiple timesteps"""
        self.dt = self.cfg.default_dt
        self.set_env_timestep(self.dt)
        assert self.get_env_timestep() == self.cfg.default_dt

        print(colored('Eval using dt =', 'green'), self.get_env_timestep())
        
        print(colored('Training Step:', 'green'), self._step)
        print(colored('Total training success states:', 'green'), self._train_success_states)

        """ adjust _max_episode_steps to accomodate default timestepping """
        self.env.env._max_episode_steps = self.cfg.episode_length

        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, -1
            obs = obs.view(-1)
            
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i==0))
            while not done:
                t += 1
                obs_tensor = self.get_obs_tensor(obs)
                dt = self.dt if self.cfg.multi_dt else None
                action = self.agent.act(obs_tensor, t0=t==0, eval_mode=True, timestep=dt)
                obs, reward, done, info = self.env.step(action)
                done = done or (t >= self.cfg.episode_length-1)
                ep_reward += reward
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
                    
            ep_rewards.append(ep_reward.detach().cpu().numpy())
            ep_successes.append(info['success'])
            
            if self.cfg.save_video:
                self.logger.video.save(self._step)

        # training success ratio
        train_sr = round(self._train_success_states/max(1, self._step), 4)

        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            train_success_ratio=np.array(train_sr),
        )

    def to_td(self, obs, action=None, reward=None, dt=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()

        
        # action = torch.full_like(self.env.rand_act(), float('nan'))
        if action is None:
            try:
                action = torch.full_like(self.env.rand_act(), float('nan'))
            except:
                self.env.action_space.sample().astype(np.float32)
                if isinstance(self.env.action_space, gymnasium.spaces.Discrete): 
                    # discrete -> continuous action space
                    if self.env.action_space.n == 2: action = torch.from_numpy(action).unsqueeze(0)
                    else: raise NotImplementedError('Expect Discrete action space to be binary!')
                else:
                    # continous action space
                    action = torch.from_numpy(action)

                action = torch.full_like(action, float('nan'))
        if not isinstance(action, torch.Tensor):
            action = torch.Tensor(action.astype(np.float32))

        if reward is None:
            reward = torch.tensor(float('nan'))
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)

        if dt is None:
            dt = torch.tensor(float('nan'))
        else:
            dt = torch.tensor(dt)

        """ NOTE: save everything in buffer to cpu to avoid crashing GPU"""
        """ NOTE: use torch.no_grad() and clone() obs, action, reward to avoid backward() graph error"""
        with torch.no_grad():
            obs = obs.detach().clone().to('cpu')
            action = action.detach().clone().to('cpu')
            reward = reward.detach().clone().to('cpu')
        
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            timestep=dt.unsqueeze(0),
        ), batch_size=(1,))
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, True

        """ train agent for `self.cfg.steps` number of steps
            process:
                * eval periodically
                1. obs = env.reset() if done
                2. action ~ MPC
                3. obs, reward, done, info = env.step(action)
                4. agent.update()
                5. self._steps += 1
        """
        while self._step <= self.cfg.steps:

            """ * Evaluate agent periodically"""
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True
                
            """ 
                1. Reset environment if DONE: 
                     (1) log performance; 
                    (2) buffer exp; 
                    (3) env.reset()
            """
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False

                    # if True: # eval_metrics['episode_reward'] >= self.best_eval_reward:
                    #     # 1. save weights
                    #     self.agent.save(self.cfg.checkpoint)
                    #     # 2. update best eval_reward
                    #     # self.best_eval_reward = eval_metrics['episode_reward']
                    #     print('New checkpoint with ep_reward =', eval_metrics['episode_reward'])
                    
                    # save model weights at current eval step
                    self.agent.save(f'{self.cfg.checkpoint}/step_{self._step}.pt')

                if self._step > 0:
                    """ NOTE: modified code to compatible with Nivida's DiffRL envs"""
                    try:
                        info_success = info['success']
                    except:
                        info_success = -1

                    train_metrics.update(
                        episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
                        episode_success=info_success,
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                
                """ MAIN COMPONENT OF TAWM: Adaptive Time Stepping
                    reset simulation: adaptive time step size sampling
                        (1) set new env timestep; 
                        (2) adjust _max_episode_steps to accomodate different timestepping;
                        (3) env.reset()
                """
                if self.cfg.multi_dt:
                    """ (1) set new env timestep """
                    if self.cfg.task[:2] == 'mw':
                        """ Meta-World observation delta t """
                        # self.dt = np.random.uniform(low=np.log(0.001), high=np.log(max(0.05, 2*self.cfg.default_dt))) # Log sampling; Max dt = max(0.05, 2 * default_dt)
                        if self.cfg.dt_sampler == 'uniform':
                            # uniform sampling
                            self.dt = round(np.random.uniform(low=0.001, high=0.05), 4) # Uniform sampling; Max dt = 50ms
                        elif self.cfg.dt_sampler == 'log-uniform':
                            # log-uniform sampling (updated)
                            if self._step < 100000: # max dt = 20ms
                                self.dt = np.random.uniform(low=np.log(0.001), high=np.log(0.02)) # Log sampling; Max dt = 20ms
                            elif self._step < 200000: # max dt = 30ms
                                self.dt = np.random.uniform(low=np.log(0.001), high=np.log(0.03)) # Log sampling; Max dt = 30ms
                            else:   # max dt = 50ms
                                self.dt = np.random.uniform(low=np.log(0.001), high=np.log(0.05)) # Log sampling; Max dt = 50ms
                            self.dt = round(float(np.exp(self.dt)), 4)
                        # ==========================================================
                    elif self.cfg.task[:3] == 'pde':
                        """ ControlGym observation rate """
                        _ , task_name = self.cfg.task.split('-', maxsplit=1)
                        max_dt = 1.0

                        if self.cfg.dt_sampler == 'uniform':
                            # uniform sampling
                            self.dt = round(np.random.uniform(low=0.01, high=max_dt), 4) # Uniform sampling; Max dt = 1.0 default
                        elif self.cfg.dt_sampler == 'log-uniform':
                            self.dt = np.random.uniform(low=np.log(0.01), high=np.log(max_dt)) # Log sampling; Max dt = 1.0 default
                            self.dt = round(float(np.exp(self.dt)), 4)
                    else:
                        raise NotImplementedError
                    

                    # self.dt = self.cfg.default_dt # experiment: fixed dt
                    # set seed timestep to default timestep in pretraining round
                    if self._step < self.cfg.seed_steps:
                        self.dt = self.cfg.default_dt

                    # set simulation timestep for the next episode
                    self.set_env_timestep(self.dt)

                    """ 2. adjust _max_episode_steps to accomodate different timestepping """
                    # adjust episode length to accomodate different timestep:
                    # a long simulation timestep = multiple default steps
                    n_steps = int(self.dt // self.cfg.default_dt)
                    dt_0 = round(self.dt - n_steps * self.cfg.default_dt, 4)
                    n = int(dt_0 > 0) + n_steps # num actual simulation steps per algorithmic step
                    if self.cfg.task[:2] == 'mw':
                        # specific to Meta-World envs
                        self.env.env._max_episode_steps = self.cfg.episode_length * n # env -> TimeLimit() wrapper -> _max_episode_steps
                        self.env.max_episode_steps = self.cfg.episode_length * n      # env -> max_episode_steps
                    else:
                        # specific to dmcontrol envs
                        self.env.max_episode_steps = self.cfg.episode_length * n

                """ (3) env.reset() """
                print(f'\tTraining episode using dt =', self.get_env_timestep())
                obs = self.env.reset()
                t = 0
                
                # obs = obs.view(-1)
                obs_tensor = self.get_obs_tensor(obs)
                self._tds = [self.to_td(obs_tensor)]

            """ 2. Sample action using MPC """
            # Collect experience
            dt = self.dt if self.cfg.multi_dt else self.cfg.default_dt
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs_tensor, t0=len(self._tds)==1, timestep=dt)
            else:
                if self.cfg.task[:2] == 'f1':
                    # pretraining on classic planner's actions
                    action = self.env.plan_classic()
                else:
                    # pretraining on random actions
                    action = self.env.rand_act()
            
            # """ 3. Take action & move to next state: env.step(action)"""
            # obs, reward, done, info = self.env.step(action)
            """ 3. Take action & move to next state under sim.timestep dt: env_step_adaptive_dt(action, dt)"""
            t += 1
            obs, reward, done, info = self.env_step_adaptive_dt(action, dt)
            done = done or (t >= self.cfg.episode_length-1)
            self._train_success_states += info['success']
            
            obs_tensor = self.get_obs_tensor(obs)
            self._tds.append(self.to_td(obs_tensor, action, reward, dt))
                
            """ 
                4. Update agent
                    a. collect experiences for `seed_steps` steps first
                        -> only start updating when self._step >= self.cfg.seed_steps
                    b. start trainning on buffer exp for `num_updates` iterations
            """
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    # Save buffer as offline dataset
                    # os.system(f'mkdir -p {self.cfg.data_dir}')
                    # self.buffer._buffer._storage.dumps(f'{self.cfg.data_dir}/offline-data-{self.cfg.task}-seed{self.cfg.seed}.pt')
                    num_updates = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1
    
        self.logger.finish(self.agent)
