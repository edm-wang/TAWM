""" ***********************************************************
        Trainer to train state decoder for computing the 
        env gradients of TDMPC-2
********************************************************** """

from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
import jax.numpy as jp

from trainer.base import Trainer

class DecoderTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    """ Convert jax.numpy.array to torch.Tensor """
    def jax_array_to_tensor(self, x):
        x = np.asarray(x)
        x = torch.from_numpy(x)
        return x

    """ Evaluate TD-MPC2 agent."""
    def eval(self):
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            (obs, done, ep_reward), t = self.env.reset(), 0
            
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i==0))

            """ Simulate an Episode for evaluation """
            while not done:
                action = self.agent.act(self.jax_array_to_tensor(obs), 
                                        t0=t==0, eval_mode=True)
                action = jp.array(action) # convert action to jax.numpy.array for MJX physics step

                """ modified code to compatible with MJX envs """
                obs, reward, done, info, new_mjx_data = self.env.step(obs, action)
                self.env.mjx_data = new_mjx_data
                    
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)

            """ modified code to compatible with Nivida's DiffRL envs"""
            try:
                ep_successes.append(info['success'])
            except:
                # ignore: some MJX envs may not have info['successes']
                ep_successes.append(-1)
            
            if self.cfg.save_video:
                self.logger.video.save(self._step)
                
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None):
        """ convert all observations from jax.ndarray to Tensor"""
        if isinstance(obs, jp.ndarray):
            obs = self.jax_array_to_tensor(obs)
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()

        """ convert all action from jax.ndarray to Tensor """
        if isinstance(action, jp.ndarray):
            action = self.jax_array_to_tensor(action)
        if action is None:
            action = self.env.rand_act()
            action = self.jax_array_to_tensor(action)
            action = torch.full_like(action, float('nan'))

        """ convert all rewards to Tensor"""
        if reward is None:
            reward = torch.tensor(float('nan'))
        else:
            reward = torch.tensor(reward)
        
        """ create TensorDict() to be added to buffered experiences"""
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
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

            """ * Evaluate & save agent periodically"""
            if self._step % self.cfg.eval_freq == 0:
                # 1. save weights
                self.agent.save(self.cfg.checkpoint)
                # 2. set mode to eval()
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

                obs, reward, done = self.env.reset()
                self._tds = [self.to_td(obs)]

            """ 2. Sample action using MPC """
            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(self.jax_array_to_tensor(obs), 
                                        t0=len(self._tds)==1)
                action = jp.array(action)
            else:
                action = self.env.rand_act()
            
            """ 3. Take action & move to next state: env.step(action)"""
            """ NOTE: modified code to compatible with MJX envs"""
            obs, reward, done, info, new_mjx_data = self.env.step(obs, action)
            self.env.mjx_data = new_mjx_data

            """ 4. Add experiences to TensorDict() 
                    -> train world models
                    -> convert obs from jax.numpy.array to numpy.array        
            """
            self._tds.append(self.to_td(obs, action, reward))

            
            """ 
                4. Update agent
                    a. collect experiences for `seed_steps` steps first
                        -> only start updating when self._step >= self.cfg.seed_steps
                    b. start trainning on buffer exp for `num_updates` iterations
            """
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1
    
        self.logger.finish(self.agent)