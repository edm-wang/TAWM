from collections import defaultdict

import gym
import gymnasium
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.
    """

    def __init__(self, env):
        super().__init__(env)
    
    def rand_act(self):
        if isinstance(self.env.action_space, gymnasium.spaces.Discrete): 
            # discrete -> continuous action space
            if self.env.action_space.n == 2:
                action = np.array(self.action_space.sample()).astype(np.float32)
                return torch.from_numpy(action).unsqueeze(0)
            else:
                raise NotImplementedError('Expect Discrete action space to be binary!')
        else:
            # continous action space
            return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _try_f32_tensor(self, x):
        x = torch.from_numpy(x)
        if x.dtype == torch.float64:
            x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, task_idx=None):
        return self._obs_to_tensor(self.env.reset())

    def step(self, action):
        # Convert action to numpy if it's a torch tensor
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        obs, reward, done, info = self.env.step(action)
        info = defaultdict(float, info)
        info['success'] = float(info['success'])
        return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info
