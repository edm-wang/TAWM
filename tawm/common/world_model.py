from copy import deepcopy

import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore

from common import layers, math, init


class WorldModel(nn.Module):
    """
    TAWM with TD-MPC2's implicit world model architecture.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.action_dims[i]] = 1.
        
        """ state encoder: obs -> z"""
        self._encoder = layers.enc(cfg)
        
        """ state dynamics model: 
            1. default   : (z, a, task) -> z' 
            2. time-aware: (z, a, task, dt) -> z'; z' = z + d1(z,a,task,dt) * dt

            reward model:
            1. default   : (z, a, task)  -> r
            2. time-aware: (z, a, task, dt) -> r

            planner pi: Gaussian prior (mu & std) of action space
            1. default   : (z, task) -> (mu_action, std_action)
            2. time-aware: (z, task, dt) -> (mu_action, std_action)
            
            q-value model:
            1. default   : (z, a, task) -> q-value
            2. time-aware: (z, a, task, dt) -> q-value
        """
        if cfg.multi_dt:
            # Time-Aware World Model (TAWM)
            self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim + 1, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
            self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim + 1, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
            self._pi = layers.mlp(cfg.latent_dim + 1 + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
            self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + 1 + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        else:
            # Non Time-Aware Baseline
            self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
            self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
            self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
            self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
        # init.zero_([self._state_reward.weight, self._action_reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        if self.cfg.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        return self
    
    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        This method also enables/disables gradients for task embeddings.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.cfg.multitask:
            for p in self._task_emb.parameters():
                p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.cfg.tau)
    
    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task, timestep):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])

        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task, timestep):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        
        if self.cfg.multi_dt:
            """ Time-aware model: z' = z + d1(z,a) * dt
                -> linearize non-linear dynamics
            """
            dt = timestep if (len(timestep.shape) == len(z.shape)) else timestep.repeat(z.shape[0], 1)

            if self.cfg.integrator == 'euler':
                # TAWM: Euler integration method: 
                #       z' = z + d(z,a,dt) * (log(dt)+5)
                z_dot = self._dynamics(torch.cat([z, a, dt], dim=-1))
                dt_norm = (torch.log10(dt) + 5) * (dt > 0) # take log10 where dt > 0
                dt_norm = torch.nan_to_num(dt_norm, nan=0.) # replace nan (log(dt) where dt=0) as 0
                return z + z_dot * dt_norm
            elif self.cfg.integrator == 'rk4':
                # TAWM: 4th-order Runge-Kutta method
                # normalize dt
                dt_norm = (torch.log10(dt) + 5) * (dt > 0) # take log10 where dt > 0
                dt_norm = torch.nan_to_num(dt_norm, nan=0.) # replace nan (log(dt) where dt=0) as 0
                # normalize dt/2
                half_dt_norm = (torch.log10(dt/2) + 5) * (dt/2 > 0) # take log10 where dt/2 > 0
                half_dt_norm = torch.nan_to_num(half_dt_norm, nan=0.) # replace nan (log(dt/2) where dt/2=0) as 0
                
                # NOTE: we have to handle multitask emb z = [latent, task_emb] differently from single task z = latent
                if not self.cfg.multitask:
                    # start from initial point z
                    d1 = self._dynamics(torch.cat([z, a, dt], dim=-1))          # slope at z in log(dt)
                    # move to middle point z1
                    z1 = z + self._dynamics(torch.cat([z, a, dt/2], dim=-1)) * half_dt_norm 
                    d2 = self._dynamics(torch.cat([z1, a, dt], dim=-1))         # slope at z1 in log(dt)
                    # move to middle point z2
                    z2 = z + self._dynamics(torch.cat([z1, a, dt/2], dim=-1)) * half_dt_norm 
                    d3 = self._dynamics(torch.cat([z2, a, dt], dim=-1))         # slope at z2 in log(dt)
                    # move to end point z3
                    z3 = z + d3 * dt_norm 
                    d4 = self._dynamics(torch.cat([z3, a, dt], dim=-1))         # slope at z3 in log(dt)
                    return z + 1/6 * (d1 + 2*d2 + 2*d3 + d4) * dt_norm      # z' = z + 1/6 * (d1+2*d2+2*d3+d4) * dt
                else:
                    # extract only latent z (exclude task emb)
                    z_latent = z[:,:-self.cfg.task_dim]
                    # start from initial point z
                    d1 = self._dynamics(torch.cat([z, a, dt], dim=-1))          # slope at z in log(dt)
                    # move to middle point z1
                    z1 = z_latent + self._dynamics(torch.cat([z, a, dt/2], dim=-1)) * half_dt_norm 
                    z1 = self.task_emb(z1, task) # task-emb z1 for dynamics model
                    d2 = self._dynamics(torch.cat([z1, a, dt], dim=-1))         # slope at z1 in log(dt)
                    # move to middle point z2
                    z2 = z_latent + self._dynamics(torch.cat([z1, a, dt/2], dim=-1)) * half_dt_norm 
                    z2 = self.task_emb(z2, task) # task-emb z2 for dynamics model
                    d3 = self._dynamics(torch.cat([z2, a, dt], dim=-1))         # slope at z2 in log(dt)
                    # move to end point z3
                    z3 = z_latent + d3 * dt_norm 
                    z3 = self.task_emb(z3, task) # task-emb z3 for dynamics model
                    d4 = self._dynamics(torch.cat([z3, a, dt], dim=-1))         # slope at z3 in log(dt)
                    return z_latent + 1/6 * (d1 + 2*d2 + 2*d3 + d4) * dt_norm      # z' = z + 1/6 * (d1+2*d2+2*d3+d4) * dt
        else:
            """ Original model: z' = d(z,a) """
            return self._dynamics(torch.cat([z, a], dim=-1))
    
    def reward(self, z, a, task, timestep):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        
        if self.cfg.multi_dt:
            dt = timestep if (len(timestep.shape) == len(z.shape)) else timestep.repeat(z.shape[0], 1)
            return self._reward(torch.cat([z, a, dt], dim=-1)) # @debugging
        else:
            """ Original model: r = r(z,a)"""
            return self._reward(torch.cat([z, a], dim=-1))

    def pi(self, z, task, timestep):

        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        
        """ if multi-dt -> use [z, dt]"""
        if self.cfg.multi_dt:
            dt = timestep if (len(timestep.shape) == len(z.shape)) else timestep.repeat(z.shape[0], 1)
            z = torch.cat([z, dt], dim=-1)

        # Gaussian policy prior
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask: # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else: # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type='min', target=False, timestep=None):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        """ if multi-dt -> use [z, a, dt]"""
        if self.cfg.multi_dt:
            dt = timestep if (len(timestep.shape) == len(z.shape)) else timestep.repeat(z.shape[0], 1)
            z = torch.cat([z, a, dt], dim=-1)
        else:
            z = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)

        if return_type == 'all':
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
