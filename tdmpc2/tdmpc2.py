import numpy as np # type: ignore
import torch # type: ignore
import torch.nn.functional as F # type: ignore

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel


class TDMPC2:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.model = WorldModel(cfg).to(self.device)
        
        """ time-aware world model has different architecture:
            _dynamics -> _dynamics_1 & _dynamics_2
            _reward -> _reward with different architecture
        """
        if self.cfg.multi_dt:
            # TAWM
            self.optim = torch.optim.Adam([
                {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
                {'params': self.model._dynamics.parameters()},
                {'params': self.model._reward.parameters()},
                {'params': self.model._Qs.parameters()},
                {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
            ], lr=self.cfg.lr)
        else:
            # Non Time-Aware baseline
            self.optim = torch.optim.Adam([
                {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
                {'params': self.model._dynamics.parameters()},
                {'params': self.model._reward.parameters()},
                {'params': self.model._Qs.parameters()},
                {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
            ], lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
        
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None, timestep=None):
        """
        Select an action by planning in the latent space of the world model.
        
        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: Action to take in the environment.
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
            
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)

        if timestep is not None:
            dt = torch.tensor([timestep], device=self.device)
        else:
            dt = None

        z = self.model.encode(obs, task, timestep=dt)
        # print('act:', 'obs', obs.shape, 'z', z.shape, 'dt', dt.shape)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task, timestep=dt)
        else:
            a = self.model.pi(z, task, timestep=dt)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, timestep):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        dt = timestep
        for t in range(self.cfg.horizon):
            # "Imagine" immediate reward & next state under trandition Delta t = eval_dt
            if (self.cfg.multi_dt) or (not self.cfg.eval_steps_adjusted):
                # Case 1: One-step evaluation (no adjustment) -> either time-aware model or no evaluation adjustment
                reward = math.two_hot_inv(self.model.reward(z, actions[t], task, timestep), self.cfg)
                z = self.model.next(z, actions[t], task, timestep=dt)
            else:
                # Case 2: Adjusted multi-step evaluation -> must be non-time-aware model (baseline)
                train_dt = self.cfg.train_dt if (self.cfg.train_dt is not None) else self.cfg.default_dt
                eval_dt = self.cfg.eval_dt
                pred_steps = int(np.ceil(eval_dt/train_dt)) # adjusted number of prediction steps (#times the model is applied)

                reward = math.two_hot_inv(self.model.reward(z, actions[t], task, timestep), self.cfg)
                for _ in range(pred_steps):
                    z = self.model.next(z, actions[t], task, timestep=dt)

            # print('_estimate_value:', 'z', z.shape, 'action', actions[t].shape, 'dt', dt.shape)
            G += discount * reward
            discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
        return G + discount * self.model.Q(z, self.model.pi(z, task, timestep=dt)[1], task, return_type='avg', timestep=dt)


    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None, timestep=None):
        """
        Plan a sequence of actions using the learned world model.
        
        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        dt = timestep
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t] = self.model.pi(_z, task, timestep=dt)[1]
                _z = self.model.next(_z, pi_actions[t], task, timestep=dt)
            pi_actions[-1] = self.model.pi(_z, task, timestep=dt)[1]

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
    
        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
                .clamp(-1, 1)
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task, timestep=dt).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)
        
    def update_pi(self, zs, task, timestep):
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        dt = timestep
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task, timestep=dt)
        qs = self.model.Q(zs, pis, task, return_type='avg', timestep=dt)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, reward, task, timestep):
        """
        Compute the TD-target from a reward and the observation at the following time step.
        
        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: TD-target.
        """
        dt = timestep
        pi = self.model.pi(next_z, task, timestep=dt)[1]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True, timestep=dt)

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        if not self.cfg.use_grad_reg:
            # default: no Jacobian regularizer
            obs, action, reward, task, dt = buffer.sample()
        else:
            # gradient-informed: use Jacobian regularizer
            obs, action, reward, dx2_dx1, dx2_da1, dr_dx1, dr_da1, task, dt = buffer.sample()
        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task, timestep=dt)
            td_targets = self._td_target(next_z, reward, task, timestep=dt)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        if dt is None:
            dt = [None] * self.cfg.horizon
            
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task, timestep=dt[0])
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task, timestep=dt[t])
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all', timestep=dt)
        reward_preds = self.model.reward(_zs, action, task, timestep=dt)
        
        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
        consistency_loss *= (1/self.cfg.horizon)
        reward_loss *= (1/self.cfg.horizon)
        value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))

        
        """ env Jacobian regularizer
            NOTE 1: when optimize Jacobian loss, freeze encoder -> focus on dynamics errors
            NOTE 2: current version: use L2 loss between ground-truth & world model Jacobian
        """
        # encoder Jacobian w.r.t (obs)
        def get_latent(obs, dt):
            # with torch.no_grad():
            z = self.model.encode(obs, task=0, timestep=dt) # encode raw observation to latent state
            return z
        encoder_jacobian = torch.func.vmap(torch.func.jacfwd(get_latent, argnums=0)) 

        # latent dynamics Jacobian w.r.t (obs, action)
        def get_next_latent(obs, action, dt):
            # with torch.no_grad():
            z = self.model.encode(obs, task=None, timestep=dt) # encode raw observation to latent state
            z_next = self.model.next(z, action, task=None, timestep=dt) # predict the next latent state
            return z_next
        model_dynamic_jacobian = torch.func.vmap(torch.func.jacfwd(get_next_latent, argnums=(0,1)))
        
        if self.cfg.use_grad_reg:
            gradient_loss = 0
            # gradients at different timestep t in the horizon
            for t in range(self.cfg.horizon):
                # encoder Jacobian: (ds_next / dx_next)
                ds2_dx2_t = encoder_jacobian(obs[t+1])
                """ 1. ground-truth gradients: (ds_next/dx , ds_next/da)
                        -> detach() to avoid updating encoder() -> focus on dynamics()
                """
                ds2_dx1_true_t = torch.matmul(ds2_dx2_t, dx2_dx1[t]).detach()
                ds2_da1_true_t = torch.matmul(ds2_dx2_t, dx2_da1[t]).detach()
                """ 2. world model gradients: (df/dx, df/da) at current timestep (t)"""
                ds2_dx1_model_t, ds2_da1_model_t = model_dynamic_jacobian(obs[t], action[t])
                
                """ 3. compute gradient loss: L2 error: 
                        Loss = L2(ds_next/dx, df/dx) + L2(ds_next/da, df/da)
                    NOTE: ignore NAN values resulting from JAX Env gradients"""
                not_nan_mask_ds2_dx1_t = (~torch.isnan(ds2_dx1_true_t)).float()
                not_nan_mask_ds2_da1_t = (~torch.isnan(ds2_da1_true_t)).float()
                gradient_loss += F.mse_loss(ds2_dx1_model_t * not_nan_mask_ds2_dx1_t, 
                                            torch.nan_to_num(ds2_dx1_true_t, nan=0.))
                gradient_loss += F.mse_loss(ds2_da1_model_t * not_nan_mask_ds2_da1_t, 
                                            torch.nan_to_num(ds2_da1_true_t, nan=0.))
                
            gradient_loss *= (1/self.cfg.horizon)
        else:
            """ NOTE: torch.tensor() will detach gradient_loss
                -> DO NOT USE IT WHEN cfg.use_grad_reg=true
            """
            gradient_loss = torch.tensor(0.)

        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss + 
            self.cfg.grad_reg_coeff * gradient_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Update policy
        # (T, B, D) -> (T+1, B, D); T: temporal; B: batch; D: feature dim
        if (dt is not None) and (dt[0] is not None):
            dt = dt[0].unsqueeze(0).repeat(zs.shape[0], 1, 1)
        pi_loss = self.update_pi(zs.detach(), task, timestep=dt)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "gradient_loss": float(gradient_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }
