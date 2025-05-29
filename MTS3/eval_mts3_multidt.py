"""
    This is strictly for evaluation on 1 environment: mw-basketball
    -> TODO: implement generic function for all envs

    MTS3:
    1. requires having access to past 2H observation/action trajectory
        -> do nothing in the first 2H episodes: take zero-action: 
            -> a = np.zero()
            -> env.step(zero-action) for 2H episodes
        -> take action ~ MTS3+MPPI for step >= 2H
            -> a = plan(obs_history, action_history)
            -> env.step(a)

    MTS3+MPPI:
    1. copy tdmpc2's plan() function here
        a. replace all self. with tdmpc2. (global variable)
    2. copy tdmpc2's _estimate_value() function here
        a. replace all self. with tdmpc2. (global variable)
    3. change dynamic prediction:
        tdmpc2: take z_t
            (1) q = q-value(z,a)  
            (2) z' = tdmpc2.model.next(z, a) # latent dynamic
            (3) z = z' # repeat
        mts3: take obs_history, act_history
            (1) obs, act = obs_history[-1], act_rollout[t] # take obs,act from historical & rollout trajetory
            (2) z = tdmpc2.model.encode(obs) # use tdmpc2 encoder
            (3) q = q-value(z,act) # use tdmpc2's value function
            NOTE: most change is in dynamic model
            (4) act_history.append(act)  # take action a and push to action history
            (5) obs' = mts3(obs_history, act_history, ...)
            (6) obs_history.append(obs') # add current obs to obs history
            (7) obs = obs'

        TODO: Test implementation correctness
            (1) check mst3 dynamic implementation: rmse on interactive env
                -> compare mts3(obs_history, act_history) vs obs_next WITHIN the env.step() loop
            (2) [VERIFIED] check planner implemntation: replace back tdmpc2's dynamic model
                -> the performance should be decent
"""

import sys
sys.path.append('.')
from omegaconf import DictConfig, OmegaConf
import hydra
import os
cwd = os.path.dirname(__file__)

import warnings
warnings.filterwarnings('ignore')

import torch
import pickle
import gdown
import numpy as np

from experiments.exp_prediction_mts3 import Experiment
from hydra.utils import get_original_cwd
from agent.worldModels.MTS3 import MTS3
from agent.Infer.repre_infer_mts3 import Infer

sys.path.append('../world_models_diff_envs/tdmpc2/tdmpc2/')
import pandas as pd
import hydra
from termcolor import colored # type: ignore
from tdmpc2 import TDMPC2
from common.parser import parse_cfg
from common.layers import dec
from common import math
from common.seed import set_seed
from envs import make_env
from tqdm import tqdm

""" (I) Select model version (H: slow dynamic)"""
# mts3_H = 'H3'
# mts3_H = 'H11'
# mts3_H = 'H33'
mts3_H = 'H50'

""" (II) Select task to be evaluated """
tasks = ['mw-basketball']
model_path = '/nfshomes/anhu/MTS3/experiments/saved_models/mts3-mw-basketball.ckpt'
model_path = f'/nfshomes/anhu/MTS3/experiments/saved_models/mts3-mw-basketball-{mts3_H}.ckpt'
tdmpc2_weights = '/fs/nexus-scratch/anhu/world-model-checkpoints/mw-basketball/singledt-0.0025/1/step_1450053.pt'

# tasks = ['mw-faucet-open']
# model_path = '/nfshomes/anhu/MTS3/experiments/saved_models/mts3-mw-faucet-open.ckpt'
# model_path = f'/nfshomes/anhu/MTS3/experiments/saved_models/mts3-mw-faucet-open-{mts3_H}.ckpt'
# tdmpc2_weights = '/fs/nexus-scratch/anhu/world-model-checkpoints/mw-faucet-open/singledt-0.0025/1/step_1450053.pt'


model_cfg = None
mts3_model = None
tdmpc2 = None

""" (III) Select eval_dt to be evaluated """
# eval_dts = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05]
# eval_dts = [0.001, 0.0025, 0.005, 0.0075]
# eval_dts = [0.0025, 0.005, 0.01]
# eval_dts = [0.02, 0.03, 0.05]
# eval_dts = [0.001]
# eval_dts = [0.0025]
# eval_dts = [0.005, 0.0075]
# eval_dts = [0.01]
# eval_dts = [0.02]
# eval_dts = [0.03]
eval_dts = [0.05]

@hydra.main(config_path='experiments/basketball/conf',config_name="config")
def eval(cfg)->OmegaConf:
    global config
    global model_cfg
    global mts3_model
    global tdmpc2_model
    model_cfg = cfg

    # Initialize Experiments to obtain mw-basketball dataset for model initialization (wandb)
    exp = Experiment(model_cfg)
    wandb_run = exp._wandb_init()
    # Retrieve data for model initialization
    # data_path = get_original_cwd() + '/dataFolder/mts3_datasets_processed/mts3-data-mw-basketball.pkl'
    data_path = '/fs/nexus-scratch/anhu/mts3-data-mw-basketball.pkl'
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
        train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer = \
            data_dict['train_obs'], data_dict['train_act'], data_dict['train_targets'], data_dict['test_obs'], \
            data_dict['test_act'], data_dict['test_targets'], data_dict['normalizer']

    # Initialize model & load trained weights
    
    print('Using', model_path)
    mts3_model = MTS3(input_shape=[train_obs.shape[-1]], action_dim=train_act.shape[-1], config=model_cfg.model)
    mts3_model.load_state_dict(torch.load(model_path))
    mts3_model.H = int(mts3_H[1:])
    print('Inference H=', mts3_model.H)

    # Setup interactive environment for model prediction
    # dp_infer = Infer(mts3_model, normalizer=normalizer, config=model_cfg, run=wandb_run, log=model_cfg.model.wandb['log'])

    # evaluate model on offline dataset
    # eval_offline(test_obs, test_act, test_targets, normalizer)

    # evaluate model on online interaction with the environment
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    eval_online()

""" ====================================================
        Evaluate model using offline dataset
====================================================="""
def eval_offline(test_obs, test_act, test_targets, normalizer):
    """
        NOTE: MTS3 can only predict step >= 2H: padded history list
            -> when initialize interactive env, set env.max_episodes = 2*H + env.max_episodes
            -> use the first 2*H steps to for the model to accumulate (obs,action)
            OR:
            -> keep env.max_episodes: just do nothing in the first 2*H steps
    """
    n_episodes = test_obs.shape[0]
    for episode in range(0, n_episodes, n_episodes//10):
        obs_history = []
        act_history = []
        obs_next_history = []
        pred_history = []
        
        for step in range(test_obs.shape[1]):
            # obs, action, ground-truth obs_next in EACH STEP
            obs, action, obs_next = test_obs[episode, step], test_act[episode, step], test_targets[episode, step]
            # concatenate new obs,action to history list
            obs_history.append(obs)
            act_history.append(action)
            obs_next_history.append(obs_next)
            n_steps = len(obs_history)
            
            # NOTE: MTS3 can only predict step >= 2H: padded history list
            if step < 2*mts3_model.H:
                continue # continue collecting (obs,action) from the environment; cannot generate prediction yet
            else:
                # input the whole obs & action history
                obs = torch.Tensor(np.array(obs_history)).to(mts3_model._device)
                action = torch.Tensor(np.array(act_history)).to(mts3_model._device)
                obs_next = torch.Tensor(np.array(obs_next_history)).to(mts3_model._device)
                assert action[:,4:].sum() == 0 # NOTE: issue: action padding from multitask training
                
                # mts3 implementation: validity flag to mask out future observations in batch
                # -> we don't need this validity flag for step-by-step observation
                obs_valid = torch.Tensor([True for _ in range(n_steps)])
                obs_valid = obs_valid.bool().to(mts3_model._device)

                # mts3 prediction: STACK PAST OBSERVATIONS
                out_mean, out_var, mu_l_prior, cov_l_prior, mu_l_post, cov_l_post, act_abs = mts3_model(obs.reshape((1,n_steps,-1)), 
                                                                                                        action.reshape((1,n_steps,-1)), 
                                                                                                        obs_valid.reshape((1,n_steps,-1)))
                pred_next = out_mean[:,-1,:].detach().cpu()
                pred_history.append(pred_next)

        print(obs.shape, action.shape, obs_next.shape)     
        print(obs.reshape((1,n_steps,-1)).repeat(512,1,1).shape, 
              action.reshape((1,n_steps,-1)).repeat(512,1,1).shape, 
              obs_valid.reshape((1,n_steps,-1)).repeat(512,1,1).shape)   
        
        """ Evaluation """
        obs_next_history = torch.vstack(obs_next_history[2*mts3_model.H:]) # exclude padded 0.'s
        pred_history = torch.vstack(pred_history) # exclude padded 0.'s
        print(f'Episode {episode}: rmse =', torch.sqrt(((obs_next_history-pred_history)**2).mean()))
        print(f'Episode {episode}: mean(obs) =', (obs_next_history-pred_history).mean())


""" ====================================================
        Evaluate model using online interaction 
====================================================="""
@hydra.main(config_name='config', config_path='/nfshomes/anhu/world_models_diff_envs/tdmpc2/tdmpc2/')
def eval_online(cfg: dict)->OmegaConf:
    global tdmpc2
    cfg = parse_cfg(cfg)
    cfg.multitask = False
    cfg.multi_dt = False # use non time-aware model to use its encoder -> better for MTS3
    cfg.num_pi_trajs = 0
    cfg.eval_steps_adjusted = True # adjust stepping dt_eval/dt_train times for fair evaluation
    
    for task in tasks:
        cfg.task = task
        env = make_env(cfg)
        cfg.default_dt = env.env.default_dt
        print(colored(f'{cfg.task}\'s Default Δt:', 'green'), cfg.default_dt)

        # Initialize tdmpc2 model & load trained weights
        tdmpc2 = TDMPC2(cfg)
        tdmpc2.model = tdmpc2.model.to(mts3_model._device)
        tdmpc2.load(tdmpc2_weights)
        tdmpc2.model.eval()

        """ Evaluate on different observation rates """
        # for dt in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05]:
        for dt in eval_dts:
            print(colored('Evaluating on eval Δt:', 'green'), dt)
            dt = round(dt, 4)
            tdmpc2.cfg.eval_dt = dt

            """
                NOTE: MTS3 can only predict step >= 2H: padded history list
                    -> when initialize interactive env, set env.max_episodes = 2*H + env.max_episodes
                    -> use the first 2*H steps to for the model to accumulate (obs,action):
                        step <= 2*H: use default Δt
                        else: use new Δt
                    OR:
                    -> keep env.max_episodes: just do nothing in the first 2*H steps
            """
            ep_successes = []
            ep_rewards = []
            rows = []
            for episode in range(10):
                obs_history = []
                act_history = []
                obs_next_history = []
                pred_history = []
                
                obs, ep_reward, done, t = env.reset(), 0, False, -1
                obs_history.append(obs)
                
                for step in tqdm(range(100+2*mts3_model.H), desc=f'Episode {episode}'):
                    # step <  2H: do nothing (taking zero-value actions)
                    # step >= 2H: action ~ MTS3+MPPI
                    if step < 2*mts3_model.H:
                        # eval step 1 -> 2*H: do nothing (action 0) with default dt stepping
                        #   => required for MTS3
                        action = torch.zeros((env.action_space.shape))
                        obs, reward, done, info = env.env.step_adaptive_dt(action, cfg.default_dt)
                    else:
                        # action = ... # use MTS3 + MPPI: 1st planning step is 2*H (not 0): take eval dt stepping
                        action = plan(obs_history, act_history, t0=step==2*mts3_model.H, eval_mode=True, timestep=dt)
                        obs, reward, done, info = env.env.step_adaptive_dt(action, dt)
                    
                    # take action 
                    ep_reward += reward
                    # MTS3 requires input as the whole historical
                    obs_history.append(obs.detach().cpu())
                    act_history.append(action.detach().cpu())

                # monitor success state at the end of episode
                ep_successes.append(info['success'])
                ep_rewards.append(ep_reward)
                print('\tSuccess state:', info['success'], '| ep_reward:', ep_reward)

                rows.append({'Timestep': dt, 
                            'reward': float(ep_reward.detach().cpu().numpy()), 
                            'Success': info['success']})
                
                df_eval = pd.DataFrame(rows)
                df_eval.to_csv(f'{cwd}/logs/MTS3-{mts3_H}-{task}-multidt-{dt}.csv', index=False)
            
            print(colored(f'\tAvg success rate on env dt={dt}:', 'green'), np.array(ep_successes).mean())
            print(colored(f'\tAvg reward rate on env dt={dt}:', 'green'), np.array(ep_rewards).mean())

""" 
    Planner for MTS3: o -> MTS3's dynamic -> o' -> TDMPC2's encoder -> s' => planner
    Implementation: change all "self" to "tdmpc2"
"""
@torch.no_grad()
def plan(obs_history, act_history, t0=False, eval_mode=False, task=None, timestep=None):
    """
    Plan a sequence of actions using the learned world model.
    
    Args:
        obs_history (list): historical observation trajectory -> required for MTS3
        act_history (list): historical action trajectory -> required for MTS3
        t0 (bool): Whether this is the first observation in the episode.
        eval_mode (bool): Whether to use the mean of the action distribution.
        task (Torch.Tensor): Task index (only used for multi-task experiments).

    Returns:
        torch.Tensor: Action to take in the environment.
    """
    # Sample policy trajectories
    dt = timestep

    # Initialize state and parameters
    # z = tdmpc2.model.encode(obs, task, timestep=dt)
    # z = z.repeat(tdmpc2.cfg.num_samples, 1)
    mean = torch.zeros(tdmpc2.cfg.horizon, tdmpc2.cfg.action_dim, device=tdmpc2.device)
    std = tdmpc2.cfg.max_std*torch.ones(tdmpc2.cfg.horizon, tdmpc2.cfg.action_dim, device=tdmpc2.device)
    if not t0:
        mean[:-1] = tdmpc2._prev_mean[1:]
    actions = torch.empty(tdmpc2.cfg.horizon, tdmpc2.cfg.num_samples, tdmpc2.cfg.action_dim, device=tdmpc2.device)

    # Iterate MPPI
    for _ in range(tdmpc2.cfg.iterations):

        # Sample actions
        actions[:, tdmpc2.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
            torch.randn(tdmpc2.cfg.horizon, tdmpc2.cfg.num_samples-tdmpc2.cfg.num_pi_trajs, tdmpc2.cfg.action_dim, device=std.device)) \
            .clamp(-1, 1)
        if tdmpc2.cfg.multitask:
            actions = actions * tdmpc2.model._action_masks[task]

        # Compute elite actions
        value = _estimate_value(obs_history, act_history, act_rollout=actions, 
                                task=task, timestep=dt).nan_to_num_(0)
        elite_idxs = torch.topk(value.squeeze(1), tdmpc2.cfg.num_elites, dim=0).indices
        elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

        # Update parameters
        max_value = elite_value.max(0)[0]
        score = torch.exp(tdmpc2.cfg.temperature*(elite_value - max_value))
        score /= score.sum(0)
        mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
        std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
            .clamp_(tdmpc2.cfg.min_std, tdmpc2.cfg.max_std)
        if tdmpc2.cfg.multitask:
            mean = mean * tdmpc2.model._action_masks[task]
            std = std * tdmpc2.model._action_masks[task]

    # Select action
    score = score.squeeze(1).cpu().numpy()
    actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
    tdmpc2._prev_mean = mean
    a, std = actions[0], std[0]
    if not eval_mode:
        a += std * torch.randn(tdmpc2.cfg.action_dim, device=std.device)
    return a.clamp_(-1, 1)


""" Estimate value of a trajectory starting at latent state z and executing given actions. 
    NOTE: MTS3 requires access to historical trajectories of (obs,action) for inference

    :param obs_history: trajectory past & current observations obs[0:t]
    :param act_history: trajectory past & current action act[0:t]
    :param act_rollout: new rollout actions to be imagined
    :param task: task id (multitask settings)
    :param timestep: eval Δt

    :return expected value of current obs = obs_history[-1]
"""
@torch.no_grad()
def _estimate_value(obs_history, act_history, act_rollout, task, timestep):
    G, discount = 0, 1
    dt = timestep

    """ MTS3 requires list of (obs, action) history to predict"""
    obs_history = torch.stack(obs_history.copy())
    act_history = torch.stack(act_history.copy())

    """ Duplicate (T,D) -> (N_trajectories, T,D): roll out N_trajectories """
    obs_history = obs_history.repeat(tdmpc2.cfg.num_samples, 1, 1).to(tdmpc2.device)
    act_history = act_history.repeat(tdmpc2.cfg.num_samples, 1, 1).to(tdmpc2.device)
    
    assert obs_history.shape[1] == act_history.shape[1]+1 # current action not taken yet

    for t in range(tdmpc2.cfg.horizon):
        # "Imagine" immediate reward & next state under trandition Delta t = eval_dt
        if (tdmpc2.cfg.multi_dt) or (not tdmpc2.cfg.eval_steps_adjusted):
            n_steps = obs_history.shape[1]
            
            # Case 1: One-step evaluation (no adjustment) -> either time-aware model or no evaluation adjustment
            # encode observation using tdmpc2's encoder
            obs = obs_history[:,-1,:] # current obs
            z = tdmpc2.model.encode(obs, task=task, timestep=dt)
            # print(colored('HERE', 'blue'), obs.shape, z.shape, act_rollout[t].shape)
            reward = math.two_hot_inv(tdmpc2.model.reward(z, act_rollout[t], task, timestep), tdmpc2.cfg)
            
            """ NOTE: Modification HERE: change TDMPC2's dynamic model with MTS3"""
            # z = tdmpc2.model.next(z, actions[t], task, timestep=dt)
            # z = self.model.next(z, actions[t], task, timestep=dt)

            # add current rollout action to action history (shape N_traj x T x D) -> concat dim=1 (T)
            act_history = torch.cat([act_history, torch.permute(act_rollout[[t]], (1,0,2))], dim=1)
            # mts3 implementation: validity flag to mask out future observations in batch
            # -> we don't need this validity flag for step-by-step observation
            obs_valid = torch.Tensor([True]).bool().to(tdmpc2.device)
            obs_valid = obs_valid.repeat(tdmpc2.cfg.num_samples, n_steps, 1)
            # print(colored('HERE', 'blue'), obs_history.shape, act_history.shape, obs_valid.shape)

            """ IMPORTANT CORRECTION: pad 0's to action space of mts3: 
                -> mts3 was trained on data collected for tdmpc2 multitask => action padding
                -> this is done to save data collection time, but I wasn't aware of this
                TODO: collect new dataset to train MTS3
            """
            zero_tensor = torch.zeros((act_history.shape[0], act_history.shape[1], 2)).to(tdmpc2.device)
            act_history_corrected = torch.cat([act_history, zero_tensor], dim=-1)
            # print(colored('HERE', 'blue'), obs_history.shape, act_history_corrected.shape, obs_valid.shape)
            out_mean, out_var, mu_l_prior, cov_l_prior, mu_l_post, cov_l_post, act_abs = mts3_model(obs_history, act_history_corrected, obs_valid)
            # add current rollout action to action history (shape N_traj x T x D) -> concat dim=1 (T)
            # print(colored('HERE', 'blue'), out_mean.shape)
            obs_next = out_mean[:,[-1],:]
            obs_history = torch.cat([obs_history, obs_next], dim=1)
            obs = obs_next
        else:
            # Case 2: Adjusted multi-step evaluation -> must be non-time-aware model (baseline)
            train_dt = tdmpc2.cfg.train_dt if (tdmpc2.cfg.train_dt is not None) else tdmpc2.cfg.default_dt
            eval_dt = tdmpc2.cfg.eval_dt
            pred_steps = int(np.ceil(eval_dt/train_dt)) # adjusted number of prediction steps (#times the model is applied)
            
            # Case 1: One-step evaluation (no adjustment) -> either time-aware model or no evaluation adjustment
            # encode observation using tdmpc2's encoder
            obs = obs_history[:,-1,:] # current obs
            # z = tdmpc2.model.encode(obs, task=task, timestep=dt) if (t==0) else z # this is for testing tdmpc2's dynamic model
            z = tdmpc2.model.encode(obs, task=task, timestep=dt)
            # print(colored('HERE', 'blue'), obs.shape, z.shape, act_rollout[t].shape)
            reward = math.two_hot_inv(tdmpc2.model.reward(z, act_rollout[t], task, timestep), tdmpc2.cfg)
            
            for _ in range(pred_steps):
                n_steps = obs_history.shape[1]

                """ NOTE: Modification HERE: change TDMPC2's dynamic model with MTS3"""
                # z = tdmpc2.model.next(z, act_rollout[t], task, timestep=dt)
                # z = self.model.next(z, act_rollout[t], task, timestep=dt)

                # add current rollout action to action history (shape N_traj x T x D) -> concat dim=1 (T)
                act_history = torch.cat([act_history, torch.permute(act_rollout[[t]], (1,0,2))], dim=1)
                # mts3 implementation: validity flag to mask out future observations in batch
                # -> we don't need this validity flag for step-by-step observation
                obs_valid = torch.Tensor([True]).bool().to(tdmpc2.device)
                obs_valid = obs_valid.repeat(tdmpc2.cfg.num_samples, n_steps, 1)
                # print(colored('HERE', 'blue'), obs_history.shape, act_history.shape, obs_valid.shape)

                """ IMPORTANT CORRECTION: pad 0's to action space of mts3: 
                    -> mts3 was trained on data collected for tdmpc2 multitask => action padding
                    -> this is done to save data collection time, but I wasn't aware of this
                    TODO: collect new dataset to train MTS3
                """
                zero_tensor = torch.zeros((act_history.shape[0], act_history.shape[1], 2)).to(tdmpc2.device)
                act_history_corrected = torch.cat([act_history, zero_tensor], dim=-1)
                # print(colored('HERE', 'blue'), obs_history.shape, act_history_corrected.shape, obs_valid.shape)
                out_mean, out_var, mu_l_prior, cov_l_prior, mu_l_post, cov_l_post, act_abs = mts3_model(obs_history, act_history_corrected, obs_valid)
                # add current rollout action to action history (shape N_traj x T x D) -> concat dim=1 (T)
                # print(colored('HERE', 'blue'), out_mean.shape)
                obs_next = out_mean[:,[-1],:]
                obs_history = torch.cat([obs_history, obs_next], dim=1)
                obs = obs_next

        # print('_estimate_value:', 'z', z.shape, 'action', actions[t].shape, 'dt', dt.shape)
        G += discount * reward
        discount *= tdmpc2.discount[torch.tensor(task)] if tdmpc2.cfg.multitask else tdmpc2.discount
    return G + discount * tdmpc2.model.Q(z, tdmpc2.model.pi(z, task, timestep=dt)[1], task, return_type='avg', timestep=dt)

if __name__ == '__main__':
    # eval_offline()
    eval()