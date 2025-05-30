"""
    Generate:
    (1) Each task: Reward curve (time-aware model vs baselines)
    (2) Each task: Performance-by-dt (time-aware model vs baselines)
"""

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
plt.rcParams["font.family"] = "serif"

import hydra # type: ignore
from termcolor import colored # type: ignore
import warnings
warnings.filterwarnings('ignore')
from envs import make_env
from common import TASK_SET

import os

cwd = os.getcwd()
homedir = '/nfshomes/anhu'
model_dir = '/fs/nexus-scratch/anhu/world-model-checkpoints'
logdir_base = f'{cwd}/../results'
logdir_tawm = f'{cwd}/logs'

# tasks = ['mw-assembly', 'mw-basketball', 'mw-box-close', 
#          'mw-faucet-open', 'mw-hammer', 'mw-handle-pull',  
#          'mw-lever-pull', 'mw-pick-out-of-hole', 'mw-sweep-into']
# tasks = ['mw-assembly', 'mw-basketball', 'mw-box-close', 
#          'mw-faucet-open', 'mw-hammer', 'mw-handle-pull']
# tasks = ['mw-assembly', 'mw-basketball', 'mw-box-close']
# tasks = ['mw-lever-pull', 'mw-pick-out-of-hole', 'mw-sweep-into']
# tasks = ['mw-faucet-open', 'mw-hammer', 'mw-lever-pull']
# tasks = ['mw-faucet-open']
# tasks = ['mw-basketball']
# eval_dts = [0.001, 0.0025, 0.01, 0.02, 0.03, 0.05]

# tasks = ['mw-assembly']
# eval_dts = [0.0025]

# tasks = ['pygame-flappybird']
# eval_dts = [0.01, 0.0333, 0.05, 0.1, 0.2]

tasks = ['pde-allen_cahn', 'pde-burgers', 'pde-wave']
# tasks = ['pde-allen_cahn']
eval_dts = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

""" ==================================================
    PLOT REWARD CURVE FOR ALL TASKS FOR EACH eval_dt (simulation timestep)
    
    --------------------------------------------------------------------------
    assumptions:
        Return curve results are saved at 
            `{cwd}/logs/{task}/return_curve_{model_type}.csv`,
         where:
            - model_type = 'baseline' or 'timeaware'
            - eval_dt    = evaluation time step size (simulation time step size)

    --------------------------------------------------------------------------
    save_dir: `{cwd}/plots/return_curve_{task}_{eval_dt}.png`
     
     where:
        - model_type = 'baseline' or 'timeaware'
        - eval_dt    = evaluation time step size (simulation time step size)
=================================================== """
# @hydra.main(config_name='config', config_path='.')
def plot_return_curves():
    for task in tasks:
        for eval_dt in eval_dts:
            print('Saving reward curves for', colored(f'{task}', 'green'), 
                  'at', colored(f'`eval_dt={eval_dt}`', 'green'), 
                  'saving at', colored(f'`plots/return_curve_{task}_{eval_dt}.png`', 'green'))
            
            result_file_baseline = f'{cwd}/logs/{task}/return_curve_baseline_eval_dt={eval_dt}.csv'
            result_file_timeaware = f'{cwd}/logs/{task}/return_curve_timeaware_eval_dt={eval_dt}.csv'
            result_file_timeaware_euler = f'{cwd}/logs/{task}/return_curve_timeaware-euler_eval_dt={eval_dt}.csv'
            result_file_plot_curve = f'plots/return_curve_{task}_{eval_dt}.png'
            
            # """ 1.  load saved learning curves for baseline (non-time-aware model) 
            #         @ `{cwd}/logs/{task}/return_curve_baseline_eval_dt={eval_dt}.csv`
            #         -> results trained on fixed, default dt=2.5ms (all seeds)
            # """
            # # baseline results
            # df_base = pd.read_csv(result_file_baseline)
            # df_base = df_base[df_base['step'] < 1_500_000] # cut off extra last-step result

            # """ 2.  load saved learning curves for time-aware model
            #         @ `{cwd}/logs/{task}/return_curve_timeaware_eval_dt={eval_dt}.csv`
            #         -> results trained on varying dt's
            # """
            # # new results for all seeds
            # df_tawm = pd.read_csv(result_file_timeaware)
            # df_tawn = df_tawm[df_tawn['step'] < 1_500_000] # cut off extra last-step result
            
            """ 1.  load saved learning curves for baseline (non-time-aware model) 
                        @ `{cwd}/logs/{task}/return_curve_baseline_eval_dt={eval_dt}.csv`
                        -> results trained on fixed, default dt=2.5ms (all seeds)
            """
            if eval_dt != 0.0025: # default time step size
                result_file_baseline = f'{cwd}/logs/{task}/return_curve_baseline_eval_dt={eval_dt}.csv'
                result_file_plot_curve = f'plots/return_curve_{task}_{eval_dt}.png'
                
                # baseline results
                df_base = pd.read_csv(result_file_baseline)
                df_base = df_base[df_base['step'] <= 1_500_400] # cut off extra last-step result
                df_base['step'] = df_base['step'] - df_base['step']%1000 # step: round to thounsand digits
            else: # use learning curves from eval.csv
                df_base = pd.read_csv(f'{logdir_base}/{task}.csv')
                df_base = df_base.rename(columns={'success': 'episode_success'})
                df_base = df_base[df_base['step'] <= 1_500_400]
                df_base['step'] = df_base['step'] - df_base['step']%1000 # step: round to thounsand digits
                result_file_plot_curve = f'plots/return_curve_{task}_{eval_dt}.png'

            """ 2.  load saved learning curves for time-aware model
                    @ `{cwd}/logs/{task}/return_curve_timeaware_eval_dt={eval_dt}.csv`
                    -> results trained on varying dt's
            """
            # new results for all seeds
            df_tawm = pd.read_csv(result_file_timeaware)
            df_tawm = df_tawm[df_tawm['step'] < 1_500_400] # cut off extra last-step result
            df_tawm['step'] = df_tawm['step'] - df_tawm['step']%1000 # step: round to thounsand digits

            """ 3.  load saved learning curves for time-aware euler model
                    @ `{cwd}/logs/{task}/return_curve_timeaware-euler_eval_dt={eval_dt}.csv`
                    -> results trained on varying dt's
            """
            # new results for all seeds
            if os.path.exists(result_file_timeaware_euler):
                df_euler = pd.read_csv(result_file_timeaware_euler)
                df_euler = df_euler[df_euler['step'] < 1_500_400] # cut off extra last-step result
                df_euler['step'] = df_euler['step'] - df_euler['step']%1000 # step: round to thounsand digits
            else:
                df_euler = None
            
            
            
            
            """ 2. compute mean curve & max/min shades """
            metric = 'episode_success' if task[:2] == 'mw' else 'episode_reward'
            
            if metric =='episode_success':
                if df_base is not None:
                    eval_base = df_base.groupby('step').agg(
                            avg_metric=(metric, 'mean'),
                            lower_ci=(metric, lambda x: confidence_interval_from_success_ratio(x, eval_episodes=10)[0]),
                            upper_ci=(metric, lambda x: confidence_interval_from_success_ratio(x, eval_episodes=10)[1])
                    ).reset_index()

                if df_tawm is not None:
                    eval_tawm = df_tawm.groupby('step').agg(
                            avg_metric=(metric, 'mean'),
                            lower_ci=(metric, lambda x: confidence_interval_from_success_ratio(x, eval_episodes=10)[0]),
                            upper_ci=(metric, lambda x: confidence_interval_from_success_ratio(x, eval_episodes=10)[1])
                    ).reset_index()

                if df_euler is not None:
                    eval_euler = df_euler.groupby('step').agg(
                            avg_metric=(metric, 'mean'),
                            lower_ci=(metric, lambda x: confidence_interval_from_success_ratio(x, eval_episodes=10)[0]),
                            upper_ci=(metric, lambda x: confidence_interval_from_success_ratio(x, eval_episodes=10)[1])
                    ).reset_index()
            elif metric == 'episode_reward':
                if df_base is not None:
                    eval_base = df_base.groupby('step').agg(
                            avg_metric=(metric, 'mean'),
                            lower_ci=(metric, lambda x: confidence_interval(x)[0]),
                            upper_ci=(metric, lambda x: confidence_interval(x)[1])
                    ).reset_index()
                
                if df_tawm is not None:
                    eval_tawm = df_tawm.groupby('step').agg(
                            avg_metric=(metric, 'mean'),
                            lower_ci=(metric, lambda x: confidence_interval(x)[0]),
                            upper_ci=(metric, lambda x: confidence_interval(x)[1])
                    ).reset_index()

                if df_euler is not None:
                    eval_euler = df_euler.groupby('step').agg(
                            avg_metric=(metric, 'mean'),
                            lower_ci=(metric, lambda x: confidence_interval(x)[0]),
                            upper_ci=(metric, lambda x: confidence_interval(x)[1])
                    ).reset_index()
            else:
                raise NotImplementedError

            # retrieve corresponding training steps for plots
            base_steps = df_base[df_base['seed'] == 1]['step']
            tawm_steps = df_tawm[df_tawm['seed'] == 1]['step']
            # compare same training steps for fair comparison
            n_steps = min(max(base_steps), max(tawm_steps))
            if df_euler is not None:
                euler_steps = df_euler[df_euler['seed'] == 1]['step']
                n_steps = min(n_steps, max(euler_steps))

            # extract same number of training steps
            base_steps = base_steps[base_steps <= n_steps]
            tawm_steps = tawm_steps[tawm_steps <= n_steps]
            eval_base = eval_base[eval_base['step'] <= n_steps]
            eval_tawm = eval_tawm[eval_tawm['step'] <= n_steps]

            if df_euler is not None:
                euler_steps = euler_steps[euler_steps <= n_steps]
                eval_euler = eval_euler[eval_euler['step'] <= n_steps]

            # # ignore 1st step because PDE has very low LQ Error at step 0
            # if task[:3] == 'pde':
            #     base_steps = base_steps[1:]
            #     tawm_steps = tawm_steps[1:]
            #     eval_base = eval_base.iloc[1:]
            #     eval_tawm = eval_tawm.iloc[1:]

            """ 3. plot curves """
            os.system(f'mkdir -p {cwd}/plots')
            plt.figure(figsize=(8,5))
            if eval_dt == eval_dts[0]: # only plot title on first eval_dt for writing purpose
                plt.title(f'{task}', fontsize=24)
            # a. plot baseline learning curve
            plt.plot(base_steps, eval_base['avg_metric'], linewidth=4, color='blue')
            plt.fill_between(base_steps, eval_base['lower_ci'], eval_base['upper_ci'], alpha=0.2, color='blue')
            # b. plot tawm learning curve
            plt.plot(tawm_steps, eval_tawm['avg_metric'], linewidth=4, color='red')
            plt.fill_between(tawm_steps, eval_tawm['lower_ci'], eval_tawm['upper_ci'], alpha=0.2, color='red')
            # c. plot euler learning curve
            if df_euler is not None:
                plt.plot(euler_steps, eval_euler['avg_metric'], linewidth=4, color='maroon')
                plt.fill_between(euler_steps, eval_euler['lower_ci'], eval_euler['upper_ci'], alpha=0.2, color='maroon')
            # limit y axis of PDE control for easier visualization
            if task[:3] == 'pde':
                ymin = min(eval_base['lower_ci'][1:].min(),  eval_tawm['lower_ci'][1:].min())
                ymax = max(eval_base['upper_ci'].max(),  eval_tawm['upper_ci'].max())
                if df_euler is not None:
                    ymin = min(ymin, eval_euler['lower_ci'][1:].min())
                    ymax = max(ymax, eval_euler['upper_ci'].max())
                ymax = ymax + (ymax-ymin)*0.1
                plt.ylim([ymin, ymax])
            # compute xticks (training steps)
            xticks = [0, 500_000, 1_000_000, 1_500_000, 2_000_000]
            xticks_labels = ['0', '0.5M', '1M', '1.5M', '2M']
            # adjust the maximum number of steps based on trained steps
            xticks = [e for e in xticks if e <= base_steps.max()]
            xticks_labels = xticks_labels[:len(xticks)]
            plt.xticks(xticks, xticks_labels, fontsize=20)
            plt.yticks(fontsize=20)
            # if task[:3] == 'pde':
            #     plt.yscale('symlog')
            plt.grid(True)

            # compute legend position on graph
            xpos_legend = 0.5 * base_steps.max()
            if task[:3] == 'pde':
                # # ignore 1st step because PDE has very low LQ Error at step 0
                ymin = min(eval_tawm['avg_metric'][1:].min(), eval_base['avg_metric'][1:].min())
                ymax = max(eval_tawm['avg_metric'][1:].max(), eval_base['avg_metric'][1:].max())
                if eval_euler is not None:
                    ymin = min(ymin, eval_euler['avg_metric'][1:].min())
                    ymax = max(ymax, eval_euler['avg_metric'][1:].max())
            else:
                ymin = min(eval_tawm['avg_metric'].min(), eval_base['avg_metric'].min())
                ymax = max(eval_tawm['avg_metric'].max(), eval_base['avg_metric'].max())
                if eval_euler is not None:
                    ymin = min(ymin, eval_euler['avg_metric'].min())
                    ymax = max(ymax, eval_euler['avg_metric'].max())
            ypos_legend = ymin - 0.05*(ymax-ymin)
            plt.text(xpos_legend, ypos_legend, f'Eval $\Delta t$ = {eval_dt*1000}ms', fontsize=22, color='black', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
            # save figure
            plt.savefig(result_file_plot_curve, bbox_inches='tight', pad_inches=0.1)
            plt.close()

""" ==================================================
    PLOT REWARD CURVE FOR ALL TASKS
    save_dir:
        case 1: `{cwd}/plots/timeaware-{task}.png`
        -> comparison with non-time-aware model 
            trained on fixed, default dt
        case 2: `{cwd}/plots/timeaware-{task}-all.png`
        -> comparison with non-time-aware model 
            trained on different fixed dt's
=================================================== """
@hydra.main(config_name='config', config_path='.', version_base=None)
def plot_time_performance(cfg: dict):
    for task in tasks:
        # Retrieve default timestep
        cfg.task = task
        cfg.multitask = False
        env = make_env(cfg)
        if task[:2] == 'mw':
            default_dt = env.env.env.sim.model.opt.timestep
        elif task in TASK_SET['dmcontrol']:
            default_dt = env.env.env.physics.model.opt.timestep
        elif task[:6] == 'pygame':
            default_dt = env.default_dt
        elif task[:3] == 'pde':
            default_dt = env.default_dt
        else:
            raise Exception(f'Need to implement timestep retriever for {task}.')

        """ Iterate through model pair for pair-wise comparison
            
            load saved Performance-by-dt curves for model (model \in ['timeaware', 'baseline_0.001', 'baseline', 'baseline_0.01', 'baseline_0.05'])
                @ `logs/{task}/eval_multidt_{model}_{seed}.csv`
            
            --------------------------------------------------------
            Input: Evaluation Results
            -> for each {task} & {seed}:
                    (1A) logs/{task}/eval_multidt_timeaware_{seed}.csv              : time-aware model (trained on log-uniform sampling)
                    (1B) logs/{task}/eval_multidt_timeaware_{dt_sampler}_{seed}.csv : time-aware model (trained on {dt_sampler} sampling strategy: uniform or others)

                    (2) logs/{task}/eval_multidt_baseline_0.001_{seed}.csv          : non-time-aware model trained only on dt=1ms
                    (3) logs/{task}/eval_multidt_baseline_0.001_adjusted_{seed}.csv : non-time-aware model trained only on dt=1ms; adjusted inference step

                    (4) logs/{task}/eval_multidt_baseline_{seed}.csv                : non-time-aware model trained only on dt=2.5ms (default)
                    (5) logs/{task}/eval_multidt_baseline_adjusted_{seed}.csv       : non-time-aware model trained only on dt=2.5ms (default); adjusted inference step

                    (6) logs/{task}/eval_multidt_baseline_0.01_{seed}.csv           : non-time-aware model trained only on dt=10ms
                    (7) logs/{task}/eval_multidt_baseline_0.01_adjusted_{seed}.csv  : non-time-aware model trained only on dt=10ms; adjusted inference step

                    (8) logs/{task}/eval_multidt_baseline_0.05_{seed}.csv           : non-time-aware model trained only on dt=50ms
                    (9) logs/{task}/eval_multidt_baseline_0.05_adjusted_{seed}.csv  : non-time-aware model trained only on dt=50ms; adjusted inference step

                    (10) ../../MTS3/logs/MTS3-H{mts3.H}-{task}-multidt.csv          : MTS3 model (H = SSM_slow discretion factor) trained on offline dataset with default dt
            --------------------------------------------------------
            Output: Plots
            -> saved at:
                (a) `plots/time-aware-{task}.png`: time-aware vs non-time-aware (trained on default dt)
                    OR
                (b) `plots/time-aware-{task}-all.png`: time-aware vs ALL non-time-aware (trained on different fixed dt's)
        """
        for i, model_pair in enumerate([['timeaware', 'baseline', 'baseline_adjusted'], 
                                        ['timeaware', 'baseline_0.001', 'baseline', 'baseline_0.01', 'baseline_0.05'],
                                        ['timeaware', 'timeaware_uniform', 'baseline'], 
                                        ['timeaware', 'MTS3-H3', 'MTS3-H11', 'MTS3-H33', 'MTS3-H50'], 
                                        ['timeaware', 'timeaware_euler', 'baseline', 'baseline_adjusted']]):
            # Set figure name 
            if i == 0:
                continue
                savefig_name = f'{cwd}/plots/timeaware-{task}.png'
                print('Saving performance-by-delta-t curves for', colored(f'{task}', 'green'), 'at', colored(f'plots/timeaware-{task}.png', 'blue'))
            if i == 1:
                continue
                savefig_name = f'{cwd}/plots/timeaware-{task}-all.png'
                print('Saving performance-by-delta-t curves for', colored(f'{task}', 'green'), 'at', colored(f'plots/timeaware-{task}-all.png', 'blue'))
            if i == 2:
                continue
                savefig_name = f'{cwd}/plots/timeaware-{task}-sampling-strategy.png'
                print('Saving performance-by-delta-t curves for', colored(f'{task}', 'green'), 'at', colored(f'plots/timeaware-{task}-sampling-strategy.png', 'blue'))
            if i == 3:
                continue
                savefig_name = f'{cwd}/plots/timeaware-{task}-mts3.png'
                print('Saving performance-by-delta-t curves for', colored(f'{task}', 'green'), 'at', colored(f'plots/timeaware-{task}-mts3.png', 'blue'))
            if i == 4:
                savefig_name = f'{cwd}/plots/timeaware-{task}-integrator.png'
                print('Saving performance-by-delta-t curves for', colored(f'{task}', 'green'), 'at', colored(f'plots/timeaware-{task}-integrator.png', 'blue'))

            # Plotting functions
            plt.figure(figsize=(10,8))
            for model in model_pair:
                if 'MTS3' not in model :
                    df_eval = None
                    # 1. aggregate evaluation results across different seeds
                    for seed in [1,2,3]:
                        eval_dir = f'{cwd}/logs/{task}/eval_multidt_{model}_{seed}.csv'
                        if os.path.exists(eval_dir):
                            df = pd.read_csv(eval_dir)
                            df_eval = pd.concat([df_eval, df]) if df_eval is not None else df
                            del df

                    if df_eval is None:
                        break
                else:
                    df_eval = pd.read_csv(f'{homedir}/MTS3/logs/{model}-{task}-multidt.csv')
                    
                # 2. plot eval results for current task
                # Group by 'timestep' and calculate the mean and standard deviation of rewards
                metric = 'Success' if task[:2] == 'mw' else 'Reward'
                
                # * POST-PROCESSING eval metrics for some envs
                if task == 'pygame-flappybird':
                    # flappy bird: measure in # holes passed => offset terminating reward of -5
                    df_eval[metric] = df_eval[metric] + 5

                grouped_data = df_eval.groupby('Timestep').agg(
                        avg_metric=(metric, 'mean'),
                        lower_ci=(metric, lambda x: confidence_interval(x)[0]),
                        upper_ci=(metric, lambda x: confidence_interval(x)[1])
                ).reset_index()
                grouped_data['Timestep'] *= 1000 # convert eval_dt from s to ms


                # 3. Plot the average success rate / reward with shading for 95% C.I level
                if model == 'timeaware':
                    label = 'Time-aware (ours 1: RK4 Method)\n(training Δt~Log-Uniform(1,50)ms)' # if i==2 else 'Time-aware Model\n(training Δt~Log-Uniform(1,50)ms)'
                    color = 'r'
                    linestyle = '-'
                elif model == 'timeaware_uniform':
                    label = 'Time-aware (ours 2)\n(training Δt~Uniform(1,50)ms)'
                    color = 'darkviolet'
                    linestyle = '-'
                elif model == 'timeaware_euler':
                    label = 'Time-aware (ours 2: Euler Method)\n(training Δt~Log-Uniform(1,50)ms)'
                    color = 'maroon'
                    linestyle = '-'
                elif model == 'MTS3-H3':
                    label = 'MTS3 Model (Δt=2.5ms, $H=3$)'
                    color = 'violet'
                    linestyle = '--'
                elif model == 'MTS3-H11':
                    label = 'MTS3 Model (Δt=2.5ms, $H=11$)'
                    color = 'navy'
                    linestyle = '--'
                elif model == 'MTS3-H33':
                    label = 'MTS3 Model (Δt=2.5ms, $H=33$)'
                    color = 'darkgreen'
                    linestyle = '--'
                elif model == 'MTS3-H50':
                    label = 'MTS3 Model (Δt=2.5ms, $H=50$)'
                    color = 'darkgray'
                    linestyle = '--'
                elif model == 'baseline':
                    label = 'Non Time-aware (training $\Delta t$=2.5ms);\nOne-step Prediction' if i==0 else 'Non Time-aware (training $\Delta t$=2.5ms)'
                    color = 'b'
                    linestyle = '--'
                elif model == 'baseline_adjusted':
                    label = 'Non Time-aware (training $\Delta t$=2.5ms);\nAdjusted Multi-step Prediction' if i==0 else 'Non Time-aware (training $\Delta t$=2.5ms)'
                    color = 'purple'
                    linestyle = '-.'
                elif model == 'baseline_0.001':
                    label = 'Non Time-aware (training $\Delta t$=1ms)'
                    color = 'g'
                    linestyle = '--'
                elif model == 'baseline_0.01':
                    label = 'Non Time-aware (training $\Delta t$=10ms)'
                    color = 'darkorange'
                    linestyle = '-.'
                elif model == 'baseline_0.05':
                    label = 'Non Time-aware (training $\Delta t$=50ms)'
                    color = 'purple'
                    linestyle = ':'
                
                vmin, vmax = df_eval[metric].min(), df_eval[metric].max()
                plt.plot(grouped_data['Timestep'], grouped_data['avg_metric'], linewidth=4, linestyle=linestyle, label=label, color=color)
                plt.fill_between(grouped_data['Timestep'], 
                                grouped_data['lower_ci'].clip(vmin,vmax), 
                                grouped_data['upper_ci'].clip(vmin,vmax), 
                                color=color, alpha=0.1)
            
            # scale of total return/reward metric for different environments
            if task in TASK_SET['dmcontrol']:
                yticks = [n for n in np.arange(0,1100,100)]  
            elif task[:2] == 'mw': 
                yticks = [n for n in np.arange(0,1.2,0.2)]
            elif task == 'pygame-flappybird':
                yticks = [n for n in np.arange(0,30,5)]
            elif task[:3] == 'pde':
                y_low = grouped_data['lower_ci'].clip(vmin,vmax).min()
                y_high = 0.0 # for PDE ControlGym envs, max reward is 0. (no error)
                if task == 'pde-allen_cahn':
                    y_high = -210. # temporary for visualization purpose

                yticks = [n for n in np.arange(y_low, y_high, (y_high-y_low)/10)]
            else:
                raise Exception(f'Plotting for {task} is not implemented.')
            # for draft organization purpose
            plot_legend = False if (i==0 and task not in ['mw-lever-pull', 'mw-pick-out-of-hole', 'mw-sweep-into']) \
                                else True
            
            plt.title(task, fontsize=36)    
            plt.axvline(default_dt*1000, color='black', linestyle='--', linewidth=2)
            plt.yticks(yticks, fontsize=25)
            plt.xticks(fontsize=25)
            ncol = 1 # ncol = 2 if (i==0) else 1
            if plot_legend: 
                plt.legend(fontsize=25, bbox_to_anchor=(1.0, -0.05), ncol=ncol)
            plt.grid(True)
            plt.savefig(savefig_name, bbox_inches='tight', pad_inches=0.1)
            plt.close()

""" 
    Compute 95 CI level from raw values
"""
def confidence_interval(data):
    n = len(data)*10 # 10 eval episode per data point
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)  # Standard Error
    margin_of_error = 1.96 * std_err  # 95% confidence interval margin
    return (mean - margin_of_error, mean + margin_of_error)

""" 
    Compute 95 CI level from only summarized success ratio

    E.g:
        seed 1: 0.9 -> [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        seed 2: 1.0 -> [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        seed 3: 0.1 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"""
def confidence_interval_from_success_ratio(data, eval_episodes):
    # convert summarized success ratio to raw episode-wise success binary (0/1)
    raw_data = []
    for success_ratio in data:
        n_successes = int(success_ratio * eval_episodes)
        n_failures = eval_episodes - n_successes
        raw_data += [1.0 for _ in range(n_successes)]
        raw_data += [0.0 for _ in range(n_failures)]
    data = np.array(raw_data)

    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)  # Standard Error
    margin_of_error = 1.96 * std_err  # 95% confidence interval margin
    return (mean - margin_of_error, mean + margin_of_error)

if __name__ == '__main__':
    """ Plot success rate / reward curves (for all tasks) """
    # plot_return_curves() # comment out for quick experiment
    """ Plot success rate / reward across observation rates (for all tasks) """
    plot_time_performance() # comment out for quick experiment

    
