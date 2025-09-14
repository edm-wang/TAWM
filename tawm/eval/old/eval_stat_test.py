import numpy as np  # type: ignore
import pandas as pd # type: ignore
import scipy        # type: ignore
from scipy import stats # type: ignore
from scipy.stats import ttest_ind, binomtest, fisher_exact, chi2_contingency # type: ignore
from collections import defaultdict
import os
from termcolor import colored # type: ignore


tasks = ['mw-assembly', 'mw-basketball', 'mw-box-close', 
         'mw-faucet-open', 'mw-hammer', 'mw-lever-pull']
eval_dts = [0.001, 0.0025, 0.01, 0.02, 0.03, 0.05]

# tasks = ['pde-allen_cahn', 'pde-burgers', 'pde-wave']
# eval_dts = [0.01, 0.05, 0.1, 0.5, 1.0]

for task in tasks:
    print(colored(task, 'green'))

    """ performance metric"""
    metric = 'Success' if (task[:2] == 'mw') else 'Reward'
    """ model performance by eval dt"""
    metric_eval_dt_1 = defaultdict(list)
    metric_eval_dt_2 = defaultdict(list)
    metric_eval_dt_3 = defaultdict(list)

    
    """ collect each model performance by time"""
    for seed in [1,2,3]:
        log_rk4_loguniform = f'logs/{task}/eval_multidt_timeaware_{seed}.csv'
        log_euler_loguniform = f'logs/{task}/eval_multidt_timeaware_euler_{seed}.csv'
        log_rk4_uniform = f'logs/{task}/eval_multidt_timeaware_uniform_{seed}.csv'
        
        # 1. collect eval results for model 1 (rk4 + log uniform sampling)
        if os.path.exists(log_rk4_loguniform):
            df = pd.read_csv(log_rk4_loguniform)
            for dt in eval_dts:
                # print(eval_dt)
                metric_eval_dt_1[dt] += list(df[df['Timestep'] == dt][metric])

        # 2. collect eval results for model 1 (rk4 + log uniform sampling)
        if os.path.exists(log_euler_loguniform):
            df = pd.read_csv(log_euler_loguniform)
            for dt in eval_dts:
                # print(eval_dt)
                metric_eval_dt_2[dt] += list(df[df['Timestep'] == dt][metric])

        # 3. collect eval results for model 1 (rk4 + log uniform sampling)
        if os.path.exists(log_rk4_uniform):
            df = pd.read_csv(log_rk4_uniform)
            for dt in eval_dts:
                # print(eval_dt)
                metric_eval_dt_3[dt] += list(df[df['Timestep'] == dt][metric])
    
    # print(metric_eval_dt_1)
    # print(metric_eval_dt_2)
    # print(metric_eval_dt_3)

    """ Statistical test of performance"""
    """     1. models: euler vs rk4"""
    print(colored('Euler vs RK4', 'blue'))
    for dt in eval_dts:
        if metric == 'Success':
            # Count successes and failures for each model
            a_success = sum(metric_eval_dt_3[dt])
            a_failure = len(metric_eval_dt_3[dt]) - a_success
            b_success = sum(metric_eval_dt_1[dt])
            b_failure = len(metric_eval_dt_1[dt]) - b_success
            # Create the 2x2 contingency table
            contingency_table = np.array([[a_success, a_failure], 
                                        [b_success, b_failure]])
            # Haldane-Anscombe correction
            contingency_table += 1
            # fisher exact test
            test_res = stats.fisher_exact(contingency_table, alternative='greater')
        else:
            # pair-wise t-test
            test_res = ttest_ind(metric_eval_dt_2[dt], metric_eval_dt_1[dt])
        
        stat, p_val = test_res[:2]
        stat, p_val = round(stat, 2), round(p_val, 2)
        print(colored(f'\tdt = {dt}:', 'blue'), f'stats = {stat}; p-value = {p_val}')

    """     2. models: uniform vs log-uniform"""
    print(colored('Uniform vs Log-Uniform', 'blue'))
    for dt in eval_dts:
        if metric == 'Success':
            # Count successes and failures for each model
            a_success = sum(metric_eval_dt_3[dt])
            a_failure = len(metric_eval_dt_3[dt]) - a_success
            b_success = sum(metric_eval_dt_1[dt])
            b_failure = len(metric_eval_dt_1[dt]) - b_success
            # Create the 2x2 contingency table
            contingency_table = np.array([[a_success, a_failure], 
                                        [b_success, b_failure]])
            # Haldane-Anscombe correction
            contingency_table += 1
            # fisher exact test
            test_res = stats.fisher_exact(contingency_table, alternative='greater')
        else:
            # pair-wise t-test
            test_res = ttest_ind(metric_eval_dt_3[dt], metric_eval_dt_1[dt])
        
        stat, p_val = test_res[:2]
        stat, p_val = round(stat, 2), round(p_val, 2)
        print(colored(f'\tdt = {dt}:', 'blue'), f'stats = {stat}; p-value = {p_val}')
    print()