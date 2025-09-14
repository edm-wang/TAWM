"""
    Test model performance on various inference-time observation rates
    across different tasks
"""

import os
import re

cwd = os.path.dirname(__file__)

task_default_dt = {
    'pde-burgers': 0.05,
    'pde-wave': 0.1,
    'pde-allen_cahn': 0.01,
}

""" Retrieve the last model checkpoint from given model_path """
def get_last_checkpoint(model_path, max_epoch=99999999):
    # 1. Retrieve the latest trained model
    num_checkpoints = []
    for filename in os.listdir(model_path):
        # Use regular expression to find numbers after 'file_'
        match = re.match(r'step_(\d+)', filename)
        if match:
            # Append the extracted number as an integer
            num_checkpoints.append(int(match.group(1)))
    last_ckpt = max([n for n in num_checkpoints if n <= max_epoch])

    return last_ckpt

if __name__ == '__main__':
    """ Comment out task set not being evaluated """
    tasks = [
        'mw-assembly', 'mw-basketball', 'mw-box-close', 'mw-faucet-open', 'mw-hammer', 
        'mw-handle-pull', 'mw-lever-pull', 'mw-pick-out-of-hole', 'mw-sweep-into',
        'pde-allen_cahn', 'pde-burgers', 'pde-wave'
    ]
    
    for task in tasks:
        for seed in [1,2,3]:
            """ ==============================================================
                (1A) Evaluate non-time-aware models trained on default dt 
            ============================================================== """
            print('Baseline model (training default dt)')
            
            # define model path
            if task[:2] == 'mw':
                # metaworld tasks: tdmpc2's pretrained model 
                default_dt = 0.0025
                model_path = f'{cwd}/checkpoints/baseline/{task}-{seed}.pt'
            else:
                # other envs: trained baseline
                default_dt = task_default_dt[task]
                model_path = f'/fs/nexus-scratch/anhu/world-model-checkpoints/{task}/singledt-{default_dt}/{seed}'
                # time-aware model for pde-burgers was only trained till 750k steps
                if task == 'pde-burgers':
                    last_ckpt = get_last_checkpoint(model_path, max_epoch=751000)
                else:
                    last_ckpt = get_last_checkpoint(model_path)
                model_path = f'{model_path}/step_{last_ckpt}.pt'
            
            # evaluate model on different eval_dt's
            if os.path.exists(model_path):
                """ 1(a) single-step prediction"""
                print(f'Model Path: {model_path}')
                os.system(f'python eval_model_multidt.py task={task} checkpoint={model_path} seed={seed} multi_dt=false')
                """ 1(b) adjusted-step prediction"""
                os.system(f'python eval_model_multidt.py task={task} checkpoint={model_path} seed={seed} multi_dt=false eval_steps_adjusted=true')
            print()

            """ ==============================================================
                (1B) Evaluate non-time-aware model trained on various non-default dt's 
                    => not all tasks have model for this ablation study
            ============================================================== """
            for train_dt in [0.001, 0.01, 0.05]:
                """ Collect baseline results on different fixed dt """
                print(f'Baseline model (training dt = {train_dt})')
                model_path = f'/fs/nexus-scratch/anhu/world-model-checkpoints/{task}/singledt-{train_dt}/{seed}'
                if os.path.exists(model_path):
                    if task == 'pde-burgers':
                        last_ckpt = get_last_checkpoint(model_path, max_epoch=751000)
                    else:
                        last_ckpt = get_last_checkpoint(model_path)
                    os.system(f'python eval_model_multidt.py task={task} checkpoint={model_path}/step_{last_ckpt}.pt seed={seed} multi_dt=false train_dt={train_dt}')
            print()
                
            
            
            """ ==============================================================
                (2A) Evaluate time-aware models: dt ~ log-uniform(min_dt, max_dt) 
                     with RK4 integrator
            ============================================================== """
            print('Time-aware model (log-uniform)')
            model_path = f'/fs/nexus-scratch/anhu/world-model-checkpoints/{task}/multidt-log-uniform-rk4/{seed}'
            # model_path = f'/fs/nexus-scratch/anhu/world-model-checkpoints/{task}/multidt/{seed}'
            if os.path.exists(model_path):
                last_ckpt = get_last_checkpoint(model_path)
                if task == 'pde-burgers':
                    last_ckpt = get_last_checkpoint(model_path, max_epoch=751000) # pde-burgers: 750k steps
                elif task[:3] == 'pde':
                    last_ckpt = get_last_checkpoint(model_path, max_epoch=1010000) # other pde: 1M steps
                else:
                    last_ckpt = get_last_checkpoint(model_path)

                model_path = f'{model_path}/step_{last_ckpt}.pt'
                print(f'Model Path: {model_path}')
                os.system(f'python eval_model_multidt.py task={task} checkpoint={model_path} seed={seed} multi_dt=true integrator=rk4')
            print()

            """ ==============================================================
                (3) Evaluate time-aware models: dt ~ log-uniform(min_dt, max_dt) 
                     with Euler integrator
            ============================================================== """
            print('Time-aware model (euler log-uniform)')
            model_path = f'/fs/nexus-scratch/anhu/world-model-checkpoints/{task}/multidt-log-uniform-euler/{seed}'
            if os.path.exists(model_path):
                last_ckpt = get_last_checkpoint(model_path)
                if task == 'pde-burgers':
                    last_ckpt = get_last_checkpoint(model_path, max_epoch=751000) # pde-burgers: 750k steps
                elif task[:3] == 'pde':
                    last_ckpt = get_last_checkpoint(model_path, max_epoch=1010000) # other pde: 1M steps
                else:
                    last_ckpt = get_last_checkpoint(model_path)

                model_path = f'{model_path}/step_{last_ckpt}.pt'
                print(f'Model Path: {model_path}')
                os.system(f'python eval_model_multidt.py task={task} checkpoint={model_path} seed={seed} dt_sampler=log-uniform multi_dt=true integrator=euler')
            print()