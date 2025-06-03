<h1> Introduction </h1>
This is the code for Time-Aware World Model (TAWM), a model-agnostic and more efficient training method for world model. In this work, TAWM is built on top of TD-MPC2 world model architecture as the basis of the experiments. <br><br>

Since TAWM's core contribution is the time-aware training method, which is architecture-agnostic, it can be incorporated into any world model training pipeline, including but is not limited to TD-MPC2 and Dreamers. <br><br>

To incorporate **TAWM** into any world model architecture:

1. **Modify the dynamics or temporal state-space model to condition on the time step** $\Delta t$  
   Example (Euler integration model):  
   $$z_{t+\Delta t} = z_t + d_{\theta}(z_t, a_t, \Delta t) \cdot \tau(\Delta t)$$

2. **Train using a mixture of time step sizes:**  
   $$\Delta t \sim \text{Log-Uniform}(\Delta t_{\min}, \Delta t_{\max}) \quad \text{(or Uniform sampling)}$$


<h1> 1. Dependencies Installations </h1>

1. Installation:<br>
   **Base Conda env (recommended)**
   install Miniconda3 + dependencies:
   ```sh
   conda env create -f environment.yaml
   pip3 install gym==0.21.0
   pip3 install torch==2.3.1
   pip3 install torchvision==0.18.1
   ```
   


   **Additional dependecies for control tasks**:
   
   ```sh
   # install metaworld envs
   pip3 install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
   pip3 install gymnasium

   # install controlgym envs (PDE)
   cd tdmpc2/envs
   git clone https://github.com/xiangyuan-zhang/controlgym.git
   rm -r controlgym/.git/
   ```

   ---
   <b> IMPORTANT NOTE </b>: <br>
   If for some reasons, `import controlgym` causes the following warning to TERMINATE the program (which it shouldn't):
   `UserWarning: We've integrated functorch into PyTorch. As the final step of the integration, functorch.combine_state_for_ensemble is deprecated as of PyTorch 2.0 and will be deleted in a future version of PyTorch >= 2.3.` <br>
   <b> Normally, the warning MUST NOT TERMINATE the program (just a warning) </b> <br><br>
   The following fix works for us:
   * see the error log like this: `~/miniconda3/envs/tdmpc2/lib/python3.9/site-packages/torch/_functorch/deprecated.py:38, in warn_deprecated(api, new_api)` on your machine / environment
   * execute `vim <path to _functorch/deprecated.py>` <br>
      in our case, it was `vim ~/miniconda3/envs/tdmpc2/lib/python3.9/site-packages/torch/_functorch/deprecated.py`
   * comment out line 38: `# warnings.warn(warning, stacklevel=2)`
   



<h1> 2. TAWM and baseline training </h1>

1. Activate conda env
   ```sh
   conda activate tawm
   ```

2. TAWM Training Examples:
   ```sh
   cd tdmpc2
   # TAWM: meta-world environments
   python train.py task=mw-basketball multi_dt=true steps=1500000 seed=3
   # TAWM: pde-control environments: no rendering/video
   python train.py task=pde-burgers save_video=false multi_dt=true steps=1500000 seed=5
   ```

   ---
   For the non-time-aware baseline models trained on fixed default $\Delta t$, we used the trained weights of the original TD-MPC2 model for each Meta-World control task.
   The trained weights are available here: https://huggingface.co/nicklashansen/tdmpc2/tree/main/metaworld.

   Baseline Training Examples:
   ```sh
   cd tdmpc2
   # baseline: meta-world environments
   python train.py task=mw-basketball multi_dt=false steps=2000000 seed=3
   # baseline: pde-control environments: no rendering/video
   python train.py task=pde-burgers save_video=false multi_dt=false steps=2000000 seed=5
   ```

   Every 50,000 steps, the model checkpoints are saved in `<cfg.checkpoint>/<task>/<model_type>-<dt_sampler>-<integrator>/<seed>`, where:
   * `cfg.checkpoint`: saving directory of model checkpoints
   * `task`: control task (e.g. `mw-assembly`, `mw-basketball`, `pde-burgers`)
   * `model_type`: `multidt` (TAWM) or `singledt` (baseline)
   * `dt_sampler`: $\Delta t$ sampling method; `log-uniform` (default) or `uniform`
   * `integrator`: integration method; `euler` (default) or `rk4`

<!-- 5. Evaluation: <br>
   The evaluation code evalutate the performance of the world model on specified task with different simulation time steps. For `meta-world` environment, it also provides success rate for the task (success rate ranges from 0.0 to 1.0).

   Example evaluation on `mw-faucet-open` using MPC planner (better performance for some tasks):
   ```sh
   python eval_model_multidt.py task=mw-faucet-open multitask=false checkpoint=<path to .pt file>
   ```

   Example evaluation on `mw-faucet-open` using model-free planner $\pi$ (significantly faster):
   ```sh
   python eval_model_multidt.py task=mw-faucet-open multitask=false checkpoint=<path to .pt file> mpc=false
   ``` -->

<h1> 3. Model Evaluation </h1>

<h2> 3a. Evaluation Scripts </h2>
The evaluation scripts are provided as reference script for TAWM experiment/deployments. <br>

The evaluation code evalutate the performance of the world model on specified task with different simulation time steps. For `meta-world` environment, it also provides success rate for the task (success rate ranges from 0.0 to 1.0). <br>

**NOTE: You need to update your local `model_path`**
1. `eval_model_multidt.py`: test model performance on `task` on various inference-time observation rates <br>
   ```sh
   # test TAWM (Euler + Log-Uniform) on `mw-basketball`
   python eval_model_multidt.py task=mw-basketball checkpoint={model_path} seed={seed} dt_sampler=log-uniform multi_dt=true integrator=euler
   ```

   ```sh
   # test non-time-aware baseline on `mw-basketball`
   python eval_model_multidt.py task=mw-basketball checkpoint={model_path} seed={seed} multi_dt=false eval_steps_adjusted=true
   ```
2. `eval_model_multidt_all`: comprehensively test all model performance across all tasks on various inference-time observation rates <br>
   * Models: <br>
      (1a) Baseline (non-time-aware) trained on $\Delta t_{default}$ <br>
      (1b) Baseline (non-time-aware) trained on $\Delta t \neq \Delta t_{default}$ <br>
      (2a) TAWM (RK4 + Log-Uniform Sampling) <br>
      (2b) TAWM (Euler + Log-Uniform Sampling) <br>
   * Use `eval_model_multidt.py`

3. `eval_model_learning_curve`: Evaluate intermediate models and save learning curves on each task across seeds
   * **NOTE: You need to have model saved at each step for this evaluation.** By default, a model checkpoint is saved every 50,000 steps.
<br>

<h2> 3b. Evaluation Results </h2>

The evaluation results are saved in `tdmpc2/logs/<task>/<eval-type>.csv`.
* `task`: the control task evaluated on
* `eval_type`: evaluated model type (e.g. baseline, TAWM-RK4, TAWM-Euler, etc.)

<h1> (Optional) Experiments with MTS3 </h1>
The original MTS3 is prediction-only world model and does not support evaluation on control tasks. If you are interested in experimenting with MTS3 as a comparison to our TAWM, please use our modified MTS3+MPC for comparison on control tasks. 

1. **Offline data collection** <br>
   The MTS3 model is prediction-only world model, so it does not interact with environments like `Meta-World`. Therefore, we need to collect offline dataset for it before training MTS3.

2. **Offline Data collection**: <br>
   **NOTE**: to collect data for an individual task only, specify `specific_task=<task name>`. <br>
   a. Collect offline data for **Time-Aware World Model** <br>
   ```sh
   python collect_offline_dataset.py task_set=mt9 specific_task=mw-basketball num_eps=40000 ep_length=100 multitask=false multi_dt=true data_dir=/fs/nexus-scratch/anhu/mt9_multidt_40k
   ```

   b. Collect data for **baseline world model**:
   ```sh
   cd tdmpc2/tdmpc2
   python collect_offline_dataset.py task_set=mt9 specific_task=mw-basketball num_eps=40000 ep_length=100 multitask=false multi_dt=false data_dir=/fs/nexus-scratch/anhu/mt9_singledt_40k
   ```

3. **MTS3 settings:** `H`: slow time scale factor for MTS3 
   * access config file in `MTS3/experiments/basketball/conf/model/default_mts3.yaml`
   * set `time_scale_multiplier = <H>`

4. **Example Training MTS3** for `mw-basketball`:
   * go to `MTS3`
   * `python MTS3/experiments/basketball/mts3_exp.py`
   

   <!-- It is very likely that when running the training process with adaptive timesteps, you will get an error like this:

   ```
   Maximum path length allowed by the benchmark has been exceeded
   ```


   This is a mujoco check that we can disable. To do this, simply go to ~/anaconda3/envs/mtrl/lib/python3.8/site-packages/metaworld/envs/mujoco/mujoco_env.py and comment the lines 107 and 108:

   ```
   if getattr(self, 'curr_path_length', 0) > self.max_path_length:
      raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
   ``` -->