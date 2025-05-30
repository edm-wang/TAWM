#!/bin/bash
#SBATCH -c 32                   # 32 CPUs
#SBATCH --mem=32G               # 128 GB RAM
#SBATCH --gres=gpu:1            # 1 GPUs
#SBATCH --time=3-00:00:00       # 3 days
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long

source activate tdmpc2
cd ~/world_models_diff_envs/tdmpc2/tdmpc2

python eval_model_learning_curve.py
