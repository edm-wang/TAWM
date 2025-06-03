#!/bin/bash
#SBATCH -c 32                   # 32 CPUs
#SBATCH --mem=32G               # 128 GB RAM
#SBATCH --gres=gpu:1            # 1 GPUs
#SBATCH --time=3-00:00:00       # 3 days
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long

source activate tdmpc2
cd /fs/nexus-scratch/anhu/tawm/tawm

# # 1) baseline model: non-time-aware model
# python train.py task=mw-assembly multi_dt=false checkpoint=/fs/nexus-scratch/anhu/world-model-checkpoints steps=1500000 seed=3

# # 2) Time-Aware model: rk4 integrator
# python train.py task=mw-assembly multi_dt=true checkpoint=/fs/nexus-scratch/anhu/world-model-checkpoints steps=1500000 seed=3

# # 3) Time-Aware model: Euler integrator
python train.py task=mw-assembly multi_dt=true checkpoint=/fs/nexus-scratch/anhu/world-model-checkpoints steps=1500000 integrator=euler seed=3
