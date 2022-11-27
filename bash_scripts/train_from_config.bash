#!/bin/bash
#
#SBATCH --job-name=sPhysNet-MT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=8:00:00
#SBATCH --mem=30GB

CONFIG=$1
python train.py --config_name $CONFIG