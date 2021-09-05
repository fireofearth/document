#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=blob_dqn
#SBATCH --output=/scratch/cchen795/slurm/%x-%j.out
#SBATCH --error=/scratch/cchen795/slurm/%x-%j.out

source $HOME/scratch/py38init.sh
source ../cc.env.sh

echo "train blob dqn"
export MPLBACKEND="agg"
python blob_dqn.py
