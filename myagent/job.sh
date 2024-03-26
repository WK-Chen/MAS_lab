#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -c 1
#SBATCH --time=12:00:00
#SBATCH --partition=t4v1,t4v2,rtx6000
#SBATCH --qos=normal
#SBATCH --output=log
#SBATCH --gres=gpu:1

echo Running on $(hostname)
date

python train.py