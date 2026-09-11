#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=test-tf-%j.log
#SBATCH --error=test-tf-%j.log
#SBATCH -p long

module load cuda/8.0
module load powerAI-4.0/Tensorflow-1.1.0

srun python -u cifar10_train.py
