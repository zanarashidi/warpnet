#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=3
#SBATCH --gres=gpu:4
#SBATCH --output=test-tf-%j.log
#SBATCH --error=test-tf-%j.log
#SBATCH --partition=long

module load cuda/8.0
module load powerAI-4.0/Tensorflow-1.1.0

srun -N1 python -u 6termswithinputsonce/cifar100_train.py  > 6termswithinputsonce/6termswithinputsonce.txt &
srun -N1 python -u 7termswithFs/cifar100_train.py  > 7termswithFs/7termswithFs.txt &
srun -N1 python -u 3terms/cifar100_train.py  > 3terms/3terms.txt &

wait
