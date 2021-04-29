#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=CovidData
#SBATCH --nodes=1

## The number of tasks per node should be the same number as requested GPUS per node.
#SBATCH --ntasks-per-node=1

## The number of cpus per task should be the same number as dataloader workers.
#SBATCH --cpus-per-task=20

#SBATCH --time=24:00:00
#SBATCH --account=machnitz
#SBATCH --partition=p2GPU32
#SBATCH --exclusive
#SBATCH --output=slurm_output/slurm-%j.out

# 300 seconds before training ends resubmit the job
#SBATCH --signal=SIGUSR1@300

module load compilers/cuda/11.0
nvidia-smi
srun /gpfs/home/machnitz/miniconda3/envs/pytorch/bin/python main.py --gpus 1 --num_workers 20 --max_epochs 100 --data_path "/gpfs/work/machnitz/HIDA/HIDA-COVID/HackathonCovidData" --learning_rate 0.0001 --accelerator "ddp"
