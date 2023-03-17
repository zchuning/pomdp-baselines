#!/bin/bash

#SBATCH --job-name=icml-mppi
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zchuning@cs.washington.edu

#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=24:00:00

#SBATCH --chdir=/gscratch/weirdlab/zchuning/pomdp-baselines
#SBATCH --export=all
#SBATCH --output=slurm/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm/%j-err.txt    # where STDERR goes

# Your programs to run.
export WANDB_MODE=online
export WANDB_PROJECT=ramp-icml-rebuttal

for seed in 0 1
do
    srun --exclusive -N 1 -n 1 python policies/main.py --cfg configs/meta/metaworld/reach_wall.yml --algo sac --seed $seed &
    srun --exclusive -N 1 -n 1 python policies/main.py --cfg configs/meta/metaworld/button_press.yml --algo sac --seed $seed &
    srun --exclusive -N 1 -n 1 python policies/main.py --cfg configs/meta/metaworld/faucet_open.yml --algo sac --seed $seed &
    srun --exclusive -N 1 -n 1 python policies/main.py --cfg configs/meta/metaworld/door_open.yml --algo sac --seed $seed &
    srun --exclusive -N 1 -n 1 python policies/main.py --cfg configs/meta/dclaw/dclaw.yml --algo sac --seed $seed &
    wait
done