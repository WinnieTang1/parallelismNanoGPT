#!/bin/bash
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:15:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=wt222@scarletmail.rutgers.edu
#SBATCH --mail-type=begin
ml cudatoolkit/12.9

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)


source "/pscratch/sd/w/wtang99/parallelismNanoGPT"

srun -N 2 --ntasks-per-node=1 --gpus-per-node=4 torchrun \
        --nnodes=2 \
        --nproc_per_node=8 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        /pscratch/sd/w/wtang99/nanoGPT/train.py /pscratch/sd/w/wtang/parallelismNanoGPT/config/train_shakespeare_char.py \
        --n_gpus=$1