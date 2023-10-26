#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/classification/BE_sup_vits16_100_%j.out
#SBATCH --error=srun_outputs/classification/BE_sup_vits16_100_%j.err
#SBATCH --time=06:00:00
#SBATCH --job-name=BE_sup_vits16
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
module load Stages/2022
module load GCCcore/.11.2.0
module load Python

# activate virtual environment
source /p/project/hai_dm4eo/wang_yi/env2/bin/activate

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# run script as slurm job
srun python -u linear_BE_mae.py \
--is_slurm_job \
--data_path /p/project/hai_dm4eo/wang_yi/data/BigEarthNet/ \
--output_dir /p/project/hai_ssl4eo/wang_yi/MAE-MFP/src/benchmark/transfer_classification/checkpoints/BE_sup_vits16_100_70 \
--log_dir /p/project/hai_ssl4eo/wang_yi/MAE-MFP/src/benchmark/transfer_classification/checkpoints/BE_sup_vits16_100_70/log \
--model vit_small_patch16 \
--nb_classes 19 \
--train_frac 1.0 \
--num_workers 10 \
--batch_size 64 \
--epochs 50 \
--lr 0.05 \
--warmup_epochs 0 \
--dist_url $dist_url \
--dist_backend 'nccl' \
--seed 42 \
--fine_tune \
#--finetune /p/project/hai_ssl4eo/wang_yi/MAE-MFP/src/benchmark/pretrain_ssl/checkpoints/mae_vits16_raw_B13/checkpoint-99.pth \