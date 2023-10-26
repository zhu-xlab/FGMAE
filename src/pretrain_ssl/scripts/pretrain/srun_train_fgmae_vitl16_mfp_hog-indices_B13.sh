#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/pretrain/pretrain_mfp_vitl16_70_hog-indices_B13_%j.out
#SBATCH --error=srun_outputs/pretrain/pretrain_mfp_vitl16_70_hog-indices_B13_%j.err
#SBATCH --time=23:00:00
#SBATCH --job-name=mfp-both
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
#module load Stages/2022
#module load GCCcore/.11.2.0
#module load Python

# activate virtual environment
#source /p/project/hai_dm4eo/wang_yi/env2/bin/activate

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

# run script as slurm job
srun python -u pretrain_fgmae.py \
--root_s2 /p/project/hai_dm4eo/wang_yi/data/251k_ms.lmdb \
--output_dir /p/project/hai_ssl4eo/wang_yi/MAE-MFP/src/benchmark/pretrain_ssl/checkpoints/mae_vitl16_hog-indices_B13/ \
--log_dir /p/project/hai_ssl4eo/wang_yi/MAE-MFP/src/benchmark/pretrain_ssl/checkpoints/mae_vitl16_hog-indices_B13/ \
--model mae_vit_large_patch16 \
--mask_ratio 0.7 \
--num_workers 10 \
--batch_size 64 \
--epochs 100 \
--warmup_epochs 10 \
--weight_decay 0.05 \
--dist_url $dist_url \
--dist_backend 'nccl' \
--seed 42 \
--input_size 224 \
--feature hog+ndi \
--frac 1 \
--blr 1.5e-4 \
--in_channels 13 \
--norm_pix_loss \
--hog_norm \
#--resume /p/project/hai_ssl4eo/wang_yi/MAE-MFP/src/benchmark/pretrain_ssl/checkpoints/mfp_vits16_canny_B13/checkpoint-85.pth
