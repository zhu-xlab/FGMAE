#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/pretrain/pretrain_mfp_vitl16_70_hog_B2_%j.out
#SBATCH --error=srun_outputs/pretrain/pretrain_mfp_vitl16_70_hog_B2_%j.err
#SBATCH --time=23:00:00
#SBATCH --job-name=mfp-hog
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

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# run script as slurm job
srun python -u pretrain_fgmae.py \
--root_s2 /p/project/hai_dm4eo/wang_yi/data/251k_sar.lmdb \
--output_dir /p/project/hai_ssl4eo/wang_yi/MAE-MFP/src/benchmark/pretrain_ssl/checkpoints/mfp_vitl16_hog_B2/ \
--log_dir /p/project/hai_ssl4eo/wang_yi/MAE-MFP/src/benchmark/pretrain_ssl/checkpoints/mfp_vitl16_hog_B2/ \
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
--feature hog \
--frac 1 \
--blr 1.5e-4 \
--in_channels 2 \
--norm_pix_loss \
--hog_norm \
#--resume /p/project/hai_dm4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/moco/B13_rn18_int16/checkpoint_0059.pth.tar
