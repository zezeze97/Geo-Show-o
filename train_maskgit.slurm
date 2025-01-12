#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J MaskGit
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64          # number of cores per tasks
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu26   # 添加这一行排除节点

module load cuda/11.8
module load gcc/12.2.0
module load openmpi



# export PYTHONPATH=$PYTHONPATH:/lustre/home/2201210053/Geo-Show-o
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/Show-o
export WANDB_MODE=offline

# 设置每个 GPU 任务使用的核心数
export OMP_NUM_THREADS=16
# source activate showo
source activate show-o
# accelerate launch --config_file accelerate_configs/1_gpu.yaml --main_process_port=8888 training/finetuning_T2I.py config=configs/showo_finetuning_512x512_t2i.yaml
accelerate launch --config_file accelerate_configs/4_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/finetuning_T2I.py config=configs/formalgeo-maskgit-32Down-512x512-t2i-0112/train.yaml

