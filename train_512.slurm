#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J GEOSHOW-O
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=4          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64          # number of cores per tasks
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu29,l12gpu25   # 添加这一行排除节点

module load cuda/11.8
module load gcc/12.2.0
module load openmpi



export PYTHONPATH=$PYTHONPATH:/lustre/home/2201210053/Geo-Show-o
#export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/Show-o
export WANDB_MODE=offline

# 设置每个 GPU 任务使用的核心数
export OMP_NUM_THREADS=16
source activate showo
#source activate show-o

accelerate launch --config_file accelerate_configs/4_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/finetuning_Geo_T2I.py config=configs/showo_finetuning_512x512_t2i_1209_2.yaml
#accelerate launch --config_file accelerate_configs/1_gpu.yaml --main_process_port=8888 training/finetuning_T2I.py config=configs/showo_finetuning_512x512_t2i_1209_1.yaml
#accelerate launch --config_file accelerate_configs/2_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/finetuning_T2I.py config=configs/showo_finetuning_512x512_t2i.yaml
# accelerate launch --config_file accelerate_configs/1_gpu.yaml --main_process_port=8888 training/finetuning.py config=configs/showo_finetuning_512x512.yaml
#accelerate launch --config_file accelerate_configs/1_gpu.yaml --main_process_port=8888 training/finetuning_Geo_T2I.py config=configs/showo_finetuning_512x512_t2i_1209_1.yaml
