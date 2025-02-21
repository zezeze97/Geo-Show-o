#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=normal
#SBATCH -J GeoUni-GRPO-Test
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64          # number of cores per tasks
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu25   # 添加这一行排除节点

module load cuda/12.1
module load gcc/12.2.0

export WANDB_MODE=offline
export MASTER_ADDR=localhost
export MASTER_PORT=9909
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/GeoUni-GRPO
# export CUDA_HOME=/usr/local/cuda-12.1
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source activate GRPO
# 运行任务
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/open_r1/grpo.py \
    --geo_config_path configs/geouni_512x512_0221.yaml \
    --dataset_name data/2formalgeo7k \
    --image_root_path data/ \
    --deepspeed scripts/zero3.json \
    --output_dir checkpoints/geo-grpo-0221 \
    --model_name_or_path GeoUni \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --save_total_limit 1 \
    --num_train_epochs 2 \
    --num_generations 8 \
    --run_name GeoUni_GRPO_0221 \
    --save_steps 10 \
    --learning_rate 5e-5 \

