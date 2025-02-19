#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=high
#SBATCH -J GeoUni-GRPO-Test
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16          # number of cores per tasks
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu25   # 添加这一行排除节点


export WANDB_MODE=offline
export MASTER_ADDR=localhost
export MASTER_PORT=9909
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/GeoUni-GRPO
export CUDA_HOME=/usr/local/cuda-12.1
source activate GRPO
# 运行任务
torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/open_r1/grpo.py \
    --geo_config_path configs/geouni_512x512_0218.yaml \
    --image_root_path data/ \
    --deepspeed scripts/zero3.json \
    --output_dir checkpoints/geo-grpo-0218 \
    --model_name_or_path GeoUni \
    --dataset_name data/smalltest \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --save_total_limit 1 \
    --num_train_epochs 100 \
    --num_generations 8 \
    --run_name GeoUni_GRPO_0218 \
    --save_steps 10 \
    --learning_rate 5e-5

