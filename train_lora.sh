module load cuda/12.1
module load gcc/12.2.0

export WANDB_MODE=offline
export MASTER_ADDR=localhost
export MASTER_PORT=9909
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/GeoUni-GRPO
# export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 运行任务
torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/open_r1/grpo.py \
    --geo_config_path configs/geouni_512x512_32down.yaml \
    --dataset_name data/smalltest \
    --image_root_path data/ \
    --deepspeed scripts/zero2.json \
    --output_dir checkpoints/geo-grpo-0318-lora-debug \
    --model_name_or_path GeoUni \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --save_total_limit 1 \
    --num_train_epochs 200 \
    --num_generations 8 \
    --run_name GeoUni_GRPO_0318-lora-debug \
    --save_steps 10 \
    --learning_rate 5e-5 \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 512 \
    --beta 0.01 \
    --lora_target_modules q_proj v_proj k_proj o_proj \