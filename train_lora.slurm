#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J GeoUni-GRPO-Lora
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32          # number of cores per tasks
#SBATCH --gres=gpu:2
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu25   # 添加这一行排除节点

module load cuda/12.1
module load gcc/12.2.0

export WANDB_MODE=offline
export MASTER_ADDR=localhost
export MASTER_PORT=9909
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/GeoUni-GRPO
export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source activate GRPO
# 运行任务
torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/open_r1/grpo.py \
    --geo_config_path configs/geouni_512x512_32down.yaml \
    --dataset_name data/grpo \
    --image_root_path data/ \
    --deepspeed scripts/zero2.json \
    --output_dir checkpoints/geo-grpo-0316-lora \
    --model_name_or_path GeoUni \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --save_total_limit 1 \
    --num_train_epochs 1 \
    --num_generations 8 \
    --run_name GeoUni_GRPO_0316-lora \
    --save_steps 10 \
    --learning_rate 5e-5 \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 512 \
    --beta 0.01 \
    --lora_target_modules q_proj v_proj k_proj o_proj \



CUDA_VISIBLE_DEVICES=0 python inference_reasoning.py config=configs/geouni_test_32down.yaml \
pretrained_geouni_model_path='/lustre/home/2001110054/Geo-Show-o/outputs/geouni-512x512-0313-32down' \
lora_weights_path='checkpoints/geo-grpo-0316-lora' \
output_dir='outputs/geo-grpo-0316-lora' \
save_file_name='test_reasoning_en' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/geouni_mixing_data/reasoning/test_reasoning_en.jsonl' \
language='en' \
formalization=False \
max_new_tokens=3000 > logs/test_reasoning_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python inference_reasoning.py config=configs/geouni_test_32down.yaml \
pretrained_geouni_model_path='/lustre/home/2001110054/Geo-Show-o/outputs/geouni-512x512-0313-32down' \
lora_weights_path='checkpoints/geo-grpo-0316-lora' \
output_dir='outputs/geo-grpo-0316-lora' \
save_file_name='test_reasoning_cn' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/geouni_mixing_data/reasoning/test_reasoning_cn.jsonl' \
language='cn' \
formalization=False \
max_new_tokens=3000 > logs/test_reasoning_cn.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python inference_reasoning.py config=configs/geouni_test_32down.yaml \
pretrained_geouni_model_path='/lustre/home/2001110054/Geo-Show-o/outputs/geouni-512x512-0313-32down' \
lora_weights_path='checkpoints/geo-grpo-0316-lora' \
output_dir='outputs/geo-grpo-0316-lora' \
save_file_name='test_reasoning_choice_en' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/geouni_mixing_data/reasoning/test_reasoning_choice_en.jsonl' \
language='en' \
formalization=False \
max_new_tokens=3000 > logs/test_reasoning_choice_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python inference_reasoning.py config=configs/geouni_test_32down.yaml \
pretrained_geouni_model_path='/lustre/home/2001110054/Geo-Show-o/outputs/geouni-512x512-0313-32down' \
lora_weights_path='checkpoints/geo-grpo-0316-lora' \
output_dir='outputs/geo-grpo-0316-lora' \
save_file_name='test_reasoning_choice_cn' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/geouni_mixing_data/reasoning/test_reasoning_choice_cn.jsonl' \
language='cn' \
formalization=False \
max_new_tokens=3000 > logs/test_reasoning_choice_cn.log 2>&1 &




CUDA_VISIBLE_DEVICES=0 python inference_reasoning.py config=configs/geouni_test_32down.yaml \
pretrained_geouni_model_path='/lustre/home/2001110054/Geo-Show-o/outputs/geouni-512x512-0313-32down' \
lora_weights_path='checkpoints/geo-grpo-0316-lora' \
output_dir='outputs/geo-grpo-0316-lora' \
save_file_name='test_reasoning_en_pre_formalization' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/geouni_mixing_data/reasoning/test_reasoning_en.jsonl' \
language='en' \
formalization=True \
max_new_tokens=3000 > logs/test_reasoning_en_pre_formalization.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python inference_reasoning.py config=configs/geouni_test_32down.yaml \
pretrained_geouni_model_path='/lustre/home/2001110054/Geo-Show-o/outputs/geouni-512x512-0313-32down' \
lora_weights_path='checkpoints/geo-grpo-0316-lora' \
output_dir='outputs/geo-grpo-0316-lora' \
save_file_name='test_reasoning_cn_pre_formalization' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/geouni_mixing_data/reasoning/test_reasoning_cn.jsonl' \
language='cn' \
formalization=True \
max_new_tokens=3000 > logs/test_reasoning_cn_pre_formalization.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python inference_reasoning.py config=configs/geouni_test_32down.yaml \
pretrained_geouni_model_path='/lustre/home/2001110054/Geo-Show-o/outputs/geouni-512x512-0313-32down' \
lora_weights_path='checkpoints/geo-grpo-0316-lora' \
output_dir='outputs/geo-grpo-0316-lora' \
save_file_name='test_reasoning_choice_en_pre_formalization' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/geouni_mixing_data/reasoning/test_reasoning_choice_en.jsonl' \
language='en' \
formalization=True \
max_new_tokens=3000 > logs/test_reasoning_choice_en_pre_formalization.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python inference_reasoning.py config=configs/geouni_test_32down.yaml \
pretrained_geouni_model_path='/lustre/home/2001110054/Geo-Show-o/outputs/geouni-512x512-0313-32down' \
lora_weights_path='checkpoints/geo-grpo-0316-lora' \
output_dir='outputs/geo-grpo-0316-lora' \
save_file_name='test_reasoning_choice_cn_pre_formalization' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/geouni_mixing_data/reasoning/test_reasoning_choice_cn.jsonl' \
language='cn' \
formalization=True \
max_new_tokens=3000 > logs/test_reasoning_choice_cn_pre_formalization.log 2>&1 &


wait