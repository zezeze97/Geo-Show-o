#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J Infer-GRPO-Lora
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16          # number of cores per tasks
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu25   # 添加这一行排除节点

python inference_reasoning.py config=configs/geouni_test.yaml \
pretrained_geouni_model_path='/lustre/home/2001110054/Geo-Show-o/outputs/geouni-512x512-0217' \
lora_weights_path='checkpoints/geo-grpo-0307-lora' \
output_dir='outputs/geo-grpo-0307-lora' \
save_file_name='test_reasoning_choice_cn' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/reasoning/test_reasoning_choice_cn.jsonl' \
language='cn' \
formalization=False \
max_new_tokens=3000
