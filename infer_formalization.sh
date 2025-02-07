#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU40G
#SBATCH --qos=low
#SBATCH -J Infer-formalization
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64          # number of cores per tasks
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu26   # 添加这一行排除节点

module load cuda/11.8
module load gcc/12.2.0
module load openmpi
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/Geo-Show-o
export OMP_NUM_THREADS=64
source activate show-o


CUDA_VISIBLE_DEVICES=0 python3 inference_mmu.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=600 \
mmu_image_root='./data/formalgeo7k/formalgeo7k_v2' \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/formalization/formalization_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0206' \
save_file_name='formalization_en' > logs/formalization_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_mmu.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=600 \
mmu_image_root='./data/formalgeo7k/formalgeo7k_v2' \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/formalization/formalization_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0206' \
save_file_name='formalization_cn' > logs/formalization_cn.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_mmu.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=600 \
mmu_image_root='./data/formalgeo7k/formalgeo7k_v2' \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/formalization/formalization_cali_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0206' \
save_file_name='formalization_cali_en' > logs/formalization_cali_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_mmu.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=600 \
mmu_image_root='./data/formalgeo7k/formalgeo7k_v2' \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/formalization/formalization_cali_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0206' \
save_file_name='formalization_cali_cn' > logs/formalization_cali_cn.log 2>&1 &

wait  # 等待所有任务完成