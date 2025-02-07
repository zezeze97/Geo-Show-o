#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU40G
#SBATCH --qos=low
#SBATCH -J Infer
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

CUDA_VISIBLE_DEVICES=0 python3 inference_t2i.py config=configs/geouni_test_512x512.yaml \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/t2i/test_cdl2i.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0206/gen_imgs_cdl2i > logs/test_cdl2i.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_t2i.py config=configs/geouni_test_512x512.yaml \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/t2i/test_caption2i_en.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0206/gen_imgs_caption2i_en > logs/test_caption2i_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_t2i.py config=configs/geouni_test_512x512.yaml \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/t2i/test_caption2i_cn.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0206/gen_imgs_caption2i_cn > logs/test_caption2i_cn.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_t2i.py config=configs/geouni_test_512x512.yaml \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/t2i/test_gpt_caption2i.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0206/gen_imgs_gpt_caption2i > logs/test_gpt_caption2i.log 2>&1 &


wait

