#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J Infer-mix
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

CUDA_VISIBLE_DEVICES=0 python3 inference_mix.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/mixing/test_mixing_en_problem.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0210/ \
save_file_name='test_mixing_en_problem' > logs/test_mixing_en_problem.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_mix.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/mixing/test_mixing_cn_problem.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0210/ \
save_file_name='test_mixing_cn_problem' > logs/test_mixing_cn_problem.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_mix.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/mixing/test_mixing_en_problem_ans.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0210/ \
save_file_name='test_mixing_en_problem_ans' > logs/test_mixing_en_problem_ans.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_mix.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/mixing/test_mixing_cn_problem_ans.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0210/ \
save_file_name='test_mixing_cn_problem_ans' > logs/test_mixing_cn_problem_ans.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python3 inference_mix.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/mixing/test_mixing_en_problem_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0210/ \
save_file_name='test_mixing_en_problem_choice' > logs/test_mixing_en_problem_choice.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_mix.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/mixing/test_mixing_cn_problem_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0210/ \
save_file_name='test_mixing_cn_problem_choice' > logs/test_mixing_cn_problem_choice.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_mix.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/mixing/test_mixing_en_problem_ans_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0210/ \
save_file_name='test_mixing_en_problem_ans_choice' > logs/test_mixing_en_problem_ans_choice.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_mix.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/mixing/test_mixing_cn_problem_ans_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0210/ \
save_file_name='test_mixing_cn_problem_ans_choice' > logs/test_mixing_cn_problem_ans_choice.log 2>&1 &


wait

