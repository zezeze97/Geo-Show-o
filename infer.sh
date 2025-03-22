#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J Infer-Reasoning-and-Mixing
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64          # number of cores per tasks
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu26   # 添加这一行排除节点

module load cuda/12.1
module load gcc/12.2.0
module load openmpi
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/Geo-Show-o
source activate show-o

# Reasoning
CUDA_VISIBLE_DEVICES=0 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0320-32down' \
save_file_name='test_reasoning_cn' \
language='cn' \
formalization=False > logs/test_reasoning_cn.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0320-32down' \
save_file_name='test_reasoning_en' \
language='en' \
formalization=False > logs/test_reasoning_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_choice_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0320-32down' \
save_file_name='test_reasoning_choice_cn' \
language='cn' \
formalization=False > logs/test_reasoning_choice_cn.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_choice_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0320-32down' \
save_file_name='test_reasoning_choice_en' \
language='en' \
formalization=False > logs/test_reasoning_choice_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0320-32down' \
save_file_name='test_reasoning_cn_pre_formalization' \
language='cn' \
formalization=True > logs/test_reasoning_cn_pre_formalization.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0320-32down' \
save_file_name='test_reasoning_en_pre_formalization' \
language='en' \
formalization=True > logs/test_reasoning_en_pre_formalization.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_choice_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0320-32down' \
save_file_name='test_reasoning_choice_cn_pre_formalization' \
language='cn' \
formalization=True > logs/test_reasoning_choice_cn_pre_formalization.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_choice_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0320-32down' \
save_file_name='test_reasoning_choice_en_pre_formalization' \
language='en' \
formalization=True > logs/test_reasoning_choice_en_pre_formalization.log 2>&1 &


# Mixing
CUDA_VISIBLE_DEVICES=0 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_en_problem.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0320-32down/ \
save_file_name='test_mixing_en_problem' > logs/test_mixing_en_problem.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_cn_problem.jsonl \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
output_dir=outputs/model_predict/geouni-512x512-0320-32down/ \
save_file_name='test_mixing_cn_problem' > logs/test_mixing_cn_problem.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_en_problem_ans.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0320-32down/ \
save_file_name='test_mixing_en_problem_ans' > logs/test_mixing_en_problem_ans.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_cn_problem_ans.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0320-32down/ \
save_file_name='test_mixing_cn_problem_ans' > logs/test_mixing_cn_problem_ans.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_en_problem_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0320-32down/ \
save_file_name='test_mixing_en_problem_choice' > logs/test_mixing_en_problem_choice.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_cn_problem_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0320-32down/ \
save_file_name='test_mixing_cn_problem_choice' > logs/test_mixing_cn_problem_choice.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_en_problem_ans_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0320-32down/ \
save_file_name='test_mixing_en_problem_ans_choice' > logs/test_mixing_en_problem_ans_choice.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0320-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_cn_problem_ans_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0320-32down/ \
save_file_name='test_mixing_cn_problem_ans_choice' > logs/test_mixing_cn_problem_ans_choice.log 2>&1 &

wait  # 等待所有任务完成