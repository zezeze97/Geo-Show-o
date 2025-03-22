#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J GeoUni-32Down
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64          # number of cores per tasks
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00

module load cuda/12.1
module load gcc/12.2.0
module load openmpi



# export PYTHONPATH=$PYTHONPATH:/lustre/home/2201210053/Geo-Show-o
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/Geo-Show-o
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置每个 GPU 任务使用的核心数
export OMP_NUM_THREADS=16
# source activate showo
source activate show-o

# accelerate launch --config_file accelerate_configs/1_gpu.yaml --main_process_port=8888 training/train.py config=configs/geouni_sft_512x512_32Down.yaml
accelerate launch --config_file accelerate_configs/4_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train.py config=configs/geouni_sft_512x512_32Down.yaml



# T2I
CUDA_VISIBLE_DEVICES=0 python3 inference_t2i.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
validation_prompts_file=data/geouni_mixing_data/t2i/test_cdl2i_cn.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/gen_imgs_cdl2i_cn > logs/test_cdl2i_cn.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_t2i.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
validation_prompts_file=data/geouni_mixing_data/t2i/test_cdl2i_en.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/gen_imgs_cdl2i_en > logs/test_cdl2i_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_t2i.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
validation_prompts_file=data/geouni_mixing_data/t2i/test_caption2i_cn.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/gen_imgs_caption2i_cn > logs/test_caption2i_cn.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_t2i.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
validation_prompts_file=data/geouni_mixing_data/t2i/test_caption2i_en.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/gen_imgs_caption2i_en > logs/test_caption2i_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python3 inference_t2i.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
validation_prompts_file=data/geouni_mixing_data/t2i/test_gpt_caption2i_cn.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/gen_imgs_gpt_caption2i_cn > logs/test_gpt_caption2i_cn.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_t2i.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
validation_prompts_file=data/geouni_mixing_data/t2i/test_gpt_caption2i_en.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/gen_imgs_gpt_caption2i_en > logs/test_gpt_caption2i_en.log 2>&1 &

# Mixing
CUDA_VISIBLE_DEVICES=0 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_en_problem.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/ \
save_file_name='test_mixing_en_problem' > logs/test_mixing_en_problem.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_cn_problem.jsonl \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/ \
save_file_name='test_mixing_cn_problem' > logs/test_mixing_cn_problem.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_en_problem_ans.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/ \
save_file_name='test_mixing_en_problem_ans' > logs/test_mixing_en_problem_ans.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_cn_problem_ans.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/ \
save_file_name='test_mixing_cn_problem_ans' > logs/test_mixing_cn_problem_ans.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_en_problem_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/ \
save_file_name='test_mixing_en_problem_choice' > logs/test_mixing_en_problem_choice.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_cn_problem_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/ \
save_file_name='test_mixing_cn_problem_choice' > logs/test_mixing_cn_problem_choice.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_en_problem_ans_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/ \
save_file_name='test_mixing_en_problem_ans_choice' > logs/test_mixing_en_problem_ans_choice.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_mix.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
validation_prompts_file=data/geouni_mixing_data/mixing/test_mixing_cn_problem_ans_choice.jsonl \
output_dir=outputs/model_predict/geouni-512x512-0322-32down/ \
save_file_name='test_mixing_cn_problem_ans_choice' > logs/test_mixing_cn_problem_ans_choice.log 2>&1 &


# Reasoning
CUDA_VISIBLE_DEVICES=0 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0322-32down' \
save_file_name='test_reasoning_cn' \
language='cn' \
formalization=False > logs/test_reasoning_cn.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0322-32down' \
save_file_name='test_reasoning_en' \
language='en' \
formalization=False > logs/test_reasoning_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_choice_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0322-32down' \
save_file_name='test_reasoning_choice_cn' \
language='cn' \
formalization=False > logs/test_reasoning_choice_cn.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_choice_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0322-32down' \
save_file_name='test_reasoning_choice_en' \
language='en' \
formalization=False > logs/test_reasoning_choice_en.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0322-32down' \
save_file_name='test_reasoning_cn_pre_formalization' \
language='cn' \
formalization=True > logs/test_reasoning_cn_pre_formalization.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0322-32down' \
save_file_name='test_reasoning_en_pre_formalization' \
language='en' \
formalization=True > logs/test_reasoning_en_pre_formalization.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_choice_cn.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0322-32down' \
save_file_name='test_reasoning_choice_cn_pre_formalization' \
language='cn' \
formalization=True > logs/test_reasoning_choice_cn_pre_formalization.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 inference_reasoning.py config=configs/geouni_test_512x512_32Down.yaml \
pretrained_geouni_model_path='outputs/geouni-512x512-0322-32down' \
max_new_tokens=3000 \
mmu_image_root='./data' \
validation_prompts_file=data/geouni_mixing_data/reasoning/test_reasoning_choice_en.jsonl \
output_dir='outputs/model_predict/geouni-512x512-0322-32down' \
save_file_name='test_reasoning_choice_en_pre_formalization' \
language='en' \
formalization=True > logs/test_reasoning_choice_en_pre_formalization.log 2>&1 &
wait  # 等待所有任务完成