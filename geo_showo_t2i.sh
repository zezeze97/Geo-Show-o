#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU40G
#SBATCH --qos=low
#SBATCH -J Infer
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu26   # 添加这一行排除节点

module load cuda/11.8
module load gcc/12.2.0
module load openmpi
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/Show-o
export OMP_NUM_THREADS=16
source activate show-o

# 将任务输出分别重定向到不同日志文件
# srun python3 geo_inference_t2i.py config=configs/formalgeo-maskgit-16Down-512x512-t2i-0114/test.yaml \
# batch_size=8 validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/t2i/test_2cdl2i.json \
# guidance_scale=0.0 generation_timesteps=50 output_dir=outputs/formalgeo-maskgit-16Down-512x512-t2i-0114/gen_imgs_2cdl2i \
# mode='t2i' 

# srun python3 geo_inference_t2i.py config=configs/formalgeo-maskgit-16Down-512x512-t2i-0114/test.yaml \
# batch_size=8 validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/t2i/test_caption2i.json \
# guidance_scale=0.0 generation_timesteps=50 output_dir=outputs/formalgeo-maskgit-16Down-512x512-t2i-0114/gen_imgs_caption2i \
# mode='t2i'

# srun python3 geo_inference_t2i.py config=configs/formalgeo-maskgit-16Down-512x512-t2i-0114/test.yaml \
# batch_size=8 validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/t2i/test_caption_2cdl2i.json \
# guidance_scale=0.0 generation_timesteps=50 output_dir=outputs/formalgeo-maskgit-16Down-512x512-t2i-0114/gen_imgs_caption_2cdl2i \
# mode='t2i'

srun python3 geo_inference_t2i.py config=configs/formalgeo-maskgit-32Down-512x512-t2i-0112/test.yaml \
batch_size=8 validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/t2i_noAug_v2/train.json \
guidance_scale=0.0 generation_timesteps=50 output_dir=outputs/formalgeo-maskgit-32Down-512x512-t2i-0112/gen_imgs_2cdl2i_train \
mode='t2i'