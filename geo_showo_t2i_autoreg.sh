#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU40G
#SBATCH --qos=low
#SBATCH -J Infer
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16          # number of cores per tasks
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu26   # 添加这一行排除节点

module load cuda/11.8
module load gcc/12.2.0
module load openmpi
export PYTHONPATH=$PYTHONPATH:/lustre/home/2001110054/Show-o
export OMP_NUM_THREADS=16
source activate show-o

srun python3 geo_inference_t2i_autoreg.py config=configs/formalgeo-autoreg-32Down-512x512-t2i-0112/test.yaml \
batch_size=8 validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/t2i_noAug_v2/test.json \
mode='t2i'