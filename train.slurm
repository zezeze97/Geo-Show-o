#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH -J GeoGRPO_0212
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1  
#SBATCH --time=5-00:00:00
#SBATCH --exclude=l12gpu28

module load cuda/12.1
module load gcc/12.2.0
module load openmpi

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# Set NCCL_DEBUG to WARN or ERROR to reduce verbosity
export NCCL_DEBUG=WARN
# Set NCCL_DEBUG_SUBSYS to DEFAULT to avoid detailed output
export NCCL_DEBUG_SUBSYS=DEFAULT
export NCCL_SOCKET_IFNAME=eno3np0

export NCCL_INCLUDE_DIR=/lustre/home/2201210053/software/miniconda3/envs/openr1/include/nccl.h  # 修改为你实际的路径
export NCCL_LIB_DIR=/lustre/home/2201210053/software/miniconda3/envs/openr1/lib/libnccl.so         # 修改为你实际的路径
export CXXFLAGS="-I$NCCL_INCLUDE_DIR"
export LDFLAGS="-L$NCCL_LIB_DIR"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$((15000 + $RANDOM % 5000))

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"

cd /lustre/home/2201210053/geo-grpo

export WANDB_MODE=offline
export PYTHONPATH=/lustre/home/2201210053/geo-grpo/src:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda-12.1
source /lustre/home/2201210053/software/miniconda3/bin/activate openr1

export DS_ACCELERATOR=cuda
TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=INFO TORCH_SHOW_CPP_STACKTRACES=1

torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo.py \
    --deepspeed scripts/zero2.json \
    --output_dir checkpoints/geo-grpo0212_1 \
    --model_name_or_path GeoUni \
    --dataset_name data/2formalgeo7k \
    --max_prompt_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2359296 \
    --save_total_limit 8 \
    --num_train_epochs 1 \
    --run_name GeoUni_GRPO_0212_1
