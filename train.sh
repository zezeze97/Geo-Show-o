export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export CXXFLAGS="-I/lustre/home/2201210053/software/miniconda3/envs/openr1/include"
export LDFLAGS="-L/lustre/home/2201210053/software/miniconda3/envs/openr1/lib"

# NCCL 相关设置，减少调试信息
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=DEFAULT
export NCCL_SOCKET_IFNAME=eno3np0

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"

# 进入工作目录
cd /lustre/home/2201210053/geo-grpo

# 设置 Python 环境
export WANDB_MODE=offline
export PYTHONPATH=/lustre/home/2201210053/geo-grpo/src:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda-12.1


export DS_ACCELERATOR=cuda
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

# 运行任务
torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    src/open_r1/grpo.py \
    --deepspeed scripts/zero2.json \
    --output_dir checkpoints/geo-grpo \
    --model_name_or_path GeoUni \
    --dataset_name data/formalgeo7k \
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
    --run_name GeoUni_GRPO
