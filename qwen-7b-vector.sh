#!/bin/bash
#SBATCH --qos=a100_sahiravi
#SBATCH --job-name=qwen2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --partition=a100
#SBATCH --time=6:00:00
module load cuda-12.4
export WANDB_API_KEY=3596e10c718e17ba4c1ba6fc462b2ad582eb0dcc
export WANDB_PROJECT=Qwen2-VL-7B-Video-GRPO
export WANDB_NAME=llava-video-4k-remove-formatreward-matchletterreward-f16-full
export FLASH_ATTENTION_USE_TILED=1
export FLASH_ATTENTION_BLOCK_HEURISTIC=2
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

mkdir -p /h/sahiravi/scratch/data/wangxd/ckpt/$WANDB_PROJECT/$WANDB_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    src/open_r1_video/grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir data/ckpt/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name xxx \
    --jsonl_path /h/sahiravi/scratch/data/LLaVA-Video-large-swift-beliefs.jsonl \
    --max_prompt_length 16384 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --torch_dtype 'bfloat16' \
    --bf16 true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 100 \
    --num_generations 4\
    --save_only_model true

