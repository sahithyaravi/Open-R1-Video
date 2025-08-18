#!/bin/bash
#SBATCH --account=aip-vshwartz
#SBATCH --job-name=qwen2
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=10:10:00
deactivate
module load python/3.10.13 
module load scipy-stack/2024a
module load gcc opencv/4.11.0
module load cuda
module load gcc arrow/16.1.0 
source /home/sahiravi/projects/aip-vshwartz/sahiravi/videor1/bin/activate

# display current python venv
echo "Current Python Virtual Environment:"
which python
# display current python version
echo "Current Python Version:"
python --version

export WANDB_API_KEY=3596e10c718e17ba4c1ba6fc462b2ad582eb0dcc
export WANDB_PROJECT=Qwen2-VL-7B-Video-GRPO-dummy
export WANDB_NAME=llava-video-4k-remove-formatreward-matchletterreward-f16-full
export FLASH_ATTENTION_USE_TILED=1
export FLASH_ATTENTION_BLOCK_HEURISTIC=2

mkdir -p data/wangxd/ckpt/$WANDB_PROJECT/$WANDB_NAME
# conda install av -c conda-forgeQwen/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    src/open_r1_video/grpo.py \
    --deepspeed scripts/zero2.json \
    --output_dir data/ckpt/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name xxx \
    --jsonl_path /home/sahiravi/scratch/data/LLaVA-Video-large-swift-beliefs.jsonl \
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
    --max_pixels 200704\
    --save_only_model true


# conda install av -c conda-forge