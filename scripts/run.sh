# date=$(date +%Y%m%d_%H%M%S)

# model_name_or_path=/mnt/nas_data2/chenjn_workspace/datasets/HF/llama/Llama-3.2-3B-Instruct

# output_name=${date}-Llama-3.2-3B-Instruct
# output_dir=output/llm_pro/train/${output_name}

# mkdir -p ${output_dir}

# # --packing
# CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_name json \
#     --dataset_file output/llm_pro/datasets/law_project/mixed_train_cleaned.jsonl \
#     --learning_rate 2.0e-5 \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --bf16 \
#     --max_seq_length 4096 \
#     --attn_implementation flash_attention_2 \
#     --logging_steps 5 \
#     --eval_strategy no \
#     --eval_steps 10000 \
#     --output_dir ${output_dir} \
#     --wandb_project law_project \
#     --run_name ${output_name}


#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: bash run.sh <train_type> <config_file>"
    exit 1
fi

train_type=$1
config_file=$2

if [ "$train_type" == "sft" ]; then
    CUDA_VISIBLE_DEVICES=2 \
        accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes 1 \
        src/open_r1/sft.py --config recipes/Qwen2.5-1.5B-Instruct/sft/config_custom.yaml

elif [ "$train_type" == "grpo" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
        accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
        src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml
fi
