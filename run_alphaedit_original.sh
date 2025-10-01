#!/bin/bash

# Set environment variables
export PYTHONPATH="/root/code/AnyEdit"
export CUDA_VISIBLE_DEVICES="0"

# Change to the project directory
cd /root/code/AnyEdit

# Run the evaluation script with Qwen2.5-7B-Instruct
python experiments/evaluate_uns.py \
    --alg_name=AlphaEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B-Instruct.json \
    --ds_name=unke \
    --dataset_size_limit=1000 \
    --num_edits=1

echo "llama evaluation completed!"


python experiments/evaluate_uns.py \
    --alg_name=AlphaEdit \
    --model_name=qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=Qwen2.5-7B-Instruct.json \
    --ds_name=unke \
    --dataset_size_limit=1000 \
    --num_edits=1

echo "qwen evaluation completed!"