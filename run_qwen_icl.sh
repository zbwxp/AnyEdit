#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/code/AnyEdit"
export PYTHONPATH="$ROOT:${PYTHONPATH-}"
cd "$ROOT"

python3 -u experiments/evaluate_icl.py \
  --alg_name=AlphaEdit_ARE \
  --model_name=qwen/Qwen2.5-7B-Instruct \
  --hparams_fname=Qwen2.5-7B-Instruct.json \
  --ds_name=unke \
  --dataset_size_limit=1000 \
  --num_edits=1 \
  "$@"

