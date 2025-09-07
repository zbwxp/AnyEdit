#!/usr/bin/env bash
set -euo pipefail

# Go to repo root (directory of this script)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Optional: activate local virtualenv if present
if [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

# Match VS Code launch env
export PYTHONPATH="${DIR}:${PYTHONPATH:-}"

python experiments/evaluate_uns.py \
  --alg_name=AlphaEdit_ARE \
  --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname=Llama3-8B-Instruct.json \
  --ds_name=unke \
  --dataset_size_limit=1000 \
  --num_edits=1