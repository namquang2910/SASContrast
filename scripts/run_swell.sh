#!/usr/bin/env bash
set -euo pipefail

WESAD_CONFIG="configs/pretrain_swell.json"
SWELL_CONFIG="configs/pretrain_swell.json"
PORT=23505
NPROC=2
echo "Running Dual Branch Subject-aware contrastive learning for dataset SWELL..."


torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${WESAD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "SWELLDataset" 

echo "All runs completed."