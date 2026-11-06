#!/usr/bin/env bash
# One host-owned process runs both policies in the frozen order, then validates all artifacts.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$1"
shift
cd "$project_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
scripts/run_b0_rollout.sh python -u scripts/evaluate_clean_comparison.py \
  --policy b0 --output-root "$output_root" "$@"
scripts/run_reference_policy.sh python -u scripts/evaluate_clean_comparison.py \
  --policy reference --output-root "$output_root" "$@"
CUDA_VISIBLE_DEVICES="" scripts/run_b0_rollout.sh python -u scripts/evaluate_clean_comparison.py \
  --policy analyze --output-root "$output_root" "$@"
