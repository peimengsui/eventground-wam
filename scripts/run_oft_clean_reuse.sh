#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$1"
shift
mkdir -p "$output_root"
# MuJoCo diagnostics belong with the run, never in the immutable source checkout.
cd "$output_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$project_root/scripts/run_b0_rollout.sh" python -u \
  "$project_root/scripts/evaluate_oft_clean_reuse.py" --output-root "$output_root" "$@"
