#!/usr/bin/env bash
# Explicit Feifei environment; no launch or scheduling implied by unpacking this file.
set -euo pipefail
EVAL_SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$EVAL_SOURCE_ROOT/src:$EVAL_SOURCE_ROOT/scripts:/home/peimengsui/code/vendor/LIBERO-plus"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=/home/peimengsui/data/eventground-wam/libero-plus/config
export EVENTGROUND_LIBERO_ROOT=/home/peimengsui/code/vendor/LIBERO-plus
export EVENTGROUND_LIBERO_DATA_ROOT=/home/peimengsui/data/eventground-wam/libero-plus
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MAGICK_HOME=/home/peimengsui/data/eventground-wam/libero-plus/system/imagemagick/usr
export MAGICK_CONFIGURE_PATH=/home/peimengsui/data/eventground-wam/libero-plus/system/imagemagick/etc/ImageMagick-7
export LD_LIBRARY_PATH=/home/peimengsui/data/eventground-wam/libero-plus/system/imagemagick/usr/lib/x86_64-linux-gnu
exec /home/peimengsui/code/eventground-wam-week4-run-v2/environments/b0/.venv/bin/python \
  "$EVAL_SOURCE_ROOT/scripts/evaluate_coupled_dev.py" "$@"
