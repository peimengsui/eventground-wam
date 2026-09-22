#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
data_root="${EVENTGROUND_LIBERO_DATA_ROOT:-${HOME}/data/eventground-wam/libero-plus}"
libero_root="${EVENTGROUND_LIBERO_ROOT:-${HOME}/code/vendor/LIBERO-plus}"
python_bin="${EVENTGROUND_B0_PYTHON:-${HOME}/code/eventground-wam-b0-oft-v1/environments/b0/.venv/bin/python}"
# Reuse the frozen installed runtime, with source taken only from this checkout.
export PYTHONPATH="${project_root}/src:${project_root}/scripts:${libero_root}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LIBERO_CONFIG_PATH="${data_root}/config"
export EVENTGROUND_LIBERO_ROOT="${libero_root}"
export EVENTGROUND_LIBERO_DATA_ROOT="${data_root}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
imagemagick_root="${data_root}/system/imagemagick"
if [[ -d "${imagemagick_root}/usr" ]]; then
  export MAGICK_HOME="${imagemagick_root}/usr"
  export MAGICK_CONFIGURE_PATH="${imagemagick_root}/etc/ImageMagick-7"
  export LD_LIBRARY_PATH="${imagemagick_root}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
stage="${1:?expected train or evaluate}"
shift
case "${stage}" in
  train) exec "${python_bin}" "${script_dir}/train_b0_continuation.py" "$@" ;;
  evaluate) exec "${python_bin}" "${script_dir}/evaluate_b0_continuation.py" "$@" ;;
  *) echo "unknown continuation stage: ${stage}" >&2; exit 78 ;;
esac
