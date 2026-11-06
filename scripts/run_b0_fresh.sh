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
stage="${1:?expected pipeline, train, clean, dev-b0, or dev-reference}"
shift
case "${stage}" in
  pipeline)
    work_root="${1:?work root required}"
    stage_pid=""
    stop_pipeline() {
      if [[ -n "${stage_pid}" ]]; then
        kill -TERM "${stage_pid}" 2>/dev/null || true
        wait "${stage_pid}" || true
      fi
      exit 75
    }
    trap stop_pipeline TERM INT
    run_stage() {
      "${BASH_SOURCE[0]}" "$@" &
      stage_pid=$!
      wait "${stage_pid}"
      stage_pid=""
    }
    run_stage train --phase train --until-epoch 1 --work-root "${work_root}"
    run_stage clean --epoch 1 --mode development --work-root "${work_root}"
    run_stage train --phase train --until-epoch 2 --work-root "${work_root}"
    run_stage clean --epoch 2 --mode development --work-root "${work_root}"
    run_stage clean --epoch 2 --mode clean-2000 --work-root "${work_root}"
    run_stage dev-b0 --work-root "${work_root}"
    run_stage dev-reference --work-root "${work_root}"
    run_stage analyze --work-root "${work_root}"
    ;;
  train) exec "${python_bin}" "${script_dir}/train_b0_fresh.py" "$@" ;;
  clean) exec "${python_bin}" "${script_dir}/evaluate_b0_fresh_clean.py" "$@" ;;
  dev-b0) exec "${python_bin}" "${script_dir}/evaluate_b0_fresh_dev.py" --policy b0 "$@" ;;
  dev-reference)
    export PYTHONPATH="${project_root}/src:${project_root}/scripts:${HOME}/code/vendor/openvla-oft:${libero_root}"
    export TF_CPP_MIN_LOG_LEVEL=2
    exec "${HOME}/code/eventground-wam-reference-clean-v1/environments/reference-policy/.venv/bin/python" "${script_dir}/evaluate_b0_fresh_dev.py" --policy reference "$@"
    ;;
  analyze) exec "${python_bin}" "${script_dir}/analyze_b0_fresh.py" "$@" ;;
  *) echo "unknown fresh stage: ${stage}" >&2; exit 78 ;;
esac
