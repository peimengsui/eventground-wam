#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
data_root="${EVENTGROUND_LIBERO_DATA_ROOT:-${HOME}/data/eventground-wam/libero-plus}"
uv_bin="${EVENTGROUND_UV_BIN:-${HOME}/.local/bin/uv}"
imagemagick_root="${data_root}/system/imagemagick"

export EVENTGROUND_LIBERO_ROOT="${EVENTGROUND_LIBERO_ROOT:-${HOME}/code/vendor/LIBERO-plus}"
export EVENTGROUND_LIBERO_DATA_ROOT="${data_root}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${data_root}/config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
# LIBERO's pinned, trusted initial-state files predate PyTorch's weights-only
# loading default. Keep the compatibility override scoped to this runner.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTHONPATH="${project_root}/src:${EVENTGROUND_LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -d "${imagemagick_root}/usr" ]]; then
  export MAGICK_HOME="${MAGICK_HOME:-${imagemagick_root}/usr}"
  export MAGICK_CONFIGURE_PATH="${MAGICK_CONFIGURE_PATH:-${imagemagick_root}/etc/ImageMagick-7}"
  export LD_LIBRARY_PATH="${imagemagick_root}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec "${uv_bin}" run \
  --project "${project_root}/environments/libero" \
  --no-sync \
  "$@"
