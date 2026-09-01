#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
data_root="${EVENTGROUND_LIBERO_DATA_ROOT:-${HOME}/data/eventground-wam/libero-plus}"
uv_bin="${EVENTGROUND_UV_BIN:-${HOME}/.local/bin/uv}"

export EVENTGROUND_LIBERO_ROOT="${EVENTGROUND_LIBERO_ROOT:-${HOME}/code/vendor/LIBERO-plus}"
export EVENTGROUND_LIBERO_DATA_ROOT="${data_root}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${data_root}/config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

exec "${uv_bin}" run \
  --project "${project_root}/environments/libero" \
  --no-sync \
  "$@"
