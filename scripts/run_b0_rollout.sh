#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
data_root="${EVENTGROUND_LIBERO_DATA_ROOT:-${HOME}/data/eventground-wam/libero-plus}"
libero_root="${EVENTGROUND_LIBERO_ROOT:-${HOME}/code/vendor/LIBERO-plus}"
uv_bin="${EVENTGROUND_UV_BIN:-${HOME}/.local/bin/uv}"
imagemagick_root="${data_root}/system/imagemagick"
libero_revision="4976dc30028e805ff8094b55501d532c48fec182"

verify_clean_source_tree() {
  local label="$1"
  local source_root="$2"
  local expected_revision="$3"
  local actual_revision
  local status

  if ! git -C "${source_root}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "${label} checkout is missing: ${source_root}" >&2
    exit 1
  fi
  actual_revision="$(git -C "${source_root}" rev-parse HEAD)"
  if [[ -n "${expected_revision}" && "${actual_revision}" != "${expected_revision}" ]]; then
    echo "${label} revision mismatch: expected ${expected_revision}, found ${actual_revision}" >&2
    exit 1
  fi
  status="$(git -C "${source_root}" status --porcelain=v1 --untracked-files=all)"
  if [[ -n "${status}" ]]; then
    echo "${label} checkout is not clean: ${source_root}" >&2
    echo "${status}" >&2
    exit 1
  fi
}

verify_clean_source_tree "EventGround-WAM" "${project_root}" ""
verify_clean_source_tree "LIBERO-Plus" "${libero_root}" "${libero_revision}"

export EVENTGROUND_LIBERO_ROOT="${libero_root}"
export EVENTGROUND_LIBERO_DATA_ROOT="${data_root}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${data_root}/config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTHONPATH="${project_root}/src:${project_root}/scripts:${libero_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -d "${imagemagick_root}/usr" ]]; then
  export MAGICK_HOME="${MAGICK_HOME:-${imagemagick_root}/usr}"
  export MAGICK_CONFIGURE_PATH="${MAGICK_CONFIGURE_PATH:-${imagemagick_root}/etc/ImageMagick-7}"
  export LD_LIBRARY_PATH="${imagemagick_root}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec "${uv_bin}" run \
  --project "${project_root}/environments/b0" \
  --no-sync \
  "$@"
