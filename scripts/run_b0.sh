#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
uv_bin="${EVENTGROUND_UV_BIN:-${HOME}/.local/bin/uv}"

if [[ ! -x "${uv_bin}" ]]; then
  echo "uv is not executable at ${uv_bin}" >&2
  exit 1
fi

export PYTHONPATH="${project_root}/src:${project_root}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

exec "${uv_bin}" run \
  --project "${project_root}/environments/b0" \
  --no-sync \
  "$@"
