#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
data_root="${EVENTGROUND_LIBERO_DATA_ROOT:-${HOME}/data/eventground-wam/libero-plus}"
uv_bin="${EVENTGROUND_UV_BIN:-${HOME}/.local/bin/uv}"
runtime_project="${project_root}/environments/demonstrations"
python_bin="${runtime_project}/.venv/bin/python"
max_workers="${EVENTGROUND_DOWNLOAD_WORKERS:-8}"

if [[ ! -x "${uv_bin}" ]]; then
  echo "uv is not executable at ${uv_bin}" >&2
  exit 1
fi

"${uv_bin}" sync --project "${runtime_project}" --locked
"${python_bin}" "${project_root}/scripts/download_demonstrations.py" \
  --data-root "${data_root}" \
  --max-workers "${max_workers}"
"${python_bin}" "${project_root}/scripts/check_demonstrations.py" \
  --data-root "${data_root}"
