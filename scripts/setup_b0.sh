#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
uv_bin="${EVENTGROUND_UV_BIN:-${HOME}/.local/bin/uv}"

if [[ ! -x "${uv_bin}" ]]; then
  echo "uv is not executable at ${uv_bin}" >&2
  exit 1
fi

"${uv_bin}" sync --project "${project_root}/environments/b0" --locked

EVENTGROUND_UV_BIN="${uv_bin}" "${script_dir}/run_b0.sh" python -c \
  'import av, numpy, pyarrow; print(f"av={av.__version__} numpy={numpy.__version__} pyarrow={pyarrow.__version__}")'
