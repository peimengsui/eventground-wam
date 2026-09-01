#!/usr/bin/env bash
set -euo pipefail

compute_host="${1:-feifei}"

ssh -o BatchMode=yes -o ConnectTimeout=10 "${compute_host}" '
set -eu
echo "host: $(hostname)"
echo "user: $(id -un)"
echo "home: ${HOME}"
echo "kernel: $(uname -srm)"
git --version
python3 --version
if command -v uv >/dev/null 2>&1; then
  uv --version
elif [ -x "${HOME}/.local/bin/uv" ]; then
  "${HOME}/.local/bin/uv" --version
else
  echo "uv: not found"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi: not found"
fi
'
