#!/usr/bin/env bash
# CPU analysis and previews continue on the host after the dev evaluation completes.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dev_root="$1"
clean_root="$2"
output_root="$3"
mkdir -p "$output_root"
cd "$output_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
args=(--dev-root "$dev_root" --clean-root "$clean_root" --output-root "$output_root")
"$project_root/scripts/run_b0_rollout.sh" python -u "$project_root/scripts/analyze_week4_failures.py" --mode clean "${args[@]}"
while [[ ! -f "$dev_root/evaluation-report.json" ]]; do
  if systemctl --user is-failed --quiet eventground-week4-b0-dev210.service; then
    echo 'dev-210 service failed; retain analysis and inspect the evaluation log.' >&2
    exit 1
  fi
  sleep 20
done
"$project_root/scripts/run_b0_rollout.sh" python -u "$project_root/scripts/analyze_week4_failures.py" --mode dev "${args[@]}"
"$project_root/scripts/run_b0_rollout.sh" python -u "$project_root/scripts/analyze_week4_failures.py" --mode previews "${args[@]}"
