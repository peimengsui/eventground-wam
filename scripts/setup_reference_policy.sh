#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
evaluator_root="${EVENTGROUND_OPENVLA_OFT_ROOT:-${HOME}/code/vendor/openvla-oft}"
libero_root="${EVENTGROUND_LIBERO_ROOT:-${HOME}/code/vendor/LIBERO-plus}"
data_root="${EVENTGROUND_REFERENCE_POLICY_DATA_ROOT:-${HOME}/data/eventground-wam/reference-policy}"
libero_data_root="${EVENTGROUND_LIBERO_DATA_ROOT:-${HOME}/data/eventground-wam/libero-plus}"
checkpoint_dir="${EVENTGROUND_REFERENCE_POLICY_CHECKPOINT:-${data_root}/checkpoints/openvla-7b-oft-libero-plus-a85655ec}"
uv_bin="${EVENTGROUND_UV_BIN:-${HOME}/.local/bin/uv}"
evaluator_revision="e4287e94541f459edc4feabc4e181f537cd569a8"
libero_revision="4976dc30028e805ff8094b55501d532c48fec182"

if [[ ! -x "${uv_bin}" ]]; then
  echo "uv is not executable at ${uv_bin}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${evaluator_root}")" "${data_root}"
if [[ ! -d "${evaluator_root}/.git" ]]; then
  git clone --filter=blob:none https://github.com/moojink/openvla-oft.git "${evaluator_root}"
fi
if [[ -n "$(git -C "${evaluator_root}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "OpenVLA-OFT checkout is not clean: ${evaluator_root}" >&2
  exit 1
fi
if ! git -C "${evaluator_root}" cat-file -e "${evaluator_revision}^{commit}" 2>/dev/null; then
  git -C "${evaluator_root}" fetch origin "${evaluator_revision}"
fi
git -C "${evaluator_root}" checkout --detach "${evaluator_revision}"

if [[ ! -d "${libero_root}/.git" ]]; then
  echo "LIBERO-Plus checkout is missing; run scripts/setup_libero.sh first" >&2
  exit 1
fi
if [[ "$(git -C "${libero_root}" rev-parse HEAD)" != "${libero_revision}" ]]; then
  echo "LIBERO-Plus revision mismatch: expected ${libero_revision}" >&2
  exit 1
fi

# setup_libero.sh keeps the large machine-local assets outside the source tree
# and links them into the upstream package. Mark only that verified link as a
# local Git exclude, then require every non-ignored path to be clean.
asset_link="${libero_root}/libero/libero/assets"
asset_dir="${libero_data_root}/assets"
if [[ ! -L "${asset_link}" ]] || \
  [[ "$(readlink -f "${asset_link}")" != "$(readlink -f "${asset_dir}")" ]]; then
  echo "LIBERO-Plus asset link is missing or points elsewhere: ${asset_link}" >&2
  exit 1
fi
if ! git -C "${libero_root}" check-ignore --quiet -- libero/libero/assets; then
  exclude_file="$(git -C "${libero_root}" rev-parse --path-format=absolute --git-path info/exclude)"
  printf '\n# EventGround machine-local LIBERO assets\n/libero/libero/assets\n' >> "${exclude_file}"
fi
if [[ -n "$(git -C "${libero_root}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "LIBERO-Plus checkout is not clean: ${libero_root}" >&2
  exit 1
fi

"${uv_bin}" sync --project "${project_root}/environments/reference-policy" --locked
"${project_root}/scripts/run_reference_policy.sh" \
  python "${project_root}/scripts/check_reference_policy.py" \
  --evaluator-root "${evaluator_root}" \
  --checkpoint-dir "${checkpoint_dir}" \
  --download \
  --full-hash \
  --require-cuda

echo "OpenVLA-OFT revision: ${evaluator_revision}"
echo "OpenVLA-OFT source: ${evaluator_root}"
echo "Reference checkpoint: ${checkpoint_dir}"
