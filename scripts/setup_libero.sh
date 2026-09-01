#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
libero_root="${EVENTGROUND_LIBERO_ROOT:-${HOME}/code/vendor/LIBERO-plus}"
data_root="${EVENTGROUND_LIBERO_DATA_ROOT:-${HOME}/data/eventground-wam/libero-plus}"
uv_bin="${EVENTGROUND_UV_BIN:-${HOME}/.local/bin/uv}"
libero_revision="4976dc30028e805ff8094b55501d532c48fec182"
asset_url="https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip?download=true"
asset_sha256="96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
asset_archive="${data_root}/assets.zip"
asset_dir="${data_root}/assets"
source_asset_link="${libero_root}/libero/libero/assets"
runtime_project="${project_root}/environments/libero"

if [[ ! -x "${uv_bin}" ]]; then
  echo "uv is not executable at ${uv_bin}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${libero_root}")" "${data_root}"

if [[ ! -d "${libero_root}/.git" ]]; then
  git clone https://github.com/sylvestf/LIBERO-plus.git "${libero_root}"
fi

if [[ -n "$(git -C "${libero_root}" status --short --untracked-files=no)" ]]; then
  echo "LIBERO-Plus checkout has tracked changes: ${libero_root}" >&2
  exit 1
fi

if ! git -C "${libero_root}" cat-file -e "${libero_revision}^{commit}" 2>/dev/null; then
  git -C "${libero_root}" fetch origin "${libero_revision}"
fi
git -C "${libero_root}" checkout --detach "${libero_revision}"

"${uv_bin}" sync --project "${runtime_project}" --locked
"${uv_bin}" pip install \
  --python "${runtime_project}/.venv/bin/python" \
  --no-deps \
  --editable "${libero_root}"

if [[ ! -f "${asset_archive}" ]] || ! printf '%s  %s\n' "${asset_sha256}" "${asset_archive}" | sha256sum --check --status; then
  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 5 \
    --retry-delay 2 \
    --output "${asset_archive}" \
    "${asset_url}"
fi
printf '%s  %s\n' "${asset_sha256}" "${asset_archive}" | sha256sum --check

if [[ ! -d "${asset_dir}" ]]; then
  if zipinfo -1 "${asset_archive}" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
    echo "Unsafe path found in asset archive" >&2
    exit 1
  fi
  extraction_root="$(mktemp -d "${data_root}/.assets-extract.XXXXXX")"
  unzip -q "${asset_archive}" -d "${extraction_root}"
  if [[ ! -d "${extraction_root}/assets" ]]; then
    echo "Asset archive does not contain the expected assets directory" >&2
    exit 1
  fi
  mv "${extraction_root}/assets" "${asset_dir}"
  rmdir "${extraction_root}"
fi

if [[ -L "${source_asset_link}" ]]; then
  if [[ "$(readlink -f "${source_asset_link}")" != "$(readlink -f "${asset_dir}")" ]]; then
    echo "Existing asset symlink points elsewhere: ${source_asset_link}" >&2
    exit 1
  fi
elif [[ -e "${source_asset_link}" ]]; then
  echo "Asset path exists and is not a symlink: ${source_asset_link}" >&2
  exit 1
else
  ln -s "${asset_dir}" "${source_asset_link}"
fi

"${runtime_project}/.venv/bin/python" "${project_root}/scripts/configure_libero.py" \
  --libero-root "${libero_root}" \
  --data-root "${data_root}"

echo "LIBERO-Plus revision: ${libero_revision}"
echo "LIBERO-Plus source: ${libero_root}"
echo "LIBERO-Plus data: ${data_root}"
