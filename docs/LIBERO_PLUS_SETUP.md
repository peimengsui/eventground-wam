# LIBERO-Plus setup

LIBERO-Plus runs on the `feifei` Linux compute host. Its simulator environment is
kept separate from the root model environment because the upstream benchmark
uses substantially older dependency pins.

## Pinned inputs

- Repository: `https://github.com/sylvestf/LIBERO-plus.git`
- Revision: `4976dc30028e805ff8094b55501d532c48fec182`
- Asset archive: `https://huggingface.co/datasets/Sylvest/LIBERO-plus`
- Asset SHA-256: `96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`

## Remote layout

```text
~/code/eventground-wam/                 project source
~/code/vendor/LIBERO-plus/              pinned upstream checkout
~/data/eventground-wam/libero-plus/     assets, datasets, config, smoke output
```

## Ubuntu libraries

The upstream project requires ImageMagick at runtime. The setup script installs
the needed Ubuntu runtime packages into the project-owned data directory without
`sudo` or system changes. A system-wide installation is optional:

```bash
sudo apt update
sudo apt install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
```

## Reproduce the installation

From the project checkout on `feifei`:

```bash
./scripts/setup_libero.sh
./scripts/run_libero.sh python scripts/check_libero.py
```

The setup script is idempotent: it verifies the pinned Git revision and asset
checksum, reuses completed downloads, synchronizes the locked `uv` environment,
provides the legacy CMake compatibility setting required by `egl-probe`, and
refuses to replace unexpected files or directories. It also normalizes the
archive's deeply nested upstream path into the expected local `assets/` layout.

The upstream repository's editable-package metadata does not expose its nested
namespace with modern setuptools. The runner therefore adds the pinned upstream
checkout to `PYTHONPATH` explicitly instead of modifying upstream code.

The smoke test creates and resets a perturbed task, applies an initial simulator
state, advances five zero-action steps, and writes a rendered image plus JSON
report under `~/data/eventground-wam/libero-plus/smoke`.

## Environment variables

The scripts accept these optional overrides:

| Variable | Default |
|---|---|
| `EVENTGROUND_LIBERO_ROOT` | `~/code/vendor/LIBERO-plus` |
| `EVENTGROUND_LIBERO_DATA_ROOT` | `~/data/eventground-wam/libero-plus` |
| `EVENTGROUND_UV_BIN` | `~/.local/bin/uv` |
| `MUJOCO_EGL_DEVICE_ID` | `0` |
