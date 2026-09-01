"""Write the machine-local LIBERO-Plus path configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero-root",
        type=expanded_path,
        default=expanded_path("~/code/vendor/LIBERO-plus"),
    )
    parser.add_argument(
        "--data-root",
        type=expanded_path,
        default=expanded_path("~/data/eventground-wam/libero-plus"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = args.libero_root / "libero" / "libero"
    required_paths = {
        "benchmark root": benchmark_root,
        "BDDL files": benchmark_root / "bddl_files",
        "initial states": benchmark_root / "init_files",
        "assets": args.data_root / "assets",
    }
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not path.exists()]
    if missing:
        raise SystemExit("Missing LIBERO-Plus paths:\n" + "\n".join(missing))

    config_dir = args.data_root / "config"
    datasets_dir = args.data_root / "datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "assets": str(args.data_root / "assets"),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "benchmark_root": str(benchmark_root),
        "datasets": str(datasets_dir),
        "init_states": str(benchmark_root / "init_files"),
    }
    config_path = config_dir / "config.yaml"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"LIBERO_CONFIG_PATH={config_dir}")
    print(f"configuration: {config_path}")


if __name__ == "__main__":
    main()
