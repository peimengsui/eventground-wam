#!/usr/bin/env python3
"""Generate the complete Week 2 trajectory-contract index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eventground_wam.artifact_provenance import project_state
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.trajectory_index import write_trajectory_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_VERSION = "week2-v1"


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def default_data_root() -> Path:
    return expanded_path(
        os.environ.get(
            "EVENTGROUND_LIBERO_DATA_ROOT",
            "~/data/eventground-wam/libero-plus",
        )
    )


def parse_args() -> argparse.Namespace:
    data_root = default_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=expanded_path,
        default=data_root / "datasets" / "libero-plus-lerobot" / "current",
    )
    parser.add_argument(
        "--output",
        type=expanded_path,
        default=data_root / "derived" / "week2-v1" / "trajectory-contracts.jsonl",
    )
    parser.add_argument("--annotation-version", default=DEFAULT_ANNOTATION_VERSION)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow non-publishable output from an uncommitted implementation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision, clean = project_state(PROJECT_ROOT)
    if not clean and not args.allow_dirty:
        raise SystemExit(
            "project tree is dirty; commit the generator or pass --allow-dirty for "
            "explicitly non-publishable output"
        )
    reader = DemonstrationReader(args.snapshot)
    manifest = write_trajectory_index(
        reader.iter_episodes(),
        reader.repository_root,
        args.output,
        project_git_revision=revision,
        project_tree_clean=clean,
        annotation_version=args.annotation_version,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
