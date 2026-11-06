#!/usr/bin/env python3
"""Validate all pair rows and frozen split/source relationships without media I/O."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventground_wam.future_pairs import (
    expand_split_groups,
    read_sealed,
    sha256_file,
    verify_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    config = read_sealed(ROOT / "configs/datasets/trajectory_v2_future_pairs_v1.json")
    split_root = ROOT / "configs/datasets/libero-plus-trajectory-v2"
    for name, expected in config["inputs"]["split_files"].items():
        if sha256_file(split_root / name) != expected:
            raise ValueError("frozen split changed")
    assignments = expand_split_groups(
        map(json.loads, (split_root / "groups.jsonl").read_text().splitlines()))
    provenance = read_sealed(args.bundle / "manifest.json")["provenance"]
    if (provenance["config_digest"] != config["digest"] or provenance["split_manifest_sha256"]
            != config["inputs"]["split_files"]["manifest.json"]):
        raise ValueError("bundle belongs to another data/split contract")
    print(json.dumps(verify_bundle(args.bundle, assignments, provenance), indent=2))


if __name__ == "__main__":
    main()
