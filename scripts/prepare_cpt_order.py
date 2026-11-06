#!/usr/bin/env python3
"""Freeze the one-pass Pixel/JEPA order from already approved train metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.future_pairs import read_sealed, sha256_file
from eventground_wam.pixel_training import build_order


def main(args):
    manifest = read_sealed(args.pairs/"manifest.json")
    expected = sha256_file(Path(__file__).resolve().parents[1]/
                           "configs/datasets/trajectory-v2-future-pairs-v1/manifest.json")
    if sha256_file(args.pairs/"manifest.json") != expected:
        raise ValueError("pair manifest differs from accepted Step 1")
    if sha256_file(args.pairs/"sources.jsonl") != manifest["outputs"]["sources.jsonl"]["sha256"]:
        raise ValueError("frozen source registry changed")
    sources = [json.loads(line) for line in (args.pairs/"sources.jsonl").read_text().splitlines()]
    if any(s["digest"] != contract_digest(s) for s in sources):
        raise ValueError("unsealed source record")
    report = build_order([s for s in sources if s["split"] == "train"], args.output,
                          pair_manifest_sha256=expected)
    if report["pairs_by_horizon"] != manifest["counts"]["train"]["pairs_by_horizon"]:
        raise ValueError("CPT exposure differs from all frozen train pairs")
    print(json.dumps({key: report[key] for key in ("digest", "updates", "anchors", "pair_presentations",
                                                   "minimum_pairs_per_update", "maximum_pairs_per_update")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
