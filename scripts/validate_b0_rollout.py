#!/usr/bin/env python3
"""Validate a B0 training-task HDF5 artifact and its sealed rollout report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventground_wam.b0_rollout_validation import validate_b0_rollout_bundle
from eventground_wam.evaluation_protocol import (
    B0_TRAINING_ROLLOUT_PROTOCOL_PATH,
    load_evaluation_protocol,
)


def _expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=_expanded_path)
    parser.add_argument("--report", type=_expanded_path, required=True)
    parser.add_argument(
        "--protocol",
        type=_expanded_path,
        default=B0_TRAINING_ROLLOUT_PROTOCOL_PATH,
    )
    args = parser.parse_args()
    result = validate_b0_rollout_bundle(
        args.artifact,
        args.report,
        load_evaluation_protocol(args.protocol),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
