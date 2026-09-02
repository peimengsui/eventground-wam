"""Validate and summarize a locked evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventground_wam.evaluation_protocol import (
    DEFAULT_PROTOCOL_PATH,
    derive_episode_seed,
    load_evaluation_protocol,
    protocol_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Path to the protocol JSON file",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_evaluation_protocol(args.protocol)
    suites = protocol["benchmark"]["suites"]
    summary = {
        "protocol_id": protocol["protocol_id"],
        "status": protocol["status"],
        "sha256": protocol_sha256(protocol),
        "benchmark_revision": protocol["benchmark"]["revision"],
        "total_task_count": protocol["benchmark"]["total_task_count"],
        "suite_task_counts": {
            suite: specification["task_count"] for suite, specification in suites.items()
        },
        "control_frequency_hz": protocol["simulation"]["control_frequency_hz"],
        "capture_resolution": protocol["observation"]["capture_resolution"],
        "action_chunk_length": protocol["simulation"]["action"]["chunk_length"],
        "example_episode_seed": derive_episode_seed(protocol, "libero_spatial", 0),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print(f"protocol: {summary['protocol_id']} ({summary['status']})")
    print(f"sha256: {summary['sha256']}")
    print(f"benchmark revision: {summary['benchmark_revision']}")
    print(f"tasks: {summary['total_task_count']} across {len(suites)} suites")
    print(
        "simulation: "
        f"{summary['capture_resolution'][0]}x{summary['capture_resolution'][1]} RGB, "
        f"{summary['control_frequency_hz']} Hz, "
        f"{summary['action_chunk_length']}-action chunks"
    )


if __name__ == "__main__":
    main()
