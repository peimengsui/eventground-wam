"""Validate one unified EventGround-WAM HDF5 episode artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventground_wam.episode_recorder import sha256_file, validate_episode_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=lambda value: Path(value).expanduser().resolve())
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a recording or aborted status while retaining structural checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_episode_file(
        args.episode,
        allow_incomplete=args.allow_incomplete,
    )
    summary["sha256"] = sha256_file(args.episode)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
