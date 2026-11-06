"""Seal prospective train-only review selection; never opens episode payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.event_supervision import (
    SPLIT_DIGEST,
    STUDY_DIGEST,
    build_review_queue,
    preparation_contract,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evaluation/week10-event-supervision-v1/preparation"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    study = json.loads(
        (root / "configs/training/week10_eventground_experiment_v1.json").read_text()
    )
    split = json.loads(
        (root / "configs/datasets/libero-plus-trajectory-v2/manifest.json").read_text()
    )
    for value, expected in [(study, STUDY_DIGEST), (split, SPLIT_DIGEST)]:
        if value["digest"] != expected or contract_digest(value) != expected:
            raise ValueError("Study or split identity changed")
    path = root / "configs/datasets/libero-plus-trajectory-v2/groups.jsonl"
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != study["source_files_sha256"][str(path.relative_to(root))]
    ):
        raise ValueError("Group payload changed")
    groups = [json.loads(line) for line in path.read_text().splitlines()]
    queue = build_review_queue(groups)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [("contract.json", preparation_contract()), ("review-queue.json", queue)]:
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "queue_digest": queue["digest"],
                "initial_episodes": 200,
                "reserve": 100,
                "payloads_opened": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
