#!/usr/bin/env python3
"""Bounded standard-LIBERO numeric overlap probe: first published episode per task."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from audit_b0_training_source import fetch

from eventground_wam.b0_diagnostics import numeric_digest
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract


def main(root):
    started = time.monotonic()
    audit = read_sealed(root / "data-report.json")
    train = defaultdict(list)
    for line in (root / "numeric-episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "train":
            train[(row["instruction"], row["state_action_digest"])].append(row["episode_index"])
    selected = []
    for suite, source in audit["standard_libero_metadata"].items():
        episodes = [json.loads(line) for line in
                    (root / "public" / suite / "episodes.jsonl").read_text().splitlines()]
        seen = set()
        for row in episodes:
            task = row["tasks"][0]
            if task not in seen:
                selected.append((suite, source, row))
                seen.add(task)
        assert len(seen) == 10

    def compare(item):
        suite, source, episode = item
        i = episode["episode_index"]
        relative = source["info"]["data_path"].format(
            episode_index=i, episode_chunk=i // source["info"]["chunks_size"])
        path = root / "public" / suite / relative
        fetch(f'https://huggingface.co/datasets/{source["repo"]}/resolve/'
              f'{source["revision"]}/{relative}', path)
        table = pq.read_table(path, columns=["observation.state", "action"], use_threads=False)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        assert states.shape == (episode["length"], 8)
        assert actions.shape == (episode["length"], 7)
        assert set(np.unique(actions[:, 6])) <= {0., 1.}
        # RLDS action is gripper-open (1=open); B0 uses -1=open,+1=close.
        # Leave every motion/state float unchanged, with no tolerance matching.
        actions[:, 6] = 1 - 2 * actions[:, 6]
        digest = numeric_digest(states, actions)
        matches = train.get((episode["tasks"][0], digest), [])
        return {"suite": suite, "public_episode": i, "instruction": episode["tasks"][0],
                "length": episode["length"], "public_repo": source["repo"],
                "public_revision": source["revision"], "public_file_sha256": sha256_file(path),
                "state_action_digest_after_gripper_conversion": digest,
                "matching_training_episode_indices": matches}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(compare, selected))
    result = seal_contract({
        "artifact_type": "eventground_wam.standard_libero_numeric_overlap_probe",
        "status": "complete", "source_audit_digest": audit["digest"],
        "selection": "First published episode by metadata order for each of 40 tasks; "
                     "chosen before observing overlap, not a random population sample.",
        "public_episodes_checked": len(rows),
        "public_episodes_with_exact_training_match": sum(
            bool(r["matching_training_episode_indices"]) for r in rows),
        "conversion": "Only gripper: 1 - 2 * public_gripper_open; other floats unchanged",
        "limitation": "Numeric sequence overlap does not establish matching camera appearance, "
                      "full-dataset overlap, or paper-era dataset identity.",
        "elapsed_seconds": time.monotonic() - started, "episodes": rows,
    })
    atomic_json(root / "clean-overlap-report.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    main(parser.parse_args().audit_root)
