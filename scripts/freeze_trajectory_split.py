#!/usr/bin/env python3
"""Freeze the v2 numeric trajectory split without training or policy evaluation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from eventground_wam.b0_data import load_frozen_b0_split_index
from eventground_wam.b0_diagnostics import numeric_digest
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import (
    SOURCE_DATASET_REVISION,
    SOURCE_EPISODE_COUNT,
    SOURCE_FRAME_COUNT,
    seal_contract,
)
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.trajectory_split import (
    SPLIT_ID,
    SPLITS,
    NumericTrajectory,
    build_trajectory_groups,
    expand_groups,
    split_rule,
)


def jsonl_bytes(rows):
    return b"".join((json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n").encode()
                    for r in rows)


def main(args):
    started = time.monotonic()
    rule_path = args.source / "configs/datasets/libero_plus_trajectory_v2.json"
    rule = read_sealed(rule_path)
    if rule != split_rule():
        raise ValueError("rule file and executable implementation differ")
    old = load_frozen_b0_split_index(args.data / "derived/frozen/week2-dataset-v1")
    prior = read_sealed(args.source / "docs/evaluation/week4-b0-training-source-audit-v1.json")
    repository = args.data / f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo"
    for name, digest in prior["source_metadata"]["metadata_sha256"].items():
        if sha256_file(repository / "meta" / name) != digest:
            raise ValueError(f"source metadata changed: {name}")
    reader = DemonstrationReader(repository)
    episodes = list(reader.iter_episodes())
    clean = read_sealed(args.source / "evaluation/splits/libero-clean-v1/clean-40.json")
    tasks = {r["instruction"]: f'{r["suite"]}/task-{r["runtime_task_index"]:02d}'
             for r in clean["tasks"]}
    assert len(episodes) == SOURCE_EPISODE_COUNT and len(tasks) == 40
    assert [e.episode_index for e in episodes] == list(range(SOURCE_EPISODE_COUNT))
    assert sum(e.length for e in episodes) == SOURCE_FRAME_COUNT

    def read_episode(episode):
        table = pq.read_table(episode.parquet_path, columns=["observation.state", "action"],
                              use_threads=False)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if len(states) != episode.length:
            raise ValueError("source episode length mismatch")
        trajectory = NumericTrajectory(episode.episode_index, tasks[episode.instruction],
                                       states, actions)
        row = {"episode_index": episode.episode_index, "task_key": trajectory.task_key,
               "episode_length": episode.length, "old_split": old.assignments[
                   episode.episode_index].split,
               "state_action_digest": numeric_digest(states, actions),
               "action_digest": numeric_digest(actions),
               "parquet_sha256": sha256_file(episode.parquet_path)}
        return trajectory, row

    with ThreadPoolExecutor(max_workers=8) as pool:
        pairs = list(pool.map(read_episode, episodes))
    trajectories, fingerprints = map(list, zip(*pairs, strict=True))
    prior_rows = Path(prior["numeric_row_file"])
    assert sha256_file(prior_rows) == prior["numeric_row_sha256"]
    for line in prior_rows.read_text().splitlines():
        previous = json.loads(line)
        current = fingerprints[previous["episode_index"]]
        assert current["parquet_sha256"] == previous["source_parquet_sha256"]
        assert current["state_action_digest"] == previous["state_action_digest"]
        assert current["action_digest"] == previous["action_digest"]
    print(json.dumps({"episodes_read": len(trajectories), "old_test_arrays_for_identity_only": 1430}),
          flush=True)
    groups, grouping = build_trajectory_groups(trajectories)
    assignments = expand_groups(groups)
    assert [r["episode_index"] for r in assignments] == list(range(SOURCE_EPISODE_COUNT))
    # Independent invocation with reversed input must reproduce exact output bytes.
    replica_groups, replica_audit = build_trajectory_groups(list(reversed(trajectories)))
    assert jsonl_bytes(groups) == jsonl_bytes(replica_groups) and grouping == replica_audit
    del replica_groups, replica_audit
    episode_split = {r["episode_index"]: r["split"] for r in assignments}
    episode_group = {r["episode_index"]: r["group_digest"] for r in assignments}
    overlaps = {}
    for field in ("state_action_digest", "action_digest"):
        splits = defaultdict(set)
        for row in fingerprints:
            splits[row[field]].add(episode_split[row["episode_index"]])
        overlaps[field] = sum(len(values) > 1 for values in splits.values())
        assert overlaps[field] == 0
    assert all(episode_group[a] == episode_group[b]
               for a, b in grouping["near_edges_between_distinct_exact_sequences"])

    def counts(task=None):
        result = {}
        for split in SPLITS:
            selected = [r for r in assignments if r["split"] == split
                        and (task is None or r["task_key"] == task)]
            result[split] = {"groups": len({r["group_digest"] for r in selected}),
                             "episodes": len(selected),
                             "frames": sum(r["episode_length"] for r in selected),
                             "valid_action_chunks": sum(r["valid_action_chunks"] for r in selected)}
        return result

    per_task = {task: counts(task) for task in sorted(tasks.values())}
    assert all(per_task[task][split]["groups"] >= 1 for task in per_task for split in SPLITS)
    old_train_digests = {r["state_action_digest"] for r in fingerprints if r["old_split"] == "train"}
    previously_seen = {}
    for split in ("validation", "test"):
        selected = [r for r in fingerprints if episode_split[r["episode_index"]] == split]
        matched = [r for r in selected if r["state_action_digest"] in old_train_digests]
        previously_seen[split] = {
            "episodes": len(selected), "episodes_matching_old_training": len(matched),
            "groups_with_member_matching_old_training": len({episode_group[r["episode_index"]]
                                                             for r in matched}),
        }
    migration = {old_split: dict(Counter(episode_split[r["episode_index"]] for r in fingerprints
                                       if r["old_split"] == old_split)) for old_split in SPLITS}
    for filename, rows in [("groups.jsonl", groups), ("assignments.jsonl", assignments),
                           ("fingerprints.jsonl", fingerprints)]:
        (args.output / filename).write_bytes(jsonl_bytes(rows))
    source_hashes = {
        str(p.relative_to(args.source)): sha256_file(p)
        for directory in ["src/eventground_wam", "scripts"]
        for p in sorted((args.source / directory).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("._")
    }
    manifest = seal_contract({
        "artifact_type": "eventground_wam.trajectory_grouped_split", "schema_version": 1,
        "split_id": SPLIT_ID, "status": "frozen_split_definition",
        "training_launch_ready": False,
        "launch_readiness_note": "Future version-aware loader, train-only normalization, "
                                 "recipe and recovery pilot still required; v1 loader unchanged",
        "rule_digest": rule["digest"], "rule_sha256": sha256_file(rule_path),
        "source": prior["source"],
        "prior_train_validation_numeric_audit_digest": prior["digest"],
        "prior_train_validation_source_parquet_hashes_unchanged": True,
        "source_scope_note": "source refers to immutable snapshot and historical freeze identities; "
                             "this split replaces assignments only for future fresh training",
        "canonical_task_panel_sha256": sha256_file(
            args.source / "evaluation/splits/libero-clean-v1/clean-40.json"),
        "counts": counts(), "per_task": per_task, "grouping": grouping,
        "cross_split_numeric_overlap_counts": overlaps,
        "cross_split_group_overlap": 0, "cross_split_near_edges": 0,
        "old_to_new_episode_counts": migration,
        "old_training_exposure_in_new_holdouts": previously_seen,
        "outputs": {name: {"sha256": sha256_file(args.output / name),
                           "bytes": (args.output / name).stat().st_size}
                    for name in ["groups.jsonl", "assignments.jsonl", "fingerprints.jsonl"]},
        "executed_source_sha256": source_hashes,
        "reversed_input_reproduction_byte_identical": True,
        "training_launched": False, "policy_evaluations": 0,
        "old_test_arrays_read_for_structural_identity": True,
        "old_test_performance_measured": False,
    })
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "manifest_digest": manifest["digest"],
                      "counts": manifest["counts"], "grouping": grouping,
                      "old_training_exposure": previously_seen,
                      "elapsed_seconds": time.monotonic() - started}), flush=True)
    (args.output / "manifest.sha256").write_text(
        hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "data", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "manifest.json").exists():
            raise FileExistsError("completed split already exists; use a new output directory")
        main(args)
