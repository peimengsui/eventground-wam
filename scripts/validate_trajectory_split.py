#!/usr/bin/env python3
"""Independently verify the frozen v2 split against its numeric fingerprint audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.demonstration_contracts import (
    SOURCE_EPISODE_COUNT,
    SOURCE_FRAME_COUNT,
    canonical_json_bytes,
    seal_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path, evidence: Path) -> dict:
    split_root = root / "configs/datasets/libero-plus-trajectory-v2"
    manifest = read_sealed(split_root / "manifest.json")
    rule_path = root / "configs/datasets/libero_plus_trajectory_v2.json"
    rule = read_sealed(rule_path)
    assert rule["digest"] == manifest["rule_digest"]
    assert sha256_file(rule_path) == manifest["rule_sha256"]
    for name, pin in manifest["outputs"].items():
        path = split_root / name if name == "groups.jsonl" else evidence / name
        assert sha256_file(path) == pin["sha256"] and path.stat().st_size == pin["bytes"], name
    groups = [json.loads(line) for line in (split_root / "groups.jsonl").read_text().splitlines()]
    fingerprints = [json.loads(line) for line in (evidence / "fingerprints.jsonl").read_text().splitlines()]
    assignments = [json.loads(line) for line in (evidence / "assignments.jsonl").read_text().splitlines()]
    expected_ids = set(range(SOURCE_EPISODE_COUNT))
    for rows in (fingerprints, assignments):
        assert len(rows) == SOURCE_EPISODE_COUNT
        assert {r["episode_index"] for r in rows} == expected_ids
    fp = {r["episode_index"]: r for r in fingerprints}
    assignment = {r["episode_index"]: r for r in assignments}
    seen = set()
    per_task = defaultdict(list)
    for group in groups:
        members = group["episode_indices"]
        assert members and not seen.intersection(members)
        assert len(set(members)) == len(members)
        seen.update(members)
        assert {fp[i]["task_key"] for i in members} == {group["task_key"]}
        assert {fp[i]["episode_length"] for i in members} == {group["episode_length"]}
        digests = sorted({fp[i]["state_action_digest"] for i in members})
        assert group["numeric_digests"] == digests
        payload = {"split_id": rule["split_id"], "task_key": group["task_key"],
                   "numeric_digests": digests}
        assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == group["group_digest"]
        for i in members:
            assert assignment[i] == {
                "episode_index": i, "task_key": group["task_key"], "split": group["split"],
                "group_digest": group["group_digest"], "episode_length": group["episode_length"],
                "valid_action_chunks": group["episode_length"] - 7,
            }
        per_task[group["task_key"]].append(group)
    assert seen == expected_ids and len(per_task) == 40
    names = ("train", "validation", "test")
    for task, selected in per_task.items():
        def rank(group, task=task):
            payload = {"namespace": rule["namespace"], "task_key": task,
                       "group_digest": group["group_digest"]}
            return hashlib.sha256(canonical_json_bytes(payload)).hexdigest(), group["group_digest"]
        ordered = sorted(selected, key=rank)
        count = len(ordered)
        quotas = [count * w // 10 for w in (8, 1, 1)]
        remainders = sorted(range(3), key=lambda i: (-(count * (8, 1, 1)[i] % 10), i))
        for i in remainders[:count - sum(quotas)]:
            quotas[i] += 1
        expected_splits = [name for name, n in zip(names, quotas, strict=True) for _ in range(n)]
        assert [g["split"] for g in ordered] == expected_splits
        for name in names:
            rows = [r for r in assignments if r["task_key"] == task and r["split"] == name]
            totals = {"groups": sum(g["split"] == name for g in selected), "episodes": len(rows),
                      "frames": sum(r["episode_length"] for r in rows),
                      "valid_action_chunks": sum(r["valid_action_chunks"] for r in rows)}
            assert totals == manifest["per_task"][task][name]
            assert totals["groups"] > 0
    for name in names:
        totals = {key: sum(t[name][key] for t in manifest["per_task"].values())
                  for key in ("groups", "episodes", "frames", "valid_action_chunks")}
        assert totals == manifest["counts"][name]
    assert sum(r["episode_length"] for r in fingerprints) == SOURCE_FRAME_COUNT
    for field in ("state_action_digest", "action_digest"):
        by_digest = defaultdict(set)
        for row in fingerprints:
            by_digest[row[field]].add(assignment[row["episode_index"]]["split"])
        assert all(len(splits) == 1 for splits in by_digest.values())
    for a, b in manifest["grouping"]["near_edges_between_distinct_exact_sequences"]:
        assert assignment[a]["group_digest"] == assignment[b]["group_digest"]
    for old in names:
        counts = Counter(assignment[r["episode_index"]]["split"]
                         for r in fingerprints if r["old_split"] == old)
        assert dict(counts) == manifest["old_to_new_episode_counts"][old]
    return seal_contract({
        "artifact_type": "eventground_wam.trajectory_split_verification",
        "status": "verified", "split_id": rule["split_id"],
        "manifest_digest": manifest["digest"], "rule_digest": rule["digest"],
        "counts": manifest["counts"], "task_count_in_every_split": 40,
        "checks": {"source_episode_coverage_exactly_once": True,
                   "all_group_numeric_identities_recomputed": True,
                   "independent_hash_ranking_and_quotas_match": True,
                   "per_task_and_global_totals_match": True,
                   "zero_exact_state_action_overlap": True,
                   "zero_exact_action_only_overlap": True,
                   "recorded_near_edges_stay_within_groups": True,
                   "output_hashes_and_canonical_digests_valid": True,
                   "historical_split_migration_recomputed": True},
        "near_check_scope": "Generator scanned 9640 equal-length distinct sequence pairs. "
                            "This offline check verifies edge placement, not raw-array distances.",
        "training_launched": False, "policy_evaluations": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "outputs/week4-trajectory-split-v2")
    parser.add_argument("--write-receipt", type=Path)
    args = parser.parse_args()
    result = verify(ROOT, args.evidence)
    if args.write_receipt:
        args.write_receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, indent=2))
