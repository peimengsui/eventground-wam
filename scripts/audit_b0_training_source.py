#!/usr/bin/env python3
"""Audit source membership, task balance, no-ops and exact numeric duplication."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from eventground_wam.b0_data import load_frozen_b0_split_index
from eventground_wam.b0_diagnostics import noop_mask, numeric_digest
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader

CATEGORY_REPO = "katsukiono/libero-plus-episode-categories"
CATEGORY_REVISION = "64d4c3244865528f8842e4a2ccc3e7a12fc51530"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def fetch(url, destination):
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(raw)
        temporary.replace(destination)
    return destination.read_bytes()


def numeric_episode(pair):
    episode, assignment = pair
    table = pq.read_table(episode.parquet_path, columns=["action", "observation.state"],
                          use_threads=False)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    assert action.shape == (episode.length, 7) and state.shape == (episode.length, 8)
    assert np.isfinite(action).all() and np.isfinite(state).all()
    assert set(np.unique(action[:, 6])) <= {-1., 1.}
    return {
        "episode_index": episode.episode_index, "task_index": episode.task_index,
        "instruction": episode.instruction, "split": assignment.split,
        "length": episode.length, "chunks": max(0, episode.length - 7),
        "noop_frames": int(noop_mask(action).sum()),
        "zero_motion_frames": int((np.linalg.norm(action[:, :6], axis=1) < 1e-4).sum()),
        "gripper_transitions": int(np.sum(action[1:, 6] != action[:-1, 6])),
        "closed_frames": int(np.sum(action[:, 6] == 1)),
        "out_of_range_motion_components": int(np.sum(np.abs(action[:, :6]) > 1)),
        "action_digest": numeric_digest(action),
        "state_action_digest": numeric_digest(state, action),
        "source_parquet_sha256": sha256_file(episode.parquet_path),
    }


def duplicate_summary(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["instruction"], row[key])].append(row)
    duplicated = [v for v in groups.values() if len(v) > 1]
    crossing = [v for v in duplicated if len({r["split"] for r in v}) > 1]
    train_groups = [[r for r in v if r["split"] == "train"] for v in groups.values()]
    train_groups = [v for v in train_groups if v]
    return {
        "scope": "train and validation numeric sequences only; held-out test arrays unread",
        "identity": "canonical task instruction and exact float32 sequence, including shape",
        "train_unique_sequences": len(train_groups),
        "train_duplicate_excess_episodes": sum(len(v) - 1 for v in train_groups),
        "train_episodes_in_duplicate_groups": sum(len(v) for v in train_groups if len(v) > 1),
        "train_max_multiplicity": max(map(len, train_groups)),
        "train_validation_crossing_groups": len(crossing),
        "validation_episodes_matching_train": sum(
            sum(r["split"] == "validation" for r in v) for v in crossing),
        "examples": [{"instruction": v[0]["instruction"],
                      "episodes": [{k: r[k] for k in ("episode_index", "split")} for r in v]}
                     for v in crossing[:5]],
        "caution": "Numeric equality is not RGB equality or proof of duplicate visual experience.",
    }


def main(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    index = load_frozen_b0_split_index(args.data / "derived/frozen/week2-dataset-v1")
    repository = args.data / ("datasets/libero-plus-lerobot/snapshots/"
                              "22c57433fef692b5b9ecc0795344daac7fa867a5/repo")
    reader = DemonstrationReader(repository)
    episodes = list(reader.iter_episodes())
    clean = read_sealed(args.source / "evaluation/splits/libero-clean-v1/clean-40.json")
    task_map = {r["instruction"]: f'{r["suite"]}/task-{r["runtime_task_index"]:02d}'
                for r in clean["tasks"]}
    assert len(task_map) == 40 and all(e.instruction in task_map for e in episodes)
    print("Fetching and pinning public metadata", flush=True)
    public = {}
    for suite in SUITES:
        repo = f"IPEC-COMMUNITY/{suite}_no_noops_1.0.0_lerobot"
        base = args.output / "public" / suite
        metadata = json.loads(fetch(f"https://huggingface.co/api/datasets/{repo}",
                                    base / "hub.json"))
        revision = metadata["sha"]
        files = {}
        for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
            path = base / name
            fetch(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/meta/{name}", path)
            files[name] = sha256_file(path)
        info = json.loads((base / "info.json").read_text())
        rows = [json.loads(line) for line in (base / "episodes.jsonl").read_text().splitlines()]
        counts = Counter(r["tasks"][0] for r in rows)
        assert sum(r["length"] for r in rows) == info["total_frames"]
        public[suite] = {"repo": repo, "revision": revision, "file_sha256": files,
                         "info": info, "task_episode_counts": dict(counts),
                         "task_frames": {task: sum(r["length"] for r in rows
                                                   if r["tasks"][0] == task) for task in counts}}
    categories = {}
    for name in ("README.md", "episode_categories.csv", "recover_episode_categories.py"):
        path = args.output / "public/categories" / name
        fetch(f"https://huggingface.co/datasets/{CATEGORY_REPO}/resolve/"
              f"{CATEGORY_REVISION}/{name}", path)
        categories[name] = sha256_file(path)
    rows = list(csv.DictReader(io.StringIO(
        (args.output / "public/categories/episode_categories.csv").read_text())))
    assert len(rows) == len(episodes) == 14347
    for episode, row in zip(episodes, rows, strict=True):
        assert int(row["episode_index"]) == episode.episode_index
        assert int(row["length"]) == episode.length and row["task"] == episode.instruction
        assert row["suite"] == task_map[episode.instruction].split("/")[0]
    category_counts = {}
    for split in ("all", "train", "validation", "test"):
        selected = [r for r in rows if split == "all"
                    or index.assignments[int(r["episode_index"])].split == split]
        category_counts[split] = dict(Counter(r["category"] for r in selected))
    metadata_by_task = {}
    for instruction, task in task_map.items():
        subset = [e for e in episodes if e.instruction == instruction]
        metadata_by_task[task] = {
            "instruction": instruction,
            "splits": {split: {
                "episodes": sum(index.assignments[e.episode_index].split == split for e in subset),
                "frames": sum(e.length for e in subset
                              if index.assignments[e.episode_index].split == split),
                "chunks": sum(max(0, e.length - 7) for e in subset
                              if index.assignments[e.episode_index].split == split),
            } for split in ("train", "validation", "test")}}
    numeric_pairs = [(e, index.assignments[e.episode_index]) for e in episodes
                     if index.assignments[e.episode_index].split != "test"]
    numeric = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(numeric_episode, numeric_pairs), 1):
            numeric.append(row)
            if i % 1000 == 0:
                print(json.dumps({"numeric_episodes": i, "total": len(numeric_pairs)}), flush=True)
    rows_path = args.output / "numeric-episodes.jsonl"
    rows_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in numeric))
    numeric_stats = {}
    for split in ("train", "validation"):
        selected = [r for r in numeric if r["split"] == split]
        fields = ("length", "chunks", "noop_frames", "zero_motion_frames",
                  "gripper_transitions", "closed_frames", "out_of_range_motion_components")
        numeric_stats[split] = {k: sum(r[k] for r in selected) for k in fields}
        numeric_stats[split]["episodes"] = len(selected)
        numeric_stats[split]["noop_fraction"] = (
            numeric_stats[split]["noop_frames"] / numeric_stats[split]["length"])
    assert numeric_stats["train"]["length"] == 1_796_601
    assert numeric_stats["train"]["chunks"] == 1_715_954
    report = seal_contract({
        "artifact_type": "eventground_wam.b0_training_source_audit", "status": "complete",
        "source": index.source_identity, "source_metadata": {
            "info": reader.info.__dict__, "metadata_sha256": {
                name: sha256_file(repository / "meta" / name)
                for name in ("info.json", "episodes.jsonl", "tasks.jsonl")}},
        "task_match": "All 40 canonical instructions match exactly; this is not scene matching.",
        "per_task": metadata_by_task, "numeric_statistics": numeric_stats,
        "numeric_row_file": str(rows_path), "numeric_row_sha256": sha256_file(rows_path),
        "duplicates": {key: duplicate_summary(numeric, key)
                       for key in ("action_digest", "state_action_digest")},
        "category_supplement": {
            "repo": CATEGORY_REPO, "revision": CATEGORY_REVISION, "file_sha256": categories,
            "all_14347_episode_id_task_length_suite_checks_pass": True,
            "counts": category_counts,
            "confidence": "PROVISIONAL independent community recovery, not original provenance; "
                          "RLDS path recovery not rerun. Task/length checks do not prove label semantics.",
            "env_semantics": "Unknown; do not label env as clean or any particular perturbation.",
        },
        "standard_libero_metadata": public,
        "standard_metadata_scope": "Published IPEC snapshots as fetched; paper-era file identity "
                                   "and numeric/visual overlap are not established.",
        "test_numeric_arrays_read": False,
        "noop_definition": "norm(action[:6]) <1e-4 and unchanged gripper versus previous source "
                           "frame; first frame tests motion only; no data removed.",
        "missing_original_provenance": ["perturbation category and severity", "original demo ID",
                                        "simulator initial-state ID", "success flags"],
        "elapsed_seconds": time.monotonic() - started,
    })
    atomic_json(args.output / "data-report.json", report)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "data", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
