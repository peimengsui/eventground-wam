"""Versioned, fixed-horizon dual-camera references for Pixel and JEPA CPT.

The binary rows contain references, never images, actions, or model features.
All multi-byte integers are little endian; SHA-256 fields are 32 raw bytes.
Only the Python standard library is required for generation and verification.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from eventground_wam.demonstration_contracts import (
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_MANIFEST_SHA256,
    canonical_json_bytes,
    contract_digest,
    seal_contract,
)

PAIR_VERSION = "trajectory-v2-future-pairs-v1"
HORIZONS = (2, 4, 8, 16, 32)
SPLITS = {"train": 0, "validation": 1}
IDENTITY = struct.Struct("<IIIBB")
RECORD = struct.Struct("<IIIBB32s32s")
DOMAIN = b"eventground_wam.future_image_pair.v1\0"
CAMERAS = ("observation.images.front", "observation.images.wrist")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sealed(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("digest") != contract_digest(value):
        raise ValueError(f"invalid contract digest: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pair_rule() -> dict:
    """The data rule is independent of predictor architecture and sampling weights."""
    return {
        "version": PAIR_VERSION,
        "horizons": list(HORIZONS),
        "control_frequency_hz": 20,
        "splits": SPLITS,
        "cameras": list(CAMERAS),
        "anchors": "Every integer t with 0 <= t < episode_length - h",
        "target": "Frame t+h from the same episode and both cameras",
        "actions": "Raw source actions [t, t+h); exactly h actions, target action excluded",
        "conditioning": "Instruction, state and both images at t; no future observations",
        "order": "Per split: ascending episode_index, then horizon, then anchor",
        "record_struct": RECORD.format,
        "record_bytes": RECORD.size,
        "record_fields": ["episode_index", "anchor", "target", "horizon", "split_code",
                          "source_digest", "pair_digest"],
        "pair_digest": "SHA256(ASCII eventground_wam.future_image_pair.v1 + NUL + "
                       "raw source_digest + struct.pack('<IIIBB', episode,anchor,target,h,split))",
        "next_event_pairs": 0,
        "test_payloads": "Forbidden; test assignment metadata is used only for isolation checks",
        "transforms": "None; source 256x256 dual-camera frame references, no cache or resize",
        "weighting": "Index coverage only; training mixture and exposure budget remain unfrozen",
        "policy_samples": "Does not filter B0 action anchors; later auxiliary losses need masks",
    }


def expand_split_groups(groups: Iterable[dict]) -> list[dict]:
    """Expand exactly the frozen assignment schema without loading test payloads."""
    rows = []
    seen_episodes: set[int] = set()
    seen_groups: set[str] = set()
    for group in groups:
        digest = group["group_digest"]
        _digest_bytes(digest)
        if digest in seen_groups:
            raise ValueError("duplicate or cross-split trajectory group")
        seen_groups.add(digest)
        if group["split"] not in (*SPLITS, "test"):
            raise ValueError("unknown split")
        length = group["episode_length"]
        if type(length) is not int or length < 8 or not group["episode_indices"]:
            raise ValueError("invalid episode length or empty group")
        for episode in group["episode_indices"]:
            if type(episode) is not int or episode < 0 or episode in seen_episodes:
                raise ValueError("invalid or duplicate episode")
            seen_episodes.add(episode)
            rows.append({"episode_index": episode, "task_key": group["task_key"],
                         "split": group["split"], "group_digest": digest,
                         "episode_length": length, "valid_action_chunks": length - 7})
    if not rows:
        raise ValueError("empty split catalog")
    return sorted(rows, key=lambda row: row["episode_index"])


def _digest_bytes(value: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError("invalid SHA-256")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("invalid SHA-256") from error
    if len(raw) != 32:
        raise ValueError("invalid SHA-256")
    return raw


def _relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or not path.parts:
        raise ValueError("unsafe source path")


def validate_sources(sources: Iterable[dict], assignments: list[dict]) -> list[dict]:
    if len({row["episode_index"] for row in assignments}) != len(assignments):
        raise ValueError("duplicate assignment episode")
    group_splits: dict[str, str] = {}
    for row in assignments:
        if row["split"] not in (*SPLITS, "test"):
            raise ValueError("unknown split")
        if group_splits.setdefault(row["group_digest"], row["split"]) != row["split"]:
            raise ValueError("cross-split trajectory group")
    expected = {row["episode_index"]: row for row in assignments if row["split"] in SPLITS}
    rows = sorted(sources, key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in rows] != sorted(expected):
        raise ValueError("source coverage differs from train/validation assignments")
    for source in rows:
        if source.get("digest") != contract_digest(source):
            raise ValueError("invalid source digest")
        if source.get("artifact_type") != "eventground_wam.future_pair_source":
            raise ValueError("invalid source artifact type")
        if source.get("version") != PAIR_VERSION:
            raise ValueError("invalid future pair version")
        assignment = expected[source["episode_index"]]
        for key in ("episode_index", "episode_length", "task_key", "split", "group_digest"):
            if source[key] != assignment[key]:
                raise ValueError(f"source assignment mismatch: {key}")
        if source.get("assignment_digest") != hashlib.sha256(
            canonical_json_bytes(assignment)
        ).hexdigest():
            raise ValueError("source assignment digest mismatch")
        if source.get("dataset") != {
            "repo_id": SOURCE_DATASET_REPO_ID, "revision": SOURCE_DATASET_REVISION,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        }:
            raise ValueError("source dataset changed")
        if (not isinstance(source.get("instruction"), str) or not source["instruction"]
                or type(source.get("task_index")) is not int or source["task_index"] < 0):
            raise ValueError("invalid source instruction/task")
        _digest_bytes(source["split_manifest_sha256"])
        _digest_bytes(source["group_digest"])
        files = source["files"]
        if set(files) != {"parquet", *CAMERAS}:
            raise ValueError("source must contain telemetry and both cameras")
        episode = source["episode_index"]
        stem = f"chunk-{episode // 1000:03d}"
        expected_paths = {"parquet": f"data/{stem}/episode_{episode:06d}.parquet"}
        expected_paths.update({key: f"videos/{stem}/{key}/episode_{episode:06d}.mp4"
                              for key in CAMERAS})
        for key, value in files.items():
            _relative_path(value["path"])
            if value["path"] != expected_paths[key]:
                raise ValueError("camera or episode source path mismatch")
            _digest_bytes(value["sha256"])
            if type(value["bytes"]) is not int or value["bytes"] <= 0:
                raise ValueError("invalid source file size")
    return rows


def coverage(sources: list[dict]) -> dict:
    result: dict[str, Any] = {}
    for split in SPLITS:
        rows = [row for row in sources if row["split"] == split]
        if not rows:
            raise ValueError(f"no {split} sources")
        horizon_counts = {str(h): sum(max(0, row["episode_length"] - h) for row in rows)
                          for h in HORIZONS}
        tasks: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            for h in HORIZONS:
                tasks[row["task_key"]][str(h)] += max(0, row["episode_length"] - h)
        result[split] = {
            "episodes": len(rows), "groups": len({row["group_digest"] for row in rows}),
            "tasks": len(tasks), "source_frames": sum(row["episode_length"] for row in rows),
            "pairs": sum(horizon_counts.values()), "pairs_by_horizon": horizon_counts,
            "pairs_by_camera": {key: sum(horizon_counts.values()) for key in CAMERAS},
            "excluded_tail_anchors_by_horizon": {
                str(h): sum(min(h, row["episode_length"]) for row in rows) for h in HORIZONS},
            "per_task_pairs_by_horizon": dict(sorted(tasks.items())),
        }
    return result


def pair_bytes(source: dict, anchor: int, horizon: int) -> bytes:
    if (type(anchor) is not int or type(horizon) is not int or horizon not in HORIZONS
            or not 0 <= anchor < source["episode_length"] - horizon
            or source["split"] not in SPLITS):
        raise ValueError("pair violates split, horizon or same-episode bounds")
    identity = IDENTITY.pack(source["episode_index"], anchor, anchor + horizon,
                             horizon, SPLITS[source["split"]])
    source_digest = _digest_bytes(source["digest"])
    digest = hashlib.sha256(DOMAIN + source_digest + identity).digest()
    return identity + source_digest + digest


def write_bundle(output: Path, sources: Iterable[dict], assignments: list[dict],
                 provenance: dict) -> dict:
    """Write deterministic binary indexes and sources; refuse existing output."""
    sources = validate_sources(sources, assignments)
    counts = coverage(sources)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "sources.jsonl").open("xb") as handle:
        for source in sources:
            handle.write(canonical_json_bytes(source) + b"\n")
    outputs = {}
    for split, split_code in SPLITS.items():
        name = f"{split}.bin"
        with (output / name).open("xb", buffering=1024 * 1024) as handle:
            for source in sources:
                if source["split"] != split:
                    continue
                # Bind invariants once per episode; preserve exact pair_bytes format.
                source_digest = _digest_bytes(source["digest"])
                prefix = DOMAIN + source_digest
                for horizon in HORIZONS:
                    for anchor in range(max(0, source["episode_length"] - horizon)):
                        identity = IDENTITY.pack(source["episode_index"], anchor,
                                                 anchor + horizon, horizon, split_code)
                        handle.write(identity + source_digest
                                     + hashlib.sha256(prefix + identity).digest())
        outputs[name] = {"sha256": sha256_file(output / name),
                         "bytes": (output / name).stat().st_size,
                         "records": counts[split]["pairs"]}
    outputs["sources.jsonl"] = {"sha256": sha256_file(output / "sources.jsonl"),
                                "bytes": (output / "sources.jsonl").stat().st_size,
                                "records": len(sources)}
    manifest = seal_contract({
        "artifact_type": "eventground_wam.future_pair_index", "schema_version": 1,
        "version": PAIR_VERSION, "rule": pair_rule(), "provenance": provenance,
        "counts": counts, "outputs": outputs,
        "test_episodes_excluded": sum(row["split"] == "test" for row in assignments),
        "test_payloads_read": 0, "next_event_pairs": 0,
        "shared_consumers": ["Pixel CPT", "JEPA CPT"],
    })
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_bundle(root: Path, assignments: list[dict], expected_provenance: dict) -> dict:
    """Read every row, recompute its identity, and check exact coverage/order."""
    manifest = read_sealed(root / "manifest.json")
    if (manifest.get("rule") != pair_rule() or manifest.get("version") != PAIR_VERSION
            or manifest.get("artifact_type") != "eventground_wam.future_pair_index"
            or manifest.get("schema_version") != 1
            or manifest.get("provenance") != expected_provenance):
        raise ValueError("manifest rule or provenance differs")
    if set(manifest["outputs"]) != {"sources.jsonl", "train.bin", "validation.bin"}:
        raise ValueError("unexpected index output files")
    for name, pin in manifest["outputs"].items():
        path = root / name
        if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
            raise ValueError(f"output hash or size mismatch: {name}")
    sources = validate_sources(
        [json.loads(line) for line in (root / "sources.jsonl").read_text().splitlines()],
        assignments,
    )
    if len(sources) != manifest["outputs"]["sources.jsonl"]["records"]:
        raise ValueError("source record count mismatch")
    if manifest["counts"] != coverage(sources):
        raise ValueError("coverage counts differ")
    if any(source["split_manifest_sha256"] != expected_provenance["split_manifest_sha256"]
           for source in sources):
        raise ValueError("source split manifest differs")
    if (manifest["test_payloads_read"] != 0 or manifest["next_event_pairs"] != 0
            or manifest["test_episodes_excluded"] != sum(
                row["split"] == "test" for row in assignments)):
        raise ValueError("test/event scope changed")
    for split, split_code in SPLITS.items():
        count = 0
        with (root / f"{split}.bin").open("rb", buffering=1024 * 1024) as handle:
            for source in sources:
                if source["split"] != split:
                    continue
                source_digest = bytes.fromhex(source["digest"])
                for horizon in HORIZONS:
                    for anchor in range(max(0, source["episode_length"] - horizon)):
                        raw = handle.read(RECORD.size)
                        if len(raw) != RECORD.size:
                            raise ValueError("truncated pair stream")
                        ep, start, target, h, code, source_sha, pair_sha = RECORD.unpack(raw)
                        if ((ep, start, target, h, code) !=
                                (source["episode_index"], anchor, anchor + horizon,
                                 horizon, split_code) or source_sha != source_digest):
                            raise ValueError("pair bounds, source, order or coverage mismatch")
                        if hashlib.sha256(DOMAIN + source_sha + raw[:IDENTITY.size]).digest() != pair_sha:
                            raise ValueError("invalid pair digest")
                        count += 1
            if handle.read(1):
                raise ValueError("unexpected trailing pair bytes")
        if count != manifest["outputs"][f"{split}.bin"]["records"]:
            raise ValueError("pair record count mismatch")
    return seal_contract({"artifact_type": "eventground_wam.future_pair_verification",
                          "status": "verified", "manifest_digest": manifest["digest"],
                          "counts": manifest["counts"], "all_pair_digests_verified": True,
                          "all_source_foreign_keys_verified": True,
                          "exact_order_coverage_and_bounds_verified": True,
                          "test_payloads_read": 0})


def read_pair(root: Path, split: str, ordinal: int) -> dict:
    """Resolve a row to raw file references, usable by both future predictors.

    Consumers must validate the bundle once before sampling. This function checks
    the selected row's digest and source relationship, not the entire file.
    """
    if split not in SPLITS or type(ordinal) is not int or ordinal < 0:
        raise ValueError("invalid split or ordinal")
    manifest = read_sealed(root / "manifest.json")
    if manifest["rule"] != pair_rule():
        raise ValueError("unsupported pair rule")
    if ordinal >= manifest["counts"][split]["pairs"]:
        raise IndexError(ordinal)
    with (root / f"{split}.bin").open("rb") as handle:
        handle.seek(ordinal * RECORD.size)
        raw = handle.read(RECORD.size)
    if len(raw) != RECORD.size:
        raise ValueError("truncated pair stream")
    ep, anchor, target, horizon, code, source_sha, pair_sha = RECORD.unpack(raw)
    with (root / "sources.jsonl").open() as handle:
        source = next((row for row in map(json.loads, handle) if row["episode_index"] == ep), None)
    if (source is None or source["digest"] != contract_digest(source)
            or source["digest"] != source_sha.hex() or source["split"] != split
            or code != SPLITS[split] or pair_bytes(source, anchor, horizon) != raw):
        raise ValueError("invalid pair/source identity")
    return {"episode_index": ep, "anchor": anchor, "target": target, "horizon": horizon,
            "split": split, "action_start": anchor, "action_stop_exclusive": target,
            "pair_digest": pair_sha.hex(), "source": source}
