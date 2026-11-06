#!/usr/bin/env python3
"""Prepare, independently regenerate, and verify the Week 5/6 future-pair index."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eventground_wam.demonstration_contracts import (
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_MANIFEST_SHA256,
    canonical_json_bytes,
    seal_contract,
)
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.future_pairs import (
    CAMERAS,
    PAIR_VERSION,
    SPLITS,
    expand_split_groups,
    pair_rule,
    read_sealed,
    sha256_file,
    verify_bundle,
    write_bundle,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "scripts/prepare_future_pairs.py", "scripts/validate_future_pairs.py",
    "src/eventground_wam/__init__.py", "src/eventground_wam/future_pairs.py",
    "src/eventground_wam/demonstration_contracts.py",
    "src/eventground_wam/demonstration_reader.py",
)


def load_inputs(root: Path, data_root: Path) -> tuple:
    config = read_sealed(root / "configs/datasets/trajectory_v2_future_pairs_v1.json")
    if config["rule"] != pair_rule():
        raise ValueError("config and executable data rules differ")
    pins = config["inputs"]
    split_root = root / "configs/datasets/libero-plus-trajectory-v2"
    for name in ("manifest.json", "groups.jsonl"):
        if sha256_file(split_root / name) != pins["split_files"][name]:
            raise ValueError(f"frozen v2 split changed: {name}")
    split_manifest = read_sealed(split_root / "manifest.json")
    groups = [json.loads(line) for line in (split_root / "groups.jsonl").read_text().splitlines()]
    assignments = expand_split_groups(groups)
    encoded = b"".join(canonical_json_bytes(row) + b"\n" for row in assignments)
    if hashlib.sha256(encoded).hexdigest() != pins["assignments_sha256"]:
        raise ValueError("expanded assignments differ from frozen v2")
    if [row["episode_index"] for row in assignments] != list(range(14347)):
        raise ValueError("source assignment coverage differs")
    for split, expected in split_manifest["counts"].items():
        rows = [row for row in assignments if row["split"] == split]
        if (len(rows) != expected["episodes"]
                or sum(row["episode_length"] for row in rows) != expected["frames"]
                or len({row["group_digest"] for row in rows}) != expected["groups"]):
            raise ValueError("split coverage differs")
    snapshot = data_root / f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}"
    inventory = snapshot / "source-manifest.tsv"
    if sha256_file(inventory) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source file inventory changed")
    files = {}
    for line in inventory.read_text().splitlines():
        name, size, digest = line.split("\t")
        if name in files:
            raise ValueError("duplicate source inventory entry")
        files[name] = {"path": name, "bytes": int(size), "sha256": digest}
    repository = snapshot / "repo"
    for name, digest in pins["metadata_sha256"].items():
        if sha256_file(repository / "meta" / name) != digest:
            raise ValueError(f"source metadata changed: {name}")
    panel_path = root / "evaluation/splits/libero-clean-v1/clean-40.json"
    if sha256_file(panel_path) != pins["canonical_task_panel_sha256"]:
        raise ValueError("canonical task mapping changed")
    panel = read_sealed(panel_path)
    tasks = {row["instruction"]: f'{row["suite"]}/task-{row["runtime_task_index"]:02d}'
             for row in panel["tasks"]}
    if len(tasks) != 40:
        raise ValueError("canonical task panel is incomplete")
    reader = DemonstrationReader(repository)
    episodes = list(reader.iter_episodes())  # Metadata only, including test assignments.
    if len(episodes) != len(assignments):
        raise ValueError("source episode coverage differs")
    for episode, assignment in zip(episodes, assignments, strict=True):
        if (episode.episode_index != assignment["episode_index"]
                or episode.length != assignment["episode_length"]
                or tasks[episode.instruction] != assignment["task_key"]):
            raise ValueError("source descriptor and v2 assignment disagree")
    return config, assignments, reader, episodes, files


def video_header(path: Path, length: int) -> None:
    import av

    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError(f"unexpected video streams: {path}")
        stream = container.streams.video[0]
        codec = stream.codec_context
        if (stream.frames != length or stream.average_rate != 20
                or (codec.width, codec.height) != (256, 256)
                or codec.name not in {"av1", "libdav1d"}):
            raise ValueError(f"video header/frame count differs: {path}")


def inspect_source(reader, episode, assignment, inventory, split_sha) -> dict:
    # Reject test before opening/stat'ing/hash-reading any episode payload.
    if assignment["split"] not in SPLITS:
        raise ValueError("test payload access is forbidden")
    files = {}
    for key, path in (("parquet", episode.parquet_path), *episode.video_paths):
        relative = path.relative_to(reader.repository_root).as_posix()
        pin = inventory[relative]
        if not pin["sha256"] or path.stat().st_size != pin["bytes"]:
            raise ValueError(f"source size/hash pin missing or changed: {relative}")
        if sha256_file(path) != pin["sha256"]:
            raise ValueError(f"source bytes changed: {relative}")
        files[key] = pin
        if key in CAMERAS:
            video_header(path, episode.length)
    # Complete Parquet row/schema/time alignment, without decoding every video.
    reader.validate_episode(episode)
    return seal_contract({
        "artifact_type": "eventground_wam.future_pair_source", "version": PAIR_VERSION,
        "episode_index": episode.episode_index, "episode_length": episode.length,
        "task_index": episode.task_index, "task_key": assignment["task_key"],
        "instruction": episode.instruction, "group_digest": assignment["group_digest"],
        "split": assignment["split"], "split_manifest_sha256": split_sha,
        "assignment_digest": hashlib.sha256(canonical_json_bytes(assignment)).hexdigest(),
        "dataset": {"repo_id": SOURCE_DATASET_REPO_ID, "revision": SOURCE_DATASET_REVISION,
                    "source_manifest_sha256": SOURCE_MANIFEST_SHA256},
        "files": files,
    })


def prepare(args) -> None:
    started = time.monotonic()
    output = args.output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite an existing result or partial run")
    config, assignments, reader, episodes, inventory = load_inputs(ROOT, args.data_root)
    provenance = {
        "config_digest": config["digest"],
        "split_manifest_sha256": config["inputs"]["split_files"]["manifest.json"],
        "assignments_sha256": config["inputs"]["assignments_sha256"],
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "metadata_sha256": config["inputs"]["metadata_sha256"],
        "source_code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "runtime_lock_sha256": sha256_file(ROOT / "environments/demonstrations/uv.lock"),
    }
    partial.mkdir(parents=True)

    def progress(phase, **extra):
        row = {"phase": phase, "elapsed_seconds": round(time.monotonic() - started, 3), **extra}
        temporary = partial / "progress.json.tmp"
        write_json(temporary, row)
        temporary.replace(partial / "progress.json")
        print(json.dumps(row), flush=True)

    progress("verify-source-payloads", selected_episodes=12963, test_payloads_read=0)
    selected = [(episode, assignment) for episode, assignment in zip(episodes, assignments)
                if assignment["split"] in SPLITS]

    def inspect(item):
        return inspect_source(reader, *item, inventory, provenance["split_manifest_sha256"])

    sources = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for source in pool.map(inspect, selected):
            sources.append(source)
            if len(sources) % 1000 == 0:
                progress("verify-source-payloads", completed_episodes=len(sources),
                         selected_episodes=len(selected), test_payloads_read=0)
    # Deterministic task x split sample, including both cameras and every frame.
    # Full media checksums/headers above cover all selected videos; decoding here
    # tests actual RGB conversion and synchronization without claiming a full decode.
    chosen = {}
    for episode, assignment in selected:
        chosen.setdefault((assignment["split"], assignment["task_key"]), episode)
    decoded = []
    progress("decode-alignment-smoke", episodes=len(chosen))
    for key, episode in sorted(chosen.items()):
        rgb = {camera: hashlib.sha256() for camera in CAMERAS}
        frames = 0
        for step in reader.iter_episode_steps(episode):
            for camera, frame in zip(CAMERAS, (step.front_frame, step.wrist_frame), strict=True):
                converted = frame.payload.reformat(format="rgb24")
                if (converted.width, converted.height) != (256, 256):
                    raise ValueError("decoded RGB shape differs")
                plane = converted.planes[0]
                pixels = memoryview(plane)
                for line in range(256):
                    start = line * plane.line_size
                    rgb[camera].update(pixels[start:start + 256 * 3])
            frames += 1
        decoded.append({"split": key[0], "task_key": key[1], "episode_index": episode.episode_index,
                        "frames": frames, "rgb_sequence_sha256": {
                            camera: value.hexdigest() for camera, value in rgb.items()}})
    if len(decoded) != 80:
        raise ValueError("decode smoke must cover all 40 tasks in train and validation")
    progress("generate-primary")
    primary = write_bundle(partial / "primary", sources, assignments, provenance)
    progress("verify-every-pair")
    verification = verify_bundle(partial / "primary", assignments, provenance)
    progress("regenerate-from-source-metadata")
    second_config, second_assignments, second_reader, second_episodes, second_inventory = load_inputs(
        ROOT, args.data_root)
    if second_config != config or second_assignments != assignments:
        raise ValueError("inputs changed during preparation")
    # Reconstruct the source registry independently from the inventory/metadata.
    # The first pass verified all inventory hashes; do not repeat media decoding.
    replica_sources = []
    for episode, assignment in reversed(list(zip(second_episodes, second_assignments))):
        if assignment["split"] not in SPLITS:
            continue
        files = {key: second_inventory[path.relative_to(second_reader.repository_root).as_posix()]
                 for key, path in (("parquet", episode.parquet_path), *episode.video_paths)}
        replica_sources.append(seal_contract({
            "artifact_type": "eventground_wam.future_pair_source", "version": PAIR_VERSION,
            **{key: assignment[key] for key in
               ("episode_index", "episode_length", "task_key", "group_digest", "split")},
            "task_index": episode.task_index, "instruction": episode.instruction,
            "assignment_digest": hashlib.sha256(canonical_json_bytes(assignment)).hexdigest(),
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "dataset": {"repo_id": SOURCE_DATASET_REPO_ID, "revision": SOURCE_DATASET_REVISION,
                        "source_manifest_sha256": SOURCE_MANIFEST_SHA256}, "files": files,
        }))
    replica = write_bundle(partial / "replica", replica_sources, second_assignments, provenance)
    if replica != primary:
        raise ValueError("independent regeneration differs")
    for name in (*primary["outputs"], "manifest.json"):
        if sha256_file(partial / "primary" / name) != sha256_file(partial / "replica" / name):
            raise ValueError(f"regenerated artifact bytes differ: {name}")
    if any(sha256_file(ROOT / name) != digest
           for name, digest in provenance["source_code_sha256"].items()):
        raise ValueError("executed source changed during run")
    import av
    import pyarrow

    receipt = seal_contract({
        "artifact_type": "eventground_wam.future_pair_preparation", "schema_version": 1,
        "version": PAIR_VERSION, "status": "complete", "config_digest": config["digest"],
        "manifest_digest": primary["digest"], "manifest_sha256": sha256_file(
            partial / "primary/manifest.json"),
        "verification": verification, "counts": primary["counts"],
        "source_verification": {"episodes": len(sources), "payload_files_hashed": len(sources) * 3,
                                "parquet_rows_validated": sum(s["episode_length"] for s in sources),
                                "video_headers_validated": len(sources) * 2,
                                "decoded_episode_count": len(decoded), "decoded_episodes": decoded,
                                "full_video_decode": False, "test_payloads_read": 0},
        "determinism": {"independent_metadata_regeneration": True,
                        "reversed_episode_input": True, "byte_identical_files": [
                            *primary["outputs"], "manifest.json"]},
        "provenance": provenance,
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "av": av.__version__, "pyarrow": pyarrow.__version__},
        "training_launched": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    write_json(partial / "preparation.json", receipt)
    progress("complete", manifest_digest=primary["digest"])
    partial.rename(output)
    print(json.dumps({"status": "complete", "output": str(output),
                      "receipt_digest": receipt["digest"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "data/eventground-wam/libero-plus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    prepare(args)
