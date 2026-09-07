#!/usr/bin/env python3
"""Generate, independently regenerate, validate, and freeze the Week 2 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eventground_wam.artifact_provenance import project_state
from eventground_wam.dataset_freeze import validate_and_seal_dataset_freeze
from eventground_wam.demonstration_contracts import SOURCE_EPISODE_COUNT, SOURCE_FRAME_COUNT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_VERSION = "week2-dataset-v1"
MINIMUM_FREE_BYTES = 30 * 1024**3
STAGE_FILES = {
    "candidate_generation": (
        "selection.json",
        "trajectory-contracts.jsonl",
        "event-candidates.jsonl",
        "review-items.jsonl",
        "feasibility-report.json",
    ),
    "event_extraction": (
        "event-candidates.jsonl",
        "event-intervals.jsonl",
        "event-decisions.jsonl",
        "manifest.json",
    ),
    "split_sample_indexing": (
        "split-assignments.jsonl",
        "multi-horizon-samples.jsonl",
        "split-sample-manifest.json",
    ),
}


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def default_data_root() -> Path:
    return expanded_path(
        os.environ.get(
            "EVENTGROUND_LIBERO_DATA_ROOT",
            "~/data/eventground-wam/libero-plus",
        )
    )


def parse_args() -> argparse.Namespace:
    data_root = default_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=expanded_path,
        default=data_root / "datasets" / "libero-plus-lerobot" / "current",
    )
    parser.add_argument(
        "--output-root",
        type=expanded_path,
        default=data_root / "derived" / "frozen" / DEFAULT_DATASET_VERSION,
    )
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--fingerprint-workers", type=int, default=8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only stages already recorded complete in freeze-progress.json",
    )
    return parser.parse_args()


def run_freeze(
    *,
    snapshot: Path,
    output_root: Path,
    dataset_version: str,
    fingerprint_workers: int,
    resume: bool,
) -> dict[str, Any]:
    """Run two complete builds and independently validate the primary artifacts."""

    if not dataset_version.strip():
        raise ValueError("dataset_version must be nonempty")
    if fingerprint_workers < 1:
        raise ValueError("fingerprint_workers must be positive")
    revision, clean = project_state(PROJECT_ROOT)
    if not clean:
        raise RuntimeError("dataset freezing requires a clean committed project revision")

    output_root = output_root.expanduser().resolve()
    if output_root.exists() and not resume:
        raise FileExistsError(f"dataset freeze refuses to overwrite {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root.parent).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            "dataset freeze requires at least 30 GiB free for primary and replica indexes"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    progress_path = output_root / "freeze-progress.json"
    freeze_manifest_path = output_root / "dataset-freeze-manifest.json"
    if freeze_manifest_path.exists():
        raise FileExistsError(f"dataset freeze manifest already exists: {freeze_manifest_path}")
    progress = _load_progress(progress_path, dataset_version, revision, resume=resume)

    for run_name in ("primary", "replica"):
        run_root = output_root / run_name
        candidate_dir = run_root / "candidates"
        event_dir = run_root / "events"
        split_dir = run_root / "splits"
        commands = {
            "candidate_generation": (
                candidate_dir,
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_event_labeling_pilot.py"),
                    "--snapshot",
                    str(snapshot),
                    "--output-dir",
                    str(candidate_dir),
                    "--all-episodes",
                    "--annotation-version",
                    f"{dataset_version}.candidates",
                ],
            ),
            "event_extraction": (
                event_dir,
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "generate_event_intervals.py"),
                    "--snapshot",
                    str(snapshot),
                    "--trajectory-contracts",
                    str(candidate_dir / "trajectory-contracts.jsonl"),
                    "--event-candidates",
                    str(candidate_dir / "event-candidates.jsonl"),
                    "--output-dir",
                    str(event_dir),
                    "--annotation-version",
                    f"{dataset_version}.events",
                ],
            ),
            "split_sample_indexing": (
                split_dir,
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "generate_split_sample_indexes.py"),
                    "--snapshot",
                    str(snapshot),
                    "--trajectory-index",
                    str(candidate_dir / "trajectory-contracts.jsonl"),
                    "--event-index",
                    str(event_dir / "event-intervals.jsonl"),
                    "--output-dir",
                    str(split_dir),
                    "--annotation-version",
                    f"{dataset_version}.splits",
                    "--namespace",
                    dataset_version,
                    "--fingerprint-workers",
                    str(fingerprint_workers),
                ],
            ),
        }
        for stage_name, (stage_dir, command) in commands.items():
            expected = tuple(stage_dir / name for name in STAGE_FILES[stage_name])
            prior = progress["stages"].get(run_name, {}).get(stage_name)
            if resume and prior is not None and all(path.is_file() for path in expected):
                print(f"[{run_name}] reusing completed {stage_name}", flush=True)
                continue
            conflicts = [path for path in expected if path.exists()]
            if conflicts:
                raise FileExistsError(
                    f"cannot safely resume incomplete {run_name}.{stage_name}: {conflicts}"
                )
            log_path = logs_dir / f"{run_name}-{stage_name}.log"
            print(f"[{run_name}] starting {stage_name}", flush=True)
            started = time.perf_counter()
            _run_stage(command, log_path)
            wall_seconds = time.perf_counter() - started
            if not all(path.is_file() for path in expected):
                raise RuntimeError(f"{run_name}.{stage_name} did not publish every artifact")
            measurement = _stage_measurement(stage_name, stage_dir, wall_seconds)
            progress["stages"].setdefault(run_name, {})[stage_name] = measurement
            _write_json_atomic(progress_path, progress)
            print(
                f"[{run_name}] completed {stage_name} in {wall_seconds:.2f}s",
                flush=True,
            )

    benchmark = {"stages": progress["stages"]}
    print("validating primary artifacts and byte-comparing the replica", flush=True)
    manifest = validate_and_seal_dataset_freeze(
        output_root / "primary",
        output_root / "replica",
        dataset_version=dataset_version,
        project_git_revision=revision,
        benchmark=benchmark,
    )
    _write_json_atomic(freeze_manifest_path, manifest)
    print(
        f"dataset freeze complete: {manifest['digest']} ({freeze_manifest_path})",
        flush=True,
    )
    return manifest


def _run_stage(command: Sequence[str], log_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        with log_path.open("x", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(f"freeze stage failed; tail of {log_path}:\n{tail}") from error


def _stage_measurement(stage: str, output_dir: Path, seconds: float) -> dict[str, Any]:
    measurement: dict[str, Any] = {
        "wall_seconds": seconds,
        "source_episode_count": SOURCE_EPISODE_COUNT,
        "source_frame_count": SOURCE_FRAME_COUNT,
        "episodes_per_second": SOURCE_EPISODE_COUNT / seconds,
        "source_frames_per_second": SOURCE_FRAME_COUNT / seconds,
    }
    if stage == "split_sample_indexing":
        manifest = json.loads(
            (output_dir / "split-sample-manifest.json").read_text(encoding="utf-8")
        )
        sample_count = manifest["outputs"]["multi_horizon_samples"]["record_count"]
        measurement["sample_record_count"] = sample_count
        measurement["samples_per_second"] = sample_count / seconds
        measurement["decoded_initial_camera_frame_count"] = SOURCE_EPISODE_COUNT * 2
        measurement["decoded_initial_camera_frames_per_second"] = (
            SOURCE_EPISODE_COUNT * 2 / seconds
        )
    return measurement


def _load_progress(
    path: Path,
    dataset_version: str,
    revision: str,
    *,
    resume: bool,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "artifact_type": "eventground_wam.dataset_freeze_progress",
            "dataset_version": dataset_version,
            "project_git_revision": revision,
            "stages": {},
        }
    if not resume:
        raise FileExistsError(f"freeze progress already exists: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resume invalid freeze progress {path}") from error
    if not isinstance(value, dict) or (
        value.get("dataset_version") != dataset_version
        or value.get("project_git_revision") != revision
        or not isinstance(value.get("stages"), Mapping)
    ):
        raise ValueError("freeze progress does not match the requested version and revision")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def main() -> None:
    args = parse_args()
    manifest = run_freeze(
        snapshot=args.snapshot,
        output_root=args.output_root,
        dataset_version=args.dataset_version,
        fingerprint_workers=args.fingerprint_workers,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
