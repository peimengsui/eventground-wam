#!/usr/bin/env python3
"""Fit and smoke-test the frozen Week 3 B0 data layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from eventground_wam.b0_data import (
    PINNED_TRAIN_ACTION_COUNT,
    PINNED_TRAIN_SAMPLE_COUNT,
    B0DataConfig,
    B0DataError,
    B0StreamingDataset,
    fit_b0_action_statistics,
    load_b0_action_normalizer,
    load_frozen_b0_split_index,
    validate_b0_action_roundtrip,
    write_b0_action_statistics,
)
from eventground_wam.demonstration_reader import DemonstrationReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    statistics = subparsers.add_parser(
        "statistics",
        help="fit and seal train-only B0 action statistics",
    )
    _add_source_arguments(statistics)
    statistics.add_argument("--output", type=Path, required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="validate all action round trips and decode the first training episode",
    )
    _add_source_arguments(smoke)
    smoke.add_argument("--normalization", type=Path, required=True)
    smoke.add_argument("--output", type=Path)
    smoke.add_argument("--seed", type=int, default=0)
    return parser


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)


def _git_identity() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, not bool(status.strip())


def _statistics(args: argparse.Namespace) -> dict[str, object]:
    revision, clean = _git_identity()
    reader = DemonstrationReader(args.repository_root)
    split_index = load_frozen_b0_split_index(args.frozen_root)
    artifact = fit_b0_action_statistics(
        reader,
        split_index,
        project_git_revision=revision,
        project_tree_clean=clean,
    )
    if artifact["split"]["action_count"] != PINNED_TRAIN_ACTION_COUNT:
        raise B0DataError("training action count differs from the audited frozen count")
    if artifact["split"]["valid_action_chunk_count"] != PINNED_TRAIN_SAMPLE_COUNT:
        raise B0DataError("B0 sample count differs from the audited frozen count")
    output = write_b0_action_statistics(args.output, artifact)
    return {
        "command": "statistics",
        "output": str(output),
        "artifact_digest": artifact["digest"],
        "project_git_revision": revision,
        "project_tree_clean": clean,
        "split": artifact["split"],
        "statistics": artifact["statistics"],
    }


def _smoke(args: argparse.Namespace) -> dict[str, object]:
    revision, clean = _git_identity()
    reader = DemonstrationReader(args.repository_root)
    split_index = load_frozen_b0_split_index(args.frozen_root)
    normalizer = load_b0_action_normalizer(args.normalization)
    dataset = B0StreamingDataset(
        reader,
        split_index,
        normalizer,
        config=B0DataConfig(seed=args.seed, shuffle_episodes=False),
    )
    if dataset.sample_count != PINNED_TRAIN_SAMPLE_COUNT:
        raise B0DataError(
            f"loader exposes {dataset.sample_count} samples, expected {PINNED_TRAIN_SAMPLE_COUNT}"
        )
    first_digest = dataset.ordered_identity_digest()
    second_digest = dataset.ordered_identity_digest()
    if first_digest != second_digest:
        raise B0DataError("same configuration and seed produced different identity digests")

    if not dataset.episodes:
        raise B0DataError("training dataset is empty")
    episode = dataset.episodes[0]
    if episode.episode_index != 0 or episode.length != 277:
        raise B0DataError(
            f"first training episode is {(episode.episode_index, episode.length)}, expected (0, 277)"
        )
    raw_rows = tuple(reader.iter_episode_rows(episode))
    samples = []
    sample_iterator = dataset.iter_samples()
    try:
        for sample in sample_iterator:
            if sample.identity.episode_index != episode.episode_index:
                break
            samples.append(sample)
    finally:
        sample_iterator.close()
    expected_episode_samples = episode.length - 8 + 1
    if len(samples) != expected_episode_samples:
        raise B0DataError(
            f"episode 0 produced {len(samples)} samples, expected {expected_episode_samples}"
        )
    for anchor in (0, len(samples) // 2, len(samples) - 1):
        sample = samples[anchor]
        expected_actions = np.asarray(
            [row.action for row in raw_rows[anchor : anchor + 8]],
            dtype=np.float64,
        )
        actual_actions = normalizer.denormalize(sample.actions)
        np.testing.assert_allclose(actual_actions, expected_actions, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            sample.proprioception,
            np.asarray(raw_rows[anchor].state, dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        if (
            sample.front_rgb.shape != (3, 256, 256)
            or sample.wrist_rgb.shape != (3, 256, 256)
            or sample.front_rgb.dtype != np.uint8
            or sample.wrist_rgb.dtype != np.uint8
            or not sample.front_rgb.flags.c_contiguous
            or not sample.wrist_rgb.flags.c_contiguous
        ):
            raise B0DataError("decoded sample images violate RGB uint8 CHW contract")

    roundtrip = validate_b0_action_roundtrip(reader, split_index, normalizer)
    report: dict[str, object] = {
        "artifact_type": "eventground_wam.b0_dataloader_smoke",
        "schema_version": 1,
        "project_git_revision": revision,
        "project_tree_clean": clean,
        "normalization_path": str(args.normalization.expanduser().resolve()),
        "normalization_file_sha256": _sha256_file(args.normalization),
        "normalization_artifact_digest": normalizer.artifact_digest,
        "source": split_index.source_identity,
        "config": {
            "seed": args.seed,
            "shuffle_episodes": False,
            "camera_orientation": "identity",
            "rgb_conversion": "pyav-rgb24",
            "image_layout": "CHW",
            "resize_crop": "none-source-256x256",
            "action_chunk_length": 8,
        },
        "training": {
            "episode_count": split_index.episode_counts["train"],
            "sample_count": dataset.sample_count,
            "ordered_identity_digest": first_digest,
        },
        "episode_zero": {
            "length": episode.length,
            "sample_count": len(samples),
            "first_identity": samples[0].identity.to_dict(),
            "last_identity": samples[-1].identity.to_dict(),
            "direct_alignment_anchors_checked": [0, len(samples) // 2, len(samples) - 1],
            "front_shape": list(samples[0].front_rgb.shape),
            "wrist_shape": list(samples[0].wrist_rgb.shape),
        },
        "roundtrip": roundtrip,
        "passed": True,
    }
    payload_without_digest = dict(report)
    report["digest"] = hashlib.sha256(
        json.dumps(
            payload_without_digest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if args.output is not None:
        _write_exclusive_json(args.output, report)
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive_json(path: Path, value: dict[str, object]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _statistics(args) if args.command == "statistics" else _smoke(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
