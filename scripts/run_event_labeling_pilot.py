#!/usr/bin/env python3
"""Run the conservative Week 2 event-labeling feasibility pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eventground_wam.demonstration_contracts import (
    EVENT_TYPES,
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_MANIFEST_SHA256,
)
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.event_labeling import (
    GripperCloseDetectorConfig,
    GripperPolarityAudit,
    extract_gripper_close_candidates,
    review_item,
    select_task_stratified_episodes,
    trajectory_contract_for_episode,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_VERSION = "week2-pilot-v1"
OUTPUT_FILENAMES = (
    "selection.json",
    "trajectory-contracts.jsonl",
    "event-candidates.jsonl",
    "review-items.jsonl",
    "feasibility-report.json",
)


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
        "--output-dir",
        type=expanded_path,
        default=data_root / "derived" / "event-labeling" / DEFAULT_ANNOTATION_VERSION,
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=1,
        help="length-stratified episodes to select per task (default: 1)",
    )
    parser.add_argument("--annotation-version", default=DEFAULT_ANNOTATION_VERSION)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow non-publishable pilot output from an uncommitted implementation",
    )
    return parser.parse_args()


def project_state(project_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, not bool(status.strip())


def run_pilot(
    *,
    snapshot: Path,
    output_dir: Path,
    episodes_per_task: int,
    annotation_version: str,
    project_revision: str,
    project_tree_clean: bool,
) -> dict[str, Any]:
    output_paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"pilot refuses to overwrite existing outputs: {existing}")

    reader = DemonstrationReader(snapshot)
    selected = select_task_stratified_episodes(
        reader.iter_episodes(),
        episodes_per_task=episodes_per_task,
    )
    detector_config = GripperCloseDetectorConfig()
    polarity_audit = GripperPolarityAudit(
        response_window_frames=detector_config.response_window_frames
    )
    trajectory_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    extraction_totals = {
        "close_transition_count": 0,
        "candidate_count": 0,
        "rejected_short_command": 0,
        "rejected_incomplete_response": 0,
        "rejected_weak_aperture_response": 0,
    }

    for episode in selected:
        for _, video_path in episode.video_paths:
            if not video_path.is_file():
                raise FileNotFoundError(f"pilot video is missing: {video_path}")
        rows = tuple(reader.iter_episode_rows(episode))
        polarity_audit.update(rows)
        trajectory = trajectory_contract_for_episode(
            episode,
            project_git_revision=project_revision,
            annotation_version=annotation_version,
        )
        extraction = extract_gripper_close_candidates(
            rows,
            trajectory,
            config=detector_config,
        )
        trajectory_records.append(trajectory.to_dict())
        candidate_records.extend(candidate.to_dict() for candidate in extraction.candidates)
        review_records.extend(
            review_item(candidate, trajectory) for candidate in extraction.candidates
        )
        selection_records.append(
            {
                "source_episode_index": episode.episode_index,
                "source_task_index": episode.task_index,
                "instruction": episode.instruction,
                "length": episode.length,
                "trajectory_digest": trajectory.digest,
                "candidate_count": extraction.summary.candidate_count,
            }
        )
        for key in extraction_totals:
            extraction_totals[key] += int(getattr(extraction.summary, key))

    transition_count = extraction_totals["close_transition_count"]
    candidate_count = extraction_totals["candidate_count"]
    confirmation_rate = candidate_count / transition_count if transition_count else None
    task_count = len({episode.task_index for episode in selected})
    report = {
        "artifact_type": "eventground_wam.event_labeling_feasibility_report",
        "schema_version": 1,
        "status": "pilot_complete",
        "publishable": project_tree_clean,
        "source": {
            "dataset_repo_id": SOURCE_DATASET_REPO_ID,
            "dataset_revision": SOURCE_DATASET_REVISION,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "snapshot": str(reader.repository_root),
        },
        "generator": {
            "project_git_revision": project_revision,
            "project_tree_clean": project_tree_clean,
            "annotation_version": annotation_version,
        },
        "selection": {
            "algorithm": "per-task-length-strata-v1",
            "episodes_per_task": episodes_per_task,
            "selected_episode_count": len(selected),
            "selected_task_count": task_count,
            "selected_frame_count": sum(episode.length for episode in selected),
            "episode_indices": [episode.episode_index for episode in selected],
        },
        "gripper_convention": {
            "source_convention": "LIBERO -1=open, +1=close",
            "aperture_definition": "abs(state[-2]) + abs(state[-1])",
            "pilot_transition_evidence": polarity_audit.to_dict(),
        },
        "detector": {
            "name": "gripper-close-telemetry",
            "version": "v1",
            "config": asdict(detector_config),
            **extraction_totals,
            "telemetry_confirmation_rate": confirmation_rate,
        },
        "ontology_decision": {
            "candidate_event_types": ["gripper_close"],
            "reliable_event_types": [],
            "deferred_event_types": [
                event_type for event_type in EVENT_TYPES if event_type != "gripper_close"
            ],
            "deferred_reason": (
                "object-relative evidence is absent from the source telemetry; "
                "RGB, simulator-sidecar, or human evidence is required"
            ),
            "next_action": "human_review_required_before_reliable_ontology_admission",
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_paths["selection.json"], selection_records)
    _write_jsonl(output_paths["trajectory-contracts.jsonl"], trajectory_records)
    _write_jsonl(output_paths["event-candidates.jsonl"], candidate_records)
    _write_jsonl(output_paths["review-items.jsonl"], review_records)
    _write_json(output_paths["feasibility-report.json"], report)
    return report


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    payload = "".join(
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    )
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    revision, clean = project_state(PROJECT_ROOT)
    if not clean and not args.allow_dirty:
        raise SystemExit(
            "project tree is dirty; commit the detector or pass --allow-dirty for "
            "explicitly non-publishable pilot output"
        )
    report = run_pilot(
        snapshot=args.snapshot,
        output_dir=args.output_dir,
        episodes_per_task=args.episodes_per_task,
        annotation_version=args.annotation_version,
        project_revision=revision,
        project_tree_clean=clean,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
