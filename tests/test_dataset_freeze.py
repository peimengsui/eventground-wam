from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import eventground_wam.dataset_freeze as freeze_module
from eventground_wam.dataset_freeze import DatasetFreezeError, validate_and_seal_dataset_freeze
from eventground_wam.demonstration_contracts import (
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_MANIFEST_SHA256,
    EventCandidate,
    EventInterval,
    MultiHorizonSample,
    Provenance,
    SplitAssignment,
    TrajectoryRecord,
    contract_digest,
    seal_contract,
)
from eventground_wam.event_extraction import EventDecision

REVISION = "a" * 40


def _jsonl(path: Path, records: list[dict[str, object]]) -> tuple[str, int]:
    payload = "".join(
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source() -> dict[str, str]:
    return {
        "dataset_repo_id": SOURCE_DATASET_REPO_ID,
        "dataset_revision": SOURCE_DATASET_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
    }


def _build_tiny_run(root: Path) -> None:
    candidates_dir = root / "candidates"
    events_dir = root / "events"
    splits_dir = root / "splits"
    trajectory = TrajectoryRecord.from_episode(
        Provenance(
            source_episode_index=0,
            source_task_index=0,
            source_episode_length=6,
            project_git_revision=REVISION,
            annotation_version="tiny.candidates",
        ),
        "instruction",
    )
    candidate = EventCandidate(
        provenance=trajectory.provenance,
        trajectory_digest=trajectory.digest,
        frame_start=1,
        frame_end=4,
        event_type="gripper_close",
        confidence=0.95,
        evidence=("action:test",),
        detector_name="test",
        detector_version="v1",
    )
    interval = EventInterval(
        provenance=Provenance(
            source_episode_index=0,
            source_task_index=0,
            source_episode_length=6,
            project_git_revision=REVISION,
            annotation_version="tiny.events",
        ),
        trajectory_digest=trajectory.digest,
        frame_start=1,
        frame_end=4,
        event_type="gripper_close",
        confidence=0.95,
        evidence=("action:test",),
        candidate_digests=(candidate.digest,),
        ontology_tier="weak",
        review_status="unreviewed",
    )
    decision = EventDecision(
        source_episode_index=0,
        source_task_index=0,
        event_type="gripper_close",
        status="admitted_weak",
        reason="test",
        frame_start=1,
        frame_end=4,
        candidate_digests=(candidate.digest,),
        interval_digests=(interval.digest,),
    )
    assignment = SplitAssignment(
        provenance=Provenance(
            source_episode_index=0,
            source_task_index=0,
            source_episode_length=6,
            project_git_revision=REVISION,
            annotation_version="tiny.splits",
        ),
        trajectory_digest=trajectory.digest,
        frame_start=0,
        frame_end=6,
        split="train",
        group_strategy="task-state-rgb-surrogate-v1",
        initial_condition_digest="b" * 64,
    )
    sample = MultiHorizonSample(
        provenance=assignment.provenance,
        trajectory_digest=trajectory.digest,
        split_assignment_digest=assignment.digest,
        frame_start=0,
        frame_end=3,
        anchor_frame_index=0,
        target_frame_index=2,
        horizon=2,
        split="train",
    )

    _json(candidates_dir / "selection.json", [])
    trajectory_sha, _ = _jsonl(
        candidates_dir / "trajectory-contracts.jsonl", [trajectory.to_dict()]
    )
    candidate_sha, _ = _jsonl(
        candidates_dir / "event-candidates.jsonl", [candidate.to_dict()]
    )
    _json(candidates_dir / "review-items.jsonl", [])
    _json(
        candidates_dir / "feasibility-report.json",
        {
            "status": "full_source_complete",
            "publishable": True,
            "source": _source(),
            "generator": {
                "project_git_revision": REVISION,
                "project_tree_clean": True,
                "annotation_version": "tiny.candidates",
            },
            "selection": {
                "complete_source": True,
                "selected_episode_count": 1,
                "selected_frame_count": 6,
                "selected_task_count": 1,
            },
        },
    )

    _jsonl(events_dir / "event-candidates.jsonl", [candidate.to_dict()])
    interval_sha, _ = _jsonl(events_dir / "event-intervals.jsonl", [interval.to_dict()])
    decision_sha, _ = _jsonl(events_dir / "event-decisions.jsonl", [decision.to_dict()])
    event_manifest = seal_contract(
        {
            "artifact_type": "eventground_wam.event_extraction",
            "schema_version": 1,
            "source": _source(),
            "generator": {
                "project_git_revision": REVISION,
                "annotation_version": "tiny.events",
            },
            "episode_count": 1,
            "candidate_count": 1,
            "interval_count": 1,
            "decision_count": 1,
            "decision_counts": {
                "admitted_weak": 1,
                "absent": 0,
                "ambiguous": 0,
                "deferred": 0,
            },
            "records_sha256": {
                "event-candidates.jsonl": candidate_sha,
                "event-intervals.jsonl": interval_sha,
                "event-decisions.jsonl": decision_sha,
            },
            "publishable": True,
        }
    )
    _json(events_dir / "manifest.json", event_manifest)

    split_sha, _ = _jsonl(splits_dir / "split-assignments.jsonl", [assignment.to_dict()])
    sample_sha, _ = _jsonl(splits_dir / "multi-horizon-samples.jsonl", [sample.to_dict()])
    split_manifest = seal_contract(
        {
            "artifact_type": "eventground_wam.split_sample_indexes",
            "schema_version": 1,
            "source": _source(),
            "generator": {
                "project_git_revision": REVISION,
                "project_tree_clean": True,
                "annotation_version": "tiny.splits",
            },
            "grouping": {
                "strategy": "task-state-rgb-surrogate-v1",
                "cross_split_group_overlap_count": 0,
                "groups_by_split": {"train": 1, "validation": 0, "test": 0},
                "episodes_by_split": {"train": 1, "validation": 0, "test": 0},
            },
            "sampling": {
                "horizons": [2, 4, 8, 16, 32, "next-event"],
                "sample_counts_by_horizon": {
                    "2": 1,
                    "4": 0,
                    "8": 0,
                    "16": 0,
                    "32": 0,
                    "next-event": 0,
                },
                "sample_counts_by_split": {"train": 1, "validation": 0, "test": 0},
                "event_targets": {
                    "input_event_count": 1,
                    "eligible_event_count": 0,
                    "excluded_event_count": 1,
                },
            },
            "outputs": {
                "split_assignments": {
                    "sha256": split_sha,
                    "record_count": 1,
                },
                "multi_horizon_samples": {
                    "sha256": sample_sha,
                    "record_count": 1,
                },
            },
            "complete_source": True,
            "publishable": True,
        }
    )
    _json(splits_dir / "split-sample-manifest.json", split_manifest)
    assert trajectory_sha


def _benchmark() -> dict[str, object]:
    stages = {
        run: {
            stage: {"wall_seconds": 1.0}
            for stage in (
                "candidate_generation",
                "event_extraction",
                "split_sample_indexing",
            )
        }
        for run in ("primary", "replica")
    }
    return {"stages": stages}


def test_complete_replica_is_validated_and_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(freeze_module, "SOURCE_EPISODE_COUNT", 1)
    monkeypatch.setattr(freeze_module, "SOURCE_FRAME_COUNT", 6)
    monkeypatch.setattr(freeze_module, "SOURCE_TASK_COUNT", 1)
    primary = tmp_path / "primary"
    replica = tmp_path / "replica"
    _build_tiny_run(primary)
    shutil.copytree(primary, replica)

    manifest = validate_and_seal_dataset_freeze(
        primary,
        replica,
        dataset_version="tiny-v1",
        project_git_revision=REVISION,
        benchmark=_benchmark(),
    )

    assert manifest["status"] == "frozen"
    assert manifest["publishable"] is True
    assert manifest["validation"]["replica_byte_identical"] is True
    assert manifest["event_conditioned_training_ready"] is False
    assert manifest["digest"] == contract_digest(manifest)


def test_replica_byte_difference_blocks_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(freeze_module, "SOURCE_EPISODE_COUNT", 1)
    monkeypatch.setattr(freeze_module, "SOURCE_FRAME_COUNT", 6)
    monkeypatch.setattr(freeze_module, "SOURCE_TASK_COUNT", 1)
    primary = tmp_path / "primary"
    replica = tmp_path / "replica"
    _build_tiny_run(primary)
    shutil.copytree(primary, replica)
    (replica / "candidates" / "selection.json").write_text("[1]\n", encoding="utf-8")

    with pytest.raises(DatasetFreezeError, match="replica differs"):
        validate_and_seal_dataset_freeze(
            primary,
            replica,
            dataset_version="tiny-v1",
            project_git_revision=REVISION,
            benchmark=_benchmark(),
        )


def test_committed_week2_freeze_evidence_is_digest_sealed() -> None:
    path = Path(__file__).parents[1] / "docs" / "freeze" / "week2-dataset-v1.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["digest"] == contract_digest(evidence)
    assert evidence["status"] == "frozen"
    assert evidence["scope"] == {
        "episode_count": 14_347,
        "frame_count": 2_238_036,
        "task_count": 40,
    }
    assert evidence["splits"]["cross_split_group_overlap_count"] == 0
    assert evidence["samples"]["record_count"] == 10_300_666
    assert evidence["samples"]["by_horizon"]["next-event"] == 0
    assert evidence["validation"]["replica_byte_identical"] is True
