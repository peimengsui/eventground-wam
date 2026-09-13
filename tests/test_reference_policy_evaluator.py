import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_reference_policy as evaluator

from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.libero_sensor_noise import codec_independent_motion_blur


def test_select_tasks_applies_offset_and_limit() -> None:
    manifest = {"tasks": [{"id": index} for index in range(5)]}

    assert evaluator.select_tasks(manifest, offset=1, limit=2) == [
        {"id": 1},
        {"id": 2},
    ]
    assert evaluator.select_tasks(manifest, offset=3, limit=None) == [
        {"id": 3},
        {"id": 4},
    ]
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluator.select_tasks(manifest, offset=-1, limit=1)


def test_process_openvla_action_chunk_converts_gripper() -> None:
    raw = np.zeros((8, 7), dtype=np.float64)
    raw[:, -1] = [0.0, 0.25, 0.49, 0.5, 0.51, 0.75, 1.0, 0.0]

    processed = evaluator.process_openvla_action_chunk(raw)

    np.testing.assert_array_equal(
        processed[:, -1],
        np.asarray([1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 1.0]),
    )
    with pytest.raises(evaluator.InvalidPolicyAction, match="8x7"):
        evaluator.process_openvla_action_chunk(np.zeros((7, 7)))
    raw[0, 0] = np.nan
    with pytest.raises(evaluator.InvalidPolicyAction, match="non-finite"):
        evaluator.process_openvla_action_chunk(raw)


def test_motion_blur_rejects_invalid_inputs_without_loading_wand() -> None:
    with pytest.raises(ValueError, match="severity"):
        codec_independent_motion_blur(np.zeros((4, 4, 3), dtype=np.uint8), severity=0)
    with pytest.raises(ValueError, match="HxWx3"):
        codec_independent_motion_blur(np.zeros((4, 4), dtype=np.uint8), severity=1)


def test_wilson_interval_contains_observed_rate() -> None:
    interval = evaluator.wilson_interval(5, 10)

    assert interval is not None
    assert interval[0] < 0.5 < interval[1]
    assert evaluator.wilson_interval(0, 0) is None


def test_find_resumable_episode_prefers_completed_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "episode.hdf5"
    Path(str(base) + ".partial").touch()
    retry = evaluator.attempt_path(base, 1)
    retry.touch()
    validated: list[Path] = []

    def fake_validate(path: Path, protocol: object) -> dict[str, object]:
        validated.append(path)
        return {"status": "complete"}

    monkeypatch.setattr(evaluator, "validate_episode_file", fake_validate)

    assert evaluator.find_resumable_episode(base, {}, retry_limit=1) == (retry, 1)
    assert validated == [retry]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("project_git_revision", "wrong-project"),
        ("checkpoint_id", "wrong-checkpoint"),
        ("checkpoint_revision", "wrong-revision"),
        ("suite", "libero_goal"),
        ("runtime_task_index", 75),
        ("classification_task_id", 76),
        ("perturbation_category", "Camera Viewpoints"),
        ("difficulty_level", 2),
        ("episode_seed", 999),
        ("instruction", "wrong instruction"),
    ],
)
def test_recovered_episode_rejects_provenance_mismatch(
    field: str,
    bad_value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_task = {
        "suite": "libero_spatial",
        "runtime_task_index": 74,
        "classification_task_id": 75,
        "task_name": "task_name",
        "perturbation_category": "Background Textures",
        "difficulty_level": 3,
        "episode_seed": 152158716,
    }
    metadata = {
        "project_git_revision": "abc123",
        "checkpoint_id": evaluator.MODEL_ID,
        "checkpoint_revision": evaluator.MODEL_REVISION,
        **{key: manifest_task[key] for key in manifest_task if key != "task_name"},
        "instruction": "pick up the bowl",
    }
    metadata[field] = bad_value
    monkeypatch.setattr(evaluator, "read_episode_metadata", lambda path: metadata)
    monkeypatch.setattr(
        evaluator,
        "validate_episode_file",
        lambda path, protocol: {"status": "complete"},
    )

    with pytest.raises(RuntimeError, match=field):
        evaluator.recovered_result(
            tmp_path / "episode.hdf5",
            {},
            retry_count=0,
            expected_project_revision="abc123",
            manifest_task=manifest_task,
            expected_instruction="pick up the bowl",
        )


def test_episode_metadata_accepts_a_policy_specific_checkpoint_identity(tmp_path: Path) -> None:
    protocol = load_evaluation_protocol()
    bddl = tmp_path / "task.bddl"
    bddl.write_text("fixture", encoding="utf-8")
    manifest_task = {
        "suite": "libero_spatial",
        "runtime_task_index": 74,
        "classification_task_id": 75,
        "perturbation_category": "Background Textures",
        "difficulty_level": 3,
        "episode_seed": 152158716,
    }

    metadata = evaluator.make_episode_metadata(
        protocol=protocol,
        manifest_task=manifest_task,
        task=SimpleNamespace(language="pick up the bowl"),
        bddl_source_path=bddl,
        initial_state=np.zeros(10, dtype=np.float64),
        project_revision="a" * 40,
        checkpoint_id="eventground-wam/b0-ten-demonstration-bce-v1",
        checkpoint_revision="b" * 64,
        policy_instruction="sealed cohort instruction",
    )

    assert metadata.checkpoint_id == "eventground-wam/b0-ten-demonstration-bce-v1"
    assert metadata.checkpoint_revision == "b" * 64
    assert metadata.instruction == "sealed cohort instruction"


def test_partial_summary_filename_is_selection_qualified() -> None:
    manifest = {"task_count": 35}

    assert (
        evaluator.summary_filename(
            manifest,
            selected_task_offset=3,
            selected_task_count=4,
        )
        == "summary-offset-0003-count-0004.json"
    )
    assert (
        evaluator.summary_filename(
            manifest,
            selected_task_offset=0,
            selected_task_count=35,
        )
        == "summary.json"
    )


def test_summary_with_resumed_episode_is_strict_json(tmp_path: Path) -> None:
    manifest = {
        "split_id": "sanity-35",
        "task_count": 35,
        "protocol_id": "libero-plus-v1",
        "protocol_sha256": "0" * 64,
    }
    result = evaluator.EpisodeResult(
        suite="libero_spatial",
        runtime_task_index=74,
        perturbation_category="Background Textures",
        difficulty_level=3,
        success=True,
        termination_reason="environment_done",
        artifact="episode-retry-1.hdf5",
        artifact_sha256="1" * 64,
        policy_step_count=32,
        episode_wall_time_seconds=None,
        policy_latencies_ms=[10.0, 20.0],
        retry_count=1,
        exception="recovered_after_infrastructure_failure",
    )
    summary = evaluator.summarize_results(
        manifest=manifest,
        results=[result],
        project_revision="abc123",
        runtime_metadata={"torch": "2.13.0+cu130", "cuda_device": "RTX 5090"},
        environment_lock_sha256="2" * 64,
    )
    output = tmp_path / "summary.json"

    evaluator.write_json(output, summary)

    decoded = json.loads(output.read_text(encoding="utf-8"))
    assert decoded["micro_success_rate"] == 1.0
    assert decoded["episode_wall_time_seconds"] is None
    assert decoded["retry_count"] == 1
    assert decoded["manifest_task_count"] == 35
    assert decoded["selected_task_offset"] == 0
    assert decoded["selected_task_count"] == 1
    assert decoded["selection_is_complete_manifest"] is False
    assert decoded["environment_lock"]["sha256"] == "2" * 64
    assert decoded["runtime"]["cuda_device"] == "RTX 5090"
    assert decoded["breakdowns"]["difficulty_level"]["3"]["success_rate"] == 1.0


def test_committed_sanity_35_evidence_is_digest_sealed() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "evaluation"
        / "sanity-35-reference-policy-v1.json"
    )
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["digest"] == contract_digest(evidence)
    assert evidence["status"] == "complete"
    assert evidence["split"]["selected_task_count"] == 35
    assert evidence["result"]["selection_is_complete_manifest"] is True
    assert evidence["result"]["episode_count"] == 35
    assert evidence["result"]["success_count"] == 28
    assert evidence["result"]["exception_count"] == 0
    assert evidence["result"]["retry_count"] == 0
    assert evidence["validation"]["artifact_count"] == 35
    assert evidence["validation"]["partial_file_count"] == 0
    assert evidence["validation"]["all_episode_contracts_valid"] is True
    assert evidence["validation"]["all_artifact_hashes_match_report"] is True
