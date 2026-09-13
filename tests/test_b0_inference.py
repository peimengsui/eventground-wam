from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import eventground_wam.b0_inference as inference
from eventground_wam.b0_data import ActionNormalizer
from eventground_wam.demonstration_contracts import seal_contract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _write_with_sidecar(path: Path, payload: bytes | dict[str, object]) -> str:
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        _write_json(path, payload)
    digest = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def _normalizer(*, artifact_digest: str | None = None) -> ActionNormalizer:
    return ActionNormalizer(
        center=(0.0,) * 7,
        scale=(2.0,) * 6 + (1.0,),
        lower=(-1.0,) * 7,
        upper=(1.0,) * 7,
        artifact_digest=artifact_digest,
    )


def test_decode_b0_action_chunk_denormalizes_clips_and_preserves_gripper() -> None:
    normalized = np.zeros((8, 7), dtype=np.float64)
    normalized[:, 6] = [-1.0, 1.0] * 4
    normalized[0, :3] = [1.0, -1.0, 0.25]

    actions, summary = inference.decode_b0_action_chunk(normalized, _normalizer())

    assert actions.shape == (8, 7)
    assert actions.flags.c_contiguous
    np.testing.assert_array_equal(actions[:, 6], normalized[:, 6])
    np.testing.assert_allclose(actions[0, :3], [1.0, -1.0, 0.5])
    assert summary == {
        "continuous_value_count": 48,
        "clipped_continuous_value_count": 2,
        "maximum_continuous_clipping_delta": 1.0,
        "gripper_values": [-1.0, 1.0],
    }


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (np.zeros((7, 7)), "8x7"),
        (np.full((8, 7), np.nan), "non-finite"),
        (np.zeros((8, 7)), r"exact -1 or \+1"),
    ],
)
def test_decode_b0_action_chunk_rejects_invalid_predictions(
    actions: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(inference.B0InferenceError, match=message):
        inference.decode_b0_action_chunk(actions, ActionNormalizer.identity())


def test_verify_b0_checkpoint_bundle_requires_the_exact_accepted_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    checkpoint_path = checkpoint_root / "trainable-state.pt"
    checkpoint_sha256 = _write_with_sidecar(checkpoint_path, b"trainable state")

    manifest = seal_contract(
        {
            "artifact_type": inference.B0_COHORT_MANIFEST_ARTIFACT_TYPE,
            "schema_version": 1,
            "episodes": [
                {
                    "episode_index": index,
                    "task_index": index,
                    "instruction": f"instruction {index}",
                }
                for index in range(10)
            ],
        }
    )
    manifest_path = checkpoint_root / "cohort-manifest.json"
    manifest_sha256 = _write_with_sidecar(manifest_path, manifest)

    normalizer_digest = "a" * 64
    normalization_path = tmp_path / "action-statistics.json"
    normalization_path.write_text("normalization fixture\n", encoding="utf-8")
    normalization_sha256 = _sha256(normalization_path)
    monkeypatch.setattr(
        inference,
        "load_b0_action_normalizer",
        lambda path: _normalizer(artifact_digest=normalizer_digest),
    )

    report = seal_contract(
        {
            "artifact_type": inference.B0_TRAINING_REPORT_ARTIFACT_TYPE,
            "schema_version": 1,
            "configuration": {
                "objective": inference.B0_ROLLOUT_OBJECTIVE,
                "gripper_bce_weight": inference.B0_ROLLOUT_GRIPPER_BCE_WEIGHT,
            },
            "cohort": {
                "manifest_digest": manifest["digest"],
                "manifest_file_sha256": manifest_sha256,
            },
            "checkpoint": {
                "file_sha256": checkpoint_sha256,
                "format": inference.B0_COHORT_CHECKPOINT_FORMAT,
                "reload_max_absolute_error": 0.0,
            },
            "source": {
                "normalization_file_sha256": normalization_sha256,
                "normalization_artifact_digest": normalizer_digest,
            },
            "project": {"git_revision": "b" * 40},
        }
    )
    report_path = checkpoint_root / "overfit-report.json"
    report_sha256 = _write_with_sidecar(report_path, report)

    review = seal_contract(
        {
            "artifact_type": inference.B0_ACCEPTED_REVIEW_ARTIFACT_TYPE,
            "schema_version": 1,
            "status": "accepted-with-documented-residuals",
            "reviewed_disposition": {"accepted_with_documented_residuals": True},
            "artifacts": {
                "training": {
                    "overfit_report": {
                        "artifact_digest": report["digest"],
                        "file_sha256": report_sha256,
                    },
                    "cohort_manifest": {
                        "artifact_digest": manifest["digest"],
                        "file_sha256": manifest_sha256,
                    },
                    "checkpoint": {"file_sha256": checkpoint_sha256},
                }
            },
        }
    )
    review_path = tmp_path / "accepted-review.json"
    _write_json(review_path, review)

    verified = inference.verify_b0_checkpoint_bundle(
        checkpoint_root,
        normalization_path,
        review_path,
    )

    assert verified.checkpoint_revision == checkpoint_sha256
    assert verified.training_report_digest == report["digest"]
    assert verified.cohort_manifest_digest == manifest["digest"]
    assert verified.cohort_matches_for_task_name("SCENE_instruction_3_table_17") == (
        {
            "episode_index": 3,
            "task_index": 3,
            "instruction": "instruction 3",
        },
    )
    assert verified.cohort_matches_for_task_name("unseen_task_table_1") == ()
    assert verified.normalization_artifact_digest == normalizer_digest
    assert verified.objective == inference.B0_ROLLOUT_OBJECTIVE

    checkpoint_path.write_bytes(b"tampered")
    with pytest.raises(inference.B0InferenceError, match="fails its SHA-256 sidecar"):
        inference.verify_b0_checkpoint_bundle(
            checkpoint_root,
            normalization_path,
            review_path,
        )


class _FakePolicy(torch.nn.Module):
    def forward(
        self,
        backbone_inputs: dict[str, torch.Tensor],
        proprioception: torch.Tensor,
    ) -> SimpleNamespace:
        assert backbone_inputs["pixel_values"].dtype == torch.bfloat16
        assert proprioception.shape == (1, 8)
        output = torch.zeros((1, 8, 7), dtype=torch.float32)
        output[0, :, 6] = torch.tensor([-1.0, 1.0] * 4)
        return SimpleNamespace(normalized_actions=output)


def test_inference_policy_matches_shared_evaluator_boundary() -> None:
    policy = inference.B0InferencePolicy(
        policy=_FakePolicy(),  # type: ignore[arg-type]
        input_builder=lambda instructions, front, wrist: {
            "pixel_values": torch.ones((1, 2), dtype=torch.float32)
        },
        normalizer=ActionNormalizer.identity(),
        device=torch.device("cpu"),
        checkpoint=object(),  # type: ignore[arg-type]
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    actions = policy.predict_chunk(
        agent_rgb=image,
        wrist_rgb=image,
        proprioception=np.zeros(8, dtype=np.float32),
        instruction="pick up the object",
        suite="libero_spatial",
    )

    assert actions.shape == (8, 7)
    assert actions[:, 6].tolist() == [-1.0, 1.0] * 4
    assert policy.inference_summary()["prediction_count"] == 1
    assert policy.inference_summary()["decoded_action_count"] == 8


def test_committed_b0_evaluator_integration_evidence_is_sealed() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs/evaluation/week3-b0-evaluator-integration-v1.json"
    )
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["digest"] == inference.contract_digest(evidence)
    assert evidence["status"] == "passed"
    assert evidence["implementation"]["project_tree_clean"] is True
    assert evidence["validation"]["local_test_count"] == 180
    assert evidence["validation"]["accepted_hash_chain_verified"] is True
    assert evidence["validation"]["strict_qwen_lora_checkpoint_restore_passed"] is True
    assert evidence["validation"]["synthetic_query"]["output_shape"] == [8, 7]
    assert evidence["validation"]["synthetic_query"]["all_values_finite"] is True
    assert evidence["validation"]["real_simulator_episode_executed"] is False
