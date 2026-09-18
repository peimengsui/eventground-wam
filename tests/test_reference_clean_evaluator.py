from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from torch.torch_version import TorchVersion

from eventground_wam.demonstration_contracts import seal_contract

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import evaluate_reference_clean as evaluator


@pytest.mark.parametrize("field", evaluator.PAIRED_METADATA_FIELDS)
def test_paired_metadata_rejects_every_changed_evaluation_input(field):
    left = {name: f"same-{name}" for name in evaluator.PAIRED_METADATA_FIELDS}
    right = {**left, field: "changed"}
    with pytest.raises(ValueError, match=field):
        evaluator.match_metadata(left, right)


def test_paired_metadata_allows_different_policy_and_source_identities():
    shared = {name: f"same-{name}" for name in evaluator.PAIRED_METADATA_FIELDS}
    evaluator.match_metadata(
        {**shared, "checkpoint_id": "b0", "project_git_revision": "old-source"},
        {**shared, "checkpoint_id": "openvla", "project_git_revision": "new-source"},
    )


def test_reference_evaluator_rejects_modified_sealed_report(tmp_path):
    path = tmp_path / "report.json"
    value = seal_contract({"success_count": 26})
    path.write_text(json.dumps(value))
    assert evaluator.read_sealed(path) == value
    value["success_count"] = 40
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid sealed record"):
        evaluator.read_sealed(path)


def test_paired_verification_rejects_missing_episode(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        evaluator.verify_pairs({"episodes": [{}]}, {"episodes": []}, None, {})


def test_final_report_accepts_torch_version_and_preserves_saved_metrics():
    progress = {"episode_count": 40, "episodes": [{}] * 40,
                "selection_is_complete_manifest": True, "project_git_revision": "source",
                "protocol_sha256": "protocol", "checkpoint_revision": "checkpoint",
                "success_count": 39, "runtime": {"torch": TorchVersion("2.13.0+cu130")}}
    selection = {"project_git_revision": "source", "protocol_sha256": "protocol",
                 "checkpoint_revision": "checkpoint", "task_records_sha256": "tasks",
                 "digest": "selection", "b0_report_digest": "b0", "policy_profile": {},
                 "comparison_scope": "external", "acceptance_gate": "open"}
    report = evaluator.finalize_report(progress, selection)
    assert report["success_count"] == 39
    assert report["runtime"]["torch"] == "2.13.0+cu130"
    assert type(report["runtime"]["torch"]) is str
    assert report["status"] == "complete"
    progress["episode_count"] = 39
    with pytest.raises(ValueError, match="incomplete or mismatched"):
        evaluator.finalize_report(progress, selection)
