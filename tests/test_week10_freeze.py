"""Adversarial checks for the scientific admission boundary, without GPU/data reads."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "week10_verifier", ROOT / "scripts/verify_week10_experiment.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def frozen_copy(tmp_path):
    study = json.loads((ROOT / MODULE.CONTRACT).read_text())
    paths = list(study["source_files_sha256"]) + [str(MODULE.CONTRACT)]
    paths += [row["snapshot_path"] for row in study["document_history"]]
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def rewrite_contract(root, mutate):
    path = root / MODULE.CONTRACT
    study = json.loads(path.read_text())
    mutate(study)
    study["digest"] = MODULE.digest(study)
    path.write_text(json.dumps(study))


def test_real_freeze_keeps_labels_and_training_unadmitted():
    result = MODULE.verify(ROOT)
    assert result["status"] == "passed"
    assert sum(result["validation_groups"].values()) == 180
    assert not result["training_ready"]
    assert not result["contact_grasp_labels_admitted"]


def test_changed_parent_weights_cannot_be_admitted_by_resealing(frozen_copy):
    rewrite_contract(frozen_copy, lambda x: x["parent"].update(policy_sha256="0" * 64))
    with pytest.raises(ValueError, match="Policy payload mismatch"):
        MODULE.verify(frozen_copy)


def test_resealed_close_only_scope_is_rejected(frozen_copy):
    rewrite_contract(frozen_copy, lambda x: x["event_scope"].update(
        required_positive_types=["gripper_close"]))
    with pytest.raises(ValueError, match="ontology removed"):
        MODULE.verify(frozen_copy)


def test_extra_action_exposure_is_rejected(frozen_copy):
    rewrite_contract(frozen_copy, lambda x: x["parent"].update(additional_action_updates=1))
    with pytest.raises(ValueError, match="Extra action/CPT"):
        MODULE.verify(frozen_copy)


def test_corrupted_qa_evidence_fails_closed(frozen_copy):
    path = frozen_copy / "docs/freeze/week2-dataset-v1-human-qa.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        MODULE.verify(frozen_copy)


def test_resealed_partition_cannot_move_a_group_between_panels(frozen_copy):
    relative = str(MODULE.EVIDENCE / "validation-groups.json")
    path = frozen_copy / relative
    partition = json.loads(path.read_text())
    row = partition["assignments"][0]
    row["partition"] = "event-report" if row["partition"] == "event-dev" else "event-dev"
    partition["digest"] = MODULE.digest(partition)
    path.write_text(json.dumps(partition))

    def mutate(study):
        study["data"]["validation_partition_digest"] = partition["digest"]
        study["source_files_sha256"][relative] = MODULE.hashlib.sha256(path.read_bytes()).hexdigest()

    rewrite_contract(frozen_copy, mutate)
    with pytest.raises(ValueError, match="Validation partition membership mismatch"):
        MODULE.verify(frozen_copy)
