from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from test_b0_training_task_evaluator import _write_base_inventory

from eventground_wam.b0_training import TrainingContractError, atomic_json, sha256_file
from eventground_wam.clean_evaluation import BASE_SUITES, clean_instruction, clean_manifest
from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.evaluation_protocol import (
    ProtocolValidationError,
    load_evaluation_protocol,
    validate_evaluation_protocol,
)

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_b0_full as evaluator


def bundle(root: Path):
    recipe = json.loads((ROOT / "configs/training/week4_b0_5epoch_v2.json").read_text())
    normalization = root / "normalizer.json"
    normalization.write_text("test normalizer")
    checkpoint = root / "B0.ckpt"
    checkpoint.write_bytes(b"test checkpoint bytes")
    manifest = seal_contract({
        "artifact_type": "eventground_wam.b0_full_training_manifest", "mode": "production",
        "recipe": recipe,
        "source": {"normalization_file_sha256": sha256_file(normalization)},
    })
    report = seal_contract({
        "artifact_type": "eventground_wam.b0_full_training_report", "status": "complete",
        "identity_digest": manifest["digest"], "epochs_completed": 5,
        "samples_seen": 8_579_770, "reload_validation_passed": True,
        "checkpoint_sha256": sha256_file(checkpoint), "selected_epoch": 3,
    })
    atomic_json(root / "training-manifest.json", manifest)
    atomic_json(root / "training-report.json", report)
    return normalization, manifest, report


def test_full_bundle_uses_its_own_acceptance_chain_without_cohort_review(tmp_path):
    normalization, manifest, report = bundle(tmp_path)
    actual, _, identity = evaluator.verify_training_bundle(tmp_path, normalization)
    assert actual == manifest
    assert identity.selected_epoch == 3
    assert identity.training_report_digest == report["digest"]
    assert "cohort" not in identity.to_dict()
    (tmp_path / "B0.ckpt").write_bytes(b"changed checkpoint")
    with pytest.raises(TrainingContractError, match="checkpoint SHA-256"):
        evaluator.verify_training_bundle(tmp_path, normalization)


@pytest.mark.parametrize("field,value", [
    ("epochs_completed", 4), ("samples_seen", 5), ("reload_validation_passed", False),
    ("identity_digest", "0" * 64), ("status", "interrupted"),
])
def test_full_bundle_rejects_incomplete_or_mismatched_reports(tmp_path, field, value):
    normalization, _, report = bundle(tmp_path)
    report.pop("digest")
    report[field] = value
    atomic_json(tmp_path / "training-report.json", seal_contract(report))
    with pytest.raises(TrainingContractError, match="acceptance bundle"):
        evaluator.verify_training_bundle(tmp_path, normalization)


def test_clean_manifest_pins_all_forty_tasks_seeds_and_file_hashes(tmp_path):
    for suite in BASE_SUITES:
        _write_base_inventory(tmp_path, suite, [f"pick_up_object_{index}" for index in range(10)])
    protocol = load_evaluation_protocol(evaluator.CLEAN_PROTOCOL)
    manifest, suites = clean_manifest(tmp_path, protocol)
    assert manifest["digest"] == contract_digest(manifest)
    assert manifest["task_count"] == len(manifest["tasks"]) == len(suites) == 40
    assert len({(t["suite"], t["runtime_task_index"]) for t in manifest["tasks"]}) == 40
    assert all(t["perturbation_category"] == "Clean" for t in manifest["tasks"])
    assert all(t["initial_states_file_sha256"] for t in manifest["tasks"])
    assert manifest == clean_manifest(tmp_path, protocol)[0]
    assert clean_instruction("KITCHEN_SCENE10_close_the_drawer") == "close the drawer"
    assert clean_instruction("KITCHEN_SCENE1_close_the_drawer") == "close the drawer"
    protocol["benchmark"]["initial_state"]["state_index"] = 1
    with pytest.raises(ProtocolValidationError):
        validate_evaluation_protocol(protocol)


def test_oft_bundle_accepts_budgeted_exposure_but_requires_full_validation(tmp_path):
    normalization, manifest, _ = bundle(tmp_path)
    config = json.loads((ROOT / "configs/training/week4_b0_oft_v1.json").read_text())
    manifest["recipe"] = config
    manifest = seal_contract({key: value for key, value in manifest.items() if key != "digest"})
    report = seal_contract({
        "artifact_type": "eventground_wam.b0_oft_training_report", "status": "complete",
        "policy_family": config["policy_family"], "identity_digest": manifest["digest"],
        "global_step": 1000, "samples_seen": 8000, "selected_step": 1000,
        "reload_validation_passed": True,
        "checkpoint_sha256": sha256_file(tmp_path / "B0.ckpt"),
        "validation_history": [{"global_step": 1000, "validation": {"sample_count": 212471}}],
        "final_reload_validation": {"sample_count": 212471},
    })
    atomic_json(tmp_path / "training-manifest.json", manifest)
    atomic_json(tmp_path / "training-report.json", report)
    _, _, identity = evaluator.verify_training_bundle(tmp_path, normalization)
    assert identity.selected_epoch is None and identity.selected_step == 1000
    report["final_reload_validation"]["sample_count"] = 8
    atomic_json(tmp_path / "training-report.json",
                seal_contract({key: value for key, value in report.items() if key != "digest"}))
    with pytest.raises(TrainingContractError, match="acceptance bundle"):
        evaluator.verify_training_bundle(tmp_path, normalization)
