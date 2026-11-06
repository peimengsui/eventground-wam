import copy
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_week10_weak_supervision import make_contract, make_protocol
from qualify_week10_weak_supervision import tree_digest

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_training import require_training_admission
from eventground_wam.eventground_weak_supervision import (
    audit_row,
    build_dataset,
    overfit_gates,
    sealed,
    verify_pilot,
    verify_protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def reseal(value):
    return seal_contract({k: v for k, v in value.items() if k != "digest"})


@pytest.fixture(scope="module")
def artifacts():
    protocol = make_protocol(ROOT)
    dataset = build_dataset(ROOT, protocol)
    recipe = sealed(ROOT / "configs/training/week10_eventground_numerical_v2.json")
    return protocol, dataset, recipe, make_contract(protocol, dataset, recipe)


def test_actual_pool_reproduces_conservative_diagnosis_and_keeps_provenance(artifacts):
    protocol, dataset, recipe, contract = artifacts
    assert dataset["coverage"]["class_counts"] == {"none": 500, "contact": 327, "grasp": 29, "gripper_close": 476}
    assert dataset["coverage"]["selected_tasks"] == 40
    assert dataset["coverage"]["ambiguities_retained"] == 1816
    rows = dataset["main_rows"]
    assert len({r["prompt_id"] for r in rows}) == 1332
    assert max(Counter(r["source_id"] for r in rows).values()) <= 16
    assert max(Counter(r["event_id"] for r in rows if r["event_id"]).values()) <= 4
    assert len(set(dataset["qualification_prompt_ids"])) == 16
    assert not dataset["independent_qa_passed"] and not dataset["labels_admitted"]
    assert not dataset["formal_training_authorized"]
    assert verify_pilot(contract, protocol, dataset, recipe) == contract
    assert build_dataset(ROOT, protocol) == dataset
    with pytest.raises(ValueError, match="Formal training"):
        require_training_admission(dataset, study_digest=protocol["parent_study_digest"],
                                   recipe_digest=recipe["digest"], manifest_digest=dataset["digest"])


@pytest.mark.parametrize("field,value", [("discard_ambiguities", True), ("allow_ongoing_event_anchors", True),
    ("independent_qa_passed", True), ("heldout_payload_reads", 1), ("formal_training_authorized", True)])
def test_protocol_cannot_silently_change_scope(artifacts, field, value):
    changed = copy.deepcopy(artifacts[0]); changed[field] = value
    with pytest.raises(ValueError):
        verify_protocol(reseal(changed))


@pytest.mark.parametrize("field,value", [("logical_updates", 65), ("attempts", 2),
    ("maximum_executed_updates_including_replay", 66), ("wall_seconds", 3600),
    ("independent_qa_passed", True), ("prior_qualification_active_seconds", 10000)])
def test_pilot_bounds_fail_closed(artifacts, field, value):
    protocol, dataset, recipe, contract = artifacts
    changed = copy.deepcopy(contract); changed[field] = value
    with pytest.raises(ValueError):
        verify_pilot(reseal(changed), protocol, dataset, recipe)


def test_current_window_ambiguity_ongoing_target_and_input_rejections(artifacts):
    protocol, dataset, _, _ = artifacts
    row = next(r for r in dataset["main_rows"] if r["target"]["next_event"] == "grasp")
    original = sealed(ROOT / protocol["inputs"]["candidate_root"] / row["recording_id"] / "event-candidates.json")
    assert audit_row(row, original)["event_id"] == row["event_id"]
    changed = copy.deepcopy(original)
    changed["ambiguities"].append({"start": row["anchor"], "end": row["anchor"]+1})
    with pytest.raises(ValueError, match="ambiguity"):
        audit_row(row, reseal(changed))
    changed = copy.deepcopy(original)
    changed["events"].append({"start": row["anchor"], "end": row["anchor"]+1})
    with pytest.raises(ValueError, match="ongoing"):
        audit_row(row, reseal(changed))
    changed_row = copy.deepcopy(row); changed_row["target"]["interval"][0] += 1
    with pytest.raises(ValueError, match="earliest-onset"):
        audit_row(changed_row, original)
    changed_row = copy.deepcopy(row); changed_row["input_datasets"].append("action")
    with pytest.raises(ValueError, match="Noncausal"):
        audit_row(changed_row, original)


def test_pinned_review_bytes_cannot_be_substituted(artifacts):
    protocol = copy.deepcopy(artifacts[0])
    protocol["input_files_sha256"][protocol["inputs"]["review_snapshot"]] = "0"*64
    with pytest.raises(ValueError, match="input/code changed"):
        build_dataset(ROOT, reseal(protocol))


def test_thresholds_and_bitwise_state_digest(artifacts):
    import torch
    thresholds = artifacts[3]["thresholds"]
    initial = {"loss": 2., "event_accuracy": .25, "joint_two_frame_accuracy": 0.}
    final = {"loss": .9, "event_accuracy": .875, "joint_two_frame_accuracy": .75}
    assert all(overfit_gates(initial, final, thresholds).values())
    final["joint_two_frame_accuracy"] = .749
    assert not overfit_gates(initial, final, thresholds)["joint_two_frame_accuracy"]
    final["loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        overfit_gates(initial, final, thresholds)
    x = {"parameter": torch.tensor([1., 2.]), "step": 9}
    assert tree_digest(x) == tree_digest(copy.deepcopy(x))
    y = copy.deepcopy(x); y["parameter"][0] += 1e-6
    assert tree_digest(x) != tree_digest(y)
    y = copy.deepcopy(x); y["step"] = 10
    assert tree_digest(x) != tree_digest(y)
