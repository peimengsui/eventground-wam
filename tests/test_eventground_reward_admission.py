"""Fictional reference fixtures only; these never constitute real annotation QA."""

import copy
import hashlib
import json

import pytest

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.eventground_qa import (
    ACTIVE_STUDY_DIGEST,
    MATCHING_RULE,
    REVIEW_ATTESTATION,
    assess_label_review,
    make_review_template,
)
from eventground_wam.eventground_reward_admission import (
    RewardAdmissionError,
    _reference_prompt_binding,
    build_reward_evidence,
    verify_reward_evidence_binding,
)


def reseal(record):
    return seal_contract({key: value for key, value in record.items() if key != "digest"})


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def reference_fixture(tmp_path):
    episodes = [
        {"recording_id": f"synthetic-{i:03}", "source_id": digest(f"source-{i}"),
         "task_key": f"task-{i % 40:02}", "frames": 128, "instruction": "synthetic fixture",
         "evidence_sha256": {name: digest(f"{name}:{i}")
                             for name in ("recording.h5", "contacts.jsonl", "geometry-map.json")}}
        for i in range(200)
    ]
    source = seal_contract(
        {"artifact_type": "eventground_wam.week10_blind_review_manifest", "version": 1,
         "study_digest": ACTIVE_STUDY_DIGEST, "matching_rule": MATCHING_RULE,
         "preparation_contract_digest": digest("synthetic budget"),
         "capture_completion_digest": digest("synthetic capture"), "episodes": episodes}
    )
    review = make_review_template(source)
    review.update(
        status="complete", reviewer_id="fictional-test-person", reviewer_kind="human",
        created_by="fictional-test-person", annotation_origin="human_blind_full_episode",
        candidate_exposure_before_reference_sealed=False, independent_of_labeler_and_training=True,
        started_at="2026-10-18T00:00:00Z", completed_at="2026-10-18T01:00:00Z",
        attestation={"statement": REVIEW_ATTESTATION, "reviewer_id": "fictional-test-person",
                     "signed_at": "2026-10-18T01:01:00Z"},
    )
    candidates = {}
    rows = []
    for index, (item, ref) in enumerate(zip(episodes, review["episodes"], strict=True)):
        events = [
            {"reference_id": f"manual-{kind}", "event_type": kind,
             "object_id": None if kind == "gripper_close" else "object",
             "start": start, "end": end, "left_censored": False, "right_censored": False}
            for kind, start, end in (("gripper_close", 20, 25), ("contact", 60, 66), ("grasp", 85, 90))
        ]
        ref.update(
            reviewed_frame_ranges=[[0, 128]], both_camera_views_reviewed=True,
            physical_evidence_reviewed=True, missed_event_search_completed=True,
            negative_cases_reviewed=True, events=events,
            negative_frame_ranges_by_type={e["event_type"]: [[0, e["start"]], [e["end"], 128]]
                                          for e in events},
        )
        predicted = [{**event, "event_id": f"pred-{event['event_type']}"} for event in events]
        candidate = seal_contract(
            {"recording_id": item["recording_id"], "source_id": item["source_id"],
             "frames": 128, "events": predicted, "ambiguities": []}
        )
        candidates[item["recording_id"]] = candidate
        count = 3 if index < 100 else 2
        for kind, base, start, end in (
            ("gripper_close", 0, 20, 25), ("contact", 30, 60, 66),
            ("grasp", 66, 85, 90), ("none", 91, None, None),
        ):
            for anchor in range(base, base + count):
                target = {"next_event": kind,
                          "interval": [start - anchor, min(33, end - anchor)] if start else None,
                          "horizon": start - anchor if start else None}
                rows.append(
                    {"prompt_id": digest(f"{item['source_id']}:{anchor}"),
                     "recording_id": item["recording_id"], "source_id": item["source_id"],
                     "recording_sha256": item["evidence_sha256"]["recording.h5"],
                     "anchor": anchor, "target": target,
                     "label_metadata": {"candidate_digest": candidate["digest"]}}
                )
    review = reseal(review)
    qa = assess_label_review(source, review, candidates,
                             trusted_reviewer_ids=["fictional-test-person"],
                             labeler_actor_ids=["fictional-test-labeler"])
    assert qa["status"] == "pass"
    manifest = seal_contract(
        {"artifact_type": "eventground_wam.admitted_eventground_manifest",
         "study_digest": ACTIVE_STUDY_DIGEST, "independent_qa_digest": qa["digest"],
         "labels_admitted": True, "training_authorized": False, "main_rows": rows}
    )
    write(tmp_path / "sources.json", source)
    write(tmp_path / "references.json", review)
    return {"root": tmp_path, "source": source, "review": review, "qa": qa,
            "manifest": manifest, "candidates": candidates}


def build(fixture):
    return build_reward_evidence(
        fixture["manifest"], fixture["qa"], annotation_root=fixture["root"],
        review_manifest_path="sources.json", reference_review_path="references.json",
    )


def verify(fixture, evidence):
    return verify_reward_evidence_binding(
        evidence, fixture["manifest"], fixture["qa"], annotation_root=fixture["root"]
    )


def test_all_prompts_are_reconstructed_from_actual_reference_bytes(reference_fixture):
    evidence = build(reference_fixture)
    result = verify(reference_fixture, evidence)
    assert len(result) == 2000 and result == evidence["prompts"]
    assert evidence["reference_annotations_created"] == 0
    assert evidence["training_authorized"] is False
    assert set(evidence["annotation_files_sha256"]) == {"sources.json", "references.json"}
    for row in result.values():
        raw = {key: value for key, value in row.items() if key != "evidence_digest"}
        assert row["evidence_digest"] == contract_digest(raw)
        assert row["future_evidence_is_model_input"] is False
        if row["target"]["next_event"] == "none":
            assert row["verified_interval"] is None
        else:
            assert row["verified_interval"][0] == row["target"]["horizon"]


def test_copied_candidate_only_evidence_cannot_pass(reference_fixture):
    evidence = build(reference_fixture)
    for row in evidence["prompts"].values():
        for key in ("reference_event_id", "reference_episode_digest", "reference_review_digest"):
            row.pop(key)
    with pytest.raises(RewardAdmissionError, match="reconstructed"):
        verify(reference_fixture, reseal(evidence))


@pytest.mark.parametrize("field,value", [
    ("evidence_digest", "0" * 64), ("source_id", "0" * 64), ("anchor", 4),
    ("recording_sha256", "0" * 64), ("reference_event_id", "invented event"),
    ("reference_object_id", "wrong object"), ("complete_window", False),
    ("verified_interval", [20, 26]),
])
def test_changed_declared_binding_rejected_even_when_resealed(reference_fixture, field, value):
    evidence = build(reference_fixture)
    next(iter(evidence["prompts"].values()))[field] = value
    with pytest.raises(RewardAdmissionError, match="reconstructed"):
        verify(reference_fixture, reseal(evidence))


def test_annotation_hash_is_not_decorative(reference_fixture):
    evidence = build(reference_fixture)
    path = reference_fixture["root"] / "references.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RewardAdmissionError, match="checksum"):
        verify(reference_fixture, evidence)


def test_unrelated_hashed_file_cannot_substitute_for_reference(reference_fixture):
    evidence = build(reference_fixture)
    write(reference_fixture["root"] / "unrelated.json", seal_contract({"unrelated": True}))
    evidence["reference_review_path"] = "unrelated.json"
    evidence["annotation_files_sha256"] = {
        "sources.json": evidence["annotation_files_sha256"]["sources.json"],
        "unrelated.json": hashlib.sha256((reference_fixture["root"] / "unrelated.json").read_bytes()).hexdigest(),
    }
    with pytest.raises(RewardAdmissionError, match="bind"):
        verify(reference_fixture, reseal(evidence))


def test_changed_reference_cannot_reuse_old_qa_receipt(reference_fixture):
    review = copy.deepcopy(reference_fixture["review"])
    review["episodes"][0]["events"][1]["start"] += 1
    write(reference_fixture["root"] / "references.json", reseal(review))
    with pytest.raises(RewardAdmissionError, match="bind"):
        build(reference_fixture)


def test_passing_average_qa_does_not_admit_a_mismatched_prompt(reference_fixture):
    review = copy.deepcopy(reference_fixture["review"])
    ref = review["episodes"][0]
    ref["events"][1]["start"] += 1
    ref["negative_frame_ranges_by_type"]["contact"][0][1] += 1
    review = reseal(review)
    qa = assess_label_review(reference_fixture["source"], review, reference_fixture["candidates"],
                             trusted_reviewer_ids=["fictional-test-person"], labeler_actor_ids=[])
    assert qa["status"] == "pass"  # Small average error is permitted by global QA.
    reference_fixture["qa"] = qa
    reference_fixture["manifest"]["independent_qa_digest"] = qa["digest"]
    reference_fixture["manifest"] = reseal(reference_fixture["manifest"])
    write(reference_fixture["root"] / "references.json", review)
    with pytest.raises(RewardAdmissionError, match="disagrees"):
        build(reference_fixture)


@pytest.mark.parametrize("patch", [{"reviewer_kind": "model"},
                                    {"annotation_origin": "candidate_copy"},
                                    {"candidate_exposure_before_reference_sealed": True}])
def test_generated_reference_origin_rejected(reference_fixture, patch):
    review = copy.deepcopy(reference_fixture["review"])
    review.update(patch)
    review = reseal(review)
    qa = copy.deepcopy(reference_fixture["qa"])
    qa["review_digest"] = review["digest"]
    reference_fixture["qa"] = reseal(qa)
    manifest = copy.deepcopy(reference_fixture["manifest"])
    manifest["independent_qa_digest"] = reference_fixture["qa"]["digest"]
    reference_fixture["manifest"] = reseal(manifest)
    write(reference_fixture["root"] / "references.json", review)
    with pytest.raises(RewardAdmissionError, match="not independent"):
        build(reference_fixture)


def direct_binding(fixture, *, anchor, target, reference=None):
    item = fixture["source"]["episodes"][0]
    row = copy.deepcopy(fixture["manifest"]["main_rows"][0])
    row.update(anchor=anchor, prompt_id=digest(f"{item['source_id']}:{anchor}"), target=target)
    return _reference_prompt_binding(
        row, item, reference or fixture["review"]["episodes"][0],
        review_manifest_digest=fixture["source"]["digest"], review_digest=fixture["review"]["digest"],
    )


@pytest.mark.parametrize("anchor,error", [(20, "ongoing"), (100, "complete-window")])
def test_ongoing_and_incomplete_windows_rejected(reference_fixture, anchor, error):
    with pytest.raises(RewardAdmissionError, match=error):
        direct_binding(reference_fixture, anchor=anchor,
                       target={"next_event": "none", "interval": None, "horizon": None})


def test_ambiguous_window_is_never_negative(reference_fixture):
    ref = copy.deepcopy(reference_fixture["review"]["episodes"][0])
    ref["ambiguities"] = [{"event_type": "contact", "object_id": "object", "start": 92,
                            "end": 94, "reason": "Uncertain evidence"}]
    with pytest.raises(RewardAdmissionError, match="ambiguity"):
        direct_binding(reference_fixture, anchor=91, reference=ref,
                       target={"next_event": "none", "interval": None, "horizon": None})


def test_negative_requires_explicit_review_of_all_three_types(reference_fixture):
    ref = copy.deepcopy(reference_fixture["review"]["episodes"][0])
    ref["negative_frame_ranges_by_type"]["grasp"] = [[0, 85], [92, 128]]
    with pytest.raises(RewardAdmissionError, match="clear frames"):
        direct_binding(reference_fixture, anchor=91, reference=ref,
                       target={"next_event": "none", "interval": None, "horizon": None})


def test_frozen_tie_priority_and_clipped_right_censoring(reference_fixture):
    ref = copy.deepcopy(reference_fixture["review"]["episodes"][0])
    grasp = ref["events"][2]
    grasp.update(start=60, end=128, right_censored=True)
    result = direct_binding(reference_fixture, anchor=30, reference=ref,
                            target={"next_event": "grasp", "interval": [30, 33], "horizon": 30})
    assert result["reference_event_id"] == "manual-grasp"
    assert result["verified_interval"] == [30, 98]
    assert result["right_censored_reference"] is True


def test_first_event_at_window_stop_is_outside_future_horizon(reference_fixture):
    result = direct_binding(reference_fixture, anchor=27,
                            target={"next_event": "none", "interval": None, "horizon": None})
    assert result["verified_interval"] is None  # Contact starts at stop=60, excluded.


def test_missing_reference_media_link_rejected(reference_fixture):
    reference_fixture["manifest"]["main_rows"][0]["recording_sha256"] = "0" * 64
    reference_fixture["manifest"] = reseal(reference_fixture["manifest"])
    with pytest.raises(RewardAdmissionError, match="source/media"):
        build(reference_fixture)


def test_relative_reference_path_cannot_escape_root(reference_fixture):
    with pytest.raises(RewardAdmissionError, match="escape"):
        build_reward_evidence(reference_fixture["manifest"], reference_fixture["qa"],
                              annotation_root=reference_fixture["root"],
                              review_manifest_path="../sources.json", reference_review_path="references.json")
