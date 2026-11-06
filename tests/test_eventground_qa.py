"""Synthetic-only independent-review gate tests; none are production annotations."""

import copy
import json
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_qa import (
    EVENT_TYPES,
    REVIEW_ATTESTATION,
    LabelReviewError,
    _matched,
    assess_label_review,
    build_review_manifest,
    make_review_template,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "docs/evaluation/week10-production-capture-visual-v2"


def reseal(value):
    return seal_contract({key: item for key, item in value.items() if key != "digest"})


def negative_ranges(events, ambiguities, n, kind):
    mask = [True] * n
    for event in events + ambiguities:
        if event["event_type"] == kind:
            mask[event["start"] : event["end"]] = [False] * (event["end"] - event["start"])
    edges = []
    start = None
    for index, negative in enumerate(mask + [False]):
        if negative and start is None:
            start = index
        elif not negative and start is not None:
            edges.append([start, index])
            start = None
    return edges


@pytest.fixture
def synthetic_case():
    contract = json.loads((CAPTURE / "recording-contract.json").read_text())
    completion = json.loads((CAPTURE / "run/completion.json").read_text())
    manifest = build_review_manifest(
        contract, completion,
        preparation_contract=seal_contract({"test_budget_only": True}),
    )
    review = make_review_template(manifest)
    review.update(
        status="complete", reviewer_id="synthetic-reviewer", reviewer_kind="human",
        annotation_origin="human_blind_full_episode", created_by="synthetic-reviewer",
        started_at="2026-10-18T00:00:00+00:00", completed_at="2026-10-18T01:00:00+00:00",
        candidate_exposure_before_reference_sealed=False,
        independent_of_labeler_and_training=True,
        attestation={"statement": REVIEW_ATTESTATION, "reviewer_id": "synthetic-reviewer",
                     "signed_at": "2026-10-18T01:01:00+00:00"},
    )
    candidates = {}
    for row, item in zip(review["episodes"], manifest["episodes"], strict=True):
        n = item["frames"]
        events = [
            {"reference_id": f"manual-{kind}", "event_type": kind,
             "object_id": None if kind == "gripper_close" else "synthetic_object",
             "start": start, "end": end, "left_censored": False, "right_censored": False}
            for kind, start, end in (("contact", 10, 20), ("grasp", 12, 22), ("gripper_close", 3, 9))
        ]
        row.update(
            reviewed_frame_ranges=[[0, n]], both_camera_views_reviewed=True,
            physical_evidence_reviewed=True, missed_event_search_completed=True,
            negative_cases_reviewed=True, events=events,
            negative_frame_ranges_by_type={
                kind: negative_ranges(events, [], n, kind) for kind in EVENT_TYPES
            },
        )
        predicted = []
        for event in events:
            pred = dict(event)
            pred["event_id"] = pred.pop("reference_id").replace("manual", "predicted")
            predicted.append(pred)
        candidates[row["recording_id"]] = seal_contract(
            {"recording_id": row["recording_id"], "source_id": row["source_id"],
             "frames": n, "events": predicted, "ambiguities": []}
        )
    return manifest, review, candidates


def assess(case, **kwargs):
    manifest, review, candidates = case
    return assess_label_review(
        manifest, reseal(review), candidates,
        trusted_reviewer_ids=kwargs.get("trusted", ["synthetic-reviewer"]),
        labeler_actor_ids=kwargs.get("actors", ["synthetic-labeler"]),
    )


def test_templates_are_unreviewed_and_hide_candidates(synthetic_case):
    manifest, _, _ = synthetic_case
    template = make_review_template(manifest)
    assert template["status"] == "unreviewed"
    assert template["reviewer_id"] is None
    assert len(template["episodes"]) == 200
    assert all(not row["events"] and not row["reviewed_frame_ranges"] for row in template["episodes"])
    assert "final_environment_done" not in json.dumps(manifest)
    with pytest.raises(LabelReviewError, match="Untrusted"):
        assess((manifest, template, synthetic_case[2]))


def test_full_synthetic_review_measures_without_admitting(synthetic_case):
    result = assess(synthetic_case)
    assert result["status"] == "pass"
    assert all(result["gates"].values())
    assert result["labels_admitted"] is False
    assert result["training_authorized"] is False
    for metric in result["metrics_by_type"].values():
        assert (metric["tp"], metric["fp"], metric["fn"]) == (200, 0, 0)
        assert metric["start_mae_frames"] == metric["end_mae_frames"] == 0
        assert metric["precision_wilson_95"][0] < 1


@pytest.mark.parametrize("patch", [
    {"reviewer_kind": "model"}, {"annotation_origin": "model_generated"},
    {"created_by": "synthetic-labeler"}, {"candidate_exposure_before_reference_sealed": True},
    {"independent_of_labeler_and_training": False}, {"status": "partial"},
])
def test_generated_or_nonindependent_review_rejected(synthetic_case, patch):
    synthetic_case[1].update(patch)
    with pytest.raises(LabelReviewError, match="non-independent"):
        assess(synthetic_case)


def test_reviewer_registration_and_generator_identity_are_external(synthetic_case):
    with pytest.raises(LabelReviewError):
        assess(synthetic_case, trusted=[])
    with pytest.raises(LabelReviewError):
        assess(synthetic_case, actors=["synthetic-reviewer"])


def test_falsified_hash_rejected(synthetic_case):
    manifest, review, candidates = synthetic_case
    candidates[next(iter(candidates))]["events"][0]["start"] = 9
    with pytest.raises(LabelReviewError, match="changed"):
        assess((manifest, review, candidates))


@pytest.mark.parametrize("mutation", ["missing_episode", "duplicate_episode", "partial_frames",
                                      "negative_check", "missed_check", "camera_check",
                                      "source", "evidence", "negative_range"])
def test_incomplete_coverage_rejected(synthetic_case, mutation):
    review = synthetic_case[1]
    row = review["episodes"][0]
    if mutation == "missing_episode":
        review["episodes"].pop()
    elif mutation == "duplicate_episode":
        review["episodes"].append(copy.deepcopy(row))
    elif mutation == "partial_frames":
        row["reviewed_frame_ranges"][0][1] -= 1
    elif mutation == "negative_check":
        row["negative_cases_reviewed"] = False
    elif mutation == "missed_check":
        row["missed_event_search_completed"] = False
    elif mutation == "camera_check":
        row["both_camera_views_reviewed"] = False
    elif mutation == "source":
        row["source_id"] = "0" * 64
    elif mutation == "evidence":
        row["evidence_sha256"]["recording.h5"] = "0" * 64
    elif mutation == "negative_range":
        row["negative_frame_ranges_by_type"]["contact"] = [[0, 245]]
    with pytest.raises(LabelReviewError):
        assess(synthetic_case)


def test_false_negatives_and_false_positives_fail_point_gates(synthetic_case):
    candidates = synthetic_case[2]
    for index, (rid, old) in enumerate(candidates.items()):
        candidate = copy.deepcopy(old)
        if index < 11:
            candidate["events"] = [e for e in candidate["events"] if e["event_type"] != "contact"]
            extra = copy.deepcopy(candidate["events"][0])
            extra.update(start=30, end=40, event_id="false-positive-grasp")
            candidate["events"].append(extra)
        candidates[rid] = reseal(candidate)
    result = assess(synthetic_case)
    assert result["status"] == "fail"
    assert result["metrics_by_type"]["contact"]["fn"] == 11
    assert result["metrics_by_type"]["grasp"]["fp"] == 11
    assert "contact_recall" in result["failures"]
    assert "grasp_precision" in result["failures"]


def test_boundary_metrics_are_separate_start_and_end(synthetic_case):
    for rid, candidate in synthetic_case[2].items():
        candidate["events"][0]["start"] += 3
        synthetic_case[2][rid] = reseal(candidate)
    result = assess(synthetic_case)
    assert result["metrics_by_type"]["contact"]["precision"] == 1
    assert result["metrics_by_type"]["contact"]["start_mae_frames"] == 3
    assert result["metrics_by_type"]["contact"]["end_mae_frames"] == 0
    assert "contact_boundary_mae" in result["failures"]


def test_censored_intervals_not_used_for_boundary_accuracy(synthetic_case):
    for row in synthetic_case[1]["episodes"]:
        event = row["events"][0]
        event.update(start=0, left_censored=True)
        row["negative_frame_ranges_by_type"]["contact"] = [[20, row["reviewed_frame_ranges"][0][1]]]
        rid = row["recording_id"]
        synthetic_case[2][rid]["events"][0].update(start=0, left_censored=True)
        synthetic_case[2][rid] = reseal(synthetic_case[2][rid])
    result = assess(synthetic_case)
    metric = result["metrics_by_type"]["contact"]
    assert metric["censored_matches"] == 200
    assert metric["start_mae_frames"] is None
    assert "contact_boundary_mae" in result["failures"]


def test_missing_censor_flag_rejected(synthetic_case):
    synthetic_case[1]["episodes"][0]["events"][0].pop("right_censored")
    with pytest.raises(LabelReviewError, match="censoring"):
        assess(synthetic_case)


def test_ambiguous_predictions_preserved_in_report_and_not_negative(synthetic_case):
    row = synthetic_case[1]["episodes"][0]
    row["ambiguities"] = [
        {"event_type": "contact", "object_id": "synthetic_object", "start": 30, "end": 40,
         "reason": "Independent reviewer cannot determine collision boundary"}
    ]
    n = row["reviewed_frame_ranges"][0][1]
    row["negative_frame_ranges_by_type"]["contact"] = negative_ranges(
        row["events"], row["ambiguities"], n, "contact"
    )
    rid = row["recording_id"]
    predicted = copy.deepcopy(synthetic_case[2][rid]["events"][0])
    predicted.update(start=30, end=40, event_id="uncertain-contact")
    synthetic_case[2][rid]["events"].append(predicted)
    synthetic_case[2][rid] = reseal(synthetic_case[2][rid])
    result = assess(synthetic_case)
    metric = result["metrics_by_type"]["contact"]
    assert metric["reference_ambiguities"] == 1
    assert metric["predictions_excluded_for_reference_ambiguity"] == 1
    assert metric["fp"] == 0


def test_matching_is_one_to_one_and_maximum_cardinality():
    def interval(start, end):
        return {"event_type": "contact", "object_id": "o", "start": start, "end": end,
                "left_censored": False, "right_censored": False}
    predictions = [interval(1, 15), interval(2, 5)]
    references = [interval(2, 6), interval(10, 16)]
    assert _matched(predictions, references) == [(0, 1), (1, 0)]


def test_low_positive_coverage_is_not_fixed_by_duplicates(synthetic_case):
    for row in synthetic_case[1]["episodes"][99:]:
        row["events"] = [e for e in row["events"] if e["event_type"] != "grasp"]
        row["negative_frame_ranges_by_type"]["grasp"] = row["reviewed_frame_ranges"]
        rid = row["recording_id"]
        synthetic_case[2][rid]["events"] = [
            e for e in synthetic_case[2][rid]["events"] if e["event_type"] != "grasp"
        ]
        synthetic_case[2][rid] = reseal(synthetic_case[2][rid])
    result = assess(synthetic_case)
    assert result["metrics_by_type"]["grasp"]["precision"] == 1
    assert "grasp_100_reviewed_positives" in result["failures"]


def test_manifest_requires_all_sources_and_all_tasks():
    contract = json.loads((CAPTURE / "recording-contract.json").read_text())
    completion = json.loads((CAPTURE / "run/completion.json").read_text())
    completion["results"].pop()
    with pytest.raises(LabelReviewError, match="200"):
        build_review_manifest(contract, reseal(completion))


def test_review_without_sealed_preparation_budget_is_rejected(synthetic_case):
    manifest, review, candidates = synthetic_case
    manifest["preparation_contract_digest"] = None
    manifest = reseal(manifest)
    review["review_manifest_digest"] = manifest["digest"]
    with pytest.raises(LabelReviewError, match="budget"):
        assess((manifest, review, candidates))
