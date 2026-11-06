"""Blind, full-episode Week 10 review preparation and fail-closed assessment.

This module never supplies reference annotations. SHA seals establish integrity,
not reviewer authenticity: callers must register trusted independent reviewers.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from eventground_wam.demonstration_contracts import contract_digest, seal_contract

EVENT_TYPES = ("contact", "grasp", "gripper_close")
NEW_TYPES = ("contact", "grasp")
ACTIVE_STUDY_DIGEST = "080cca5bd1068992556d8fba6c60215d5136f69d8a6106aee47692d209c704fa"
REVIEW_ATTESTATION = (
    "I independently reviewed every listed episode in full, including negative frames "
    "and missed events, using both cameras and the physical evidence. I wrote the "
    "reference events without viewing the extractor candidates or model predictions. "
    "I have recorded uncertainties and exclusions, and did not copy or generate "
    "reference annotations from the labeler or a model."
)
MATCHING_RULE = "maximum-cardinality same-type/object positive-overlap; matching censor flags"


class LabelReviewError(ValueError):
    """A review cannot be trusted or assessed against the frozen population."""


def _sealed(value: Mapping[str, Any], name: str) -> None:
    if not value.get("digest") or contract_digest(value) != value["digest"]:
        raise LabelReviewError(f"Unsealed or changed {name}")


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and all(
        char.isalnum() or char in "-_" for char in value
    )


def build_review_manifest(capture_contract, capture_completion, *, preparation_contract=None):
    """Select every completed initial source; expose no predictions or task outcomes."""
    _sealed(capture_contract, "capture contract")
    _sealed(capture_completion, "capture completion")
    if capture_contract.get("study_digest") != ACTIVE_STUDY_DIGEST:
        raise LabelReviewError("Review must inherit the unchanged active v3 study")
    if (
        capture_completion.get("status") != "complete"
        or capture_completion.get("contract_digest") != capture_contract["digest"]
    ):
        raise LabelReviewError("Capture completion does not bind the complete capture")
    targets = {
        item["recording_id"]: item
        for item in capture_contract["targets"]
        if item["queue_role"] == "initial_review"
    }
    rows = capture_completion["results"]
    if len(rows) != len(targets) or len(targets) != 200:
        raise LabelReviewError("Initial review must include exactly all 200 recorded sources")
    episodes = []
    seen = set()
    for row in rows:
        rid = row["recording_id"]
        if rid in seen or not _safe_id(rid) or rid not in targets:
            raise LabelReviewError("Unknown or repeated recording identity")
        seen.add(rid)
        item = targets[rid]
        if (
            row.get("status") != "completed"
            or row["source_id"] != item["source_id"]
            or row["frames"] != item["frames"]
            or not _sha(row["source_id"])
        ):
            raise LabelReviewError("Capture source/frame identity mismatch")
        hashes = {
            name: row["evidence_sha256"][name]
            for name in ("recording.h5", "contacts.jsonl", "geometry-map.json")
        }
        if not all(_sha(value) for value in hashes.values()):
            raise LabelReviewError("Missing evidence checksums")
        episodes.append(
            {
                "recording_id": rid,
                "source_id": row["source_id"],
                "task_key": item["task_key"],
                "frames": row["frames"],
                "instruction": item["instruction"],
                "evidence_sha256": hashes,
            }
        )
    if len({row["source_id"] for row in episodes}) != 200:
        raise LabelReviewError("Sources must be unique")
    task_counts = Counter(row["task_key"] for row in episodes)
    if len(task_counts) != 40 or set(task_counts.values()) != {5}:
        raise LabelReviewError("Initial review must retain five sources from each of 40 tasks")
    if preparation_contract is not None:
        _sealed(preparation_contract, "preparation contract")
    return seal_contract(
        {
            "artifact_type": "eventground_wam.week10_blind_review_manifest",
            "version": 1,
            "study_digest": ACTIVE_STUDY_DIGEST,
            "capture_contract_digest": capture_contract["digest"],
            "capture_completion_digest": capture_completion["digest"],
            "preparation_contract_digest": (
                preparation_contract["digest"] if preparation_contract else None
            ),
            "episodes": episodes,
            "control_hz": 20,
            "candidate_data_included": False,
            "matching_rule": MATCHING_RULE,
            "boundary_gate": "both start and end MAE <=2; only fully observed matches",
            "population_quality_confidence_guarantee": False,
            "labels_admitted": False,
            "training_authorized": False,
        }
    )


def make_review_template(manifest):
    """Return an explicitly unfinished form; no false negative or positive is filled in."""
    _sealed(manifest, "review manifest")
    return {
        "artifact_type": "eventground_wam.week10_independent_reference_review",
        "version": 1,
        "status": "unreviewed",
        "review_manifest_digest": manifest["digest"],
        "reviewer_id": None,
        "reviewer_kind": None,
        "annotation_origin": None,
        "created_by": None,
        "started_at": None,
        "completed_at": None,
        "candidate_exposure_before_reference_sealed": None,
        "independent_of_labeler_and_training": None,
        "attestation": {"statement": REVIEW_ATTESTATION, "reviewer_id": None, "signed_at": None},
        "episodes": [
            {
                "recording_id": row["recording_id"],
                "source_id": row["source_id"],
                "evidence_sha256": row["evidence_sha256"].copy(),
                "reviewed_frame_ranges": [],
                "both_camera_views_reviewed": None,
                "physical_evidence_reviewed": None,
                "missed_event_search_completed": None,
                "negative_cases_reviewed": None,
                "events": [],
                "ambiguities": [],
                "negative_frame_ranges_by_type": {kind: [] for kind in EVENT_TYPES},
                "notes": "",
            }
            for row in manifest["episodes"]
        ],
        "labels_admitted": False,
    }


def _datetime(value):
    if not isinstance(value, str):
        raise LabelReviewError("Missing reviewer timestamps")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabelReviewError("Invalid reviewer timestamp") from exc
    if stamp.tzinfo is None:
        raise LabelReviewError("Reviewer timestamps need a timezone")
    return stamp


def _ranges(ranges, frames):
    mask = [False] * frames
    if not isinstance(ranges, list):
        raise LabelReviewError("Frame ranges must be explicit half-open intervals")
    for span in ranges:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] < span[1] <= frames
        ):
            raise LabelReviewError("Invalid half-open frame range")
        if any(mask[span[0] : span[1]]):
            raise LabelReviewError("Repeated review/negative frame coverage")
        mask[span[0] : span[1]] = [True] * (span[1] - span[0])
    return mask


def _interval(row, frames, *, ambiguity=False):
    if row.get("event_type") not in EVENT_TYPES:
        raise LabelReviewError("Unknown event type")
    start, end = row.get("start"), row.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= frames:
        raise LabelReviewError("Invalid event boundaries")
    if row["event_type"] == "gripper_close":
        if row.get("object_id") is not None:
            raise LabelReviewError("Close events cannot claim an object identity")
    elif not isinstance(row.get("object_id"), str) or not row["object_id"]:
        raise LabelReviewError("Contact/grasp requires an independently identified object")
    if ambiguity:
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise LabelReviewError("Ambiguity must have an explicit reason")
    elif (
        type(row.get("left_censored")) is not bool
        or type(row.get("right_censored")) is not bool
        or row["left_censored"] != (start == 0)
        or row["right_censored"] != (end == frames)
    ):
        raise LabelReviewError("Missing or inconsistent censoring flags")


def _overlap(left, right):
    return max(0, min(left["end"], right["end"]) - max(left["start"], right["start"]))


def _same_identity(left, right):
    return (left["event_type"], left["object_id"]) == (right["event_type"], right["object_id"])


def _matched(predictions, references):
    """Deterministic maximum-cardinality bipartite match; timing assessed separately."""
    neighbours = {}
    for pi, pred in enumerate(predictions):
        neighbours[pi] = sorted(
            (
                ri
                for ri, ref in enumerate(references)
                if _same_identity(pred, ref)
                and _overlap(pred, ref)
                and pred["left_censored"] == ref["left_censored"]
                and pred["right_censored"] == ref["right_censored"]
            ),
            key=lambda ri: (-_overlap(pred, references[ri]), ri),
        )
    owners = {}

    def augment(pi, seen):
        for ri in neighbours[pi]:
            if ri in seen:
                continue
            seen.add(ri)
            if ri not in owners or augment(owners[ri], seen):
                owners[ri] = pi
                return True
        return False

    for pi in range(len(predictions)):
        augment(pi, set())
    return sorted((pi, ri) for ri, pi in owners.items())


def _wilson(success, count):
    if not count:
        return None
    z = 1.959963984540054
    p = success / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def assess_label_review(
    manifest, review, candidates_by_recording, *, trusted_reviewer_ids, labeler_actor_ids
):
    """Measure honest independent references; fail gates, never manufacture admission.

    A malformed, incomplete, untrusted or self-generated review raises. A complete
    valid review with insufficient positives or low quality returns status=fail.
    """
    _sealed(manifest, "review manifest")
    _sealed(review, "independent review")
    if (
        manifest.get("artifact_type") != "eventground_wam.week10_blind_review_manifest"
        or manifest.get("study_digest") != ACTIVE_STUDY_DIGEST
        or manifest.get("matching_rule") != MATCHING_RULE
        or not manifest.get("preparation_contract_digest")
    ):
        raise LabelReviewError("Unqualified review design/budget")
    reviewer = review.get("reviewer_id")
    if (
        not isinstance(reviewer, str)
        or reviewer not in set(trusted_reviewer_ids)
        or reviewer in set(labeler_actor_ids)
        or review.get("reviewer_kind") != "human"
        or review.get("created_by") != reviewer
        or review.get("annotation_origin") != "human_blind_full_episode"
        or review.get("candidate_exposure_before_reference_sealed") is not False
        or review.get("independent_of_labeler_and_training") is not True
        or review.get("status") != "complete"
    ):
        raise LabelReviewError("Untrusted, incomplete, non-independent or generated review")
    if review.get("review_manifest_digest") != manifest["digest"]:
        raise LabelReviewError("Review does not bind this source manifest")
    attestation = review.get("attestation", {})
    if (
        attestation.get("reviewer_id") != reviewer
        or attestation.get("statement") != REVIEW_ATTESTATION
        or not (
            _datetime(review.get("started_at"))
            < _datetime(review.get("completed_at"))
            <= _datetime(attestation.get("signed_at"))
        )
    ):
        raise LabelReviewError("Missing independent reviewer attestation")
    episodes = manifest["episodes"]
    if len({row["recording_id"] for row in episodes}) != len(episodes):
        raise LabelReviewError("Duplicate review-manifest recording")
    if len({row["source_id"] for row in episodes}) != len(episodes):
        raise LabelReviewError("Repeated physical source in review")
    expected = {row["recording_id"]: row for row in episodes}
    references = {row["recording_id"]: row for row in review["episodes"]}
    if len(references) != len(review["episodes"]) or set(references) != set(expected):
        raise LabelReviewError("Missing, duplicated or extra full-episode references")
    if set(candidates_by_recording) != set(expected):
        raise LabelReviewError("Candidate population differs from the reviewed population")
    counters = {
        kind: {
            "tp": 0, "fp": 0, "fn": 0, "reviewed_positives": 0,
            "predictions_excluded_for_reference_ambiguity": 0,
            "reference_ambiguities": 0, "extractor_ambiguities": 0,
            "negative_frames_reviewed": 0, "negative_episodes_reviewed": 0,
            "censored_matches": 0, "start_errors": [], "end_errors": [],
        }
        for kind in EVENT_TYPES
    }
    match_ledger = []
    for rid, episode in expected.items():
        ref = references[rid]
        n = episode["frames"]
        candidate = candidates_by_recording[rid]
        _sealed(candidate, f"candidate {rid}")
        if (
            ref.get("source_id") != episode["source_id"]
            or ref.get("evidence_sha256") != episode["evidence_sha256"]
            or candidate.get("source_id") != episode["source_id"]
            or candidate.get("recording_id") != rid
            or candidate.get("frames") != n
            or not all(_ranges(ref.get("reviewed_frame_ranges"), n))
        ):
            raise LabelReviewError("Incomplete review or changed source/evidence identity")
        for key in (
            "both_camera_views_reviewed", "physical_evidence_reviewed",
            "missed_event_search_completed", "negative_cases_reviewed",
        ):
            if ref.get(key) is not True:
                raise LabelReviewError(f"Missing full-episode review check: {key}")
        for rows, ambiguous in (
            (ref["events"], False), (ref["ambiguities"], True),
            (candidate["events"], False), (candidate["ambiguities"], True),
        ):
            for row in rows:
                _interval(row, n, ambiguity=ambiguous)
        ids = [row.get("reference_id") for row in ref["events"]]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
            raise LabelReviewError("Reference events need unique explicit identities")
        for index, event in enumerate(ref["events"]):
            if any(_same_identity(event, other) and _overlap(event, other)
                   for other in ref["events"][index + 1 :]):
                raise LabelReviewError("Overlapping duplicate reference events")
            if any(_same_identity(event, other) and _overlap(event, other)
                   for other in ref["ambiguities"]):
                raise LabelReviewError("A reference event cannot also be uncertain")
        for kind in EVENT_TYPES:
            count = counters[kind]
            truth = [row for row in ref["events"] if row["event_type"] == kind]
            ambiguities = [row for row in ref["ambiguities"] if row["event_type"] == kind]
            predicted = [row for row in candidate["events"] if row["event_type"] == kind]
            negative = _ranges(ref["negative_frame_ranges_by_type"].get(kind), n)
            occupied = [False] * n
            for event in truth + ambiguities:
                occupied[event["start"] : event["end"]] = [True] * (event["end"] - event["start"])
            if any(neg == pos for neg, pos in zip(negative, occupied, strict=True)):
                raise LabelReviewError("Negative coverage must be exactly the clear reference complement")
            eligible = [
                row for row in predicted
                if not any(_same_identity(row, amb) and _overlap(row, amb) for amb in ambiguities)
            ]
            count["predictions_excluded_for_reference_ambiguity"] += len(predicted) - len(eligible)
            count["reference_ambiguities"] += len(ambiguities)
            count["extractor_ambiguities"] += sum(
                row["event_type"] == kind for row in candidate["ambiguities"]
            )
            count["reviewed_positives"] += len(truth)
            count["negative_frames_reviewed"] += sum(negative)
            count["negative_episodes_reviewed"] += int(any(negative))
            pairs = _matched(eligible, truth)
            count["tp"] += len(pairs)
            count["fp"] += len(eligible) - len(pairs)
            count["fn"] += len(truth) - len(pairs)
            for pi, ri in pairs:
                pred, true = eligible[pi], truth[ri]
                censored = true["left_censored"] or true["right_censored"]
                count["censored_matches"] += int(censored)
                if not censored:
                    count["start_errors"].append(abs(pred["start"] - true["start"]))
                    count["end_errors"].append(abs(pred["end"] - true["end"]))
                match_ledger.append(
                    {"recording_id": rid, "event_type": kind,
                     "candidate_event_id": pred.get("event_id"),
                     "reference_id": true["reference_id"], "censored": censored,
                     "absolute_start_error": abs(pred["start"] - true["start"]),
                     "absolute_end_error": abs(pred["end"] - true["end"])}
                )
    metrics = {}
    for kind, count in counters.items():
        start_errors, end_errors = count.pop("start_errors"), count.pop("end_errors")
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        metrics[kind] = {
            **count,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision_wilson_95": _wilson(tp, tp + fp),
            "recall_wilson_95": _wilson(tp, tp + fn),
            "fully_observed_matches": len(start_errors),
            "start_mae_frames": sum(start_errors) / len(start_errors) if start_errors else None,
            "end_mae_frames": sum(end_errors) / len(end_errors) if end_errors else None,
        }
    gates = {
        "episode_count_200_to_300": 200 <= len(episodes) <= 300,
        "all_40_tasks": len({row["task_key"] for row in episodes}) == 40,
        "independent_full_episode_review": True,
        "negative_and_missed_event_review": True,
    }
    for kind, metric in metrics.items():
        if kind in NEW_TYPES:
            gates[f"{kind}_100_reviewed_positives"] = metric["reviewed_positives"] >= 100
            gates[f"{kind}_precision"] = metric["precision"] is not None and metric["precision"] >= 0.95
            gates[f"{kind}_recall"] = metric["recall"] is not None and metric["recall"] >= 0.95
            gates[f"{kind}_boundary_mae"] = all(
                metric[key] is not None and metric[key] <= 2
                for key in ("start_mae_frames", "end_mae_frames")
            )
        gates[f"{kind}_negative_frames_reviewed"] = metric["negative_frames_reviewed"] > 0
    failures = [name for name, passed in gates.items() if not passed]
    return seal_contract(
        {
            "artifact_type": "eventground_wam.week10_label_qa_assessment",
            "version": 1,
            "study_digest": ACTIVE_STUDY_DIGEST,
            "status": "fail" if failures else "pass",
            "review_manifest_digest": manifest["digest"],
            "review_digest": review["digest"],
            "capture_completion_digest": manifest["capture_completion_digest"],
            "preparation_contract_digest": manifest["preparation_contract_digest"],
            "candidate_digests": {rid: row["digest"] for rid, row in candidates_by_recording.items()},
            "source_ids": [row["source_id"] for row in episodes],
            "reviewed_recording_ids": list(expected),
            "reviewer_id": reviewer,
            "metrics_by_type": metrics,
            "matching_rule": MATCHING_RULE,
            "match_ledger": match_ledger,
            "gates": gates,
            "failures": failures,
            "population_quality_confidence_guarantee": False,
            "labels_admitted": False,
            "training_authorized": False,
        }
    )
