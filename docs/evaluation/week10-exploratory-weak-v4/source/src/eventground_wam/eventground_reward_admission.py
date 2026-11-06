"""Bind every RL reward window to preserved independent reference annotations.

No extraction, simulation, inference or annotation occurs here. Passing global
label QA is insufficient when a particular prompt disagrees with the independent
reference: this verifier rejects it without editing or resampling the manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.eventground_qa import (
    ACTIVE_STUDY_DIGEST,
    EVENT_TYPES,
    REVIEW_ATTESTATION,
    LabelReviewError,
    _datetime,
    _interval,
    _ranges,
)

ARTIFACT_TYPE = "eventground_wam.admitted_eventground_reward_evidence"
BINDING_TYPE = "eventground_wam.independent_reference_reward_window"
TIE_ORDER = {"grasp": 0, "contact": 1, "gripper_close": 2}
MAX_ANNOTATION_FILE_BYTES = 64 * 1024**2


class RewardAdmissionError(ValueError):
    """A selected reward window cannot be certified from independent references."""


def _sealed(value, name):
    if not isinstance(value, dict) or value.get("digest") != contract_digest(value):
        raise RewardAdmissionError(f"Changed or unsealed {name}")


def _context(manifest, qa, review_manifest, review):
    for name, record in (("manifest", manifest), ("QA", qa),
                         ("review manifest", review_manifest), ("reference review", review)):
        _sealed(record, name)
    if (
        manifest.get("artifact_type") != "eventground_wam.admitted_eventground_manifest"
        or manifest.get("labels_admitted") is not True
        or manifest.get("training_authorized") is not False
        or manifest.get("study_digest") != ACTIVE_STUDY_DIGEST
        or qa.get("artifact_type") != "eventground_wam.week10_label_qa_assessment"
        or qa.get("study_digest") != ACTIVE_STUDY_DIGEST
        or qa.get("status") != "pass"
        or qa.get("failures") != []
        or qa.get("labels_admitted") is not False
        or qa.get("training_authorized") is not False
        or manifest.get("independent_qa_digest") != qa["digest"]
        or qa.get("review_digest") != review["digest"]
        or qa.get("review_manifest_digest") != review_manifest["digest"]
        or review.get("review_manifest_digest") != review_manifest["digest"]
        or review_manifest.get("capture_completion_digest") != qa.get("capture_completion_digest")
        or review_manifest.get("artifact_type") != "eventground_wam.week10_blind_review_manifest"
        or review_manifest.get("study_digest") != ACTIVE_STUDY_DIGEST
        or not review_manifest.get("preparation_contract_digest")
        or qa.get("preparation_contract_digest") != review_manifest["preparation_contract_digest"]
    ):
        raise RewardAdmissionError("Reward references do not bind admitted training QA/source artifacts")
    required = {
        "episode_count_200_to_300", "all_40_tasks", "independent_full_episode_review",
        "negative_and_missed_event_review",
        *(f"{kind}_negative_frames_reviewed" for kind in EVENT_TYPES),
        *(f"{kind}_{suffix}" for kind in ("contact", "grasp")
          for suffix in ("100_reviewed_positives", "precision", "recall", "boundary_mae")),
    }
    if set(qa.get("gates", {})) != required or any(qa["gates"][key] is not True for key in required):
        raise RewardAdmissionError("Independent QA gates are incomplete")
    reviewer = review.get("reviewer_id")
    attestation = review.get("attestation", {})
    if (
        not reviewer or reviewer != qa.get("reviewer_id")
        or review.get("artifact_type") != "eventground_wam.week10_independent_reference_review"
        or review.get("status") != "complete"
        or review.get("labels_admitted") is not False
        or review.get("reviewer_kind") != "human"
        or review.get("created_by") != reviewer
        or review.get("annotation_origin") != "human_blind_full_episode"
        or review.get("candidate_exposure_before_reference_sealed") is not False
        or review.get("independent_of_labeler_and_training") is not True
        or attestation.get("reviewer_id") != reviewer
        or attestation.get("statement") != REVIEW_ATTESTATION
        or not (_datetime(review.get("started_at")) < _datetime(review.get("completed_at"))
                <= _datetime(attestation.get("signed_at")))
    ):
        raise RewardAdmissionError("Candidate-only, generated or unsigned reference is not independent")
    episodes = {item["recording_id"]: item for item in review_manifest["episodes"]}
    refs = {item["recording_id"]: item for item in review["episodes"]}
    if (
        len(episodes) != len(review_manifest["episodes"])
        or len(refs) != len(review["episodes"])
        or set(episodes) != set(refs)
        or not 200 <= len(episodes) <= 300
        or len({item["task_key"] for item in episodes.values()}) != 40
        or len({item["source_id"] for item in episodes.values()}) != len(episodes)
        or sorted(qa.get("reviewed_recording_ids", [])) != sorted(episodes)
        or sorted(qa.get("source_ids", [])) != sorted(item["source_id"] for item in episodes.values())
        or set(qa.get("candidate_digests", {})) != set(episodes)
    ):
        raise RewardAdmissionError("Independent reference source coverage differs from QA")
    for rid, item in episodes.items():
        ref, frames = refs[rid], item["frames"]
        if (
            ref.get("source_id") != item["source_id"]
            or ref.get("evidence_sha256") != item["evidence_sha256"]
            or not all(_ranges(ref.get("reviewed_frame_ranges"), frames))
            or any(ref.get(key) is not True for key in (
                "both_camera_views_reviewed", "physical_evidence_reviewed",
                "missed_event_search_completed", "negative_cases_reviewed",
            ))
        ):
            raise RewardAdmissionError("Independent episode/source/media/full-review evidence differs")
        for event in ref["events"]:
            _interval(event, frames)
        for ambiguity in ref["ambiguities"]:
            _interval(ambiguity, frames, ambiguity=True)
        ids = [event.get("reference_id") for event in ref["events"]]
        if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
            raise RewardAdmissionError("Reference event identities must be unique and explicit")
        for kind in EVENT_TYPES:
            negatives = _ranges(ref["negative_frame_ranges_by_type"].get(kind), frames)
            occupied = [False] * frames
            for event in ref["events"] + ref["ambiguities"]:
                if event["event_type"] == kind:
                    occupied[event["start"] : event["end"]] = [True] * (event["end"] - event["start"])
            if any(negative == positive for negative, positive in zip(negatives, occupied, strict=True)):
                raise RewardAdmissionError("Reference negative ranges do not cover exactly clear frames")
    rows = manifest["main_rows"]
    if (
        len(rows) != 2000 or len({row["prompt_id"] for row in rows}) != 2000
        or Counter(row["target"]["next_event"] for row in rows)
        != dict.fromkeys((*EVENT_TYPES, "none"), 500)
    ):
        raise RewardAdmissionError("Reward admission requires the exact balanced 2000-prompt universe")
    return episodes, refs


def _reference_prompt_binding(row, item, reference, *, review_manifest_digest, review_digest):
    """Reconstruct the frozen earliest-event target directly from a blind reference."""
    anchor, n = row["anchor"], item["frames"]
    if (
        type(anchor) is not int or not 0 <= anchor < n - 32
        or row["source_id"] != item["source_id"]
        or row["recording_id"] != item["recording_id"]
        or row["recording_sha256"] != item["evidence_sha256"]["recording.h5"]
        or row["prompt_id"] != hashlib.sha256(f"{row['source_id']}:{anchor}".encode()).hexdigest()
    ):
        raise RewardAdmissionError("Prompt source/media/anchor identity or complete-window bound differs")
    stop = anchor + 33
    events = sorted(reference["events"], key=lambda event: (
        event["start"], TIE_ORDER[event["event_type"]], event["object_id"] or "",
    ))
    if any(event["start"] <= anchor < event["end"] for event in events):
        raise RewardAdmissionError("Independent reference has an ongoing event at this anchor")
    if any(event["start"] < stop and event["end"] > anchor for event in reference["ambiguities"]):
        raise RewardAdmissionError("Independent reference has ambiguity in this reward window")
    future = [event for event in events if anchor < event["start"] < stop and not event["left_censored"]]
    first = future[0] if future else None
    if first:
        interval = [first["start"] - anchor, first["end"] - anchor]
        target = {"next_event": first["event_type"],
                  "interval": [interval[0], min(33, interval[1])], "horizon": interval[0]}
    else:
        interval = None
        target = {"next_event": "none", "interval": None, "horizon": None}
        for kind in EVENT_TYPES:
            negatives = _ranges(reference["negative_frame_ranges_by_type"][kind], n)
            if not all(negatives[anchor:stop]):
                raise RewardAdmissionError("None requires independently reviewed clear frames for every event type")
    if row["target"] != target:
        raise RewardAdmissionError(
            "Frozen prompt target disagrees with independent reference; no automatic relabel/resampling"
        )
    binding = {
        "artifact_type": BINDING_TYPE, "version": 1,
        "prompt_id": row["prompt_id"], "source_id": item["source_id"],
        "recording_id": item["recording_id"], "anchor": anchor,
        "recording_sha256": item["evidence_sha256"]["recording.h5"],
        "target": target, "verified_interval": interval, "complete_window": True,
        "reviewed_frame_window": [anchor, stop],
        "review_manifest_digest": review_manifest_digest, "reference_review_digest": review_digest,
        "reference_episode_digest": contract_digest(reference),
        "reference_event_id": first["reference_id"] if first else None,
        "reference_object_id": first["object_id"] if first else None,
        "right_censored_reference": first["right_censored"] if first else False,
        "future_evidence_is_model_input": False,
    }
    return {**binding, "evidence_digest": contract_digest(binding)}


def _resolved(path, annotation_root):
    if not isinstance(path, str) or not path or "://" in path:
        raise RewardAdmissionError("Reference paths must identify local preserved JSON files")
    relative = Path(path)
    if not relative.is_absolute() and ".." in relative.parts:
        raise RewardAdmissionError("Relative reference paths cannot escape annotation root")
    return relative if relative.is_absolute() else Path(annotation_root) / relative


def _read_annotation(path, annotation_root, expected_sha256=None):
    resolved = _resolved(path, annotation_root)
    if resolved.stat().st_size > MAX_ANNOTATION_FILE_BYTES:
        raise RewardAdmissionError("Reference JSON exceeds the bounded annotation-file size")
    with resolved.open("rb") as stream:
        raw = stream.read(MAX_ANNOTATION_FILE_BYTES + 1)
    if len(raw) > MAX_ANNOTATION_FILE_BYTES:
        raise RewardAdmissionError("Reference JSON grew beyond its size bound")
    checksum = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise RewardAdmissionError("Independent annotation file checksum changed")
    value = json.loads(raw)
    _sealed(value, "annotation file")
    return value, checksum


def _build(manifest, qa, review_manifest, review, *, review_manifest_path,
           reference_review_path, annotation_files_sha256):
    try:
        episodes, references = _context(manifest, qa, review_manifest, review)
        mapping = {}
        for row in manifest["main_rows"]:
            rid = row["recording_id"]
            if rid not in episodes or row["label_metadata"]["candidate_digest"] != qa["candidate_digests"][rid]:
                raise RewardAdmissionError("Prompt is not linked to the source candidates reviewed by QA")
            mapping[row["prompt_id"]] = _reference_prompt_binding(
                row, episodes[rid], references[rid],
                review_manifest_digest=review_manifest["digest"], review_digest=review["digest"],
            )
    except (KeyError, TypeError, LabelReviewError) as error:
        raise RewardAdmissionError("Malformed independent reference evidence") from error
    return seal_contract(
        {
            "artifact_type": ARTIFACT_TYPE, "version": 1,
            "manifest_digest": manifest["digest"], "qa_digest": qa["digest"],
            "review_manifest_digest": review_manifest["digest"], "reference_review_digest": review["digest"],
            "review_manifest_path": review_manifest_path, "reference_review_path": reference_review_path,
            "annotation_files_sha256": annotation_files_sha256,
            "reference_reviewer_id": review["reviewer_id"], "prompts": mapping,
            "independently_reviewed": True, "labels_admitted": True,
            "training_authorized": False, "future_evidence_is_model_input": False,
            "reference_annotations_created": 0, "labeler_or_model_called": False,
            "mismatch_policy": "fail_closed_no_relabel_no_resampling",
        }
    )


def build_reward_evidence(manifest, qa, *, review_manifest_path, reference_review_path,
                          annotation_root):
    """Return an immutable reward bundle only from real preserved reference files.

    The caller's QA receipt must already come from trusted independent review;
    SHA seals prove integrity, not a reviewer's identity or honesty.
    """
    if review_manifest_path == reference_review_path:
        raise RewardAdmissionError("Reference review and source manifest must be distinct artifacts")
    review_manifest, manifest_sha = _read_annotation(review_manifest_path, annotation_root)
    review, review_sha = _read_annotation(reference_review_path, annotation_root)
    return _build(
        manifest, qa, review_manifest, review,
        review_manifest_path=review_manifest_path, reference_review_path=reference_review_path,
        annotation_files_sha256={review_manifest_path: manifest_sha, reference_review_path: review_sha},
    )


def verify_reward_evidence_binding(evidence, manifest, qa, *, annotation_root):
    """Recompute all 2000 bindings from bytes, reject copied candidate-only receipts."""
    _sealed(evidence, "reward evidence")
    try:
        source_path, reference_path = evidence["review_manifest_path"], evidence["reference_review_path"]
        hashes = evidence["annotation_files_sha256"]
        if source_path == reference_path or set(hashes) != {source_path, reference_path}:
            raise RewardAdmissionError("Exact independent source-manifest/reference file links required")
        review_manifest, _ = _read_annotation(source_path, annotation_root, hashes[source_path])
        review, _ = _read_annotation(reference_path, annotation_root, hashes[reference_path])
        rebuilt = _build(
            manifest, qa, review_manifest, review,
            review_manifest_path=source_path, reference_review_path=reference_path,
            annotation_files_sha256=hashes,
        )
    except (KeyError, TypeError) as error:
        raise RewardAdmissionError("Missing independent reference file bindings") from error
    if rebuilt != evidence:
        raise RewardAdmissionError("Declared reward evidence differs from independently reconstructed windows")
    return rebuilt["prompts"]
