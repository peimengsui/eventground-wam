"""Exploratory weak-label preparation, separate from the original independent-QA gate."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.eventground_manifest import CLASSES, _candidate_rows, _sample


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if contract_digest(value) != value.get("digest"):
        raise ValueError(f"Unsealed or changed artifact: {path}")
    return value


def verify_protocol(protocol):
    if protocol.get("digest") != contract_digest(protocol):
        raise ValueError("Unsealed exploratory protocol")
    expected = {
        "artifact_type": "eventground_wam.week10_exploratory_weak_supervision",
        "version": 4, "independent_qa_passed": False,
        "formal_training_authorized": False, "qualification_authorized": True,
        "allow_ongoing_event_anchors": False, "discard_ambiguities": False,
        "heldout_payload_reads": 0, "maximum_prompts_per_source": 16,
        "maximum_anchors_per_event": 4, "maximum_prompts_per_class": 500,
        "qualification_prompts_per_class": 4,
    }
    if any(protocol.get(k) != v or type(protocol.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("Exploratory scope, provenance or conservative eligibility changed")
    return protocol


def audit_row(row, candidate):
    """Re-derive target from preserved physical proposals; never call it independent truth."""
    anchor, stop = row["anchor"], row["anchor"] + 33
    if (candidate["digest"] != contract_digest(candidate)
            or row["source_id"] != candidate["source_id"]
            or row["recording_id"] != candidate["recording_id"]
            or type(anchor) is not int or anchor < 0 or stop > candidate["frames"]):
        raise ValueError("Source identity or complete future window mismatch")
    if any(e["start"] <= anchor < e["end"] for e in candidate["events"]):
        raise ValueError("Anchor is inside an ongoing event")
    if any(e["start"] < stop and e["end"] > anchor for e in candidate["ambiguities"]):
        raise ValueError("Unresolved ambiguity overlaps the current/future window")
    future = [e for e in candidate["events"]
              if anchor < e["start"] < stop and not e["left_censored"]]
    future.sort(key=lambda e: (e["start"], {"grasp": 0, "contact": 1, "gripper_close": 2}[e["event_type"]],
                               e["object_id"] or ""))
    event = future[0] if future else None
    expected = ({"next_event": event["event_type"],
                 "interval": [event["start"]-anchor, min(33, event["end"]-anchor)],
                 "horizon": event["start"]-anchor} if event else
                {"next_event": "none", "interval": None, "horizon": None})
    expected_censored = bool(event and (event["right_censored"] or event["end"] > stop))
    if (row["target"] != expected or row["event_id"] != (event["event_id"] if event else None)
            or row["label_metadata"]["end_censored"] != expected_censored):
        raise ValueError("Target is not the exact earliest-onset weak label")
    if row["input_datasets"] != ["observation/front", "observation/wrist", "observation/state"]:
        raise ValueError("Noncausal input datasets")
    return event


def build_dataset(root, protocol):
    root = Path(root)
    verify_protocol(protocol)
    for mapping in (protocol["input_files_sha256"], protocol["code_sha256"]):
        for name, checksum in mapping.items():
            if Path(name).is_absolute() or ".." in Path(name).parts or sha(root / name) != checksum:
                raise ValueError(f"Exploratory input/code changed: {name}")
    paths = protocol["inputs"]
    study = sealed(root / paths["parent_study"])
    partitions = sealed(root / paths["partitions"])
    blind = sealed(root / paths["blind_manifest"])
    completion = sealed(root / paths["candidate_completion"])
    diagnostic = sealed(root / paths["original_diagnostic"])
    review_manifest = sealed(root / paths["assisted_manifest"])
    review_evidence = sealed(root / paths["review_evidence"])
    review_raw = (root / paths["review_snapshot"]).read_bytes()
    review_record = json.loads(review_raw)
    if (study["digest"] != protocol["parent_study_digest"]
            or partitions["digest"] != study["data"]["partition_contract_digest"]
            or completion["status"] != "complete"
            or completion["capture_completion_digest"] != blind["capture_completion_digest"]
            or completion["diagnostic_manifest_digest"] != diagnostic["digest"]
            or review_manifest["candidate_completion_digest"] != completion["digest"]
            or review_manifest["blind_manifest_digest"] != blind["digest"]
            or review_record["manifest_digest"] != review_manifest["digest"]
            or review_evidence["revision"] != review_record["revision"]
            or review_evidence["draft_file_sha256"] != hashlib.sha256(review_raw).hexdigest()
            or review_evidence["manifest_digest"] != review_manifest["digest"]):
        raise ValueError("Capture, candidates, study or review linkage mismatch")
    draft = review_record["draft"]
    encoded = (json.dumps(draft, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if (hashlib.sha256(encoded).hexdigest() != review_record["draft_sha256"]
            or draft["independent_reference"] is not False
            or draft["candidate_exposure_before_review"] is not True
            or draft["labels_admitted"] is not False):
        raise ValueError("Review must retain its assisted and unadmitted origin")
    cases = {r["recording_id"]: r for r in completion["cases"]}
    if (len(cases) != 200 or len(blind["episodes"]) != 200
            or set(cases) != {r["recording_id"] for r in blind["episodes"]}
            or len(review_manifest["episodes"]) != 200 or len(draft["episodes"]) != 200):
        raise ValueError("Exact initial 200-source population required")
    records, candidates, human = [], {}, {}
    review_counts = Counter()
    for row, assisted, decision_episode in zip(blind["episodes"], review_manifest["episodes"],
                                               draft["episodes"], strict=True):
        rid = row["recording_id"]
        if rid != assisted["recording_id"] or rid != decision_episode["recording_id"]:
            raise ValueError("Review source order changed")
        case = cases[rid]
        file = root / paths["candidate_root"] / rid / "event-candidates.json"
        if (sha(file) != case["candidate_file_sha256"]
                or completion["output_files_sha256"][f"{rid}/event-candidates.json"] != sha(file)):
            raise ValueError("Candidate bytes differ from extraction completion")
        candidate = sealed(file)
        if candidate["digest"] != case["candidate_digest"] or candidate["digest"] != assisted["candidate_digest"]:
            raise ValueError("Candidate digest differs from reviewed proposal")
        for key in ("recording_id", "source_id", "frames", "task_key"):
            if case[key] != row[key]:
                raise ValueError("Candidate metadata differs from capture")
        if any(candidate[k] != row[k] for k in ("recording_id", "source_id", "frames")):
            raise ValueError("Candidate payload differs from capture")
        original = [("event", e) for e in candidate["events"]] + [("ambiguity", e) for e in candidate["ambiguities"]]
        if len(original) != len(assisted["proposals"]) or len(original) != len(decision_episode["decisions"]):
            raise ValueError("Review omitted a proposal or uncertainty flag")
        for (kind, event), proposal, decision in zip(original, assisted["proposals"], decision_episode["decisions"], strict=True):
            if (proposal["kind"] != kind or proposal["original"] != event
                    or proposal["proposal_id"] != decision["proposal_id"]):
                raise ValueError("Assisted proposal identity changed")
            review_counts[(kind, event["event_type"], decision["decision"])] += 1
            if decision["decision"] not in ("pending", "correct"):
                raise ValueError("Human corrections/uncertainty require separate adjudication before this child")
            if kind == "event":
                human[event["event_id"]] = decision["decision"]
        candidates[rid] = candidate
        records.append({"candidate": candidate, "task_key": row["task_key"], "instruction": row["instruction"],
                        "recording_path": str(Path(protocol["capture_base"]) / "run" / rid / "recording.h5"),
                        "recording_sha256": row["evidence_sha256"]["recording.h5"]})
    rows, excluded, events, identities = _candidate_rows(records, partitions)
    selected = _sample(rows, protocol["maximum_prompts_per_class"])
    if selected != diagnostic["main_rows"]:
        raise ValueError("Recomputed conservative sample differs from preserved v3 diagnosis")
    enriched = []
    for row in selected:
        audit_row(row, candidates[row["recording_id"]])
        item = copy.deepcopy(row)
        item["weak_label_provenance"] = {
            "origin": "automatic_physical_predicate", "independent_qa_passed": False,
            "assisted_review_revision": review_record["revision"],
            "event_proposal_feedback": human[row["event_id"]] if row["event_id"] else "not_a_reviewed_negative",
            "reviewed_correct_does_not_establish_recall": True,
        }
        enriched.append(item)
    pilot = _sample(enriched, protocol["qualification_prompts_per_class"])
    if Counter(r["target"]["next_event"] for r in pilot) != {k: 4 for k in CLASSES}:
        raise ValueError("Balanced 16-prompt qualification set unavailable")
    coverage = {"source_pool": 200, "selected_sources": len({r["source_id"] for r in selected}),
                "selected_tasks": len({r["task_key"] for r in selected}), "unique_prompts": len(selected),
                "class_counts": dict(Counter(r["target"]["next_event"] for r in selected)),
                "lead_bins": {k: dict(Counter(r["lead_time_bin"] for r in selected if r["target"]["next_event"] == k)) for k in CLASSES},
                "excluded_anchors": dict(excluded), "raw_event_proposals": dict(events),
                "ambiguities_retained": sum(len(c["ambiguities"]) for c in candidates.values()),
                "selected_feedback_counts": dict(Counter(r["weak_label_provenance"]["event_proposal_feedback"] for r in enriched)),
                "all_source_proposal_decisions": {"|".join(k): v for k, v in sorted(review_counts.items())}}
    return seal_contract({"artifact_type": "eventground_wam.week10_weak_label_dataset",
                          "protocol_digest": protocol["digest"], "parent_study_digest": study["digest"],
                          "partition_digest": partitions["digest"], "capture_completion_digest": completion["capture_completion_digest"],
                          "candidate_completion_digest": completion["digest"], "review_evidence_digest": review_evidence["digest"],
                          "status": "qualified_provenance_for_bounded_weak_label_pilot",
                          "independent_qa_passed": False, "labels_admitted": False,
                          "weak_labels_authorized_for_qualification": True,
                          "formal_training_authorized": False, "future_evidence_is_model_input": False,
                          "coverage": coverage, "source_candidates": identities,
                          "main_rows": enriched, "qualification_prompt_ids": [r["prompt_id"] for r in pilot]})


def overfit_gates(initial, final, thresholds):
    if any(not math.isfinite(x) for metrics in (initial, final) for x in
           (metrics["loss"], metrics["event_accuracy"], metrics["joint_two_frame_accuracy"])):
        raise ValueError("Nonfinite qualification metric")
    return {"loss_reduced": final["loss"] <= initial["loss"] * thresholds["maximum_loss_ratio"],
            "event_accuracy": final["event_accuracy"] >= thresholds["minimum_event_accuracy"],
            "joint_two_frame_accuracy": final["joint_two_frame_accuracy"] >= thresholds["minimum_joint_two_frame_accuracy"]}


def verify_pilot(contract, protocol, dataset, recipe):
    if contract.get("digest") != contract_digest(contract):
        raise ValueError("Unsealed qualification contract")
    expected = {
        "operation": "exploratory_weak_label_overfit_qualification", "attempts": 1,
        "protocol_digest": protocol["digest"], "dataset_digest": dataset["digest"],
        "recipe_digest": recipe["digest"], "logical_updates": 64,
        "maximum_executed_updates_including_replay": 65, "recovery_checkpoint_step": 8,
        "logical_batch_size": 16, "microbatch_size": 1, "seed": 17,
        "formal_training_authorized": False, "independent_qa_passed": False,
        "heldout_payload_reads": 0, "wall_seconds": 1800,
    }
    if any(contract.get(k) != v or type(contract.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("Qualification scope, identities or fixed endpoint changed")
    if (contract["prompt_ids"] != dataset["qualification_prompt_ids"]
            or len(set(contract["prompt_ids"])) != 16
            or recipe["logical_batch_size"] != 16 or recipe["microbatch_size"] != 1
            or recipe["compute_dtype"] != "float32"
            or contract["prior_qualification_active_seconds"] + contract["wall_seconds"] > 3 * 3600):
        raise ValueError("Qualification inputs, precision or cumulative resource ceiling changed")
    if contract["thresholds"] != {"maximum_loss_ratio": .5, "minimum_event_accuracy": .875,
                                  "minimum_joint_two_frame_accuracy": .75}:
        raise ValueError("Predeclared overfit thresholds changed")
    if contract["resources"] != {"output_bytes": 8 * 1024**3, "minimum_free_bytes": 50 * 1024**3,
                                 "maximum_gpu_fraction": .9, "cpu_threads": 4}:
        raise ValueError("Qualification resource caps changed")
    return contract
