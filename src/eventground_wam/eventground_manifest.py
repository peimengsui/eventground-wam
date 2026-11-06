"""Deterministic Week 10 prompt coverage and fail-closed dataset admission.

Candidate generation remains the frozen instrumented labeler. This module only
selects unique current-frame references and separate targets; it does not copy
future evidence into model inputs or manufacture examples to cover shortfalls.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.instrumented_labeling import prompt_previews
from eventground_wam.instrumented_partitions import require_train_source
from eventground_wam.instrumented_supervision import labeler_contract

CLASSES = ("contact", "grasp", "gripper_close", "none")
LEAD_BINS = ((1, 4), (5, 8), (9, 16), (17, 32))


def require_sealed(value, name="record"):
    if value.get("digest") != contract_digest(value):
        raise ValueError(f"Changed or unsealed {name}")
    return value


def lead_bin(horizon):
    if horizon is None:
        return "none"
    for low, high in LEAD_BINS:
        if type(horizon) is int and low <= horizon <= high:
            return f"{low}-{high}"
    raise ValueError("Invalid event horizon")


def _candidate_rows(records, partitions):
    seen_sources, seen_recordings = set(), set()
    eligible, excluded, events, identities = [], Counter(), Counter(), {}
    for record in sorted(records, key=lambda r: r["candidate"]["source_id"]):
        candidate = require_sealed(record["candidate"], "event candidates")
        source = candidate["source_id"]
        rid = candidate["recording_id"]
        if source in seen_sources or rid in seen_recordings:
            raise ValueError("Duplicate source or recording cannot fill coverage")
        seen_sources.add(source)
        seen_recordings.add(rid)
        partition = require_train_source(partitions, source)
        if (
            partition["task_key"] != record["task_key"]
            or partition["frames"] != candidate["frames"]
        ):
            raise ValueError("Candidate source metadata does not match training partition")
        if candidate["labeler_digest"] != labeler_contract()["digest"]:
            raise ValueError("Candidate rules differ from frozen physical labeler")
        if candidate.get("labels_admitted") or candidate.get("future_evidence_is_model_input"):
            raise ValueError("Expected unadmitted offline candidates")
        ordered = sorted(
            candidate["events"],
            key=lambda r: (
                r["start"],
                {"grasp": 0, "contact": 1, "gripper_close": 2}[r["event_type"]],
                r["object_id"] or "",
            ),
        )
        if ordered != candidate["events"]:
            raise ValueError("Candidate event order violates frozen earliest/tie order")
        ids = [row["event_id"] for row in ordered]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate physical event identity")
        identities[source] = {
            "recording_id": rid,
            "candidate_digest": candidate["digest"],
            "task_key": record["task_key"],
            "frames": candidate["frames"],
            "recording_sha256": record["recording_sha256"],
        }
        events.update(row["event_type"] for row in ordered)
        for row in prompt_previews(candidate)["rows"]:
            if row["status"] != "unreviewed_preview":
                excluded[row["reason"]] += 1
                continue
            anchor = row["anchor"]
            unique = f"{source}:{anchor}"
            eligible.append(
                {
                    "prompt_id": hashlib.sha256(unique.encode()).hexdigest(),
                    "source_id": source,
                    "recording_id": rid,
                    "task_key": record["task_key"],
                    "anchor": anchor,
                    "event_id": row["event_id"],
                    "lead_time_bin": lead_bin(row["horizon"]),
                    "recording_path": record["recording_path"],
                    "recording_sha256": record["recording_sha256"],
                    "instruction": record["instruction"],
                    "input_datasets": [
                        "observation/front",
                        "observation/wrist",
                        "observation/state",
                    ],
                    "target": {
                        "next_event": row["event_type"],
                        "interval": row["interval"],
                        "horizon": row["horizon"],
                    },
                    "label_metadata": {
                        "end_censored": row["end_censored"],
                        "candidate_digest": candidate["digest"],
                    },
                }
            )
    return eligible, excluded, events, identities


def _sample(rows, count):
    """Round-robin task then source, interleaving classes and lead-time bins.

    Scarce classes are visited first each round. Within each task a source receives
    at most one prompt per class per round. Bins already represented least often in
    that class are preferred, then unused event bins, stable event ID and anchor.
    No class or duplicate substitution is performed if this fixed schedule falls short.
    """
    by_class = {kind: defaultdict(lambda: defaultdict(list)) for kind in CLASSES}
    for row in rows:
        by_class[row["target"]["next_event"]][row["task_key"]][row["source_id"]].append(row)
    class_order = sorted(
        CLASSES,
        key=lambda k: (
            sum(len(values) for sources in by_class[k].values() for values in sources.values()),
            CLASSES.index(k),
        ),
    )
    selected, source_counts, event_counts = [], Counter(), Counter()
    class_counts, bin_counts, event_bins = Counter(), Counter(), Counter()
    while True:
        progress = False
        for kind in class_order:
            if class_counts[kind] >= count:
                continue
            for task in sorted(by_class[kind]):
                for source in sorted(by_class[kind][task]):
                    if class_counts[kind] >= count or source_counts[source] >= 16:
                        continue
                    candidates = [
                        r
                        for r in by_class[kind][task][source]
                        if r["event_id"] is None or event_counts[r["event_id"]] < 4
                    ]
                    if not candidates:
                        continue
                    row = min(
                        candidates,
                        key=lambda r: (
                            bin_counts[(kind, r["lead_time_bin"])],
                            event_bins[(r["event_id"], r["lead_time_bin"])],
                            r["event_id"] or "",
                            r["anchor"],
                            r["prompt_id"],
                        ),
                    )
                    by_class[kind][task][source].remove(row)
                    selected.append(row)
                    source_counts[source] += 1
                    class_counts[kind] += 1
                    bin_counts[(kind, row["lead_time_bin"])] += 1
                    if row["event_id"] is not None:
                        event_counts[row["event_id"]] += 1
                        event_bins[(row["event_id"], row["lead_time_bin"])] += 1
                    progress = True
        if not progress:
            return selected


def build_diagnostic_manifest(records, study, partitions):
    """Return coverage and deterministic prospective samples; never admit labels."""
    require_sealed(study, "study")
    require_sealed(partitions, "partitions")
    data = study["data"]
    expected = {
        "per_class_prompts": 500,
        "rl_pilot_per_class_prompts": 125,
        "maximum_anchors_per_physical_event": 4,
        "maximum_prompts_per_group": 16,
        "sft_unique_prompts": 2000,
        "rl_pilot_unique_prompts": 500,
    }
    if any(data[k] != v for k, v in expected.items()):
        raise ValueError("Only the frozen Week 10 prompt counts and caps are supported")
    if data["partition_contract_digest"] != partitions["digest"]:
        raise ValueError("Wrong source partition")
    if data["lead_time_bins"] != [list(x) for x in LEAD_BINS]:
        raise ValueError("Lead-time bins changed")
    rows, excluded, events, identities = _candidate_rows(records, partitions)
    main = _sample(rows, 500)
    # The nested set follows the same algorithm using only already-selected main rows.
    pilot = _sample(main, 125)
    raw = Counter(r["target"]["next_event"] for r in rows)
    selected = Counter(r["target"]["next_event"] for r in main)
    per_event = Counter(
        (r["target"]["next_event"], r["source_id"], r["event_id"])
        for r in rows
        if r["event_id"] is not None
    )
    per_source = Counter()
    for (kind, source, _), value in per_event.items():
        per_source[kind, source] += min(4, value)
    for row in rows:
        if row["target"]["next_event"] == "none":
            per_source["none", row["source_id"]] += 1
    upper = {
        kind: sum(min(16, value) for (k, _), value in per_source.items() if k == kind)
        for kind in CLASSES
    }
    coverage = {
        "recordings": len(records),
        "tasks": len({r["task_key"] for r in records}),
        "frames": sum(r["candidate"]["frames"] for r in records),
        "candidate_events": {k: events[k] for k in CLASSES if k != "none"},
        "excluded_anchor_reasons": dict(sorted(excluded.items())),
        "eligible_anchors": {k: raw[k] for k in CLASSES},
        "single_class_cap_upper_bound": upper,
        "selected_main_per_class": {k: selected[k] for k in CLASSES},
        "main_shortfall_per_class": {k: max(0, 500 - selected[k]) for k in CLASSES},
        "selected_pilot_per_class": {
            k: sum(r["target"]["next_event"] == k for r in pilot) for k in CLASSES
        },
        "main_lead_bins": {
            k: dict(
                sorted(
                    Counter(
                        r["lead_time_bin"] for r in main if r["target"]["next_event"] == k
                    ).items()
                )
            )
            for k in CLASSES
        },
        "schedule_is_global_feasibility_proof": False,
        "quality_estimated_from_candidate_counts": False,
    }
    return seal_contract(
        {
            "artifact_type": "eventground_wam.eventground_diagnostic_manifest",
            "schema_version": 1,
            "study_digest": study["digest"],
            "partition_digest": partitions["digest"],
            "source_candidates": identities,
            "coverage": coverage,
            "sampling": "deterministic_task_group_round_robin_v1",
            "main_rows": main,
            "pilot_prompt_ids": [r["prompt_id"] for r in pilot],
            "labels_admitted": False,
            "training_authorized": False,
            "future_evidence_is_model_input": False,
            "status": "diagnostic_only_independent_qa_required",
            "shortfall_policy": "block_no_duplicate_or_class_substitution",
        }
    )


def admit_manifest(diagnostic, qa):
    """Admit only complete balanced candidates linked to an independent QA receipt.

    The QA receipt must be generated from preserved independent annotations, not
    candidate agreement or a second execution of the same physical predicates.
    """
    require_sealed(diagnostic, "diagnostic manifest")
    require_sealed(qa, "independent QA receipt")
    if qa.get("artifact_type") != "eventground_wam.week10_label_qa_assessment":
        raise ValueError("Independent event-label QA receipt required")
    identities = diagnostic["source_candidates"]
    candidate_digests = {r["recording_id"]: r["candidate_digest"] for r in identities.values()}
    if (
        qa.get("capture_completion_digest") != diagnostic.get("capture_completion_digest")
        or not diagnostic.get("capture_completion_digest")
        or qa.get("candidate_digests") != candidate_digests
        or sorted(qa.get("source_ids", [])) != sorted(identities)
        or sorted(qa.get("reviewed_recording_ids", [])) != sorted(candidate_digests)
    ):
        raise ValueError("QA must bind every source, recording and candidate digest")
    gates = {
        "episode_count_200_to_300",
        "all_40_tasks",
        "independent_full_episode_review",
        "negative_and_missed_event_review",
    }
    for kind in ("contact", "grasp"):
        gates.update(
            f"{kind}_{suffix}"
            for suffix in ("100_reviewed_positives", "precision", "recall", "boundary_mae")
        )
    gates.update(f"{kind}_negative_frames_reviewed" for kind in CLASSES if kind != "none")
    if (
        qa.get("status") != "pass"
        or set(qa.get("gates", {})) != gates
        or any(qa["gates"][key] is not True for key in gates)
        or not qa.get("review_manifest_digest")
        or not qa.get("review_digest")
    ):
        raise ValueError("Independent full-episode QA gate has not passed")
    if len(identities) < 200 or len({r["task_key"] for r in identities.values()}) != 40:
        raise ValueError("Independent QA coverage shortfall")
    for kind in ("contact", "grasp"):
        metrics = qa.get("metrics_by_type", {}).get(kind, {})
        if (
            metrics.get("reviewed_positives", 0) < 100
            or metrics.get("precision", 0) < 0.95
            or metrics.get("recall", 0) < 0.95
            or metrics.get("start_mae_frames", 99) > 2
            or metrics.get("end_mae_frames", 99) > 2
        ):
            raise ValueError(f"Independent {kind} quality gate failed")
    if any(diagnostic["coverage"]["main_shortfall_per_class"].values()):
        raise ValueError("Unique class-balanced prompt coverage shortfall")
    rows = diagnostic["main_rows"]
    pilot = diagnostic["pilot_prompt_ids"]
    ids = {r["prompt_id"] for r in rows}
    if len(rows) != 2000 or len(ids) != 2000 or len(pilot) != 500 or len(set(pilot)) != 500:
        raise ValueError("Fixed unique main/pilot size required")
    if not set(pilot) <= ids:
        raise ValueError("Pilot must be nested in main")
    if Counter(r["target"]["next_event"] for r in rows) != dict.fromkeys(CLASSES, 500):
        raise ValueError("Exact main class counts required")
    if Counter(
        r["target"]["next_event"] for r in rows if r["prompt_id"] in set(pilot)
    ) != dict.fromkeys(CLASSES, 125):
        raise ValueError("Exact nested pilot class counts required")
    if (
        max(Counter(r["source_id"] for r in rows).values()) > 16
        or max(Counter(r["event_id"] for r in rows if r["event_id"]).values(), default=0) > 4
    ):
        raise ValueError("Source/event cap exceeded")
    for row in rows:
        source = identities.get(row["source_id"])
        if (
            source is None
            or row["recording_id"] != source["recording_id"]
            or row["recording_sha256"] != source["recording_sha256"]
            or row["label_metadata"]["candidate_digest"] != source["candidate_digest"]
            or type(row["anchor"]) is not int
            or not 0 <= row["anchor"] < source["frames"] - 32
            or row["prompt_id"]
            != hashlib.sha256(f"{row['source_id']}:{row['anchor']}".encode()).hexdigest()
            or row["input_datasets"]
            != ["observation/front", "observation/wrist", "observation/state"]
        ):
            raise ValueError("Prompt identity or causal input boundary changed")
    return seal_contract(
        {
            "artifact_type": "eventground_wam.admitted_eventground_manifest",
            "schema_version": 1,
            "diagnostic_digest": diagnostic["digest"],
            "independent_qa_digest": qa["digest"],
            "study_digest": diagnostic["study_digest"],
            "partition_digest": diagnostic["partition_digest"],
            "main_rows": diagnostic["main_rows"],
            "pilot_prompt_ids": diagnostic["pilot_prompt_ids"],
            "labels_admitted": True,
            "training_authorized": False,
            "status": "data_admitted_training_launch_gate_still_required",
        }
    )
