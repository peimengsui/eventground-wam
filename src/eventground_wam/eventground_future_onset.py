"""Versioned next-new-onset task; original no-ongoing experiments stay immutable."""
from __future__ import annotations

import hashlib
from collections import Counter

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_manifest import CLASSES, _sample, lead_bin, require_sealed
from eventground_wam.eventground_weak_evaluation import valid_prediction
from eventground_wam.instrumented_supervision import labeler_contract

TASK = "week10-future-onset-v1"
INPUTS = ["observation/front", "observation/wrist", "observation/state"]
ORDER = {"grasp": 0, "contact": 1, "gripper_close": 2}


def target_at(candidate, anchor):
    """Earliest NEW onset in (anchor, anchor+32], including ongoing-event anchors.

    Uncertainty still excludes the entire current/future interval. End offsets
    are half-open and clipped to 33; no new onset means none, even while active.
    """
    if type(anchor) is not int or anchor < 0 or anchor + 33 > candidate["frames"]:
        raise ValueError("incomplete_future_window")
    stop = anchor + 33
    if any(e["start"] < stop and e["end"] > anchor for e in candidate["ambiguities"]):
        raise ValueError("unresolved_ambiguity")
    future = sorted((e for e in candidate["events"]
                     if anchor < e["start"] < stop and not e["left_censored"]),
                    key=lambda e: (e["start"], ORDER[e["event_type"]], e["object_id"] or ""))
    event = future[0] if future else None
    target = ({"next_event": event["event_type"],
               "interval": [event["start"] - anchor, min(33, event["end"] - anchor)],
               "horizon": event["start"] - anchor} if event else
              {"next_event": "none", "interval": None, "horizon": None})
    return target, event, bool(event and (event["right_censored"] or event["end"] > stop))


def audit_row(row, candidate):
    require_sealed(candidate)
    if (candidate["labeler_digest"] != labeler_contract()["digest"]
            or row.get("task_definition") != TASK
            or row["source_id"] != candidate["source_id"]
            or row["recording_id"] != candidate["recording_id"]
            or row["label_metadata"]["candidate_digest"] != candidate["digest"]
            or row["prompt_id"] != hashlib.sha256(f"{row['source_id']}:{row['anchor']}".encode()).hexdigest()
            or row["input_datasets"] != INPUTS):
        raise ValueError("Task/source/candidate/current-input binding differs")
    target, event, censored = target_at(candidate, row["anchor"])
    if (row["target"] != target or row["event_id"] != (event["event_id"] if event else None)
            or row["label_metadata"]["end_censored"] != censored
            or row["lead_time_bin"] != lead_bin(target["horizon"])):
        raise ValueError("Target is not the exact next-new-onset label")
    return event


def summarize(rows):
    return {"n": len(rows), "sources": len({r["source_id"] for r in rows}),
            "tasks": len({r["task_key"] for r in rows}),
            "class_counts": {k: sum(r["target"]["next_event"] == k for r in rows) for k in CLASSES},
            "lead_bins": {k: dict(Counter(r["lead_time_bin"] for r in rows
                                           if r["target"]["next_event"] == k)) for k in CLASSES}}


def build_manifest(index, panel):
    require_sealed(index)
    records = index["panels"][panel]
    if len({r["candidate"]["source_id"] for r in records}) != len(records):
        raise ValueError("Duplicate physical source")
    eligible, excluded = [], Counter()
    for record in records:
        c = require_sealed(record["candidate"])
        if c["labeler_digest"] != labeler_contract()["digest"] or c["labels_admitted"]:
            raise ValueError("Physical rules or candidate provenance changed")
        for anchor in range(c["frames"]):
            try:
                target, event, censored = target_at(c, anchor)
            except ValueError as e:
                excluded[str(e)] += 1
                continue
            row = dict(prompt_id=hashlib.sha256(f"{c['source_id']}:{anchor}".encode()).hexdigest(),
                       task_definition=TASK, source_id=c["source_id"], recording_id=c["recording_id"],
                       task_key=record["task_key"], anchor=anchor, event_id=event["event_id"] if event else None,
                       lead_time_bin=lead_bin(target["horizon"]), target=target,
                       recording_path=record["recording_path"], recording_sha256=record["recording_sha256"],
                       instruction=record["instruction"], input_datasets=INPUTS,
                       label_metadata=dict(candidate_digest=c["digest"], end_censored=censored),
                       weak_label_provenance=record["weak_label_provenance"])
            audit_row(row, c)
            eligible.append(row)
    selected = _sample(eligible, 500 if panel == "train" else 250)
    if panel != "train":
        count = min(Counter(r["target"]["next_event"] for r in selected).get(k, 0) for k in CLASSES)
        kept, counts = [], Counter()
        for row in selected:
            kind = row["target"]["next_event"]
            if counts[kind] < count:
                kept.append(row); counts[kind] += 1
        selected = kept
    if (len({r["prompt_id"] for r in selected}) != len(selected)
            or max(Counter(r["source_id"] for r in selected).values(), default=0) > 16
            or max(Counter(r["event_id"] for r in selected if r["event_id"]).values(), default=0) > 4):
        raise ValueError("Duplicate/cap violation")
    return seal_contract(dict(artifact_type="eventground_wam.future_onset_manifest", task_definition=TASK,
        source_index_digest=index["digest"], panel=panel, source_count=len(records),
        summary=summarize(selected), eligible_summary=summarize(eligible), exclusions=dict(excluded),
        rows=selected, independent_qa_passed=False,
        interpretation="Exploratory automatic-label agreement; previously inspected labels, not untouched confirmation"))


def schedule(manifest, stage):
    require_sealed(manifest)
    rows = manifest["rows"]
    if stage.endswith("qualification"):
        selected = _sample(rows, 4)
        if len(selected) != 16: raise ValueError("Incomplete qualification")
        batches = [selected, selected]
    elif stage in ("primary", "calibration"):
        selected = _sample(rows, 125)
        if Counter(r["target"]["next_event"] for r in selected) != {k:125 for k in CLASSES}:
            raise ValueError("Incomplete nested pilot")
        ordered = sorted(selected, key=lambda r: (hashlib.sha256(f"17:rlvr:{r['prompt_id']}".encode()).hexdigest(), r["prompt_id"]))
        batches = [ordered[i:i+16] for i in range(0, len(ordered), 16)]
    elif stage in ("sft", "main", "control"):
        batches = []
        for epoch in range(2 if stage == "sft" else 1):
            # Main and control have exactly the same prompt exposure and ordering.
            ordered = sorted(rows, key=lambda r: (hashlib.sha256(f"17:{epoch}:{r['prompt_id']}".encode()).hexdigest(), r["prompt_id"]))
            batches.extend(ordered[i:i+16] for i in range(0, len(ordered), 16))
    else:
        raise ValueError("Unknown fixed stage")
    return seal_contract(dict(artifact_type="eventground_wam.future_onset_schedule", task_definition=TASK,
        manifest_digest=manifest["digest"], stage=stage, seed=17, logical_batch_size=16,
        microbatch_size=1, batches=[[r["prompt_id"] for r in b] for b in batches],
        batch_sizes=[len(b) for b in batches], updates=len(batches), presentations=sum(map(len,batches))))


def weak_reward(prediction, row, candidate, weights=(.4,.25,.2,.15)):
    event = audit_row(row, candidate)
    if tuple(weights) not in ((.4,.25,.2,.15), (.3,.35,.2,.15)):
        raise ValueError("Unfrozen weights")
    if not valid_prediction(prediction): return 0.
    target = row["target"]
    iou = timing = predicate = 0.
    if prediction["next_event"] == target["next_event"]:
        if event is None: iou = timing = predicate = 1.
        else:
            a,b = prediction["interval"]; c,d = target["interval"]
            intersection = max(0, min(b,d)-max(a,c))
            iou = intersection/(b-a+d-c-intersection)
            timing = max(0., 1-abs(a-c)/32)
            predicate = float(event["start"]-row["anchor"] <= a < event["end"]-row["anchor"])
    return sum(w*v for w,v in zip(weights,(iou,timing,predicate,1.),strict=True))
