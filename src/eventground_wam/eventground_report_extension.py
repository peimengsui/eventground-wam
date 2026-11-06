"""Separate, prospective coverage probe; no changes to the frozen report or labels."""
from collections import Counter
import hashlib

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_manifest import CLASSES, _sample, lead_bin, require_sealed
from eventground_wam.eventground_weak_supervision import audit_row
from eventground_wam.instrumented_labeling import prompt_previews
from eventground_wam.instrumented_supervision import labeler_contract, physical_source_id

PANEL = "event-report-extension"
GATE = {"prompts_per_class": 20, "grasp_events": 10, "grasp_sources": 10, "grasp_tasks": 5}


def panel_sources(partitions, metadata):
    require_sealed(partitions)
    require_sealed(metadata)
    if metadata["panel"] != PANEL or metadata["partition_digest"] != partitions["digest"]:
        raise ValueError("Wrong parent partition or child panel")
    excluded = set(metadata["excluded_source_ids"])
    parent_ids = {r["source_id"] for r in partitions["rows"]}
    if not parent_ids <= excluded:
        raise ValueError("Parent sources not fully excluded")
    rows = metadata["rows"]
    ids = [r["source_id"] for r in rows]
    tasks = Counter(r["task_key"] for r in rows)
    if len(rows) != 80 or len(set(ids)) != 80 or set(ids) & excluded:
        raise ValueError("Incomplete, duplicate or previously accessed source")
    if set(tasks) != {r["task_key"] for r in partitions["rows"]} or set(tasks.values()) != {2}:
        raise ValueError("Expected two new sources for each of 40 tasks")
    for row in rows:
        if (row["partition"] != PANEL or row["frames"] <= 32 or
                row["source_id"] != physical_source_id(row["repository"], row["revision"],
                                                      row["hdf5_path"], row["demo_id"])):
            raise ValueError("Invalid canonical source binding")
    return rows


def validate_panel_contract(contract, partitions, metadata):
    require_sealed(contract)
    rows = panel_sources(partitions, metadata)
    if (contract["panel"] != PANEL or contract["source_metadata_digest"] != metadata["digest"]
            or contract["partition_digest"] != partitions["digest"]
            or contract["source_ids"] != [r["source_id"] for r in rows]
            or contract["coverage_gate"] != GATE
            or contract["independent_qa_passed"] is not False
            or contract["automatic_retries"] != 0
            or contract["training_authorized"] is not False
            or contract["report_model_inference_authorized"] is not False
            or not contract["training_decision_digest"]):
        raise ValueError("Changed coverage-probe scope or identity")
    return rows


def coverage_gate(rows):
    counts = Counter(r["target"]["next_event"] for r in rows)
    grasp = [r for r in rows if r["target"]["next_event"] == "grasp"]
    actual = dict(prompts_per_class=min(counts.get(k, 0) for k in CLASSES),
                  grasp_events=len({r["event_id"] for r in grasp}),
                  grasp_sources=len({r["source_id"] for r in grasp}),
                  grasp_tasks=len({r["task_key"] for r in grasp}))
    checks = {k: actual[k] >= v for k, v in GATE.items()}
    return dict(passed=all(checks.values()), checks=checks, actual=actual, minimum=GATE,
                interpretation="Minimum exploratory coverage, not a power calculation or independent label QA",
                on_failure="Stop after the fixed 80-source probe; no replenishment or threshold reduction")


def heldout_rows(records, partitions, metadata):
    sources = panel_sources(partitions, metadata)
    if [r["candidate"]["source_id"] for r in records] != [r["source_id"] for r in sources]:
        raise ValueError("Missing or reordered sources")
    eligible, exclusions = [], Counter()
    raw = Counter()
    for record, source in zip(records, sources, strict=True):
        c = require_sealed(record["candidate"])
        if (c["labeler_digest"] != labeler_contract()["digest"] or c["frames"] != source["frames"]
                or record["task_key"] != source["task_key"] or c["labels_admitted"]):
            raise ValueError("Changed labeler or source")
        raw.update(e["event_type"] for e in c["events"])
        for p in prompt_previews(c)["rows"]:
            if p["status"] != "unreviewed_preview":
                exclusions[p["reason"]] += 1
                continue
            row = dict(prompt_id=hashlib.sha256(f"{source['source_id']}:{p['anchor']}".encode()).hexdigest(),
                       source_id=source["source_id"], recording_id=c["recording_id"],
                       task_key=source["task_key"], anchor=p["anchor"], event_id=p["event_id"],
                       lead_time_bin=lead_bin(p["horizon"]), recording_path=record["recording_path"],
                       recording_sha256=record["recording_sha256"], instruction=record["instruction"],
                       input_datasets=["observation/front", "observation/wrist", "observation/state"],
                       target=dict(next_event=p["event_type"], interval=p["interval"], horizon=p["horizon"]),
                       label_metadata=dict(end_censored=p["end_censored"], candidate_digest=c["digest"]),
                       weak_label_provenance=dict(origin="automatic_physical_predicate", independent_qa_passed=False,
                                                  human_reviewed=False, negative_windows_human_reviewed=False))
            audit_row(row, c)
            eligible.append(row)
    # Same capped task/source/lead-bin sampler; this new panel uses dense eligible anchors.
    preliminary = _sample(eligible, 250)
    count = min(Counter(r["target"]["next_event"] for r in preliminary).get(k, 0) for k in CLASSES)
    selected, counts = [], Counter()
    for row in preliminary:
        kind = row["target"]["next_event"]
        if counts[kind] < count:
            selected.append(row)
            counts[kind] += 1
    return seal_contract(dict(artifact_type="eventground_wam.weak_report_extension_manifest", panel=PANEL,
        partition_digest=partitions["digest"], source_metadata_digest=metadata["digest"],
        source_count=len(records), source_candidate_digests={r["candidate"]["source_id"]: r["candidate"]["digest"] for r in records},
        exclusions=dict(exclusions), raw_event_counts=dict(raw), eligible_count=len(eligible),
        eligible_by_class=dict(Counter(r["target"]["next_event"] for r in eligible)),
        selected_count=len(selected), selected_by_class=dict(counts), rows=selected,
        capped_or_balance_omissions=len(eligible)-len(selected),
        grasp_eligible_sources=len({r["source_id"] for r in eligible if r["target"]["next_event"] == "grasp"}),
        grasp_eligible_events=len({r["event_id"] for r in eligible if r["target"]["next_event"] == "grasp"}),
        grasp_lead_bins=dict(Counter(r["lead_time_bin"] for r in selected if r["target"]["next_event"] == "grasp")),
        coverage_gate=coverage_gate(selected), training_authorized=False, report_model_inference_authorized=False,
        independent_qa_passed=False, original_primary_restored=False,
        label_quality="Unreviewed automatic labels; all original ambiguity and ongoing-event exclusions retained",
        reporting="Separate class-balanced coverage supplement; never pool with or replace the original stride-eight report"))
