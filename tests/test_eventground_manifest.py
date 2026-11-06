import copy
from collections import Counter

import pytest

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_manifest import (
    CLASSES,
    admit_manifest,
    build_diagnostic_manifest,
    lead_bin,
)
from eventground_wam.instrumented_supervision import labeler_contract


def reseal(value):
    return seal_contract({k: v for k, v in value.items() if k != "digest"})


def setup(count=200):
    records, split = [], []
    for index in range(count):
        source = f"{index:064x}"
        rid = f"recording-{index:04d}"
        task = f"task-{index % 40:02d}"
        events = []
        for ei, (kind, start) in enumerate(
            (("contact", 40), ("grasp", 100), ("gripper_close", 160))
        ):
            events.append(
                {
                    "event_type": kind,
                    "object_id": None if kind == "gripper_close" else "bowl",
                    "start": start,
                    "end": start + 10,
                    "left_censored": False,
                    "right_censored": False,
                    "event_id": f"{rid}:event-{ei:04d}",
                }
            )
        candidate = seal_contract(
            {
                "source_id": source,
                "recording_id": rid,
                "frames": 220,
                "labeler_digest": labeler_contract()["digest"],
                "events": events,
                "ambiguities": [],
                "labels_admitted": False,
                "future_evidence_is_model_input": False,
            }
        )
        records.append(
            {
                "candidate": candidate,
                "task_key": task,
                "instruction": "pick up bowl",
                "recording_path": f"/capture/{rid}/recording.h5",
                "recording_sha256": "f" * 64,
            }
        )
        split.append(
            {
                "source_id": source,
                "task_key": task,
                "frames": 220,
                "partition": "train",
                "queue_role": "initial_review",
            }
        )
    partitions = seal_contract({"rows": split})
    study = seal_contract(
        {
            "data": {
                "per_class_prompts": 500,
                "rl_pilot_per_class_prompts": 125,
                "maximum_anchors_per_physical_event": 4,
                "maximum_prompts_per_group": 16,
                "sft_unique_prompts": 2000,
                "rl_pilot_unique_prompts": 500,
                "partition_contract_digest": partitions["digest"],
                "lead_time_bins": [[1, 4], [5, 8], [9, 16], [17, 32]],
            }
        }
    )
    return records, study, partitions


def test_complete_balance_nested_and_all_caps():
    records, study, partition = setup()
    result = build_diagnostic_manifest(records, study, partition)
    assert result["coverage"]["selected_main_per_class"] == dict.fromkeys(CLASSES, 500)
    assert result["coverage"]["selected_pilot_per_class"] == dict.fromkeys(CLASSES, 125)
    rows = result["main_rows"]
    assert len(rows) == len({r["prompt_id"] for r in rows}) == 2000
    assert max(Counter(r["source_id"] for r in rows).values()) <= 16
    assert max(Counter(r["event_id"] for r in rows if r["event_id"]).values()) <= 4
    assert set(result["pilot_prompt_ids"]) <= {r["prompt_id"] for r in rows}
    assert not result["labels_admitted"] and not result["training_authorized"]
    assert build_diagnostic_manifest(list(reversed(records)), study, partition) == result


def test_shortfall_cannot_be_filled_by_duplicates_or_classes():
    records, study, partition = setup(3)
    result = build_diagnostic_manifest(records, study, partition)
    assert result["coverage"]["single_class_cap_upper_bound"]["grasp"] == 12
    assert result["coverage"]["main_shortfall_per_class"]["grasp"] >= 488
    with pytest.raises(ValueError, match="Duplicate source"):
        build_diagnostic_manifest(records + records[:1], study, partition)


def test_lead_bins_and_causal_model_inputs():
    for value, expected in [
        (1, "1-4"),
        (4, "1-4"),
        (5, "5-8"),
        (16, "9-16"),
        (32, "17-32"),
        (None, "none"),
    ]:
        assert lead_bin(value) == expected
    for value in (0, 33, True, 1.5):
        with pytest.raises(ValueError):
            lead_bin(value)
    records, study, partition = setup(1)
    manifest = build_diagnostic_manifest(records, study, partition)
    for row in manifest["main_rows"]:
        assert row["input_datasets"] == [
            "observation/front",
            "observation/wrist",
            "observation/state",
        ]
        assert "action" not in row and "contacts" not in row and "object_position" not in row
        assert set(row["target"]) == {"next_event", "interval", "horizon"}
        assert row["anchor"] >= 0


def test_exact_earliest_tie_ongoing_ambiguity_and_tail_rules():
    records, study, partition = setup(1)
    candidate = records[0]["candidate"]
    candidate["events"][0]["start"] = 100
    candidate["events"][0]["end"] = 110
    candidate["events"] = [candidate["events"][1], candidate["events"][0], candidate["events"][2]]
    candidate["ambiguities"] = [{"start": 30, "end": 35}]
    records[0]["candidate"] = reseal(candidate)
    manifest = build_diagnostic_manifest(records, study, partition)
    coverage = manifest["coverage"]
    assert coverage["eligible_anchors"]["contact"] == 0  # grasp wins simultaneous onset
    assert coverage["excluded_anchor_reasons"]["inside_ongoing_event"] == 20
    assert coverage["excluded_anchor_reasons"]["incomplete_future_window"] == 32
    assert coverage["excluded_anchor_reasons"]["unresolved_candidate_in_window"] > 0
    assert all(row["anchor"] + 32 < 220 for row in manifest["main_rows"])


@pytest.mark.parametrize("damage", ["heldout", "reserve", "task", "rules", "seal", "ties"])
def test_invalid_sources_and_rules_rejected(damage):
    records, study, partition = setup(1)
    if damage == "heldout":
        partition["rows"][0]["partition"] = "event-dev"
    elif damage == "reserve":
        partition["rows"][0]["queue_role"] = "coverage_reserve"
    elif damage == "task":
        records[0]["task_key"] = "other"
    elif damage == "rules":
        records[0]["candidate"]["labeler_digest"] = "0" * 64
        records[0]["candidate"] = reseal(records[0]["candidate"])
    elif damage == "seal":
        records[0]["candidate"]["frames"] = 221
    elif damage == "ties":
        records[0]["candidate"]["events"].reverse()
        records[0]["candidate"] = reseal(records[0]["candidate"])
    partition = reseal(partition)
    study["data"]["partition_contract_digest"] = partition["digest"]
    study = reseal(study)
    with pytest.raises(ValueError):
        build_diagnostic_manifest(records, study, partition)


def qa_fixture(manifest):
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
    return seal_contract(
        {
            "artifact_type": "eventground_wam.week10_label_qa_assessment",
            "status": "pass",
            "capture_completion_digest": manifest["capture_completion_digest"],
            "candidate_digests": {
                v["recording_id"]: v["candidate_digest"]
                for v in manifest["source_candidates"].values()
            },
            "source_ids": list(manifest["source_candidates"]),
            "reviewed_recording_ids": [
                v["recording_id"] for v in manifest["source_candidates"].values()
            ],
            "review_digest": "a" * 64,
            "review_manifest_digest": "b" * 64,
            "gates": dict.fromkeys(gates, True),
            "metrics_by_type": {
                k: {
                    "reviewed_positives": 100,
                    "precision": 0.95,
                    "recall": 0.95,
                    "start_mae_frames": 2,
                    "end_mae_frames": 2,
                }
                for k in ("contact", "grasp")
            },
        }
    )


def test_admission_requires_exact_independent_qa_and_complete_counts():
    records, study, partition = setup()
    manifest = build_diagnostic_manifest(records, study, partition)
    manifest = reseal(manifest | {"capture_completion_digest": "c" * 64})
    qa = qa_fixture(manifest)
    admitted = admit_manifest(manifest, qa)
    assert admitted["labels_admitted"] and not admitted["training_authorized"]
    for damage in ("empty", "coverage", "candidate", "recall", "fail"):
        bad = copy.deepcopy(qa)
        if damage == "empty":
            bad["gates"] = {}
        elif damage == "coverage":
            bad["source_ids"].pop()
        elif damage == "candidate":
            bad["candidate_digests"]["recording-0000"] = "0" * 64
        elif damage == "recall":
            bad["metrics_by_type"]["contact"]["recall"] = 0.94
        else:
            bad["status"] = "fail"
        with pytest.raises(ValueError):
            admit_manifest(manifest, reseal(bad))
