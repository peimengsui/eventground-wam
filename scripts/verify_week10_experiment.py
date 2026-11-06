"""Verify Week 10 scientific freeze and parent evidence without executing models."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

CONTRACT = Path("configs/training/week10_eventground_experiment_v1.json")
EVIDENCE = Path("docs/evaluation/week10-freeze-v1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    payload = {key: item for key, item in value.items() if key != "digest"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    require(value.get("digest") == digest(value), f"Digest mismatch: {path}")
    return value


def verify(root, contract=CONTRACT):
    root = Path(root).resolve()
    study = sealed(root / contract)
    require(study["experiment_id"] == "week10-eventground-v1", "Wrong study")
    for name, expected in study["source_files_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), f"Source outside root: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"Source hash mismatch: {name}")
    for record in study["document_history"]:
        path = (root / record["snapshot_path"]).resolve()
        require(path.is_relative_to(root), "Snapshot outside root")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
                f"Historical snapshot mismatch: {path}")

    check = sealed(root / "docs/evaluation/week9-progress/20261015T1438Z/check.json")
    numeric = sealed(root / "docs/evaluation/week9-final-v1/numerical-verification.json")
    remote = sealed(root / EVIDENCE / "feifei-entry.json")
    completion = sealed(root / EVIDENCE / "M4-completion.json")
    endpoint = sealed(root / EVIDENCE / "M4-endpoint.json")
    identity = sealed(root / EVIDENCE / "M4-training-identity.json")
    admission = sealed(root / "docs/evaluation/week9-progress/20261014T0443Z/M4-admission.json")
    controls = sealed(root / "configs/training/phase2_two_epoch_v1.json")
    prereq, parent = study["prerequisites"], study["parent"]
    require(check["remaining_week9_evaluation_work"] == [] and check["outcomes"] == 15000,
            "Week 9 incomplete")
    require(check["all_ten_panels_audited"] and numeric["all_ten_panel_reports_verified"],
            "Missing audited panels")
    require(numeric["both_panel_bootstraps_recomputed"], "Missing numerical verification")
    require(check["digest"] == prereq["week9_completion_digest"]
            == remote["week9_completion_digest"], "Week 9 completion link mismatch")
    require(numeric["analysis_digest"] == prereq["week9_analysis_digest"],
            "Analysis link mismatch")
    require(controls["digest"] == prereq["phase2_controls_digest"], "Action control mismatch")
    require(prereq["original_jepa_gate"] == "FAIL"
            and prereq["jepa_research_use_exception_preserved"], "JEPA qualification changed")
    require(remote["digest"] == prereq["remote_entry_digest"] and remote["status"] == "passed",
            "Missing fresh endpoint check")
    require(not remote["model_executed"] and not remote["remote_files_modified"],
            "Entry check scope changed")
    require(parent["arm"] == admission["arm"] == "M4", "Parent arm changed")
    require(parent["selection"] == admission["selection"] == "fixed-epoch-2-endpoint",
            "Parent selection changed")
    require(parent["admission_digest"] == admission["digest"], "Admission link changed")
    require(parent["identity_digest"] == admission["training_identity_digest"]
            == completion["identity_digest"] == endpoint["identity_digest"]
            == identity["digest"] == remote["M4_identity_digest"], "Parent identity mismatch")
    require(parent["completion_digest"] == completion["digest"]
            == remote["M4_completion_digest"], "Completion mismatch")
    require(parent["endpoint_digest"] == endpoint["digest"]
            == completion["endpoint_report_digest"] == admission["endpoint_report_digest"]
            == remote["M4_endpoint_report_digest"], "Endpoint report mismatch")
    require(parent["policy_sha256"] == endpoint["checkpoint_sha256"]
            == completion["checkpoint_sha256"] == admission["checkpoint_sha256"],
            "Policy payload mismatch")
    require(parent["predictor_sha256"] == endpoint["predictor_sha256"],
            "Predictor payload mismatch")
    path = Path(parent["checkpoint_path"])
    require(str(path) == admission["checkpoint_path"], "Checkpoint path changed")
    require(remote["files"][str(path)]["sha256"] == parent["policy_sha256"],
            "Remote policy hash mismatch")
    require(remote["files"][str(path.parent / "full-training-predictor.pt")]["sha256"]
            == parent["predictor_sha256"], "Remote predictor hash mismatch")
    for inherited, field, expected in [
        ("inherited_action_updates", "global_step", 117132),
        ("inherited_action_presentations", "samples_seen", 3412044),
        ("inherited_future_pairs", "future_pairs_seen", 16259770),
    ]:
        require(parent[inherited] == endpoint[field] == completion[field] == expected,
                f"Exposure mismatch: {field}")
    require(parent["additional_action_updates"] == 0
            and parent["additional_world_model_cpt_updates"] == 0
            and not parent["mutate_parent"], "Extra action/CPT training or parent mutation")

    scope = study["event_scope"]
    require(set(scope["required_positive_types"]) == {"gripper_close", "contact", "grasp"},
            "Required contact/grasp ontology removed")
    require(not scope["allow_close_only_fallback"]
            and scope["aligned_physical_evidence_required_before_new_labels_admitted"],
            "User-selected label prerequisite weakened")
    qa = sealed(root / "docs/freeze/week2-dataset-v1-human-qa.json")
    require(qa["quality_gate"]["admitted_event_type"] == "gripper_close"
            and qa["review"]["reviewed_prediction_count"] == 79,
            "Existing QA evidence changed")
    require(scope["new_labeler_thresholds"] is None
            and scope["labeler_contract_required_before_extraction_and_qa"],
            "Unimplemented labeler falsely admitted")

    split = sealed(root / "configs/datasets/libero-plus-trajectory-v2/manifest.json")
    partition = sealed(root / EVIDENCE / "validation-groups.json")
    require(study["data"]["split_digest"] == split["digest"]
            == partition["parent_split_digest"], "Wrong split inheritance")
    require(study["data"]["validation_partition_digest"] == partition["digest"],
            "Validation partition link changed")
    by_task = defaultdict(list)
    for line in (root / "configs/datasets/libero-plus-trajectory-v2/groups.jsonl").read_text().splitlines():
        group = json.loads(line)
        if group["split"] == "validation":
            by_task[group["task_key"]].append(group["group_digest"])
    expected = []
    for task, groups in sorted(by_task.items()):
        ordered = sorted(groups, key=lambda group: hashlib.sha256(
            f"week10-event-validation-v1:10:{group}".encode()).hexdigest())
        expected.extend({"task_key": task, "trajectory_group_digest": group,
                         "partition": "event-dev" if index % 2 == 0 else "event-report"}
                        for index, group in enumerate(ordered))
    require(partition["assignments"] == expected, "Validation partition membership mismatch")
    require(len(expected) == split["counts"]["validation"]["groups"] == 180,
            "Validation group coverage mismatch")
    require(len({row["trajectory_group_digest"] for row in expected}) == 180,
            "Duplicate or leaking group")
    counts = {part: sum(row["partition"] == part for row in expected)
              for part in ["event-dev", "event-report"]}
    require(partition["counts"] == counts, "Partition count mismatch")
    require(study["data"]["test_payloads_locked"]
            and not study["data"]["week9_rollouts_used_for_training"], "Data boundary changed")

    require(study["training"]["group_size"] == 4, "RL group size changed")
    require(study["training"]["numerical_recipe"] is None
            and study["training"]["recipe_required_before_any_optimization_pilot"],
            "Recipe pending status changed")
    require(study["reward"]["weights"] == [0.40, 0.25, 0.20, 0.15]
            and study["reward"]["calibration_weights"] == [0.30, 0.35, 0.20, 0.15]
            and study["reward"]["maximum_calibrations"] == 1,
            "Frozen reward/bounded calibration changed")
    require(study["data"]["sft_unique_prompts"] == study["data"]["main_rl_unique_prompts"]
            == 2000 and study["data"]["rl_pilot_unique_prompts"] == 500,
            "Exposure budget changed")
    readiness = study["readiness"]
    require(readiness["design_complete"] and all(not value for key, value in readiness.items()
                                                 if key != "design_complete"),
            "Step 1 incorrectly marked ready for training")
    return {
        "artifact_type": "eventground_wam.week10_experiment_verification", "status": "passed",
        "experiment_digest": study["digest"],
        "pinned_sources_verified": len(study["source_files_sha256"]),
        "historical_documents_preserved": len(study["document_history"]),
        "parent_M4_completion_links_verified": True,
        "parent_payloads_freshly_rehashed_on_feifei": 2,
        "validation_groups": counts, "cross_partition_overlap": 0,
        "test_payloads_read": False, "contact_grasp_labels_admitted": False,
        "training_ready": False, "training_launched": False,
        "scope": "Scientific contract and evidence integrity only; no model/data-quality/resource certification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.contract), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
