"""Check the Week 7 study freeze against its pinned inputs without loading models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed(path: Path) -> dict:
    value = json.loads(path.read_text())
    payload = {key: item for key, item in value.items() if key != "digest"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    require(value.get("digest") == hashlib.sha256(encoded).hexdigest(), f"Bad digest: {path}")
    return value


def verify(root: Path, contract: Path) -> dict:
    study = sealed(contract)
    require(study["experiment_id"] == "week7-auxiliary-v1", "Unexpected experiment")
    for name, expected in study["source_files_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), f"Source outside repository: {name}")
        require(sha256(path) == expected, f"Source hash mismatch: {name}")

    controls = sealed(root / "configs/training/phase2_two_epoch_v1.json")
    modules = sealed(root / "configs/training/phase2_future_modules_v1.json")
    decision = sealed(root / "docs/evaluation/week6-progression-v1/decision.json")
    progression = sealed(root / "docs/evaluation/week6-progression-v1/completion.json")
    entry = sealed(root / "docs/evaluation/week7-freeze-v1/week6-entry-verification.json")
    baseline = sealed(root / "docs/evaluation/week4-completion-v1.json")["selected_baseline"]

    require(progression["status"] == "complete", "Week 6 progression is incomplete")
    require(progression["week6_status"] == decision["week6_status"] == "closed", "Week 6 open")
    for value in (progression, decision):
        require(value["remaining_week6_tasks"] == [], "Week 6 work remains")
    require(decision["accepted_for_downstream_research_use"], "Missing research-use exception")
    require(decision["frozen_gate_status"] == "FAIL", "Original JEPA gate was relabeled")
    require(modules["authorization_decision_digest"] == decision["digest"], "Decision differs")
    require(progression["module_registry_digest"] == modules["digest"], "Registry differs")
    require(study["prerequisites"]["module_registry_digest"] == modules["digest"],
            "Study registry differs")
    require(study["prerequisites"]["progression_decision_digest"] == decision["digest"],
            "Study progression decision differs")
    require(entry["status"] == "passed", "Entry audit failed")
    require(study["prerequisites"]["week6_entry_verification_digest"] == entry["digest"],
            "Entry audit differs")

    for row in entry["document_history"]:
        require(sha256(root / row["preserved_snapshot_path"]) == row["original_sha256"],
                f"Changed pre-freeze document snapshot: {row['original_relative_path']}")

    require(set(study["arms"]) == {"B0", "M1", "M2"}, "Wrong Week 7 arms")
    require(study["arms"]["B0"]["checkpoint_sha256"] == baseline["checkpoint_sha256"],
            "B0 checkpoint differs")
    require(study["arms"]["B0"]["new_action_training"] is False, "Extra B0 training enabled")
    for arm, paired_arm in (("M1", "M3"), ("M2", "M4")):
        branch = study["arms"][arm]
        module_id = modules["branch_to_module"][arm]
        require(branch["future_module_id"] == module_id == modules["branch_to_module"][paired_arm],
                f"Module mapping differs: {arm}")
        require(branch["future_module_sha256"] == modules["modules"][module_id]["sha256"],
                f"Module hash differs: {arm}")
        require(branch["future_module_gate"] == modules["modules"][module_id]["original_gate"],
                f"Module acceptance relabeled: {arm}")
        require(branch["train_loaded_future_predictor"], f"Disconnected experiment: {arm}")
        require(not branch["future_tokens_in_action_path"] and not branch["future_module_at_inference"],
                f"Unexpected inference coupling: {arm}")

    jepa = modules["modules"]["C_jepa-study-v1"]
    require(study["arms"]["M2"]["research_use_exception"] is True, "JEPA exception omitted")
    require(sha256(root / jepa["local_path"]) == jepa["sha256"], "JEPA export differs")
    require(study["shared_controls"]["digest"] == controls["digest"], "Action controls differ")
    for field in ("initialization", "action_training", "batching_and_sampling"):
        require(study["shared_controls"][field] == controls[field], f"Changed control: {field}")
    action = controls["action_training"]
    require((action["epochs"], action["optimizer_updates"], action["presentations"])
            == (2, 117132, 3412044), "Wrong action budget")

    supervision = study["future_supervision"]
    require(supervision["horizons_frames"] == [2, 4, 8, 16, 32], "Wrong horizons")
    require(supervision["preserve_all_action_samples"], "Action samples can be removed")
    require(supervision["outer_loss_weights"] == {"M1": None, "M2": None},
            "Step 1 must not invent measured pilot weights")
    pixel = sealed(root / "configs/training/week5_pixel_cpt_v1.json")
    jepa_recipe = sealed(root / "configs/training/week6_jepa_cpt_v1.json")
    require(supervision["pixel_loss"] == pixel["loss"], "Pixel objective differs")
    require(supervision["jepa_loss"] == {"cosine_weight": 1.0, **jepa_recipe["loss"]},
            "JEPA objective differs")
    for field in ("pair_presentations", "total_updates", "order_manifest_sha256", "order_rule"):
        require(pixel[field] == jepa_recipe[field], f"Unmatched original CPT: {field}")

    evaluation = study["evaluation"]
    panel = json.loads((root / evaluation["manifest_path"]).read_text())
    report = sealed(root / evaluation["baseline_report_path"])
    require(panel["task_count"] == evaluation["task_count"] == len(panel["tasks"]) == 210,
            "Wrong development panel")
    require(panel["task_records_sha256"] == evaluation["task_records_sha256"], "Panel differs")
    require(panel["protocol_sha256"] == evaluation["protocol_sha256"], "Protocol differs")
    task_keys = [(t["suite"], t["runtime_task_index"]) for t in panel["tasks"]]
    report_keys = [(t["suite"], t["runtime_task_index"]) for t in report["episodes"]]
    require(len(set(task_keys)) == 210 and report_keys == task_keys, "B0 identities differ")
    require(sum(bool(t["success"]) for t in report["episodes"])
            == report["success_count"] == evaluation["baseline_successes"] == 162,
            "B0 result differs")
    require(evaluation["primary_contrasts"] == ["M1-B0", "M2-B0"], "Wrong primary contrasts")
    require(not evaluation["test_payload_access"] and not evaluation["larger_confirmation_panel_access"],
            "Confirmation data unlocked")
    require(not evaluation["rollout_outcomes_may_select_recipe"], "Rollout tuning permitted")
    require(not study["completion"]["positive_policy_gain_required"], "Negative result excluded")
    require(study["production"]["ready"] is False and study["production"]["launched"] is False,
            "Step 1 is incorrectly marked production-ready")
    return {
        "artifact_type": "eventground_wam.week7_experiment_verification",
        "status": "passed",
        "experiment_digest": study["digest"],
        "source_files_verified": len(study["source_files_sha256"]),
        "document_snapshots_verified": len(entry["document_history"]),
        "week6_entry_verification_digest": entry["digest"],
        "week6_status": "closed",
        "jepa_original_gate": "FAIL",
        "jepa_research_use_authorized": True,
        "b0_task_identities_verified": len(task_keys),
        "shared_action_controls_match": True,
        "inherited_cpt_exposure_and_objectives_match": True,
        "production_ready": False,
        "scope": "Local contract, source hashes, snapshots, JEPA export and report metadata; no model execution or remote rehash. Archive checks are in the pinned entry audit.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--contract", type=Path,
                        default=Path("configs/training/week7_auxiliary_experiment_v1.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(verify(root, root / args.contract), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
