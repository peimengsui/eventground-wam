"""Seal the approved exploratory child and one bounded qualification before optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import build_dataset, sealed, sha, verify_pilot

PROTOCOL_PATH = "configs/training/week10_eventground_exploratory_v4.json"
RECIPE_PATH = "configs/training/week10_eventground_numerical_v2.json"
BUNDLE_PATH = "docs/evaluation/week10-exploratory-weak-v4"


def make_protocol(root):
    root = Path(root)
    inputs = {
        "parent_study": "configs/training/week10_eventground_experiment_v3.json",
        "numerical_parent": RECIPE_PATH,
        "partitions": "docs/evaluation/week10-source-split-v3/partitions.json",
        "blind_manifest": "docs/evaluation/week10-independent-review-v1/materials/manifest.json",
        "candidate_completion": "docs/evaluation/week10-production-labels-v1/run/completion.json",
        "original_diagnostic": "docs/evaluation/week10-production-labels-v1/run/diagnostic-manifest.json",
        "assisted_manifest": "docs/evaluation/week10-assisted-review-v1/manifest.json",
        "review_evidence": "docs/evaluation/week10-assisted-review-v1/review-snapshots/evidence-000177.json",
        "review_snapshot": "docs/evaluation/week10-assisted-review-v1/review-snapshots/draft-000177.json",
        "candidate_root": "docs/evaluation/week10-production-labels-v1/run",
        "capture_contract": "docs/evaluation/week10-production-capture-visual-v2/recording-contract.json",
        "capture_completion": "docs/evaluation/week10-production-capture-visual-v2/run/completion.json",
        "capture_verification": "docs/evaluation/week10-production-capture-visual-v2/verification.json",
    }
    study, recipe = sealed(root / inputs["parent_study"]), sealed(root / RECIPE_PATH)
    for name, checksum in {**study["source_files_sha256"], **recipe["code_sha256"]}.items():
        if sha(root / name) != checksum:
            raise ValueError(f"Original pinned implementation changed: {name}")
    code = dict(recipe["code_sha256"])
    for name in ("src/eventground_wam/eventground_weak_supervision.py",
                 "scripts/prepare_week10_weak_supervision.py", "scripts/qualify_week10_weak_supervision.py"):
        code[name] = sha(root / name)
    return seal_contract({
        "artifact_type": "eventground_wam.week10_exploratory_weak_supervision", "version": 4,
        "parent_study_digest": study["digest"], "numerical_parent_digest": recipe["digest"],
        "authorization": "User accepted the proposed exploratory path after reviewing 20 episodes: Ok, can you continue the path forward?",
        "independent_qa_passed": False, "qualification_authorized": True,
        "formal_training_authorized": False, "heldout_payload_reads": 0,
        "inputs": inputs, "input_files_sha256": {p: sha(root / p) for k, p in inputs.items() if k != "candidate_root"},
        "code_sha256": code,
        "capture_base": "/home/peimengsui/data/eventground-wam/libero-plus/results/week10-production-capture-visual-v2",
        "source_pool": "Exact initial 200 train recordings; no reserve or held-out payloads",
        "label_origin": "Automatic physical predicates with limited candidate-assisted human spot-checking",
        "review_limitations": {"episodes": 20, "tasks": 4, "accepted_positives": 113,
                               "whole_episode_missed_event_checks": 0, "active_human_minutes": None,
                               "unresolved_ambiguities": 1816, "independent_precision_recall_established": False},
        "allow_ongoing_event_anchors": False, "discard_ambiguities": False,
        "target_rule": "Earliest non-left-censored onset in offsets 1..32, grasp/contact/close tie order; end clipped at 33 with censoring retained",
        "ambiguity_rule": "Exclude any ambiguity intersecting [anchor,anchor+33), regardless of event class",
        "none_interpretation": "No eligible automatic onset in a complete window; not a human-verified negative",
        "maximum_prompts_per_source": 16, "maximum_anchors_per_event": 4,
        "maximum_prompts_per_class": 500, "qualification_prompts_per_class": 4,
        "sampling": "Unchanged deterministic v3 task/source round-robin with lead-bin balancing and actual class shortfalls; no duplication",
        "claims": "Exploratory feasibility and fitting to weak labels only; no blind-QA pass, no generalization claim from overfit",
        "earlier_future_onset_adjudication_proposal_activated": False,
        "qualification": "One fixed 16-example overfit run, 64 logical updates plus one step-9 recovery replay; discard optimization state for any later formal run",
        "later_sft_plan": "If qualification passes, prepare a separately approved two-pass endpoint on the actual unique prompt pool, freshly initialized from M4",
        "later_rlvr_and_evaluation": "Separate stage qualification and explicit weak-label reward/reporting contracts required; existing independent-evidence gates are unchanged",
    })


def make_contract(protocol, dataset, recipe):
    return seal_contract({
        "artifact_type": "eventground_wam.week10_weak_qualification_contract",
        "operation": "exploratory_weak_label_overfit_qualification", "attempts": 1,
        "protocol_digest": protocol["digest"], "dataset_digest": dataset["digest"], "recipe_digest": recipe["digest"],
        "prompt_ids": dataset["qualification_prompt_ids"], "logical_updates": 64,
        "maximum_executed_updates_including_replay": 65, "recovery_checkpoint_step": 8,
        "logical_batch_size": 16, "microbatch_size": 1, "seed": 17,
        "wall_seconds": 1800, "prior_qualification_active_seconds": 62.73182583646849,
        "resources": {"output_bytes": 8*1024**3, "minimum_free_bytes": 50*1024**3,
                      "maximum_gpu_fraction": .9, "cpu_threads": 4},
        "thresholds": {"maximum_loss_ratio": .5, "minimum_event_accuracy": .875,
                       "minimum_joint_two_frame_accuracy": .75},
        "metrics": "Greedy predictions on the same 16 examples; joint accuracy requires correct class and both boundaries within two frames, or correct none",
        "recovery_test": "Save after update 8; execute update 9; restore update 8; replay update 9 and require bitwise parameters/optimizer/RNG/counters plus exact scalar metrics",
        "formal_training_authorized": False, "independent_qa_passed": False, "heldout_payload_reads": 0,
        "checkpoint_selection": "Fixed update 64 regardless of intermediate scores; no automatic retry or hyperparameter search",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    protocol = make_protocol(args.root)
    dataset = build_dataset(args.root, protocol)
    recipe = sealed(args.root / RECIPE_PATH)
    contract = make_contract(protocol, dataset, recipe)
    verify_pilot(contract, protocol, dataset, recipe)
    destinations = {PROTOCOL_PATH: protocol, f"{BUNDLE_PATH}/dataset.json": dataset,
                    f"{BUNDLE_PATH}/qualification-contract.json": contract}
    if any((args.root / path).exists() for path in destinations):
        raise FileExistsError("Exploratory freeze already exists; never overwrite it")
    for path, value in destinations.items():
        target = args.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"protocol_digest": protocol["digest"], "dataset_digest": dataset["digest"],
                      "contract_digest": contract["digest"], "coverage": dataset["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
