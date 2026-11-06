"""Explicit weak-label SFT admission and fixed two-pass schedule.

This child never changes the original independent-QA admission functions.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.eventground_weak_supervision import (
    build_dataset,
    overfit_gates,
    sealed,
    sha,
    verify_pilot,
)

CLASSES = {"contact": 327, "grasp": 29, "gripper_close": 476, "none": 500}
QUALIFICATION_GATES = {
    "current_observation_only", "exact_parameter_allowlist", "bitwise_checkpoint_recovery",
    "loss_reduced", "event_accuracy", "joint_two_frame_accuracy", "frozen_parameters_unchanged",
    "trainable_parameters_changed", "stored_parent_unchanged", "finite_losses_gradients_parameters",
}


def schedule_rows(dataset):
    if dataset.get("digest") != contract_digest(dataset):
        raise ValueError("Unsealed weak dataset")
    rows = dataset["main_rows"]
    if (len(rows) != 1332 or len({r["prompt_id"] for r in rows}) != 1332
            or Counter(r["target"]["next_event"] for r in rows) != CLASSES):
        raise ValueError("Weak SFT requires the unchanged 1332-prompt pool")
    batches = []
    for epoch in range(2):
        ordered = sorted(rows, key=lambda r: (
            hashlib.sha256(f"17:{epoch}:{r['prompt_id']}".encode()).hexdigest(), r["prompt_id"]))
        batches.extend(ordered[i:i+16] for i in range(0, len(ordered), 16))
    return batches


def schedule_contract(dataset):
    batches = schedule_rows(dataset)
    return seal_contract({"artifact_type": "eventground_wam.week10_weak_sft_schedule",
                          "dataset_digest": dataset["digest"], "seed": 17, "epochs": 2,
                          "logical_batch_size": 16, "total_updates": 168,
                          "total_prompt_presentations": 2664,
                          "batch_sizes": [len(b) for b in batches],
                          "batches": [[r["prompt_id"] for r in b] for b in batches]})


def cursor(step, prompts_seen, schedule):
    if (type(step) is not int or not 0 <= step <= 168
            or prompts_seen != sum(schedule["batch_sizes"][:step])):
        raise ValueError("Checkpoint cursor differs from exact prompt exposure")
    return {"schedule_digest": schedule["digest"], "global_step": step,
            "prompts_seen": prompts_seen, "completed_epochs": step // 84,
            "next_epoch_index": step // 84 if step < 168 else None,
            "next_batch_index": step % 84 if step < 168 else None,
            "next_prompt_ids": schedule["batches"][step] if step < 168 else []}


def load_inputs(root, launch):
    return {key: sealed(Path(root) / path) for key, path in launch["inputs"].items()}


def validate_launch(root, launch, inputs=None):
    if launch.get("digest") != contract_digest(launch):
        raise ValueError("Unsealed formal weak-label launch")
    expected = {
        "artifact_type": "eventground_wam.week10_weak_sft_launch", "version": 1,
        "stage": "weak-sft", "formal_sft_authorized": True, "rlvr_authorized": False,
        "independent_qa_passed": False, "initialization": "fresh_M4_fixed_epoch_2_plus_new_head",
        "pilot_weights_reused": False, "optimizer_state_reused": False,
        "epochs": 2, "logical_updates": 168, "prompt_presentations": 2664,
        "logical_batch_size": 16, "microbatch_size": 1, "seed": 17,
        "heldout_payload_reads": 0, "maximum_attempts": 1, "automatic_retries": 0,
        "host": "feifei", "checkpoint_selection": "fixed_epoch_2_update_168",
    }
    if any(launch.get(k) != v or type(launch.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError("Formal SFT scope, initialization or fixed exposure changed")
    for mapping in (launch["input_files_sha256"], launch["code_sha256"]):
        for name, checksum in mapping.items():
            if Path(name).is_absolute() or ".." in Path(name).parts or sha(Path(root)/name) != checksum:
                raise ValueError(f"Sealed formal input/code changed: {name}")
    data = load_inputs(root, launch) if inputs is None else inputs
    if any(value.get("digest") != contract_digest(value) for value in data.values()):
        raise ValueError("Unsealed linked input")
    p, d, r, q, qc = (data[k] for k in ("protocol", "dataset", "recipe", "qualification", "qualification_contract"))
    if build_dataset(root,p) != d:
        raise ValueError("Weak dataset differs from reproducible source reconstruction")
    verify_pilot(qc,p,d,r)
    if (q["contract_digest"] != qc["digest"] or q["protocol_digest"] != p["digest"]
            or q["dataset_digest"] != d["digest"] or q["recipe_digest"] != r["digest"]
            or q["status"] != "passed_exploratory_qualification"
            or set(q["gates"]) != QUALIFICATION_GATES or not all(v is True for v in q["gates"].values())
            or q["logical_updates"] != 64 or q["executed_updates"] != 65
            or q["independent_qa_passed"] is not False or q["heldout_payload_reads"] != 0
            or q["qualification_weights_reusable_for_formal_training"] is not False
            or q["recovery"]["expected_digest"] != q["recovery"]["actual_digest"]
            or q["recovery"]["metrics_identical"] is not True):
        raise ValueError("Linked bounded qualification did not pass exactly")
    if ([m["global_step"] for m in q["evaluations"]] != [0,16,32,48,64]
            or not all(overfit_gates(q["evaluations"][0],q["evaluations"][-1],qc["thresholds"]).values())
            or q["wall_seconds"] > qc["wall_seconds"]
            or q["peak_reserved_bytes"] > .9*q["gpu_total_bytes"]
            or q["output_bytes"] > qc["resources"]["output_bytes"]):
        raise ValueError("Qualification measurements do not meet the sealed gates")
    if (data["schedule"] != schedule_contract(d) or launch["protocol_digest"] != p["digest"]
            or launch["dataset_digest"] != d["digest"] or launch["recipe_digest"] != r["digest"]
            or launch["qualification_digest"] != q["digest"]
            or launch["schedule_digest"] != data["schedule"]["digest"]):
        raise ValueError("Launch, data, recipe or schedule linkage differs")
    if r["compute_dtype"] != "float32" or r["logical_batch_size"] != 16 or r["microbatch_size"] != 1:
        raise ValueError("Numerical settings differ from qualified implementation")
    caps = launch["resources"]
    if (caps["wall_seconds"] != 3600 or caps["output_bytes"] != 16*1024**3
            or caps["minimum_free_bytes"] != 50*1024**3 or caps["maximum_gpu_fraction"] != .9
            or caps["cpu_threads"] != 4 or caps["maximum_cached_observation_bytes"] != 600*1024**2
            or caps["prior_active_gpu_seconds"] != q["cumulative_qualification_seconds"]
            or caps["prior_active_gpu_seconds"] + caps["wall_seconds"] > 40*3600
            or caps["prior_output_bytes"] != 1024**3
            or caps["prior_output_bytes"] + caps["output_bytes"] + 1024**3 > 100*1024**3):
        raise ValueError("Formal SFT resource ceiling or cumulative accounting changed")
    if launch["checkpoint_steps"] != [25,50,75,84,100,125,150,168]:
        raise ValueError("Checkpoint schedule changed")
    return schedule_rows(d)


def admission(launch):
    """Explicit approval of exploratory weak labels; never an independent-QA pass."""
    return seal_contract({"artifact_type": "eventground_wam.week10_weak_sft_admission",
        "launch_digest": launch["digest"], "protocol_digest": launch["protocol_digest"],
        "dataset_digest": launch["dataset_digest"], "qualification_digest": launch["qualification_digest"],
        "schedule_digest": launch["schedule_digest"], "status": "approved_exploratory_sft",
        "formal_sft_authorized": True, "independent_qa_passed": False,
        "label_evidence": "automatic physical predicates with limited assisted spot-checking",
        "initialization": "fresh_M4_fixed_epoch_2_plus_new_head", "rlvr_authorized": False})
