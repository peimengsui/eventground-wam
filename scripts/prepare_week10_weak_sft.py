"""Freeze the user's explicitly authorized two-pass exploratory SFT launch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_sft import admission, schedule_contract, validate_launch
from eventground_wam.eventground_weak_supervision import sealed, sha

BUNDLE = "docs/evaluation/week10-weak-sft-v1"
LAUNCH = "configs/training/week10_weak_sft_launch_v1.json"
QUALIFICATION = "docs/evaluation/week10-exploratory-weak-v4"


def make_launch(root, schedule):
    root = Path(root)
    inputs = {"protocol": "configs/training/week10_eventground_exploratory_v4.json",
              "dataset": f"{QUALIFICATION}/dataset.json",
              "recipe": "configs/training/week10_eventground_numerical_v2.json",
              "qualification": f"{QUALIFICATION}/run/completion.json",
              "qualification_contract": f"{QUALIFICATION}/qualification-contract.json",
              "schedule": f"{BUNDLE}/schedule.json"}
    p,d,r,q = (sealed(root/inputs[k]) for k in ("protocol","dataset","recipe","qualification"))
    code = dict(p["code_sha256"])
    for name in ("src/eventground_wam/eventground_weak_sft.py", "scripts/prepare_week10_weak_sft.py",
                 "scripts/train_week10_weak_sft.py"):
        code[name] = sha(root/name)
    return seal_contract({
        "artifact_type": "eventground_wam.week10_weak_sft_launch", "version": 1, "stage": "weak-sft",
        "user_request": "Can you Finish the formal SFT launch setup, and start Train SFT from a fresh copy of M4? How long will the training need to run?",
        "formal_sft_authorized": True, "rlvr_authorized": False, "independent_qa_passed": False,
        "protocol_digest": p["digest"], "dataset_digest": d["digest"], "recipe_digest": r["digest"],
        "qualification_digest": q["digest"], "schedule_digest": schedule["digest"],
        "inputs": inputs, "input_files_sha256": {path: sha(root/path) for path in inputs.values()},
        "code_sha256": code, "initialization": "fresh_M4_fixed_epoch_2_plus_new_head",
        "pilot_weights_reused": False, "optimizer_state_reused": False, "epochs": 2,
        "logical_updates": 168, "prompt_presentations": 2664, "logical_batch_size": 16,
        "microbatch_size": 1, "seed": 17, "heldout_payload_reads": 0,
        "numerical_recipe_inheritance": "Unchanged FP32 model, exact parameter allowlist, seed, optimizer and learning rates; replace only the old 2000-prompt/250-update schedule with this frozen 1332-prompt/168-update schedule",
        "partial_batch_rule": "Retain four-prompt final batch of each pass; normalize by its actual size; no padding, duplication or drop",
        "host": "feifei", "maximum_attempts": 1, "automatic_retries": 0,
        "checkpoint_selection": "fixed_epoch_2_update_168", "checkpoint_steps": [25,50,75,84,100,125,150,168],
        "output": "/home/peimengsui/data/eventground-wam/libero-plus/results/week10-weak-sft-v1/run",
        "qualification_root": "/home/peimengsui/data/eventground-wam/libero-plus/results/week10-exploratory-weak-v4/run",
        "resources": {"wall_seconds": 3600, "output_bytes": 16*1024**3,
                      "minimum_free_bytes": 50*1024**3, "maximum_gpu_fraction": .9,
                      "cpu_threads": 4, "maximum_cached_observation_bytes": 600*1024**2,
                      "prior_active_gpu_seconds": q["cumulative_qualification_seconds"],
                      "prior_output_bytes": 1024**3},
        "initial_estimate": {"pilot_seconds_per_update": q["mean_seconds_per_update"],
                             "optimizer_minutes": q["mean_seconds_per_update"] * 2664/16/60,
                             "planning_wall_minutes_range": [15,25],
                             "qualification_is_not_a_full_data_throughput_measurement": True},
        "limitations": "Exploratory weak labels; no independent QA or generalization claim; 29 grasp prompts, no grasp lead times above eight frames",
    })


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();root=args.root
    if (root/BUNDLE/"schedule.json").exists() or (root/LAUNCH).exists():
        raise FileExistsError("Formal launch already frozen; never overwrite")
    dataset=sealed(root/QUALIFICATION/"dataset.json")
    schedule=schedule_contract(dataset)
    write(root/BUNDLE/"schedule.json",schedule)
    launch=make_launch(root,schedule)
    validate_launch(root,launch)
    write(root/LAUNCH,launch)
    write(root/BUNDLE/"admission.json",admission(launch))
    print(json.dumps({"launch_digest":launch["digest"],"schedule_digest":schedule["digest"],
                      "updates":168,"presentations":2664,"estimate":launch["initial_estimate"]},indent=2))


if __name__=="__main__":
    main()
