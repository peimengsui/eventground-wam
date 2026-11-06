"""One fixed exploratory overfit pilot; no independent-QA or formal-run admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import torch
from train_week10_eventground import (
    check_gpu_exclusive,
    require_output_capacity,
    save_bounded_checkpoint,
    write_completion,
)

from eventground_wam.b0_training import capture_rng, configure_deterministic_cuda, seed_training
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_data import (
    collate_current_observations,
    load_current_observation,
    target_tokens,
)
from eventground_wam.eventground_objectives import structured_sft_loss
from eventground_wam.eventground_runtime import build_eventground_from_m4, verify_eventground_parent
from eventground_wam.eventground_training import EventGroundTrainer
from eventground_wam.eventground_weak_supervision import (
    build_dataset,
    overfit_gates,
    sealed,
    sha,
    verify_pilot,
)
from eventground_wam.pixel_predictor import tensor_state_digest


def tree_digest(value):
    """Bitwise identity of parameters, optimizer, RNG and counters, without torch.save metadata."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            value = item.detach().cpu().contiguous()
            digest.update(json.dumps(["tensor", str(value.dtype), list(value.shape)]).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())

    visit(value)
    return digest.hexdigest()


def recovery_digest(trainer):
    params = dict(trainer.model.named_parameters())
    return tree_digest({"parameters": {n: params[n] for n in trainer.names},
                        "optimizer": trainer.optimizer.state_dict(), "rng": capture_rng(),
                        "step": trainer.global_step, "prompts": trainer.prompts_seen})


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    rows, loss = [], 0.
    for encoding, state, tokens in batches:
        features = model(encoding, state)
        scores = model.head.teacher_forced(features, tokens)
        loss += float(structured_sft_loss(scores, logical_batch_size=len(batches)))
        pred = model.head.sample(features, group_size=1, greedy=True).tokens[0].tolist()
        target = tokens[0].tolist()
        correct_event = pred[0] == target[0]
        exact = pred == target
        joint = correct_event and (target[0] == 0 or max(abs(pred[i]-target[i]) for i in (1, 2)) <= 2)
        intersection = max(0, min(pred[2], target[2]) - max(pred[1], target[1]))
        union = pred[2]-pred[1] + target[2]-target[1] - intersection
        iou = intersection / union if target[0] and correct_event and union else 0.
        rows.append({"target": target, "prediction": pred, "correct_event": correct_event,
                     "exact": exact, "joint_two_frame": joint, "positive_iou": iou})
    n = len(rows)
    positives = [r for r in rows if r["target"][0]]
    return {"loss": loss, "event_accuracy": sum(r["correct_event"] for r in rows)/n,
            "exact_triple_accuracy": sum(r["exact"] for r in rows)/n,
            "joint_two_frame_accuracy": sum(r["joint_two_frame"] for r in rows)/n,
            "mean_positive_iou": sum(r["positive_iou"] for r in positives)/len(positives),
            "event_accuracy_by_class": {name: sum(r["correct_event"] for r in rows if r["target"][0] == k)
                                         / sum(r["target"][0] == k for r in rows)
                                         for k, name in enumerate(("none", "gripper_close", "contact", "grasp"))},
            "predictions": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "protocol", "dataset", "contract", "recipe", "cache", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    protocol, dataset, contract, recipe = map(sealed, (args.protocol, args.dataset, args.contract, args.recipe))
    if build_dataset(args.root, protocol) != dataset:
        raise ValueError("Saved weak dataset differs from complete metadata rebuild")
    verify_pilot(contract, protocol, dataset, recipe)
    if sha(Path(__file__)) != protocol["code_sha256"]["scripts/qualify_week10_weak_supervision.py"]:
        raise ValueError("Running pilot script differs from sealed source")
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"artifact_type": "eventground_wam.week10_weak_supervision_qualification",
              "contract_digest": contract["digest"], "protocol_digest": protocol["digest"],
              "dataset_digest": dataset["digest"], "recipe_digest": recipe["digest"],
              "status": "started", "independent_qa_passed": False, "formal_training_authorized": False,
              "heldout_payload_reads": 0, "qualification_weights_reusable_for_formal_training": False,
              "executed_updates": 0, "logical_updates": 0, "evaluations": [], "gates": {}}
    resources = contract["resources"]
    active_start = None

    def stop(*_):
        raise TimeoutError("Qualification wall ceiling; no automatic retry")

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(contract["wall_seconds"])
    try:
        require_output_capacity(args.output, resources, 0)
        check_gpu_exclusive()
        configure_deterministic_cuda()
        torch.set_num_threads(resources["cpu_threads"])
        seed_training(contract["seed"])
        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(resources["maximum_gpu_fraction"], device)
        torch.cuda.reset_peak_memory_stats()
        row_map = {r["prompt_id"]: r for r in dataset["main_rows"]}
        rows = [row_map[k] for k in contract["prompt_ids"]]
        verified_files = {}
        observations = [load_current_observation(r, verified_files=verified_files) for r in rows]
        report["observation_reads"] = len(observations)
        report["source_files_verified"] = len(verified_files)
        active_start = time.monotonic()
        model, builder, allowlist, parent_identity = build_eventground_from_m4(args.root, recipe, args.cache, device)
        if {k: list(v) for k, v in allowlist.items()} != recipe["exact_allowlist"]:
            raise ValueError("Trainable/frozen parameter groups differ from numerical parent")
        trainer = EventGroundTrainer(model, allowlist, recipe, stage="sft")
        identity = seal_contract({"artifact_type": "eventground_wam.week10_weak_qualification_identity",
                                  "stage": "weak-qualification", "contract_digest": contract["digest"],
                                  "parent_identity": parent_identity,
                                  "independent_qa_passed": False, "formal_training_authorized": False})
        with (args.output / "identity.json").open("x") as stream:
            json.dump(identity, stream, indent=2, sort_keys=True)
            stream.write("\n")
        batches = []
        for obs, row in zip(observations, rows, strict=True):
            encoded, state = collate_current_observations([obs], builder, device)
            batches.append((encoded, state, target_tokens([row], device)))
        del observations
        frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
        frozen_before = tensor_state_digest(frozen)
        trainable_before = tensor_state_digest({n: p for n, p in model.named_parameters() if p.requires_grad})
        report["gates"].update(current_observation_only=True, exact_parameter_allowlist=True)
        initial = evaluate(model, batches)
        report["evaluations"].append({"global_step": 0, **initial})
        schedule = [rows] * contract["logical_updates"]
        recovery = args.output / "step-000008.pt"
        recovery_digest_expected = recovery_metric_expected = None
        durations = []

        def update(kind):
            if report["executed_updates"] >= contract["maximum_executed_updates_including_replay"]:
                raise ValueError("Physical optimizer-update budget exhausted")
            require_output_capacity(args.output, resources, 16384)
            tick = time.monotonic()
            report["executed_updates"] += 1  # conservatively charge even a partial failed update
            metric = trainer.sft_update(batches, prompt_count=16)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            durations.append(elapsed)
            report["logical_updates"] = trainer.global_step
            record = {**metric, "kind": kind, "executed_updates": report["executed_updates"], "seconds": elapsed}
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
            return metric

        while trainer.global_step < contract["logical_updates"]:
            metric = update("scheduled")
            if trainer.global_step == contract["recovery_checkpoint_step"]:
                save_bounded_checkpoint(trainer, recovery, identity=identity, batches=schedule,
                                        output=args.output, resources=resources)
            if trainer.global_step == contract["recovery_checkpoint_step"] + 1:
                recovery_digest_expected = recovery_digest(trainer)
                recovery_metric_expected = metric
                trainer.resume(recovery, identity=identity)
                replay = update("recovery_replay")
                actual = recovery_digest(trainer)
                report["recovery"] = {"expected_digest": recovery_digest_expected, "actual_digest": actual,
                                       "metrics_identical": replay == recovery_metric_expected,
                                       "restored_step": 8, "replayed_step": 9}
                report["gates"]["bitwise_checkpoint_recovery"] = actual == recovery_digest_expected and replay == recovery_metric_expected
                if not report["gates"]["bitwise_checkpoint_recovery"]:
                    raise ValueError("Step-9 checkpoint replay differs; no further updates permitted")
            if trainer.global_step % 16 == 0:
                measured = evaluate(model, batches)
                report["evaluations"].append({"global_step": trainer.global_step, **measured})
                print(json.dumps({"evaluation_step": trainer.global_step,
                                  **{k: v for k, v in measured.items() if k != "predictions"}}), flush=True)
        final = report["evaluations"][-1]
        report["gates"].update(overfit_gates(initial, final, contract["thresholds"]))
        report["gates"]["frozen_parameters_unchanged"] = tensor_state_digest(frozen) == frozen_before
        report["gates"]["trainable_parameters_changed"] = tensor_state_digest(
            {n: p for n, p in model.named_parameters() if p.requires_grad}) != trainable_before
        verify_eventground_parent(sealed(args.root / protocol["inputs"]["parent_study"]))
        report["gates"]["stored_parent_unchanged"] = True
        endpoint = args.output / "qualification-step-000064.pt"
        checksum = save_bounded_checkpoint(trainer, endpoint, identity=identity, batches=schedule,
                                          output=args.output, resources=resources)
        report.update(endpoint_file=endpoint.name, endpoint_sha256=checksum,
                      physical_prompt_presentations=16 * report["executed_updates"],
                      logical_prompt_presentations=trainer.prompts_seen,
                      mean_seconds_per_update=sum(durations)/len(durations),
                      frozen_parameter_digest=frozen_before)
        report["gates"]["finite_losses_gradients_parameters"] = all(math.isfinite(v)
            for row in report["evaluations"] for v in [row["loss"]])
        report["status"] = "passed_exploratory_qualification" if all(report["gates"].values()) else "failed_predeclared_overfit_gate"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        report["wall_seconds"] = time.monotonic() - start
        report["conservative_gpu_occupancy_seconds"] = time.monotonic()-active_start if active_start else 0
        report["cumulative_qualification_seconds"] = contract["prior_qualification_active_seconds"] + report["conservative_gpu_occupancy_seconds"]
        if torch.cuda.is_initialized():
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["files_sha256"] = {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob("*") if p.is_file()}
        write_completion(args.output, report, resources)
        print(json.dumps({"status": report["status"], "gates": report["gates"]}), flush=True)


if __name__ == "__main__":
    main()
