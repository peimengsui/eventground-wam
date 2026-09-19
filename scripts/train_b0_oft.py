#!/usr/bin/env python3
"""Train the separately versioned OFT B0 within a fixed single-GPU time budget."""

from __future__ import annotations

import fcntl
import signal
import socket
import sys
import time
from typing import Any

import torch
import train_b0 as common

from eventground_wam.b0_model import (
    QWEN_REVISION,
    B0ModelError,
    b0_trainable_state_dict,
    load_b0_trainable_state_dict,
)
from eventground_wam.b0_oft import POLICY_FAMILY
from eventground_wam.b0_training import (
    POLICY_FORMAT,
    CheckpointStore,
    TrainingContractError,
    TrainingEngine,
    atomic_json,
    atomic_torch_save,
    build_batch_plan,
    configure_deterministic_cuda,
    evaluate_validation,
    load_recipe,
    plan_digest,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.demonstration_contracts import seal_contract


def main(args: Any) -> int:
    recipe = load_recipe(args.recipe)
    if recipe.get("policy_family") != POLICY_FAMILY:
        raise TrainingContractError("OFT runner requires a versioned OFT recipe")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args, recipe)


def run(args: Any, recipe: Any) -> int:
    started = time.monotonic()
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise TrainingContractError("OFT training requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed_training(recipe["seed"])
    train, validation, plans, identity = common.inspect_data(args, recipe)
    for key, value in recipe["state_normalization"]["source"].items():
        if identity["source"].get(key) != value:
            raise TrainingContractError("state statistics belong to another dataset split")
    if recipe["maximum_updates"] > identity["steps_per_epoch"] * recipe["epochs"]:
        raise TrainingContractError("update budget exceeds the available epoch plans")
    output = args.output_root
    manifest_path = output / "training-manifest.json"
    if manifest_path.exists() and read_sealed(manifest_path)["digest"] != identity["digest"]:
        raise TrainingContractError("output identity changed; use a new output directory")
    atomic_json(manifest_path, identity)
    if args.preflight_only:
        return 0
    if (output / "training-report.json").exists():
        report = read_sealed(output / "training-report.json")
        if (report["identity_digest"] != identity["digest"]
                or report["checkpoint_sha256"] != sha256_file(output / "B0.ckpt")):
            raise TrainingContractError("completed OFT training bundle changed")
        print("OFT training is already complete.", flush=True)
        return 0
    if not args.pilot_steps:
        pilot = read_sealed(args.pilot_report)
        if (pilot["status"] != "passed" or pilot["recipe"] != recipe
                or pilot["project_git_revision"] != identity["project_git_revision"]
                or pilot["b0_runtime_lock_sha256"] != identity["b0_runtime_lock_sha256"]):
            raise TrainingContractError("production requires a matching passing OFT pilot")
    policy, names = common.build_policy(recipe, str(args.cache_dir), device)
    engine = TrainingEngine(policy, names, recipe, identity_digest=identity["digest"],
                            steps_per_epoch=identity["steps_per_epoch"], device=device)
    store = CheckpointStore(output / "recovery", identity["digest"], recipe["checkpoint_keep_last"])
    saved = store.load()
    if saved is not None:
        engine.restore(saved)
    previous_active = saved.get("active_seconds", 0.0) if saved else 0.0
    validations = saved.get("validation_history", []) if saved else []

    def active_seconds() -> float:
        return previous_active + time.monotonic() - started

    def save() -> None:
        store.save({**engine.snapshot(), "active_seconds": active_seconds(),
                    "validation_history": validations})

    save()
    common.write_event(output, {
        "event": "started", "host": socket.gethostname(), "global_step": engine.global_step,
        "epoch": engine.epoch, "next_batch": engine.next_batch,
        "recovery_notes": store.recovery_notes,
        "trainable_parameters": sum(p.numel() for p in engine.parameters),
        "lora_parameter_tensors": len(names), "resumed": saved is not None,
    })
    first_plan, validation_plan = plans
    torch.cuda.reset_peak_memory_stats()
    pilot_step = engine.global_step
    measured_seconds = measured_samples = 0.0
    last_save = last_log = time.monotonic()

    def validate_and_export() -> None:
        save()
        common.write_event(output, {"event": "validation_started", "global_step": engine.global_step})
        val_started = time.monotonic()
        metrics = evaluate_validation(
            policy, common.make_loader(validation, validation_plan, args, recipe), device)
        if metrics["sample_count"] != validation.sample_count:
            raise TrainingContractError("validation did not cover the complete frozen split")
        path = output / "policies" / f"step-{engine.global_step:09d}.pt"
        digest = atomic_torch_save(path, {
            "format": POLICY_FORMAT, "identity_digest": identity["digest"],
            "qwen_revision": QWEN_REVISION, "policy_family": POLICY_FAMILY,
            "state_dict": b0_trainable_state_dict(policy, allowed_exact_names=names),
        })
        candidate = {"global_step": engine.global_step, "samples_seen": engine.samples_seen,
                     "epoch_fraction": engine.samples_seen / train.sample_count,
                     "path": str(path.relative_to(output)), "sha256": digest,
                     "validation": metrics, "validation_seconds": time.monotonic() - val_started}
        atomic_json(path.with_suffix(".json"), seal_contract(candidate))
        validations.append(candidate)
        if engine.best is None or metrics["objective"] < engine.best["validation"]["objective"]:
            engine.best = candidate
        save()
        common.write_event(output, {"event": "validation_complete", **candidate})

    if (not args.pilot_steps and engine.global_step > 0
            and engine.global_step % recipe["validation_interval_steps"] == 0
            and (not validations or validations[-1]["global_step"] != engine.global_step)):
        validate_and_export()
    stop_reason = "maximum-updates"
    while engine.epoch < recipe["epochs"] and engine.global_step < recipe["maximum_updates"]:
        plan = first_plan if engine.epoch == 0 else build_batch_plan(
            train.ordered_episodes(engine.epoch), batch_size=recipe["batch_size"],
            seed=recipe["seed"], epoch=engine.epoch, shuffle=True)
        if plan_digest(plan) != identity["epoch_plan_digests"][engine.epoch]:
            raise TrainingContractError("regenerated sample order changed")
        loader = common.make_loader(train, plan, args, recipe, start=engine.next_batch)
        previous = time.monotonic()
        interrupted = budget_stop = False
        for batch in loader:
            if (not args.pilot_steps and
                    active_seconds() >= recipe["training_active_hours"] * 3600):
                budget_stop = True
                stop_reason = "active-time-budget"
                break
            record = engine.step(batch)
            now = time.monotonic()
            if engine.global_step - pilot_step > 20:
                measured_seconds += now - previous
                measured_samples += batch["sample_count"]
            previous = now
            if engine.global_step % recipe["log_interval_steps"] == 0 or now - last_log >= 60:
                event = {**record, "active_seconds": active_seconds(),
                         "steady_samples_per_second": measured_samples / max(measured_seconds, 1e-9),
                         "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated()}
                common.write_event(output, {"event": "progress", **event})
                atomic_json(output / "status.json", {"status": "running", **event})
                last_log = now
            if now - last_save >= recipe["checkpoint_interval_seconds"]:
                save()
                last_save = time.monotonic()
            if common.STOP_REQUESTED:
                save()
                interrupted = True
                break
            if args.pilot_steps and engine.global_step - pilot_step >= args.pilot_steps:
                result = common.verify_reload(engine, store, batch)
                save()
                throughput = measured_samples / measured_seconds
                update_seconds = measured_seconds / max(1, engine.global_step - pilot_step - 20)
                common.write_event(output, {"event": "validation_probe_started"})
                val_start = time.monotonic()
                val_probe = evaluate_validation(
                    policy, common.make_loader(validation, validation_plan[:128], args, recipe),
                    device)
                val_throughput = val_probe["sample_count"] / (time.monotonic() - val_start)
                val_count = (recipe["maximum_updates"] // recipe["validation_interval_steps"]
                             + recipe["maximum_updates"] // engine.steps_per_epoch + 2)
                projected = (update_seconds * recipe["maximum_updates"]
                             + val_count * validation.sample_count / val_throughput) / 3600
                projected *= recipe["pilot_projection_margin"]
                report = seal_contract({
                    "artifact_type": "eventground_wam.b0_oft_pilot",
                    "status": ("passed" if projected <= recipe["maximum_active_hours"]
                               else "over-time-budget"),
                    "recipe": recipe, "project_git_revision": identity["project_git_revision"],
                    "b0_runtime_lock_sha256": identity["b0_runtime_lock_sha256"],
                    "identity_digest": identity["digest"], "samples_per_second": throughput,
                    "seconds_per_update": update_seconds, "projected_training_hours": projected,
                    "validation_samples_per_second": val_throughput,
                    "projected_full_validation_seconds": validation.sample_count / val_throughput,
                    "planned_full_validation_passes_upper_bound": val_count,
                    "validation_probe": val_probe,
                    "budget_enforced_at_runtime": True, "reload_validation": result,
                    "updates": engine.global_step - pilot_step,
                    "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
                    "device": torch.cuda.get_device_name(),
                    "trainable_parameters": sum(p.numel() for p in engine.parameters),
                })
                atomic_json(output / "pilot-report.json", report)
                common.write_event(output, {"event": "pilot_complete", **report})
                return 0 if report["status"] == "passed" else 78
            if engine.global_step >= recipe["maximum_updates"]:
                break
            if engine.global_step % recipe["validation_interval_steps"] == 0:
                validate_and_export()
                previous = time.monotonic()
                if common.STOP_REQUESTED:
                    interrupted = True
                    break
        del loader
        if interrupted:
            atomic_json(output / "status.json", {"status": "interrupted",
                                                "global_step": engine.global_step})
            return 75
        if budget_stop or engine.global_step >= recipe["maximum_updates"]:
            break
        if not validations or validations[-1]["global_step"] != engine.global_step:
            validate_and_export()
        engine.finish_epoch(validations[-1]["validation"])
        save()
    if not validations or validations[-1]["global_step"] != engine.global_step:
        validate_and_export()
    if common.STOP_REQUESTED:
        save()
        return 75
    best = engine.best
    best_path = output / best["path"]
    if sha256_file(best_path) != best["sha256"]:
        raise TrainingContractError("selected OFT policy hash mismatch")
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    load_b0_trainable_state_dict(policy, selected["state_dict"], allowed_exact_names=names)
    final_validation = evaluate_validation(
        policy, common.make_loader(validation, validation_plan, args, recipe), device)
    if abs(final_validation["objective"] - best["validation"]["objective"]) > 1e-6:
        raise TrainingContractError("selected policy reload changed full validation objective")
    digest = atomic_torch_save(output / "B0.ckpt", selected)
    report = seal_contract({
        "artifact_type": "eventground_wam.b0_oft_training_report", "status": "complete",
        "policy_family": POLICY_FAMILY, "identity_digest": identity["digest"],
        "epochs_completed": engine.epoch, "epoch_equivalent_exposure":
        engine.samples_seen / train.sample_count, "global_step": engine.global_step,
        "samples_seen": engine.samples_seen, "sample_chain": engine.sample_chain,
        "stop_reason": stop_reason, "active_seconds": active_seconds(),
        "within_resource_budget": active_seconds() <= recipe["maximum_active_hours"] * 3600,
        "validation_history": validations, "selected_step": best["global_step"],
        "checkpoint_sha256": digest, "final_reload_validation": final_validation,
        "reload_validation_passed": True,
    })
    atomic_json(output / "training-report.json", report)
    atomic_json(output / "status.json", {"status": "complete", "global_step": engine.global_step})
    common.write_event(output, {"event": "complete", "checkpoint_sha256": digest})
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, common.request_stop)
    signal.signal(signal.SIGINT, common.request_stop)
    try:
        sys.exit(main(common.parse_args()))
    except (TrainingContractError, B0ModelError, torch.OutOfMemoryError) as error:
        print(f"NONRETRYABLE: {error}", file=sys.stderr, flush=True)
        sys.exit(78)
