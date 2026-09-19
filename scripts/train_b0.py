#!/usr/bin/env python3
"""Train five complete B0 epochs with durable, exact-cursor recovery on one GPU."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from eventground_wam.artifact_provenance import project_state
from eventground_wam.b0_data import (
    B0DataConfig,
    B0StreamingDataset,
    load_b0_action_normalizer,
    load_frozen_b0_split_index,
)
from eventground_wam.b0_model import (
    QWEN_REVISION,
    ActionQueryTransformer,
    ActionQueryTransformerConfig,
    B0ActionPolicy,
    B0ObjectiveConfig,
    QwenConditioningAdapter,
    QwenConditioningConfig,
    b0_trainable_state_dict,
    install_qwen_upper_lora,
    load_b0_trainable_state_dict,
    load_pinned_qwen_model,
)
from eventground_wam.b0_training import (
    POLICY_FORMAT,
    CheckpointStore,
    EpochBatches,
    TrainingContractError,
    TrainingEngine,
    atomic_json,
    atomic_torch_save,
    build_batch_plan,
    configure_deterministic_cuda,
    evaluate_validation,
    load_recipe,
    move_batch,
    plan_digest,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256
from eventground_wam.evaluation_splits import load_manifest_family

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"signal {signum}: saving at the next completed update", flush=True)


def build_policy(recipe: dict[str, Any], cache_dir: str, device: torch.device) -> tuple[Any, Any]:
    if recipe.get("policy_family") == "qwen35-oft-b0-v1":
        from eventground_wam.b0_oft import build_oft_policy

        return build_oft_policy(recipe, cache_dir, device)
    full_model = load_pinned_qwen_model(cache_dir=cache_dir, local_files_only=True)
    core_names = install_qwen_upper_lora(full_model)
    policy = B0ActionPolicy(
        QwenConditioningAdapter(
            full_model.model,
            config=QwenConditioningConfig(
                initialization_seed=recipe["projector_initialization_seed"],
            ),
            trainable_backbone_parameter_names=core_names,
        ),
        ActionQueryTransformer(config=ActionQueryTransformerConfig(
            initialization_seed=recipe["decoder_initialization_seed"],
        )),
        objective_config=B0ObjectiveConfig(recipe["objective"], recipe["gripper_bce_weight"]),
    ).to(device)
    return policy, tuple(f"conditioning_adapter.backbone.{name}" for name in core_names)


def make_loader(dataset: Any, plan: Any, args: Any, recipe: Any, *, start: int = 0) -> Any:
    batches = EpochBatches(dataset, plan, cache_dir=str(args.cache_dir),
                           maximum_cached_episode_frames=recipe["maximum_cached_episode_frames"],
                           policy_family=recipe.get("policy_family"))
    # Separate generator avoids modifying training RNG when workers are recreated on resume.
    generator = torch.Generator().manual_seed(recipe["seed"])
    return DataLoader(
        batches, batch_size=None, sampler=range(start, len(plan)),
        num_workers=recipe["workers"], prefetch_factor=recipe["prefetch_factor"],
        multiprocessing_context="spawn", pin_memory=True, generator=generator,
        timeout=300,
    )


def write_event(output: Path, event: dict[str, Any]) -> None:
    event = {"time_unix": time.time(), "pid": os.getpid(), **event}
    line = json.dumps(event, sort_keys=True, allow_nan=False)
    with (output / "events.jsonl").open("a") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def inspect_data(args: Any, recipe: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    revision, clean = project_state(PROJECT_ROOT)
    if not clean:
        raise TrainingContractError("training requires a clean, committed source checkout")
    if sha256_file(args.normalization) != recipe["normalization_sha256"]:
        raise TrainingContractError("train-only normalization differs from the recipe")
    index = load_frozen_b0_split_index(args.frozen_root)
    reader = DemonstrationReader(args.repository_root)
    normalizer = load_b0_action_normalizer(args.normalization)
    train = B0StreamingDataset(reader, index, normalizer, config=B0DataConfig(seed=recipe["seed"]))
    validation = B0StreamingDataset(reader, index, normalizer, config=B0DataConfig(
        split="validation", seed=recipe["seed"], shuffle_episodes=False,
    ))
    if train.sample_count != recipe["training_sample_count"]:
        raise TrainingContractError("full training sample count differs from recipe")
    protocol = load_evaluation_protocol()
    family = load_manifest_family(PROJECT_ROOT / "evaluation/splits/libero-plus-v1", protocol)
    if (protocol_sha256(protocol) != recipe["evaluation_protocol_sha256"]
            or family["dev-210"]["task_records_sha256"] != recipe["dev_task_records_sha256"]):
        raise TrainingContractError("evaluation protocol or dev task IDs changed")
    clean_protocol = load_evaluation_protocol(PROJECT_ROOT / "configs/evaluation/libero_clean_v1.json")
    clean_manifest = read_sealed(PROJECT_ROOT / "evaluation/splits/libero-clean-v1/clean-40.json")
    if (protocol_sha256(clean_protocol) != recipe["clean_protocol_sha256"]
            or clean_manifest["task_records_sha256"] != recipe["clean_task_records_sha256"]
            or clean_manifest["protocol_sha256"] != recipe["clean_protocol_sha256"]):
        raise TrainingContractError("clean evaluation task IDs or protocol changed")
    model = read_sealed(PROJECT_ROOT / "docs/model/Qwen3.5-2B-manifest.json")
    model_cache = (args.cache_dir / "models--Qwen--Qwen3.5-2B" / "snapshots" / QWEN_REVISION)
    # Verify downloaded checkpoint bytes before a new process can train or resume.
    for record in model["files"]:
        if record["path"] in {".gitattributes", "LICENSE", "README.md"}:
            continue  # Repository metadata is pinned by the manifest, not a runtime input.
        source = model_cache / record["path"]
        if not source.is_file() or source.stat().st_size != record["size"]:
            raise TrainingContractError(f"missing or size-mismatched pinned model file: {source}")
        if "lfs_sha256" in record and sha256_file(source) != record["lfs_sha256"]:
            raise TrainingContractError(f"pinned model SHA-256 mismatch: {source}")
        if "lfs_sha256" not in record:
            data = source.read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if blob != record["git_blob_id"]:
                raise TrainingContractError(f"pinned model Git blob mismatch: {source}")
    plans = [build_batch_plan(
        train.ordered_episodes(epoch), batch_size=recipe["batch_size"],
        seed=recipe["seed"], epoch=epoch, shuffle=True,
    ) for epoch in range(recipe["epochs"])]
    validation_plan = build_batch_plan(
        validation.ordered_episodes(), batch_size=recipe["batch_size"],
        seed=recipe["seed"], epoch=0, shuffle=False,
    )
    budget_identity = {}
    if recipe.get("policy_family") == "qwen35-oft-b0-v1":
        remaining = recipe["maximum_updates"]
        exposure = 0
        for plan in plans:
            selected = plan[:remaining]
            exposure += sum(len(anchors) for _, anchors in selected)
            remaining -= len(selected)
            if remaining == 0:
                break
        budget_identity = {"maximum_planned_updates": recipe["maximum_updates"],
                           "maximum_planned_sample_exposures": exposure,
                           "maximum_epoch_equivalent_exposure": exposure / train.sample_count}
    identity = seal_contract({
        "artifact_type": "eventground_wam.b0_full_training_manifest",
        "schema_version": 1,
        **budget_identity,
        "mode": "pilot" if args.pilot_steps else "production",
        "recipe": recipe,
        "project_git_revision": revision,
        "source": {**index.source_identity,
                   "normalization_file_sha256": sha256_file(args.normalization),
                   "normalization_artifact_digest": normalizer.artifact_digest},
        "data_config": asdict(train.config),
        "model_manifest_digest": model["digest"],
        "b0_runtime_lock_sha256": sha256_file(PROJECT_ROOT / "environments/b0/uv.lock"),
        "training_sample_count": train.sample_count,
        "validation_sample_count": validation.sample_count,
        "epoch_plan_digests": [plan_digest(plan) for plan in plans],
        "steps_per_epoch": len(plans[0]),
        "validation_plan_digest": plan_digest(validation_plan),
        "validation_batch_count": len(validation_plan),
        "initialization": ("pinned-qwen-with-fresh-oft-head-state-projector-and-broad-rank32-lora"
                           if recipe.get("policy_family") == "qwen35-oft-b0-v1" else
                           "pinned-qwen-with-fresh-projectors-decoder-and-rank16-lora"),
    })
    # Avoid keeping all five Python batch plans in memory during training.
    return train, validation, (plans[0], validation_plan), identity


def export_policy(output: Path, engine: TrainingEngine, manifest: Any, validation: Any) -> Any:
    policy_path = output / "policies" / f"epoch-{engine.epoch + 1:02d}.pt"
    digest = atomic_torch_save(policy_path, {
        "format": POLICY_FORMAT, "identity_digest": manifest["digest"],
        "qwen_revision": QWEN_REVISION,
        "state_dict": b0_trainable_state_dict(engine.policy,
                                              allowed_exact_names=engine.exact_lora_names),
    })
    descriptor = {"epoch": engine.epoch + 1, "path": str(policy_path.relative_to(output)),
                  "sha256": digest, "validation": validation}
    atomic_json(policy_path.with_suffix(".json"), seal_contract({
        "artifact_type": "eventground_wam.b0_epoch_policy",
        "identity_digest": manifest["digest"], **descriptor,
    }))
    return descriptor


def verify_reload(engine: TrainingEngine, store: CheckpointStore, batch: Any) -> dict[str, Any]:
    """Prove disk reload changes perturbed predictions back to the exact saved result."""
    engine.policy.eval()
    inputs, state, _ = move_batch(batch, engine.device)
    with torch.no_grad():
        expected = engine.policy(inputs, state).normalized_actions.detach().float().cpu()
    store.save(engine.snapshot())
    with torch.no_grad():
        for parameter in engine.parameters:
            parameter.zero_()
        changed = engine.policy(inputs, state).normalized_actions.detach().float().cpu()
    if float((expected - changed).abs().max()) <= 1e-4:
        raise TrainingContractError("reload probe perturbation did not change predictions")
    engine.restore(store.load())
    engine.policy.eval()
    with torch.no_grad():
        restored = engine.policy(inputs, state).normalized_actions.detach().float().cpu()
    maximum_error = float((expected - restored).abs().max())
    if maximum_error > 1e-3:
        raise TrainingContractError("reload changed policy predictions beyond tolerance")
    # Exercise a real next optimizer update on both sides of the disk restore.
    probe = {**batch, "index": engine.next_batch}
    engine.step(probe)
    expected_state = {name: tensor.cpu() for name, tensor in b0_trainable_state_dict(
        engine.policy, allowed_exact_names=engine.exact_lora_names,
    ).items()}
    engine.restore(store.load())
    engine.step(probe)
    actual_state = b0_trainable_state_dict(engine.policy,
                                          allowed_exact_names=engine.exact_lora_names)
    update_error = max(float((expected_state[name] - tensor.cpu()).abs().max())
                       for name, tensor in actual_state.items())
    engine.restore(store.load())
    if update_error > 1e-6:
        raise TrainingContractError("a resumed optimizer update differs from uninterrupted execution")
    return {"reload_max_absolute_error": maximum_error,
            "resumed_update_max_absolute_error": update_error,
            "restored_optimizer_and_scheduler": True, "restored_rng_and_cursor": True}


def run(args: Any) -> int:
    recipe = load_recipe(args.recipe)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root
    with (output / "job.lock").open("a") as job_lock:
        try:
            fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TrainingContractError("another process already owns this training output") from error
        return run_locked(args, recipe, output)


def run_locked(args: Any, recipe: Any, output: Path) -> int:
    configure_deterministic_cuda()
    if not torch.cuda.is_available():
        raise TrainingContractError("full B0 training requires an available CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    seed_training(recipe["seed"])
    train, validation, plans, identity = inspect_data(args, recipe)
    manifest_path = output / "training-manifest.json"
    if manifest_path.exists():
        if read_sealed(manifest_path)["digest"] != identity["digest"]:
            raise TrainingContractError("output identity differs; use a new output directory")
    else:
        atomic_json(manifest_path, identity)
    if args.preflight_only:
        print(json.dumps(identity, indent=2), flush=True)
        return 0
    if (output / "training-report.json").exists():
        report = read_sealed(output / "training-report.json")
        if report["identity_digest"] != identity["digest"]:
            raise TrainingContractError("completed report identity mismatch")
        if sha256_file(output / "B0.ckpt") != report["checkpoint_sha256"]:
            raise TrainingContractError("completed checkpoint hash mismatch")
        print("Five-epoch training is already complete.", flush=True)
        return 0
    if not args.pilot_steps:
        pilot = read_sealed(args.pilot_report)
        if (pilot["status"] != "passed" or pilot["recipe"] != recipe
                or pilot["project_git_revision"] != identity["project_git_revision"]
                or pilot["b0_runtime_lock_sha256"] != identity["b0_runtime_lock_sha256"]
                or pilot["projected_training_hours"] > recipe["maximum_projected_training_hours"]):
            raise TrainingContractError("production requires a matching passing pilot within 120h")
    policy, exact_names = build_policy(recipe, str(args.cache_dir), device)
    engine = TrainingEngine(policy, exact_names, recipe, identity_digest=identity["digest"],
                            steps_per_epoch=identity["steps_per_epoch"], device=device)
    store = CheckpointStore(output / "recovery", identity["digest"],
                            recipe["checkpoint_keep_last"])
    saved = store.load()
    if saved is not None:
        engine.restore(saved)
    else:
        store.save(engine.snapshot())
    write_event(output, {"event": "started", "host": socket.gethostname(),
                         "global_step": engine.global_step, "epoch": engine.epoch,
                         "next_batch": engine.next_batch, "recovery_notes": store.recovery_notes})
    torch.cuda.reset_peak_memory_stats(device)
    pilot_started = time.monotonic()
    pilot_start_samples = engine.samples_seen
    pilot_start_step = engine.global_step
    measured_seconds = measured_samples = 0.0
    last_save = last_log = time.monotonic()
    first_plan, validation_plan = plans
    while engine.epoch < recipe["epochs"]:
        plan = first_plan if engine.epoch == 0 else build_batch_plan(
            train.ordered_episodes(engine.epoch), batch_size=recipe["batch_size"],
            seed=recipe["seed"], epoch=engine.epoch, shuffle=True,
        )
        if plan_digest(plan) != identity["epoch_plan_digests"][engine.epoch]:
            raise TrainingContractError("regenerated epoch sample order changed")
        loader = make_loader(train, plan, args, recipe, start=engine.next_batch)
        previous = time.monotonic()
        for batch in loader:
            record = engine.step(batch)
            now = time.monotonic()
            # Include input wait in timing; omit worker/model warmup from the steady sample.
            if engine.global_step - pilot_start_step > 20:
                measured_seconds += now - previous
                measured_samples += batch["sample_count"]
            previous = now
            if (engine.global_step % recipe["log_interval_steps"] == 0
                    or now - last_log >= 60):
                write_event(output, {"event": "progress", **record,
                                     "steady_samples_per_second":
                                     measured_samples / max(measured_seconds, 1e-9),
                                     "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated()})
                atomic_json(output / "status.json", {"status": "running", **record,
                                                       "updated_unix": time.time()})
                last_log = now
            if (now - last_save >= recipe["checkpoint_interval_seconds"]
                    or engine.global_step % recipe["checkpoint_interval_steps"] == 0):
                store.save(engine.snapshot())
                last_save = time.monotonic()
            if STOP_REQUESTED:
                store.save(engine.snapshot())
                atomic_json(output / "status.json", {"status": "interrupted", **record})
                return 75
            if args.pilot_steps and engine.global_step - pilot_start_step >= args.pilot_steps:
                elapsed = time.monotonic() - pilot_started
                throughput = measured_samples / measured_seconds
                # Conservatively charge validation at training throughput plus a 20% margin.
                projected = (recipe["epochs"] * (train.sample_count + validation.sample_count)
                             / throughput / 3600 * recipe["pilot_projection_margin"])
                reload_result = verify_reload(engine, store, batch)
                pilot = seal_contract({
                    "artifact_type": "eventground_wam.b0_training_pilot",
                    "status": "passed" if projected <= recipe["maximum_projected_training_hours"]
                    else "over-time-budget",
                    "recipe": recipe, "project_git_revision": identity["project_git_revision"],
                    "b0_runtime_lock_sha256": identity["b0_runtime_lock_sha256"],
                    "identity_digest": identity["digest"],
                    "samples_per_second": throughput, "projected_training_hours": projected,
                    "updates": engine.global_step - pilot_start_step,
                    "sample_exposures": engine.samples_seen - pilot_start_samples,
                    "elapsed_seconds_including_startup": elapsed,
                    "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
                    "device": torch.cuda.get_device_name(), "reload_validation": reload_result,
                })
                atomic_json(output / "pilot-report.json", pilot)
                write_event(output, {"event": "pilot_complete", **pilot})
                return 0 if pilot["status"] == "passed" else 78
        # Save before potentially long validation; a restart repeats only validation.
        store.save(engine.snapshot())
        write_event(output, {"event": "validation_started", "epoch": engine.epoch + 1})
        metrics = evaluate_validation(policy, make_loader(validation, validation_plan, args, recipe),
                                      device)
        if metrics["sample_count"] != validation.sample_count:
            raise TrainingContractError("validation did not cover every frozen chunk")
        candidate = export_policy(output, engine, identity, metrics)
        if engine.best is None or metrics["objective"] < engine.best["validation"]["objective"]:
            engine.best = candidate
        engine.finish_epoch(metrics)
        store.save(engine.snapshot())
        write_event(output, {"event": "epoch_complete", **engine.history[-1]})
        if STOP_REQUESTED:
            return 75
    if engine.samples_seen != recipe["epochs"] * train.sample_count or engine.best is None:
        raise TrainingContractError("completed run has wrong sample exposure or no selected policy")
    best_path = output / engine.best["path"]
    if sha256_file(best_path) != engine.best["sha256"]:
        raise TrainingContractError("selected epoch policy hash mismatch")
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    load_b0_trainable_state_dict(policy, selected["state_dict"], allowed_exact_names=exact_names)
    # Validate the selected checkpoint's predictions after disk reload over the full validation set.
    final_validation = evaluate_validation(
        policy, make_loader(validation, validation_plan, args, recipe), device,
    )
    if abs(final_validation["objective"] - engine.best["validation"]["objective"]) > 1e-6:
        raise TrainingContractError("selected checkpoint reload changed validation objective")
    checkpoint_sha256 = atomic_torch_save(output / "B0.ckpt", selected)
    report = seal_contract({
        "artifact_type": "eventground_wam.b0_full_training_report", "status": "complete",
        "identity_digest": identity["digest"], "epochs_completed": engine.epoch,
        "global_step": engine.global_step, "samples_seen": engine.samples_seen,
        "sample_chain": engine.sample_chain, "history": engine.history,
        "selected_epoch": engine.best["epoch"], "checkpoint_sha256": checkpoint_sha256,
        "final_reload_validation": final_validation, "reload_validation_passed": True,
        "runtime": {key: importlib.metadata.version(key)
                    for key in ("torch", "transformers", "peft", "numpy", "av", "pyarrow")},
    })
    atomic_json(output / "training-report.json", report)
    atomic_json(output / "status.json", {"status": "complete", "global_step": engine.global_step})
    write_event(output, {"event": "complete", "checkpoint_sha256": checkpoint_sha256})
    return 0


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    for name in ("repository-root", "frozen-root", "normalization", "cache-dir", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.pilot_steps and args.pilot_steps < 100:
        parser.error("pilot requires at least 100 updates")
    if not args.pilot_steps and not args.pilot_report and not args.preflight_only:
        parser.error("production requires --pilot-report")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        sys.exit(run(parse_args()))
    except TrainingContractError as error:
        print(f"NONRETRYABLE: {error}", file=sys.stderr, flush=True)
        sys.exit(78)
