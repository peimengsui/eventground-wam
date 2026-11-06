#!/usr/bin/env python3
"""Launch the fixed all-L1 B continuation, after a matching host recovery pilot."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import signal
import sys
import time
from pathlib import Path

import torch
import train_b0 as common

from eventground_wam.b0_continuation import (
    EXPERIMENT,
    OBJECTIVE,
    ActiveBudget,
    action_metrics,
    adopt_parent,
    continuation_plans,
    l1_gradient_panel,
    mixed_loader,
    validation_metrics,
)
from eventground_wam.b0_diagnostics import tensor_tree_digest
from eventground_wam.b0_model import QWEN_REVISION, b0_trainable_state_dict
from eventground_wam.b0_training import (
    POLICY_FORMAT,
    CheckpointStore,
    TrainingContractError,
    TrainingEngine,
    atomic_json,
    atomic_torch_save,
    configure_deterministic_cuda,
    load_recipe,
    move_batch,
    read_sealed,
    sha256_file,
)
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/week4_b0_oft_all_l1_v2.json"


def experiment():
    config = read_sealed(CONFIG)
    if (config["experiment_id"] != EXPERIMENT or config["objective"] != OBJECTIVE
            or config["launch_arm"] != "B" or config["additional_updates"] != 15000):
        raise TrainingContractError("unexpected continuation specification")
    for pin in config["sources"].values():
        if sha256_file(ROOT / pin["path"]) != pin["sha256"]:
            raise TrainingContractError(f"frozen dependency changed: {pin['path']}")
    return config


def prepare(args, config):
    base = load_recipe(ROOT / config["sources"]["parent_recipe"]["path"])
    train, validation, _, source = common.inspect_data(args, base)
    parent_root = args.data_root / "derived/b0/week4-b0-oft-v1"
    parent_manifest = read_sealed(parent_root / "training-manifest.json")
    audit = read_sealed(ROOT / config["sources"]["parent_recovery_audit"]["path"])
    if parent_manifest["digest"] != audit["training_manifest_digest"]:
        raise TrainingContractError("parent manifest changed")
    recovery = Path(config["initialization"]["recovery_path"])
    if (sha256_file(recovery) != config["initialization"]["recovery_sha256"]
            or sha256_file(parent_root / "B0.ckpt") != config["initialization"]["checkpoint_sha256"]):
        raise TrainingContractError("parent checkpoint bytes changed")
    blocked, plan = continuation_plans(train, parent_manifest, audit)
    del blocked
    recipe = {**base, "recipe_id": EXPERIMENT + "-B", "objective": OBJECTIVE,
              "epochs": 1, "maximum_updates": 15000, "warmup_fraction": 0.,
              "minimum_learning_rate_ratio": 1., "decoder_projector_learning_rate": 1e-5,
              "lora_learning_rate": 1e-5, "maximum_cached_episode_frames": 8192,
              "checkpoint_selection": "fixed-additional-update-15000",
              "sample_order": "continuation-v1-mixed-128-update-windows",
              "validation": "frozen-validation-2560-at-parent-and-fixed-endpoint"}
    identity = seal_contract({
        "artifact_type": "eventground_wam.b0_all_l1_continuation_manifest",
        "experiment_digest": config["digest"], "arm": "B", "mode": args.phase,
        "recipe": recipe, "project_git_revision": source["project_git_revision"],
        "source": source["source"], "model_manifest_digest": source["model_manifest_digest"],
        "b0_runtime_lock_sha256": source["b0_runtime_lock_sha256"],
        "parent_manifest_digest": parent_manifest["digest"],
        "initialization": config["initialization"], "plan_digest": batches_digest(plan),
        "sample_count": sum(map(len, plan)), "additional_updates": len(plan),
    })
    return train, validation, plan, identity, recovery


def check_space(args, config):
    if shutil.disk_usage(args.work_root).free < config["budget"]["minimum_free_bytes"]:
        raise TrainingContractError("insufficient free disk for recovery")
    used = sum(p.stat().st_size for p in args.work_root.rglob("*") if p.is_file())
    if used > config["budget"]["output_limit_bytes"]:
        raise TrainingContractError("continuation output cap reached")


def export_policy(output, engine, names):
    return atomic_torch_save(output / "B0.ckpt", {
        "format": POLICY_FORMAT, "identity_digest": engine.identity_digest,
        "qwen_revision": QWEN_REVISION, "objective": OBJECTIVE,
        "policy_family": engine.recipe["policy_family"],
        "state_dict": b0_trainable_state_dict(engine.policy, allowed_exact_names=names),
    })


def exact_replay(engine, store, next_batch):
    """Two copies of the actual next mixed update must agree in their complete state."""
    store.save(engine.snapshot())
    saved = store.load()
    before = tensor_tree_digest(saved)
    inputs, state, _ = move_batch(next_batch, engine.device)
    engine.policy.eval()
    with torch.no_grad():
        expected = engine.policy(inputs, state).raw_action_predictions.float().cpu()
    engine.restore(saved)
    engine.policy.eval()
    with torch.no_grad():
        actual = engine.policy(inputs, state).raw_action_predictions.float().cpu()
    if not torch.equal(expected, actual):
        raise TrainingContractError("raw action reload is not exact")
    engine.step(next_batch)
    first = tensor_tree_digest(engine.snapshot())
    engine.restore(saved)
    engine.step(next_batch)
    second = tensor_tree_digest(engine.snapshot())
    engine.restore(saved)
    if first != second or tensor_tree_digest(saved) != before:
        raise TrainingContractError("resumed next update differs or mutates saved parent snapshot")
    return {"raw_predictions_bitwise_equal": True, "next_update_complete_state_bitwise_equal": True,
            "state_digest_after_probe_update": first, "probe_token_digest": next_batch["token_digest"],
            "probe_optimizer_updates_executed": 2, "probe_updates_rolled_back": True}


def measurement(args, config, engine, validation, output, name):
    panel = read_sealed(ROOT / config["sources"]["validation_panel"]["path"])
    plan = [(r["episode_index"], tuple(r["anchors"])) for r in panel["episodes"]]
    before = tensor_tree_digest(engine.snapshot())
    metrics = validation_metrics(engine.policy, common.make_loader(
        validation, plan, args, engine.recipe), engine.device)
    if metrics["sample_count"] != 2560 or before != tensor_tree_digest(engine.snapshot()):
        raise TrainingContractError("measurement altered training state or panel coverage")
    result = seal_contract({"artifact_type": "eventground_wam.all_l1_validation_panel",
                            "identity_digest": engine.identity_digest,
                            "additional_updates": engine.global_step,
                            "panel_digest": panel["digest"], "metrics": metrics,
                            "state_preserved": True, "scope": "seen-trajectory descriptive validation"})
    atomic_json(output / f"validation-{name}.json", result)
    common.write_event(output, {"event": "validation_complete", "name": name, **metrics})
    return result


def gradients(args, config, engine, validation, output, name):
    path = output / f"gradients-{name}.json"
    if path.exists():
        return read_sealed(path)
    panel = read_sealed(ROOT / config["sources"]["gradient_panel"]["path"])
    plan = [(r["episode_index"], tuple(r["anchors"])) for r in panel["episodes"]]
    before = tensor_tree_digest(engine.snapshot())
    result = l1_gradient_panel(engine.policy, common.make_loader(
        validation, plan, args, engine.recipe), engine.device)
    if before != tensor_tree_digest(engine.snapshot()) or result["batch_count"] != 40:
        raise TrainingContractError("gradient measurement changed training state or panel")
    result = seal_contract({"artifact_type": "eventground_wam.all_l1_gradient_panel",
                            "identity_digest": engine.identity_digest, "panel_digest": panel["digest"],
                            "additional_updates": engine.global_step, **result})
    atomic_json(path, result)
    common.write_event(output, {"event": "gradient_panel_complete", "name": name,
                                "backbone_gripper_motion_ratio": result[
                                    "backbone_median_gripper_motion_ratio"]})
    return result


def run(args, config, budget):
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise TrainingContractError("CUDA unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    train, validation, plan, identity, parent_path = prepare(args, config)
    output = args.work_root / ("pilot" if args.phase == "pilot" else "training")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "training-manifest.json"
    if manifest_path.exists() and read_sealed(manifest_path) != identity:
        raise TrainingContractError("continuation source/runtime/data identity changed")
    atomic_json(manifest_path, identity)
    if args.preflight_only:
        common.write_event(output, {"event": "preflight_complete", "manifest_digest": identity["digest"]})
        return 0
    if (output / "training-report.json").exists():
        report = read_sealed(output / "training-report.json")
        if (report["identity_digest"] != identity["digest"] or report["status"] != "complete"
                or sha256_file(output / "B0.ckpt") != report["checkpoint_sha256"]):
            raise TrainingContractError("completed continuation bundle changed")
        return 0
    if args.phase == "train":
        pilot = read_sealed(args.work_root / "pilot/pilot-report.json")
        smoke = read_sealed(args.work_root / "pilot/rollouts/evaluation-report.json")
        if (pilot["status"] != "passed" or pilot["experiment_digest"] != config["digest"]
                or pilot["project_git_revision"] != identity["project_git_revision"]
                or pilot["runtime_lock_sha256"] != identity["b0_runtime_lock_sha256"]
                or pilot["process_resume_observed"] is not True
                or smoke["all_pairs_valid"] is not True or smoke["completed"] != 2
                or smoke["checkpoint_sha256"] != pilot["checkpoint_sha256"]):
            raise TrainingContractError("production requires matching pilot and observed process resume")
    policy, names = common.build_policy(identity["recipe"], str(args.cache_dir), device)
    engine = TrainingEngine(policy, names, identity["recipe"], identity_digest=identity["digest"],
                            steps_per_epoch=15000, device=device)
    store = CheckpointStore(output / "recovery", identity["digest"], keep=3)
    recovered = store.load()
    if recovered is None:
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        if parent["identity_digest"] != identity["parent_manifest_digest"]:
            raise TrainingContractError("parent recovery envelope changed")
        adoption = adopt_parent(engine, parent)
        del parent
        atomic_json(output / "parent-adoption.json", seal_contract(adoption))
    else:
        engine.restore(recovered)
        del recovered
        atomic_json(output / "process-resume.json", seal_contract({
            "identity_digest": identity["digest"], "additional_step": engine.global_step,
            "next_batch": engine.next_batch, "samples_seen": engine.samples_seen,
            "complete_state_digest": tensor_tree_digest(engine.snapshot()),
            "recovery_notes": store.recovery_notes,
        }))
    store.save(engine.snapshot())
    common.write_event(output, {"event": "started", "phase": args.phase,
                                "additional_updates": engine.global_step,
                                "resumed": (output / "process-resume.json").exists(),
                                "objective": OBJECTIVE, "parent_updates": 72000,
                                "budget_active_seconds": budget.seconds})
    if args.phase == "train" and not (output / "validation-parent.json").exists():
        if engine.global_step != 0:
            raise TrainingContractError("missing unchanged-parent validation")
        measurement(args, config, engine, validation, output, "parent")
    if args.phase == "train" and not (output / "gradients-parent.json").exists():
        if engine.global_step != 0:
            raise TrainingContractError("missing unchanged-parent gradient panel")
        gradients(args, config, engine, validation, output, "parent")
    target = config["pilot_updates"] if args.phase == "pilot" else 15000
    start_step = engine.global_step
    loader = mixed_loader(train, plan, args.cache_dir, identity["recipe"], start=engine.next_batch)
    iterator = iter(loader)
    last_save = last_log = previous = time.monotonic()
    profile_path = output / "profiling.json"
    profile = read_sealed(profile_path) if profile_path.exists() else {}
    measured_time = profile.get("measured_seconds", 0.)
    measured_samples = profile.get("measured_samples", 0.)
    measured_updates = profile.get("measured_updates", 0.)

    def save_profile():
        atomic_json(profile_path, seal_contract({"measured_seconds": measured_time,
                                                 "measured_samples": measured_samples,
                                                 "measured_updates": measured_updates}))
    torch.cuda.reset_peak_memory_stats()
    last_batch = None
    while engine.global_step < target:
        if budget.seconds >= config["budget"]["training_and_pilot_seconds"]:
            store.save(engine.snapshot())
            raise TrainingContractError("training active budget reached before fixed endpoint")
        b = next(iterator)
        last_batch = b
        record = engine.step(b)
        now = time.monotonic()
        if engine.global_step - start_step > 20:
            measured_time += now - previous
            measured_samples += b["sample_count"]
            measured_updates += 1
        previous = now
        if engine.global_step % 100 == 0 or now - last_log >= 60:
            event = {"event": "progress", **record, "additional_updates": engine.global_step,
                     "cumulative_updates": 72000 + engine.global_step,
                     "cumulative_samples_seen": 2096581 + engine.samples_seen,
                     "unique_task_count": b["unique_task_count"], "batch_token_digest": b["token_digest"],
                     "budget_active_seconds": budget.seconds,
                     "steady_samples_per_second": measured_samples / max(measured_time, 1e-9),
                     "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
            common.write_event(output, event)
            atomic_json(output / "status.json", {"status": "running", **event})
            check_space(args, config)
            save_profile()
            last_log = now
        if (engine.global_step == 1 or now - last_save >= 300
                or engine.global_step % 2000 == 0 or common.STOP_REQUESTED):
            store.save(engine.snapshot())
            last_save = time.monotonic()
        if common.STOP_REQUESTED:
            save_profile()
            common.write_event(output, {"event": "interrupted_checkpoint_saved",
                                        "additional_updates": engine.global_step})
            return 75
    store.save(engine.snapshot())
    save_profile()
    if args.phase == "pilot":
        probe = next(iterator)
        replay = exact_replay(engine, store, probe)
        gradient_probe = l1_gradient_panel(policy, [probe], device)
        inputs, state, targets = move_batch(probe, device)
        with torch.no_grad():
            metrics = action_metrics(policy(inputs, state, normalized_action_targets=targets), targets)
        projected = measured_time / measured_updates * 15000 * 1.25
        passed = (projected + budget.seconds + 1800 < config["budget"]["training_and_pilot_seconds"]
                  and (output / "process-resume.json").exists())
        report = seal_contract({
            "artifact_type": "eventground_wam.all_l1_mixed_pilot", "status": "passed" if passed else "failed",
            "experiment_digest": config["digest"], "project_git_revision": identity["project_git_revision"],
            "runtime_lock_sha256": identity["b0_runtime_lock_sha256"],
            "identity_digest": identity["digest"], "updates": engine.global_step,
            "process_resume_observed": (output / "process-resume.json").exists(),
            "reload_validation": replay, "metrics": metrics,
            "gradient_probe": gradient_probe,
            "measured_updates": measured_updates,
            "samples_per_second": measured_samples / measured_time,
            "seconds_per_update": measured_time / measured_updates,
            "projected_training_seconds_with_25_percent_margin": projected,
            "budget_active_seconds": budget.seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "checkpoint_sha256": export_policy(output, engine, names),
        })
        atomic_json(output / "pilot-report.json", report)
        common.write_event(output, {"event": "pilot_complete", **report})
        return 0 if passed else 78
    del iterator, loader
    if engine.samples_seen != 435945:
        raise TrainingContractError("fixed endpoint has incorrect exposure")
    endpoint = measurement(args, config, engine, validation, output, "endpoint")
    gradients(args, config, engine, validation, output, "endpoint")
    checkpoint = {"format": POLICY_FORMAT, "identity_digest": identity["digest"],
                  "qwen_revision": QWEN_REVISION, "objective": OBJECTIVE,
                  "policy_family": identity["recipe"]["policy_family"],
                  "state_dict": b0_trainable_state_dict(policy, allowed_exact_names=names)}
    digest = atomic_torch_save(output / "B0.ckpt", checkpoint)
    loaded = torch.load(output / "B0.ckpt", map_location="cpu", weights_only=True)
    if tensor_tree_digest(loaded["state_dict"]) != tensor_tree_digest(checkpoint["state_dict"]):
        raise TrainingContractError("endpoint checkpoint reload changed tensors")
    from eventground_wam.b0_model import load_b0_trainable_state_dict

    load_b0_trainable_state_dict(policy, loaded["state_dict"], allowed_exact_names=names)
    repeated = measurement(args, config, engine, validation, output, "reload")
    if endpoint["metrics"] != repeated["metrics"]:
        raise TrainingContractError("endpoint reload validation changed")
    report = seal_contract({
        "artifact_type": "eventground_wam.all_l1_continuation_report", "status": "complete",
        "identity_digest": identity["digest"], "experiment_digest": config["digest"],
        "additional_updates": engine.global_step, "additional_samples": engine.samples_seen,
        "cumulative_updates": 87000, "cumulative_samples": 2532526,
        "selected_additional_step": 15000, "checkpoint_sha256": digest,
        "validation": endpoint, "reload_validation_passed": True,
        "budget_active_seconds": budget.seconds, "rollouts": "pending development-200",
    })
    store.save(engine.snapshot())
    atomic_json(output / "training-report.json", report)
    atomic_json(output / "status.json", {"status": "training-complete", **report})
    common.write_event(output, {"event": "training_complete", **report})
    del last_batch
    return 0


def parse_args():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "train"), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--data-root", type=Path, default=home / "data/eventground-wam/libero-plus")
    parser.add_argument("--cache-dir", type=Path, default=home / "data/eventground-wam/huggingface-cache")
    args = parser.parse_args()
    args.pilot_steps = 128 if args.phase == "pilot" else 0
    args.frozen_root = args.data_root / "derived/frozen/week2-dataset-v1"
    args.normalization = args.data_root / "derived/b0/week3-b0-data-v1/action-statistics.json"
    args.repository_root = args.data_root / (
        "datasets/libero-plus-lerobot/snapshots/22c57433fef692b5b9ecc0795344daac7fa867a5/repo")
    return args


if __name__ == "__main__":
    args = parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, common.request_stop)
    signal.signal(signal.SIGINT, common.request_stop)
    try:
        with (args.work_root / "owner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = experiment()
            budget = ActiveBudget(args.work_root / "budget.json", config["digest"])
            try:
                code = run(args, config, budget)
            finally:
                budget.close()
        raise SystemExit(code)
    except (TrainingContractError, BlockingIOError, ValueError) as error:
        print(json.dumps({"integrity_stop": str(error)}), file=sys.stderr, flush=True)
        raise SystemExit(78) from error
