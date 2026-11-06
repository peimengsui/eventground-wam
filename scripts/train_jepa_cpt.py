#!/usr/bin/env python3
"""Pinned JEPA CPT runner; pilot and production have separate recovery identities."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import inspect
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from check_pixel_predictor import verify_base_files

from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.b0_training import (
    atomic_json,
    atomic_torch_save,
    configure_deterministic_cuda,
    seed_training,
)
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import read_sealed, sha256_file
from eventground_wam.jepa_data import JepaPairDataset
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_predictor import JepaPredictor
from eventground_wam.jepa_training import (
    JepaTrainingEngine,
    PoolTargetCache,
    encode_training_batch,
    verify_recipe,
)
from eventground_wam.pixel_data import FutureInputBuilder
from eventground_wam.pixel_predictor import FrozenQwenFeatures, tensor_state_digest
from eventground_wam.pixel_training import (
    CPTOrder,
    PixelCheckpointStore,
    append_event,
    pilot_indices,
    tree_digest,
)

ROOT = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def code_pins():
    files = sorted(ROOT.glob("src/eventground_wam/*.py"))
    files += [ROOT/"scripts"/name for name in ("train_jepa_cpt.py", "check_jepa_training.py", "check_pixel_training.py",
                                               "prepare_cpt_order.py", "check_pixel_predictor.py")]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in files}


def configure():
    configure_deterministic_cuda()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("the frozen recipe requires one CUDA device")
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "transformers": importlib.metadata.version("transformers"),
            "av": importlib.metadata.version("av"), "pyarrow": importlib.metadata.version("pyarrow"),
            "pillow": importlib.metadata.version("pillow"), "cuda": str(torch.version.cuda),
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                                "--format=csv,noheader"], text=True).strip(),
            "gpu": torch.cuda.get_device_name(0), "torch_threads": 2,
            "deterministic_algorithms": True, "tf32": False, "mha_fastpath": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def run(args):
    started = time.monotonic()
    runtime = configure()
    if runtime["torch"] != "2.13.0+cu130" or runtime["transformers"] != "5.17.0":
        raise ValueError("locked JEPA teacher runtime required")
    recipe = verify_recipe(args.recipe, ROOT)
    if shutil.disk_usage(args.output).free < recipe["resource_limits"]["minimum_free_disk_bytes"]:
        raise ValueError("insufficient free disk reserve for durable recovery")
    source = code_pins()
    order = CPTOrder(args.order, recipe["order_manifest_sha256"])
    if (len(order) != recipe["total_updates"] or order.manifest["pair_presentations"] != recipe["pair_presentations"]
            or order.manifest["pair_manifest_sha256"] != recipe["pair_manifest_sha256"]):
        raise ValueError("frozen CPT exposure changed")
    dataset = JepaPairDataset(args.pairs, args.data_root/
        f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo", split="train",
        expected_manifest_sha256=recipe["pair_manifest_sha256"], cache_episodes=4,
        normalizer=load_b0_action_normalizer(ROOT/"docs/model/week4-fresh-v2/action-statistics.json"),
        state_bounds=read_sealed(ROOT/"configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"])
    indices = pilot_indices(order, dataset.sources) if args.phase == "pilot" else list(range(len(order)))
    identity = seal_contract({"artifact_type": "eventground_wam.jepa_cpt_training_identity",
                              "phase": args.phase, "recipe_digest": recipe["digest"],
                              "order_digest": order.manifest["digest"], "indices_digest": tree_digest(indices),
                              "updates": len(indices), "source_sha256": source, "runtime": runtime})
    if args.phase == "train":
        if args.pilot_report is None:
            raise ValueError("production requires the passed matching Step 4 pilot report")
        pilot = read_sealed(args.pilot_report)
        if (pilot["status"] != "passed" or pilot["recipe_digest"] != recipe["digest"]
                or pilot["order_digest"] != order.manifest["digest"] or pilot["source_sha256"] != source
                or pilot["runtime"] != runtime):
            raise ValueError("production pilot gate does not match this exact recipe/code/runtime")
        if args.pause_after_microbatch_at is not None:
            raise ValueError("fault injection is forbidden in production")
    manifest_path = args.output/"identity.json"
    existed = manifest_path.exists()
    if existed and read_sealed(manifest_path) != identity:
        raise ValueError("refusing changed recipe, phase, source, runtime or sample order on resume")
    atomic_json(manifest_path, identity)
    budget = ActiveBudget(args.output/"active-budget.json", identity["digest"])
    try:
        return train(args, recipe, runtime, identity, dataset, order, indices, budget, started)
    finally:
        budget.close()


def train(args, recipe, runtime, identity, dataset, order, indices, budget, started):
    maximum_seconds = recipe["maximum_active_training_hours"]*3600 if args.phase == "train" else 3600
    if budget.seconds >= maximum_seconds:
        raise ValueError("active runtime budget exhausted; no incomplete checkpoint is accepted")
    seed_training(recipe["seed"])
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    verified = verify_base_files(args.cache, read_sealed(ROOT/"docs/model/Qwen3.5-2B-manifest.json"))
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).to(device)
    encoder = FrozenQwenFeatures(full.model).to(device)
    processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True)
    builder = FutureInputBuilder(processor)
    teacher = FrozenQwenVisualTargets(full.model.visual, processor.image_processor)
    accepted = read_sealed(ROOT/"docs/evaluation/week6-jepa-targets-v2/report.json")
    teacher_before = tensor_state_digest(teacher.visual.state_dict())
    if teacher_before != accepted["vision_state_after"]:
        raise ValueError("teacher differs from the accepted target contract")
    for name, obj in (("vision", type(teacher.visual)), ("processor", type(teacher.image_processor))):
        if sha256_file(Path(inspect.getfile(obj))) != accepted["implementation_files"][name]["sha256"]:
            raise ValueError("teacher implementation changed")
    target_cache = PoolTargetCache(teacher, maximum_bytes=recipe["target_cache"]["maximum_bytes"])
    base_before = tensor_state_digest(encoder.core.state_dict())
    model = JepaPredictor().to(device)
    if model.parameter_budget() != recipe["parameter_budget"]:
        raise ValueError("trainable parameter budget changed")
    engine = JepaTrainingEngine(model, recipe, identity_digest=identity["digest"], indices=indices, order=order)
    store = PixelCheckpointStore(args.output/"recovery", identity["digest"], keep=recipe["checkpoint_keep_last"])
    def save_checkpoint():
        if shutil.disk_usage(args.output).free < recipe["resource_limits"]["minimum_free_disk_bytes"]:
            raise ValueError("disk reserve exhausted; previous committed checkpoint retained")
        return store.save(engine.snapshot())
    saved = store.load()
    if saved is not None:
        # Deliberately disturb startup RNGs; restoring the checkpoint must repair all of them.
        seed_training(recipe["seed"]+991)
        engine.restore(saved)
        atomic_json(args.output/"process-resume.json", seal_contract({
            "identity_digest": identity["digest"], "pid": os.getpid(),
            "global_step": engine.global_step, "pairs_seen": engine.pairs_seen,
            "snapshot_digest": tree_digest(engine.snapshot()), "recovery_notes": store.notes}))
    else:
        atomic_json(args.output/"initialization.json", seal_contract({
            "identity_digest": identity["digest"], "model_state_digest": tensor_state_digest(model.state_dict()),
            "base_state_digest": base_before, "teacher_state_digest": teacher_before, "trainable_base_parameters": 0,
            "parent_or_diagnostic_weights_loaded": False, "verified_base_files": verified,
            "optimizer_state_entries": len(engine.optimizer.state)}))
        save_checkpoint()
    resume_step = engine.global_step
    last_save = time.monotonic()
    history = []
    checkpoint_seconds = 0.
    interval = recipe["pilot"]["checkpoint_interval_updates"] if args.phase == "pilot" else recipe["checkpoint_interval_updates"]
    atomic_json(args.output/"status.json", {"status": "running", "phase": args.phase,
                                           "global_step": engine.global_step, "pid": os.getpid()})
    while engine.global_step < len(indices):
        if STOP_REQUESTED or budget.seconds >= maximum_seconds:
            save_checkpoint()
            atomic_json(args.output/"status.json", {"status": "interrupted" if STOP_REQUESTED else "budget-exhausted",
                                                   "global_step": engine.global_step,
                                                   "production_endpoint_reached": False})
            return 0 if STOP_REQUESTED else 75
        index = indices[engine.global_step]
        torch.cuda.synchronize(); begin = time.monotonic()
        decoded_before = len(dataset.decoded_episodes)
        samples = order.samples(dataset, index)
        loaded = time.monotonic()
        condition, targets, groups, info = encode_training_batch(
            samples, encoder, builder, target_cache, device, pool=int(order.pools[index]),
            feature_microbatch=recipe["feature_microbatch"])
        torch.cuda.synchronize(); encoded = time.monotonic()
        logical_count = len(targets)
        def injection(start, stop, logical_count=logical_count):
            if args.pause_after_microbatch_at == engine.global_step and start == 0:
                atomic_json(args.output/"fault-injection-ready.json", seal_contract({
                    "pid": os.getpid(), "global_step": engine.global_step,
                    "microbatch_pairs_backpropagated": stop, "logical_batch_pairs": logical_count,
                    "optimizer_update_complete": False}))
                # A pilot supervisor sends SIGKILL. No partial update is checkpointed.
                signal.pause()
                raise RuntimeError("fault-injection probe unexpectedly resumed")
        event = engine.step(condition, targets, [s.identity for s in samples], groups=groups,
                            hook=injection if args.pause_after_microbatch_at is not None else None)
        torch.cuda.synchronize(); finished = time.monotonic()
        event.update({"data_seconds": loaded-begin, "feature_seconds": encoded-loaded,
                      "update_seconds": finished-encoded, "full_step_seconds": finished-begin,
                      "newly_decoded_episodes": len(dataset.decoded_episodes)-decoded_before,
                      "active_seconds": budget.seconds, **info})
        if args.phase == "pilot":
            history.append(event)
        if args.phase == "pilot" or engine.global_step % recipe["log_interval_updates"] == 0:
            append_event(args.output/"events.jsonl", event)
            atomic_json(args.output/"status.json", {"status": "running", **event})
        if engine.global_step % interval == 0 or time.monotonic()-last_save >= recipe["checkpoint_interval_seconds"]:
            t = time.monotonic(); record = save_checkpoint()
            checkpoint_seconds += time.monotonic()-t; last_save = time.monotonic()
            append_event(args.output/"events.jsonl", {"event": "checkpoint", "global_step": engine.global_step,
                                                     "checkpoint": record["checkpoint"], "bytes": record["bytes"]})
        if engine.global_step % 16 == 0 or engine.global_step == len(indices):
            print(json.dumps({"phase": args.phase, "global_step": engine.global_step,
                              "updates": len(indices), "pairs_seen": engine.pairs_seen,
                              "loss": event["loss"], "full_step_seconds": event["full_step_seconds"]}), flush=True)
        del condition, targets, groups, samples
    torch.cuda.synchronize()
    base_after = tensor_state_digest(encoder.core.state_dict())
    if base_before != base_after or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
        raise ValueError("frozen base changed during CPT")
    if code_pins() != identity["source_sha256"] or read_sealed(args.recipe)["digest"] != recipe["digest"]:
        raise ValueError("training code or recipe changed while running")
    teacher_after = tensor_state_digest(teacher.visual.state_dict())
    if teacher_after != teacher_before:
        raise ValueError("target teacher changed during CPT")
    final = save_checkpoint()
    if args.phase == "train":
        if engine.pairs_seen != recipe["pair_presentations"]:
            raise ValueError("production endpoint has incorrect exposure")
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        from dataclasses import asdict
        header = seal_contract({"artifact_type": "eventground_wam.jepa_predictor_weights",
                                "config": asdict(model.config), "metadata": {
                                    "purpose": "fixed-final-C_jepa-candidate-pending-step6-gate",
                                    "training_identity": identity["digest"], "pairs_seen": engine.pairs_seen,
                                    "global_step": engine.global_step}, "state_sha256": tensor_state_digest(state)})
        atomic_torch_save(args.output/"C_jepa-candidate.pt", {"header": header, "state_dict": state})
    report = seal_contract({"artifact_type": "eventground_wam.jepa_cpt_run", "status": "complete",
                            "phase": args.phase, "identity_digest": identity["digest"],
                            "recipe_digest": recipe["digest"], "order_digest": order.manifest["digest"],
                            "source_sha256": identity["source_sha256"], "runtime": runtime,
                            "global_step": engine.global_step, "pairs_seen": engine.pairs_seen,
                            "sample_chain": engine.sample_chain, "snapshot_digest": tree_digest(engine.snapshot()),
                            "state_component_digests": {k: tree_digest(v) for k, v in engine.snapshot().items()},
                            "checkpoint": final, "resume_step": resume_step,
                            "base_state_sha256_before": base_before, "base_state_sha256_after": base_after,
                            "teacher_state_sha256_before": teacher_before, "teacher_state_sha256_after": teacher_after,
                            "target_cache_peak_bytes": target_cache.peak_bytes,
                            "trainable_base_parameters": 0, "validation_payloads_read": 0, "test_payloads_read": 0,
                            "decoded_episodes": dataset.decoded_episodes,
                            "total_process_seconds": time.monotonic()-started,
                            "checkpoint_seconds": checkpoint_seconds,
                            "active_seconds_all_processes": budget.seconds,
                            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                            "profile": history})
    atomic_json(args.output/"report.json", report)
    atomic_json(args.output/"status.json", {"status": "complete", "phase": args.phase,
                                           "global_step": engine.global_step, "report_digest": report["digest"]})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "train"), required=True)
    parser.add_argument("--recipe", type=Path, default=ROOT/"configs/training/week6_jepa_cpt_v1.json")
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home()/"data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--pause-after-microbatch-at", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    with (args.output/"owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
