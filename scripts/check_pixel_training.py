#!/usr/bin/env python3
"""Independent resource, accumulation and actual process-crash checks for pixel CPT."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.b0_training import atomic_json
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import HORIZONS, read_sealed
from eventground_wam.pixel_data import FutureInputBuilder, FuturePairDataset
from eventground_wam.pixel_predictor import FrozenQwenFeatures, PixelPredictor
from eventground_wam.pixel_training import (
    CPTOrder,
    PixelCheckpointStore,
    accumulate_loss,
    build_order,
    encode_training_batch,
    pilot_indices,
    verify_recipe,
)

ROOT = Path(__file__).resolve().parents[1]


def progress(root, phase, **values):
    record = {"phase": phase, **values}
    atomic_json(root/"progress.json", record)
    print(json.dumps(record), flush=True)


def verify_order(args, order, sources):
    rebuilt = build_order(list(reversed(list(sources.values()))), args.output/"order-regenerated",
                           pair_manifest_sha256=order.manifest["pair_manifest_sha256"])
    if rebuilt != order.manifest:
        raise ValueError("order regeneration differs")
    anchors = np.asarray(order.anchors)
    maximum = max(sources)+1
    lengths = np.zeros(maximum, dtype=np.int64)
    for ep, source in sources.items():
        lengths[ep] = source["episode_length"]
    ep, t = anchors[:, 0].astype(np.int64), anchors[:, 1].astype(np.int64)
    packed = (ep.astype(np.uint64) << np.uint64(32)) | t.astype(np.uint64)
    counts = np.bincount(ep, minlength=maximum)
    if (len(np.unique(packed)) != len(anchors) or np.any(t >= lengths[ep]-2)
            or any(counts[e] != s["episode_length"]-2 for e, s in sources.items())):
        raise ValueError("independent coverage check found repeated, missing or invalid anchors")
    horizon_counts = {str(h): int(np.count_nonzero(t+h < lengths[ep])) for h in HORIZONS}
    if horizon_counts != order.manifest["pairs_by_horizon"]:
        raise ValueError("independent horizon coverage differs")
    counts_per_anchor = sum((t+h < lengths[ep]).astype(np.int64) for h in HORIZONS)
    prefix = np.concatenate(([0], np.cumsum(counts_per_anchor)))
    batch_counts = prefix[order.offsets[1:]]-prefix[order.offsets[:-1]]
    if not np.array_equal(batch_counts, order.pair_counts):
        raise ValueError("independent logical-batch target counts differ")
    max_episodes = max(len({e for e, _ in order.references(i)}) for i in range(len(order)))
    if max_episodes > 4:
        raise ValueError("episode cache budget exceeded")
    record = seal_contract({"artifact_type": "eventground_wam.cpt_order_verification", "status": "passed",
                             "manifest_digest": order.manifest["digest"], "byte_identical_regeneration": True,
                             "independent_exact_anchor_and_pair_coverage": True,
                             "anchors": len(anchors), "updates": len(order),
                             "pair_presentations": int(batch_counts.sum()), "pairs_by_horizon": horizon_counts,
                             "maximum_episodes_per_batch": max_episodes,
                             "train_metadata_only": True, "validation_payloads_read": 0, "test_payloads_read": 0})
    atomic_json(args.output/"order-verification.json", record)
    return record


def child_command(args, output, inject=False):
    command = [sys.executable, str(ROOT/"scripts/train_pixel_cpt.py"), "--phase", "pilot",
               "--recipe", str(args.recipe), "--order", str(args.order), "--pairs", str(args.pairs),
               "--output", str(output), "--data-root", str(args.data_root), "--cache", str(args.cache)]
    if inject:
        command += ["--pause-after-microbatch-at", "19"]
    return command


def launch(args, output, *, inject=False):
    output.mkdir(parents=True, exist_ok=True)
    log = (output/"process.log").open("a")
    process = subprocess.Popen(child_command(args, output, inject), stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONPATH=str(ROOT/"src"), HF_HUB_OFFLINE="1",
                                        TOKENIZERS_PARALLELISM="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
    return process, log


def wait_success(process, log):
    try:
        status = process.wait(timeout=1800)
        if status != 0:
            raise ValueError(f"pilot child exited {status}; inspect process.log")
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        log.close()


def crash_and_resume(args):
    baseline = args.output/"uninterrupted"
    progress(args.output, "uninterrupted-full-path-pilot")
    p, log = launch(args, baseline); baseline_pid = p.pid; wait_success(p, log)
    interrupted = args.output/"resumed"
    progress(args.output, "crash-during-accumulation")
    p, log = launch(args, interrupted, inject=True)
    killed_pid = p.pid; deadline = time.monotonic()+600
    try:
        while not (interrupted/"fault-injection-ready.json").exists():
            if p.poll() is not None:
                raise ValueError("fault-injection child failed before reaching partial update")
            if time.monotonic() > deadline:
                raise TimeoutError("fault injection was not reached")
            time.sleep(.1)
        ready = read_sealed(interrupted/"fault-injection-ready.json")
        if (ready["pid"] != p.pid or ready["global_step"] != 19 or ready["optimizer_update_complete"]
                or not 0 < ready["microbatch_pairs_backpropagated"] < ready["logical_batch_pairs"]):
            raise ValueError("process did not stop inside a partial accumulation")
        committed = read_sealed(interrupted/"recovery/latest.json")
        if committed["global_step"] != 16:
            raise ValueError("unexpected recovery boundary for kill test")
        p.kill()
        if p.wait(timeout=30) != -signal.SIGKILL:
            raise ValueError("pilot was not terminated by SIGKILL")
    finally:
        if p.poll() is None:
            p.kill(); p.wait()
        log.close()
    progress(args.output, "new-process-resume-from-update-16")
    p, log = launch(args, interrupted); resumed_pid = p.pid; wait_success(p, log)
    a, b = read_sealed(baseline/"report.json"), read_sealed(interrupted/"report.json")
    if a["identity_digest"] != b["identity_digest"] or a["state_component_digests"] != b["state_component_digests"]:
        raise ValueError("uninterrupted/resumed complete training state differs")
    if a["snapshot_digest"] != b["snapshot_digest"] or b["resume_step"] != 16:
        raise ValueError("process resume did not reproduce the complete endpoint")
    if len({baseline_pid, killed_pid, resumed_pid}) != 3:
        raise ValueError("recovery test requires three distinct processes")
    return a, b, {"baseline_pid": baseline_pid, "killed_pid": killed_pid, "resumed_pid": resumed_pid,
                 "signal": "SIGKILL", "partial_update": ready, "last_committed_update": 16,
                 "resume_update": b["resume_step"], "complete_state_bitwise_equal": True,
                 "components": list(a["state_component_digests"])}


def corruption_probe(args, report):
    copied = args.output/"corruption-probe"
    shutil.copytree(args.output/"resumed/recovery", copied)
    records = sorted(copied.glob("checkpoint-*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in records[:2]:
        record = read_sealed(path); (copied/record["checkpoint"]).write_bytes(b"deliberate damaged-copy probe")
    (copied/".unfinished.partial").write_bytes(b"incomplete write")
    store = PixelCheckpointStore(copied, report["identity_digest"])
    value = store.load()
    expected = read_sealed(records[2])["global_step"]
    if value["global_step"] != expected or not store.notes:
        raise ValueError("damaged-generation fallback failed")
    return {"corrupted_copied_generations": 2, "recovered_update": expected,
            "original_recovery_files_unchanged": True, "notes": store.notes}


def accumulation_probe(args, recipe, order, sources, report):
    from train_pixel_cpt import configure

    configure(); torch.cuda.reset_peak_memory_stats()
    dataset = FuturePairDataset(args.pairs, args.data_root/
        f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo", split="train",
        expected_manifest_sha256=recipe["pair_manifest_sha256"], cache_episodes=4,
        normalizer=load_b0_action_normalizer(ROOT/"docs/model/week4-fresh-v2/action-statistics.json"),
        state_bounds=read_sealed(ROOT/"configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"])
    candidates = pilot_indices(order, sources)
    # Cover both a maximum logical batch and an actual retained short batch.
    indices = [next(i for i in candidates if int(order.pair_counts[i]) == 40),
               int(np.argmin(order.pair_counts))]
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).cuda()
    encoder = FrozenQwenFeatures(full.model).cuda()
    builder = FutureInputBuilder(load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True))
    saved = PixelCheckpointStore(args.output/"uninterrupted/recovery", report["identity_digest"]).load()
    model = PixelPredictor().cuda(); model.load_state_dict(saved["model"]); model.train()
    results = []
    for index in indices:
        samples = order.samples(dataset, index)
        condition, targets, info = encode_training_batch(samples, encoder, builder, torch.device("cuda"),
                                                          feature_microbatch=recipe["feature_microbatch"])
        model.zero_grad(set_to_none=True)
        whole = accumulate_loss(model, condition, targets, microbatch=len(targets), ssim_weight=recipe["loss"]["ssim_weight"])
        gradients = [p.grad.detach().cpu().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        micro = accumulate_loss(model, condition, targets, microbatch=recipe["predictor_microbatch"],
                                ssim_weight=recipe["loss"]["ssim_weight"])
        error2, reference2, maximum = 0., 0., 0.
        for expected, p in zip(gradients, model.parameters(), strict=True):
            actual = p.grad.detach().cpu(); delta = (actual-expected).double()
            error2 += float(delta.square().sum()); reference2 += float(expected.double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        relative = (error2/max(reference2, 1e-30))**.5
        passed = (abs(whole-micro) <= recipe["pilot"]["accumulation_loss_absolute_tolerance"]
                  and relative <= recipe["pilot"]["accumulation_gradient_relative_l2_tolerance"]
                  and maximum <= recipe["pilot"]["accumulation_gradient_max_absolute_tolerance"])
        results.append({"plan_index": index, "pairs": len(targets), "full_loss": whole,
                        "accumulated_loss": micro, "loss_absolute_difference": abs(whole-micro),
                        "gradient_relative_l2_difference": relative, "gradient_max_absolute_difference": maximum,
                        "passed": passed, **info})
        if not passed:
            atomic_json(args.output/"accumulation-failure.json", {"results": results})
            raise ValueError("full/microbatch gradient equivalence failed")
    return {"status": "passed", "batches": results,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_updates": 0}


def resource_projection(args, recipe, order, report, observed_gpu_mib):
    events = report["profile"]
    warm = [e for e in events if e["newly_decoded_episodes"] == 0]
    cold = [e for e in events if e["newly_decoded_episodes"] > 0]
    if len(warm) < 32 or len(cold) < 8:
        raise ValueError("resource pilot lacks sufficient warm and cold batches")
    warm_data = float(np.quantile([e["data_seconds"] for e in warm], .95))
    core = float(np.quantile([e["feature_seconds"]+e["update_seconds"] for e in events], .95))
    per_episode = float(np.quantile([e["data_seconds"]/e["newly_decoded_episodes"] for e in cold], .95))
    save_seconds = report["checkpoint_seconds"]/max(1, report["global_step"]//recipe["pilot"]["checkpoint_interval_updates"])
    # Include scheduled saves and additional wall-time-triggered saves conservatively.
    count_saves = int(np.ceil(recipe["total_updates"]/recipe["checkpoint_interval_updates"]))
    raw = recipe["total_updates"]*(core+warm_data)+order.manifest["episodes"]*per_episode
    count_saves += int(np.ceil(raw/recipe["checkpoint_interval_seconds"]))+1
    projected = (raw+count_saves*save_seconds+report["total_process_seconds"])*recipe["resource_limits"]["projection_safety_factor"]
    checkpoint_bytes = report["checkpoint"]["bytes"]
    order_bytes = sum(pin["bytes"] for pin in order.manifest["files"].values())
    persistent = checkpoint_bytes*(recipe["checkpoint_keep_last"]+1)+4*recipe["parameter_budget"]["total_trainable"]+order_bytes+64*1024**2
    free = shutil.disk_usage(args.output).free
    device_bytes = torch.cuda.get_device_properties(0).total_memory
    peak_device = int(observed_gpu_mib*1024**2)
    checks = {"training_time_with_safety_margin": projected < recipe["maximum_projected_training_hours"]*3600,
              "gpu_headroom": peak_device < device_bytes*recipe["resource_limits"]["maximum_gpu_memory_fraction"],
              "host_memory": report["host_peak_rss_bytes"] < recipe["resource_limits"]["maximum_host_rss_bytes"],
              "persistent_disk_budget": persistent < recipe["resource_limits"]["maximum_new_persistent_bytes"],
              "free_disk_reserve": free >= recipe["resource_limits"]["minimum_free_disk_bytes"]}
    return {"checks": checks, "observed_steps": len(events), "cold_batches": len(cold), "warm_batches": len(warm),
            "pilot_pairs_per_second_including_cold_decodes": report["pairs_seen"]/sum(e["full_step_seconds"] for e in events),
            "p95_feature_plus_update_seconds": core, "p95_warm_data_seconds": warm_data,
            "p95_decode_seconds_per_episode": per_episode, "mean_checkpoint_seconds": save_seconds,
            "projected_training_hours_with_1_5_margin": projected/3600,
            "projection_scope": "One full pass, online current-feature extraction, full-pair loss/backward, source decoding, checkpoint saves. No full validation or JEPA-target computation.",
            "observed_full_device_peak_bytes": peak_device, "device_total_bytes": device_bytes,
            "torch_peak_allocated_bytes": report["gpu_peak_allocated_bytes"],
            "torch_peak_reserved_bytes": report["gpu_peak_reserved_bytes"],
            "host_peak_rss_bytes": report["host_peak_rss_bytes"],
            "one_recovery_checkpoint_bytes": checkpoint_bytes,
            "projected_new_persistent_bytes": persistent, "disk_free_bytes": free,
            "feature_cache_disk_bytes": 0, "maximum_batch_feature_bytes": max(e["feature_bytes"] for e in events),
            "maximum_feature_tokens": max(e["maximum_feature_tokens"] for e in events)}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = verify_recipe(args.recipe, ROOT)
    order = CPTOrder(args.order, recipe["order_manifest_sha256"])
    raw = [json.loads(line) for line in (args.pairs/"sources.jsonl").read_text().splitlines()]
    sources = {s["episode_index"]: s for s in raw if s["split"] == "train"}
    progress(args.output, "verify-complete-common-order")
    order_check = verify_order(args, order, sources)
    telemetry = (args.output/"gpu-memory.csv").open("w")
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "250"],
                               stdout=telemetry, stderr=subprocess.DEVNULL)
    try:
        reference, resumed, recovery = crash_and_resume(args)
        progress(args.output, "damaged-checkpoint-copy-fallback")
        corruption = corruption_probe(args, resumed)
        progress(args.output, "full-and-microbatch-gradient-comparison")
        accumulation = accumulation_probe(args, recipe, order, sources, reference)
        torch.cuda.synchronize(); time.sleep(.3)
    finally:
        monitor.terminate(); monitor.wait(timeout=10); telemetry.close()
    observed = [float(line.strip()) for line in (args.output/"gpu-memory.csv").read_text().splitlines() if line.strip()]
    if not observed:
        raise ValueError("GPU telemetry is empty")
    resources = resource_projection(args, recipe, order, reference, max(observed))
    from train_pixel_cpt import code_pins
    if code_pins() != reference["source_sha256"]:
        raise ValueError("source changed during pilot verification")
    report = seal_contract({"artifact_type": "eventground_wam.pixel_cpt_pilot", "schema_version": 1,
                            "status": "passed" if all(resources["checks"].values()) else "failed",
                            "recipe_digest": recipe["digest"], "order_digest": order.manifest["digest"],
                            "source_sha256": reference["source_sha256"], "runtime": reference["runtime"],
                            "order_verification_digest": order_check["digest"],
                            "uninterrupted_report_digest": reference["digest"], "resumed_report_digest": resumed["digest"],
                            "recovery": recovery, "corruption_fallback": corruption,
                            "accumulation": accumulation, "resources": resources,
                            "base_state_unchanged": reference["base_state_sha256_before"] == reference["base_state_sha256_after"] == resumed["base_state_sha256_after"],
                            "pilot_logical_updates": reference["global_step"], "pilot_unique_pairs": reference["pairs_seen"],
                            "validation_payloads_read": 0, "test_payloads_read": 0,
                            "production_training_started": False, "pilot_checkpoint_is_production_initializer": False})
    atomic_json(args.output/"pilot-report.json", report)
    progress(args.output, "complete", status=report["status"], report_digest=report["digest"],
             projected_training_hours=resources["projected_training_hours_with_1_5_margin"])
    if report["status"] != "passed":
        raise ValueError("pilot resource requirements failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=ROOT/"configs/training/week5_pixel_cpt_v1.json")
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home()/"data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    main(parser.parse_args())
