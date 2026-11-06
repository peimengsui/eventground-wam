#!/usr/bin/env python3
"""Independent resource, accumulation and actual process-crash checks for JEPA CPT."""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from check_pixel_training import corruption_probe, verify_order

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.b0_training import atomic_json
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import read_sealed
from eventground_wam.jepa_data import JepaPairDataset
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_predictor import JepaPredictor, jepa_prediction_loss
from eventground_wam.jepa_training import (
    PoolTargetCache,
    accumulate_loss,
    encode_training_batch,
    verify_recipe,
)
from eventground_wam.pixel_data import FutureInputBuilder
from eventground_wam.pixel_predictor import FrozenQwenFeatures
from eventground_wam.pixel_training import (
    CPTOrder,
    PixelCheckpointStore,
    pilot_indices,
)

ROOT = Path(__file__).resolve().parents[1]


def progress(root, phase, **values):
    record = {"phase": phase, **values}
    atomic_json(root/"progress.json", record)
    print(json.dumps(record), flush=True)


def child_command(args, output, inject=False):
    command = [sys.executable, str(ROOT/"scripts/train_jepa_cpt.py"), "--phase", "pilot",
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


def accumulation_probe(args, recipe, order, sources, report):
    from train_jepa_cpt import configure

    configure(); torch.cuda.reset_peak_memory_stats()
    dataset = JepaPairDataset(args.pairs, args.data_root/
        f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo", split="train",
        expected_manifest_sha256=recipe["pair_manifest_sha256"], cache_episodes=4,
        normalizer=load_b0_action_normalizer(ROOT/"docs/model/week4-fresh-v2/action-statistics.json"),
        state_bounds=read_sealed(ROOT/"configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"])
    candidates = pilot_indices(order, sources)
    indices = [next(i for i in candidates if int(order.pair_counts[i]) == 40),
               int(np.argmin(order.pair_counts))]
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).cuda()
    encoder = FrozenQwenFeatures(full.model).cuda()
    processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True)
    builder = FutureInputBuilder(processor)
    cache = PoolTargetCache(FrozenQwenVisualTargets(full.model.visual, processor.image_processor),
                            maximum_bytes=recipe["target_cache"]["maximum_bytes"])
    saved = PixelCheckpointStore(args.output/"uninterrupted/recovery", report["identity_digest"]).load()
    model = JepaPredictor().cuda(); model.load_state_dict(saved["model"]); model.train()
    results = []
    for index in indices:
        samples = order.samples(dataset, index)
        condition, targets, groups, info = encode_training_batch(
            samples, encoder, builder, cache, torch.device("cuda"), pool=int(order.pools[index]),
            feature_microbatch=recipe["feature_microbatch"])
        model.zero_grad(set_to_none=True)
        whole = jepa_prediction_loss(model(condition).features, targets, groups, **recipe["loss"])
        whole.total.backward()
        whole_value = float(whole.total.detach())
        gradients = [p.grad.detach().cpu().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        micro = accumulate_loss(model, condition, targets, groups,
                                microbatch=recipe["predictor_microbatch"], loss_rule=recipe["loss"])
        error2, reference2, maximum = 0., 0., 0.
        for expected, parameter in zip(gradients, model.parameters(), strict=True):
            actual = parameter.grad.detach().cpu(); delta = (actual-expected).double()
            error2 += float(delta.square().sum()); reference2 += float(expected.double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        relative = (error2/max(reference2, 1e-30))**.5
        passed = (abs(whole_value-micro["loss"]) <= recipe["pilot"]["accumulation_loss_absolute_tolerance"]
                  and relative <= recipe["pilot"]["accumulation_gradient_relative_l2_tolerance"]
                  and maximum <= recipe["pilot"]["accumulation_gradient_max_absolute_tolerance"])
        results.append({"plan_index": index, "pairs": len(targets), "full_loss": whole_value,
                        "accumulated_loss": micro["loss"], "variance": micro["variance"],
                        "independent_anchors": micro["independent_anchors"],
                        "loss_absolute_difference": abs(whole_value-micro["loss"]),
                        "gradient_relative_l2_difference": relative, "gradient_max_absolute_difference": maximum,
                        "passed": passed, **info})
        if not passed:
            atomic_json(args.output/"accumulation-failure.json", {"results": results})
            raise ValueError("full/two-pass gradient equivalence failed")
    cache_result = cache_probe(dataset, order, cache)
    return {"status": "passed", "batches": results, "cache": cache_result,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "optimizer_updates": 0}


def cache_probe(dataset, order, cache):
    # Bound the entire epoch's peak from metadata, then fill its largest real pool.
    starts = order.manifest["pool_first_updates"]
    pools = []
    for j, start in enumerate(starts):
        stop = starts[j+1] if j+1 < len(starts) else len(order)
        eps = sorted({int(e) for e in order.anchors[order.offsets[start]:order.offsets[stop], 0]})
        pools.append((sum(dataset.sources[e]["episode_length"]-2 for e in eps), j, eps))
    frames, pool, eps = max(pools)
    cache.clear()
    initial_samples, initial_values = None, None
    started = time.monotonic()
    encoded = 0
    for ep in eps:
        length = dataset.sources[ep]["episode_length"]
        for begin in range(0, length-2, 16):
            samples = [dataset[dataset.ordinal(ep, t, 2)] for t in range(begin, min(begin+16, length-2))]
            values, info = cache.encode(samples, pool=pool, device=torch.device("cpu"))
            encoded += info["target_cache_misses"]
            if initial_samples is None:
                initial_samples, initial_values = samples, values.clone()
    elapsed = time.monotonic()-started
    assert encoded == frames and len(cache.values) == frames
    hit, hit_info = cache.encode(initial_samples, pool=pool, device=torch.device("cpu"))
    assert hit_info["target_cache_misses"] == 0 and torch.equal(hit, initial_values)
    assembly = []
    for _ in range(20):
        torch.cuda.synchronize(); begin = time.monotonic()
        # Full-size all-hit bank exercises 40-pair gather plus host-to-device copy.
        cache.encode((initial_samples*40)[:40], pool=pool, device=torch.device("cuda"))
        torch.cuda.synchronize(); assembly.append(time.monotonic()-begin)
    fresh = cache.teacher(np.stack([s.target_rgb for s in initial_samples])).cpu()
    assert torch.equal(fresh, initial_values)
    cache.clear()
    restarted, _ = cache.encode(initial_samples, pool=pool, device=torch.device("cpu"))
    assert torch.equal(restarted, initial_values)
    return {"status": "passed", "maximum_pool_target_frames": frames, "maximum_pool": pool,
            "episodes": eps, "full_pool_encoded_frames": encoded, "full_pool_seconds": elapsed,
            "observed_cache_peak_bytes": cache.peak_bytes, "maximum_pool_bytes": frames*128*2048*4,
            "cache_assembly_p95_seconds": float(np.quantile(assembly, .95)), "cache_hit_exact": True, "fresh_encoding_exact": True, "empty_cache_restart_exact": True,
            "replayed_dual_frames": 3*len(initial_samples),
            "full_epoch_unique_target_frames": order.manifest["anchors"],
            "full_epoch_fp32_target_disk_bytes": order.manifest["anchors"]*128*2048*4,
            "persistent_cache_bytes": 0}


def resource_projection(args, recipe, order, report, observed_gpu_mib):
    events = report["profile"]
    warm = [e for e in events if e["newly_decoded_episodes"] == 0]
    cold = [e for e in events if e["newly_decoded_episodes"] > 0]
    if len(warm) < 32 or len(cold) < 8:
        raise ValueError("resource pilot lacks sufficient warm and cold batches")
    warm_data = float(np.quantile([e["data_seconds"] for e in warm], .95))
    core = float(np.quantile([e["current_feature_seconds"]+e["update_seconds"] for e in events], .95))
    target_per_frame = float(np.quantile([e["target_seconds"]/e["target_cache_misses"]
                                          for e in events if e["target_cache_misses"]], .95))
    # Charge all target assembly/transfer overhead again at every logical update;
    # this makes amortization conservative even when most frames become cache hits.
    target_assembly = report["cache_assembly_p95_seconds"]
    per_episode = float(np.quantile([e["data_seconds"]/e["newly_decoded_episodes"] for e in cold], .95))
    save_seconds = report["checkpoint_seconds"]/max(1, report["global_step"]//recipe["pilot"]["checkpoint_interval_updates"])
    # Include scheduled saves and additional wall-time-triggered saves conservatively.
    count_saves = int(np.ceil(recipe["total_updates"]/recipe["checkpoint_interval_updates"]))
    raw = recipe["total_updates"]*(core+warm_data+target_assembly)+order.manifest["episodes"]*per_episode+order.manifest["anchors"]*target_per_frame
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
            "p95_current_feature_plus_update_seconds": core, "p95_target_seconds_per_miss": target_per_frame,
            "p95_target_assembly_seconds": target_assembly, "p95_warm_data_seconds": warm_data,
            "p95_decode_seconds_per_episode": per_episode, "mean_checkpoint_seconds": save_seconds,
            "projected_training_hours_with_1_5_margin": projected/3600,
            "projection_scope": "One full pass, online current-feature extraction, full-pair loss/backward, source decoding, checkpoint saves. Canonical teacher computation with one encoding per distinct target in each pool, measured cache assembly, no full validation.",
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
    reference_for_resources = dict(reference)
    reference_for_resources["host_peak_rss_bytes"] = max(reference["host_peak_rss_bytes"], accumulation["host_peak_rss_bytes"], resumed["host_peak_rss_bytes"])
    reference_for_resources["cache_assembly_p95_seconds"] = accumulation["cache"]["cache_assembly_p95_seconds"]
    resources = resource_projection(args, recipe, order, reference_for_resources, max(observed))
    resources["target_cache"] = accumulation["cache"]
    resources["checks"]["target_cache_ram_bound"] = accumulation["cache"]["maximum_pool_bytes"] <= recipe["target_cache"]["maximum_bytes"]
    from train_jepa_cpt import code_pins
    if code_pins() != reference["source_sha256"]:
        raise ValueError("source changed during pilot verification")
    report = seal_contract({"artifact_type": "eventground_wam.jepa_cpt_pilot", "schema_version": 1,
                            "status": "passed" if all(resources["checks"].values()) else "failed",
                            "recipe_digest": recipe["digest"], "order_digest": order.manifest["digest"],
                            "source_sha256": reference["source_sha256"], "runtime": reference["runtime"],
                            "order_verification_digest": order_check["digest"],
                            "uninterrupted_report_digest": reference["digest"], "resumed_report_digest": resumed["digest"],
                            "recovery": recovery, "corruption_fallback": corruption,
                            "accumulation": accumulation, "resources": resources,
                            "teacher_state_unchanged": reference["teacher_state_sha256_before"] == reference["teacher_state_sha256_after"] == resumed["teacher_state_sha256_after"],
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
    parser.add_argument("--recipe", type=Path, default=ROOT/"configs/training/week6_jepa_cpt_v1.json")
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home()/"data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    main(parser.parse_args())
