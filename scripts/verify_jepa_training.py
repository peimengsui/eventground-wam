#!/usr/bin/env python3
"""Audit saved JEPA pilot evidence and compare actual CPU recovery payloads."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.future_pairs import read_sealed, sha256_file, write_json
from eventground_wam.pixel_training import tree_digest


def equal_tree(a, b):
    """Compare actual values independently of reported state/content digests."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def audit(args):
    root, out = args.source, args.evidence
    report = read_sealed(out/"pilot-report.json")
    recipe = read_sealed(root/"configs/training/week6_jepa_cpt_v1.json")
    inventory = read_sealed(out/"source-inventory.json")
    order = read_sealed(root/"configs/datasets/trajectory-v2-cpt-order-v1/manifest.json")
    common = read_sealed(out/"order-verification.json")
    assert report["status"] == "passed"
    assert recipe["digest"] == report["recipe_digest"] == inventory["recipe_digest"]
    assert report["order_digest"] == order["digest"] == common["manifest_digest"]
    assert report["order_verification_digest"] == common["digest"]
    assert common["byte_identical_regeneration"] and common["independent_exact_anchor_and_pair_coverage"]
    assert common["pair_presentations"] == recipe["pair_presentations"] == 8221365
    assert common["updates"] == recipe["total_updates"] == 221633
    for name, digest in {**inventory["files"], **recipe["input_sha256"], **report["source_sha256"]}.items():
        assert sha256_file(root/name) == digest, name
    reference, resumed = (read_sealed(out/name/"report.json") for name in ("uninterrupted", "resumed"))
    assert reference["digest"] == report["uninterrupted_report_digest"]
    assert resumed["digest"] == report["resumed_report_digest"]
    assert reference["identity_digest"] == resumed["identity_digest"]
    assert reference["runtime"] == resumed["runtime"] == report["runtime"]
    snapshots = []
    for name, run in (("uninterrupted", reference), ("resumed", resumed)):
        identity = read_sealed(out/name/"identity.json")
        initial = read_sealed(out/name/"initialization.json")
        assert identity["digest"] == run["identity_digest"] == initial["identity_digest"]
        assert identity["phase"] == "pilot" and initial["optimizer_state_entries"] == 0
        assert not initial["parent_or_diagnostic_weights_loaded"]
        checkpoint = args.payloads/name/"final.pt"
        assert sha256_file(checkpoint) == run["checkpoint"]["sha256"]
        value = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert tree_digest(value) == run["snapshot_digest"]
        assert {k: tree_digest(v) for k, v in value.items()} == run["state_component_digests"]
        assert value["global_step"] == 128 and value["pairs_seen"] == run["pairs_seen"]
        assert value["scheduler"]["last_epoch"] == 128
        snapshots.append(value)
    assert equal_tree(*snapshots)
    assert resumed["resume_step"] == 16 and reference["resume_step"] == 0
    fault = read_sealed(out/"resumed/fault-injection-ready.json")
    resume = read_sealed(out/"resumed/process-resume.json")
    assert fault["global_step"] == 19 and not fault["optimizer_update_complete"]
    assert fault["microbatch_pairs_backpropagated"] == 10 < fault["logical_batch_pairs"]
    assert resume["global_step"] == 16 and report["recovery"]["signal"] == "SIGKILL"
    assert report["corruption_fallback"]["recovered_update"] == 120
    lines = [json.loads(s) for s in (out/"resumed/events.jsonl").read_text().splitlines()]
    resumed_updates = [x for x in lines if "loss" in x]
    keys = ("global_step", "plan_index", "pairs", "pairs_seen", "batch_digest", "sample_chain", "loss",
            "cosine", "variance", "independent_anchors", "learning_rate", "gradient_norm")
    lookup = {e["global_step"]: e for e in reference["profile"]}
    for event in resumed_updates:
        assert all(event[k] == lookup[event["global_step"]][k] for k in keys)
    assert len(resumed_updates) == 19+112
    assert all(math.isfinite(e[k]) for e in reference["profile"] for k in ("loss", "cosine", "variance", "gradient_norm"))
    assert all(e["independent_anchors"] == e["unique_current_observations"] for e in reference["profile"])
    criteria = recipe["pilot"]
    batches = report["accumulation"]["batches"]
    assert [b["pairs"] for b in batches] == [40, 1]
    assert batches[0]["independent_anchors"] == 8 and batches[1]["independent_anchors"] == 1
    assert batches[1]["variance"] == 0
    for b in batches:
        assert b["loss_absolute_difference"] <= criteria["accumulation_loss_absolute_tolerance"]
        assert b["gradient_relative_l2_difference"] <= criteria["accumulation_gradient_relative_l2_tolerance"]
        assert b["gradient_max_absolute_difference"] <= criteria["accumulation_gradient_max_absolute_tolerance"]
    cache = report["accumulation"]["cache"]
    assert cache["full_pool_encoded_frames"] == cache["maximum_pool_target_frames"] == 1193
    assert cache["maximum_pool_bytes"] == 1193*128*2048*4 <= recipe["target_cache"]["maximum_bytes"]
    assert cache["observed_cache_peak_bytes"] == cache["maximum_pool_bytes"]
    assert cache["cache_hit_exact"] and cache["fresh_encoding_exact"] and cache["empty_cache_restart_exact"]
    assert cache["full_epoch_fp32_target_disk_bytes"] == order["anchors"]*128*2048*4
    resources = report["resources"]
    events = reference["profile"]
    q95 = lambda values: float(np.quantile(values, .95))
    core = q95([e["current_feature_seconds"]+e["update_seconds"] for e in events])
    warm = q95([e["data_seconds"] for e in events if e["newly_decoded_episodes"] == 0])
    decode = q95([e["data_seconds"]/e["newly_decoded_episodes"] for e in events if e["newly_decoded_episodes"]])
    target = q95([e["target_seconds"]/e["target_cache_misses"] for e in events if e["target_cache_misses"]])
    raw = recipe["total_updates"]*(core+warm+cache["cache_assembly_p95_seconds"])+order["episodes"]*decode+order["anchors"]*target
    saves = math.ceil(recipe["total_updates"]/recipe["checkpoint_interval_updates"])+math.ceil(raw/recipe["checkpoint_interval_seconds"])+1
    seconds_per_save = reference["checkpoint_seconds"]/(128//criteria["checkpoint_interval_updates"])
    projected = (raw+saves*seconds_per_save+reference["total_process_seconds"])*1.5/3600
    assert abs(projected-resources["projected_training_hours_with_1_5_margin"]) < 1e-10
    assert projected < recipe["maximum_projected_training_hours"]
    telemetry = [float(x) for x in (out/"gpu-memory.csv").read_text().splitlines() if x.strip()]
    assert max(telemetry)*1024**2 == resources["observed_full_device_peak_bytes"]
    assert resources["observed_full_device_peak_bytes"] < resources["device_total_bytes"]*.85
    assert resources["host_peak_rss_bytes"] < 24*1024**3
    persistent = reference["checkpoint"]["bytes"]*4+4*recipe["parameter_budget"]["total_trainable"]+sum(x["bytes"] for x in order["files"].values())+64*1024**2
    assert persistent == resources["projected_new_persistent_bytes"] < 2*1024**3
    assert resources["disk_free_bytes"] >= 5*1024**3
    assert all(resources["checks"].values())
    accepted = read_sealed(root/"docs/evaluation/week6-jepa-targets-v2/report.json")
    assert reference["teacher_state_sha256_before"] == reference["teacher_state_sha256_after"] == resumed["teacher_state_sha256_after"] == accepted["vision_state_after"]
    assert reference["base_state_sha256_before"] == reference["base_state_sha256_after"] == resumed["base_state_sha256_after"]
    assert report["validation_payloads_read"] == report["test_payloads_read"] == 0
    completed_pairs = reference["pairs_seen"]+sum(e["pairs"] for e in resumed_updates)
    return seal_contract({"artifact_type": "eventground_wam.jepa_cpt_independent_local_audit", "schema_version": 1,
                          "passed": True, "pilot_report_digest": report["digest"], "recipe_digest": recipe["digest"],
                          "auditor_sha256": sha256_file(Path(__file__)), "actual_recovery_payloads_bitwise_equal": True,
                          "all_reported_state_digests_recomputed": True, "replayed_update_records_exact": 112,
                          "all_recovery_update_records_compared": len(resumed_updates),
                          "gradient_thresholds_verified": True, "cache_bounds_and_exact_replay_verified": True,
                          "runtime_projection_recomputed_hours": projected, "gpu_telemetry_maximum_verified": True,
                          "persistent_disk_projection_verified": True, "source_recipe_and_order_pins_verified": True,
                          "diagnostic_completed_optimizer_updates": 128+len(resumed_updates),
                          "diagnostic_completed_pair_presentations": completed_pairs,
                          "killed_partial_backward_presentations": 10,
                          "gradient_equivalence_backward_presentations": 2*sum(b["pairs"] for b in batches),
                          "validation_payloads_read": 0, "test_payloads_read": 0,
                          "limits": "Independent CPU audit of actual recovery files, trace equality, source identity and resource arithmetic. CUDA gradient/cache probes are checked against their frozen report thresholds, not rerun by this audit. No held-out predictor acceptance."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, audit(args))
    print("Independent JEPA training recipe verification passed")
