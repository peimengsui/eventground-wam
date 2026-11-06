#!/usr/bin/env python3
"""Measure frozen parent L1/BCE gradients without constructing an optimizer."""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from collections import Counter
from pathlib import Path

import torch
from train_b0 import build_policy

from eventground_wam.b0_data import (
    B0DataConfig,
    B0StreamingDataset,
    load_b0_action_normalizer,
    load_frozen_b0_split_index,
)
from eventground_wam.b0_diagnostics import (
    gradient_statistics,
    parameter_group,
    summarize_gradients,
    tensor_tree_digest,
)
from eventground_wam.b0_model import load_b0_trainable_state_dict
from eventground_wam.b0_training import (
    EpochBatches,
    atomic_json,
    capture_rng,
    configure_deterministic_cuda,
    move_batch,
    read_sealed,
    restore_rng,
    sha256_file,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader


def main(args):
    started = time.monotonic()
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    experiment = read_sealed(args.source / "configs/training/week4_b0_oft_continuation_v1.json")
    panel_path = args.source / "evaluation/splits/week4-oft-continuation-v1/gradient-640.json"
    panel = read_sealed(panel_path)
    recipe_path = args.source / "configs/training/week4_b0_oft_v1.json"
    recipe = json.loads(recipe_path.read_text())
    root = args.data / "derived/b0/week4-b0-oft-v1"
    checkpoint = root / "B0.ckpt"
    recovery = Path(experiment["initialization"]["recovery_path"])
    assert sha256_file(checkpoint) == experiment["initialization"]["checkpoint_sha256"]
    assert sha256_file(recovery) == experiment["initialization"]["recovery_sha256"]
    normalization = args.data / "derived/b0/week3-b0-data-v1/action-statistics.json"
    assert sha256_file(normalization) == recipe["normalization_sha256"]
    index = load_frozen_b0_split_index(args.data / "derived/frozen/week2-dataset-v1")
    repository = args.data / (
        "datasets/libero-plus-lerobot/snapshots/"
        "22c57433fef692b5b9ecc0795344daac7fa867a5/repo")
    dataset = B0StreamingDataset(
        DemonstrationReader(repository), index, load_b0_action_normalizer(normalization),
        config=B0DataConfig(split="validation", seed=17, shuffle_episodes=False))
    selected = panel["episodes"]
    assert len(selected) == 40 and sum(len(r["anchors"]) for r in selected) == 640
    assert len({r["task_key"] for r in selected}) == 40
    assert all(index.assignments[r["episode_index"]].split == "validation" for r in selected)
    plan = [(r["episode_index"], tuple(r["anchors"])) for r in selected]
    batches = EpochBatches(dataset, plan, cache_dir=str(args.cache),
                           maximum_cached_episode_frames=4096,
                           policy_family=recipe["policy_family"])
    policy, names = build_policy(recipe, str(args.cache), device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_b0_trainable_state_dict(policy, saved["state_dict"], allowed_exact_names=names)
    del saved
    saved = torch.load(recovery, map_location="cpu", weights_only=False)
    restore_rng(saved["rng"])
    del saved
    parameters = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    names, tensors = zip(*parameters, strict=True)
    counts = Counter()
    for name, parameter in parameters:
        counts[parameter_group(name)] += parameter.numel()
    assert sum(counts.values()) == 92_907_527
    assert all(p.grad is None for p in policy.parameters())
    state_before = tensor_tree_digest(policy.state_dict())
    rng_before = capture_rng()
    modes_before = {n: m.training for n, m in policy.named_modules()}
    policy.train()  # Exactly the parent TrainingEngine.step mode, including checkpointing.
    rows = []
    superposition = None
    try:
        for i, selection in enumerate(selected):
            inputs, state, targets = move_batch(batches[i], device)
            output = policy(inputs, state, normalized_action_targets=targets)
            left = torch.autograd.grad(output.continuous_l1_loss, tensors,
                                       retain_graph=True, allow_unused=True)
            right = torch.autograd.grad(output.gripper_bce_loss, tensors,
                                        retain_graph=i == 0, allow_unused=True)
            groups = gradient_statistics(names, left, right)
            if i == 0:
                direct = torch.autograd.grad(output.loss, tensors, allow_unused=True)
                error2 = direct2 = 0.0
                maximum = 0.0
                for a, b, c in zip(left, right, direct, strict=True):
                    if c is None:
                        assert a is None and b is None
                        continue
                    expected = c.double()
                    residual = expected - (0 if a is None else a.double())
                    residual = residual - (0 if b is None else b.double())
                    error2 += float(residual.square().sum())
                    direct2 += float(expected.square().sum())
                    maximum = max(maximum, float(residual.abs().max()))
                superposition = {"max_absolute_error": maximum,
                                 "relative_l2_error": (error2 / direct2) ** .5,
                                 "note": "BF16 backward sums need not be bitwise additive"}
                # A gross discrepancy signals wrong gradient accounting; tiny BF16
                # differences are retained explicitly, never silently made exact.
                assert superposition["relative_l2_error"] < .05
                del direct
            row = {**selection, "motion_l1": float(output.continuous_l1_loss.detach()),
                   "gripper_bce": float(output.gripper_bce_loss.detach()),
                   "groups": groups}
            rows.append(row)
            atomic_json(args.output / f"batch-{i:02d}.json", seal_contract(row))
            print(json.dumps({"batch": i + 1, "task": selection["task_key"],
                              "backbone": groups["backbone_lora"]}), flush=True)
            assert all(p.grad is None for p in policy.parameters())
            del inputs, state, targets, output, left, right
    finally:
        restore_rng(rng_before)
        for name, module in policy.named_modules():
            module.training = modes_before[name]
    state_after = tensor_tree_digest(policy.state_dict())
    checks = {
        "all_parameters_and_persistent_buffers_unchanged": state_after == state_before,
        "rng_restored": tensor_tree_digest(capture_rng()) == tensor_tree_digest(rng_before),
        "modes_restored": modes_before == {n: m.training for n, m in policy.named_modules()},
        "no_parameter_grad_populated": all(p.grad is None for p in policy.parameters()),
        "parent_policy_file_unchanged": sha256_file(checkpoint)
        == experiment["initialization"]["checkpoint_sha256"],
        "parent_recovery_file_unchanged": sha256_file(recovery)
        == experiment["initialization"]["recovery_sha256"],
    }
    assert all(checks.values()), checks
    summary = summarize_gradients(rows)
    report = seal_contract({
        "artifact_type": "eventground_wam.b0_parent_gradient_diagnostic",
        "status": "complete", "checkpoint_sha256": sha256_file(checkpoint),
        "experiment_digest": experiment["digest"], "panel_digest": panel["digest"],
        "recipe_sha256": sha256_file(recipe_path), "sample_count": 640, "batch_count": 40,
        "trainable_parameter_counts": dict(counts), "summary": summary,
        "by_suite": {s: summarize_gradients([r for r in rows if r["task_key"].startswith(s + "/")])
                     for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10")},
        "bce_imbalance_flag": summary["backbone_lora"]["ratio_at_least_two_batches"] >= 30,
        "flag_definition": "backbone BCE/motion norm >=2 on at least 30 of 40 fixed batches",
        "limitations": "Endpoint validation-batch gradients, not training-history or causal evidence; "
                       "Euclidean norms before clipping, not Adam-preconditioned update magnitudes.",
        "superposition_check": superposition, "invariance_checks": checks,
        "model_state_digest_before_and_after": state_before,
        "optimizer_constructed": False, "optimizer_updates": 0,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "batches": rows,
    })
    atomic_json(args.output / "gradient-report.json", report)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "data", "cache", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "job.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "gradient-report.json").exists():
            raise FileExistsError("completed report already exists")
        main(args)
