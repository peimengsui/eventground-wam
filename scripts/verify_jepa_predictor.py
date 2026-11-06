#!/usr/bin/env python3
"""Reconstruct real inputs and independently recompute JEPA diagnostic metrics."""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import HORIZONS, read_sealed, sha256_file, write_json
from eventground_wam.jepa_data import JepaPairDataset, collate_jepa_samples
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_predictor import load_jepa_checkpoint
from eventground_wam.pixel_predictor import PixelCondition, tensor_state_digest


def numpy_metrics(prediction, targets, group_ids, rule):
    p, t = np.asarray(prediction, np.float64), np.asarray(targets, np.float64)
    cosine = np.clip(1-np.sum(p*t, axis=-1), 0, 2).mean()
    groups = sorted(set(group_ids.tolist()))
    penalty = 0.
    if len(groups) > 1:
        p = np.stack([p[group_ids == g].mean(axis=0) for g in groups])
        t = np.stack([t[group_ids == g].mean(axis=0) for g in groups])
        pv = np.mean((p-p.mean(axis=0))**2, axis=0)
        tv = np.mean((t-t.mean(axis=0))**2, axis=0)
        p_std, t_std = np.sqrt(pv*2048+rule["variance_epsilon"]), np.sqrt(tv*2048+rule["variance_epsilon"])
        penalty = np.maximum(rule["variance_floor_ratio"]*t_std-p_std, 0).mean()
    return {"cosine": float(cosine), "variance": float(penalty),
            "total": float(cosine+rule["variance_weight"]*penalty)}


def audit(args):
    root, out = args.source, args.run
    recipe = read_sealed(root/"configs/training/week6_jepa_predictor_v3.json")
    report = read_sealed(out/"report.json")
    identity = read_sealed(out/"identity.json")
    assert report["status"] == "passed" and report["recipe_digest"] == recipe["digest"] == identity["recipe_digest"]
    assert report["identity_digest"] == identity["digest"]
    for name, digest in report["files_sha256"].items():
        assert sha256_file(out/name) == digest, name
    for name, digest in {**recipe["input_sha256"], **identity["source_sha256"]}.items():
        assert sha256_file(root/name) == digest, name
    bank = torch.load(out/"diagnostic-bank.pt", map_location="cpu", weights_only=True)
    saved = torch.load(out/"predictions.pt", map_location="cpu", weights_only=True)
    assert bank["identities"] == read_sealed(out/"sample-identities.json")["samples"]
    assert bank["identities"] == read_sealed(root/"docs/evaluation/week5-pixel-predictor-v3/sample-identities.json")["samples"]
    dataset = JepaPairDataset(args.pairs, args.data_root/f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo",
                              split="train", expected_manifest_sha256=recipe["pair_manifest_sha256"],
                              normalizer=load_b0_action_normalizer(root/"docs/model/week4-fresh-v2/action-statistics.json"),
                              state_bounds=read_sealed(root/"configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"])
    rank = lambda value: hashlib.sha256(f"pixel-step3-diagnostic-v1:{value}".encode()).hexdigest()
    tasks = sorted({s["task_key"] for s in dataset.sources.values()}, key=rank)[:4]
    samples = []
    for task in tasks:
        source = min((s for s in dataset.sources.values() if s["task_key"] == task), key=lambda s: rank(s["episode_index"]))
        for h in HORIZONS:
            samples.append(dataset[dataset.ordinal(source["episode_index"], (source["episode_length"]-33)//2, h)])
    assert [s.identity for s in samples] == bank["identities"]
    rebuilt = collate_jepa_samples(samples, bank["condition"]["features"], bank["condition"]["feature_padding"], bank["targets"])
    assert torch.equal(rebuilt.anchor_groups, bank["anchor_groups"])
    for name, tensor in bank["condition"].items():
        assert torch.equal(getattr(rebuilt.condition, name), tensor), name
    for start in range(0, 20, 5):
        for i in range(start+1, start+5):
            assert torch.equal(bank["condition"]["features"][start], bank["condition"]["features"][i])
            assert torch.equal(bank["copy"][start], bank["copy"][i])
    torch.set_num_threads(1)
    model, header = load_jepa_checkpoint(out/"diagnostic-jepa.pt")
    assert header["digest"] == report["checkpoint_header_digest"]
    assert header["metadata"]["purpose"] == "step3-diagnostic-only-not-C_jepa"
    model.eval()
    condition = PixelCondition(**bank["condition"])
    with torch.no_grad():
        prediction = model(condition)
        restored = model(condition)
    assert torch.equal(prediction.features, restored.features)
    output_error = (prediction.features-saved["features"]).abs().max().item()
    token_error = (prediction.future_tokens-saved["future_tokens"]).abs().max().item()
    assert output_error < 1e-5 and token_error < 1e-4
    target, group_ids = bank["targets"].numpy(), bank["anchor_groups"].numpy()
    errors = []
    recomputed = numpy_metrics(prediction.features.numpy(), target, group_ids, recipe["loss"])
    for key, value in recomputed.items():
        errors.append(abs(value-report["real_train"]["final"][key]))
    for kind, saved_ablation in report["real_train"]["ablations"].items():
        with torch.no_grad():
            changed = model(condition, ablation=kind).features
        metrics = numpy_metrics(changed.numpy(), target, group_ids, recipe["loss"])
        metrics["mean_absolute_prediction_change"] = float(np.abs(changed.double().numpy()-prediction.features.double().numpy()).mean())
        for key, value in metrics.items():
            errors.append(abs(value-saved_ablation[key]))
    assert max(errors) < 1e-6
    # Fresh canonical teacher replay for every real current/target observation.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True)
    visual = full.model.visual
    del full
    gc.collect()
    teacher = FrozenQwenVisualTargets(visual.cuda(), load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True).image_processor)
    receipt = read_sealed(out/"feature-receipt.json")
    assert tensor_state_digest(visual.state_dict()) == receipt["teacher_before"] == receipt["teacher_after"]
    teacher_error = 0.
    for start in range(0, 20, 5):
        target_replay = teacher(np.stack([s.target_rgb for s in samples[start:start+5]])).cpu()
        current_replay = teacher(samples[start].observation.current_rgb[None]).cpu()
        teacher_error = max(teacher_error, (target_replay-bank["targets"][start:start+5]).abs().max().item(),
                            (current_replay[0]-bank["copy"][start]).abs().max().item())
    assert teacher_error == 0
    assert tensor_state_digest(visual.state_dict()) == receipt["teacher_after"]
    assert report["validation_payload_reads"] == report["test_payload_reads"] == 0
    return seal_contract({"artifact_type": "eventground_wam.jepa_predictor_independent_verification", "passed": True,
                          "report_digest": report["digest"], "recipe_digest": recipe["digest"],
                          "auditor_sha256": sha256_file(Path(__file__)), "training_pairs": 20,
                          "training_episodes": 4, "same_pairs_as_pixel_step3": True,
                          "condition_reconstruction_exact": True, "loss_arithmetic_max_error": max(errors),
                          "cpu_gpu_features_max_error": output_error, "cpu_gpu_future_tokens_max_error": token_error,
                          "fresh_teacher_replay_max_error": teacher_error, "cpu_repeat_exact": True,
                          "validation_payload_reads": 0, "test_payload_reads": 0,
                          "limits": "All 20 saved real pairs and their targets verified; training-only feasibility, not held-out predictor acceptance or production recovery."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    parser.add_argument("--data-root", type=Path, default=Path.home()/"data/eventground-wam/libero-plus")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, audit(args))
    print("Independent JEPA predictor verification passed", flush=True)
