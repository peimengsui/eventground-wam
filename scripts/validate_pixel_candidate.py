#!/usr/bin/env python3
"""Independent aggregates/bootstrap, real-image metric checks, and inference replay."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from train_pixel_cpt import configure

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed, sha256_file, write_json
from eventground_wam.pixel_baselines import FIELDS, PixelMetrics
from eventground_wam.pixel_data import FutureInputBuilder, FuturePairDataset
from eventground_wam.pixel_predictor import (
    FrozenQwenFeatures,
    load_pixel_checkpoint,
    tensor_state_digest,
)
from eventground_wam.pixel_training import encode_training_batch, slice_condition


def independent_aggregate(baseline, candidate, report, criteria):
    """Use explicit nested means and sampled groups, not production weight matrices."""
    joined = {}
    for row in baseline:
        key = row["episode_index"], row["camera"], row["horizon"]
        assert key not in joined
        joined[key] = dict(row)
    for row in candidate:
        key = row["episode_index"], row["camera"], row["horizon"]
        ref = joined[key]
        for field in ("source_digest", "group_digest", "task_key", "pairs"):
            assert row[field] == ref[field]
        ref["prediction"] = row["prediction"]
    assert all("prediction" in row for row in joined.values())
    groups = defaultdict(lambda: defaultdict(list))
    for row in joined.values():
        groups[row["task_key"]][row["group_digest"]].append(row)
    methods = ("copy", "background", "prediction")
    point = {m: np.zeros((2, 5, 6)) for m in methods}
    draws = {m: np.zeros((10000, 2, 5, 6)) for m in methods}
    rng = np.random.default_rng(20260925)
    motion_groups = 0
    motion_tasks = set()
    for task in sorted(groups):
        numeric = groups[task]
        sampled = rng.integers(len(numeric), size=(10000, len(numeric)))
        for method in methods:
            arrays = []
            for group in sorted(numeric):
                rows = numeric[group]
                array = np.array([[np.mean([[r[method][f] for f in FIELDS] for r in rows
                                           if r["camera"] == camera and r["horizon"] == horizon], axis=0)
                                   for horizon in HORIZONS] for camera in CAMERAS])
                arrays.append(array)
            arrays = np.stack(arrays)
            if method == "copy":
                supported = arrays[:, :, 2:, 4].mean(axis=(1, 2)) > 0
                motion_groups += int(supported.sum())
                if supported.any():
                    motion_tasks.add(task)
            point[method] += arrays.mean(axis=0)/len(groups)
            draws[method] += arrays[sampled].mean(axis=1)/len(groups)
    def values(masses):
        return {"l1": masses[..., 0], "ssim": masses[..., 1],
                "motion_l1": masses[..., 2]/masses[..., 4], "static_l1": masses[..., 3]/masses[..., 5],
                "motion_area": masses[..., 4]}
    overall = {m: values(point[m].mean(axis=(0, 1))) for m in methods}
    bootstrap = {m: values(draws[m].mean(axis=(1, 2))) for m in methods}
    def check_summary(estimate, resamples, expected):
        for key, value in estimate.items():
            assert abs(float(value)-expected[key]["value"]) < 1e-10
            np.testing.assert_allclose(np.quantile(resamples[key], [.025, .975]),
                                       expected[key]["group_bootstrap_95"], atol=1e-10, rtol=0)
    check_summary(overall["prediction"], bootstrap["prediction"], report["overall"])
    motion_points = {m: values(point[m][:, 2:].mean(axis=(0, 1))) for m in methods}
    check_summary(motion_points["prediction"], values(draws["prediction"][:, :, 2:].mean(axis=(1, 2))),
                  report["motion_h8_h16_h32"])
    assert len(report["by_camera_horizon"]) == 10
    for cell in report["by_camera_horizon"]:
        c, h = CAMERAS.index(cell["camera"]), HORIZONS.index(cell["horizon"])
        check_summary(values(point["prediction"][c, h]), values(draws["prediction"][:, c, h]), cell["prediction"])
    assert motion_groups >= criteria["minimum_motion_groups"]
    assert len(motion_tasks) == criteria["required_motion_task_count"]
    assert report["numeric_groups"] == sum(len(g) for g in groups.values())
    assert report["tasks"] == len(groups)
    for method in ("copy", "prediction"):
        np.testing.assert_allclose(point[method][..., 0], report["gate"]["per_camera_horizon_l1"][method], atol=1e-10, rtol=0)
    effects = {}
    for name, base, motion, ssim in (("l1_vs_copy", "copy", False, False),
                                    ("motion_l1_vs_copy", "copy", True, False),
                                    ("ssim_vs_copy", "copy", False, True),
                                    ("l1_vs_background", "background", False, False)):
        field = 1 if ssim else (2 if motion else 0)
        span = slice(2, None) if motion else slice(None)
        a = point[base][:, span, field].mean(); b = point["prediction"][:, span, field].mean()
        x = draws[base][:, :, span, field].mean(axis=(1, 2))
        y = draws["prediction"][:, :, span, field].mean(axis=(1, 2))
        actual = {"value": float(b-a if ssim else (a-b)/a),
                  "paired_95": np.quantile(y-x if ssim else (x-y)/x, [.025, .975]).tolist()}
        assert abs(actual["value"]-report["gate"]["effects"][name]["value"]) < 1e-10
        np.testing.assert_allclose(actual["paired_95"], report["gate"]["effects"][name]["paired_95"], atol=1e-10, rtol=0)
        effects[name] = actual
    checks = {
        "overall_l1": effects["l1_vs_copy"]["value"] >= criteria["l1_relative_gain_vs_copy_minimum"] and effects["l1_vs_copy"]["paired_95"][0] > 0,
        "motion_l1": effects["motion_l1_vs_copy"]["value"] >= criteria["motion_l1_relative_gain_vs_copy_minimum"] and effects["motion_l1_vs_copy"]["paired_95"][0] > 0,
        "ssim": effects["ssim_vs_copy"]["value"] >= 0 and effects["ssim_vs_copy"]["paired_95"][0] >= -.005,
        "beats_background": effects["l1_vs_background"]["value"] > 0 and effects["l1_vs_background"]["paired_95"][0] > 0,
        "all_camera_horizon_cells": bool(np.all(point["prediction"][..., 0] <= criteria["maximum_per_camera_horizon_l1_ratio_vs_copy"]*point["copy"][..., 0])),
    }
    assert checks == report["gate"]["checks"]
    assert report["gate"]["status"] == ("PASS" if all(checks.values()) else "FAIL")
    return {"point_estimates": {m: {k: float(v) for k, v in row.items()} for m, row in overall.items()},
            "medium_long_horizon_point_estimates": {m: {k: float(v) for k, v in row.items()} for m, row in motion_points.items()},
            "motion_groups": motion_groups, "motion_tasks": len(motion_tasks),
            "camera_horizon_summaries_verified": 10,
            "paired_effects": effects, "independently_recomputed_checks": checks, "bootstrap_resamples": 10000}


@torch.inference_mode()
def audit(args):
    from skimage.metrics import structural_similarity
    runtime = configure()
    root = args.source
    completion = read_sealed(args.output / "completion.json")
    report = read_sealed(args.output / "report.json")
    identity = read_sealed(args.output / "identity.json")
    assert runtime == identity["runtime"]
    assert completion["report_digest"] == report["digest"]
    assert completion["identity_digest"] == report["identity_digest"] == identity["digest"]
    for name, digest in identity["source_sha256"].items():
        assert sha256_file(root / name) == digest
    assert completion["predictor_and_base_unchanged"] and completion["optimizer_updates"] == completion["test_payloads_read"] == 0
    protocol = read_sealed(root / "configs/evaluation/week5_pixel_baselines_v2.json")
    assert protocol["digest"] == identity["protocol_digest"]
    assert sha256_file(args.output / "episode-metrics.jsonl") == report["episode_metrics_sha256"]
    candidate_rows = [json.loads(line) for line in (args.output / "episode-metrics.jsonl").read_text().splitlines()]
    baseline_report = read_sealed(args.baselines / "report.json")
    assert baseline_report["digest"] == identity["baseline_report_digest"]
    assert sha256_file(args.baselines / "episode-metrics.jsonl") == baseline_report["episode_metrics_sha256"]
    baseline_rows = [json.loads(line) for line in (args.baselines / "episode-metrics.jsonl").read_text().splitlines()]
    dataset = FuturePairDataset(args.pairs, args.data_root / f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo",
                               split="validation", expected_manifest_sha256=protocol["pair_manifest_sha256"], cache_episodes=1,
                               normalizer=load_b0_action_normalizer(root / "docs/model/week4-fresh-v2/action-statistics.json"),
                               state_bounds=read_sealed(root / "configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"])
    expected = {(ep, c, h): s for ep, s in dataset.sources.items() for c in CAMERAS for h in HORIZONS}
    by_episode = defaultdict(list)
    for row in candidate_rows:
        source = expected.pop((row["episode_index"], row["camera"], row["horizon"]))
        assert row["pairs"] == source["episode_length"]-row["horizon"]
        assert row["source_digest"] == source["digest"]
        by_episode[row["episode_index"]].append(row)
    assert not expected and len(candidate_rows) == 15280 and len(by_episode) == 1528
    assert sum(row["pairs"] for row in candidate_rows) == 2*1109994
    assert len(completion["episode_receipts"]) == 1528
    for pin in completion["episode_receipts"]:
        path = args.output / "episodes" / f"{pin['episode_index']:06d}.json"
        assert sha256_file(path) == pin["sha256"]
        saved = read_sealed(path)
        assert saved["digest"] == pin["digest"] and saved["identity_digest"] == identity["digest"]
        assert saved["rows"] == by_episode[pin["episode_index"]]
        for name, digest in saved["qualitative_sha256"].items():
            assert sha256_file(args.output / "qualitative" / name) == digest
    aggregates = independent_aggregate(baseline_rows, candidate_rows, report, protocol["measurement"]["exit_gate"])
    panel = read_sealed(root / "configs/evaluation/week5_pixel_qualitative_v2.json")
    assert panel["digest"] == identity["qualitative_panel_digest"]
    examples = defaultdict(list)
    for e in panel["examples"]:
        examples[e["episode_index"]].append(e)
    scorer = PixelMetrics("cuda")
    maximum_error, maximum_ssim_error, checked = 0., 0., 0
    for ep in sorted(examples):
        saved = read_sealed(args.output / "episodes" / f"{ep:06d}.json")
        assert saved["qualitative_examples"] == examples[ep]
        predictions = np.load(args.output / "qualitative" / f"episode-{ep:06d}.npz", allow_pickle=False)["predictions"]
        assert predictions.shape == (12, 3, 128, 128) and np.isfinite(predictions).all()
        for example, predicted in zip(examples[ep], predictions, strict=True):
            sample = dataset[dataset.ordinal(ep, example["anchor"], example["horizon"])]
            c = CAMERAS.index(example["camera"])
            current = sample.observation.current_images[c].numpy(); target = sample.target_images[c].numpy()
            current_sum = np.rint(current*1020).astype(np.int32)
            target_sum = np.rint(target*1020).astype(np.int32)
            mask = np.abs(target_sum-current_sum).sum(axis=0) > 120
            if mask.sum()*100 < mask.size:
                mask[:] = False
            error = np.abs(predicted-target).mean(axis=0).astype(np.float64)
            expected_values = {"l1": error.mean(), "motion_error_mass": (error*mask).mean(),
                               "static_error_mass": (error*~mask).mean(), "motion_area": mask.mean(), "static_area": (~mask).mean()}
            reference = structural_similarity(predicted.transpose(1, 2, 0).astype(np.float64),
                                               target.transpose(1, 2, 0).astype(np.float64), gaussian_weights=True,
                                               sigma=1.5, use_sample_covariance=False, data_range=1., channel_axis=2)
            actual = scorer.paired(*[torch.from_numpy(x)[None].to("cuda") for x in (current, target, predicted)])
            maximum_error = max(maximum_error, *(abs(actual[k].item()-float(v)) for k, v in expected_values.items()))
            maximum_ssim_error = max(maximum_ssim_error, abs(actual["ssim"].item()-float(reference)))
            checked += 1
        print(json.dumps({"audited_qualitative_episode": ep, "images_checked": checked}), flush=True)
    assert checked == 480 and maximum_error <= 5e-7 and maximum_ssim_error <= 5e-5
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).to("cuda")
    encoder = FrozenQwenFeatures(full.model).to("cuda")
    builder = FutureInputBuilder(load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True))
    model, header = load_pixel_checkpoint(args.candidate)
    assert sha256_file(args.candidate) == identity["candidate_sha256"]
    model = model.eval().requires_grad_(False).to("cuda")
    baseline_completion = read_sealed(args.baselines / "completion.json")
    replay_ids = sorted(int(name.removeprefix("episode-").removesuffix(".png")) for name in baseline_completion["preview_sha256"])
    replayed = 0; max_prediction_error = 0.
    for ep in replay_ids:
        source = dataset.sources[ep]
        predictions = np.load(args.output / "qualitative" / f"episode-{ep:06d}.npz", allow_pickle=False)["predictions"]
        selected = {(e["anchor"], e["horizon"], CAMERAS.index(e["camera"])): i for i, e in enumerate(examples[ep])}
        for begin in sorted({e["anchor"]//8*8 for e in examples[ep]}):
            samples = [dataset[dataset.ordinal(ep, t, h)] for t in range(begin, min(begin+8, source["episode_length"]-2))
                       for h in HORIZONS if t+h < source["episode_length"]]
            condition, _, _ = encode_training_batch(samples, encoder, builder, torch.device("cuda"), feature_microbatch=4)
            for offset in range(0, len(samples), 10):
                images = model(slice_condition(condition, offset, offset+10)).images.cpu().numpy()
                for j, sample in enumerate(samples[offset:offset+10]):
                    for c in range(2):
                        key = sample.identity["anchor"], sample.identity["horizon"], c
                        if key in selected:
                            max_prediction_error = max(max_prediction_error, float(np.abs(images[j, c]-predictions[selected[key]]).max()))
                            replayed += 1
        print(json.dumps({"inference_replayed_episode": ep, "maximum_pixel_error": max_prediction_error}), flush=True)
    assert replayed == 96 and max_prediction_error <= 1e-6
    assert tensor_state_digest(model.state_dict()) == header["state_sha256"]
    return seal_contract({"artifact_type": "eventground_wam.pixel_candidate_independent_verification", "status": "passed",
                          "report_digest": report["digest"], "completion_digest": completion["digest"],
                          "validator_sha256": sha256_file(Path(__file__)), "candidate_sha256": identity["candidate_sha256"],
                          "episode_rows_verified": len(candidate_rows), "dual_camera_pairs_verified": 1109994,
                          "independent_aggregation": aggregates, "qualitative_metric_reference_images": checked,
                          "maximum_numpy_metric_error": maximum_error, "maximum_skimage_ssim_error": maximum_ssim_error,
                          "inference_replay_episode_indices": replay_ids, "inference_replay_images": replayed,
                          "maximum_inference_replay_pixel_error": max_prediction_error,
                          "test_payloads_read": 0, "optimizer_updates": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "baselines", "pairs", "candidate", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home() / "data/eventground-wam/huggingface-cache")
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    receipt = audit(args)
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))
