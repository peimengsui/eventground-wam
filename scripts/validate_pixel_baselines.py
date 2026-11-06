#!/usr/bin/env python3
"""Independently audit baseline artifact coverage and replay selected real videos."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed, sha256_file, write_json
from eventground_wam.pixel_baselines import FIELDS, fixed_examples, measurement_rule

ROOT = Path(__file__).resolve().parents[1]


def primary_estimates(values):
    result = {"l1": values[0], "ssim": values[1], "motion_l1": values[2]/values[4],
              "static_l1": values[3]/values[5], "motion_area": values[4]}
    return {key: float(value) for key, value in result.items()}


def audit(args):
    from skimage.metrics import structural_similarity

    spec = importlib.util.spec_from_file_location("baseline_runner", ROOT / "scripts/run_pixel_baselines.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    torch.set_num_threads(1)
    completion = read_sealed(args.output / "completion.json")
    report = read_sealed(args.output / "report.json")
    identity = read_sealed(args.output / "identity.json")
    config = read_sealed(ROOT / "configs/evaluation/week5_pixel_baselines_v2.json")
    pair_manifest = read_sealed(args.pairs / "manifest.json")
    templates_manifest = read_sealed(args.output / "template-manifest.json")
    panel = read_sealed(args.output / "qualitative-panel.json")
    provenance = identity["provenance"]
    assert completion["status"] == report["status"] == "complete"
    assert completion["provenance"] == report["provenance"] == templates_manifest["provenance"] == provenance
    assert completion["report_sha256"] == sha256_file(args.output / "report.json")
    assert completion["report_digest"] == report["digest"]
    assert report["measurement"] == config["measurement"] == measurement_rule()
    assert config["digest"] == provenance["config_digest"] == panel["config_digest"]
    assert config["pair_manifest_sha256"] == sha256_file(args.pairs / "manifest.json")
    assert config["pair_manifest_digest"] == pair_manifest["digest"] == provenance["pair_manifest_digest"]
    assert sha256_file(args.pairs / "sources.jsonl") == provenance["sources_sha256"]
    for name, digest in provenance["source_code_sha256"].items():
        assert sha256_file(ROOT / name) == digest, name
    assert sha256_file(ROOT / "environments/b0/uv.lock") == provenance["runtime_lock_sha256"]
    sources = [json.loads(line) for line in (args.pairs / "sources.jsonl").read_text().splitlines()]
    assert {source["split"] for source in sources} == {"train", "validation"}
    train = [source for source in sources if source["split"] == "train"]
    validation = [source for source in sources if source["split"] == "validation"]
    assert not ({source["group_digest"] for source in train}
                & {source["group_digest"] for source in validation})
    assert (len(train), len(validation)) == (11435, 1528)
    expected_task_counts = {}
    for task in sorted({source["task_key"] for source in train}):
        selected = [source for source in train if source["task_key"] == task]
        expected_task_counts[task] = {"episodes": len(selected),
                                      "groups": len({source["group_digest"] for source in selected})}
    assert templates_manifest["task_counts"] == expected_task_counts
    assert templates_manifest["task_order"] == sorted(expected_task_counts)
    assert templates_manifest["training_camera_frames"] == 2 * sum(s["episode_length"] for s in train)
    assert templates_manifest["validation_payloads_used_for_fitting"] == 0
    assert templates_manifest["test_payloads_read"] == completion["test_payloads_read"] == 0
    assert completion["training_launched"] is False
    assert templates_manifest["templates_sha256"] == report["templates_sha256"] == sha256_file(args.output / "templates.npy")
    templates = np.load(args.output / "templates.npy", allow_pickle=False)
    assert templates.shape == (40, 2, 3, 128, 128) and templates.dtype == np.float32
    assert np.isfinite(templates).all() and templates.min() >= 0 and templates.max() <= 1
    assert panel["digest"] == completion["qualitative_panel_digest"]
    assert panel["examples"] == fixed_examples(validation) and len(panel["examples"]) == 480
    for name, digest in completion["preview_sha256"].items():
        assert sha256_file(args.output / "previews" / name) == digest
    assert len(completion["preview_sha256"]) == 8
    assert sha256_file(args.output / "episode-metrics.jsonl") == report["episode_metrics_sha256"]
    rows = [json.loads(line) for line in (args.output / "episode-metrics.jsonl").read_text().splitlines()]
    expected = {}
    for source in validation:
        for camera in CAMERAS:
            for h in HORIZONS:
                expected[(source["episode_index"], camera, h)] = source
    assert len(rows) == len(expected) == 15280
    by_episode = defaultdict(list)
    cells = defaultdict(int)
    for row in rows:
        source = expected.pop((row["episode_index"], row["camera"], row["horizon"]))
        assert row["source_digest"] == source["digest"]
        assert row["group_digest"] == source["group_digest"] and row["task_key"] == source["task_key"]
        assert row["pairs"] == source["episode_length"] - row["horizon"]
        cells[(row["camera"], row["horizon"])] += row["pairs"]
        by_episode[row["episode_index"]].append(row)
    assert not expected
    for (camera, h), count in cells.items():
        assert count == pair_manifest["counts"]["validation"]["pairs_by_horizon"][str(h)]
    for source in validation:
        saved = read_sealed(args.output / "episodes" / f"{source['episode_index']:06d}.json")
        assert saved["provenance"] == provenance and saved["source_digest"] == source["digest"]
        assert saved["templates_sha256"] == templates_manifest["templates_sha256"]
        assert saved["rows"] == by_episode[source["episode_index"]]

    # Independent nested means reproduce the primary point estimates, without
    # importing the optimized grouping/bootstrap implementation under audit.
    aggregates = {}
    for method in ("copy", "background"):
        task_values = []
        for task in sorted({row["task_key"] for row in rows}):
            task_rows = [row for row in rows if row["task_key"] == task]
            group_values = []
            for group in sorted({row["group_digest"] for row in task_rows}):
                group_rows = [row for row in task_rows if row["group_digest"] == group]
                # Every episode has all ten cells, so flattening these equal
                # axes is equivalent to their successive arithmetic means.
                group_values.append(np.mean([[row[method][f] for f in FIELDS]
                                             for row in group_rows], axis=0))
            task_values.append(np.mean(group_values, axis=0))
        values = np.mean(task_values, axis=0)
        result = primary_estimates(values)
        for field, value in result.items():
            assert abs(value - report["overall"][method][field]["value"]) < 1e-10
        aggregates[method] = result

    repository = args.data_root / f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo"
    replay_ids = sorted({int(name.removeprefix("episode-").removesuffix(".png"))
                         for name in completion["preview_sha256"]})
    max_replay_error = 0.
    max_ssim_error = 0.
    ssim_pairs = 0
    for index in replay_ids:
        source = next(source for source in validation if source["episode_index"] == index)
        _, videos = runner.validation_images(repository, source)
        template = templates[templates_manifest["task_order"].index(source["task_key"])]
        replay = runner.rows_for_episode(source, videos, template, "cpu")
        for actual, saved in zip(replay, by_episode[index], strict=True):
            for method in ("copy", "background"):
                for field in FIELDS:
                    max_replay_error = max(max_replay_error, abs(actual[method][field]-saved[method][field]))
        # Both cameras and horizons on the fixed middle-anchor panel, against
        # the independently implemented public SSIM reference on actual data.
        for example in panel["examples"]:
            if example["episode_index"] != index or example["anchor_fraction"] != .5:
                continue
            c = CAMERAS.index(example["camera"])
            current = videos[c][example["anchor"]]
            target = videos[c][example["target"]]
            for prediction in (current, template[c]):
                reference = structural_similarity(
                    prediction.transpose(1, 2, 0).astype(np.float64), target.transpose(1, 2, 0).astype(np.float64),
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                    data_range=1., channel_axis=2)
                actual = runner.PixelMetrics().ssim(torch.from_numpy(prediction)[None],
                                                    torch.from_numpy(target)[None]).item()
                max_ssim_error = max(max_ssim_error, abs(float(reference)-actual))
                ssim_pairs += 1
        print(json.dumps({"replayed_episode": index, "maximum_metric_error": max_replay_error}), flush=True)
    assert max_replay_error <= 5e-5 and max_ssim_error <= 5e-5
    return seal_contract({
        "artifact_type": "eventground_wam.pixel_baseline_verification", "status": "verified",
        "report_digest": report["digest"], "completion_digest": completion["digest"],
        "validator_sha256": sha256_file(Path(__file__)), "measurement_rows": len(rows),
        "matched_dual_camera_pairs": sum(cells.values())//2,
        "independent_primary_point_estimates": aggregates,
        "cpu_replayed_episode_indices": replay_ids, "maximum_cpu_replay_metric_error": max_replay_error,
        "real_image_ssim_reference_pairs": ssim_pairs, "maximum_real_image_ssim_error": max_ssim_error,
        "absolute_tolerance": 5e-5, "source_template_episode_panel_preview_hashes_verified": True,
        "training_and_validation_isolated": True, "test_payloads_read": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "data/eventground-wam/libero-plus")
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("verification receipt already exists")
    receipt = audit(args)
    write_json(args.receipt, receipt)
    print(json.dumps({"status": receipt["status"], "digest": receipt["digest"]}), flush=True)
