#!/usr/bin/env python3
"""Training-only Week 6 target feasibility; no optimizer or learned predictor."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from check_pixel_predictor import verify_base_files

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import HORIZONS, read_sealed, sha256_file, write_json
from eventground_wam.jepa_features import (
    TARGET_VERSION,
    FrozenQwenVisualTargets,
    anchor_feature_diagnostics,
    normalize_visual_targets,
)
from eventground_wam.pixel_data import FuturePairDataset
from eventground_wam.pixel_predictor import tensor_state_digest

ROOT = Path(__file__).resolve().parents[1]


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "numpy", "av", "pillow")}
    # TorchVersion subclasses str, but sealed contracts require plain JSON types.
    versions["torch_loaded_build"] = str(torch.__version__)
    return versions


def progress(output, phase, **details):
    record = {"phase": phase, **details}
    write_json(output/"progress.json", record)
    print(json.dumps(record), flush=True)


def patch_layout_check(teacher):
    # Every 32x32 merged region carries its camera/row/column identity.
    rows, cols = np.indices((8, 8))
    grids = np.stack([np.stack([rows*31, cols*31, np.full((8, 8), c*255)], axis=-1)
                      for c in range(2)]).astype(np.uint8)
    rgb = grids.repeat(32, axis=1).repeat(32, axis=2)[None]
    pixels = teacher.preprocess(rgb)["pixel_values"].reshape(2, 64, 4, 3, 2, 16, 16)
    actual = pixels.mean(dim=(2, 4, 5, 6))
    expected = torch.from_numpy(grids.reshape(2, 64, 3)).float()/127.5-1
    return float((actual-expected).abs().max())


def pair_metrics(current, future):
    return (1-(current.double()*future.double()).sum(dim=-1)).reshape(2, 64).mean(dim=1)


def main(args):
    contract = read_sealed(ROOT/"configs/training/week6_jepa_targets_v2.json")
    if contract["target"]["version"] != TARGET_VERSION:
        raise ValueError("target contract version mismatch")
    for name, expected in contract["input_sha256"].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError(f"pinned input changed: {name}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError("target feasibility requires the pinned CUDA host")
    versions = runtime_versions()
    if (versions["transformers"] != "5.17.0" or versions["torch"].split("+")[0] != "2.13.0"
            or torch.__version__ != "2.13.0+cu130"):
        raise ValueError("use the locked Week 5 CUDA runtime")
    source_files = {str(p.relative_to(ROOT)): sha256_file(p)
                    for p in sorted((ROOT/"src/eventground_wam").glob("*.py"))}
    source_files.update({p: sha256_file(ROOT/p) for p in
                         ("scripts/check_jepa_targets.py", "scripts/check_pixel_predictor.py")})
    identity = seal_contract({"artifact_type": "eventground_wam.jepa_target_diagnostic_identity",
                              "contract_digest": contract["digest"], "versions": versions,
                              "source_files": source_files})
    write_json(args.output/"identity.json", identity)
    progress(args.output, "verify-frozen-inputs")
    model_files = verify_base_files(args.cache, read_sealed(ROOT/"docs/model/Qwen3.5-2B-manifest.json"))
    state = read_sealed(ROOT/"configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"]
    dataset = FuturePairDataset(
        args.pairs,
        args.data_root/f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo",
        split="train", expected_manifest_sha256=contract["pair_manifest_sha256"],
        normalizer=load_b0_action_normalizer(ROOT/"docs/model/week4-fresh-v2/action-statistics.json"),
        state_bounds=state)
    rank = lambda episode: hashlib.sha256(f"jepa-target-step1-v1:{episode}".encode()).hexdigest()
    tasks = sorted({source["task_key"] for source in dataset.sources.values()})
    if len(tasks) != 40:
        raise ValueError("expected all 40 training tasks")
    selected = [min((s for s in dataset.sources.values()
                     if s["task_key"] == task and s["episode_length"] >= 37),
                    key=lambda s: rank(s["episode_index"])) for task in tasks]
    samples = []
    for source in selected:
        anchors = [((source["episode_length"]-33)*n)//4 for n in (1, 2, 3)]
        if len(set(anchors)) != 3:
            raise ValueError("diagnostic anchors must be distinct")
        for anchor in anchors:
            samples.append({"task_key": source["task_key"], "episode_index": source["episode_index"],
                            "anchor": anchor, "horizons": list(HORIZONS),
                            "source_digest": source["digest"], "split": "train"})
    write_json(args.output/"selection.json", seal_contract({"artifact_type": "eventground_wam.jepa_target_selection",
                                                           "samples": samples,
                                                           "contract_digest": contract["digest"]}))
    progress(args.output, "load-frozen-vision", anchors=len(samples))
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True)
    visual = full.model.visual
    del full  # Keep only the target vision tower; no text model is executed.
    teacher = FrozenQwenVisualTargets(visual.to("cuda"), load_pinned_qwen_processor(
        cache_dir=str(args.cache), local_files_only=True).image_processor).eval()
    implementation_files = {}
    for label, obj in (("vision", type(visual)), ("processor", type(teacher.image_processor))):
        path = Path(inspect.getfile(obj))
        implementation_files[label] = {"path": str(path), "sha256": sha256_file(path)}
    before = tensor_state_digest(teacher.visual.state_dict())
    layout_error = patch_layout_check(teacher)
    metrics, bank, bank_ids, anchor_bank_indices = [], [], [], []
    repeat_error = batch_error = permutation_error = maximum_norm_error = 0.
    norm_min, norm_max = float("inf"), 0.
    for index, selection in enumerate(samples):
        ep, anchor = selection["episode_index"], selection["anchor"]
        source = dataset.sources[ep]
        # Resolve each pair through the frozen index before accessing its raw RGB.
        resolved = [dataset[dataset.ordinal(ep, anchor, h)] for h in HORIZONS]
        _, _, videos = dataset._episode(source)
        times = [anchor, *[anchor+h for h in HORIZONS]]
        rgb = videos[times].copy()
        raw = teacher.encode_raw(rgb)
        norm_min = min(norm_min, float(raw.float().norm(dim=-1).min()))
        norm_max = max(norm_max, float(raw.float().norm(dim=-1).max()))
        features = normalize_visual_targets(raw)
        maximum_norm_error = max(maximum_norm_error, float((features.norm(dim=-1)-1).abs().max()))
        if index == 0:
            repeat_error = float((features-teacher(rgb)).abs().max())
            singles = torch.cat([teacher(rgb[i:i+1]) for i in range(len(rgb))])
            batch_error = float((features-singles).abs().max())
            swapped = teacher(rgb[:, ::-1].copy()).reshape(6, 2, 64, 2048).flip(1).reshape(6, 128, 2048)
            permutation_error = float((features-swapped).abs().max())
            torch.save({"rgb": torch.from_numpy(rgb), "raw": raw.cpu(),
                        "normalized": features.cpu()}, args.output/"first-observation-check.pt")
        features = features.cpu()
        anchor_bank_indices.append(len(bank_ids))
        bank.append(features)
        bank_ids.extend({**selection, "frame": frame} for frame in times)
        for i, sample in enumerate(resolved, start=1):
            error = pair_metrics(features[0], features[i])
            metrics.append({**sample.identity, "front_cosine_distance": float(error[0]),
                            "wrist_cosine_distance": float(error[1])})
        if (index+1) % 3 == 0:
            progress(args.output, "encode-training-sample", tasks_complete=(index+1)//3,
                     anchors_complete=index+1)
    all_features = torch.cat(bank)
    bank_path = args.output/"feature-bank.pt"
    torch.save({"features": all_features, "identities": bank_ids,
                "anchor_indices": anchor_bank_indices, "contract_digest": contract["digest"]}, bank_path)
    anchor_features = all_features[anchor_bank_indices]
    diagnostics = {camera: anchor_feature_diagnostics(anchor_features[:, c*64:(c+1)*64])
                   for c, camera in enumerate(("front", "wrist"))}
    cells = {f"{camera}/h{h}": float(np.mean([row[f"{camera}_cosine_distance"]
                                            for row in metrics if row["horizon"] == h]))
             for camera in ("front", "wrist") for h in HORIZONS}
    # Independently reproducible rank spectrum, without downloading the feature bank.
    for c, camera in enumerate(("front", "wrist")):
        pooled = anchor_features[:, c*64:(c+1)*64].double().mean(dim=1)
        centered = pooled-pooled.mean(dim=0)
        diagnostics[camera]["pooled_centered_singular_values"] = torch.linalg.svdvals(centered).tolist()
    after = tensor_state_digest(teacher.visual.state_dict())
    gates = contract["diagnostic"]["criteria"]
    checks = {
        "coverage": len(metrics) == 600 and len(dataset.decoded_episodes) == 40,
        "train_only": all(row["split"] == "train" for row in metrics),
        "finite_unit_targets": maximum_norm_error <= gates["unit_norm_error_max"],
        "patch_camera_row_major_alignment": layout_error <= gates["preprocess_error_max"],
        "exact_repeat": repeat_error == 0,
        "batch_composition_consistency": batch_error <= gates["normalized_batch_max_absolute_error"],
        "camera_permutation_consistency": permutation_error <= gates["normalized_batch_max_absolute_error"],
        "frozen_base": before == after and not any(p.requires_grad or p.grad is not None
                                                   for p in teacher.parameters()),
        "nonconstant_across_observations": all(
            d["pooled_effective_rank"] >= gates["pooled_effective_rank_min"]
            and d["fixed_position_across_example_std_rms"] >= gates["fixed_position_std_rms_min"]
            for d in diagnostics.values()),
        "temporal_signal_both_cameras": all(cells[f"{c}/h32"] >= gates["h32_copy_cosine_distance_min"]
                                           for c in ("front", "wrist")),
    }
    report = seal_contract({
        "artifact_type": "eventground_wam.jepa_target_feasibility", "contract_digest": contract["digest"],
        "identity_digest": identity["digest"], "checks": checks, "passed": all(checks.values()),
        "tasks": len(tasks), "episodes": len(dataset.decoded_episodes), "anchors": len(samples),
        "pairs": len(metrics), "encoded_dual_observations": len(all_features),
        "target_shape_per_pair": [128, 2048], "target_dtype": "float32",
        "target_encoder_parameters": sum(p.numel() for p in teacher.parameters()),
        "trainable_parameters": 0, "optimizer_updates": 0,
        "validation_payload_reads": 0, "test_payload_reads": 0,
        "model_files": model_files, "implementation_files": implementation_files,
        "vision_state_before": before, "vision_state_after": after,
        "raw_token_norm_range": [norm_min, norm_max], "maximum_unit_norm_error": maximum_norm_error,
        "patch_layout_max_error": layout_error, "repeat_max_error": repeat_error,
        "batch_max_error": batch_error, "camera_permutation_max_error": permutation_error,
        "camera_diagnostics": diagnostics, "copy_cosine_distance_by_camera_horizon": cells,
        "feature_bank_sha256": sha256_file(bank_path), "feature_bank_bytes": bank_path.stat().st_size,
        "elapsed_seconds": time.monotonic()-started,
        "torch_cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "limits": "Training-only teacher feasibility; no predictor acceptance, retrieval gate or policy-quality claim.",
    })
    write_json(args.output/"pair-diagnostics.json", seal_contract({"artifact_type": "eventground_wam.jepa_target_pairs",
                                                                  "pairs": metrics}))
    write_json(args.output/"report.json", report)
    progress(args.output, "complete" if report["passed"] else "failed", passed=report["passed"])
    if not report["passed"]:
        raise ValueError("target feasibility failed; retain evidence and revise explicitly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home()/"data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
