#!/usr/bin/env python3
"""Fit frozen sampled means and score every validation pair with durable receipts."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from check_jepa_targets import runtime_versions
from check_pixel_predictor import verify_base_files

from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.demonstration_reader import iter_video_frames
from eventground_wam.future_pairs import (
    CAMERAS,
    HORIZONS,
    expand_split_groups,
    read_sealed,
    sha256_file,
    verify_bundle,
    write_json,
)
from eventground_wam.jepa_baselines import (
    FIELDS,
    RETRIEVAL_HORIZONS,
    baseline_report,
    collapse_diagnostics,
    cosine_errors,
    error_masses,
    measurement_rule,
    representative_sources,
    retrieval_metrics,
    retrieval_pool,
    retrieval_scores,
    token_motion_weights,
)
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.pixel_baselines import box_rgb
from eventground_wam.pixel_predictor import tensor_state_digest

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    write_json(temp, value)
    os.replace(temp, path)


def atomic_array(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return sha256_file(path)


def progress(output, phase, **details):
    value = {"phase": phase, "time_unix": time.time(), **details}
    atomic_json(output/"progress.json", value)
    print(json.dumps(value), flush=True)


def verify_pilot(pilot, pilot_identity, full_identity):
    if (not pilot["passed"] or not pilot["teacher_unchanged"]
            or pilot["identity_digest"] != pilot_identity["digest"]
            or pilot_identity["phase"] != "pilot" or full_identity["phase"] != "full"
            or any(pilot_identity[key] != full_identity[key] for key in
                   ("config_digest", "pair_manifest_digest", "source_sha256", "versions"))):
        raise ValueError("passed pilot for this exact code/config/runtime/data identity required")


def decode(repository, source, split, frames=None):
    if split not in ("train", "validation") or source["split"] != split:
        raise ValueError("forbidden or cross-split video access")
    selected = set(range(source["episode_length"])) if frames is None else set(frames)
    if not selected or min(selected) < 0 or max(selected) >= source["episode_length"]:
        raise ValueError("invalid selected frames")
    videos = []
    for camera in CAMERAS:
        pin = source["files"][camera]
        path = repository/pin["path"]
        if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
            raise ValueError("source camera payload changed")
        images, count = [], 0
        for index, frame in enumerate(iter_video_frames(path)):
            count += 1
            if index in selected:
                rgb = frame.payload.to_ndarray(format="rgb24")
                if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                    raise ValueError("unexpected native RGB layout")
                images.append(rgb)
        if count != source["episode_length"] or len(images) != len(selected):
            raise ValueError("source frame count or selected coverage changed")
        videos.append(np.stack(images))
    return np.stack(videos, axis=1)


def encode(teacher, rgb):
    features = np.empty((len(rgb), 128, 2048), np.float32)
    for start in range(0, len(rgb), 8):
        features[start:start+8] = teacher(rgb[start:start+8]).cpu().numpy()
    norms = np.linalg.norm(features, axis=-1)
    if not np.isfinite(features).all() or np.max(np.abs(norms-1)) > 1e-6:
        raise ValueError("target teacher emitted invalid normalized features")
    return features


def episode_rows(source, features, boxes, template):
    rows = []
    length = source["episode_length"]
    for c, camera in enumerate(CAMERAS):
        view = features[:, c*64:(c+1)*64]
        for h in HORIZONS:
            sums = {m: np.zeros(5, np.float64) for m in ("copy", "mean")}
            count = length-h
            for start in range(0, count, 32):
                stop = min(start+32, count)
                target = view[start+h:stop+h]
                weights = token_motion_weights(boxes[start:stop, c], boxes[start+h:stop+h, c])
                for method, values in (("copy", view[start:stop]),
                                       ("mean", np.broadcast_to(template[c*64:(c+1)*64], target.shape))):
                    mass = error_masses(cosine_errors(values, target), weights)
                    sums[method] += np.asarray([mass[f] for f in FIELDS])*(stop-start)
            rows.append({"episode_index": source["episode_index"], "source_digest": source["digest"],
                         "group_digest": source["group_digest"], "task_key": source["task_key"],
                         "camera": camera, "horizon": h, "pairs": count,
                         **{m: dict(zip(FIELDS, (s/count).tolist(), strict=True)) for m, s in sums.items()}})
    return rows


def episode_retrieval(source, rgb, features, template):
    queries, current_panel, target_panel = [], [], []
    hashes = [[hashlib.sha256(rgb[t, c].tobytes()).hexdigest() for t in range(len(rgb))] for c in range(2)]
    for h in RETRIEVAL_HORIZONS:
        for fraction in (.25, .5, .75):
            anchor = int(fraction*(len(rgb)-h-1))
            target = anchor+h
            current_panel.append(features[anchor])
            target_panel.append(features[target])
            for c, camera in enumerate(CAMERAS):
                view = features[:, c*64:(c+1)*64]
                pool = retrieval_pool(view, hashes[c], episode=source["episode_index"],
                                      camera=camera, anchor=anchor, horizon=h)
                row = {"episode_index": source["episode_index"], "source_digest": source["digest"],
                       "group_digest": source["group_digest"], "task_key": source["task_key"],
                       "camera": camera, "horizon": h, "anchor_fraction": fraction,
                       "anchor": anchor, "target": target, "pool": pool}
                if pool["eligible"]:
                    for method, query in (("copy", view[anchor]), ("mean", template[c*64:(c+1)*64])):
                        scores = retrieval_scores(query, view[pool["frames"]])
                        row[method] = {**retrieval_metrics(scores, pool["target_slot"]), "scores": scores.tolist()}
                queries.append(row)
    means = np.stack([np.mean(current_panel, axis=0, dtype=np.float64),
                      np.mean(target_panel, axis=0, dtype=np.float64), template]).astype(np.float32)
    return queries, means


def load_inputs(args):
    config = read_sealed(ROOT/"configs/evaluation/week6_jepa_baselines_v1.json")
    if config["measurement"] != measurement_rule():
        raise ValueError("measurement rules differ from frozen protocol")
    for name, expected in config["input_sha256"].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError(f"frozen dependency changed: {name}")
    manifest = read_sealed(args.pairs/"manifest.json")
    if sha256_file(args.pairs/"manifest.json") != config["pair_manifest_sha256"]:
        raise ValueError("future pair manifest changed")
    split_root = ROOT/"configs/datasets/libero-plus-trajectory-v2"
    split = read_sealed(split_root/"manifest.json")
    if (sha256_file(split_root/"manifest.json") != manifest["provenance"]["split_manifest_sha256"]
            or sha256_file(split_root/"groups.jsonl") != split["outputs"]["groups.jsonl"]["sha256"]):
        raise ValueError("trajectory split identities changed")
    assignments = expand_split_groups(map(json.loads, (split_root/"groups.jsonl").read_text().splitlines()))
    verification = verify_bundle(args.pairs, assignments, manifest["provenance"])
    sources = [json.loads(line) for line in (args.pairs/"sources.jsonl").read_text().splitlines()]
    if any(s["split"] not in ("train", "validation") for s in sources):
        raise ValueError("test payload source admitted")
    train = representative_sources([s for s in sources if s["split"] == "train"], "train", "week6-jepa-mean-v1")
    validation = sorted([s for s in sources if s["split"] == "validation"], key=lambda s: s["episode_index"])
    panel = representative_sources(validation, "validation", "week6-jepa-retrieval-v1")
    if (len(train), len(validation), len(panel)) != (1341, 1528, 180):
        raise ValueError("frozen source coverage changed")
    return config, manifest, verification, train, validation, panel


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


def run(args):
    if (args.output/"completion.json").exists():
        raise FileExistsError("completed baseline output is immutable")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    versions = runtime_versions()
    if versions["torch_loaded_build"] != "2.13.0+cu130" or versions["transformers"] != "5.17.0":
        raise ValueError("locked Week 5 CUDA runtime required")
    started = time.monotonic()
    progress(args.output, "verify-inputs")
    config, manifest, bundle_check, train, validation, panel = load_inputs(args)
    code = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted((ROOT/"src/eventground_wam").glob("*.py"))}
    code.update({p: sha256_file(ROOT/p) for p in ("scripts/run_jepa_baselines.py", "scripts/check_jepa_targets.py", "scripts/check_pixel_predictor.py")})
    identity = seal_contract({"artifact_type": "eventground_wam.jepa_baseline_run_identity",
                              "phase": args.phase, "config_digest": config["digest"],
                              "pair_manifest_digest": manifest["digest"], "source_sha256": code, "versions": versions})
    if (args.output/"identity.json").exists() and read_sealed(args.output/"identity.json") != identity:
        raise ValueError("cannot resume a run with changed identities")
    atomic_json(args.output/"identity.json", identity)
    repository = args.data_root/f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo"
    model_files = verify_base_files(args.cache, read_sealed(ROOT/"docs/model/Qwen3.5-2B-manifest.json"))
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True)
    visual = full.model.visual
    del full
    teacher = FrozenQwenVisualTargets(visual.to("cuda"), load_pinned_qwen_processor(
        cache_dir=str(args.cache), local_files_only=True).image_processor).eval()
    before = tensor_state_digest(visual.state_dict())
    target_receipt = read_sealed(ROOT/"docs/evaluation/week6-jepa-targets-v2/report.json")
    if before != target_receipt["vision_state_after"]:
        raise ValueError("teacher weights differ from Step 1 acceptance")
    for label, obj in (("vision", type(visual)), ("processor", type(teacher.image_processor))):
        if sha256_file(Path(inspect.getfile(obj))) != target_receipt["implementation_files"][label]["sha256"]:
            raise ValueError("teacher runtime implementation changed")
    if args.phase == "pilot":
        # Entire short/long TRAIN episodes; no validation payloads for profiling.
        chosen = [min(train, key=lambda s: s["episode_length"]), max(train, key=lambda s: s["episode_length"])]
        measurements = []
        for source in chosen:
            start = time.monotonic()
            rgb = decode(repository, source, "train")
            features = encode(teacher, rgb)
            mean = features.mean(axis=0, dtype=np.float64)
            mean = (mean/np.linalg.norm(mean, axis=-1, keepdims=True)).astype(np.float32)
            boxes = np.stack([[box_rgb(x) for x in frame] for frame in rgb])
            rows = episode_rows(source, features, boxes, mean)
            queries, _ = episode_retrieval(source, rgb, features, mean)
            elapsed = time.monotonic()-start
            measurements.append({"episode": source["episode_index"], "frames": len(rgb),
                                 "seconds": elapsed, "seconds_per_frame": elapsed/len(rgb),
                                 "rows": len(rows), "retrieval_queries": len(queries)})
            progress(args.output, "pilot", **measurements[-1])
        total_frames = sum(s["episode_length"] for s in validation)+16*len(train)
        projection = max(m["seconds_per_frame"] for m in measurements)*total_frames*1.5
        free = shutil.disk_usage(args.output).free
        passed = projection < 6*3600 and torch.cuda.max_memory_allocated() < 4*1024**3 and free > 12*1024**3
        result = seal_contract({"artifact_type": "eventground_wam.jepa_baseline_pilot", "identity_digest": identity["digest"],
                                "config_digest": config["digest"], "measurements": measurements,
                                "projected_seconds_with_1_5_margin": projection,
                                "planned_encoded_dual_frames": total_frames, "free_disk_bytes": free,
                                "maximum_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "teacher_unchanged": before == tensor_state_digest(visual.state_dict()), "passed": passed})
        atomic_json(args.output/"pilot.json", result)
        if not passed or not result["teacher_unchanged"]:
            raise ValueError("baseline pilot failed")
        return
    pilot = read_sealed(args.pilot)
    verify_pilot(pilot, read_sealed(args.pilot.parent/"identity.json"), identity)
    for subdir in ("train-means", "episodes", "panels", "replay"):
        (args.output/subdir).mkdir(exist_ok=True)
    tasks = sorted({s["task_key"] for s in train})
    sums = {t: np.zeros((128, 2048), np.float64) for t in tasks}
    counts = {t: 0 for t in tasks}
    train_records = []
    for index, source in enumerate(train):
        frame_ids = np.unique(np.linspace(0, source["episode_length"]-1, 16).astype(int)).tolist()
        path = args.output/"train-means"/f"{source['episode_index']:06d}.npy"
        receipt_path = path.with_suffix(".json")
        expected = {"episode_index": source["episode_index"], "source_digest": source["digest"],
                    "group_digest": source["group_digest"], "task_key": source["task_key"],
                    "split": "train", "frames": frame_ids, "identity_digest": identity["digest"]}
        if receipt_path.exists():
            receipt = read_sealed(receipt_path)
            if any(receipt[k] != v for k, v in expected.items()) or sha256_file(path) != receipt["array_sha256"]:
                raise ValueError("training mean receipt changed")
            mean = np.load(path, allow_pickle=False)
        else:
            rgb = decode(repository, source, "train", frame_ids)
            mean = encode(teacher, rgb).mean(axis=0, dtype=np.float64)
            digest = atomic_array(path, mean)
            receipt = seal_contract({"artifact_type": "eventground_wam.jepa_train_group_mean", **expected, "array_sha256": digest})
            atomic_json(receipt_path, receipt)
        if mean.shape != (128, 2048) or not np.isfinite(mean).all():
            raise ValueError("bad saved training mean")
        sums[source["task_key"]] += mean
        counts[source["task_key"]] += 1
        train_records.append(receipt)
        if (index+1) % 20 == 0 or index+1 == len(train):
            progress(args.output, "fit-training-means", groups_complete=index+1, groups_total=len(train),
                     elapsed_seconds=time.monotonic()-started)
    templates = np.stack([sums[t]/counts[t] for t in tasks])
    norms = np.linalg.norm(templates, axis=-1, keepdims=True)
    if np.any(norms <= 1e-6):
        raise ValueError("degenerate training mean direction")
    templates = (templates/norms).astype(np.float32)
    template_sha = atomic_array(args.output/"templates.npy", templates)
    template_record = seal_contract({"artifact_type": "eventground_wam.jepa_mean_templates",
                                     "identity_digest": identity["digest"], "task_order": tasks,
                                     "templates_sha256": template_sha, "training_groups": len(train),
                                     "training_frames": sum(len(r["frames"]) for r in train_records),
                                     "training_records": train_records})
    atomic_json(args.output/"templates.json", template_record)
    del sums
    panel_ids = {s["episode_index"] for s in panel}
    replay_ids = {validation[0]["episode_index"], max(validation, key=lambda s: s["episode_length"])["episode_index"]}
    rows, queries, panel_values, panel_groups = [], [], [], []
    for index, source in enumerate(validation):
        path = args.output/"episodes"/f"{source['episode_index']:06d}.json"
        panel_path = args.output/"panels"/f"{source['episode_index']:06d}.npy"
        if path.exists():
            saved = read_sealed(path)
            if (saved["identity_digest"] != identity["digest"] or saved["source_digest"] != source["digest"]
                    or saved["templates_sha256"] != template_sha):
                raise ValueError("validation receipt identity changed")
            for relative, digest in saved["arrays"].items():
                if sha256_file(args.output/relative) != digest:
                    raise ValueError("saved panel or replay array changed")
        else:
            if shutil.disk_usage(args.output).free < 8*1024**3:
                raise ValueError("insufficient artifact storage")
            rgb = decode(repository, source, "validation")
            features = encode(teacher, rgb)
            boxes = np.stack([[box_rgb(x) for x in frame] for frame in rgb])
            template = templates[tasks.index(source["task_key"])]
            measured = episode_rows(source, features, boxes, template)
            episode_queries, arrays = [], {}
            if source["episode_index"] in panel_ids:
                episode_queries, panel_mean = episode_retrieval(source, rgb, features, template)
                arrays[str(panel_path.relative_to(args.output))] = atomic_array(panel_path, panel_mean)
            if source["episode_index"] in replay_ids:
                for name, value in (("features", features), ("boxes", boxes)):
                    dest = args.output/"replay"/f"{source['episode_index']:06d}-{name}.npy"
                    arrays[str(dest.relative_to(args.output))] = atomic_array(dest, value)
            saved = seal_contract({"artifact_type": "eventground_wam.jepa_baseline_episode", "identity_digest": identity["digest"],
                                    "source_digest": source["digest"], "templates_sha256": template_sha,
                                    "rows": measured, "queries": episode_queries, "arrays": arrays})
            atomic_json(path, saved)
            del features, rgb, boxes
        rows.extend(saved["rows"])
        queries.extend(saved["queries"])
        if source["episode_index"] in panel_ids:
            panel_values.append(np.load(panel_path, allow_pickle=False))
            panel_groups.append((source["task_key"], source["group_digest"]))
        if (index+1) % 5 == 0 or index+1 == len(validation):
            progress(args.output, "score-validation", episodes_complete=index+1, episodes_total=len(validation),
                     pairs_complete=sum(r["pairs"] for r in rows)//2, elapsed_seconds=time.monotonic()-started)
    if sum(r["pairs"] for r in rows)//2 != 1109994 or len(rows) != 15280 or len(queries) != 3240:
        raise ValueError("incomplete full validation or fixed retrieval panel")
    progress(args.output, "aggregate-and-bootstrap")
    panel_bank = np.stack(panel_values)
    collapse = {m: collapse_diagnostics(panel_bank[:, i], panel_groups) for i, m in enumerate(("copy", "target", "mean"))}
    report = baseline_report(rows, queries, collapse)
    after = tensor_state_digest(visual.state_dict())
    if before != after or any(p.requires_grad or p.grad is not None for p in visual.parameters()):
        raise ValueError("teacher changed during baseline evaluation")
    atomic_json(args.output/"episode-metrics.json", seal_contract({"artifact_type": "eventground_wam.jepa_baseline_rows", "rows": rows}))
    atomic_json(args.output/"retrieval-panel.json", seal_contract({"artifact_type": "eventground_wam.jepa_retrieval_panel",
                                                                   "identity_digest": identity["digest"], "queries": queries}))
    report = seal_contract({"artifact_type": "eventground_wam.jepa_baselines_report", **report,
                            "identity_digest": identity["digest"], "config_digest": config["digest"],
                            "dual_camera_pairs": 1109994, "episodes": len(validation),
                            "templates_digest": template_record["digest"], "target_encoder_state_sha256": before,
                            "target_encoder_unchanged": True, "test_payload_reads": 0, "optimizer_updates": 0,
                            "elapsed_seconds_this_process": time.monotonic()-started,
                            "maximum_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "maximum_torch_reserved_bytes": torch.cuda.max_memory_reserved()})
    atomic_json(args.output/"report.json", report)
    completion = seal_contract({"artifact_type": "eventground_wam.jepa_baselines_run_completion",
                                "report_digest": report["digest"], "identity_digest": identity["digest"],
                                "bundle_verification": bundle_check, "verified_model_files": model_files,
                                "replay_episode_indices": sorted(replay_ids), "panel_groups": [list(g) for g in panel_groups],
                                "files_sha256": {name: sha256_file(args.output/name) for name in
                                                 ("templates.npy", "templates.json", "episode-metrics.json", "retrieval-panel.json", "report.json")}})
    atomic_json(args.output/"completion.json", completion)
    progress(args.output, "complete", report_digest=report["digest"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home()/"data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    if args.phase == "full" and args.pilot is None:
        parser.error("--pilot is required for the full run")
    main(args)
