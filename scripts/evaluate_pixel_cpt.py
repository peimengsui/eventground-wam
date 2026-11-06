#!/usr/bin/env python3
"""Score the fixed final pixel candidate on every frozen validation pair."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from check_pixel_predictor import verify_base_files
from run_pixel_baselines import reference_check
from train_pixel_cpt import configure

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.b0_training import atomic_json
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed, sha256_file
from eventground_wam.pixel_baselines import (
    assess_candidate,
    fixed_examples,
    measurement_rule,
    motion_mask,
)
from eventground_wam.pixel_data import FutureInputBuilder, FuturePairDataset
from eventground_wam.pixel_evaluation import EpisodeAccumulator, prediction_summary
from eventground_wam.pixel_predictor import (
    FrozenQwenFeatures,
    load_pixel_checkpoint,
    tensor_state_digest,
)
from eventground_wam.pixel_training import encode_training_batch, slice_condition

ROOT = Path(__file__).resolve().parents[1]


def source_pins():
    paths = sorted(ROOT.glob("src/eventground_wam/*.py"))
    paths += [ROOT / "scripts" / name for name in ("evaluate_pixel_cpt.py", "train_pixel_cpt.py",
              "run_pixel_baselines.py", "check_pixel_predictor.py")]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def render_panel(path, source, examples, images, predictions, template):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    width, row_height = 866, 173
    canvas = Image.new("RGB", (width, 100 + row_height * len(examples)), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{source['task_key']} | episode {source['episode_index']} | fixed validation panel", fill="#172b46", font=font)
    instruction = source["instruction"]
    for line, start in enumerate(range(0, min(len(instruction), 210), 105)):
        draw.text((12, 31 + 16*line), instruction[start:start+105], fill="#34445b", font=font)
    labels = ("Current / copy", "Train-only constant", "Prediction", "Future target", "Change mask")
    for col, label in enumerate(labels):
        draw.text((12 + col*171, 77), label, fill="#172b46", font=font)
    for i, example in enumerate(examples):
        c = CAMERAS.index(example["camera"])
        current, target = images[example["anchor"], c], images[example["target"], c]
        mask = motion_mask(torch.from_numpy(current)[None], torch.from_numpy(target)[None])[0].numpy()
        y = 100 + i*row_height
        draw.text((12, y), f"{example['camera'].split('.')[-1]}  h={example['horizon']}  t={example['anchor']}->{example['target']}  fraction={example['anchor_fraction']}", fill="#34445b", font=font)
        for col, value in enumerate((current, template[c], predictions[i], target, np.repeat(mask[None], 3, axis=0))):
            rgb = np.round(np.clip(value.transpose(1, 2, 0), 0, 1)*255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb).resize((148, 148), Image.Resampling.NEAREST), (12 + col*171, y+19))
    canvas.save(path)


@torch.inference_mode()
def run(args):
    started = time.monotonic()
    runtime = configure()
    protocol = read_sealed(ROOT / "configs/evaluation/week5_pixel_baselines_v2.json")
    assert protocol["measurement"] == measurement_rule()
    verified = read_sealed(args.verification)
    training = read_sealed(args.run / "report.json")
    assert verified["status"] == "passed" and verified["report_digest"] == training["digest"]
    assert training["runtime"] == runtime
    candidate = args.run / "C_pixel-candidate.pt"
    candidate_sha = sha256_file(candidate)
    assert candidate_sha == verified["candidate_sha256"]
    baseline_completion = read_sealed(args.baselines / "completion.json")
    baseline_report = read_sealed(args.baselines / "report.json")
    assert baseline_completion["report_digest"] == baseline_report["digest"]
    assert baseline_report["digest"] == read_sealed(ROOT / "docs/evaluation/week5-pixel-baselines-report-v2.json")["digest"]
    assert baseline_completion["provenance"]["config_digest"] == protocol["digest"]
    assert sha256_file(args.baselines / "episode-metrics.jsonl") == baseline_report["episode_metrics_sha256"]
    baseline_rows = [json.loads(line) for line in (args.baselines / "episode-metrics.jsonl").read_text().splitlines()]
    baseline_by_episode = {}
    for row in baseline_rows:
        baseline_by_episode.setdefault(row["episode_index"], []).append(row)
    template_manifest = read_sealed(args.baselines / "template-manifest.json")
    assert sha256_file(args.baselines / "templates.npy") == template_manifest["templates_sha256"] == baseline_report["templates_sha256"]
    templates = np.load(args.baselines / "templates.npy", allow_pickle=False)
    dataset = FuturePairDataset(args.pairs, args.data_root / f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo",
                               split="validation", expected_manifest_sha256=protocol["pair_manifest_sha256"], cache_episodes=1,
                               normalizer=load_b0_action_normalizer(ROOT / "docs/model/week4-fresh-v2/action-statistics.json"),
                               state_bounds=read_sealed(ROOT / "configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"])
    panel = read_sealed(ROOT / "configs/evaluation/week5_pixel_qualitative_v2.json")
    assert panel["examples"] == fixed_examples(list(dataset.sources.values()))
    assert panel["digest"] == baseline_completion["qualitative_panel_digest"]
    examples_by_episode = {}
    for example in panel["examples"]:
        examples_by_episode.setdefault(example["episode_index"], []).append(example)
    code = source_pins()
    identity = seal_contract({"artifact_type": "eventground_wam.pixel_candidate_evaluation_identity",
                              "candidate_sha256": candidate_sha, "training_report_digest": training["digest"],
                              "verification_digest": verified["digest"], "protocol_digest": protocol["digest"],
                              "pair_manifest_digest": dataset.manifest["digest"], "baseline_report_digest": baseline_report["digest"],
                              "qualitative_panel_digest": panel["digest"], "source_sha256": code, "runtime": runtime,
                              "anchor_batch": 8, "feature_microbatch": 4, "prediction_microbatch": 10,
                              "split": "validation", "checkpoint_selection": "fixed-final-update-221633"})
    identity_path = args.output / "identity.json"
    if identity_path.exists():
        assert read_sealed(identity_path) == identity
        if (args.output / "completion.json").exists():
            raise FileExistsError("completed evaluation is immutable")
    else:
        if any(p.name not in {"owner.lock", "service.log"} for p in args.output.iterdir()):
            raise FileExistsError("unrecognized evaluation output")
        atomic_json(identity_path, identity)
    for name in ("episodes", "qualitative"):
        (args.output / name).mkdir(exist_ok=True)
    reference = reference_check("cuda")
    verify_base_files(args.cache, read_sealed(ROOT / "docs/model/Qwen3.5-2B-manifest.json"))
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).to("cuda")
    encoder = FrozenQwenFeatures(full.model).to("cuda")
    builder = FutureInputBuilder(load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True))
    base_before = tensor_state_digest(encoder.core.state_dict())
    assert base_before == training["base_state_sha256_before"]
    model, header = load_pixel_checkpoint(candidate)
    assert header == verified["candidate_header"]
    model = model.eval().requires_grad_(False).to("cuda")
    predictor_before = tensor_state_digest(model.state_dict())
    torch.cuda.reset_peak_memory_stats()
    sources = sorted(dataset.sources.values(), key=lambda s: s["episode_index"])
    assert len(sources) == 1528 and len(dataset) == 1109994
    all_rows, episode_receipts = [], []
    pairs_done, newly_scored = 0, 0
    started_scoring = time.monotonic()
    for source in sources:
        ep = source["episode_index"]
        path = args.output / "episodes" / f"{ep:06d}.json"
        expected_pairs = sum(source["episode_length"] - h for h in HORIZONS)
        if path.exists():
            saved = read_sealed(path)
            assert saved["identity_digest"] == identity["digest"] and saved["source_digest"] == source["digest"]
            for name, digest in saved["qualitative_sha256"].items():
                assert sha256_file(args.output / "qualitative" / name) == digest
        else:
            episode_start = time.monotonic()
            accumulator = EpisodeAccumulator(source, "cuda")
            examples = examples_by_episode.get(ep, [])
            selected = {(e["anchor"], e["horizon"], CAMERAS.index(e["camera"])): i for i, e in enumerate(examples)}
            predictions = np.empty((len(examples), 3, 128, 128), np.float32)
            collected = set()
            for begin in range(0, source["episode_length"] - 2, 8):
                samples = [dataset[dataset.ordinal(ep, anchor, h)]
                           for anchor in range(begin, min(begin+8, source["episode_length"]-2))
                           for h in HORIZONS if anchor+h < source["episode_length"]]
                condition, targets, _ = encode_training_batch(samples, encoder, builder, torch.device("cuda"), feature_microbatch=4)
                for offset in range(0, len(samples), 10):
                    stop = min(offset+10, len(samples))
                    part = slice_condition(condition, offset, stop)
                    images = model(part).images
                    identities = [s.identity for s in samples[offset:stop]]
                    accumulator.add(identities, part.current_images, targets[offset:stop], images)
                    for i, item in enumerate(identities):
                        for camera in range(2):
                            key = item["anchor"], item["horizon"], camera
                            if key in selected:
                                panel_index = selected[key]
                                predictions[panel_index] = images[i, camera].float().cpu().numpy()
                                collected.add(panel_index)
                del condition, targets, samples, part, images
            rows = accumulator.finish()
            reference_rows = {(r["camera"], r["horizon"]): r for r in baseline_by_episode[ep]}
            for row in rows:
                ref = reference_rows[row["camera"], row["horizon"]]
                assert all(row[k] == ref[k] for k in ("episode_index", "source_digest", "group_digest", "task_key", "pairs"))
                assert all(abs(row["prediction"][k]-ref["copy"][k]) <= 1e-7 for k in ("motion_area", "static_area"))
            qualitative_sha = {}
            if examples:
                assert collected == set(range(len(examples)))
                archive = args.output / "qualitative" / f"episode-{ep:06d}.npz"
                temp = archive.with_suffix(".partial")
                with temp.open("wb") as handle:
                    np.savez_compressed(handle, predictions=predictions)
                    handle.flush(); os.fsync(handle.fileno())
                temp.replace(archive)
                from eventground_wam.pixel_baselines import box_rgb
                videos = dataset._episode(source)[2]
                current_images = np.stack([np.stack([box_rgb(frame[c]) for c in range(2)]) for frame in videos])
                image_path = archive.with_suffix(".png")
                render_panel(image_path, source, examples, current_images, predictions,
                             templates[template_manifest["task_order"].index(source["task_key"])] )
                qualitative_sha = {p.name: sha256_file(p) for p in (archive, image_path)}
            torch.cuda.synchronize()
            saved = seal_contract({"artifact_type": "eventground_wam.pixel_candidate_episode",
                                   "identity_digest": identity["digest"], "source_digest": source["digest"],
                                   "rows": rows, "pair_count": expected_pairs, "qualitative_examples": examples,
                                   "qualitative_sha256": qualitative_sha, "elapsed_seconds": time.monotonic()-episode_start})
            atomic_json(path, saved)
            newly_scored += expected_pairs
        assert saved["pair_count"] == expected_pairs and len(saved["rows"]) == 10
        all_rows.extend(saved["rows"])
        episode_receipts.append({"episode_index": ep, "digest": saved["digest"], "sha256": sha256_file(path)})
        pairs_done += expected_pairs
        elapsed = time.monotonic()-started_scoring
        projected = elapsed/max(1, newly_scored)*len(dataset)
        gpu_free, gpu_total = torch.cuda.mem_get_info()
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 >= 24*1024**3 or (gpu_total-gpu_free)/gpu_total >= .85:
            raise RuntimeError("evaluation resource cap exceeded at committed episode boundary")
        if shutil.disk_usage(args.output).free < 5*1024**3 or time.monotonic()-started > 8*3600:
            raise RuntimeError("evaluation disk reserve or eight-hour process limit reached")
        progress = {"status": "running", "episodes": len(episode_receipts), "total_episodes": len(sources),
                    "pairs": pairs_done, "total_pairs": len(dataset), "elapsed_seconds": time.monotonic()-started,
                    "projected_full_scoring_hours": projected/3600,
                    "gpu_allocated_bytes": torch.cuda.memory_allocated(), "last_episode": ep}
        atomic_json(args.output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    assert pairs_done == 1109994 and len(all_rows) == 15280
    before = args.output / "episode-metrics.jsonl.partial"
    with before.open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    before.replace(args.output / "episode-metrics.jsonl")
    report = seal_contract({"artifact_type": "eventground_wam.pixel_candidate_validation", "status": "complete",
                            "identity_digest": identity["digest"], "candidate_sha256": candidate_sha,
                            "episode_metrics_sha256": sha256_file(args.output / "episode-metrics.jsonl"),
                            "gate": assess_candidate(baseline_rows, all_rows), **prediction_summary(all_rows),
                            "validation_episodes": len(sources), "dual_camera_pairs": pairs_done,
                            "test_payloads_read": 0, "qualitative_review": "pending-human-visible-inspection"})
    assert tensor_state_digest(model.state_dict()) == predictor_before
    assert tensor_state_digest(encoder.core.state_dict()) == base_before
    assert sha256_file(candidate) == candidate_sha and source_pins() == code
    atomic_json(args.output / "report.json", report)
    completion = seal_contract({"artifact_type": "eventground_wam.pixel_candidate_evaluation_completion",
                                "status": "complete", "identity_digest": identity["digest"], "report_digest": report["digest"],
                                "candidate_sha256": candidate_sha, "reference_check": reference,
                                "episode_receipts": episode_receipts, "predictor_and_base_unchanged": True,
                                "total_process_seconds": time.monotonic()-started,
                                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                "test_payloads_read": 0, "optimizer_updates": 0})
    atomic_json(args.output / "completion.json", completion)
    atomic_json(args.output / "progress.json", {"status": "complete", "report_digest": report["digest"], "gate": report["gate"]["status"]})
    print(json.dumps({"status": "complete", "gate": report["gate"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "baselines", "run", "verification", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home() / "data/eventground-wam/huggingface-cache")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "owner.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
