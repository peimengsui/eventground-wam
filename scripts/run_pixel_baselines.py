#!/usr/bin/env python3
"""Fit train-only image constants and score the complete frozen validation pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from eventground_wam.demonstration_contracts import (
    SOURCE_DATASET_REVISION,
    canonical_json_bytes,
    seal_contract,
)
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
from eventground_wam.pixel_baselines import (
    PixelMetrics,
    baseline_report,
    box_rgb,
    fit_templates,
    fixed_examples,
    measure_video,
    measurement_rule,
    motion_mask,
)

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "scripts/run_pixel_baselines.py", "src/eventground_wam/pixel_baselines.py",
    "src/eventground_wam/future_pairs.py", "src/eventground_wam/demonstration_reader.py",
    "src/eventground_wam/demonstration_contracts.py", "src/eventground_wam/__init__.py",
)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, value)
    temp.replace(path)


def runtime_metadata(device):
    # TorchVersion is a str subclass; the contract deliberately accepts only
    # plain JSON types, so normalize library version wrappers at the boundary.
    return {"torch": str(torch.__version__), "numpy": str(np.__version__), "device": str(device)}


def bounded_map(function, items, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(items)
        pending = deque()
        for item in iterator:
            pending.append(pool.submit(function, item))
            if len(pending) >= workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def decoded_camera(repository, source, camera, allowed_split):
    if source["split"] != allowed_split or allowed_split not in {"train", "validation"}:
        raise ValueError("source split is forbidden for this operation")
    pin = source["files"][camera]
    path = repository / pin["path"]
    if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
        raise ValueError(f"source video changed: {source['episode_index']}/{camera}")
    count = 0
    for frame in iter_video_frames(path):
        yield box_rgb(frame.payload.to_ndarray(format="rgb24"))
        count += 1
    if count != source["episode_length"]:
        raise ValueError("decoded video length differs from pair source")


def training_mean(repository, source):
    means = []
    for camera in CAMERAS:
        total = np.zeros((3, 128, 128), dtype=np.float64)
        for image in decoded_camera(repository, source, camera, "train"):
            total += image
        means.append(total / source["episode_length"])
    return source, np.stack(means)


def validation_images(repository, source):
    return source, [np.stack(list(decoded_camera(repository, source, camera, "validation")))
                    for camera in CAMERAS]


def load_inputs(args):
    config = read_sealed(ROOT / "configs/evaluation/week5_pixel_baselines_v2.json")
    if config["measurement"] != measurement_rule():
        raise ValueError("executable measurement rule changed")
    if sha256_file(args.pairs / "manifest.json") != config["pair_manifest_sha256"]:
        raise ValueError("pair manifest changed")
    manifest = read_sealed(args.pairs / "manifest.json")
    if manifest["digest"] != config["pair_manifest_digest"]:
        raise ValueError("pair version differs")
    split_root = ROOT / "configs/datasets/libero-plus-trajectory-v2"
    if sha256_file(split_root / "manifest.json") != manifest["provenance"]["split_manifest_sha256"]:
        raise ValueError("frozen split changed")
    split = read_sealed(split_root / "manifest.json")
    if sha256_file(split_root / "groups.jsonl") != split["outputs"]["groups.jsonl"]["sha256"]:
        raise ValueError("frozen groups changed")
    assignments = expand_split_groups(map(json.loads, (split_root / "groups.jsonl").read_text().splitlines()))
    verification = verify_bundle(args.pairs, assignments, manifest["provenance"])
    sources = list(map(json.loads, (args.pairs / "sources.jsonl").read_text().splitlines()))
    provenance = {
        "config_digest": config["digest"], "pair_manifest_digest": manifest["digest"],
        "sources_sha256": manifest["outputs"]["sources.jsonl"]["sha256"],
        "source_code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "runtime_lock_sha256": sha256_file(ROOT / "environments/b0/uv.lock"),
    }
    return config, manifest, sources, verification, provenance


@torch.inference_mode()
def reference_check(device):
    from skimage.metrics import structural_similarity

    rng = np.random.default_rng(17)
    pairs = []
    for _ in range(3):
        left = rng.random((128, 128, 3), dtype=np.float32)
        right = np.clip(left + rng.normal(0, .05, left.shape), 0, 1).astype(np.float32)
        pairs.append((left, right))
    pairs += [(np.zeros((128, 128, 3), np.float32), np.ones((128, 128, 3), np.float32)),
              (np.full((128, 128, 3), .4, np.float32), np.full((128, 128, 3), .4, np.float32))]
    almost_flat = np.broadcast_to(.73 + np.linspace(0, .0001, 128, dtype=np.float32)[None, :, None],
                                  (128, 128, 3)).copy()
    pairs.append((almost_flat, np.full_like(almost_flat, .73005)))
    scorer = PixelMetrics(device)
    errors = []
    for left, right in pairs:
        reference = structural_similarity(left.astype(np.float64), right.astype(np.float64), gaussian_weights=True, sigma=1.5,
                                          use_sample_covariance=False, data_range=1., channel_axis=2)
        actual = scorer.ssim(torch.from_numpy(left.transpose(2, 0, 1).copy())[None].to(device),
                             torch.from_numpy(right.transpose(2, 0, 1).copy())[None].to(device)).item()
        errors.append(abs(actual - float(reference)))
    if max(errors) > 5e-5:
        raise ValueError(f"SSIM reference mismatch: {errors}")
    return {"reference": "skimage.metrics.structural_similarity", "pairs": len(pairs),
            "maximum_absolute_error": max(errors), "tolerance": 5e-5}


def rows_for_episode(source, videos, template, device):
    rows = []
    for c, camera in enumerate(CAMERAS):
        measurements = measure_video(torch.from_numpy(videos[c]).to(device),
                                     torch.from_numpy(template[c]).to(device))
        for horizon in HORIZONS:
            rows.append({"episode_index": source["episode_index"], "source_digest": source["digest"],
                         "group_digest": source["group_digest"], "task_key": source["task_key"],
                         "camera": camera, "horizon": horizon,
                         "pairs": source["episode_length"] - horizon, **measurements[horizon]})
    return rows


def preview_sheet(output, source, videos, template, examples):
    from PIL import Image, ImageDraw

    selected = [row for row in examples if row["episode_index"] == source["episode_index"]
                and row["anchor_fraction"] == .5]
    if not selected:
        return
    canvas = Image.new("RGB", (1072, 1190), "#101827")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), f"{source['task_key']} | episode {source['episode_index']}", fill="white")
    draw.text((16, 30), source["instruction"], fill="white")
    for row_index, row in enumerate(selected):
        c = CAMERAS.index(row["camera"])
        current, target = videos[c][row["anchor"]], videos[c][row["target"]]
        mask = motion_mask(torch.from_numpy(current)[None], torch.from_numpy(target)[None])[0].numpy()
        overlay = np.repeat(mask[None], 3, axis=0).astype(np.float32)
        y = 70 + row_index * 278
        draw.text((16, y), f"{row['camera'].split('.')[-1]} | h={row['horizon']} | "
                  f"t={row['anchor']} -> {row['target']}", fill="white")
        for column, (label, image) in enumerate((("Current / copy", current),
                                                ("Train-only constant", template[c]),
                                                ("Future target", target), ("Change mask", overlay))):
            x = 16 + column * 264
            draw.text((x, y + 16), label, fill="#a7bad6")
            pixels = np.round(np.clip(image.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(pixels).resize((240, 240), Image.Resampling.NEAREST),
                         (x, y + 32))
    canvas.save(output / f"episode-{source['episode_index']:06d}.png")


def main(args):
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device unavailable")
        torch.cuda.reset_peak_memory_stats()
    config, manifest, sources, verification, provenance = load_inputs(args)
    repository = args.data_root / f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo"
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / "identity.json"
    if identity_path.exists():
        if read_sealed(identity_path)["provenance"] != provenance:
            raise ValueError("existing output belongs to another source/config")
        if (args.output / "completion.json").exists():
            raise FileExistsError("completed output is immutable")
    else:
        if list(args.output.iterdir()):
            raise FileExistsError("refusing unrecognized existing output")
        atomic_json(identity_path, seal_contract({"artifact_type": "eventground_wam.pixel_baseline_run",
                                                  "provenance": provenance}))

    def progress(phase, **extra):
        row = {"phase": phase, "elapsed_seconds": round(time.monotonic() - started, 3), **extra}
        atomic_json(args.output / "progress.json", row)
        print(json.dumps(row), flush=True)

    reference = reference_check(device)
    validation = [source for source in sources if source["split"] == "validation"]
    training = [source for source in sources if source["split"] == "train"]
    examples = fixed_examples(validation)
    panel = seal_contract({"artifact_type": "eventground_wam.pixel_qualitative_panel",
                           "config_digest": config["digest"], "examples": examples})
    atomic_json(args.output / "qualitative-panel.json", panel)
    # Bounded pilot exercises decode + optimized metrics before any large run.
    pilot_started = time.monotonic()
    first, videos = validation_images(repository, validation[0])
    pilot_rows = rows_for_episode(first, videos, np.zeros((2, 3, 128, 128), np.float32), device)
    pilot = seal_contract({"artifact_type": "eventground_wam.pixel_metric_pilot",
                           "reference_check": reference, "episode_index": first["episode_index"],
                           "pairs_per_camera": sum(first["episode_length"] - h for h in HORIZONS),
                           "wall_seconds": time.monotonic() - pilot_started,
                           "rows": len(pilot_rows), "synthetic_zero_background": True,
                           "not_production_baseline_evidence": True})
    atomic_json(args.output / "pilot.json", pilot)
    del videos, pilot_rows
    if args.pilot_only:
        progress("pilot-complete", **{k: pilot[k] for k in ("wall_seconds", "pairs_per_camera")})
        return
    template_path = args.output / "templates.npy"
    template_manifest_path = args.output / "template-manifest.json"
    if args.templates_from is not None and not template_manifest_path.exists():
        prior_path = args.templates_from / "template-manifest.json"
        prior = read_sealed(prior_path)
        reuse = config["template_reuse"]
        if (sha256_file(prior_path) != reuse["manifest_sha256"]
                or sha256_file(args.templates_from / "templates.npy") != reuse["templates_sha256"]
                or prior["templates_sha256"] != reuse["templates_sha256"]
                or any(prior["provenance"][k] != provenance[k] for k in
                       ("pair_manifest_digest", "sources_sha256", "runtime_lock_sha256"))
                or prior["training_episodes"] != len(training)
                or prior["validation_payloads_used_for_fitting"] != 0
                or prior["test_payloads_read"] != 0):
            raise ValueError("approved train-only template reuse identity changed")
        shutil.copyfile(args.templates_from / "templates.npy", template_path)
        cached = {key: value for key, value in prior.items() if key not in {"digest", "provenance"}}
        cached.update(provenance=provenance, fit_reused_from={
            "path": str(args.templates_from), "manifest_digest": prior["digest"],
            "manifest_sha256": reuse["manifest_sha256"], "reason": reuse["reason"],
        })
        atomic_json(template_manifest_path, seal_contract(cached))
        progress("reused-verified-train-only-templates", episodes=len(training))
    if template_manifest_path.exists():
        template_manifest = read_sealed(template_manifest_path)
        if (template_manifest["provenance"] != provenance
                or sha256_file(template_path) != template_manifest["templates_sha256"]):
            raise ValueError("cached templates changed")
        templates = np.load(template_path, allow_pickle=False)
        tasks = template_manifest["task_order"]
    else:
        progress("fit-train-only-templates", episodes=len(training))

        def means():
            for count, item in enumerate(bounded_map(lambda source: training_mean(repository, source),
                                                      training, args.workers), start=1):
                if count % 500 == 0:
                    progress("fit-train-only-templates", completed_episodes=count,
                             episodes=len(training))
                yield item

        templates, tasks, task_counts = fit_templates(training, means())
        temporary = template_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, templates, allow_pickle=False)
        temporary.replace(template_path)
        template_manifest = seal_contract({
            "artifact_type": "eventground_wam.pixel_background_templates", "provenance": provenance,
            "templates_sha256": sha256_file(template_path), "task_order": tasks,
            "task_counts": task_counts, "training_episodes": len(training),
            "training_groups": sum(value["groups"] for value in task_counts.values()),
            "training_camera_frames": 2 * sum(source["episode_length"] for source in training),
            "validation_payloads_used_for_fitting": 0, "test_payloads_read": 0,
        })
        atomic_json(template_manifest_path, template_manifest)
    task_index = {task: i for i, task in enumerate(tasks)}
    episode_root = args.output / "episodes"
    episode_root.mkdir(exist_ok=True)
    preview_root = args.output / "previews"
    preview_root.mkdir(exist_ok=True)
    preview_tasks = sorted(tasks, key=lambda task: hashlib.sha256(task.encode()).hexdigest())[:8]
    preview_ids = {row["episode_index"] for row in examples if row["task_key"] in preview_tasks}
    remaining = []
    results = {}
    for source in validation:
        path = episode_root / f"{source['episode_index']:06d}.json"
        if path.exists():
            result = read_sealed(path)
            if (result["provenance"] != provenance or result["source_digest"] != source["digest"]
                    or result["templates_sha256"] != template_manifest["templates_sha256"]):
                raise ValueError("cached episode result changed")
            results[source["episode_index"]] = result
        else:
            remaining.append(source)
    progress("measure-validation", completed_episodes=len(results), episodes=len(validation))
    for source, videos in bounded_map(lambda source: validation_images(repository, source), remaining, 2):
        template = templates[task_index[source["task_key"]]]
        rows = rows_for_episode(source, videos, template, device)
        if source["episode_index"] in preview_ids:
            preview_sheet(preview_root, source, videos, template, examples)
        result = seal_contract({"artifact_type": "eventground_wam.pixel_episode_baselines",
                                "provenance": provenance, "source_digest": source["digest"],
                                "templates_sha256": template_manifest["templates_sha256"], "rows": rows})
        atomic_json(episode_root / f"{source['episode_index']:06d}.json", result)
        results[source["episode_index"]] = result
        if len(results) % 50 == 0:
            progress("measure-validation", completed_episodes=len(results), episodes=len(validation))
    rows = [row for index in sorted(results) for row in results[index]["rows"]]
    if len(rows) != len(validation) * len(CAMERAS) * len(HORIZONS):
        raise ValueError("measurement row coverage changed")
    for camera in CAMERAS:
        for h in HORIZONS:
            if sum(row["pairs"] for row in rows if row["camera"] == camera and row["horizon"] == h) != (
                manifest["counts"]["validation"]["pairs_by_horizon"][str(h)]
            ):
                raise ValueError("measured pair coverage differs")
    rows_path = args.output / "episode-metrics.jsonl"
    with rows_path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
    progress("aggregate-paired-group-uncertainty")
    report = baseline_report(rows)
    if (report["numeric_groups"] != 180 or report["tasks"] != 40
            or report["motion_groups"] < 120 or report["motion_tasks"] != 40):
        raise ValueError("insufficient evaluation/motion coverage for the frozen gate")
    report = seal_contract({"artifact_type": "eventground_wam.pixel_baselines", "status": "complete",
                            "provenance": provenance, "measurement": config["measurement"],
                            "episode_metrics_sha256": sha256_file(rows_path),
                            "templates_sha256": template_manifest["templates_sha256"],
                            "validation_pairs_per_camera": manifest["counts"]["validation"]["pairs"],
                            "validation_episodes": len(validation), **report})
    atomic_json(args.output / "report.json", report)
    if any(sha256_file(ROOT / name) != digest for name, digest in provenance["source_code_sha256"].items()):
        raise ValueError("executed source changed during run")
    previews = {path.name: sha256_file(path) for path in sorted(preview_root.glob("*.png"))}
    if len(previews) != 8:
        raise ValueError("qualitative previews are incomplete")
    completion = seal_contract({
        "artifact_type": "eventground_wam.pixel_baseline_completion", "status": "complete",
        "provenance": provenance, "report_digest": report["digest"],
        "report_sha256": sha256_file(args.output / "report.json"),
        "pair_verification_digest": verification["digest"],
        "reference_check": reference, "qualitative_panel_digest": panel["digest"],
        "preview_sha256": previews,
        "source_video_hashes_and_decodes": {
            "train": 0 if "fit_reused_from" in template_manifest else len(training) * 2,
            "validation": len(validation) * 2},
        "template_fit_reused": template_manifest.get("fit_reused_from"),
        "test_payloads_read": 0, "training_launched": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gpu_peak_torch_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "gpu_peak_torch_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else 0,
        "runtime": runtime_metadata(device),
    })
    atomic_json(args.output / "completion.json", completion)
    progress("complete", report_digest=report["digest"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "data/eventground-wam/libero-plus")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--templates-from", type=Path,
                        help="Reuse only the exact train-only template artifact pinned in protocol v2")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main(args)
