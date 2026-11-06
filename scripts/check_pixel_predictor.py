#!/usr/bin/env python3
"""Bounded Step 3 implementation checks; never production CPT or policy training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import (
    QWEN_REVISION,
    load_pinned_qwen_model,
    load_pinned_qwen_processor,
)
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import HORIZONS, read_sealed, sha256_file, write_json
from eventground_wam.pixel_baselines import PixelMetrics
from eventground_wam.pixel_data import (
    FEATURE_PROMPT,
    FutureInputBuilder,
    FuturePairDataset,
    collate_future_samples,
)
from eventground_wam.pixel_predictor import (
    FrozenQwenFeatures,
    PixelCondition,
    PixelPredictor,
    PixelPredictorConfig,
    load_pixel_checkpoint,
    pixel_reconstruction_loss,
    save_pixel_checkpoint,
    tensor_state_digest,
)

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ["scripts/check_pixel_predictor.py", *[f"src/eventground_wam/{name}.py" for name in
              ("pixel_predictor", "pixel_data", "pixel_baselines", "future_pairs", "b0_model",
               "b0_data", "split_sample_index", "demonstration_reader", "demonstration_contracts", "__init__")]]


def progress(output, phase, **extra):
    record = {"phase": phase, **extra}
    temporary = output / "progress.tmp"
    write_json(temporary, record)
    temporary.replace(output / "progress.json")
    print(json.dumps(record), flush=True)


@torch.inference_mode()
def score(model, condition, targets, ablation="none"):
    model.eval()
    predictions = model(condition, ablation=ablation).images
    metrics = PixelMetrics(predictions.device).paired(
        condition.current_images.flatten(0, 1), targets.flatten(0, 1), predictions.flatten(0, 1))
    area = metrics["motion_area"].mean().item()
    return {"l1": metrics["l1"].mean().item(), "ssim": metrics["ssim"].mean().item(),
            "motion_l1": metrics["motion_error_mass"].mean().item()/area if area else None}, predictions


def fit(model, condition, targets, *, steps, learning_rate, output, phase):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    initial, _ = score(model, condition, targets)
    gradient_activity = {"action_projection": False, "action_sequence_projection": False,
                         "horizon_projection": False,
                         "feature_projection": False, "current_patches": False,
                         "state_projection": False}
    started = time.monotonic()
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(condition).images
        loss = pixel_reconstruction_loss(prediction, targets)
        if not torch.isfinite(loss):
            raise ValueError("non-finite diagnostic loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError("non-finite diagnostic gradient")
        for name in gradient_activity:
            module = getattr(model, name)
            gradient_activity[name] |= any(p.grad is not None and p.grad.abs().max().item() > 0
                                           for p in module.parameters())
        optimizer.step()
        if (step+1) % 50 == 0 or step+1 == steps:
            progress(output, phase, step=step+1, steps=steps, l1=loss.item(),
                     elapsed_seconds=round(time.monotonic()-started, 3))
    final, predictions = score(model, condition, targets)
    ablations = {}
    for kind in ("zero-actions", "reverse-actions", "zero-horizon", "zero-features"):
        metrics, altered = score(model, condition, targets, kind)
        ablations[kind] = {**metrics, "mean_absolute_prediction_change": (predictions-altered).abs().mean().item()}
    return {"initial_copy": initial, "final": final, "ablations": ablations,
            "relative_l1_reduction": 1-final["l1"]/initial["l1"],
            "gradient_activity": gradient_activity, "steps": steps, "samples": len(targets),
            "sample_presentations": steps*len(targets), "elapsed_seconds": time.monotonic()-started}, predictions


def controlled_batch(device):
    # Identical current images, features, state, and unordered action multisets.
    # Only action ORDER and requested horizon distinguish these four targets.
    horizons = torch.tensor([8, 8, 32, 32], device=device)
    actions = torch.zeros(4, 32, 7, device=device)
    current = torch.full((4, 2, 3, 128, 128), .5, device=device)
    targets = current.clone()
    for i, h in enumerate(horizons.tolist()):
        direction = 1 if i % 2 else -1
        actions[i, :h//2, 0] = -direction
        actions[i, h//2:h, 0] = direction
        targets[i, 0, 0] += direction*.2
        targets[i, 0, 1] += .1*h/32
        targets[i, 1, 1] -= direction*.2
        targets[i, 1, 2] += .1*h/32
    condition = PixelCondition(torch.zeros(4, 4, 2048, device=device),
                               torch.zeros(4, 4, dtype=torch.bool, device=device), current,
                               torch.zeros(4, 8, device=device), actions, horizons)
    return condition, targets


def verify_base_files(cache, manifest):
    snapshot = cache / "models--Qwen--Qwen3.5-2B" / "snapshots" / QWEN_REVISION
    checked = {}
    for pin in manifest["files"]:
        if pin["path"] in {".gitattributes", "LICENSE", "README.md"}:
            continue
        path = snapshot / pin["path"]
        if path.stat().st_size != pin["size"]:
            raise ValueError("base model file size changed")
        if "lfs_sha256" in pin:
            digest = sha256_file(path)
            if digest != pin["lfs_sha256"]:
                raise ValueError("base model checkpoint hash changed")
        else:
            data = path.read_bytes()
            digest = hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
            if digest != pin["git_blob_id"]:
                raise ValueError("base model processing/config changed")
        checked[pin["path"]] = digest
    return checked


def render_examples(output, samples, predictions):
    from PIL import Image, ImageDraw

    for episode in sorted({sample.identity["episode_index"] for sample in samples}):
        selected = [(i, sample) for i, sample in enumerate(samples)
                    if sample.identity["episode_index"] == episode and sample.identity["horizon"] in (8, 32)]
        canvas = Image.new("RGB", (1072, 1190), "#101827")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f"Diagnostic TRAIN overfit only | episode {episode}", fill="white")
        draw.text((16, 30), selected[0][1].observation.instruction, fill="white")
        for j, (i, sample) in enumerate(selected):
            for camera in range(2):
                y = 65+(j*2+camera)*278
                draw.text((16, y), f"{'front' if camera == 0 else 'wrist'} | h={sample.identity['horizon']} "
                          f"t={sample.identity['anchor']} -> {sample.identity['target']}", fill="white")
                current, target, predicted = sample.observation.current_images[camera], sample.target_images[camera], predictions[i, camera].cpu()
                for column, (label, tensor) in enumerate((("Current / copy", current), ("Target", target),
                                                         ("Prediction", predicted), ("Abs error x4", (predicted-target).abs()*4))):
                    x = 16+column*264
                    draw.text((x, y+16), label, fill="#a7bad6")
                    pixels = (tensor.clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
                    canvas.paste(Image.fromarray(pixels).resize((240, 240)), (x, y+32))
        canvas.save(output / f"diagnostic-episode-{episode:06d}.png")


def main(args):
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("this bounded real-base diagnostic requires the approved CUDA host")
    recipe = read_sealed(ROOT / "configs/training/week5_pixel_predictor_v3.json")
    config = PixelPredictorConfig(**recipe["architecture"])
    if asdict(config) != asdict(PixelPredictorConfig()) or FEATURE_PROMPT != recipe["feature_prompt"]:
        raise ValueError("implementation recipe changed")
    for name, digest in recipe["input_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    code = {name: sha256_file(ROOT / name) for name in CODE_FILES}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(args.output / "identity.json", seal_contract({"artifact_type": "eventground_wam.pixel_step3_run",
                                                            "recipe_digest": recipe["digest"], "source_sha256": code}))
    progress(args.output, "controlled-conditioning-check")
    model = PixelPredictor(config).to(device)
    condition, targets = controlled_batch(device)
    synthetic, _ = fit(model, condition, targets, steps=recipe["diagnostics"]["synthetic_steps"],
                       learning_rate=recipe["diagnostics"]["learning_rate"], output=args.output,
                       phase="controlled-conditioning-check")
    synthetic["checks"] = {
        "learned_conditional_targets": synthetic["relative_l1_reduction"] >= .80,
        "action_order_changes_prediction": synthetic["ablations"]["reverse-actions"]["l1"] > synthetic["final"]["l1"] + .02,
        "explicit_horizon_changes_prediction": synthetic["ablations"]["zero-horizon"]["mean_absolute_prediction_change"] > .005,
    }
    write_json(args.output / "synthetic-check.json", seal_contract({"artifact_type": "eventground_wam.pixel_conditioning_diagnostic", **synthetic}))
    if not all(synthetic["checks"].values()):
        raise ValueError("controlled action/horizon learning check failed; see synthetic-check.json")
    del model, condition, targets
    gc.collect(); torch.cuda.empty_cache()

    progress(args.output, "verify-base-and-train-inputs")
    model_manifest = read_sealed(ROOT / "docs/model/Qwen3.5-2B-manifest.json")
    verified_model_files = verify_base_files(args.cache, model_manifest)
    action_normalizer = load_b0_action_normalizer(ROOT / "docs/model/week4-fresh-v2/action-statistics.json")
    state_bounds = read_sealed(ROOT / "configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"]
    repository = args.data_root / f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo"
    dataset = FuturePairDataset(args.pairs, repository, split="train",
                                expected_manifest_sha256=recipe["pair_manifest_sha256"],
                                normalizer=action_normalizer, state_bounds=state_bounds)
    rank = lambda value: hashlib.sha256(f"pixel-step3-diagnostic-v1:{value}".encode()).hexdigest()
    tasks = sorted({source["task_key"] for source in dataset.sources.values()}, key=rank)[:4]
    selected = [min((source for source in dataset.sources.values() if source["task_key"] == task),
                    key=lambda source: rank(source["episode_index"])) for task in tasks]
    samples = []
    for source in selected:
        anchor = (source["episode_length"]-33)//2
        samples.extend(dataset[dataset.ordinal(source["episode_index"], anchor, h)] for h in HORIZONS)
    write_json(args.output / "sample-identities.json", seal_contract({"artifact_type": "eventground_wam.pixel_step3_samples",
                                                                    "selection": recipe["diagnostics"]["selection"],
                                                                    "samples": [s.identity for s in samples]}))

    progress(args.output, "extract-frozen-base-features", current_observations=4)
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).to(device)
    encoder = FrozenQwenFeatures(full.model).to(device)
    builder = FutureInputBuilder(load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True))
    before = tensor_state_digest(encoder.core.state_dict())
    features, repeat_error = [], 0.
    for i in range(0, len(samples), len(HORIZONS)):
        observation = samples[i].observation
        inputs = builder([observation.instruction], [observation.current_rgb[0]], [observation.current_rgb[1]])
        inputs = {key: value.to(device) for key, value in inputs.items()}
        encoded, padding = encoder(inputs)
        again, _ = encoder(inputs)
        repeat_error = max(repeat_error, (encoded-again).float().abs().max().item())
        active = encoded[0, ~padding[0]].cpu()
        features.extend([active.clone() for _ in HORIZONS])
    after = tensor_state_digest(encoder.core.state_dict())
    if before != after or repeat_error != 0 or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
        raise ValueError("base feature extractor changed or is nondeterministic")
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    padding = torch.arange(padded_features.shape[1])[None] >= torch.tensor([len(f) for f in features])[:, None]
    torch.save({"features": padded_features, "padding": padding,
                "identities": [s.identity for s in samples]}, args.output / "diagnostic-features.pt")
    feature_receipt = seal_contract({"artifact_type": "eventground_wam.pixel_step3_features",
                                     "base_manifest_digest": model_manifest["digest"],
                                     "verified_model_files": verified_model_files,
                                     "base_state_sha256_before": before, "base_state_sha256_after": after,
                                     "maximum_repeat_difference": repeat_error, "base_trainable_parameters": 0,
                                     "prompt": FEATURE_PROMPT, "token_counts": [len(f) for f in features],
                                     "cache_sha256": sha256_file(args.output / "diagnostic-features.pt"),
                                     "only_current_images_and_instruction_encoded": True})
    write_json(args.output / "feature-receipt.json", feature_receipt)
    del encoder, full, builder, encoded, again, inputs
    gc.collect(); torch.cuda.empty_cache()

    batch = collate_future_samples(samples, padded_features, padding)
    condition, targets = batch.condition.to(device), batch.targets.to(device)
    model = PixelPredictor(config).to(device)
    torch.cuda.reset_peak_memory_stats()
    real, predictions = fit(model, condition, targets, steps=recipe["diagnostics"]["real_steps"],
                            learning_rate=recipe["diagnostics"]["learning_rate"], output=args.output,
                            phase="real-train-small-batch-overfit")
    checkpoint = args.output / "diagnostic-pixel.pt"
    header = save_pixel_checkpoint(checkpoint, model, metadata={"purpose": "step3-diagnostic-only-not-C_pixel",
                                                              "recipe_digest": recipe["digest"],
                                                              "feature_receipt_digest": feature_receipt["digest"]})
    restored, _ = load_pixel_checkpoint(checkpoint)
    restored = restored.to(device)
    _, replay = score(restored, condition, targets)
    reload_error = (replay-predictions).abs().max().item()
    real["checks"] = {"small_batch_l1_reduction": real["relative_l1_reduction"] >= .20,
                       "all_conditioning_branches_receive_gradients": all(real["gradient_activity"].values()),
                       "weight_reload_exact": reload_error == 0,
                       "action_ablation_changes_prediction": real["ablations"]["zero-actions"]["mean_absolute_prediction_change"] > 1e-5,
                       "horizon_ablation_changes_prediction": real["ablations"]["zero-horizon"]["mean_absolute_prediction_change"] > 1e-5}
    render_examples(args.output, samples, predictions)
    if any(sha256_file(ROOT / name) != digest for name, digest in code.items()):
        raise ValueError("executed implementation changed")
    report = seal_contract({"artifact_type": "eventground_wam.pixel_step3_implementation_check",
                            "status": "passed" if all(real["checks"].values()) else "failed",
                            "recipe_digest": recipe["digest"], "source_sha256": code,
                            "parameter_budget": model.parameter_budget(), "synthetic": synthetic, "real_train": real,
                            "checkpoint_header_digest": header["digest"], "checkpoint_sha256": sha256_file(checkpoint),
                            "maximum_checkpoint_reload_difference": reload_error,
                            "feature_receipt_digest": feature_receipt["digest"],
                            "training_episode_indices": dataset.decoded_episodes,
                            "validation_payloads_read": 0, "test_payloads_read": 0,
                            "production_cpt_started": False, "action_policy_training_started": False,
                            "step2_generalization_gate": "NOT_ASSESSED_DIAGNOSTIC_TRAIN_OVERFIT_ONLY",
                            "elapsed_seconds": time.monotonic()-started,
                            "diagnostic_gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "diagnostic_gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
                            "preview_sha256": {p.name: sha256_file(p) for p in args.output.glob("diagnostic-episode-*.png")}})
    write_json(args.output / "report.json", report)
    progress(args.output, "complete", status=report["status"], report_digest=report["digest"])
    if report["status"] != "passed":
        raise ValueError("real-data implementation checks failed; see report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home() / "data/eventground-wam/huggingface-cache")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
