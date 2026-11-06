#!/usr/bin/env python3
"""Bounded training-only JEPA implementation verification, never production CPT."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from check_jepa_targets import runtime_versions
from check_pixel_predictor import progress, verify_base_files

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import HORIZONS, read_sealed, sha256_file, write_json
from eventground_wam.jepa_data import JepaPairDataset, collate_jepa_samples
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_predictor import (
    JepaPredictor,
    JepaPredictorConfig,
    jepa_prediction_loss,
    load_jepa_checkpoint,
    normalize_predictions,
    save_jepa_checkpoint,
)
from eventground_wam.pixel_baselines import box_rgb
from eventground_wam.pixel_data import FEATURE_PROMPT, FutureInputBuilder
from eventground_wam.pixel_predictor import (
    FrozenQwenFeatures,
    PixelCondition,
    PixelPredictor,
    tensor_state_digest,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPE = "configs/training/week6_jepa_predictor_v3.json"


def controlled_batch(device):
    horizons = torch.tensor([8, 8, 32, 32], device=device)
    actions = torch.zeros(4, 32, 7, device=device)
    raw = torch.zeros(4, 128, 2048, device=device)
    raw[:, :, 0], raw[:, :, 1] = 1, -1
    for i, h in enumerate(horizons.tolist()):
        direction = 1 if i % 2 else -1
        actions[i, :h//2, 0], actions[i, h//2:h, 0] = -direction, direction
        raw[i, :64, 2], raw[i, :64, 3] = .7*direction, -.7*direction
        raw[i, 64:, 2], raw[i, 64:, 3] = -.7*direction, .7*direction
        raw[i, :, 4], raw[i, :, 5] = .6*h/32, -.6*h/32
    condition = PixelCondition(torch.zeros(4, 4, 2048, device=device),
                               torch.zeros(4, 4, dtype=torch.bool, device=device),
                               torch.full((4, 2, 3, 128, 128), .5, device=device),
                               torch.zeros(4, 8, device=device), actions, horizons)
    return condition, normalize_predictions(raw).detach(), torch.arange(4, device=device)


@torch.no_grad()
def score(model, condition, targets, groups, loss_rule, ablation="none"):
    model.eval()
    predicted = model(condition, ablation=ablation)
    loss = jepa_prediction_loss(predicted.features, targets, groups, **loss_rule)
    return {"total": loss.total.item(), "cosine": loss.cosine.item(), "variance": loss.variance.item(),
            "independent_anchors": loss.independent_anchors}, predicted


def fit(model, condition, targets, groups, *, steps, recipe, output, phase):
    initial, _ = score(model, condition, targets, groups, recipe["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["diagnostics"]["learning_rate"], weight_decay=0.)
    activity = dict.fromkeys(("feature_projection", "current_patches", "action_projection",
                             "action_sequence_projection", "state_projection", "horizon_projection",
                             "output_projection"), False)
    started = time.monotonic()
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = jepa_prediction_loss(model(condition).features, targets, groups, **recipe["loss"])
        if not torch.isfinite(loss.total):
            raise ValueError("nonfinite training loss")
        loss.total.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("missing or nonfinite trainable gradient")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite gradient norm")
        for name in activity:
            activity[name] |= any(p.grad.abs().max().item() > 0 for p in getattr(model, name).parameters())
        optimizer.step()
        if (step+1) % 50 == 0 or step+1 == steps:
            progress(output, phase, step=step+1, steps=steps, cosine=loss.cosine.item(),
                     variance=loss.variance.item(), elapsed_seconds=time.monotonic()-started)
    final, predicted = score(model, condition, targets, groups, recipe["loss"])
    ablations = {}
    for ablation in ("zero-actions", "reverse-actions", "zero-horizon", "zero-features"):
        metrics, altered = score(model, condition, targets, groups, recipe["loss"], ablation)
        ablations[ablation] = {**metrics, "mean_absolute_prediction_change":
                              (altered.features-predicted.features).abs().mean().item()}
    return {"initial": initial, "final": final, "ablations": ablations,
            "relative_initial_cosine_reduction": 1-final["cosine"]/initial["cosine"],
            "gradient_activity": activity, "steps": steps, "samples": len(targets),
            "sample_presentations": steps*len(targets), "elapsed_seconds": time.monotonic()-started}, predicted


def selected_samples(dataset):
    # Exactly the accepted Pixel implementation-check identities, for a matched check.
    rank = lambda value: hashlib.sha256(f"pixel-step3-diagnostic-v1:{value}".encode()).hexdigest()
    tasks = sorted({s["task_key"] for s in dataset.sources.values()}, key=rank)[:4]
    selected = [min((s for s in dataset.sources.values() if s["task_key"] == task),
                    key=lambda s: rank(s["episode_index"])) for task in tasks]
    samples = []
    for source in selected:
        anchor = (source["episode_length"]-33)//2
        samples.extend(dataset[dataset.ordinal(source["episode_index"], anchor, h)] for h in HORIZONS)
    return samples


def render_examples(output, samples, targets, copy, predicted):
    from PIL import Image, ImageDraw

    errors = {"Copy cosine error": (1-(copy.double()*targets.double()).sum(-1)).clamp(0, 2),
              "Predicted cosine error": (1-(predicted.double()*targets.double()).sum(-1)).clamp(0, 2)}
    for start in range(0, len(samples), 5):
        canvas = Image.new("RGB", (1050, 1170), "#111827")
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 12), f"TRAIN overfit only | episode {samples[start].identity['episode_index']} | error scale 0..1", fill="white")
        for row, (offset, camera) in enumerate(((2, 0), (2, 1), (4, 0), (4, 1))):
            i, y = start+offset, 45+row*280
            s = samples[i]
            draw.text((15, y), f"{'front' if camera == 0 else 'wrist'} h={s.identity['horizon']} t={s.identity['anchor']} -> {s.identity['target']}", fill="white")
            panels = [("Current", s.observation.current_rgb[camera]), ("Future target", s.target_rgb[camera])]
            for label, error in errors.items():
                field = error[i, camera*64:(camera+1)*64].reshape(8, 8).cpu().numpy().clip(0, 1)
                pixels = np.stack((field, .15*field, .1*field), axis=-1)
                panels.append((label, np.round(255*pixels).astype(np.uint8)))
            for col, (label, pixels) in enumerate(panels):
                x = 15+col*260
                draw.text((x, y+18), label, fill="white")
                canvas.paste(Image.fromarray(pixels).resize((240, 240), Image.Resampling.NEAREST), (x, y+35))
        canvas.save(output/f"diagnostic-episode-{samples[start].identity['episode_index']:06d}.png")


def main(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise ValueError("approved CUDA host required")
    recipe = read_sealed(ROOT/RECIPE)
    config = JepaPredictorConfig(**recipe["architecture"])
    if asdict(config) != asdict(JepaPredictorConfig()) or recipe["feature_prompt"] != FEATURE_PROMPT:
        raise ValueError("JEPA architecture or input prompt differs from recipe")
    for name, digest in recipe["input_sha256"].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    versions = runtime_versions()
    if versions["torch_loaded_build"] != "2.13.0+cu130" or versions["transformers"] != "5.17.0":
        raise ValueError("locked CUDA runtime required")
    code = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted((ROOT/"src/eventground_wam").glob("*.py"))}
    code.update({name: sha256_file(ROOT/name) for name in
                 ("scripts/check_jepa_predictor.py", "scripts/check_jepa_targets.py", "scripts/check_pixel_predictor.py")})
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    identity = seal_contract({"artifact_type": "eventground_wam.jepa_step3_run", "recipe_digest": recipe["digest"],
                              "source_sha256": code, "runtime": versions})
    write_json(args.output/"identity.json", identity)
    criteria = recipe["diagnostics"]["criteria"]
    progress(args.output, "controlled-conditioning")
    model = JepaPredictor(config).cuda()
    condition, targets, groups = controlled_batch("cuda")
    synthetic, _ = fit(model, condition, targets, groups, steps=recipe["diagnostics"]["synthetic_steps"],
                       recipe=recipe, output=args.output, phase="controlled-conditioning")
    synthetic["checks"] = {
        "conditional_learning": synthetic["relative_initial_cosine_reduction"] >= criteria["synthetic_relative_reduction_min"],
        "action_order": synthetic["ablations"]["reverse-actions"]["cosine"]-synthetic["final"]["cosine"] > criteria["synthetic_reverse_cosine_increase_min"],
        "explicit_horizon": synthetic["ablations"]["zero-horizon"]["mean_absolute_prediction_change"] > criteria["synthetic_horizon_prediction_change_min"]}
    write_json(args.output/"synthetic-check.json", seal_contract({"artifact_type": "eventground_wam.jepa_controlled_diagnostic", **synthetic}))
    if not all(synthetic["checks"].values()):
        raise ValueError("controlled conditioning failed; real data not accessed")
    del model, condition, targets, groups
    gc.collect()
    torch.cuda.empty_cache()
    progress(args.output, "verify-and-encode-training-pairs")
    model_manifest = read_sealed(ROOT/"docs/model/Qwen3.5-2B-manifest.json")
    model_files = verify_base_files(args.cache, model_manifest)
    dataset = JepaPairDataset(args.pairs, args.data_root/f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo",
                              split="train", expected_manifest_sha256=recipe["pair_manifest_sha256"],
                              normalizer=load_b0_action_normalizer(ROOT/"docs/model/week4-fresh-v2/action-statistics.json"),
                              state_bounds=read_sealed(ROOT/"configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"])
    samples = selected_samples(dataset)
    write_json(args.output/"sample-identities.json", seal_contract({"artifact_type": "eventground_wam.jepa_step3_samples", "samples": [s.identity for s in samples]}))
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).cuda()
    processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True)
    encoder, builder = FrozenQwenFeatures(full.model), FutureInputBuilder(processor)
    teacher = FrozenQwenVisualTargets(full.model.visual, processor.image_processor)
    accepted = read_sealed(ROOT/"docs/evaluation/week6-jepa-targets-v2/report.json")
    before, vision_before = tensor_state_digest(encoder.core.state_dict()), tensor_state_digest(teacher.visual.state_dict())
    if vision_before != accepted["vision_state_after"]:
        raise ValueError("teacher weights changed from Step 1")
    for name, obj in (("vision", type(teacher.visual)), ("processor", type(teacher.image_processor))):
        if sha256_file(Path(inspect.getfile(obj))) != accepted["implementation_files"][name]["sha256"]:
            raise ValueError("teacher implementation changed")
    features, copy, target, repeat = [], [], [], 0.
    for start in range(0, len(samples), 5):
        observation = samples[start].observation
        inputs = {k: v.cuda() for k, v in builder([observation.instruction], [observation.current_rgb[0]], [observation.current_rgb[1]]).items()}
        encoded, padding = encoder(inputs)
        again, _ = encoder(inputs)
        repeat = max(repeat, (again-encoded).abs().max().item())
        features.extend([encoded[0, ~padding[0]].cpu().clone() for _ in HORIZONS])
        current_target = teacher(observation.current_rgb[None])[0].cpu()
        copy.extend([current_target.clone() for _ in HORIZONS])
        rgb = np.stack([s.target_rgb for s in samples[start:start+5]])
        values = teacher(rgb)
        repeat = max(repeat, (values-teacher(rgb)).abs().max().item())
        target.extend(values.cpu().unbind())
        for sample in samples[start:start+5]:
            for camera in range(2):
                if not np.array_equal(box_rgb(sample.target_rgb[camera]), sample.target_images[camera].numpy()):
                    raise ValueError("native target/camera alignment changed")
    after, vision_after = tensor_state_digest(encoder.core.state_dict()), tensor_state_digest(teacher.visual.state_dict())
    if before != after or vision_before != vision_after or repeat != 0 or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
        raise ValueError("frozen encoders changed or extraction is not repeatable")
    feature_receipt = seal_contract({"artifact_type": "eventground_wam.jepa_step3_features", "verified_model_files": model_files,
                                     "base_before": before, "base_after": after, "teacher_before": vision_before,
                                     "teacher_after": vision_after, "repeat_max_error": repeat,
                                     "base_trainable_parameters": 0, "teacher_trainable_parameters": 0,
                                     "target_contract_digest": recipe["target_contract_digest"],
                                     "token_counts": [len(f) for f in features], "camera_target_alignment_exact": True})
    write_json(args.output/"feature-receipt.json", feature_receipt)
    padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    padding = torch.arange(padded.shape[1])[None] >= torch.tensor([len(f) for f in features])[:, None]
    batch = collate_jepa_samples(samples, padded, padding, torch.stack(target))
    bank = {"condition": {f.name: getattr(batch.condition, f.name) for f in fields(batch.condition)},
            "targets": batch.targets, "copy": torch.stack(copy), "anchor_groups": batch.anchor_groups,
            "identities": list(batch.identities)}
    torch.save(bank, args.output/"diagnostic-bank.pt")
    del teacher, encoder, full, builder, processor, encoded, again, inputs, values
    gc.collect()
    torch.cuda.empty_cache()
    condition, targets, groups = batch.condition.to("cuda"), batch.targets.cuda(), batch.anchor_groups.cuda()
    model, pixel = JepaPredictor(config).cuda().eval(), PixelPredictor(config.pixel_config()).cuda().eval()
    with torch.no_grad():
        token_error = (model(condition).future_tokens-pixel(condition).future_tokens).abs().max().item()
    if token_error != 0:
        raise ValueError("initial JEPA/Pixel core output mismatch")
    del pixel
    torch.cuda.reset_peak_memory_stats()
    real, predicted = fit(model, condition, targets, groups, steps=recipe["diagnostics"]["real_steps"],
                          recipe=recipe, output=args.output, phase="real-train-overfit")
    real["copy_cosine"] = jepa_prediction_loss(bank["copy"].cuda(), targets, groups, variance_weight=0).cosine.item()
    real["relative_copy_cosine_reduction"] = 1-real["final"]["cosine"]/real["copy_cosine"]
    header = save_jepa_checkpoint(args.output/"diagnostic-jepa.pt", model,
                                  metadata={"purpose": "step3-diagnostic-only-not-C_jepa", "recipe_digest": recipe["digest"],
                                            "identity_digest": identity["digest"], "feature_receipt_digest": feature_receipt["digest"]})
    restored, _ = load_jepa_checkpoint(args.output/"diagnostic-jepa.pt")
    _, replay = score(restored.cuda(), condition, targets, groups, recipe["loss"])
    reload_error = (replay.features-predicted.features).abs().max().item()
    real["checks"] = {"initial_loss_reduction": real["relative_initial_cosine_reduction"] >= criteria["real_relative_initial_reduction_min"],
                       "copy_loss_reduction": real["relative_copy_cosine_reduction"] >= criteria["real_relative_copy_reduction_min"],
                       "all_conditioning_gradients": all(real["gradient_activity"].values()),
                       "exact_reload": reload_error == 0,
                       "action_sensitivity": real["ablations"]["zero-actions"]["mean_absolute_prediction_change"] > criteria["real_prediction_change_min"],
                       "action_order_sensitivity": real["ablations"]["reverse-actions"]["mean_absolute_prediction_change"] > criteria["real_prediction_change_min"],
                       "horizon_sensitivity": real["ablations"]["zero-horizon"]["mean_absolute_prediction_change"] > criteria["real_prediction_change_min"],
                       "feature_sensitivity": real["ablations"]["zero-features"]["mean_absolute_prediction_change"] > criteria["real_prediction_change_min"]}
    torch.save({"features": predicted.features.cpu(), "future_tokens": predicted.future_tokens.cpu()}, args.output/"predictions.pt")
    render_examples(args.output, samples, batch.targets, bank["copy"], predicted.features.cpu())
    if any(sha256_file(ROOT/name) != digest for name, digest in code.items()):
        raise ValueError("executed source changed")
    report = seal_contract({"artifact_type": "eventground_wam.jepa_step3_implementation_check",
                            "status": "passed" if all(real["checks"].values()) else "failed",
                            "identity_digest": identity["digest"], "recipe_digest": recipe["digest"],
                            "parameter_budget": model.parameter_budget(), "synthetic": synthetic, "real_train": real,
                            "initial_pixel_core_token_max_error": token_error, "checkpoint_reload_max_error": reload_error,
                            "checkpoint_header_digest": header["digest"], "feature_receipt_digest": feature_receipt["digest"],
                            "training_episode_indices": dataset.decoded_episodes, "validation_payload_reads": 0, "test_payload_reads": 0,
                            "production_cpt_started": False, "action_policy_training_started": False,
                            "step2_gate": "NOT_ASSESSED_DIAGNOSTIC_TRAIN_OVERFIT_ONLY",
                            "elapsed_seconds": time.monotonic()-started,
                            "predictor_gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "predictor_gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "files_sha256": {p.name: sha256_file(p) for p in args.output.iterdir() if p.suffix in (".pt", ".png", ".json") and p.name != "progress.json"}})
    write_json(args.output/"report.json", report)
    progress(args.output, "complete", status=report["status"], report_digest=report["digest"])
    if report["status"] != "passed":
        raise ValueError("real implementation check failed; preserve report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home()/"data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
