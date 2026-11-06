"""Bounded zero-update diagnosis of M4 padding/batch/precision sensitivity.

The frozen v1 recipe and failed evidence remain unchanged. This diagnostic cannot
admit training, labels, a new precision recipe, or a relaxed numerical tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import time
from pathlib import Path

import torch

from eventground_wam.b0_training import (
    capture_rng,
    configure_deterministic_cuda,
    read_sealed,
    restore_rng,
    seed_training,
    sha256_file,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_data import (
    collate_current_observations,
    load_current_observation,
    target_tokens,
)
from eventground_wam.eventground_model import ENCODING_KEYS
from eventground_wam.eventground_objectives import structured_sft_loss
from eventground_wam.eventground_runtime import build_eventground_from_m4
from eventground_wam.pixel_predictor import tensor_state_digest

GIB = 1024 ** 3


def backend_metadata(device, parameter_dtypes):
    return {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "registered_parameter_dtypes": sorted({str(value) for value in parameter_dtypes}),
    }


def slice_encoding(encoded, start, stop):
    """Slice examples while retaining shared padding and both flattened image blocks.

    Qwen pixel_values rows are raw spatiotemporal patches, not examples or merged
    image tokens. Grid products delimit them; every example owns front then wrist.
    """
    if set(encoded) != ENCODING_KEYS:
        raise ValueError("Expected exactly the five OFT encoding fields")
    ids, mask, kinds = (encoded[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids"))
    grid, pixels = encoded["image_grid_thw"], encoded["pixel_values"]
    if (ids.ndim != 2 or mask.shape != ids.shape or kinds.shape != ids.shape
            or type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(ids)
            or grid.shape != (2 * len(ids), 3) or grid.dtype not in (torch.int32, torch.int64)
            or (grid <= 0).any() or pixels.ndim != 2):
        raise ValueError("Malformed OFT example/text/image layout")
    counts = grid.to(torch.int64).prod(-1)
    if int(counts.sum()) != len(pixels):
        raise ValueError("Flattened image patches disagree with raw grid products")
    first = int(counts[:2 * start].sum())
    last = int(counts[:2 * stop].sum())
    return {
        "input_ids": ids[start:stop].clone(),
        "attention_mask": mask[start:stop].clone(),
        "mm_token_type_ids": kinds[start:stop].clone(),
        "image_grid_thw": grid[2 * start:2 * stop].clone(),
        "pixel_values": pixels[first:last].clone(),
    }


def tensor_comparison(reference, actual):
    if reference.shape != actual.shape:
        return {"same_shape": False, "reference_shape": list(reference.shape),
                "actual_shape": list(actual.shape)}
    reference, actual = reference.detach().double().cpu(), actual.detach().double().cpu()
    difference = actual - reference
    denominator = float(reference.square().sum())
    squared = float(difference.square().sum())
    return {"same_shape": True, "bitwise_equal": bool(torch.equal(reference, actual)),
            "maximum_absolute_error": float(difference.abs().max()) if difference.numel() else 0.,
            "relative_l2": math.sqrt(squared / max(denominator, 1e-24)),
            "reference_l2": math.sqrt(denominator), "difference_l2": math.sqrt(squared)}


def gradient_comparison(reference, actual):
    if set(reference) != set(actual):
        raise ValueError("Gradient names differ between diagnostic executions")
    totals = {"all": [0., 0.], "head": [0., 0.], "language_lora": [0., 0.]}
    per_parameter = []
    for name in sorted(reference):
        value = tensor_comparison(reference[name], actual[name])
        if not value["same_shape"]:
            raise ValueError(f"Gradient shape changed: {name}")
        per_parameter.append({"name": name, **value})
        group = "head" if name.startswith("head.") else "language_lora"
        for kind in ("all", group):
            totals[kind][0] += value["difference_l2"] ** 2
            totals[kind][1] += value["reference_l2"] ** 2
    return {"relative_l2_by_group": {
        kind: math.sqrt(error / max(scale, 1e-24)) for kind, (error, scale) in totals.items()
    }, "per_parameter": per_parameter}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "recipe", "contract", "capture-base", "cache", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    contract, recipe = read_sealed(args.contract), read_sealed(args.recipe)
    if (contract["operation"] != "current_observation_batch_numerics_diagnostic"
            or contract["recipe_digest"] != recipe["digest"]
            or contract["optimizer_updates"] != 0 or contract["labels_admitted"] is not False
            or contract["heldout_payload_reads"] != 0 or contract["maximum_sources"] != 4
            or contract["maximum_observation_reads"] != 4
            or not 0 < contract["wall_seconds"] <= 1800
            or not 0 < contract["output_bytes"] <= GIB
            or contract["minimum_free_bytes"] < 50 * GIB
            or not 0 < contract["maximum_gpu_fraction"] <= .9
            or type(contract["allow_fp32_diagnostic"]) is not bool
            or contract["script_sha256"] != sha256_file(Path(__file__))):
        raise ValueError("Invalid or changed bounded numerical diagnostic contract")
    if (contract.get("prior_qualification_active_gpu_seconds", 0) + contract["wall_seconds"]
            > recipe["resources"]["qualification_active_gpu_hours_maximum"] * 3600):
        raise ValueError("Diagnostic exceeds cumulative qualification budget")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {
        "artifact_type": "eventground_wam.week10_batch_numerics_diagnostic",
        "contract_digest": contract["digest"], "recipe_digest": recipe["digest"],
        "status": "started", "optimizer_updates": 0, "heldout_payload_reads": 0,
        "labels_admitted": False, "training_admission_passed": False,
        "recipe_or_tolerances_changed": False, "cases": [], "comparisons": [],
    }
    model = None
    original_dtypes = None
    original_digest = None

    def write_report():
        payload = json.dumps(seal_contract(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
        if len(payload.encode()) > contract["output_bytes"]:
            raise ValueError("Diagnostic report exceeds output budget")
        with (args.output / "report.json").open("x") as stream:
            stream.write(payload)

    def timeout(*_):
        raise TimeoutError("Numerical diagnostic wall ceiling")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["wall_seconds"])
    try:
        configure_deterministic_cuda()
        seed_training(17)
        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(contract["maximum_gpu_fraction"], device)
        torch.cuda.reset_peak_memory_stats(device)

        def resource_check():
            if shutil.disk_usage(args.output).free < contract["minimum_free_bytes"]:
                raise ValueError("Diagnostic free disk floor exceeded")
            torch.cuda.synchronize(device)
            if (torch.cuda.max_memory_reserved(device)
                    > contract["maximum_gpu_fraction"] * torch.cuda.get_device_properties(device).total_memory):
                raise ValueError("Diagnostic GPU memory ceiling exceeded")

        resource_check()
        capture = read_sealed(args.capture_base / "recording-contract.json")
        completion = read_sealed(args.capture_base / "run/completion.json")
        if (capture["digest"] != contract["capture_contract_digest"]
                or completion["digest"] != contract["capture_completion_digest"]
                or completion["status"] != "complete"):
            raise ValueError("Capture identity differs from diagnostic contract")
        targets = {row["source_id"]: row for row in capture["targets"]
                   if row["queue_role"] == "initial_review"}
        results = {row["source_id"]: row for row in completion["results"]}
        source_ids = contract["source_ids"]
        if len(source_ids) != 4 or len(set(source_ids)) != 4 or set(source_ids) - set(targets):
            raise ValueError("Only four pinned initial training sources can be read")
        observations = []
        for source in source_ids:
            target, result = targets[source], results[source]
            observations.append(load_current_observation({
                "source_id": source, "anchor": 0, "instruction": target["instruction"],
                "recording_path": str(args.capture_base / "run" / target["recording_id"] / "recording.h5"),
                "recording_sha256": result["evidence_sha256"]["recording.h5"],
            }))
        report["current_observations_read"] = len(observations)
        model, builder, allowlist, identity = build_eventground_from_m4(
            args.root, recipe, args.cache, device,
        )
        if {key: list(value) for key, value in allowlist.items()} != recipe["exact_allowlist"]:
            raise ValueError("Actual trainable allowlist differs from frozen recipe")
        report["loaded_identity"] = identity
        original_dtypes = {name: value.dtype for name, value in model.state_dict().items()}
        original_digest = tensor_state_digest(model.state_dict())
        trainable = tuple(sorted((*allowlist["head"], *allowlist["language_lora"])))
        parameters = dict(model.named_parameters())
        report["backend"] = backend_metadata(device, [value.dtype for value in parameters.values()])
        indices = [0, 3]
        dummy = [
            {"target": {"next_event": "none", "interval": None, "horizon": None}},
            {"target": {"next_event": "grasp", "interval": [17, 33], "horizon": 17}},
        ]
        tokens = target_tokens(dummy, device)
        combined, combined_state = collate_current_observations(
            [observations[i] for i in indices], builder, device,
        )
        separate = [collate_current_observations([observations[i]], builder, device) for i in indices]
        sliced = [(slice_encoding(combined, i, i + 1), combined_state[i:i + 1].clone()) for i in range(2)]
        adapter = model.conditioning_adapter

        def rope(encoded):
            positions, _ = adapter.backbone.get_rope_index(
                encoded["input_ids"], image_grid_thw=encoded["image_grid_thw"],
                attention_mask=encoded["attention_mask"],
                mm_token_type_ids=encoded["mm_token_type_ids"],
            )
            return positions.detach().cpu()

        combined_rope = rope(combined)
        input_audit = []
        report["input_audit"] = input_audit
        for index, ((individual, state), (padded, padded_state)) in enumerate(zip(separate, sliced, strict=True)):
            valid = combined["attention_mask"][index].bool()
            independent_valid = individual["attention_mask"][0].bool()
            comparisons = {
                key: tensor_comparison(padded[key][0, valid], individual[key][0, independent_valid])
                for key in ("input_ids", "attention_mask", "mm_token_type_ids")
            }
            for key in ("pixel_values", "image_grid_thw"):
                comparisons[key] = tensor_comparison(padded[key], individual[key])
            comparisons["proprioception"] = tensor_comparison(padded_state, state)
            comparisons["rope_active_independent"] = tensor_comparison(
                combined_rope[:, index:index+1, valid.cpu()], rope(individual)[:, :, independent_valid.cpu()],
            )
            comparisons["rope_full_padded_slice"] = tensor_comparison(
                combined_rope[:, index:index+1, :], rope(padded),
            )
            input_audit.append({"source_id": source_ids[indices[index]],
                                "combined_padded_tokens": combined["input_ids"].shape[1],
                                "individual_padded_tokens": individual["input_ids"].shape[1],
                                "active_tokens": int(valid.sum()), "comparisons": comparisons})
            if any(not item.get("bitwise_equal", False) for item in comparisons.values()):
                raise ValueError("Active source pixels/tokens/state/native positions differ across collations")
        report["combined_encoding_digest"] = tensor_state_digest(combined)
        report["dummy_targets"] = dummy
        rng = capture_rng()

        def run_case(name, batches, *, fp32=False):
            report["current_case"] = name
            resource_check()
            restore_rng(rng)
            model.train()
            model.zero_grad(set_to_none=True)
            features, logs, logits, loss_total = [], [], [], 0.
            torch.cuda.synchronize(device)
            tick = time.monotonic()
            for offset, (encoded, state) in batches:
                if fp32:
                    encoded = {key: value.float() if value.is_floating_point() else value
                               for key, value in encoded.items()}
                slots = model(encoded, state)
                score = model.head.teacher_forced(slots, tokens[offset:offset + len(state)])
                loss = structured_sft_loss(score, logical_batch_size=2)
                loss.backward()
                features.append(slots.detach().float().cpu())
                logs.append(score.log_probs.detach().float().cpu())
                logits.append(score.event_logits.detach().float().cpu())
                loss_total += float(loss.detach())
                del loss, score, slots
            gradients = {}
            for name_, parameter in model.named_parameters():
                if name_ in trainable:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ValueError(f"Missing/nonfinite diagnostic gradient: {name_}")
                    gradients[name_] = parameter.grad.detach().float().cpu().clone()
                elif parameter.grad is not None:
                    raise ValueError(f"Frozen parameter gradient: {name_}")
            resource_check()
            seconds = time.monotonic() - tick
            report["cases"].append({"name": name, "loss": loss_total, "seconds": seconds,
                                    "diagnostic_all_fp32": fp32,
                                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device)})
            return {"name": name, "loss": loss_total, "gradients": gradients,
                    "slots": torch.cat(features), "log_probs": torch.cat(logs),
                    "event_logits": torch.cat(logits)}

        def compare(reference, actual):
            gradient = gradient_comparison(reference["gradients"], actual["gradients"])
            report["comparisons"].append({
                "reference": reference["name"], "actual": actual["name"],
                "loss_absolute_error": abs(reference["loss"] - actual["loss"]),
                "slots": tensor_comparison(reference["slots"], actual["slots"]),
                "slots_per_observation": [tensor_comparison(reference["slots"][i], actual["slots"][i])
                                          for i in range(2)],
                "log_probs": tensor_comparison(reference["log_probs"], actual["log_probs"]),
                "event_logits": tensor_comparison(reference["event_logits"], actual["event_logits"]),
                "gradients": gradient,
                "passes_unchanged_loss_tolerance": abs(reference["loss"] - actual["loss"])
                    <= recipe["verification"]["loss_atol"],
                "passes_unchanged_gradient_tolerance": gradient["relative_l2_by_group"]["all"]
                    <= recipe["verification"]["gradient_relative_l2"],
            })

        whole = [(0, (combined, combined_state))]
        independent = list(enumerate(separate))
        padded = list(enumerate(sliced))
        reference = run_case("bf16_full2", whole)
        repeat = run_case("bf16_full2_repeat", whole)
        compare(reference, repeat)
        del repeat
        independent_result = run_case("bf16_independent_micro1", independent)
        compare(reference, independent_result)
        repeated_micro = run_case("bf16_independent_micro1_repeat", independent)
        compare(independent_result, repeated_micro)
        del repeated_micro
        padded_result = run_case("bf16_shared_padding_micro1", padded)
        compare(reference, padded_result)
        compare(independent_result, padded_result)
        del reference, independent_result, padded_result
        model.zero_grad(set_to_none=True)
        if tensor_state_digest(model.state_dict()) != original_digest:
            raise ValueError("BF16 diagnostic changed model weights/buffers")
        report["bf16_parameters_unchanged"] = True
        if contract["allow_fp32_diagnostic"]:
            # A diagnostic cast of the child only; it is never an admitted recipe.
            # Existing BF16 values are exactly representable in FP32 and restored
            # to their original dtypes before the final unchanged-state checksum.
            model.float()
            resource_check()
            fp32_reference = run_case("diagnostic_fp32_full2", whole, fp32=True)
            fp32_padded = run_case("diagnostic_fp32_shared_padding_micro1", padded, fp32=True)
            compare(fp32_reference, fp32_padded)
            del fp32_padded
            fp32_independent = run_case("diagnostic_fp32_independent_micro1", independent, fp32=True)
            compare(fp32_reference, fp32_independent)
            del fp32_reference, fp32_independent
        report["status"] = "completed_diagnostic_not_training_admission"
    except Exception as error:
        report.update(status="failed_preserved", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        if model is not None and original_dtypes is not None:
            try:
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, value in model.state_dict(keep_vars=True).items():
                        if value.dtype != original_dtypes[name]:
                            value.data = value.data.to(original_dtypes[name])
                report["parameters_and_buffers_unchanged"] = (
                    tensor_state_digest(model.state_dict()) == original_digest)
                if not report["parameters_and_buffers_unchanged"]:
                    report.update(status="failed_preserved", integrity_error="Model state changed")
            except Exception as error:  # noqa: BLE001 - preserve cleanup failure in durable evidence
                report.update(status="failed_preserved", integrity_error=repr(error))
        report["elapsed_seconds"] = time.monotonic() - start
        if torch.cuda.is_initialized():
            report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        write_report()
        if "integrity_error" in report and "error" not in report:
            raise RuntimeError("Diagnostic model-state integrity check failed; see report.json")


if __name__ == "__main__":
    main()
