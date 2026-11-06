"""Bounded real-M4 causal/gradient/resource probe; no optimizer update or label admission."""

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
    configure_deterministic_cuda,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_data import (
    collate_current_observations,
    load_current_observation,
    target_tokens,
)
from eventground_wam.eventground_model import serialize_tokens
from eventground_wam.eventground_objectives import structured_sft_loss
from eventground_wam.eventground_runtime import build_eventground_from_m4
from eventground_wam.pixel_predictor import tensor_state_digest


def write(path, value):
    with path.open("x") as stream:
        json.dump(seal_contract(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--capture-base", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract, recipe = read_sealed(args.contract), read_sealed(args.recipe)
    if (contract["operation"] != "current_observation_gradient_only_diagnostic"
            or contract["recipe_digest"] != recipe["digest"]
            or contract["optimizer_updates"] != 0 or contract["labels_admitted"] is not False
            or contract["maximum_sources"] != 4 or contract["maximum_observation_reads"] != 4
            or not 0 < contract["wall_seconds"] <= 1800):
        raise ValueError("Wrong bounded diagnostic contract")
    if sha256_file(Path(__file__)) != recipe["code_sha256"]["scripts/check_week10_training.py"]:
        raise ValueError("Qualification script changed")
    start = time.monotonic()

    def stop(*_):
        raise TimeoutError("Gradient diagnostic wall ceiling")

    signal.signal(signal.SIGALRM, stop)
    signal.alarm(contract["wall_seconds"])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"artifact_type": "eventground_wam.week10_gradient_resource_diagnostic",
              "contract_digest": contract["digest"], "recipe_digest": recipe["digest"],
              "status": "started", "optimizer_updates": 0, "labels_admitted": False,
              "training_admission_passed": False, "heldout_payload_reads": 0}
    try:
        if shutil.disk_usage(args.output).free < recipe["resources"]["minimum_free_bytes"]:
            raise ValueError("Insufficient free disk")
        configure_deterministic_cuda()
        seed_training(17)
        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(recipe["resources"]["maximum_gpu_fraction"], device)
        torch.cuda.reset_peak_memory_stats()
        capture = read_sealed(args.capture_base / "recording-contract.json")
        completed = read_sealed(args.capture_base / "run/completion.json")
        if (capture["digest"] != contract["capture_contract_digest"]
                or completed["digest"] != contract["capture_completion_digest"]
                or completed["status"] != "complete"):
            raise ValueError("Capture identity changed")
        targets = {t["source_id"]: t for t in capture["targets"] if t["queue_role"] == "initial_review"}
        results = {r["source_id"]: r for r in completed["results"]}
        source_ids = contract["source_ids"]
        if len(source_ids) != 4 or len(set(source_ids)) != 4 or set(source_ids) - set(targets):
            raise ValueError("Only four pinned initial training sources may be read")
        # Explicit dummy outputs exercise all categorical decisions. They are NOT
        # annotations, event candidates, measured targets or optimization data.
        dummy = [
            {"next_event": "none", "interval": None, "horizon": None},
            {"next_event": "gripper_close", "interval": [3, 8], "horizon": 3},
            {"next_event": "contact", "interval": [9, 16], "horizon": 9},
            {"next_event": "grasp", "interval": [17, 33], "horizon": 17},
        ]
        observations = []
        for source in source_ids:
            t, result = targets[source], results[source]
            row = {"source_id": source, "anchor": 0, "instruction": t["instruction"],
                   "recording_path": str(args.capture_base / "run" / t["recording_id"] / "recording.h5"),
                   "recording_sha256": result["evidence_sha256"]["recording.h5"]}
            observations.append(load_current_observation(row))
        model, builder, allowlist, identity = build_eventground_from_m4(
            args.root, recipe, args.cache, device,
        )
        if {k: list(v) for k, v in allowlist.items()} != recipe["exact_allowlist"]:
            raise ValueError("Actual parameter allowlist differs from sealed inventory")
        write(args.output / "loaded-identity.json", identity)
        parameters = dict(model.named_parameters())
        trainable = tuple(sorted((*allowlist["head"], *allowlist["language_lora"])))
        trainable_before = tensor_state_digest({n: parameters[n] for n in trainable})
        frozen_before = tensor_state_digest({n: p for n, p in parameters.items() if n not in trainable})
        optimizer_state_bytes = 2 * sum(parameters[n].numel() * parameters[n].element_size()
                                        for n in trainable)
        # Retain both Adam moments DURING the backward passes so activation and
        # optimizer residency overlap. No parameter update is performed.
        moment_buffers = [torch.zeros_like(parameters[n]) for n in trainable for _ in range(2)]
        model.train()

        def gradients(indices, microbatch):
            model.zero_grad(set_to_none=True)
            total, durations = 0., []
            for offset in range(0, len(indices), microbatch):
                chosen = indices[offset:offset + microbatch]
                encoded, state = collate_current_observations(
                    [observations[i] for i in chosen], builder, device,
                )
                tokens = target_tokens([{"target": dummy[i]} for i in chosen], device)
                torch.cuda.synchronize()
                tick = time.monotonic()
                scores = model.score(encoded, state, tokens)
                loss = structured_sft_loss(scores, logical_batch_size=len(indices))
                loss.backward()
                torch.cuda.synchronize()
                durations.append(time.monotonic() - tick)
                total += float(loss.detach())
                for name, parameter in parameters.items():
                    if parameter.grad is not None and (
                        name not in trainable or not torch.isfinite(parameter.grad).all()
                    ):
                        raise ValueError(f"Unexpected/nonfinite gradient: {name}")
            return total, durations

        # Two observations compare actual Qwen full-batch and accumulated gradients.
        full_loss, _ = gradients([0, 3], 2)
        full = {n: parameters[n].grad.detach().float().cpu().clone()
                for n in trainable if parameters[n].grad is not None}
        if set(full) != set(trainable):
            raise ValueError("Full-batch gradients do not cover the exact trainable allowlist")
        accumulated_loss, _ = gradients([0, 3], 1)
        errors = []
        squared_error, squared_reference = 0., 0.
        for name, gradient in full.items():
            actual = parameters[name].grad
            if actual is None:
                raise ValueError("Microbatch gradient disappeared")
            actual = actual.detach().float().cpu()
            absolute = float((actual - gradient).abs().max())
            scale = float(gradient.abs().max())
            squared_error += float((actual.double() - gradient.double()).square().sum())
            squared_reference += float(gradient.double().square().sum())
            errors.append({"name": name, "maximum_absolute_error": absolute,
                           "reference_maximum": scale})
        del full
        relative_l2 = math.sqrt(squared_error / max(squared_reference, 1e-24))
        report.update(full_batch_loss=full_loss, accumulated_loss=accumulated_loss,
                      gradient_comparisons=errors, gradient_relative_l2=relative_l2)
        if relative_l2 > recipe["verification"]["gradient_relative_l2"]:
            raise ValueError("Actual-host microbatch gradient equivalence failed")
        if abs(full_loss - accumulated_loss) > recipe["verification"]["loss_atol"]:
            raise ValueError("Actual-host microbatch loss equivalence failed")
        # One complete batch with unchanged sample identities/order, no optimizer step.
        _, timings = gradients([i % 4 for i in range(recipe["logical_batch_size"])], 1)
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        del moment_buffers
        if peak_reserved > recipe["resources"]["maximum_gpu_fraction"] * total_memory:
            raise ValueError("Actual-host GPU memory ceiling exceeded")
        model.eval()
        encoded, state = collate_current_observations(observations[:1], builder, device)
        with torch.no_grad():
            first = model.sample(encoded, state, group_size=4)
        samples = serialize_tokens(first.tokens)
        if tensor_state_digest({n: parameters[n] for n in trainable}) != trainable_before:
            raise ValueError("Gradient-only probe changed trainable weights")
        if tensor_state_digest({n: p for n, p in parameters.items() if n not in trainable}) != frozen_before:
            raise ValueError("Gradient-only probe changed frozen weights")
        report.update({
            "status": "passed_gradient_only_probe_not_training_admission",
            "parent_identity": identity, "current_observations_read": 4,
            "dummy_target_scope": "Synthetic categorical diagnostics, not physical event labels",
            "full_batch_loss": full_loss, "accumulated_loss": accumulated_loss,
            "gradient_comparisons": errors,
            "full_logical_batch_size": recipe["logical_batch_size"],
            "microbatch_size": 1, "logical_batch_forward_backward_seconds": sum(timings),
            "microbatch_seconds": timings, "optimizer_moment_bytes_measured": optimizer_state_bytes,
            "gpu_peak_allocated_bytes": peak_allocated, "gpu_peak_reserved_bytes": peak_reserved,
            "gpu_total_bytes": total_memory, "sampled_outputs": samples,
            "trainable_weights_unchanged": True, "frozen_weights_unchanged": True,
            "projected_sft_compute_hours_with_2x_margin": (
                2 * math.ceil(2000 / recipe["logical_batch_size"]) * sum(timings) * 2 / 3600),
            "estimate_limitations": "No measured optimizer/checkpoint cost or admitted-label overfit; RLVR resource qualification remains separate.",
        })
    except Exception as exc:
        report.update(status="failed_preserved", error=repr(exc))
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - start
        write(args.output / "report.json", report)


if __name__ == "__main__":
    main()
