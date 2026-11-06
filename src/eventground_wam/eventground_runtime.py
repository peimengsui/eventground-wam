"""Verified M4 initialization for current-observation-only event prediction."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from eventground_wam.b0_fresh import load_fresh_recipe
from eventground_wam.b0_model import load_b0_trainable_state_dict, load_pinned_qwen_processor
from eventground_wam.b0_oft import OFTInputBuilder, build_oft_policy
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.pixel_predictor import tensor_state_digest


def verify_eventground_parent(study):
    """Check both immutable epoch-2 exports and all completion links before loading."""
    if study.get("digest") != contract_digest(study):
        raise ValueError("Unsealed Week 10 study")
    parent = study["parent"]
    checkpoint = Path(parent["checkpoint_path"])
    endpoint = checkpoint.parent
    training = checkpoint.parents[2]
    identity = read_sealed(training / "training-manifest.json")
    report = read_sealed(endpoint / "training-report.json")
    completion = read_sealed(training / "training-report.json")
    for value, key in ((identity, "identity_digest"), (report, "endpoint_digest"),
                       (completion, "completion_digest")):
        if value["digest"] != parent[key]:
            raise ValueError(f"M4 {key} changed")
    if read_sealed(endpoint / "training-manifest.json") != identity:
        raise ValueError("M4 endpoint identity differs")
    if (parent["arm"] != "M4" or report["epoch"] != 2
            or report["selection"] != "fixed-epoch-2-endpoint"
            or completion["endpoint_report_digest"] != report["digest"]):
        raise ValueError("M4 fixed completion is not verified")
    for value in (report, completion):
        for key, expected in {
            "identity_digest": identity["digest"],
            "global_step": parent["inherited_action_updates"],
            "samples_seen": parent["inherited_action_presentations"],
            "future_pairs_seen": parent["inherited_future_pairs"],
        }.items():
            if value[key] != expected:
                raise ValueError(f"M4 exposure differs: {key}")
    for key in ("sample_chain", "future_chain", "checkpoint_sha256"):
        if report[key] != completion[key]:
            raise ValueError(f"M4 completion chain differs: {key}")
    for path, expected, report_key in (
        (checkpoint, parent["policy_sha256"], "checkpoint_sha256"),
        (endpoint / "full-training-predictor.pt", parent["predictor_sha256"], "predictor_sha256"),
    ):
        if sha256_file(path) != expected or report[report_key] != expected:
            raise ValueError(f"M4 export checksum differs: {path.name}")
    return checkpoint


def build_eventground_from_m4(root: Path, recipe, cache_dir: Path, device="cuda:0"):
    """Copy the parent adapter; never run or mutate its action/future/coupling path."""
    from eventground_wam.eventground_model import (
        EventGroundHead,
        EventGroundModel,
        configure_eventground_trainables,
    )

    root = Path(root)
    study = read_sealed(root / "configs/training/week10_eventground_experiment_v3.json")
    if recipe.get("digest") != contract_digest(recipe) or recipe["study_digest"] != study["digest"]:
        raise ValueError("Wrong/unsealed EventGround recipe")
    for name, expected in recipe["code_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Recipe code changed: {name}")
    for name, expected in study["source_files_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Study source changed: {name}")
    checkpoint = verify_eventground_parent(study)
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    header, states = artifact["header"], artifact["states"]
    if (header.get("digest") != contract_digest(header) or header["arm"] != "M4"
            or header["format"] != "eventground-wam-coupled-inference-state-v1"
            or header["metadata"] != {
                "identity_digest": study["parent"]["identity_digest"],
                "global_step": study["parent"]["inherited_action_updates"],
                "epoch": 2, "checkpoint_selection": "fixed-epoch-2-endpoint",
            } or set(states) != {"policy", "future_core", "coupling"}):
        raise ValueError("M4 tensor envelope identity changed")
    for group, tensors in states.items():
        if tensor_state_digest(tensors) != header["state_sha256"][group]:
            raise ValueError(f"M4 tensor checksum changed: {group}")
    base_recipe = load_fresh_recipe(root / "configs/training/week4_b0_fresh_all_l1_v2.json")
    policy, inherited = build_oft_policy(base_recipe, str(cache_dir), device)
    load_b0_trainable_state_dict(policy, states["policy"], allowed_exact_names=inherited)
    head = EventGroundHead(**recipe["head"]).to(device)
    model = EventGroundModel(policy.conditioning_adapter, head)
    compute_dtype = recipe.get("compute_dtype", "bfloat16")
    if compute_dtype not in ("bfloat16", "float32"):
        raise ValueError("Unsupported EventGround compute precision")
    if compute_dtype == "float32":
        # Exact representation of inherited values in the child; the stored M4
        # endpoint stays untouched. This avoids observed BF16 batch-shape drift.
        model.float()
    allowlist = configure_eventground_trainables(model, inherited)
    del artifact, states, policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    builder = OFTInputBuilder(load_pinned_qwen_processor(
        cache_dir=str(cache_dir), local_files_only=True,
    ))
    builder.eventground_compute_dtype = getattr(torch, compute_dtype)
    return model, builder, allowlist, {
        "parent_policy_sha256": study["parent"]["policy_sha256"],
        "parent_predictor_sha256": study["parent"]["predictor_sha256"],
        "parent_completion_digest": study["parent"]["completion_digest"],
        "recipe_digest": recipe["digest"],
        "inherited_exact_lora_names": list(inherited),
        "allowlist": {k: list(v) for k, v in allowlist.items()},
        "compute_dtype": compute_dtype,
        "future_predictor_or_action_decoder_executed": False,
    }
