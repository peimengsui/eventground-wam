#!/usr/bin/env python3
"""Audit preserved production startup checkpoints without touching the live trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.future_pairs import HORIZONS, RECORD, read_sealed, sha256_file, write_json
from eventground_wam.jepa_predictor import JepaPredictor
from eventground_wam.jepa_training import verify_recipe
from eventground_wam.pixel_predictor import tensor_state_digest
from eventground_wam.pixel_training import CPTOrder, tree_digest


def verify(args):
    torch.set_num_threads(1)
    identity = read_sealed(args.run / "identity.json")
    initial = read_sealed(args.run / "initialization.json")
    preflight = read_sealed(args.evidence / "preflight.json")
    pilot = read_sealed(args.evidence / "pilot-report.json")
    recipe = verify_recipe(args.source / "configs/training/week6_jepa_cpt_v1.json", args.source)
    inventory = read_sealed(args.evidence / "source-inventory.json")
    order = CPTOrder(args.order, recipe["order_manifest_sha256"])
    assert identity["phase"] == "train" and identity["updates"] == len(order) == 221633
    assert identity["indices_digest"] == tree_digest(list(range(len(order))))
    assert preflight["fresh_output_empty"] and preflight["source_read_only"]
    assert preflight["status"] == pilot["status"] == "passed"
    assert recipe["digest"] == identity["recipe_digest"] == pilot["recipe_digest"] == preflight["recipe_digest"]
    assert order.manifest["digest"] == identity["order_digest"] == pilot["order_digest"]
    assert identity["runtime"] == pilot["runtime"] == preflight["runtime"]
    assert identity["source_sha256"] == pilot["source_sha256"] == preflight["source_sha256"]
    assert preflight["source_inventory_digest"] == inventory["digest"]
    for name, digest in {**inventory["files"], **identity["source_sha256"]}.items():
        assert sha256_file(args.source / name) == digest, name
    assert initial["identity_digest"] == identity["digest"]
    assert not initial["parent_or_diagnostic_weights_loaded"]
    assert initial["optimizer_state_entries"] == initial["trainable_base_parameters"] == 0
    assert initial["verified_base_files"] == preflight["verified_base_files"]
    pilot_initial = read_sealed(args.pilot_run / "initialization.json")
    accepted = read_sealed(args.source / "docs/evaluation/week6-jepa-targets-v2/report.json")
    assert initial["base_state_digest"] == pilot_initial["base_state_digest"]
    assert initial["teacher_state_digest"] == accepted["vision_state_after"]
    assert identity["digest"] != pilot_initial["identity_digest"]
    fresh_model = JepaPredictor()
    assert fresh_model.parameter_budget() == recipe["parameter_budget"]
    fresh_digest = tensor_state_digest(fresh_model.state_dict())
    assert fresh_digest == initial["model_state_digest"] == preflight["fresh_model_state_digest"]
    payloads, commits = [], []
    for name, step in (("initial-checkpoint", 0), ("first-periodic-checkpoint", 500)):
        record = read_sealed(args.evidence / f"{name}-commit.json")
        path = args.evidence / f"{name}.pt"
        assert path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"]
        value = torch.load(path, map_location="cpu", weights_only=True)
        assert value["identity_digest"] == record["identity_digest"] == identity["digest"]
        assert value["global_step"] == value["scheduler"]["last_epoch"] == record["global_step"] == step
        assert value["pairs_seen"] == int(order.pair_counts[:step].sum())
        assert set(value["rng"]) == {"python", "numpy", "torch", "cuda"}
        assert len(value["rng"]["cuda"]) == 1
        assert math.isfinite(value["loss_sum"])
        assert all(torch.isfinite(v).all() for v in value["model"].values())
        assert all(torch.isfinite(v).all() for s in value["optimizer"]["state"].values()
                   for v in s.values() if isinstance(v, torch.Tensor))
        payloads.append(value)
        commits.append(record)
    zero, periodic = payloads
    assert zero["optimizer"]["state"] == {} and zero["loss_sum"] == 0
    assert zero["sample_chain"] == "0" * 64
    assert tensor_state_digest(zero["model"]) == fresh_digest
    assert {int(s["step"]) for s in periodic["optimizer"]["state"].values()} == {500}
    assert len(periodic["optimizer"]["state"]) == len(list(fresh_model.parameters()))
    trained_digest = tensor_state_digest(periodic["model"])
    assert trained_digest != fresh_digest
    pair_manifest = read_sealed(args.pairs / "manifest.json")
    assert sha256_file(args.pairs / "manifest.json") == recipe["pair_manifest_sha256"]
    for name in ("sources.jsonl", "train.bin"):
        assert sha256_file(args.pairs / name) == pair_manifest["outputs"][name]["sha256"]
    sources = {s["episode_index"]: s for s in map(json.loads, (args.pairs / "sources.jsonl").read_text().splitlines())
               if s["split"] == "train"}
    offsets, count = {}, 0
    for ep, source in sorted(sources.items()):
        for horizon in HORIZONS:
            offsets[ep, horizon] = count
            count += max(0, source["episode_length"] - horizon)
    assert count == recipe["pair_presentations"] == 8221365
    records = np.memmap(args.pairs / "train.bin", dtype=np.uint8, mode="r", shape=(count, RECORD.size))
    visited, chain = set(), "0" * 64
    for index in range(500):
        digests = []
        for ep, anchor in order.references(index):
            for horizon in HORIZONS:
                if anchor + horizon >= sources[ep]["episode_length"]:
                    continue
                ordinal = offsets[ep, horizon] + anchor
                assert ordinal not in visited
                visited.add(ordinal)
                fields = RECORD.unpack(records[ordinal].tobytes())
                assert fields[:4] == (ep, anchor, anchor + horizon, horizon)
                digests.append(fields[-1].hex())
        assert len(digests) == int(order.pair_counts[index])
        batch = hashlib.sha256("\n".join(digests).encode()).digest()
        chain = hashlib.sha256(bytes.fromhex(chain) + batch).hexdigest()
    assert chain == periodic["sample_chain"] and len(visited) == periodic["pairs_seen"]
    return seal_contract({
        "artifact_type": "eventground_wam.jepa_production_launch_verification", "schema_version": 1,
        "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity_digest": identity["digest"], "recipe_digest": recipe["digest"],
        "preflight_digest": preflight["digest"], "pilot_report_digest": pilot["digest"],
        "source_inventory_digest": inventory["digest"],
        "fresh_seeded_model_digest": fresh_digest, "fresh_actual_checkpoint_matches_seeded_model": True,
        "fresh_optimizer_empty": True, "production_identity_distinct_from_pilot": True,
        "base_and_teacher_initialization_match_pinned_pilot": True,
        "initial_checkpoint": commits[0], "first_periodic_checkpoint": commits[1],
        "first_periodic_model_digest": trained_digest,
        "all_model_and_optimizer_tensors_finite": True, "complete_rng_state_present": True,
        "optimizer_updates_verified": 500, "unique_training_pairs_verified": len(visited),
        "independently_replayed_sample_chain": chain,
        "state_component_digests": [{k: tree_digest(v) for k, v in p.items()} for p in payloads],
        "snapshot_digests": [tree_digest(p) for p in payloads],
        "validation_payload_reads": 0, "test_payload_reads": 0,
        "production_training_complete": False, "candidate_accepted": False,
        "limits": "CPU audit of preserved startup payloads and training metadata. No interruption or GPU replay of the live run. Final coverage, frozen-weight and export checks remain pending.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "run", "evidence", "pilot-run", "pairs", "order", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    result = verify(args)
    write_json(args.receipt, result)
    print(json.dumps(result, indent=2), flush=True)
