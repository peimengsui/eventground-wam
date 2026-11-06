#!/usr/bin/env python3
"""Read-only verification of the fixed production endpoint and exported candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.future_pairs import HORIZONS, RECORD, read_sealed, sha256_file, write_json
from eventground_wam.pixel_predictor import load_pixel_checkpoint, tensor_state_digest
from eventground_wam.pixel_training import CPTOrder, tree_digest, verify_recipe


def verify(args):
    torch.set_num_threads(2)
    report = read_sealed(args.run / "report.json")
    identity = read_sealed(args.run / "identity.json")
    initial = read_sealed(args.run / "initialization.json")
    budget = read_sealed(args.run / "active-budget.json")
    recipe = verify_recipe(args.source / "configs/training/week5_pixel_cpt_v1.json", args.source)
    order = CPTOrder(args.order, recipe["order_manifest_sha256"])
    assert report["status"] == "complete" and report["phase"] == identity["phase"] == "train"
    assert report["identity_digest"] == identity["digest"] == initial["identity_digest"]
    assert report["recipe_digest"] == identity["recipe_digest"] == recipe["digest"]
    assert report["order_digest"] == identity["order_digest"] == order.manifest["digest"]
    assert report["global_step"] == recipe["total_updates"] == len(order) == 221633
    assert report["pairs_seen"] == recipe["pair_presentations"] == int(order.pair_counts.sum()) == 8221365
    assert report["source_sha256"] == identity["source_sha256"]
    for name, digest in identity["source_sha256"].items():
        assert sha256_file(args.source / name) == digest, name
    assert report["base_state_sha256_before"] == report["base_state_sha256_after"] == initial["base_state_digest"]
    assert report["trainable_base_parameters"] == report["validation_payloads_read"] == report["test_payloads_read"] == 0
    assert initial["optimizer_state_entries"] == 0 and not initial["parent_or_diagnostic_weights_loaded"]
    assert budget["identity"] == identity["digest"] and not budget["running"]
    assert budget["active_seconds"] < recipe["maximum_active_training_hours"] * 3600
    status = json.loads((args.run / "status.json").read_text())
    assert status["status"] == "complete" and status["report_digest"] == report["digest"]
    record = read_sealed(args.run / "recovery/latest.json")
    assert record == report["checkpoint"]
    path = args.run / "recovery" / record["checkpoint"]
    assert path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert tree_digest(payload) == report["snapshot_digest"]
    assert {k: tree_digest(v) for k, v in payload.items()} == report["state_component_digests"]
    assert payload["identity_digest"] == identity["digest"]
    assert payload["global_step"] == payload["scheduler"]["last_epoch"] == report["global_step"]
    assert payload["pairs_seen"] == report["pairs_seen"]
    assert {int(s["step"]) for s in payload["optimizer"]["state"].values()} == {221633}
    assert set(payload["rng"]) == {"python", "numpy", "torch", "cuda"} and len(payload["rng"]["cuda"]) == 1
    assert all(torch.isfinite(v).all() for v in payload["model"].values())
    assert all(torch.isfinite(v).all() for s in payload["optimizer"]["state"].values()
               for v in s.values() if isinstance(v, torch.Tensor))
    candidate = args.run / "C_pixel-candidate.pt"
    model, header = load_pixel_checkpoint(candidate)
    assert header["metadata"]["training_identity"] == identity["digest"]
    assert header["metadata"]["global_step"] == 221633 and header["metadata"]["pairs_seen"] == 8221365
    assert tensor_state_digest(model.state_dict()) == tensor_state_digest(payload["model"]) == header["state_sha256"]
    assert model.parameter_budget() == recipe["parameter_budget"]
    pair_manifest = read_sealed(args.pairs / "manifest.json")
    assert sha256_file(args.pairs / "manifest.json") == recipe["pair_manifest_sha256"]
    for name in ("sources.jsonl", "train.bin"):
        assert sha256_file(args.pairs / name) == pair_manifest["outputs"][name]["sha256"]
    sources = {s["episode_index"]: s for s in map(json.loads, (args.pairs / "sources.jsonl").read_text().splitlines())
               if s["split"] == "train"}
    assert set(report["decoded_episodes"]) == set(sources)
    offsets = {}; count = 0
    for ep, source in sorted(sources.items()):
        for horizon in HORIZONS:
            offsets[ep, horizon] = count
            count += max(0, source["episode_length"] - horizon)
    assert count == 8221365
    records = np.memmap(args.pairs / "train.bin", dtype=np.uint8, mode="r", shape=(count, RECORD.size))
    visited = np.zeros(count, dtype=np.bool_)
    chain = "0" * 64
    # Independently reconstruct every acknowledged pair and the complete batch chain.
    for index in range(len(order)):
        digests = []
        for ep, anchor in order.references(index):
            for horizon in HORIZONS:
                if anchor + horizon >= sources[ep]["episode_length"]:
                    continue
                ordinal = offsets[ep, horizon] + anchor
                assert not visited[ordinal]
                visited[ordinal] = True
                raw = records[ordinal].tobytes()
                fields = RECORD.unpack(raw)
                assert fields[:4] == (ep, anchor, anchor + horizon, horizon)
                digests.append(fields[-1].hex())
        assert len(digests) == int(order.pair_counts[index])
        batch = hashlib.sha256("\n".join(digests).encode()).digest()
        chain = hashlib.sha256(bytes.fromhex(chain) + batch).hexdigest()
    assert visited.all() and chain == payload["sample_chain"] == report["sample_chain"]
    return seal_contract({"artifact_type": "eventground_wam.pixel_cpt_final_verification", "status": "passed",
                          "report_digest": report["digest"], "identity_digest": identity["digest"],
                          "recipe_digest": recipe["digest"], "checkpoint": record,
                          "candidate_sha256": sha256_file(candidate), "candidate_header": header,
                          "independently_replayed_sample_chain": chain, "unique_training_pairs_verified": count,
                          "optimizer_updates_verified": len(order), "training_episodes_verified": len(sources),
                          "full_recovery_state_matches_report": True, "export_matches_final_recovery_weights": True,
                          "frozen_base_unchanged": True, "fresh_initialization_verified": True,
                          "test_payloads_read": 0, "validator_sha256": sha256_file(Path(__file__)),
                          "active_training_hours": report["active_seconds_all_processes"] / 3600,
                          "candidate_accepted": False, "acceptance_requires": "Step 6 held-out gate"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "order", "pairs", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    result = verify(args)
    write_json(args.receipt, result)
    print(json.dumps(result, indent=2), flush=True)
