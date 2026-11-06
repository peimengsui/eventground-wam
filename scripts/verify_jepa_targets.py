#!/usr/bin/env python3
"""Recompute target diagnostic measurements from saved features using NumPy."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.future_pairs import read_sealed, sha256_file, write_json


def verify(run, contract_path, output):
    if output.exists():
        raise FileExistsError(output)
    contract = read_sealed(contract_path)
    report = read_sealed(run/"report.json")
    identity = read_sealed(run/"identity.json")
    selection = read_sealed(run/"selection.json")["samples"]
    pairs = read_sealed(run/"pair-diagnostics.json")["pairs"]
    assert report["contract_digest"] == contract["digest"] == identity["contract_digest"]
    assert report["identity_digest"] == identity["digest"]
    assert sha256_file(run/"feature-bank.pt") == report["feature_bank_sha256"]
    bank = torch.load(run/"feature-bank.pt", map_location="cpu", weights_only=True)
    assert bank["contract_digest"] == contract["digest"]
    values = bank["features"].numpy()
    assert values.dtype == np.float32 and values.shape == (720, 128, 2048)
    assert np.isfinite(values).all()
    assert len(selection) == 120 and len(pairs) == 600
    assert bank["anchor_indices"] == list(range(0, 720, 6))
    assert len({s["task_key"] for s in selection}) == 40
    assert len({s["episode_index"] for s in selection}) == 40
    assert all(s["split"] == "train" for s in selection)
    by_cell = {f"{c}/h{h}": [] for c in ("front", "wrist") for h in (2, 4, 8, 16, 32)}
    pair_error = norm_error = 0.
    for index, sample in enumerate(selection):
        block = values[index*6:(index+1)*6].astype(np.float64)
        norm_error = max(norm_error, float(np.max(np.abs(np.linalg.norm(block, axis=-1)-1))))
        for k, delta in enumerate((0, 2, 4, 8, 16, 32)):
            assert bank["identities"][index*6+k] == {**sample, "frame": sample["anchor"]+delta}
        for hindex, h in enumerate((2, 4, 8, 16, 32), start=1):
            row = pairs[index*5+hindex-1]
            for key in ("task_key", "episode_index", "anchor", "split", "source_digest"):
                assert row[key] == sample[key]
            assert row["horizon"] == h and row["target"] == sample["anchor"]+h
            scores = (1-np.sum(block[0]*block[hindex], axis=-1)).reshape(2, 64).mean(axis=1)
            for c, camera in enumerate(("front", "wrist")):
                pair_error = max(pair_error, abs(float(scores[c])-row[f"{camera}_cosine_distance"]))
                by_cell[f"{camera}/h{h}"].append(float(scores[c]))
    cells = {name: float(np.mean(scores)) for name, scores in by_cell.items()}
    cell_error = max(abs(value-report["copy_cosine_distance_by_camera_horizon"][name])
                     for name, value in cells.items())
    recomputed = {}
    for camera_index, camera in enumerate(("front", "wrist")):
        anchors = values[::6, camera_index*64:(camera_index+1)*64].astype(np.float64)
        pooled = anchors.mean(axis=1)
        centered = pooled-pooled.mean(axis=0)
        eigenvalues = np.maximum(np.linalg.eigvalsh(centered@centered.T), 0)
        probabilities = eigenvalues/eigenvalues.sum()
        positive = probabilities[probabilities > 0]
        rank = float(np.exp(-np.sum(positive*np.log(positive))))
        stats = {"pooled_effective_rank": rank,
                 "pooled_across_example_std_rms": float(np.sqrt(np.mean(centered**2))),
                 "fixed_position_across_example_std_rms": float(np.sqrt(anchors.var(axis=0).mean())),
                 "within_image_spatial_std_rms": float(np.sqrt(anchors.var(axis=1).mean()))}
        for key, value in stats.items():
            assert math.isclose(value, report["camera_diagnostics"][camera][key], abs_tol=1e-9)
        recomputed[camera] = stats
    first = torch.load(run/"first-observation-check.pt", map_location="cpu", weights_only=True)
    assert torch.equal(first["normalized"], bank["features"][:6])
    raw = first["raw"].float().numpy().astype(np.float64)
    centered = raw-raw.mean(axis=-1, keepdims=True)
    layer_norm = centered/np.sqrt((centered**2).mean(axis=-1, keepdims=True)+1e-5)
    normalized = layer_norm/np.linalg.norm(layer_norm, axis=-1, keepdims=True)
    normalization_error = float(np.max(np.abs(normalized-first["normalized"].numpy())))
    gates = contract["diagnostic"]["criteria"]
    checks = {
        "report_and_bank_identity": True,
        "pair_and_camera_horizon_recomputation": pair_error < 1e-10 and cell_error < 1e-10,
        "independent_normalization": normalization_error < 1e-7,
        "unit_norms": norm_error <= gates["unit_norm_error_max"],
        "across_observation_diversity": all(
            d["pooled_effective_rank"] >= gates["pooled_effective_rank_min"]
            and d["fixed_position_across_example_std_rms"] >= gates["fixed_position_std_rms_min"]
            for d in recomputed.values()),
        "temporal_signal": all(cells[f"{c}/h32"] >= gates["h32_copy_cosine_distance_min"]
                               for c in ("front", "wrist")),
        "reported_integration_checks": all(report["checks"].values()) and report["passed"],
        "frozen_weight_hashes": report["vision_state_before"] == report["vision_state_after"],
    }
    result = seal_contract({
        "artifact_type": "eventground_wam.jepa_target_independent_verification",
        "report_digest": report["digest"], "contract_digest": contract["digest"],
        "auditor_sha256": sha256_file(Path(__file__)), "checks": checks,
        "passed": all(checks.values()), "maximum_pair_error": pair_error,
        "maximum_cell_error": cell_error, "independent_normalization_max_error": normalization_error,
        "unit_norm_max_error": norm_error, "camera_diagnostics": recomputed,
        "copy_cosine_distance_by_camera_horizon": cells,
        "limits": "Independent NumPy arithmetic over saved feature bank and identity checks; not a second full encoder/data replay.",
    })
    write_json(output, result)
    print(f"Independent verification passed: {result['passed']}", flush=True)
    if not result["passed"]:
        raise ValueError("independent target verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run, args.contract, args.output)
