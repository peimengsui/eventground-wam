from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import (
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_MANIFEST_SHA256,
    canonical_json_bytes,
    seal_contract,
)
from eventground_wam.future_pairs import (
    CAMERAS,
    DOMAIN,
    HORIZONS,
    IDENTITY,
    PAIR_VERSION,
    RECORD,
    expand_split_groups,
    pair_bytes,
    read_pair,
    read_sealed,
    sha256_file,
    validate_sources,
    verify_bundle,
    write_bundle,
    write_json,
)

ROOT = Path(__file__).parents[1]
PROVENANCE = {"split_manifest_sha256": "a" * 64}


@pytest.fixture()
def catalog():
    groups = [{"group_digest": str(i + 1) * 64, "task_key": f"task-{i}",
               "episode_length": length, "episode_indices": [i], "split": split}
              for i, (length, split) in enumerate(
                  [(8, "train"), (35, "validation"), (50, "test")])]
    assignments = expand_split_groups(groups)
    sources = []
    for row in assignments[:2]:
        i = row["episode_index"]
        files = {"parquet": {"path": f"data/chunk-000/episode_{i:06d}.parquet",
                             "sha256": "b" * 64, "bytes": 42}}
        files.update({camera: {"path": f"videos/chunk-000/{camera}/episode_{i:06d}.mp4",
                               "sha256": "c" * 64, "bytes": 84} for camera in CAMERAS})
        sources.append(seal_contract({
            "artifact_type": "eventground_wam.future_pair_source", "version": PAIR_VERSION,
            **{key: row[key] for key in
               ("episode_index", "episode_length", "task_key", "group_digest", "split")},
            "assignment_digest": hashlib.sha256(canonical_json_bytes(row)).hexdigest(),
            "task_index": i, "instruction": f"task {i}", **PROVENANCE,
            "dataset": {"repo_id": SOURCE_DATASET_REPO_ID, "revision": SOURCE_DATASET_REVISION,
                        "source_manifest_sha256": SOURCE_MANIFEST_SHA256}, "files": files,
        }))
    return sources, assignments


def reseal(value):
    return seal_contract({key: item for key, item in value.items() if key != "digest"})


def reseal_file_hash(bundle, name):
    manifest = read_sealed(bundle / "manifest.json")
    manifest["outputs"][name]["sha256"] = sha256_file(bundle / name)
    manifest["outputs"][name]["bytes"] = (bundle / name).stat().st_size
    write_json(bundle / "manifest.json", reseal(manifest))


def test_complete_horizon_coverage_and_action_interval(catalog, tmp_path):
    sources, assignments = catalog
    bundle = tmp_path / "pairs"
    manifest = write_bundle(bundle, sources, assignments, PROVENANCE)
    assert manifest["counts"]["train"]["pairs_by_horizon"] == {
        "2": 6, "4": 4, "8": 0, "16": 0, "32": 0}
    assert manifest["counts"]["validation"]["pairs"] == sum(35 - h for h in HORIZONS)
    assert manifest["counts"]["train"]["excluded_tail_anchors_by_horizon"]["32"] == 8
    assert manifest["test_episodes_excluded"] == 1
    assert verify_bundle(bundle, assignments, PROVENANCE)["status"] == "verified"
    first = read_pair(bundle, "train", 0)
    assert (first["anchor"], first["target"], first["horizon"]) == (0, 2, 2)
    assert first["action_stop_exclusive"] - first["action_start"] == 2
    last = read_pair(bundle, "validation", manifest["counts"]["validation"]["pairs"] - 1)
    assert (last["anchor"], last["target"], last["horizon"]) == (2, 34, 32)
    assert set(last["source"]["files"]) == {"parquet", *CAMERAS}


def test_pair_digest_binds_both_cameras_assignment_and_horizon(catalog):
    source = catalog[0][0]
    row = pair_bytes(source, 1, 2)
    identity = IDENTITY.pack(0, 1, 3, 2, 0)
    expected = hashlib.sha256(DOMAIN + bytes.fromhex(source["digest"]) + identity).digest()
    assert RECORD.unpack(row)[-1] == expected
    changed = deepcopy(source)
    changed["files"][CAMERAS[1]]["sha256"] = "d" * 64
    assert pair_bytes(reseal(changed), 1, 2) != row
    assert pair_bytes(source, 1, 4) != row


@pytest.mark.parametrize("anchor,horizon", [(-1, 2), (6, 2), (0, 8), (0, 3), (True, 2)])
def test_reject_invalid_boundaries_and_horizons(catalog, anchor, horizon):
    with pytest.raises(ValueError, match="bounds"):
        pair_bytes(catalog[0][0], anchor, horizon)


def test_byte_determinism_for_reversed_source_and_assignment_input(catalog, tmp_path):
    sources, assignments = catalog
    first = write_bundle(tmp_path / "a", sources, assignments, PROVENANCE)
    second = write_bundle(tmp_path / "b", reversed(sources), list(reversed(assignments)), PROVENANCE)
    assert first == second
    for name in (*first["outputs"], "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    with pytest.raises(FileExistsError):
        write_bundle(tmp_path / "a", sources, assignments, PROVENANCE)


@pytest.mark.parametrize("damage", ["duplicate", "bounds", "digest", "trailing", "truncate"])
def test_semantic_validator_rejects_corruption_even_with_resealed_file_hash(
    catalog, tmp_path, damage,
):
    sources, assignments = catalog
    write_bundle(tmp_path / "pairs", sources, assignments, PROVENANCE)
    path = tmp_path / "pairs/train.bin"
    data = bytearray(path.read_bytes())
    if damage == "duplicate":
        data[RECORD.size:2 * RECORD.size] = data[:RECORD.size]
    elif damage == "bounds":
        data[8:12] = (8).to_bytes(4, "little")
        data[46:78] = hashlib.sha256(DOMAIN + data[14:46] + data[:14]).digest()
    elif damage == "digest":
        data[77] ^= 1
    elif damage == "trailing":
        data += data[:RECORD.size]
    else:
        data = data[:-1]
    path.write_bytes(data)
    reseal_file_hash(tmp_path / "pairs", "train.bin")
    with pytest.raises(ValueError):
        verify_bundle(tmp_path / "pairs", assignments, PROVENANCE)


def test_split_leakage_and_test_sources_are_rejected(catalog):
    sources, assignments = catalog
    leaky = deepcopy(assignments)
    leaky[2]["group_digest"] = leaky[0]["group_digest"]
    with pytest.raises(ValueError, match="cross-split"):
        validate_sources(sources, leaky)
    with pytest.raises(ValueError, match="coverage"):
        validate_sources(sources + [sources[0]], assignments)
    injected = deepcopy(sources[0])
    injected["episode_index"] = 2
    injected["split"] = "test"
    with pytest.raises(ValueError, match="coverage"):
        validate_sources(sources + [reseal(injected)], assignments)


@pytest.mark.parametrize("damage", ["source_seal", "camera_swap", "group", "assignment", "path"])
def test_source_integrity_is_checked(catalog, damage):
    sources, assignments = deepcopy(catalog)
    if damage == "source_seal":
        sources[0]["instruction"] = "changed"
    else:
        if damage == "camera_swap":
            sources[0]["files"][CAMERAS[0]] = sources[0]["files"][CAMERAS[1]]
        elif damage == "group":
            sources[0]["group_digest"] = "d" * 64
        elif damage == "assignment":
            sources[0]["assignment_digest"] = "d" * 64
        else:
            sources[0]["files"]["parquet"]["path"] = "../test.parquet"
        sources[0] = reseal(sources[0])
    with pytest.raises(ValueError):
        validate_sources(sources, assignments)


def test_test_guard_runs_before_any_source_access():
    spec = importlib.util.spec_from_file_location("prepare_future_pairs",
                                                 ROOT / "scripts/prepare_future_pairs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="test payload access"):
        module.inspect_source(None, None, {"split": "test"}, None, None)


def test_checked_in_rule_expands_to_exact_frozen_split():
    config = read_sealed(ROOT / "configs/datasets/trajectory_v2_future_pairs_v1.json")
    path = ROOT / "configs/datasets/libero-plus-trajectory-v2/groups.jsonl"
    groups = list(map(json.loads, path.read_text().splitlines()))
    rows = expand_split_groups(groups)
    assert hashlib.sha256(b"".join(canonical_json_bytes(row) + b"\n" for row in rows)).hexdigest() == (
        config["inputs"]["assignments_sha256"])
    counts = {split: sum(sum(max(0, row["episode_length"] - h) for h in HORIZONS)
                         for row in rows if row["split"] == split)
              for split in ("train", "validation")}
    assert counts == {"train": 8221365, "validation": 1109994}
    assert expand_split_groups(reversed(groups)) == rows
