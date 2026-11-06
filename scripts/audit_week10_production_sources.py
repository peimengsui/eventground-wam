"""Recompute source-action bindings and prepare a non-admitted capture proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.instrumented_supervision import numeric_digest, physical_source_id


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sealed(path):
    d = json.loads(path.read_text())
    assert d["digest"] == contract_digest(d), str(path)
    return d


def write(path, value):
    with path.open("x") as f:
        json.dump(seal_contract(value), f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.first, args.recovery, args.output = (
        args.first.resolve(),
        args.recovery.resolve(),
        args.output.resolve(),
    )
    root = Path(__file__).resolve().parents[1]
    first, recovery = (
        read_sealed(args.first / "contract.json"),
        read_sealed(args.recovery / "contract.json"),
    )
    assert recovery["parent_digest"] == first["digest"]
    bundles = [(args.first, first), (args.recovery, recovery)]
    continuation = None
    if args.continuation:
        args.continuation = args.continuation.resolve()
        continuation = read_sealed(args.continuation / "contract.json")
        assert continuation["parent_digest"] == recovery["digest"]
        bundles.append((args.continuation, continuation))
    groups = {
        g["group_digest"]: g
        for g in map(
            json.loads,
            (root / "configs/datasets/libero-plus-trajectory-v2/groups.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    for base, contract in bundles:
        for name, h in contract["bundle_sha256"].items():
            assert sha(base / name) == h, name
    totals = Counter()
    audited = []
    selected = []
    reused = []
    final_ids = set()
    summaries = []
    for task in recovery["tasks"]:
        folder = args.recovery / "run" / task["task_key"].replace("/", "-")
        if continuation and task["task_key"] in {t["task_key"] for t in continuation["tasks"]}:
            previous = read_sealed(folder / "result.json")
            folder = args.continuation / "run" / folder.name
            current = read_sealed(folder / "result.json")
            assert current["results"][: len(previous["results"])] == previous["results"]
        r = read_sealed(folder / "result.json")
        for name, h in r["evidence_sha256"].items():
            assert sha(folder / name) == h, str(folder / name)
        old = args.first / "run" / folder.name / "progress.jsonl"
        old_rows = [json.loads(x) for x in old.read_text().splitlines()] if old.exists() else []
        assert r["results"][: len(old_rows)] == old_rows
        assert len({v["group_digest"] for v in r["results"]}) == len(r["results"])
        assert r["exact_bindings"] == sum(
            v["status"] == "exact_action_binding" for v in r["results"]
        )
        expected_candidates = {v["group"]["group_digest"]: v for v in task["candidates"]}
        matches = []
        for v in r["results"]:
            totals[v["status"]] += 1
            g = groups[v["group_digest"]]
            candidate = expected_candidates[v["group_digest"]]
            assert (
                g == candidate["group"]
                and v["episode"] == candidate["episode"]
                and g["split"] == "train"
            )
            assert g["episode_length"] not in {
                p["episode_length"]
                for p in groups.values()
                if p["split"] != "train" and p["task_key"] == g["task_key"]
            }
            if "action_digest" not in v:
                assert v["status"] in ("no_equal_length_source", "ambiguous_source_length")
                continue
            arrays = np.load(folder / f"target-{v['episode']}.npz")
            a, s = arrays["actions"], arrays["states"]
            assert a.shape == (g["episode_length"], 7) and s.shape == (g["episode_length"], 8)
            assert numeric_digest(s, a) == v["target_numeric_digest"] in g["numeric_digests"]
            assert numeric_digest(a) == v["action_digest"]
            actual = []
            for demo in v["equal_length_source_demos"]:
                source = np.load(folder / f"{demo}-actions.npz")["actions"]
                assert source.shape == a.shape and np.isfinite(source).all()
                if numeric_digest(source) == numeric_digest(a):
                    actual.append(demo)
            assert actual == v["source_matches"]
            if v["status"] != "exact_action_binding":
                assert len(actual) != 1
                continue
            assert len(actual) == 1
            sid = physical_source_id(
                recovery["repository"], recovery["revision"], task["hdf5_path"], actual[0]
            )
            assert (
                sid == v["source_id"]
                and sha(folder / f"{actual[0]}-actions.npz") == v["source_actions_sha256"]
            )
            assert v["known_alias_episode_ids"] == g["episode_indices"]
            assert sid not in final_ids, "Duplicate physical source would inflate review coverage"
            final_ids.add(sid)
            matches.append(
                {
                    "task_key": task["task_key"],
                    "source_id": sid,
                    "repository": recovery["repository"],
                    "revision": recovery["revision"],
                    "hdf5_path": task["hdf5_path"],
                    "demo_id": actual[0],
                    "frames": len(a),
                    "group_digest": g["group_digest"],
                    "known_alias_episode_ids": g["episode_indices"],
                    "target_episode": v["episode"],
                    "action_digest": numeric_digest(a),
                    "evidence_path": str(folder.relative_to(root) / "result.json"),
                    "evidence_sha256": sha(folder / "result.json"),
                    "known_full_trace_partition": "train",
                    "historical_transformed_alias_closure": False,
                    "production_admitted": False,
                }
            )
        selected.extend(dict(x, queue_role="initial_review") for x in matches[:5])
        reused.extend(
            dict(x, queue_role="coverage_reserve") for x in matches[5 : task["target_bindings"]]
        )
        summaries.append(
            {
                "task_key": task["task_key"],
                "status": r["status"],
                "exact_bindings": len(matches),
                "initial": min(5, len(matches)),
                "reserve": max(0, len(matches) - 5),
                "error": r.get("error"),
            }
        )
        audited.append(
            {
                "task_key": task["task_key"],
                "result_digest": r["digest"],
                "preserved_completed_comparisons": len(old_rows),
                "all_saved_numeric_comparisons_recomputed": True,
            }
        )
    range_bytes = sum(
        json.loads(line)["length"]
        for base, _ in bundles
        for p in (base / "run").rglob("http-ranges.jsonl")
        for line in p.read_text().splitlines()
    )
    assert range_bytes <= first["resources"]["cumulative_http_bytes"]
    output_bytes = sum(
        p.stat().st_size for base, _ in bundles for p in base.rglob("*") if p.is_file()
    )
    assert output_bytes <= first["resources"]["output_bytes"]
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "mapping-audit.json",
        {
            "artifact_type": "eventground_wam.production_mapping_audit",
            "first_contract_digest": first["digest"],
            "recovery_contract_digest": recovery["digest"],
            "continuation_contract_digest": continuation["digest"] if continuation else None,
            "audited_tasks": audited,
            "task_summary": summaries,
            "comparison_outcomes": dict(totals),
            "unique_exact_sources": len(final_ids),
            "initial_review_sources": len(selected),
            "reserve_sources": len(reused),
            "source_tasks": sum(x["exact_bindings"] > 0 for x in summaries),
            "tasks_with_five_initial_sources": sum(x["initial"] == 5 for x in summaries),
            "cumulative_http_bytes": range_bytes,
            "mapping_output_bytes": output_bytes,
            "heldout_payload_reads": 0,
            "simulator_steps": 0,
            "labels_extracted_by_mapper": False,
            "production_partition_closure_verified": False,
            "qualification": "Exact complete float32 action identity, all declared candidates independently recomputed; no claim about transformed historical aliases or scene equivalence.",
            "production_admitted": False,
            "verifier_sha256": sha(Path(__file__)),
        },
    )
    queue = selected + reused
    write(
        args.output / "recording-plan-proposal.json",
        {
            "artifact_type": "eventground_wam.production_recording_plan_proposal",
            "status": "prepared_not_admitted_not_a_frozen_capture_manifest",
            "parent_study_digest": first["study_digest"],
            "source_mapping_digest": read_sealed(args.output / "mapping-audit.json")["digest"],
            "selection": "Same prospective per-task length-tercile/hash order; first five unique exact source bindings initial, remaining up to predeclared 8/7 per task reserved for coverage. No event labels or robot success used.",
            "initial_recordings": len(selected),
            "coverage_reserve_recordings": len(reused),
            "rows": queue,
            "proposed_resource_ceilings": {
                "new_control_steps": sum(x["frames"] + 10 for x in queue),
                "observed_source_frames": sum(x["frames"] for x in queue),
                "wall_seconds": 7200,
                "gpu_hours_upper_bound": 2,
                "output_bytes": 100 * 1024**3,
                "minimum_free_bytes": 150 * 1024**3,
                "maximum_recordings": 300,
                "attempts_per_source": 1,
                "automatic_trajectory_retries": 0,
                "heldout_payload_reads": 0,
                "training_updates": 0,
            },
            "budget_basis": "Three-case corrected pilot took 15.235 seconds for 545 control steps; compiled models dominate 184.6-241.7 MiB per recording. Two hours/100 GiB are proposed ceilings, not a measured production guarantee or authorization. No model deduplication/compression change assumed.",
            "capture_requirements": [
                "Before launch resolve source-level partition closure and seal exact inputs, source state/XML/action hashes, recorder/environment and manifest",
                "Ten settling steps; fresh independent normalized snapshot immediately before each complete source action; 20 Hz front/wrist 256 RGB + 8D proprio + object/gripper pose + raw contacts + full state/model",
                "Preserve every attempt and task-failure outcome; stop on runtime/integrity failure; no success-based source replacement",
                "All new variants of a physical source remain together; no historical protected source reassignment",
                "Separate complete-episode independent QA before label admission; do not satisfy class quotas by duplicating prompts",
            ],
            "blocking_conditions": [
                "Historical cropped/filtered/resampled aliases remain unresolved; full-trace identity alone does not prove protected source closure",
                "Exact production capture payload identities and admission are not sealed",
                "Independent label-quality gate and per-class prompt coverage remain unmeasured",
            ],
            "production_manifest_sealed": False,
            "production_capture_authorized": False,
            "labels_admitted": False,
            "training_authorized": False,
        },
    )
    print(
        json.dumps(
            {
                "mapping": dict(totals),
                "initial": len(selected),
                "reserve": len(reused),
                "tasks": len(summaries),
                "bytes": range_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
