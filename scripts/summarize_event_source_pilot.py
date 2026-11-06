"""Independently verify saved source-pilot evidence and summarize every candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract


def read_sealed(path):
    data = json.loads(path.read_text())
    if contract_digest(data) != data["digest"]:
        raise ValueError(f"Invalid digest: {path}")
    return data


def numeric_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f4")
        digest.update(str(value.shape).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def verify(base, bundle_name="bundle", run_name="run"):
    contract = read_sealed(base / bundle_name / "contract.json")
    result = read_sealed(base / run_name / "results.json")
    assert result["contract_digest"] == contract["digest"]
    for name, expected in contract["bundle_files_sha256"].items():
        assert hashlib.sha256((base / bundle_name / name).read_bytes()).hexdigest() == expected
    rows = result["candidate_results"]
    assert [r["candidate"] for r in rows] == contract["candidates"]
    groups = {
        g["group_digest"]: g
        for g in map(json.loads, (base / bundle_name / "groups.jsonl").read_text().splitlines())
    }
    verified_files = comparisons = 0
    action_matches = []
    details = []
    for row in rows:
        rank = row["candidate"]["queue_rank"]
        folder = base / run_name / f"candidate-{rank:02d}"
        group = groups[row["candidate"]["group_digest"]]
        n = row["candidate"]["frame_count"]
        assert group["split"] == row["candidate"]["split"] == "train"
        assert row["candidate"]["source_episode_index"] in group["episode_indices"]
        assert n == group["episode_length"]
        saved = read_sealed(folder / "result.json")
        saved.pop("digest")
        assert saved == row
        for name, expected in row["evidence_sha256"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
            verified_files += 1
        if rank in contract["heldout_protection"]["excluded_queue_ranks"]:
            assert row["status"] == "excluded_heldout_task_length_collision"
            assert row["target_payload_reads"] == row["original_demo_payload_reads"] == 0
            assert not list(folder.glob("*.npz"))
            assert not (folder / "http-ranges.jsonl").exists()
        if (folder / "target-numeric.npz").exists():
            with np.load(folder / "target-numeric.npz", allow_pickle=False) as target:
                digest = numeric_digest(target["states"], target["actions"])
                assert digest == row["target_numeric_digest"] and digest in group["numeric_digests"]
                assert target["states"].shape == (n, 8) and target["actions"].shape == (n, 7)
        if row["comparisons"]:
            with np.load(folder / "target-numeric.npz", allow_pickle=False) as target:
                for comparison in row["comparisons"]:
                    with np.load(
                        folder / f"{comparison['demo_id']}-numeric.npz", allow_pickle=False
                    ) as source:
                        assert source["states"].shape == (n, 8) and source["actions"].shape == (
                            n,
                            7,
                        )
                        assert (
                            numeric_digest(source["states"], source["actions"])
                            == comparison["source_numeric_digest"]
                        )
                        action_error = np.abs(
                            target["actions"].astype(np.float64) - source["actions"]
                        )
                        state_error = np.abs(target["states"].astype(np.float64) - source["states"])
                        assert np.isfinite(action_error).all() and np.isfinite(state_error).all()
                        passed = (
                            np.all(action_error <= 1e-6)
                            and np.all(state_error <= 1e-5)
                            and np.array_equal(target["actions"][:, 6], source["actions"][:, 6])
                        )
                        recorded = comparison["alignment"]
                        assert contract_digest(recorded) == recorded["digest"]
                        assert recorded["frame_count"] == n
                        assert (
                            recorded["source_first_frame"] == 0
                            and recorded["source_last_frame"] == n - 1
                        )
                        assert bool(passed) == recorded["numeric_alignment_passed"]
                        assert float(action_error.max()) == recorded["max_action_absolute_error"]
                        assert float(state_error.max()) == recorded["max_state_absolute_error"]
                        if np.all(action_error <= 1e-6) and np.array_equal(
                            target["actions"][:, 6], source["actions"][:, 6]
                        ):
                            offsets = []
                            for offset in (-1, 0, 1):
                                left = (
                                    target["states"][:-1]
                                    if offset == -1
                                    else target["states"][1:]
                                    if offset == 1
                                    else target["states"]
                                )
                                right = (
                                    source["states"][1:]
                                    if offset == -1
                                    else source["states"][:-1]
                                    if offset == 1
                                    else source["states"]
                                )
                                error = np.abs(left.astype(np.float64) - right)
                                offsets.append(
                                    {
                                        "offset": offset,
                                        "overlap_frames": len(left),
                                        "omitted_boundaries_per_sequence": abs(offset),
                                        "maximum_absolute_error": float(error.max()),
                                        "per_channel_max_error": error.max(axis=0).tolist(),
                                    }
                                )
                            action_matches.append(
                                {
                                    "rank": rank,
                                    "episode_index": row["candidate"]["source_episode_index"],
                                    "original_demo_id": comparison["demo_id"],
                                    "max_action_error": float(action_error.max()),
                                    "max_state_error": float(state_error.max()),
                                    "max_eef_position_distance_m": float(
                                        np.linalg.norm(state_error[:, :3], axis=1).max()
                                    ),
                                    "phase_diagnostics_not_admission": offsets,
                                }
                            )
                        comparisons += 1
        http = row.get("http", {})
        assert http.get("bytes", 0) <= contract["resources"]["per_file_http_bytes"]
        assert http.get("requests", 0) <= contract["resources"]["per_file_http_requests"]
        details.append(
            {
                "rank": rank,
                "episode_index": row["candidate"]["source_episode_index"],
                "task": row["candidate"]["task_key"],
                "status": row["status"],
                "equal_length_original_demos": row.get("equal_length_demo_ids", []),
                "comparisons": len(row["comparisons"]),
            }
        )
    counts = dict(sorted(Counter(r["status"] for r in rows).items()))
    matched = sum(
        c["alignment"]["numeric_alignment_passed"] for r in rows for c in r["comparisons"]
    )
    return seal_contract(
        {
            "artifact_type": "eventground_wam.source_matching_pilot_completion",
            "schema_version": 1,
            "contract_digest": contract["digest"],
            "result_digest": result["digest"],
            "execution_complete": len(rows) == len(contract["candidates"]),
            "candidate_count": len(rows),
            "status_counts": counts,
            "unique_numeric_match_targets": counts.get("unique_numeric_match_scene_unverified", 0),
            "passing_numeric_comparisons": matched,
            "action_matching_comparisons": len(action_matches),
            "action_matches_with_state_phase_diagnostics": action_matches,
            "independently_recomputed_comparisons": comparisons,
            "target_payloads_read": sum(r["target_payload_reads"] for r in rows),
            "original_demo_payloads_read": sum(r["original_demo_payload_reads"] for r in rows),
            "full_simulator_states_read": sum(r["full_simulator_state_reads"] for r in rows),
            "http_bytes": sum(r.get("http", {}).get("bytes", 0) for r in rows),
            "http_requests": sum(r.get("http", {}).get("requests", 0) for r in rows),
            "wall_seconds": result["wall_seconds"],
            "evidence_files_verified": verified_files,
            "physical_scene_identity_verified": False,
            "physical_labels_admitted": False,
            "simulator_steps": 0,
            "training_updates": 0,
            "heldout_target_payload_reads": 0,
            "conclusion": (
                "Numeric matches require independent physical-scene and observation-phase qualification."
                if matched
                else "This bounded complete/equal-length matching route did not recover an admitted physical source."
            ),
            "limitations": [
                "Not an exhaustive search of every demonstration source or conversion.",
                "Cropped, no-op-filtered, resampled and regenerated mappings were not attempted.",
                "Metadata-only exclusions and source access failures are not numerical mismatches.",
                "No contact/grasp label extraction or quality review has occurred.",
            ],
            "candidates": details,
        }
    )


def verify_recorded_diagnostic(base):
    contract = read_sealed(base / "recorded-state-diagnostic-contract.json")
    result = read_sealed(base / "recorded-state-diagnostic/results.json")
    primary_contract = read_sealed(base / "bundle/contract.json")
    primary_results = read_sealed(base / "run/results.json")
    assert contract["parent_pilot_digest"] == primary_contract["digest"]
    action_matches = [
        {"rank": r["candidate"]["queue_rank"], "demo_id": c["demo_id"]}
        for r in primary_results["candidate_results"]
        for c in r["comparisons"]
        if c["alignment"]["max_action_absolute_error"] <= 1e-6
        and c["alignment"]["gripper_commands_exact"]
    ]
    assert contract["targets"] == action_matches[:3]
    assert result["contract_digest"] == contract["digest"]
    assert (
        result["executed_source_sha256"]
        == hashlib.sha256((base / "check_source_recorded_gripper.py").read_bytes()).hexdigest()
    )
    assert [{"rank": r["rank"], "demo_id": r["demo_id"]} for r in result["results"]] == contract[
        "targets"
    ]
    checked = []
    for row in result["results"]:
        folder = base / "recorded-state-diagnostic" / f"candidate-{row['rank']:02d}"
        saved = read_sealed(folder / "result.json")
        saved.pop("digest")
        assert saved == row
        original = next(
            r
            for r in primary_results["candidate_results"]
            if r["candidate"]["queue_rank"] == row["rank"]
        )
        for metric, bound in [
            ("bytes", "per_file_http_bytes"),
            ("requests", "per_file_http_requests"),
        ]:
            assert row["http"][metric] <= contract["resources"][bound]
            assert (
                row["http"][metric] + original["http"][metric]
                <= primary_contract["resources"][bound]
            )
        for name, expected in row["evidence_sha256"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
        assert row["status"] == "recorded_gripper_checked"
        nq = nv = 0
        columns = {}
        world = ET.fromstring((folder / "model.xml").read_text()).find("worldbody")
        for node in world.iter():
            if node.tag not in {"joint", "freejoint"}:
                continue
            kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
            q, v = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}[kind]
            columns[node.get("name")] = nq + 1
            nq += q
            nv += v
        indices = [columns[name] for name in row["gripper_joint_names"]]
        assert indices == row["gripper_state_columns"]
        states = np.load(folder / "recorded-states.npy", allow_pickle=False)
        assert states.shape[1] == 1 + nq + nv
        assert np.isfinite(states).all()
        assert np.allclose(np.diff(states[:, 0]), 0.05, atol=1e-6, rtol=0)
        with np.load(
            base / "run" / f"candidate-{row['rank']:02d}" / "target-numeric.npz", allow_pickle=False
        ) as target:
            gripper = target["states"][:, 6:8].astype(np.float64)
        source = states[:, indices]
        for item in row["offset_diagnostics"]:
            shift = item["offset"]
            left = gripper[:-1] if shift == -1 else gripper[1:] if shift == 1 else gripper
            right = source[1:] if shift == -1 else source[:-1] if shift == 1 else source
            assert float(np.abs(left - right).max()) == item["max_error_m"]
        errors = np.abs(gripper - source)
        checked.append(
            {
                "rank": row["rank"],
                "demo_id": row["demo_id"],
                "frames": len(gripper),
                "frames_exceeding_state_tolerance": int((errors.max(axis=1) > 1e-5).sum()),
                "max_gripper_error_m": float(errors.max()),
                "max_gripper_error_after_first_frame_m": float(errors[1:].max()),
                "offset_diagnostics": row["offset_diagnostics"],
            }
        )
    return {
        "contract_digest": contract["digest"],
        "result_digest": result["digest"],
        "independently_verified": checked,
        "http_bytes": sum(r["http"]["bytes"] for r in result["results"]),
        "http_requests": sum(r["http"]["requests"] for r in result["results"]),
    }


def verify_all(base):
    primary = verify(base)
    recovery = verify(base, "recovery-bundle", "recovery-run")
    plan = read_sealed(base / "recovery-plan.json")
    contract = read_sealed(base / "bundle/contract.json")
    second = read_sealed(base / "recovery-bundle/contract.json")
    initial_result = read_sealed(base / "run/results.json")
    recovery_result = read_sealed(base / "recovery-run/results.json")
    receipt = read_sealed(base / "recovery-receipt.json")
    assert receipt["status"] == "complete" and receipt["returncode"] == 0
    assert receipt["plan_digest"] == plan["digest"]
    assert second["parent_pilot_digest"] == plan["parent_pilot_digest"] == contract["digest"]
    assert second["parent_result_digest"] == initial_result["digest"]
    assert second["recovery_plan_digest"] == plan["digest"]
    assert second["numeric_mapping"] == contract["numeric_mapping"]
    assert second["heldout_protection"] == contract["heldout_protection"]
    assert second["metadata_length_strategy"] == "num_samples_attribute"
    assert (
        hashlib.sha256((base / "recovery-controller.py").read_bytes()).hexdigest()
        == plan["controller_sha256"]
    )
    assert (
        hashlib.sha256((base / "recovery-bundle/runner.py").read_bytes()).hexdigest()
        == plan["runner_sha256"]
    )
    # Independently reconstruct recovery eligibility from the original artifacts.
    eligible = []
    for row in initial_result["candidate_results"]:
        if (
            row["status"] != "access_or_integrity_failure"
            or row.get("error") != "TimeoutError: Per-file wall limit reached"
        ):
            continue
        if any(
            row[key]
            for key in [
                "target_payload_reads",
                "original_demo_payload_reads",
                "full_simulator_state_reads",
                "comparisons",
            ]
        ):
            continue
        folder = base / "run" / f"candidate-{row['candidate']['queue_rank']:02d}"
        ranges = [json.loads(x) for x in (folder / "http-ranges.jsonl").read_text().splitlines()]
        if ranges and all(r["phase"] == "metadata" for r in ranges):
            eligible.append(row["candidate"])
    assert eligible == second["candidates"]
    final_rows = {r["candidate"]["queue_rank"]: r for r in initial_result["candidate_results"]}
    for row in recovery_result["candidate_results"]:
        rank = row["candidate"]["queue_rank"]
        original = final_rows[rank]
        assert row["candidate"] == original["candidate"]
        for metric, bound in [
            ("bytes", "per_file_http_bytes"),
            ("requests", "per_file_http_requests"),
        ]:
            assert (
                original["http"][metric] + row.get("http", {}).get(metric, 0)
                <= contract["resources"][bound]
            )
        final_rows[rank] = row
    diagnostic = verify_recorded_diagnostic(base)
    total_bytes = primary["http_bytes"] + recovery["http_bytes"] + diagnostic["http_bytes"]
    assert total_bytes <= contract["resources"]["total_network_upper_bound_bytes"]
    launch = json.loads((base / "launch.json").read_text())
    assert (
        receipt["finished_unix"] <= launch["started_unix"] + contract["resources"]["wall_seconds"]
    )
    output_bytes = sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
    assert output_bytes < contract["resources"]["output_cap_bytes"]
    matches = (
        primary["action_matches_with_state_phase_diagnostics"]
        + recovery["action_matches_with_state_phase_diagnostics"]
    )
    summaries = {r["rank"]: r for r in primary["candidates"]}
    summaries.update({r["rank"]: r for r in recovery["candidates"]})
    return seal_contract(
        {
            "artifact_type": "eventground_wam.source_matching_pilot_final_completion",
            "schema_version": 1,
            "execution_complete": len(final_rows) == 40,
            "primary_verification": primary,
            "recovery_verification": recovery,
            "recorded_state_diagnostic": diagnostic,
            "candidate_count": len(final_rows),
            "final_status_counts": dict(
                sorted(Counter(r["status"] for r in final_rows.values()).items())
            ),
            "action_matching_comparisons": len(matches),
            "action_matching_target_count": len({r["rank"] for r in matches}),
            "passing_numeric_comparisons": primary["passing_numeric_comparisons"]
            + recovery["passing_numeric_comparisons"],
            "independently_recomputed_comparisons": primary["independently_recomputed_comparisons"]
            + recovery["independently_recomputed_comparisons"],
            "all_attempts_preserved": True,
            "http_bytes_all_attempts_and_diagnostics": total_bytes,
            "http_requests_all_attempts_and_diagnostics": primary["http_requests"]
            + recovery["http_requests"]
            + diagnostic["http_requests"],
            "bounded_scan_wall_seconds": receipt["finished_unix"] - launch["started_unix"],
            "output_bytes_before_final_report": output_bytes,
            "original_time_and_download_limits_preserved": True,
            "heldout_target_payload_reads": 0,
            "physical_labels_admitted": False,
            "simulator_steps": 0,
            "training_updates": 0,
            "candidates": [summaries[i] for i in sorted(summaries)],
            "limitations": primary["limitations"],
            "conclusion": "Full numerical matches still require scene/phase verification; zero admitted physical contact/grasp sources in this pilot.",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    report = verify_all(args.base)
    with (args.base / "completion.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in {
                    "candidates",
                    "primary_verification",
                    "recovery_verification",
                    "recorded_state_diagnostic",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
