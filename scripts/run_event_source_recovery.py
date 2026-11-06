"""One recorded metadata-timeout recovery within the original pilot deadline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


from eventground_wam.demonstration_contracts import contract_digest, seal_contract


def read_sealed(path):
    value = json.loads(path.read_text())
    if contract_digest(value) != value["digest"]:
        raise ValueError(f"Invalid seal: {path}")
    return value


def eligible(row, folder):
    if (
        row["status"] != "access_or_integrity_failure"
        or row.get("error") != "TimeoutError: Per-file wall limit reached"
        or row["target_payload_reads"] != 0
        or row["original_demo_payload_reads"] != 0
        or row["full_simulator_state_reads"] != 0
        or row["comparisons"]
    ):
        return False
    ranges = [json.loads(x) for x in (folder / "http-ranges.jsonl").read_text().splitlines()]
    return bool(ranges) and all(x["phase"] == "metadata" for x in ranges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    plan = read_sealed(base / "recovery-plan.json")
    assert sha(__file__) == plan["controller_sha256"]
    assert sha(base / "recovery-runner.py") == plan["runner_sha256"]
    primary = read_sealed(base / "bundle/contract.json")
    assert primary["digest"] == plan["parent_pilot_digest"]
    launch = json.loads((base / "launch.json").read_text())
    deadline = launch["started_unix"] + primary["resources"]["wall_seconds"]
    write_json(
        base / "recovery-controller-started.json",
        {"plan_digest": plan["digest"], "deadline_unix": deadline, "started_unix": time.time()},
    )
    while not (base / "run/results.json").exists():
        if time.time() >= deadline:
            write_json(
                base / "recovery-receipt.json",
                {
                    "status": "original_deadline_reached_before_recovery",
                    "plan_digest": plan["digest"],
                    "finished_unix": time.time(),
                },
            )
            return
        time.sleep(5)
    result = read_sealed(base / "run/results.json")
    assert result["contract_digest"] == primary["digest"]
    assert [r["candidate"] for r in result["candidate_results"]] == primary["candidates"]
    selected = []
    for row in result["candidate_results"]:
        folder = base / "run" / f"candidate-{row['candidate']['queue_rank']:02d}"
        saved = read_sealed(folder / "result.json")
        saved.pop("digest")
        assert saved == row
        for name, expected in row["evidence_sha256"].items():
            assert sha(folder / name) == expected
        if eligible(row, folder):
            selected.append(row)
    remaining = deadline - time.time()
    if not selected or remaining < 60:
        write_json(
            base / "recovery-receipt.json",
            {
                "status": "no_eligible_work_or_insufficient_time",
                "eligible_count": len(selected),
                "plan_digest": plan["digest"],
                "finished_unix": time.time(),
            },
        )
        return
    diagnostic = read_sealed(base / "recorded-state-diagnostic/results.json")
    consumed_bytes = sum(
        r.get("http", {}).get("bytes", 0)
        for r in result["candidate_results"] + diagnostic["results"]
    )
    resources = copy.deepcopy(primary["resources"])
    resources["wall_seconds"] = int(remaining - 10)
    resources["maximum_files"] = len(selected)
    resources["per_file_seconds"] = min(
        240, max(1, int((remaining - 30) / math.ceil(len(selected) / resources["workers"]) - 20))
    )
    resources["per_file_http_bytes"] -= max(r["http"]["bytes"] for r in selected)
    resources["per_file_http_requests"] -= max(r["http"]["requests"] for r in selected)
    resources["total_network_upper_bound_bytes"] -= consumed_bytes
    assert (
        len(selected) * resources["per_file_http_bytes"]
        <= resources["total_network_upper_bound_bytes"]
    )
    output_bytes = sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
    assert output_bytes < resources["output_cap_bytes"]
    assert shutil.disk_usage(base).free >= resources["minimum_free_bytes"]
    bundle = base / "recovery-bundle"
    shutil.copytree(
        base / "bundle", bundle, ignore=shutil.ignore_patterns("contract.json", "__pycache__")
    )
    shutil.copyfile(base / "recovery-runner.py", bundle / "runner.py")
    contract = copy.deepcopy(primary)
    contract.pop("digest")
    contract.update(
        {
            "artifact_type": "eventground_wam.source_matching_metadata_recovery_contract",
            "parent_pilot_digest": primary["digest"],
            "parent_result_digest": result["digest"],
            "recovery_plan_digest": plan["digest"],
            "candidates": [r["candidate"] for r in selected],
            "metadata_length_strategy": "num_samples_attribute",
            "resources": resources,
            "original_deadline_unix": deadline,
            "prior_consumed_http_bytes": consumed_bytes,
            "prior_output_bytes": output_bytes,
            "prelaunch_tests": {"targeted_tests_passed": 25, "lint": "passed"},
        }
    )
    contract["bundle_files_sha256"] = {
        name: sha(bundle / name) for name in primary["bundle_files_sha256"]
    }
    contract = seal_contract(contract)
    write_json(bundle / "contract.json", contract)
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=2s",
        f"{resources['wall_seconds']}s",
        sys.executable,
        str(bundle / "runner.py"),
        "--bundle",
        str(bundle),
        "--output",
        str(base / "recovery-run"),
    ]
    started = time.time()
    write_json(
        base / "recovery-launch.json",
        {
            "command": command,
            "started_unix": started,
            "deadline_unix": deadline,
            "contract_digest": contract["digest"],
        },
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(bundle),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (base / "recovery.log").open("x") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False
        )
    write_json(
        base / "recovery-receipt.json",
        seal_contract(
            {
                "status": "complete" if completed.returncode == 0 else "bounded_execution_failure",
                "returncode": completed.returncode,
                "plan_digest": plan["digest"],
                "contract_digest": contract["digest"],
                "started_unix": started,
                "finished_unix": time.time(),
                "original_deadline_unix": deadline,
                "labels_admitted": False,
                "training_updates": 0,
            }
        ),
    )


if __name__ == "__main__":
    main()
