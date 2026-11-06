"""Bounded source-action identity mapping; no simulator, images, labels or training."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import signal
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
from run_event_source_pilot import BoundedRange

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.instrumented_supervision import numeric_digest, physical_source_id


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    with Path(path).open("x") as f:
        json.dump(seal_contract(obj), f, indent=2, sort_keys=True)
        f.write("\n")


def work(task, contract, base):
    import pyarrow.parquet as pq

    folder = base / "run" / task["task_key"].replace("/", "-")
    folder.mkdir()
    prior = base / "prior-run" / task["task_key"].replace("/", "-")
    if prior.exists():
        for p in prior.iterdir():
            if p.is_file() and (p.suffix == ".npz" or p.name == "progress.jsonl"):
                shutil.copy2(p, folder / p.name)
    reader = None
    results = (
        [json.loads(x) for x in (folder / "progress.jsonl").read_text().splitlines()]
        if (folder / "progress.jsonl").exists()
        else []
    )
    completed_groups = {r["group_digest"] for r in results}
    matched = sum(r["status"] == "exact_action_binding" for r in results)
    reads = 0
    begin = time.monotonic()
    summary = {"task_key": task["task_key"], "status": "started"}
    try:
        limits = contract["resources"]
        assert shutil.disk_usage(base).free >= limits["minimum_free_bytes"]
        reader = BoundedRange(task["url"], folder, limits)
        reader.fetch(0, 1)
        with h5py.File(reader, "r") as source:
            data = source["data"]
            if (prior / "source-lengths.json").exists():
                lengths = json.loads((prior / "source-lengths.json").read_text())["lengths"]
            elif task.get("cached_schema"):
                schema = json.loads((base / task["cached_schema"]).read_text())
                lengths = {
                    name: int(row.get("declared_num_samples", row.get("actions", [0])[0]))
                    for name, row in schema.items()
                }
            else:
                names = sorted(data.keys())
                assert len(names) <= 50
                lengths = {name: int(data[name].attrs["num_samples"]) for name in names}
            save(folder / "source-lengths.json", {"source_url": task["url"], "lengths": lengths})
            action_cache = {
                p.name.removesuffix("-actions.npz"): np.load(p)["actions"]
                for p in folder.glob("demo_*-actions.npz")
            }
            for candidate in task["candidates"]:
                if matched >= task["target_bindings"]:
                    break
                if candidate["group"]["group_digest"] in completed_groups:
                    continue
                if time.monotonic() - begin > limits["per_file_seconds"]:
                    raise TimeoutError("Per-task time ceiling")
                g = candidate["group"]
                n = g["episode_length"]
                ep = candidate["episode"]
                assert g["split"] == "train" and ep in g["episode_indices"]
                assert n not in task["protected_lengths"]
                result = {
                    "group_digest": g["group_digest"],
                    "episode": ep,
                    "frames": n,
                    "status": "started",
                    "source_matches": [],
                }
                matches = sorted(name for name, length in lengths.items() if length == n)
                result["equal_length_source_demos"] = matches
                if not matches:
                    result["status"] = "no_equal_length_source"
                elif len(matches) > 3:
                    result["status"] = "ambiguous_source_length"
                else:
                    target = (
                        Path(contract["target_snapshot"])
                        / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
                    )
                    table = pq.read_table(
                        target,
                        columns=["observation.state", "action", "episode_index", "frame_index"],
                    )
                    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
                    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
                    assert set(table["episode_index"].to_pylist()) == {ep} and table[
                        "frame_index"
                    ].to_pylist() == list(range(n))
                    nd = numeric_digest(states, actions)
                    assert nd in g["numeric_digests"]
                    assert np.isfinite(states).all() and np.isfinite(actions).all()
                    result.update(
                        {
                            "target_numeric_digest": nd,
                            "action_digest": numeric_digest(actions),
                            "target_parquet_sha256": sha(target),
                        }
                    )
                    np.savez_compressed(folder / f"target-{ep}.npz", states=states, actions=actions)
                    reads += 1
                    for demo in matches:
                        if demo not in action_cache:
                            reader.phase = f"eligible_train_source_actions:{demo}"
                            assert data[demo]["actions"].shape == (n, 7)
                            arr = data[demo]["actions"][()]
                            assert np.isfinite(arr).all()
                            action_cache[demo] = arr.copy()
                            np.savez_compressed(folder / f"{demo}-actions.npz", actions=arr)
                        arr = action_cache[demo]
                        if numeric_digest(arr) == numeric_digest(actions):
                            result["source_matches"].append(demo)
                    if len(result["source_matches"]) == 1:
                        demo = result["source_matches"][0]
                        result.update(
                            {
                                "status": "exact_action_binding",
                                "demo_id": demo,
                                "source_id": physical_source_id(
                                    contract["repository"],
                                    contract["revision"],
                                    task["hdf5_path"],
                                    demo,
                                ),
                                "known_alias_episode_ids": g["episode_indices"],
                                "partition": "train",
                                "source_actions_sha256": sha(folder / f"{demo}-actions.npz"),
                                "source_closure_production_admission": False,
                            }
                        )
                        matched += 1
                    else:
                        result["status"] = (
                            "no_exact_action_binding"
                            if not result["source_matches"]
                            else "ambiguous_exact_source_binding"
                        )
                results.append(result)
                with (folder / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")
                if matched >= task["target_bindings"]:
                    break
            summary.update(
                {
                    "status": "target_reached"
                    if matched >= task["target_bindings"]
                    else "candidate_pool_exhausted"
                }
            )
    except Exception as exc:  # noqa: BLE001 -- preserve bounded failure, no retry
        summary.update(
            {"status": "stopped", "error": repr(exc), "traceback": traceback.format_exc()}
        )
    summary.update(
        {
            "results": results,
            "exact_bindings": matched,
            "target_payload_reads": reads,
            "source_action_payloads": len(list(folder.glob("demo_*-actions.npz"))),
            "http_bytes": reader.downloaded if reader else 0,
            "http_requests": reader.requests if reader else 0,
            "elapsed_seconds": time.monotonic() - begin,
            "preserved_prior_results": len(completed_groups),
            "heldout_payload_reads": 0,
            "source_full_state_reads": 0,
            "simulator_steps": 0,
            "labels_extracted": False,
            "production_admitted": False,
        }
    )
    summary["evidence_sha256"] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
    save(folder / "result.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("task_key", "status", "exact_bindings", "http_bytes", "elapsed_seconds")
            }
        ),
        flush=True,
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    a = p.parse_args()
    base = a.base.resolve()
    c = json.loads((base / "contract.json").read_text())
    assert contract_digest(c) == c["digest"]
    for rel, h in c["bundle_sha256"].items():
        assert sha(base / rel) == h, rel
    (base / "run").mkdir(exist_ok=False)
    start = time.monotonic()

    def deadline(signum, frame):
        raise TimeoutError("Global mapping time ceiling")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(max(1, int(c["absolute_deadline_unix"] - time.time())))
    save(base / "run/launch.json", {"contract_digest": c["digest"], "started_unix": time.time()})
    results = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(work, t, c, base) for t in c["tasks"]]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
    finally:
        signal.alarm(0)
        save(
            base / "run/completion.json",
            {
                "contract_digest": c["digest"],
                "tasks": [
                    {
                        k: r[k]
                        for k in (
                            "task_key",
                            "status",
                            "exact_bindings",
                            "target_payload_reads",
                            "source_action_payloads",
                            "http_bytes",
                            "elapsed_seconds",
                        )
                    }
                    for r in sorted(results, key=lambda r: r["task_key"])
                ],
                "exact_bindings": sum(r["exact_bindings"] for r in results),
                "http_bytes": sum(r["http_bytes"] for r in results),
                "elapsed_seconds": time.monotonic() - start,
                "production_admitted": False,
                "preserved_prior_results": sum(r["preserved_prior_results"] for r in results),
                "heldout_payload_reads": 0,
                "simulator_steps": 0,
                "training_updates": 0,
            },
        )


if __name__ == "__main__":
    main()
