"""Stage selected v3 train initial states/XML; never simulate or read held-out payloads."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import signal
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from run_event_source_pilot import BoundedRange

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.instrumented_partitions import require_train_source
from eventground_wam.instrumented_supervision import numeric_digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    d = json.loads(Path(path).read_text())
    if d.get("digest") != contract_digest(d):
        raise ValueError(f"Changed record: {path}")
    return d


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(seal_contract(value), f, indent=2, sort_keys=True)
        f.write("\n")


def check_job(task, split, base):
    for item in task["sources"]:
        row = require_train_source(split, item["source_id"], allow_reserve=True)
        for k in ("task_key", "hdf5_path", "demo_id", "frames", "queue_role"):
            if row[k] != item[k]:
                raise ValueError(f"Unapproved source field {k}")
        actions = base / item["actions_file"]
        if sha(actions) != item["actions_sha256"]:
            raise ValueError("Changed action payload")
        arr = np.load(actions, allow_pickle=False)["actions"]
        if (
            arr.shape != (row["frames"], 7)
            or not np.isfinite(arr).all()
            or numeric_digest(arr) != item["action_digest"]
        ):
            raise ValueError("Wrong action identity or shape")


def work(task, contract, split, base):
    out = base / "run" / task["task_key"].replace("/", "-")
    out.mkdir()
    begin = time.monotonic()
    rows = []
    reader = None
    failure = None
    try:
        check_job(task, split, base)  # Complete selection checks before opening a remote container.
        if shutil.disk_usage(base).free < contract["resources"]["minimum_free_bytes"]:
            raise RuntimeError("Insufficient space")
        reader = BoundedRange(task["url"], out, contract["resources"])
        reader.fetch(0, 1)
        with h5py.File(reader, "r") as h:
            for item in task["sources"]:
                reader.phase = f"train_initial_state_and_model_xml:{item['demo_id']}"
                d = h["data"][item["demo_id"]]
                if (
                    d["states"].ndim != 2
                    or d["states"].shape[0] != item["frames"]
                    or not 1 < d["states"].shape[1] <= 4096
                ):
                    raise ValueError("Unexpected state shape")
                initial = d["states"][0]
                if not np.isfinite(initial).all():
                    raise ValueError("Nonfinite initial state")
                xml = d.attrs["model_file"]
                if isinstance(xml, bytes):
                    xml = xml.decode("utf-8")
                if not isinstance(xml, str) or len(xml.encode()) > 2 * 1024**2:
                    raise ValueError("Invalid XML size/type")
                tree = ET.fromstring(xml)
                body = tree.find("worldbody")
                if body is None:
                    raise ValueError("Missing source worldbody")
                joints = [e.attrib["name"] for e in body.iter() if e.tag in ("joint", "freejoint")]
                if not joints or len(joints) != len(set(joints)):
                    raise ValueError("Ambiguous source joint ordering")
                target = out / item["source_id"]
                target.mkdir()
                np.save(target / "initial-state.npy", initial, allow_pickle=False)
                (target / "source-model.xml").write_text(xml)
                shutil.copy2(base / item["actions_file"], target / "actions.npz")
                result = {
                    "source_id": item["source_id"],
                    "demo_id": item["demo_id"],
                    "task_key": item["task_key"],
                    "queue_role": item["queue_role"],
                    "frames": item["frames"],
                    "action_digest": item["action_digest"],
                    "source_state_dataset_shape": list(d["states"].shape),
                    "source_state_rows_read": [0],
                    "initial_state_shape": list(initial.shape),
                    "source_joint_names": joints,
                    "input_sha256": {p.name: sha(p) for p in target.iterdir() if p.is_file()},
                    "status": "staged",
                    "source_model_is_new_capture_scene": False,
                    "simulator_steps": 0,
                    "heldout_payload_reads": 0,
                }
                save(target / "input-identity.json", result)
                rows.append(sealed(target / "input-identity.json"))
                if shutil.disk_usage(base).free < contract["resources"]["minimum_free_bytes"]:
                    raise RuntimeError("Insufficient space")
                if (
                    sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
                    > contract["resources"]["per_task_output_bytes"]
                ):
                    raise RuntimeError("Per-task output ceiling")
    except Exception as exc:  # noqa: BLE001 -- preserve bounded failures; never change selection
        failure = {"error": repr(exc), "traceback": traceback.format_exc()}
    result = {
        "task_key": task["task_key"],
        "status": "complete"
        if failure is None and len(rows) == len(task["sources"])
        else "incomplete",
        "inputs": rows,
        "failure": failure,
        "http_bytes": reader.downloaded if reader else 0,
        "http_requests": reader.requests if reader else 0,
        "elapsed_seconds": time.monotonic() - begin,
        "heldout_payload_reads": 0,
        "simulator_steps": 0,
        "training_updates": 0,
    }
    save(out / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("task_key", "status", "http_bytes", "elapsed_seconds", "failure")
            }
        ),
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, type=Path)
    a = p.parse_args()
    base = a.base.resolve()
    c = sealed(base / "staging-contract.json")
    split = sealed(base / "partitions.json")
    if split["digest"] != c["partition_digest"]:
        raise ValueError("Wrong active partition")
    for rel, h in c["bundle_sha256"].items():
        if sha(base / rel) != h:
            raise ValueError(f"Changed bundle file {rel}")
    ids = [i["source_id"] for task in c["tasks"] for i in task["sources"]]
    if (
        len(ids) != 300
        or len(set(ids)) != 300
        or set(ids) != {r["source_id"] for r in split["rows"] if r["partition"] == "train"}
    ):
        raise ValueError("Staging membership mismatch")
    for task in c["tasks"]:
        check_job(task, split, base)
    (base / "run").mkdir(exist_ok=False)
    begin = time.monotonic()

    def deadline(signum, frame):
        raise TimeoutError("Global source-staging wall limit")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(c["resources"]["global_seconds"])
    save(
        base / "run/launch.json",
        {"staging_contract_digest": c["digest"], "started_unix": time.time()},
    )
    rows = []
    failure = None
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            for f in concurrent.futures.as_completed(
                [pool.submit(work, t, c, split, base) for t in c["tasks"]]
            ):
                rows.append(f.result())
    except Exception as exc:  # noqa: BLE001 -- preserve process/runtime failures in completion evidence
        failure = repr(exc)
    finally:
        signal.alarm(0)
        save(
            base / "run/completion.json",
            {
                "staging_contract_digest": c["digest"],
                "partition_digest": split["digest"],
                "status": "complete"
                if len(rows) == 40
                and all(r["status"] == "complete" for r in rows)
                and failure is None
                else "incomplete",
                "tasks": sorted(rows, key=lambda r: r["task_key"]),
                "failure": failure,
                "elapsed_seconds": time.monotonic() - begin,
                "source_inputs_staged": sum(len(r["inputs"]) for r in rows),
                "http_bytes": sum(r["http_bytes"] for r in rows),
                "heldout_payload_reads": 0,
                "new_simulator_steps": 0,
                "training_updates": 0,
            },
        )
    if failure or len(rows) != 40 or any(r["status"] != "complete" for r in rows):
        raise RuntimeError("Incomplete source staging; preserved outcomes require review")


if __name__ == "__main__":
    main()
