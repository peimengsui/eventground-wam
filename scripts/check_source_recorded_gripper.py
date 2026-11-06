"""Compare recorded gripper qpos without replay, simulation, or label extraction."""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from runner import BoundedRange, sha, write_json

from eventground_wam.demonstration_contracts import contract_digest, seal_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    contract = json.loads((base / "recorded-state-diagnostic-contract.json").read_text())
    assert contract_digest(contract) == contract["digest"]
    primary = json.loads((base / "bundle/contract.json").read_text())
    assert primary["digest"] == contract["parent_pilot_digest"]
    start = time.monotonic()
    output = base / "recorded-state-diagnostic"
    output.mkdir(exist_ok=False)
    results = []
    for item in contract["targets"]:
        row = {**item, "status": "started", "labels_admitted": False}
        reader = None
        folder = output / f"candidate-{item['rank']:02d}"
        folder.mkdir()
        try:
            if time.monotonic() - start > contract["resources"]["wall_seconds"]:
                raise TimeoutError("Diagnostic deadline")
            launch = json.loads((base / "launch.json").read_text())
            if time.time() > launch["started_unix"] + primary["resources"]["wall_seconds"]:
                raise TimeoutError("Original global pilot deadline")
            case = base / "run" / f"candidate-{item['rank']:02d}"
            result = json.loads((case / "result.json").read_text())
            assert contract_digest(result) == result["digest"]
            assert item["rank"] not in primary["heldout_protection"]["excluded_queue_ranks"]
            comparison = next(c for c in result["comparisons"] if c["demo_id"] == item["demo_id"])
            assert comparison["alignment"]["max_action_absolute_error"] <= 1e-6
            assert comparison["alignment"]["gripper_commands_exact"]
            for name, expected in result["evidence_sha256"].items():
                assert sha(case / name) == expected
            with np.load(case / "target-numeric.npz", allow_pickle=False) as target:
                target_gripper = target["states"][:, 6:8].copy()
            reader = BoundedRange(
                result["candidate"]["candidate_original_hdf5_url"], folder, contract["resources"]
            )
            reader.fetch(0, 1)
            reader.phase = "selected_recorded_state_diagnostic"
            with h5py.File(reader, "r") as handle:
                demo = handle[f"data/{item['demo_id']}"]
                states = demo["states"][()]
                xml = demo.attrs["model_file"]
                xml = xml.decode() if isinstance(xml, bytes) else str(xml)
                np.save(folder / "recorded-states.npy", states, allow_pickle=False)
                (folder / "model.xml").write_text(xml)
            root = ET.fromstring(xml)
            world = root.find("worldbody")
            if world is None:
                raise ValueError("No worldbody")
            # MuJoCo generalized coordinates follow body-tree/joint order.
            # Reject unrecognized types or extra state blocks rather than guess.
            nq = nv = 0
            addresses = {}
            for node in world.iter():
                if node.tag not in {"joint", "freejoint"}:
                    continue
                kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
                q, v = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}[kind]
                addresses[node.get("name")] = {"qpos_address": nq, "qpos_size": q}
                nq += q
                nv += v
            row.update(
                {
                    "nq_from_xml": nq,
                    "nv_from_xml": nv,
                    "recorded_state_shape": list(states.shape),
                    "joint_addresses": addresses,
                }
            )
            if states.ndim != 2 or states.shape != (len(target_gripper), 1 + nq + nv):
                raise ValueError("Full-state layout is not unambiguous time/qpos/qvel")
            if not np.isfinite(states).all() or not np.allclose(
                np.diff(states[:, 0]), 0.05, atol=1e-6, rtol=0
            ):
                raise ValueError("Recorded state clock/finite check failed")
            names = ["gripper0_finger_joint1", "gripper0_finger_joint2"]
            indices = [1 + addresses[name]["qpos_address"] for name in names]
            assert all(addresses[name]["qpos_size"] == 1 for name in names)
            gripper = states[:, indices]
            diagnostics = []
            for offset in (-1, 0, 1):
                left = (
                    target_gripper[:-1]
                    if offset == -1
                    else target_gripper[1:]
                    if offset == 1
                    else target_gripper
                )
                right = gripper[1:] if offset == -1 else gripper[:-1] if offset == 1 else gripper
                error = np.abs(left.astype(np.float64) - right)
                diagnostics.append(
                    {
                        "offset": offset,
                        "frames": len(left),
                        "max_error_m": float(error.max()),
                        "per_finger_max_error_m": error.max(axis=0).tolist(),
                    }
                )
            row.update(
                {
                    "status": "recorded_gripper_checked",
                    "gripper_joint_names": names,
                    "gripper_state_columns": indices,
                    "offset_diagnostics": diagnostics,
                    "same_index_gripper_agreement": diagnostics[1]["max_error_m"] <= 1e-5,
                }
            )
        except (OSError, ValueError, KeyError, RuntimeError, AssertionError, StopIteration) as exc:
            row.update({"status": "diagnostic_unresolved", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if reader is not None:
                row["http"] = {
                    "bytes": reader.downloaded,
                    "requests": reader.requests,
                    "seconds": time.monotonic() - reader.started,
                }
                reader.close()
            row["evidence_sha256"] = {
                p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()
            }
            write_json(folder / "result.json", seal_contract(row))
        results.append(row)
        print(json.dumps(row), flush=True)
    write_json(
        output / "results.json",
        seal_contract(
            {
                "contract_digest": contract["digest"],
                "results": results,
                "executed_source_sha256": sha(__file__),
                "elapsed_seconds": time.monotonic() - start,
                "simulator_steps": 0,
                "labels_admitted": False,
            }
        ),
    )


if __name__ == "__main__":
    main()
