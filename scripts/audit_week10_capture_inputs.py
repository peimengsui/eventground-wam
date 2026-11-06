"""Independently audit staged train inputs against the approved split and canonical schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.instrumented_partitions import require_train_source
from eventground_wam.instrumented_supervision import numeric_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    value = json.loads(path.read_text())
    assert value["digest"] == contract_digest(value), str(path)
    return value


def audit(root):
    evidence = root / "docs/evaluation"
    base = evidence / "week10-capture-source-staging-v1"
    schema_base = evidence / "week10-capture-schema-preflight-corrected-v1"
    c = load(base / "staging-contract.json")
    split = load(evidence / "week10-source-split-v3/partitions.json")
    plan = load(evidence / "week10-production-plan-v2/recording-plan-proposal.json")
    done = load(base / "run/completion.json")
    schemas = load(schema_base / "run/completion.json")
    schema_contract = load(schema_base / "contract.json")
    assert done["staging_contract_digest"] == c["digest"]
    assert done["partition_digest"] == c["partition_digest"] == split["digest"]
    assert done["status"] == schemas["status"] == "complete"
    assert len(done["tasks"]) == len(schemas["tasks"]) == 40
    assert schemas["contract_digest"] == schema_contract["digest"]
    for rel, expected in c["bundle_sha256"].items():
        assert sha(base / rel) == expected, rel
    by_task = {r["task_key"]: r for r in schemas["tasks"]}
    for key, row in by_task.items():
        folder = schema_base / "run" / key.replace("/", "-")
        assert load(folder / "result.json") == row
        assert sha(folder / "canonical-model.xml") == row["canonical_model_sha256"]
        assert row["status"] == "passed" and row["physics_steps"] == row["rendered_images"] == 0
    found = {}
    http_bytes = 0
    for task in done["tasks"]:
        folder = base / "run" / task["task_key"].replace("/", "-")
        assert seal_contract(task) == load(folder / "result.json")
        assert task["status"] == "complete"
        ranges = [
            json.loads(line) for line in (folder / "http-ranges.jsonl").read_text().splitlines()
        ]
        assert sum(r["length"] for r in ranges) == task["http_bytes"]
        assert len(ranges) == task["http_requests"] <= c["resources"]["per_file_http_requests"]
        assert task["http_bytes"] <= c["resources"]["per_file_http_bytes"]
        assert task["elapsed_seconds"] <= c["resources"]["per_file_seconds"]
        http_bytes += task["http_bytes"]
        for item in task["inputs"]:
            sid = item["source_id"]
            assert sid not in found
            member = require_train_source(split, sid, allow_reserve=True)
            src = folder / sid
            assert item == load(src / "input-identity.json")
            for key in ("task_key", "demo_id", "frames", "queue_role"):
                assert item[key] == member[key]
            assert item["status"] == "staged" and item["source_state_rows_read"] == [0]
            assert item["heldout_payload_reads"] == item["simulator_steps"] == 0
            assert set(item["input_sha256"]) == {
                "actions.npz",
                "initial-state.npy",
                "source-model.xml",
            }
            for rel, expected in item["input_sha256"].items():
                assert sha(src / rel) == expected
            with np.load(src / "actions.npz", allow_pickle=False) as saved:
                actions = saved["actions"]
            assert actions.shape == (item["frames"], 7) and np.isfinite(actions).all()
            assert numeric_digest(actions) == item["action_digest"]
            initial = np.load(src / "initial-state.npy", allow_pickle=False)
            schema = by_task[item["task_key"]]
            assert np.isfinite(initial).all()
            assert (
                list(initial.shape)
                == item["initial_state_shape"]
                == schema["flattened_state_shape"]
            )
            joints = [
                e.attrib["name"]
                for e in ET.parse(src / "source-model.xml").getroot().find("worldbody").iter()
                if e.tag in ("joint", "freejoint")
            ]
            assert joints == item["source_joint_names"] == schema["joint_names"]
            assert len(joints) == len(set(joints))
            found[sid] = {
                "source_id": sid,
                "input_identity_digest": item["digest"],
                "action_digest": item["action_digest"],
                "canonical_schema_digest": schema["digest"],
                "joint_order_and_state_shape_match": True,
                "input_sha256": item["input_sha256"],
            }
    assert set(found) == {r["source_id"] for r in split["rows"] if r["partition"] == "train"}
    assert len(found) == done["source_inputs_staged"] == 300
    for item in plan["rows"]:
        assert found[item["source_id"]]["action_digest"] == item["action_digest"]
    assert http_bytes == done["http_bytes"] <= c["resources"]["cumulative_http_bytes"]
    assert done["elapsed_seconds"] <= c["resources"]["global_seconds"]
    output_bytes = sum(p.stat().st_size for p in (base / "run").rglob("*") if p.is_file())
    assert output_bytes <= c["resources"]["output_bytes"]
    return seal_contract(
        {
            "artifact_type": "eventground_wam.production_input_verification",
            "status": "passed_source_identity_and_schema_not_recording_or_label_QA",
            "partition_digest": split["digest"],
            "staging_contract_digest": c["digest"],
            "staging_completion_digest": done["digest"],
            "schema_completion_digest": schemas["digest"],
            "recording_plan_digest": plan["digest"],
            "sources_verified": len(found),
            "tasks_verified": len(by_task),
            "http_bytes_independently_summed": http_bytes,
            "staging_output_bytes": output_bytes,
            "rows": [found[r["source_id"]] for r in plan["rows"]],
            "heldout_payload_reads": 0,
            "source_full_trace_scene_identity_claimed": False,
            "historical_alias_closure_claimed": False,
            "recordings_created": 0,
            "labels_admitted": False,
            "training_updates": 0,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
