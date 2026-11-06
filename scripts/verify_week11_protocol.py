#!/usr/bin/env python3
"""Verify the Step 1 freeze using local metadata only; never load a model."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = "configs/evaluation/week11_integration_v1.json"
CONTRACT_DIGEST = "5d7e919252104f9814086474f6a92633edc1d86bff984608c0bbb80a7b4c4a0a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(record):
    payload = {key: value for key, value in record.items() if key != "digest"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, *, sealed=False):
    result = json.loads(path.read_text())
    if sealed:
        require(result.get("digest") == digest(result), f"Invalid seal: {path}")
    return result


def task_key(row):
    return row["suite"], row["runtime_task_index"]


def verify(root=ROOT):
    root = Path(root)
    contract = read(root / CONTRACT_PATH, sealed=True)
    require(contract["digest"] == CONTRACT_DIGEST,
            "Week 11 freeze changed; use a versioned amendment")
    for name, expected in contract["source_files_sha256"].items():
        require(sha(root / name) == expected, f"Frozen input changed: {name}")

    panels = {}
    for name, identity in contract["panels"].items():
        path = root / identity["path"]
        require(sha(path) == identity["sha256"], f"Panel bytes changed: {name}")
        panel = read(path, sealed=True)
        require(panel["digest"] == identity["digest"], f"Panel identity changed: {name}")
        require(panel["count"] == identity["count"], f"Panel count changed: {name}")
        require(sha(root / panel["parent_path"]) == panel["parent_sha256"],
                f"Panel parent changed: {name}")
        panels[name] = panel

    targeted = panels["targeted-467"]
    plus = read(root / targeted["parent_path"])["tasks"]
    target_rows = [row for row in plus if row["perturbation_category"]
                   in targeted["selection"]["categories"]]
    require(targeted["tasks"] == target_rows and len(target_rows) == 467,
            "Targeted task records/order differ from the parent filter")
    require(len({task_key(row) for row in target_rows}) == 467,
            "Duplicate targeted task")
    dev = read(root / "evaluation/splits/libero-plus-v1/dev-210.json")["tasks"]
    dev_keys = {task_key(row) for row in dev}
    overlap = sum(task_key(row) in dev_keys for row in target_rows)
    require(overlap == contract["panel_accounting"]["dev210_overlap"] == 90,
            "Development overlap changed")
    require(dict(Counter(row["perturbation_category"] for row in target_rows))
            == contract["panel_accounting"]["targeted_category_counts"],
            "Category accounting changed")

    clean = panels["clean-200"]
    parent_clean = read(root / clean["parent_path"], sealed=True)["pairs"]
    clean_rows = [row for row in parent_clean if row["initial_state_index"] in range(5)]
    require(clean["pairs"] == clean_rows and len(clean_rows) == 200,
            "Clean panel records, seeds or order changed")
    clusters = defaultdict(list)
    for row in clean_rows:
        clusters[task_key(row)].append(row)
    require(len(clusters) == 40 and all(
        len(rows) == 5 and {r["initial_state_index"] for r in rows} == set(range(5))
        for rows in clusters.values()), "Clean canonical-task clusters changed")
    qualification = panels["qualification-8"]
    first_tasks = {suite: min(row["runtime_task_index"] for row in parent_clean
                             if row["suite"] == suite)
                   for suite in {row["suite"] for row in parent_clean}}
    smoke_rows = [row for row in parent_clean if row["initial_state_index"] in (5, 6)
                  and row["runtime_task_index"] == first_tasks[row["suite"]]]
    require(qualification["pairs"] == smoke_rows and len(smoke_rows) == 8,
            "Qualification panel differs from its deterministic selector")
    require(not ({row["pair_id"] for row in clean_rows}
                 & {row["pair_id"] for row in smoke_rows}), "Smoke/formal episode overlap")
    proxy = panels["contact-proxy-98"]
    fragments = proxy["selection"]["task_name_lowercase_contains_any"]
    proxy_rows = [row for row in target_rows
                  if any(fragment in row["task_name"].lower() for fragment in fragments)]
    require(proxy["tasks"] == proxy_rows and len(proxy_rows) == 98,
            "Contact-proxy membership changed")
    require(len(contract["arms"]) * (len(target_rows) + len(clean_rows))
            == contract["panel_accounting"]["formal_episodes_all_arms"] == 2001,
            "Formal episode accounting changed")

    handoff = read(root / contract["handoff_path"], sealed=True)
    require(handoff["digest"] == contract["handoff_digest"], "Handoff identity mismatch")
    require(handoff["parent_m4"] == contract["endpoints"]["m4"], "M4 parent mismatch")
    event = contract["endpoints"]["event"]
    require(all(handoff[key] == value for key, value in event.items()),
            "Selected event endpoint mismatch")
    parent_verification = contract["parent_verification"]
    completion = read(root / parent_verification["event_completion_path"], sealed=True)
    require(completion["digest"] == event["completion_digest"]
            and completion["endpoint_sha256"] == event["endpoint_sha256"]
            and completion["identity_digest"] == event["identity_digest"]
            and completion["inference_header"]["digest"] == event["inference_header_digest"]
            and completion["status"] == "complete" and completion["global_step"] == 32
            and completion["prompts_seen"] == 500 and not completion["partial_update_in_progress"]
            and all(completion["gates"].values()), "Selected completion cross-link mismatch")
    require(digest(completion["inference_header"]) == event["inference_header_digest"],
            "Endpoint header seal mismatch")
    media = read(root / parent_verification["last_remote_verification_path"], sealed=True)
    require(media["digest"] == parent_verification["last_remote_verification_digest"]
            and media["handoff_digest"] == handoff["digest"]
            and media["selected_endpoint_sha256"] == event["endpoint_sha256"]
            and media["selected_endpoint_verified"] and media["parent_m4_verified"],
            "Historical remote verification cross-link mismatch")
    closeout = read(root / "docs/evaluation/week10-closeout-v1/completion.json", sealed=True)
    require(closeout["status"] == "complete" and closeout["handoff_digest"] == handoff["digest"]
            and closeout["remote_checkpoint_media_verification_digest"] == media["digest"]
            and not closeout["independent_qa_passed"], "Closeout scope changed")
    study = read(root / "configs/training/week10_eventground_experiment_v3.json", sealed=True)
    require(study["parent"] == handoff["parent_m4"], "M4 loading-contract mismatch")

    predictions_path = contract["diagnostics"]["weak_timing_source"]
    predictions = [json.loads(line) for line in (root / predictions_path).read_text().splitlines()]
    dev_completion = read((root / predictions_path).parent / "completion.json", sealed=True)
    dev_manifest = read(root / contract["diagnostics"]["weak_timing_manifest"], sealed=True)
    require(len(predictions) == contract["diagnostics"]["weak_timing_prompts"] == 396
            and dev_completion["predictions_sha256"] == sha(root / predictions_path)
            and dev_completion["endpoint_completion_digest"] == completion["digest"]
            and dev_completion["manifest_digest"] == dev_manifest["digest"],
            "Recorded diagnostic prediction provenance mismatch")

    entry = read(root / "docs/evaluation/week11-freeze-v1/entry.json", sealed=True)
    require(entry["contract_digest"] == contract["digest"]
            and not entry["remote_bytes_freshly_rehashed"]
            and entry["new_model_inferences"] == entry["training_updates"] == entry["rollouts"] == 0,
            "Entry scope mismatch")
    for name, expected in entry["original_evidence_files_sha256"].items():
        require(sha(root / name) == expected, f"Prior evidence changed: {name}")

    result = {
        "artifact_type": "eventground_wam.week11_protocol_verification",
        "contract_digest": contract["digest"], "passed": True,
        "verified_source_files": len(contract["source_files_sha256"]),
        "targeted_tasks": 467, "development_overlap": overlap, "contact_proxy_tasks": 98,
        "clean_canonical_tasks": 40, "clean_episodes_per_arm": 200,
        "formal_episodes_all_arms": 2001, "qualification_episodes_all_arms": 24,
        "scope": "Local records, seals, selectors and cross-links only",
        "remote_bytes_freshly_rehashed": False, "hardware_qualified": False,
        "mapping_frozen": False, "models_loaded": False, "rollouts_launched": False,
    }
    result["digest"] = digest(result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
