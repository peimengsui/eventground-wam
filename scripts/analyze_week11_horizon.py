"""Apply the already sealed rule to existing dev predictions, without model calls."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from verify_week11_horizon import CONTRACT_PATH, ROOT, parent_verifier, verify

from eventground_wam.eventground_horizon import SUPPORTED_HORIZONS, select_horizon


def timing(errors):
    absolute = [abs(error) for error in errors]
    ordered = sorted(absolute)
    if not ordered:
        return {"n": 0, "mae_frames": None, "median_absolute_frames": None,
                "p90_absolute_frames": None, "mean_signed_frames": None,
                "within_two_frames": 0, "more_than_eight_frames": 0}
    position = (len(ordered) - 1) * .9
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    p90 = ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    return {"n": len(errors), "mae_frames": statistics.mean(absolute),
            "median_absolute_frames": statistics.median(absolute), "p90_absolute_frames": p90,
            "mean_signed_frames": statistics.mean(errors),
            "within_two_frames": sum(error <= 2 for error in absolute),
            "more_than_eight_frames": sum(error > 8 for error in absolute)}


def summarize(records):
    predicted_positive = [r for r in records if r["decision"]["raw_start"] is not None]
    target_positive = [r for r in records if r["target"]["next_event"] != "none"]
    matched = [r for r in target_positive if r["correct_event"]]

    def paired(rows):
        return {"raw": timing([r["raw_signed_error"] for r in rows]),
                "mapped": timing([r["mapped_signed_error"] for r in rows])}

    selected = Counter(str(r["decision"]["selected_horizon"]) for r in records)
    return {
        "prompts": len(records), "sources": len({r["source_id"] for r in records}),
        "target_positive": len(target_positive), "correctly_typed_positive": len(matched),
        "positive_type_coverage": len(matched) / len(target_positive) if target_positive else None,
        "predicted_positive": len(predicted_positive),
        "valid_none_fallbacks": len(records) - len(predicted_positive),
        "raw_positive_out_of_supported_set": sum(
            r["decision"]["raw_start"] not in SUPPORTED_HORIZONS for r in predicted_positive),
        "raw_positive_start_counts": dict(sorted(Counter(
            str(r["decision"]["raw_start"]) for r in predicted_positive).items())),
        "selected_horizon_counts_all_prompts": {str(h): selected[str(h)] for h in SUPPORTED_HORIZONS},
        "selected_horizon_different_from_eight": sum(
            r["decision"]["selected_horizon"] != 8 for r in records),
        "quantization_delta_counts_positive_only": dict(sorted(Counter(
            str(r["decision"]["quantization_delta_frames"]) for r in predicted_positive).items())),
        "conditional_onset_errors": paired(matched),
        "by_target_lead_bin": {name: paired([r for r in matched if r["lead_time_bin"] == name])
                               for name in ("1-4", "5-8", "9-16", "17-32")},
        "interpretation": "Descriptive balanced weak-label dev reuse; no rule selection, new "
                          "model inference, on-policy calibration, independent QA or robot benefit",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root
    parent = parent_verifier(root)
    verification = verify(root)
    child = parent.read(root / CONTRACT_PATH, sealed=True)
    started_at = datetime.now(timezone.utc).isoformat()
    parent.require(started_at >= child["created_at_utc"], "Diagnostic precedes mapping seal")
    diagnostic = child["diagnostic"]
    manifest = parent.read(root / diagnostic["manifest_path"], sealed=True)
    predictions = [json.loads(line) for line in
                   (root / diagnostic["predictions_path"]).read_text().splitlines()]
    parent.require(len(predictions) == len(manifest["rows"]) == diagnostic["prompts"],
                   "Diagnostic count mismatch")
    records = []
    for saved, example in zip(predictions, manifest["rows"], strict=True):
        parent.require(all(saved[key] == example[key]
                           for key in ("prompt_id", "source_id", "target", "lead_time_bin")),
                       "Prediction/manifest mismatch")
        decision = select_horizon(saved["prediction"])
        correct = decision.event_type == example["target"]["next_event"]
        parent.require(correct == saved["correct_event"], "Recorded event correctness mismatch")
        matched = correct and decision.raw_start is not None
        target_start = example["target"]["horizon"]
        records.append({"prompt_id": saved["prompt_id"], "source_id": saved["source_id"],
                        "target": example["target"], "prediction": saved["prediction"],
                        "lead_time_bin": saved["lead_time_bin"], "correct_event": correct,
                        "decision": asdict(decision),
                        "raw_signed_error": decision.raw_start - target_start if matched else None,
                        "mapped_signed_error": decision.selected_horizon - target_start
                        if matched else None})
    result = summarize(records)
    result.update({"artifact_type": "eventground_wam.week11_recorded_horizon_diagnostic",
                   "contract_digest": child["digest"], "verification_digest": verification["digest"],
                   "started_at_utc": started_at, "model_calls": 0, "training_updates": 0,
                   "robot_rollouts": 0, "report_panel_read": False, "rule_changed": False,
                   "inputs_sha256": {name: parent.sha(root / name) for name in
                                     (diagnostic["predictions_path"], diagnostic["manifest_path"])}})
    out = args.out or root / "docs/evaluation/week11-horizon-v1/diagnostic"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "decisions.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    result["decisions_sha256"] = parent.sha(out / "decisions.jsonl")
    result["digest"] = parent.digest(result)
    with (out / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
